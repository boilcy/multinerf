import functools
import os
import flax
import gin
import datetime
from absl import app
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, value_and_grad
from flax.training import checkpoints
import matplotlib.pyplot as plt
from internal import camera_utils
from internal import train_utils
from internal import models
from internal import datasets
from internal import configs
from internal import utils
from inerf_helper import setup_model
from utils import get_noised_pose, extract_delta, find_Edge, find_EdgeRegion, create_alpha_fn

configs.define_common_flags()
jax.config.parse_flags_with_absl()

def create_train_step(model: models.Model,
                      modelState,
                      poseModel,
                      camera,
                      config: configs.Config,
                      camtype):
    
    def train_step(batch, poseState, alpha):
        def loss_fn(variables):
            rays = batch.rays
            c2w = poseModel.apply(variables, camera[1])
            _camera = (camera[0], c2w, camera[2],camera[3])
            rays = camera_utils.cast_ray_batch(_camera, rays, camtype, xnp=jnp)

            renderings, _ = model.apply(
                modelState.params,
                None,
                alpha,
                rays,
                train_frac=1.0,
                compute_extras=False,
                zero_glo=False)

            mse = jnp.mean(jnp.square(renderings[-1]['rgb'] - batch.rgb[..., :3]))
            return mse
        
        loss_grad_fn = jax.value_and_grad(loss_fn, argnums=(0,), has_aux=False)
        mse, grad = loss_grad_fn(poseState.params)
        pmean = lambda x: jax.lax.pmean(x, axis_name='batch')
        grad = pmean(grad)

        grad = train_utils.clip_gradients(grad, config)
        grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)

        new_poseState = poseState.apply_gradients(grads=grad)

        return new_poseState

    train_vstep = jax.vmap(
        train_step,
        axis_name='batch',
        in_axes=(None, 0, None))
    return train_vstep

def load_model(config, rng):
    dummy_rays = utils.dummy_rays(include_exposure_idx=config.rawnerf_mode, include_exposure_values=True)
    model, variables = models.construct_model(rng, dummy_rays, config)
    state, _ = train_utils.create_optimizer(config, variables)
    state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
    step = int(state.step)
    print(f'Load checkpoint at step {step}.')
    render_eval_pfn = train_utils.create_render_fn(model)
    return model, state, render_eval_pfn

def save_pose_config(out_dir, config):
    fn = os.path.join(out_dir, 'poseconfig.txt')
    with open(fn, 'w') as f:
        f.write(f'render_train={config.pose_render_train}\n')
        f.write(f'batch_size={config.batch_size}\n')
        f.write(f'patch_size={config.patch_size}\n')
        f.write(f'max_steps={config.pose_max_steps}\n')
        f.write(f'lr_init={config.pose_lr_init}\n')
        f.write(f'lr_final={config.pose_lr_final}\n')
        f.write(f'sampling_strategy={config.pose_sampling_strategy}\n')
        f.write(f'with_filter={config.pose_w_alpha}\n')
        f.write(f'optim_method={config.pose_optim_method}\n')
        f.write(f'alpha0={config.pose_alpha0}\n')
        f.write(f'delta_phi={config.pose_delta_phi}\n')
        f.write(f'delta_theta={config.pose_delta_theta}\n')
        f.write(f'delta_psi={config.pose_delta_psi}\n')
        f.write(f'delta_x={config.pose_delta_x}\n')
        f.write(f'delta_y={config.pose_delta_y}\n')
        f.write(f'delta_z={config.pose_delta_z}\n')

def save_final_err(out_dir, *args):
    fn = os.path.join(out_dir, 'err.txt')
    with open(fn, 'w+') as f:
        for err in args:
            if isinstance(err, list):
                f.write("|".join(err))
            else:
                f.write(f"{err}")
            f.write("\n")
            
def main(unused_arg):
    config = configs.load_config(save_config=False)
    
    render_dir = config.render_dir
    exam_id = config.pose_exam_id
    out_dir = os.path.join(render_dir, f'{exam_id}/')
    if not utils.isdir(out_dir):
        utils.makedirs(out_dir)
    path_fn = lambda x: os.path.join(out_dir, x)
    
    save_pose_config(out_dir, config)
    
    rng = random.PRNGKey(0)
    model, modelState, render_eval_pfn = load_model(config, rng)

    # 指定相机
    cam_idx = 0

    dataset = datasets.load_dataset('test', config.data_dir, config)
    p2c, distortion_param, p2c_ndc, camtype = dataset.load_data_property(cam_idx)
    obs_img, obs_img_c2w = dataset.load_obs_data(cam_idx)

    delta = (config.pose_delta_x, config.pose_delta_y, config.pose_delta_z, config.pose_delta_phi, config.pose_delta_theta, config.pose_delta_psi)
    start_c2w = get_noised_pose(obs_img_c2w, delta)

    start_camera = (p2c, start_c2w, distortion_param, p2c_ndc)

    np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
    start_camera = tuple(np_to_jax(x) for x in start_camera)

    #euler_ref, t_ref = extract_delta(obs_img_c2w)

    poseModel, poseState, lr_fn = setup_model(config, rng, start_camera[1])
    poseState = flax.jax_utils.replicate(poseState)
    
    # (N, 2)
    if config.pose_sampling_strategy == 'edge':
        obs_img_prime = (np.array(obs_img) * 255).astype(np.uint8)
        POI = find_Edge(obs_img_prime, False)
    elif config.pose_sampling_strategy == 'edge_region':
        obs_img_prime = (np.array(obs_img) * 255).astype(np.uint8)
        POI = find_EdgeRegion(obs_img_prime, False)
    else:
        POI = None

    train_vstep = create_train_step(model, modelState, poseModel, start_camera, config, camtype)

    #是否在训练时render图片
    render_train = config.pose_render_train
    
    Nt = config.pose_max_steps
    #是否使用低通滤波器
    w_alpha = config.pose_w_alpha
    #初始频率阈值
    alpha0 = config.pose_alpha0

    #创建alpha_fn
    alpha_fn = create_alpha_fn(alpha0, 0.95, config.pose_alpha_linear)

    # 迭代更新位姿
    for Nc in range(Nt+1):
        
        alpha = alpha_fn(Nc/Nt) if w_alpha else 1.0
            
        lr = lr_fn(Nc)
        
        batch = dataset.generate_pose_batch(cam_idx, POI)

        poseState = train_vstep(batch, poseState, alpha)
        print(poseState.param)
        
    # A hack that forces Jax to keep all TPUs alive until every TPU is finished.
    x = jax.numpy.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    print(x)

if __name__ == '__main__':
    with gin.config_scope('eval'):  # Use the same scope as eval.py
        app.run(main)