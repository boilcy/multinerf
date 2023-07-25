import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from flax import linen as nn
from flax.core.frozen_dict import unfreeze
from flax.training.train_state import TrainState
from internal import math
import optax
from scipy.spatial.transform import Rotation

from utils import extract_delta

@jit
def vec2ss_matrix(vector):
    return jnp.array([[0, -vector[2], vector[1]],
                      [vector[2], 0, -vector[0]],
                      [-vector[1], vector[0], 0]])


class camera_transf(nn.Module):
    def setup(self):
        self.w = self.param('w',  jax.nn.initializers.normal(stddev=1e-6), (3,))
        self.v = self.param('v',  jax.nn.initializers.zeros, (3,))
        self.theta = self.param('theta',  jax.nn.initializers.normal(stddev=1e-6), ())

    def __call__(self, x):
        bottom = jnp.array([0, 0, 0, 1]).reshape(1, 4)
        y = jnp.concatenate([x, bottom], axis=0)
        w_skewsym = vec2ss_matrix(self.w)
        a = jnp.eye(3) + jnp.sin(self.theta) * w_skewsym + (1 - jnp.cos(self.theta)) * jnp.matmul(w_skewsym, w_skewsym)
        b = jnp.matmul(jnp.eye(3) * self.theta + (1 - jnp.cos(self.theta)) * w_skewsym + (self.theta - jnp.sin(self.theta)) * jnp.matmul(w_skewsym, w_skewsym), self.v)
        b = b.reshape(3,1)

        exp_i = jnp.concatenate([a, b], axis=1)
        T_i = jnp.matmul(exp_i, y)[:3, :4]
        return T_i


# 也可以使用jax.pure_callback来完成，但需要自己定义求导规则
# https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html#example-pure-callback-with-custom-jvp
@jit
def rotation_matrix(vector):
    vector = jnp.radians(vector)
    phi = vector[2]
    theta = vector[1]
    psi = vector[0]

    Rx = jnp.array([[1, 0, 0], [0, jnp.cos(phi), -jnp.sin(phi)], [0, jnp.sin(phi), jnp.cos(phi)]])
    Ry = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)], [0, 1, 0], [-jnp.sin(theta), 0, jnp.cos(theta)]])
    Rz = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0], [jnp.sin(psi), jnp.cos(psi), 0], [0, 0, 1]])
    R = jnp.dot(jnp.dot(Rx, Ry), Rz)
    return R

class direct_pose(nn.Module):
    def setup(self):
        self.euler = self.param('euler',  jax.nn.initializers.zeros, (3,))
        self.t = self.param('t',  jax.nn.initializers.zeros, (3,))

    def __call__(self):
        r = rotation_matrix(self.euler)
        t = self.t.reshape(3,1)

        return jnp.concatenate([r,t], axis=1)

class direct_se3(nn.Module):
    def setup(self):
        self.se3 = self.param('t', jax.nn.initializers.zeros, (3, 4))

    def __call__(self):
        return self.se3

class camera_transf_batch(nn.Module):
    n: int  # Declare 'n' as a class attribute

    def setup(self):
        # Create a list of camera_transf models
        self.camera_transfs = [camera_transf(name=f'camera_transf_{i}') for i in range(self.n)]

    def __call__(self, cams):
        # Apply each camera_transf model to corresponding input
        output_cams = []
        for i in range(self.n):
            output_cams.append(self.camera_transfs[i](cams[i]))
        return jnp.asarray(output_cams)

def setup_union_optim(config, rng, n: int):
    dummy_cams = np.ones((n,3,4))

    model = camera_transf_batch(n)
    init_variables = model.init(rng, dummy_cams)

    state, lr_fn = create_optimizer(init_variables, config)
    return model, state, lr_fn

def create_optimizer(variables, config):
    adam_kwargs = {
        'b1': 0.9,
        'b2': 0.999,
        'eps': 1e-6,
    }
    
    lr_kwargs = {
        'max_steps': config.pose_max_steps,
        'lr_delay_steps': config.lr_delay_steps,
        'lr_delay_mult': config.lr_delay_mult,
    }
    
    max_steps = config.pose_max_steps
    lr_init = config.pose_lr_init
    lr_final = config.pose_lr_final
    
    def get_lr_fn1(max_steps, lr_init, lr_final, step):
        return lr_init * (0.8 ** (3*step/max_steps))
    
    def get_lr_fn2(lr_init, lr_final):
        return functools.partial(
            math.learning_rate_decay,
            lr_init=lr_init,
            lr_final=lr_final,
            **lr_kwargs)
    
    def get_lr_fn3(max_steps, lr_init, lr_final, step):
        return math.log_lerp(step/max_steps, lr_init, lr_final)
    
    lr_fn_main = functools.partial(get_lr_fn3, max_steps, lr_init, lr_final)
    tx = optax.adam(learning_rate=lr_fn_main, **adam_kwargs)

    return TrainState.create(apply_fn=None, params=variables, tx=tx), lr_fn_main

def setup_model(config, rng, params=None):
    if config.pose_optim_method=="manifold":
        dummy_c2w = jnp.array([[1., 0., 0., 1.],
                    [0., 1., 0., 1.],
                    [0., 0., 1., 1.]])
        model = camera_transf()
        init_variables = model.init(rng, dummy_c2w)

        state, lr_fn = create_optimizer(init_variables, config)

        return model, state, lr_fn
    elif config.pose_optim_method=="direct_pose":
        model = direct_pose()
        init_variables = model.init(rng)
        init_variables = unfreeze(init_variables)
        euler, t = extract_delta(params)
        init_variables['params']['euler'] = euler
        init_variables['params']['t'] = t
        state, lr_fn = create_optimizer(init_variables, config)
        return model, state, lr_fn
    elif config.pose_optim_method=="direct_se3":
        model = direct_se3()
        init_variables = model.init(rng)
        init_variables = unfreeze(init_variables)
        init_variables['params']['se3'] = params
        state, lr_fn = create_optimizer(init_variables, config)
        return model, state, lr_fn
    else:
        return ValueError("optim_method is not valid")
