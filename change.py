rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

trans_t = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]])

def load_blender() -> obs_img, obs_img_pose, [H, W, focal]:
    img_rgb = np.asarray(img_rgb*255, dtype=np.uint8)
    obs_img_pose = np.array(frames[obs_img_num]['transform_matrix']).astype(np.float32)
    phi, theta, psi, t = kwargs
    start_pose =  trans_t(t) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi)  @ obs_img_pose
    pass

def generate_start_pose(obs_img_pose, [H, W, focal]):
    psi = np.random.uniform(0, np.pi)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, np.pi)

    t = (x,y,z)

optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))
pose = cam_transf(start_pose)

def render(pose) -> rendered_img:
    pass

for k in range(N):
    rendered_img = render(pose)

    optimizer.zero_grad()
    loss = img2mse(rgb, target_s)
    loss.backward()

out_dir = "/home/yc/code/multinerf/obs/"
path_fn = lambda x: os.path.join(out_dir, x)

utils.save_img_u8(im, path_fn(f'{k}.png'))
