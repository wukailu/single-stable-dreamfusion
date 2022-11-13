import numpy as np
import torch


@torch.no_grad()
def sample_ray(rays_o, rays_d, near, far, xyz_min, xyz_max, voxel_size, world_size, stepsize, is_train=False, **kwargs):
    '''Sample query points on rays'''
    # 1. determine the maximum number of query points to cover all possible rays
    N_samples = int(far / voxel_size / stepsize) + 1
    # 2. determine the two end-points of ray bbox intersection
    vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
    rate_a = (xyz_max - rays_o) / vec
    rate_b = (xyz_min - rays_o) / vec
    t_min: torch.Tensor = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
    t_max: torch.Tensor = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
    # 3. check wheter a raw intersect the bbox or not
    mask_outbbox = (t_max <= t_min)
    # 4. sample points on each ray
    rng = torch.arange(N_samples)[None].float().to(xyz_min)
    if is_train:
        rng = rng.repeat(rays_d.shape[-2], 1)
        rng += torch.rand_like(rng[:, [0]])  # uniform
    step = stepsize * voxel_size * rng
    interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)
    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
    # 5. update mask for query points outside bbox
    mask_outbbox = mask_outbbox[..., None] | ((xyz_min > rays_pts) | (rays_pts > xyz_max)).any(dim=-1)
    return rays_pts, mask_outbbox


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center', img_type='plane', **kwargs):
    if img_type == 'panoramic':
        rays_o, rays_d = get_rays_omni(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    else:
        rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    # ray_origin, ray_direction, view_direction
    return rays_o, rays_d, viewdirs


def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i + 0.5, j + 0.5
    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))

    dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
    if not inverse_y:
        dirs = dirs * torch.tensor([1, -1, -1]).to(c2w)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, 3], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_omni(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    # 正+z,上+y,右+x
    import math
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device))
    i = i.t().float()
    j = j.t().float()

    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i + 0.5, j + 0.5
    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))

    # mapping the plane to omnidirectional
    i, j = 2 * math.pi * (i / W - 0.5), math.pi * (j / H - 0.5)  # 全景图, (phi, theta), y 向上 z 向前, [0,0,1] 为图像中心
    dirs = torch.stack([torch.cos(j) * torch.sin(i), torch.sin(j), -torch.cos(j) * torch.cos(i)], dim=-1)
    # location by PICCOLO
    # 看了看 piccolo, 做的是直接的uv坐标到[-1,1]x[-1,1]
    # i, j = 2 * math.pi * (1 - i / W), math.pi * (j / H)
    # dirs = torch.stack([torch.sin(j) * torch.sin(i), -torch.cos(j), torch.sin(j) * torch.cos(i)], dim=-1)

    if not inverse_y:
        dirs = dirs * torch.tensor([1, -1, -1]).to(c2w)

    # print("forward: ", dirs[dirs.size(0)//2, dirs.size(1)//2])
    # print("up: ", dirs[0, dirs.size(1) // 2])
    # print("down: ", dirs[-1, dirs.size(1) // 2])
    # print("left: ", dirs[dirs.size(0)//2, dirs.size(1) // 4])
    # print("right: ", dirs[dirs.size(0)//2, dirs.size(1) // 4 * 3])
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def voxel_count(model, cfg_model, cfg_train, use_depths, pcd_path, near, far, nerfData):
    if len(pcd_path) > 0:
        import pickle
        with open(pcd_path, 'rb') as f:
            pts = torch.tensor(pickle.load(f))
        cnt = model._voxel_count_pcd(pts)
    elif use_depths:
        cnt = model._voxel_count_views_depth(
            rays_o_tr=nerfData.rays_o_tr, rays_d_tr=nerfData.rays_d_tr, rays_depth=nerfData.rgb_tr[..., -1],
            stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate)
    else:
        cnt = model._voxel_count_views(
            rays_o_tr=nerfData.rays_o_tr, rays_d_tr=nerfData.rays_d_tr, near=near,
            far=far, stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate, imsz=nerfData.imsz)
    return cnt