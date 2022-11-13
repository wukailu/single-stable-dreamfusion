import argparse
import os
import sys

import imageio
import numpy as np
import torch

if os.getcwd() not in sys.path:
    sys.path = [os.getcwd()] + sys.path
from frameworks.nerf.renderers.image_renderer import ImageRenderer
from frameworks.nerf.modules import load_nerf

pos_keys = {
    '-x': torch.tensor([-1.0, 0.0, 0.0]),
    '-y': torch.tensor([0.0, -1.0, 0.0]),
    '-z': torch.tensor([0.0, 0.0, -1.0]),
    'x': torch.tensor([1.0, 0.0, 0.0]),
    'y': torch.tensor([0.0, 1.0, 0.0]),
    'z': torch.tensor([0.0, 0.0, 1.0]),
}


def c2w_to_w2c(mat):
    ret = torch.zeros(4, 4)
    ret[3, 3] = 1
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, [3]] = -mat[:3, :3].T @ mat[:3, [3]]
    return ret


# 注意区分 extrinsic matrix 和 camera's pose matrix
def extrinsic_to_look_at(mat):
    R = mat[:3, :3]
    t = mat[:3, [3]]
    C = -R.inverse() @ t
    L = -R.T[:, [2]]
    p = C + L
    return C[:, 0], p[:, 0]


# using methods from https://ksimek.github.io/2012/08/22/extrinsic/
def look_at_to_w2c(C: torch.Tensor, p: torch.Tensor):
    up = torch.nn.functional.normalize(torch.tensor([0.0, 0.0, 1.0]), dim=0)
    L = (p - C)
    s = torch.cross(L, up)
    u = torch.cross(s, L)
    R = torch.nn.functional.normalize(torch.stack([s, u, -L]), dim=1)
    t = -R @ C[None].T
    ret = torch.zeros((4, 4))
    ret[:3, :3] = R
    ret[:3, [3]] = t
    ret[3, 3] = 1.0
    return ret


def look_at_to_c2w(C: torch.Tensor, p: torch.Tensor, up=torch.tensor([0.1, 0.1, 1.0])):
    """from C to look at p, the up direction is z+"""
    up = torch.nn.functional.normalize(up, dim=0)
    L = (p - C)
    s = torch.cross(L, up)
    u = torch.cross(s, L)
    R = torch.nn.functional.normalize(torch.stack([s, u, -L]), dim=1).T
    ret = torch.zeros((4, 4))
    ret[:3, :3] = R
    ret[:3, 3] = C
    ret[3, 3] = 1.0
    return ret


# see Spherical coordinate system
def cord_spherical(radius, theta, phi):
    import math
    theta = theta / 180 * math.pi
    phi = phi / 180 * math.pi
    return torch.tensor([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]) * radius


def render_round_views(nerf, inverse_y, flip_x, flip_y, H, W, focal, num_imgs=16, P=torch.tensor([0.0, 0.0, 0.0]),
                       dis=1, up='z'):
    render_poses = torch.stack([
        look_at_to_c2w(cord_spherical(dis, 60, angle) + P, P, up=pos_keys[up])
        for angle in np.linspace(-180, 180, num_imgs + 1)[:-1]], 0)
    # render_poses = torch.stack([look_at_to_extrinsic(p, o).inverse()])

    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    Ks = K[None].repeat(len(render_poses), axis=0)
    HWs = np.array([[H, W]] * len(render_poses))

    renderer = ImageRenderer(inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, ndc=False)
    with torch.no_grad():
        return renderer.renderViews(render_poses=render_poses, HW_list=HWs, K_list=Ks, nerf=nerf)


def snap_shot(nerf, inverse_y=False, flip_x=False, flip_y=False, H=400, W=400, focal=1000, C=None, P=None,
              pos='x', up='z', dis_coe=2, img_type='plane', render_pose=None, key='rgb_marched', **kwargs):
    """
        get a snapshot (from C to look at P) from a nerf.
    :param up: up direction of snap_shot, default z
    :param dis_coe: dis to center = dis_coe * scenes size
    :param pos: where to look at the object if C, P is None, 'x' by default
    :param nerf: the nerf
    :param inverse_y: bool
    :param flip_x: bool
    :param flip_y: bool
    :param H: int
    :param W: int
    :param focal: int
    :param C: view position, if it's None, it will be calculated by dis_coe and pos
    :param P: look at P from C, if it's None, it will be the center of bbox
    :param img_type: whether it's a common image or a panoramic one
    :param render_pose: c2w matrix, default None, if it's None it will be calculated by P,C
    :param key: the value that need to be saved.
    :return: rgb data in range [0, 1] if key = rgb_marched.
    """
    if render_pose is None:
        if C is None or P is None:
            P = (nerf.xyz_min.cpu() + nerf.xyz_max.cpu()) / 2
            dis_p = (nerf.xyz_max.cpu() - nerf.xyz_min.cpu()).norm(p=2).item() * float(dis_coe)
            v_pos = torch.tensor([0, 0, 0]).float()

            all_mismatch = False
            while len(pos) > 0 and not all_mismatch:
                all_mismatch = True
                for k, v in pos_keys.items():
                    if pos.startswith(k):
                        v_pos += v
                        pos = pos[len(k):]
                        all_mismatch = False

            assert torch.norm(v_pos, p=2) > 0
            v_pos = v_pos / torch.norm(v_pos, p=2) * dis_p
            C = P + v_pos

        render_pose = look_at_to_c2w(C, P, up=pos_keys[up] + torch.tensor([1e-3, 1e-3, 1e-3]))

    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    renderer = ImageRenderer(inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, ndc=False, img_type=img_type, key=key)

    with torch.no_grad():
        return renderer.renderView(H, W, K, render_pose, nerf)


def save_rgbs(img, path):
    img = (255 * np.clip(img.cpu().numpy(), 0, 1)).astype(np.uint8)
    imageio.imsave(path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gridNerf_save_path', default='/data/tmp/jade_small.gnrf')
    parser.add_argument('--export_video_name', default='test.mp4')
    parser.add_argument('--inverse_y', default=True)
    parser.add_argument('--flip_x', default=False, action='store_true')
    parser.add_argument('--flip_y', default=False, action='store_true')
    parser.add_argument('--H', default=400)
    parser.add_argument('--W', default=400)
    parser.add_argument('--focal', default=1000, type=int)
    opt = parser.parse_args()

    loaded_nerf = load_nerf(opt.gridNerf_save_path).cuda()
    loaded_nerf.render_kwargs['far'] *= 5
    pos = (loaded_nerf.xyz_min.cpu() + loaded_nerf.xyz_max.cpu()) / 2
    dis = (loaded_nerf.xyz_min.cpu() - loaded_nerf.xyz_max.cpu()).norm(p=2) * 2
    print("dis=", dis)
    rgbs = render_round_views(loaded_nerf, opt.inverse_y, opt.flip_x, opt.flip_y, opt.H, opt.W, opt.focal, P=pos,
                              dis=dis)
    imageio.mimwrite(os.path.join(os.path.dirname(opt.gridNerf_save_path), opt.export_video_name),
                     [(255 * np.clip(rgb.cpu().numpy(), 0, 1)).astype(np.uint8) for rgb in rgbs], fps=4, quality=5)
