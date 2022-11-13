import glob
import os

import imageio
import numpy as np
import torch

from datasets.nerf.lib.load_blender import pose_spherical


def load_nsvf_data(basedir):
    pose_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*txt')))
    rgb_paths = sorted(glob.glob(os.path.join(basedir, 'rgb', '*png')))

    all_poses = []
    all_imgs = []
    i_split = [[], [], []]
    for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, rgb_paths)):
        i_set = int(os.path.split(rgb_path)[-1][0])
        all_imgs.append((imageio.imread(rgb_path) / 255.).astype(np.float32))
        all_poses.append(np.loadtxt(pose_path).astype(np.float32))
        i_split[i_set].append(i)

    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)

    H, W = imgs[0].shape[:2]
    with open(os.path.join(basedir, 'intrinsics.txt')) as f:
        focal = float(f.readline().split()[0])

    render_poses = torch.stack([pose_spherical(angle, -30.0, 1.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    return imgs, poses, render_poses, [H, W, focal], i_split
