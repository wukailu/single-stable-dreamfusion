import json
import os

import imageio
import numpy as np
import torch

from datasets.nerf.lib.load_blender import pose_spherical

fix_rot = np.array([1, 0, 0,
                    0, -1, 0,
                    0, 0, -1]).reshape(3, 3)


def load_split(path):
    cluster_lines = open(path).read().splitlines()
    return [int(line) for line in cluster_lines if len(line) > 0]


def load_toydesk_data(basedir):
    with open(os.path.join(basedir, 'transforms_full.json'), 'r') as fp:
        meta = json.load(fp)

    if basedir.endswith("our_desk_1"):
        near, far = 0.3, 9.0
    elif basedir.endswith("our_desk_2"):
        near, far = 0.8, 24.0
    else:
        raise NotImplementedError()

    imgs = []
    available_idxes = []
    poses = []
    for frame in meta['frames']:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        imgs.append(imageio.imread(fname) / 255.)
        pose = np.array(frame['transform_matrix'])
        pose[:3, :3] = pose[:3, :3] @ fix_rot
        available_idxes.append(frame['idx'])
        poses.append(pose)
    idx_convert = {i: idx for idx, i in enumerate(available_idxes)}
    suffix = basedir.split('/')[-1]
    split_path = os.path.join(basedir, '..', '..', 'split', suffix+'_train_0.8')
    train_idx = np.array([idx_convert[i] for i in load_split(os.path.join(split_path, 'train.txt')) if i in available_idxes])
    test_idx = np.array([idx_convert[i] for i in load_split(os.path.join(split_path, 'test.txt')) if i in available_idxes])
    i_split = [train_idx, test_idx, test_idx]
    # for toy_desk_1, there will be 79 train_idx and 8 test_idx at this line, while total img number is 96
    # i_split = [np.arange(len(imgs)), np.arange(len(imgs)//10), np.arange(len(imgs)//10)]  # for fine training

    imgs = np.array(imgs).astype(np.float32)
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    return imgs, poses, render_poses, [H, W, focal], i_split, near, far