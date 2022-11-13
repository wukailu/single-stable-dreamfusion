import glob
import os

import imageio
import numpy as np


def load_cam(file):
    """Read txt file into lines.
    """
    cam = np.zeros((4, 4)).astype(np.float32)
    words = file.read().split()
    # read extrinsic
    for i in range(0, 3):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j
            cam[i][j] = words[extrinsic_index]
    cam[3][3] = 1
    cam = np.linalg.inv(cam)
    return cam


# dir example: /data/omniscenes_ver_1_1_release/change_handheld_pose/handheld_pyebaekRoom_1_scene_2
def load_omniScenes_data(basedir):
    cam_paths = sorted(glob.glob(os.path.join(basedir.replace('pano', 'pose'), 'seq_*txt')))
    img_paths = sorted(glob.glob(os.path.join(basedir.replace('pose', 'pano'), 'seq_*jpg')))

    images = [(imageio.imread(path) / 255.).astype(np.float32) for path in img_paths]
    cams = [load_cam(open(path)) for path in cam_paths]
    intrinsics = np.zeros((3, 3))

    # 为了去除底座影响，球的上下10%不要(因为分不清楚上下，所以直接都裁了好了)
    # 似乎有bug
    # images = [im[int(im.shape[0]*0.1):-int(im.shape[0]*0.1)] for im in images]

    # return mvs input
    images = np.stack(images, axis=0)
    cams = np.stack(cams, axis=0)

    # generate i_split
    tot_len, len_train = len(images), int(len(images) * 0.8)
    perm = np.random.RandomState(seed=233).permutation(tot_len)
    i_split = [perm[:len_train], perm[len_train:], perm[len_train:]]

    H, W = images[0].shape[:2]
    focal = 1

    return images, cams, cams[i_split[-1]], [H, W, focal], intrinsics, i_split
