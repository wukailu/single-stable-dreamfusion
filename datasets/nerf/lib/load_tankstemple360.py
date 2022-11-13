import os
import glob
import torch
import numpy as np
import imageio


def w2c_RT_2_c2w(R, T):
    ret = np.zeros((4, 4))
    ret[:3, :3] = R
    ret[:3, 3] = T
    ret[3, 3] = 1
    return np.linalg.inv(ret)


def load_tankstemple360_data(basedir):
    rgb_paths = sorted(glob.glob(os.path.join(basedir, 'im_*.jpg')))
    depth_paths = sorted(glob.glob(os.path.join(basedir, 'dm_*.npy')))
    Ks = np.load(os.path.join(basedir, "Ks.npy"))
    Rs = np.load(os.path.join(basedir, "Rs.npy"))
    Ts = np.load(os.path.join(basedir, "ts.npy"))

    assert (Ks.max(axis=0) - Ks.min(axis=0)).max() == 0 and len(Ks) == len(Rs) and len(Rs) == len(Ts)

    all_poses = []
    all_imgs = []
    all_depths = []
    for i in range(len(Ks)):
        all_poses.append(w2c_RT_2_c2w(Rs[i], Ts[i]).astype(np.float32))
        all_imgs.append((imageio.imread(rgb_paths[i]) / 255.).astype(np.float32))
        all_depths.append(np.load(depth_paths[i]))

    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)
    depths = np.stack(all_depths, 0)

    # generate i_split
    tot_len, len_train = len(imgs), int(len(imgs) * 0.8)
    perm = np.random.RandomState(seed=233).permutation(tot_len)
    i_split = [perm[:len_train], perm[len_train:], perm[len_train:]]

    H, W = imgs[0].shape[:2]
    K = Ks[0]
    focal = float(K[0, 0])

    render_poses = poses[i_split[-1]]
    return imgs, poses, depths, render_poses, [H, W, focal], K, i_split
