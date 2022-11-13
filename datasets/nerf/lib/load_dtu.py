import os
import imageio
import numpy as np
import cv2
import glob


def load_K_Rt(world_mat, scale_mat):
    P = (world_mat @ scale_mat)[:3, :4]
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def load_dtu_data(basedir):
    imgs = [imageio.imread(f) / 255. for f in sorted(glob.glob(os.path.join(basedir, 'image', '*.png')))]
    masks = [imageio.imread(f) / 255. for f in sorted(glob.glob(os.path.join(basedir, 'mask', '*.png')))]
    assert len(imgs) == len(masks)
    for i in range(len(imgs)):
        imgs[i][masks[i] == np.array([0, 0, 0])] = 1

    camera_dict = np.load(os.path.join(basedir, 'cameras.npz'))
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(imgs))]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(imgs))]
    Ks, poses = list(zip(*[load_K_Rt(scale_mat, world_mat) for scale_mat, world_mat in zip(scale_mats, world_mats)]))
    Ks = np.stack(Ks)[..., :3, :3]
    poses = np.stack(poses)

    # generate i_split
    tot_len, len_train = len(imgs), int(len(imgs) * 0.8)
    perm = np.random.RandomState(seed=233).permutation(tot_len)
    # i_split = [perm[:len_train], perm[len_train:], perm[len_train:]]
    i_split = [perm, perm[len_train:], perm[len_train:]]
    print("Warning! All images are used for training!!!")

    H, W = imgs[0].shape[:2]
    focal = float(Ks[0][0, 0])

    return np.stack(imgs).astype(np.float32), poses, poses[i_split[2]], [H, W, focal], Ks[0], i_split