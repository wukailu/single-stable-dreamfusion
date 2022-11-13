import os
import imageio
import numpy as np


def load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4)).astype(np.float32)
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    cam[1][3][0] = 0
    cam[1][3][1] = 0
    cam[1][3][2] = 0
    cam[1][3][3] = 1

    cam[0] = np.linalg.inv(cam[0])
    return cam


def gen_blendedmvs_path(dataset_folder):
    """ mvs input path list """
    cluster_path = os.path.join(dataset_folder, 'cams', 'pair.txt')
    cluster_lines = open(cluster_path).read().splitlines()
    image_num = int(cluster_lines[0])

    # get per-image info
    img_paths = []
    cam_paths = []
    for idx in range(0, image_num):
        ref_idx = int(cluster_lines[2 * idx + 1])
        img_path = os.path.join(dataset_folder, 'blended_images', '%08d_masked' % ref_idx)
        for suffix in ['.jpg', '.png']:
            if os.path.isfile(img_path + suffix):
                img_paths.append(img_path + suffix)
                # ref_image_path = os.path.join(dataset_folder, 'blended_images', '%08d.jpg' % ref_idx)
                ref_cam_path = os.path.join(dataset_folder, 'cams', '%08d_cam.txt' % ref_idx)
                cam_paths.append(ref_cam_path)
                break

    return img_paths, cam_paths


def load_standard_blendedmvs_data(basedir):
    img_paths, cam_paths = gen_blendedmvs_path(basedir)

    images = [(imageio.imread(path) / 255.).astype(np.float32) for path in img_paths]
    cams = [load_cam(open(path))[0] for path in cam_paths]
    intrinsics = load_cam(open(cam_paths[0]))[1]

    # return mvs input
    images = np.stack(images, axis=0)
    cams = np.stack(cams, axis=0)

    # generate i_split
    tot_len, len_train = len(images), int(len(images) * 0.8)
    perm = np.random.RandomState(seed=233).permutation(tot_len)
    i_split = [perm[:len_train], perm[len_train:], perm[len_train:]]

    H, W = images[0].shape[:2]
    focal = float(intrinsics[0, 0])

    return images, cams, cams[i_split[-1]], [H, W, focal], intrinsics, i_split
