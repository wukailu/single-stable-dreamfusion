import json
import os
import glob
import numpy as np
import imageio


def load_cam(file):
    """return c2w matrix
    """
    cfg = json.load(file)
    return np.linalg.inv(np.array(cfg['camera_rt_matrix'] + [[0.0, 0.0, 0.0, 1.0]]))


def convert_d(d):
    d[d == 65535] = 0
    d = d / 512.0  # accurate dis
    return d


def fill_d(d):
    H, W = d.shape[:2]

    d[d == 65535] = 0
    d = d / 512.0  # accurate dis

    loops = 0
    while d.min() == 0:
        loops += 1
        if loops > 1000:
            assert False, "impossible to fill depth img"
        idx_0, idx_1 = np.where(d == 0)
        d_fill = np.zeros(d.shape)
        d_fill[idx_0, idx_1] = 1

        for i in range(H):
            y_idx = np.where(d_fill[i] > 0)[0]

            if len(y_idx) == 0: continue
            if len(y_idx) == 1:
                d_fill[i, y_idx[0]] = (d[i, y_idx[0] - 1] + d[i, (y_idx[0] + 1) % W]) / 2
                continue
            if len(y_idx) == W:
                d_fill[i] = 0
                if i != 0 and d[i - 1, 0] != 0:
                    d[i, 0] = d[i - 1, 0]
                else:
                    d[i, 0] = d[min(i + 1, H - 1), 0]
                continue

            gaps = [[s, e] for s, e in zip(y_idx, y_idx[1:]) if s + 1 < e]
            edges = np.concatenate([y_idx[:1], np.array(sum(gaps, [])), y_idx[-1:]])

            interval = [[int(s), int(e) + 1] for s, e in zip(edges[::2], edges[1:][::2])]
            if interval[0][0] == 0:
                interval[0][0] = interval[-1][0] - W
                interval = interval[:-1]

            for s, e in interval:
                if s < 0:
                    interp = np.linspace(d[i, s - 1], d[i, (e + 1) % W], e - s)
                    d_fill[i, s:] = interp[:-s]
                    d_fill[i, :e] = interp[-s:]
                else:
                    d_fill[i, s:e] = np.linspace(d[i, s - 1], d[i, (e + 1) % W], e - s)
        d = d + d_fill
    return d


# dir example: /data/stanford/area_3/pano/
def load_stanford3d_data(basedir):
    cam_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*.json')))
    img_paths = sorted(glob.glob(os.path.join(basedir, 'rgb', '*.png')))
    depth_paths = sorted(glob.glob(os.path.join(basedir, 'depth', '*.png')))

    images = [(imageio.imread(path)[..., :3] / 255.).astype(np.float32) for path in img_paths]
    depths = [(imageio.imread(path)) for path in depth_paths]
    cams = [load_cam(open(path)) for path in cam_paths]

    images = np.stack(images, axis=0)
    # depths = [fill_d(d) for d in depths]
    depths = [convert_d(d) for d in depths]  # for accurate mask
    depths = np.stack(depths, axis=0)
    cams = np.stack(cams, axis=0)

    # generate i_split
    tot_len, len_train = len(images), int(len(images) * 0.8)
    perm = np.random.RandomState(seed=233).permutation(tot_len)
    # i_split = [perm[:len_train], perm[len_train:], perm[len_train:]]
    i_split = [perm, perm[len_train:], perm[len_train:]]  # 例如 piccolo 的点云本质上也用了所有的数据

    H, W = images[0].shape[:2]
    return images, cams, depths, cams[i_split[-1]], [H, W, 1], np.zeros((3, 3)), i_split
