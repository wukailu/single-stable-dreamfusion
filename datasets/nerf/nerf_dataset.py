import time

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.nerf.utils import sample_ray, get_rays_of_a_view
from datasets.utils import FullDatasetBase


class Visualizer:
    def __init__(self):
        pass

    def singleView(self, data):
        pass

    def videoView(self, data):
        pass


class NeRFFullDataset(FullDatasetBase):
    num_classes = 1
    name = "NeRF"

    def __init__(self, cfg_train, cfg_model, cfg_data, data_dict, model=None, **kwargs):
        super().__init__()
        self.cfg_train = cfg_train
        self.model = model
        self.cfg_data = cfg_data
        self.data_dict = data_dict
        self.render_kwargs = get_render_kwargs(self.cfg_data, cfg_model.stepsize, self.data_dict)

    def gen_train_transforms(self):
        return None, None

    def gen_test_transforms(self):
        return None, None

    def gen_train_datasets(self, transform=None, target_transform=None):
        return NeRFData(self.data_dict, self.cfg_data, self.cfg_train, self.model, self.render_kwargs, 'train')

    # reduce validation size to speed up training
    def gen_val_datasets(self, transform=None, target_transform=None):
        return NeRFData(self.data_dict, self.cfg_data, self.cfg_train, self.model, self.render_kwargs, 'val',
                        maxsize=819200)

    # reduce test size to speed up training
    def gen_test_datasets(self, transform=None, target_transform=None):
        return NeRFData(self.data_dict, self.cfg_data, self.cfg_train, self.model, self.render_kwargs, 'test',
                        maxsize=819200)

    @staticmethod
    def is_dataset_name(name: str):
        import re
        return re.match("(nerf)$", name.lower())


class NeRFData(Dataset):
    def __init__(self, data_dict, cfg_data, cfg_train, model, render_kwargs, split, maxsize=None):
        assert split in ['train', 'val', 'test']
        test_sampler = "stanford" if "stanford" in cfg_train.ray_sampler else "random"
        ray_sampler = cfg_train.ray_sampler if split == "train" else test_sampler
        data_info = self._gather_training_rays(data_dict, cfg_data, model, render_kwargs, 'i_' + split, ray_sampler)
        self.rgb_tr, self.rays_o_tr, self.rays_d_tr, self.viewdirs_tr, self.imsz = data_info
        assert len(self.rays_o_tr) == len(self.rays_d_tr)
        if maxsize is not None and len(self.rgb_tr) > maxsize:
            perm = torch.randperm(len(self.rgb_tr))[:maxsize]
            self.rgb_tr = self.rgb_tr[perm]
            self.rays_o_tr = self.rays_o_tr[perm]
            self.rays_d_tr = self.rays_d_tr[perm]
            self.viewdirs_tr = self.viewdirs_tr[perm]

    def __getitem__(self, index):
        target = self.rgb_tr[index]
        rays_o = self.rays_o_tr[index]
        rays_d = self.rays_d_tr[index]
        viewdirs = self.viewdirs_tr[index]
        return (rays_d, rays_o, viewdirs), target

    def __len__(self):
        ret = len(self.rgb_tr)
        # return ret // 8  # TODO: comment this line
        return ret

    @staticmethod
    def _gather_training_rays(data_dict, cfg_data, model, render_kwargs, split, ray_sampler):
        HW, Ks, indexes, poses, images = [data_dict[k] for k in ['HW', 'Ks', split, 'poses', 'images']]
        if cfg_data.load_depths:
            assert not data_dict['irregular_shape']
            assert data_dict['depths'].shape == images.shape[:-1]
            images = torch.cat([images, data_dict['depths'][..., None]], dim=-1)
            if ray_sampler == "random":
                ray_sampler = "random_depth"
            elif ray_sampler not in ["ray_sampler", "stanford", "in_maskcache_stanford"]:
                raise NotImplementedError("only support random sample when using depth information")

        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu') for i in indexes]
        else:
            rgb_tr_ori = images[indexes].to('cpu')

        if ray_sampler == 'in_maskcache':
            assert model is not None
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays_in_maskcache_sampling(
                rgb_tr_ori=rgb_tr_ori, train_poses=poses[indexes],
                HW=HW[indexes], Ks=Ks[indexes],
                model=model.cuda(), render_kwargs=render_kwargs, **cfg_data)
        elif ray_sampler == 'in_maskcache_stanford':  # for stanford3D dataset
            assert model is not None
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays_in_maskcache_sampling_stanford(
                rgb_tr_ori=rgb_tr_ori, train_poses=poses[indexes],
                HW=HW[indexes], Ks=Ks[indexes],
                model=model.cuda(), render_kwargs=render_kwargs, **cfg_data)
        elif ray_sampler == 'stanford':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays_stanford(
                rgb_tr=rgb_tr_ori, train_poses=poses[indexes],
                HW=HW[indexes], Ks=Ks[indexes], **cfg_data)
        elif ray_sampler == 'random_depth':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays_depth(
                rgb_tr=rgb_tr_ori, train_poses=poses[indexes],
                HW=HW[indexes], Ks=Ks[indexes], **cfg_data)
        elif ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori, train_poses=poses[indexes],
                HW=HW[indexes], Ks=Ks[indexes], **cfg_data)
        elif ray_sampler == 'in_alpha_channel':
            # 这个类型是用于只保留部分图片中的光线，透明的部分为不加载的部分
            # 注意训练时需要物体的光线来约束物体以及物体以外的光线约束背景和空气。
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays_alpha(
                rgb_tr=rgb_tr_ori, train_poses=poses[indexes],
                HW=HW[indexes], Ks=Ks[indexes], **cfg_data)
        elif ray_sampler == "random":
            print("Using random ray sampler!")
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays(
                rgb_tr=rgb_tr_ori, train_poses=poses[indexes],
                HW=HW[indexes], Ks=Ks[indexes], **cfg_data)
        else:
            raise NotImplementedError()
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def get_render_kwargs(cfg_data, stepsize, data_dict):
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': torch.tensor(cfg_data.bkgd).float(),
        'stepsize': stepsize,
        'inverse_y': cfg_data.inverse_y,
        'flip_x': cfg_data.flip_x,
        'flip_y': cfg_data.flip_y,
    }
    return render_kwargs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, **kwargs):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks), -1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3])
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3])
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3])
    imsz = [H * W] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w, **kwargs)
        rays_o_tr[i] = rays_o
        rays_d_tr[i] = rays_d
        viewdirs_tr[i] = viewdirs
    rgb_tr = torch.reshape(rgb_tr, (-1, *rgb_tr.shape[3:]))
    rays_o_tr = rays_o_tr.reshape((-1, 3))
    rays_d_tr = rays_d_tr.resize_as(rays_o_tr)
    viewdirs_tr = viewdirs_tr.resize_as(rays_o_tr)
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_stanford(*args, **kwargs):
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays(*args, **kwargs)
    mask = ~((rgb_tr[:, 3] == 0) | ((rgb_tr[:, 0] == 0) & (rgb_tr[:, 1] == 0) & (rgb_tr[:, 2] == 0)))
    return rgb_tr[mask], rays_o_tr[mask], rays_d_tr[mask], viewdirs_tr[mask], imsz


@torch.no_grad()
def get_training_rays_depth(*args, **kwargs):
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays(*args, **kwargs)
    mask = ~(rgb_tr[:, 3] == 0)
    return rgb_tr[mask], rays_o_tr[mask], rays_d_tr[mask], viewdirs_tr[mask], imsz


@torch.no_grad()
def get_training_rays_alpha(rgb_tr, train_poses, HW, Ks, **kwargs):
    print('get_training_rays_mask: start')
    assert rgb_tr.shape[-1] == 4
    # implement this
    raise NotImplementedError()


@torch.no_grad()
def get_training_rays_in_maskcache_sampling_stanford(*args, **kwargs):
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = get_training_rays_in_maskcache_sampling(*args, **kwargs)
    mask = ~((rgb_tr[:, 3] == 0) | ((rgb_tr[:, 0] == 0) & (rgb_tr[:, 1] == 0) & (rgb_tr[:, 2] == 0)))
    return rgb_tr[mask], rays_o_tr[mask], rays_d_tr[mask], viewdirs_tr[mask], imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, model, render_kwargs, **kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    dim = rgb_tr_ori.shape[-1]
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, dim], device=DEVICE)
    rays_o_tr = torch.zeros([N, 3], device=DEVICE)
    rays_d_tr = torch.zeros_like(rays_o_tr)
    viewdirs_tr = torch.zeros_like(rays_o_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w, **kwargs)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        model = model.cuda()
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox = sample_ray(
                rays_o=rays_o[i:i + CHUNK].cuda(), rays_d=rays_d[i:i + CHUNK].cuda(),
                xyz_min=model.xyz_min, xyz_max=model.xyz_max,
                voxel_size=model.voxel_size, world_size=model.world_size, **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.mask_cache(rays_pts[~mask_outbbox]))
            mask[i:i + CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top + n].copy_(img[mask])
        rays_o_tr[top:top + n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top + n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top + n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, **kwargs):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    dim = rgb_tr_ori.shape[-1]
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, dim], device=DEVICE)
    rays_o_tr = torch.zeros([N, 3], device=DEVICE)
    rays_d_tr = torch.zeros_like(rays_o_tr)
    viewdirs_tr = torch.zeros_like(rays_o_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w, **kwargs)
        n = H * W
        rgb_tr[top:top + n].copy_(img.flatten(0, 1))
        rays_o_tr[top:top + n].copy_(rays_o.flatten(0, 1).to(DEVICE))
        rays_d_tr[top:top + n].copy_(rays_d.flatten(0, 1).to(DEVICE))
        viewdirs_tr[top:top + n].copy_(viewdirs.flatten(0, 1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz
