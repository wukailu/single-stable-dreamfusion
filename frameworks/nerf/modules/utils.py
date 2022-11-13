import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from datasets.nerf.nerf_dataset import NeRFData
from datasets.nerf.utils import voxel_count, get_rays_of_a_view

''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''


class MaskCache(nn.Module):
    def __init__(self, xyz_min, xyz_max, density, act_shift, voxel_size_ratio, mask_cache_thres, ks=3):
        super().__init__()
        self.mask_cache_thres = mask_cache_thres
        self.register_buffer('xyz_min', torch.FloatTensor(xyz_min))
        self.register_buffer('xyz_max', torch.FloatTensor(xyz_max))
        self.register_buffer('density', F.max_pool3d(density, kernel_size=ks, padding=ks // 2, stride=1))
        self.act_shift = act_shift
        self.voxel_size_ratio = voxel_size_ratio

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        assert (-1 <= ind_norm).all() and (ind_norm <= 1).all()
        density = F.grid_sample(self.density, ind_norm, align_corners=True)
        alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        alpha = alpha.reshape(*shape)
        return alpha >= self.mask_cache_thres

def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[..., [0]]), p.clamp_min(1e-10).cumprod(-1)], -1)


def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum


def total_variation(v, mask=None):
    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()

    if mask is not None:
        tv2 = tv2[(mask[:, :, :-1] & mask[:, :, 1:]).expand_as(tv2)]
        tv3 = tv3[(mask[:, :, :, :-1] & mask[:, :, :, 1:]).expand_as(tv3)]
        tv4 = tv4[(mask[:, :, :, :, :-1] & mask[:, :, :, :, 1:]).expand_as(tv4)]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3


def metric_loss(v, mask=None):
    shuffle_v = v[:, :, torch.randperm(v.size(2))]
    shuffle_v = shuffle_v[:, :, :, torch.randperm(v.size(3))]
    shuffle_v = shuffle_v[:, :, :, :, torch.randperm(v.size(4))]
    return total_variation(v, mask) - (v[mask.expand_as(v)] - shuffle_v[mask.expand_as(shuffle_v)]).abs().mean()


def compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip, **kwargs):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w, **cfg.data)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0, 1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm(cfg, HW, Ks, poses, i_train, near, far, depths, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    dep = HW if depths is None else depths
    for (H, W), K, c2w, d in zip(HW[i_train], Ks[i_train], poses[i_train], dep[i_train]):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, **cfg.data)
        if depths is None:
            pts_nf = torch.stack([rays_o + viewdirs * near, rays_o + viewdirs * far])
        else:
            pts_nf = (rays_o + viewdirs * d[..., None])[None, ...]
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))
    return xyz_min, xyz_max


@torch.no_grad()
def compute_bbox_by_coarse_geo(model, thres):
    print('compute_bbox_by_coarse_geo: start')
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
    ), -1)
    dense_xyz = model.xyz_min * (1 - interp) + model.xyz_max * interp
    density = model.grid_sampler(dense_xyz, model.density)[..., 0]
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    assert active_xyz.nelement() > 0
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    return xyz_min, xyz_max


def per_voxel_init(model, cfg_model, cfg_train, use_depth, pcd_path, near, far, optimizer, nerfData: NeRFData):
    cnt = voxel_count(model, cfg_model, cfg_train, use_depth, pcd_path, near, far, nerfData)
    optimizer.set_pervoxel_lr(cnt)
    with torch.no_grad():
        model.density[cnt < 2] = -100
    ratio = torch.sum(model.density == -100) / model.density.nelement() * 100
    assert ratio != 100
    print(f"empty voxel ratio {ratio}%")


def position_encoding(input, frequency):
    emb = (input.unsqueeze(-1) * frequency).flatten(-2)
    return torch.cat([input, emb.sin(), emb.cos()], -1)
