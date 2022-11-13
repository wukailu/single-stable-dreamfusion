import time
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict

from datasets.nerf.utils import sample_ray
from datasets.nerf.nerf_dataset import NeRFData
from .lightning_base import NeRFModule
from .utils import MaskCache, cumprod_exclusive, total_variation, metric_loss, per_voxel_init


class DVGO_Coarse(NeRFModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        # create model
        self.cfg_data, self.cfg_model, self.cfg_train = self.parse_config(EasyDict(self.params['cfg']))
        # init model
        self.num_voxels = self.num_voxels_base = self.cfg_model.num_voxels
        self.fast_color_thres = self.cfg_model.fast_color_thres
        self.density_noise = self.cfg_model.density_noise
        self.act_shift = np.log(1 / (1 - self.cfg_model.alpha_init) - 1)
        self.near, self.far = self.params['near'], self.params['far']
        self.stepsize = self.cfg_model.stepsize

        # set parameters
        self.register_buffer('xyz_min', torch.tensor(self.params['xyz_min']).float())
        self.register_buffer('xyz_max', torch.tensor(self.params['xyz_max']).float())
        self.register_buffer('voxel_size_each', torch.ones(3))
        self.register_buffer('voxel_size_ratio', torch.tensor(1))
        self.register_buffer('voxel_size', torch.tensor(1))
        self.register_buffer('world_size', torch.ones(3).long())

        self._set_grid_resolution(self.num_voxels_base)

        self.density, self.k0, self.rgbnet = None, None, None
        self.started_training = False
        self.my_train_dataset = None
        self.use_mask_cache = False
        self.nonempty_mask = None
        self.init_parameters(self.cfg_model)

    ######### training config setup ######
    def init_parameters(self, cfg_model):
        self.density = torch.nn.Parameter(torch.randn([1, 1, *self.world_size]))
        self.k0 = torch.nn.Parameter(torch.randn([1, 3, *self.world_size]))

    def parse_config(self, cfg):
        return cfg.data, cfg.coarse_model_and_render, cfg.coarse_train

    def setup_dataset(self, nerfData: NeRFData):
        self.my_train_dataset = nerfData

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / voxel_size).long()
        self.voxel_size_ratio = voxel_size / voxel_size_base
        self.voxel_size = voxel_size
        self.voxel_size_each = (self.xyz_max - self.xyz_min) / (self.world_size - 1)
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def grid_sampler(self, xyz, grid, mode='bilinear', align_corners=True):
        '''Wrapper for the interp operation'''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        return F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1], -1).T.\
            reshape(*shape, grid.shape[1])

    def _voxel_count_views_depth(self, rays_o_tr, rays_d_tr, rays_depth, stepsize, downrate=1):
        print('dvgo: voxel_count_views with depth start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.detach())

        rays_o_, rays_d_, depth_ = rays_o_tr.to(self.xyz_max), rays_d_tr.to(self.xyz_max), rays_depth.to(self.xyz_max)
        ones = torch.ones_like(self.density).requires_grad_()
        block_size = len(rays_o_tr) // 100
        # if irregular_shape:
        if len(rays_o_.shape) == 2:
            rays_o_ = rays_o_.split(block_size)
            rays_d_ = rays_d_.split(block_size)
            depth_ = depth_.split(block_size)
        else:
            rays_o_ = rays_o_[::downrate, ::downrate].flatten(0, -2).split(block_size)
            rays_d_ = rays_d_[::downrate, ::downrate].flatten(0, -2).split(block_size)
            depth_ = depth_[::downrate, ::downrate].flatten(0, -2).split(block_size)

        for rays_o, rays_d, depth_info in zip(rays_o_, rays_d_, depth_):
            N_samples = ((depth_info * 0.2).max() / (stepsize * self.voxel_size)).long().item() + 1
            rng = torch.arange(N_samples)[None].float().to(self.xyz_max) * stepsize * self.voxel_size
            interpx = (depth_info * 0.9)[..., None] + rng
            rays_pts = rays_o[..., None, :] + (rays_d / rays_d.norm(dim=-1, keepdim=True))[..., None, :] * interpx[
                ..., None]
            self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 2) * 2
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def _voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:]) + 1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float().to(self.xyz_max)
        count = torch.zeros_like(self.density.detach())
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            rays_o_, rays_d_ = rays_o_.to(self.xyz_max), rays_d_.to(self.xyz_max)
            ones = torch.ones_like(self.density).requires_grad_()
            # if irregular_shape:
            if len(rays_o_.shape) == 2:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].flatten(0, -2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].flatten(0, -2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True))
                rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def _voxel_count_pcd(self, pts):
        count = torch.zeros_like(self.density.detach())
        for rays_pts in pts.split(100000):
            ones = torch.ones_like(self.density).requires_grad_()
            self.grid_sampler(rays_pts.to(ones), ones).sum().backward()
            with torch.no_grad():
                count += ones.grad
        count[count >= 2] = 4.2
        count[(count < 2) & (count > 1e-2)] = 2.1
        count[count <= 1e-2] = 0
        return count

    def _scale_volume_grid(self, new_num_voxels):
        ori_world_size = self.world_size
        self._set_grid_resolution(new_num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        if (ori_world_size == self.world_size).all():
            return

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        self.k0 = torch.nn.Parameter(
            F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))

    def choose_optimizer(self):
        # init optimizer
        from frameworks.nerf import utils
        optimizer = utils.create_optimizer_or_freeze_model(self, self.cfg_train)

        # view-count-based learning rate, only used in coarse training
        if self.cfg_train.pervoxel_lr and self.started_training:
            assert isinstance(self.my_train_dataset, NeRFData)
            per_voxel_init(self, self.cfg_model, self.cfg_train, self.cfg_data.load_depths, self.cfg_data.pcd_path,
                           self.near, self.far, optimizer, self.my_train_dataset)
            self.cfg_train.pervoxel_lr = False  # use only once
        return optimizer

    def _reconfigure_optimizer_and_scheduler(self, new_num_voxels):
        self._scale_volume_grid(new_num_voxels)
        self.trainer.accelerator.setup_optimizers(self.trainer)
        self.density.data.sub_(1)

    def on_train_epoch_start(self):
        # progress scaling checkpoint
        self.started_training = True
        need_rescale = (self.current_epoch == 0) or (self.current_epoch in self.cfg_train.pg_scale)
        if self.current_epoch in self.cfg_train.pg_scale:
            self.cfg_train.pg_scale = self.cfg_train.pg_scale[1:]
        if need_rescale:
            num_voxels = self.num_voxels_base // (2 ** len(self.cfg_train.pg_scale))
            self._reconfigure_optimizer_and_scheduler(num_voxels)

    ######### training part ##############
    def forward(self, rays_o, rays_d, viewdirs):
        return self.render(rays_o, rays_d, viewdirs, **self.get_render_args())

    def step(self, batch, phase: str):
        (rays_d, rays_o, viewdirs), rgb_gt = batch
        rays_d, rays_o = rays_d.to(self.xyz_min), rays_o.to(self.xyz_min)
        viewdirs, rgb_gt = viewdirs.to(self.xyz_min), rgb_gt.to(self.xyz_min)
        render_result = self.forward(rays_o, rays_d, viewdirs)
        loss = self.compute_loss(render_result, rgb_gt)
        if self.cfg_data.load_depths:
            rgb_gt = rgb_gt[..., :-1]
        for key, metric in self.metrics.items():
            self.log(phase + '/' + key, metric(render_result['rgb_marched'].detach(), rgb_gt))
        return loss

    def compute_loss(self, render_result, rgb_gt):
        if self.cfg_data.load_depths:
            rgb, depth = rgb_gt[..., :-1], rgb_gt[..., -1]
        else:
            rgb, depth = rgb_gt, None
        return self._compute_loss(render_result, rgb, depth)

    def _loss_main(self, rgb, target):
        return F.mse_loss(rgb, target)

    def _compute_loss(self, render_result, target, target_depth):
        cfg_train = self.cfg_train
        loss = cfg_train.weight_main * self._loss_main(render_result['rgb_marched'], target)
        self.log("loss_main", loss)
        if cfg_train.weight_entropy_last > 0:  # 鼓励要么打穿，要么不穿
            pout = render_result['alphainv_cum'][..., -1].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
            loss_entropy_last = cfg_train.weight_entropy_last * entropy_last_loss
            loss += loss_entropy_last
            self.log("loss_entropy_last", loss_entropy_last)
        if cfg_train.dvp_feature_entropy > 0:  # 鼓励要么打穿，要么不穿
            dist = torch.sigmoid(self.k0)
            entropy_loss = -(dist * torch.log(dist) + (1 - dist) * torch.log(1 - dist)).mean()
            loss_feature_entropy = cfg_train.dvp_feature_entropy * entropy_loss
            loss += loss_feature_entropy
            self.log("loss_feature_entropy", loss_feature_entropy)
        if cfg_train.weight_rgbper > 0:  # 鼓励颜色内外一致性
            rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss_rgbper = cfg_train.weight_rgbper * rgbper_loss
            loss += loss_rgbper
            self.log('loss_rgbper', loss_rgbper)
        if cfg_train.entropy_weight > 0:  # 希望权重分布是单峰
            loss_reg = cfg_train.entropy_weight * self._ray_entropy_loss(render_result['weights'])
            loss += loss_reg
            self.log('loss_reg', loss_reg)
        # TODO: 对 activated density 的行列纵 也添加 entropy_loss
        if cfg_train.weight_tv_density > 0:
            loss_tv_dens = cfg_train.weight_tv_density * self._density_total_variation()
            loss += loss_tv_dens
            self.log('loss_tv_dens', loss_tv_dens)
        if cfg_train.weight_tv_k0 > 0:
            loss_tv_k0 = cfg_train.weight_tv_k0 * self._k0_total_variation()
            loss += loss_tv_k0
            self.log('loss_tv_k0', loss_tv_k0)
        # metric learning
        if cfg_train.weight_metric_k0 > 0:
            loss_metric_k0 = cfg_train.weight_metric_k0 * self._k0_metric_loss()
            loss += loss_metric_k0
            self.log('loss_metric_k0', loss_metric_k0)
        if cfg_train.weight_depth > 0:
            assert target_depth is not None
            if target_depth.shape != render_result['depths'].shape:
                print(target_depth.shape, render_result['depths'].shape)
            d_loss = F.mse_loss(render_result['depths'], target_depth)
            dist_loss = render_result['weights'][render_result['dists'] < target_depth[..., None] * 0.99].mean()
            loss_depth = d_loss * cfg_train.weight_depth
            loss_dist = dist_loss * cfg_train.weight_depth
            loss += loss_depth + loss_dist
            self.log('loss_depth', loss_depth)
            self.log('loss_dist', loss_dist)
        return loss

    def _density_total_variation(self):
        tv = total_variation(self.activate_density(self.density), self.nonempty_mask)
        return tv

    def _k0_total_variation(self):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)

    def _k0_metric_loss(self):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        return metric_loss(v, self.nonempty_mask)

    def _ray_entropy_loss(self, ray_batch, sum_thres=1e-2):
        assert ray_batch.min() >= 0
        normalized_ray = ray_batch / (torch.sum(ray_batch, -1).unsqueeze(-1)+1e-10)
        entropy = - normalized_ray * torch.log2(normalized_ray + 1e-10)
        entropy_ray = torch.sum(entropy, -1)
        entropy_ray *= (ray_batch.sum(-1) > sum_thres).detach().float()  # mask out empty rays
        return entropy_ray.mean()

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.stepsize * self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)

    ######### render part ##########
    def get_render_args(self):
        render_kwargs = {
            'near': self.near,
            'far': self.far,
            'bg': torch.tensor(self.cfg_data.bkgd).float(),
            'stepsize': self.stepsize,
        }
        return render_kwargs

    def render(self, rays_o, rays_d, viewdirs, **render_kwargs):
        """Volume rendering"""
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=self.training,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max, voxel_size=self.voxel_size,
            world_size=torch.tensor(self.density.shape[-3:]), **render_kwargs)

        # update mask for query points in known free space
        if self.use_mask_cache:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))

        alpha, alphainv_cum, rgb, weights = self._render_core(mask_outbbox, rays_pts, viewdirs)

        # Ray marching
        rgb_marched = (weights[..., None] * rgb).sum(-2) + alphainv_cum[..., [-1]] * render_kwargs['bg'].to(rgb)
        dists = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        depth = (weights * dists).sum(-1) + alphainv_cum[..., -1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched.clamp(0, 1),
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depths': depth,
            'disp': disp,
            'dists': dists,
        })
        return ret_dict

    def _render_core(self, mask_outbbox, rays_pts, viewdirs):
        alpha = torch.zeros_like(rays_pts[..., 0])
        alpha[~mask_outbbox] = self.query_alpha(rays_pts[~mask_outbbox])
        # compute accumulated transmittance
        alphainv_cum = cumprod_exclusive(1 - alpha)
        weights = alpha * alphainv_cum[..., :-1]
        # query for color
        mask = (weights > self.fast_color_thres)
        rgb = torch.ones(*weights.shape, 3).to(weights) * 0.5  # 0.5 as default value
        rgb[mask] = self.query_rgb(rays_pts[mask], viewdirs[:, None, :].expand_as(rays_pts)[mask])
        return alpha, alphainv_cum, rgb, weights

    def query_rgb(self, rays_pts, viewdirs):
        return torch.sigmoid(self.grid_sampler(rays_pts, self.k0))

    def query_alpha(self, rays_pts):
        interval = self.stepsize * self.voxel_size_ratio

        # post-activation
        noise = 0 if (not self.training) else torch.randn_like(self.density) * self.density_noise
        density = self.grid_sampler(rays_pts, self.density + noise)[..., 0]
        return self.activate_density(density, interval)

    ######################################################################

    def generate_maskCache(self, mask_cache_thres):
        return MaskCache(self.xyz_min, self.xyz_max, self.density.data, self.act_shift, self.voxel_size_ratio,
                         mask_cache_thres)

    def maskCache_from_coarse(self, coarse_model):
        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        self.use_mask_cache = True
        self.mask_cache = coarse_model.generate_maskCache(self.cfg_model.mask_cache_thres)
        self._set_nonempty_mask()

    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1).to(self.xyz_max)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None, None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        with torch.no_grad():
            self.density[~self.nonempty_mask] = -100

    def scale_bound(self, scale):
        if abs(scale - 1) > 1e-9:
            xyz_shift = (self.xyz_max - self.xyz_min) * (scale - 1) / 2
            self.xyz_min -= xyz_shift
            self.xyz_max += xyz_shift