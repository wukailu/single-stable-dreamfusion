import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from datasets.nerf.utils import sample_ray
from frameworks.nerf.decoders import get_MLP
from frameworks.nerf.modules import DVGO_Fine
from frameworks.nerf.modules.utils import cumprod_exclusive, position_encoding, metric_loss


class OSR_Fine(DVGO_Fine):
    def init_parameters(self, cfg_model):
        super(OSR_Fine, self).init_parameters(cfg_model)
        self.env = torch.nn.Parameter(torch.randn([9, 3]))
        self.use_shadow_jitter = True
        self.use_shadow = True

    def get_rgbnet(self, cfg_model):
        return get_MLP(cfg_model.rgbnet, in_dim=cfg_model.rgbnet_dim + 9, out_dim=4, width=cfg_model.rgbnet_width,
                       depth=cfg_model.rgbnet_depth, k0_dim=cfg_model.rgbnet_dim)

    def grid_sampler_with_grad(self, xyz, grid):
        '''Wrapper for the interp operation'''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        return grid_sample_3d(grid, ind_norm).reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1])

    def query_alpha_with_grad(self, rays_pts):
        interval = self.stepsize * self.voxel_size_ratio
        density = self.grid_sampler_with_grad(rays_pts, self.density)[..., 0]
        return self.activate_density(density, interval)

    def query_rgb(self, rays_pts, viewdirs):
        if len(rays_pts) == 0:
            return 0, 0, 0
        env_gray = self.env[..., 0] * 0.2126 + self.env[..., 1] * 0.7152 + self.env[..., 2] * 0.0722
        fg_sph = env_gray.reshape((1, -1)).expand([rays_pts.size(0), 9])
        if self.use_shadow_jitter:
            fg_sph = fg_sph + torch.randn_like(fg_sph) * 0.01

        # query normal
        with torch.enable_grad():
            query_pts = rays_pts.detach()
            query_pts.requires_grad_(True)
            key_alpha = self.query_alpha_with_grad(query_pts)
            normal_map = torch.autograd.grad(
                outputs=key_alpha, inputs=query_pts,
                grad_outputs=torch.ones_like(key_alpha, requires_grad=False),
                retain_graph=True, create_graph=True)[0]
        # query rgb and shadow
        k0_pts = self.grid_sampler(rays_pts, self.k0)
        output = self.rgbnet(torch.cat([k0_pts, fg_sph], dim=-1))
        rgb, shadow = torch.sigmoid(output[..., :3]), torch.sigmoid(output[..., -1:])
        return rgb, shadow, normal_map

    def render(self, rays_o, rays_d, viewdirs, **render_kwargs):
        """Volume rendering"""
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=self.training,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max, voxel_size=self.voxel_size,
            world_size=self.world_size, **render_kwargs)

        # update mask for query points in known free space
        if self.use_mask_cache:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))

        alpha, alphainv_cum, rgb, weights, shadow, normal = self._render_core(mask_outbbox, rays_pts, viewdirs)

        # Ray marching
        rgb_albedo_marched = (weights[..., None] * rgb).sum(-2)
        normal_marched = F.normalize((weights[..., None] * normal).sum(-2), dim=-1)
        shadow_marched = (weights[..., None] * shadow).sum(-2)

        irradiance = illuminate_vec(normal_marched, self.env)
        irradiance = torch.relu(irradiance)  # can't be < 0
        irradiance = irradiance ** (1 / 2.2)  # linear to srgb
        if self.use_shadow:
            rgb_marched = irradiance * rgb_albedo_marched * shadow_marched \
                          + alphainv_cum[..., [-1]] * render_kwargs['bg'].to(rgb)
        else:
            rgb_marched = rgb_albedo_marched + alphainv_cum[..., [-1]] * render_kwargs['bg'].to(rgb)

        dists = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        depth = (weights * dists).sum(-1) + alphainv_cum[..., -1] * render_kwargs['far']
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'irradiance': irradiance,
            'shadow_marched': shadow_marched,
            'normal_marched': normal_marched,
            'raw_rgb': rgb,
            'depths': depth,
            'dists': dists,
        })
        return ret_dict

    def _render_core(self, mask_outbbox, rays_pts, viewdirs):
        alpha = torch.zeros_like(rays_pts[..., 0])
        alpha[~mask_outbbox] = self.query_alpha(rays_pts[~mask_outbbox])
        # compute accumulated transmittance
        alphainv_cum = cumprod_exclusive(1 - alpha)
        weights = alpha * alphainv_cum[..., :-1]
        mask = (weights > self.fast_color_thres)
        # query for color
        rgb = torch.ones(*weights.shape, 3).to(weights) * 0.5  # 0.5 as default value
        normal = torch.ones(*weights.shape, 3).to(weights) * ((1 / 3) ** 0.5)
        shadow = torch.ones(*weights.shape, 1).to(weights)
        rgb[mask], shadow[mask], normal[mask] = self.query_rgb(rays_pts[mask],
                                                               viewdirs[:, None, :].expand_as(rays_pts)[mask])
        normal = F.normalize(normal, dim=-1)
        return alpha, alphainv_cum, rgb, weights, shadow, normal


# point-wise shadow and irradiance
class OSR_Fine_V2(OSR_Fine):
    def query_rgb(self, rays_pts, viewdirs):
        if len(rays_pts) == 0:
            return 0, 0, 0
        env_gray = self.env[..., 0] * 0.2126 + self.env[..., 1] * 0.7152 + self.env[..., 2] * 0.0722
        fg_sph = env_gray.reshape((1, -1)).expand([rays_pts.size(0), 9])
        if self.use_shadow_jitter:
            fg_sph = fg_sph + torch.randn_like(fg_sph) * 0.01

        # query normal
        with torch.enable_grad():
            query_pts = rays_pts.detach()
            query_pts.requires_grad_(True)
            key_alpha = self.query_alpha_with_grad(query_pts)
            normal_map = torch.autograd.grad(
                outputs=key_alpha, inputs=query_pts,
                grad_outputs=torch.ones_like(key_alpha, requires_grad=False),
                retain_graph=True, create_graph=True)[0]
        # query rgb and shadow
        k0_pts = self.grid_sampler(rays_pts, self.k0)
        output = self.rgbnet(torch.cat([k0_pts, fg_sph], dim=-1))
        rgb, shadow = torch.sigmoid(output[..., :3]), torch.sigmoid(output[..., -1:])

        normal_map = F.normalize(normal_map, dim=-1)
        irradiance = illuminate_vec(normal_map, self.env)
        irradiance = torch.relu(irradiance)  # can't be < 0
        irradiance = irradiance ** (1 / 2.2)  # linear to srgb
        if self.use_shadow:
            rgb = irradiance * rgb * shadow
        return rgb, shadow, normal_map

    def render(self, rays_o, rays_d, viewdirs, **render_kwargs):
        """Volume rendering"""
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=self.training,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max, voxel_size=self.voxel_size,
            world_size=self.world_size, **render_kwargs)

        # update mask for query points in known free space
        if self.use_mask_cache:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))

        alpha, alphainv_cum, rgb, weights, shadow, normal = self._render_core(mask_outbbox, rays_pts, viewdirs)

        # Ray marching
        rgb_marched = (weights[..., None] * rgb).sum(-2) + alphainv_cum[..., [-1]] * render_kwargs['bg'].to(rgb)
        normal_marched = F.normalize((weights[..., None] * normal).sum(-2), dim=-1)
        shadow_marched = (weights[..., None] * shadow).sum(-2)

        dists = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        depth = (weights * dists).sum(-1) + alphainv_cum[..., -1] * render_kwargs['far']
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'shadow_marched': shadow_marched,
            'normal_marched': normal_marched,
            'raw_rgb': rgb,
            'depths': depth,
            'dists': dists,
        })
        return ret_dict


# only apply shadow prediction
# color still depends on view direction
class OSR_Fine_V3(DVGO_Fine):
    use_shadow = True

    def get_rgbnet(self, cfg_model):
        return get_MLP(cfg_model.rgbnet, in_dim=self.dim0, out_dim=4, width=cfg_model.rgbnet_width,
                       depth=cfg_model.rgbnet_depth, k0_dim=cfg_model.rgbnet_dim, shadow_dim=cfg_model.rgbnet_shadow)

    def _render_core(self, mask_outbbox, rays_pts, viewdirs):
        alpha = torch.zeros_like(rays_pts[..., 0])
        alpha[~mask_outbbox] = self.query_alpha(rays_pts[~mask_outbbox])
        # compute accumulated transmittance
        alphainv_cum = cumprod_exclusive(1 - alpha)
        weights = alpha * alphainv_cum[..., :-1]
        # query for color
        mask = (weights > self.fast_color_thres)
        rgb = torch.ones(*weights.shape, 3).to(weights) * 0.5  # 0.5 as default value
        shadow = torch.ones(*weights.shape, 1).to(weights)  # 1 as default value
        rgb[mask], shadow[mask] = self.query_rgb(rays_pts[mask], viewdirs[:, None, :].expand_as(rays_pts)[mask])
        return alpha, alphainv_cum, rgb, shadow, weights

    def render(self, rays_o, rays_d, viewdirs, **render_kwargs):
        """Volume rendering"""
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=self.training,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max, voxel_size=self.voxel_size,
            world_size=self.world_size, **render_kwargs)

        # update mask for query points in known free space
        if self.use_mask_cache:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))

        alpha, alphainv_cum, rgb, shadow, weights = self._render_core(mask_outbbox, rays_pts, viewdirs)

        # Ray marching
        rgb_marched = (weights[..., None] * rgb * shadow).sum(-2) + alphainv_cum[..., [-1]] * render_kwargs['bg'].to(rgb).clamp(
            0, 1)
        dists = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        depth = (weights * dists).sum(-1) + alphainv_cum[..., -1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'raw_shadow': shadow,
            'depths': depth,
            'disp': disp,
            'dists': dists,
        })
        return ret_dict

    def query_rgb(self, rays_pts, viewdirs, use_shadow=True, calc_loss=True):
        # view-dependent color emission
        if len(rays_pts) == 0:
            return 0, 1
        rays_xyz = (rays_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        k0_view = [self.query_k0(rays_pts)]
        viewdirs_emb = [position_encoding(viewdirs, self.viewfreq)] if hasattr(self, 'viewfreq') else []
        xyz_emb = [position_encoding(rays_xyz, self.posfreq)] if hasattr(self, 'posfreq') else []
        rgb_feat = torch.cat(k0_view + xyz_emb + viewdirs_emb, -1)
        output = self.rgbnet(rgb_feat)
        rgb, shadow = torch.sigmoid(output[..., :3]), torch.sigmoid(output[..., -1:])
        if calc_loss:
            self.ex_reg_loss(rays_pts, viewdirs, rgb, shadow)
        if self.use_shadow and use_shadow:
            shadow = (self.cfg_train.shadow_bar + (1 - self.cfg_train.shadow_bar) * shadow)
            return rgb, shadow
        else:
            return rgb, 1

    def query_k0(self, rays_pts):
        k0_view = self.grid_sampler(rays_pts, self.k0)
        return k0_view

    # this should be called only once, per training inference
    def ex_reg_loss(self, rays_pts, viewdirs, rgb, shadow):
        if not self.training:
            return
        self.ex_loss = 0
        if self.cfg_train.weight_metric_k0_v2 > 0:
            # 每个查询点，再去查询附近随机一个点，然后期望颜色相同
            rand_perturbation = (torch.rand_like(rays_pts)-0.5) * (self.xyz_max - self.xyz_min) * self.cfg_train.metric_k0_dis
            near_rgb, _ = self.query_rgb(rays_pts + rand_perturbation, viewdirs, use_shadow=False, calc_loss=False)
            perm = torch.randperm(len(rgb))
            metric_loss = F.mse_loss(rgb, near_rgb) - F.mse_loss(rgb, near_rgb[perm])
            self.ex_loss += self.cfg_train.weight_metric_k0_v2 * metric_loss
        if self.cfg_train.weight_metric_k0_v3 > 0:
            # 每个查询点，再去查询附近随机一个点，然后期望 feature 相同
            rand_perturbation = (torch.rand_like(rays_pts)-0.5) * (self.xyz_max - self.xyz_min) * self.cfg_train.metric_k0_dis
            rand_pts = rays_pts + rand_perturbation
            ori_k0 = self.query_k0(rays_pts)
            near_k0 = self.query_k0(rand_pts)
            perm = torch.randperm(len(ori_k0))
            metric_loss = F.l1_loss(ori_k0, near_k0) - F.l1_loss(ori_k0, near_k0[perm])
            self.ex_loss += self.cfg_train.weight_metric_k0_v3 * metric_loss
        if self.cfg_train.weight_metric_k0_v4 > 0:
            # apply to rgb only
            rand_perturbation = (torch.rand_like(rays_pts)-0.5) * (self.xyz_max - self.xyz_min) * self.cfg_train.metric_k0_dis
            rand_pts = rays_pts + rand_perturbation
            ori_k0 = self.query_k0(rays_pts)[self.cfg_model.rgbnet_shadow:]
            near_k0 = self.query_k0(rand_pts)[self.cfg_model.rgbnet_shadow:]
            perm = torch.randperm(len(ori_k0))
            metric_loss = F.l1_loss(ori_k0, near_k0) - F.l1_loss(ori_k0, near_k0[perm])
            self.ex_loss += self.cfg_train.weight_metric_k0_v4 * metric_loss
        if self.cfg_train.weight_metric_k0_v5 > 0:
            # apply to color feature and tune distance
            rand_perturbation = (torch.rand_like(rays_pts)-0.5) * (self.xyz_max - self.xyz_min) * self.cfg_train.metric_k0_dis
            rand_pts = rays_pts + rand_perturbation
            ori_k0 = self.query_k0(rays_pts)[self.cfg_model.rgbnet_shadow:]
            near_k0 = self.query_k0(rand_pts)[self.cfg_model.rgbnet_shadow:]
            perm = torch.randperm(len(ori_k0))
            metric_loss = (ori_k0 - near_k0).abs().mean() - (ori_k0 - near_k0[perm]).abs().mean()
            self.ex_loss += self.cfg_train.weight_metric_k0_v5 * metric_loss
        if self.cfg_train.more_shadow > 0:
            self.ex_loss += -self.cfg_train.more_shadow * F.mse_loss(shadow, torch.ones_like(shadow))
        if self.cfg_train.more_shadow_l1 > 0:
            self.ex_loss += -self.cfg_train.more_shadow_l1 * F.l1_loss(shadow, torch.ones_like(shadow))
        if self.cfg_train.reg_shadow > 0:
            self.ex_loss += self.cfg_train.reg_shadow * F.mse_loss(shadow, torch.ones_like(shadow))
        if self.cfg_train.reg_shadow_l1 > 0:
            self.ex_loss += self.cfg_train.reg_shadow * F.l1_loss(shadow, torch.ones_like(shadow))

    def _compute_loss(self, render_result, target, target_depth):
        loss = super(OSR_Fine_V3, self)._compute_loss(render_result, target, target_depth)
        if hasattr(self, "ex_loss"):
            return loss + self.ex_loss
        if self.cfg_train.weight_rgbper_v2 > 0:  # 鼓励颜色内外一致性
            weights_sum = render_result['weights'].detach().sum(-1)[..., None]
            rgb_target = ((render_result['raw_rgb'].detach() * (render_result['weights'].detach()[..., None])).sum(-2)) / weights_sum
            rgbper = (render_result['raw_rgb'] - rgb_target[:, None, :]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss += self.cfg_train.weight_rgbper_v2 * rgbper_loss
        return loss


# dvp version
class OSR_Fine_V4(OSR_Fine_V3):
    def query_k0(self, rays_pts):
        k0_view = torch.sigmoid(self.grid_sampler(rays_pts, self.k0))
        return k0_view

    def _k0_metric_loss(self):
        v = torch.sigmoid(self.k0)
        return metric_loss(v, self.nonempty_mask)


# V5 metric loss on rgb only
class OSR_Fine_V5(OSR_Fine_V3):
    def query_k0(self, rays_pts):
        k0_view = torch.sigmoid(self.grid_sampler(rays_pts, self.k0))
        return k0_view

    def _k0_metric_loss(self):
        v = torch.sigmoid(self.k0[:, self.cfg_model.rgbnet_shadow:])
        return metric_loss(v, self.nonempty_mask)


# no better than v5
class OSR_Fine_V6(OSR_Fine_V4):
    def init_parameters(self, cfg_model):
        super(OSR_Fine_V6, self).init_parameters(cfg_model)
        self.k0 = torch.nn.Parameter(torch.zeros_like(self.k0.data))

    def _k0_metric_loss(self):
        v = self.k0[:, self.cfg_model.rgbnet_shadow:]
        return metric_loss(v, self.nonempty_mask)


# RGI format, R = r * I * 3, G = g * I * 3, B = (1-r-g) * I * 3 where 0 <= r+g <= 1, 0<= r, g <=1, 0<= I <= 1
class OSR_Fine_RGI(DVGO_Fine):
    use_shadow = True
    base_illuminance = 0.5

    def get_rgbnet(self, cfg_model):
        return get_MLP(cfg_model.rgbnet, in_dim=self.dim0, out_dim=3, width=cfg_model.rgbnet_width,
                       depth=cfg_model.rgbnet_depth, k0_dim=cfg_model.rgbnet_dim, shadow_dim=cfg_model.rgbnet_shadow)

    def query_k0(self, rays_pts):
        k0_view = torch.sigmoid(self.grid_sampler(rays_pts, self.k0))
        return k0_view

    def _k0_metric_loss(self):
        v = torch.sigmoid(self.k0[:, self.cfg_model.rgbnet_shadow:])
        return metric_loss(v, self.nonempty_mask)

    def _render_core(self, mask_outbbox, rays_pts, viewdirs):
        alpha = torch.zeros_like(rays_pts[..., 0])
        alpha[~mask_outbbox] = self.query_alpha(rays_pts[~mask_outbbox])
        # compute accumulated transmittance
        alphainv_cum = cumprod_exclusive(1 - alpha)
        weights = alpha * alphainv_cum[..., :-1]
        # query for color
        mask = (weights > self.fast_color_thres)
        rg = torch.zeros(*weights.shape, 2).to(weights)  # 0 as default value
        illuminance = torch.zeros(*weights.shape, 1).to(weights)  # 0 as default value
        rg[mask], illuminance[mask] = self.query_rgb(rays_pts[mask], viewdirs[:, None, :].expand_as(rays_pts)[mask])
        return alpha, alphainv_cum, rg, illuminance, weights

    def render(self, rays_o, rays_d, viewdirs, **render_kwargs):
        """Volume rendering"""
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=self.training,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max, voxel_size=self.voxel_size,
            world_size=self.world_size, **render_kwargs)

        # update mask for query points in known free space
        if self.use_mask_cache:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))

        alpha, alphainv_cum, rg, illuminance, weights = self._render_core(mask_outbbox, rays_pts, viewdirs)

        # initial bias

        illuminance = torch.nn.functional.relu(illuminance + 0.5)
        rg += torch.nn.functional.relu(rg + 1/3)

        # conversion
        rgb_converted = torch.zeros_like(rays_pts)
        # print('rg shape', rg.shape, 'rgb shape', rgb_converted.shape, 'illuminance shape', illuminance.shape)
        rgb_converted[..., 0] = rg[..., 0] * illuminance[..., 0] * 3
        rgb_converted[..., 1] = rg[..., 1] * illuminance[..., 0] * 3
        rgb_converted[..., 2] = torch.nn.functional.relu(1 - rg[..., 0] - rg[..., 1]) * illuminance[..., 0] * 3

        # Ray marching
        rgb_marched = (weights[..., None] * rgb_converted).sum(-2) + alphainv_cum[..., [-1]] * render_kwargs['bg'].to(rg).clamp(
            0, 1)
        dists = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        depth = (weights * dists).sum(-1) + alphainv_cum[..., -1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rg': rg,
            'raw_shadow': illuminance,
            'depths': depth,
            'disp': disp,
            'dists': dists,
        })
        return ret_dict

    def query_rgb(self, rays_pts, viewdirs, use_shadow=True):
        # view-dependent color emission
        if len(rays_pts) == 0:
            return 0, 1
        rays_xyz = (rays_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        k0_view = [self.query_k0(rays_pts)]
        viewdirs_emb = [position_encoding(viewdirs, self.viewfreq)] if hasattr(self, 'viewfreq') else []
        xyz_emb = [position_encoding(rays_xyz, self.posfreq)] if hasattr(self, 'posfreq') else []
        rgb_feat = torch.cat(k0_view + xyz_emb + viewdirs_emb, -1)
        output = self.rgbnet(rgb_feat)
        rg = output[..., :2]
        illuminance = torch.clamp(output[..., -1:], 0, 1) * (1-self.cfg_train.shadow_bar) + self.cfg_train.shadow_bar
        if self.use_shadow and use_shadow:
            return rg, illuminance
        else:
            return rg, self.base_illuminance

    def _compute_loss(self, render_result, target, target_depth):
        loss = super(OSR_Fine_RGI, self)._compute_loss(render_result, target, target_depth)
        if self.cfg_train.more_shadow != 0:
            loss += -self.cfg_train.more_shadow * F.mse_loss(render_result['raw_shadow'], torch.ones_like(render_result['raw_shadow']))
        if self.cfg_train.weight_rgbper_v3 > 0:  # 鼓励颜色内外一致性
            I3_target = target.sum(dim=-1, keepdim=True) + 1e-6
            rg_target = target[..., :2] / I3_target
            rgbper = (render_result['raw_rg'] - rg_target.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
            loss += self.cfg_train.weight_rgbper * rgbper_loss
        return loss


class AnnealingPosEmbedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos),
                 N_anneal=100000, N_anneal_min_freq=0,
                 use_annealing=True):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.use_annealing = use_annealing

        self.N_anneal = N_anneal
        self.N_anneal_min_freq = N_anneal_min_freq

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, iteration):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        alpha = (len(self.freq_bands) - self.N_anneal_min_freq) * iteration / self.N_anneal
        for i in range(len(self.freq_bands)):
            w = (1 - np.cos(np.pi * np.clip(alpha - i + self.N_anneal_min_freq, 0, 1))) / 2.

            if not self.use_annealing:
                w = 1

            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq) * w)
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def illuminate_vec(n, env):  # sh-coef
    c1 = 0.282095
    c2 = 0.488603
    c3 = 1.092548
    c4 = 0.315392
    c5 = 0.546274

    c = env.unsqueeze(1)
    x, y, z = n[..., 0, None], n[..., 1, None], n[..., 2, None]

    irradiance = (
            c[0] * c1 +
            c[1] * c2 * y +
            c[2] * c2 * z +
            c[3] * c2 * x +
            c[4] * c3 * x * y +
            c[5] * c3 * y * z +
            c[6] * c4 * (3 * z * z - 1) +
            c[7] * c3 * x * z +
            c[8] * c5 * (x * x - y * y)
    )
    return irradiance


# this one has gradiant
# code from https://github.com/pytorch/pytorch/issues/34704
def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():
        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2,
                           (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2,
                           (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2,
                           (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2,
                           (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2,
                           (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2,
                           (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2,
                           (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2,
                           (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val