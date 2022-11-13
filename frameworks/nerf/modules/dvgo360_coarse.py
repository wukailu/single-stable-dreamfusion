import torch
import torch.nn.functional as F

from frameworks.nerf.modules import DVGO_Coarse
from frameworks.nerf.modules.utils import MaskCache, cumprod_exclusive


class DVGO360_Coarse(DVGO_Coarse):
    """
    Distance mapping for 360 scenes. Convert world coordinates to contrast coordinates.
    For each dimension, map x -> x if center - r <= x <= center + r else center + sgn(x) * r * (1 + bg_dis - bg_dis/x)
    And center = (xyz_min + xyz_max) / 2, r = (xyz_max - xyz_min) / 2.
    """

    ######### training config setup ######
    def _set_grid_resolution(self, num_voxels):
        self.bg_dis = self.cfg_model.bg_dis
        # Determine grid resolution
        self.num_voxels = num_voxels
        total_volume = ((self.xyz_max - self.xyz_min) * (1 + self.bg_dis)).prod()
        voxel_size = (total_volume / num_voxels).pow(1 / 3)
        voxel_size_base = (total_volume / self.num_voxels_base).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) * (1 + self.bg_dis) / voxel_size).long()
        self.voxel_size_ratio = voxel_size / voxel_size_base
        self.voxel_size = voxel_size
        self.voxel_size_each = (self.xyz_max - self.xyz_min) * (1 + self.bg_dis) / (self.world_size - 1)
        print('voxel_size      ', self.voxel_size)
        print('world_size      ', self.world_size)
        print('voxel_size_ratio', self.voxel_size_ratio)

    def get_center_r(self):
        return (self.xyz_min + self.xyz_max) / 2, (self.xyz_max - self.xyz_min) / 2

    # individual dim mapping
    def world_to_contrast(self, xyz: torch.Tensor):
        device = xyz.device
        xyz = xyz.to(self.xyz_min)
        center, r = self.get_center_r()
        inner_mask = (self.xyz_min <= xyz) & (xyz <= self.xyz_max)
        relative = (1 + self.bg_dis - self.bg_dis / ((xyz - center) / r).abs()) * (xyz - center).sign()
        ray_pts = torch.where(inner_mask, xyz, center + r * relative)
        return ray_pts.to(device)

    def contrast_to_world(self, xyz: torch.Tensor):
        center, r = self.get_center_r()
        inner_mask = (self.xyz_min <= xyz) & (xyz <= self.xyz_max)
        relative = (xyz - center) / r
        ray_pts = torch.where(inner_mask, xyz,
                              center + r * relative.sign() * (self.bg_dis / (1 + self.bg_dis - relative.abs())))
        return ray_pts

    def grid_sampler(self, xyz, grid, mode='bilinear', align_corners=True):
        """
        Wrapper for the interp operation, map to [-1, 1]
        """
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        center, r = self.get_center_r()
        xyz = (xyz - center) / (r * (1 + self.bg_dis))
        ind_norm = xyz.flip((-1,))
        assert (-1-1e-9 <= ind_norm).all() and (ind_norm <= 1+1e-9).all()
        return F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1], -1).T. \
            reshape(*shape, grid.shape[1])

    def _voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1):
        print('dvgo: voxel_count_views start')
        count = torch.zeros_like(self.density.detach())
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density).requires_grad_()
            for rays_o, rays_d in zip(rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts = self.sample_ray(rays_o=rays_o.to(self.xyz_max), rays_d=rays_d.to(self.xyz_max))
                self.grid_sampler(ray_pts, ones).sum().backward()
            count.data += (ones.grad > 1)
        return count

    def _voxel_count_views_depth(self, rays_o_tr, rays_d_tr, rays_depth, stepsize, downrate=1):
        print('dvgo: voxel_count_views with depth start')
        count = torch.zeros_like(self.density.detach())

        rays_o_, rays_d_, depth_ = rays_o_tr.to(self.xyz_max), rays_d_tr.to(self.xyz_max), rays_depth.to(self.xyz_max)
        ones = torch.ones_like(self.density).requires_grad_()
        for rays_o, rays_d, depth_info in zip(rays_o_, rays_d_, depth_):
            N_samples = ((depth_info * 0.2).max() / (stepsize * self.voxel_size)).long().item() + 1
            rng = torch.arange(N_samples)[None].float().to(self.xyz_max) * stepsize * self.voxel_size
            interpx = (depth_info * 0.9)[..., None] + rng
            rays_pts = rays_o[..., None, :] + (rays_d / rays_d.norm(dim=-1, keepdim=True))[..., None, :] * interpx[
                ..., None]
            rays_pts = self.world_to_contrast(rays_pts)
            self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        # add extra points
        added_pts = self._get_boarder_points(boarder_ratio=1)
        self.grid_sampler(added_pts, ones).sum().backward()
        with torch.no_grad():
            count += (ones.grad > 1) * 2
        return count

    def _voxel_count_pcd(self, pts):
        pts = self.world_to_contrast(pts)
        print("pts max, min", pts.max(dim=0)[0], pts.min(dim=0)[0])
        # add boarder points
        added_xyz = self._get_boarder_points(boarder_ratio=1)
        print("added_xyz.shape", added_xyz.shape)
        return super()._voxel_count_pcd(torch.cat([pts, added_xyz.to(pts)], dim=0))

    def _get_boarder_points(self, boarder_ratio=1.0):
        center, r = self.get_center_r()
        real_min = center - r * (1 + self.bg_dis)
        real_max = center + r * (1 + self.bg_dis)
        bound_min = center - r * (1 + self.bg_dis * (1 - boarder_ratio))
        bound_max = center + r * (1 + self.bg_dis * (1 - boarder_ratio))
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(real_min[0], real_max[0], self.density.shape[2]),
            torch.linspace(real_min[1], real_max[1], self.density.shape[3]),
            torch.linspace(real_min[2], real_max[2], self.density.shape[4]),
        ), -1).to(self.xyz_max)
        return self_grid_xyz[((self_grid_xyz < bound_min) | (self_grid_xyz > bound_max)).any(dim=-1)]

    def sample_ray(self, rays_o, rays_d):
        """
        Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rays_d:   both in [N, 3] indicating ray configurations.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [N, P, 3] storing all the sampled points.
        """
        assert len(rays_o.shape) == 2 and rays_o.shape == rays_d.shape
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # normalize
        N_inner = int(1 / (1 + self.bg_dis) * (self.world_size + 1).float().norm().cpu() / self.stepsize) + 1
        N_inner = max(N_inner, 128)
        N_outer = N_inner
        b_inner = torch.linspace(0, 2, N_inner + 1).to(rays_o)
        b_outer = 2 / torch.linspace(1, 1 / 128, N_outer + 1).to(rays_o)
        t = torch.cat([
            (b_inner[1:] + b_inner[:-1]) * 0.5,
            (b_outer[1:] + b_outer[:-1]) * 0.5,
        ])
        ray_pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None] * (self.xyz_max - self.xyz_min).norm() / 2
        ray_pts = self.world_to_contrast(ray_pts)
        return ray_pts

    ######### render part ##########
    def get_render_args(self):
        render_kwargs = {
            'near': self.near,
            'far': self.far,
            'bg': torch.tensor(self.cfg_data.bkgd).float(),
            'stepsize': self.stepsize,
            'rand_bkgd': self.cfg_data.unbounded,
        }
        return render_kwargs

    def query_alpha(self, rays_pts, interval=1):
        # post-activation
        noise = 0 if (not self.training) else torch.randn_like(self.density) * self.density_noise
        density = self.grid_sampler(rays_pts, self.density + noise)[..., 0]
        return self.activate_density(density, interval)

    def _render_core(self, mask_outbbox, rays_pts, viewdirs, interval=1):
        alpha = torch.zeros_like(rays_pts[..., 0])
        interval = interval if isinstance(interval, torch.Tensor) else torch.ones_like(alpha) * interval
        alpha[~mask_outbbox] = self.query_alpha(rays_pts[~mask_outbbox], interval[~mask_outbbox])
        # compute accumulated transmittance
        alphainv_cum = cumprod_exclusive(1 - alpha)
        weights = alpha * alphainv_cum[..., :-1]
        # query for color
        mask = (weights > self.fast_color_thres)
        rgb = torch.ones(*weights.shape, 3).to(weights) * 0.5  # 0.5 as default value
        rgb[mask] = self.query_rgb(rays_pts[mask], viewdirs[:, None, :].expand_as(rays_pts)[mask])
        return alpha, alphainv_cum, rgb, weights

    def render(self, rays_o, rays_d, viewdirs, **render_kwargs):
        assert 'rand_bkgd' in render_kwargs
        assert 'bg' in render_kwargs
        """Volume rendering"""
        ret_dict = {}

        # sample points on rays
        rays_pts = self.sample_ray(rays_o=rays_o, rays_d=rays_d)
        ori_ray = self.contrast_to_world(rays_pts)
        interval = torch.cat(
            [(ori_ray[:, 1:] - ori_ray[:, :-1]).norm(dim=-1), torch.ones_like(ori_ray[:, [0], 0]) * 1e6], dim=1)

        # take away query points in known free space
        mask_out = torch.zeros_like(rays_pts[..., 0]).bool()
        if self.use_mask_cache:
            mask_out[~mask_out] |= (~self.mask_cache(rays_pts[~mask_out]))

        alpha, alphainv_cum, rgb, weights = self._render_core(mask_out, rays_pts, viewdirs, interval)

        # Ray marching
        if render_kwargs['rand_bkgd']:
            bkgd = torch.rand(3).to(rgb)
        else:
            bkgd = render_kwargs['bg'].to(rgb)
        rgb_marched = (weights[..., None] * rgb).sum(-2) + alphainv_cum[..., [-1]] * bkgd
        dists = (rays_o[..., None, :] - ori_ray).norm(dim=-1)
        depth = (weights * dists).sum(-1)
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

    ######################################################################

    def generate_maskCache(self, mask_cache_thres):
        center, r = self.get_center_r()
        return MaskCache(center - r * (1 + self.bg_dis), center + r * (1 + self.bg_dis), self.density.data,
                         self.act_shift, 1, mask_cache_thres)

    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        center, r = self.get_center_r()
        real_min = center - r * (1 + self.bg_dis)
        real_max = center + r * (1 + self.bg_dis)
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(real_min[0], real_max[0], self.density.shape[2]),
            torch.linspace(real_min[1], real_max[1], self.density.shape[3]),
            torch.linspace(real_min[2], real_max[2], self.density.shape[4]),
        ), -1).to(self.xyz_max)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None, None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        with torch.no_grad():
            self.density[~self.nonempty_mask] = -100
            print("nonempty_mask empty ratio", 1 - self.nonempty_mask.float().mean())
