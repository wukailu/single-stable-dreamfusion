import torch

from .utils import get_ray_marching_ray, position_encoding
from .dvgo_fine import DVGO_Fine
from frameworks.nerf.decoders.mlps import get_MLP


class FFL(DVGO_Fine):

    def get_rgbnet(self, cfg_model):  # FFL_MLP
        return get_MLP(cfg_model.rgbnet, in_dim=self.dim0, out_dim=4, width=cfg_model.rgbnet_width,
                       depth=cfg_model.rgbnet_depth, viewdir_dim=len(self.viewfreq) * 6 + 3)

    def fine_forward(self, alpha, rays_pts, viewdirs):
        mask: torch.Tensor = (alpha > 1e-4)  # [rays, pts]
        # render video use
        if not mask.any():
            rgb = torch.zeros(*alpha.shape, 3).to(alpha)
            alpha_pred = torch.zeros(*alpha.shape).to(alpha)
            weights, alphainv_cum = get_ray_marching_ray(alpha * alpha_pred)
            return alpha * alpha_pred, alphainv_cum, rgb, weights

        # rays_pts: [rays, pts, 3]
        cell = ((rays_pts - self.xyz_min) / self.voxel_size_each).floor()
        cell[cell == (self.world_size.to(cell) - 1)] -= 1  # solve the case that equal to xyz_max
        assert (cell[mask] >= 0).all()
        assert (cell[mask] < self.world_size.to(cell)).all()
        cells = cell[..., None, :] + CELL_BASE.to(cell)  # [rays, samples, 8, 3]
        relatives = (rays_pts[..., None, :] - self.xyz_min) - cells * self.voxel_size_each
        LIIF_weights = relatives[mask].prod(dim=-1).flip(
            dims=[1]).abs() / self.voxel_size_each.prod()  # [query_pts, 8]
        assert (LIIF_weights.sum(dim=-1) < 1.001).all()
        assert (LIIF_weights.sum(dim=-1) > 0.999).all()
        ######################################################################
        indexes = cells[mask].long().reshape(-1, 3)
        # find corresponding k0
        k0s = self.k0[:, :, indexes[:, 0], indexes[:, 1], indexes[:, 2]].transpose(-1, -2).reshape(
            (*cells[mask].shape[:-1], -1))  # [query_pts, 8, feat]
        viewdirs = viewdirs[:, None, None, :].expand(
            (*rays_pts.shape[:-1], 8, 3))  # [rays, samples, 8, 3]
        viewdirs_emb = position_encoding(viewdirs[mask], self.viewfreq)  # [query_pts, 8, emb]
        xyz_emb = position_encoding(relatives[mask], self.posfreq)  # shape of [query_pts 8, emb]
        rgb_feat = torch.cat([k0s, xyz_emb, viewdirs_emb], -1)
        rgb = torch.zeros(*alpha.shape, 3).to(alpha)
        alpha_pred = torch.zeros(*alpha.shape).to(alpha)
        pred = self.rgbnet(rgb_feat.flatten(0, -2)).reshape((*rgb_feat.shape[:-1], -1))  # [query_pts, 8, dims]
        rgb[mask] = (torch.sigmoid(pred[..., :3]) * LIIF_weights[..., None]).sum(dim=-2)
        alpha_pred[mask] = (torch.sigmoid(pred[..., 3]) * LIIF_weights).sum(dim=-1)
        # compute accumulated transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha * alpha_pred)
        return alpha * alpha_pred, alphainv_cum, rgb, weights

    def _render_core(self, mask_outbbox, rays_pts, viewdirs):
        alpha = torch.zeros_like(rays_pts[..., 0])
        alpha[~mask_outbbox] = self.query_alpha(rays_pts[~mask_outbbox])
        return self.fine_forward(alpha, rays_pts, viewdirs)


# 支持选用 view Dependent Emission
# Feature 部分每个位置的预测同LIIF 一样先预测再插值
# Density 部分区分为整体密度与局部密度的乘积， density = density_voxel * sigmoid(f_liif(density_feature)['dens'])
# f_liif 同时以 LIIF 的形式编码密度和颜色
# 所以这里i和8-i必须对应

CELL_BASE = torch.tensor([
    [0, 0, 1],
    [0, 0, 0],
    [0, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 0, 0],
    [1, 1, 1],
    [1, 1, 0],
]).float()


