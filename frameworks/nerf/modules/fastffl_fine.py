import torch

from frameworks.nerf.modules.utils import get_ray_marching_ray, position_encoding
from frameworks.nerf.modules.ffl_fine import FFL, CELL_BASE


class FastFFL(FFL):
    def fine_forward(self, alpha, rays_pts, viewdirs):
        mask: torch.Tensor = (alpha > 1e-4)  # [rays, pts]
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
        ########################################################################
        # probabilistic way
        choices = torch.searchsorted(LIIF_weights.cumsum(dim=1) + 1e-3,
                                     torch.rand((LIIF_weights.size(0), 1)).to(LIIF_weights))[:, 0]
        assert (choices.max() < 8)
        indexes = cells[mask].long()[torch.arange(len(choices)), choices]  # [query_pts, 3]
        # find corresponding k0
        k0s = self.k0[0, :, indexes[:, 0], indexes[:, 1], indexes[:, 2]].transpose(-1, -2)  # [query_pts, feat]
        viewdirs = viewdirs[:, None, :].expand((*rays_pts.shape[:-1], 3))  # [rays, samples, 3]
        viewdirs_emb = position_encoding(viewdirs[mask], self.viewfreq)  # [query_pts, emb]
        xyz_emb = position_encoding(relatives[mask][torch.arange(len(choices)), choices],
                                    self.posfreq)  # [query_pts, emb]
        rgb_feat = torch.cat([k0s, xyz_emb, viewdirs_emb], -1)  # [query_pts, eeemmmbbb]
        rgb = torch.zeros(*alpha.shape, 3).to(alpha)
        alpha_pred = torch.zeros(*alpha.shape).to(alpha)
        pred = self.rgbnet(rgb_feat)  # [query_pts, dims]
        rgb[mask] = torch.sigmoid(pred[..., :3])
        alpha_pred[mask] = torch.sigmoid(pred[..., 3])
        # compute accumulated transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha * alpha_pred)
        return alpha * alpha_pred, alphainv_cum, rgb, weights