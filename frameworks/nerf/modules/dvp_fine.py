import torch

from frameworks.nerf.modules.utils import total_variation, position_encoding
from frameworks.nerf.modules.dvgo_fine import DVGO_Fine


class DVGO_Plus(DVGO_Fine):
    def _k0_total_variation(self):
        v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)

    def query_rgb(self, rays_pts, viewdirs):
        # view-dependent color emission
        if len(rays_pts) == 0:
            return 0
        rays_xyz = (rays_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        ####### only thing new #######
        k0_view = [torch.sigmoid(self.grid_sampler(rays_pts, self.k0))]
        ##############################
        viewdirs_emb = [position_encoding(viewdirs, self.viewfreq)] if hasattr(self, 'viewfreq') else []
        xyz_emb = [position_encoding(rays_xyz, self.posfreq)] if hasattr(self, 'posfreq') else []
        rgb_feat = torch.cat(k0_view + xyz_emb + viewdirs_emb, -1)
        return torch.sigmoid(self.rgbnet(rgb_feat))