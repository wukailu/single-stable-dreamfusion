import torch
import torch.nn.functional as F

from frameworks.nerf.modules.utils import position_encoding
from frameworks.nerf.modules.dvgo_coarse import DVGO_Coarse
from frameworks.nerf.decoders.mlps import get_MLP


class DVGO_Fine(DVGO_Coarse):
    def parse_config(self, cfg):
        return cfg.data, cfg.fine_model_and_render, cfg.fine_train

    def init_parameters(self, cfg_model):
        self.density = torch.nn.Parameter(torch.randn([1, 1, *self.world_size]))
        self.k0 = torch.nn.Parameter(torch.randn([1, cfg_model.rgbnet_dim, *self.world_size]))
        dim0 = cfg_model.rgbnet_dim
        if cfg_model.posbase_pe != 0:
            self.register_buffer('posfreq', torch.FloatTensor([(2 ** i) for i in range(cfg_model.posbase_pe)]))
            dim0 += (3 + 3 * cfg_model.posbase_pe * 2)
        else:
            print("posbase_pe is 0!")
        if cfg_model.viewbase_pe != 0:
            self.register_buffer('viewfreq', torch.FloatTensor([(2 ** i) for i in range(cfg_model.viewbase_pe)]))
            dim0 += (3 + 3 * cfg_model.viewbase_pe * 2)
        else:
            print("viewbase_pe is 0!")
        self.dim0 = dim0
        self.rgbnet = self.get_rgbnet(cfg_model)

    def get_rgbnet(self, cfg_model):
        return get_MLP(cfg_model.rgbnet, in_dim=self.dim0, out_dim=3, width=cfg_model.rgbnet_width,
                       depth=cfg_model.rgbnet_depth, k0_dim=cfg_model.rgbnet_dim)

    def _scale_volume_grid(self, num_voxels):
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        self.k0 = torch.nn.Parameter(
            F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if hasattr(self, 'mask_cache') and self.mask_cache is not None:
            self._set_nonempty_mask()

    def query_rgb(self, rays_pts, viewdirs):
        # view-dependent color emission
        if len(rays_pts) == 0:
            return 0
        rays_xyz = (rays_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        k0_view = [self.grid_sampler(rays_pts, self.k0)]
        viewdirs_emb = [position_encoding(viewdirs, self.viewfreq)] if hasattr(self, 'viewfreq') else []
        xyz_emb = [position_encoding(rays_xyz, self.posfreq)] if hasattr(self, 'posfreq') else []
        rgb_feat = torch.cat(k0_view + xyz_emb + viewdirs_emb, -1)
        return torch.sigmoid(self.rgbnet(rgb_feat))

    ######################################################################
