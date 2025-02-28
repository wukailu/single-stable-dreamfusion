import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.norm = nn.LayerNorm(self.dim_out)
        self.activation = nn.SiLU()

        if self.dim_in != self.dim_out:
            self.skip = nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        # x: [B, C]
        identity = x

        out = self.dense(x)
        out = self.norm(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.activation(out)

        return out


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l != num_layers - 1:
                net.append(ResBlock(self.dim_in if l == 0 else self.dim_hidden, self.dim_hidden, bias=bias))
            else:
                net.append(nn.Linear(self.dim_hidden, self.dim_out, bias=bias))

        self.net = nn.ModuleList(net)
        
    
    def forward(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)
            
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=5,
                 hidden_dim=128,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, self.in_dim = get_encoder('frequency', input_dim=3)
        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def gaussian(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        h = self.encoder(x, bound=self.bound)

        h = self.sigma_net(h)

        sigma = trunc_exp(h[..., 0] + self.gaussian(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def normal(self, x):

        with torch.enable_grad():
            x.requires_grad_(True)
            sigma, albedo = self.common_forward(x)
            # query gradient
            normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]

        # normalize...
        normal = safe_normalize(normal)
        normal[torch.isnan(normal)] = 0
        return normal
        
    def forward(self, x, d, l=None, ratio=1, shading='albedo', weight=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if shading == 'albedo':
            # no need to query normal
            sigma, color = self.common_forward(x)
            normal = None
        
        else:
            # query normal

            # sigma, albedo = self.common_forward(x)
            # normal = self.finite_difference_normal(x)

            with torch.enable_grad():
                x.requires_grad_(True)
                sigma, albedo = self.common_forward(x)
                # query gradient
                normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]

            # normalize...
            normal = safe_normalize(normal)
            normal[torch.isnan(normal)] = 0

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            # {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]        

        if self.bg_radius > 0:
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params


class NeRFNetwork_Kailu(NeRFNetwork):
    def __init__(self, opt, num_layers_bg=2, hidden_dim_bg=64, pretrained_load_from=""):
        # NeRFRenderer.__init__(self, opt)

        super(NeRFNetwork, self).__init__(opt)
        from frameworks.nerf.modules import load_nerf
        self.main_net = load_nerf(pretrained_load_from)
        # to enable backward on grid_sample3d
        import types
        self.main_net.grid_sampler = types.MethodType(grid_sampler_with_grad, self.main_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)

        else:
            self.bg_net = None

    def to_our_coor(self, x):
        scaled = ((x + self.bound) / (2 * self.bound))[..., [0, 2, 1]]  # swap y-z
        scaled = (scaled - 0.5) * 1.25 + 0.5  # scale
        scaled = scaled * (self.main_net.xyz_max - self.main_net.xyz_min) + self.main_net.xyz_min
        return scaled

    def common_forward(self, x, weight=None):
        if weight is None:
            weight = torch.ones_like(x[..., 0])
        # 返回x对应的密度和颜色，最终的密度计算是 1-exp(sigma * dis)
        # x: [N, 3], in [-bound, bound]
        rays_pts = self.to_our_coor(x)
        inside_mask = ((rays_pts <= self.main_net.xyz_max) & (self.main_net.xyz_min <= rays_pts)).all(dim=-1)
        density = torch.zeros_like(x[..., 0])
        density[inside_mask] = self.main_net.grid_sampler(rays_pts[inside_mask], self.main_net.density)[..., 0]
        sigma = F.softplus(density + self.main_net.act_shift) * 10
        # sigma
        albedo = torch.ones_like(x).float() * 0.5
        valid_mask = (weight > (1e-2 + self.main_net.act_shift)) & inside_mask
        masked_x = rays_pts[valid_mask]
        masked_viewdirs = torch.ones_like(rays_pts[valid_mask])/(3**0.5)
        albedo[valid_mask] = self.main_net.query_rgb(masked_x, masked_viewdirs).float()

        return sigma, albedo

    # optimizer utils
    def get_params(self, lr):
        self.main_net.density.requires_grad = False
        self.main_net.k0.requires_grad = False
        # freeze(self.main_net.rgbnet)
        params = [
            # {'params': self.main_net.parameters(), 'lr': lr},
            {'params': self.main_net.rgbnet.parameters(), 'lr': lr},
        ]

        if self.bg_radius > 0:
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params

    def forward(self, x, d, l=None, ratio=1, shading='albedo', weight=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if shading == 'albedo':
            # no need to query normal
            sigma, color = self.common_forward(x, weight)
            normal = None

        else:
            # query normal

            sigma, albedo = self.common_forward(x, weight)
            normal = self.normal(x)

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0)  # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else:  # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)

        return sigma, color, normal

def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def grid_sampler_with_grad(self, xyz, grid):
    '''Wrapper for the interp operation'''
    shape = xyz.shape[:-1]
    xyz = xyz.reshape(1, 1, 1, -1, 3)
    ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
    from frameworks.nerf.modules.osr_fine import grid_sample_3d
    return grid_sample_3d(grid, ind_norm).reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1])