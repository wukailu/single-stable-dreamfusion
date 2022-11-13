import torch
from torch import nn

## utils

__all__ = ["register_model", "get_MLP", "model_dict"]

model_dict = {}


def register_model(cls):
    key = cls.__name__.lower()
    if key not in model_dict:
        model_dict[key] = cls
    elif model_dict[key] != cls:
        raise KeyError(f"Duplicated key {cls.__name__} from {model_dict[cls.__name__]} and {cls}!!!!")


def get_MLP(model_name, *args, **kwargs):
    return model_dict[model_name.lower()](*args, **kwargs)


## models


class Res1D(nn.Module):
    def __init__(self, width, act):
        super().__init__()
        self.net = nn.Linear(width, width)
        self.act = act

    def forward(self, x):
        return self.act(self.net(x)) + x


@register_model
class ResMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim=3, width=128, depth=3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(inplace=True),
            *[
                Res1D(width, nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, out_dim),
        )
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        return self.net(x)


@register_model
def MLP(*args, **kwargs):
    return BasicMLP(*args, **kwargs)


class BasicMLP(torch.nn.Module):
    def __init__(self, in_dim=0, out_dim=3, width=128, depth=3, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, out_dim),
        )
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        return self.net(x)


# 预测 RGB, shadow
@register_model
class ShadowMLP(BasicMLP):
    def __init__(self, in_dim=0, out_dim=4, width=128, depth=3, k0_dim=6, **kwargs):
        super().__init__(k0_dim, 3, width, depth)
        assert out_dim == 4
        self.input_ch = in_dim
        self.k0_dim = k0_dim

        shadow_layers = [nn.Linear(k0_dim + 9, width // 2), nn.ReLU(), nn.Linear(width // 2, 1)]
        self.shadow_layers = nn.Sequential(*shadow_layers)
        print(self)

    def forward(self, input):
        '''
        :param input: [..., k0_dim+fg_sph]
        :return [..., 4]
        '''
        base_remap = input[..., :self.k0_dim]
        rgb = self.net(base_remap)

        input_sph = input[..., -9:]
        shadow = self.shadow_layers(torch.cat((base_remap, input_sph), dim=-1))
        return torch.cat([rgb, shadow], dim=-1)


# 预测 RGB, shadow; 放弃 Relight，只考虑 albedo 的重建
@register_model
class DirectShadowMLP(BasicMLP):
    def __init__(self, in_dim=0, out_dim=4, width=128, depth=3, k0_dim=6, **kwargs):
        super().__init__(k0_dim - 1, 3, width, depth)
        assert out_dim == 4
        self.input_ch = in_dim
        self.k0_dim = k0_dim - 1

    def forward(self, input):
        rgb = self.net(input[..., :self.k0_dim])
        shadow = input[..., self.k0_dim:self.k0_dim + 1]
        return torch.cat([rgb, shadow], dim=-1)


@register_model
class ShadowV3MLP(BasicMLP):
    def __init__(self, in_dim=0, out_dim=4, width=128, depth=3, k0_dim=6, shadow_dim=1, **kwargs):
        super().__init__(in_dim - shadow_dim, 3, width, depth)
        assert out_dim == 4
        self.shadow_dim = shadow_dim
        self.shadowNet = nn.Linear(shadow_dim, 1)
        nn.init.constant_(self.shadowNet.bias, 0)

    def forward(self, input):
        rgb = self.net(input[..., self.shadow_dim:])
        shadow = self.shadowNet(input[..., :self.shadow_dim])
        return torch.cat([rgb, shadow], dim=-1)


@register_model
class ShadowV4MLP(BasicMLP):
    def __init__(self, in_dim=0, out_dim=4, width=128, depth=3, k0_dim=6, shadow_dim=1, **kwargs):
        super().__init__(in_dim - shadow_dim, 3, width, depth)
        assert out_dim == 4
        self.shadow_dim = shadow_dim
        self.shadowNet = nn.Sequential(nn.Linear(shadow_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.constant_(self.shadowNet[-1].bias, 0)

    def forward(self, input):
        rgb = self.net(input[..., self.shadow_dim:])
        shadow = self.shadowNet(input[..., :self.shadow_dim])
        return torch.cat([rgb, shadow], dim=-1)


# view-dependent shadow
@register_model
class ShadowV5MLP(BasicMLP):
    def __init__(self, in_dim=0, out_dim=4, width=128, depth=3, k0_dim=6, shadow_dim=1, **kwargs):
        super().__init__(k0_dim - shadow_dim, 3, width, depth)
        assert out_dim == 4
        self.shadow_dim = shadow_dim
        self.k0_dim = k0_dim

        self.shadowNet = nn.Sequential(
            nn.Linear(in_dim-k0_dim+shadow_dim, width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, 1),
        )
        nn.init.constant_(self.shadowNet[-1].bias, 0)
        print(self)

    def forward(self, input):
        rgb = self.net(input[..., self.shadow_dim:self.k0_dim])
        shadow = self.shadowNet(torch.cat([input[..., :self.shadow_dim], input[..., self.k0_dim:]], dim=-1))
        return torch.cat([rgb, shadow], dim=-1)


@register_model
class RobustShadowMLP(BasicMLP):
    def __init__(self, in_dim=0, out_dim=4, width=128, depth=3, k0_dim=6, shadow_dim=1, ensemble=2, noise_beta=0.1, **kwargs):
        super().__init__(k0_dim - shadow_dim, 3, width, depth)
        assert out_dim == 4
        self.shadow_dim = shadow_dim
        self.k0_dim = k0_dim
        self.ensemble = ensemble  # number of samples taken
        self.noise_beta = noise_beta  # the scale of the gaussian noise

        self.shadowNet = nn.Sequential(
            nn.Linear(in_dim-k0_dim+shadow_dim, width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, 1),
        )
        nn.init.constant_(self.shadowNet[-1].bias, 0)

    def forward(self, input):
        shadow = self.shadowNet(torch.cat([input[..., :self.shadow_dim], input[..., self.k0_dim:]], dim=-1))
        rgb_inp = input[..., self.shadow_dim:self.k0_dim]
        if self.training:
            noises = [torch.randn_like(rgb_inp) * self.noise_beta for i in range(self.ensemble)]
            rgb_inp = torch.cat([rgb_inp + noi for noi in noises], dim=0)
            out = self.net(rgb_inp)
            rgb = sum(out.split(noises[0].shape[0], 0)) / self.ensemble
        else:
            rgb = self.net(rgb_inp)
        return torch.cat([rgb, shadow], dim=-1)


@register_model
class ShadowRGIMLP(BasicMLP):
    def __init__(self, in_dim=0, out_dim=3, width=128, depth=3, k0_dim=6, shadow_dim=1, **kwargs):
        super().__init__(k0_dim - shadow_dim, 2, width, depth)
        assert out_dim == 3
        self.shadow_dim = shadow_dim
        self.k0_dim = k0_dim

        self.shadowNet = nn.Sequential(
            nn.Linear(in_dim-k0_dim+shadow_dim, width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, 1),
        )
        nn.init.constant_(self.shadowNet[-1].bias, 0)
        print(self)

    def forward(self, input):
        rg = self.net(input[..., self.shadow_dim:self.k0_dim])
        luminance = self.shadowNet(torch.cat([input[..., :self.shadow_dim], input[..., self.k0_dim:]], dim=-1))
        return torch.cat([rg, luminance], dim=-1)


@register_model
class SeparateShadowMLP(BasicMLP):
    def __init__(self, in_dim=0, out_dim=4, width=128, depth=3, k0_dim=12, shadow_dim=6, **kwargs):
        super().__init__(k0_dim - shadow_dim, 3, width, depth)
        assert out_dim == 4
        self.input_ch = in_dim
        self.k0_dim = k0_dim - shadow_dim
        self.shadow_dim = shadow_dim

        shadow_layers = [nn.Linear(shadow_dim + 9, width // 2), nn.ReLU(), nn.Linear(width // 2, 1)]
        self.shadow_layers = nn.Sequential(*shadow_layers)
        print(self)

    def forward(self, input):
        '''
        :param input: [..., k0_dim+fg_sph]
        :return [..., 3], [..., 1]
        '''
        rgb = self.net(input[..., :self.k0_dim])
        shadow = self.shadow_layers(input[..., - 9 - self.shadow_dim:])
        return torch.cat([rgb, shadow], dim=-1)


@register_model
class DVP_MLP(BasicMLP):
    def __init__(self, *args, in_dim=0, k0_dim=0, **kwargs):
        super().__init__(*args, in_dim=in_dim, **kwargs)
        self.k0_dim = k0_dim
        self.mapping_network = nn.Sequential(
            nn.Linear(k0_dim, k0_dim), nn.ReLU(inplace=True),
            nn.Linear(k0_dim, k0_dim),
        )

    def forward(self, x):
        x = torch.cat([self.mapping_network(x[:, :self.k0_dim]), x[:, self.k0_dim:]], dim=1)
        return self.net(x)


@register_model
class RobustMLP(BasicMLP):
    def __init__(self, *args, k0_dim=0, ensemble=2, noise_beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.k0_dim = k0_dim  # first k0_dim of input is the feature, where we will add the noise
        self.ensemble = ensemble  # number of samples taken
        self.noise_beta = noise_beta  # the scale of the gaussian noise

    def forward(self, x):
        x = torch.cat([x[:, :self.k0_dim].clamp(-1, 1), x[:, self.k0_dim:]], dim=1)
        # return self.net(x)
        if self.training:
            noises = [torch.cat([
                torch.randn_like(x[:, :self.k0_dim]) * self.noise_beta,
                torch.zeros_like(x[:, self.k0_dim:])],
                dim=1) for i in range(self.ensemble)]
            x = torch.cat([x + noi for noi in noises], dim=0)
            out = self.net(x)
            return sum(out.split(noises[0].shape[0], 0)) / self.ensemble
        else:
            return self.net(x)


#####################LIIF######################
# 区别在于这里要同时得到 density 和 color， 其中 density 与视角无关，也就是结构会接近于标准NeRF
@register_model
class LIIF_MLP(torch.nn.Module):
    def __init__(self, in_dim=0, out_dim=4, width=128, depth=3, viewdir_dim=0, **kwargs):
        super().__init__()
        self.view_dim = viewdir_dim
        self.net1 = nn.Sequential(
            nn.Linear(in_dim - viewdir_dim, width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 3)
            ], )
        self.net2 = nn.Sequential(
            nn.Sequential(nn.Linear(width + viewdir_dim, width), nn.ReLU(inplace=True)),
            nn.Linear(width, out_dim - 1),
        )
        nn.init.constant_(self.net2[-1].bias, 0)

    def forward(self, x):
        emb, view_emb = x[:, :-self.view_dim], x[:, -self.view_dim:]
        hidden = self.net1(emb)
        density = hidden[:, -1:]
        color = self.net2(torch.cat([hidden, view_emb], -1))
        return torch.cat([color, density], -1)
