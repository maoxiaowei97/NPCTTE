import math
from inspect import isfunction
from functools import partial

import torch
from torch import nn
from torch import einsum
from einops import rearrange
import numpy as np


def exists(x):
    """
    Judge whether the input exists.
    """
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



class Residual(nn.Module):
    """
    Adds the input to the output of a particular function.
    """

    def __init__(self, fn):
        """
        :param fn: the function residual connection add to.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    """
    Applies a 2D transposed convolution operator over an input image to upsample it.
    The transposed convolution can be seen as the gradient of Conv2d with respect to its input.
    """
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    """
    Applies a 2D convolution operator over an input image to downsample it.
    """
    return nn.Conv2d(dim, dim, 4, 2, 1)



class Block(nn.Module):
    """
    The basic building block of U-Net.
    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    Convolutional layers with conv-based residual connection.
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """
    A special type of convolutional block.
    https://arxiv.org/abs/2201.03545
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        # self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.traffic_cov = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), padding=2)
        # self.traffic_mlp = (
        #     nn.Sequential(nn.GELU(), nn.Linear(20 * 20 * 1, dim))
        # )

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)  # h, [b, 4, 20, 20]

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            # traffic = self.traffic_cov(traffic)  # [b, 1, 20, 20]
            # traffic = self.traffic_mlp(traffic.reshape(x.shape[0], -1))
            # h = h + rearrange(condition, "b c -> b c 1 1") + rearrange(traffic,
            #                                                            "b c -> b c 1 1")  # h + condition, condition可以变成odt + traffic_state(b, c, n, n). [b, 4, 20, 20], [b, 4, 1, 1]
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.net(h)
        return h + self.res_conv(x)


class PreNorm(nn.Module):
    """
    A block used for applying groupnorm before the attention blocks.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Attention(nn.Module):
    """
    Multi-head attention block, the same used in the Transformer.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    A more efficient attention block, where time- and memory requirements scale linear in the sequence length,
    as opposed to quadratic for regular attention.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)





class ETA_Unet(nn.Module):
    """
    A convolution-based bottleneck network. The network is built up as follows:
    - First, a convolutional layer is applied on the batch of noisy images,
        and position embeddings are computed for the noise levels.
    - Next, a sequence of downsampling stages are applied. Each downsampling stage consists of
        2 ResNet/ConvNeXT blocks + groupnorm + attention + residual connection + a downsample operation.
    - At the middle of the network, again ResNet or ConvNeXT blocks are applied, interleaved with attention.
    - Next, a sequence of upsampling stages are applied. Each upsampling stage consists of
        2 ResNet/ConvNeXT blocks + groupnorm + attention + residual connection + an upsample operation.
    - Finally, a ResNet/ConvNeXT block followed by a convolutional layer is applied.

    """

    def __init__(
            self,
            dim, # 128
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=None,
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2,
            condition='odt',
            split = 20
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.condition = condition
        self.split = split
        self.d_model = dim

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        y_dim = 5
        time_dim = dim * 4
        # self.y_linear = nn.Linear(y_dim, 128)
        self.y_linear = nn.Linear(y_dim, time_dim)
        self.out_state_linear =  nn.Sequential(nn.Linear(self.channels * self.split * self.split, dim),
                                    nn.LeakyReLU(), nn.Dropout(0.1),
                                    nn.Linear(dim, dim))

        self.final_layer_mean =  nn.Sequential(nn.Linear(dim, dim),
                                    nn.LeakyReLU(), nn.Dropout(0.1),
                                    nn.Linear(dim, 1))

        self.final_layer_sigma = nn.Sequential(nn.Linear(dim, dim),
                                              nn.LeakyReLU(), nn.Dropout(0.1),
                                              nn.Linear(dim, 1))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

        self.name = 'unet'
        self.num_layers = len(dim_mults)

        #####simple
        self.state_conv = nn.Conv2d(channels, 1, 7, padding=3)
        self.mean_layer = nn.Sequential(nn.Linear(128 + self.split * self.split, 128),
                                              nn.LeakyReLU(), nn.Dropout(0.1),
                                              nn.Linear(128, 1))
        self.sigma_layer = nn.Sequential(nn.Linear(128 + self.split * self.split, 128),
                                              nn.LeakyReLU(), nn.Dropout(0.1),
                                              nn.Linear(128, 1))
    def forward(self, x, time=None, y=None):  # images, odt
        # x: b, k, c, h, w
        # x = x[:, [2, 3, 4]] # b, 3, c, h, w 用一个channel试试 chengdu
        alpha_list = [3, 4, 5] # xian
        mean_multi = []
        sigma_multi = []
        for a in alpha_list:
            route = x[:, [a]]  # b, 3, c, h, w 用一个channel试试
            mask = route[:, :, 0] < 0 # [b, 3, h, w]
            x_t_diff_max = route[:, :, 2] # [b, 3, h, w]
            x_t_diff_max[mask] = -1  # [b, 3, h, w]
            state = self.init_conv(x_t_diff_max) # [b, 3, n, n] -> [b, c_out, n, n]
            y = y[:, :5] # odt 特征
            t = self.y_linear(y)
            h = []
            # downsample
            for block1, block2, attn, downsample in self.downs:
                state = block1(state, t)  # x: [b, 4, h, w]
                state = block2(state, t)
                state = attn(state)
                h.append(state)
                state = downsample(state)

            # bottleneck
            state = self.mid_block1(state, t)
            state = self.mid_attn(state)
            state = self.mid_block2(state, t)

            # upsample
            for block1, block2, attn, upsample in self.ups:
                state = torch.cat((state, h.pop()), dim=1)
                state = block1(state, t)
                state = block2(state, t)
                state = attn(state)
                state = upsample(state)

            state = self.final_conv(state)  # [b, k*c, n, n]
            state = state.reshape(state.size(0), -1) # [b, k*c*n*n]
            state = self.out_state_linear(state) # [b, 1]
            mean_out = self.final_layer_mean(state).squeeze(1)  # b,
            sigma_out = self.final_layer_sigma(state).squeeze(1)  # b,
            mean_multi.append(mean_out.reshape(-1, 1))
            sigma_multi.append(sigma_out.reshape(-1, 1))

        return torch.stack(mean_multi, dim=1).squeeze(-1), torch.stack(sigma_multi, dim=1).squeeze(-1)

    # def forward(self, x, time=None, y=None):  # images, odt -> conv
    #     # x: b, k, c, h, w
    #     x = x[:, [2, 3, 6]]  # b, 3, c, h, w
    #     mask = x[:, :, 0] < 0  # [b, 3, h, w]
    #     x_t_diff_max = x[:, :, 2]  # [b, 3, h, w]
    #     # x_t_diff_max = (x_t_diff_max + 1) * 30  # 转为分钟
    #     # x_t_diff_max[mask] = 0  # [b, 3, h, w]
    #     x_t_diff_max[mask] = -1
    #     # state = self.init_conv(x_t_diff_max)  # [b, 3, n, n] -> [b, c_out, n, n]
    #     y = y[:, :7]  # odt 特征
    #     t = self.y_linear(y) # [b, d] -> [b, h]
    #     state = self.state_conv(x_t_diff_max).squeeze(1) # [b, 1, n, n]
    #     state = torch.cat([t, state.reshape(x.size(0), -1)], dim=1) # [b, dim + n * n ]
    #     mean = self.mean_layer(state)
    #     sigma = self.sigma_layer(state)
    #     return mean, sigma


    # input: X_noise, odt
    # def forward(self, x, time=None, y=None):  # images, odt
    #     # x: b, k, c, h, w
    #     x = x[:, [2, 3, 6]] # b, 3, c, h, w
    #     mask = x[:, :, 0] < 0 # [b, 3, h, w]
    #     x_t_diff_max = x[:, :, 2] # [b, 3, h, w]
    #     x_t_diff_max = (x_t_diff_max + 1) * 30 # 转为分钟
    #     x_t_diff_max[mask] = 0 # [b, 3, h, w]
    #     state = self.init_conv(x_t_diff_max) # [b, 3, n, n] -> [b, c_out, n, n]
    #     y = y[:, :7] # odt 特征
    #     t = self.y_linear(y)
    #     h = []
    #     # downsample
    #     for block1, block2, attn, downsample in self.downs:
    #         state = block1(state, t)  # x: [b, 4, h, w]
    #         state = block2(state, t)
    #         state = attn(state)
    #         h.append(state)
    #         state = downsample(state)
    #
    #     # bottleneck
    #     state = self.mid_block1(state, t)
    #     state = self.mid_attn(state)
    #     state = self.mid_block2(state, t)
    #
    #     # upsample
    #     for block1, block2, attn, upsample in self.ups:
    #         state = torch.cat((state, h.pop()), dim=1)
    #         state = block1(state, t)
    #         state = block2(state, t)
    #         state = attn(state)
    #         state = upsample(state)
    #
    #     state = self.final_conv(state)  # [b, k*c, n, n]
    #     state = state.reshape(state.size(0), -1) # [b, k*c*n*n]
    #     state = self.out_state_linear(state) # [b, 1]
    #     mean_out = self.final_layer_mean(state).squeeze(1)  # b,
    #     sigma_out = self.final_layer_sigma(state).squeeze(1)  # b,
    #
    #     return mean_out, sigma_out

    # def forward(self, x, time=None, y=None):  # images, odt
    #     # x: b, k, c, h, w
    #     x = x[:, [2, 3, 6]]  # b, 3, c, h, w
    #     mask = x[:, :, 0] < 0  # [b, 3, h, w]
    #     x_t_diff_max = x[:, :, 2]  # [b, 3, h, w]
    #     x_t_diff_max[mask] = -1
    #     # x_t_diff_max = (x_t_diff_max + 1) * 30  # 转为分钟
    #     # x_t_diff_max[mask] = 0  # [b, 3, h, w]
    #     # state = self.init_conv(x_t_diff_max)  # [b, 3, n, n] -> [b, c_out, n, n]
    #     y = y[:, :7]  # odt 特征
    #     t = self.y_linear(y) # [b, d] -> [b, h]
    #     state = self.state_conv(x_t_diff_max).squeeze(1) # [b, 1, n, n]
    #     state = torch.cat([t, state.reshape(x.size(0), -1)], dim=1) # [b, dim + n * n ]
    #     mean = self.mean_layer(state)
    #     sigma = self.sigma_layer(state)
    #     return mean, sigma
    #     h = []
    #     # downsample
    #     for block1, block2, attn, downsample in self.downs:
    #         state = block1(state, t)  # x: [b, 4, h, w]
    #         state = block2(state, t)
    #         state = attn(state)
    #         h.append(state)
    #         state = downsample(state)
    #
    #     # bottleneck
    #     state = self.mid_block1(state, t)
    #     state = self.mid_attn(state)
    #     state = self.mid_block2(state, t)
    #
    #     # upsample
    #     for block1, block2, attn, upsample in self.ups:
    #         state = torch.cat((state, h.pop()), dim=1)
    #         state = block1(state, t)
    #         state = block2(state, t)
    #         state = attn(state)
    #         state = upsample(state)
    #
    #     state = self.final_conv(state)  # [b, k*c, n, n]
    #     state = state.reshape(state.size(0), -1)  # [b, k*c*n*n]
    #     state = self.out_state_linear(state)  # [b, 1]
    #     mean_out = self.final_layer_mean(state).squeeze(1)  # b,
    #     sigma_out = self.final_layer_sigma(state).squeeze(1)  # b,
    #
    #     return mean_out, sigma_out
