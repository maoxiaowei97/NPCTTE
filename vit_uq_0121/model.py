"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math
from .transformer import Transformer
from .utils import load_pretrained_weights, as_tuple
from .configs import PRETRAINED_MODELS
from einops import rearrange, repeat

class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding

class PositionalEncoding(nn.Module):
    """
    A type of trigonometric encoding for indicating items' positions in sequences.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()

        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0.1, output_layer=True):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))  # 输出下一个节点的eta
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x.float())

class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
            self,
            name: str = 'vit',
            patches: int = 4,
            dim: int = 128,
            ff_dim: int = 128,
            num_heads: int = 8,
            num_layers: int = 2,
            dropout_rate: float = 0.1,
            in_channels: int = 3,
            image_size: int = 20,
            predict_type: str = 'only_mean'
    ):
        super().__init__()


        self.image_size = image_size
        self.name = name
        self.d_model = dim
        self.embed_dim = 32

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw


        # Patch embedding
        in_channels = 2 + self.embed_dim
        # self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw))
        self.grid_embed = nn.Embedding(image_size * image_size + 1,  self.embed_dim) # 每个格子有一个embedding
        self.positional_embedding = PositionalEmbedding1D(seq_len, dim)

        # Transformer
        self.transformer = Transformer(num_layers=2, dim=dim, num_heads=num_heads,
                                       ff_dim=ff_dim, dropout=dropout_rate)
        # self.pos_encode = PositionalEncoding(dim, dim)
        self.gh = gh
        self.gw = gw

        self.out_linear = nn.Linear(dim * self.gh * self.gw, dim)
        self.final_layer_mean = MLP(2 * dim + 11 +  self.embed_dim * 3, (dim,), output_layer=True) #  2 * d + 9 + 32 * 3
        self.final_layer_sigma = MLP(2 * dim + 11 +  self.embed_dim * 3, (dim,), output_layer=True)
        self.ts_embed = nn.Embedding(144 + 1,  self.embed_dim)
        self.od_seq = nn.GRU(input_size = self.embed_dim, hidden_size = self.d_model,batch_first = True)
        self.predict_type = predict_type

    def forward(self, x, odt):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
            odt (tensor): b, 11
        """

        x = x[:, [2, 3, 6]] # 只取-1.5 * sigma_1 * w_1, -2 * sigma_1 * w_1, mean三个值
        b, k, c, fh, fw = x.shape
        mask = x[:, :, 0] < 0
        #选出条件信息，和PiT信息一起聚合

        mask = x[:, 0, :, :] < 0
        mask = repeat(mask, 'b h w -> b h w d', d = self.embed_dim + 2)
        #mask -> b, fh, fw, d+1

        #grid embedding
        grid = torch.arange(0, fh * fw).long().to(x.device)
        grid = repeat(grid, 'g -> b g', b=b)  # (b, num_grid)
        grid = self.grid_embed(grid)  # (b, num_grid, d_model)

        # 图中通过网格先后顺序特征，通过网格时间和出发时间差
        x[:, 1] = (x[:, 1] + 1) / 2 * 60 # 转化为起点到网格的分钟时间差
        x_seq = x[:, [1,2], :, :].reshape(b, fh * fw, 2) # time_diff_min, time_diff_max

        x_input = torch.cat([x_seq, grid], 2).reshape(b, fh, fw,  self.embed_dim + 2) # b fh, fw, d_embed + 2
        x_input[mask] = 0 # 网格mask
        x_input = rearrange(x_input, 'b h w d -> b d h w')

        x_input = self.patch_embedding(x_input)  # b,d,gh,gw
        x_input = x_input.flatten(2).transpose(1, 2)  # b,gh*gw,d, gh = image_h / patch

        route = self.transformer(x_input)  # b,gh*gw,d
        route = self.out_linear(route.reshape(b, self.gh * self.gw, -1).mean(1))# b, d

        #o, d的cell index
        o_cell_index = odt[:,7].reshape(b, 1).long()
        d_cell_index = odt[:,8].reshape(b, 1).long()
        ts_10 = odt[:, 10].reshape(b, 1)
        ts_10 = self.ts_embed(ts_10.long())
        o_cell_embed = self.grid_embed(o_cell_index)
        d_cell_embed = self.grid_embed(d_cell_index)
        # o, d过一个LSTM
        _, hn = self.od_seq(torch.cat([o_cell_embed, d_cell_embed], 1)) # [b, 2, d]

        fused = torch.cat([route, odt, ts_10.squeeze(1), o_cell_embed.squeeze(1), d_cell_embed.squeeze(1), hn.squeeze(0)], dim=1)# b, 2 * d + 9 + 32 * 3
        mean_out = self.final_layer_mean(fused).squeeze(1) # b,
        sigma_out = self.final_layer_sigma(fused).squeeze(1)  # b,

        return mean_out, sigma_out

