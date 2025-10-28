"""Microscopy transformer model architecture.
Forked from github.com/facebookresearch/DiT (MIT license).

We make four small changes from the vanilla DiT:
1. enable dropout on the attention projection (improves stability)
2. add UNet-style skip connections, inspired by SkipDiT (improves stability)
3. replace LayerNorm with RMSNorm (for slightly improved memory bandwidth)
4. separate out conditioning into swappable adaptors
"""

import functools
import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed  # type: ignore


class ScalarEmbedder(nn.Module):
    """Embeds scalar conditions into vector representations. Useful for timesteps, zoom levels, etc."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations. Also handles dropout for cfg."""

    def __init__(self, num_classes, hidden_size, dropout_prob=0.15):
        super().__init__()
        # +1 for null token
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class BasicAdaptor(nn.Module):
    """Adaptor module for conditioning layers in DiT generator.
    Uses OH embeddings (useful for pretraining but cannot generalise to new conditions).
    Assumes the following conditioning:
    - t: a scalar time condition, float range [0,1]
    - y: a class label, int range [0, y_dim-1]
    - e: an experiment label, int range [0, e_dim-1]
    - c: a cell type label, int range [0, c_dim-1]
    """

    def __init__(self, hidden_size, y_dim, e_dim, c_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embedder = ScalarEmbedder(hidden_size, frequency_embedding_size)
        self.y_embedder = LabelEmbedder(num_classes=y_dim, hidden_size=hidden_size)
        self.e_embedder = LabelEmbedder(num_classes=e_dim, hidden_size=hidden_size)
        self.c_embedder = LabelEmbedder(num_classes=c_dim, hidden_size=hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)  # type: ignore
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)  # type: ignore
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.e_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.c_embedder.embedding_table.weight, std=0.02)

    def forward(self, t, y, e, c):
        t_emb = self.t_embedder(t)

        # allow passing None to use unconditional embedding for cfg
        if y is None:
            y = torch.full((t.shape[0],), self.y_embedder.num_classes, device=t.device)
        if e is None:
            e = torch.full((t.shape[0],), self.e_embedder.num_classes, device=t.device)
        if c is None:
            c = torch.full((t.shape[0],), self.c_embedder.num_classes, device=t.device)

        y_emb = self.y_embedder(y, train=self.training)
        e_emb = self.e_embedder(e, train=self.training)
        c_emb = self.c_embedder(c, train=self.training)
        return t_emb + y_emb + e_emb + c_emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.1,
        )
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # noqa
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,  # type: ignore
            drop=0.0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class SkipLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(2 * hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x, skip):
        cat = torch.cat([x, skip], dim=-1)
        cat = self.norm(cat)
        return self.linear(cat)


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.skip_layers = nn.ModuleList(
            [SkipLayer(hidden_size) for _ in range(depth // 2)]
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)  # type: ignore

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # type: ignore
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)  # type: ignore

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)  # type: ignore
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)  # type: ignore
        nn.init.constant_(self.final_layer.linear.weight, 0)  # type: ignore
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """(N, T, patch_size**2 * C) -> (N, C, H, W)"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))  # type: ignore
        return imgs

    def forward(self, x, c):
        """Forward pass of DiT.
        x: (B, C, H, W) float32 tensor of spatial inputs (images or latent representations of images)
        c: (B, hidden_size) float32 tensor of conditioning inputs used as the residual stream."""
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)

        skips = []
        index = 0
        depth = len(self.blocks)
        for block in self.blocks:
            if index >= depth // 2:
                skip_layer = self.skip_layers[index - depth // 2]
                skip = skips.pop()
                x = skip_layer(x, skip)

            x = block(x, c)  # (N, T, D)

            if index < depth // 2:
                skips.append(x)
            index += 1

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


class DiTWrapper(nn.Module):
    """Wraps DiT to include a BasicAdaptor for conditioning."""

    def __init__(
        self,
        y_dim,
        e_dim,
        c_dim,
        input_size=32,
        patch_size=2,
        in_channels=6,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        frequency_embedding_size=256,
    ):
        super().__init__()
        self.dit = DiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            learn_sigma=learn_sigma,
        )
        self.adaptor = BasicAdaptor(
            hidden_size=hidden_size,
            y_dim=y_dim,
            e_dim=e_dim,
            c_dim=c_dim,
            frequency_embedding_size=frequency_embedding_size,
        )

    def forward(self, *, x, t, y=None, e=None, c=None):
        cond = self.adaptor(t, y, e, c)  # (B, hidden_size)
        return self.dit(x, cond)  # (B, out_channels, H, W)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def get_model_cls(
    name: Literal[
        "DiT-XL/2", "DiT-XL/4", "DiT-XL/8",
        "DiT-L/2", "DiT-L/4", "DiT-L/8",
        "DiT-B/2", "DiT-B/4", "DiT-B/8",
        "DiT-S/2", "DiT-S/4", "DiT-S/8",
    ],
) -> functools.partial[DiTWrapper]:  # fmt: skip
    model_variant, patch_size = name.split("/")
    match model_variant:
        case "DiT-S":
            depth = 12
            hidden_size = 384
            num_heads = 6
        case "DiT-B":
            depth = 12
            hidden_size = 768
            num_heads = 12
        case "DiT-L":
            depth = 24
            hidden_size = 1024
            num_heads = 16
        case "DiT-XL":
            depth = 28
            hidden_size = 1152
            num_heads = 16
        case _:
            raise ValueError(f"Unknown model name: {name}")

    patch_size = int(patch_size)
    assert patch_size in [2, 4, 8], f"Invalid patch size: {patch_size}"
    return functools.partial(
        DiTWrapper,
        depth=depth,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_heads=num_heads,
    )
