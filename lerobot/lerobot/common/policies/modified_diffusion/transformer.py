import torch
import torch.nn as nn
from timm.layers import Mlp
import math
from torch.nn import functional as F

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        condition_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_heads,
        block_type,
        mlp_ratio=4.0,
        dropout=0.1,
        time_embed_dim=256,
        rope_max_seq_length=16,  # max sequence length for RoPE
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.cond_embed = nn.Linear(condition_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            get_flow_transformer_block(block_type)(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rope_max_seq_length=rope_max_seq_length
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # Basic initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)
        
    def forward(self, x, t, condition):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(condition)

        for block in self.transformer_blocks:
            x = block(x, t, c)

        x = self.norm(x)
        x = self.out_proj(x)
        return x


class CrossAttentionAdaLNTransformerBlock(nn.Module):
    """
    A conventional Transformer block using pre-layernorm.
    The block consists of self-attention, cross-attention, and MLP.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        dropout=0.,
        rope_max_seq_length=16,  
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6*dim)
        )

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=rope_max_seq_length,
        )

        # Cross-attention
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=rope_max_seq_length,
        )

        # MLP
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=0
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    def forward(self, x, t, c):

        # inject time embedding to adaln
        B = x.shape[0]
        features = self.ada_ln(nn.SiLU()(t)).view(B, 6, 1, self.dim).unbind(1)
        gamma1, gamma2, scale1, scale2, shift1, shift2 = features

        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1.mul(scale1.add(1)).add_(shift1)
        x = x + self.self_attn(x_norm1).mul_(gamma1)

        x = x + self.cross_attn(x, c).mul_(gamma2)

        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2.mul(scale2.add(1)).add_(shift2)
        x = x + self.mlp(x_norm2).mul_(gamma2)

        return x
    

class AdaLNTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        dropout=0.,
        rope_max_seq_length=16,  # max sequence length for RoPE
    ):
        super().__init__()
        # self.norm1 = RMSNorm(dim)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
            max_seq_len=rope_max_seq_length,
        )

        def approx_gelu(): return nn.GELU(approximate="tanh")

        # self.norm2 = RMSNorm(dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=0
        )

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6*dim)
        )
        self.dim = dim

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    def forward(self, x, t, c):
        B = x.shape[0]
        features = self.ada_ln(nn.SiLU()(t+c)).view(B, 6, 1, self.dim).unbind(1)
        gamma1, gamma2, scale1, scale2, shift1, shift2 = features

        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1.mul(scale1.add(1)).add_(shift1)
        x = x + self.attn(x_norm1).mul_(gamma1)

        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2.mul(scale2.add(1)).add_(shift2)
        x = x + self.mlp(x_norm2).mul_(gamma2)

        return x
    

FLOW_TRANFORMER_BLOCK_TYPES = {
    "adaln": AdaLNTransformerBlock,
    "cross_adaln": CrossAttentionAdaLNTransformerBlock,
}

def get_flow_transformer_block(block_type): return FLOW_TRANFORMER_BLOCK_TYPES[block_type]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-6, weight=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if weight else None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is None:
            return output
        return output * self.weight


class RotaryPositionalEmbeddings(nn.Module):
    """ RoPE implementation from torchtune """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 256,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self._build_rope_cache(self.max_seq_len)

    def _build_rope_cache(self, max_seq_len: int = 256) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x) -> torch.Tensor:
        """
        Inputs: x: [B, num_heads, S, head_dim]
        Returns: [B, num_heads, S, head_dim]
        """
        x = x.permute(0, 2, 1, 3)  # [B, S, num_heads, head_dim]
        B, S, num_heads, head_dim = x.size()

        rope_cache = (self.cache[:S])
        xshaped = x.float().reshape(*x.shape[:-1], head_dim // 2, 2)
        rope_cache = rope_cache.view(1, S, num_heads, head_dim // 2, 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3)
        x_out = x_out.permute(0, 2, 1, 3)
        return x_out.type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        max_seq_len=16,
        qk_norm=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.attn_norm = self.head_dim ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPositionalEmbeddings(dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, S, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)
        if self.training:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, S, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        max_seq_len=16,
        qk_norm=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPositionalEmbeddings(dim, max_seq_len=max_seq_len)

    def forward(self, x, context, mask=None):
        """
        x: [B, S, D] - Query input
        context: [B, S_ctx, D] - Key/Value input
        mask: Optional attention mask
        """
        B, S, C = x.shape
        _, S_ctx, _ = context.shape

        # Project queries from x
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Project keys and values from context
        kv = self.kv_proj(context).reshape(B, S_ctx, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # Each [B, num_heads, S_ctx, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.rope(q)
        # Condition is not time-series. There there is no k = self.rope(k)
        if self.training:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, S, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


