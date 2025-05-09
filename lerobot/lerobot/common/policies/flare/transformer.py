import math
import einops
import torch
from torch import Tensor, nn
from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from lerobot.common.policies.flare.configuration_flare import FlareConfig
from functools import partial
from lerobot.common.policies.gaze_vision_encoder import GazeVisionEncoder
from timm.models.vision_transformer import Mlp, RmsNorm
from lerobot.common.policies.flare.attention import Attention

class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FlowTransformer(nn.Module):

    def __init__(self, config: FlareConfig):
        super().__init__()

        self.config = config

        robot_state_dim = self.config.robot_state_feature.shape[0]
        self.robot_state_proj = nn.Sequential(
            nn.Linear(robot_state_dim, robot_state_dim),
            nn.GELU(),
            nn.Dropout(config.state_dropout),
        )
        global_cond_dim = robot_state_dim
        if self.config.image_features:
            self.vision_encoder = GazeVisionEncoder(config.vision)
            global_cond_dim += config.vision.out_dim * len(config.vision.image_keys) * (2 if config.vision.use_gaze and config.vision.gaze_add_small_features else 1)
            if config.vision.use_gaze:
                global_cond_dim += 2 * len(config.vision.eye_keys)
        if self.config.env_state_feature:
            env_state_dim = self.config.env_state_feature.shape[0]
            self.env_state_proj = nn.Sequential(
                nn.Linear(env_state_dim, env_state_dim),
                nn.GELU(),
                nn.Dropout(config.state_dropout),
            )
            global_cond_dim += env_state_dim
        global_cond_dim *= config.n_obs_steps
        self.global_cond_proj = nn.Linear(global_cond_dim, config.dim_model)

        # Transformer
        self.noise_input_proj = nn.Linear(self.config.action_feature.shape[0], config.dim_model)
        self.decoder = TransformerDecoder(config)
        action_dim = self.config.action_feature.shape[0]
        self.action_head = nn.Linear(config.dim_model, action_dim)
        
        # Encoder for the diffusion timestep.
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(config.time_dim),
            nn.Linear(config.time_dim, config.time_dim * 4),
            nn.Mish(),
            nn.Linear(config.time_dim * 4, config.dim_model),
        )

    def prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        global_cond_feats = [torch.flatten(self.robot_state_proj(batch[OBS_ROBOT]), start_dim=1)]
        vision_ret = {}
        if self.config.image_features:
            img_features, vision_ret = self.vision_encoder(batch)
            global_cond_feats.append(torch.flatten(img_features, start_dim=1))
        if self.config.env_state_feature:
            global_cond_feats.append(torch.flatten(self.env_state_proj(batch[OBS_ENV]), start_dim=1))
        global_cond_feats = self.global_cond_proj(torch.cat(global_cond_feats, dim=1))
        return global_cond_feats, vision_ret
    
    def forward(self, x: Tensor, timestep: Tensor | int, global_cond: dict) -> Tensor:
        x = self.noise_input_proj(x)
        t = self.timestep_encoder(timestep)
        c = global_cond
        x = self.decoder(x,t,c)
        x = self.action_head(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config: FlareConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.n_decoder_layers)])

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, t, c)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TransformerDecoderLayer(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, config: FlareConfig):
        super().__init__()

        norm_cls = partial(RmsNorm, eps=1e-5, affine=True)

        self.norm1 = norm_cls(config.dim_model)
        self.attn = Attention(
            dim=config.dim_model, 
            num_heads=config.n_heads, 
            max_seq_len=config.rope_max_seq_len,
            qkv_bias=True,
        )

        self.norm2 = norm_cls(config.dim_model)
        mlp_hidden_dim = int(config.dim_model * config.mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=config.dim_model, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            # nn.Linear(config.dim_model, config.dim_model, bias=True),
            nn.SiLU(),
            nn.Linear(config.dim_model, 6 * config.dim_model, bias=True)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t+c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x