from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.policies.gaze_vision_encoder import GazeVisionEncoderConfig


@PreTrainedConfig.register_subclass("flare")
@dataclass
class FlareConfig(PreTrainedConfig):

    # Inputs / output structure.
    fps: float = 8.333333333
    n_obs_steps: int = 1
    n_obs_step_size: int = 1
    chunk_size: int = 16
    n_action_steps: int = 8

    drop_n_last_frames: int = 8*3

    observation_keys: list[str] = field(default_factory=lambda: [
        "observation.images.zed_cam_left",
        "observation.state", 
    ])

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture / modeling.
    vision: GazeVisionEncoderConfig = field(
        default_factory=lambda: GazeVisionEncoderConfig()
    )
    
    flow_net: str = "transformer"

    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Transformer
    dim_model: int = 512
    n_heads: int = 8
    mlp_ratio: float = 4.0
    n_decoder_layers: int = 6
    dropout: float = 0.1
    time_dim: int = 128
    rope_max_seq_len: int = 16
    state_dropout: float = 0.3

    # Flow matching
    flow_matcher: str = "target"
    num_sampling_steps: int = 6
    flow_matcher_kwargs: dict = field(default_factory=lambda: {
        "sigma": 0.0,
    })

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    grad_clip_norm: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.chunk_size % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        # return list(range(1 - self.n_obs_steps, 1))
        return list(reversed(range(0, -self.n_obs_steps*self.n_obs_step_size, -self.n_obs_step_size)))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None