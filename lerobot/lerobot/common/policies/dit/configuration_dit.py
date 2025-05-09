from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.policies.gaze_vision_encoder import GazeVisionEncoderConfig


@PreTrainedConfig.register_subclass("dit")
@dataclass
class DiTConfig(PreTrainedConfig):
    # Inputs / output structure.
    fps: float = 8.333333333
    n_obs_steps: int = 2
    n_obs_step_size: int = 1
    n_action_steps: int = 8
    chunk_size: int = 8

    observation_keys: list[str] = field(default_factory=lambda: [
        "observation.images.zed_cam_left",
        # "observation.images.zed_cam_right", 
        "observation.state", 
    ])

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 8*3

    # Architecture / modeling.
    # Vision backbone.
    vision: GazeVisionEncoderConfig = field(
        default_factory=lambda: GazeVisionEncoderConfig()
    )

    # Transformer
    dim_model: int = 512
    dim_feedforward: int = 2048
    n_heads: int = 8
    n_blocks: int = 6
    dropout: float = 0.1
    time_dim: int = 256
    state_dropout: float = 0.2

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    grad_clip_norm: float = 1.0

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