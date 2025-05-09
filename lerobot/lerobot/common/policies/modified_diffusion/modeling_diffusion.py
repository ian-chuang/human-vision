#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.common.policies.gaze_vision_encoder import GazeVisionEncoder
from lerobot.common.policies.diffusion.transformer import DiffusionTransformer
from lerobot.common.policies.diffusion.unet import DiffusionConditionalUnet1d
from lerobot.common.policies.diffusion.utils import SpatialSoftmax, SinusoidalPositionEmbedding2d
from lerobot.common.policies.diffusion.cfm import get_flow_matcher
from lerobot.common.constants import OBS_ENV, OBS_ROBOT, ACTION

class DiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("diffusion.vision_encoder.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("diffusion.vision_encoder.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            # "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        for key in self.config.observation_keys:
            self._queues[key] = deque(maxlen=self.config.n_obs_steps)
        # if self.config.env_state_feature:
        #     self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}

        if len(self._queues["action"]) == 0:
            obs = self.normalize_inputs(batch)
            # stack n latest observations from the queue
            actions, _ = self.diffusion.generate_actions(obs)

            # TODO(rcadene): make above methods return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action, batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss, vision_ret = self.diffusion.compute_loss(batch)
        loss_dict = {"diffusion_loss": loss.item()}
        # no output_dict so returning None
        if len(vision_ret) > 0 and "loss" in vision_ret:
            loss += vision_ret["loss"]
            # add any keys that have "loss" in them to the loss_dict
            if "attn_loss" in vision_ret:
                loss_dict["attn_loss"] = vision_ret["attn_loss"]
            if "gaze_loss" in vision_ret:
                loss_dict["gaze_loss"] = vision_ret["gaze_loss"]

        return loss, loss_dict
    
    @torch.no_grad
    def get_vision_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)

        _, vision_ret = self.diffusion.vision_encoder(batch)

        return vision_ret


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        if not self.config.tokenize_condition:
            if self.config.robot_state_feature:
                condition_dim = self.config.robot_state_feature.shape[0]

            if self.config.image_features:
                self.vision_encoder = GazeVisionEncoder(self.config.vision)
                if config.use_spatial_softmax:
                    self.pool = SpatialSoftmax(self.vision_encoder.get_feature_map_shape(), num_kp=config.spatial_softmax_num_keypoints)
                    feature_dim = config.spatial_softmax_num_keypoints * 2
                else:
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                    feature_dim = self.vision_encoder.backbone_out_dim
                self.out = nn.Linear(feature_dim, feature_dim)
                num_images = len(self.config.image_features)
                condition_dim += feature_dim * (num_images if not self.config.vision.fuse_images else 1)

            if self.config.env_state_feature:
                condition_dim += self.config.env_state_feature.shape[0]

            condition_dim *= self.config.n_obs_steps

        else:
            # Transformer encoder input projections. The tokens will be structured like
            # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
            if self.config.robot_state_feature:
                self.encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], self.config.difft_hidden_dim
                )
            if self.config.env_state_feature:
                self.encoder_env_state_input_proj = nn.Linear(
                    self.config.env_state_feature.shape[0], self.config.difft_hidden_dim
                )
            if self.config.image_features:
                self.vision_encoder = GazeVisionEncoder(self.config.vision)
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    self.vision_encoder.backbone_out_dim, self.config.difft_hidden_dim, kernel_size=1
                )
            # Transformer encoder positional embeddings.
            self.n_1d_tokens = 0  # for the latent
            if self.config.robot_state_feature:
                self.n_1d_tokens += self.config.n_obs_steps
            if self.config.env_state_feature:
                self.n_1d_tokens += self.config.n_obs_steps
            self.encoder_1d_feature_pos_embed = nn.Embedding(self.n_1d_tokens, self.config.difft_hidden_dim)
            if self.config.image_features:
                self.encoder_cam_feat_pos_embed = SinusoidalPositionEmbedding2d(self.config.difft_hidden_dim // 2)

            condition_dim = self.config.difft_hidden_dim

            
        if self.config.diffusion_net == 'unet':
            self.diffusion_net = DiffusionConditionalUnet1d(
                global_cond_dim=condition_dim,
                action_dim=self.config.action_feature.shape[0],
                down_dims=self.config.unet_down_dims,
                kernel_size=self.config.unet_kernel_size,
                n_groups=self.config.unet_n_groups,
                diffusion_step_embed_dim=self.config.unet_diffusion_step_embed_dim,
                use_film_scale_modulation=self.config.unet_use_film_scale_modulation,
            )
        elif self.config.diffusion_net == 'diffusion_transformer':
            self.diffusion_net = DiffusionTransformer(
                input_dim=self.config.action_feature.shape[0],
                condition_dim=condition_dim,
                hidden_dim=self.config.difft_hidden_dim,
                output_dim=self.config.action_feature.shape[0],
                num_layers=self.config.difft_num_layers,
                num_heads=self.config.difft_num_heads,
                block_type=self.config.difft_block_type,
                mlp_ratio=self.config.difft_mlp_ratio,
                dropout=self.config.difft_dropout,
                time_embed_dim=self.config.difft_time_embed_dim, 
                rope_max_seq_length=self.config.difft_rope_max_seq_length,
            )
        else:
            raise ValueError(f"Unsupported flow network type: {self.config.diffusion_net}")     

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        if not self.config.tokenize_condition:
            """Encode image features and concatenate them all together along with the state vector."""
            if self.config.robot_state_feature:
                global_cond_feats = [batch[OBS_ROBOT]]
            # Extract image features.
            vision_ret = {}
            if self.config.image_features:
                x, vision_ret = self.vision_encoder(batch)
                b, s, n = x.shape[:3]
                x = einops.rearrange(x, "b s n ... -> (b s n) ...")
                x = torch.flatten(self.pool(x), start_dim=1)
                x = nn.ReLU()(self.out(x))
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                x = einops.rearrange(x, "(b s n) ... -> b s (n ...)", b=b, s=s, n=n)
                global_cond_feats.append(x)

            if self.config.env_state_feature:
                global_cond_feats.append(batch[OBS_ENV])

            # Concatenate features then flatten to (B, global_cond_dim).
            return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1), vision_ret

        else:
            # Prepare transformer encoder inputs.
            encoder_in_tokens = []
            encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
            # Robot state token.
            if self.config.robot_state_feature:
                encoder_in_tokens.extend(self.encoder_robot_state_input_proj(batch[OBS_ROBOT].permute(1, 0, 2)))
            # Environment state token.
            if self.config.env_state_feature:
                encoder_in_tokens.extend(
                    self.encoder_env_state_input_proj(batch[OBS_ENV].permute(1, 0, 2))
                )

            # Camera observation features and positional embeddings.
            vision_ret = {}
            if self.config.image_features:
                cam_features, vision_ret = self.vision_encoder(batch)
                b, s, n = cam_features.shape[:3]

                cam_features = einops.rearrange(cam_features, "b s n d h w -> (b s n) d h w")
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                
                cam_features = einops.rearrange(cam_features, "(b s n) d h w -> (s n h w) b d", b=b, s=s, n=n)
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b d h w -> (h w) b d").repeat(s*n, 1, 1)

                encoder_in_tokens.extend(cam_features)
                encoder_in_pos_embed.extend(cam_pos_embed)

            # Stack all tokens along the sequence dimension.
            encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
            encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

            condition = (encoder_in_tokens + encoder_in_pos_embed).permute(1, 0, 2)  # (B, T, D) where T is the total number of tokens.

            return condition, vision_ret

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond, vision_ret = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions, vision_ret

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond, vision_ret = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.diffusion_net(noisy_trajectory, timesteps, global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean(), vision_ret


class GenerativeModel():
    def compute_loss(self, net, target, cond):
        raise NotImplementedError

    def sample(self, model, cond, shape, num_steps):
        raise NotImplementedError
    

class FlowMatchingGenerator(GenerativeModel):
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.FM = get_flow_matcher(
            name=self.config.flow_matcher,
            sigma=self.config.sigma,
            num_sampling_steps=self.config.num_sampling_steps,
        )

    def compute_loss(self, batch, net, condition):
        if ACTION not in batch:
            raise ValueError("Actions must be provided during training")
        loss, metrics = self.FM.compute_loss(net, batch[ACTION], condition)
        return loss

    def sample(self, batch_size, net, condition, num_steps=None):
        shape = (
            batch_size,
            self.config.horizon,
            self.config.action_feature.shape[0],
        )
        return self.FM.sample(net, condition, shape, num_steps=num_steps)
    
class DiffusionGenerator(GenerativeModel):
    def __init__(self, config: DiffusionConfig):
        self.config = config

        self.noise_scheduler = _make_noise_scheduler(
            config.diff_noise_scheduler_type,
            num_train_timesteps=config.diff_num_train_timesteps,
            beta_start=config.diff_beta_start,
            beta_end=config.diff_beta_end,
            beta_schedule=config.diff_beta_schedule,
            clip_sample=config.diff_clip_sample,
            clip_sample_range=config.diff_clip_sample_range,
            prediction_type=config.diff_prediction_type,
        )

        if config.diff_num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.diff_num_inference_steps

    def compute_loss(self, batch, net, condition):
        return self.diffusion.compute_loss(batch)

    def sample(
        self, batch_size, net, condition, num_steps=None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(net)
        dtype = get_dtype_from_parameters(net)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps if not num_steps else num_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = net(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                condition,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample


    