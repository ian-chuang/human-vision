#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig

import matplotlib.pyplot as plt
import cv2
import imageio
import os

def visualize_policy(
    env: gym.Env,
    policy: PreTrainedPolicy,
    videos_dir: Path,
    options: dict | None = None,
    seed: int | None = None,
) -> dict:
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    policy.eval()

    observation, info = env.reset(seed=seed, options=options)

    step = 0
    # Keep track of which environments are done.
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    gaze_features_video = []
    cropped_images_video = []
    small_images_video = []
    images_video = []
    attn_video = []
    # features_video = []
    while step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)

        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }

        with torch.inference_mode():
            action, batch = policy.select_action(observation)
            vision_ret = {}
            if batch: vision_ret = policy.get_vision_features(batch)

        # Convert to CPU / numpy.
        action = action.to("cpu").numpy()
        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action)

        step += 1
        progbar.update()

        if hasattr(policy.config, "vision"):
            # if "features" in vision_ret:
            #     mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
            #     std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
            #     images = vision_ret['images'].cpu() * std + mean
            #     images = einops.rearrange(images, "b c h w -> h (b w) c")
            #     images = images.numpy()
            #     images = (images * 255).astype(np.uint8)
            #     images = cv2.cvtColor(images, cv2.COLOR_RGB2RGBA)

            #     features = einops.rearrange(vision_ret["features"].cpu(), "b c h w -> h (b w) c").norm(dim=-1).numpy()
            #     features = (features - features.min()) / (features.max() - features.min())
            #     features = plt.cm.plasma(features)
            #     features = (features * 255).astype(np.uint8)
            #     features = cv2.resize(features, (images.shape[1], images.shape[0]), interpolation=cv2.INTER_NEAREST)
            #     features = cv2.addWeighted(images, 0.4, features, 0.6, 0)
            #     features_video.append(features)
                
            if "gaze_features" in vision_ret:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                images = vision_ret['images'].cpu() * std + mean
                images = einops.rearrange(images, "b c h w -> h (b w) c")
                images = images.numpy()
                images = (images * 255).astype(np.uint8)
                images = cv2.cvtColor(images, cv2.COLOR_RGB2RGBA)

                pred_gaze = einops.rearrange(vision_ret["gaze_features"].cpu().squeeze(1), "b h w -> h (b w)").numpy()
                pred_gaze = (pred_gaze - pred_gaze.min()) / (pred_gaze.max() - pred_gaze.min())
                pred_gaze = plt.cm.plasma(pred_gaze)
                pred_gaze = (pred_gaze * 255).astype(np.uint8)
                pred_gaze = cv2.resize(pred_gaze, (images.shape[1], images.shape[0]), interpolation=cv2.INTER_NEAREST)
                pred_gaze = cv2.addWeighted(images, 0.4, pred_gaze, 0.6, 0)
                gaze_features_video.append(pred_gaze)

            if "cropped_images" in vision_ret:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                images = vision_ret['cropped_images'].cpu() * std + mean
                images = einops.rearrange(images, "b c h w -> h (b w) c")
                images = images.numpy()
                images = (images * 255).astype(np.uint8)
                images = cv2.cvtColor(images, cv2.COLOR_RGB2RGBA)
                cur_h, cur_w = vision_ret['cropped_images'].shape[-2:]
                new_h, new_w = vision_ret['images'].shape[-2:]
                h = images.shape[0] * new_h // cur_h
                w = images.shape[1] * new_w // cur_w
                images = cv2.resize(images, (w, h), interpolation=cv2.INTER_NEAREST)
                cropped_images_video.append(images)

            if "small_images" in vision_ret:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                images = vision_ret['small_images'].cpu() * std + mean
                images = einops.rearrange(images, "b c h w -> h (b w) c")
                images = images.numpy()
                images = (images * 255).astype(np.uint8)
                images = cv2.cvtColor(images, cv2.COLOR_RGB2RGBA)
                cur_h, cur_w = vision_ret['small_images'].shape[-2:]
                new_h, new_w = vision_ret['images'].shape[-2:]
                h = images.shape[0] * new_h // cur_h
                w = images.shape[1] * new_w // cur_w
                images = cv2.resize(images, (w, h), interpolation=cv2.INTER_NEAREST)
                small_images_video.append(images)

            if "images" in vision_ret:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                images = vision_ret['images'].cpu() * std + mean
                images = einops.rearrange(images, "b c h w -> h (b w) c")
                images = images.numpy()
                images = (images * 255).astype(np.uint8)
                images = cv2.cvtColor(images, cv2.COLOR_RGB2RGBA)
                images_video.append(images)

            if "attention" in vision_ret and vision_ret["attention"] is not None:
                b, l, a, n, h, w = vision_ret["attention"].shape

                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # Shape (1, C, 1, 1)
                images = vision_ret['images'].cpu() * std + mean
                images = einops.rearrange(images, "b c h w -> h (b w) c").repeat(l, 1, 1)
                images = images.numpy()
                images = (images * 255).astype(np.uint8)
                images = cv2.cvtColor(images, cv2.COLOR_RGB2RGBA)

                attn = einops.rearrange(vision_ret['attention'].cpu(), 'b l a n h w -> b l a n (h w)')
                attn = attn.mean(dim=2) 
                attn = attn[:,:,0,:] 
                attn = torch.nn.functional.softmax(attn, dim=-1) 
                attn = einops.rearrange(attn, 'b l (h w) -> (l h) (b w)', h=h)
                attn = attn.numpy()
                attn = (attn - attn.min()) / (attn.max() - attn.min())
                attn = plt.cm.plasma(attn)
                attn = (attn * 255).astype(np.uint8)
                attn = cv2.resize(attn, (images.shape[1], images.shape[0]), interpolation=cv2.INTER_NEAREST)
                attn = cv2.addWeighted(images, 0.4, attn, 0.6, 0)
                attn_video.append(attn)


    video_paths = []    
    if len(cropped_images_video) > 0:
        video_path = videos_dir / f"gaze.mp4"
        os.makedirs(str(videos_dir), exist_ok=True)
        video_paths.append(str(video_path))
        imageio.mimsave(str(video_path), cropped_images_video, fps=env.unwrapped.metadata["render_fps"])
    if len(attn_video) > 0:
        video_path = videos_dir / f"attn.mp4"
        os.makedirs(str(videos_dir), exist_ok=True)
        video_paths.append(str(video_path))
        imageio.mimsave(str(video_path), attn_video, fps=env.unwrapped.metadata["render_fps"])
    if len(gaze_features_video) > 0:
        video_path = videos_dir / f"gaze_heatmap.mp4"
        os.makedirs(str(videos_dir), exist_ok=True)
        video_paths.append(str(video_path))
        imageio.mimsave(str(video_path), gaze_features_video, fps=env.unwrapped.metadata["render_fps"])
    if len(small_images_video) > 0:
        video_path = videos_dir / f"small_images.mp4"
        os.makedirs(str(videos_dir), exist_ok=True)
        video_paths.append(str(video_path))
        imageio.mimsave(str(video_path), small_images_video, fps=env.unwrapped.metadata["render_fps"])
    if len(images_video) > 0:
        video_path = videos_dir / f"images.mp4"
        os.makedirs(str(videos_dir), exist_ok=True)
        video_paths.append(str(video_path))
        imageio.mimsave(str(video_path), images_video, fps=env.unwrapped.metadata["render_fps"])

    return video_paths


class NoTerminationWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, False, False, info  # Always return False for termination/truncation

@parser.wrap()
def visualize_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    env = NoTerminationWrapper(make_env(cfg.env, n_envs=1, use_async_envs=cfg.eval.use_async_envs))

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        video_paths = visualize_policy(
            env,
            policy,
            videos_dir=Path(cfg.output_dir) / "videos",
            seed=cfg.seed,
        )

    env.close()

    logging.info("End of visualize")


if __name__ == "__main__":
    init_logging()
    visualize_main()
