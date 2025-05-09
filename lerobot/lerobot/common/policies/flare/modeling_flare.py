from collections import deque
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from lerobot.common.constants import OBS_ENV, OBS_ROBOT, ACTION
from lerobot.common.policies.flare.configuration_flare import FlareConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.common.policies.flare.unet import DiffusionConditionalUnet1d
from lerobot.common.policies.flare.transformer import FlowTransformer
import torchcfm.conditional_flow_matching as cfm

class FlarePolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = FlareConfig
    name = "flare"

    def __init__(
        self,
        config: FlareConfig,
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

        self.flow = FlowModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        starting_key = "flow.net.vision_encoder.backbone"
        if self.config.image_features:
            if not any(n.startswith(starting_key) for n, p in self.named_parameters()):
                raise ValueError("No parameters found for vision encoder backbone. Please check your configuration.")

        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith(starting_key) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith(starting_key) and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        for key in self.config.observation_keys:
            self._queues[key] = deque(maxlen=(self.config.n_obs_steps-1)*self.config.n_obs_step_size+1)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        self._queues = populate_queues(self._queues, batch)
        batch = {k: torch.stack(
            list(self._queues[k])[::self.config.n_obs_step_size],  # every n_obs_step_size steps
            dim=1
        ) for k in batch if k in self._queues}

        if len(self._queues["action"]) == 0:
            obs = self.normalize_inputs(batch)
            
            actions = self.flow.generate_actions(obs)[:, : self.config.n_action_steps]

            # TODO(rcadene): make above methods return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action, batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss, vision_ret = self.flow.compute_loss(batch)
        loss_dict = {"velocity_loss": loss.item()}
        # no output_dict so returning None
        if len(vision_ret) > 0 and "loss" in vision_ret:
            loss += vision_ret["loss"]
            for key in vision_ret:
                if key.endswith("_loss"):
                    loss_dict[key] = vision_ret[key]
        return loss, loss_dict
    
    @torch.no_grad
    def get_vision_features(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        batch = self.normalize_inputs(batch)
        _, vision_ret = self.flow.net.vision_encoder(batch)
        return vision_ret
    
def _make_flow_matcher(name: str, **kwargs: dict) -> cfm.ConditionalFlowMatcher:
    if name == "conditional":
        return cfm.ConditionalFlowMatcher(**kwargs)
    elif name == "target":
        return cfm.TargetConditionalFlowMatcher(**kwargs)
    elif name == "schrodinger":
        return cfm.SchrodingerBridgeConditionalFlowMatcher(**kwargs)
    elif name == "exact":
        return cfm.ExactOptimalTransportConditionalFlowMatcher(**kwargs)
    else:
        raise ValueError(f"Unsupported flow matcher type {name}")

class FlowModel(nn.Module):
    def __init__(self, config: FlareConfig):
        super().__init__()
        self.config = config

        if config.flow_net == "unet":
            self.net = DiffusionConditionalUnet1d(config)
        elif config.flow_net == "transformer":
            self.net = FlowTransformer(config)
        else:
            raise ValueError(f"Unsupported flow network type {config.flow_net}")

        self.flow_matcher = _make_flow_matcher(
            config.flow_matcher,
            **config.flow_matcher_kwargs,
        )

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        num_steps = self.config.num_sampling_steps
        shape = (batch_size, self.config.chunk_size, self.config.action_feature.shape[0])
        x = torch.randn(shape, device=device, dtype=dtype)
        dt = 1.0 / num_steps

        for t in range(num_steps):
            timestep = torch.ones(x.shape[0], device=x.device) * (t / num_steps)
            vt = self.net(x, timestep, global_cond=global_cond)
            x = x + vt * dt

        return x

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
        global_cond, _ = self.net.prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        return actions

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
        chunk_size = batch["action"].shape[1]
        assert chunk_size == self.config.chunk_size
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond, vision_ret = self.net.prepare_global_conditioning(batch)  # (B, global_cond_dim)

        x0 = torch.randn_like(batch[ACTION])
        timestep, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, batch[ACTION])
        vt = self.net(xt, timestep, global_cond=global_cond)
        loss = F.mse_loss(vt, ut, reduction="none")

        return loss.mean(), vision_ret
    

