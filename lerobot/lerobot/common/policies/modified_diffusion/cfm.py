import torch
import torch.nn.functional as F
import torchcfm.conditional_flow_matching as cfm
import numpy as np

class BaseFlowMatcher():
    def compute_loss(self, model, target, cond):
        raise NotImplementedError

    def sample(self, model, cond, shape, num_steps, return_traces=False):
        raise NotImplementedError


class ConsistencyFlowMatcher(BaseFlowMatcher):
    def __init__(
        self,
        eps=1e-2,
        num_segments=2,
        boundary=1,
        delta=1e-3,
        alpha=1e-5,
        noise_scale=1.0,
        sigma_var=0.0,
        ode_tol=1e-5,
        num_sampling_steps=1,
    ):
        super().__init__()
        self.eps = eps
        self.num_segments = num_segments
        self.boundary = boundary
        self.delta = delta
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.sigma_var = sigma_var
        self.ode_tol = ode_tol
        self.sigma_t = lambda t: (1. - t) * sigma_var
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model, target, cond):
        """Compute the CFM loss for training."""
        batch_size = target.shape[0]
        device = target.device

        a0 = torch.randn_like(target)
        t = torch.rand(batch_size, device=device) * (1 - self.eps) + self.eps
        r = torch.clamp(t + self.delta, max=1.0)

        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        xt = t_expand * target + (1 - t_expand) * a0
        xr = r_expand * target + (1 - r_expand) * a0

        segments = torch.linspace(0, 1, self.num_segments + 1, device=device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
        segment_ends = segments[seg_indices]
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        x_at_segment_ends = segment_ends_expand * target + (1 - segment_ends_expand) * a0

        vt = model(xt, t, cond)
        vr = model(xr, r, cond)
        vr = torch.nan_to_num(vr)

        ft = self._f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = self._threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, self.boundary, x_at_segment_ends)

        losses_f = torch.mean(torch.square(ft - fr).reshape(batch_size, -1), dim=-1)
        losses_v = self._masked_losses_v(vt, vr, self.boundary, segment_ends, t, batch_size)

        loss = torch.mean(losses_f + self.alpha * losses_v)
        return loss, {
            'loss': loss.item(),
            'flow_loss': torch.mean(losses_f).item(),
            'velocity_loss': torch.mean(losses_v).item()
        }

    def sample(self, model, cond, shape, num_steps=None, return_traces=False):
        """Generate samples, optionally returning traces."""
        if num_steps is None:
            num_steps = self.num_sampling_steps
        noise = torch.randn(shape, device=cond.device)
        z = noise.detach().clone()
        dt = 1.0 / num_steps
        eps = self.eps

        if return_traces:
            traj_history = []
            vel_history = []

        for i in range(num_steps):
            num_t = i / num_steps * (1 - eps) + eps
            t = torch.ones(shape[0], device=cond.device) * num_t
            vt = model(z, t, cond)
            sigma_t = self.sigma_t(num_t)
            if sigma_t > 0:
                pred_sigma = vt + (sigma_t**2) / (2 * (self.noise_scale**2) * ((1-num_t)**2)) * \
                    (0.5 * num_t * (1-num_t) * vt - 0.5 * (2-num_t) * z.detach().clone())
                z = z.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma)
            else:
                z = z.detach().clone() + vt * dt

            if return_traces:
                traj_history.append(z.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return z, (traj_history, vel_history)
        return z

    def _f_euler(self, t_expand, segment_ends_expand, xt, vt):
        return xt + (segment_ends_expand - t_expand) * vt

    def _threshold_based_f_euler(self, t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
        if isinstance(threshold, int) and threshold == 0:
            return x_at_segment_ends
        less_than_threshold = t_expand < threshold
        return less_than_threshold * self._f_euler(t_expand, segment_ends_expand, xt, vt) + \
            (~less_than_threshold) * x_at_segment_ends

    def _masked_losses_v(self, vt, vr, threshold, segment_ends, t, batch_size):
        if isinstance(threshold, int) and threshold == 0:
            return torch.tensor(0.0, device=vt.device)
        t_expand = t.view(-1, 1, 1).repeat(1, vt.shape[1], vt.shape[2])
        less_than_threshold = t_expand < threshold
        far_from_segment_ends = (segment_ends - t) > 1.01 * self.delta
        far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(1, vt.shape[1], vt.shape[2])
        losses_v = torch.square(vt - vr)
        losses_v = less_than_threshold * far_from_segment_ends * losses_v
        return torch.mean(losses_v.reshape(batch_size, -1), dim=-1)


class TorchFlowMatcher(BaseFlowMatcher):
    def __init__(self, fm, num_sampling_steps=6):
        """
        Flow matcher wrapper for torchcfm.
        """
        super().__init__()
        self.fm = fm
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model, target, cond):
        """
        Compute the training loss using the flow matcher.

        Args:
            model: The flow network (e.g., ConditionalUnet1D or FlowTransformer).
            target: Target actions for training.
            cond: Conditioning input.

        Returns:
            Tuple of (loss tensor, dictionary of metrics).
        """
        x0 = torch.randn_like(target)
        timestep, xt, ut = self.fm.sample_location_and_conditional_flow(x0, target)
        vt = model(xt, timestep, cond)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {}

    def sample(self, model, cond, shape, num_steps=None,return_traces=False):
        """
        Generate samples using the flow matcher.

        Args:
            model: The flow network.
            cond: Conditioning input.
            shape: Shape of the output tensor (batch_size, pred_horizon, action_dim).
            return_traces: If True, return trajectory and velocity histories.

        Returns:
            Sampled actions, or (actions, (traj_history, vel_history)) if return_traces is True.
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps
        device = cond.device
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        if return_traces:
            traj_history = []
            vel_history = []

        for t in range(num_steps):
            timestep = torch.ones(x.shape[0], device=x.device) * (t / num_steps)
            vt = model(x, timestep, cond)
            x = x + vt * dt

            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return x, (traj_history, vel_history)
        return x


def get_flow_matcher(**kwargs):
    name = kwargs.pop('name', 'conditional')

    # Customized flow matcher that implements sampling and loss computation
    if name == 'consistency':
        return ConsistencyFlowMatcher(**kwargs)

    num_sampling_steps = kwargs.pop('num_sampling_steps', 6)
    # Wrap torchcfm flow matchers for sampling and loss computation
    CFM_CLASSES = {
        'conditional': cfm.ConditionalFlowMatcher,
        'target': cfm.TargetConditionalFlowMatcher,
        'schrodinger': cfm.SchrodingerBridgeConditionalFlowMatcher,
        'exact': cfm.ExactOptimalTransportConditionalFlowMatcher
    }
    if name not in CFM_CLASSES:
        raise ValueError(f'Invalid flow matcher name: {name}')
    flow_matcher = CFM_CLASSES[name](**kwargs)
    return TorchFlowMatcher(flow_matcher, num_sampling_steps=num_sampling_steps)