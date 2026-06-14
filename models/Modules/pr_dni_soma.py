import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_value, max_value):
        return x.clamp(min_value, max_value).round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def clip_round_ste(x, min_value=0.0, max_value=8.0):
    return ClipRoundSTE.apply(x, float(min_value), float(max_value))


def _valid_group_count(channels, preferred_groups):
    groups = min(int(preferred_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class PRDNISoma(nn.Module):
    """Phase-relational dynamic normalized integer soma.

    The module consumes temporal features in [N, B, C, H, W] order and keeps all
    membrane state local to the current forward call. It does not register
    persistent membrane buffers, so DDP buffer synchronization cannot leak state
    across iterations or ranks.
    """

    def __init__(
        self,
        channels,
        capacity=8,
        theta=1.0,
        rho_min=0.05,
        rho_max=1.0,
        gamma_min=0.0,
        gamma_max=1.0,
        pre_norm="group",
        num_groups=8,
        relation_channels=None,
        gate_hidden_channels=16,
        gate_bias_init=-2.0,
        v_clamp=None,
        eps=1e-6,
    ):
        super(PRDNISoma, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")
        if not (0.0 <= rho_min <= rho_max <= 1.0):
            raise ValueError("rho range must satisfy 0 <= rho_min <= rho_max <= 1")
        if not (0.0 <= gamma_min <= gamma_max):
            raise ValueError("gamma range must satisfy 0 <= gamma_min <= gamma_max")

        self.channels = int(channels)
        self.capacity = int(capacity)
        self.theta = float(theta)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.eps = float(eps)
        self.v_clamp = None if v_clamp is None else float(v_clamp)

        if pre_norm == "group":
            groups = _valid_group_count(self.channels, num_groups)
            self.pre_norm = nn.GroupNorm(groups, self.channels)
        elif pre_norm == "batch":
            self.pre_norm = nn.BatchNorm2d(self.channels)
        elif pre_norm == "identity":
            self.pre_norm = nn.Identity()
        else:
            raise ValueError("pre_norm must be one of: group, batch, identity")

        if relation_channels is None:
            relation_channels = self.channels
        if relation_channels <= 0:
            raise ValueError("relation_channels must be positive")
        if relation_channels == self.channels:
            self.feature_proj = nn.Identity()
        else:
            self.feature_proj = nn.Conv2d(self.channels, int(relation_channels), kernel_size=1, bias=False)

        if gate_hidden_channels is not None and gate_hidden_channels > 0:
            hidden = int(gate_hidden_channels)
            self.gate = nn.Sequential(
                nn.Conv2d(5, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )
            nn.init.constant_(self.gate[-1].bias, float(gate_bias_init))
        else:
            self.gate = nn.Sequential(nn.Conv2d(5, 1, kernel_size=1))
            nn.init.constant_(self.gate[0].bias, float(gate_bias_init))

    def gate_from_relation_cue(self, relation_cue):
        if relation_cue.dim() != 4 or relation_cue.size(1) != 5:
            raise ValueError("relation_cue must have shape [B, 5, H, W]")

        gate_param = next(self.gate.parameters())
        gate_input = relation_cue.to(dtype=gate_param.dtype)
        gate_logits = self.gate(gate_input).clamp(-12.0, 12.0)
        c = torch.sigmoid(gate_logits)
        r = 1.0 - c
        rho = self.rho_min + r * (self.rho_max - self.rho_min)
        gamma = self.gamma_min + c * (self.gamma_max - self.gamma_min)
        return {
            "c": c,
            "r": r,
            "rho": rho,
            "gamma": gamma,
            "gate_logits": gate_logits,
        }

    def _phase_pre(self, frame):
        return F.relu(self.pre_norm(frame.float()))

    def _spatial_activity(self, x):
        if x.size(-2) > 1:
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            dy = F.pad(dy, (0, 0, 1, 0))
        else:
            dy = torch.zeros_like(x)
        if x.size(-1) > 1:
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dx = F.pad(dx, (1, 0, 0, 0))
        else:
            dx = torch.zeros_like(x)
        grad = torch.sqrt(dx.pow(2) + dy.pow(2) + self.eps)
        return grad.mean(dim=1, keepdim=True)

    def _relation_cue(self, prev_frame, frame, prev_s, s_pre, prev_v_post, v_pre):
        theta_eps = self.theta + self.eps
        psi_prev = self.feature_proj(prev_frame.float())
        psi_curr = self.feature_proj(frame.float())

        delta_s = (prev_s.detach() - s_pre.detach()).abs().mean(dim=1, keepdim=True)
        delta_f = (psi_prev - psi_curr).abs().mean(dim=1, keepdim=True)
        delta_v = (prev_v_post / theta_eps - v_pre / theta_eps).abs().mean(dim=1, keepdim=True)
        eta = torch.max(prev_s.detach().abs(), s_pre.detach().abs()).mean(dim=1, keepdim=True)
        beta = torch.max(self._spatial_activity(psi_prev), self._spatial_activity(psi_curr))

        return torch.cat([delta_s, delta_f, delta_v, eta, beta], dim=1)

    def _first_phase_gate(self, frame):
        b, _, h, w = frame.shape
        cue = frame.new_zeros((b, 5, h, w), dtype=torch.float32)
        c = frame.new_zeros((b, 1, h, w), dtype=torch.float32)
        r = frame.new_ones((b, 1, h, w), dtype=torch.float32)
        rho = frame.new_full((b, 1, h, w), self.rho_max, dtype=torch.float32)
        gamma = frame.new_full((b, 1, h, w), self.gamma_min, dtype=torch.float32)
        logits = frame.new_full((b, 1, h, w), -12.0, dtype=torch.float32)
        return {
            "relation_cue": cue,
            "c": c,
            "r": r,
            "rho": rho,
            "gamma": gamma,
            "gate_logits": logits,
        }

    def forward(self, x, return_state=False, return_mixed=False):
        if x.dim() != 5:
            raise ValueError("PRDNISoma expects input shape [N, B, C, H, W]")
        n_phase, batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError("input channel count does not match module channels")

        x_float = x.float()
        theta_eps = self.theta + self.eps

        s_list = []
        k_list = []
        d_list = []
        v_pre_list = []
        u_list = []
        v_post_list = []
        c_list = []
        r_list = []
        rho_list = []
        gamma_list = []
        gate_logits_list = []
        relation_cue_list = []

        prev_s = None
        prev_v_post = None

        for idx in range(n_phase):
            frame = x_float[idx]
            v_pre = self._phase_pre(frame)
            s_pre = (v_pre / theta_eps).clamp(0.0, 1.0)

            if idx == 0:
                gate = self._first_phase_gate(frame)
                u = v_pre
            else:
                relation_cue = self._relation_cue(x_float[idx - 1], frame, prev_s, s_pre, prev_v_post, v_pre)
                gate = self.gate_from_relation_cue(relation_cue)
                gate["relation_cue"] = relation_cue
                rho = gate["rho"].to(dtype=v_pre.dtype)
                u = rho * prev_v_post + v_pre

            k = clip_round_ste(F.relu(u) / theta_eps, 0.0, float(self.capacity))
            d = x_float.new_full((batch, 1, 1, 1), float(self.capacity))
            s = k / (d + self.eps)
            gamma = gate["gamma"].to(dtype=u.dtype)
            v_post = u - gamma * self.theta * k
            if self.v_clamp is not None:
                v_post = v_post.clamp(-self.v_clamp, self.v_clamp)

            s_list.append(s)
            k_list.append(k)
            d_list.append(d)
            v_pre_list.append(v_pre)
            u_list.append(u)
            v_post_list.append(v_post)
            c_list.append(gate["c"].to(dtype=x_float.dtype))
            r_list.append(gate["r"].to(dtype=x_float.dtype))
            rho_list.append(gate["rho"].to(dtype=x_float.dtype))
            gamma_list.append(gate["gamma"].to(dtype=x_float.dtype))
            gate_logits_list.append(gate["gate_logits"].to(dtype=x_float.dtype))
            relation_cue_list.append(gate["relation_cue"].to(dtype=x_float.dtype))

            prev_s = s
            prev_v_post = v_post

        state = {
            "S": torch.stack(s_list, dim=0),
            "K": torch.stack(k_list, dim=0),
            "D": torch.stack(d_list, dim=0),
            "V_pre": torch.stack(v_pre_list, dim=0),
            "U": torch.stack(u_list, dim=0),
            "V_post": torch.stack(v_post_list, dim=0),
            "c": torch.stack(c_list, dim=0),
            "r": torch.stack(r_list, dim=0),
            "rho": torch.stack(rho_list, dim=0),
            "gamma": torch.stack(gamma_list, dim=0),
            "gate_logits": torch.stack(gate_logits_list, dim=0),
            "relation_cue": torch.stack(relation_cue_list, dim=0),
        }

        if return_mixed:
            v_proxy = (state["V_post"] / theta_eps).clamp(-1.0, 1.0)
            state["mixed"] = torch.cat([state["S"], v_proxy], dim=2)

        if return_state:
            return state
        if return_mixed:
            return state["mixed"]
        return state["S"]


__all__ = ["PRDNISoma", "ClipRoundSTE", "clip_round_ste"]
