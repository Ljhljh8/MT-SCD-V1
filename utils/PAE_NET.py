import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, surrogate, functional

class PAENTE(nn.Module):
    """
    输入:  x [B, N, C, H, W]，WUSU 中 N=3
    输出: j_seq [T, B, Ce, H, W]，以及时间元数据
    """
    def __init__(self, c_in, c_e, n_phase=3, K=4, R=2):
        super().__init__()
        self.n_phase = n_phase
        self.K = K
        self.R = R

        self.stem = nn.Sequential(
            nn.Conv2d(c_in, c_e, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_e),
            nn.GELU(),
            nn.Conv2d(c_e, c_e, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_e),
            nn.GELU(),
        )

        # 从图像/特征差异构建先验
        self.prior_net = nn.Sequential(
            nn.Conv2d(c_in + c_e + c_e, c_e, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_e),
            nn.GELU(),
            nn.Conv2d(c_e, c_e, 1)
        )

        self.phase_emb = nn.Parameter(torch.randn(n_phase + (n_phase - 1), c_e, 1, 1))
        self.local_emb = nn.Parameter(torch.randn(max(K, R), c_e, 1, 1))

        # 时相内动态的通道级系数
        self.a = nn.Parameter(torch.zeros(K, c_e, 1, 1))
        self.b = nn.Parameter(torch.zeros(K, c_e, 1, 1))
        if R > 0:
            self.beta = nn.Parameter(torch.zeros(R, c_e, 1, 1))
            self.omega = nn.Parameter(torch.linspace(0.25, 0.75, steps=R).view(R, 1, 1, 1))
        else:
            self.beta = None
            self.omega = None

        self.feat_mod = nn.Conv2d(c_e, c_e, 1)
        self.prior_mod = nn.Conv2d(c_e, c_e, 1)

    def build_prior(self, xi, xj, fi, fj):
        xdiff = (xi - xj).abs()
        fdiff = (fi - fj).abs()
        prod = fi * fj
        prior = torch.sigmoid(self.prior_net(torch.cat([xdiff, fdiff, prod], dim=1)))
        return prior

    def forward(self, x):
        B, N, C, H, W = x.shape
        feats = [self.stem(x[:, i]) for i in range(N)]

        p12 = self.build_prior(x[:, 0], x[:, 1], feats[0], feats[1])
        p23 = self.build_prior(x[:, 1], x[:, 2], feats[1], feats[2])
        p13 = self.build_prior(x[:, 0], x[:, 2], feats[0], feats[2])

        local_priors = [p12, 0.5 * (p12 + p23), p23]
        seq, phase_idx, step_type, anchor_mask = [], [], [], []

        transition_phase_offset = N  # t1->t2, t2->t3 的嵌入

        for i in range(N):
            fi = feats[i]
            # 硬锚点
            seq.append(fi + self.phase_emb[i])
            phase_idx.append(i)
            step_type.append("anchor")
            anchor_mask.append(1)

            # 时相内动态伴随步
            for m in range(1, self.K):
                j = (
                    fi
                    + self.a[m] * self.feat_mod(fi)
                    + self.b[m] * self.prior_mod(local_priors[i])
                    + self.phase_emb[i]
                    + self.local_emb[m]
                )
                seq.append(j)
                phase_idx.append(i)
                step_type.append("intra")
                anchor_mask.append(0)

            # 仅用于转移的时间步
            if i < N - 1 and self.R > 0:
                fj = feats[i + 1]
                pij = p12 if i == 0 else p23
                trans_id = transition_phase_offset + i
                avg_f = 0.5 * (fi + fj)
                delta_f = fj - fi

                for r in range(self.R):
                    w = torch.sigmoid(self.omega[r])  # 类似标量，可广播
                    j = (
                        (1.0 - pij) * avg_f
                        + pij * ((1.0 - w) * fi + w * fj + self.beta[r] * delta_f)
                        + self.phase_emb[trans_id]
                        + self.local_emb[r]
                    )
                    seq.append(j)
                    phase_idx.append(trans_id)
                    step_type.append("transition")
                    anchor_mask.append(0)

        j_seq = torch.stack(seq, dim=0)  # [T, B, Ce, H, W]
        meta = {
            "phase_idx": phase_idx,
            "step_type": step_type,
            "anchor_mask": torch.tensor(anchor_mask, device=j_seq.device),
            "priors": {"p12": p12, "p23": p23, "p13": p13},
        }
        return j_seq, meta
