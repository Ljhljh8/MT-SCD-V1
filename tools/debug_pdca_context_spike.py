import sys
import types
from pathlib import Path

import torch


repo_root = Path(__file__).resolve().parents[1]
mmseg = types.ModuleType("mmseg")
mmseg.__path__ = [str(repo_root / "mmseg")]
sys.modules.setdefault("mmseg", mmseg)

from models.Encoders.phase_deformable_context_attention import PhaseDeformableContextAttention


def phase_names(n):
    return tuple("t%d" % (idx + 1) for idx in range(n))


def adjacent_pairs(names):
    return tuple((names[idx], names[idx + 1]) for idx in range(len(names) - 1))


def assert_stats_finite(stats_by_target):
    assert stats_by_target
    for target, stats in stats_by_target.items():
        assert stats["mode"] in ("none", "weights", "values", "both", "context"), target
        for key, value in stats.items():
            if torch.is_tensor(value):
                assert not value.requires_grad, key
                assert torch.isfinite(value).all(), key


def run_case(mode, n, use_null_source, device):
    names = phase_names(n)
    x = torch.randn(n, 2, 32, 6, 5, device=device, requires_grad=True)
    model = PhaseDeformableContextAttention(
        channels=32,
        phase_names=names,
        context_pairs=adjacent_pairs(names),
        num_heads=4,
        num_points=2,
        hidden_channels=16,
        use_null_source=use_null_source,
        pdca_context_spike_mode=mode,
    ).to(device)

    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    with autocast(enabled=device.type == "cuda"):
        out, aux = model(x, return_aux=True, detach_aux=True)
    assert out.shape == x.shape, (out.shape, x.shape)
    assert torch.isfinite(out).all()
    assert "context_spike" in aux
    assert_stats_finite(aux["context_spike"])
    out.float().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for mode in ("none", "weights", "values", "both", "context"):
        for n in (3, 6):
            for use_null_source in (True, False):
                run_case(mode, n, use_null_source, device)
                print("ok mode=%s N=%d null=%s device=%s" % (mode, n, use_null_source, device))


if __name__ == "__main__":
    main()
