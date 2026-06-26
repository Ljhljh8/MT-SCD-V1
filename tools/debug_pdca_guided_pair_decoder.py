import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def install_decoder_import_shims():
    mmseg = types.ModuleType("mmseg")
    mmseg.__path__ = [str(REPO_ROOT / "mmseg")]
    sys.modules.setdefault("mmseg", mmseg)

    qtrick = types.ModuleType("mmseg.Qtrick_architecture")
    clock = types.ModuleType("mmseg.Qtrick_architecture.clock_driven")
    neuron = types.ModuleType("mmseg.Qtrick_architecture.clock_driven.neuron")
    surrogate = types.ModuleType("mmseg.Qtrick_architecture.clock_driven.surrogate")
    dend = types.ModuleType("models.dendsn_lifFADC_Snn_v2")

    class Q_IFNode(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    class Quant:
        pass

    class Quant4:
        pass

    class DendFADCConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )

        def forward(self, x, k=None):
            t, b, c, h, w = x.shape
            y = self.conv(x.flatten(0, 1)).view(t, b, -1, h, w)
            return y, k

    neuron.Q_IFNode = Q_IFNode
    surrogate.Quant = Quant
    surrogate.Quant4 = Quant4
    dend.DendFADCConv2d = DendFADCConv2d

    sys.modules.setdefault("mmseg.Qtrick_architecture", qtrick)
    sys.modules.setdefault("mmseg.Qtrick_architecture.clock_driven", clock)
    sys.modules["mmseg.Qtrick_architecture.clock_driven.neuron"] = neuron
    sys.modules["mmseg.Qtrick_architecture.clock_driven.surrogate"] = surrogate
    sys.modules["models.dendsn_lifFADC_Snn_v2"] = dend


install_decoder_import_shims()

from models.Decoders.Snn_Mtscd_Decoder_V2 import MTSCDDecoderNet  # noqa: E402
from utils.loss import (  # noqa: E402
    PairwiseBinaryChangeLoss,
    make_pairwise_change_targets,
)


PAIR_KEYS = ("t1_to_t2", "t2_to_t3", "t1_to_t3")
IN_CHANNELS = (4, 8, 12, 16)


def make_features(batch_size):
    sizes = (16, 8, 4, 2)
    return [
        torch.randn(3, batch_size, channels, size, size)
        for channels, size in zip(IN_CHANNELS, sizes)
    ]


def make_pdca_aux(feature_xy, requires_grad=False):
    source_names = {
        "t1": ("t2", "t3", "__null__"),
        "t2": ("t1", "t3", "__null__"),
        "t3": ("t1", "t2", "__null__"),
    }
    source_weights = {}
    source_names_by_scale = {}
    for scale in (2, 3):
        feat = feature_xy[scale]
        _, b, _, h, w = feat.shape
        per_target = {}
        for target_name in ("t1", "t2", "t3"):
            weights = torch.softmax(torch.rand(b, 2, 3, h, w), dim=2)
            per_target[target_name] = weights.detach().requires_grad_(requires_grad)
        source_weights[str(scale)] = per_target
        source_names_by_scale[str(scale)] = source_names
    return {
        "pdca_source_weights": source_weights,
        "pdca_source_names_by_target": source_names_by_scale,
    }


def build_model(use_pair_decoder, use_pdca_guidance=True):
    return MTSCDDecoderNet(
        in_channels=IN_CHANNELS,
        decoder_channels=8,
        num_sem_classes=13,
        num_change_classes=1,
        input_size=(32, 32),
        phase_windows={"t1": [0], "t2": [1], "t3": [2]},
        transition_windows={"t1_to_t2": None, "t2_to_t3": None, "t1_to_t3": None},
        temporal_readout="mean",
        diff_mode="abs_signed",
        share_semantic_decoder=True,
        feature_order="high_to_low",
        use_transition_fusion=False,
        return_intermediates_default=False,
        use_pdca_guided_pair_decoder=use_pair_decoder,
        detach_pdca_guidance=True,
        use_pdca_guidance=use_pdca_guidance,
    )


def assert_pair_logits(outputs, batch_size):
    assert tuple(outputs["change_logits_dict"].keys()) == PAIR_KEYS
    for key in PAIR_KEYS:
        assert outputs["change_logits_dict"][key].shape == (batch_size, 1, 32, 32)
        assert torch.isfinite(outputs["change_logits_dict"][key]).all().item()
    assert torch.allclose(outputs["chg_logits"], outputs["change_logits_dict"]["t1_to_t3"])


def run_legacy_path():
    model = build_model(use_pair_decoder=False)
    outputs = model(make_features(batch_size=1), input_size=(32, 32))
    assert outputs["chg_logits"].shape == (1, 1, 32, 32)
    assert tuple(outputs["change_logits_dict"]) == ("t1_to_t3",)
    assert not any(name.startswith("pair_change_decoder.") for name, _ in model.named_parameters())


def run_pair_path(batch_size, use_pdca_guidance):
    model = build_model(use_pair_decoder=True, use_pdca_guidance=use_pdca_guidance)
    assert not any(
        name.startswith("change_decoder.") or name.startswith("change_head.")
        for name, _ in model.named_parameters()
    )
    feature_xy = make_features(batch_size=batch_size)
    pdca_aux = make_pdca_aux(feature_xy, requires_grad=True) if use_pdca_guidance else {}
    outputs = model(feature_xy, input_size=(32, 32), pdca_aux=pdca_aux)
    assert_pair_logits(outputs, batch_size)
    assert tuple(outputs["pair_gate_debug"]["gate"].keys()) == PAIR_KEYS
    for key in PAIR_KEYS:
        assert len(outputs["pair_gate_debug"]["gate"][key]) == len(IN_CHANNELS)
        for gate in outputs["pair_gate_debug"]["gate"][key]:
            assert gate.shape[0] == batch_size
            assert gate.shape[1] == 1
            assert torch.isfinite(gate).all().item()

    sem_targets = {
        "t1": torch.randint(-1, 13, (batch_size, 32, 32)),
        "t2": torch.randint(-1, 13, (batch_size, 32, 32)),
        "t3": torch.randint(-1, 13, (batch_size, 32, 32)),
    }
    pair_targets = make_pairwise_change_targets(sem_targets, ignore_index=-1)
    loss, stats = PairwiseBinaryChangeLoss()(outputs["change_logits_dict"], pair_targets)
    assert loss.ndim == 0 and torch.isfinite(loss).item()
    assert "pair_bcd_loss_t1_to_t3" in stats
    loss.backward()
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)
    if use_pdca_guidance:
        for scale_weights in pdca_aux["pdca_source_weights"].values():
            for weights in scale_weights.values():
                assert weights.grad is None


def run_missing_guidance_error():
    model = build_model(use_pair_decoder=True, use_pdca_guidance=True)
    try:
        model(make_features(batch_size=1), input_size=(32, 32), pdca_aux={})
    except RuntimeError as exc:
        assert "pdca_source_weights" in str(exc)
        return
    raise AssertionError("missing PDCA guidance should raise RuntimeError")


def run_git_diff_check():
    safe_directory = str(REPO_ROOT).replace("\\", "/")
    result = subprocess.run(
        ["git", "-c", "safe.directory=" + safe_directory, "diff", "--check"],
        cwd=str(REPO_ROOT),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout)


def main():
    torch.manual_seed(7)
    run_legacy_path()
    run_pair_path(batch_size=1, use_pdca_guidance=True)
    run_pair_path(batch_size=2, use_pdca_guidance=True)
    run_missing_guidance_error()
    run_pair_path(batch_size=1, use_pdca_guidance=False)
    run_git_diff_check()
    print("debug_pdca_guided_pair_decoder passed")


if __name__ == "__main__":
    main()
