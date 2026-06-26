from pathlib import Path

import torch

from utils.loss import PairwiseRelationAuxLoss_V11
from train_WUSU_ddp_accum_v9 import build_parser, pairrel_warmup


STAT_METRICS = (
    "loss",
    "loss_unchanged",
    "loss_changed",
    "valid_ratio",
    "valid_weight_sum",
    "unchanged_ratio",
    "changed_ratio",
    "ambiguous_ratio",
    "unchanged_weight_sum",
    "changed_weight_sum",
    "dist_unchanged",
    "dist_changed",
    "dist_gap_changed_minus_unchanged",
    "hinge_active_ratio",
    "skipped",
)


def make_features(close_pair=False):
    features = {
        2: torch.randn(3, 2, 128, 32, 32),
        3: torch.randn(3, 2, 360, 16, 16),
    }
    if close_pair:
        features[2][2].copy_(features[2][0])
        features[3][2].copy_(features[3][0])
    return {scale: feature.requires_grad_() for scale, feature in features.items()}


def expected_keys(scales):
    return {
        "pairrel_%s_s%d" % (metric, scale)
        for scale in scales
        for metric in STAT_METRICS
    }


def assert_stats(stats, scales):
    assert expected_keys(scales).issubset(stats)
    assert all(not value.requires_grad for value in stats.values())
    assert all(torch.isfinite(value).all().item() for value in stats.values())


def assert_finite_grad(feature):
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all().item()


def assert_raises_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main():
    torch.manual_seed(0)
    change_mask = torch.zeros(2, 256, 256)
    change_mask[:, 64:160, 64:160] = 1

    features = make_features(close_pair=True)
    criterion = PairwiseRelationAuxLoss_V11()
    assert criterion.mode == "unchanged_only"
    assert not list(criterion.parameters())
    loss, stats = criterion(features, change_mask)
    assert loss.ndim == 0 and torch.isfinite(loss).item()
    assert_stats(stats, (3,))
    assert torch.allclose(loss.detach(), stats["pairrel_loss_unchanged_s3"])
    assert stats["pairrel_loss_changed_s3"].item() > 0.0
    loss.backward()
    assert_finite_grad(features[3])
    assert features[2].grad is None

    features = make_features(close_pair=True)
    loss, stats = PairwiseRelationAuxLoss_V11(
        mode="weak_contrastive",
        changed_weight=0.1,
    )(features, change_mask)
    expected = (
        stats["pairrel_loss_unchanged_s3"]
        + 0.1 * stats["pairrel_loss_changed_s3"]
    )
    assert torch.allclose(loss.detach(), expected)
    assert_stats(stats, (3,))
    loss.backward()
    assert_finite_grad(features[3])

    features = make_features()
    loss, stats = PairwiseRelationAuxLoss_V11(scales=(2, 3))(features, change_mask)
    expected = (
        stats["pairrel_loss_s2"] + stats["pairrel_loss_s3"]
    ) / 2.0
    assert torch.allclose(loss.detach(), expected)
    assert_stats(stats, (2, 3))
    loss.backward()
    assert_finite_grad(features[2])
    assert_finite_grad(features[3])

    missing = {2: torch.randn(3, 2, 128, 32, 32, requires_grad=True)}
    loss, stats = PairwiseRelationAuxLoss_V11()(missing, change_mask)
    assert loss.item() == 0.0 and loss.requires_grad
    assert stats["pairrel_skipped_s3"].item() == 1.0
    assert_stats(stats, (3,))
    loss.backward()
    assert_finite_grad(missing[2])

    no_valid = {3: torch.randn(3, 2, 360, 16, 16, requires_grad=True)}
    ambiguous_mask = torch.full((2, 1, 256, 256), 0.1)
    loss, stats = PairwiseRelationAuxLoss_V11()(no_valid, ambiguous_mask)
    assert loss.item() == 0.0 and loss.requires_grad
    assert stats["pairrel_valid_weight_sum_s3"].item() == 0.0
    assert stats["pairrel_ambiguous_ratio_s3"].item() == 1.0
    assert stats["pairrel_skipped_s3"].item() == 1.0
    assert_stats(stats, (3,))
    loss.backward()
    assert_finite_grad(no_valid[3])

    changed_only = {3: torch.randn(3, 2, 360, 16, 16, requires_grad=True)}
    loss, stats = PairwiseRelationAuxLoss_V11()(changed_only, torch.ones(2, 256, 256))
    assert loss.item() == 0.0 and loss.requires_grad
    assert stats["pairrel_changed_ratio_s3"].item() == 1.0
    assert stats["pairrel_changed_weight_sum_s3"].item() > 0.0
    assert stats["pairrel_valid_weight_sum_s3"].item() == 0.0
    assert stats["pairrel_skipped_s3"].item() == 1.0
    assert_stats(stats, (3,))
    loss.backward()
    assert_finite_grad(changed_only[3])

    features = make_features()
    loss, stats = PairwiseRelationAuxLoss_V11()(features, change_mask.unsqueeze(1))
    assert torch.isfinite(loss).item()
    assert_stats(stats, (3,))
    assert_raises_value_error(
        lambda: PairwiseRelationAuxLoss_V11()(make_features(), change_mask - 0.1)
    )
    assert_raises_value_error(
        lambda: PairwiseRelationAuxLoss_V11()(make_features(), change_mask + 1.1)
    )

    assert pairrel_warmup(9, 10, 10) == 0.0
    assert pairrel_warmup(10, 10, 10) == 0.1
    assert pairrel_warmup(19, 10, 10) == 1.0
    assert pairrel_warmup(10, 10, 0) == 1.0

    args = build_parser().parse_args([])
    assert args.pairrel_mode == "unchanged_only"
    assert args.pairrel_aux_weight == 0.02
    assert args.pairrel_aux_start_epoch == 10
    assert args.pairrel_aux_warmup_epochs == 10
    assert args.pairrel_aux_scales == "3"
    assert args.pairrel_margin == 0.5
    assert args.pairrel_tau_unchanged == 0.05
    assert args.pairrel_tau_changed == 0.50
    assert args.pairrel_changed_weight == 0.0

    model_source = (Path(__file__).resolve().parents[1] / "models" / "GSTMSCD_MTSCD_Snn.py").read_text(
        encoding="utf-8"
    )
    assert "from models.Encoders.FDPC_Encoder import FDPCEncoder" in model_source
    assert "from utils.PAE_NET import PAENTE" in model_source
    assert "relation_mode=\"pdca\"" in model_source
    assert "def forward(self, x, return_aux: bool = False):" in model_source
    assert 'aux["encoder_features"] = {2: feature_xy[2], 3: feature_xy[3]}' in model_source

    train_source = (
        Path(__file__).resolve().parents[1] / "train_WUSU_ddp_accum_v9.py"
    ).read_text(encoding="utf-8")
    assert "MTSCD-PairRelAux-V1.1" in train_source
    assert "PairwiseRelationAuxLoss_V11" in train_source
    assert "PairRel_active" in train_source
    assert "Loss_pairrel_effective" in train_source
    validation_source = train_source.split("def validate_t1_t3", 1)[1].split("def checkpoint_dir", 1)[0]
    assert "return_aux=True" not in validation_source

    print("debug_pairrel_aux V1.1 passed")


if __name__ == "__main__":
    main()
