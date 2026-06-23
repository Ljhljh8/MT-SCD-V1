from pathlib import Path

import torch

from utils.loss import PairwiseRelationAuxLoss


def make_features():
    return {
        2: torch.randn(3, 2, 128, 32, 32, requires_grad=True),
        3: torch.randn(3, 2, 360, 16, 16, requires_grad=True),
    }


def expected_keys(scales):
    metrics = (
        "loss",
        "valid_ratio",
        "valid_weight_sum",
        "dist_unchanged",
        "dist_changed",
        "skipped",
    )
    return {"pairrel_%s_s%d" % (metric, scale) for scale in scales for metric in metrics}


def main():
    torch.manual_seed(0)
    change_mask = torch.zeros(2, 256, 256)
    change_mask[:, 64:160, 64:160] = 1

    features = make_features()
    criterion = PairwiseRelationAuxLoss()
    assert not list(criterion.parameters())
    loss, stats = criterion(features, change_mask)
    assert loss.ndim == 0 and torch.isfinite(loss).item()
    assert expected_keys((3,)).issubset(stats)
    assert all(not value.requires_grad for value in stats.values())
    loss.backward()
    assert features[3].grad is not None
    assert features[2].grad is None

    features = make_features()
    loss, stats = PairwiseRelationAuxLoss(scales=(2, 3))(features, change_mask)
    assert loss.ndim == 0 and torch.isfinite(loss).item()
    assert expected_keys((2, 3)).issubset(stats)
    loss.backward()
    assert features[2].grad is not None and features[3].grad is not None

    missing = {2: torch.randn(3, 2, 128, 32, 32, requires_grad=True)}
    loss, stats = PairwiseRelationAuxLoss()(missing, change_mask)
    assert loss.item() == 0.0 and loss.requires_grad
    assert stats["pairrel_skipped_s3"].item() == 1.0
    loss.backward()
    assert missing[2].grad is not None

    no_valid = {3: torch.randn(3, 2, 360, 16, 16, requires_grad=True)}
    threshold_gap_mask = torch.full((2, 1, 256, 256), 0.1)
    loss, stats = PairwiseRelationAuxLoss()(no_valid, threshold_gap_mask)
    assert loss.item() == 0.0 and loss.requires_grad
    assert stats["pairrel_valid_weight_sum_s3"].item() == 0.0
    assert stats["pairrel_skipped_s3"].item() == 1.0
    loss.backward()
    assert no_valid[3].grad is not None

    model_source = (Path(__file__).resolve().parents[1] / "models" / "GSTMSCD_MTSCD_Snn.py").read_text(
        encoding="utf-8"
    )
    assert "from models.Encoders.FDPC_Encoder import FDPCEncoder" in model_source
    assert "from utils.PAE_NET import PAENTE" in model_source
    assert "relation_mode=\"pdca\"" in model_source
    assert "def forward(self, x, return_aux: bool = False):" in model_source
    assert "detach_aux=True" in model_source
    assert '"encoder_features": {2: feature_xy[2], 3: feature_xy[3]}' in model_source

    train_source = (
        Path(__file__).resolve().parents[1] / "train_WUSU_ddp_accum_v8.py"
    ).read_text(encoding="utf-8")
    assert "MTSCD-PairRelAux-V1 experimental training entrypoint" in train_source
    assert 'choices=["prg", "pdca", "none"], default="pdca"' in train_source
    assert '"--enable-pairrel-aux", action="store_true"' in train_source
    assert '"--pairrel-aux-scales", type=str, default="3"' in train_source
    assert "relation_mode=args.relation_mode" in train_source
    assert "ctx.model(x, return_aux=True)" in train_source
    validation_source = train_source.split("def validate_t1_t3", 1)[1].split("def checkpoint_dir", 1)[0]
    assert "return_aux=True" not in validation_source

    print("debug_pairrel_aux passed")


if __name__ == "__main__":
    main()
