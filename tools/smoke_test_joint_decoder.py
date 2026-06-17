from pathlib import Path
import sys
import types

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The reused decoder blocks import the local mmseg package, whose package
# initializer only needs mmcv/mmengine version metadata before this smoke path
# falls back away from mmcv.ops.
if "mmcv" not in sys.modules:
    mmcv_stub = types.ModuleType("mmcv")
    mmcv_stub.__version__ = "2.0.0rc4"
    sys.modules["mmcv"] = mmcv_stub
if "mmengine" not in sys.modules:
    mmengine_stub = types.ModuleType("mmengine")
    mmengine_stub.__version__ = "0.5.0"
    sys.modules["mmengine"] = mmengine_stub
if "packaging.version" not in sys.modules:
    packaging_stub = types.ModuleType("packaging")
    version_stub = types.ModuleType("packaging.version")

    class _ParsedVersion:
        def __init__(self, version_str):
            self.is_postrelease = False
            self.post = None
            if "rc" in version_str:
                release, rc = version_str.split("rc", 1)
                self.is_prerelease = True
                self.pre = ("rc", int(rc))
            else:
                release = version_str
                self.is_prerelease = False
                self.pre = None
            self.release = tuple(int(part) for part in release.split(".") if part)

    def _parse_version(version_str):
        return _ParsedVersion(version_str)

    version_stub.parse = _parse_version
    packaging_stub.version = version_stub
    sys.modules["packaging"] = packaging_stub
    sys.modules["packaging.version"] = version_stub

from models.Decoders.Snn_Mtscd_JointTemporal_Decoder_V1 import (  # noqa: E402
    MTSCDJointTemporalDecoderNet,
)


def main():
    channels = [32, 64, 128, 360]
    feature_xy = [
        torch.randn(3, 2, 32, 64, 64),
        torch.randn(3, 2, 64, 32, 32),
        torch.randn(3, 2, 128, 16, 16),
        torch.randn(3, 2, 360, 8, 8),
    ]

    model = MTSCDJointTemporalDecoderNet(
        in_channels=channels,
        decoder_channels=64,
        num_sem_classes=13,
        num_change_classes=1,
        input_size=(128, 128),
        feature_order="high_to_low",
    )
    model.eval()

    with torch.no_grad():
        outputs = model(feature_xy)

    assert outputs["sem_logits"].shape == (2, 3, 13, 128, 128), outputs["sem_logits"].shape
    assert outputs["chg_logits"].shape == (2, 1, 128, 128), outputs["chg_logits"].shape
    for phase_name in ("t1", "t2", "t3"):
        assert outputs["sem_logits_dict"][phase_name].shape == (2, 13, 128, 128)

    print("smoke_test_joint_decoder passed")


if __name__ == "__main__":
    main()
