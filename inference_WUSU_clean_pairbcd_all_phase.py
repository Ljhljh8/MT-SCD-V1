#!/usr/bin/env python3
"""Single-GPU all-phase validation for the clean WUSU PairBCD model.

Example:
    python inference_WUSU_clean_pairbcd_all_phase.py --checkpoint checkpoints_clean_pairbcd/latest.pth --data-root /path/to/WUSU_data --output-dir pred_clean_pairbcd
"""

import argparse
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from spikingjelly.clock_driven import functional
from torch.utils.data import DataLoader

import datasets.MultiSiamese_RS_ST_TL as RS
from models.GSTMSCD_MTSCD_Snn_ForDecoder_clean import GSTMSCD_WUSU
from utils.metric import IOUandSek
from utils.palette import color_map_WUSU13


PAIR_SPECS = {
    "t1_to_t2": ("t1", "t2"),
    "t2_to_t3": ("t2", "t3"),
    "t1_to_t3": ("t1", "t3"),
}
PHASE_DIRS = {"t1": "time_1", "t2": "time_2", "t3": "time_3"}
METRIC_FIELDS = (
    ("miou", "mIoU"),
    ("sek", "SeK"),
    ("Fscd", "Fscd"),
    ("OA", "OA"),
    ("score", "Score"),
    ("SC_Precision", "SC Precision"),
    ("SC_Recall", "SC Recall"),
    ("change_ratio", "Change Ratio"),
)


def str2bool(value):
    """Parse common command-line boolean spellings."""
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("invalid boolean value: %r" % value)


def build_parser():
    """Build the standalone inference argument parser."""
    parser = argparse.ArgumentParser("Clean PairBCD WUSU all-phase validation")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="pred_results_clean_pairbcd_all_phase")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--change-threshold", type=float, default=0.5)
    parser.add_argument("--backbone", default="sdtv2")
    parser.add_argument(
        "--dend-spatial-conv-type",
        choices=(
            "fadc",
            "structure_routed_v1",
            "structure_routed_v2",
            "structure_routed_v3",
        ),
        default="fadc",
    )
    parser.add_argument(
        "--routeconv-ablation-mode",
        choices=(
            "full",
            "uniform_route",
            "global_route",
            "no_axis_descriptor",
            "isotropic_direction_pool",
        ),
        default="full",
    )
    parser.add_argument(
        "--routeconv-v2-mode",
        choices=("v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v2_6"),
        default="v2_6",
    )
    parser.add_argument(
        "--routeconv-v3-mode",
        choices=("v3_1", "v3_2", "v3_3", "v3_4", "v3_5", "v3_6"),
        default="v3_6",
    )
    parser.add_argument("--dend-residual-init", type=float, default=0.0)

    parser.add_argument(
        "--pdca-dend-prior-mode",
        default="offset_residual",
        choices=(
            "none",
            "source",
            "source_gain",
            "offset_sim",
            "offset_dual",
            "offset_residual",
            "offset_improve",
            "offset_gate",
        ),
    )
    parser.add_argument("--pdca-dend-prior-alpha", type=float, default=1e-3)
    parser.add_argument("--pdca-dend-prior-detach", type=str2bool, default=True)
    parser.add_argument(
        "--pdca-dend-prior-descriptor",
        default="mean_std",
        choices=("mean", "mean_std", "raw", "delta", "gain"),
    )
    parser.add_argument(
        "--pdca-dend-prior-normalize",
        default="zscore",
        choices=("none", "zscore"),
    )
    parser.add_argument("--pdca-dend-prior-source-weight", type=float, default=1.0)
    parser.add_argument("--pdca-dend-prior-point-weight", type=float, default=0.25)
    parser.add_argument("--pdca-dend-prior-sim-weight", type=float, default=1.0)
    parser.add_argument("--pdca-dend-prior-diff-weight", type=float, default=0.25)
    parser.add_argument("--pdca-dend-prior-use-conf-gate", type=str2bool, default=True)
    parser.add_argument("--pdca-dend-prior-conf-beta", type=float, default=4.0)
    parser.add_argument("--pdca-dend-prior-conf-tau", type=float, default=0.10)
    parser.add_argument("--pdca-dend-prior-use-offset-gate", type=str2bool, default=True)
    parser.add_argument("--pdca-dend-prior-center-point", type=str2bool, default=True)
    parser.add_argument("--pdca-dend-prior-clip", type=float, default=2.0)
    parser.add_argument("--pdca-dend-prior-affect-null", type=str2bool, default=False)
    parser.add_argument("--pdca-dend-prior-stats", type=str2bool, default=False)
    return parser


def build_model(args, device):
    """Construct the clean model with the training-time forward configuration."""
    model = GSTMSCD_WUSU(
        backbone=args.backbone,
        pretrained=False,
        nclass=len(RS.ST_CLASSES),
        relation_mode="pdca",
        use_pdca_guided_pair_decoder=True,
        detach_pdca_guidance=True,
        use_pdca_guidance=True,
        pdca_dend_prior_mode=args.pdca_dend_prior_mode,
        pdca_dend_prior_alpha=args.pdca_dend_prior_alpha,
        pdca_dend_prior_detach=args.pdca_dend_prior_detach,
        pdca_dend_prior_descriptor=args.pdca_dend_prior_descriptor,
        pdca_dend_prior_normalize=args.pdca_dend_prior_normalize,
        pdca_dend_prior_source_weight=args.pdca_dend_prior_source_weight,
        pdca_dend_prior_point_weight=args.pdca_dend_prior_point_weight,
        pdca_dend_prior_sim_weight=args.pdca_dend_prior_sim_weight,
        pdca_dend_prior_diff_weight=args.pdca_dend_prior_diff_weight,
        pdca_dend_prior_use_conf_gate=args.pdca_dend_prior_use_conf_gate,
        pdca_dend_prior_conf_beta=args.pdca_dend_prior_conf_beta,
        pdca_dend_prior_conf_tau=args.pdca_dend_prior_conf_tau,
        pdca_dend_prior_use_offset_gate=args.pdca_dend_prior_use_offset_gate,
        pdca_dend_prior_center_point=args.pdca_dend_prior_center_point,
        pdca_dend_prior_clip=args.pdca_dend_prior_clip,
        pdca_dend_prior_affect_null=args.pdca_dend_prior_affect_null,
        pdca_dend_prior_stats=args.pdca_dend_prior_stats,
        dend_spatial_conv_type=args.dend_spatial_conv_type,
        routeconv_ablation_mode=args.routeconv_ablation_mode,
        routeconv_v2_mode=args.routeconv_v2_mode,
        routeconv_v3_mode=args.routeconv_v3_mode,
        dend_residual_init=args.dend_residual_init,
    )
    return model.to(device)


def load_checkpoint(model, checkpoint_path):
    """Load a model/model-state/raw-state checkpoint strictly."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if state_dict and any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module.") :] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)


def build_dataloader(args, device):
    """Build the WUSU validation dataset and its single inference loader."""
    if args.data_root is not None:
        RS.root = args.data_root
    with redirect_stdout(io.StringIO()):
        dataset = RS.Data(mode="val", random_flip=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=args.workers,
        drop_last=False,
    )
    return dataset, loader


def prepare_output_dirs(output_dir):
    """Create the requested output tree once and return its paths."""
    root = Path(output_dir)
    paths = {
        "root": root,
        "semantic_predictions": {
            phase: root / "semantic_predictions" / phase_dir
            for phase, phase_dir in PHASE_DIRS.items()
        },
        "semantic_ground_truth": {
            phase: root / "semantic_ground_truth" / phase_dir
            for phase, phase_dir in PHASE_DIRS.items()
        },
        "pairs": {},
    }
    directories = [root]
    directories.extend(paths["semantic_predictions"].values())
    directories.extend(paths["semantic_ground_truth"].values())
    for pair_key, (phase_i, phase_j) in PAIR_SPECS.items():
        pair_root = root / ("pair_" + pair_key)
        pair_paths = {
            "change_prediction": pair_root / "change_prediction",
            "change_ground_truth": pair_root / "change_ground_truth",
            "semantic_i_masked": pair_root / ("semantic_" + phase_i + "_masked"),
            "semantic_j_masked": pair_root / ("semantic_" + phase_j + "_masked"),
        }
        paths["pairs"][pair_key] = pair_paths
        directories.extend(pair_paths.values())
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def save_index_png(array, path, palette):
    """Save one uint8 index map as a paletted PNG."""
    image = Image.fromarray(array.astype(np.uint8, copy=False)).convert("P")
    image.putpalette(palette)
    image.save(path, format="PNG")


def save_binary_png(array, path):
    """Save one binary map as a 0/255 grayscale PNG."""
    Image.fromarray(array.astype(np.uint8, copy=False) * 255).save(path, format="PNG")


def metric_result(metric):
    """Return evaluate_SECOND outputs with their exact project names."""
    change_ratio, score, miou, sek, fscd, oa, sc_precision, sc_recall = (
        metric.evaluate_SECOND()
    )
    return {
        "score": float(score),
        "miou": float(miou),
        "sek": float(sek),
        "Fscd": float(fscd),
        "OA": float(oa),
        "SC_Precision": float(sc_precision),
        "SC_Recall": float(sc_recall),
        "change_ratio": float(change_ratio),
    }


def sample_png_name(sample_id):
    """Preserve the dataset sample stem while writing an actual PNG file."""
    return Path(Path(str(sample_id)).name).with_suffix(".png").name


@torch.inference_mode()
def run_validation(args, model, loader, device, output_paths, palette):
    """Run one forward per batch, save outputs, and accumulate all metrics."""
    model.eval()
    primary_metric = IOUandSek(num_classes=len(RS.ST_CLASSES))
    pair_metrics = {
        pair_key: IOUandSek(num_classes=len(RS.ST_CLASSES)) for pair_key in PAIR_SPECS
    }
    non_blocking = device.type == "cuda"
    if non_blocking:
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for batch in loader:
        img1, img2, img3, label1, label2, label3, label_bn, sample_ids = batch
        img1 = img1.float().to(device, non_blocking=non_blocking)
        img2 = img2.float().to(device, non_blocking=non_blocking)
        img3 = img3.float().to(device, non_blocking=non_blocking)
        x = torch.stack([img1, img2, img3], dim=0)

        with torch.autocast(device_type=device.type, enabled=args.amp):
            out1, out2, out3, change13, change_logits_dict = model(
                x,
                return_change_logits_dict=True,
            )

        missing_keys = set(PAIR_SPECS).difference(change_logits_dict)
        if missing_keys:
            raise KeyError("missing change logits: %s" % sorted(missing_keys))

        semantic_predictions_zero_based = {
            "t1": torch.argmax(out1.float(), dim=1).cpu(),
            "t2": torch.argmax(out2.float(), dim=1).cpu(),
            "t3": torch.argmax(out3.float(), dim=1).cpu(),
        }
        semantic_predictions_label_space = {
            phase: prediction + 1
            for phase, prediction in semantic_predictions_zero_based.items()
        }
        primary_change_prediction = (
            torch.sigmoid(change13.float()) > args.change_threshold
        ).cpu()
        pair_change_predictions = {
            pair_key: (
                torch.sigmoid(change_logits_dict[pair_key].float().squeeze(1))
                > args.change_threshold
            ).cpu()
            for pair_key in PAIR_SPECS
        }
        labels = {
            "t1": label1.long(),
            "t2": label2.long(),
            "t3": label3.long(),
        }
        label_bn = label_bn.long()
        for phase, label in labels.items():
            if ((label < 0) | (label > len(RS.ST_CLASSES))).any():
                raise ValueError("%s label is outside WUSU label space [0, %d]" % (phase, len(RS.ST_CLASSES)))

        primary_pred1 = semantic_predictions_label_space["t1"].clone()
        primary_pred3 = semantic_predictions_label_space["t3"].clone()
        primary_gt1 = labels["t1"].clone()
        primary_gt3 = labels["t3"].clone()
        primary_gt1[label_bn == 0] = 0
        primary_gt3[label_bn == 0] = 0
        primary_pred1[~primary_change_prediction] = 0
        primary_pred3[~primary_change_prediction] = 0
        primary_metric.add_batch(primary_pred1.numpy(), primary_gt1.numpy())
        primary_metric.add_batch(primary_pred3.numpy(), primary_gt3.numpy())

        pair_batch_outputs = {}
        for pair_key, (phase_i, phase_j) in PAIR_SPECS.items():
            label_i = labels[phase_i]
            label_j = labels[phase_j]
            valid = (label_i > 0) & (label_j > 0)
            gt_change = (label_i != label_j) & valid
            pred_change = pair_change_predictions[pair_key]

            pred_i = semantic_predictions_label_space[phase_i].clone()
            pred_j = semantic_predictions_label_space[phase_j].clone()
            gt_i = label_i.clone()
            gt_j = label_j.clone()
            pred_i[~pred_change] = 0
            pred_j[~pred_change] = 0
            gt_i[~gt_change] = 0
            gt_j[~gt_change] = 0
            pred_i[~valid] = 0
            pred_j[~valid] = 0
            gt_i[~valid] = 0
            gt_j[~valid] = 0
            pair_metrics[pair_key].add_batch(pred_i.numpy(), gt_i.numpy())
            pair_metrics[pair_key].add_batch(pred_j.numpy(), gt_j.numpy())

            masked_i = semantic_predictions_zero_based[phase_i].clone()
            masked_j = semantic_predictions_zero_based[phase_j].clone()
            masked_i[~pred_change] = 0
            masked_j[~pred_change] = 0
            pair_batch_outputs[pair_key] = {
                "change_prediction": pred_change.numpy(),
                "change_ground_truth": gt_change.numpy(),
                "semantic_i_masked": masked_i.numpy(),
                "semantic_j_masked": masked_j.numpy(),
            }

        semantic_predictions_np = {
            phase: prediction.numpy()
            for phase, prediction in semantic_predictions_zero_based.items()
        }
        semantic_ground_truth_np = {
            phase: torch.where(label > 0, label - 1, torch.zeros_like(label)).numpy()
            for phase, label in labels.items()
        }
        for index, sample_id in enumerate(sample_ids):
            filename = sample_png_name(sample_id)
            for phase in PHASE_DIRS:
                save_index_png(
                    semantic_predictions_np[phase][index],
                    output_paths["semantic_predictions"][phase] / filename,
                    palette,
                )
                save_index_png(
                    semantic_ground_truth_np[phase][index],
                    output_paths["semantic_ground_truth"][phase] / filename,
                    palette,
                )
            for pair_key, pair_outputs in pair_batch_outputs.items():
                pair_paths = output_paths["pairs"][pair_key]
                save_binary_png(
                    pair_outputs["change_prediction"][index],
                    pair_paths["change_prediction"] / filename,
                )
                save_binary_png(
                    pair_outputs["change_ground_truth"][index],
                    pair_paths["change_ground_truth"] / filename,
                )
                save_index_png(
                    pair_outputs["semantic_i_masked"][index],
                    pair_paths["semantic_i_masked"] / filename,
                    palette,
                )
                save_index_png(
                    pair_outputs["semantic_j_masked"][index],
                    pair_paths["semantic_j_masked"] / filename,
                    palette,
                )

        functional.reset_net(model)

    if non_blocking:
        torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - start_time
    primary = metric_result(primary_metric)
    pairwise = {pair_key: metric_result(metric) for pair_key, metric in pair_metrics.items()}
    macro = {
        key: float(np.mean([pairwise[pair_key][key] for pair_key in PAIR_SPECS]))
        for key in primary
    }
    return primary, pairwise, macro, inference_time


def write_metrics(output_dir, payload):
    """Write raw JSON metrics and percentage-formatted text metrics."""
    output_dir = Path(output_dir)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as handle:
        sections = [("Primary t1_to_t3", payload["primary_t1_to_t3"])]
        sections.extend(
            ("Pair " + pair_key, stats) for pair_key, stats in payload["pairwise"].items()
        )
        sections.append(("Pairwise macro average", payload["pairwise_macro_average"]))
        for section_index, (title, stats) in enumerate(sections):
            if section_index:
                handle.write("\n")
            handle.write(title + "\n")
            for key, label in METRIC_FIELDS:
                handle.write("%s: %.2f%%\n" % (label, stats[key] * 100.0))


def format_metric_line(stats):
    """Format one concise percentage metric line for the console."""
    return ", ".join(
        "%s=%.2f%%" % (label, stats[key] * 100.0)
        for key, label in METRIC_FIELDS[:-1]
    )


def main(argv=None):
    """Run clean PairBCD validation and save all requested outputs."""
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("--batch-size must be >= 1 and --workers must be >= 0")
    if not 0.0 <= args.change_threshold <= 1.0:
        raise ValueError("--change-threshold must be in [0, 1]")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    palette_rgb = color_map_WUSU13()
    if not np.array_equal(palette_rgb, np.asarray(RS.ST_COLORMAP, dtype=np.uint8)):
        raise ValueError("color_map_WUSU13 and RS.ST_COLORMAP do not match")
    palette_table = np.zeros((256, 3), dtype=np.uint8)
    palette_table[: len(palette_rgb)] = palette_rgb
    palette = palette_table.reshape(-1).tolist()

    output_paths = prepare_output_dirs(args.output_dir)
    dataset, loader = build_dataloader(args, device)
    model = build_model(args, device)
    load_checkpoint(model, args.checkpoint)

    print("Checkpoint loaded: %s" % args.checkpoint)
    print("Validation samples: %d" % len(dataset))
    print("Output directory: %s" % output_paths["root"].resolve())

    primary, pairwise, macro, inference_time = run_validation(
        args,
        model,
        loader,
        device,
        output_paths,
        palette,
    )
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "change_threshold": float(args.change_threshold),
        "num_samples": len(dataset),
        "inference_time_seconds": float(inference_time),
        "primary_t1_to_t3": primary,
        "pairwise": pairwise,
        "pairwise_macro_average": macro,
    }
    write_metrics(output_paths["root"], payload)

    print("Primary t1_to_t3: " + format_metric_line(primary))
    for pair_key, stats in pairwise.items():
        print("Pair %s: %s" % (pair_key, format_metric_line(stats)))
    print("Total inference time: %.2f s" % inference_time)


if __name__ == "__main__":
    main()
