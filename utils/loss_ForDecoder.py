import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from functools import partial
from typing import Dict, Sequence, Tuple

import utils.loss_functions as fc
from utils.loss_functions import sigmoid_focal_loss, reduced_focal_loss


class PairwiseRelationAuxLoss(nn.Module):
    """T1/t3 encoder relation supervision with no learnable parameters."""

    def __init__(
        self,
        phase_names=("t1", "t2", "t3"),
        pair=("t1", "t3"),
        scales=(3,),
        tau_unchanged=0.05,
        tau_changed=0.30,
        margin=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.phase_names = tuple(phase_names)
        self.pair = tuple(pair)
        self.scales = tuple(int(scale) for scale in scales)
        self.tau_unchanged = float(tau_unchanged)
        self.tau_changed = float(tau_changed)
        self.margin = float(margin)
        self.eps = float(eps)

        if len(set(self.phase_names)) != len(self.phase_names):
            raise ValueError("phase_names must be unique")
        if len(self.pair) != 2 or any(name not in self.phase_names for name in self.pair):
            raise ValueError("pair must contain two names from phase_names")
        if not self.scales:
            raise ValueError("scales must not be empty")
        if not 0.0 <= self.tau_unchanged <= 1.0 or not 0.0 <= self.tau_changed <= 1.0:
            raise ValueError("pairrel thresholds must be within [0, 1]")
        if self.margin < 0.0 or self.eps <= 0.0:
            raise ValueError("margin must be non-negative and eps must be positive")

        self.phase_indices = tuple(self.phase_names.index(name) for name in self.pair)

    @staticmethod
    def _stat_keys(scale):
        prefix = "_s%d" % scale
        return {
            "loss": "pairrel_loss" + prefix,
            "valid_ratio": "pairrel_valid_ratio" + prefix,
            "valid_weight_sum": "pairrel_valid_weight_sum" + prefix,
            "dist_unchanged": "pairrel_dist_unchanged" + prefix,
            "dist_changed": "pairrel_dist_changed" + prefix,
            "skipped": "pairrel_skipped" + prefix,
        }

    def _skipped_stats(self, scale, zero):
        keys = self._stat_keys(scale)
        detached_zero = zero.detach()
        return {
            keys["loss"]: detached_zero,
            keys["valid_ratio"]: detached_zero,
            keys["valid_weight_sum"]: detached_zero,
            keys["dist_unchanged"]: detached_zero,
            keys["dist_changed"]: detached_zero,
            keys["skipped"]: detached_zero + 1.0,
        }

    def forward(self, encoder_features, change_mask):
        if not encoder_features:
            raise ValueError("encoder_features must not be empty")
        if change_mask.ndim == 3:
            change_mask = change_mask.unsqueeze(1)
        elif change_mask.ndim != 4 or change_mask.shape[1] != 1:
            raise ValueError("change_mask must be [B,H,W] or [B,1,H,W]")
        change_mask = change_mask.float().clamp(0.0, 1.0)

        first_feature = next(iter(encoder_features.values()))
        graph_zero = first_feature.float().sum() * 0.0
        weighted_losses = []
        valid_scale_flags = []
        stats = {}
        phase_i, phase_j = self.phase_indices

        for scale in self.scales:
            feature = encoder_features.get(scale)
            if feature is None:
                stats.update(self._skipped_stats(scale, graph_zero))
                continue
            if feature.ndim != 5:
                raise ValueError("encoder_features[%d] must be [N,B,C,H,W]" % scale)
            if feature.shape[0] != len(self.phase_names):
                raise ValueError("encoder_features[%d] phase count does not match phase_names" % scale)
            if feature.shape[1] != change_mask.shape[0]:
                raise ValueError("encoder_features[%d] batch size does not match change_mask" % scale)

            feature_i = feature[phase_i].float()
            feature_j = feature[phase_j].float()
            feature_i = F.normalize(feature_i, p=2, dim=1, eps=self.eps)
            feature_j = F.normalize(feature_j, p=2, dim=1, eps=self.eps)
            cosine = (feature_i * feature_j).sum(dim=1, keepdim=True)
            distance = torch.sqrt(torch.clamp(2.0 - 2.0 * cosine, min=self.eps))

            changed_ratio = F.interpolate(change_mask, size=distance.shape[-2:], mode="area")
            weight_unchanged = (
                (changed_ratio < self.tau_unchanged).float() * (1.0 - changed_ratio)
            ).detach()
            weight_changed = (
                (changed_ratio > self.tau_changed).float() * changed_ratio
            ).detach()
            valid_weight = weight_unchanged + weight_changed
            valid_mask = valid_weight > 0
            valid_ratio = valid_mask.float().mean()
            valid_weight_sum = valid_weight.sum()

            loss_map = (
                weight_unchanged * distance.pow(2)
                + weight_changed * F.relu(self.margin - distance).pow(2)
            )
            raw_loss = loss_map.sum() / valid_weight_sum.clamp_min(self.eps)
            scale_is_valid = (valid_weight_sum > 0).to(raw_loss.dtype).detach()
            zero_for_scale = feature.float().sum() * 0.0
            loss_scale = raw_loss * scale_is_valid + zero_for_scale * (1.0 - scale_is_valid)

            mean_unchanged = (
                (weight_unchanged * distance).sum()
                / weight_unchanged.sum().clamp_min(self.eps)
            )
            mean_changed = (
                (weight_changed * distance).sum()
                / weight_changed.sum().clamp_min(self.eps)
            )

            keys = self._stat_keys(scale)
            stats.update(
                {
                    keys["loss"]: loss_scale.detach(),
                    keys["valid_ratio"]: valid_ratio.detach(),
                    keys["valid_weight_sum"]: valid_weight_sum.detach(),
                    keys["dist_unchanged"]: mean_unchanged.detach(),
                    keys["dist_changed"]: mean_changed.detach(),
                    keys["skipped"]: (1.0 - scale_is_valid).detach(),
                }
            )
            weighted_losses.append(loss_scale * scale_is_valid)
            valid_scale_flags.append(scale_is_valid)

        if not weighted_losses:
            return graph_zero, stats
        total_loss = torch.stack(weighted_losses).sum()
        valid_scale_count = torch.stack(valid_scale_flags).sum()
        total_loss = total_loss / valid_scale_count.clamp_min(1.0)
        return total_loss, stats


class PairwiseRelationAuxLoss_V11(nn.Module):
    """Parameter-free t1/t3 relation supervision with mode-aware weighting."""

    MODES = ("unchanged_only", "weak_contrastive", "contrastive")

    def __init__(
        self,
        phase_names=("t1", "t2", "t3"),
        pair=("t1", "t3"),
        scales=(3,),
        mode="unchanged_only",
        changed_weight=0.0,
        tau_unchanged=0.05,
        tau_changed=0.50,
        margin=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.phase_names = tuple(phase_names)
        self.pair = tuple(pair)
        self.scales = tuple(int(scale) for scale in scales)
        self.mode = str(mode)
        self.changed_weight = float(changed_weight)
        self.tau_unchanged = float(tau_unchanged)
        self.tau_changed = float(tau_changed)
        self.margin = float(margin)
        self.eps = float(eps)

        if len(set(self.phase_names)) != len(self.phase_names):
            raise ValueError("phase_names must be unique")
        if len(self.pair) != 2 or any(name not in self.phase_names for name in self.pair):
            raise ValueError("pair must contain two names from phase_names")
        if not self.scales:
            raise ValueError("scales must not be empty")
        if self.mode not in self.MODES:
            raise ValueError("mode must be one of %r" % (self.MODES,))
        if self.changed_weight < 0.0:
            raise ValueError("changed_weight must be non-negative")
        if not 0.0 <= self.tau_unchanged <= self.tau_changed <= 1.0:
            raise ValueError("pairrel thresholds must satisfy 0 <= unchanged <= changed <= 1")
        if self.margin < 0.0 or self.eps <= 0.0:
            raise ValueError("margin must be non-negative and eps must be positive")

        self.phase_indices = tuple(self.phase_names.index(name) for name in self.pair)
        self.effective_changed_weight = {
            "unchanged_only": 0.0,
            "weak_contrastive": self.changed_weight,
            "contrastive": 1.0,
        }[self.mode]

    @staticmethod
    def _stat_keys(scale):
        prefix = "_s%d" % scale
        return {
            "loss": "pairrel_loss" + prefix,
            "loss_unchanged": "pairrel_loss_unchanged" + prefix,
            "loss_changed": "pairrel_loss_changed" + prefix,
            "valid_ratio": "pairrel_valid_ratio" + prefix,
            "valid_weight_sum": "pairrel_valid_weight_sum" + prefix,
            "unchanged_ratio": "pairrel_unchanged_ratio" + prefix,
            "changed_ratio": "pairrel_changed_ratio" + prefix,
            "ambiguous_ratio": "pairrel_ambiguous_ratio" + prefix,
            "unchanged_weight_sum": "pairrel_unchanged_weight_sum" + prefix,
            "changed_weight_sum": "pairrel_changed_weight_sum" + prefix,
            "dist_unchanged": "pairrel_dist_unchanged" + prefix,
            "dist_changed": "pairrel_dist_changed" + prefix,
            "dist_gap": "pairrel_dist_gap_changed_minus_unchanged" + prefix,
            "hinge_active_ratio": "pairrel_hinge_active_ratio" + prefix,
            "skipped": "pairrel_skipped" + prefix,
        }

    @staticmethod
    def _finite_stat(value):
        return torch.nan_to_num(value.detach(), nan=0.0, posinf=0.0, neginf=0.0)

    def _skipped_stats(self, scale, zero):
        keys = self._stat_keys(scale)
        finite_zero = self._finite_stat(zero)
        return {
            key: finite_zero + (1.0 if name == "skipped" else 0.0)
            for name, key in keys.items()
        }

    def forward(self, encoder_features, change_mask):
        if not encoder_features:
            raise ValueError("encoder_features must not be empty")
        if change_mask.ndim == 3:
            change_mask = change_mask.unsqueeze(1)
        elif change_mask.ndim != 4 or change_mask.shape[1] != 1:
            raise ValueError("change_mask must be [B,H,W] or [B,1,H,W]")
        change_mask = change_mask.float()
        if change_mask.numel() == 0 or not torch.isfinite(change_mask).all().item():
            raise ValueError("change_mask must contain finite values within [0, 1]")
        if change_mask.min().item() < 0.0 or change_mask.max().item() > 1.0:
            raise ValueError("change_mask values must be within [0, 1]")

        first_feature = next(iter(encoder_features.values()))
        graph_zero = first_feature.float().sum() * 0.0
        losses = []
        valid_scale_flags = []
        stats = {}
        phase_i, phase_j = self.phase_indices
        changed_is_active = self.effective_changed_weight > 0.0

        for scale in self.scales:
            feature = encoder_features.get(scale)
            if feature is None:
                stats.update(self._skipped_stats(scale, graph_zero))
                continue
            if feature.ndim != 5:
                raise ValueError("encoder_features[%d] must be [N,B,C,H,W]" % scale)
            if feature.shape[0] != len(self.phase_names):
                raise ValueError("encoder_features[%d] phase count does not match phase_names" % scale)
            if feature.shape[1] != change_mask.shape[0]:
                raise ValueError("encoder_features[%d] batch size does not match change_mask" % scale)

            feature_i = F.normalize(feature[phase_i].float(), p=2, dim=1, eps=self.eps)
            feature_j = F.normalize(feature[phase_j].float(), p=2, dim=1, eps=self.eps)
            cosine = (feature_i * feature_j).sum(dim=1, keepdim=True)
            distance = torch.sqrt(torch.clamp(2.0 - 2.0 * cosine, min=self.eps))

            changed_ratio = F.interpolate(change_mask, size=distance.shape[-2:], mode="area")
            unchanged_mask = changed_ratio < self.tau_unchanged
            changed_mask = changed_ratio > self.tau_changed
            ambiguous_mask = ~(unchanged_mask | changed_mask)
            weight_unchanged = (unchanged_mask.float() * (1.0 - changed_ratio)).detach()
            weight_changed = (changed_mask.float() * changed_ratio).detach()
            unchanged_weight_sum = weight_unchanged.sum()
            changed_weight_sum = weight_changed.sum()
            valid_weight = weight_unchanged + (weight_changed if changed_is_active else 0.0)
            valid_mask = unchanged_mask | (changed_mask if changed_is_active else False)
            valid_weight_sum = valid_weight.sum()

            loss_unchanged = (
                (weight_unchanged * distance.pow(2)).sum()
                / unchanged_weight_sum.clamp_min(self.eps)
            )
            hinge = F.relu(self.margin - distance)
            loss_changed = (
                (weight_changed * hinge.pow(2)).sum()
                / changed_weight_sum.clamp_min(self.eps)
            )
            raw_loss = loss_unchanged
            if changed_is_active:
                raw_loss = raw_loss + self.effective_changed_weight * loss_changed
            scale_is_valid = (valid_weight_sum > 0).to(raw_loss.dtype).detach()
            zero_for_scale = feature.float().sum() * 0.0
            loss_scale = raw_loss * scale_is_valid + zero_for_scale * (1.0 - scale_is_valid)

            mean_unchanged = (
                (weight_unchanged * distance).sum()
                / unchanged_weight_sum.clamp_min(self.eps)
            )
            mean_changed = (
                (weight_changed * distance).sum()
                / changed_weight_sum.clamp_min(self.eps)
            )
            hinge_active_ratio = (
                (weight_changed * (distance < self.margin).float()).sum()
                / changed_weight_sum.clamp_min(self.eps)
            )

            keys = self._stat_keys(scale)
            values = {
                "loss": loss_scale,
                "loss_unchanged": loss_unchanged,
                "loss_changed": loss_changed,
                "valid_ratio": valid_mask.float().mean(),
                "valid_weight_sum": valid_weight_sum,
                "unchanged_ratio": unchanged_mask.float().mean(),
                "changed_ratio": changed_mask.float().mean(),
                "ambiguous_ratio": ambiguous_mask.float().mean(),
                "unchanged_weight_sum": unchanged_weight_sum,
                "changed_weight_sum": changed_weight_sum,
                "dist_unchanged": mean_unchanged,
                "dist_changed": mean_changed,
                "dist_gap": mean_changed - mean_unchanged,
                "hinge_active_ratio": hinge_active_ratio,
                "skipped": 1.0 - scale_is_valid,
            }
            stats.update({keys[name]: self._finite_stat(value) for name, value in values.items()})
            losses.append(loss_scale)
            valid_scale_flags.append(scale_is_valid)

        if not losses:
            return graph_zero, stats
        total_loss = torch.stack(losses).sum()
        valid_scale_count = torch.stack(valid_scale_flags).sum()
        return total_loss / valid_scale_count.clamp_min(1.0), stats


PAIRWISE_CHANGE_PAIRS = (
    ("t1", "t2"),
    ("t2", "t3"),
    ("t1", "t3"),
)


def _pair_key(pair: Tuple[str, str]) -> str:
    return "%s_to_%s" % (pair[0], pair[1])


def _as_bchw(tensor, name):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError("%s must be [B,H,W] or [B,1,H,W], got %r" % (name, tuple(tensor.shape)))
    return tensor


def make_pairwise_change_targets(
    sem_targets: Dict[str, torch.Tensor],
    pairs: Sequence[Tuple[str, str]] = PAIRWISE_CHANGE_PAIRS,
    ignore_index: int = -1,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    sem_targets[phase]: CE labels [B,H,W] with ignore_index.
    returns target/valid maps [B,1,H,W] for fixed pairwise binary change.
    """

    out = {}
    for phase_i, phase_j in pairs:
        if phase_i not in sem_targets or phase_j not in sem_targets:
            raise KeyError("sem_targets must contain %s and %s" % (phase_i, phase_j))
        yi = sem_targets[phase_i]
        yj = sem_targets[phase_j]
        if yi.shape != yj.shape or yi.ndim != 3:
            raise ValueError("semantic targets for %s/%s must match [B,H,W]" % (phase_i, phase_j))
        valid = (yi != ignore_index) & (yj != ignore_index)
        target = ((yi != yj) & valid).float().unsqueeze(1)
        out[_pair_key((phase_i, phase_j))] = {
            "target": target,
            "valid": valid.float().unsqueeze(1),
        }
    return out


def pairwise_c13_mismatch_stats(pair_targets, change13_target):
    if "t1_to_t3" not in pair_targets:
        raise KeyError("pair_targets must contain t1_to_t3")
    target = _as_bchw(pair_targets["t1_to_t3"]["target"], "pair target").float()
    valid = _as_bchw(pair_targets["t1_to_t3"]["valid"], "pair valid").float()
    reference = _as_bchw(change13_target, "change13 target").float()
    if reference.shape[-2:] != target.shape[-2:]:
        reference = F.interpolate(reference, size=target.shape[-2:], mode="nearest")
    reference = (reference > 0.5).float()
    mismatch = ((target > 0.5) != (reference > 0.5)).float() * valid
    valid_sum = valid.sum().clamp_min(1.0)
    return {
        "pair_bcd_c13_mismatch_ratio": (mismatch.sum() / valid_sum).detach(),
        "pair_bcd_c13_valid_ratio": valid.mean().detach(),
    }


class MaskedBCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    def forward(self, logits, target, valid):
        logits = _as_bchw(logits, "logits").float()
        target = _as_bchw(target, "target").float()
        valid = _as_bchw(valid, "valid").float()
        if logits.shape != target.shape or logits.shape != valid.shape:
            raise ValueError("logits, target, and valid must have the same [B,1,H,W] shape")

        valid_sum = valid.sum()
        if valid_sum.item() <= 0.0:
            return logits.sum() * 0.0

        loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss_bce = (loss_map * valid).sum() / valid_sum.clamp_min(1.0)
        prob = torch.sigmoid(logits) * valid
        target = target * valid
        inter = (prob * target).sum()
        denom = prob.sum() + target.sum()
        loss_dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        return loss_bce + self.dice_weight * loss_dice


class PairwiseBinaryChangeLoss(nn.Module):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]] = PAIRWISE_CHANGE_PAIRS,
        lambda_adj: float = 0.5,
        lambda_13: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.pairs = tuple((str(a), str(b)) for a, b in pairs)
        self.pair_keys = tuple(_pair_key(pair) for pair in self.pairs)
        if self.pair_keys != ("t1_to_t2", "t2_to_t3", "t1_to_t3"):
            raise ValueError("PairwiseBinaryChangeLoss expects fixed pair order")
        self.lambda_adj = float(lambda_adj)
        self.lambda_13 = float(lambda_13)
        if self.lambda_adj < 0.0 or self.lambda_13 < 0.0:
            raise ValueError("pairwise loss weights must be non-negative")
        self.criterion = MaskedBCEDiceLoss(dice_weight=dice_weight)

    def forward(self, change_logits_dict, pair_targets):
        losses = []
        stats = {}
        for key in self.pair_keys:
            if key not in change_logits_dict or key not in pair_targets:
                raise KeyError("missing pairwise change key: %s" % key)
            target = pair_targets[key]["target"]
            valid = pair_targets[key]["valid"]
            pair_loss = self.criterion(change_logits_dict[key], target, valid)
            weight = self.lambda_13 if key == "t1_to_t3" else self.lambda_adj
            losses.append(weight * pair_loss)
            valid_f = _as_bchw(valid, "valid").float()
            target_f = _as_bchw(target, "target").float()
            valid_sum = valid_f.sum().clamp_min(1.0)
            stats["pair_bcd_loss_" + key] = pair_loss.detach()
            stats["pair_bcd_valid_ratio_" + key] = valid_f.mean().detach()
            stats["pair_bcd_pos_ratio_" + key] = ((target_f * valid_f).sum() / valid_sum).detach()

        if not losses:
            first_logit = next(iter(change_logits_dict.values()))
            return first_logit.float().sum() * 0.0, stats
        return torch.stack(losses).sum(), stats


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    pos = (truth > 0.5).float()
    neg = (truth < 0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()

    return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """

    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        # ~表示布尔值取反
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss


class js_div(_Loss):
    def __init__(self):
        super().__init__()
        self.KLDivLoss = nn.KLDivLoss(reduction='none')
    def forward(self, p_out, q_out, get_softmax=True):
        if get_softmax:
            p_out = F.softmax(p_out, dim=1)
            q_out = F.softmax(q_out, dim=1)
        log_mean_out = ((p_out + q_out)/2).log()
        js = 0.5*self.KLDivLoss(log_mean_out, p_out) + 0.5*self.KLDivLoss(log_mean_out, q_out)
        return js

class Similarity(_Loss):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    def forward(self, p_out, q_out, mask_bin):
        # mask_bin = 1 - mask_bin
        p = torch.argmax(p_out, dim=1)
        q = torch.argmax(q_out, dim=1)

        # print('mask_bin', mask_bin.shape)
        # exit(0)

        p[mask_bin == 1] = -1
        q[mask_bin == 1] = -1
        # q_out = (mask_bin * q_out).long()
        loss = 0.5*self.criterion(p_out, q.long()) + 0.5*self.criterion(q_out, p.long())
        return loss



class TverskyLoss(_Loss):
    __name__ = "dice_loss"

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, y_pr, y_gt):
        return 1 - fc.tversky(y_pr, y_gt, beta=1, eps=self.eps, threshold=None)

class BCELoss(_Loss):
    __name__ = "bce_loss"

    def __int__(self, reduction="mean"):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, target):
        if type(outputs) in [tuple, list]:
            outputs = outputs[0]

        bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return bce_loss(outputs, target)


class WBCELoss(_Loss):
    __name__ = "wbce_loss"

    def __int__(self, reduction="mean", betas=[0.8, 0.2]):
        super(WBCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, target):
        bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return bce_loss(outputs, target)


class NLLLoss(_Loss):
    __name__ = "nll_loss"

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, outputs, target):
        """outputs.shape = [B, 2, H, W], target.shape = [B, H, W]"""
        nll_loss = nn.NLLLoss(weight=self.weight, reduction=self.reduction)
        # print(f"type: {type(outputs)}")
        # print(f"len: {len(outputs)}")
        # if type(outputs) in [tuple, list]:
        #     outputs = outputs[0]

        return nll_loss(outputs, target.long().squeeze(dim=1))


class OhemBCELoss(_Loss):
    __name__ = "OhemBCELoss"

    def __init__(self, weight=None, threshold=0.7, min_kept=1000, reduction="mean"):
        super(OhemBCELoss, self).__init__()
        self.reduction = reduction
        self.weight = weight
        self.ths = threshold
        self.min_kept = min_kept

    def forward(self, predict, target):
        bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        bce_loss_matrix = bce_loss.contiguous().view(-1, )

        # ========================================================== #
        batch_kept = self.min_kept * target.size(0)
        prob_out = torch.sigmoid(predict)

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        # gather: 以tmp_target [n, 1, h, w]的spatial dim[h, w]的值作为prob_out通道维dim=1的索引，取出相应通道的值
        prob = prob_out.gather(dim=1, index=tmp_target.unsqueeze(dim=1))
        mask = target.contiguous().view(-1,) == 1
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, self.thresh)
        # ========================================================== #

        sort_loss_matrix = bce_loss_matrix[mask][sort_indices]
        select_loss_matrix = sort_loss_matrix[sort_prob < threshold]

        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


class DiceLoss(_Loss):
    __name__ = "dice_loss"

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - fc.f_score(y_pr, y_gt, beta=1, eps=self.eps, threshold=None, activation=self.activation)


class BCEDiceLoss(_Loss):

    __name__ = "bce_dice_loss"

    def __init__(self, dice_weight=0.5, pos_weight=None, eps=1e-7):
        super().__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        self.dice_loss = DiceLoss(eps)
        self.dice_weight = dice_weight

    def forward(self, outputs, target):

        return self.bce_loss(outputs, target) + self.dice_weight * self.dice_loss(outputs, target)


class MultiBCEDiceLoss(_Loss):

    __name__ = "multi_bcedice_loss"

    def __init__(self, dice_weight=1., pos_weight=False):
        super().__init__()

        if pos_weight:
            self.pos_weight = torch.ones([1]) * 20
        else:
            self.pos_weight = None

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=self.pos_weight)
        self.dice_loss = DiceLoss(eps=1e-7)
        self.dice_weight = dice_weight

    def forward(self, outputs, target):

        if type(outputs) is list and len(outputs) == 5:
            loss = 0.
            # print('*'*50)
            # print(self.pos_weight)
            # print(outputs[0].shape)
            for index, loss_weight in enumerate([1., 1., 1., 1., 1.]):
                loss += loss_weight * (self.bce_loss(outputs[index], target) + self.dice_weight * self.dice_loss(outputs[index], target))

        elif type(outputs) is list and len(outputs) == 1:
            loss = self.bce_loss(outputs[-1], target) + self.dice_weight * self.dice_loss(outputs[-1], target)
        else:
            raise ValueError

        return loss


class MultiSEGLoss(_Loss):

    __name__ = "multi_seg_loss"

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, outputs, target):

        if type(outputs) is tuple and len(outputs) == 3:
            loss = 0.
            for index, loss_weight in enumerate([1.0, 0.2, 0.2]):
                loss += loss_weight * self.bce_loss(outputs[index], target)

        elif type(outputs) is tuple and len(outputs) == 1:
            loss = self.bce_loss(outputs[-1], target)
        else:
            raise ValueError

        return loss


class JaccardLoss(_Loss):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - fc.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class JaccardLogLoss(_Loss):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        iou = fc.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)
        return - torch.log(iou)


class BCEJaccardLoss(_Loss):
    """
    Loss = -\alpha * SoftJaccard + (1 - \alpha) * BCE

    """

    __name__ = "bce_jaccard_loss"

    def __init__(self, jaccard_weight=0.3, use_ohem=False):
        super().__init__()
        if use_ohem:
            print("=> use ohem")
            self.bce_loss = OhemBCELoss(weight=None, threshold=0.7, min_kept=100000, reduction="mean")
        else:
            self.bce_loss = nn.BCELoss(reduction="mean")

        self.jaccard_weight = jaccard_weight
        # self.jaccard_weight = True

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.bce_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()


            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss

    def forward(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.bce_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            # forward output without nn.Sigmoid
            # jaccard_output = torch.sigmoid(outputs)
            jaccard_output = outputs
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class BCEFocalLoss(_Loss):
    __name__ = "bce_focal_loss"

    def __init__(
            self,
            alpha=0.5,
            gamma=2,
            ignore_index=None,
            reduction='mean',
            reduced=False,
            threshold=0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if reduced:
            self.focal_loss = partial(reduced_focal_loss, gamma=gamma, threshold=threshold, reduction=reduction)
        else:
            self.focal_loss = partial(sigmoid_focal_loss, gamma=gamma, alpha=alpha, reduction=reduction)

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem."""
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

        loss = self.focal_loss(label_input, label_target)
        return loss


class FocalJaccardLoss(BCEFocalLoss):
    """Loss = - \alpha * SoftJaccard + (1 - \alpha) Focal"""

    __name__ = "focal_jaccard_loss"

    def __init__(self, alpha=0.5, gamma=2, reduction='mean', reduced=False, jaccard_weight=0.7, threshold=0.5):
        super().__init__()
        self.loss = BCEFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction, reduced=reduced, threshold=threshold)
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = (1 - self.jaccard_weight) * self.loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class MultiBCELoss(torch.nn.BCEWithLogitsLoss):
    """2D Cross Entropy Loss with Multi-Loss"""

    __name__ = "multi_bce_loss"

    def __init__(self, weight=None, reduction="mean"):
        super(MultiBCELoss, self).__init__(weight, reduction)

    def forward(self, preds, target):

        pred1, pred2, pred3 = tuple(preds)

        loss1 = super(MultiBCELoss, self).forward(pred1, target)
        loss2 = super(MultiBCELoss, self).forward(pred2, target)
        loss3 = super(MultiBCELoss, self).forward(pred3, target)
        loss = loss1 + loss2 + loss3
        return loss


class MultiBCEJaccardLoss(BCEJaccardLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""

    __name__ = "multi_bcejaccard_loss"

    def __init__(self, jaccard_weight=0.3, aux_weight=0., pos_weight=0., use_ohem=False):
        super(MultiBCEJaccardLoss, self).__init__(jaccard_weight, use_ohem)
        self.aux_weight = aux_weight
        self.pos_weight = pos_weight

        if pos_weight is not None:
            # self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
            self.bce_loss = nn.BCELoss(reduction="mean")
    def __call__(self, *preds, target):
        """
         __call__(self, *preds, target) for dp: loss = self.loss(predictions, target=y)
         __call__(self, preds, target) for ddp: loss = self.loss(predictions, target=y)
        """

        assert type(preds) in [list, tuple], "net forward preds must return list or tuple."
        # print(f"type_preds = {type(preds[0])}, len_preds = {len(preds[0])}")
        # print(f"type_preds = {type(preds)}, len_preds = {len(preds)}")
        # exit(1)
        # print('loss中的维度', preds)
        # print('*'*50)
        # print('pred[0]的维度', len(preds[0]), len(preds))
        # print('len(preds[0]', len(preds[0]))
        if len(preds[0]) >= 1:
            preds = preds[0]
            # print(preds[0].shape)
        # print('len(preds)', len(preds))
        # exit(0)
        if len(preds) == 3:
            # print(f"preds(type, len) = ({type(preds)}, {len(preds)})")
            # preds(type, len) = (<class 'tuple'>, 3)

            pred1, pred2, pred3 = tuple(preds)

            loss1 = super(MultiBCEJaccardLoss, self).forward(pred1, target)
            loss2 = super(MultiBCEJaccardLoss, self).forward(pred2, target)

            if self.pos_weight is not None:
                loss3 = self.bce_loss(pred3, target)
                loss = loss1 + self.aux_weight*loss2 + self.pos_weight*loss3
            else:
                loss3 = super(MultiBCEJaccardLoss, self).forward(pred3, target)
                loss = loss1 + loss2 + loss3

        elif len(preds) == 2:
            pred1, pred2 = tuple(preds)
            loss1 = super(MultiBCEJaccardLoss, self).forward(pred1, target)

            """
            if self.pos_weight is not None:
                loss2 = self.bce_loss(pred2, target)
                loss = loss1 + self.pos_weight * loss2
            elif self.aux_weight is not None:
                loss2 = super(MultiBCEJaccardLoss, self).forward(pred2, target)
                loss = loss1 + self.aux_weight * loss2
            """

            loss2 = super(MultiBCEJaccardLoss, self).forward(pred2, target)
            loss = loss1 + self.aux_weight * loss2

        else:
            loss = super(MultiBCEJaccardLoss, self).forward(preds[0], target)

        return loss


class AffinityLoss(_Loss):

    __name__ = "affinity_loss"

    def __init__(self, eps=1e-15, ths=0.5):
        super().__init__()
        self.eps = eps
        self.ths = ths

    def __call__(self, pred_affinity_org, gt_affinity):
        pred_affinity_org = torch.sigmoid(pred_affinity_org.detach())
        pred_affinity = pred_affinity_org
        pred_affinity[pred_affinity_org >= self.ths] = 1.0
        pred_affinity[pred_affinity_org < self.ths] = 0.0
        gt_affinity = gt_affinity.float()

        # [N, HW, HW] => [N, HW, 1] 行和sum(dim=-1), 同时去掉最后一维dims = dims - 1
        tp = (pred_affinity * gt_affinity).sum(dim=-1)
        tp_plus_fp = pred_affinity.sum(dim=-1)
        tp_plus_fn = gt_affinity.sum(dim=-1)

        tn = ((torch.ones_like(pred_affinity) - pred_affinity) * (torch.ones_like(gt_affinity) - gt_affinity)).sum(dim=-1)
        tn_plus_fp = (1 - gt_affinity).sum(dim=-1)

        intra_loss_precision = torch.log((tp + self.eps) / (tp_plus_fp + self.eps))
        intra_loss_recall = torch.log((tp + self.eps) / (tp_plus_fn + self.eps))
        inter_loss_specifity = torch.log((tn + self.eps) / (tn_plus_fp + self.eps))

        # mean(dim=1), 对每一行的和取平均，同时dims减一
        gloabal_affinity_loss = -(intra_loss_precision + intra_loss_recall + inter_loss_specifity).mean(dim=1)

        # mean(dim=0), 对批量求平均
        gloabal_affinity_loss = gloabal_affinity_loss.mean(dim=0)

        return gloabal_affinity_loss


class MultiAffinityBCEJaccardLoss(BCEJaccardLoss):
    __name__ = "multi_affinity_bcejaccard_loss"

    def __init__(
        self, jaccard_weight=0.7,
        aux_weight=0.4,
        aff_weight=1.0,
        aff_global_weight=1.0,
        aff_unary_weight=1.0,
        aff_eps=1e-15, aff_ths=0.5
    ):
        super(MultiAffinityBCEJaccardLoss, self).__init__(jaccard_weight)
        self.aux_weight = aux_weight

        self.aff_weight = aff_weight
        self.aff_global_weight = aff_global_weight
        self.aff_unary_weight = aff_unary_weight
        if None not in [aff_unary_weight, aff_global_weight]:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
            self.aff_loss = AffinityLoss(aff_eps, aff_ths)

    def __call__(self, preds, target):

        assert type(preds) in [list, tuple], "net forward preds must return list or tuple."

        if len(preds) == 3:
            # torch.autograd.set_detect_anomaly(True)

            pred1, pred2, pred3 = tuple(preds)

            loss1 = super(MultiAffinityBCEJaccardLoss, self).forward(pred1, target)
            loss2 = super(MultiAffinityBCEJaccardLoss, self).forward(pred2, target)

            b, c, h, w = target.shape
            downsample_target = F.interpolate(target, size=(int(h // 16), int(w // 16)), mode="bilinear", align_corners=True).view(b, c, -1)
            idea_aff_map = torch.matmul(
                downsample_target.permute(0, 2, 1).contiguous(),
                downsample_target
            )
            # print("idea_aff_map : ", idea_aff_map.shape, idea_aff_map.max(), idea_aff_map.min())

            if self.aff_global_weight is not None:
                # loss3 = self.aff_unary_weight * self.bce_loss(pred3, idea_aff_map) + \
                #         self.aff_global_weight * self.aff_loss(pred3, idea_aff_map)

                loss3 = self.aff_unary_weight * self.bce_loss(pred3, idea_aff_map) + \
                        self.aff_global_weight * self.aff_loss(pred3, idea_aff_map)

                loss = loss1 + self.aux_weight * loss2 + self.aff_weight * loss3
            else:
                # loss3 = super(MultiAffinityBCEJaccardLoss, self).forward(pred3, target)
                loss3 = self.aff_unary_weight * self.bce_loss(pred3, target)
                loss = loss1 + self.aux_weight * loss2 + loss3

        elif len(preds) == 2:
            pred1, pred2 = tuple(preds)

            loss1 = super(MultiAffinityBCEJaccardLoss, self).forward(pred1, target)
            loss2 = super(MultiAffinityBCEJaccardLoss, self).forward(pred2, target)
            loss = loss1 + self.aux_weight * loss2

        else:
            loss = super(MultiAffinityBCEJaccardLoss, self).forward(preds[0], target)

        return loss


class MultiJacSEGLoss(_Loss):

    __name__ = "multi_jacseg_loss"

    def __init__(self):
        super().__init__()
        self.bce_loss = BCEJaccardLoss()

    def forward(self, outputs, target):

        if type(outputs) is tuple and len(outputs) == 3:
            loss = 0.
            for index, loss_weight in enumerate([1.0, 0.2, 0.2]):
                loss += loss_weight * self.bce_loss(outputs[index], target)

        elif type(outputs) is tuple and len(outputs) == 1:
            loss = self.bce_loss(outputs[-1], target)
        else:
            raise ValueError

        return loss


class DSIFN_Loss(_Loss):

    __name__ = "DSIFN_loss"

    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        # self.bce_loss = torch.nn.BCELoss()
        self.dice_loss = DiceLoss(eps=1e-7)
        self.sigmoid = torch.nn.Sigmoid()
    def __dice__(self, output, target):

        smooth = 1.
        iflat = self.sigmoid(output).view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        return dice_loss

    def forward(self, outputs, target):
        # print(target)
        
        if type(outputs) is tuple and len(outputs) == 5:
            loss = 0.
            for index, loss_weight in enumerate([1.0] * 5):
                loss += loss_weight * (self.bce_loss(outputs[index], target) + self.__dice__(outputs[index], target))
                # loss += loss_weight * (self.bce_loss(outputs[index], target) + self.dice_loss(outputs[index], target))
        elif type(outputs) is tuple and len(outputs) == 1:
            loss = self.bce_loss(outputs[-1], target) + self.__dice__(outputs[-1], target)
        else:
            raise ValueError

        return loss
