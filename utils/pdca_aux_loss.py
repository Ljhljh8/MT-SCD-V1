import torch
import torch.nn.functional as F


def downsample_change_ratio(mask_bn, out_size, eps=1e-6):
    if eps <= 0:
        raise ValueError("eps must be positive")
    if mask_bn.ndim == 3:
        mask_bn = mask_bn.unsqueeze(1)
    elif mask_bn.ndim != 4 or mask_bn.shape[1] != 1:
        raise ValueError("mask_bn must be [B,H,W] or [B,1,H,W]")
    return F.interpolate(mask_bn.float(), size=out_size, mode="area")


def build_ratio_weight(
    ratio,
    tau_neg=0.05,
    tau_pos=0.20,
    ambiguous_weight=0.25,
):
    if ratio.ndim != 4 or ratio.shape[1] != 1:
        raise ValueError("ratio must be [B,1,H,W]")
    if not 0.0 <= tau_neg <= tau_pos <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= tau_neg <= tau_pos <= 1")
    if ambiguous_weight < 0:
        raise ValueError("ambiguous_weight must be non-negative")
    weight = torch.ones_like(ratio)
    ambiguous = (ratio > tau_neg) & (ratio < tau_pos)
    return weight.masked_fill(ambiguous, float(ambiguous_weight))


def _masked_mean(values, mask, eps):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def pdca_relation_aux_loss(
    encoder_aux_list,
    mask_bn,
    scale_key="3",
    pair_keys=("t1<-t3", "t3<-t1"),
    lambda_last_only=True,
    tau_neg=0.05,
    tau_pos=0.20,
    ambiguous_weight=0.25,
    eps=1e-6,
):
    if not isinstance(encoder_aux_list, (list, tuple)) or not encoder_aux_list:
        raise RuntimeError("encoder_aux_list is empty; PDCA relation logits are required")
    if not pair_keys:
        raise ValueError("pair_keys must not be empty")
    if eps <= 0:
        raise ValueError("eps must be positive")

    selected = [(len(encoder_aux_list) - 1, encoder_aux_list[-1])]
    if not lambda_last_only:
        selected = list(enumerate(encoder_aux_list))

    losses = []
    target_means = []
    logit_means = []
    positive_ratios = []
    prob_means = []
    changed_losses = []
    unchanged_losses = []
    scale_key = str(scale_key)

    for block_index, block_aux in selected:
        try:
            relation_logits = block_aux["pdca_relation_logits"][scale_key]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "Missing PDCA relation logits at encoder_aux[%d]['pdca_relation_logits'][%r]"
                % (block_index, scale_key)
            ) from exc

        for pair_key in pair_keys:
            if pair_key not in relation_logits:
                raise RuntimeError(
                    "Missing PDCA relation logit at encoder_aux[%d]['pdca_relation_logits'][%r][%r]"
                    % (block_index, scale_key, pair_key)
                )
            relation_logit = relation_logits[pair_key]
            if relation_logit.ndim != 4 or relation_logit.shape[1] != 1:
                raise RuntimeError(
                    "PDCA relation logit %r must be [B,1,H,W], got %r"
                    % (pair_key, tuple(relation_logit.shape))
                )
            if relation_logit.shape[0] != mask_bn.shape[0]:
                raise RuntimeError("PDCA relation logit batch size does not match mask_bn")

            logit = relation_logit.float()
            ratio = downsample_change_ratio(mask_bn, logit.shape[-2:], eps=eps)
            weight = build_ratio_weight(
                ratio,
                tau_neg=tau_neg,
                tau_pos=tau_pos,
                ambiguous_weight=ambiguous_weight,
            )
            loss_map = F.binary_cross_entropy_with_logits(logit, ratio, reduction="none")
            losses.append((loss_map * weight).sum() / weight.sum().clamp_min(eps))
            target_means.append(ratio.mean())
            logit_means.append(logit.mean())
            positive_mask = ratio >= tau_pos
            unchanged_mask = ratio <= tau_neg
            positive_ratios.append(positive_mask.float().mean())
            prob_means.append(torch.sigmoid(logit).mean())
            changed_losses.append(_masked_mean(loss_map, positive_mask, eps))
            unchanged_losses.append(_masked_mean(loss_map, unchanged_mask, eps))

    loss_aux = torch.stack(losses).mean()
    stats = {
        "pdca_aux_loss": loss_aux.detach(),
        "pdca_aux_target_mean": torch.stack(target_means).mean().detach(),
        "pdca_aux_logit_mean": torch.stack(logit_means).mean().detach(),
        "pdca_aux_positive_ratio": torch.stack(positive_ratios).mean().detach(),
        "pdca_aux_prob_mean": torch.stack(prob_means).mean().detach(),
        "pdca_aux_changed_loss": torch.stack(changed_losses).mean().detach(),
        "pdca_aux_unchanged_loss": torch.stack(unchanged_losses).mean().detach(),
    }
    return loss_aux, stats
