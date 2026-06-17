"""Auxiliary losses: ranking (τ/ρ), query contrastive (QFVS)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    max_pairs: int = 32,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Vectorized hinge on random pairs: target[i] > target[j] => logit_i should exceed logit_j.
    All computation in FP32 to avoid AMP overflow.
    """
    logits_f = logits.detach().float().view(-1)
    logits_g = logits.float().view(-1)
    target_f = target.float().view(-1)
    n = logits_f.numel()
    if n < 2:
        return logits.new_zeros(())
    device = logits.device
    idx_i = torch.randint(0, n, (max_pairs,), device=device)
    idx_j = torch.randint(0, n, (max_pairs,), device=device)
    ti, tj = target_f[idx_i], target_f[idx_j]
    mask = ti > tj
    if not mask.any():
        swap = tj > ti
        if not swap.any():
            return logits.new_zeros(())
        idx_i, idx_j = idx_j[swap], idx_i[swap]
    else:
        idx_i, idx_j = idx_i[mask], idx_j[mask]
    diff = logits_g[idx_i] - logits_g[idx_j]
    return F.relu(margin - diff).mean()


def topology_regularization_loss(
    all_weights: list[list[torch.Tensor]],
    lambda_sparse: float = 0.01,
    lambda_div: float = 0.005,
    rho_min: float = 0.3,
) -> torch.Tensor:
    """
    TAMS topology regularization:

    L_topo = λ_s · sparsity + λ_d · diversity

    Sparsity: mean edge weight across all layers/modalities (prevents fully-connected).
    Diversity: KL-divergence lower bound between modality pairs (forces different topologies).

    Args:
        all_weights: list (per layer) of list (per modality) of [E_m] weight tensors.
        lambda_sparse: sparsity coefficient.
        lambda_div: diversity coefficient.
        rho_min: minimum KL divergence threshold between modality pairs.
    """
    if not all_weights:
        return torch.zeros((), dtype=torch.float32)

    device = all_weights[0][0].device

    # --- Sparsity: average edge weight across all layers and modalities ---
    sparse_sum = torch.zeros((), device=device)
    count = 0
    for layer_weights in all_weights:
        for w in layer_weights:
            sparse_sum = sparse_sum + w.mean()
            count += 1
    sparsity = sparse_sum / max(count, 1)

    # --- Diversity: KL lower bound between modality pairs per layer ---
    div_loss = torch.zeros((), device=device)
    div_count = 0
    for layer_weights in all_weights:
        n_mod = len(layer_weights)
        if n_mod < 2:
            continue
        for m1 in range(n_mod):
            for m2 in range(m1 + 1, n_mod):
                w1 = layer_weights[m1]
                w2 = layer_weights[m2]
                min_len = min(w1.shape[0], w2.shape[0])
                p = w1[:min_len].clamp(1e-6, 1 - 1e-6)
                q = w2[:min_len].clamp(1e-6, 1 - 1e-6)
                kl = (p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())).mean()
                div_loss = div_loss + F.relu(rho_min - kl)
                div_count += 1

    if div_count > 0:
        div_loss = div_loss / div_count

    return lambda_sparse * sparsity + lambda_div * div_loss


def gate_entropy_loss(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean entropy of row-wise softmax `alpha` (N, M). Minimizing encourages sharper modality gates.
    Use with small lambda_gate.
    """
    p = alpha.float().clamp_min(eps)
    ent = -(p * p.log()).sum(dim=-1).mean()
    return ent


def margin_separation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.3,
    top_frac: float = 0.15,
) -> torch.Tensor:
    """
    Encourages a clear gap between high-importance and low-importance frames.

    Selects top-k and bottom-k frames by ground truth, then penalizes when
    the mean predicted score gap is smaller than `margin`.  This directly
    benefits knapsack segment selection (F1) by making the score distribution
    more discriminative.

    Safe: bounded in [0, margin], never diverges.
    """
    s = logits.float().view(-1)
    t = target.float().view(-1)
    n = s.numel()
    if n < 4:
        return s.new_zeros(())
    k = max(int(n * top_frac), 1)
    _, top_idx = t.topk(k)
    _, bot_idx = t.topk(k, largest=False)
    gap = s[top_idx].mean() - s[bot_idx].mean()
    return F.relu(margin - gap)


def segment_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    picks: torch.Tensor,
    change_points: torch.Tensor,
    max_pairs: int = 48,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Segment-level pairwise ranking: groups frame predictions by change_point
    segments, computes segment average scores, then applies hinge loss on
    segment pairs.

    Directly optimizes the segment ordering that knapsack uses for F1.
    Also benefits Tau/Rho by encouraging globally coherent importance ranking.
    """
    device = logits.device
    logits_f = logits.float().view(-1)
    target_f = target.float().view(-1)

    if change_points is None or change_points.numel() == 0:
        return logits.new_zeros(())

    picks_d = picks.long().to(device)
    n_seg = change_points.shape[0]
    seg_preds: list[torch.Tensor] = []
    seg_gts: list[torch.Tensor] = []

    for s in range(n_seg):
        a = int(change_points[s, 0].item())
        b = int(change_points[s, 1].item())
        mask = (picks_d >= a) & (picks_d < b)
        if mask.sum().item() < 1:
            continue
        seg_preds.append(logits_f[mask].mean())
        seg_gts.append(target_f[mask].mean())

    if len(seg_preds) < 2:
        return logits.new_zeros(())

    seg_preds_t = torch.stack(seg_preds)
    seg_gts_t = torch.stack(seg_gts).detach()

    n_s = len(seg_preds_t)
    idx_i = torch.randint(0, n_s, (max_pairs,), device=device)
    idx_j = torch.randint(0, n_s, (max_pairs,), device=device)
    gi, gj = seg_gts_t[idx_i], seg_gts_t[idx_j]
    valid = gi > gj
    if not valid.any():
        swap = gj > gi
        if not swap.any():
            return logits.new_zeros(())
        idx_i, idx_j = idx_j[swap], idx_i[swap]
    else:
        idx_i, idx_j = idx_i[valid], idx_j[valid]

    diff = seg_preds_t[idx_i] - seg_preds_t[idx_j]
    return F.relu(margin - diff).mean()


def list_mle_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    max_items: int = 256,
) -> torch.Tensor:
    """
    ListMLE: listwise ranking loss that optimizes the full permutation probability.
    Much stronger ranking signal than random pairwise sampling — considers global
    ordering context rather than isolated pairs.

    For efficiency, samples a mix of top/bottom/random items when n > max_items.
    """
    s = logits.float().view(-1)
    t = target.float().view(-1)
    n = s.numel()
    if n < 2:
        return s.new_zeros(())
    if n > max_items:
        k = max_items // 3
        _, top_idx = t.topk(min(k, n))
        _, bot_idx = t.topk(min(k, n), largest=False)
        rand_idx = torch.randperm(n, device=s.device)[: max_items - 2 * k]
        sel = torch.cat([top_idx, bot_idx, rand_idx]).unique()
        s, t = s[sel], t[sel]
    sorted_idx = t.argsort(descending=True)
    y_sorted = s[sorted_idx]
    cumsums = torch.logcumsumexp(y_sorted.flip(0), dim=0).flip(0)
    return -(y_sorted - cumsums).mean()


def query_contrastive_loss(
    score_pos: torch.Tensor,
    score_neg: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    max(0, m - s_pos + s_neg) on scalar matching scores (higher = more match).
    """
    return F.relu(margin - score_pos + score_neg)
