
from __future__ import annotations

import numpy as np
import torch




def ranks_from_scores(scores: torch.Tensor, target_idx: torch.Tensor) -> np.ndarray:
   
    assert int(target_idx.max()) < scores.shape[1], \
        "target index outside the catalog -- a target product is missing"
    target_scores = scores.gather(1, target_idx.unsqueeze(1))       # (Q, 1)
    better = (scores > target_scores).sum(dim=1)                    # (Q,)
    return (better + 1).numpy()


def rank_metrics(ranks: np.ndarray, ks: list[int],
                 n_catalog: int | np.ndarray) -> dict:
   
    
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks))
    for k in ks:
        dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
        out[f"ndcg@{k}"] = float(dcg.mean())
   
    n = np.broadcast_to(np.asarray(n_catalog, dtype=float), ranks.shape)
    usable = n > 1
    n_u, r_u = n[usable], ranks[usable]          # mask first: n == 1 would divide by zero
    out["auc_full"] = float(np.mean((n_u - r_u) / (n_u - 1))) if usable.any() else float("nan")
    out["n_auc_queries"] = int(usable.sum())
    out["mean_rank"] = float(ranks.mean())
    out["median_rank"] = float(np.median(ranks))
    return out


def ranks_within_mask(scores: torch.Tensor, target_idx: torch.Tensor,
                      mask: torch.Tensor) -> np.ndarray:

    target_scores = scores.gather(1, target_idx.unsqueeze(1))
    better = ((scores > target_scores) & mask).sum(dim=1)
    return (better + 1).numpy()


# --------------------------------------------------------------------------
# beyond accuracy
# --------------------------------------------------------------------------


def catalog_coverage(topk: np.ndarray, n_catalog: int) -> float:
    
    return float(len(np.unique(topk)) / n_catalog)


def exposure_counts(topk: np.ndarray, n_catalog: int) -> np.ndarray:
   
    return np.bincount(topk.ravel(), minlength=n_catalog)


def gini(counts: np.ndarray) -> float:
    
    x = np.sort(np.asarray(counts, dtype=float))
    total = x.sum()
    if total == 0:
        return 0.0
    n = len(x)
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * x).sum() / (n * total))


def head_share(counts: np.ndarray, fraction: float = 0.01) -> float:
   
    total = counts.sum()
    if total == 0:
        return 0.0
    n_head = max(1, int(round(len(counts) * fraction)))
    return float(np.sort(counts)[::-1][:n_head].sum() / total)


def intra_list_diversity(topk: np.ndarray, ref_emb: torch.Tensor) -> float:
    
    emb = ref_emb[torch.from_numpy(topk)]              # (Q, K, D)
    sim = emb @ emb.transpose(1, 2)                    # (Q, K, K)
    k = emb.shape[1]
    if k < 2:
        return 0.0
    iu = torch.triu_indices(k, k, offset=1)
    pairwise = sim[:, iu[0], iu[1]]                    # (Q, n_pairs)
    return float((1.0 - pairwise).mean())


def beyond_accuracy_metrics(scores: torch.Tensor, k: int, n_catalog: int,
                            ref_emb: torch.Tensor) -> dict:
   
    topk = torch.topk(scores, k=k, dim=1).indices.numpy()
    counts = exposure_counts(topk, n_catalog)
    return {
        f"coverage@{k}": catalog_coverage(topk, n_catalog),
        f"diversity@{k}": intra_list_diversity(topk, ref_emb),
        f"gini@{k}": gini(counts),
        f"head1pct_share@{k}": head_share(counts, 0.01),
    }
