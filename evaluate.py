
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import baselines
import config
from dataset import ScenePairDataset, SceneOnlyDataset, eval_transform, load_pairs
from metrics import (beyond_accuracy_metrics, rank_metrics, ranks_from_scores,
                     ranks_within_mask)
from model import CompatibilityNet
from train import evaluate_auc, split_by_scene


def load_model(device: torch.device) -> tuple[CompatibilityNet, dict]:
    ckpt = torch.load(config.CHECKPOINT, map_location=device, weights_only=True)
    model = CompatibilityNet(ckpt["embed_dim"], ckpt["color_embed_dim"],
                             ckpt["hist_bins"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def embed_scenes(model: CompatibilityNet, pairs: list[dict], device: torch.device,
                 style_dim: int, color_dim: int, batch_size: int = 64,
                 num_workers: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(SceneOnlyDataset(pairs), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    style = torch.zeros(len(pairs), style_dim)
    color = torch.zeros(len(pairs), color_dim)
    for imgs, idxs in loader:
        s, c = model.embed_scene(imgs.to(device))
        style[idxs] = s.cpu()
        color[idxs] = c.cpu()
    return style, color


def blended_scores(q_style: torch.Tensor, q_color: torch.Tensor,
                   c_style: torch.Tensor, c_color: torch.Tensor,
                   color_weight: float) -> torch.Tensor:
    return ((1 - color_weight) * (q_style @ c_style.T)
            + color_weight * (q_color @ c_color.T))





def print_table(results: dict[str, dict], columns: list[str]) -> None:
    name_w = max(len(n) for n in results) + 2
    header = "method".ljust(name_w) + "".join(c.rjust(16) for c in columns)
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        row = name.ljust(name_w)
        for c in columns:
            value = m.get(c)
            if value is None:
                row += "-".rjust(16)
            elif "rank" in c and "mrr" not in c:
                row += f"{value:>16.1f}"
            else:
                row += f"{value:>16.4f}"
        print(row)


def plot_recall(results: dict[str, dict], ks: list[int], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, m in results.items():
        ax.plot(ks, [m[f"recall@{k}"] for k in ks], marker="o", label=name)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Full-catalog retrieval")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--color-weight", type=float, default=config.COLOR_SCORE_WEIGHT,
                        help="blend weight of the color/pattern score")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50],
                        help="cutoffs for Recall@K / NDCG@K")
    parser.add_argument("--div-k", type=int, default=10,
                        help="cutoff for the beyond-accuracy metrics")
    parser.add_argument("--n-sampled-neg", type=int, default=50,
                        help="random negatives per query for the sampled-AUC protocol")
    parser.add_argument("--frozen-arch", default="resnet18",
                        choices=["resnet18", "resnet50"],
                        help="backbone for the untrained visual-similarity baseline")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 is safest on Windows)")
    parser.add_argument("--skip-sampled-auc", action="store_true",
                        help="skip protocol 2 (it re-reads every image)")
    parser.add_argument("--skip-category", action="store_true",
                        help="skip protocol 3 (category-restricted retrieval)")
    parser.add_argument("--output", default=None, help="write all metrics as JSON")
    parser.add_argument("--plot", default=None, help="write a Recall@K chart, e.g. recall.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, ckpt = load_model(device)
    print(f"checkpoint val AUC (sampled, blended): {ckpt.get('val_auc', float('nan')):.4f}")

    pairs = load_pairs()
    train_pairs, val_pairs = split_by_scene(pairs)

    catalog = torch.load(config.CATALOG_EMB, weights_only=True)
    signatures = catalog["signatures"]
    sig_to_idx = {sig: i for i, sig in enumerate(signatures)}
    c_style, c_color = catalog["embeddings"], catalog["color_embeddings"]

    eval_pairs = [p for p in val_pairs if p["product"] in sig_to_idx]
    if dropped := len(val_pairs) - len(eval_pairs):
        print(f"skipping {dropped} held-out pairs whose product is not in the catalog")
    target_idx = torch.tensor([sig_to_idx[p["product"]] for p in eval_pairs])
    n_q, n_cat = len(eval_pairs), len(signatures)
    print(f"{n_q} queries against {n_cat} candidate products "
          f"({len(train_pairs)} training pairs)")

    print("\npreparing the frozen-backbone baseline")
    frozen_scene, frozen_prod = baselines.frozen_features(
        eval_pairs, signatures, device, arch=args.frozen_arch,
        num_workers=args.num_workers)

    print("embedding scenes with the trained model")
    q_style, q_color = embed_scenes(model, eval_pairs, device,
                                    ckpt["embed_dim"], ckpt["color_embed_dim"],
                                    num_workers=args.num_workers)


    methods: dict[str, callable] = {
        "random":
            lambda: baselines.random_scores(n_q, n_cat, config.SEED),
        "popularity":
            lambda: baselines.popularity_scores(train_pairs, signatures, n_q, config.SEED),
        f"frozen {args.frozen_arch}":
            lambda: baselines.similarity_scores(frozen_scene, frozen_prod),
        "learned (style)":
            lambda: blended_scores(q_style, q_color, c_style, c_color, 0.0),
        f"learned (style+color w={args.color_weight})":
            lambda: blended_scores(q_style, q_color, c_style, c_color, args.color_weight),
    }

    
    cat_mask = None
    cat_sizes = None
    if not args.skip_category and "categories" in catalog:
        leaves = [c.split("|")[-1] for c in catalog["categories"]]
        vocab = {c: i for i, c in enumerate(sorted(set(leaves)))}
        cat_id = torch.tensor([vocab[c] for c in leaves])
        target_cat = cat_id[target_idx]                      # (Q,)
        cat_mask = cat_id.unsqueeze(0) == target_cat.unsqueeze(1)   # (Q, N)
        cat_sizes = cat_mask.sum(dim=1).numpy()
        n_unknown = sum(1 for c in leaves if c == "Unknown")
        print(f"category-restricted protocol: {len(vocab)} leaf categories, "
              f"median {int(np.median(cat_sizes))} candidates per query "
              f"(min {cat_sizes.min()}, max {cat_sizes.max()}"
              + (f"; {n_unknown} catalog items have no category)" if n_unknown else ")"))

    results: dict[str, dict] = {}
    results_cat: dict[str, dict] = {}
    for name, build_scores in methods.items():
        scores = build_scores()
        ranks = ranks_from_scores(scores, target_idx)
        m = rank_metrics(ranks, args.ks, n_cat)
        m |= beyond_accuracy_metrics(scores, args.div_k, n_cat, frozen_prod)
        results[name] = m
        if cat_mask is not None:
            results_cat[name] = rank_metrics(
                ranks_within_mask(scores, target_idx, cat_mask), args.ks, cat_sizes)
        del scores

    print("\n=== Full-catalog retrieval: accuracy ===")
    print_table(results, [f"recall@{k}" for k in args.ks]
                + ["mrr", "auc_full", "median_rank"])

    print("\n=== Beyond accuracy (top-%d lists) ===" % args.div_k)
    print_table(results, [f"coverage@{args.div_k}", f"diversity@{args.div_k}",
                          f"gini@{args.div_k}", f"head1pct_share@{args.div_k}"])

    if results_cat:
        print(f"\n=== Protocol 3: category-restricted retrieval "
              f"(median {int(np.median(cat_sizes))} candidates) ===")
        print("This is the setting the web app operates in once a category is "
              "chosen,\nso it is the closest estimate of what a user actually sees.")
        print_table(results_cat, [f"recall@{k}" for k in args.ks]
                    + ["mrr", "auc_full", "median_rank"])

    sampled = None
    if not args.skip_sampled_auc:
        val_loader = DataLoader(ScenePairDataset(eval_pairs, eval_transform()),
                                batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=(device.type == "cuda"))
        style_auc, blend_auc = evaluate_auc(model, val_loader, device,
                                            n_neg=args.n_sampled_neg)
        sampled = {"style_only": style_auc, "blended": blend_auc,
                   "n_neg": args.n_sampled_neg}
        print(f"\n=== Protocol 2: sampled-negative AUC "
              f"({args.n_sampled_neg} negatives/query) ===")
        print(f"  learned (style)        {style_auc:.4f}")
        print(f"  learned (style+color)  {blend_auc:.4f}")
        print("  (both protocols estimate the same quantity -- P(true product scored "
              "above a random\n   non-relevant one) -- so close agreement with "
              "auc_full above is the expected\n   outcome and cross-validates the two "
              "implementations)")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "n_queries": n_q,
                "n_catalog": n_cat,
                "n_train_pairs": len(train_pairs),
                "color_weight": args.color_weight,
                "frozen_arch": args.frozen_arch,
                "div_k": args.div_k,
                "seed": config.SEED,
                "full_catalog": results,
                "by_category": results_cat or None,
                "category_candidates": None if cat_sizes is None else {
                    "median": float(np.median(cat_sizes)),
                    "mean": float(cat_sizes.mean()),
                    "min": int(cat_sizes.min()),
                    "max": int(cat_sizes.max()),
                },
                "sampled_auc": sampled,
            }, f, indent=2)
        print(f"\nwrote {args.output}")

    if args.plot:
        plot_recall(results, args.ks, args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
