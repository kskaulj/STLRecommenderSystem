
import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from dataset import eval_transform, load_pairs
from model import CompatibilityNet

TRUNK_DIM = 512


def load_labels() -> dict[str, str]:
    
    with open(config.FASHION_CAT_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    on_disk = {p.stem for p in config.PRODUCTS_DIR.glob("*.jpg")}
    return {sig: cat.split("|")[-1] for sig, cat in raw.items() if sig in on_disk}


def crop_bbox(img: Image.Image, bbox: list[float]) -> Image.Image:

    w, h = img.size
    left, top, right, bottom = bbox
    pad_w, pad_h = (right - left) * config.PROBE_BBOX_PAD, (bottom - top) * config.PROBE_BBOX_PAD
    box = (int(max(0.0, left - pad_w) * w), int(max(0.0, top - pad_h) * h),
           int(min(1.0, right + pad_w) * w), int(min(1.0, bottom + pad_h) * h))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:      # degenerate box
        return img
    return img.crop(box)


def build_records(labels: dict[str, str], use_scene_crops: bool) -> list[dict]:
    records = [{"source": "catalog", "signature": sig, "label": lab}
               for sig, lab in sorted(labels.items())]

    dropped = 0
    for pair in load_pairs():
        label = labels.get(pair["product"])
        if label is None:
            continue
        left, top, right, bottom = pair["bbox"]
        if right <= left or bottom <= top or \
                (right - left) * (bottom - top) < config.PROBE_MIN_BBOX_AREA:
            dropped += 1
            continue
        if not (config.SCENES_DIR / f"{pair['scene']}.jpg").exists():
            continue
        records.append({"source": "scene", "signature": pair["product"],
                        "scene": pair["scene"], "bbox": pair["bbox"], "label": label})
    if dropped:
        print(f"  skipped {dropped} scene crops (box too small or degenerate)")
    return records


def split_records(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    
    sigs_by_class = defaultdict(set)
    for r in records:
        sigs_by_class[r["label"]].add(r["signature"])

    rng = random.Random(seed)
    val_sigs: set[str] = set()
    for label in sorted(sigs_by_class):
        sigs = sorted(sigs_by_class[label])
        rng.shuffle(sigs)
        n_val = max(1, int(len(sigs) * config.PROBE_VAL_FRACTION))
        val_sigs.update(sigs[:n_val])

    train = [r for r in records if r["signature"] not in val_sigs]
    val = [r for r in records if r["signature"] in val_sigs]
    return train, val


class ProbeImages(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records
        self.transform = eval_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        if r["source"] == "catalog":
            img = Image.open(config.PRODUCTS_DIR / f"{r['signature']}.jpg").convert("RGB")
        else:
            img = Image.open(config.SCENES_DIR / f"{r['scene']}.jpg").convert("RGB")
            img = crop_bbox(img, r["bbox"])
        return self.transform(img), idx


@torch.no_grad()
def extract_features(model, records: list[dict], device, flip: bool = False) -> torch.Tensor:
    loader = DataLoader(ProbeImages(records), batch_size=64, num_workers=4,
                        pin_memory=True)
    feats = torch.zeros(len(records), TRUNK_DIM)
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=True)
        if flip:
            imgs = torch.flip(imgs, dims=[3])
        feats[idxs] = model.trunk_features(imgs).float().cpu()
    return feats




def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def slot_of(label: str) -> str:
    for name, leaves, _ in config.OUTFIT_SLOTS:
        if label in leaves:
            return name
    return label


def score(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    
    cm = confusion_matrix(y_true, y_pred, len(classes))
    precision, recall, f1 = per_class_prf(cm)

    
    slots = sorted({slot_of(c) for c in classes})
    slot_map = np.array([slots.index(slot_of(c)) for c in classes])
    slot_hits = int((slot_map[y_true] == slot_map[y_pred]).sum()) if len(y_true) else 0

    return {
        "n": int(cm.sum()),
        "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0,
        "macro_f1": float(f1.mean()),
        "slot_accuracy": slot_hits / len(y_true) if len(y_true) else 0.0,
        "precision": precision, "recall": recall, "f1": f1, "cm": cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.PROBE_EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no-scene-crops", action="store_true")                 )
    parser.add_argument("--backbone", choices=("compat", "imagenet"), default="compat",                      )
    parser.add_argument("--output", default="category_results.json")
    parser.add_argument("--no-save", action="store_true"                        )
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    labels = load_labels()
    records = build_records(labels, use_scene_crops=True)
    classes = sorted({r["label"] for r in records})
    class_idx = {c: i for i, c in enumerate(classes)}
    by_source = Counter(r["source"] for r in records)
    print(f"{len(records)} training images over {len(classes)} classes "
          f"({by_source['catalog']} catalog shots, {by_source['scene']} scene crops)")

    train_recs, val_recs = split_records(records, args.seed)
    if args.no_scene_crops:
        
        train_recs = [r for r in train_recs if r["source"] == "catalog"]
    print(f"{len(train_recs)} train / {len(val_recs)} val "
          f"(disjoint product signatures)")

    if args.backbone == "compat":
        ckpt = torch.load(config.CHECKPOINT, map_location=device, weights_only=True)
        model = CompatibilityNet(ckpt["embed_dim"], ckpt["color_embed_dim"],
                                 ckpt["hist_bins"], pretrained=False).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = CompatibilityNet(config.EMBED_DIM, config.COLOR_EMBED_DIM,
                                 config.HIST_BINS, pretrained=True).to(device)
    model.eval()
    print(f"trunk: {args.backbone}")

    print("extracting frozen trunk features…")
    X_train = torch.cat([extract_features(model, train_recs, device),
                         extract_features(model, train_recs, device, flip=True)])
    y_train = torch.tensor([class_idx[r["label"]] for r in train_recs]).repeat(2)
    X_val = extract_features(model, val_recs, device)
    y_val = torch.tensor([class_idx[r["label"]] for r in val_recs])

    
    feat_mean = X_train.mean(dim=0)
    feat_std = X_train.std(dim=0).clamp_min(1e-6)
    X_train = ((X_train - feat_mean) / feat_std).to(device)
    X_val = ((X_val - feat_mean) / feat_std).to(device)
    y_train, y_val = y_train.to(device), y_val.to(device)

    counts = torch.bincount(y_train, minlength=len(classes)).float()
    weights = (len(y_train) / (len(classes) * counts.clamp_min(1))).to(device)
    print("class balance (train images): "
          + ", ".join(f"{c}={int(n)}" for c, n in zip(classes, counts.tolist())))

    clf = nn.Linear(TRUNK_DIM, len(classes)).to(device)
    optimiser = torch.optim.AdamW(clf.parameters(), lr=config.PROBE_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best = {"macro_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        clf.train()
        perm = torch.randperm(len(X_train), device=device)
        total = 0.0
        for start in range(0, len(perm), config.PROBE_BATCH):
            batch = perm[start:start + config.PROBE_BATCH]
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(clf(X_train[batch]), y_train[batch])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)

        clf.eval()
        with torch.no_grad():
            pred = clf(X_val).argmax(dim=1)
        cm = confusion_matrix(y_val.cpu().numpy(), pred.cpu().numpy(), len(classes))
        macro_f1 = float(per_class_prf(cm)[2].mean())
        if macro_f1 > best["macro_f1"]:
            best = {"macro_f1": macro_f1, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in clf.state_dict().items()}}
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}  loss {total / len(perm):.4f}  "
                  f"val macro-F1 {macro_f1:.4f}", flush=True)

    clf.load_state_dict(best["state"])
    with torch.no_grad():
        pred = clf(X_val).argmax(dim=1).cpu().numpy()
    truth = y_val.cpu().numpy()
    source = np.array([r["source"] for r in val_recs])

    overall = score(truth, pred, classes)
    by_source = {s: score(truth[source == s], pred[source == s], classes)
                 for s in sorted(set(source))}
    cm = overall["cm"]
    precision, recall, f1 = overall["precision"], overall["recall"], overall["f1"]
    accuracy, slot_accuracy = overall["accuracy"], overall["slot_accuracy"]

    print(f"\nbest epoch {best['epoch']}")
    print(f"{'validation subset':<22}{'n':>7}{'acc':>8}{'macroF1':>9}{'slot acc':>10}")
    print(f"{'all':<22}{overall['n']:>7}{accuracy:>8.3f}"
          f"{overall['macro_f1']:>9.3f}{slot_accuracy:>10.3f}")
    for s, m in by_source.items():
        name = "catalog product shots" if s == "catalog" else "in-scene bbox crops"
        print(f"{name:<22}{m['n']:>7}{m['accuracy']:>8.3f}"
              f"{m['macro_f1']:>9.3f}{m['slot_accuracy']:>10.3f}")

    
    with torch.no_grad():
        confidence = clf(X_val).softmax(dim=1).max(dim=1).values.cpu().numpy()
    slots_sorted = sorted({slot_of(c) for c in classes})
    slot_map = np.array([slots_sorted.index(slot_of(c)) for c in classes])
    correct_slot = slot_map[truth] == slot_map[pred]
    catalog = source == "catalog"

    coverage_table = []
    for threshold in (0.0, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9):
        keep = catalog & (confidence >= threshold)
        cov = keep.sum() / max(catalog.sum(), 1)
        acc_slot = correct_slot[keep].mean() if keep.any() else float("nan")
        acc_cat = (truth[keep] == pred[keep]).mean() if keep.any() else float("nan")
        print(f"{threshold:>10.2f}{cov:>10.3f}{acc_slot:>10.3f}{acc_cat:>9.3f}")
        coverage_table.append({"threshold": threshold, "coverage": float(cov),
                               "slot_accuracy": float(acc_slot),
                               "category_accuracy": float(acc_cat)})

    print(f"\n{'category':<28}{'prec':>7}{'rec':>7}{'F1':>7}{'n':>7}")
    for i, c in enumerate(classes):
        print(f"{c:<28}{precision[i]:>7.3f}{recall[i]:>7.3f}{f1[i]:>7.3f}{cm[i].sum():>7}")

    print("\nconfusion matrix (rows = true, cols = predicted)")
    width = max(len(c) for c in classes)
    print(" " * (width + 2) + "".join(f"{c[:6]:>7}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<{width + 2}}" + "".join(f"{v:>7}" for v in cm[i]))

    probe = {
        "weight": clf.weight.detach().cpu(),
        "bias": clf.bias.detach().cpu(),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "classes": classes,
        "val_accuracy": accuracy,
        "val_macro_f1": overall["macro_f1"],
        "val_slot_accuracy": slot_accuracy,
        "val_catalog_slot_accuracy": by_source.get("catalog", {}).get("slot_accuracy"),
        "trained_with_scene_crops": not args.no_scene_crops,
        "backbone": args.backbone,
    }
    if args.no_save:
        print(f"\n--no-save: left {config.CATEGORY_PROBE.name} untouched")
    else:
        torch.save(probe, config.CATEGORY_PROBE)
        print(f"\nwrote {config.CATEGORY_PROBE}")

    results = {
        "classes": classes,
        "n_train_images": len(train_recs), "n_val_images": len(val_recs),
        "trained_with_scene_crops": not args.no_scene_crops,
        "backbone": args.backbone,
        "best_epoch": best["epoch"],
        "accuracy": accuracy, "macro_f1": overall["macro_f1"],
        "slot_accuracy": slot_accuracy,
        "by_source": {s: {k: m[k] for k in ("n", "accuracy", "macro_f1", "slot_accuracy")}
                      for s, m in by_source.items()},
        "risk_coverage_catalog": coverage_table,
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(cm[i].sum())}
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(norm, cmap="magma_r", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Category probe — accuracy {accuracy:.3f}, "
                     f"macro-F1 {overall['macro_f1']:.3f}")
        for i in range(len(classes)):
            for j in range(len(classes)):
                if norm[i, j] >= 0.01:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, label="share of true class")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
