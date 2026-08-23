
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import ScenePairDataset, eval_transform, load_pairs, train_transform
from model import CompatibilityNet, batch_hinge_loss


def split_by_scene(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    scenes = sorted({p["scene"] for p in pairs})
    random.Random(config.SEED).shuffle(scenes)
    n_val = max(1, int(len(scenes) * config.VAL_FRACTION))
    val_scenes = set(scenes[:n_val])
    train = [p for p in pairs if p["scene"] not in val_scenes]
    val = [p for p in pairs if p["scene"] in val_scenes]
    return train, val


@torch.no_grad()
def evaluate_auc(model: CompatibilityNet, loader: DataLoader,
                 device: torch.device, n_neg: int = 50) -> tuple[float, float]:
 
    model.eval()
    scene_embs, prod_embs = [], []
    scene_color, prod_color = [], []
    for scenes, products in loader:
        s, sc, p, pc = model(scenes.to(device), products.to(device))
        scene_embs.append(s.cpu())
        prod_embs.append(p.cpu())
        scene_color.append(sc.cpu())
        prod_color.append(pc.cpu())
    S, P = torch.cat(scene_embs), torch.cat(prod_embs)
    SC, PC = torch.cat(scene_color), torch.cat(prod_color)
    n = len(S)
    rng = np.random.default_rng(config.SEED)
    w = config.COLOR_SCORE_WEIGHT
    wins_style, wins_blend, total = 0.0, 0.0, 0
    pos_style = (S * P).sum(dim=1)           # cosine; higher = more compatible
    pos_blend = (1 - w) * pos_style + w * (SC * PC).sum(dim=1)
    for i in range(n):
        neg_idx = rng.choice(n - 1, size=min(n_neg, n - 1), replace=False)
        neg_idx = np.where(neg_idx >= i, neg_idx + 1, neg_idx)
        neg_idx = torch.from_numpy(neg_idx)
        neg_style = S[i] @ P[neg_idx].T
        neg_blend = (1 - w) * neg_style + w * (SC[i] @ PC[neg_idx].T)
        wins_style += (pos_style[i] > neg_style).sum().item() \
            + 0.5 * (pos_style[i] == neg_style).sum().item()
        wins_blend += (pos_blend[i] > neg_blend).sum().item() \
            + 0.5 * (pos_blend[i] == neg_blend).sum().item()
        total += len(neg_idx)
    return wins_style / total, wins_blend / total


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    pairs = load_pairs()
    train_pairs, val_pairs = split_by_scene(pairs)
    print(f"{len(train_pairs)} train pairs, {len(val_pairs)} val pairs")

    train_ds = ScenePairDataset(train_pairs, train_transform())
    val_ds = ScenePairDataset(val_pairs, eval_transform())
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="continue from the saved checkpoint")
    args = parser.parse_args()

    model = CompatibilityNet(config.EMBED_DIM, config.COLOR_EMBED_DIM,
                             config.HIST_BINS).to(device)
    if args.resume and config.CHECKPOINT.exists():
        ckpt = torch.load(config.CHECKPOINT, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"resumed from checkpoint (AUC {ckpt.get('val_auc', 0):.4f})")
    # fine-tune the last two residual stages of the backbone + all heads
    for name, param in model.backbone.named_parameters():
        param.requires_grad = name.startswith(("layer3", "layer4"))
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.layer3.parameters(), "lr": config.LR_BACKBONE},
        {"params": model.backbone.layer4.parameters(), "lr": config.LR_BACKBONE},
        {"params": model.scene_head.parameters(), "lr": config.LR_HEAD},
        {"params": model.product_head.parameters(), "lr": config.LR_HEAD},
        {"params": model.scene_color_head.parameters(), "lr": config.LR_HEAD},
        {"params": model.product_color_head.parameters(), "lr": config.LR_HEAD},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_auc = 0.0
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (scenes, products) in enumerate(train_loader, 1):
            scenes = scenes.to(device, non_blocking=True)
            products = products.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=device.type == "cuda"):
                s_emb, s_col, p_emb, p_col = model(scenes, products)
                loss = (batch_hinge_loss(s_emb.float(), p_emb.float(), config.MARGIN)
                        + config.COLOR_LOSS_WEIGHT
                        * batch_hinge_loss(s_col.float(), p_col.float(), config.MARGIN))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f}", flush=True)
        style_auc, blend_auc = evaluate_auc(model, val_loader, device)
        print(f"epoch {epoch}: loss {running / len(train_loader):.4f} "
              f"val AUC style {style_auc:.4f} blended {blend_auc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if blend_auc > best_auc:
            best_auc = blend_auc
            torch.save({"state_dict": model.state_dict(),
                        "embed_dim": config.EMBED_DIM,
                        "color_embed_dim": config.COLOR_EMBED_DIM,
                        "hist_bins": config.HIST_BINS,
                        "val_auc": blend_auc,
                        "val_auc_style": style_auc}, config.CHECKPOINT)
            print(f"  saved checkpoint (blended AUC {blend_auc:.4f})")
    print(f"done. best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
