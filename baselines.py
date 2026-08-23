
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

import config
from dataset import ProductImageDataset, SceneOnlyDataset

_ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def _frozen_backbone(arch: str, device: torch.device) -> torch.nn.Module:
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_ARCHS)}")
    ctor, weights = _ARCHS[arch]
    net = ctor(weights=weights)
    net.fc = torch.nn.Identity()
    return net.eval().to(device)


@torch.no_grad()
def _embed(net: torch.nn.Module, dataset, device: torch.device,
           batch_size: int, num_workers: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    out: torch.Tensor | None = None
    for imgs, idxs in loader:
        feats = net(imgs.to(device)).cpu()
        if out is None:
            out = torch.zeros(len(dataset), feats.shape[1])
        out[idxs] = feats
    assert out is not None, "empty dataset"
    return F.normalize(out, dim=-1)


def frozen_features(pairs: list[dict], signatures: list[str], device: torch.device,
                    arch: str = "resnet18", batch_size: int = 64,
                    num_workers: int = 0, use_cache: bool = True
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    
    net = _frozen_backbone(arch, device)

    cache = config.MODELS_DIR / f"frozen_{arch}_products.pt"
    prod: torch.Tensor | None = None
    if use_cache and cache.exists():
        blob = torch.load(cache, weights_only=True)
        if blob.get("signatures") == signatures:
            prod = blob["features"]
            print(f"  reusing cached {arch} product features ({cache.name})")
    if prod is None:
        print(f"  embedding {len(signatures)} products with frozen {arch} ...")
        prod = _embed(net, ProductImageDataset(signatures), device, batch_size, num_workers)
        if use_cache:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"signatures": signatures, "features": prod}, cache)

    print(f"  embedding {len(pairs)} scenes with frozen {arch} ...")
    scene = _embed(net, SceneOnlyDataset(pairs), device, batch_size, num_workers)
    return scene, prod



# score matrices  (Q, N): higher = recommended earlier



def _jitter(scores: torch.Tensor, seed: int, scale: float = 1e-6) -> torch.Tensor:
    
    g = torch.Generator().manual_seed(seed)
    return scores + torch.rand(scores.shape, generator=g) * scale


def random_scores(n_queries: int, n_catalog: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_queries, n_catalog, generator=g)


def popularity_scores(train_pairs: list[dict], signatures: list[str],
                      n_queries: int, seed: int) -> torch.Tensor:
   
    sig_to_idx = {sig: i for i, sig in enumerate(signatures)}
    counts = np.zeros(len(signatures), dtype=np.float32)
    for pair in train_pairs:
        idx = sig_to_idx.get(pair["product"])
        if idx is not None:
            counts[idx] += 1.0
    row = torch.from_numpy(counts)
    return _jitter(row.unsqueeze(0).repeat(n_queries, 1), seed)


def similarity_scores(scene_feats: torch.Tensor, prod_feats: torch.Tensor) -> torch.Tensor:
    
    return scene_feats @ prod_feats.T
