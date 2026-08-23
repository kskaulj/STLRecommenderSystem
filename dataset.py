
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_pairs() -> list[dict]:
    with open(config.PAIRS_FILE, encoding="utf-8") as f:
        return json.load(f)


def mask_bbox(img: Image.Image, bbox: list[float]) -> Image.Image:
    
    w, h = img.size
    left, top, right, bottom = bbox
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [left * w, top * h, right * w, bottom * h], fill=(128, 128, 128)
    )
    return img


class ScenePairDataset(Dataset):
    def __init__(self, pairs: list[dict], transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        scene = Image.open(config.SCENES_DIR / f"{pair['scene']}.jpg").convert("RGB")
        scene = mask_bbox(scene, pair["bbox"])
        product = Image.open(config.PRODUCTS_DIR / f"{pair['product']}.jpg").convert("RGB")
        return self.transform(scene), self.transform(product)


class SceneOnlyDataset(Dataset):
    

    def __init__(self, pairs: list[dict], transform=None):
        self.pairs = pairs
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        scene = Image.open(config.SCENES_DIR / f"{pair['scene']}.jpg").convert("RGB")
        scene = mask_bbox(scene, pair["bbox"])
        return self.transform(scene), idx


class ProductImageDataset(Dataset):
    

    def __init__(self, signatures: list[str], transform=None):
        self.signatures = signatures
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.signatures)

    def __getitem__(self, idx: int):
        img = Image.open(config.PRODUCTS_DIR / f"{self.signatures[idx]}.jpg").convert("RGB")
        return self.transform(img), idx
