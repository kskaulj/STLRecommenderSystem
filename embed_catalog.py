import json

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from dataset import eval_transform
from model import CompatibilityNet


class ProductImages(Dataset):
    def __init__(self, signatures: list[str]):
        self.signatures = signatures
        self.transform = eval_transform()

    def __len__(self) -> int:
        return len(self.signatures)

    def __getitem__(self, idx: int):
        sig = self.signatures[idx]
        img = Image.open(config.PRODUCTS_DIR / f"{sig}.jpg").convert("RGB")
        return self.transform(img), idx


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(config.CHECKPOINT, map_location=device, weights_only=True)
    model = CompatibilityNet(ckpt["embed_dim"], ckpt["color_embed_dim"],
                             ckpt["hist_bins"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    signatures = sorted(p.stem for p in config.PRODUCTS_DIR.glob("*.jpg"))
    print(f"embedding {len(signatures)} products (val AUC of model: "
          f"{ckpt.get('val_auc', float('nan')):.4f})")

    with open(config.FASHION_CAT_JSON, encoding="utf-8") as f:
        categories = json.load(f)

    loader = DataLoader(ProductImages(signatures), batch_size=64,
                        num_workers=4, pin_memory=True)
    embs = torch.zeros(len(signatures), ckpt["embed_dim"])
    color_embs = torch.zeros(len(signatures), ckpt["color_embed_dim"])
    for imgs, idxs in loader:
        style, color = model.embed_product(imgs.to(device))
        embs[idxs] = style.cpu()
        color_embs[idxs] = color.cpu()

    torch.save({
        "signatures": signatures,
        "embeddings": embs,
        "color_embeddings": color_embs,
        "categories": [categories.get(s, "Unknown") for s in signatures],
    }, config.CATALOG_EMB)
    print(f"wrote {config.CATALOG_EMB}")


if __name__ == "__main__":
    main()
