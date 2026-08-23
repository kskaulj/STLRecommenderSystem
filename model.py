
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _projection_head(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


class CompatibilityNet(nn.Module):
    def __init__(self, embed_dim: int = 128, color_embed_dim: int = 64,
                 hist_bins: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features       
        shallow_dim = 128                        
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.hist_bins = hist_bins

        self.scene_head = _projection_head(feat_dim, embed_dim)
        self.product_head = _projection_head(feat_dim, embed_dim)

        color_in = hist_bins ** 3 + 2 * shallow_dim   # histogram + mean/std stats
        self.scene_color_head = _projection_head(color_in, color_embed_dim, hidden=128)
        self.product_color_head = _projection_head(color_in, color_embed_dim, hidden=128)

        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _backbone_feats(self, x: torch.Tensor):
       
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer1(x)
        shallow = b.layer2(x)                        
        deep = b.layer4(b.layer3(shallow))
        return b.avgpool(deep).flatten(1), shallow

    def _color_hist(self, x: torch.Tensor) -> torch.Tensor:
       
        bins = self.hist_bins
        with torch.no_grad():
            img = (x * self._std + self._mean).clamp(0, 1)
            q = (img * bins).long().clamp(max=bins - 1)      # (B, 3, H, W)
            idx = ((q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]).flatten(1)
            hist = torch.zeros(x.size(0), bins ** 3, device=x.device)
            hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=hist.dtype))
            hist = (hist / idx.size(1)).sqrt()               # soften dominant bins
        return hist

    def _appearance_feats(self, x: torch.Tensor, shallow: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([shallow.mean(dim=(2, 3)), shallow.std(dim=(2, 3))], dim=1)
        return torch.cat([self._color_hist(x).to(stats.dtype), stats], dim=1)

    def trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        
        return self._backbone_feats(x)[0]

    def embed_scene(self, x: torch.Tensor):
        feat, shallow = self._backbone_feats(x)
        style = F.normalize(self.scene_head(feat), dim=-1)
        color = F.normalize(self.scene_color_head(self._appearance_feats(x, shallow)), dim=-1)
        return style, color

    def embed_product(self, x: torch.Tensor):
        feat, shallow = self._backbone_feats(x)
        style = F.normalize(self.product_head(feat), dim=-1)
        color = F.normalize(self.product_color_head(self._appearance_feats(x, shallow)), dim=-1)
        return style, color

    def forward(self, scenes: torch.Tensor, products: torch.Tensor):
        s_style, s_color = self.embed_scene(scenes)
        p_style, p_color = self.embed_product(products)
        return s_style, s_color, p_style, p_color


def batch_hinge_loss(scene_emb: torch.Tensor, prod_emb: torch.Tensor,
                     margin: float) -> torch.Tensor:
    
    d2 = torch.cdist(scene_emb, prod_emb).pow(2)          # (B, B)
    pos = d2.diag()                                        # (B,)
    eye = torch.eye(len(d2), dtype=torch.bool, device=d2.device)
    
    viol_p = F.relu(pos.unsqueeze(1) - d2 + margin).masked_fill(eye, 0)
    viol_s = F.relu(pos.unsqueeze(0) - d2 + margin).masked_fill(eye, 0)
    n_neg = len(d2) - 1
    return (viol_p.sum(dim=1) + viol_s.sum(dim=0)).mean() / (2 * n_neg)
