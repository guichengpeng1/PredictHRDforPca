import timm
import torch
import torch.nn as nn


def build_torchvision_resnet(backbone_name: str, pretrained: bool):
    from torchvision.models import (
        ResNet18_Weights,
        ResNet34_Weights,
        ResNet50_Weights,
        resnet18,
        resnet34,
        resnet50,
    )

    builders = {
        "tv_resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        "tv_resnet34": (resnet34, ResNet34_Weights.DEFAULT),
        "tv_resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    }
    if backbone_name not in builders:
        raise ValueError(f"Unsupported torchvision backbone: {backbone_name}")

    builder, default_weights = builders[backbone_name]
    weights = default_weights if pretrained else None
    model = builder(weights=weights)
    num_features = int(model.fc.in_features)
    model.fc = nn.Identity()
    model.num_features = num_features
    return model


class GatedAttentionPool(nn.Module):
    def __init__(self, embed_dim: int, attention_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.v = nn.Linear(embed_dim, attention_dim)
        self.u = nn.Linear(embed_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor):
        gated = torch.tanh(self.v(features)) * torch.sigmoid(self.u(features))
        attn_logits = self.w(self.dropout(gated)).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * features, dim=1)
        return pooled, attn_weights


class WSIAttentionMIL(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if backbone_name.startswith("tv_"):
            self.backbone = build_torchvision_resnet(backbone_name, pretrained)
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
            )
        self.embed_dim = int(getattr(self.backbone, "num_features"))
        self.pool = GatedAttentionPool(self.embed_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(self.embed_dim, 1)
        self.status_head = nn.Linear(self.embed_dim, 1)

    def forward(self, bags: torch.Tensor):
        batch_size, num_tiles, channels, height, width = bags.shape
        flat_bags = bags.view(batch_size * num_tiles, channels, height, width)
        features = self.backbone(flat_bags)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        features = features.view(batch_size, num_tiles, -1)
        pooled, attention = self.pool(features)
        pooled = self.head(pooled)
        score = self.score_head(pooled).squeeze(-1)
        status_logits = self.status_head(pooled).squeeze(-1)
        return {
            "score": score,
            "status_logits": status_logits,
            "attention": attention,
        }
