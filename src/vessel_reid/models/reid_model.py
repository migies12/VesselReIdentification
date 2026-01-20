from typing import Optional

import torch
from torch import nn
from torchvision import models


class ReIDModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dim: int = 256,
        use_length: bool = False,
        length_embed_dim: int = 16,
        pretrained: bool = True,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            feature_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.use_length = use_length
        self.length_embed_dim = length_embed_dim
        self.dropout_p = dropout_p

        self.embed = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        if self.use_length:
            self.length_proj = nn.Sequential(
                nn.Linear(1, length_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(length_embed_dim, length_embed_dim),
            )
            fused_dim = embedding_dim + length_embed_dim
            self.fuse = nn.Linear(fused_dim, embedding_dim)
        else:
            self.length_proj = None
            self.fuse = None

    def forward(self, x: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.backbone(x)
        emb = self.embed(features)

        if self.use_length:
            if length is None:
                raise ValueError("length tensor is required when use_length=true")
            length_emb = self.length_proj(length)
            emb = torch.cat([emb, length_emb], dim=1)
            emb = self.fuse(emb)

        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb
