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

    def load_state_dict(self, state_dict, strict: bool = True):
        if (
            isinstance(self.embed[0], nn.Dropout)
            and "embed.0.weight" in state_dict
            and "embed.2.weight" not in state_dict
        ):
            remapped = {}
            for key, value in state_dict.items():
                if key.startswith("embed.0."):
                    remapped[key.replace("embed.0.", "embed.1.", 1)] = value
                elif key.startswith("embed.1."):
                    remapped[key.replace("embed.1.", "embed.2.", 1)] = value
                else:
                    remapped[key] = value
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)
