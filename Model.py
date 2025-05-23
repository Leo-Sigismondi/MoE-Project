import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SingleModel(nn.Module):
    def __init__(
        self,
        trunk_channels: int = 256,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        num_classes: int = 10,
        trunk: 'Optional[nn.Module]' = None,
    ):
        super().__init__()
        # Use the same trunk as in MoE for fair comparison
        if trunk is None:
            self.trunk = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),  # 32→16
                nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.MaxPool2d(2),  # 16→8
                nn.AdaptiveAvgPool2d(1),  # → [B,256,1,1]
                nn.Flatten(),             # → [B,256]
            )
        else:
            self.trunk = trunk

        # Single "expert" MLP head
        self.head = nn.Sequential(
            nn.Linear(trunk_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        feats = self.trunk(x)      # [B, trunk_channels]
        logits = self.head(feats)  # [B, num_classes]
        return logits