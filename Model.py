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
    

class SingleImprovedModel(nn.Module):
    def __init__(
        self,
        trunk_channels: int = 256,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        num_classes: int = 10,
        trunk: Optional[nn.Module] = None,
        expert_dropout: float = 0.1,
    ):
        super().__init__()
        # Use the same improved trunk as in ImprovedMoE
        if trunk is None:
            self.trunk = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 64, 3, padding=1), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 32→16
                nn.Dropout2d(0.1),
                # Block 2
                nn.Conv2d(64, 128, 3, padding=1), 
                nn.BatchNorm2d(128), 
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), 
                nn.BatchNorm2d(128), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 16→8
                nn.Dropout2d(0.1),
                # Block 3
                nn.Conv2d(128, 256, 3, padding=1), 
                nn.BatchNorm2d(256), 
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1), 
                nn.BatchNorm2d(256), 
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
            )
            trunk_channels = 256
        else:
            self.trunk = trunk

        # Use the same expert MLP as in ImprovedMoE
        self.expert = nn.Sequential(
            nn.Linear(trunk_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(expert_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(expert_dropout / 2),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

        # Classifier head (same as MoE, but only one expert)
        self.project = nn.Linear(trunk_channels, embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.trunk(x)           # [B, trunk_channels]
        # expert_out = self.expert(feats) # [B, embed_dim]
        expert_out = self.project(feats)
        logits = self.classifier(expert_out)  # [B, num_classes]
        return logits