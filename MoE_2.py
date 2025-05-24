import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ExpertModel(nn.Module):
    """A simple example expert: two hidden‐layer MLP."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        # x: [N, in_dim]
        return self.net(x)  # [N, out_dim]

class MoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        capacity: int,
        k: int = 2,
        trunk: 'Optional[nn.Module]' = None,
        trunk_channels: int = 256,
        expert_hidden: int = 256,
        embed_dim: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity = capacity
        self.k = k
        # Shared convolutional trunk for CIFAR-10 → produces [B, trunk_channels, 1, 1]
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

        # Each expert “scorer” → single logit of confidence
        self.scorers = nn.ModuleList([
            nn.Linear(trunk_channels, 1) 
            for _ in range(num_experts)
        ])

        # Each expert’s heavy‐lifting network
        self.experts = nn.ModuleList([
            ExpertModel(in_dim=trunk_channels, hidden_dim=expert_hidden, out_dim=embed_dim)
            for _ in range(num_experts)
        ])

        # Final classifier: takes the concatenated expert outputs per example
        self.classifier = nn.Linear(num_experts * embed_dim, num_classes)

    def forward(self, x, temp: float = 1.0):
        B = x.size(0)
        device = x.device
        E = self.num_experts
        k = self.k
        C = self.capacity

        # 1) Shared trunk → semantic features
        feats = self.trunk(x)             # [B, C_feat]

        # 2) Per-expert confidence scores
        logits = torch.cat([s(feats) for s in self.scorers], dim=1)
        if self.training:
            # a) temperature scaling
            logits = logits / temp
            # b) optional clamping
            logits = logits.clamp(-10, +10)
            # c) tiny exploration noise
            logits = logits + torch.randn_like(logits) * 1e-2
        
        all_scores = F.softmax(logits, dim=1)  # [B, E]

        # 3) 4-cycle waterfall routing
        D = torch.zeros(B, E, device=device)         # dispatch mask
        expert_count = [0] * E
        sample_count = [0] * B

        current_scores = all_scores.clone()
        max_assign = B * k
        # split total desired assignments into 4 roughly equal chunks
        per_cycle = (max_assign + 3) // 4

        total_assigned = 0
        for cycle in range(4):
            if total_assigned >= max_assign:
                break

            # flatten & sort by descending score
            flat_scores, flat_idx = current_scores.view(-1).sort(descending=True)
            assigned_this = 0

            for score, idx in zip(flat_scores.tolist(), flat_idx.tolist()):
                b = idx // E
                j = idx % E
                # skip if already dispatched or caps reached
                if D[b, j] == 1 or expert_count[j] >= C or sample_count[b] >= k:
                    continue

                # assign
                D[b, j] = 1
                expert_count[j] += 1
                sample_count[b] += 1
                total_assigned += 1
                assigned_this += 1

                if assigned_this >= per_cycle or total_assigned >= max_assign:
                    break

            # penalize experts for fullness before next cycle
            penalty = torch.tensor([expert_count[j] / C for j in range(E)],
                                device=device)            # [E]
            current_scores = all_scores * (1.0 - penalty.unsqueeze(0))

        # 4) Expert inference & scatter back
        embed_dim = self.classifier.in_features // self.num_experts
        expert_outs = torch.zeros(B, E * embed_dim, device=device)

        for j, expert in enumerate(self.experts):
            idxs = D[:, j].nonzero(as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                continue
            out_j = expert(feats[idxs])                   # [Nj, embed_dim]
            start, end = j * embed_dim, (j + 1) * embed_dim
            expert_outs[idxs, start:end] = out_j.to(expert_outs.dtype)

        # 5) Final classification
        logits = self.classifier(expert_outs)            # [B, num_classes]
        return logits, all_scores, D
