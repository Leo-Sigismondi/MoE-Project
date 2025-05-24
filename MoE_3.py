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
        trunk: Optional[nn.Module] = None,
        trunk_channels: int = 256,
        expert_hidden: int = 256,
        embed_dim: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity = capacity
        self.k = k

        # 1) Shared trunk → [B, trunk_channels]
        if trunk is None:
            self.trunk = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32,64,3,padding=1),   nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),  # 32→16
                nn.Conv2d(64,128,3,padding=1),  nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.MaxPool2d(2),  # 16→8
                nn.AdaptiveAvgPool2d(1),        # → [B,256,1,1]
                nn.Flatten(),                    # → [B,256]
            )
        else:
            self.trunk = trunk

        # 2) Each expert’s own classification head on trunk features
        self.expert_cls = nn.ModuleList([
            nn.Linear(trunk_channels, num_classes)
            for _ in range(num_experts)
        ])

        # 3) Each expert’s heavy‐lifting network (to produce embeddings)
        self.experts = nn.ModuleList([
            ExpertModel(in_dim=trunk_channels, hidden_dim=expert_hidden, out_dim=embed_dim)
            for _ in range(num_experts)
        ])

        # 4) Final MoE classifier on concatenated embeddings
        self.classifier = nn.Linear(num_experts * embed_dim, num_classes)

    def forward(self, x, temp: float = 1.0):
        B, device, E, k, C = x.size(0), x.device, self.num_experts, self.k, self.capacity

        # 1) Shared trunk
        feats = self.trunk(x)  # [B, trunk_channels]

        # 2) Per-expert classification logits & confidence
        #    -> all_cls_logits: [B, E, num_classes]
        all_cls_logits = torch.stack([cls(feats) for cls in self.expert_cls], dim=1)
        if self.training:
            # optional temp scaling & noise on logits
            logits = all_cls_logits / temp
            logits = logits.clamp(-10, +10) + torch.randn_like(logits) * 1e-2
            all_cls_logits = logits

        # probabilities and negative-entropy confidence: conf_scores[b,j]
        probs = F.softmax(all_cls_logits, dim=2)      # [B, E, num_classes]
        ent = -(probs * probs.clamp(min=1e-12).log()).sum(dim=2)  # [B, E]
        conf_scores = -ent                            # higher = more confident

        # flatten to [B, E]
        current = conf_scores

        # 3) 4-cycle waterfall routing on conf_scores
        D = torch.zeros(B, E, device=device, dtype=torch.bool)
        expert_count = [0]*E
        sample_count = [0]*B
        total_needed = B * k
        per_cycle = (total_needed + 3) // 4
        assigned = 0

        for _ in range(4):
            if assigned >= total_needed:
                break
            flat_vals, flat_idx = current.view(-1).sort(descending=True)
            cycle_assigned = 0
            for val, idx in zip(flat_vals.tolist(), flat_idx.tolist()):
                b, j = divmod(idx, E)
                if D[b,j] or expert_count[j]>=C or sample_count[b]>=k:
                    continue
                D[b,j] = True
                expert_count[j] += 1
                sample_count[b] += 1
                assigned += 1
                cycle_assigned += 1
                if cycle_assigned >= per_cycle or assigned>=total_needed:
                    break
            # penalize fullness
            penalty = torch.tensor([expert_count[j]/C for j in range(E)], device=device)
            current = conf_scores * (1.0 - penalty.unsqueeze(0))

        # 4) gated ensemble of expert logits → final logits
        num_classes = all_cls_logits.size(2)
        final_logits = torch.zeros(B, num_classes, device=device)
        # weight = conf[b,j] before normalization
        weight = conf_scores

        for j in range(E):
            idxs = D[:,j].nonzero(as_tuple=True)[0]
            if idxs.numel()==0: continue
            # fetch each expert’s own logits on the samples it won
            logj = all_cls_logits[idxs, j, :]           # [Nj, C]
            wj   = weight[idxs, j].unsqueeze(1)         # [Nj, 1]
            final_logits[idxs] += wj * logj             # gated sum

        # optional: normalize by number of experts assigned (or sum of weights)
        norm = D.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        final_logits = final_logits / norm

        return final_logits, conf_scores, D
