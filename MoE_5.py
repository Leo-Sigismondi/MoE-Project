import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

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
        self.capacity    = capacity
        self.k           = k
        self.embed_dim   = embed_dim
        self.num_classes = num_classes

        # 1) Shared CNN trunk → [B, trunk_channels]
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

        # Gate network: learns to score experts based on features
        self.gate = nn.Sequential(
            nn.Linear(trunk_channels, trunk_channels // 2),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(trunk_channels // 2, num_experts)
        )

        # 2) Fused expert‐classifier heads: weight shape [E, C_out, C_in]
        self.cls_weight = nn.Parameter(
            torch.randn(num_experts, num_classes, trunk_channels) * 0.01
        )
        self.cls_bias = nn.Parameter(torch.zeros(num_experts, num_classes))

        # 3) Each expert’s embedding network (unused in final logits)
        self.experts = nn.ModuleList([
            ExpertModel(in_dim=trunk_channels, hidden_dim=expert_hidden, out_dim=embed_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x, temp: float = 1.0):
        B, device = x.size(0), x.device
        E, k, C = self.num_experts, self.k, self.capacity

        # 1) Shared trunk
        feats = self.trunk(x)  # [B, trunk_channels]

        # 2) Batched expert logits via einsum
        logits_e = torch.einsum("bd,ecd->bec", feats, self.cls_weight) \
                 + self.cls_bias.unsqueeze(0)

        if self.training:
            logits_e = (logits_e / temp).clamp(-10,10)
            logits_e = logits_e + torch.randn_like(logits_e)*1e-2

        # 3) Expert confidence = negative entropy of each expert’s softmax
        probs_e = F.softmax(logits_e, dim=2)              # [B, E, C]
        ent     = -(probs_e * probs_e.clamp(min=1e-12).log()).sum(dim=2)  # [B,E]
        # conf_norm = (-ent) / math.log(self.num_classes)          # [B,E] in [0,1]
        confidence = -ent 

        # 4) learned gate → [B,E]
        gate_logits = self.gate(feats)
        # if self.training:
        #     gate_logits = (gate_logits/temp).clamp(-10,10) \
        #                 + torch.randn_like(gate_logits)*1e-2
        # gate_probs = F.softmax(gate_logits, dim=1)

        # 5) combine on equal footing
        alpha = 1.0 
        # routing_scores = alpha * gate_probs + (1 - alpha) * confidence
        routing_scores = gate_logits * confidence

        # 6) Vectorized per-expert top-C routing
        C_eff = min(C, B)
        _, topk_idx = routing_scores.topk(C_eff, dim=0)        # [C_eff, E]
        D = torch.zeros(B, E, device=device, dtype=torch.bool)
        D[topk_idx, torch.arange(E, device=device)] = True

        # 7) Enforce at most k experts per sample:
        if k < E:
            _, sample_idx = routing_scores.masked_fill(~D, -1e9).topk(k, dim=1)
            D2 = torch.zeros_like(D)
            D2[torch.arange(B).unsqueeze(1), sample_idx] = True
            D = D2

        # 8) Gated ensemble: weighted sum of expert logits
        final_logits = torch.zeros(B, self.num_classes, device=device)
        weights = routing_scores.unsqueeze(2)  # [B, E, 1]
        weights = weights * D.unsqueeze(2).float()
        final_logits = (weights * logits_e).sum(dim=1)  # [B, C_out]
        norm = D.sum(dim=1, keepdim=True).clamp(min=1.0)
        final_logits = final_logits / norm

        return final_logits, routing_scores, D

    def diversity_penalty(self):
        """
        Orthogonality penalty on expert classifier weights:
          encourage W_i ⋅ W_j ≈ 0 for i ≠ j.
        """
        # reshape to [E, C_out*C_in]
        W = self.cls_weight.view(self.num_experts, -1)  # [E, D]
        # compute Gram matrix [E, E]
        G = W @ W.t()  # inner products
        # zero out diagonal
        diag = torch.eye(self.num_experts, device=G.device)
        offdiag = G * (1 - diag)
        # L2 norm of off-diagonal
        return offdiag.pow(2).sum()
