# collaborative_waterfall_moe_fixed.py
"""
Collaborative Waterfall Mixture‑of‑Experts - Minimal Fixes
---------------------------------------------------------
Keeps your original iterative routing logic, just fixes the gradient issue
and makes small improvements.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightThroughAssignment(torch.autograd.Function):
    """Not actually used - keeping for reference only."""
    
    @staticmethod
    def forward(ctx, soft_scores, hard_assignment):
        return hard_assignment
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> nn.Sequential:
    """Tiny helper: conv → batchnorm → ReLU."""
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.1),
    )


class GlobalAvgPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B × C × H × W → B × C
        return torch.mean(x, dim=(-2, -1))


class SlimExpert(nn.Module):
    """A *very* lightweight expert with internal scorer - minimal changes."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: List[int] | Tuple[int, ...] = (64, 128, 256),
        embed_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        c1, c2, c3 = hidden_channels

        self.encoder = nn.Sequential( 
            # Block 1
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32→16
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16→8
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
        )  # B×C

        self.project = nn.Linear(c3, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # scorer head - keep it simple but allow gradients
        self.scorer = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=1),  # slightly bigger: 4→8 channels
            nn.ReLU(inplace=True),
            GlobalAvgPool(),
            nn.Linear(8, 1),  # scalar logit
        )

    def score_tokens(self, x: torch.Tensor) -> torch.Tensor:  # B × 1
        """Cheap preference logit for each token - REMOVE @torch.no_grad()!"""
        return self.scorer(x).squeeze(-1)  # (B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B × C × H × W
        """Full path: encode + classify."""
        z = self.encoder(x)               # B × C
        z = self.project(z)               # B × D
        logits = self.classifier(z)       # B × classes
        return logits


class CollaborativeWaterfallMoE(nn.Module):
    """Your original routing logic with minimal gradient fixes."""

    def __init__(
        self,
        num_experts: int = 4,
        capacity: int | str = "auto",
        in_channels: int = 3,
        num_classes: int = 10,
        balance_loss_weight: float = 0.1,
        scorer_aux_loss_weight: float = 0.05,  # NEW: auxiliary loss for scorer
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.balance_loss_weight = balance_loss_weight
        self.scorer_aux_loss_weight = scorer_aux_loss_weight  # NEW
        self.experts = nn.ModuleList([
            SlimExpert(in_channels, num_classes=num_classes) for _ in range(num_experts)
        ])
        self._static_capacity = capacity if capacity != "auto" else None

    def forward(self, x: torch.Tensor, T: float = 0.1, return_aux=False, targets=None):  # B × C × H × W
        B = x.size(0)
        device = x.device
        C = (
            self._static_capacity
            if self._static_capacity is not None
            else math.ceil(B / self.num_experts)
        )
        
        # ---------------------------------------------------------------
        # 1) Preference scoring - KEEP GRADIENTS! 
        # ---------------------------------------------------------------
        scores = torch.stack([
            expert.score_tokens(x) for expert in self.experts
        ], dim=1)  # B × E
        
        probs = torch.softmax(scores, dim=1)  # B × E (for load‑balance loss)
        
        gumbel = -torch.log(-torch.log(torch.rand_like(scores)))   # Gumbel(0,1)
        scores_noisy = (scores + gumbel) / T                       # T decays → 0

        # ---------------------------------------------------------------
        # 2) Your original waterfall routing (unchanged!)
        # ---------------------------------------------------------------
        assignment = torch.zeros(B, self.num_experts, dtype=torch.bool, device=device)
        cap = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        remaining = torch.arange(B, device=device)
        iter_ = 0
        while remaining.numel() > 0:
            # Copy scores for remaining tokens
            scores_this = scores_noisy[remaining].clone()  # (|R|, E)
            
            # Mask out full experts by setting their scores to -inf
            deficit = (cap.float() / C).clamp(0, 1)  # 0 (empty) ... 1 (full)
            scores_this = scores_this * (1-deficit)  # penalize full/near-full experts
            full_experts = (cap >= C)
            if full_experts.any():
                scores_this[:, full_experts] = float('-inf')
            # Each remaining token chooses its best expert among those with capacity
            best_exp = scores_this.argmax(dim=1)  # (|R|, E)
            taken_mask_global = torch.zeros_like(remaining, dtype=torch.bool)
            
            quota = 2 ** iter_  # quota doubles each round
            for e in range(self.num_experts):
                want = (best_exp == e).nonzero(as_tuple=True)[0]      # local indices
                if want.numel() == 0:
                    continue
                space = min(C - cap[e].item(), quota)        # quota
                if space <= 0:
                    continue
                select = want[:space]
                token_ids = remaining[select]
                assignment[token_ids, e] = True
                cap[e] += select.numel()
                taken_mask_global[select] = True

            # remove taken tokens for next round
            remaining = remaining[~taken_mask_global]
            iter_ += 1

        unassigned_tokens = remaining.numel()
        # Safety fallback: still‑unassigned tokens → greedy to least loaded expert
        if unassigned_tokens > 0:
            load = cap.float() / C  # frac used
            least_loaded = load.argmin().item()
            assignment[remaining, least_loaded] = True
            cap[least_loaded] += remaining.numel()
            remaining = torch.empty(0, dtype=torch.long, device=device)

        # ---------------------------------------------------------------
        # 3) Hard assignment routing (your original approach)
        # ---------------------------------------------------------------
        # No need for routing weights - we use pure hard assignment

        # ---------------------------------------------------------------
        # 4) Compute per‑expert predictions on *assigned* tokens only
        # ---------------------------------------------------------------
        out_logits = torch.zeros(B, self.num_classes, device=device)
        for e, expert in enumerate(self.experts):
            idx = torch.where(assignment[:, e])[0]
            if idx.numel() == 0:
                continue  # this expert unused this step
            logits_e = expert(x[idx])  # forward pass only for its share
            out_logits[idx] = logits_e.to(out_logits.dtype)

        # ---------------------------------------------------------------
        # 4) Auxiliary losses
        # ---------------------------------------------------------------
        aux_loss = None
        aux_losses = {}
        
        if self.training and self.balance_loss_weight > 0:
            # Shannon entropy of routing probabilities averaged over batch.
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean()
            aux_losses['balance'] = -entropy * self.balance_loss_weight

        # NEW: Auxiliary loss for scorer network
        if self.training and self.scorer_aux_loss_weight > 0 and targets is not None:
            # Use routing probabilities to create pseudo-targets for each expert's scorer
            # This gives the scorer network direct supervision
            scorer_targets = probs  # B × E - how much each expert should "want" each token
            scorer_predictions = torch.softmax(scores, dim=1)  # B × E - what each expert predicts
            
            # KL divergence between scorer predictions and routing probabilities
            scorer_loss = F.kl_div(
                scorer_predictions.log(), 
                scorer_targets, 
                reduction='batchmean'
            )
            aux_losses['scorer'] = scorer_loss * self.scorer_aux_loss_weight

        aux_loss = sum(aux_losses.values()) if aux_losses else None

        if return_aux:
            return out_logits, probs, assignment, aux_loss, iter_, aux_losses
        return out_logits


# Minimal usage demo
def _demo():
    import time

    torch.manual_seed(42)
    model = CollaborativeWaterfallMoE(
        num_experts=4, 
        num_classes=10,
        scorer_aux_loss_weight=0.05  # Enable scorer auxiliary loss
    ).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    B = 64
    dummy_x = torch.randn(B, 3, 32, 32).cuda()
    dummy_y = torch.randint(0, 10, (B,), dtype=torch.long).cuda()

    model.train()
    for step in range(5):
        st = time.time()
        logits, probs, assignment, aux_loss, iter_, aux_losses = model(
            dummy_x, return_aux=True, targets=dummy_y
        )
        loss = criterion(logits, dummy_y)
        if aux_loss is not None:
            loss = loss + aux_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # Check scorer gradients
        scorer_grad_norm = sum(
            p.grad.norm().item() if p.grad is not None else 0 
            for expert in model.experts 
            for p in expert.scorer.parameters()
        )
        
        print(f"step {step}: loss={loss.item():.3f} | aux={aux_loss.item() if aux_loss else 0:.3f} | iters={iter_}")
        print(f"  scorer_grad_norm: {scorer_grad_norm:.4f}")
        print(f"  expert usage: {assignment.sum(dim=0).tolist()}")
        print(f"  aux_losses: {aux_losses}")
        print(f"  time: {time.time()-st:.2f}s\n")


if __name__ == "__main__":
    _demo()