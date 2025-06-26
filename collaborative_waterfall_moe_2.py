import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Optional: Expert orthogonality loss ---
def orthogonality_loss(z_experts: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    num = 0
    for i in range(len(z_experts)):
        for j in range(i + 1, len(z_experts)):
            if z_experts[i].size(0) == 0 or z_experts[j].size(0) == 0:
                continue
            zi = F.normalize(z_experts[i], dim=1)
            zj = F.normalize(z_experts[j], dim=1)
            sim = torch.mm(zi, zj.T).pow(2).mean()
            loss += sim
            num += 1
    return loss / max(num, 1)

# --- SlimExpert with scorer and class head sharing the same sub-features ---
class SlimExpert(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Tuple[int, int, int],
        embed_dim: int,
        num_classes: int,
        scorer_hidden: int = 128,
    ):
        super().__init__()
        _, c2, c3 = hidden_channels

        # Classifier head: operates on expert's own conv block
        self.encoder = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
        )
        self.project = nn.Sequential(
            nn.Linear(c3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Scorer/class specialization head: takes shared features, projects, splits
        self.scorer_project = nn.Linear(c2, scorer_hidden)
        self.scorer_head = nn.Linear(scorer_hidden, 1)
        self.class_head = nn.Linear(scorer_hidden, num_classes)

    def score_tokens(self, features):  # features: (B, c2)
        h = self.scorer_project(features)
        return self.scorer_head(h).squeeze(-1)  # (B,)

    def predict_class(self, features):
        h = self.scorer_project(features)
        return self.class_head(h)  # (B, num_classes)

    def forward(self, x):  # x: (B, c2, H, W)
        feats = self.encoder(x)
        z = self.project(feats)
        logits = self.classifier(z)
        return logits

# --- CollaborativeWaterfallMoE ---
class CollaborativeWaterfallMoE(nn.Module):
    def __init__(
        self,
        num_experts: int = 4,
        capacity: int | str = "auto",
        in_channels: int = 3,
        hidden_channels: Tuple[int, int, int] = (64, 128, 256),
        embed_dim: int = 256,
        num_classes: int = 10,
        balance_loss_weight: float = 0.0,
        scorer_aux_loss_weight: float = 0.1,
        orthogonality_weight: float = 0.1,
        class_entropy_weight: float = 0.5,
        diversity_weight: float = 0.1,
        scorer_hidden: int = 128,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._static_capacity = capacity if capacity != "auto" else None

        # Shared trunk
        c1, c2, _ = hidden_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
        )

        # Experts (heads)
        self.experts = nn.ModuleList([
            SlimExpert(
                in_channels,
                hidden_channels,
                embed_dim,
                num_classes,
                scorer_hidden=scorer_hidden,
            ) for _ in range(num_experts)
        ])

        # Loss weights
        self.balance_loss_weight = balance_loss_weight
        self.scorer_aux_loss_weight = scorer_aux_loss_weight
        self.orthogonality_weight = orthogonality_weight
        self.class_entropy_weight = class_entropy_weight
        self.diversity_weight = diversity_weight

    def forward(self, x, T=0.1, return_aux=False, targets=None):
        B = x.size(0)
        device = x.device
        C = (self._static_capacity if self._static_capacity is not None else math.ceil(B / self.num_experts))
        features = self.encoder(x)  # (B, c2, H, W)
        flat_features = features.mean(dim=(-2, -1))  # (B, c2), for scorer/class

        # Per-expert scores and class logits
        scores = torch.stack([expert.score_tokens(flat_features) for expert in self.experts], dim=1)  # (B, E)
        class_logits = torch.stack([expert.predict_class(flat_features) for expert in self.experts], dim=1)  # (B, E, C)
        class_probs = torch.softmax(class_logits, dim=2)  # (B, E, C)

        # Routing: bonus for ground-truth class probability
        if targets is not None and self.training:
            targets_expand = targets.view(B, 1).expand(B, self.num_experts).to(device)
            gt_probs = class_probs.gather(2, targets_expand.unsqueeze(-1)).squeeze(-1)
        else:
            gt_probs = torch.zeros(B, self.num_experts, device=device)

        beta = 1.0  # Weight for ground-truth bonus
        scores -= scores.mean(dim=0, keepdim=True)  # Center scores
        combined_score = scores + beta * gt_probs.clamp_min(1e-9).log()

        # Routing probabilities (for aux loss and logging)
        probs = torch.softmax(combined_score, dim=1)
        if self.training: 
            gumbel = -torch.log(-torch.log(torch.rand_like(scores)))   # Gumbel(0,1)
            scores_noisy = (combined_score + gumbel) / T                       # T decays → 0
        else:
            scores_noisy = combined_score/T
        # scores_noisy = combined_score / T  # Optionally add Gumbel noise for exploration
        scores_noisy = scores_noisy.clamp(min=-1e9, max=1e9)  # Avoid NaNs in softmax

        # --- Waterfall Routing ---
        assignment = torch.zeros(B, self.num_experts, dtype=torch.bool, device=device)
        cap = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        remaining = torch.arange(B, device=device)
        iter_ = 0
        if self.training:
            alpha = 5.0  # Capacity penalty
        else:
            alpha = 0.0 
            
        while remaining.numel() > 0 and iter_ < 15:
            scores_this = scores_noisy[remaining].clone()
            deficit = (cap.float() / C).clamp(0, 1)
            scores_this = scores_this - alpha * deficit

            full_experts = (cap >= C)
            if full_experts.any():
                scores_this[:, full_experts] = float('-inf')
            best_exp = scores_this.argmax(dim=1)
            taken_mask_global = torch.zeros_like(remaining, dtype=torch.bool)
            quota = 2 ** iter_
            for e in range(self.num_experts):
                want = (best_exp == e).nonzero(as_tuple=True)[0]
                if want.numel() == 0: continue
                space = min(C - cap[e].item(), quota)
                if space <= 0: continue
                select = want[:space]
                token_ids = remaining[select]
                assignment[token_ids, e] = True
                cap[e] += select.numel()
                taken_mask_global[select] = True
            remaining = remaining[~taken_mask_global]
            iter_ += 1

        # Fallback: assign remaining to least-loaded expert
        if remaining.numel() > 0:
            least_loaded = (cap.float() / C).argmin().item()
            assignment[remaining, least_loaded] = True
            cap[least_loaded] += remaining.numel()

        # --- Compute per-expert predictions ---
        out_logits = torch.zeros(B, self.num_classes, device=device)
        z_expert_outputs = []
        for e, expert in enumerate(self.experts):
            idx = torch.where(assignment[:, e])[0]
            if idx.numel() == 0: continue
            feats = expert.encoder(features[idx])
            z = expert.project(feats)
            logits = expert.classifier(z)
            out_logits[idx] = logits.to(out_logits.dtype)
            z_expert_outputs.append(z)
        # --- Auxiliary losses ---
        aux_loss = None
        aux_losses = {}

        # Scorer auxiliary loss (KL)
        if self.training and self.scorer_aux_loss_weight > 0:
            scorer_targets = probs
            scorer_predictions = torch.softmax(scores, dim=1)
            scorer_loss = F.kl_div(
                scorer_predictions.log(), scorer_targets, reduction='batchmean'
            )
            aux_losses['scorer'] = scorer_loss * self.scorer_aux_loss_weight

        # Orthogonality loss (optional, only if needed)
        if self.training and self.orthogonality_weight > 0:
            orth_loss = orthogonality_loss(z_expert_outputs) * self.orthogonality_weight
            aux_losses['orthogonality'] = orth_loss

        # Class entropy loss (for specialization)
        if self.training and self.class_entropy_weight > 0:
            class_entropy = -(class_probs * (class_probs + 1e-9).log()).sum(dim=2).mean()
            aux_losses['class_entropy'] = class_entropy * self.class_entropy_weight

        # Diversity loss (optional, for encouraging experts to specialize on different classes)
        if self.training and self.diversity_weight > 0:
            mean_class_probs = class_probs.mean(dim=0)
            diversity = 0.0
            num_pairs = 0
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    pi = mean_class_probs[i].clamp_min(1e-8)
                    pj = mean_class_probs[j].clamp_min(1e-8)
                    kl = F.kl_div(pi.log(), pj, reduction='batchmean')
                    diversity += kl
                    num_pairs += 1
            aux_losses['diversity'] = -diversity / max(num_pairs, 1) * self.diversity_weight

        aux_loss = sum(aux_losses.values()) if aux_losses else None

        if return_aux:
            return out_logits, probs, assignment, aux_loss, iter_, scores, aux_losses
        return out_logits

# --- Usage Example / Test ---
if __name__ == "__main__":
    torch.manual_seed(42)
    model = CollaborativeWaterfallMoE(
        num_experts=4,
        num_classes=10,
        scorer_aux_loss_weight=0.05,
        class_entropy_weight=0.01,
        diversity_weight=0.01,
    ).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    B = 64
    dummy_x = torch.randn(B, 3, 32, 32).cuda()
    dummy_y = torch.randint(0, 10, (B,), dtype=torch.long).cuda()

    model.train()
    for step in range(5):
        logits, probs, assignment, aux_loss, iter_, scores, aux_losses = model(
            dummy_x, return_aux=True, targets=dummy_y
        )
        loss = criterion(logits, dummy_y)
        if aux_loss is not None:
            loss = loss + aux_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"step {step}: loss={loss.item():.3f} | aux={aux_loss.item() if aux_loss else 0:.3f} | iters={iter_}")
        print(f"  expert usage: {assignment.sum(dim=0).tolist()}")

    # Eval test
    model.eval()
    with torch.no_grad():
        eval_logits = model(dummy_x, return_aux=True, targets=dummy_y)
        print(f"Eval mode output: {eval_logits[0].shape}")

