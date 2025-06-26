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
        capacity: int = 32,  # Reduced from 64
        k: int = 2,
        trunk: Optional[nn.Module] = None,
        trunk_channels: int = 256,
        expert_hidden: int = 256,
        embed_dim: int = 128,
        num_classes: int = 10,
        max_iterations: int = 3,
        load_penalty_factor: float = 2.0,  # Increased from 0.5
        diversity_temp: float = 2.0,  # New parameter for routing diversity
        min_expert_usage: float = 0.05,  # Minimum fraction of tokens per expert
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity = capacity
        self.k = k
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        self.load_penalty_factor = load_penalty_factor
        self.diversity_temp = diversity_temp
        self.min_expert_usage = min_expert_usage

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

        # Gate network per ogni esperto: ogni esperto ha il suo gate personale
        self.expert_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_channels, trunk_channels // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(trunk_channels // 2, 1)  # Output scalare per ogni esperto
            ) for _ in range(num_experts)
        ])

        # 2) Fused expert‐classifier heads: weight shape [E, C_out, C_in]
        self.cls_weight = nn.Parameter(
            torch.randn(num_experts, num_classes, trunk_channels) * 0.01
        )
        self.cls_bias = nn.Parameter(torch.zeros(num_experts, num_classes))

        # 3) Each expert's embedding network (per analisi)
        self.experts = nn.ModuleList([
            ExpertModel(in_dim=trunk_channels, hidden_dim=expert_hidden, out_dim=embed_dim)
            for _ in range(num_experts)
        ])

        # 4) Running average of expert usage for load balancing
        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)
        self.usage_momentum = 0.9

    def balanced_iterative_routing(self, feats, confidence, temp=1.0):
        """
        Improved routing with better load balancing and fairness
        """
        B, device = feats.size(0), feats.device
        E = self.num_experts
        
        # 1) Expert Preference Scoring with diversity temperature
        expert_scores = torch.zeros(B, E, device=device)
        for j in range(E):
            raw_score = self.expert_gates[j](feats).squeeze(-1)  # [B]
            expert_scores[:, j] = raw_score / self.diversity_temp  # Apply temperature
        
        # Add controlled noise for exploration during training
        if self.training:
            noise = torch.randn_like(expert_scores) * 0.1
            expert_scores = expert_scores + noise
        
        # Combine with confidence
        alpha = 0.6  # Weight for expert scores vs confidence
        routing_scores = alpha * expert_scores + (1 - alpha) * confidence
        
        # 2) Strong load balancing penalty based on historical usage
        usage_penalty = self.load_penalty_factor * self.expert_usage_ema.unsqueeze(0)
        routing_scores = routing_scores - usage_penalty
        
        # Initialize structures
        D = torch.zeros(B, E, device=device, dtype=torch.bool)
        expert_loads = torch.zeros(E, device=device)
        available_tokens = torch.ones(B, device=device, dtype=torch.bool)
        
        # 3) Randomized expert claiming order to ensure fairness
        expert_order = torch.randperm(E, device=device) if self.training else torch.arange(E, device=device)
        
        for iteration in range(self.max_iterations):
            if not available_tokens.any():
                break
            
            # Dynamic load penalty that increases with iterations
            iteration_penalty = (iteration + 1) * self.load_penalty_factor * expert_loads.unsqueeze(0)
            penalized_scores = routing_scores - iteration_penalty
            
            # Mask unavailable tokens
            masked_scores = penalized_scores.clone()
            masked_scores[~available_tokens] = -float('inf')
            
            # Process experts in random order
            for j_idx in expert_order:
                j = j_idx.item()
                remaining_capacity = max(0, self.capacity - int(expert_loads[j].item()))
                if remaining_capacity == 0:
                    continue
                
                expert_preferences = masked_scores[:, j]
                valid_mask = available_tokens & (expert_preferences > -float('inf'))
                
                if not valid_mask.any():
                    continue
                
                # Ensure minimum expert usage
                min_tokens = max(1, int(B * self.min_expert_usage))
                num_to_claim = min(remaining_capacity, valid_mask.sum().item())
                num_to_claim = max(num_to_claim, min(min_tokens, valid_mask.sum().item()))
                
                if num_to_claim > 0:
                    _, top_indices = expert_preferences[valid_mask].topk(num_to_claim)
                    actual_indices = torch.nonzero(valid_mask, as_tuple=True)[0][top_indices]
                    
                    D[actual_indices, j] = True
                    expert_loads[j] += len(actual_indices)
                    available_tokens[actual_indices] = False
        
        # 4) Enhanced waterfall with load balancing consideration
        if available_tokens.any():
            remaining_indices = torch.nonzero(available_tokens, as_tuple=True)[0]
            for idx in remaining_indices:
                # Choose expert with lowest current load among top-k preferences
                top_k_experts = routing_scores[idx].topk(min(3, E)).indices
                loads_of_top_k = expert_loads[top_k_experts]
                best_expert = top_k_experts[loads_of_top_k.argmin()]
                D[idx, best_expert] = True
                expert_loads[best_expert] += 1
        
        # Update EMA of expert usage
        if self.training:
            current_usage = expert_loads / B
            self.expert_usage_ema = (self.usage_momentum * self.expert_usage_ema + 
                                   (1 - self.usage_momentum) * current_usage)
        
        return routing_scores, D

    def forward(self, x, temp: float = 1.0):
        B, device = x.size(0), x.device
        E = self.num_experts

        # 1) Shared trunk
        feats = self.trunk(x)  # [B, trunk_channels]

        # 2) Batched expert logits via einsum
        logits_e = torch.einsum("bd,ecd->bec", feats, self.cls_weight) \
                 + self.cls_bias.unsqueeze(0)

        if self.training:
            logits_e = (logits_e / temp).clamp(-10, 10)
            # Reduced noise for more stable training
            logits_e = logits_e + torch.randn_like(logits_e) * 5e-3

        # 3) Expert confidence = negative entropy
        probs_e = F.softmax(logits_e, dim=2)  # [B, E, C]
        ent = -(probs_e * probs_e.clamp(min=1e-12).log()).sum(dim=2)  # [B,E]
        confidence = -ent

        # 4) Balanced Iterative Routing
        routing_scores, D = self.balanced_iterative_routing(feats, confidence, temp)

        # 5) Gated ensemble with normalization
        final_logits = torch.zeros(B, self.num_classes, device=device)
        
        # Softmax normalization of routing scores for selected experts
        active_routing_scores = routing_scores * D.float()
        active_routing_scores = active_routing_scores - active_routing_scores.max(dim=1, keepdim=True)[0]
        routing_weights = F.softmax(active_routing_scores + (D.float() - 1) * 1e9, dim=1)
        
        weights = routing_weights.unsqueeze(2)  # [B, E, 1]
        final_logits = (weights * logits_e).sum(dim=1)  # [B, C_out]

        return final_logits, routing_scores, D

    def diversity_penalty(self):
        """
        Enhanced orthogonality penalty with expert usage regularization
        """
        # Orthogonality penalty
        W = self.cls_weight.view(self.num_experts, -1)  # [E, D]
        G = W @ W.t()  # inner products
        diag = torch.eye(self.num_experts, device=G.device)
        offdiag = G * (1 - diag)
        ortho_penalty = offdiag.pow(2).sum()
        
        # Usage balance penalty
        usage_std = self.expert_usage_ema.std()
        usage_penalty = usage_std * 10.0  # Strong penalty for unbalanced usage
        
        return ortho_penalty + usage_penalty

    def get_routing_stats(self, D):
        """
        Enhanced routing statistics
        """
        expert_loads = D.sum(dim=0).float()  # [E]
        tokens_per_expert = expert_loads / D.size(0)  # Fraction of tokens per expert
        load_balance = tokens_per_expert.std()  # How balanced the load is
        coverage = (expert_loads > 0).sum().item()  # How many experts are used
        
        # Additional stats
        max_load = expert_loads.max().item()
        min_load = expert_loads.min().item()
        load_ratio = max_load / (min_load + 1e-6)  # Ratio between most and least used
        
        return {
            'expert_loads': expert_loads.cpu().numpy(),
            'tokens_per_expert': tokens_per_expert.cpu().numpy(),
            'load_balance_std': load_balance.item(),
            'expert_coverage': coverage,
            'total_assignments': D.sum().item(),
            'usage_ema': self.expert_usage_ema.cpu().numpy(),
            'load_ratio': load_ratio,
            'max_load': max_load,
            'min_load': min_load
        }