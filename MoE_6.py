import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class ExpertModel(nn.Module):
    # ... (rest of your ExpertModel class is unchanged)
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
        max_iterations: int = 3,
        iteration_capacity_fraction: float = 0.5,
        load_penalty_factor: float = 0.1, # NEW: Factor for penalizing full experts
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity    = capacity
        self.k           = k
        self.embed_dim   = embed_dim
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        self.iteration_capacity_fraction = iteration_capacity_fraction
        self.load_penalty_factor = load_penalty_factor # Store the penalty factor

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
        E, C = self.num_experts, self.capacity

        # 1) Shared trunk
        feats = self.trunk(x)  # [B, trunk_channels]

        # Compute initial gate probabilities for the whole batch (needed for forced assignment)
        initial_gate_logits = self.gate(feats)
        initial_gate_probs = F.softmax(initial_gate_logits, dim=1)

        D_aggregated = torch.zeros(B, E, device=device, dtype=torch.bool)
        routing_scores_aggregated = torch.zeros(B, E, device=device) # To store the actual score that led to the assignment

        available_tokens_mask = torch.ones(B, dtype=torch.bool, device=device)
        
        # Track initial (batch) load for each expert based on tokens *already assigned*
        # This starts at 0 for all experts for a new batch
        expert_current_load = torch.zeros(E, dtype=torch.float, device=device)
        
        # Calculate expert_total_capacity only once per forward pass
        expert_total_capacity_for_batch = torch.full((E,), C, dtype=torch.float, device=device)


        for iteration in range(self.max_iterations):
            current_B = available_tokens_mask.sum()
            if current_B == 0:
                break

            current_feats = feats[available_tokens_mask]

            # Calculate expert logits and confidence for available tokens
            current_logits_e = torch.einsum("bd,ecd->bec", current_feats, self.cls_weight) \
                             + self.cls_bias.unsqueeze(0)

            if self.training:
                current_logits_e = (current_logits_e / temp).clamp(-10,10)
                current_logits_e = current_logits_e + torch.randn_like(current_logits_e)*1e-2

            current_probs_e = F.softmax(current_logits_e, dim=2)
            current_ent = -(current_probs_e * current_probs_e.clamp(min=1e-12).log()).sum(dim=2)
            current_conf_norm = (-current_ent) / math.log(self.num_classes)

            current_gate_logits = self.gate(current_feats)
            if self.training:
                current_gate_logits = (current_gate_logits/temp).clamp(-10,10) \
                                    + torch.randn_like(current_gate_logits)*1e-2
            current_gate_probs = F.softmax(current_gate_logits, dim=1)

            current_routing_scores = current_gate_probs * current_conf_norm # [current_B, E]

            # --- NEW: Apply penalty based on current expert load ---
            # Calculate how "full" each expert is.
            # Using the `expert_current_load` (tokens assigned so far)
            # Normalize load by capacity: 0 means empty, 1 means full
            expert_fullness = expert_current_load / expert_total_capacity_for_batch.clamp(min=1e-8)
            expert_fullness = torch.clamp(expert_fullness, 0, 1) # Ensure values are between 0 and 1

            # Create a penalty factor for each expert
            # A higher fullness leads to a lower penalty factor (more penalty, i.e., score reduction)
            # Example: (1 - fullness * load_penalty_factor)
            # If load_penalty_factor is 0.1:
            #   - Expert 0% full: factor = 1.0 (no penalty)
            #   - Expert 50% full: factor = 1 - 0.5 * 0.1 = 0.95 (5% penalty)
            #   - Expert 100% full: factor = 1 - 1.0 * 0.1 = 0.90 (10% penalty)
            penalty_factors = 1.0 - (expert_fullness * self.load_penalty_factor)
            penalty_factors = torch.clamp(penalty_factors, min=0.0) # Scores can't go below 0 due to penalty

            # Apply the penalty to the routing scores (broadcasting across tokens)
            current_routing_scores = current_routing_scores * penalty_factors.unsqueeze(0) # [current_B, E] * [1, E]


            # --- Waterfall mechanism for claiming tokens ---
            expert_token_scores = current_routing_scores.transpose(0, 1) # [E, current_B]
            sorted_scores_per_expert, token_indices_per_expert = expert_token_scores.sort(descending=True, dim=1)

            # Calculate effective capacity for claiming in this iteration based on original capacity
            # It's better to calculate remaining capacity based on the initial capacity minus `expert_current_load`
            expert_remaining_capacity_this_iter = expert_total_capacity_for_batch - expert_current_load
            experts_current_claim_capacity = (expert_remaining_capacity_this_iter * self.iteration_capacity_fraction).floor().int()
            experts_current_claim_capacity = torch.clamp(experts_current_claim_capacity, min=0, max=int(current_B.item()))

            claimed_mask_this_iter = torch.zeros(int(current_B.item()), E, dtype=torch.bool, device=device)
            
            for j in range(E):
                num_to_claim = experts_current_claim_capacity[j].item()
                if num_to_claim > 0:
                    expert_preferred_token_local_indices = token_indices_per_expert[j, :num_to_claim]
                    claimed_mask_this_iter[expert_preferred_token_local_indices, j] = True
            
            # Conflict resolution: assign token to the highest-scoring expert *among those who claimed it this iter*
            conflict_resolution_scores = current_routing_scores * claimed_mask_this_iter.float()
            max_scores_per_token, best_expert_local_idx = conflict_resolution_scores.max(dim=1)
            
            D_this_iteration = torch.zeros_like(claimed_mask_this_iter)
            assigned_locally_mask = (max_scores_per_token > 0) # Tokens that were actually assigned this iter

            if assigned_locally_mask.sum() > 0:
                assigned_indices = assigned_locally_mask.nonzero(as_tuple=True)[0]
                D_this_iteration[assigned_indices, best_expert_local_idx[assigned_indices]] = True

                # Update global D_aggregated
                global_indices_available = available_tokens_mask.nonzero(as_tuple=True)[0]
                global_assigned_indices = global_indices_available[assigned_locally_mask]
                global_assigned_experts = best_expert_local_idx[assigned_locally_mask]
                
                # Make sure we only set to True, not unset if already True from a prior iteration
                D_aggregated[global_assigned_indices, global_assigned_experts] = True
                
                # Update expert_current_load (how many tokens each expert has received so far)
                tokens_per_expert_this_iter = D_this_iteration.sum(dim=0).float()
                expert_current_load += tokens_per_expert_this_iter

                # Update available_tokens_mask: remove tokens that were truly assigned in this iteration
                available_tokens_mask[global_assigned_indices] = False
                
                # Populate routing_scores_aggregated for the assigned tokens
                scores_for_assigned_tokens = conflict_resolution_scores[assigned_indices, best_expert_local_idx[assigned_indices]]
                routing_scores_aggregated[global_assigned_indices, global_assigned_experts] = scores_for_assigned_tokens


        # Ensure full token coverage (waterfall part 2): Assign remaining unassigned tokens
        unassigned_tokens_mask = available_tokens_mask
        if unassigned_tokens_mask.sum() > 0:
            unassigned_feats = feats[unassigned_tokens_mask]
            unassigned_gate_logits = self.gate(unassigned_feats)
            unassigned_gate_probs = F.softmax(unassigned_gate_logits, dim=1)
            
            # Assign each unassigned token to its top-1 expert based on these gate probs
            _, top1_expert_idx_unassigned = unassigned_gate_probs.max(dim=1)
            
            global_unassigned_indices = unassigned_tokens_mask.nonzero(as_tuple=True)[0]
            
            D_aggregated[global_unassigned_indices, top1_expert_idx_unassigned] = True
            
            # Use initial_gate_probs for the original gate scores
            original_gate_scores_for_unassigned = initial_gate_probs[global_unassigned_indices, top1_expert_idx_unassigned]
            routing_scores_aggregated[global_unassigned_indices, top1_expert_idx_unassigned] = original_gate_scores_for_unassigned
            
            # print(f"Warning: {unassigned_tokens_mask.sum().item()} tokens unassigned after {self.max_iterations} iterations. Forcing assignment.")


        # Re-calculate `logits_e` for the *entire* batch for the ensemble
        logits_e_full_batch = torch.einsum("bd,ecd->bec", feats, self.cls_weight) \
                            + self.cls_bias.unsqueeze(0)

        # Gated ensemble: weighted sum of expert logits based on final D_aggregated
        final_logits = torch.zeros(B, self.num_classes, device=device)
        
        weights_for_ensemble = D_aggregated.float()
        
        norm = D_aggregated.sum(dim=1, keepdim=True).clamp(min=1.0)
        weights_for_ensemble = weights_for_ensemble / norm

        final_logits = (weights_for_ensemble.unsqueeze(2) * logits_e_full_batch).sum(dim=1)

        # Return `routing_scores_aggregated` as the `scores` for metrics, and `D_aggregated` as the mask.
        # `routing_scores_aggregated` now reflects the scores that led to the final hard assignments.
        # This is more accurate than just returning `original_gate_probs` if `original_gate_probs` isn't what drove the decision.
        return final_logits, routing_scores_aggregated, D_aggregated

    def diversity_penalty(self):
        # ... (unchanged)
        W = self.cls_weight.view(self.num_experts, -1)  # [E, D]
        G = W @ W.t()
        diag = torch.eye(self.num_experts, device=G.device)
        offdiag = G * (1 - diag)
        return offdiag.pow(2).sum()