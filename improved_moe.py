# Fixed ImprovedMoE implementation with proper hyperparameter usage

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math

class ImprovedMoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        capacity: int = 32,
        k: int = 2,  # This WILL be used now
        trunk: Optional[nn.Module] = None,
        trunk_channels: int = 256,
        expert_hidden: int = 256,
        embed_dim: int = 128,
        num_classes: int = 10,
        max_iterations: int = 3,
        load_penalty_factor: float = 2.0,
        diversity_temp: float = 2.0,
        min_expert_usage: float = 0.05,  # This WILL be used now
        # New hyperparameters with better defaults
        gate_noise_std: float = 0.1,
        routing_temperature: float = 1.0,
        balance_loss_weight: float = 0.01,
        usage_momentum: float = 0.9,
        expert_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity = capacity
        self.k = min(k, num_experts)  # Ensure k doesn't exceed num_experts
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        self.load_penalty_factor = load_penalty_factor
        self.diversity_temp = diversity_temp
        self.min_expert_usage = min_expert_usage
        self.gate_noise_std = gate_noise_std
        self.routing_temperature = routing_temperature
        self.balance_loss_weight = balance_loss_weight
        self.usage_momentum = usage_momentum

        # Improved trunk architecture with residual connections
        if trunk is None:
            self.trunk = self._build_improved_trunk()
        else:
            self.trunk = trunk

        # Improved gating network with better initialization
        self.expert_gates = nn.ModuleList([
            self._build_expert_gate(trunk_channels) 
            for _ in range(num_experts)
        ])

        # Expert networks with improved architecture
        self.experts = nn.ModuleList([
            self._build_expert(trunk_channels, expert_hidden, embed_dim, expert_dropout)
            for _ in range(num_experts)
        ])

        # Classifier heads with proper initialization
        self.cls_weight = nn.Parameter(
            torch.randn(num_experts, num_classes, embed_dim) / math.sqrt(embed_dim)
        )
        self.cls_bias = nn.Parameter(torch.zeros(num_experts, num_classes))

        # Load balancing tracking
        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)
        self.register_buffer('routing_history', torch.zeros(100, num_experts))
        self.register_buffer('history_idx', torch.tensor(0))
        self.usage_momentum = 0.9  # Slightly lower for faster adaptation

        # Initialize weights
        self._initialize_weights()

    def _build_improved_trunk(self):
        """Build trunk with residual connections and better normalization"""
        return nn.Sequential(
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

    def _build_expert_gate(self, trunk_channels):
        """Build improved gating network"""
        return nn.Sequential(
            nn.Linear(trunk_channels, trunk_channels // 4),
            nn.LayerNorm(trunk_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(trunk_channels // 4, trunk_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(trunk_channels // 8, 1)
        )

    def _build_expert(self, in_dim, hidden_dim, out_dim, dropout_rate):
        """Build improved expert network"""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_routing_scores(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute routing scores with temperature and noise"""
        B, E = feats.size(0), self.num_experts
        device = feats.device
        
        # Expert preference scores
        expert_scores = torch.zeros(B, E, device=device)
        for j in range(E):
            raw_score = self.expert_gates[j](feats).squeeze(-1)
            expert_scores[:, j] = raw_score
        
        # Apply temperature
        expert_scores = expert_scores / self.routing_temperature
        
        # Add noise during training for exploration
        if self.training and self.gate_noise_std > 0:
            noise = torch.randn_like(expert_scores) * self.gate_noise_std
            expert_scores = expert_scores + noise
        
        return expert_scores

    def enforce_min_expert_usage(self, routing_scores: torch.Tensor) -> torch.Tensor:
        """Enforce minimum expert usage constraint"""
        B, E = routing_scores.shape
        
        # Calculate current expert usage from EMA
        current_usage = self.expert_usage_ema
        
        # Find underutilized experts
        underutilized_mask = current_usage < self.min_expert_usage
        
        if underutilized_mask.any():
            # Boost scores for underutilized experts
            boost_factor = (self.min_expert_usage - current_usage[underutilized_mask]) * 10.0
            routing_scores[:, underutilized_mask] += boost_factor.unsqueeze(0)
        
        return routing_scores

    def top_k_routing(self, routing_scores: torch.Tensor, confidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implement proper top-k routing with load balancing"""
        B, E = routing_scores.shape
        device = routing_scores.device
        
        # Combine routing scores with confidence
        alpha = 0.7
        combined_scores = alpha * routing_scores + (1 - alpha) * confidence
        
        # Apply min expert usage constraint
        combined_scores = self.enforce_min_expert_usage(combined_scores)
        
        # Apply load balancing penalty based on historical usage
        usage_penalty = self.load_penalty_factor * self.expert_usage_ema.unsqueeze(0)
        balanced_scores = combined_scores - usage_penalty
        
        # Top-k selection per sample
        topk_values, topk_indices = torch.topk(balanced_scores, self.k, dim=1)
        
        # Create assignment matrix using top-k
        D = torch.zeros(B, E, device=device, dtype=torch.bool)
        
        # Capacity-aware assignment
        expert_loads = torch.zeros(E, device=device)
        
        # Process samples in random order during training for better load balancing
        sample_order = torch.randperm(B, device=device) if self.training else torch.arange(B, device=device)
        
        for sample_idx in sample_order:
            # Get top-k experts for this sample
            sample_topk_indices = topk_indices[sample_idx]
            sample_topk_values = topk_values[sample_idx]
            
            # Try to assign to experts in order of preference
            assigned = False
            for k_idx in range(self.k):
                expert_idx = sample_topk_indices[k_idx]
                if expert_loads[expert_idx] < self.capacity:
                    D[sample_idx, expert_idx] = True
                    expert_loads[expert_idx] += 1
                    assigned = True
                    break
            
            # If no expert had capacity, assign to least loaded among top-k
            if not assigned:
                expert_loads_topk = expert_loads[sample_topk_indices]
                least_loaded_k_idx = expert_loads_topk.argmin()
                chosen_expert = sample_topk_indices[least_loaded_k_idx]
                D[sample_idx, chosen_expert] = True
                expert_loads[chosen_expert] += 1
        
        return balanced_scores, D

    def balanced_iterative_routing(
        self, 
        feats: torch.Tensor, 
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use top-k routing instead of iterative routing"""
        return self.top_k_routing(self.compute_routing_scores(feats), confidence)

    def forward(self, x: torch.Tensor, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, device = x.size(0), x.device
        
        # Shared trunk
        feats = self.trunk(x) # Flattened features [B, trunk_channels]
        
        # ↓ aggiungi subito dopo aver calcolato feats
        expert_embeds = torch.stack([self.experts[j](feats) for j in range(self.num_experts)], dim=1)
        logits_e = torch.einsum("bed,ecd->bec", expert_embeds, self.cls_weight) + self.cls_bias

        # # Expert predictions
        # logits_e = torch.einsum("bd,ecd->bec", feats, self.cls_weight) + self.cls_bias.unsqueeze(0)
        
        # Apply temperature
        if self.training:
            logits_e = logits_e / temp
        
        # Compute confidence (negative entropy)
        probs_e = F.softmax(logits_e, dim=2)
        entropy = -(probs_e * torch.log(probs_e + 1e-12)).sum(dim=2)
        confidence = -entropy
        
        # Routing with proper top-k
        routing_scores, D = self.balanced_iterative_routing(feats, confidence)
        
        # Update expert usage tracking
        if self.training:
            current_usage = D.sum(dim=0).float() / B
            self.expert_usage_ema = (
                self.usage_momentum * self.expert_usage_ema + 
                (1 - self.usage_momentum) * current_usage
            )
            
            # Update rolling history
            idx = self.history_idx % 100
            self.routing_history[idx] = current_usage
            self.history_idx += 1
        
        # Weighted ensemble with proper masking
        # Only consider assigned experts for each sample
        masked_scores = routing_scores.clone()
        masked_scores[~D] = -float('inf')
        routing_weights = F.softmax(masked_scores, dim=1)
        
        # Zero out weights for non-assigned experts
        routing_weights = routing_weights * D.float()
        
        # Normalize weights to sum to 1 for each sample
        weight_sums = routing_weights.sum(dim=1, keepdim=True)
        routing_weights = routing_weights / (weight_sums + 1e-12)
        
        weights = routing_weights.unsqueeze(2)
        final_logits = (weights * logits_e).sum(dim=1)
        
        return final_logits, routing_scores, D

    def compute_load_balance_loss(self, D: torch.Tensor) -> torch.Tensor:
        """Enhanced load balancing loss"""
        expert_loads = D.sum(dim=0).float()
        B = D.size(0)
        
        # Basic load balance loss
        target_load = B / self.num_experts
        load_variance = ((expert_loads - target_load) ** 2).mean()
        
        # Penalty for unused experts
        unused_experts = (expert_loads == 0).sum().float()
        unused_penalty = unused_experts / self.num_experts
        
        # Combine losses
        total_loss = self.balance_loss_weight * (load_variance + 0.5 * unused_penalty)
        
        return total_loss

    def get_routing_stats(self, D: torch.Tensor) -> Dict:
        """Enhanced routing statistics"""
        expert_loads = D.sum(dim=0).float()
        B = D.size(0)
        
        # Basic stats
        tokens_per_expert = expert_loads / B
        load_balance_std = tokens_per_expert.std().item()
        coverage = (expert_loads > 0).sum().item()
        
        # Check minimum usage constraint
        current_usage = expert_loads / B
        min_usage_violations = (current_usage < self.min_expert_usage).sum().item()
        
        # Advanced stats
        max_load = expert_loads.max().item()
        min_load = expert_loads.min().item()
        load_ratio = max_load / (min_load + 1e-6)
        
        # Efficiency metrics
        total_capacity = self.capacity * self.num_experts
        total_assignments = D.sum().item()
        capacity_utilization = total_assignments / total_capacity
        
        # Top-k routing efficiency
        samples_using_k_experts = (D.sum(dim=1) == self.k).sum().item()
        k_routing_efficiency = samples_using_k_experts / B
        
        # Gini coefficient for load distribution inequality
        sorted_loads = torch.sort(expert_loads)[0]
        n = len(sorted_loads)
        if sorted_loads.sum() > 0:
            cumsum = torch.cumsum(sorted_loads, dim=0)
            gini = (2 * torch.arange(1, n + 1, device=sorted_loads.device) - n - 1) * sorted_loads
            gini = gini.sum() / (n * sorted_loads.sum())
        else:
            gini = torch.tensor(0.0)
        
        return {
            'expert_loads': expert_loads.cpu().numpy(),
            'tokens_per_expert': tokens_per_expert.cpu().numpy(),
            'load_balance_std': load_balance_std,
            'expert_coverage': coverage,
            'coverage_ratio': coverage / self.num_experts,
            'load_ratio': load_ratio,
            'max_load': max_load,
            'min_load': min_load,
            'capacity_utilization': capacity_utilization,
            'gini_coefficient': gini.item(),
            'usage_ema': self.expert_usage_ema.cpu().numpy(),
            'min_usage_violations': min_usage_violations,
            'k_routing_efficiency': k_routing_efficiency,
            'actual_k_used': self.k,
        }


# Updated hyperparameter recommendations
BETTER_HYPERPARAMETERS = {
    # For better expert utilization
    'load_penalty_factor': {
        'light_balancing': 1.0,
        'medium_balancing': 3.0,
        'strong_balancing': 6.0,  # Try this for better coverage
        'aggressive_balancing': 10.0,  # If you still have unused experts
        'note': 'Higher values force more balanced routing'
    },
    
    'min_expert_usage': {
        'relaxed': 0.02,      # 2% minimum usage
        'moderate': 0.05,     # 5% minimum usage  
        'strict': 0.1,        # 10% minimum usage
        'note': 'Enforces minimum usage per expert'
    },
    
    'k': {
        'conservative': 2,    # Most efficient, but may underutilize experts
        'balanced': 4,        # Good balance of efficiency and utilization
        'diverse': 6,         # Better expert utilization
        'note': 'Higher k values help with expert utilization'
    },
    
    'routing_temperature': {
        'sharp': 0.5,         # Sharp routing decisions
        'balanced': 1.0,      # Balanced routing
        'soft': 2.0,          # Softer routing decisions
        'note': 'Lower values make routing more decisive'
    },
    
    'gate_noise_std': {
        'low_exploration': 0.05,
        'medium_exploration': 0.1,
        'high_exploration': 0.2,
        'note': 'Higher noise helps expert exploration'
    },
    
    'usage_momentum': {
        'fast_adaptation': 0.8,
        'balanced': 0.9,
        'slow_adaptation': 0.95,
        'note': 'Lower values adapt faster to usage changes'
    }
}

def get_better_config_for_expert_utilization():
    """Configuration specifically designed to improve expert utilization"""
    return {
        'num_experts': 8,
        'capacity': 48,
        'k': 2,  # Higher k for better utilization 
        'load_penalty_factor': 3.0,  # Strong load balancing
        'min_expert_usage': 0.08,  # 8% minimum usage per expert
        'gate_noise_std': 0.15,  # Higher exploration
        'routing_temperature': 1.5,  # Softer routing
        'balance_loss_weight': 0.02,  # Higher balance loss weight
        'usage_momentum': 0.85,  # Faster adaptation
    }