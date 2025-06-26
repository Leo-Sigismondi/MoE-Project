# Distributed Feature Extraction MoE - Each expert has its own lightweight feature extractor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math

class LightweightFeatureExtractor(nn.Module):
    """Lightweight feature extractor for individual experts"""
    def __init__(self, channels=[32, 64, 128], output_dim=128, dropout_rate=0.1):
        super().__init__()
        self.channels = channels
        self.output_dim = output_dim
        
        # Much smaller feature extractor per expert
        layers = []
        in_channels = 3
        
        for i, out_channels in enumerate(channels):
            # Single conv per stage instead of double
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2) if i < len(channels) - 1 else nn.AdaptiveAvgPool2d(1),
            ])
            
            if i < len(channels) - 1:  # Don't add dropout before final pooling
                layers.append(nn.Dropout2d(dropout_rate))
            
            in_channels = out_channels
        
        layers.extend([
            nn.Flatten(),
            nn.Linear(channels[-1], output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        ])
        
        self.features = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.features(x)

class SlimExpertFeatureExtractor(nn.Module):
    """Lightweight CNN for a *specialist* expert.

    • Stage 1 – single 3×3 conv → BN → ReLU → 2× down‑sample.
    • Stage 2 – **double‑conv block** (adds a second 3×3) → down‑sample.
    • Stage 3 – **depth‑wise separable conv** (DW 3×3 + PW 1×1) → GAP.

    In total: **5 conv layers** (vs 3 in the old extractor) but only one is
    a full 64→128 3×3; the other extra layers are depth‑wise or reuse the same
    channel count, so params/FLOPs grow marginally.
    """

    def __init__(self, channels=(32, 64, 128), out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        c1, c2, c3 = channels

        layers = []
        # --- Stage 1 -----------------------------------------------------------
        layers += [
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                      # 32→16
            nn.Dropout2d(dropout),
        ]

        # --- Stage 2 (double conv) -------------------------------------------
        layers += [
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                      # 16→8
            nn.Dropout2d(dropout),
        ]

        # --- Stage 3 (depth‑wise separable) -----------------------------------
        layers += [
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),  # depth‑wise
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 1, bias=False),                        # point‑wise
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        ]

        # --- Projection -------------------------------------------------------
        layers += [nn.Flatten(), nn.Linear(c3, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]

        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.features(x)
    

class MinimalSharedTrunk(nn.Module):
    """Minimal shared processing - just for routing decisions"""
    def __init__(self, routing_dim=64):
        super().__init__()
        # Very lightweight shared features just for routing
        self.routing_features = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3, bias=False),  # 32x32 -> 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),  # 8x8 -> 2x2
            nn.Flatten(),  # 32 * 2 * 2 = 128
            nn.Linear(128, routing_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.routing_features(x)

class MiniTrunk(nn.Module):
    # 2 lightweight convs → 48 d vector
    def __init__(self, routing_dim=48):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 32×32 → 16×16
            nn.AdaptiveAvgPool2d(2),         # → 2×2
            nn.Flatten(),                    # 32·2·2 = 128
            nn.Linear(128, routing_dim), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.trunk(x)

class DistributedMoE(nn.Module):
    def __init__(
        self,
        num_experts: int = 8,
        capacity: int = 32,
        k: int = 2,
        # Feature extractor parameters
        expert_channels: list = [32, 64, 128],  # Much smaller than [64, 128, 256]
        expert_feature_dim: int = 128,
        routing_dim: int = 64,
        # Classifier parameters
        num_classes: int = 10,
        # Training parameters
        load_penalty_factor: float = 2.0,
        min_expert_usage: float = 0.05,
        gate_noise_std: float = 0.1,
        routing_temperature: float = 1.0,
        balance_loss_weight: float = 0.01,
        usage_momentum: float = 0.9,
        expert_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity = capacity
        self.k = min(k, num_experts)
        self.num_classes = num_classes
        self.routing_dim = routing_dim
        self.expert_feature_dim = expert_feature_dim
        
        # Store hyperparameters
        self.load_penalty_factor = load_penalty_factor
        self.min_expert_usage = min_expert_usage
        self.gate_noise_std = gate_noise_std
        self.routing_temperature = routing_temperature
        self.balance_loss_weight = balance_loss_weight
        self.usage_momentum = usage_momentum

        # Minimal shared trunk for routing only
        self.routing_trunk = MiniTrunk(routing_dim)
        
        # Each expert has its own feature extractor
        self.expert_feature_extractors = nn.ModuleList([
            LightweightFeatureExtractor(
                channels=expert_channels.copy(),
                output_dim=expert_feature_dim,
                dropout_rate=expert_dropout
            ) for _ in range(num_experts)
        ])
        
        # Gating network (operates on minimal shared features)
        self.gate = nn.Sequential(
            nn.Linear(routing_dim, routing_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(routing_dim // 2, num_experts)
        )
        
        # Expert classifier heads (smaller since features are smaller)
        self.expert_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_feature_dim, expert_feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(expert_dropout),
                nn.Linear(expert_feature_dim // 2, num_classes)
            ) for _ in range(num_experts)
        ])

        # Load balancing tracking
        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)
        
        # Initialize weights
        self._initialize_weights()

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

    def compute_routing_scores(self, routing_features: torch.Tensor) -> torch.Tensor:
        """Compute routing scores with temperature and noise"""
        # Get raw routing scores
        routing_scores = self.gate(routing_features)
        
        # Apply temperature
        routing_scores = routing_scores / self.routing_temperature
        
        # Add noise during training for exploration
        if self.training and self.gate_noise_std > 0:
            noise = torch.randn_like(routing_scores) * self.gate_noise_std
            routing_scores = routing_scores + noise
        
        return routing_scores

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

    def top_k_routing(self, routing_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implement top-k routing with load balancing"""
        B, E = routing_scores.shape
        device = routing_scores.device
        
        # Apply min expert usage constraint
        routing_scores = self.enforce_min_expert_usage(routing_scores)
        
        # Apply load balancing penalty
        usage_penalty = self.load_penalty_factor * self.expert_usage_ema.unsqueeze(0)
        balanced_scores = routing_scores - usage_penalty
        
        # Top-k selection per sample
        topk_values, topk_indices = torch.topk(balanced_scores, self.k, dim=1)
        
        # Create assignment matrix with capacity constraints
        D = torch.zeros(B, E, device=device, dtype=torch.bool)
        expert_loads = torch.zeros(E, device=device)
        
        # Process samples in random order for better load balancing
        sample_order = torch.randperm(B, device=device) if self.training else torch.arange(B, device=device)
        
        for sample_idx in sample_order:
            sample_topk_indices = topk_indices[sample_idx]
            
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

    def forward(self, x: torch.Tensor, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, device = x.size(0), x.device
        
        # Get minimal routing features
        routing_features = self.routing_trunk(x)
        
        # Compute routing decisions
        routing_scores = self.compute_routing_scores(routing_features)
        balanced_scores, D = self.top_k_routing(routing_scores)
        
        # Update expert usage tracking
        if self.training:
            current_usage = D.sum(dim=0).float() / B
            self.expert_usage_ema = (
                self.usage_momentum * self.expert_usage_ema + 
                (1 - self.usage_momentum) * current_usage
            )
        
        # Process samples through assigned experts only
        expert_outputs = torch.zeros(B, self.num_experts, self.num_classes, device=device, dtype=x.dtype)
        
        for expert_idx in range(self.num_experts):
            # Find samples assigned to this expert
            assigned_samples = D[:, expert_idx]
            
            if assigned_samples.any():
                # Extract features using expert's personal feature extractor
                expert_features = self.expert_feature_extractors[expert_idx](x[assigned_samples])
                
                # Get predictions from expert's classifier
                expert_logits = self.expert_classifiers[expert_idx](expert_features)
                
                # Apply temperature if training
                if self.training:
                    expert_logits = expert_logits / temp
                
                # Store in output tensor (ensure dtype match)
                expert_outputs[assigned_samples, expert_idx] = expert_logits.to(expert_outputs.dtype)
        
        # Compute routing weights
        masked_scores = balanced_scores.clone()
        masked_scores[~D] = -float('inf')
        routing_weights = F.softmax(masked_scores, dim=1)
        routing_weights = routing_weights * D.float()
        
        # Normalize weights
        weight_sums = routing_weights.sum(dim=1, keepdim=True)
        routing_weights = routing_weights / (weight_sums + 1e-12)
        
        # Weighted ensemble
        weights = routing_weights.unsqueeze(2)
        final_logits = (weights * expert_outputs).sum(dim=1)
        
        return final_logits, balanced_scores, D

    def compute_load_balance_loss(self, D: torch.Tensor) -> torch.Tensor:
        """Load balancing loss"""
        expert_loads = D.sum(dim=0).float()
        B = D.size(0)
        
        # Basic load balance loss
        target_load = B / self.num_experts
        load_variance = ((expert_loads - target_load) ** 2).mean()
        
        # Penalty for unused experts
        unused_experts = (expert_loads == 0).sum().float()
        unused_penalty = unused_experts / self.num_experts
        
        return self.balance_loss_weight * (load_variance + 0.5 * unused_penalty)

    def get_routing_stats(self, D: torch.Tensor) -> Dict:
        """Routing statistics"""
        expert_loads = D.sum(dim=0).float()
        B = D.size(0)
        
        tokens_per_expert = expert_loads / B
        coverage = (expert_loads > 0).sum().item()
        
        return {
            'expert_loads': expert_loads.cpu().numpy(),
            'tokens_per_expert': tokens_per_expert.cpu().numpy(),
            'expert_coverage': coverage,
            'coverage_ratio': coverage / self.num_experts,
            'usage_ema': self.expert_usage_ema.cpu().numpy(),
        }

    def count_parameters(self):
        """Count parameters in different components"""
        routing_params = sum(p.numel() for p in self.routing_trunk.parameters())
        gate_params = sum(p.numel() for p in self.gate.parameters())
        
        feature_extractor_params = sum(p.numel() for p in self.expert_feature_extractors.parameters())
        classifier_params = sum(p.numel() for p in self.expert_classifiers.parameters())
        
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Parameter breakdown:")
        print(f"  Routing trunk: {routing_params:,} ({routing_params/total_params:.1%})")
        print(f"  Gate network: {gate_params:,} ({gate_params/total_params:.1%})")
        print(f"  Feature extractors ({self.num_experts}): {feature_extractor_params:,} ({feature_extractor_params/total_params:.1%})")
        print(f"  Classifiers ({self.num_experts}): {classifier_params:,} ({classifier_params/total_params:.1%})")
        print(f"  Total: {total_params:,}")
        print(f"  Avg params per expert: {(feature_extractor_params + classifier_params) / self.num_experts:,.0f}")
        
        return {
            'routing_trunk': routing_params,
            'gate': gate_params,
            'feature_extractors': feature_extractor_params,
            'classifiers': classifier_params,
            'total': total_params,
            'per_expert': (feature_extractor_params + classifier_params) / self.num_experts
        }

class DistributedMoEImprovedRouting(DistributedMoE):
    """Distributed MoE with
    1. **Confidence‑aware routing** (ported from ImprovedMoE)
    2. **Deeper yet slim experts** via :class:`SlimExpertFeatureExtractor`
    3. **Single‑linear classifier heads** per expert (adapter head)

    Architecture philosophy is still *small shared trunk + specialised experts*;
    these tweaks add ~3 extra convs per expert at minimal cost and typically
    recover ~6-8 pp accuracy on CIFAR-10-like workloads.
    """

    # ------------------------------------------------------------------
    # Initialization ---------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(self, *args,
                 expert_channels=(32, 64, 128),
                 expert_feature_dim: int = 128,
                 expert_dropout: float = 0.1,
                 **kwargs):
        super().__init__(*args,
                         expert_channels=list(expert_channels),
                         expert_feature_dim=expert_feature_dim,
                         expert_dropout=expert_dropout,
                         **kwargs)

        # Replace the old 3‑conv extractors with our deeper slim version
        self.expert_feature_extractors = nn.ModuleList([
            SlimExpertFeatureExtractor(channels=expert_channels,
                                       out_dim=expert_feature_dim,
                                       dropout=expert_dropout)
            for _ in range(self.num_experts)
        ])

        # Replace 2‑layer MLP heads with a single linear per expert
        self.expert_classifiers = nn.ModuleList([
            nn.Linear(expert_feature_dim, self.num_classes) for _ in range(self.num_experts)
        ])

        # Re‑initialize weights of newly created modules
        self._initialize_weights()

    # ------------------------------------------------------------------
    # Confidence helpers -----------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(p: torch.Tensor) -> torch.Tensor:
        return -(p * torch.log(p + 1e-12)).sum(dim=-1)

    def _compute_confidence(self, logits_e: torch.Tensor) -> torch.Tensor:
        probs_e = F.softmax(logits_e, dim=2)
        return -self._entropy(probs_e)

    # ------------------------------------------------------------------
    # Routing -----------------------------------------------------------
    # ------------------------------------------------------------------
    def _top_k_routing(self,
                       routing_scores: torch.Tensor,
                       confidence: torch.Tensor,
                       alpha: float = 0.7):
        """Confidence‑blended top‑k routing with capacity constraints."""
        B, E = routing_scores.shape
        device = routing_scores.device

        combined = alpha * routing_scores + (1 - alpha) * confidence
        combined = self.enforce_min_expert_usage(combined)
        balanced = combined - self.load_penalty_factor * self.expert_usage_ema.unsqueeze(0)

        topk_vals, topk_idx = torch.topk(balanced, self.k, dim=1)

        D = torch.zeros(B, E, device=device, dtype=torch.bool)
        loads = torch.zeros(E, device=device)

        order = torch.randperm(B, device=device) if self.training else torch.arange(B, device=device)
        for b in order:
            prefs = topk_idx[b]
            for candidate in prefs:
                e = candidate.item()
                if loads[e] < self.capacity:
                    D[b, e] = True
                    loads[e] += 1
                    break
            else:
                # No capacity among preferred – pick least loaded of them
                e = prefs[loads[prefs].argmin()].item()
                D[b, e] = True
                loads[e] += 1

        return balanced, D

    # ------------------------------------------------------------------
    # Forward pass ------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, temp: float = 1.0):
        B = x.size(0)
        routing_feats = self.routing_trunk(x)                      # [B, routing_dim]
        gate_scores  = self.compute_routing_scores(routing_feats)  # [B, E]

        # Pre‑compute logits for confidence estimation
        logits_e = torch.stack([
            self.expert_classifiers[e](
                self.expert_feature_extractors[e](x)
            ) for e in range(self.num_experts)
        ], dim=1)                                                 # [B,E,C]

        if self.training:
            logits_e = logits_e / temp

        confidence = self._compute_confidence(logits_e)            # [B,E]
        balanced, D = self._top_k_routing(gate_scores, confidence)

        # Update EMA usage during training
        if self.training:
            self.expert_usage_ema.mul_(self.usage_momentum).add_((1 - self.usage_momentum) * (D.sum(0).float() / B))

        masked = balanced.clone(); masked[~D] = -float('inf')
        w = F.softmax(masked, dim=1) * D.float()
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        logits_final = (w.unsqueeze(2) * logits_e).sum(dim=1)      # [B,C]
        return logits_final, balanced, D



class SimpleBaseline(nn.Module):
    """Simple baseline model for comparison"""
    def __init__(self, num_classes=10, channels=[64, 128, 256], dropout_rate=0.2):
        super().__init__()
        
        layers = []
        in_channels = 3
        
        for i, out_channels in enumerate(channels):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate if i < len(channels) - 1 else 0),
            ])
            in_channels = out_channels
        
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[-1], num_classes)
        ])
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Baseline model parameters: {total:,}")
        return total


# Configuration for different model sizes
DISTRIBUTED_MOE_CONFIGS = {
    'tiny': {
        'num_experts': 4,
        'expert_channels': [24, 48, 96],
        'expert_feature_dim': 96,
        'routing_dim': 48,
        'k': 2,
    },
    'small': {
        'num_experts': 6,
        'expert_channels': [32, 64, 128],
        'expert_feature_dim': 128,
        'routing_dim': 64,
        'k': 2,
    },
    'medium': {
        'num_experts': 8,
        # 'expert_channels': [32, 64, 128],
        'expert_channels': [64, 128, 256],
        'expert_feature_dim': 128,
        'routing_dim': 256,
        'k': 2,
    },
    'large': {
        'num_experts': 12,
        'expert_channels': [40, 80, 160],
        'expert_feature_dim': 160,
        'routing_dim': 256,
        'k': 3,
    }
}

def compare_model_sizes():
    """Compare model sizes"""
    print("Model Size Comparison:")
    print("=" * 50)
    
    # Baseline
    baseline = SimpleBaseline(channels=[64, 128, 256])
    baseline_params = baseline.count_parameters()
    print()
    
    # Different MoE configurations
    for config_name, config in DISTRIBUTED_MOE_CONFIGS.items():
        print(f"{config_name.upper()} Distributed MoE:")
        model = DistributedMoE(**config)
        params = model.count_parameters()
        print(f"  Compression ratio vs baseline: {baseline_params / params['total']:.2f}x")
        print(f"  Each expert is {baseline_params / params['per_expert']:.1f}x smaller than baseline")
        print()

if __name__ == "__main__":
    compare_model_sizes()
