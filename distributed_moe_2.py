import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch import Tensor

# -----------------------------
# SlimExpertFeatureExtractor
# -----------------------------
class SlimExpertFeatureExtractor(nn.Module):
    def __init__(self, channels=(32, 64, 128), out_dim=128, dropout=0.1):
        super().__init__()
        c1, c2, c3 = channels

        self.features = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, c1, 3, padding=1),
            nn.BatchNorm2d(c1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout),

            # Stage 2
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(dropout),

            # Stage 3
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3), 
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1),
            nn.BatchNorm2d(c3), 
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),

            # Projection
            nn.Flatten(),
            nn.Linear(c3, out_dim), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.features(x)


# -----------------------------
# MiniTrunk (for routing)
# -----------------------------
class MiniTrunk(nn.Module):
    def __init__(self, routing_dim=48):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(128, routing_dim), 
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.trunk(x)


# -----------------------------
# DistributedMoE
# -----------------------------
class DistributedMoE(nn.Module):
    def __init__(self, num_experts=6, capacity=32, k=2, expert_channels=(32, 64, 128),
                 expert_feature_dim=128, routing_dim=64, num_classes=10,
                 expert_dropout=0.1, load_penalty_factor=2.0,
                 min_expert_usage=0.05, gate_noise_std=0.1,
                 routing_temperature=1.0, balance_loss_weight=0.01,
                 lambda_aux =0.1,  lambda_ortho=0.1,
                 usage_momentum=0.9, uniform_routing=False):
        super().__init__()

        self.num_experts = num_experts
        self.capacity = capacity
        self.k = min(k, num_experts)
        self.num_classes = num_classes
        self.routing_dim = routing_dim
        self.expert_feature_dim = expert_feature_dim
        self.load_penalty_factor = load_penalty_factor
        self.min_expert_usage = min_expert_usage
        self.gate_noise_std = gate_noise_std
        self.routing_temperature = routing_temperature
        self.balance_loss_weight = balance_loss_weight
        self.lambda_aux = lambda_aux
        self.lambda_ortho = lambda_ortho
        self.usage_momentum = usage_momentum
        self.uniform_routing = uniform_routing

        self.routing_trunk = MiniTrunk(routing_dim)
        self.gate = nn.Sequential(
            nn.Linear(routing_dim, routing_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(routing_dim // 2, num_experts)
        )

        self.expert_feature_extractors = nn.ModuleList([
            SlimExpertFeatureExtractor(channels=expert_channels,
                                       out_dim=expert_feature_dim,
                                       dropout=expert_dropout)
            for _ in range(num_experts)
        ])

        self.expert_classifiers = nn.ModuleList([
            nn.Linear(expert_feature_dim, num_classes)
            for _ in range(num_experts)
        ])

        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
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
        
        return (load_variance + 0.5 * unused_penalty)

    def compute_auxiliary_loss(self, routing_scores: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss for routing scores"""
        # Compute entropy of routing scores
        probs = F.softmax(routing_scores, dim=1)
        entropy = -self._entropy(probs)
        
        # Encourage diversity in routing decisions
        return entropy.mean()
    
    def compute_orthogonality_loss(self, routing_scores: torch.Tensor) -> torch.Tensor:
        """Orthogonality loss for routing scores"""
        # Normalize routing scores
        norm_scores = F.normalize(routing_scores, dim=1)
        
        # Compute pairwise dot products
        dot_products = torch.mm(norm_scores, norm_scores.t())
        
        # Encourage orthogonality
        return (1 - dot_products.diag().mean())
    
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
    @staticmethod
    def _entropy(p):
        return -(p * torch.log(p + 1e-12)).sum(dim=-1)

    def _compute_confidence(self, logits_e):
        probs_e = F.softmax(logits_e, dim=2)
        return -self._entropy(probs_e)

    def compute_routing_scores(self, routing_features):
        scores = self.gate(routing_features) / self.routing_temperature
        if self.training and self.gate_noise_std > 0:
            scores += torch.randn_like(scores) * self.gate_noise_std
        return scores

    def enforce_min_expert_usage(self, scores):
        underused = self.expert_usage_ema < self.min_expert_usage
        if underused.any():
            boost = (self.min_expert_usage - self.expert_usage_ema[underused]) * 10.0
            scores[:, underused] += boost.unsqueeze(0)
        return scores

    def _top_k_routing(self, routing_scores, confidence, alpha=0.7):
        B, E = routing_scores.shape
        combined = alpha * routing_scores + (1 - alpha) * confidence
        combined = self.enforce_min_expert_usage(combined)
        balanced = combined - self.load_penalty_factor * self.expert_usage_ema.unsqueeze(0)

        topk_vals, topk_idx = torch.topk(balanced, self.k, dim=1)
        D = torch.zeros(B, E, device=routing_scores.device, dtype=torch.bool)
        loads = torch.zeros(E, device=routing_scores.device)
        order = torch.randperm(B, device=routing_scores.device) if self.training else torch.arange(B, device=routing_scores.device)

        for b in order:
            for e in topk_idx[b]:
                if loads[e] < self.capacity:
                    D[b, e] = True
                    loads[e] += 1
                    break
            else:
                fallback = topk_idx[b][loads[topk_idx[b]].argmin()]
                D[b, fallback] = True
                loads[fallback] += 1

        return balanced, D
    
    def freeze_gate(self):
        for param in self.gate.parameters():
            param.requires_grad = False

    def unfreeze_gate(self):
        for param in self.gate.parameters():
            param.requires_grad = True


    def forward(self, x, temp=1.0, uniform_routing=False):
        if uniform_routing:
            # Uniform routing: all experts get equal load
            B = x.size(0)
            balanced = torch.ones(B, self.num_experts, device=x.device) / self.num_experts
            D = balanced > 0
            logits_final = torch.stack([
                self.expert_classifiers[e](self.expert_feature_extractors[e](x))
                for e in range(self.num_experts)
            ], dim=1).mean(dim=1)
            return logits_final, balanced, D
        else:
            # Normal routing with gating
            B = x.size(0)
            routing_feats = self.routing_trunk(x)
            gate_scores = self.compute_routing_scores(routing_feats)

            logits_e = torch.stack([
                self.expert_classifiers[e](self.expert_feature_extractors[e](x))
                for e in range(self.num_experts)
            ], dim=1)

            if self.training:
                logits_e = logits_e / temp

            confidence = self._compute_confidence(logits_e)
            balanced, D = self._top_k_routing(gate_scores, confidence)

            if self.training:
                usage = D.sum(dim=0).float() / B
                self.expert_usage_ema = self.usage_momentum * self.expert_usage_ema + (1 - self.usage_momentum) * usage

            masked = balanced.clone(); masked[~D] = -float('inf')
            w = F.softmax(masked, dim=1) * D.float()
            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

            logits_final = (w.unsqueeze(2) * logits_e).sum(dim=1)
            return logits_final, balanced, D



# -----------------------------
# SimpleBaseline
# -----------------------------
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
        'routing_dim': 128,
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