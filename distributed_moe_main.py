import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json
from datetime import datetime

# Import the new Distributed MoE components
from distributed_moe_2 import DistributedMoE, DISTRIBUTED_MOE_CONFIGS, SimpleBaseline
from distributed_moe_trainer import train_distributed_moe

def get_distributed_moe_config(size='medium'):
    """Get configuration for distributed MoE model"""
    base_config = DISTRIBUTED_MOE_CONFIGS[size].copy()
    
    # Add training specific parameters
    base_config.update({
        'capacity': 64,  # Capacity per expert
        'load_penalty_factor': 2.0,
        'min_expert_usage': 0.1,  # 10% minimum usage
        'gate_noise_std': 0.1,
        'routing_temperature': 2.0,
        'balance_loss_weight': 0.01,
        'usage_momentum': 0.9,
        'expert_dropout': 0.1,
    })
    
    return base_config

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    
    # Model configuration - choose from 'tiny', 'small', 'medium', 'large'
    model_size = 'medium'
    distributed_config = get_distributed_moe_config(model_size)
    
    # Training configuration
    batch_size = 256  # Smaller batch size for more frequent updates
    epochs = 150
    base_lr = 0.001
    run_dir = f"runs/distributed_moe_{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resume training configuration
    resume_from_checkpoint = False
    starting_epoch = 0
    checkpoint_path = None

    # Advanced training hyperparameters
    advanced_config = {
        'temp_start': 2.0,
        'temp_end': 0.5,
        'patience': 25,
        'gradient_clip_value': 1.0,
    }

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Print configuration
    print("\n===== DISTRIBUTED MOE CONFIGURATION =====")
    print(f"Model size: {model_size}")
    print(f"Number of experts: {distributed_config['num_experts']}")
    print(f"Expert channels: {distributed_config['expert_channels']}")
    print(f"Expert feature dim: {distributed_config['expert_feature_dim']}")
    print(f"Routing dim: {distributed_config['routing_dim']}")
    print(f"Top-k routing (k): {distributed_config['k']}")
    print(f"Capacity per expert: {distributed_config['capacity']}")
    print(f"Minimum expert usage: {distributed_config['min_expert_usage']:.1%}")
    print(f"Load penalty factor: {distributed_config['load_penalty_factor']}")
    print("==========================================\n")

    # ===== DATA PREPARATION =====
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # ===== MODEL PREPARATION =====
    
    # Create Distributed MoE model
    model = DistributedMoE(
        num_experts=distributed_config['num_experts'],
        capacity=distributed_config['capacity'],
        k=distributed_config['k'],
        expert_channels=distributed_config['expert_channels'],
        expert_feature_dim=distributed_config['expert_feature_dim'],
        routing_dim=distributed_config['routing_dim'],
        num_classes=10,  # CIFAR-10
        load_penalty_factor=distributed_config['load_penalty_factor'],
        min_expert_usage=distributed_config['min_expert_usage'],
        gate_noise_std=distributed_config['gate_noise_std'],
        routing_temperature=distributed_config['routing_temperature'],
        balance_loss_weight=distributed_config['balance_loss_weight'],
        usage_momentum=distributed_config['usage_momentum'],
        expert_dropout=distributed_config['expert_dropout'],
    )
    
    print(f"Created Distributed MoE model:")
    model.count_parameters()
    print()
    
    # Create baseline for comparison
    baseline = SimpleBaseline(num_classes=10, channels=[64, 128, 256])
    baseline_params = baseline.count_parameters()
    moe_params = model.count_parameters()
    print(f"Parameter efficiency: {baseline_params / moe_params['total']:.2f}x compression vs baseline")
    print()
    
    # Load checkpoint if resuming
    if resume_from_checkpoint and checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                starting_epoch = checkpoint.get('epoch', 0)
            else:
                model.load_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint from epoch {starting_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            resume_from_checkpoint = False
            starting_epoch = 0

    # ===== TRAINING =====
    
    print("Starting Distributed MoE training...")
    trained_model = train_distributed_moe(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        resume_from_checkpoint=resume_from_checkpoint,
        starting_epoch=starting_epoch,
        base_lr=base_lr,
        run_dir=run_dir,
        advanced_config=advanced_config,
    )

    # ===== SAVE CONFIGURATION =====
    config_to_save = {
        "model_type": "DistributedMoE",
        "model_size": model_size,
        "dataset": "CIFAR-10",
        "batch_size": batch_size,
        "epochs": epochs,
        "base_lr": base_lr,
        "resume_from_checkpoint": resume_from_checkpoint,
        "starting_epoch": starting_epoch,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "train_transform": str(transform_train),
        "test_transform": str(transform_test),
        "distributed_config": distributed_config,
        "advanced_config": advanced_config,
        "parameter_counts": moe_params,
        "baseline_comparison": {
            "baseline_params": baseline_params,
            "moe_params": moe_params['total'],
            "compression_ratio": baseline_params / moe_params['total']
        },
        "architecture_details": {
            "routing_trunk": "Minimal shared processing for routing decisions only",
            "expert_feature_extractors": "Individual lightweight feature extractors per expert",
            "expert_classifiers": "Individual classifier heads per expert",
            "routing_mechanism": "Top-k routing with capacity constraints and load balancing"
        },
            "key_features": [
                "Distributed feature extraction - each expert has its own feature extractor",
                "Top-k routing with capacity constraints",
                "Load balancing and expert dropout",
                "Minimal shared routing trunk",
                "Individual classifier heads per expert"
            ]
        }
    
    # Optionally, save configuration to a file
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=4)