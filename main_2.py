import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from metrics import *

from Vision_trainer import train_single_model
from Model import SingleModel, SingleImprovedModel
from improved_waterfall_trainer import train_moe_waterfall_improved
from Vision_trainer_MoE import train_moe_cnn
from improved_moe import ImprovedMoE, get_better_config_for_expert_utilization
import json
from datetime import datetime

if __name__ == "__main__":
    # ===== CONFIGURATION FOR BETTER EXPERT UTILIZATION =====
    
    # Use the specialized config for better expert utilization
    expert_utilization_config = get_better_config_for_expert_utilization()
    
    # Training configuration
    batch_size = 256  # Keep this reasonable
    epochs = 150
    base_lr = 0.001
    run_dir = f"runs/fixed_moe_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resume training configuration
    resume_from_checkpoint = False
    starting_epoch = 0
    checkpoint_path = "runs/previous_experiment/checkpoints/epoch_140.pth"

    # Extract values from the specialized config
    num_experts = expert_utilization_config['num_experts']
    capacity = expert_utilization_config['capacity'] 
    k = expert_utilization_config['k']
    load_penalty_factor = expert_utilization_config['load_penalty_factor']

    # Advanced hyperparameters optimized for expert utilization
    advanced_config = {
        'gate_noise_std': expert_utilization_config['gate_noise_std'],
        'routing_temperature': expert_utilization_config['routing_temperature'],
        'balance_loss_weight': expert_utilization_config['balance_loss_weight'],
        'expert_dropout': 0.1,
        'min_expert_usage': expert_utilization_config['min_expert_usage'],
        'max_iterations': 3,
        'diversity_temp': 1.5,
        'usage_momentum': expert_utilization_config['usage_momentum'],
    }

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Print configuration for debugging
    print("\n===== EXPERT UTILIZATION CONFIGURATION =====")
    print(f"Number of experts: {num_experts}")
    print(f"Capacity per expert: {capacity}")
    print(f"Top-k routing (k): {k}")
    print(f"Load penalty factor: {load_penalty_factor}")
    print(f"Minimum expert usage: {advanced_config['min_expert_usage']:.1%}")
    print(f"Gate noise std: {advanced_config['gate_noise_std']}")
    print(f"Routing temperature: {advanced_config['routing_temperature']}")
    print("=============================================\n")

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
                             num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True, persistent_workers=True)

    # ===== MODEL PREPARATION =====
    small_expert_overrides = {
    "expert_hidden": 128,   # was 256
    "embed_dim": 64,        # was 128
    "capacity": 32,         # was 48
}
    
    if num_experts > 1:
        # Create improved MoE model with fixed hyperparameter usage
        model = ImprovedMoE(
            num_experts=num_experts,
            k=k,  # This will now be properly used!
            load_penalty_factor=load_penalty_factor,
            **small_expert_overrides,
            **advanced_config  # This includes min_expert_usage now
        )
        
        print(f"Created MoE model with {num_experts} experts")
        print(f"Model will use top-{k} routing with capacity {capacity} per expert")
        print(f"Minimum expert usage enforced: {advanced_config['min_expert_usage']:.1%}")
        
        # Load checkpoint if resuming
        if resume_from_checkpoint and os.path.isfile(checkpoint_path):
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

        # Train improved MoE model
        train_moe_waterfall_improved(
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
    else:
        # Single model training (unchanged)
        model = SingleImprovedModel()
        if resume_from_checkpoint and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        train_single_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            use_amp=True,
            lr=base_lr,
        )

    # ===== SAVE CONFIGURATION =====
    params = {
        "model_type": "FixedImprovedMoE" if num_experts > 1 else "SingleModel",
        "dataset": "CIFAR-10",
        "batch_size": batch_size,
        "num_experts": num_experts,
        "capacity": capacity,
        "k": k,  # Now properly documented
        "epochs": epochs,
        "base_lr": base_lr,
        "load_penalty_factor": load_penalty_factor,
        "resume_from_checkpoint": resume_from_checkpoint,
        "starting_epoch": starting_epoch,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "train_transform": str(transform_train),
        "test_transform": str(transform_test),
        "advanced_config": advanced_config,
        "expert_utilization_config": expert_utilization_config,
        "fixes_applied": [
            "Proper top-k routing implementation",
            "Min expert usage constraint enforcement", 
            "Enhanced load balancing penalties",
            "Better routing statistics tracking",
            "Improved expert assignment logic"
        ]
    }

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(params, f, indent=4)

    print(f"Training completed. Results saved to: {run_dir}")
    print("Configuration saved to config.json")
    
    # Print expected improvements
    print("\n===== EXPECTED IMPROVEMENTS =====")
    print("✅ All 8 experts should now be utilized")
    print("✅ Top-k routing will be properly applied") 
    print("✅ Minimum expert usage will be enforced")
    print("✅ Better load balancing with higher penalty factor")
    print("✅ Enhanced routing statistics will show utilization metrics")
    print("=====================================")