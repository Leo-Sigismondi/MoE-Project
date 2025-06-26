import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from metrics import *

from Vision_trainer import train_single_model
from Model import SingleImprovedModel
from collaborative_waterfall_moe_2 import CollaborativeWaterfallMoE
from collaborative_trainer import train_moe_waterfall
import json
from datetime import datetime


def plot_from_checkpoint(checkpoint_path, dataloader, device, num_experts, capacity, num_classes=10):
    # Load model
    model = CollaborativeWaterfallMoE(num_experts=num_experts, capacity=capacity, num_classes=num_classes)
    # Load the checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)  # or 'cuda' if using GPU
    # Restore model, optimizer, and scaler states
    model.load_state_dict(ckpt["model"])
    # optimizer.load_state_dict(ckpt["optimizer"])
    # if scaler and ckpt["scaler"] is not None:
    #     scaler.load_state_dict(ckpt["scaler"])
    model.to(device)
    # Plot
    save_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    plot_per_expert_confusion(model, dataloader, device, num_classes=num_classes,
                              save_dir=os.path.join(save_dir, "plots/confusion_matrix"))
    plot_expert_embedding_pca(model, dataloader, device,
                              save_dir=os.path.join(save_dir, "plots"),
                              filename="expert_embeddings_pca.png", figsize=(10, 10))
    plot_class_expert_heatmap(model, dataloader, device,
                              save_dir=os.path.join(save_dir, "plots"),
                              filename="class_expert_heatmap.png", figsize=(10, 10))

def readable_single(num):
                """Convert a number to a human-readable format with units."""
                for unit in ['','K','M','B','T']:
                    if abs(num) < 1000.0:
                        return f"{num:3.1f}{unit}"
                    num /= 1000.0
                return f"{num:.1f}P"

def print_model_params(model, readable, label="Model"):
                if hasattr(model, "encoder"):
                    common_params = sum(p.numel() for p in model.encoder.parameters())
                else:
                    common_params = 0
                if hasattr(model, "experts") and len(getattr(model, "experts", [])) > 0:
                    expert_params = sum(p.numel() for p in model.experts[0].parameters())
                    num_experts = len(model.experts)
                else:
                    expert_params = 0
                    num_experts = 0
                total_params = sum(p.numel() for p in model.parameters())
                # During inference: encoder + 1 expert
                inference_params = common_params + expert_params if num_experts > 0 else total_params
                print(f"{label}:")
                print(f"  Common encoder params: {readable(common_params)}")
                print(f"  One expert params:    {readable(expert_params)}")
                print(f"  Num experts:          {num_experts}")
                print(f"  Total params:         {readable(total_params)}")
                print(f"  Inference params:     {readable(inference_params)}")
                print()
                return inference_params, total_params


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 256
    num_classes = 10  # CIFAR-100 has 100 classes, CIFAR-10 has 10 classes
    # num_classes = 10  # CIFAR-10 has 10 classes, CIF
    num_experts = 8
    capacity = 64
    epochs = 150
    lr = 0.001
    run_dir = f"runs/collab_moe_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resume training from checkpoint
    train = True  # Set to True to train, False to plot from checkpoint
    resume_from_checkpoint = False
    starting_epoch = 150 if resume_from_checkpoint else 0
    checkpoint_path = "runs/collab_moe_experiment_20250626_130505/checkpoints/last.pth"

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data transforms
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
    
    # # CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # CIFAR-100 datasets
    # train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    # test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if train:
        if num_experts > 1:
            # Prepare model
            model = CollaborativeWaterfallMoE(num_experts=num_experts, capacity=capacity, num_classes=num_classes)
            if resume_from_checkpoint and os.path.isfile(checkpoint_path):
                print(f"Resuming from checkpoint: {checkpoint_path}")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            # Train MoE model

            print("Model parameter comparison:\n")
            # MoE model
            moe_infer_params, moe_total_params = print_model_params(model, readable_single, label="CollaborativeWaterfallMoE")
            # Compare with SingleImprovedModel
            single_model = SingleImprovedModel(num_classes=num_classes)
            single_infer_params, single_total_params = print_model_params(single_model, readable_single, label="SingleImprovedModel")

            # Proportion: how much smaller is the MoE during inference vs single model
            if single_infer_params > 0:
                proportion = moe_infer_params / single_infer_params
                print(f"CollaborativeWaterfallMoE inference params are {proportion:.2%} of SingleImprovedModel inference params")
                print(f"Or, {1/proportion:.2f}x smaller during inference.\n")
            else:
                print("Cannot compute proportion: SingleImprovedModel inference params is zero.\n")

            
            train_moe_waterfall(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                num_experts=num_experts,
                capacity=capacity,
                epochs=epochs,
                resume_from_checkpoint=resume_from_checkpoint,
                starting_epoch=starting_epoch,
                lr=lr,
                run_dir=run_dir,
            )
        else:
            # Prepare model
            model = SingleImprovedModel(num_classes=num_classes)
            if resume_from_checkpoint and os.path.isfile(checkpoint_path):
                print(f"Resuming from checkpoint: {checkpoint_path}")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters in the model: {total_params}")
            # Train single model
            train_single_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=epochs,
                use_amp=True,
                lr=0.001,
            )

        # Plot the trained single model on test data
        plot_from_checkpoint(
            checkpoint_path=os.path.join(run_dir, "checkpoints", "last.pth"),
            dataloader=test_loader,
            device=device,
            num_experts=num_experts,
            capacity=capacity,
            num_classes=num_classes
        )
        # Save training parameters to the run folder

        params = {
            "batch_size": batch_size,
            "num_experts": num_experts,
            "capacity": capacity,
            "epochs": epochs,
            "lr": lr,
            "resume_from_checkpoint": resume_from_checkpoint,
            "starting_epoch": starting_epoch,
            "checkpoint_path": checkpoint_path,
            "device": device,
            "train_transform": str(transform_train),
            "test_transform": str(transform_test),
        }

        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

    
    
    if checkpoint_path is not None and os.path.isfile(checkpoint_path) and not train:
        print(f"Plotting from checkpoint: {checkpoint_path}")
        plot_from_checkpoint(
            checkpoint_path=checkpoint_path,
            dataloader=test_loader,
            device=device,
            num_experts=num_experts,
            capacity=capacity,
            num_classes=10
        )




