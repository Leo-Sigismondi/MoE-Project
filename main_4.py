import argparse
import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from metrics import *
from Vision_trainer import train_single_model
from Model import SingleImprovedModel
from collaborative_waterfall_moe_2 import CollaborativeWaterfallMoE
from collaborative_trainer import train_moe_waterfall

# --- Utility Functions ---

def readable_single(num):
    """Convert a number to a human-readable format with units."""
    for unit in ['', 'K', 'M', 'B', 'T']:
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
    inference_params = common_params + expert_params if num_experts > 0 else total_params
    print(f"{label}:")
    print(f"  Common encoder params: {readable(common_params)}")
    print(f"  One expert params:    {readable(expert_params)}")
    print(f"  Num experts:          {num_experts}")
    print(f"  Total params:         {readable(total_params)}")
    print(f"  Inference params:     {readable(inference_params)}")
    print()
    return inference_params, total_params

def plot_from_checkpoint(checkpoint_path, dataloader, device, num_experts, capacity, num_classes=10, hidden_channels=(64, 128, 16)):
    model = CollaborativeWaterfallMoE(num_experts=num_experts, capacity=capacity, num_classes=num_classes, 
                                      hidden_channels=hidden_channels)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    save_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    plot_per_expert_confusion(model, dataloader, device, num_classes=num_classes,
                              save_dir=os.path.join(save_dir, "plots/confusion_matrix"))
    plot_expert_embedding_pca(model, dataloader, device,
                              save_dir=os.path.join(save_dir, "plots"),
                              filename="expert_embeddings_pca.png", figsize=(10, 10))
    plot_class_expert_heatmap(model, dataloader, device,
                              save_dir=os.path.join(save_dir, "plots"),
                              filename="class_expert_heatmap.png", figsize=(10, 10))
    plot_expert_embedding_pca_3d(model, dataloader, device,
                                 save_dir=os.path.join(save_dir, "plots"),
                                 filename="expert_embeddings_pca_3d.png", figsize=(10, 10))

# --- Main Logic ---

def get_data_loaders(batch_size, num_classes):
    """Returns train and test dataloaders for CIFAR-10 or CIFAR-100."""
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
    if num_classes == 10:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif num_classes == 100:
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError("num_classes must be 10 (CIFAR-10) or 100 (CIFAR-100)")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, transform_train, transform_test

def main():
    parser = argparse.ArgumentParser(description="Collaborative MoE Training and Evaluation")
    parser.add_argument('--train', action='store_true', help="Run training")
    parser.add_argument('--plot', action='store_true', help="Plot from checkpoint")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to model checkpoint")
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--num_experts', type=int, default=4, help="Number of experts (MoE)")
    parser.add_argument('--capacity', type=int, default=128, help="Capacity per expert")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes (10 for CIFAR-10, 100 for CIFAR-100)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--hidden_channels', type=int, default=(64, 128, 256), help="Number of hidden channels in the model")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = f"runs/collab_moe_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_loader, test_loader, transform_train, transform_test = get_data_loaders(args.batch_size, args.num_classes)

    if args.train:
        # --- TRAINING ---
        if args.num_experts > 1:
            model = CollaborativeWaterfallMoE(num_experts=args.num_experts, capacity=args.capacity, num_classes=args.num_classes, 
                                              hidden_channels=args.hidden_channels)
            starting_epoch = args.epochs if args.resume else 0

            if args.resume and args.checkpoint_path and os.path.isfile(args.checkpoint_path):
                print(f"Resuming from checkpoint: {args.checkpoint_path}")
                model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            print("Model parameter comparison:\n")
            moe_infer_params, moe_total_params = print_model_params(model, readable_single, label="CollaborativeWaterfallMoE")
            single_model = SingleImprovedModel(num_classes=args.num_classes)
            single_infer_params, single_total_params = print_model_params(single_model, readable_single, label="SingleImprovedModel")
            if single_infer_params > 0:
                proportion = moe_infer_params / single_infer_params
                print(f"CollaborativeWaterfallMoE inference params are {proportion:.2%} of SingleImprovedModel inference params")
                print(f"Or, {1/proportion:.2f}x smaller during inference.\n")
            train_moe_waterfall(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                num_experts=args.num_experts,
                
                capacity=args.capacity,
                epochs=args.epochs,
                resume_from_checkpoint=args.resume,
                starting_epoch=starting_epoch,
                lr=args.lr,
                run_dir=run_dir,
            )
        else:
            model = SingleImprovedModel(num_classes=args.num_classes)
            if args.resume and args.checkpoint_path and os.path.isfile(args.checkpoint_path):
                print(f"Resuming from checkpoint: {args.checkpoint_path}")
                model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters in the model: {total_params}")
            train_single_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                use_amp=True,
                lr=args.lr,
            )
        # Plot the trained model (for quick results after training)
        last_ckpt = os.path.join(run_dir, "checkpoints", "last.pth")
        plot_from_checkpoint(
            checkpoint_path=last_ckpt,
            dataloader=test_loader,
            device=device,
            num_experts=args.num_experts,
            capacity=args.capacity,
            num_classes=args.num_classes,
            hidden_channels=args.hidden_channels
        )
        # Save training parameters
        params = {
            "batch_size": args.batch_size,
            "num_experts": args.num_experts,
            "num_classes": args.num_classes,
            "hidden_channels": args.hidden_channels,
            "capacity": args.capacity,
            "epochs": args.epochs,
            "lr": args.lr,
            "resume_from_checkpoint": args.resume,
            "checkpoint_path": args.checkpoint_path,
            "device": device,
            "train_transform": str(transform_train),
            "test_transform": str(transform_test),
        }
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

    if args.plot and args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        print(f"Plotting from checkpoint: {args.checkpoint_path}")
        plot_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            dataloader=test_loader,
            device=device,
            num_experts=args.num_experts,
            # capacity=args.batch_size,
            capacity=args.capacity,
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes
        )
        params = {
                "batch_size": args.batch_size,
                "num_experts": args.num_experts,
                "num_classes": args.num_classes,
                "hidden_channels": args.hidden_channels,
                "capacity": args.capacity,
                "epochs": args.epochs,
                "lr": args.lr,
                "resume_from_checkpoint": args.resume,
                "checkpoint_path": args.checkpoint_path,
                "device": device,
                "train_transform": str(transform_train),
                "test_transform": str(transform_test),
        }
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

if __name__ == "__main__":
    main()
