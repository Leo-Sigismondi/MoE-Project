import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from metrics import *

from Vision_trainer import train_single_model
from Model import SingleModel
from Vision_trainer_MoE import train_moe_cnn
from MoE_5 import MoE
import json
from datetime import datetime


def plot_from_checkpoint(checkpoint_path, dataloader, device, num_experts, capacity, k, num_classes=10):
    # Load model
    model = MoE(num_experts=num_experts, capacity=capacity, k=k, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    # Plot
    plot_per_expert_confusion(model, dataloader, device, num_classes=num_classes, 
                              save_dir="runs/moe_experiment_20250527_214947/plots/confusion_matrix")


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 256
    num_experts = 8
    capacity = 64
    k = 4
    epochs = 150
    lr = 0.001
    run_dir = f"runs/moe_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resume training from checkpoint
    resume_from_checkpoint = False
    starting_epoch = 140
    checkpoint_path = "runs/moe_experiment_20250528_131108/checkpoints/epoch_140.pth"

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data transforms
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if num_experts > 1:
        # Prepare model
        model = MoE(num_experts=num_experts, capacity=capacity, k=k, num_classes=10)
        if resume_from_checkpoint and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        # Train MoE model
        train_moe_cnn(
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
        model = SingleModel()
        if resume_from_checkpoint and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    
    # Save training parameters to the run folder

    params = {
        "batch_size": batch_size,
        "num_experts": num_experts,
        "capacity": capacity,
        "k": k,
        "epochs": epochs,
        "lr": lr,
        "resume_from_checkpoint": resume_from_checkpoint,
        "starting_epoch": starting_epoch,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "train_transform": str(transform),
        "test_transform": str(transform_test),
    }

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    
        

    # plot_from_checkpoint(
    #     checkpoint_path="runs/moe_experiment_20250527_214947/checkpoints/epoch_150.pth",
    #     dataloader=test_loader,
    #     device=device,
    #     num_experts=num_experts,
    #     capacity=capacity,
    #     k=k,
    #     num_classes=10
    # )




