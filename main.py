import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


from Vision_trainer import train_single_model
from Model import SingleModel
from Vision_trainer_MoE import train_moe_cnn
from MoE_2 import MoE

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 256
    num_experts = 8
    capacity = 64
    k = 3
    epochs = 150
    lr = 0.001
    
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
        # Train MoE model
        train_moe_cnn(
            model= MoE(num_experts=num_experts, capacity=capacity, k=k, num_classes=10),
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            num_experts=num_experts,
            capacity=capacity,
            k=k,
            epochs=epochs,
            lr=lr,
        )
    else:
        # Train single model
        train_single_model(
            model=SingleModel(),
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            use_amp=True,
            lr=0.001,
        )

    