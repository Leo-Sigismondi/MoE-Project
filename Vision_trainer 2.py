import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch.nn as nn
import torch.optim as optim
from MoE_2 import MoE
from torch.utils.tensorboard import SummaryWriter
from metrics import *
from datetime import datetime


def train_moe_cnn(
    num_experts=4,
    capacity=128,
    k=2,
    batch_size=256,
    epochs=20,
    lr=0.001,
    device=None
):
    # Device setup
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Model, loss, optimizer
    model = MoE(num_experts=num_experts, capacity=capacity, k=k, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TensorBoard writer
    run_name = f"moe_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Training loop
    print("Starting training...")
    step = 0
    warmup_epochs= int(epochs * 0.1)
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # Dynamic entropy loss weight: negative at start, positive later
        if epoch < warmup_epochs:
            temp = 1.0
            entr_weight = -0.005
        else:
            temp = 1.0 + (epoch - warmup_epochs) / (epochs - warmup_epochs) * 0.3
            entr_weight = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs, scores, D = model(inputs, temp=temp)  # [B, num_classes], [B, num_experts], [B, num_experts]

            # Compute importance, load, and entropy
            imp = compute_importance(scores)             # [E]
            ld  = compute_load(D)                        # [E]
            ent = compute_entropy(scores)                # scalar

            sld = compute_sample_load(D)                 # scalar
            util = compute_expert_utilization(D, capacity)      # [E]
            ld_var = compute_load_variance(D)            # scalar
            top1_dist = compute_top1_distribution(scores)  # [E]
            mi = compute_expert_class_mi_top1(scores, targets)  # scalar

            ce_loss = criterion(outputs, targets)
            aux_loss = 0.01 * num_experts * (imp * ld).sum() # Auxiliary loss for load balancing
            entr_loss = entr_weight * ent  # Negative entropy loss for regularization
            # Combine losses
            loss = ce_loss + aux_loss + entr_loss
            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

             # --- MoE-specific logging ---

            for j, v in enumerate(imp):
                writer.add_scalar(f"importance/expert_{j}", v.item(), step)
            for j, v in enumerate(ld):
                writer.add_scalar(f"load/expert_{j}", v.item(), step)
                writer.add_scalar(f"utilization/expert_{j}", util[j].item(), step)
            writer.add_scalar("gate_entropy", ent.item(), step)
            writer.add_scalar("sample_load", sld.item(), step)
            writer.add_scalar("load_variance", ld_var.item(), step)
            for j, v in enumerate(top1_dist):
                writer.add_scalar(f"top1_distribution/expert_{j}", v.item(), step)
            writer.add_scalar("mutual_information/top1", mi.item(), step)

            

            # --- End MoE-specific logging ---
            # you can also log accuracy, aux_loss, histograms, etc.
            step += 1

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, scores, dispatch_mask = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)


        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = f"checkpoints/{run_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{checkpoint_dir}/epoch_{epoch+1}.pth")
            print(f"Model checkpoint saved at epoch {epoch+1}")
    
    writer.close()
    print("Training complete.")

    plot_class_expert_heatmap(model, test_loader, device)
    plot_expert_embedding_pca(model, test_loader, device)

    

if __name__ == "__main__":
    train_moe_cnn(num_experts=4, capacity=128, k=2, batch_size=256, epochs=100, lr=0.001)

    