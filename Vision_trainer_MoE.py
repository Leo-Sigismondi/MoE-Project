import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from MoE_2 import MoE
from torch.utils.tensorboard import SummaryWriter
from metrics import *
from datetime import datetime
import time


def train_moe_cnn(
    model,
    train_loader,
    test_loader,
    device,
    num_experts=4,
    capacity=128,
    k=2,
    epochs=20,
    lr=0.001,
    log_interval=100,
):
    # Model, loss, optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Mixed precision scaler
    scaler = torch.GradScaler()

    # TensorBoard writer
    run_name = f"moe_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Training loop
    print("Starting training...")
    warmup_epochs = int(epochs * 0.05)
    for epoch in range(epochs):
        # epoch_start_time = time.time()
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Accumulators for MoE-specific metrics
        imp_sum = 0
        ld_sum = 0
        ent_sum = 0
        sld_sum = 0
        util_sum = 0
        ld_var_sum = 0
        top1_dist_sum = 0
        mi_sum = 0
        num_batches = 0

        # Dynamic entropy loss weight: negative at start, positive later
        if epoch < warmup_epochs:
            temp = 1.0
            entr_weight = -0.005
        else:
            temp = 1.0 + (epoch - warmup_epochs) / (epochs - warmup_epochs) * 0.3
            entr_weight = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
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

            # Backpropagation with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # --- Accumulate MoE-specific metrics ---
            imp_sum += imp.detach().cpu()
            ld_sum += ld.detach().cpu()
            util_sum += util.detach().cpu()
            ent_sum += ent.item()
            sld_sum += sld.item()
            ld_var_sum += ld_var.item()
            top1_dist_sum += top1_dist.detach().cpu()
            mi_sum += mi.item()
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # --- Log MoE-specific metrics per epoch ---
        avg_imp = imp_sum / num_batches
        avg_ld = ld_sum / num_batches
        avg_util = util_sum / num_batches
        avg_ent = ent_sum / num_batches
        avg_sld = sld_sum / num_batches
        avg_ld_var = ld_var_sum / num_batches
        avg_top1_dist = top1_dist_sum / num_batches
        avg_mi = mi_sum / num_batches

        # Ensure avg_imp, avg_ld, avg_util, avg_top1_dist are iterable (1D tensor or list), else wrap as list
        def ensure_iterable(x):
            if isinstance(x, torch.Tensor) and x.dim() == 0:
                return [x]
            elif isinstance(x, float) or isinstance(x, int):
                return [torch.tensor(x)]
            return x

        avg_imp_iter = ensure_iterable(avg_imp)
        avg_ld_iter = ensure_iterable(avg_ld)
        avg_util_iter = ensure_iterable(avg_util)
        avg_top1_dist_iter = ensure_iterable(avg_top1_dist)

        for j, v in enumerate(avg_imp_iter):
            writer.add_scalar(f"importance/expert_{j}", v.item(), epoch)
        for j, v in enumerate(avg_ld_iter):
            writer.add_scalar(f"load/expert_{j}", v.item(), epoch)
            writer.add_scalar(f"utilization/expert_{j}", avg_util_iter[j].item(), epoch)
        writer.add_scalar("gate_entropy", avg_ent, epoch)
        writer.add_scalar("sample_load", avg_sld, epoch)
        writer.add_scalar("load_variance", avg_ld_var, epoch)
        for j, v in enumerate(avg_top1_dist_iter):
            writer.add_scalar(f"top1_distribution/expert_{j}", v.item(), epoch)
        writer.add_scalar("mutual_information/top1", avg_mi, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast("cuda"):
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


        # Print epoch results with time taken
        # epoch_end_time = time.time()
        # epoch_time = epoch_end_time - epoch_start_time
        # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        #       f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s")
        
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


