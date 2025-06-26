import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from metrics import *
from datetime import datetime
import numpy as np


def train_moe_waterfall(
    model,
    train_loader,
    test_loader,
    device,
    num_experts=4,
    capacity=128,
    epochs=20,
    starting_epoch=0,
    resume_from_checkpoint=False,
    lr=0.001,
    run_dir=None,
    log_interval=100,
):
    # Model, loss, optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Mixed precision scaler
    scaler = torch.GradScaler()

    # TensorBoard writer
    if run_dir is None:
        raise ValueError("run_dir must be provided")
    writer = SummaryWriter(run_dir)
    
    # Training loop
    print("Starting Waterfall MoE training at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    warmup_epochs = int(epochs * 0.05)
    temp0, temp1 = 2.0, 0.3

    for epoch in range(epochs):
        # If resuming, skip to the specified epoch
        if resume_from_checkpoint and epoch < starting_epoch:
            continue
            
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Accumulatori per metriche MoE standard
        imp_sum = 0
        ld_sum = 0
        ent_sum = 0
        sld_sum = 0
        util_sum = 0
        ld_var_sum = 0
        top1_dist_sum = 0
        mi_sum = 0
        
        # Accumulatori per metriche Waterfall specifiche
        waterfall_stats = {
            'load_balance_std': [],
            'expert_coverage': [],
            'total_assignments': [],
            'expert_loads_history': []
        }
        
        num_batches = 0

        # Annealing della temperatura
        frac = epoch / (epochs - 1)
        temp_t = temp0 + (temp1 - temp0) * frac

        # Dynamic entropy loss weight
        if epoch < warmup_epochs:
            entr_weight = -0.005
        else:
            entr_weight = 0.0

        loop = tqdm(train_loader, desc=f"Waterfall Training | Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda"):
                # Forward pass con Waterfall routing
                outputs, scores, D = model(inputs, temp=temp_t)

                # === Metriche MoE standard ===
                imp = compute_importance(scores)
                ld = compute_load(D)
                ent = compute_entropy(scores)
                sld = compute_sample_load(D)
                util = compute_expert_utilization(D, capacity)
                ld_var = compute_load_variance(D)
                top1_dist = compute_top1_distribution(scores)
                mi = compute_expert_class_mi_top1(scores, targets)

                # === Metriche Waterfall specifiche ===
                if hasattr(model, 'get_routing_stats'):
                    waterfall_metrics = model.get_routing_stats(D)
                    waterfall_stats['load_balance_std'].append(waterfall_metrics['load_balance_std'])
                    waterfall_stats['expert_coverage'].append(waterfall_metrics['expert_coverage'])
                    waterfall_stats['total_assignments'].append(waterfall_metrics['total_assignments'])
                    waterfall_stats['expert_loads_history'].append(waterfall_metrics['expert_loads'])

                # Loss computation
                ce_loss = criterion(outputs, targets)
                aux_loss = 0.01 * num_experts * (imp * ld).sum()
                entr_loss = entr_weight * ent
                
                # Waterfall-specific penalty per load balancing
                load_balance_penalty = 0.005 * ld_var  # Penalizza alta varianza nei carichi
                
                # Combine losses
                loss = ce_loss + aux_loss + entr_loss + load_balance_penalty
                
                if hasattr(model, "diversity_penalty") and callable(model.diversity_penalty):
                    penalty = model.diversity_penalty()
                    if isinstance(penalty, torch.Tensor):
                        loss = loss + penalty * 0.005

            # Backpropagation with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Accumula metriche standard
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
                # Mostra anche metriche Waterfall nel progress bar
                avg_coverage = np.mean(waterfall_stats['expert_coverage'][-10:]) if waterfall_stats['expert_coverage'] else 0
                avg_balance = np.mean(waterfall_stats['load_balance_std'][-10:]) if waterfall_stats['load_balance_std'] else 0
                loop.set_postfix(
                    loss=running_loss/total, 
                    acc=100.*correct/total,
                    coverage=f"{avg_coverage:.1f}/{num_experts}",
                    balance=f"{avg_balance:.3f}"
                )

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # === Log metriche MoE standard ===
        avg_imp = imp_sum / num_batches
        avg_ld = ld_sum / num_batches
        avg_util = util_sum / num_batches
        avg_ent = ent_sum / num_batches
        avg_sld = sld_sum / num_batches
        avg_ld_var = ld_var_sum / num_batches
        avg_top1_dist = top1_dist_sum / num_batches
        avg_mi = mi_sum / num_batches

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

        # Log metriche per esperto
        for j, v in enumerate(avg_imp_iter):
            writer.add_scalar(f"importance/expert_{j}", v.item(), epoch)
        for j, v in enumerate(avg_ld_iter):
            writer.add_scalar(f"load/expert_{j}", v.item(), epoch)
            writer.add_scalar(f"utilization/expert_{j}", avg_util_iter[j].item(), epoch)
        for j, v in enumerate(avg_top1_dist_iter):
            writer.add_scalar(f"top1_distribution/expert_{j}", v.item(), epoch)

        # Log metriche scalari
        writer.add_scalar("gate_entropy", avg_ent, epoch)
        writer.add_scalar("sample_load", avg_sld, epoch)
        writer.add_scalar("load_variance", avg_ld_var, epoch)
        writer.add_scalar("mutual_information/top1", avg_mi, epoch)

        # === Log metriche Waterfall specifiche ===
        if waterfall_stats['load_balance_std']:
            avg_waterfall_balance = np.mean(waterfall_stats['load_balance_std'])
            avg_waterfall_coverage = np.mean(waterfall_stats['expert_coverage'])
            avg_waterfall_assignments = np.mean(waterfall_stats['total_assignments'])
            
            writer.add_scalar("waterfall/load_balance_std", avg_waterfall_balance, epoch)
            writer.add_scalar("waterfall/expert_coverage", avg_waterfall_coverage, epoch)
            writer.add_scalar("waterfall/total_assignments", avg_waterfall_assignments, epoch)
            writer.add_scalar("waterfall/coverage_ratio", avg_waterfall_coverage / num_experts, epoch)
            
            # Distribuzione dei carichi per esperto (heatmap style)
            if waterfall_stats['expert_loads_history']:
                final_loads = waterfall_stats['expert_loads_history'][-1]
                for j, load in enumerate(final_loads):
                    writer.add_scalar(f"waterfall/expert_{j}_final_load", load, epoch)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Metriche di validazione per Waterfall
        val_waterfall_stats = {
            'load_balance_std': [],
            'expert_coverage': [],
            'total_assignments': []
        }
        
        with torch.no_grad():
            val_loop = tqdm(test_loader, desc=f"Validation | Epoch {epoch+1}/{epochs}", leave=False)
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast("cuda"):
                    outputs, scores, dispatch_mask = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Raccogli stats di validazione
                    if hasattr(model, 'get_routing_stats'):
                        val_metrics = model.get_routing_stats(dispatch_mask)
                        val_waterfall_stats['load_balance_std'].append(val_metrics['load_balance_std'])
                        val_waterfall_stats['expert_coverage'].append(val_metrics['expert_coverage'])
                        val_waterfall_stats['total_assignments'].append(val_metrics['total_assignments'])
                        
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Progress bar con metriche Waterfall
                if val_waterfall_stats['expert_coverage']:
                    val_coverage = np.mean(val_waterfall_stats['expert_coverage'][-10:])
                    val_balance = np.mean(val_waterfall_stats['load_balance_std'][-10:])
                    val_loop.set_postfix(
                        val_loss=val_loss/val_total if val_total > 0 else 0, 
                        val_acc=100.*val_correct/val_total if val_total > 0 else 0,
                        val_cov=f"{val_coverage:.1f}/{num_experts}",
                        val_bal=f"{val_balance:.3f}"
                    )
                else:
                    val_loop.set_postfix(
                        val_loss=val_loss/val_total if val_total > 0 else 0, 
                        val_acc=100.*val_correct/val_total if val_total > 0 else 0
                    )
                    
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total

        # Log validation metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Log validation Waterfall metrics
        if val_waterfall_stats['load_balance_std']:
            writer.add_scalar("waterfall_val/load_balance_std", np.mean(val_waterfall_stats['load_balance_std']), epoch)
            writer.add_scalar("waterfall_val/expert_coverage", np.mean(val_waterfall_stats['expert_coverage']), epoch)
            writer.add_scalar("waterfall_val/coverage_ratio", np.mean(val_waterfall_stats['expert_coverage']) / num_experts, epoch)

        # Stampa riassunto epoch con metriche Waterfall
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Expert Coverage: {avg_waterfall_coverage:.1f}/{num_experts} | "
              f"Load Balance: {avg_waterfall_balance:.3f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = f"{run_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{checkpoint_dir}/epoch_{epoch+1}.pth")
    
    writer.close()
    print("Waterfall MoE training completed at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Generate comprehensive plots
    plot_dir = f"{run_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_class_expert_heatmap(model, test_loader, device, save_dir=plot_dir)
    plot_expert_embedding_pca(model, test_loader, device, save_dir=plot_dir)
    plot_per_expert_confusion(model, test_loader, device, save_dir=os.path.join(plot_dir, "confusion_matrix"))
    
    # Plot Waterfall-specific visualizations
    plot_waterfall_specific_metrics(model, test_loader, device, save_dir=plot_dir)


def plot_waterfall_specific_metrics(model, dataloader, device, save_dir="plots"):
    """
    Crea visualizzazioni specifiche per l'algoritmo Waterfall
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()
    all_stats = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            _, _, D = model(x)
            if hasattr(model, 'get_routing_stats'):
                stats = model.get_routing_stats(D)
                all_stats.append(stats)
    
    if not all_stats:
        return
    
    # Analisi distribuzione carichi
    expert_loads_matrix = np.array([stats['expert_loads'] for stats in all_stats])
    
    # Plot 1: Distribuzione carichi per batch
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(expert_loads_matrix.T, cmap='viridis', 
                xticklabels=False, yticklabels=[f'Expert {i}' for i in range(model.num_experts)])
    plt.title('Expert Load Distribution Across Batches')
    plt.xlabel('Batch')
    plt.ylabel('Expert')
    
    # Plot 2: Load balance statistics
    plt.subplot(1, 2, 2)
    load_balance_stds = [stats['load_balance_std'] for stats in all_stats]
    expert_coverages = [stats['expert_coverage'] for stats in all_stats]
    
    plt.scatter(load_balance_stds, expert_coverages, alpha=0.6)
    plt.xlabel('Load Balance Std Dev')
    plt.ylabel('Number of Active Experts')
    plt.title('Load Balance vs Expert Coverage')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "waterfall_routing_analysis.png"))
    plt.close()
    
    # Plot 3: Expert utilization histogram
    plt.figure(figsize=(10, 6))
    mean_loads = expert_loads_matrix.mean(axis=0)
    std_loads = expert_loads_matrix.std(axis=0)
    
    x = range(len(mean_loads))
    plt.bar(x, mean_loads, yerr=std_loads, capsize=5, alpha=0.7)
    plt.xlabel('Expert ID')
    plt.ylabel('Average Load')
    plt.title('Average Expert Utilization with Standard Deviation')
    plt.xticks(x, [f'E{i}' for i in x])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "expert_utilization_waterfall.png"))
    plt.close()
    
    print(f"Waterfall-specific plots saved to {save_dir}")