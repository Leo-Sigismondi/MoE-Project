import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from metrics import *
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def train_moe_waterfall_improved(
    model,
    train_loader,
    test_loader,
    device,
    epochs=150,
    starting_epoch=0,
    resume_from_checkpoint=False,
    base_lr=0.001,
    run_dir=None,
    log_interval=100,
    advanced_config=None,
):
    """
    Improved training function with better optimization and monitoring
    """
    if run_dir is None:
        raise ValueError("run_dir must be provided")
    
    if advanced_config is None:
        advanced_config = {}

    # ===== MODEL AND OPTIMIZER SETUP =====
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    
    # Different learning rates for different components
    trunk_params = []
    expert_params = []
    gate_params = []
    
    for name, param in model.named_parameters():
        if 'trunk' in name:
            trunk_params.append(param)
        elif 'expert_gates' in name:
            gate_params.append(param)
        else:
            expert_params.append(param)
    
    # Create optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': trunk_params, 'lr': base_lr, 'weight_decay': 1e-4},
        {'params': expert_params, 'lr': base_lr * 0.5, 'weight_decay': 1e-4},
        {'params': gate_params, 'lr': base_lr * 0.1, 'weight_decay': 1e-5},
    ], eps=1e-8)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # TensorBoard writer
    writer = SummaryWriter(run_dir)
    
    # ===== TRAINING CONFIGURATION =====
    print("Starting Improved Waterfall MoE training at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Temperature annealing
    temp_start = 2.0
    temp_end = 0.5
    warmup_epochs = max(1, int(epochs * 0.1))
    
    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0
    patience = 20
    no_improve_count = 0

    # ===== TRAINING LOOP =====
    for epoch in range(epochs):
        # Skip to starting epoch if resuming
        if resume_from_checkpoint and epoch < starting_epoch:
            continue
        
        # Temperature annealing
        if epoch < warmup_epochs:
            temp = temp_start
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            temp = temp_start + (temp_end - temp_start) * progress
        
        # ===== TRAINING PHASE =====
        model.train()
        train_metrics = {
            'loss': 0.0, 'ce_loss': 0.0, 'aux_loss': 0.0, 'balance_loss': 0.0,
            'correct': 0, 'total': 0,
            'routing_stats': []
        }
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(train_loop):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs, routing_scores, D = model(inputs, temp=temp)
                
                # Compute losses
                ce_loss = criterion(outputs, targets)
                
                # MoE auxiliary losses
                importance = compute_importance(routing_scores)
                load = compute_load(D)
                aux_loss = 0.01 * model.num_experts * (importance * load).sum()
                
                # Load balancing loss (improved)
                balance_loss = model.compute_load_balance_loss(D)
                
                # Total loss
                loss = ce_loss + aux_loss + balance_loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            batch_size = inputs.size(0)
            train_metrics['loss'] += loss.item() * batch_size
            train_metrics['ce_loss'] += ce_loss.item() * batch_size
            train_metrics['aux_loss'] += aux_loss.item() * batch_size
            train_metrics['balance_loss'] += balance_loss.item() * batch_size
            train_metrics['total'] += batch_size
            
            _, predicted = outputs.max(1)
            train_metrics['correct'] += predicted.eq(targets).sum().item()
            
            # Collect routing statistics
            if hasattr(model, 'get_routing_stats'):
                routing_stats = model.get_routing_stats(D)
                train_metrics['routing_stats'].append(routing_stats)
            
            # Update progress bar
            if (batch_idx + 1) % log_interval == 0:
                current_acc = 100. * train_metrics['correct'] / train_metrics['total']
                current_loss = train_metrics['loss'] / train_metrics['total']
                recent_stats = train_metrics['routing_stats'][-10:] if train_metrics['routing_stats'] else []
                avg_coverage = np.mean([s['expert_coverage'] for s in recent_stats]) if recent_stats else 0
                avg_balance = np.mean([s['load_balance_std'] for s in recent_stats]) if recent_stats else 0
                
                train_loop.set_postfix({
                    'Loss': f"{current_loss:.4f}",
                    'Acc': f"{current_acc:.2f}%",
                    'Cov': f"{avg_coverage:.1f}/{model.num_experts}",
                    'Bal': f"{avg_balance:.3f}",
                    'Temp': f"{temp:.2f}"
                })
        
        # ===== VALIDATION PHASE =====
        model.eval()
        val_metrics = {
            'loss': 0.0, 'correct': 0, 'total': 0,
            'routing_stats': []
        }
        
        with torch.no_grad():
            val_loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs, routing_scores, D = model(inputs, temp=1.0)  # No temperature during validation
                    loss = criterion(outputs, targets)
                
                batch_size = inputs.size(0)
                val_metrics['loss'] += loss.item() * batch_size
                val_metrics['total'] += batch_size
                
                _, predicted = outputs.max(1)
                val_metrics['correct'] += predicted.eq(targets).sum().item()
                
                # Collect validation routing stats
                if hasattr(model, 'get_routing_stats'):
                    routing_stats = model.get_routing_stats(D)
                    val_metrics['routing_stats'].append(routing_stats)
                
                # Update progress bar
                current_val_acc = 100. * val_metrics['correct'] / val_metrics['total']
                current_val_loss = val_metrics['loss'] / val_metrics['total']
                val_loop.set_postfix({
                    'Val Loss': f"{current_val_loss:.4f}",
                    'Val Acc': f"{current_val_acc:.2f}%"
                })
        
        # ===== COMPUTE EPOCH METRICS =====
        train_loss = train_metrics['loss'] / train_metrics['total']
        train_acc = 100. * train_metrics['correct'] / train_metrics['total']
        val_loss = val_metrics['loss'] / val_metrics['total']
        val_acc = 100. * val_metrics['correct'] / val_metrics['total']
        
        # Average routing statistics
        if train_metrics['routing_stats']:
            avg_train_routing = average_routing_stats(train_metrics['routing_stats'])
            avg_val_routing = average_routing_stats(val_metrics['routing_stats'])
        else:
            avg_train_routing = avg_val_routing = {}
        
        # ===== LOGGING =====
        # Basic metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Temperature', temp, epoch)
        
        # Loss components
        writer.add_scalar('Loss_Components/CrossEntropy', train_metrics['ce_loss'] / train_metrics['total'], epoch)
        writer.add_scalar('Loss_Components/Auxiliary', train_metrics['aux_loss'] / train_metrics['total'], epoch)
        writer.add_scalar('Loss_Components/Balance', train_metrics['balance_loss'] / train_metrics['total'], epoch)
        
        # Routing statistics
        log_routing_stats(writer, avg_train_routing, epoch, prefix='Train_Routing')
        log_routing_stats(writer, avg_val_routing, epoch, prefix='Val_Routing')
        
        # ===== PRINT EPOCH SUMMARY =====
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        if avg_train_routing:
            print(f"  Routing: Coverage={avg_train_routing.get('expert_coverage', 0):.1f}/{model.num_experts}, "
                  f"Balance={avg_train_routing.get('load_balance_std', 0):.3f}, "
                  f"Gini={avg_train_routing.get('gini_coefficient', 0):.3f}")
        print(f"  LR={optimizer.param_groups[0]['lr']:.6f}, Temp={temp:.2f}")
        
        # ===== MODEL CHECKPOINTING =====
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0
            
            # Save best model
            best_model_path = os.path.join(run_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': advanced_config,
            }, best_model_path)
            print(f"  🎉 New best validation accuracy: {best_val_acc:.2f}%")
        else:
            no_improve_count += 1
        
        # Regular checkpoints
        if (epoch + 1) % 20 == 0:
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': advanced_config,
            }, checkpoint_path)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch + 1}")
            break
    
    # ===== TRAINING COMPLETION =====
    writer.close()
    print("Training completed at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch + 1}")
    
    # Generate final plots
    print("Generating analysis plots...")
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load best model for analysis
    best_checkpoint = torch.load(os.path.join(run_dir, "best_model.pth"))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Generate comprehensive analysis
    generate_final_analysis(model, test_loader, device, plot_dir)
    
    return model

def average_routing_stats(stats_list):
    """Average routing statistics across batches"""
    if not stats_list:
        return {}
    
    averaged = {}
    for key in stats_list[0].keys():
        if key in ['expert_loads', 'tokens_per_expert', 'usage_ema']:
            # For arrays, average across batches
            averaged[key] = np.mean([stats[key] for stats in stats_list], axis=0)
        else:
            # For scalars, simple average
            averaged[key] = np.mean([stats[key] for stats in stats_list])
    
    return averaged

def log_routing_stats(writer, routing_stats, epoch, prefix="Routing"):
    """Log routing statistics to tensorboard"""
    if not routing_stats:
        return
    
    for key, value in routing_stats.items():
        if key in ['expert_loads', 'tokens_per_expert', 'usage_ema']:
            # Log per-expert values
            for i, v in enumerate(value):
                writer.add_scalar(f"{prefix}/{key}/expert_{i}", v, epoch)
        else:
            # Log scalar values
            writer.add_scalar(f"{prefix}/{key}", value, epoch)

def generate_final_analysis(model, test_loader, device, save_dir):
    """Generate comprehensive analysis plots"""
    from metrics import (plot_class_expert_heatmap, plot_expert_embedding_pca, 
                        plot_per_expert_confusion)
    
    try:
        plot_class_expert_heatmap(model, test_loader, device, save_dir=save_dir)
        plot_expert_embedding_pca(model, test_loader, device, save_dir=save_dir)
        plot_per_expert_confusion(model, test_loader, device, save_dir=os.path.join(save_dir, "confusion_matrix"))
        
        # Additional improved plots
        plot_routing_analysis_improved(model, test_loader, device, save_dir)
        print(f"Analysis plots saved to {save_dir}")
    except Exception as e:
        print(f"Error generating plots: {e}")

def plot_routing_analysis_improved(model, test_loader, device, save_dir):
    """Generate improved routing analysis plots"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()
    all_routing_stats = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, _, D = model(x)
            if hasattr(model, 'get_routing_stats'):
                stats = model.get_routing_stats(D)
                all_routing_stats.append(stats)
    
    if not all_routing_stats:
        return
    
    # Create comprehensive routing analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Improved MoE Routing Analysis', fontsize=16)
    
    # 1. Expert load distribution
    expert_loads = np.array([stats['expert_loads'] for stats in all_routing_stats])
    axes[0, 0].boxplot([expert_loads[:, i] for i in range(model.num_experts)], 
                       labels=[f'E{i}' for i in range(model.num_experts)])
    axes[0, 0].set_title('Expert Load Distribution')
    axes[0, 0].set_ylabel('Load')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Load balance over time
    load_balance_stds = [stats['load_balance_std'] for stats in all_routing_stats]
    axes[0, 1].plot(load_balance_stds)
    axes[0, 1].set_title('Load Balance Std Dev Over Batches')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Expert coverage
    coverages = [stats['expert_coverage'] for stats in all_routing_stats]
    coverage_ratios = [c / model.num_experts for c in coverages]
    axes[0, 2].hist(coverage_ratios, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Expert Coverage Distribution')
    axes[0, 2].set_xlabel('Coverage Ratio')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(coverage_ratios), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(coverage_ratios):.2f}')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Capacity utilization
    utilizations = [stats['capacity_utilization'] for stats in all_routing_stats]
    axes[1, 0].plot(utilizations)
    axes[1, 0].set_title('Capacity Utilization Over Batches')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Utilization')
    axes[1, 0].axhline(np.mean(utilizations), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(utilizations):.2f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Gini coefficient (inequality measure)
    gini_coeffs = [stats['gini_coefficient'] for stats in all_routing_stats]
    axes[1, 1].hist(gini_coeffs, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Load Distribution Inequality (Gini)')
    axes[1, 1].set_xlabel('Gini Coefficient')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(gini_coeffs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(gini_coeffs):.3f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Expert usage heatmap
    mean_loads = expert_loads.mean(axis=0)
    std_loads = expert_loads.std(axis=0)
    usage_data = np.column_stack([mean_loads, std_loads])
    
    sns.heatmap(usage_data.T, annot=True, fmt='.2f', 
                xticklabels=[f'E{i}' for i in range(model.num_experts)],
                yticklabels=['Mean Load', 'Std Load'], 
                ax=axes[1, 2], cmap='viridis')
    axes[1, 2].set_title('Expert Usage Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improved_routing_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    print("\n=== ROUTING ANALYSIS SUMMARY ===")
    print(f"Average Expert Coverage: {np.mean(coverages):.1f}/{model.num_experts} ({np.mean(coverage_ratios):.1%})")
    print(f"Average Load Balance Std: {np.mean(load_balance_stds):.3f}")
    print(f"Average Capacity Utilization: {np.mean(utilizations):.1%}")
    print(f"Average Gini Coefficient: {np.mean(gini_coeffs):.3f}")
    print(f"Load Balance Quality: {'Good' if np.mean(load_balance_stds) < 0.5 else 'Needs Improvement'}")
    print(f"Expert Utilization: {'Balanced' if np.mean(gini_coeffs) < 0.3 else 'Unbalanced'}")
