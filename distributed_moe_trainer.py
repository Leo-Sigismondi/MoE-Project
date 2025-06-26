import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math

def train_distributed_moe(
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
    Training function for Distributed MoE with individual expert feature extractors
    """
    if run_dir is None:
        raise ValueError("run_dir must be provided")
    
    if advanced_config is None:
        advanced_config = {}

    # ===== MODEL AND OPTIMIZER SETUP =====
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Different learning rates for different components
    routing_params = []
    gate_params = []
    feature_extractor_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'routing_trunk' in name:
            routing_params.append(param)
        elif 'gate' in name:
            gate_params.append(param)
        elif 'expert_feature_extractors' in name:
            feature_extractor_params.append(param)
        elif 'expert_classifiers' in name:
            classifier_params.append(param)
        else:
            routing_params.append(param)  # Default to routing params
    
    # Create optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': routing_params, 'lr': base_lr, 'weight_decay': 1e-4},
        {'params': gate_params, 'lr': base_lr * 0.1, 'weight_decay': 1e-5},
        {'params': feature_extractor_params, 'lr': base_lr * 0.8, 'weight_decay': 1e-4},
        {'params': classifier_params, 'lr': base_lr * 0.5, 'weight_decay': 1e-4},
    ], eps=1e-8)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # TensorBoard writer
    writer = SummaryWriter(run_dir)
    
    # ===== TRAINING CONFIGURATION =====
    print("Starting Distributed MoE training at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    param_counts = model.count_parameters()
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Routing trunk: {param_counts['routing_trunk']:,}")
    print(f"Feature extractors: {param_counts['feature_extractors']:,}")
    print(f"Classifiers: {param_counts['classifiers']:,}")
    print(f"Average per expert: {param_counts['per_expert']:,.0f}")
    
    # Temperature annealing
    temp_start = advanced_config.get('temp_start', 2.0)
    temp_end = advanced_config.get('temp_end', 0.5)
    warmup_epochs = max(1, int(epochs * 0.1))
    
    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0
    patience = advanced_config.get('patience', 25)
    no_improve_count = 0

    # Stage definitions
    stage0_epochs = 5
    stage1_epochs = 30
    stage2_epochs = 10
    total_epochs = stage0_epochs + stage1_epochs + stage2_epochs
    assert epochs >= total_epochs, "Not enough epochs for all stages!"

    # Initial values
    lambda_aux = 0.01  # Stage 0
    lambda_ortho = 0.01
    balance_loss_weight = 0.01
    routing_temperature = 1.0

    # ===== TRAINING LOOP =====
    for epoch in range(epochs):
        # Skip to starting epoch if resuming
        if resume_from_checkpoint and epoch < starting_epoch:
            continue
        
        # # Stage selection
        # if epoch < stage0_epochs:
        #     # Stage 0: Uniform routing, no balance loss, λ_aux = 0.5
        #     model.uniform_routing = True  # You need to implement this flag in your model's forward
        #     balance_loss_weight = 0.0
        #     lambda_aux = 0.5
        #     lambda_ortho = 0.0
        #     routing_temperature = 1.0
        #     model.unfreeze_gate()
        # elif epoch < stage0_epochs + stage1_epochs:
        #     # Stage 1: Enable top-k, cosine anneal balance loss, anneal routing temp, λ_aux ≈ 0.2, λ_ortho ≈ 0.01
        #     model.uniform_routing = False
        #     # Cosine anneal balance_loss_weight from 0.5 to 0.05
        #     progress = (epoch - stage0_epochs) / stage1_epochs
        #     balance_loss_weight = 0.05 + 0.5 * 0.5 * (1 + math.cos(math.pi * progress))
        #     lambda_aux = 0.2
        #     lambda_ortho = 0.01
        #     # Linear anneal routing_temperature from 2.0 to 0.2
        #     routing_temperature = 2.0 - (1.8 * progress)
        #     model.unfreeze_gate()
        # else:
        #     # Stage 2: Freeze gate, train only experts with CE
        #     model.freeze_gate()
        #     balance_loss_weight = 0.0
        #     lambda_aux = 0.0
        #     lambda_ortho = 0.0
        #     routing_temperature = 0.2
        #     model.uniform_routing = False

        # Set model parameters for this epoch
        # model.balance_loss_weight = balance_loss_weight
        # # model.routing_temperature = routing_temperature
        # model.lambda_aux = lambda_aux
        # model.lambda_ortho = lambda_ortho

        # Temperature annealing
        if epoch < warmup_epochs:
            temp = temp_start
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            temp = temp_start + (temp_end - temp_start) * progress
        
        # ===== TRAINING PHASE =====
        model.train()
        train_metrics = {
            'loss': 0.0, 'ce_loss': 0.0, 'balance_loss': 0.0, 'aux_loss': 0.0, 'ortho_loss': 0.0,
            'correct': 0, 'total': 0,
            'routing_stats': []
        }
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(train_loop):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs, routing_scores, D = model(inputs, temp=temp, uniform_routing=getattr(model, 'uniform_routing', False))
                
                # Compute losses
                ce_loss = criterion(outputs, targets)
                total_loss = ce_loss
                balance_loss = model.compute_load_balance_loss(D) if balance_loss_weight > 0 else 0.0
                aux_loss = model.compute_auxiliary_loss(routing_scores) if lambda_aux > 0 else 0.0
                ortho_loss = model.compute_orthogonality_loss(routing_scores) if lambda_ortho > 0 else 0.0
                total_loss += (balance_loss_weight * balance_loss) + (lambda_aux * aux_loss) + (lambda_ortho * ortho_loss)

            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            batch_size = inputs.size(0)
            train_metrics['loss'] += total_loss.item() * batch_size
            train_metrics['ce_loss'] += ce_loss.item() * batch_size
            train_metrics['balance_loss'] += balance_loss.item() * batch_size if balance_loss_weight > 0 else 0
            train_metrics['aux_loss'] += aux_loss.item() * batch_size if lambda_aux > 0 else 0
            train_metrics['ortho_loss'] += ortho_loss.item() * batch_size if lambda_ortho > 0 else 0
            train_metrics['total'] += batch_size
            
            _, predicted = outputs.max(1)
            train_metrics['correct'] += predicted.eq(targets).sum().item()
            
            # Collect routing statistics
            routing_stats = model.get_routing_stats(D)
            train_metrics['routing_stats'].append(routing_stats)
            
            # Update progress bar
            if (batch_idx + 1) % log_interval == 0:
                current_acc = 100. * train_metrics['correct'] / train_metrics['total']
                current_loss = train_metrics['loss'] / train_metrics['total']
                recent_stats = train_metrics['routing_stats'][-10:]
                avg_coverage = np.mean([s['expert_coverage'] for s in recent_stats])
                avg_coverage_ratio = np.mean([s['coverage_ratio'] for s in recent_stats])
                
                train_loop.set_postfix({
                    'Loss': f"{current_loss:.4f}",
                    'Acc': f"{current_acc:.2f}%",
                    'Cov': f"{avg_coverage:.1f}/{model.num_experts}",
                    'CovRat': f"{avg_coverage_ratio:.2f}",
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
                    outputs, routing_scores, D = model(inputs, temp=1.0)
                    loss = criterion(outputs, targets)
                
                batch_size = inputs.size(0)
                val_metrics['loss'] += loss.item() * batch_size
                val_metrics['total'] += batch_size
                
                _, predicted = outputs.max(1)
                val_metrics['correct'] += predicted.eq(targets).sum().item()
                
                # Collect validation routing stats
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
        avg_train_routing = average_routing_stats(train_metrics['routing_stats'])
        avg_val_routing = average_routing_stats(val_metrics['routing_stats'])
        
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
        writer.add_scalar('Loss_Components/Balance', train_metrics['balance_loss'] / train_metrics['total'], epoch)
        writer.add_scalar('Loss_Components/Auxiliary', train_metrics['aux_loss'] / train_metrics['total'], epoch)
        writer.add_scalar('Loss_Components/Orthogonality', train_metrics['ortho_loss'] / train_metrics['total'], epoch)
        
        # Routing statistics
        log_routing_stats(writer, avg_train_routing, epoch, prefix='Train_Routing')
        log_routing_stats(writer, avg_val_routing, epoch, prefix='Val_Routing')
        
        # ===== PRINT EPOCH SUMMARY =====
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Routing: Coverage={avg_train_routing.get('expert_coverage', 0):.1f}/{model.num_experts} "
              f"({avg_train_routing.get('coverage_ratio', 0):.1%})")
        print(f"  Expert Usage EMA: {avg_train_routing.get('usage_ema', [])}")
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
    
    # Generate analysis
    generate_distributed_moe_analysis(model, test_loader, device, plot_dir)
    
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
            # Also log statistics
            writer.add_scalar(f"{prefix}/{key}_mean", np.mean(value), epoch)
            writer.add_scalar(f"{prefix}/{key}_std", np.std(value), epoch)
        else:
            # Log scalar values
            writer.add_scalar(f"{prefix}/{key}", value, epoch)

def generate_distributed_moe_analysis(model, test_loader, device, save_dir):
    """Generate analysis plots for Distributed MoE"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()
    all_routing_stats = []
    all_assignments = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, routing_scores, D = model(x)
            
            stats = model.get_routing_stats(D)
            all_routing_stats.append(stats)
            all_assignments.append(D.cpu().numpy())
    
    if not all_routing_stats:
        return
    
    # Combine all assignments
    all_D = np.concatenate(all_assignments, axis=0)
    
    # Create comprehensive analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distributed MoE Analysis', fontsize=16)
    
    # 1. Expert load distribution
    expert_loads = np.array([stats['expert_loads'] for stats in all_routing_stats])
    axes[0, 0].boxplot([expert_loads[:, i] for i in range(model.num_experts)], 
                       labels=[f'E{i}' for i in range(model.num_experts)])
    axes[0, 0].set_title('Expert Load Distribution')
    axes[0, 0].set_ylabel('Load')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Coverage over time
    coverages = [stats['expert_coverage'] for stats in all_routing_stats]
    coverage_ratios = [stats['coverage_ratio'] for stats in all_routing_stats]
    axes[0, 1].plot(coverages, label='Absolute Coverage')
    axes[0, 1].plot([cr * model.num_experts for cr in coverage_ratios], label='Coverage Ratio × NumExperts')
    axes[0, 1].set_title('Expert Coverage Over Batches')
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Coverage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Assignment heatmap
    assignment_freq = all_D.sum(axis=0)
    sns.barplot(x=range(model.num_experts), y=assignment_freq, ax=axes[0, 2])
    axes[0, 2].set_title('Total Expert Assignments')
    axes[0, 2].set_xlabel('Expert ID')
    axes[0, 2].set_ylabel('Total Assignments')
    
    # 4. Expert usage EMA evolution
    usage_emas = [stats['usage_ema'] for stats in all_routing_stats]
    usage_ema_array = np.array(usage_emas)
    for i in range(model.num_experts):
        axes[1, 0].plot(usage_ema_array[:, i], label=f'Expert {i}')
    axes[1, 0].set_title('Expert Usage EMA Evolution')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Usage EMA')
    if model.num_experts <= 8:
        axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Load balance statistics
    mean_loads = expert_loads.mean(axis=0)
    std_loads = expert_loads.std(axis=0)
    x_pos = np.arange(model.num_experts)
    axes[1, 1].bar(x_pos, mean_loads, yerr=std_loads, capsize=5, alpha=0.7)
    axes[1, 1].set_title('Expert Load Statistics')
    axes[1, 1].set_xlabel('Expert ID')
    axes[1, 1].set_ylabel('Load (Mean ± Std)')
    axes[1, 1].set_xticks(x_pos)
    
    # 6. Utilization efficiency
    total_capacity = model.capacity * model.num_experts
    actual_usage = [stats['expert_loads'].sum() for stats in all_routing_stats]
    utilization_efficiency = [usage / total_capacity for usage in actual_usage]
    axes[1, 2].plot(utilization_efficiency)
    axes[1, 2].set_title('Capacity Utilization Efficiency')
    axes[1, 2].set_xlabel('Batch')
    axes[1, 2].set_ylabel('Utilization Ratio')
    axes[1, 2].axhline(np.mean(utilization_efficiency), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(utilization_efficiency):.2f}')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distributed_moe_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n=== DISTRIBUTED MOE ANALYSIS SUMMARY ===")
    print(f"Average Expert Coverage: {np.mean(coverages):.1f}/{model.num_experts} ({np.mean(coverage_ratios):.1%})")
    print(f"Expert Load Balance (std): {np.mean([np.std(loads) for loads in expert_loads]):.2f}")
    print(f"Average Capacity Utilization: {np.mean(utilization_efficiency):.1%}")
    print(f"Min Expert Usage: {assignment_freq.min()}")
    print(f"Max Expert Usage: {assignment_freq.max()}")
    print(f"Expert Usage Range: {assignment_freq.max() - assignment_freq.min()}")
    
    # Check if all experts are being used
    unused_experts = (assignment_freq == 0).sum()
    if unused_experts == 0:
        print("✅ All experts are being utilized!")
    else:
        print(f"⚠️  {unused_experts} experts are not being used")
    
    print("=" * 45)
