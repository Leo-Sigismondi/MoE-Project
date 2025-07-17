# Vision_trainer_Waterfall.py – v2 with rich routing analytics
"""
Enhanced trainer for **CollaborativeWaterfallMoE**
-------------------------------------------------
Adds detailed logging so you can *see* how the routing behaves:

* **Per‑expert load** (fraction of tokens) – scalar + histogram
* **Entropy** of routing probabilities (higher ⇒ more balanced)
* **Iterations** used by the waterfall algorithm
* Everything streamed to **TensorBoard** + plain txt log

Open TensorBoard:
```bash
tensorboard --logdir runs/collab_moe_experiment_*
```
Then check the tabs *Scalars* and *Histograms*.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt



# ---------------------------------------------------------------------------
# Helper metrics (fallback)
# ---------------------------------------------------------------------------


def _accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return 100.0 * correct / target.size(0)


try:
    from metrics import accuracy as _metrics_accuracy  # type: ignore
except Exception:  # noqa: BLE001
    _metrics_accuracy = _accuracy  # type: ignore

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def log_expert_bar(writer, class_counts, expert_idx, epoch):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(class_counts)), class_counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Assigned tokens")
    ax.set_title(f"Expert {expert_idx} - class distribution")
    fig.tight_layout()
    # Convert to image for TensorBoard
    writer.add_figure(f"ExpertClassBar/expert_{expert_idx}", fig, epoch)
    plt.close(fig)

def train_moe_waterfall(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str | torch.device = "cuda",
    num_experts: int = 4,
    capacity: int | str = "auto",
    num_classes: int = 10,
    epochs: int = 100,
    resume_from_checkpoint: bool = False,
    starting_epoch: int = 0,
    lr: float = 1e-3,
    run_dir: str | Path = "runs/collab_moe",
    use_amp: bool = True,
    save_every: int = 10,
) -> None:
    """Train **CollaborativeWaterfallMoE** and log rich diagnostics."""

    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_txt = run_dir / "train_log.txt"

    writer = SummaryWriter(run_dir)

    # ------------------ model & optimiser ------------------
    device = torch.device(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer, mode='min', factor=0.01, patience=5, min_lr=1e-6
    # )
    scaler: Optional[GradScaler] = GradScaler(enabled=use_amp)

    # ------------------ checkpoint resume ------------------
    last_ckpt = ckpt_dir / "last.pth"
    if resume_from_checkpoint and last_ckpt.is_file():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        starting_epoch = ckpt.get("epoch", starting_epoch)
        print(f"[Trainer] Resumed at epoch {starting_epoch} from {last_ckpt}")
    # ------------------ static hp log ------------------
    if not log_txt.exists():
        with log_txt.open("w") as fp:
            json.dump({
                "num_experts": num_experts,
                "capacity": capacity,
                "epochs": epochs,
                "lr": lr,
                "use_amp": use_amp,
            }, fp)
            fp.write("\n")

    def get_temperature(epoch, total_epochs, T_warm=5.0, T_final=0.1, warmup_frac=0.1):
        warmup_epochs = int(total_epochs * warmup_frac)
        if epoch < warmup_epochs:
            return T_warm
        else:
            # Exponential decay
            decay_epochs = total_epochs - warmup_epochs
            decay_epoch = epoch - warmup_epochs
            decay_rate = (T_final / T_warm) ** (1 / max(1, decay_epochs))
            return max(T_final, T_warm * (decay_rate ** decay_epoch))

    # ------------------ epoch loop ------------------
    global_step = 0
    for epoch in range(starting_epoch, epochs):

        # ---------------- TRAIN ----------------
        model.train()
        ep_train_loss = ep_train_acc = 0.0
        n_train = 0
        # routing stats
        expert_tok_counter = np.zeros(num_experts)
        entropies: List[float] = []
        iter_counts: List[int] = []
        epoch_class_counts = [np.zeros(num_classes, dtype=int) for _ in range(num_experts)]


        # Anneal T from 2.0 to 0.1 over the course of training
        T = get_temperature(epoch, epochs, T_warm=5.0, T_final=0.1, warmup_frac=0.1)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, dynamic_ncols=True, unit="batch")
        for images, targets in pbar:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', enabled=use_amp):
                logits, probs, assignment, aux_loss, iters, scores, aux_losses = model(images, T=T, return_aux=True, targets=targets)
                loss = criterion(logits, targets)
                if aux_loss is not None:
                    loss = loss + aux_loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # --- stats ---
            bs = targets.size(0)
            ep_train_loss += loss.item() * bs
            ep_train_acc += _metrics_accuracy(logits.detach(), targets) * bs
            n_train += bs

            # routing analytics
            expert_tok_counter += assignment.sum(dim=0).cpu().numpy()
            batch_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean().item()
            entropies.append(batch_entropy)
            iter_counts.append(iters)

            pbar.set_postfix({"loss": ep_train_loss / n_train, "acc": ep_train_acc / n_train})
            writer.add_scalar("Routing/Entropy_step", batch_entropy, global_step)
            writer.add_scalar("Routing/Iterations_step", iters, global_step)
            # NEW: Log the distribution of expert scores
            for e in range(num_experts):
                # writer.add_histogram(f"Scores/expert_{e}", scores[:, e].detach().cpu().numpy(), global_step)
                writer.add_scalar(f"Scores/mean/expert_{e}", scores[:, e].mean().item(), global_step)
                writer.add_scalar(f"Scores/mean_abs/expert_{e}", scores[:, e].abs().mean().item(), global_step)
            # NEW: Log the distribution of expert assignments
            for e in range(num_experts):
                idx = (assignment[:, e]).cpu().numpy()
                class_counts = np.bincount(targets.cpu().numpy()[idx], minlength=num_classes)
                epoch_class_counts[e] += class_counts
            # --- Log aux_losses components (train) ---
            if isinstance(aux_losses, dict):
                for k, v in aux_losses.items():
                    if hasattr(v, "item"):
                        writer.add_scalar(f"AuxLosses/train/{k}", v.item(), epoch)

            global_step += 1

        # epoch aggregates
        ep_train_loss /= n_train
        ep_train_acc /= n_train
        avg_load = expert_tok_counter / expert_tok_counter.sum()
        for e, load in enumerate(avg_load):
            writer.add_scalar(f"Routing/AvgLoad_train/expert_{e}", load, epoch)
        # writer.add_histogram("Routing/LoadHist_train", expert_tok_counter, epoch)
        writer.add_scalar("Routing/Entropy_epoch/train", np.mean(entropies), epoch)
        writer.add_scalar("Routing/IterMean/train", np.mean(iter_counts), epoch)
        if epoch % 20 == 0 or epoch == epochs - 1:
            for e in range(num_experts):
                # class_counts as before
                log_expert_bar(writer, class_counts=epoch_class_counts[e], expert_idx=e, epoch=epoch)

        # ---------------- VAL ----------------
        model.eval()
        ep_val_loss = ep_val_acc = 0.0
        n_val = 0
        val_expert_tok_counter = np.zeros(num_experts)
        val_entropies: List[float] = []
        val_iter_counts: List[int] = []
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                logits, probs, assignment, aux_loss, iters, scores, aux_losses = model(images, T=0.1, return_aux=True, targets=targets)
                loss = criterion(logits, targets)

                bs = targets.size(0)
                ep_val_loss += loss.item() * bs
                ep_val_acc += _metrics_accuracy(logits, targets) * bs
                n_val += bs

                val_expert_tok_counter += assignment.sum(dim=0).cpu().numpy()
                batch_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=1).mean().item()
                val_entropies.append(batch_entropy)
                val_iter_counts.append(iters)

        ep_val_loss /= n_val
        ep_val_acc /= n_val
        val_avg_load = val_expert_tok_counter / val_expert_tok_counter.sum()
        for e, load in enumerate(val_avg_load):
            writer.add_scalar(f"Routing/AvgLoad_val/expert_{e}", load, epoch)
        # writer.add_histogram("Routing/LoadHist_val", val_expert_tok_counter, epoch)
        writer.add_scalar("Routing/Entropy_epoch/val", np.mean(val_entropies), epoch)
        writer.add_scalar("Routing/IterMean/val", np.mean(val_iter_counts), epoch)

        # ---------------- scalar logs ----------------
        writer.add_scalar("Loss/Train", ep_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", ep_train_acc, epoch)
        writer.add_scalar("Loss/Validation", ep_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", ep_val_acc, epoch)
        # writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("Temperature", T, epoch)

        
        # txt log
        log_line = (
            f"{epoch + 1:03d} | "
            f"train_loss {ep_train_loss:.4f} acc {ep_train_acc:.2f}% | "
            f"val_loss {ep_val_loss:.4f} acc {ep_val_acc:.2f}% | "
            f"H_ent {np.mean(entropies):.3f}/{np.mean(val_entropies):.3f} | "
            f"iters {np.mean(iter_counts):.2f}/{np.mean(val_iter_counts):.2f}"
        )
        print(log_line)
        with log_txt.open("a") as fp:
            fp.write(log_line + "\n")

        # step lr scheduler
        # scheduler.step( ep_val_loss)
        scheduler.step()

        # ---------------- checkpoint ----------------
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch+1}.pth")
            torch.save(ckpt, last_ckpt)  # rolling checkpoint

    writer.close()
    print("[Trainer] Training complete   →", ckpt_dir)