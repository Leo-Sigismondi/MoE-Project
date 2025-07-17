import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

def train_single_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs=50,
    lr=0.001,
    use_amp=True,
    log_interval=100
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.GradScaler()

    # TensorBoard writer
    run_name = f"runs/CNN_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"{run_name}")
    
    # Log file
    run_dir = Path(run_name)
    log_txt = run_dir / "train_log.txt"

    if not log_txt.exists():
        with log_txt.open("w") as fp:
            json.dump({
                "model": model.__class__.__name__,
                "epochs": epochs,
                "lr": lr,
                "use_amp": use_amp,
            }, fp)
            fp.write("\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % log_interval == 0:
                loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)

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
                if use_amp:
                    with torch.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total

        # Logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # txt log
        log_line = (
            f"{epoch + 1:03d} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val_loss {val_loss:.4f} acc {val_acc:.2f}% | "
        )
        print(log_line)
        with log_txt.open("a") as fp:
            fp.write(log_line + "\n")

        # print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
