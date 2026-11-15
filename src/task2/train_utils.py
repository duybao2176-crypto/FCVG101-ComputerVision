"""Training loop, evaluation loop and helpers (seed, device, prediction on a loader)."""
import json
import random
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import config
from .model import set_backbone_trainable


def seed_everything(seed: int = config.SEED) -> None:
    """Make runs repeatable (python, numpy and torch random generators)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU when available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    scaler: Optional[torch.amp.GradScaler] = None) -> Tuple[float, float]:
    """One pass over the training data. Returns (mean loss, accuracy)."""
    model.train()                                   # enables dropout / BatchNorm updates
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:                      # mixed precision (GPU only)
            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()                         # compute gradients
            optimizer.step()                        # update weights
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float]:
    """One pass over validation/test data without learning. Returns (loss, accuracy)."""
    model.eval()                                    # disables dropout, freezes BatchNorm
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader,
                   device: Optional[torch.device] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict a whole loader. Returns (true labels, predicted labels, probabilities)."""
    device = device or get_device()
    model.to(device).eval()
    y_true, probs = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_true.append(labels.numpy())
    probs = np.concatenate(probs)
    return np.concatenate(y_true), probs.argmax(1), probs


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, confidence: float = 0.95,
                 seed: int = config.SEED) -> Tuple[float, float]:
    """Bootstrap confidence interval of an accuracy.

    `correct` is an array of 0/1 (1 = prediction was right). We resample the test
    images with replacement n_boot times, compute the accuracy each time, and
    return the central `confidence` range of those accuracies.
    """
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    samples = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    return float(np.percentile(samples, tail)), float(np.percentile(samples, 100 - tail))


def fit(model: nn.Module, loaders: Dict[str, DataLoader],
        class_weights: Optional[torch.Tensor] = None,
        epochs: int = config.EPOCHS, lr: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY, freeze_epochs: int = config.FREEZE_EPOCHS,
        patience: int = config.PATIENCE, use_amp: bool = config.USE_AMP,
        device: Optional[torch.device] = None, out_dir: Optional[Path] = None,
        prefix: str = "", meta: Optional[dict] = None,
        log_fn: Optional[Callable[[str], None]] = None) -> dict:
    """Train with AdamW + cosine LR schedule, keep the best model by validation accuracy.

    Saves {prefix}model_best.pt (weights), {prefix}model_meta.json and
    {prefix}history.json into out_dir. Stops early when validation accuracy
    does not improve for `patience` epochs. Returns the history dict.
    """
    device = device or get_device()
    out_dir = Path(out_dir) if out_dir else config.ensure_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = log_fn or (lambda message: None)

    model.to(device)
    weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "lr": [], "epoch_seconds": []}
    best_acc, best_loss, best_epoch, bad_epochs = -1.0, float("inf"), 0, 0
    checkpoint = out_dir / f"{prefix}model_best.pt"

    for epoch in range(1, epochs + 1):
        start = time.time()
        # Warm-up: during the first epochs only the new head is trained.
        set_backbone_trainable(model, trainable=(epoch > freeze_epochs))
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion,
                                                optimizer, device, scaler)
        val_loss, val_acc = evaluate_epoch(model, loaders["val"], criterion, device)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        for key, value in zip(["train_loss", "train_acc", "val_loss", "val_acc"],
                              [train_loss, train_acc, val_loss, val_acc]):
            history[key].append(value)
        history["epoch_seconds"].append(time.time() - start)
        log(f"epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | {history['epoch_seconds'][-1]:.1f}s")

        # Best = higher val accuracy; if tied, lower val loss.
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, best_epoch, bad_epochs = val_acc, val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device))   # restore best weights
    history["best_epoch"], history["best_val_acc"] = best_epoch, best_acc

    (out_dir / f"{prefix}history.json").write_text(json.dumps(history, indent=2))
    full_meta = dict(meta or {}, best_epoch=best_epoch, best_val_acc=best_acc,
                     epochs_run=len(history["train_loss"]))
    (out_dir / f"{prefix}model_meta.json").write_text(json.dumps(full_meta, indent=2))
    return history
