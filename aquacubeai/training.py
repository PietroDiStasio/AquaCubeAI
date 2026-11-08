"""Training preserves the original losses, RNG resets, scheduler and stopping rule."""

from dataclasses import replace
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .config import Config, set_seed
from .data import Dataset
from .models import RegressNN
from .evaluation import evaluate_checkpoint


def train_mlp_regressor(
    X_train_tensor: torch.Tensor,
    Y_train_tensor: torch.Tensor,
    X_val_tensor: torch.Tensor,
    Y_val_tensor: torch.Tensor,
    hidden_layers: Tuple[int, ...],
    cfg: Config,
    best_model_path: str,
):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    input_size = X_train_tensor.shape[1]
    n_targets = Y_train_tensor.shape[1]
    model = RegressNN(input_size, list(hidden_layers), n_targets).to(device)
    criterion = nn.HuberLoss(delta=cfg.huber_delta)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    train_ds_local = Dataset(X_train_tensor, Y_train_tensor)
    val_ds_local = Dataset(X_val_tensor, Y_val_tensor)
    train_loader_local = DataLoader(
        train_ds_local,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=gen,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader_local = DataLoader(
        val_ds_local,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best_val = float("inf")
    best_epoch = 0
    train_hist, val_hist = ([], [])
    if cfg.verbose:
        print(
            f"\n[TRAIN] hidden_layers={hidden_layers} | lr={cfg.lr} | batch={cfg.batch_size} | Huber(delta={cfg.huber_delta}) | seed={cfg.seed}"
        )
    for epoch in range(cfg.n_epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader_local:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            yp = model(xb)
            loss = criterion(yp, yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            batch_losses.append(loss.item())
        tr_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        train_hist.append(tr_loss)
        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader_local:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                yp = model(xb)
                vlosses.append(criterion(yp, yb).item())
        val_loss = float(np.mean(vlosses))
        val_hist.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
        if epoch - best_epoch > cfg.early_stop_patience:
            if cfg.verbose:
                print(
                    f"[Early Stop] epoch={epoch} | best_epoch={best_epoch} | best_val={best_val:.6f}"
                )
            break
        scheduler.step(val_loss)
        if cfg.verbose and epoch % 25 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Ep {epoch:4d} | Tr {tr_loss:.6f} | Val {val_loss:.6f} | LR {lr_now:.2e}")
    return (best_val, best_epoch, train_hist, val_hist)


def architecture_search(arch_list, tensors, cfg, output_dir):
    """Run the original capped search; rank by validation Huber loss."""
    results = []
    for arch in arch_list:
        arch = tuple(arch)
        search_cfg = replace(
            cfg,
            n_epochs=min(cfg.n_epochs, 800),
            early_stop_patience=min(cfg.early_stop_patience, 120),
        )
        path = Path(output_dir) / ("best_model_" + "x".join(map(str, arch)) + ".pth")
        best_val, best_epoch, train_hist, val_hist = train_mlp_regressor(
            *tensors["train"], *tensors["val"], arch, search_cfg, str(path)
        )
        out = evaluate_checkpoint(path, arch, tensors, cfg)
        results.append(
            dict(
                arch=arch,
                checkpoint=str(path),
                best_val=best_val,
                best_epoch=best_epoch,
                seed=cfg.seed,
                evaluation=out,
                train_history=train_hist,
                val_history=val_hist,
            )
        )
    return sorted(results, key=lambda result: result["best_val"])


def select_best(results):
    """Final selection uses validation RMSE, as in the original post-hoc step."""
    if not results:
        raise ValueError("No architecture results. Run training or load a supplied checkpoint.")
    return min(results, key=lambda result: result["evaluation"]["metrics"]["val"]["rmse"])
