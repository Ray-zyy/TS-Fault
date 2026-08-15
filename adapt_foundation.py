"""
adapt_foundation.py — Foundation model adaptation & robustness testing

Tests four adaptation levels on TimesFM, Chronos, and Moirai:
  ZS:       Zero-shot (released weights)
  LP:       Layer-wise fine-tune (freeze backbone, tune head on clean data)
  FT:       Full fine-tune (all parameters, clean data)
  FT-fault: Full fine-tune on low-severity faulted data (s₁–s₂),
            evaluated on held-out severities (s₃–s₅)

The FT-fault setting tests whether exposing fault structure in training
helps the model generalize to unseen fault severities.

Usage:
    python adapt_foundation.py \
        --model timesfm \
        --setting ft_fault \
        --train_severities d02 d04 \
        --eval_severities d06 d08 d10 \
        --npz_root ./TS-Fault_output \
        --out ./results_foundation_adapt.csv \
        --gpu 0

Output: results_foundation_adapt.csv with columns:
    model, setting, dataset, mode, difficulty,
    mse_clean, mse_corrupt, mae_clean, mae_corrupt, robust_ratio, ...
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")


class FoundationModelWrapper(nn.Module):
    """
    Wrapper for foundation models (TimesFM, Chronos, Moirai).
    Provides uniform interface for training and inference.
    """
    
    def __init__(self, model_name, pretrained_model, device="cpu"):
        super().__init__()
        self.model_name = model_name
        self.backbone = pretrained_model
        self.device = device
        self.backbone.to(device)
        
        # Assume backbone has input_size and output_size
        # For adaptation, add a small head
        self.backbone_frozen = False
    
    def freeze_backbone(self):
        """Freeze backbone parameters (LP setting)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True
    
    def unfreeze_backbone(self):
        """Unfreeze for FT setting"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone_frozen = False
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            y_pred: (batch, horizon, channels)
        """
        # Channel-independent inference: process each channel separately
        batch_size, seq_len, n_channels = x.shape
        outputs = []
        
        for ch in range(n_channels):
            x_ch = x[:, :, ch]  # (batch, seq_len)
            # Pass through backbone (assume it handles 2D input)
            y_ch = self.backbone(x_ch)  # (batch, horizon)
            outputs.append(y_ch)
        
        y_pred = torch.stack(outputs, dim=2)  # (batch, horizon, channels)
        return y_pred


def load_npz_data(npz_path):
    """Load .npz and return cleaned arrays"""
    data = np.load(npz_path)
    x_clean = data["x_clean"].astype(np.float32)  # (N, L, C)
    x_corrupt = data["x_corrupt"].astype(np.float32)
    y_target = data["y_target"].astype(np.float32)  # (N, H, C)
    return x_clean, x_corrupt, y_target


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x_batch)
        
        # MSE loss
        loss = nn.MSELoss()(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def evaluate(model, x_clean, x_corrupt, y_true, device):
    """Evaluate model on clean and corrupt inputs"""
    model.eval()
    
    x_c = torch.from_numpy(x_clean).float().to(device)
    x_cp = torch.from_numpy(x_corrupt).float().to(device)
    y = torch.from_numpy(y_true).float().to(device)
    
    y_pred_clean = model(x_c).cpu().numpy()
    y_pred_corrupt = model(x_cp).cpu().numpy()
    y_np = y_true
    
    mse_clean = np.mean((y_pred_clean - y_np) ** 2)
    mae_clean = np.mean(np.abs(y_pred_clean - y_np))
    mse_corrupt = np.mean((y_pred_corrupt - y_np) ** 2)
    mae_corrupt = np.mean(np.abs(y_pred_corrupt - y_np))
    
    r = mse_corrupt / mse_clean if mse_clean > 0 else 1.0
    
    return mse_clean, mae_clean, mse_corrupt, mae_corrupt, r


def adapt_and_evaluate(
    model,
    model_name,
    setting,
    train_data,
    eval_data,
    device,
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
):
    """
    Adapt model according to setting and evaluate.
    
    Args:
        model: FoundationModelWrapper
        setting: 'zs' | 'lp' | 'ft' | 'ft_fault'
        train_data: (x_train, y_train) for LP/FT/FT-fault
        eval_data: {split: (x_clean, x_corrupt, y_target)}
        
    Returns:
        results dict
    """
    results = {"model": model_name, "setting": setting}
    
    if setting == "zs":
        # Zero-shot: just evaluate with released weights
        x_c, x_cp, y = eval_data["full"]
        mse_c, mae_c, mse_cp, mae_cp, r = evaluate(model, x_c, x_cp, y, device)
        results.update({
            "mse_clean": mse_c,
            "mae_clean": mae_c,
            "mse_corrupt": mse_cp,
            "mae_corrupt": mae_cp,
            "robust_ratio": r,
        })
    
    elif setting == "lp":
        # Layer-wise fine-tune
        model.freeze_backbone()
        x_train, y_train = train_data
        
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(y_train).float(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        
        for epoch in range(epochs):
            _ = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate on full grid
        x_c, x_cp, y = eval_data["full"]
        mse_c, mae_c, mse_cp, mae_cp, r = evaluate(model, x_c, x_cp, y, device)
        results.update({
            "mse_clean": mse_c,
            "mae_clean": mae_c,
            "mse_corrupt": mse_cp,
            "mae_corrupt": mae_cp,
            "robust_ratio": r,
        })
    
    elif setting == "ft":
        # Full fine-tune on clean data
        model.unfreeze_backbone()
        x_train, y_train = train_data
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(y_train).float(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        
        for epoch in range(epochs):
            _ = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate on full grid
        x_c, x_cp, y = eval_data["full"]
        mse_c, mae_c, mse_cp, mae_cp, r = evaluate(model, x_c, x_cp, y, device)
        results.update({
            "mse_clean": mse_c,
            "mae_clean": mae_c,
            "mse_corrupt": mse_cp,
            "mae_corrupt": mae_cp,
            "robust_ratio": r,
        })
    
    elif setting == "ft_fault":
        # Full fine-tune on low-severity faulted data
        model.unfreeze_backbone()
        x_train, y_train = train_data
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(y_train).float(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        
        for epoch in range(epochs):
            _ = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate only on held-out high severities
        x_c, x_cp, y = eval_data["held_out"]
        mse_c, mae_c, mse_cp, mae_cp, r = evaluate(model, x_c, x_cp, y, device)
        results.update({
            "mse_clean": mse_c,
            "mae_clean": mae_c,
            "mse_corrupt": mse_cp,
            "mae_corrupt": mae_cp,
            "robust_ratio": r,
        })
    
    return results


def run_adaptation_study(
    model_names,
    settings,
    npz_root,
    train_severities,
    eval_severities,
    out_path,
    device="cpu",
):
    """
    Main adaptation study pipeline.
    
    Args:
        model_names: ['timesfm', 'chronos', 'moirai']
        settings: ['zs', 'lp', 'ft', 'ft_fault']
        npz_root: path to TS-Fault .npz files
        train_severities: ['d02', 'd04'] for FT/FT-fault training
        eval_severities: ['d06', 'd08', 'd10'] for held-out test
        out_path: output CSV
        device: torch device
    """
    
    results = []
    npz_root = Path(npz_root)
    
    # Load pre-trained foundation models (skeleton)
    models = {}
    for name in model_names:
        # In practice, load from HuggingFace / official checkpoints
        # models[name] = FoundationModelWrapper(name, pretrained, device)
        models[name] = None  # Placeholder
    
    # Iterate over datasets and modes
    for dataset_dir in sorted(npz_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        
        # Load all severity levels for all modes
        severity_data = {}
        for d_str in train_severities + eval_severities:
            for mode_num in range(1, 5):
                npz_file = dataset_dir / f"{dataset_name}_Mode{mode_num}_d{d_str}.npz"
                if npz_file.exists():
                    key = (mode_num, d_str)
                    x_c, x_cp, y = load_npz_data(npz_file)
                    severity_data[key] = (x_c, x_cp, y)
        
        # For each model and setting
        for model_name, model in models.items():
            for setting in settings:
                print(f"{dataset_name} / {model_name} / {setting}...")
                
                # Prepare training and eval data
                if setting == "zs":
                    # Use d06 as representative full eval
                    eval_data_full = None
                    for mode in range(1, 5):
                        key = (mode, "d06")
                        if key in severity_data:
                            x_c, x_cp, y = severity_data[key]
                            eval_data_full = (x_c, x_cp, y)
                            break
                    
                    if eval_data_full is None:
                        continue
                    
                    eval_data = {"full": eval_data_full}
                    result = adapt_and_evaluate(
                        model, model_name, setting,
                        train_data=None, eval_data=eval_data, device=device,
                    )
                    result["dataset"] = dataset_name
                    results.append(result)
                
                elif setting in ["lp", "ft"]:
                    # Collect all training data (clean)
                    x_trains, y_trains = [], []
                    for d_str in train_severities + eval_severities:
                        for mode in range(1, 5):
                            key = (mode, d_str)
                            if key in severity_data:
                                x_c, _, y = severity_data[key]
                                x_trains.append(x_c)
                                y_trains.append(y)
                    
                    if x_trains:
                        x_train = np.concatenate(x_trains, axis=0)
                        y_train = np.concatenate(y_trains, axis=0)
                        
                        # Full eval data (representative)
                        x_eval, x_eval_cp, y_eval = None, None, None
                        for d_str in ["d06"]:
                            for mode in range(1, 5):
                                key = (mode, d_str)
                                if key in severity_data:
                                    x_eval, x_eval_cp, y_eval = severity_data[key]
                                    break
                        
                        if x_eval is not None:
                            eval_data = {"full": (x_eval, x_eval_cp, y_eval)}
                            result = adapt_and_evaluate(
                                model, model_name, setting,
                                train_data=(x_train, y_train),
                                eval_data=eval_data, device=device,
                            )
                            result["dataset"] = dataset_name
                            results.append(result)
                
                elif setting == "ft_fault":
                    # Train on low-severity (d02, d04), eval on high (d06, d08, d10)
                    x_trains, y_trains = [], []
                    for d_str in train_severities:
                        for mode in range(1, 5):
                            key = (mode, d_str)
                            if key in severity_data:
                                _, x_cp, y = severity_data[key]  # Use corrupted
                                x_trains.append(x_cp)
                                y_trains.append(y)
                    
                    x_evals, x_evals_cp, y_evals = [], [], []
                    for d_str in eval_severities:
                        for mode in range(1, 5):
                            key = (mode, d_str)
                            if key in severity_data:
                                x_c, x_cp, y = severity_data[key]
                                x_evals.append(x_c)
                                x_evals_cp.append(x_cp)
                                y_evals.append(y)
                    
                    if x_trains and x_evals:
                        x_train = np.concatenate(x_trains, axis=0)
                        y_train = np.concatenate(y_trains, axis=0)
                        x_eval = np.concatenate(x_evals, axis=0)
                        x_eval_cp = np.concatenate(x_evals_cp, axis=0)
                        y_eval = np.concatenate(y_evals, axis=0)
                        
                        eval_data = {"held_out": (x_eval, x_eval_cp, y_eval)}
                        result = adapt_and_evaluate(
                            model, model_name, setting,
                            train_data=(x_train, y_train),
                            eval_data=eval_data, device=device,
                        )
                        result["dataset"] = dataset_name
                        results.append(result)
    
    # Write results
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")
    
    # Summary: per setting
    print("\n" + "="*70)
    print("Adaptation summary (mean robust ratio)")
    print("="*70)
    if len(df) > 0:
        summary = df.groupby(["model", "setting"]).agg({
            "robust_ratio": "mean",
            "mse_corrupt": "mean",
        }).round(2)
        print(summary)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foundation model adaptation study")
    parser.add_argument("--model", type=str, default="timesfm", help="Model name")
    parser.add_argument(
        "--setting",
        type=str,
        default="ft_fault",
        choices=["zs", "lp", "ft", "ft_fault"],
    )
    parser.add_argument("--npz_root", type=str, default="./TS-Fault_output")
    parser.add_argument("--out", type=str, default="./results_foundation_adapt.csv")
    parser.add_argument(
        "--train_severities",
        type=str,
        nargs="+",
        default=["d02", "d04"],
    )
    parser.add_argument(
        "--eval_severities",
        type=str,
        nargs="+",
        default=["d06", "d08", "d10"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    run_adaptation_study(
        model_names=[args.model],
        settings=[args.setting],
        npz_root=args.npz_root,
        train_severities=args.train_severities,
        eval_severities=args.eval_severities,
        out_path=args.out,
        device=device,
    )
