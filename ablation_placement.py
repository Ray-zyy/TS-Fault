"""
ablation_placement.py — Window-placement ablation study

Tests whether S(W) (window-importance score) is actually finding the right windows.
Compares three placement policies:
  - topk:         Top-K windows ranked by S(W) (the baseline)
  - random:       Uniformly random windows (delta-matched for severity)
  - anti:         Bottom-K windows (worst ranked by S(W))

Metrics:
  - Median relative degradation (RD)
  - Mean robust ratio r
  - Catastrophic failure count
  - Seed-to-seed variability (IQR / median of r across 5 seeds)
  - Rank correlation ρ (clean vs faulted)

Run at severity s₃ (d06) over all datasets/modes/models × 5 seeds.

Usage:
    python ablation_placement.py \
        --policy topk random anti \
        --severity d06 \
        --seeds 5 \
        --npz_root ./TS-Fault_output \
        --out ./results_placement.csv \
        --gpu 0

Output: results_placement.csv with columns:
    policy, mode, median_rd, mean_r, catastrophic_pct, seed_iqr_over_median, spearman_rho
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")


def load_npz_data(npz_path):
    """Load .npz and return cleaned arrays"""
    data = np.load(npz_path)
    x_clean = data["x_clean"].astype(np.float32)  # (N, L, C)
    x_corrupt = data["x_corrupt"].astype(np.float32)
    y_target = data["y_target"].astype(np.float32)
    return x_clean, x_corrupt, y_target


def compute_window_importance(x_clean, y_target, window_size=336):
    """
    Compute S(W) for each window using heuristics:
    S(W) = weighted sum of S_cp + S_per + S_var + S_pred
    
    A simplified version; in practice, use the exact mode-specific formulas
    from Section III-C.
    
    Args:
        x_clean: (N, L, C) clean windows
        y_target: (N, H, C) targets
        window_size: lookback length
    
    Returns:
        scores: (N,) importance scores
    """
    N = len(x_clean)
    scores = np.zeros(N)
    
    for i in range(N):
        x = x_clean[i]  # (L, C)
        y = y_target[i]  # (H, C)
        
        # S_cp: change-point detection (variance spike)
        diffs = np.abs(np.diff(x, axis=0))  # (L-1, C)
        s_cp = np.max(np.mean(diffs, axis=1))
        
        # S_per: periodicity (autocorrelation)
        x_mean = x - np.mean(x, axis=0)
        if np.std(x_mean) > 0:
            acf = np.correlate(x_mean[:, 0], x_mean[:, 0], mode='full')
            s_per = np.max(acf[len(acf)//2 + 1 : len(acf)//2 + 50])
        else:
            s_per = 0
        
        # S_var: variance (volatility)
        s_var = np.mean(np.std(x, axis=0))
        
        # S_pred: predictability (future target variance)
        s_pred = np.mean(np.std(y, axis=0))
        
        # Weighted sum
        scores[i] = 0.25 * (s_cp + s_per + s_var + s_pred)
    
    return scores


def place_windows_topk(scores, k=5):
    """Top-K windows by score"""
    indices = np.argsort(-scores)[:k]  # Descending order
    return sorted(indices)


def place_windows_random(scores, k=5, seed=0):
    """Random windows (delta-matched)"""
    np.random.seed(seed)
    indices = np.random.choice(len(scores), size=k, replace=False)
    return sorted(indices)


def place_windows_anti(scores, k=5):
    """Bottom-K windows (worst by S(W))"""
    indices = np.argsort(scores)[:k]  # Ascending order
    return sorted(indices)


@torch.no_grad()
def evaluate_placement(model, x_clean, x_corrupt, y_target, placement_indices, device):
    """
    Evaluate a model on specific window placements.
    
    Args:
        model: forecasting model
        x_clean, x_corrupt: (N, L, C)
        y_target: (N, H, C)
        placement_indices: list of window indices to use
    
    Returns:
        (mse_clean, mae_clean, mse_corrupt, mae_corrupt, r, spearman_rho)
    """
    model.eval()
    
    # Filter to placement indices
    x_c = x_clean[placement_indices]
    x_cp = x_corrupt[placement_indices]
    y = y_target[placement_indices]
    
    x_c_t = torch.from_numpy(x_c).float().to(device)
    x_cp_t = torch.from_numpy(x_cp).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    
    y_pred_clean = model(x_c_t).cpu().numpy() if hasattr(model, '__call__') else x_c  # Placeholder
    y_pred_corrupt = model(x_cp_t).cpu().numpy() if hasattr(model, '__call__') else x_cp
    y_np = y
    
    # Metrics
    mse_clean = np.mean((y_pred_clean - y_np) ** 2)
    mae_clean = np.mean(np.abs(y_pred_clean - y_np))
    mse_corrupt = np.mean((y_pred_corrupt - y_np) ** 2)
    mae_corrupt = np.mean(np.abs(y_pred_corrupt - y_np))
    
    r = mse_corrupt / mse_clean if mse_clean > 0 else 1.0
    
    # Rank correlation (clean vs corrupt ranking)
    if len(placement_indices) > 2:
        clean_errors = np.mean((y_pred_clean - y_np) ** 2, axis=(1, 2))
        corrupt_errors = np.mean((y_pred_corrupt - y_np) ** 2, axis=(1, 2))
        clean_rank = np.argsort(clean_errors)
        corrupt_rank = np.argsort(corrupt_errors)
        spearman_rho, _ = spearmanr(clean_rank, corrupt_rank)
    else:
        spearman_rho = 1.0
    
    return mse_clean, mae_clean, mse_corrupt, mae_corrupt, r, spearman_rho


def run_placement_ablation(
    policies,
    severity,
    seeds,
    npz_root,
    model_list,
    out_path,
    device="cpu",
):
    """
    Main placement ablation pipeline.
    
    Args:
        policies: ['topk', 'random', 'anti']
        severity: 'd06' (run only at s₃)
        seeds: number of random seeds for 'random' policy
        npz_root: path to TS-Fault .npz files
        model_list: [(name, model_obj), ...]
        out_path: output CSV
        device: torch device
    """
    
    results = []
    npz_root = Path(npz_root)
    
    d_str = severity
    
    # Iterate over datasets and modes
    for dataset_dir in sorted(npz_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        
        # Load all four modes at this severity
        mode_data = {}
        for mode_num in range(1, 5):
            npz_file = dataset_dir / f"{dataset_name}_Mode{mode_num}_{d_str}.npz"
            if npz_file.exists():
                x_c, x_cp, y = load_npz_data(npz_file)
                mode_data[mode_num] = (x_c, x_cp, y)
        
        # For each mode
        for mode_num, (x_clean, x_corrupt, y_target) in mode_data.items():
            print(f"  {dataset_name} Mode {mode_num} ({len(x_clean)} windows)...", end="")
            
            # Compute window importance scores
            scores = compute_window_importance(x_clean, y_target)
            
            # For each model
            for model_name, model in model_list:
                # For each policy
                for policy in policies:
                    # Track seed variability
                    r_values = []
                    all_results = []
                    
                    if policy == "topk":
                        placements = [place_windows_topk(scores, k=20)]
                    elif policy == "anti":
                        placements = [place_windows_anti(scores, k=20)]
                    elif policy == "random":
                        # Multiple seeds for random
                        placements = [
                            place_windows_random(scores, k=20, seed=seed)
                            for seed in range(seeds)
                        ]
                    
                    for seed_idx, placement_idx in enumerate(placements):
                        mse_c, mae_c, mse_cp, mae_cp, r, rho = evaluate_placement(
                            model, x_clean, x_corrupt, y_target, placement_idx, device
                        )
                        
                        # Relative degradation (%)
                        rd = (r - 1.0) * 100.0
                        
                        # Catastrophic (r ≥ 10)
                        catastrophic = 1 if r >= 10 else 0
                        
                        result = {
                            "policy": policy,
                            "model": model_name,
                            "dataset": dataset_name,
                            "mode": mode_num,
                            "severity": d_str,
                            "seed": seed_idx,
                            "mse_clean": mse_c,
                            "mae_clean": mae_c,
                            "mse_corrupt": mse_cp,
                            "mae_corrupt": mae_cp,
                            "robust_ratio": r,
                            "relative_degradation_pct": rd,
                            "catastrophic": catastrophic,
                            "spearman_rho": rho,
                        }
                        all_results.append(result)
                        r_values.append(r)
                    
                    # Seed variability (IQR / median)
                    if len(r_values) > 1:
                        q75, q25 = np.percentile(r_values, [75, 25])
                        median_r = np.median(r_values)
                        iqr = q75 - q25
                        seed_iqr_over_median = iqr / median_r if median_r > 0 else 0.0
                    else:
                        seed_iqr_over_median = 0.0
                    
                    # Aggregate across seeds for this (policy, model, dataset, mode)
                    agg_result = {
                        "policy": policy,
                        "model": model_name,
                        "dataset": dataset_name,
                        "mode": mode_num,
                        "severity": d_str,
                        "median_rd_pct": np.median([r["relative_degradation_pct"] for r in all_results]),
                        "mean_r": np.mean(r_values),
                        "catastrophic_pct": np.mean([r["catastrophic"] for r in all_results]) * 100,
                        "seed_iqr_over_median": seed_iqr_over_median,
                        "mean_spearman_rho": np.mean([r["spearman_rho"] for r in all_results]),
                    }
                    results.append(agg_result)
            
            print(" done")
    
    # Write results
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")
    
    # Summary: per policy
    print("\n" + "="*70)
    print("Placement ablation summary (aggregated across all configs)")
    print("="*70)
    summary = df.groupby("policy").agg({
        "median_rd_pct": "mean",
        "mean_r": "mean",
        "catastrophic_pct": "mean",
        "seed_iqr_over_median": "mean",
        "mean_spearman_rho": "mean",
    }).round(3)
    print(summary)
    print("\nInterpretation:")
    print("  - median_rd_pct: lower is better (less degradation)")
    print("  - seed_iqr_over_median: lower is better (more stable)")
    print("  - mean_spearman_rho: higher is worse (rankings diverge)")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Window placement ablation")
    parser.add_argument(
        "--policy",
        nargs="+",
        default=["topk", "random", "anti"],
        choices=["topk", "random", "anti"],
    )
    parser.add_argument("--severity", type=str, default="d06")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--npz_root", type=str, default="./TS-Fault_output")
    parser.add_argument("--out", type=str, default="./results_placement.csv")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Placeholder model list
    model_list = [("LSTM", None), ("GRU", None), ("TimesFM", None)]
    
    run_placement_ablation(
        policies=args.policy,
        severity=args.severity,
        seeds=args.seeds,
        npz_root=args.npz_root,
        model_list=model_list,
        out_path=args.out,
        device=device,
    )
