"""
ablation_lambda.py — Window-importance weight (λ) sensitivity analysis

Tests whether the per-mode weights in S(W) = λ₁S_cp + λ₂S_per + λ₃S_var + λ₄S_pred
actually matter and in what ways.

Perturbations (one-at-a-time at severity s₃ / d06):
  - λᵢ × 0.5  (reduce by half)
  - λᵢ × 2.0  (double)
  - λᵢ → 0    (zero out)

Metrics:
  - Δ median RD (%): change in relative degradation vs baseline
  - ρ (rank correlation): Spearman between default and perturbed model ranking
  
If ρ stays high (≥0.87), the weight controls *amplitude* not *ordering*.
If ρ drops, the weight controls *which models fail*.

Run at s₃ (d06) only, over all datasets/modes/models.

Usage:
    python ablation_lambda.py \
        --severity d06 \
        --npz_root ./TS-Fault_output \
        --out ./results_lambda_sensitivity.csv \
        --gpu 0

Output: results_lambda_sensitivity.csv with columns:
    mode, lambda_index, perturbation, dataset,
    delta_median_rd_pct, spearman_rho, n_windows
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


# Per-mode default weights (from paper Sec. III-C & IV)
DEFAULT_LAMBDAS = {
    1: {"lambda1": 0.15, "lambda2": 0.15, "lambda3": 0.35, "lambda4": 0.35},  # Time-Warped Shock
    2: {"lambda1": 0.15, "lambda2": 0.20, "lambda3": 0.25, "lambda4": 0.40},  # Dependency Fracture
    3: {"lambda1": 0.40, "lambda2": 0.25, "lambda3": 0.15, "lambda4": 0.20},  # Regime Missingness
    4: {"lambda1": 0.20, "lambda2": 0.15, "lambda3": 0.30, "lambda4": 0.35},  # Cascading Failure
}


def load_npz_data(npz_path):
    """Load .npz and return cleaned arrays"""
    data = np.load(npz_path)
    x_clean = data["x_clean"].astype(np.float32)  # (N, L, C)
    x_corrupt = data["x_corrupt"].astype(np.float32)
    y_target = data["y_target"].astype(np.float32)
    return x_clean, x_corrupt, y_target


def compute_s_components(x_clean, y_target):
    """
    Compute the four components of S(W):
      S_cp:   change-point detection
      S_per:  periodicity
      S_var:  variance
      S_pred: predictability (future volatility)
    
    Returns:
        (s_cp, s_per, s_var, s_pred) each of shape (N,)
    """
    N = len(x_clean)
    s_cp = np.zeros(N)
    s_per = np.zeros(N)
    s_var = np.zeros(N)
    s_pred = np.zeros(N)
    
    for i in range(N):
        x = x_clean[i]  # (L, C)
        y = y_target[i]  # (H, C)
        
        # S_cp: max abs difference (sharpness of edges)
        diffs = np.abs(np.diff(x, axis=0))
        s_cp[i] = np.max(np.mean(diffs, axis=1))
        
        # S_per: periodicity via autocorrelation
        x_mean = x - np.mean(x, axis=0)
        if np.std(x_mean) > 1e-6:
            acf = np.correlate(x_mean[:, 0], x_mean[:, 0], mode='full')
            acf = acf[len(acf)//2:]
            if len(acf) > 50:
                s_per[i] = np.max(acf[1:50])
            else:
                s_per[i] = np.max(acf[1:]) if len(acf) > 1 else 0.0
        else:
            s_per[i] = 0.0
        
        # S_var: volatility
        s_var[i] = np.mean(np.std(x, axis=0))
        
        # S_pred: future target volatility (predictability demand)
        s_pred[i] = np.mean(np.std(y, axis=0))
    
    return s_cp, s_per, s_var, s_pred


def compute_window_importance(x_clean, y_target, mode_num, lambdas=None):
    """
    S(W) = λ₁ S_cp + λ₂ S_per + λ₃ S_var + λ₄ S_pred
    
    Args:
        lambdas: dict with keys 'lambda1', 'lambda2', 'lambda3', 'lambda4'
    
    Returns:
        scores: (N,) importance scores
    """
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS[mode_num]
    
    s_cp, s_per, s_var, s_pred = compute_s_components(x_clean, y_target)
    
    # Normalize each component to [0, 1]
    def normalize(s):
        smax = np.max(np.abs(s))
        if smax < 1e-6:
            return np.zeros_like(s)
        return s / smax
    
    s_cp_norm = normalize(s_cp)
    s_per_norm = normalize(s_per)
    s_var_norm = normalize(s_var)
    s_pred_norm = normalize(s_pred)
    
    scores = (
        lambdas["lambda1"] * s_cp_norm +
        lambdas["lambda2"] * s_per_norm +
        lambdas["lambda3"] * s_var_norm +
        lambdas["lambda4"] * s_pred_norm
    )
    
    return scores


@torch.no_grad()
def evaluate_models(model_list, x_clean, x_corrupt, y_target, window_indices, device):
    """
    Evaluate all models on selected windows.
    
    Returns:
        per_model_errors: {model_name: (N_selected,) error array}
    """
    per_model_errors = {}
    
    for model_name, model in model_list:
        model.eval()
        
        x_c = x_clean[window_indices]
        x_cp = x_corrupt[window_indices]
        y = y_target[window_indices]
        
        # Dummy prediction (placeholder)
        y_pred = np.random.randn(*y.shape)  # Mock
        errors = np.mean((y_pred - y) ** 2, axis=(1, 2))
        
        per_model_errors[model_name] = errors
    
    return per_model_errors


def run_lambda_sensitivity(
    severity,
    npz_root,
    model_list,
    out_path,
    device="cpu",
):
    """
    Main lambda sensitivity pipeline.
    
    Args:
        severity: 'd06' (run only at s₃)
        npz_root: path to TS-Fault .npz files
        model_list: [(name, model_obj), ...]
        out_path: output CSV
        device: torch device
    """
    
    results = []
    npz_root = Path(npz_root)
    
    d_str = severity
    
    # Perturbation types
    perturbations = [
        (1, 0.5, "lambda1_times_0.5"),
        (1, 2.0, "lambda1_times_2.0"),
        (1, 0.0, "lambda1_to_0"),
        (2, 0.5, "lambda2_times_0.5"),
        (2, 2.0, "lambda2_times_2.0"),
        (2, 0.0, "lambda2_to_0"),
        (3, 0.5, "lambda3_times_0.5"),
        (3, 2.0, "lambda3_times_2.0"),
        (3, 0.0, "lambda3_to_0"),
        (4, 0.5, "lambda4_times_0.5"),
        (4, 2.0, "lambda4_times_2.0"),
        (4, 0.0, "lambda4_to_0"),
    ]
    
    # Iterate over datasets
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
            print(f"  {dataset_name} Mode {mode_num}...", end="")
            
            # Compute baseline S(W) with default weights
            baseline_scores = compute_window_importance(
                x_clean, y_target, mode_num, lambdas=DEFAULT_LAMBDAS[mode_num]
            )
            baseline_ranking = np.argsort(-baseline_scores)  # Descending
            
            # Select top-K windows for evaluation
            k = 20
            top_k_indices = baseline_ranking[:k]
            
            # Evaluate all models on baseline windows
            baseline_errors = evaluate_models(
                model_list, x_clean, x_corrupt, y_target, top_k_indices, device
            )
            
            # Relative degradation (median across all models' windows)
            baseline_rd = np.median([
                np.median(errors) for errors in baseline_errors.values()
            ])
            
            # For each perturbation
            for lambda_idx, multiplier, perturb_name in perturbations:
                # Create perturbed lambdas
                perturb_lambdas = DEFAULT_LAMBDAS[mode_num].copy()
                key = f"lambda{lambda_idx}"
                
                if multiplier == 0.0:
                    perturb_lambdas[key] = 0.0
                else:
                    perturb_lambdas[key] *= multiplier
                
                # Recompute S(W) with perturbed weights
                perturb_scores = compute_window_importance(
                    x_clean, y_target, mode_num, lambdas=perturb_lambdas
                )
                perturb_ranking = np.argsort(-perturb_scores)
                perturb_top_k = perturb_ranking[:k]
                
                # Evaluate on perturbed windows
                perturb_errors = evaluate_models(
                    model_list, x_clean, x_corrupt, y_target, perturb_top_k, device
                )
                
                # Perturbed RD
                perturb_rd = np.median([
                    np.median(errors) for errors in perturb_errors.values()
                ])
                
                # Δ RD (%)
                delta_rd = ((perturb_rd - baseline_rd) / max(baseline_rd, 1e-6)) * 100.0
                
                # Rank correlation: do the models' rankings match?
                if len(model_list) > 2:
                    baseline_model_errors = list(baseline_errors.values())
                    perturb_model_errors = list(perturb_errors.values())
                    
                    baseline_model_rank = np.argsort([
                        np.median(e) for e in baseline_model_errors
                    ])
                    perturb_model_rank = np.argsort([
                        np.median(e) for e in perturb_model_errors
                    ])
                    
                    spearman_rho, _ = spearmanr(baseline_model_rank, perturb_model_rank)
                else:
                    spearman_rho = 1.0
                
                result = {
                    "mode": mode_num,
                    "lambda_index": lambda_idx,
                    "perturbation": perturb_name,
                    "dataset": dataset_name,
                    "baseline_median_rd": baseline_rd,
                    "perturbed_median_rd": perturb_rd,
                    "delta_median_rd_pct": delta_rd,
                    "spearman_rho_model_ranking": spearman_rho,
                    "n_windows": k,
                }
                results.append(result)
            
            print(" done")
    
    # Write results
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")
    
    # Summary: per mode
    print("\n" + "="*70)
    print("Lambda sensitivity summary (per mode)")
    print("="*70)
    for mode in range(1, 5):
        mode_df = df[df["mode"] == mode]
        if len(mode_df) > 0:
            print(f"\nMode {mode}:")
            agg = mode_df.groupby("lambda_index").agg({
                "delta_median_rd_pct": lambda x: f"{np.mean(x):+.1f}%",
                "spearman_rho_model_ranking": "mean",
            })
            print(agg)
    
    print("\nInterpretation:")
    print("  - delta_median_rd_pct: percentage change in degradation")
    print("    negative = less degradation (weight helps)")
    print("  - spearman_rho: correlation of model rankings (λ-perturbed vs baseline)")
    print("    high ρ (≥0.87) = weight affects amplitude not ordering")
    print("    low ρ = weight affects which models fail (model-selection risk)")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lambda weight sensitivity analysis")
    parser.add_argument("--severity", type=str, default="d06")
    parser.add_argument("--npz_root", type=str, default="./TS-Fault_output")
    parser.add_argument("--out", type=str, default="./results_lambda_sensitivity.csv")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Placeholder model list
    model_list = [("LSTM", None), ("GRU", None), ("TimesFM", None)]
    
    run_lambda_sensitivity(
        severity=args.severity,
        npz_root=args.npz_root,
        model_list=model_list,
        out_path=args.out,
        device=device,
    )
