"""
compose_faults.py — Compound-fault evaluation: T_Θ_B ∘ T_Θ_A

Tests pairwise compositions of the four fault modes at matched severity.
Both operators act on the same critical window W★; each retains its own Θ.
The difficulty δ is synchronized: κ_A(Θ_A) = κ_B(Θ_B) = δ_s.

Outputs results_compound.csv with columns:
    model, dataset, modes (e.g. "I+III"), difficulty,
    mse_corrupt, mae_corrupt, mse_clean, mae_clean, r, amplification_psi

Usage:
    python compose_faults.py \
        --pairs I+II I+III I+IV II+III II+IV III+IV \
        --npz_root ./TS-Fault_output \
        --out ./results_compound.csv \
        --gpu 0

Reference (Eq. 19):
    T_Θ = T_Θ_B ∘ T_Θ_A
    Ψ(A∘B) = r(A∘B) / max(r_A, r_B)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

# Assume Mode1.py, Mode2.py, etc. are in the same directory or imported from benchmark
# For this skeleton, we mock the mode operators
try:
    from Mode1 import Mode1
    from Mode2 import Mode2
    from Mode3 import Mode3
    from Mode4 import Mode4
except ImportError:
    Mode1 = Mode2 = Mode3 = Mode4 = None


def load_npz_pair(npz_path):
    """Load clean and corrupted pair from .npz"""
    data = np.load(npz_path)
    return {
        "x_clean": data["x_clean"],
        "x_corrupt": data["x_corrupt"],
        "y_target": data["y_target"],
    }


def compose_operators(x_clean, mode_a, mode_b, theta_a, theta_b, window_idx, difficulty_a, difficulty_b):
    """
    Apply T_Θ_B( T_Θ_A(x_clean) )
    
    Args:
        x_clean: (N, L, C) clean windows
        mode_a, mode_b: mode objects with apply() methods
        theta_a, theta_b: parameter dicts
        window_idx: which window to corrupt (0 to N-1)
        difficulty_a, difficulty_b: difficulty scalars
    
    Returns:
        x_composed: (N, L, C) with both faults applied in sequence
    """
    x_a = x_clean.copy()
    
    # Apply first operator (A)
    x_a[window_idx] = mode_a.apply(
        x_clean[window_idx], theta_a, window_idx=0, difficulty=difficulty_a
    )
    
    # Apply second operator (B) to the already-faulted result
    x_ab = x_a.copy()
    x_ab[window_idx] = mode_b.apply(
        x_a[window_idx], theta_b, window_idx=0, difficulty=difficulty_b
    )
    
    return x_ab


def evaluate_model_on_batch(model, x_clean, x_corrupt, y_true, device="cpu"):
    """
    Compute MSE and MAE on clean vs corrupt inputs predicting same target.
    
    Returns:
        (mse_clean, mae_clean, mse_corrupt, mae_corrupt)
    """
    model.eval()
    with torch.no_grad():
        # Assume model.predict(x) -> y_pred
        x_c_t = torch.from_numpy(x_clean).float().to(device)
        x_cp_t = torch.from_numpy(x_corrupt).float().to(device)
        y_t = torch.from_numpy(y_true).float().to(device)
        
        y_pred_clean = model.predict(x_c_t).cpu().numpy()
        y_pred_corrupt = model.predict(x_cp_t).cpu().numpy()
        y_np = y_true
        
        mse_clean = np.mean((y_pred_clean - y_np) ** 2)
        mae_clean = np.mean(np.abs(y_pred_clean - y_np))
        mse_corrupt = np.mean((y_pred_corrupt - y_np) ** 2)
        mae_corrupt = np.mean(np.abs(y_pred_corrupt - y_np))
    
    return mse_clean, mae_clean, mse_corrupt, mae_corrupt


def compute_amplification(r_compound, r_a, r_b):
    """
    Ψ(A∘B) = r(A∘B) / max(r_A, r_B)
    Ψ ≥ 1 means the compound is at least as bad as the worse constituent.
    """
    if max(r_a, r_b) == 0:
        return 1.0
    return r_compound / max(r_a, r_b)


def run_compound_evaluation(
    npz_root,
    model_list,
    pairs,
    difficulties,
    out_path,
    device="cpu",
):
    """
    Main pipeline for compound-fault evaluation.
    
    Args:
        npz_root: root directory containing Mode*.npz files
        model_list: list of loaded models
        pairs: list of (mode_a, mode_b) tuples to compose
        difficulties: list of severity levels to test (e.g., [0.2, 0.4, ...])
        out_path: output CSV path
        device: torch device
    """
    
    results = []
    
    # Iterate over all datasets, modes, difficulties
    npz_root = Path(npz_root)
    for dataset_dir in sorted(npz_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        
        # Load the single-mode .npz files for this dataset
        single_mode_data = {}
        for mode_num in range(1, 5):
            mode_files = list(dataset_dir.glob(f"*_Mode{mode_num}_d*.npz"))
            if mode_files:
                for npz_file in mode_files:
                    difficulty = npz_file.stem.split("_d")[-1]
                    key = (mode_num, difficulty)
                    single_mode_data[key] = load_npz_pair(npz_file)
        
        # For each pair composition
        for mode_a, mode_b in pairs:
            mode_a_num = int(mode_a)
            mode_b_num = int(mode_b)
            
            # Test at each difficulty level
            for difficulty in difficulties:
                d_str = f"{int(difficulty * 100):02d}"
                
                # Load the single-mode data for both modes at this difficulty
                key_a = (mode_a_num, d_str)
                key_b = (mode_b_num, d_str)
                
                if key_a not in single_mode_data or key_b not in single_mode_data:
                    print(f"  SKIP {dataset_name} Mode{mode_a_num}+{mode_b_num} d{d_str} (missing .npz)")
                    continue
                
                data_a = single_mode_data[key_a]
                data_b = single_mode_data[key_b]
                
                # For this demo, we assume both data_a and data_b have the same x_clean and y_target
                # In practice, you would re-generate the compound at the same window positions
                x_clean = data_a["x_clean"]
                y_target = data_a["y_target"]
                
                # Compose the faults (simplified: just take x_corrupt from mode A, then B)
                # In a real implementation, you'd re-apply the operators with matched Θ
                x_compound = data_b["x_corrupt"].copy()  # Placeholder
                
                print(f"  {dataset_name} Mode{mode_a_num}+{mode_b_num} d{d_str}...", end="")
                
                # Evaluate each model
                for model_name, model in model_list:
                    mse_c, mae_c, mse_cp, mae_cp = evaluate_model_on_batch(
                        model, x_clean, x_compound, y_target, device=device
                    )
                    
                    r = mse_cp / mse_c if mse_c > 0 else 1.0
                    # Single-mode ratios for Ψ (from single_mode_data)
                    r_a = np.mean(data_a["x_corrupt"]) / np.mean(data_a["x_clean"]) if np.mean(data_a["x_clean"]) > 0 else 1.0
                    r_b = np.mean(data_b["x_corrupt"]) / np.mean(data_b["x_clean"]) if np.mean(data_b["x_clean"]) > 0 else 1.0
                    psi = compute_amplification(r, r_a, r_b)
                    
                    results.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "modes": f"{mode_a_num}+{mode_b_num}",
                        "difficulty": difficulty,
                        "d_str": d_str,
                        "mse_corrupt": mse_cp,
                        "mae_corrupt": mae_cp,
                        "mse_clean": mse_c,
                        "mae_clean": mae_c,
                        "robust_ratio": r,
                        "amplification_psi": psi,
                    })
                
                print(" done")
    
    # Write results
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")
    
    # Summary statistics
    print("\n" + "="*70)
    print("Compound-fault summary (Ψ = amplification factor)")
    print("="*70)
    summary = df.groupby("modes").agg({
        "robust_ratio": "mean",
        "amplification_psi": "mean"
    }).round(2)
    print(summary)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compound-fault evaluation")
    parser.add_argument("--npz_root", type=str, default="./TS-Fault_output")
    parser.add_argument("--out", type=str, default="./results_compound.csv")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["I+II", "I+III", "I+IV", "II+III", "II+IV", "III+IV"],
        help="Mode pairs to compose (e.g., 'I+II' 'III+IV')",
    )
    parser.add_argument("--difficulties", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", type=str, nargs="+", default=["LSTM", "GRU", "TimesFM"])
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load models (skeleton)
    model_list = [(name, None) for name in args.models]  # Placeholder
    
    # Parse pairs
    pairs = []
    for p in args.pairs:
        parts = p.replace("+", " ").split()
        pairs.append((int(parts[0]), int(parts[1])))
    
    run_compound_evaluation(
        args.npz_root,
        model_list,
        pairs,
        args.difficulties,
        args.out,
        device=device,
    )
