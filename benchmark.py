
from __future__ import annotations
import numpy as np
import json
import os
from typing import Literal, Optional, List, Dict, Any, Tuple, Union
from dataclasses import asdict

from joltbench.utils.dataset_loader import load_dataset, make_windows, StandardScaler
from joltbench.families.family1 import apply_family1, apply_family1_multivariate, SceneParamsF1
from joltbench.families.family2 import apply_family2, SceneParamsF2
from joltbench.families.family3 import apply_family3, SceneParamsF3
from joltbench.families.family4 import apply_family4, SceneParamsF4


def _difficulty_to_f1_kwargs(d: float, window_size: int) -> dict:
    """Map scalar difficulty ∈ [0,1] to Family-1 parameters."""
    return dict(
        amplitude      = 0.5 + d * 4.5,          # 0.5 → 5.0
        event_width    = max(2, int(3 + d * 10)), # 3 → 13
        warp_delta     = d * 5.0,                 # 0 → 5
        warp_alpha     = 1.0 + d * 1.5,           # 1.0 → 2.5
        warp_noise_std = d * 0.5,                 # 0 → 0.5
        coupling_type  = ('independent' if d < 0.4
                          else 'shock_centered' if d < 0.7
                          else 'amplitude_coupled'),
    )

def _difficulty_to_f2_kwargs(d: float) -> dict:
    return dict(
        shock_amplitude = 0.5 + d * 4.0,
        delta_tau_max   = max(1, int(1 + d * 6)),
        delta_g_max     = 0.5 + d * 2.5,
        allow_sign_flip = d > 0.5,
        subset_size     = max(2, int(2 + d * 4)),
    )

def _difficulty_to_f3_kwargs(d: float) -> dict:
    return dict(
        delta_beta   = d * 0.15,
        delta_b      = d * 2.0,
        a_s          = 1.0 + d * 1.0,
        period_ratio = 1.0 + d * 0.5,
        c_r          = 1.0 + d * 1.5,
        gamma        = max(1.0, 10.0 - d * 8.0),  # sharp at high d
        a0   = -5.0 + d * 4.0,     # more missing at high d
        a1   = 1.0 + d * 4.0,
        a2   = 0.5 + d * 1.5,
        geom_q = 0.1 + d * 0.5,    # longer blocks at high d
    )

def _difficulty_to_f4_kwargs(d: float) -> dict:
    return dict(
        n_roots    = max(1, int(1 + d * 2)),
        n_down     = max(1, int(1 + d * 4)),
        fault_b0    = d * 0.5,
        fault_kappa = d * 0.1,
        gamma_range = (0.3 + d * 0.3, 0.7 + d * 0.8),
        delay_range = (1, max(2, int(1 + d * 5))),
        sec_eta1    = 0.5 + d * 2.5,
        sec_geom_q  = 0.1 + d * 0.4,
    )


class JoltBench:

    def __init__(
        self,
        dataset: str = "ETTh1",
        seq_len: int = 336,
        pred_len: int = 96,
        stride: int = 24,
        max_len: int = None,
        normalise: bool = True,
        cache_dir: str = "/tmp/joltbench_data",
    ):
        self.dataset_name = dataset
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.stride   = stride

        print(f"[JoltBench] Loading dataset: {dataset}")
        self.data, self.meta = load_dataset(
            dataset, max_len=max_len,
            normalise=normalise, cache_dir=cache_dir
        )
        self.X_win, self.Y_win = make_windows(
            self.data, seq_len=seq_len, pred_len=pred_len, stride=stride
        )
        T, C = self.data.shape
        print(f"[JoltBench] {dataset}: T={T}, C={C}, "
              f"windows={len(self.X_win)}, seq_len={seq_len}, pred_len={pred_len}")

    def apply(
        self,
        family: Literal['family1','family2','family3','family4'],
        sample_idx: int = 0,
        difficulty: float = 0.5,
        channel: Optional[int] = None,   # for f1 on one channel
        extra_kwargs: dict = None,
        random_seed: int = None,
    ) -> Tuple[np.ndarray, Any]:
        assert 0 <= sample_idx < len(self.X_win), \
            f"sample_idx {sample_idx} out of range [0, {len(self.X_win)-1}]"

        x_in = self.X_win[sample_idx].copy()   # (seq_len, C)
        d = float(np.clip(difficulty, 0.0, 1.0))
        kw = extra_kwargs or {}

        if family == 'family1':
            base = _difficulty_to_f1_kwargs(d, self.seq_len)
            base.update(kw)
            if channel is not None:
                x_c, params = apply_family1(x_in[:, channel], random_seed=random_seed, **base)
                x_in[:, channel] = x_c
                return x_in, params
            else:
                X_c, params_list = apply_family1_multivariate(
                    x_in, random_seed=random_seed, **base
                )
                return X_c, params_list

        elif family == 'family2':
            base = _difficulty_to_f2_kwargs(d)
            base.update(kw)
            return apply_family2(x_in, random_seed=random_seed, **base)

        elif family == 'family3':
            base = _difficulty_to_f3_kwargs(d)
            base.update(kw)
            x_obs, z_true, mask, params = apply_family3(
                x_in, random_seed=random_seed, **base
            )
            return x_obs, params

        elif family == 'family4':
            base = _difficulty_to_f4_kwargs(d)
            base.update(kw)
            return apply_family4(x_in, random_seed=random_seed, **base)

        else:
            raise ValueError(f"Unknown family '{family}'")


    def generate_benchmark(
        self,
        families: List[str] = None,
        n_samples_per_family: int = 50,
        difficulty_levels: List[float] = None,
        random_seed: int = 42,
        save_dir: str = None,
    ) -> Dict[str, Any]:
        if families is None:
            families = ['family1', 'family2', 'family3', 'family4']
        if difficulty_levels is None:
            difficulty_levels = [0.3, 0.6, 0.9]

        n_available = len(self.X_win)
        rng = np.random.default_rng(random_seed)
        results = {}

        for family in families:
            for diff in difficulty_levels:
                key = f"{family}_d{int(diff*10):02d}"
                print(f"[JoltBench] Generating {key} "
                      f"(n={n_samples_per_family}) ...")

                idxs = rng.choice(n_available, size=min(n_samples_per_family, n_available),
                                  replace=False).tolist()

                x_corrupts, x_cleans, y_targets, params_list = [], [], [], []

                for sample_idx in idxs:
                    seed_i = int(rng.integers(0, 2**31))
                    try:
                        x_c, params = self.apply(
                            family, sample_idx=sample_idx,
                            difficulty=diff, random_seed=seed_i
                        )
                        # params can be list (f1 multi) or single
                        if isinstance(params, list):
                            p_dict = [asdict(p) if hasattr(p, '__dataclass_fields__') else p
                                      for p in params]
                        else:
                            p_dict = asdict(params) if hasattr(params, '__dataclass_fields__') else params

                        x_corrupts.append(x_c if x_c.ndim == 2
                                          else x_c[:, None])
                        x_cleans.append(self.X_win[sample_idx])
                        y_targets.append(self.Y_win[sample_idx])
                        params_list.append(p_dict)
                    except Exception as ex:
                        print(f"  [warn] sample {sample_idx} failed: {ex}")

                if not x_corrupts:
                    continue

                results[key] = {
                    "x_corrupt":   np.stack(x_corrupts),
                    "x_clean":     np.stack(x_cleans),
                    "y_target":    np.stack(y_targets),
                    "family":      family,
                    "difficulty":  diff,
                    "params":      params_list,
                }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for key, val in results.items():
                np.savez_compressed(
                    os.path.join(save_dir, f"{self.dataset_name}_{key}.npz"),
                    x_corrupt=val['x_corrupt'],
                    x_clean=val['x_clean'],
                    y_target=val['y_target'],
                )
                with open(os.path.join(save_dir,
                          f"{self.dataset_name}_{key}_params.json"), 'w') as f:
                    json.dump({"family": val["family"],
                               "difficulty": val["difficulty"],
                               "params": val["params"]}, f, indent=2,
                              default=_json_default)
            print(f"[JoltBench] Saved to {save_dir}")

        return results


    def summarise(self, results: Dict[str, Any]) -> None:
        """Print a summary table of benchmark results."""
        print(f"\n{'='*65}")
        print(f"  JoltBench Summary — {self.dataset_name}")
        print(f"{'='*65}")
        print(f"  {'Key':<30} {'N':>5}  {'Diff':>6}  {'Avg.D_score':>12}")
        print(f"  {'-'*55}")
        for key, val in sorted(results.items()):
            n = len(val['params'])
            d = val['difficulty']
            # Extract difficulty scores from params
            d_scores = []
            for p in val['params']:
                if isinstance(p, dict) and 'difficulty' in p:
                    d_scores.append(p['difficulty'])
                elif isinstance(p, list):
                    for pi in p:
                        if isinstance(pi, dict) and 'difficulty' in pi:
                            d_scores.append(pi['difficulty'])
            avg_d = np.mean(d_scores) if d_scores else float('nan')
            print(f"  {key:<30} {n:>5}  {d:>6.2f}  {avg_d:>12.4f}")
        print(f"{'='*65}\n")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
