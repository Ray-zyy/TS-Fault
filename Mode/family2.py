
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Literal, Tuple, List, Optional

from joltbench.utils.window_selector import (
    select_critical_windows,
    compute_dominant_period,
)


@dataclass
class SharedShockParams:
    shock_type: Literal['impulse', 'burst', 'shift'] = 'burst'
    amplitude:  float = 2.0
    width:      int   = 5

@dataclass
class FractureParams:
    """Per-variable fracture: Δτ_j, Δg_j."""
    delta_tau_max: int   = 3      # max lag shift
    delta_g_max:   float = 1.5    # max gain perturbation
    allow_sign_flip: bool = True  # g'_j can become negative

@dataclass
class SceneParamsF2:
    shock:      SharedShockParams = field(default_factory=SharedShockParams)
    fracture:   FractureParams    = field(default_factory=FractureParams)
    variable_subset:  list = field(default_factory=list)   # S
    root_channel:     int  = 0
    window_start:     int  = 0
    window_end:       int  = 0
    lag_shifts:       dict = field(default_factory=dict)   # j → Δτ_j
    gain_shifts:      dict = field(default_factory=dict)   # j → Δg_j
    shock_amplitude:  float = 0.0
    difficulty:       float = 0.0

    def to_dict(self):
        return asdict(self)

def _local_lead_lag(xi: np.ndarray, xj: np.ndarray,
                    tau_max: int = 5) -> float:
    """R_ij(W) = max_{τ} |Corr(x_t^i, x_{t-τ}^j)|."""
    best = 0.0
    n = len(xi)
    for tau in range(-tau_max, tau_max + 1):
        if tau >= 0:
            a, b = xi[:n-tau] if tau else xi, xj[tau:] if tau else xj
        else:
            a, b = xi[-tau:], xj[:n+tau]
        if len(a) < 3:
            continue
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            continue
        r = abs(float(np.corrcoef(a, b)[0, 1]))
        if np.isnan(r):
            continue
        best = max(best, r)
    return best


def _select_variable_subset(
    X: np.ndarray, s: int, e: int,
    m: int = 3, tau_max: int = 5,
    neighbourhood_r: int = 10
) -> Tuple[List[int], int]:
    T, C = X.shape
    r = neighbourhood_r
    left  = X[max(0, s-r): s]
    right = X[e: min(T, e+r)]
    ctx   = np.concatenate([left, right], axis=0) if (len(left) and len(right)) \
            else X[s:e]
    if len(ctx) < 4:
        ctx = X[max(0, s-20): e+20]

    G = np.zeros(C)
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            G[i] += _local_lead_lag(ctx[:, i], ctx[:, j], tau_max)

    m = min(m, C)
    top_idx = np.argsort(G)[::-1][:m].tolist()
    root = top_idx[0]
    return top_idx, root

def _make_shock(W: int, sp: SharedShockParams) -> np.ndarray:
    b = np.zeros(W)
    c = W // 2
    if sp.shock_type == 'impulse':
        b[c] = sp.amplitude
    elif sp.shock_type == 'burst':
        sigma = max(1.0, sp.width / 2.0)
        t = np.arange(W)
        b = sp.amplitude * np.exp(-0.5 * ((t - c) / sigma) ** 2)
    elif sp.shock_type == 'shift':
        b[:] = sp.amplitude
    return b

def _estimate_normal_response(
    root_series: np.ndarray, follower_series: np.ndarray,
    tau_max: int = 5
) -> Tuple[int, float]:
    best_tau, best_r = 0, 0.0
    n = len(root_series)
    for tau in range(-tau_max, tau_max + 1):
        if tau >= 0:
            a = root_series[:n - tau] if tau else root_series
            b = follower_series[tau:] if tau else follower_series
        else:
            a = root_series[-tau:]
            b = follower_series[:n + tau]
        if len(a) < 3:
            continue
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if np.isnan(r):
            continue
        if abs(r) > abs(best_r):
            best_r, best_tau = r, tau

    std_root = float(np.std(root_series)) + 1e-8
    std_fol  = float(np.std(follower_series))
    g_star = std_fol / std_root
    return best_tau, g_star


def apply_family2(
    X: np.ndarray,
    # Window selection
    window_size: int = 24,
    top_k: int = 3,
    window_lambdas: tuple = (0.25, 0.25, 0.25, 0.25),
    horizon: int = 10,
    period: int = None,
    random_select: bool = True,
    # Variable subset
    subset_size: int = 3,
    tau_max: int = 5,
    neighbourhood_r: int = 10,
    # Shared shock
    shock_type: Literal['impulse', 'burst', 'shift', 'random'] = 'random',
    shock_amplitude: float = 2.0,
    shock_width: int = 5,
    adaptive_amplitude: bool = True,
    # Fracture
    delta_tau_max: int = 3,
    delta_g_max:   float = 1.5,
    allow_sign_flip: bool = True,
    # Difficulty betas
    difficulty_betas: tuple = (0.25, 0.25, 0.25, 0.25),
    random_seed: int = None,
) -> Tuple[np.ndarray, SceneParamsF2]:
    rng = np.random.default_rng(random_seed)
    assert X.ndim == 2, "Family 2 requires multivariate input X (T, C)"
    T, C = X.shape
    X_out = X.copy().astype(float)

    if period is None:
        period = compute_dominant_period(X[:, 0])

    windows = select_critical_windows(
        X[:, 0], window_size=window_size, top_k=top_k,
        lambdas=window_lambdas, horizon=horizon, period=period
    )
    if not windows:
        s = T // 2 - window_size // 2
        e = s + window_size
        win_info = dict(start=s, end=e, score=0.0,
                        scp_n=0.0, sper_n=0.0, svar_n=0.0, spred_n=0.0)
    else:
        win_info = rng.choice(windows) if random_select else windows[0]

    s, e = win_info['start'], win_info['end']
    W = e - s

    subset, root = _select_variable_subset(
        X, s, e, m=min(subset_size, C),
        tau_max=tau_max, neighbourhood_r=neighbourhood_r
    )

    if shock_type == 'random':
        shock_type = rng.choice(['impulse', 'burst', 'shift'])

    if adaptive_amplitude:
        local_std = float(np.std(X[s:e, root])) + 1e-8
        ampl = shock_amplitude * local_std
    else:
        ampl = shock_amplitude

    sp = SharedShockParams(shock_type=shock_type,    # type: ignore[arg-type]
                           amplitude=ampl, width=min(shock_width, W//2))
    u_t = _make_shock(W, sp)

    g_true = {i: float(rng.uniform(0.3, 1.8)) for i in subset}
    g_true[root] = 1.0

    for i in subset:
        X_out[s:e, i] = X[s:e, i] + g_true[i] * u_t

    fp = FractureParams(delta_tau_max=delta_tau_max,
                        delta_g_max=delta_g_max,
                        allow_sign_flip=allow_sign_flip)

    lag_shifts  = {}
    gain_shifts = {}

    for j in subset:
        if j == root:
            continue

        tau_star, g_star = _estimate_normal_response(
            X[s:e, root], X[s:e, j], tau_max=tau_max
        )
        Δτ = int(rng.integers(-delta_tau_max, delta_tau_max + 1))
        Δg = float(rng.uniform(-delta_g_max, delta_g_max))
        lag_shifts[j]  = Δτ
        gain_shifts[j] = Δg

        tau_prime = tau_star + Δτ
        g_prime   = g_star + Δg
        if allow_sign_flip and abs(Δg) > 0.8 * abs(g_star):
            g_prime *= -1.0

        X_out[s:e, j] = X[s:e, j]   # reset
        for t_local in range(W):
            t_src = t_local - tau_prime
            t_src = int(np.clip(t_src, 0, W - 1))
            X_out[s + t_local, j] += g_prime * u_t[t_src]
    β1, β2, β3, β4 = difficulty_betas

    D_shock = min(1.0, abs(ampl) / (abs(ampl) + 1.0))
    D_lag   = min(1.0, np.mean([abs(v) for v in lag_shifts.values()]) /
                 (delta_tau_max + 1e-4)) if lag_shifts else 0.0
    D_gain  = min(1.0, np.mean([abs(v) for v in gain_shifts.values()]) /
                 (delta_g_max + 1e-4)) if gain_shifts else 0.0
    D_scale = min(1.0, len(subset) / C)

    difficulty = float(β1*D_shock + β2*D_lag + β3*D_gain + β4*D_scale)

    params = SceneParamsF2(
        shock=sp, fracture=fp,
        variable_subset=subset, root_channel=root,
        window_start=s, window_end=e,
        lag_shifts={str(k): v for k, v in lag_shifts.items()},
        gain_shifts={str(k): v for k, v in gain_shifts.items()},
        shock_amplitude=ampl,
        difficulty=difficulty,
    )
    return X_out, params
