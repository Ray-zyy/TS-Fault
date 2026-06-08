from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Literal, Tuple, Optional

from joltbench.utils.window_selector import compute_dominant_period

@dataclass
class RegimeParams:
    """G — what changed in the new regime."""
    delta_beta:   float = 0.05     
    delta_b:      float = 0.5      
    a_s:          float = 1.3      
    period_ratio: float = 1.0      
    phase_shift:  float = 0.0     
    c_r:          float = 1.2      
    gamma:        float = 5.0      

@dataclass
class MissingnessParams:
    a0:  float = -3.0   
    a1:  float =  2.5   
    a2:  float =  1.2   
    a3:  float =  1.0 
    c_i: float =  0.0   
    geom_q: float = 0.3 
    fill_method: Literal['ffill', 'zero', 'mean'] = 'ffill'

@dataclass
class RegimeContextParams:
    tau:          int   = 0     
    L:            int   = 0    
    near_forecast: bool = False 
    transition_strength: float = 0.0   

@dataclass
class CouplingParamsF3:
    """U — how regime shift and missingness are coupled."""
    dominant_factor: Literal['transition', 'variance', 'residual'] = 'transition'
    coupling_strength: float = 0.0

@dataclass
class SceneParamsF3:
    regime:    RegimeParams       = field(default_factory=RegimeParams)
    missing:   MissingnessParams  = field(default_factory=MissingnessParams)
    context:   RegimeContextParams = field(default_factory=RegimeContextParams)
    coupling:  CouplingParamsF3   = field(default_factory=CouplingParamsF3)
    missing_rate: float = 0.0
    difficulty: float  = 0.0

    def to_dict(self):
        return asdict(self)


def _score_trend_change(x: np.ndarray, tau: int) -> float:
    hw = max(tau // 4, 5)
    left  = x[max(0, tau - hw): tau]
    right = x[tau: min(len(x), tau + hw)]
    if len(left) < 2 or len(right) < 2:
        return 0.0
    slope = lambda seg: np.polyfit(np.arange(len(seg)), seg, 1)[0]
    return abs(float(slope(right) - slope(left)))


def _score_season_change(x: np.ndarray, tau: int, period: int) -> float:
    p = max(period, 2)
    before = x[max(0, tau - p): tau]
    after  = x[tau: min(len(x), tau + p)]
    if len(before) < 2 or len(after) < 2:
        return 0.0
    amp_b = float(np.ptp(before))
    amp_a = float(np.ptp(after))
    return abs(amp_a - amp_b) / (max(amp_b, amp_a) + 1e-8)


def _score_vol_change(x: np.ndarray, tau: int) -> float:
    hw = max(tau // 4, 5)
    left  = x[max(0, tau - hw): tau]
    right = x[tau: min(len(x), tau + hw)]
    if len(left) < 2 or len(right) < 2:
        return 0.0
    return abs(float(np.std(right) - np.std(left)))


def _score_pred_sensitivity(x: np.ndarray, tau: int,
                             horizon: int = 10) -> float:
    L = len(x)
    return float(np.exp(-(L - tau) / max(horizon, 1)))


def find_transition_point(
    x: np.ndarray,
    period: int = None,
    horizon: int = 10,
    etas: tuple = (0.25, 0.25, 0.25, 0.25),
    margin: int = 10,    # ignore edges
    top_k: int = 5,
    random_select: bool = True,
    rng=None,
) -> Tuple[int, float]:
    if rng is None:
        rng = np.random.default_rng()
    L = len(x)
    η1, η2, η3, η4 = etas
    if period is None:
        period = compute_dominant_period(x)

    candidates = []
    for tau in range(margin, L - margin):
        Qt = (η1 * _score_trend_change(x, tau) +
              η2 * _score_season_change(x, tau, period) +
              η3 * _score_vol_change(x, tau) +
              η4 * _score_pred_sensitivity(x, tau, horizon))
        candidates.append((tau, Qt))

    if not candidates:
        return L // 2, 0.0

    # Normalise and pick from top-k
    scores = np.array([c[1] for c in candidates])
    scores_n = scores / (scores.max() + 1e-10)
    top_idx = np.argsort(scores_n)[::-1][:top_k]
    pick = rng.choice(top_idx) if random_select else top_idx[0]
    tau, q = candidates[pick]
    return tau, float(q)


def _simple_stl(x: np.ndarray, period: int):
    L = len(x)
    P = max(period, 2)

    # Trend: centred moving average
    hw = P // 2
    T_t = np.convolve(x, np.ones(P) / P, mode='same')
    # Fix edges
    for i in range(hw):
        T_t[i] = np.mean(x[:i + hw + 1])
    for i in range(L - hw, L):
        T_t[i] = np.mean(x[i - hw:])

    detrended = x - T_t
    S_t = np.zeros(L)
    for phase in range(P):
        idxs = np.arange(phase, L, P)
        avg  = float(np.mean(detrended[idxs]))
        S_t[idxs] = avg

    R_t = x - T_t - S_t
    return T_t, S_t, R_t


def _build_new_regime(
    x: np.ndarray, tau: int, rp: RegimeParams, period: int
) -> np.ndarray:

    L = len(x)
    T_t, S_t, R_t = _simple_stl(x, period)


    t_arr = np.arange(L, dtype=float)


    T_new = T_t.copy()
    T_new[tau:] += rp.delta_beta * (t_arr[tau:] - tau) + rp.delta_b

    P  = max(period, 2)
    Pp = max(1, int(round(P * rp.period_ratio)))
    S_new = S_t.copy()
    for t in range(tau, L):
        phi_t = tau + (P / Pp) * (t - tau) + rp.phase_shift
        phi_t_mod = phi_t % P
        lo = int(np.floor(phi_t_mod)) % P
        hi = (lo + 1) % P
        # sample S_t at fractional phase
        idxs_lo = np.arange(lo, L, P)
        idxs_hi = np.arange(hi, L, P)
        s_lo = float(np.mean(S_t[idxs_lo])) if len(idxs_lo) else 0.0
        s_hi = float(np.mean(S_t[idxs_hi])) if len(idxs_hi) else 0.0
        frac = phi_t_mod - lo
        S_new[t] = rp.a_s * ((1 - frac) * s_lo + frac * s_hi)


    R_new = R_t.copy()
    R_new[tau:] *= rp.c_r

    z_ideal = T_new + S_new + R_new


    omega = 1.0 / (1.0 + np.exp(-(t_arr - tau) / max(rp.gamma, 0.1)))
    z_t = (1 - omega) * x + omega * z_ideal
    return z_t



def _sigmoid(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(u, -30, 30)))


def _build_missing_mask(
    z: np.ndarray, tau: int,
    mp: MissingnessParams,
    rng,
) -> np.ndarray:

    L = len(z)
    T_arr, S_arr, R_arr = _simple_stl(z, max(4, L // 8))
    omega = 1.0 / (1.0 + np.exp(-(np.arange(L, dtype=float) - tau) / 5.0))

    h_t = 4.0 * omega * (1.0 - omega)


    v_t = np.array([
        float(np.std(z[max(0, t-5): t+5]))
        for t in range(L)
    ])
    v_max = v_t.max() + 1e-8
    v_t /= v_max


    mad = float(np.median(np.abs(R_arr - np.median(R_arr)))) + 1e-8
    r_t = np.abs(R_arr) / mad

    logit = (mp.a0
             + mp.a1 * h_t
             + mp.a2 * v_t
             + mp.a3 * np.tanh(r_t)   # bounded
             + mp.a4 if hasattr(mp, 'a4') else mp.a0)
    logit = mp.a0 + mp.a1*h_t + mp.a2*v_t + mp.a3*np.tanh(r_t) + mp.c_i

    p_start = _sigmoid(logit)

    mask = np.ones(L, dtype=float)
    t = 0
    while t < L:
        if rng.random() < p_start[t]:
            length = int(rng.geometric(1.0 - mp.geom_q))
            length = max(1, min(length, L - t))
            mask[t: t + length] = 0.0
            t += length
        else:
            t += 1
    return mask


def _fill_missing(z: np.ndarray, mask: np.ndarray,
                  method: str = 'ffill') -> np.ndarray:
    """Fill masked positions."""
    x_obs = z.copy()
    if method == 'ffill':
        last = float(z[0])
        for t in range(len(z)):
            if mask[t] == 0:
                x_obs[t] = last
            else:
                last = float(z[t])
    elif method == 'zero':
        x_obs[mask == 0] = 0.0
    elif method == 'mean':
        m = float(z[mask == 1].mean()) if mask.sum() > 0 else 0.0
        x_obs[mask == 0] = m
    return x_obs


def apply_family3(
    X: np.ndarray,
    period: int = None,
    horizon: int = 10,
    transition_etas: tuple = (0.25, 0.25, 0.25, 0.25),
    margin: int = 10,
    top_k: int = 5,
    random_select: bool = True,
    delta_beta:   float = 0.05,
    delta_b:      float = 0.5,
    a_s:          float = 1.3,
    period_ratio: float = 1.0,
    phase_shift:  float = 0.0,
    c_r:          float = 1.2,
    gamma:        float = 5.0,
    a0:  float = -3.0,
    a1:  float =  2.5,
    a2:  float =  1.0,
    a3:  float =  0.8,
    c_i: float =  0.0,
    geom_q:      float = 0.3,
    fill_method: Literal['ffill', 'zero', 'mean'] = 'ffill',
    # Difficulty betas
    difficulty_betas: tuple = (0.25, 0.25, 0.25, 0.25),
    random_seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SceneParamsF3]:

    rng = np.random.default_rng(random_seed)

    univariate = X.ndim == 1
    if univariate:
        X = X[:, None]
    T, C = X.shape

    if period is None:
        period = compute_dominant_period(X[:, 0])

    tau, q_score = find_transition_point(
        X[:, 0], period=period, horizon=horizon,
        etas=transition_etas, margin=margin,
        top_k=top_k, random_select=random_select, rng=rng
    )


    rp = RegimeParams(delta_beta=delta_beta, delta_b=delta_b,
                      a_s=a_s, period_ratio=period_ratio,
                      phase_shift=phase_shift, c_r=c_r, gamma=gamma)
    Z = np.zeros_like(X, dtype=float)
    for c in range(C):
        Z[:, c] = _build_new_regime(X[:, c].astype(float), tau, rp, period)

    mp = MissingnessParams(a0=a0, a1=a1, a2=a2, a3=a3, c_i=c_i,
                           geom_q=geom_q, fill_method=fill_method)
    Mask = np.ones((T, C), dtype=float)
    X_obs = Z.copy()
    for c in range(C):
        m = _build_missing_mask(Z[:, c], tau, mp, rng)
        Mask[:, c] = m
        X_obs[:, c] = _fill_missing(Z[:, c], m, fill_method)

    β1, β2, β3, β4 = difficulty_betas

    D_regime = float(np.clip(
        abs(delta_beta)*10 + abs(delta_b)*0.5 +
        abs(a_s - 1.0) + abs(period_ratio - 1.0) + abs(c_r - 1.0)*0.5,
        0, 1.0
    ))

    miss_rate = float(1.0 - Mask.mean())
    D_missing = min(1.0, miss_rate * 2.0)

    D_proximity = float(np.exp(-(T - tau) / max(horizon, 1)))

    coeff_sum = abs(a1) + abs(a2) + abs(a3) + 1e-8
    D_coupling = abs(a1) / coeff_sum

    difficulty = float(β1*D_regime + β2*D_missing +
                       β3*D_proximity + β4*D_coupling)
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    dom_factor: Literal['transition', 'variance', 'residual']
    if abs(a1) >= max(abs(a2), abs(a3)):
        dom_factor = 'transition'
    elif abs(a2) >= abs(a3):
        dom_factor = 'variance'
    else:
        dom_factor = 'residual'

    params = SceneParamsF3(
        regime=rp,
        missing=mp,
        context=RegimeContextParams(
            tau=tau, L=T,
            near_forecast=D_proximity > 0.5,
            transition_strength=q_score,
        ),
        coupling=CouplingParamsF3(
            dominant_factor=dom_factor,
            coupling_strength=D_coupling,
        ),
        missing_rate=miss_rate,
        difficulty=difficulty,
    )

    if univariate:
        return X_obs[:, 0], Z[:, 0], Mask[:, 0], params

    return X_obs, Z, Mask, params
