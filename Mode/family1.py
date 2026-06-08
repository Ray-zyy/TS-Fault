from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Tuple

from joltbench.utils.window_selector import (
    select_critical_windows,
    compute_dominant_period,
    score_changepoint, score_periodic, score_variance, score_prediction,
)

@dataclass
class EventParams:
    event_type: Literal['impulse', 'burst', 'shift'] = 'burst'
    amplitude:  float = 2.0          # α  (overall strength multiplier)
    width:      int   = 5            # for burst: half-width; for shift: full window
    sign:       float = 1.0          # +1 positive shock, -1 negative

@dataclass
class WarpParams:
    warp_type: Literal['translate', 'scale', 'nonlinear'] = 'translate'
    delta:     float = 2.0   # translate: shift steps; nonlinear: amplitude
    alpha:     float = 1.3   # scale factor  (>1 stretch, <1 compress)
    noise_std: float = 0.1   # ξ_t std for nonlinear warp

@dataclass
class ContextParams:
    """C_ctx — why this window was chosen."""
    near_changepoint: float = 0.0   # Scp score
    near_peak_valley: float = 0.0   # Sper score
    variance_energy:  float = 0.0   # Svar score
    pred_sensitivity: float = 0.0   # Spred score
    window_start: int = 0
    window_end:   int = 0

@dataclass
class CouplingParams:
    """R — relationship between shock and warp."""
    coupling_type: Literal['independent', 'shock_centered',
                           'amplitude_coupled'] = 'independent'
    coupling_strength: float = 0.0

@dataclass
class SceneParamsF1:
    """Four-tuple S = (E, W_param, C_ctx, R_couple) for Family 1."""
    event:    EventParams   = field(default_factory=EventParams)
    warp:     WarpParams    = field(default_factory=WarpParams)
    context:  ContextParams = field(default_factory=ContextParams)
    coupling: CouplingParams = field(default_factory=CouplingParams)
    difficulty: float = 0.0

    def to_dict(self):
        return asdict(self)

def _make_perturbation(window_size: int, ep: EventParams) -> np.ndarray:
    """Generate unnormalised perturbation shape b_t ∈ R^{window_size}."""
    W = window_size
    b = np.zeros(W)

    if ep.event_type == 'impulse':
        c = W // 2
        hw = max(1, ep.width // 2)
        b[max(0, c-1): min(W, c+2)] = ep.sign * ep.amplitude

    elif ep.event_type == 'burst':
        c = W // 2
        sigma = max(1.0, ep.width / 2.0)
        t = np.arange(W)
        b = ep.sign * ep.amplitude * np.exp(-0.5 * ((t - c) / sigma) ** 2)

    elif ep.event_type == 'shift':
        b[:] = ep.sign * ep.amplitude

    return b

def _apply_warp(b: np.ndarray, s: int, wp: WarpParams) -> np.ndarray:
    W = len(b)
    t = np.arange(W, dtype=float)

    if wp.warp_type == 'translate':
        delta = int(round(wp.delta))
        warped = np.empty(W)
        for i in range(W):
            src = int(i - delta)
            src = np.clip(src, 0, W - 1)
            warped[i] = b[src]
        return warped

    elif wp.warp_type == 'scale':
        α = max(0.1, wp.alpha)
        warped = np.empty(W)
        for i in range(W):
            src = s + α * (i - 0)         # φ(t) = s + α·(t-s)
            src_local = src - s
            src_local = np.clip(src_local, 0, W - 1)
            lo = int(np.floor(src_local))
            hi = min(lo + 1, W - 1)
            frac = src_local - lo
            warped[i] = (1 - frac) * b[lo] + frac * b[hi]
        return warped

    elif wp.warp_type == 'nonlinear':
        Δ = wp.delta
        ξ = np.random.randn(W) * wp.noise_std
        φ = t + Δ * np.sin(2 * np.pi * t / W) + ξ
        warped = np.empty(W)
        for i in range(W):
            src = np.clip(φ[i], 0, W - 1)
            lo = int(np.floor(src))
            hi = min(lo + 1, W - 1)
            frac = src - lo
            warped[i] = (1 - frac) * b[lo] + frac * b[hi]
        return warped

    return b  


def _difficulty(ep: EventParams, wp: WarpParams,
                ctx: ContextParams, cp: CouplingParams,
                betas: Tuple[float,float,float,float] = (0.25, 0.25, 0.25, 0.25)) -> float:
    β1, β2, β3, β4 = betas

    # D_event: amplitude × (1 + event-type factor)
    type_factor = {'impulse': 1.0, 'burst': 0.7, 'shift': 0.5}
    D_event = abs(ep.amplitude) * type_factor.get(ep.event_type, 0.7)

    # D_warp: strength of warp
    if wp.warp_type == 'translate':
        D_warp = min(1.0, abs(wp.delta) / 5.0)
    elif wp.warp_type == 'scale':
        D_warp = min(1.0, abs(wp.alpha - 1.0))
    else:
        D_warp = min(1.0, abs(wp.delta) / 3.0 + wp.noise_std)

    D_context = (ctx.near_changepoint + ctx.near_peak_valley +
                 ctx.variance_energy + ctx.pred_sensitivity) / 4.0


    coupling_factor = {'independent': 0.0, 'shock_centered': 0.5,
                       'amplitude_coupled': 1.0}
    D_coupling = coupling_factor.get(cp.coupling_type, 0.0) * cp.coupling_strength

    D = β1*D_event + β2*D_warp + β3*D_context + β4*D_coupling
    return float(np.clip(D, 0.0, 1.0))


def apply_family1(
    x: np.ndarray,
    # Window selection
    window_size: int = 24,
    top_k: int = 3,
    window_lambdas: tuple = (0.25, 0.25, 0.25, 0.25),
    horizon: int = 10,
    period: int = None,
    random_select: bool = True,        
    event_type: Literal['impulse','burst','shift','random'] = 'random',
    amplitude: float = 2.0,
    event_width: int = 5,
    event_sign: float = None,          
    warp_type: Literal['translate','scale','nonlinear','random'] = 'random',
    warp_delta: float = 2.0,
    warp_alpha: float = 1.3,
    warp_noise_std: float = 0.1,
    # Coupling
    coupling_type: Literal['independent','shock_centered','amplitude_coupled'] = 'independent',
    difficulty_betas: tuple = (0.25, 0.25, 0.25, 0.25),
    # Normalise amplitude relative to local std
    adaptive_amplitude: bool = True,
    random_seed: int = None,
) -> Tuple[np.ndarray, SceneParamsF1]:
    rng = np.random.default_rng(random_seed)

    assert x.ndim == 1,
    T = len(x)
    x_out = x.copy().astype(float)
    windows = select_critical_windows(
        x, window_size=window_size, top_k=top_k,
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

    if event_type == 'random':
        event_type = rng.choice(['impulse', 'burst', 'shift'])
    if event_sign is None:
        event_sign = float(rng.choice([-1.0, 1.0]))
    if adaptive_amplitude:
        local_std = float(np.std(x[s:e])) + 1e-8
        ampl = amplitude * local_std
    else:
        ampl = amplitude

    ep = EventParams(event_type=event_type,      
                     amplitude=ampl,
                     width=min(event_width, W // 2),
                     sign=event_sign)
    b = _make_perturbation(W, ep)

    x_shocked = x_out[s:e] + b


    if warp_type == 'random':
        warp_type = rng.choice(['translate', 'scale', 'nonlinear'])

    wp = WarpParams(warp_type=warp_type,          
                    delta=warp_delta,
                    alpha=warp_alpha,
                    noise_std=warp_noise_std)
    x_warped = _apply_warp(x_shocked, s, wp)
    x_out[s:e] = x_warped

    ctx = ContextParams(
        near_changepoint = float(win_info.get('scp_n', 0.0)),
        near_peak_valley = float(win_info.get('sper_n', 0.0)),
        variance_energy  = float(win_info.get('svar_n', 0.0)),
        pred_sensitivity = float(win_info.get('spred_n', 0.0)),
        window_start=s, window_end=e,
    )
    coupling_strength = abs(ampl) / (abs(ampl) + 1.0)
    cp = CouplingParams(coupling_type=coupling_type,      # type: ignore[arg-type]
                        coupling_strength=coupling_strength)

    diff = _difficulty(ep, wp, ctx, cp, difficulty_betas)

    params = SceneParamsF1(event=ep, warp=wp, context=ctx,
                           coupling=cp, difficulty=diff)

    return x_out, params

def apply_family1_multivariate(
    X: np.ndarray,
    channels: list = None,
    **kwargs,
) -> Tuple[np.ndarray, list]:
    T, C = X.shape
    X_out = X.copy().astype(float)
    if channels is None:
        channels = list(range(C))
    all_params = []
    for ch in channels:
        x_c, p = apply_family1(X[:, ch], **kwargs)
        X_out[:, ch] = x_c
        all_params.append(p)
    return X_out, all_params
