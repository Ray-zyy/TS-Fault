from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Literal, Tuple, List, Dict, Optional
from joltbench.utils.window_selector import select_critical_windows

@dataclass
class RootFaultParams:
    fault_type: Literal['drift', 'clipping', 'quantization', 'stuck'] = 'drift'
    b0:    float = 0.2     
    kappa: float = 0.05     
    lo:    float = -np.inf
    hi:    float =  np.inf
    bias:  float = 0.0     
    delta_q: float = 0.5    


@dataclass
class PropagParams:
    gamma:  float = 0.6     
    delay:  int   = 2       
    lam:    float = 3.0     
    K:      int   = 5       

@dataclass
class SecondaryFailureParams:
    c_j:  float = 0.2       
    eta1: float = 1.5      
    eta2: float = 0.8      
    geom_q: float = 0.3   
    fill_method: Literal['ffill', 'zero', 'mean'] = 'ffill'

@dataclass
class SceneParamsF4:
    roots:       List[int] = field(default_factory=list)
    downstreams: List[int] = field(default_factory=list)
    trigger_time: int  = 0
    window_start: int  = 0
    window_end:   int  = 0
    fault_params:   Dict[str, dict] = field(default_factory=dict)  
    propag_params:  Dict[str, dict] = field(default_factory=dict)  
    secondary_params: dict = field(default_factory=dict)
    difficulty: float = 0.0

    def to_dict(self):
        return asdict(self)



def _granger_delta(
    xi: np.ndarray, xj: np.ndarray, p: int = 5
) -> float:
    n = len(xi)
    p = min(p, n // 3)
    if p < 1:
        return 0.0

    def build_X_y(series_list, p):
        arrays = [series_list[0][p:]]   # y
        rows = []
        for t in range(p, len(series_list[0])):
            row = np.concatenate([s[t-p:t] for s in series_list])
            rows.append(row)
        return np.array(rows), series_list[0][p:]

    def mse(X_mat, y):
        A = np.column_stack([X_mat, np.ones(len(X_mat))])
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            return float(np.mean((y - pred) ** 2))
        except Exception:
            return float(np.mean(y ** 2))

    X_self, y = build_X_y([xj], p)
    X_aug, _  = build_X_y([xj, xi], p)

    err_self = mse(X_self, y)
    err_aug  = mse(X_aug, y)
    return float((err_self - err_aug) / (err_self + 1e-10))


def _select_root_downstream(
    X: np.ndarray, s: int, e: int,
    n_roots: int = 2, n_down: int = 3,
    neighbourhood_r: int = 10, ar_p: int = 4,
) -> Tuple[List[int], List[int]]:
    T, C = X.shape
    ctx = X[max(0, s - neighbourhood_r): min(T, e + neighbourhood_r)]
    if len(ctx) < ar_p + 2:
        ctx = X[s:e] if e > s else X

    Delta = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            Delta[i, j] = max(0.0, _granger_delta(ctx[:, i], ctx[:, j], ar_p))

    U = Delta.sum(axis=1)     # upstream influence
    n_roots = min(n_roots, C)
    root_idx = np.argsort(U)[::-1][:n_roots].tolist()

    V = Delta[root_idx].sum(axis=0)
    n_down = min(n_down, C - n_roots)
    down_candidates = [j for j in np.argsort(V)[::-1] if j not in root_idx]
    down_idx = down_candidates[:n_down]

    return root_idx, down_idx


def _find_trigger_time(
    X: np.ndarray, roots: List[int], s: int, e: int
) -> int:
    W = X[s:e]
    medians = np.median(W, axis=0)
    deviation = np.sum(
        np.abs(W[:, roots] - medians[None, roots]), axis=1
    )
    return s + int(np.argmax(deviation))


def _apply_root_fault(
    z_r: np.ndarray, tau1: int, fp: RootFaultParams
) -> Tuple[np.ndarray, np.ndarray]:

    T = len(z_r)
    y_r = z_r.copy().astype(float)

    if fp.fault_type == 'drift':
        for t in range(tau1, T):
            y_r[t] = z_r[t] + fp.b0 + fp.kappa * (t - tau1)

    elif fp.fault_type == 'clipping':
        for t in range(tau1, T):
            y_r[t] = float(np.clip(z_r[t] + fp.bias, fp.lo, fp.hi))

    elif fp.fault_type == 'quantization':
        dq = max(fp.delta_q, 1e-6)
        for t in range(tau1, T):
            y_r[t] = dq * np.round((z_r[t] + fp.bias) / dq)

    elif fp.fault_type == 'stuck':
        stuck_val = float(y_r[tau1]) if tau1 < T else float(z_r[-1])
        y_r[tau1:] = stuck_val

    e_r = y_r - z_r
    return y_r, e_r


def _exp_kernel(K: int, lam: float) -> np.ndarray:
    l = np.arange(K, dtype=float)
    k = np.exp(-l / max(lam, 0.1))
    return k / (k.sum() + 1e-10)


def _compute_propagation(
    E: Dict[int, np.ndarray],  
    tau1: int,
    roots: List[int],
    down: int,
    pp: Dict[str, PropagParams],  
    T: int,
) -> np.ndarray:
    delta = np.zeros(T)
    for r in roots:
        key = f"{r}->{down}"
        p = pp.get(key, PropagParams())
        kern = _exp_kernel(p.K, p.lam)
        e_r = E[r]
        for t in range(tau1 + p.delay, T):
            acc = 0.0
            for l in range(p.K):
                t_src = t - p.delay - l
                if t_src < 0:
                    continue
                acc += kern[l] * e_r[t_src]
            delta[t] += p.gamma * acc
    return delta



def _secondary_mask(
    z_prime_j: np.ndarray, delta_j: np.ndarray,
    tau1: int, delay_j: int,
    sp: SecondaryFailureParams, rng
) -> np.ndarray:
    T = len(z_prime_j)
    activate_t = tau1 + delay_j

    v_t = np.array([
        float(np.std(z_prime_j[max(0, t-5): t+5]))
        for t in range(T)
    ]) / (float(np.std(z_prime_j)) + 1e-8)

    logit = (sp.c_j
             + sp.eta1 * np.abs(delta_j) / (np.abs(delta_j).max() + 1e-8)
             + sp.eta2 * v_t)
    logit[:activate_t] = -30.0
    p_start = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))

    mask = np.ones(T, dtype=float)
    t = activate_t
    while t < T:
        if rng.random() < p_start[t]:
            length = int(rng.geometric(1.0 - sp.geom_q))
            length = max(1, min(length, T - t))
            mask[t: t + length] = 0.0
            t += length
        else:
            t += 1
    return mask


def _fill(x: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    out = x.copy()
    if method == 'ffill':
        last = float(x[0])
        for t in range(len(x)):
            if mask[t] == 0:
                out[t] = last
            else:
                last = float(x[t])
    elif method == 'zero':
        out[mask == 0] = 0.0
    elif method == 'mean':
        m = float(x[mask == 1].mean()) if mask.sum() else 0.0
        out[mask == 0] = m
    return out


def apply_family4(
    X: np.ndarray,
    # Window selection
    window_size: int = 24,
    top_k: int = 3,
    window_lambdas: tuple = (0.25, 0.25, 0.25, 0.25),
    horizon: int = 10,
    period: int = None,
    random_select: bool = True,
    # Topology
    n_roots: int = 2,
    n_down:  int = 3,
    neighbourhood_r: int = 10,
    ar_p: int = 4,
    # Root faults  (one entry per root, or broadcast single spec)
    fault_type: Literal['drift','clipping','quantization','stuck','random'] = 'random',
    fault_b0:    float = 0.2,
    fault_kappa: float = 0.05,
    fault_delta_q: float = 0.5,
    # Propagation
    gamma_range: Tuple[float,float] = (0.4, 1.2),
    delay_range: Tuple[int,int]     = (1, 4),
    lambda_range: Tuple[float,float] = (2.0, 6.0),
    K: int = 5,
    # Secondary failures
    sec_c_j:   float = 0.2,
    sec_eta1:  float = 1.5,
    sec_eta2:  float = 0.8,
    sec_geom_q: float = 0.3,
    fill_method: Literal['ffill','zero','mean'] = 'ffill',
    # Difficulty
    difficulty_betas: tuple = (0.25, 0.25, 0.25, 0.25),
    random_seed: int = None,
) -> Tuple[np.ndarray, SceneParamsF4]:

    rng = np.random.default_rng(random_seed)
    assert X.ndim == 2, "Family 4 requires multivariate input (T, C)"
    T, C = X.shape

    from joltbench.utils.window_selector import compute_dominant_period
    if period is None:
        period = compute_dominant_period(X[:, 0])

    wins = select_critical_windows(
        X[:, 0], window_size=window_size, top_k=top_k,
        lambdas=window_lambdas, horizon=horizon, period=period
    )
    if not wins:
        s = T // 2 - window_size // 2
        e = s + window_size
    else:
        w = rng.choice(wins) if random_select else wins[0]
        s, e = w['start'], w['end']

    n_roots = min(n_roots, C)
    n_down  = min(n_down,  C - n_roots)

    roots, downstreams = _select_root_downstream(
        X, s, e, n_roots=n_roots, n_down=n_down,
        neighbourhood_r=neighbourhood_r, ar_p=ar_p
    )
    tau1 = _find_trigger_time(X, roots, s, e)

    E: Dict[int, np.ndarray] = {}   # root → error
    fault_records: Dict[str, dict] = {}

    ftypes = ['drift', 'clipping', 'quantization', 'stuck']

    for r in roots:
        ft = fault_type if fault_type != 'random' else str(rng.choice(ftypes))
        hi_val = float(X[:, r].max()) if ft == 'clipping' else float(np.inf)
        lo_val = float(X[:, r].min()) if ft == 'clipping' else float(-np.inf)

        fp = RootFaultParams(
            fault_type=ft,         # type: ignore[arg-type]
            b0=fault_b0,
            kappa=fault_kappa * float(rng.choice([-1, 1])),
            lo=lo_val,
            hi=hi_val + 0.1 * abs(hi_val),  # slight compression
            delta_q=fault_delta_q * (float(np.std(X[:, r])) + 1e-8),
        )
        y_r, e_r = _apply_root_fault(X[:, r], tau1, fp)
        X_out[:, r] = y_r
        E[r] = e_r
        fault_records[str(r)] = asdict(fp)

    propag_records: Dict[str, dict] = {}
    propag_params_map: Dict[str, PropagParams] = {}
    all_deltas: Dict[int, np.ndarray] = {}

    for j in downstreams:
        for r in roots:
            key = f"{r}->{j}"
            pp_rj = PropagParams(
                gamma=float(rng.uniform(*gamma_range)),
                delay=int(rng.integers(delay_range[0], delay_range[1]+1)),
                lam=float(rng.uniform(*lambda_range)),
                K=K,
            )
            propag_params_map[key] = pp_rj
            propag_records[key] = asdict(pp_rj)

        delta_j = _compute_propagation(E, tau1, roots, j,
                                       propag_params_map, T)
        all_deltas[j] = delta_j

        z_prime_j = X[:, j] + delta_j

 
        sp = SecondaryFailureParams(
            c_j=sec_c_j, eta1=sec_eta1, eta2=sec_eta2,
            geom_q=sec_geom_q, fill_method=fill_method
        )
        min_delay = min(
            propag_params_map.get(f"{r}->{j}", PropagParams()).delay
            for r in roots
        )
        sec_mask = _secondary_mask(z_prime_j, delta_j, tau1, min_delay, sp, rng)
        X_out[:, j] = _fill(z_prime_j, sec_mask, fill_method)

    β1, β2, β3, β4 = difficulty_betas

    root_errors = np.concatenate([np.abs(E[r][tau1:]) for r in roots]) \
        if roots else np.array([0.0])
    D_root = float(np.clip(root_errors.mean() / (np.std(X) + 1e-8), 0, 1))

    if all_deltas:
        d_vals = np.concatenate([np.abs(v) for v in all_deltas.values()])
        D_cascade = float(np.clip(d_vals.mean() / (np.std(X) + 1e-8), 0, 1))
    else:
        D_cascade = 0.0

    all_delays = [propag_params_map[k].delay
                  for k in propag_params_map] or [0]
    D_delay = float(np.clip(np.std(all_delays) / (np.mean(all_delays) + 1e-6), 0, 1))

    sec_miss = float(1.0 - np.mean([
        (X_out[:, j] == (X[:, j] + all_deltas.get(j, 0))).mean()
        for j in downstreams
    ])) if downstreams else 0.0
    D_secondary = float(np.clip(sec_miss, 0, 1))

    difficulty = float(β1*D_root + β2*D_cascade + β3*D_delay + β4*D_secondary)

    sp_obj = SecondaryFailureParams(c_j=sec_c_j, eta1=sec_eta1,
                                    eta2=sec_eta2, geom_q=sec_geom_q,
                                    fill_method=fill_method)
    params = SceneParamsF4(
        roots=roots, downstreams=downstreams,
        trigger_time=tau1, window_start=s, window_end=e,
        fault_params=fault_records,
        propag_params=propag_records,
        secondary_params=asdict(sp_obj),
        difficulty=difficulty,
    )
    return X_out, params
