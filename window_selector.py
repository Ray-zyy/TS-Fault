import numpy as np
from scipy.signal import find_peaks
import warnings

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False
    warnings.warn("PyWavelets not installed – Svar falls back to windowed variance.")


def compute_dominant_period(x: np.ndarray, min_lag: int = 2) -> int:
    L = len(x)
    max_lag = max(min_lag + 1, L // 2)
    x_c = x - x.mean()
    denom = float(np.dot(x_c, x_c))
    if denom < 1e-12:
        return max(4, L // 4)

    acf = np.correlate(x_c, x_c, mode='full')[L - 1:] / denom
    search = acf[min_lag:max_lag]
    peaks, _ = find_peaks(search, height=0.05, prominence=0.03)
    if len(peaks):
        return int(peaks[0]) + min_lag
    return max(4, L // 4)


def _local_linear_predict(x_train: np.ndarray, n_pred: int) -> np.ndarray:
    n = len(x_train)
    if n < 2:
        return np.full(n_pred, x_train.mean())
    t = np.arange(n, dtype=float)
    A = np.column_stack([t, np.ones(n)])
    try:
        coef, *_ = np.linalg.lstsq(A, x_train, rcond=None)
        t_p = np.arange(n, n + n_pred, dtype=float)
        return np.column_stack([t_p, np.ones(n_pred)]) @ coef
    except Exception:
        return np.full(n_pred, x_train[-1])


def score_changepoint(x: np.ndarray, s: int, e: int) -> float:
    c = (s + e) // 2
    WL, WR = x[s:c], x[c:e]
    if len(WL) < 2 or len(WR) < 2:
        return 0.0
    pred_R = _local_linear_predict(WL, len(WR))
    pred_L = _local_linear_predict(WR, len(WL))
    return float(np.mean((WR - pred_R) ** 2) + np.mean((WL - pred_L) ** 2))


def score_periodic(x: np.ndarray, s: int, e: int,
                   period: int = None) -> float:
    L = len(x)
    c = (s + e) / 2.0
    if period is None:
        period = compute_dominant_period(x)
    P = max(period, 2)

    lp_s = max(0, L - P)
    lp = x[lp_s:]
    tpeak   = lp_s + int(np.argmax(lp))
    tvalley = lp_s + int(np.argmin(lp))

    n_rep = int(L / P) + 2
    peak_anchors   = [tpeak   - k * P for k in range(n_rep)]
    valley_anchors = [tvalley - k * P for k in range(n_rep)]

    sigma_p = 0.15 * P + 1e-4
    dp = min(abs(c - a) for a in peak_anchors)
    dv = min(abs(c - a) for a in valley_anchors)
    return float(max(np.exp(-dp ** 2 / sigma_p ** 2),
                     np.exp(-dv ** 2 / sigma_p ** 2)))


def score_variance(x: np.ndarray, s: int, e: int,
                   n_levels: int = 3) -> float:
    seg = x[s:e]
    if len(seg) < 4:
        return float(np.var(seg) + 1e-12)
    mad = float(np.median(np.abs(seg - np.median(seg)))) + 1e-8

    if _HAS_PYWT:
        ml = pywt.dwt_max_level(len(seg), 'db2')
        lvl = min(n_levels, ml)
        try:
            coeffs = pywt.wavedec(seg, 'db2', level=lvl)
            weights = np.array([2.0 ** (lvl - j) for j in range(lvl)])
            weights /= weights.sum()
            energy = sum(w * float(np.sum(d ** 2))
                         for w, d in zip(weights, coeffs[1:]))
            return float(energy / mad)
        except Exception:
            pass
    return float(np.var(seg) / mad)


def score_prediction(x: np.ndarray, s: int, e: int,
                     horizon: int = 5, ar_order: int = 6) -> float:
    L = len(x)
    H = min(horizon, L - e)
    if H <= 0:
        return 0.0

    def ar_forecast(hist: np.ndarray, steps: int, p: int) -> np.ndarray:
        p = min(p, len(hist) - 1)
        if p < 1:
            return np.full(steps, hist[-1])
        rows = np.array([hist[i - p:i][::-1] for i in range(p, len(hist))])
        ys   = hist[p:]
        A    = np.column_stack([rows, np.ones(len(rows))])
        try:
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        except Exception:
            return np.full(steps, hist[-1])
        buf = list(hist[-p:])
        out = []
        for _ in range(steps):
            row = np.array(list(reversed(buf[-p:])) + [1.0])
            v = float(row @ coef)
            out.append(v); buf.append(v)
        return np.array(out)

    pf   = ar_forecast(x[:e], H, ar_order)
    x_oc = x.copy(); x_oc[s:e] = float(np.median(x))
    po   = ar_forecast(x_oc[:e], H, ar_order)
    alpha = np.array([1.0 / (h + 1) for h in range(H)])
    alpha /= alpha.sum()
    return float(np.dot(alpha, (pf - po) ** 2))

def select_critical_windows(
    x: np.ndarray,
    window_size: int = 24,
    stride: int = None,
    top_k: int = 5,
    lambdas: tuple = (0.25, 0.25, 0.25, 0.25),
    horizon: int = 10,
    period: int = None,
) -> list:
    L = len(x)
    assert window_size >= 4, "window_size must be >= 4"
    λ1, λ2, λ3, λ4 = lambdas
    if period is None:
        period = compute_dominant_period(x)
    if stride is None:
        stride = max(1, window_size // 4)

    raw = []
    for s in range(0, L - window_size + 1, stride):
        e = s + window_size
        raw.append(dict(
            start=s, end=e,
            scp=score_changepoint(x, s, e),
            sper=score_periodic(x, s, e, period),
            svar=score_variance(x, s, e),
            spred=score_prediction(x, s, e, horizon),
        ))

    if not raw:
        return []
    for key in ('scp', 'sper', 'svar', 'spred'):
        vals = np.array([r[key] for r in raw])
        vmax = vals.max() + 1e-10
        for r in raw:
            r[key + '_n'] = r[key] / vmax
    for r in raw:
        r['score'] = (λ1 * r['scp_n'] + λ2 * r['sper_n'] +
                      λ3 * r['svar_n'] + λ4 * r['spred_n'])
    raw.sort(key=lambda r: r['score'], reverse=True)
    return raw[:top_k]


def select_window_multivariate(
    X: np.ndarray,
    ref_channel: int = 0,
    **kwargs
) -> list:
    T, C = X.shape
    if C == 1 or ref_channel >= 0:
        ch = min(ref_channel, C - 1)
        return select_critical_windows(X[:, ch], **kwargs)
    per_ch = [select_critical_windows(X[:, c], **kwargs) for c in range(C)]
    result = per_ch[0]
    for r in result:
        r['score'] = float(np.mean([
            next((rr['score'] for rr in pc
                  if rr['start'] == r['start']), r['score'])
            for pc in per_ch
        ]))
    result.sort(key=lambda r: r['score'], reverse=True)
    return result
