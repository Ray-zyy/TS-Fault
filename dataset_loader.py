
from __future__ import annotations
import os
import io
import urllib.request
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    warnings.warn("pandas not found; dataset loading limited.")


_DATASET_URLS: Dict[str, str] = {
    # ETT datasets (from Informer repo)
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

_DATASET_INFO: Dict[str, Dict] = {
    "ETTh1":  {"freq": "1H", "target": "OT",
               "feature_cols": ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]},
    "ETTh2":  {"freq": "1H", "target": "OT",
               "feature_cols": ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]},
    "ETTm1":  {"freq": "15T", "target": "OT",
               "feature_cols": ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]},
    "ETTm2":  {"freq": "15T", "target": "OT",
               "feature_cols": ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]},
    "Weather": {"freq": "10T", "target": "OT",
                "feature_cols": None},   # use all numeric cols
    "Traffic": {"freq": "1H",  "target": None,
                "feature_cols": None},
    "Electricity": {"freq": "1H", "target": None, "feature_cols": None},
    "Exchange": {"freq": "1D", "target": None, "feature_cols": None},
    "ILI":     {"freq": "1W", "target": None, "feature_cols": None},
}


def _try_download(name: str, cache_dir: str = "/tmp/joltbench_data") -> Optional[str]:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}.csv")
    if os.path.exists(path):
        return path
    url = _DATASET_URLS.get(name)
    if url is None:
        return None
    try:
        print(f"[JoltBench] Downloading {name} from {url} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[JoltBench] Saved to {path}")
        return path
    except Exception as e:
        warnings.warn(f"Could not download {name}: {e}")
        return None


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_



def load_dataset(
    name_or_path: str,
    max_len: int = None,
    normalise: bool = True,
    feature_cols: list = None,
    target_col: str = None,
    date_col: str = "date",
    cache_dir: str = "/tmp/joltbench_data",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    assert _HAS_PANDAS, "pandas is required for dataset loading"

    # Resolve path
    if os.path.isfile(name_or_path):
        csv_path = name_or_path
        info = _DATASET_INFO.get(os.path.splitext(os.path.basename(name_or_path))[0], {})
    else:
        csv_path = _try_download(name_or_path, cache_dir)
        if csv_path is None:
            raise FileNotFoundError(
                f"Dataset '{name_or_path}' not found locally and no download URL registered. "
                f"Please provide a CSV path."
            )
        info = _DATASET_INFO.get(name_or_path, {})

    df = pd.read_csv(csv_path)

    if date_col in df.columns:
        df = df.drop(columns=[date_col])

    # Select feature columns
    if feature_cols is not None:
        cols = [c for c in feature_cols if c in df.columns]
    elif info.get("feature_cols") is not None:
        cols = [c for c in info["feature_cols"] if c in df.columns]
    else:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not cols:
        raise ValueError("No numeric columns found in dataset.")

    df = df[cols].dropna()

    if max_len is not None:
        df = df.iloc[:max_len]

    data = df.values.astype(np.float32)

    scaler = None
    if normalise:
        scaler = StandardScaler()
        data = scaler.fit_transform(data).astype(np.float32)

    meta: Dict[str, Any] = {
        "name":        name_or_path,
        "n_timesteps": data.shape[0],
        "n_channels":  data.shape[1],
        "columns":     cols,
        "target_col":  target_col or info.get("target"),
        "freq":        info.get("freq", "unknown"),
        "scaler":      scaler,
        "normalised":  normalise,
    }

    return data, meta

def make_windows(
    data: np.ndarray,
    seq_len: int = 336,
    pred_len: int = 96,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    T, C = data.shape
    N = max(0, (T - seq_len - pred_len) // stride + 1)
    X = np.zeros((N, seq_len, C), dtype=data.dtype)
    Y = np.zeros((N, pred_len, C), dtype=data.dtype)
    for i in range(N):
        t0 = i * stride
        X[i] = data[t0: t0 + seq_len]
        Y[i] = data[t0 + seq_len: t0 + seq_len + pred_len]
    return X, Y
