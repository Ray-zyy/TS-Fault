import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import types

def _make_fake_package(root_dir: str):

    pkg = types.ModuleType("joltbench")
    pkg.__path__ = [root_dir]
    pkg.__package__ = "joltbench"
    sys.modules["joltbench"] = pkg

    utils_pkg = types.ModuleType("joltbench.utils")
    utils_pkg.__path__ = [root_dir]
    utils_pkg.__package__ = "joltbench.utils"
    sys.modules["joltbench.utils"] = utils_pkg

    families_pkg = types.ModuleType("joltbench.families")
    families_pkg.__path__ = [root_dir]
    families_pkg.__package__ = "joltbench.families"
    sys.modules["joltbench.families"] = families_pkg

    # 逐个把当前目录的 .py 注册为子模块
    for fname, mname in [
        ("window_selector.py", "joltbench.utils.window_selector"),
        ("dataset_loader.py",  "joltbench.utils.dataset_loader"),
        ("family1.py",         "joltbench.families.family1"),
        ("family2.py",         "joltbench.families.family2"),
        ("family3.py",         "joltbench.families.family3"),
        ("family4.py",         "joltbench.families.family4"),
        ("benchmark.py",       "joltbench.benchmark"),
    ]:
        fpath = os.path.join(root_dir, fname)
        if not os.path.exists(fpath):
            continue
        import importlib.util
        spec = importlib.util.spec_from_file_location(mname, fpath)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[mname] = mod
        spec.loader.exec_module(mod)

    return pkg

_make_fake_package(SCRIPT_DIR)

import numpy as np
import pandas as pd
import json
import argparse
import warnings
import time
warnings.filterwarnings("ignore")

from joltbench.families.family1 import apply_family1, apply_family1_multivariate
from joltbench.families.family2 import apply_family2
from joltbench.families.family3 import apply_family3
from joltbench.families.family4 import apply_family4

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[警告] 未找到 matplotlib，将跳过可视化。安装: pip install matplotlib")



CONFIG = {

    "dataset": "ETTh1",
    "csv_path": "",
    "max_len": 2000,
    "seq_len": 336,
    "pred_len": 96,
    "family": "all",
    "difficulty": [0.2, 0.4, 0.6, 0.8, 1.0],
    "n_samples": 20,
    "window_size": 24,
    "f1": {
        "event_type":    "random",    
        "amplitude":     2.0,         
        "event_width":   6,          
        "event_sign":    None,       
        "warp_type":     "random",    
        "warp_delta":    2.0,       
        "warp_alpha":    1.3,        
        "warp_noise_std":0.1,        
        "coupling_type": "independent",
        "adaptive_amplitude": True,  
    
        "window_lambdas": (0.25, 0.25, 0.25, 0.25),
    },


    "f2": {
        "shock_type":     "random",   
        "shock_amplitude": 2.0,      
        "shock_width":     5,        
        "subset_size":     3,        
        "tau_max":         5,        
        "delta_tau_max":   3,        
        "delta_g_max":     1.5,     
        "allow_sign_flip": True,     
        "adaptive_amplitude": True,
    },

    "f3": {
        # 新机制参数
        "delta_beta":    0.05,   
        "delta_b":       0.5,   
        "a_s":           1.3,   
        "period_ratio":  1.0,    
        "phase_shift":   0.0,  
        "c_r":           1.2,   
        "gamma":         5.0,   
        "a0":     -3.0,  
        "a1":      2.5,  
        "a2":      1.0,  
        "a3":      0.8,  
        "c_i":     0.0,   
        "geom_q":  0.3,   
        "fill_method": "ffill",  
    },

    "f4": {
        "n_roots":      2,              
        "n_down":       3,              
        "fault_type":   "random",       
        "fault_b0":     0.2,           
        "fault_kappa":  0.05,          
        "fault_delta_q": 0.5,         
        "gamma_range":  (0.4, 1.2),     
        "delay_range":  (1, 4),         
        "lambda_range": (2.0, 6.0),  
        "K":            5,             
        "sec_c_j":      0.2,           
        "sec_eta1":     1.5,       
        "sec_eta2":     0.8,         
        "sec_geom_q":   0.3,            
        "fill_method":  "ffill",
    },

    "output_dir": "joltbench_output",

    "save_csv": True,

    "save_npz": True,

    "save_json": True,

    "plot": True,

    "plot_n_examples": 2,

    "random_seed": 42,
}



DATASET_PATHS = {
    "ETTh1":       "dataset/ETT-small/ETTh1.csv",
    "ETTh2":       "dataset/ETT-small/ETTh2.csv",
    "ETTm1":       "dataset/ETT-small/ETTm1.csv",
    "ETTm2":       "dataset/ETT-small/ETTm2.csv",
    "Weather":     "dataset/weather/weather.csv",
    "Electricity": "dataset/electricity/electricity.csv",
}

DATASET_META = {
    "ETTh1":       {"date_col": "date", "freq": "1h",  "period_hint": 24},
    "ETTh2":       {"date_col": "date", "freq": "1h",  "period_hint": 24},
    "ETTm1":       {"date_col": "date", "freq": "15T", "period_hint": 96},
    "ETTm2":       {"date_col": "date", "freq": "15T", "period_hint": 96},
    "Weather":     {"date_col": "date", "freq": "10T", "period_hint": 144},
    "Electricity": {"date_col": "date", "freq": "1h",  "period_hint": 24},
}



class Scaler:
    """零均值单位方差归一化。"""
    def fit_transform(self, X):
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean


def load_csv(name: str, csv_path: str = "", max_len: int = None,
             normalise: bool = True):

    if csv_path and os.path.exists(csv_path):
        path = csv_path
    else:

        rel = DATASET_PATHS.get(name, "")
        path = os.path.join(SCRIPT_DIR, rel) if rel else ""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到数据集 '{name}' 的 CSV 文件。\n"
                f"  期望路径: {path}\n"
                f"  请确认 dataset/ 目录结构，或在 CONFIG['csv_path'] 中手动指定路径。"
            )

    print(f"  📂 加载: {path}")
    df = pd.read_csv(path)

    meta = DATASET_META.get(name, {"date_col": "date", "period_hint": 24})
    date_col = meta.get("date_col", "date")
    if date_col in df.columns:
        df = df.drop(columns=[date_col])


    df = df.select_dtypes(include=[np.number]).dropna()

    if max_len:
        df = df.iloc[:max_len]

    data = df.values.astype(np.float32)
    cols = list(df.columns)

    scaler = None
    if normalise:
        scaler = Scaler()
        data = scaler.fit_transform(data).astype(np.float32)

    print(f"  ✓ 数据形状: {data.shape}  (时间步 × 变量数)")
    print(f"  ✓ 变量列:   {cols[:5]}{'...' if len(cols)>5 else ''}")
    return data, cols, scaler, meta


def build_samples(data: np.ndarray, seq_len: int, pred_len: int,
                  stride: int = None, n_samples: int = 50, rng=None):

    T, C = data.shape
    if rng is None:
        rng = np.random.default_rng(42)

    total_windows = T - seq_len - pred_len
    if total_windows <= 0:
        raise ValueError(
            f"数据太短！T={T} < seq_len+pred_len={seq_len+pred_len}。\n"
            f"请减小 seq_len 或 pred_len，或增大 max_len。"
        )

    n_samples = min(n_samples, total_windows)
    starts = rng.choice(total_windows, size=n_samples, replace=False)
    starts = np.sort(starts)

    X_in  = np.stack([data[s: s+seq_len]           for s in starts])
    Y_out = np.stack([data[s+seq_len: s+seq_len+pred_len] for s in starts])
    return X_in, Y_out, starts

def _d_to_f1(d: float, base: dict) -> dict:
    kw = dict(base)
    kw["amplitude"]      = base["amplitude"]      * (0.3 + d * 1.7)
    kw["event_width"]    = max(2, int(base["event_width"] * (0.5 + d)))
    kw["warp_delta"]     = base["warp_delta"]      * d * 2.5
    kw["warp_alpha"]     = 1.0 + (base["warp_alpha"] - 1.0) * d * 2.0
    kw["warp_noise_std"] = base["warp_noise_std"]  * d * 3.0
    kw["coupling_type"]  = ("independent" if d < 0.35
                            else "shock_centered" if d < 0.7
                            else "amplitude_coupled")
    return kw

def _d_to_f2(d: float, base: dict) -> dict:
    kw = dict(base)
    kw["shock_amplitude"] = base["shock_amplitude"] * (0.3 + d * 1.7)
    kw["delta_tau_max"]   = max(1, int(base["delta_tau_max"] * (0.3 + d * 2.0)))
    kw["delta_g_max"]     = base["delta_g_max"]     * (0.2 + d * 1.8)
    kw["allow_sign_flip"] = d > 0.4
    kw["subset_size"]     = max(2, int(base["subset_size"] * (0.5 + d)))
    return kw

def _d_to_f3(d: float, base: dict) -> dict:
    kw = dict(base)
    kw["delta_beta"]   = base["delta_beta"]   * d * 3.0
    kw["delta_b"]      = base["delta_b"]      * d * 4.0
    kw["a_s"]          = 1.0 + (base["a_s"] - 1.0) * d * 2.5
    kw["period_ratio"] = 1.0 + (base["period_ratio"] - 1.0) * d * 3.0
    kw["c_r"]          = 1.0 + (base["c_r"] - 1.0) * d * 2.5
    kw["gamma"]        = max(0.5, base["gamma"] * (1.0 - d * 0.85))
    kw["a0"]           = base["a0"] + d * 3.5
    kw["a1"]           = base["a1"] * (0.5 + d * 1.5)
    kw["geom_q"]       = min(0.8, base["geom_q"] + d * 0.5)
    return kw

def _d_to_f4(d: float, base: dict) -> dict:
    kw = dict(base)
    kw["n_roots"]   = max(1, int(base["n_roots"] * (0.5 + d)))
    kw["n_down"]    = max(1, int(base["n_down"]  * (0.5 + d)))
    kw["fault_b0"]  = base["fault_b0"]  * d * 3.0
    kw["fault_kappa"] = base["fault_kappa"] * d * 3.0
    lo, hi = base["gamma_range"]
    kw["gamma_range"] = (lo * (0.5 + d), hi * (0.5 + d * 1.5))
    dl, dh = base["delay_range"]
    kw["delay_range"] = (dl, max(dl+1, int(dh * (0.5 + d * 2.0))))
    kw["sec_eta1"]  = base["sec_eta1"] * (0.3 + d * 1.7)
    kw["sec_geom_q"]= min(0.75, base["sec_geom_q"] + d * 0.4)
    return kw
═══════════════════════════════════════════════════════════════════

def corrupt_sample(x_in: np.ndarray, family: str,
                   difficulty: float, cfg: dict,
                   seed: int = None):

    T, C = x_in.shape
    ws   = cfg["window_size"]
    extra = None

    if family == "family1":
        kw = _d_to_f1(difficulty, cfg["f1"])
        # 对所有通道施加（每通道独立选窗口）
        X_out, params_list = apply_family1_multivariate(
            x_in, window_size=ws, random_seed=seed, **kw
        )
        params = {
            "family": "family1",
            "difficulty_input": difficulty,
            "difficulty_score": float(np.mean([
                p.difficulty for p in params_list
            ])),
            "per_channel": [p.to_dict() for p in params_list],
        }

    elif family == "family2":
        kw = _d_to_f2(difficulty, cfg["f2"])
        X_out, p = apply_family2(
            x_in, window_size=ws, random_seed=seed, **kw
        )
        params = {
            "family": "family2",
            "difficulty_input": difficulty,
            "difficulty_score": p.difficulty,
            **p.to_dict(),
        }

    elif family == "family3":
        kw = _d_to_f3(difficulty, cfg["f3"])
        X_out, z_true, mask, p = apply_family3(
            x_in, random_seed=seed, **kw
        )
        extra = {"z_true": z_true, "mask": mask}
        params = {
            "family": "family3",
            "difficulty_input": difficulty,
            "difficulty_score": p.difficulty,
            **p.to_dict(),
        }

    elif family == "family4":
        kw = _d_to_f4(difficulty, cfg["f4"])
        X_out, p = apply_family4(
            x_in, window_size=ws, random_seed=seed, **kw
        )
        params = {
            "family": "family4",
            "difficulty_input": difficulty,
            "difficulty_score": p.difficulty,
            **p.to_dict(),
        }
    else:
        raise ValueError(f"未知 family: {family}")

    # 确保输出是 (T, C)
    if X_out.ndim == 1:
        X_out = X_out[:, None]

    return X_out, params, extra


FAMILY_COLORS = {
    "family1": {"clean": "#1f77b4", "corrupt": "#d62728", "span": "#d62728"},
    "family2": {"clean": "#1f77b4", "corrupt": "#ff7f0e", "span": "#ff7f0e"},
    "family3": {"clean": "#1f77b4", "corrupt": "#2ca02c", "span": "#2ca02c",
                "true": "#9467bd"},
    "family4": {"clean": "#1f77b4", "corrupt": "#8c564b", "span": "#8c564b"},
}

FAMILY_NAMES = {
    "family1": "Family 1 — Time-Warped Shock",
    "family2": "Family 2 — Dependency-Fracture Shock",
    "family3": "Family 3 — Regime-Transition Missingness",
    "family4": "Family 4 — Cascading Sensor-to-System Failure",
}



def _json_safe(obj):
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, np.bool_):        return bool(obj)
    raise TypeError(type(obj))

def save_results(records: list, out_dir: str, prefix: str, cfg: dict):
    os.makedirs(out_dir, exist_ok=True)

    x_corrupts = np.stack([r["x_corrupt"] for r in records])  # (N,T,C)
    x_cleans   = np.stack([r["x_clean"]   for r in records])
    y_targets  = np.stack([r["y_target"]  for r in records])
    params_all = [r["params"] for r in records]

    # NPZ
    if cfg["save_npz"]:
        np.savez_compressed(
            os.path.join(out_dir, f"{prefix}.npz"),
            x_corrupt=x_corrupts,
            x_clean=x_cleans,
            y_target=y_targets,
        )

    if cfg["save_json"]:
        with open(os.path.join(out_dir, f"{prefix}_params.json"), "w",
                  encoding="utf-8") as f:
            json.dump(params_all, f, ensure_ascii=False,
                      indent=2, default=_json_safe)

    if cfg["save_csv"]:
        csv_dir = os.path.join(out_dir, "csv", prefix)
        os.makedirs(csv_dir, exist_ok=True)
        for i, r in enumerate(records):
            T, C = r["x_clean"].shape
            cols = {}
            for c in range(C):
                cols[f"clean_ch{c}"]   = r["x_clean"][:, c]
                cols[f"corrupt_ch{c}"] = r["x_corrupt"][:, c]
            df = pd.DataFrame(cols)
            df.to_csv(os.path.join(csv_dir, f"sample_{i:03d}.csv"), index=False)


def run_one(dataset_name: str, families: list, difficulties: list, cfg: dict):

    print(f"\n{'━'*65}")
    print(f"  📊 数据集: {dataset_name}")
    print(f"{'━'*65}")


    data, cols, scaler, meta = load_csv(
        dataset_name,
        csv_path=cfg.get("csv_path", ""),
        max_len=cfg["max_len"],
        normalise=True,
    )
    T, C = data.shape

    seq_len  = min(cfg["seq_len"],  T // 3)
    pred_len = min(cfg["pred_len"], T // 6)
    if seq_len != cfg["seq_len"]:
        print(f"  ⚠️  seq_len 自动调整为 {seq_len}（数据不足原设定）")

    rng = np.random.default_rng(cfg["random_seed"])
    X_in, Y_out, _ = build_samples(
        data, seq_len=seq_len, pred_len=pred_len,
        n_samples=cfg["n_samples"] * 2,   # 多采一些防止失败
        rng=rng,
    )

    out_base = os.path.join(cfg["output_dir"], dataset_name)
    os.makedirs(out_base, exist_ok=True)

    overview_records = []   
    summary_rows     = []   

    for family in families:
        print(f"\n  {'─'*55}")
        print(f"  🔧 {FAMILY_NAMES[family]}")

        by_diff: dict = {}

        for diff in difficulties:
            tag = f"{dataset_name}_{family}_d{int(diff*10):02d}"
            print(f"\n    难度 d={diff:.1f}  [{tag}]", end="  ", flush=True)

            records = []
            d_scores = []
            n_fail   = 0

            for idx in range(min(cfg["n_samples"], len(X_in))):
                seed_i = int(rng.integers(0, 2**31))
                try:
                    x_in = X_in[idx]       # (seq_len, C)
                    x_corrupt, params, extra = corrupt_sample(
                        x_in, family, diff, cfg, seed=seed_i
                    )
                    records.append({
                        "x_clean":   x_in,
                        "x_corrupt": x_corrupt,
                        "y_target":  Y_out[idx],
                        "params":    params,
                        "extra":     extra,
                        "family":    family,
                        "difficulty": diff,
                    })
                    d_scores.append(params.get("difficulty_score", 0.0))
                except Exception as e:
                    n_fail += 1
                    if n_fail <= 2:
                        print(f"\n    [warn] sample {idx}: {e}")

            if not records:
                print("无有效样本，跳过")
                continue

            avg_d = float(np.mean(d_scores))
            print(f"✓ {len(records)} 样本 | "
                  f"D_score 均值={avg_d:.4f} | 失败={n_fail}")


            summary_rows.append({
                "数据集": dataset_name,
                "Family": family,
                "难度输入 d": diff,
                "样本数": len(records),
                "D_score 均值": round(avg_d, 4),
                "D_score 最小": round(float(np.min(d_scores)), 4),
                "D_score 最大": round(float(np.max(d_scores)), 4),
            })


            save_results(records, out_base, tag, cfg)

           

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        print(f"\n  {'═'*65}")
        print(f"  📋 汇总  [{dataset_name}]")
        print(f"  {'═'*65}")
        print(df_sum.to_string(index=False))

        df_sum.to_csv(
            os.path.join(out_base, f"{dataset_name}_summary.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"\n  💾 汇总保存至: {out_base}/{dataset_name}_summary.csv")

    return overview_records



def main():
    args = parse_args()
    cfg  = dict(CONFIG)   

    if args.output_dir: cfg["output_dir"]  = args.output_dir
    if args.n_samples:  cfg["n_samples"]   = args.n_samples
    if args.seq_len:    cfg["seq_len"]     = args.seq_len
    if args.pred_len:   cfg["pred_len"]    = args.pred_len
    if args.seed:       cfg["random_seed"] = args.seed
    if args.no_plot:    cfg["plot"]        = False
    if args.no_csv:     cfg["save_csv"]    = False


    if args.difficulty is not None:
        cfg["difficulty"] = [args.difficulty]
    elif isinstance(cfg["difficulty"], float):
        cfg["difficulty"] = [cfg["difficulty"]]

    if args.amplitude:  cfg["f1"]["amplitude"]  = args.amplitude
    if args.event_type: cfg["f1"]["event_type"] = args.event_type
    if args.warp_type:  cfg["f1"]["warp_type"]  = args.warp_type


    ALL_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity"]
    ALL_FAMILIES = ["family1", "family2", "family3", "family4"]

    if args.mode == "batch":
        datasets = ALL_DATASETS
        families = ALL_FAMILIES
        if not isinstance(cfg["difficulty"], list):
            cfg["difficulty"] = [0.2, 0.4, 0.6, 0.8, 1.0]
    else:
        ds = args.dataset or cfg["dataset"]
        datasets = ALL_DATASETS if ds == "all" else [ds]
        fm = args.family  or cfg["family"]
        families = ALL_FAMILIES if fm == "all" else [fm]

    difficulties = cfg["difficulty"]
    if not isinstance(difficulties, list):
        difficulties = [difficulties]


if __name__ == "__main__":
    main()
