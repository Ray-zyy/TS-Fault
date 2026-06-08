import os
import sys
import glob
import time
import csv
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classical_models import DL_MODELS, STAT_MODELS, ALL_MODELS, build_model


DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity']
FAMILIES = ['family1', 'family2', 'family3', 'family4']
DIFFICULTIES = ['d02', 'd04', 'd06', 'd08', 'd10']


class Configs:
    def __init__(self, enc_in, c_out, seq_len=336, pred_len=96, label_len=48):
        self.task_name = 'long_term_forecast'
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_in = enc_in
        self.dec_in = enc_in
        self.c_out = c_out
        self.d_model = 128
        self.d_ff = 256
        self.e_layers = 2
        self.d_layers = 1
        self.n_heads = 4
        self.dropout = 0.1
        self.moving_avg = 25
        self.embed = 'timeF'
        self.freq = 'h'
        self.individual = False
        self.kernel_size = 3
        self.stack_types = ['generic', 'generic', 'generic']


def train_predict_dl(model_name, x_clean, x_corrupt, y_target, device,
                     train_epochs=10, lr=1e-3, batch_size=20):
    import torch

    B, L, C = x_clean.shape
    pred_len = y_target.shape[1]

    configs = Configs(enc_in=C, c_out=C, seq_len=L, pred_len=pred_len)
    model = build_model(model_name, configs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    x_t = torch.FloatTensor(x_clean).to(device)
    y_t = torch.FloatTensor(y_target).to(device)
    xc_t = torch.FloatTensor(x_corrupt).to(device)

    model.train()
    n = x_t.size(0)
    for epoch in range(train_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_t[idx], y_t[idx]
            optimizer.zero_grad()
            out = model(xb, None, None, None)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_clean = model(x_t, None, None, None).cpu().numpy()
        pred_corrupt = model(xc_t, None, None, None).cpu().numpy()

    del model, x_t, y_t, xc_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_clean, pred_corrupt

def fit_predict_stat(model_name, x_clean, x_corrupt, y_target):
    B, L, C = x_clean.shape
    pred_len = y_target.shape[1]
    configs = Configs(enc_in=C, c_out=C, seq_len=L, pred_len=pred_len)
    model = build_model(model_name, configs)
    model.fit(x_clean, y_target)         # 仅 Tree 模型真正训练, 其余 pass
    pred_clean = model.predict(x_clean)
    pred_corrupt = model.predict(x_corrupt)
    return pred_clean, pred_corrupt


def eval_npz(model_name, npz_path, device='cpu', train_epochs=10):
    data = np.load(npz_path)
    x_clean = data['x_clean'].astype(np.float32)
    x_corrupt = data['x_corrupt'].astype(np.float32)
    y_target = data['y_target'].astype(np.float32)

    t0 = time.time()
    if model_name in DL_MODELS:
        pred_clean, pred_corrupt = train_predict_dl(
            model_name, x_clean, x_corrupt, y_target, device, train_epochs)
    else:
        pred_clean, pred_corrupt = fit_predict_stat(
            model_name, x_clean, x_corrupt, y_target)
    elapsed = time.time() - t0

    H = y_target.shape[1]
    pred_clean = pred_clean[:, :H, :y_target.shape[2]]
    pred_corrupt = pred_corrupt[:, :H, :y_target.shape[2]]

    mse_clean = float(np.mean((pred_clean - y_target) ** 2))
    mae_clean = float(np.mean(np.abs(pred_clean - y_target)))
    mse_corrupt = float(np.mean((pred_corrupt - y_target) ** 2))
    mae_corrupt = float(np.mean(np.abs(pred_corrupt - y_target)))

    return {
        'mse_corrupt': mse_corrupt, 'mae_corrupt': mae_corrupt,
        'mse_clean': mse_clean, 'mae_clean': mae_clean,
        'n_samples': x_clean.shape[0], 'time_sec': round(elapsed, 2),
    }


def find_npz(npz_root, dataset, family, difficulty):
    candidates = [
        os.path.join(npz_root, dataset, f'{dataset}_{family}_{difficulty}.npz'),
        os.path.join(npz_root, f'{dataset}_{family}_{difficulty}.npz'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_root', type=str, default='./joltbench_output')
    parser.add_argument('--output_csv', type=str,
                        default='./tsfault_results/eval_results_extended.csv')
    parser.add_argument('--models', nargs='+', default=['all'])
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--skip_datasets', nargs='+', default=[])
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.models == ['all']:
        args.models = list(ALL_MODELS)
    if args.datasets is None:
        args.datasets = DATASETS
    args.datasets = [d for d in args.datasets if d not in args.skip_datasets]


    device = 'cpu'
    if args.gpu >= 0:
        try:
            import torch
            if torch.cuda.is_available():
                device = f'cuda:{args.gpu}'
        except ImportError:
            pass
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Output: {args.output_csv}")

    done = set()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    file_exists = os.path.exists(args.output_csv)
    if file_exists and args.resume:
        import pandas as pd
        try:
            df = pd.read_csv(args.output_csv)
            for _, r in df.iterrows():
                done.add((r['model'], r['dataset'], r['family'], r['difficulty']))
            print(f"Resume: 已完成 {len(done)} 个组合, 将跳过")
        except Exception:
            pass

    f_out = open(args.output_csv, 'a', newline='')
    fieldnames = ['model', 'dataset', 'family', 'difficulty',
                  'mse_corrupt', 'mae_corrupt', 'mse_clean', 'mae_clean',
                  'n_samples', 'time_sec']
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    total = len(args.models) * len(args.datasets) * len(FAMILIES) * len(DIFFICULTIES)
    print(f"总计算量: {total} 个组合\n")

    done_cnt = skip_cnt = fail_cnt = 0
    for model_name in args.models:
        print(f"{'=' * 64}\n模型: {model_name}\n{'=' * 64}")
        for dataset in args.datasets:
            for family in FAMILIES:
                for difficulty in DIFFICULTIES:
                    key = (model_name, dataset, family, difficulty)
                    if key in done:
                        skip_cnt += 1
                        continue
                    npz_path = find_npz(args.npz_root, dataset, family, difficulty)
                    if npz_path is None:
                        print(f"   ⚠️  {dataset}/{family}/{difficulty}: .npz 不存在")
                        fail_cnt += 1
                        continue
                    try:
                        r = eval_npz(model_name, npz_path, device, args.train_epochs)
                    except Exception as e:
                        print(f"   ❌ {dataset}/{family}/{difficulty}: {str(e)[:120]}")
                        fail_cnt += 1
                        continue
                    print(f"   {dataset:>11s} | {family} | {difficulty}: "
                          f"clean={r['mse_clean']:.4f} corrupt={r['mse_corrupt']:.4f} "
                          f"Δ={r['mse_corrupt'] - r['mse_clean']:+.4f} [{r['time_sec']}s]")
                    writer.writerow({'model': model_name, 'dataset': dataset,
                                     'family': family, 'difficulty': difficulty, **r})
                    f_out.flush()
                    done_cnt += 1
    f_out.close()
    print(f"\n{'=' * 64}")
    print(f"✅ 完成 {done_cnt}, 跳过 {skip_cnt}, 失败 {fail_cnt}")
    print(f"   输出: {args.output_csv}")


if __name__ == '__main__':
    main()
