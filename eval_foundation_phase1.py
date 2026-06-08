import os, sys, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foundation_models import build_model, FOUNDATION_MODELS

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Weather']
FAMILIES = ['family1', 'family2', 'family3', 'family4']
DIFFS = ['d02', 'd04', 'd06', 'd08', 'd10']

COLUMNS = ['model', 'dataset', 'family', 'difficulty',
           'mse_corrupt', 'mae_corrupt', 'mse_clean', 'mae_clean',
           'n_samples', 'time_sec']


def find_npz(root, ds, fam, d):
    for p in [os.path.join(root, ds, f'{ds}_{fam}_{d}.npz'),
              os.path.join(root, f'{ds}_{fam}_{d}.npz')]:
        if os.path.exists(p):
            return p
    return None


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=FOUNDATION_MODELS,
                    help='timesfm chronos moirai 任意组合')
    ap.add_argument('--npz_root', default='./joltbench_output')
    ap.add_argument('--output_csv', default='./tsfault_results/eval_foundation.csv')
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--skip_datasets', nargs='+', default=[])
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--resume', action='store_true',
                    help='若 output_csv 已存在, 跳过已完成的 (model,dataset,family,difficulty)')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if args.gpu != 'cpu' else 'cpu'

    datasets = [d for d in args.datasets if d not in args.skip_datasets]
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)

    done = set()
    rows = []
    if args.resume and os.path.exists(args.output_csv):
        prev = pd.read_csv(args.output_csv)
        rows = prev.to_dict('records')
        for _, r in prev.iterrows():
            done.add((r['model'], r['dataset'], r['family'], r['difficulty']))
        print(f"[resume] 已完成 {len(done)} 条")

    for mname in args.models:
        print(f"\n{'='*60}\n加载基础模型: {mname}\n{'='*60}")
        t0 = time.time()
        try:
            model = build_model(mname, device=device)
        except Exception as e:
            print(f"[跳过] {mname} 加载失败: {e}")
            continue
        print(f"  加载耗时 {time.time()-t0:.1f}s")

        for ds in datasets:
            for fam in FAMILIES:
                for d in DIFFS:
                    key = (mname, ds, fam, d)
                    if key in done:
                        continue
                    npz = find_npz(args.npz_root, ds, fam, d)
                    if npz is None:
                        print(f"  [缺失] {ds}/{fam}/{d}")
                        continue
                    data = np.load(npz)
                    xc, xk, yt = data['x_clean'], data['x_corrupt'], data['y_target']

                    t1 = time.time()
                    try:
                        pred_clean = model.predict(xc)
                        pred_corrupt = model.predict(xk)
                    except Exception as e:
                        print(f"  [推理失败] {mname}/{ds}/{fam}/{d}: {e}")
                        continue
                    dt = time.time() - t1

                    H = yt.shape[1]
                    pred_clean = pred_clean[:, :H, :]
                    pred_corrupt = pred_corrupt[:, :H, :]

                    rows.append({
                        'model': mname, 'dataset': ds, 'family': fam, 'difficulty': d,
                        'mse_corrupt': mse(pred_corrupt, yt),
                        'mae_corrupt': mae(pred_corrupt, yt),
                        'mse_clean': mse(pred_clean, yt),
                        'mae_clean': mae(pred_clean, yt),
                        'n_samples': int(yt.shape[0]),
                        'time_sec': round(dt, 3),
                    })
                    print(f"  ✓ {mname} {ds}/{fam}/{d}  "
                          f"clean={rows[-1]['mse_clean']:.4f} corrupt={rows[-1]['mse_corrupt']:.4f} ({dt:.1f}s)")
                    if rows:
                        pd.DataFrame(rows)[COLUMNS].to_csv(args.output_csv, index=False)

        del model
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    if rows:
        pd.DataFrame(rows)[COLUMNS].to_csv(args.output_csv, index=False)

    print(f"\n完成! 写出 {len(rows)} 行 -> {args.output_csv}")


if __name__ == '__main__':
    main()
