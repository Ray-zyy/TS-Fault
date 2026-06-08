<div align="center">

# ⚡ TS-Fault

### When Clean Accuracy Lies: Benchmarking Time Series Forecasters Against Structural Faults

**Yuyang Zhao**¹ · **Lian Xu**² · **Hao Miao**³ · **Hao Xue**¹ ✉

¹ Hong Kong University of Science and Technology (Guangzhou) &nbsp;·&nbsp; ² The University of Western Australia &nbsp;·&nbsp; ³ The Hong Kong Polytechnic University

<br>

<samp>21 models &nbsp;·&nbsp; 6 datasets &nbsp;·&nbsp; 4 failure modes &nbsp;·&nbsp; 5 severity levels &nbsp;·&nbsp; 2,500 evaluated cells</samp>

</div>

---

> **TL;DR — Clean-data leaderboards do not predict robustness.**
> TS-Fault stops treating faults as i.i.d. noise. It injects **four structurally-coupled failure modes** — abstracted from the anatomy of real grid-scale failures — into the **input window** of standard forecasting datasets, then measures how far each model degrades. Under the two *mechanism-level* modes, the ranking is **reordered**: state-of-the-art Transformers and foundation models collapse, while a plain **LSTM/GRU** — or even a naïve baseline — becomes the most robust.

<div align="center">
  <img src="figures/framework.png" width="100%" alt="The TS-Fault pipeline">
  <br>
  <em>The TS-Fault pipeline. A clean pair <code>(X, Y)</code> is turned into a structured instance: a window-importance score selects the prediction-critical window <code>W★</code>, a parameterized fault operator <code>T<sub>Θ</sub></code> acts there at a chosen severity, and the scenario parameters <code>Θ</code> and difficulty <code>δ</code> are reported with the instance and used to condition evaluation.</em>
</div>

---

## 📌 Why TS-Fault?

For two decades, long-horizon forecasting progress has been measured by **one number** — average error on clean, complete, evenly-sampled held-out data — under the implicit assumption that it predicts deployed reliability. Real faults are not white noise. They are *structured events*: a transient with onset–peak–decay, a silently broken cross-variable dependency, a regime change coupled with block-missingness, a fault cascading through a sensing pipeline. None of these can be expressed by i.i.d. perturbations, no matter how the variance or masking rate is tuned.

TS-Fault changes the **object of evaluation**, replacing the clean test pair `(X, Y)` with a structured instance `(X̃, Ỹ, Θ, δ)` produced by an explicit, parameterized fault operator. Because `Θ` is exposed, a model's degradation can be attributed to a **named mechanism** at a **tunable severity** — turning a pass/fail noise test into an ablation-style diagnostic tool. It is the time-series counterpart to **ImageNet-C**.

### Headline findings

1. **Clean accuracy *anti*-correlates with robustness** — Spearman ρ = **−0.544** (p = 0.011) across all 21 models (−0.509 over the 18 non-foundation models, so foundation models *strengthen* the effect).
2. **Observation-level faults preserve the ranking; mechanism-level faults destroy it** — ρ > **0.92** under Modes I/II vs ρ < **0.06** under Modes III/IV.
3. **Catastrophe is structural** — all **884** catastrophic failures (≥10× error inflation) fall in the two mechanism-level modes; Modes I/II never trigger one.
4. **Foundation models are strong but fragile** — TimesFM is **2nd** on clean MSE yet **worst** of all 21 on robustness (ratio ≈ 555).

---

## 🧩 The four failure modes

Each mode corrupts **only the lookback window** (`L = 336`); the forecast target (`H = 96`) is **never** touched, so degradation isolates the cost of a corrupted history. A single difficulty scalar `d ∈ {0.2, 0.4, 0.6, 0.8, 1.0}` (written `d02 … d10`) sweeps severity monotonically. The modes span a **2 × 2 taxonomy** — *observation- vs. mechanism-level* × *univariate vs. multivariate* — and form a constructive minimum: omitting any one renders an entire class of deployment failure invisible.

<div align="center">
  <img src="figures/mode_taxonomy.png" width="78%" alt="The 2x2 fault taxonomy">
</div>

| File | Mode | Taxonomy cell | What breaks | Real-world analogue |
|:---|:---|:---|:---|:---|
| `Mode1.py` | **I · Time-Warped Shock** | Observation × Univariate | a shaped transient (onset–peak–decay) + a local time-warp | demand spike / brief sensor glitch |
| `Mode2.py` | **II · Dependency-Fracture Shock** | Observation × Multivariate | cross-channel lead–lag and gain silently falsified; each series alone still looks plausible | regions that usually co-move decouple |
| `Mode3.py` | **III · Regime-Transition Missingness** | Mechanism × Univariate | a regime switch + state-dependent block-missingness around the transition *(hardest mode)* | grid emergency / firm load shed |
| `Mode4.py` | **IV · Cascading Sensor-to-System Failure** | Mechanism × Multivariate | an upstream drift propagates downstream with a lag, inducing secondary dropouts | a fault cascading through coupled subsystems |

> **Observation-level (I, II)** corrupt observations while the data-generating process stays intact → *mild*.
> **Mechanism-level (III, IV)** rewrite the process itself → *severe*, and account for **100%** of catastrophic failures.

---

## 📂 Repository contents

This repository ships the **data-generation pipeline, model implementations, evaluation drivers, and the master results table**. The four fault generators are included in full, so the perturbed dataset can be regenerated from scratch — it is **not** committed as binary `.npz`.

```
.
├── README.md
├── requirements.txt
│
├── benchmark.py                 # benchmark orchestration helpers
├── dataset_loader.py            # load + normalise the clean CSVs into windows
├── window_selector.py           # select the anchor lookback / horizon windows
│
├── Mode1.py                     # Fault Mode I   — Time-Warped Shock
├── Mode2.py                     # Fault Mode II  — Dependency-Fracture Shock
├── Mode3.py                     # Fault Mode III — Regime-Transition Missingness
├── Mode4.py                     # Fault Mode IV  — Cascading Sensor-to-System Failure
├── run_TS-Fault.py              # ▶ MAIN entry point: build the perturbed benchmark
│
├── classical_models.py          # statistical / linear / recurrent / conv. forecasters
├── foundation_models.py         # TimesFM / Chronos / Moirai (zero-shot wrappers)
├── eval_classical_phase1.py     # evaluate classical + baseline models → results CSV
├── eval_foundation_phase1.py    # evaluate foundation models → results CSV
│
├── eval_results_full_23.csv     # master results table (wide schema)
└── figures/                     # paper figures used in this README
```

> **Plotting scripts and the generated `TS-Fault_output/` `.npz` files are not committed** — the perturbed data is fully reproducible from `run_TS-Fault.py` + the four `Mode*.py` generators.

### The 21 evaluated forecasters

The paper evaluates **21 models** across **six methodological families**, so the conclusions are not an artifact of a single inductive bias:

| Family | Models |
|:---|:---|
| **Statistical** (4) | Naive · SeasonalNaive · ARIMA · ETS |
| **Linear / lightweight** (3) | DLinear · NLinear · N-BEATS |
| **Recurrent / convolutional** (3) | LSTM · GRU · TCN |
| **Decomposition Transformer** (2) | Autoformer · FEDformer |
| **Attention / SOTA** (6) | PatchTST · iTransformer · TimeXer · TimeMixer · TimesNet · Nonstationary-Transformer |
| **Foundation, zero-shot** (3) | TimesFM · Chronos · Moirai |

*Statistical, linear, and recurrent/convolutional models live in `classical_models.py`; the eight Transformer-family models are trained via the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (see [§6](#-evaluating-the-transformer-family-models)); the three foundation models run strictly zero-shot via `foundation_models.py`. The repository's `classical_models.py` additionally bundles a few exploratory baselines (e.g. RandomForest, XGBoost) that fall outside the 21-model paper subset.*

---

## 📊 Datasets (clean originals)

TS-Fault perturbs six widely-used long-horizon datasets spanning **energy, load, and climate**, deliberately chosen to cover a wide range of **dimensionality** (7 → 321 channels) and **granularity** (10-minute → hourly). Every clean window is a **length-336 history** and a **length-96 target**. We do **not** redistribute the raw data — download it and place each file at the path the loader expects.

| Dataset | Domain | Channels | Granularity | Path expected |
|:---|:---|:---:|:---|:---|
| ETTh1 / ETTh2 | Energy | 7 | hourly | `dataset/ETT-small/ETTh1.csv`, `…/ETTh2.csv` |
| ETTm1 / ETTm2 | Energy | 7 | 15-min | `dataset/ETT-small/ETTm1.csv`, `…/ETTm2.csv` |
| Electricity (ECL) | Load | 321 | hourly | `dataset/electricity/electricity.csv` |
| Weather | Climate | 21 | 10-min | `dataset/weather/weather.csv` |

<details>
<summary><b>Download links (canonical)</b></summary>

**All six datasets, one bundle (recommended)** — official Time-Series-Library mirror on HuggingFace (CC BY 4.0): <https://huggingface.co/datasets/thuml/Time-Series-Library>

```python
from huggingface_hub import hf_hub_download
for f in ["ETT-small/ETTh1.csv", "ETT-small/ETTh2.csv",
          "ETT-small/ETTm1.csv", "ETT-small/ETTm2.csv",
          "electricity/electricity.csv", "weather/weather.csv"]:
    hf_hub_download("thuml/Time-Series-Library", f, repo_type="dataset")
```

* **ETT only (original source):** <https://github.com/zhouhaoyi/ETDataset>
* **TSLib data instructions:** <https://github.com/thuml/Time-Series-Library>

Each CSV has a leading `date` column, the feature columns, and a target column `OT` (standard TSLib convention).
</details>

---

## 🚀 Quick start

### 1 · Install

```bash
git clone https://github.com/Ray-zyy/TS-Fault.git
cd TS-Fault
pip install -r requirements.txt
```

### 2 · Get the clean datasets

Download the six CSVs (see [§Datasets](#-datasets-clean-originals)) into:

```
dataset/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
dataset/electricity/electricity.csv
dataset/weather/weather.csv
```

### 3 · Generate the perturbed benchmark

`run_TS-Fault.py` slides the anchor windows (`window_selector.py`), loads & normalises the clean series (`dataset_loader.py`), and applies each of the four fault modes (`Mode1.py … Mode4.py`) across all five difficulties:

```bash
python run_TS-Fault.py \
    --data_root ./dataset \
    --out ./TS-Fault_output \
    --n_windows 20
```

This writes one `.npz` per `(dataset, Mode, difficulty)` to `TS-Fault_output/<Dataset>/<Dataset>_Mode<k>_d<dd>.npz` — **`6 datasets × 4 modes × 5 difficulties = 120 files`**, each containing:

| array | shape | meaning |
|:---|:---|:---|
| `x_clean` | `(N, 336, C)` | clean lookback windows |
| `x_corrupt` | `(N, 336, C)` | perturbed lookback windows |
| `y_target` | `(N, 96, C)` | forecast target (**never** perturbed) |

### 4 · Evaluate

**Classical / baseline models** (deep ones train once per dataset on clean windows; statistical ones need no training):

```bash
python eval_classical_phase1.py \
    --models Naive SeasonalNaive ARIMA ETS \
             DLinear NLinear NBEATS \
             LSTM GRU TCN \
    --npz_root ./TS-Fault_output \
    --out ./results/eval_classical.csv --gpu 0 --resume
```

**Foundation models** (zero-shot — see [§7](#-installing-the-foundation-models)):

```bash
python eval_foundation_phase1.py \
    --models timesfm chronos moirai \
    --npz_root ./TS-Fault_output \
    --out ./results/eval_foundation.csv --gpu 0 --resume
```

Each evaluator emits the **canonical wide schema**, one row per `(model, dataset, Mode, difficulty)`:

```
model, dataset, Mode, difficulty, mse_corrupt, mae_corrupt, mse_clean, mae_clean, n_samples, time_sec
```

`mse_clean` is the error on the clean input window, `mse_corrupt` on the perturbed window — **both predicting the same untouched target**. The **robustness ratio** `r = mse_corrupt / mse_clean` isolates the cost of corrupted history; `r = 1` is perfect robustness and `r ≥ 10` is a **catastrophic failure**. Concatenate the per-group CSVs (classical + foundation + the TSLib Transformer rows) into the master table `eval_results_full_23.csv`.

---

## 🧪 Evaluating the Transformer-family models

The eight Transformer/attention models (PatchTST, iTransformer, Autoformer, FEDformer, Nonstationary-Transformer, TimeMixer, TimeXer, TimesNet) are trained and run with the **[Time-Series-Library](https://github.com/thuml/Time-Series-Library)** (TSLib).

1. Clone TSLib and place the same six clean CSVs under its `./dataset/`.
2. Train / evaluate each model with `run.py` under the TS-Fault protocol (`seq_len=336`, `pred_len=96`, `features=M`, MSE loss). Example:

   ```bash
   python -u run.py \
       --task_name long_term_forecast --is_training 1 \
       --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
       --model_id ETTh1_336_96 --model PatchTST --data ETTh1 \
       --features M --seq_len 336 --label_len 48 --pred_len 96 \
       --e_layers 2 --d_layers 1 --enc_in 7 --dec_in 7 --c_out 7 \
       --train_epochs 10 --batch_size 32 --learning_rate 1e-4 --itr 1
   ```

3. Score each trained checkpoint on the TS-Fault `.npz` windows (clean vs. corrupt) and write rows in the same wide schema, then merge into `eval_results_full_23.csv`.

<details>
<summary><b>Shared hyperparameters & model-specific notes</b></summary>

Shared across all models: `seq_len=336, label_len=48, pred_len=96, e_layers=2, d_layers=1, d_model=128–512, train_epochs=10, batch_size=32` (16 for Electricity), MSE loss, instance/RevIN normalisation.

* **TimeMixer** — `--down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg --channel_independence 1`
* **Nonstationary-Transformer** — `--p_hidden_dims 128 128 --p_hidden_layers 2`
* **DLinear** — `--individual` is a store-true flag
* Reduce `d_model` / `batch_size` for TimesNet / FEDformer / Autoformer on 11 GB GPUs.
</details>

---

## 🧱 Installing the foundation models

These libraries pull heavy / conflicting dependencies. The recipe that works (verified on CUDA 12.1, `torch 2.5.1`):

```bash
# TimesFM + Chronos are clean pip installs
pip install timesfm chronos-forecasting

# Moirai (uni2ts) pins torch<2.5 and WILL try to downgrade torch.
# Install WITHOUT deps so it cannot touch torch, then hand-add what it needs:
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts && pip install -e . --no-deps && touch .env
pip install "gluonts==0.14.4" "einops==0.7.0" jaxtyping --no-deps
```

Behind a mirror, set `export HF_ENDPOINT=https://hf-mirror.com` before the first run. All three run **zero-shot** (no fine-tuning) and are wrapped channel-independently for multivariate inputs.

| Model | Pretrained weights |
|:---|:---|
| TimesFM | <https://huggingface.co/google/timesfm-2.0-500m-pytorch> |
| Chronos-Bolt | <https://huggingface.co/amazon/chronos-bolt-base> |
| Moirai | <https://huggingface.co/Salesforce/moirai-1.1-R-base> |

---

## 🏆 Results

### Aggregate — the clean and faulted leaderboards are near mirror images

The most robust models (**GRU**, **LSTM**, **TCN**, near-unit ratios) are only mid-pack on clean accuracy, while the clean-accuracy leaders (**N-BEATS**, **TimesFM**, **iTransformer**, **TimeXer**) sit at the bottom of the robustness order with ratios in the hundreds.

<div align="center">
  <img src="figures/result1.png" width="100%" alt="Aggregate results: robustness ratio, severity sweep, per-mode breakdown">
  <br>
  <em>Left: clean/faulted MSE-MAE and robustness ratio. Center: degradation across severities <code>d02→d10</code> and the <code>d10/d02</code> sensitivity slope. Right: per-mode relative degradation — note the jump from single-digit % under Modes I/II to <b>thousands–to–tens-of-thousands %</b> under Modes III/IV.</em>
</div>

### Rank reordering — preserved under observation-level faults, destroyed under mechanism-level

| Mode | Spearman ρ (clean vs. faulted rank) | p | Verdict |
|:---|:---:|:---:|:---|
| I · Time-Warped Shock | **+0.925** | < 0.001 | ranking preserved |
| II · Dependency Fracture | **+0.952** | < 0.001 | ranking preserved |
| III · Regime Missingness | **+0.032** | 0.889 | ranking **destroyed** |
| IV · Cascading Failure | **+0.055** | 0.814 | ranking **destroyed** |

### Per-dataset — fragility is structural; dimensionality sets the amplitude

The model ordering is highly consistent across datasets (fragility is a property of the *architecture*), but dense cross-channel correlation amplifies it: on the 321-channel **Electricity**, TimesFM's ratio peaks at **2090** and TimeXer at **1112**, while GRU/LSTM stay near **1.2** everywhere.

<div align="center">
  <img src="figures/result2.png" width="100%" alt="Per-dataset corrupted MSE and robustness ratios across six datasets">
  <br>
  <img src="figures/vis1.png" width="92%" alt="MSE on corrupted inputs per dataset">
  <br>
  <em>MSE on corrupted inputs per dataset (lower is better). Recurrent models (LSTM, GRU) sit at the bottom of every panel; foundation models (gold) sit at the top.</em>
</div>

> **Catastrophic failures (`r ≥ 10`):** 884 total — **0** in Mode I, **0** in Mode II, **537** in Mode III (85.9% of its cells), **347** in Mode IV (55.5%). Mechanism-level modes account for **100%** of them.

---

## 🔁 Reproducibility

TS-Fault is designed to be fully reproducible. Because faulted instances are produced by an **explicit operator at evaluation time**, the benchmark can be regenerated at any severity by re-sweeping `κ`, and previously-unexposed `Θ` combinations can be held out at release time to guard against benchmark gaming. We release the parameterized fault generators (with their `Θ` schemas and the unified window-importance front-end), the evaluation harness with per-model configs, and the master results table.


---

## 📜 License & acknowledgements

TS-Fault code is released under the **MIT licence**. Some baseline implementations are adapted from external sources and retain their own licences: **TCN** from [`locuslab/TCN`](https://github.com/locuslab/TCN) (MIT), **NLinear/DLinear** from [`cure-lab/LTSF-Linear`](https://github.com/cure-lab/LTSF-Linear) (Apache-2.0), and **N-BEATS** from [`ServiceNow/N-BEATS`](https://github.com/ServiceNow/N-BEATS) (**CC-BY-NC-4.0, non-commercial**). The eight Transformer baselines come from the [**Time-Series-Library**](https://github.com/thuml/Time-Series-Library). Datasets (ETT, Electricity, Weather) are released by their original authors under CC BY 4.0 and are not redistributed here.

<div align="center">
<br>
<sub>No model occupies the accurate-<i>and</i>-robust regime. That empty quadrant is the open problem TS-Fault is built to drive progress on.</sub>
</div>
