# AASP-UML

A label-free statistical **patch** baseline for univariate time-series anomaly
detection on **TSB-AD-U**.

AASP-UML describes overlapping patches with 27 hand-crafted statistical features,
scores them by k-nearest-neighbor distance to patches from the anomaly-free
training prefix, and fuses evidence across three temporal scales. It uses
**no anomaly labels** and **no neural networks**; every normalization and
calibration statistic is restricted to the training prefix, so the pipeline is
leakage-free by construction.

> Scope: this repository contains the **model** and the **exploratory data
> analysis (EDA)** only.

## Method at a glance

1. Robust prefix scaling by train-portion median/MAD.
2. Stride-1 patch extraction at lengths `L ∈ {64, 96, 128}`.
3. 27 statistical features per patch (amplitude / shape / temporal / spectral / trend).
4. kNN distance to training-prefix patches (`k = 3`) in the standardized feature space.
5. `O(n)` back-projection of patch scores to per-point scores.
6. Safe-prefix calibration and multi-scale fusion.

The core implementation is in [`src/aasp_ad/unsupervised_patch.py`](src/aasp_ad/unsupervised_patch.py).

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pulls `TSB_AD==1.5`, which provides the official VUS-PR metric.
If pip cannot resolve it, clone the official repository and expose it on `PYTHONPATH`:

```powershell
git clone https://github.com/TheDatumOrg/TSB-AD external\TSB-AD
$env:PYTHONPATH = "src;external\TSB-AD"
```

## Data setup

Download the official TSB-AD-U univariate CSV files from
<https://github.com/TheDatumOrg/TSB-AD> and place them here:

```text
data/raw/TSB-AD-U/
```

The small split lists committed in `data/*.csv` define the exact series used by
each split (`eva350`, `eva_full822`, `tuning`). The raw benchmark data itself is
not committed.

## Run the model

```powershell
$env:PYTHONPATH = "src"
python -m aasp_ad.run_aasp_eval --method knn --split eva350
python -m aasp_ad.run_aasp_eval --method iforest --split eva_full822
```

Each run writes a per-series VUS-PR CSV under `outputs/results/` with an explicit
protocol marker (`eval_protocol = TSB_AD_find_length_rank_opt_250`), the per-series
`sliding_window`, and per-series `status`/`error` fields.

## Exploratory data analysis

- [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb) — interactive EDA of TSB-AD-U.
- [`notebooks/eda_tsb_ad_u/run_all_eda.py`](notebooks/eda_tsb_ad_u/run_all_eda.py) —
  regenerates the committed EDA tables and figures.

```powershell
$env:PYTHONPATH = "src"
python notebooks/eda_tsb_ad_u/run_all_eda.py
```

Committed EDA artifacts live in `outputs/eda/tables/` and `outputs/eda/figures/`.

## Repository layout

```text
data/                     # public split lists (raw data downloaded separately)
notebooks/                # EDA notebook + regeneration script
outputs/eda/              # committed EDA tables and figures
src/aasp_ad/
  config.py               # reproducible paths and split loading
  unsupervised_patch.py   # AASP-UML patch features and kNN / iForest scorers
  run_aasp_eval.py        # model evaluator (VUS-PR, leakage-free)
  eda_features.py         # meta-feature extraction
  profile.py              # per-series profiling
  integrity.py            # dataset integrity checks
  figures.py              # figure generation
tools/                    # split-safety guard and hygiene checks
```

## Reproducibility

- Paths are derived relative to the repository (via `pathlib`), so the project runs
  after a fresh clone without editing any absolute paths.
- All model normalization and calibration statistics are fitted on the training
  prefix only; a fail-closed guard rejects series whose prefix is too short.

## Data license

TSB-AD-U is publicly available from its authors
(<https://github.com/TheDatumOrg/TSB-AD>). The raw benchmark data is not
redistributed here; only the small public split lists are committed.
