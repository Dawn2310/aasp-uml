"""Statistical Profiling + Label/Segment Analysis for TSB-AD-U (stage 5 + 6).

Computes distributional (amplitude) statistics and label/segment features per
series, writing 2 tables for the paper's Dataset Characterization section:
  - eda_series_profile.csv  : per-series (870 rows)
  - eda_dataset_summary.csv : rollup by dataset group

Read-only, numpy-only (no extra dependency).

Run:    PYTHONPATH=src python -m aasp_ad.profile
"""

from pathlib import Path

import numpy as np
import pandas as pd

from aasp_ad.config import EDA_TABLES_DIR, TSB_AD_U_DIR, ensure_output_dirs

EPS = 1e-12
OUTLIER_K = 3.0  # robust-z threshold (|x-median|/MAD) for counting outliers

# Provisional thresholds for the point/sequence descriptive label (adjustable).
POINT_MAX_MEAN_SEG = 2.0
SEQ_MIN_MEAN_SEG = 10.0


def segment_lengths(y_bool: np.ndarray) -> np.ndarray:
    """Lengths of contiguous anomaly segments (pad 0 on both ends to catch
    boundary anomalies)."""
    d = np.diff(np.concatenate(([0], y_bool.astype(int), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return ends - starts


def _anomaly_type(mean_seg_len: float) -> str:
    if mean_seg_len < POINT_MAX_MEAN_SEG:
        return "point"
    if mean_seg_len >= SEQ_MIN_MEAN_SEG:
        return "sequence"
    return "mixed"


def profile_series(path: Path) -> dict:
    df = pd.read_csv(path)
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y_bool = pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(dtype=float) == 1

    # --- Stage 5: distributional / amplitude statistics ---
    mean, std = float(np.nanmean(x)), float(np.nanstd(x))
    median = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - median)))
    z = (x - mean) / (std + EPS)
    robust_z = np.abs(x - median) / (mad + EPS)

    # --- Stage 6: label / segment ---
    n_anom = int(y_bool.sum())
    seg = segment_lengths(y_bool)
    mean_seg = float(seg.mean()) if len(seg) else 0.0
    pts_in_seq = int(seg[seg > 1].sum()) if len(seg) else 0

    return {
        "file": path.name,
        "dataset": path.name.split("_")[1],
        "length": len(x),
        # amplitude / distribution
        "mean": mean,
        "std": std,
        "median": median,
        "mad": mad,
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "range": float(np.nanmax(x) - np.nanmin(x)),
        "iqr": float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
        "skewness": float(np.nanmean(z**3)),
        "kurtosis": float(np.nanmean(z**4) - 3.0),  # excess (Fisher)
        "outlier_ratio": float(np.nanmean(robust_z > OUTLIER_K)),
        # label / segment
        "n_anomaly": n_anom,
        "contamination": float(y_bool.mean()),
        "n_segments": int(len(seg)),
        "min_seg_len": int(seg.min()) if len(seg) else 0,
        "mean_seg_len": mean_seg,
        "max_seg_len": int(seg.max()) if len(seg) else 0,
        "frac_points_in_seq": pts_in_seq / n_anom if n_anom else 0.0,
        "anomaly_type": _anomaly_type(mean_seg),
    }


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("dataset")
    summary = (
        pd.DataFrame(
            {
                "n_series": g.size(),
                "len_median": g["length"].median(),
                "contam_mean": g["contamination"].mean(),
                "n_seg_mean": g["n_segments"].mean(),
                "seg_len_mean": g["mean_seg_len"].mean(),
                "kurtosis_mean": g["kurtosis"].mean(),
                "outlier_ratio_mean": g["outlier_ratio"].mean(),
                "frac_pts_in_seq_mean": g["frac_points_in_seq"].mean(),
                "dominant_type": g["anomaly_type"].agg(lambda s: s.mode().iat[0]),
            }
        )
        .reset_index()
        .sort_values("n_series", ascending=False)
    )
    return summary


def main() -> None:
    files = sorted(TSB_AD_U_DIR.glob("*.csv"))
    print(f"Profiling {len(files)} series ...")
    df = pd.DataFrame(profile_series(f) for f in files)

    ensure_output_dirs()
    profile_path = EDA_TABLES_DIR / "eda_series_profile.csv"
    summary_path = EDA_TABLES_DIR / "eda_dataset_summary.csv"
    df.to_csv(profile_path, index=False)

    summary = build_dataset_summary(df)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {len(df)} rows -> {profile_path}")
    print(f"Wrote {len(summary)} rows -> {summary_path}")

    print("\n== Anomaly type (n series) ==")
    print(df["anomaly_type"].value_counts().to_string())

    print("\n== Heavy-tail (amplitude branch) ==")
    print("  kurtosis > 5         :", int((df["kurtosis"] > 5).sum()))
    print("  kurtosis > 20        :", int((df["kurtosis"] > 20).sum()))
    print("  outlier_ratio > 1%   :", int((df["outlier_ratio"] > 0.01).sum()))

    print("\n== Dataset summary ==")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
