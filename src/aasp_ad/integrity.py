"""Data Integrity Assessment for TSB-AD-U (read-only audit).

Scans all series, computes data-integrity indicators and writes a report
table. Does NOT modify the raw data: following the project guideline, spikes
and level-shifts are anomalies that MUST BE KEPT, not noise to be cleaned.

Run:    PYTHONPATH=src python -m aasp_ad.integrity
Output: outputs/eda/tables/data_integrity.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from aasp_ad.config import EDA_TABLES_DIR, TSB_AD_U_DIR, ensure_output_dirs

CONSTANT_STD_EPS = 1e-8


def _segment_lengths(y_bool: np.ndarray) -> np.ndarray:
    """Lengths of contiguous anomaly segments (pad 0 on both ends to catch
    boundary anomalies)."""
    d = np.diff(np.concatenate(([0], y_bool.astype(int), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return ends - starts


def analyze_series(path: Path) -> dict:
    """Compute integrity indicators for one series. Read-only, no modification."""
    df = pd.read_csv(path)
    col_data, col_label = df.iloc[:, 0], df.iloc[:, -1]

    x = pd.to_numeric(col_data, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(col_label, errors="coerce").to_numpy(dtype=float)
    y_bool = y == 1

    uniq_labels = np.unique(y[~np.isnan(y)])
    seg_len = _segment_lengths(y_bool)
    std = float(np.nanstd(x))

    return {
        "file": path.name,
        "dataset": path.name.split("_")[1],
        "length": len(x),
        "n_nan": int(np.isnan(x).sum()),
        "n_inf": int(np.isinf(x).sum()),
        "dtype_numeric": bool(pd.api.types.is_numeric_dtype(col_data)),
        "label_ok": bool(np.all(np.isin(uniq_labels, [0, 1]))),
        "n_anomaly": int(y_bool.sum()),
        "contamination": float(y_bool.mean()),
        "std": std,
        "is_constant": bool(std < CONSTANT_STD_EPS),
        "n_segments": int(len(seg_len)),
        "mean_seg_len": float(seg_len.mean()) if len(seg_len) else 0.0,
        "max_seg_len": int(seg_len.max()) if len(seg_len) else 0,
        "anomaly_at_start": bool(y_bool[0]),
        "anomaly_at_end": bool(y_bool[-1]),
    }


def main() -> None:
    files = sorted(TSB_AD_U_DIR.glob("*.csv"))
    print(f"Scanning {len(files)} series in {TSB_AD_U_DIR} ...")
    df = pd.DataFrame(analyze_series(f) for f in files)

    ensure_output_dirs()
    out_path = EDA_TABLES_DIR / "data_integrity.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")

    # ---- Problem summary (ASCII-safe for the Windows cp1252 console) ----
    print("\n== Integrity flags (n series) ==")
    print("  has NaN              :", int((df.n_nan > 0).sum()))
    print("  has Inf              :", int((df.n_inf > 0).sum()))
    print("  non-numeric data col :", int((~df.dtype_numeric).sum()))
    print("  label not in {0,1}   :", int((~df.label_ok).sum()))
    print("  constant series      :", int(df.is_constant.sum()))
    print("  contamination == 0   :", int((df.contamination == 0).sum()))
    print("  contamination == 1   :", int((df.contamination == 1).sum()))
    print("  anomaly at start     :", int(df.anomaly_at_start.sum()))
    print("  anomaly at end       :", int(df.anomaly_at_end.sum()))

    print("\n== Length ==")
    print(
        "  min / median / max   :",
        int(df.length.min()),
        "/",
        int(df.length.median()),
        "/",
        int(df.length.max()),
    )
    print("  length < 100         :", int((df.length < 100).sum()))
    print("  length < 256         :", int((df.length < 256).sum()))

    print("\n== Contamination ==")
    print(
        f"  mean / median        : {df.contamination.mean():.4f} / {df.contamination.median():.4f}"
    )


if __name__ == "__main__":
    main()
