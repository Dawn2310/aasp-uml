#!/usr/bin/env python
"""
==========================================================================
 AASP-UML: Comprehensive Exploratory Data Analysis on TSB-AD-U Eva-Full
==========================================================================

Generates 8 publication-quality figures that motivate and defend the
handcrafted-patch + multi-scale kNN anomaly detection method.

Run from project root:
    $env:PYTHONPATH="src"; python notebooks/eda_tsb_ad_u/run_all_eda.py

Output:
    outputs/eda/figures/fig01_series_length_distribution.pdf/png
    outputs/eda/figures/fig02_anomaly_ratio_distribution.pdf/png
    ...
    outputs/eda/figures/fig08_multiscale_score_example.pdf/png
    outputs/eda/tables/eda_metadata.csv

Protocol notes:
    - Labels are used ONLY for EDA characterization and evaluation,
      never for fitting the label-free models.
    - Median/MAD/scaler/kNN are fitted exclusively on TRAIN portions.
    - PCA/UMAP projections are fitted on train-normal patches only.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

# -- Project imports --
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aasp_ad.config import (
    EDA_FIGURES_DIR,
    EDA_TABLES_DIR,
    TSB_AD_U_DIR,
    ensure_output_dirs,
    eva_full_files,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).dropna()
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(dtype=np.int32)
    return x, y


def train_split_from_name(name: str) -> int | None:
    import re

    match = re.search(r"_tr_(\d+)_", name)
    return int(match.group(1)) if match else None


def rank01(score: np.ndarray) -> np.ndarray:
    rank = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    return (rank - 1.0) / (len(rank) - 1.0 + 1e-12)

# ==========================================================================
# Global style
# ==========================================================================
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
}
plt.rcParams.update(STYLE)

# Colour palette
C_NORMAL = "#3b82f6"       # blue
C_ANOMALY = "#ef4444"      # red
C_TRAIN_NORMAL = "#64748b" # slate
C_TEST_NORMAL = "#3b82f6"  # blue
C_TEST_ANOM = "#ef4444"    # red
C_PATCH_64 = "#f59e0b"     # amber
C_PATCH_96 = "#10b981"     # emerald
C_PATCH_128 = "#8b5cf6"   # violet
C_FUSED = "#ef4444"        # red

SAVEFMT = ["png", "pdf"]


def _save(fig, name):
    for fmt in SAVEFMT:
        fig.savefig(EDA_FIGURES_DIR / f"{name}.{fmt}")
    plt.close(fig)
    print(f"  -> saved {name}")


# ==========================================================================
# Step 0: Build metadata table
# ==========================================================================
def build_metadata(files: list[Path]) -> pd.DataFrame:
    """Build per-series metadata: length, train/test splits, anomaly stats."""
    rows = []
    for i, f in enumerate(files, 1):
        x, y = load_xy(f)
        ts = train_split_from_name(f.name)
        if ts is None or ts <= 0 or ts >= len(x):
            ts = len(x) // 2

        n = len(x)
        n_anom_full = int(y.sum())
        n_anom_test = int(y[ts:].sum())
        n_test = n - ts

        # Anomaly segments
        segments = []
        in_seg = False
        seg_start = 0
        for t in range(n):
            if y[t] == 1 and not in_seg:
                in_seg = True
                seg_start = t
            elif y[t] == 0 and in_seg:
                segments.append(t - seg_start)
                in_seg = False
        if in_seg:
            segments.append(n - seg_start)

        rows.append({
            "file": f.name,
            "length": n,
            "train_length": ts,
            "test_length": n_test,
            "anomaly_points": n_anom_full,
            "anomaly_ratio_full": n_anom_full / n if n > 0 else 0,
            "anomaly_points_test": n_anom_test,
            "anomaly_ratio_test": n_anom_test / n_test if n_test > 0 else 0,
            "n_segments": len(segments),
            "segment_lengths": segments,
        })
        if i % 100 == 0 or i == len(files):
            print(f"  metadata: {i}/{len(files)}", flush=True)

    return pd.DataFrame(rows)


# ==========================================================================
# Figure 1: Series length distribution
# ==========================================================================
def fig01_series_length(meta: pd.DataFrame):
    print("Figure 1: Series length distribution")
    lengths = meta["length"].values

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 4.5),
        gridspec_kw={"height_ratios": [1, 4]},
        sharex=True,
    )

    # Calculate log lengths first so we can use it for both
    log_lengths = np.log10(lengths)

    # Boxplot
    bp = ax_box.boxplot(
        log_lengths, vert=False, widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor=C_NORMAL, alpha=0.3, edgecolor=C_NORMAL),
        medianprops=dict(color=C_ANOMALY, linewidth=2),
        whiskerprops=dict(color=C_NORMAL),
        capprops=dict(color=C_NORMAL),
        flierprops=dict(marker=".", markersize=3, markerfacecolor=C_NORMAL, alpha=0.4),
    )
    ax_box.set_yticks([])
    ax_box.set_title("Distribution of Time-Series Lengths in TSB-AD-U Eva-Full", fontweight="bold")

    # Histogram (log-scale x)
    bins = np.linspace(log_lengths.min() - 0.1, log_lengths.max() + 0.1, 40)
    ax_hist.hist(log_lengths, bins=bins, color=C_NORMAL, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax_hist.set_xlabel("Series Length (log$_{10}$ scale)")
    ax_hist.set_ylabel("Number of Series")

    # Custom tick labels
    tick_vals = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    ax_hist.set_xticks(tick_vals)
    ax_hist.set_xticklabels([f"$10^{{{v}}}$" if v == int(v) else f"{10**v:.0f}" for v in tick_vals])

    # Stats annotation
    med = np.median(lengths)
    q25, q75 = np.percentile(lengths, [25, 75])
    stats_text = (
        f"N = {len(lengths)} series\n"
        f"Median = {med:,.0f}\n"
        f"IQR = [{q25:,.0f}, {q75:,.0f}]\n"
        f"Min = {lengths.min():,} | Max = {lengths.max():,}"
    )
    ax_hist.text(
        0.97, 0.95, stats_text,
        transform=ax_hist.transAxes, ha="right", va="top",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e2e8f0", alpha=0.9),
    )

    plt.tight_layout()
    _save(fig, "fig01_series_length_distribution")


# ==========================================================================
# Figure 2: Anomaly ratio distribution
# ==========================================================================
def fig02_anomaly_ratio(meta: pd.DataFrame):
    print("Figure 2: Anomaly ratio distribution")
    ratios_full = meta["anomaly_ratio_full"].values * 100
    ratios_test = meta["anomaly_ratio_test"].values * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, data, title, color in [
        (axes[0], ratios_full, "Full Series", C_NORMAL),
        (axes[1], ratios_test, "Test Portion Only", C_ANOMALY),
    ]:
        ax.hist(data, bins=50, color=color, alpha=0.65, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Anomaly Ratio (%)")
        ax.set_title(title, fontweight="bold")

        med = np.median(data)
        q25, q75 = np.percentile(data, [25, 75])
        ax.axvline(med, color="black", linestyle="--", linewidth=1, alpha=0.7, label=f"Median={med:.2f}%")
        ax.legend(fontsize=8)

        stats_text = (
            f"Median = {med:.2f}%\n"
            f"IQR = [{q25:.2f}%, {q75:.2f}%]\n"
            f"Min = {data.min():.2f}% | Max = {data.max():.1f}%"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e2e8f0", alpha=0.9),
        )

    axes[0].set_ylabel("Number of Series")
    fig.suptitle(
        "Distribution of Anomaly Ratios Across TSB-AD-U Eva-Full",
        fontweight="bold", fontsize=12, y=1.02,
    )
    plt.tight_layout()
    _save(fig, "fig02_anomaly_ratio_distribution")


# ==========================================================================
# Figure 3: Representative anomaly morphologies
# ==========================================================================
def fig03_anomaly_morphologies(files: list[Path]):
    """Select 6 representative anomaly types and plot them."""
    print("Figure 3: Representative anomaly morphologies")

    # Strategy: semi-automatic selection based on filename patterns and
    # anomaly segment characteristics
    candidates = {}
    for f in files:
        x, y = load_xy(f)
        ts = train_split_from_name(f.name) or len(x) // 2

        # Only look at series with anomalies in test portion
        if y[ts:].sum() < 1:
            continue

        # Compute segment info
        segments = []
        in_seg = False
        seg_start = 0
        for t in range(len(y)):
            if y[t] == 1 and not in_seg:
                in_seg = True
                seg_start = t
            elif y[t] == 0 and in_seg:
                segments.append((seg_start, t - 1, t - seg_start))
                in_seg = False
        if in_seg:
            segments.append((seg_start, len(y) - 1, len(y) - seg_start))

        if not segments:
            continue

        seg_lens = [s[2] for s in segments]
        max_seg = max(seg_lens)
        min_seg = min(seg_lens)
        n_segs = len(segments)

        # Compute robust z-score for anomaly amplitude
        ref = x[:ts]
        med = np.median(ref)
        mad = np.median(np.abs(ref - med)) + 1e-12
        z_anom = np.abs((x[y == 1] - med) / (1.4826 * mad))
        max_z = z_anom.max() if len(z_anom) > 0 else 0

        # Classify
        info = {
            "file": f, "x": x, "y": y, "ts": ts,
            "segments": segments, "max_z": max_z, "max_seg": max_seg,
            "min_seg": min_seg, "n_segs": n_segs,
            "length": len(x),
        }

        # Global point: very short segment, very high z-score
        if min_seg <= 5 and max_z > 5 and "global_point" not in candidates:
            candidates["global_point"] = info

        # Contextual: moderate z-score, short segment
        if 1 < max_z < 5 and min_seg <= 20 and "contextual" not in candidates:
            candidates["contextual"] = info

        # Collective: long segment
        if max_seg > 100 and "collective" not in candidates:
            candidates["collective"] = info

        # Trend: use series with clear trend anomaly (long segment, moderate z)
        if max_seg > 50 and max_z < 8 and n_segs <= 3 and "trend" not in candidates:
            if "trend" not in candidates or info["max_seg"] > candidates.get("trend", {}).get("max_seg", 0):
                candidates["trend"] = info

        # Shapelet: multiple short segments with moderate z-score
        if n_segs >= 3 and max_seg < 50 and "shapelet" not in candidates:
            candidates["shapelet"] = info

        # Seasonal: look for periodic-like patterns (YAHOO/NAB)
        if "YAHOO" in f.name and n_segs >= 2 and "seasonal" not in candidates:
            candidates["seasonal"] = info

        if len(candidates) >= 6:
            break

    # Fill missing types with whatever we have
    type_labels = ["global_point", "contextual", "shapelet", "seasonal", "collective", "trend"]
    type_titles = ["Global Point", "Contextual", "Shapelet", "Seasonal", "Collective", "Trend"]

    # If some types are missing, fill with any available
    available = list(candidates.keys())
    used_files = set()
    plot_data = []
    for tl, tt in zip(type_labels, type_titles):
        if tl in candidates and candidates[tl]["file"].name not in used_files:
            plot_data.append((tt, candidates[tl]))
            used_files.add(candidates[tl]["file"].name)
        elif available:
            # Fall back to any unused candidate
            for k in list(candidates.keys()):
                if candidates[k]["file"].name not in used_files:
                    plot_data.append((tt, candidates[k]))
                    used_files.add(candidates[k]["file"].name)
                    break

    n_plots = min(len(plot_data), 6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for idx in range(6):
        ax = axes[idx]
        if idx >= n_plots:
            ax.set_visible(False)
            continue

        title, info = plot_data[idx]
        x, y, ts = info["x"], info["y"], info["ts"]

        # Subsample if too long for clear visualization
        if len(x) > 5000:
            # Center around first anomaly in test
            anom_idx = np.where(y == 1)[0]
            if len(anom_idx) > 0:
                center = anom_idx[len(anom_idx) // 2]
                start = max(0, center - 2500)
                end = min(len(x), center + 2500)
            else:
                start, end = 0, 5000
            x_plot = x[start:end]
            y_plot = y[start:end]
        else:
            x_plot = x
            y_plot = y
            start = 0

        t = np.arange(len(x_plot))
        ax.plot(t, x_plot, color="#475569", linewidth=0.6, alpha=0.85)

        # Shade anomaly regions
        in_seg = False
        seg_start = 0
        for ti in range(len(y_plot)):
            if y_plot[ti] == 1 and not in_seg:
                in_seg = True
                seg_start = ti
            elif y_plot[ti] == 0 and in_seg:
                ax.axvspan(seg_start, ti, alpha=0.25, color=C_ANOMALY, linewidth=0)
                in_seg = False
        if in_seg:
            ax.axvspan(seg_start, len(y_plot), alpha=0.25, color=C_ANOMALY, linewidth=0)

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Representative Anomaly Morphologies in TSB-AD-U",
        fontweight="bold", fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _save(fig, "fig03_anomaly_morphologies")


# ==========================================================================
# Figure 4: Anomaly segment length distribution
# ==========================================================================
def fig04_segment_length(meta: pd.DataFrame):
    print("Figure 4: Anomaly segment length distribution")
    all_segs = []
    for segs in meta["segment_lengths"]:
        all_segs.extend(segs)
    all_segs = np.array(all_segs)

    if len(all_segs) == 0:
        print("  WARNING: no anomaly segments found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram with log x-axis
    log_segs = np.log10(np.maximum(all_segs, 1))
    bins = np.linspace(0, log_segs.max() + 0.2, 50)
    ax.hist(log_segs, bins=bins, color="#64748b", alpha=0.7, edgecolor="white", linewidth=0.5)

    # Vertical lines for patch lengths
    for pl, color, ls in [(64, C_PATCH_64, "--"), (96, C_PATCH_96, "-"), (128, C_PATCH_128, ":")]:
        ax.axvline(np.log10(pl), color=color, linestyle=ls, linewidth=2, alpha=0.85,
                   label=f"L = {pl}")

    ax.set_xlabel("Anomaly Segment Length (log$_{10}$ scale)")
    ax.set_ylabel("Number of Segments")
    ax.set_title("Distribution of Anomaly Segment Lengths", fontweight="bold")

    # Custom tick labels
    max_tick = int(np.ceil(log_segs.max()))
    tick_vals = list(range(0, max_tick + 1))
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"$10^{{{v}}}$" for v in tick_vals])

    ax.legend(loc="upper right", framealpha=0.9)

    # Stats
    med_seg = np.median(all_segs)
    q25, q75 = np.percentile(all_segs, [25, 75])
    stats_text = (
        f"N segments = {len(all_segs):,}\n"
        f"Median = {med_seg:.0f}\n"
        f"IQR = [{q25:.0f}, {q75:.0f}]\n"
        f"Min = {all_segs.min()} | Max = {all_segs.max():,}"
    )
    ax.text(
        0.03, 0.95, stats_text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e2e8f0", alpha=0.9),
    )

    plt.tight_layout()
    _save(fig, "fig04_segment_length_distribution")


# ==========================================================================
# Figure 5: Robust z-score normal vs anomaly
# ==========================================================================
def fig05_robust_zscore(files: list[Path], max_points_per_group: int = 100_000):
    print("Figure 5: Robust z-score normal vs anomaly")

    z_normal = []
    z_anomaly = []

    for f in files:
        x, y = load_xy(f)
        ts = train_split_from_name(f.name) or len(x) // 2

        # Robust z-score using TRAIN portion only
        ref = x[:ts]
        med = np.median(ref)
        mad = np.median(np.abs(ref - med)) + 1e-12
        z = np.abs((x - med) / (1.4826 * mad))
        z = np.clip(z, 0, 20)  # clip extreme values

        z_normal.extend(z[y == 0].tolist())
        z_anomaly.extend(z[y == 1].tolist())

    # Subsample if too many
    rng = np.random.default_rng(42)
    if len(z_normal) > max_points_per_group:
        z_normal = rng.choice(z_normal, max_points_per_group, replace=False).tolist()
    if len(z_anomaly) > max_points_per_group:
        z_anomaly = rng.choice(z_anomaly, max_points_per_group, replace=False).tolist()

    z_normal = np.array(z_normal)
    z_anomaly = np.array(z_anomaly)

    fig, (ax_violin, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))

    # Violin plot
    parts = ax_violin.violinplot(
        [z_normal, z_anomaly],
        positions=[0, 1],
        showmedians=True,
        showextrema=False,
    )
    for i, (body, color) in enumerate(zip(parts["bodies"], [C_NORMAL, C_ANOMALY])):
        body.set_facecolor(color)
        body.set_alpha(0.4)
        body.set_edgecolor(color)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax_violin.set_xticks([0, 1])
    ax_violin.set_xticklabels(["Normal", "Anomaly"])
    ax_violin.set_ylabel("|Robust z-score|")
    ax_violin.set_title("Violin Plot", fontweight="bold")

    # Overlapping histogram / KDE
    bins = np.linspace(0, 20, 80)
    ax_hist.hist(z_normal, bins=bins, density=True, alpha=0.5, color=C_NORMAL, label="Normal", edgecolor="white", linewidth=0.3)
    ax_hist.hist(z_anomaly, bins=bins, density=True, alpha=0.5, color=C_ANOMALY, label="Anomaly", edgecolor="white", linewidth=0.3)
    ax_hist.set_xlabel("|Robust z-score|")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Density Comparison", fontweight="bold")
    ax_hist.legend()

    # Stats annotation
    med_n, med_a = np.median(z_normal), np.median(z_anomaly)
    stats_text = (
        f"Normal:  median={med_n:.2f}\n"
        f"Anomaly: median={med_a:.2f}\n"
        f"Ratio = {med_a/med_n:.2f}x"
    )
    ax_hist.text(
        0.95, 0.95, stats_text,
        transform=ax_hist.transAxes, ha="right", va="top",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e2e8f0", alpha=0.9),
    )

    fig.suptitle(
        "Robust Amplitude: Normal vs Anomalous Points\n"
        "(z-score computed using train-portion median/MAD only)",
        fontweight="bold", fontsize=12, y=1.03,
    )
    plt.tight_layout()
    _save(fig, "fig05_robust_zscore_comparison")


# ==========================================================================
# Figure 6: PCA/UMAP patch feature space
# ==========================================================================
def fig06_pca_patch_space(files: list[Path], patch_len: int = 96, n_series: int = 80,
                          max_patches_per_group: int = 20_000):
    print("Figure 6: PCA patch feature space")
    from sklearn.decomposition import PCA
    from aasp_ad.unsupervised_patch import _patch_features, _robust_scale, _starts

    rng = np.random.default_rng(42)
    selected = rng.choice(len(files), min(n_series, len(files)), replace=False)

    train_normal_feats = []
    test_normal_feats = []
    test_anom_feats = []

    for idx in selected:
        f = files[idx]
        x, y = load_xy(f)
        ts = train_split_from_name(f.name) or len(x) // 2

        if len(x) < patch_len:
            continue

        x_s = _robust_scale(x, ts, patch_len)
        all_starts = _starts(len(x), patch_len)
        if len(all_starts) == 0:
            continue

        feats = _patch_features(x_s, all_starts, patch_len)

        # Patch labels: any overlap with anomaly
        patch_labels = np.zeros(len(all_starts), dtype=int)
        for pi, s in enumerate(all_starts):
            if y[s:s + patch_len].sum() > 0:
                patch_labels[pi] = 1

        # Split: train patches vs test patches
        train_mask = (all_starts + patch_len) <= ts
        test_mask = ~train_mask

        # Train normal
        tn_mask = train_mask & (patch_labels == 0)
        if tn_mask.sum() > 0:
            train_normal_feats.append(feats[tn_mask])

        # Test normal
        test_norm_mask = test_mask & (patch_labels == 0)
        if test_norm_mask.sum() > 0:
            test_normal_feats.append(feats[test_norm_mask])

        # Test anomaly
        test_anom_mask = test_mask & (patch_labels == 1)
        if test_anom_mask.sum() > 0:
            test_anom_feats.append(feats[test_anom_mask])

    if not train_normal_feats:
        print("  WARNING: no train normal patches collected")
        return

    train_normal = np.vstack(train_normal_feats)
    test_normal = np.vstack(test_normal_feats) if test_normal_feats else np.empty((0, train_normal.shape[1]))
    test_anom = np.vstack(test_anom_feats) if test_anom_feats else np.empty((0, train_normal.shape[1]))

    # Clean NaN/Inf from overflow in kurtosis/skewness before standardisation
    FEAT_CLIP = 1e6
    for arr in (train_normal, test_normal, test_anom):
        if len(arr) > 0:
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=FEAT_CLIP, neginf=-FEAT_CLIP)
            np.clip(arr, -FEAT_CLIP, FEAT_CLIP, out=arr)

    print(f"  Patches: train_normal={len(train_normal)}, test_normal={len(test_normal)}, test_anom={len(test_anom)}")

    # Standardize using train normal only
    mu = np.nanmedian(train_normal, axis=0, keepdims=True)
    sig = np.nanstd(train_normal, axis=0, keepdims=True) + 1e-12
    train_normal_z = np.clip((train_normal - mu) / sig, -10, 10)
    test_normal_z = np.clip((test_normal - mu) / sig, -10, 10) if len(test_normal) > 0 else test_normal
    test_anom_z = np.clip((test_anom - mu) / sig, -10, 10) if len(test_anom) > 0 else test_anom

    # Subsample for PCA/plotting
    def _sub(arr, n):
        if len(arr) > n:
            return arr[rng.choice(len(arr), n, replace=False)]
        return arr

    tn_sub = _sub(train_normal_z, max_patches_per_group)
    testn_sub = _sub(test_normal_z, max_patches_per_group)
    testa_sub = _sub(test_anom_z, max_patches_per_group)

    # Fit PCA on train normal only
    pca = PCA(n_components=2, random_state=42)
    pca.fit(tn_sub)

    tn_2d = pca.transform(tn_sub)
    testn_2d = pca.transform(testn_sub) if len(testn_sub) > 0 else np.empty((0, 2))
    testa_2d = pca.transform(testa_sub) if len(testa_sub) > 0 else np.empty((0, 2))

    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot order: train normal (back), test normal (mid), test anom (front)
    ax.scatter(tn_2d[:, 0], tn_2d[:, 1], c=C_TRAIN_NORMAL, s=3, alpha=0.15, label=f"Train Normal (n={len(tn_sub):,})", rasterized=True)
    if len(testn_2d) > 0:
        ax.scatter(testn_2d[:, 0], testn_2d[:, 1], c=C_TEST_NORMAL, s=3, alpha=0.2, label=f"Test Normal (n={len(testn_sub):,})", rasterized=True)
    if len(testa_2d) > 0:
        ax.scatter(testa_2d[:, 0], testa_2d[:, 1], c=C_TEST_ANOM, s=8, alpha=0.4, label=f"Test Anomaly (n={len(testa_sub):,})", rasterized=True, marker="x")

    ax.set_xlabel(f"PC-1 ({var_explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC-2 ({var_explained[1]:.1f}% variance)")
    ax.set_title(
        f"PCA of Handcrafted 27D Patch Features (L={patch_len})\n"
        "Scaler & PCA fitted on train-normal patches only",
        fontweight="bold",
    )
    ax.legend(loc="upper right", markerscale=3, framealpha=0.9)

    plt.tight_layout()
    _save(fig, "fig06_pca_patch_feature_space")


# ==========================================================================
# Figure 7: kNN distance normal vs anomaly
# ==========================================================================
def fig07_knn_distance(files: list[Path], patch_len: int = 96, k: int = 3, n_series: int = 80):
    print("Figure 7: kNN distance normal vs anomaly")
    from sklearn.neighbors import NearestNeighbors
    from aasp_ad.unsupervised_patch import _patch_features, _robust_scale, _starts, _standardize_features

    rng = np.random.default_rng(42)
    selected = rng.choice(len(files), min(n_series, len(files)), replace=False)

    dist_normal = []
    dist_anomaly = []

    for idx in selected:
        f = files[idx]
        x, y = load_xy(f)
        ts = train_split_from_name(f.name) or len(x) // 2

        if len(x) < patch_len:
            continue

        x_s = _robust_scale(x, ts, patch_len)
        all_starts = _starts(len(x), patch_len)
        if len(all_starts) == 0:
            continue

        feats = _patch_features(x_s, all_starts, patch_len)

        train_mask = (all_starts + patch_len) <= ts
        if train_mask.sum() < k + 1:
            continue

        train_feats = feats[train_mask]
        all_norm, train_norm = _standardize_features(feats, train_feats)

        nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        nn.fit(train_norm)

        # Only score test patches
        test_mask = ~train_mask
        test_feats = all_norm[test_mask]
        test_starts = all_starts[test_mask]

        if len(test_feats) == 0:
            continue

        dists = nn.kneighbors(test_feats)[0].mean(1)

        # Patch labels
        for pi, (s, d) in enumerate(zip(test_starts, dists)):
            overlap = y[s:s + patch_len].sum()
            if overlap > 0:
                dist_anomaly.append(d)
            else:
                dist_normal.append(d)

    dist_normal = np.array(dist_normal)
    dist_anomaly = np.array(dist_anomaly)

    # Subsample
    max_pts = 100_000
    if len(dist_normal) > max_pts:
        dist_normal = rng.choice(dist_normal, max_pts, replace=False)
    if len(dist_anomaly) > max_pts:
        dist_anomaly = rng.choice(dist_anomaly, max_pts, replace=False)

    print(f"  Normal patches: {len(dist_normal):,} | Anomaly patches: {len(dist_anomaly):,}")

    fig, (ax_kde, ax_box) = plt.subplots(1, 2, figsize=(10, 5))

    # KDE / density plot
    clip_val = np.percentile(np.concatenate([dist_normal, dist_anomaly]), 99)
    dn_clip = np.clip(dist_normal, 0, clip_val)
    da_clip = np.clip(dist_anomaly, 0, clip_val)

    bins = np.linspace(0, clip_val, 80)
    ax_kde.hist(dn_clip, bins=bins, density=True, alpha=0.5, color=C_NORMAL, label="Normal", edgecolor="white", linewidth=0.3)
    ax_kde.hist(da_clip, bins=bins, density=True, alpha=0.5, color=C_ANOMALY, label="Anomaly", edgecolor="white", linewidth=0.3)
    ax_kde.set_xlabel("kNN Distance (k=3)")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title("Density Comparison", fontweight="bold")
    ax_kde.legend()

    # Boxplot
    bp = ax_box.boxplot(
        [dn_clip, da_clip],
        labels=["Normal", "Anomaly"],
        patch_artist=True,
        widths=0.5,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    bp["boxes"][0].set_facecolor(C_NORMAL)
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor(C_ANOMALY)
    bp["boxes"][1].set_alpha(0.4)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax_box.set_ylabel("kNN Distance (k=3)")
    ax_box.set_title("Boxplot Comparison", fontweight="bold")

    # Stats
    med_n, med_a = np.median(dist_normal), np.median(dist_anomaly)
    # Mann-Whitney U
    if len(dist_normal) > 0 and len(dist_anomaly) > 0:
        u_stat, p_val = stats.mannwhitneyu(dist_anomaly, dist_normal, alternative="greater")
        p_text = f"p < 1e-10" if p_val < 1e-10 else f"p = {p_val:.2e}"
    else:
        p_text = "N/A"

    stats_text = (
        f"Normal  median = {med_n:.3f}\n"
        f"Anomaly median = {med_a:.3f}\n"
        f"Ratio = {med_a/(med_n+1e-12):.2f}x\n"
        f"Mann-Whitney U: {p_text}"
    )
    ax_kde.text(
        0.95, 0.95, stats_text,
        transform=ax_kde.transAxes, ha="right", va="top",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e2e8f0", alpha=0.9),
    )

    fig.suptitle(
        "kNN Patch Distances: Normal vs Anomaly-Overlapping Test Patches\n"
        "(kNN fitted on train-normal patches only)",
        fontweight="bold", fontsize=12, y=1.03,
    )
    plt.tight_layout()
    _save(fig, "fig07_knn_distance_comparison")


# ==========================================================================
# Figure 8: Multi-scale score example
# ==========================================================================
def fig08_multiscale_example(files: list[Path]):
    print("Figure 8: Multi-scale score example")
    from aasp_ad.unsupervised_patch import score_series_knn

    # Find 2 contrasting examples: one with short anomaly, one with long
    example_short = None
    example_long = None

    for f in files:
        x, y = load_xy(f)
        ts = train_split_from_name(f.name) or len(x) // 2

        if y.sum() < 1 or len(x) < 200:
            continue

        # Compute segment lengths
        segs = []
        in_seg = False
        seg_start = 0
        for t in range(len(y)):
            if y[t] == 1 and not in_seg:
                in_seg = True
                seg_start = t
            elif y[t] == 0 and in_seg:
                segs.append(t - seg_start)
                in_seg = False
        if in_seg:
            segs.append(len(y) - seg_start)

        if not segs:
            continue

        max_seg = max(segs)
        min_seg = min(segs)

        # Short anomaly example: single short spike
        if example_short is None and min_seg <= 10 and len(x) < 10000:
            example_short = (f, x, y, ts, "Short Spike Anomaly")

        # Long anomaly example: collective/trend
        if example_long is None and max_seg > 80 and len(x) < 10000:
            example_long = (f, x, y, ts, "Collective / Trend Anomaly")

        if example_short and example_long:
            break

    examples = [e for e in [example_short, example_long] if e is not None]
    if not examples:
        print("  WARNING: could not find suitable examples")
        return

    fig, axes = plt.subplots(len(examples) * 2, 1, figsize=(12, 4 * len(examples)),
                             gridspec_kw={"height_ratios": [2, 1.5] * len(examples)})
    if len(examples) == 1:
        axes = [axes] if not hasattr(axes, '__len__') else list(axes)

    for ei, (f, x, y, ts, title) in enumerate(examples):
        ax_ts = axes[ei * 2]
        ax_score = axes[ei * 2 + 1]

        # Compute scores per scale
        scores = {}
        for pl, color, label in [(64, C_PATCH_64, "L=64"), (96, C_PATCH_96, "L=96"), (128, C_PATCH_128, "L=128")]:
            sc = score_series_knn(x, ts, patch_len=pl, k_neighbors=3)
            scores[label] = (sc, color)

        # Fused
        fused = rank01(np.stack([scores["L=64"][0], scores["L=96"][0], scores["L=128"][0]]).mean(0))

        # Subsample for plotting
        if len(x) > 5000:
            anom_idx = np.where(y == 1)[0]
            if len(anom_idx) > 0:
                center = anom_idx[len(anom_idx) // 2]
                start = max(0, center - 2000)
                end = min(len(x), center + 2000)
            else:
                start, end = 0, 4000
        else:
            start, end = 0, len(x)

        t = np.arange(end - start)
        x_p = x[start:end]
        y_p = y[start:end]

        # Time series
        ax_ts.plot(t, x_p, color="#475569", linewidth=0.7, alpha=0.85)
        # Shade anomaly
        in_seg = False
        for ti in range(len(y_p)):
            if y_p[ti] == 1 and not in_seg:
                in_seg = True
                seg_start_t = ti
            elif y_p[ti] == 0 and in_seg:
                ax_ts.axvspan(seg_start_t, ti, alpha=0.2, color=C_ANOMALY, linewidth=0)
                in_seg = False
        if in_seg:
            ax_ts.axvspan(seg_start_t, len(y_p), alpha=0.2, color=C_ANOMALY, linewidth=0)

        ax_ts.set_title(f"{title}: {f.name[:50]}", fontweight="bold", fontsize=10)
        ax_ts.set_ylabel("Value")
        ax_ts.set_xlim(0, len(t))

        # Scores
        for label, (sc, color) in scores.items():
            ax_score.plot(t, sc[start:end], color=color, linewidth=0.8, alpha=0.7, label=label)
        ax_score.plot(t, fused[start:end], color=C_FUSED, linewidth=1.8, alpha=0.9, label="Fused")

        # Shade anomaly on score plot too
        in_seg = False
        for ti in range(len(y_p)):
            if y_p[ti] == 1 and not in_seg:
                in_seg = True
                seg_start_t = ti
            elif y_p[ti] == 0 and in_seg:
                ax_score.axvspan(seg_start_t, ti, alpha=0.1, color=C_ANOMALY, linewidth=0)
                in_seg = False
        if in_seg:
            ax_score.axvspan(seg_start_t, len(y_p), alpha=0.1, color=C_ANOMALY, linewidth=0)

        ax_score.set_ylabel("Anomaly Score")
        ax_score.set_xlabel("Time")
        ax_score.set_xlim(0, len(t))
        ax_score.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.9)

    fig.suptitle(
        "Multi-Scale kNN Anomaly Scores: L={64, 96, 128} + Fusion",
        fontweight="bold", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _save(fig, "fig08_multiscale_score_example")


# ==========================================================================
# Main
# ==========================================================================
def main():
    t0 = time.time()
    ensure_output_dirs()

    print("=" * 60)
    print(" AASP-UML: EDA on TSB-AD-U Eva-Full")
    print("=" * 60)

    files = eva_full_files()
    print(f"Eva-Full files: {len(files)}")

    # Step 0: Build metadata
    print("\n--- Building metadata table ---")
    meta = build_metadata(files)
    meta_save = meta.drop(columns=["segment_lengths"])
    meta_save.to_csv(EDA_TABLES_DIR / "eda_metadata.csv", index=False)
    print(f"  -> saved eda_metadata.csv ({len(meta)} rows)")

    # Figure 1
    print()
    fig01_series_length(meta)

    # Figure 2
    print()
    fig02_anomaly_ratio(meta)

    # Figure 3
    print()
    fig03_anomaly_morphologies(files)

    # Figure 4
    print()
    fig04_segment_length(meta)

    # Figure 5
    print()
    fig05_robust_zscore(files)

    # Figure 6
    print()
    fig06_pca_patch_space(files)

    # Figure 7
    print()
    fig07_knn_distance(files)

    # Figure 8
    print()
    fig08_multiscale_example(files)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All 8 figures generated in {elapsed:.1f}s")
    print(f"Output: {EDA_FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
