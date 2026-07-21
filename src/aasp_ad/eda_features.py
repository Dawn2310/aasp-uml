"""Time-domain + Frequency-domain meta-features for TSB-AD-U (stage 7 + 8).

Builds the meta-feature set (vector z) per series and merges it with
eda_series_profile to form a complete meta-feature table that serves both
(a) characterization and (b) the adaptive gating of AASP-AD. Also draws
Figure 2 (the meta-feature space).

Read-only, numpy-only. Does NOT run STL across all series (per the project
guideline); seasonality is captured by acf + dominant_period + spectral_entropy.

Run:    PYTHONPATH=src python -m aasp_ad.eda_features
Output: outputs/eda/tables/eda_meta_features.csv
        outputs/eda/figures/fig2_meta_feature_space.png
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aasp_ad.config import EDA_FIGURES_DIR, EDA_TABLES_DIR, TSB_AD_U_DIR, ensure_output_dirs

EPS = 1e-12
ROLL_WIN = 50  # window for rolling std (variance-shift indicator)
LOW_BAND_FRAC = 0.1  # fraction of low-frequency band used for low_freq_ratio


def load_x(name: str) -> np.ndarray:
    df = pd.read_csv(TSB_AD_U_DIR / name)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)


def time_features(x: np.ndarray) -> dict:
    n = len(x)
    t = np.arange(n)
    acf1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if np.std(x[:-1]) > 0 else 0.0
    trend = float(abs(np.corrcoef(x, t)[0, 1])) if np.std(x) > 0 else 0.0
    if n >= 2 * ROLL_WIN:
        rs = pd.Series(x).rolling(ROLL_WIN).std().to_numpy()
        rs = rs[~np.isnan(rs)]
        roll_cv = float(rs.std() / (rs.mean() + EPS)) if len(rs) else 0.0
    else:
        roll_cv = 0.0
    diff_ratio = float(np.std(np.diff(x)) / (np.std(x) + EPS))
    return {
        "acf_lag1": acf1,
        "trend_strength": trend,
        "rolling_std_cv": roll_cv,
        "diff_std_ratio": diff_ratio,
    }


def freq_features(x: np.ndarray) -> dict:
    xc = x - np.mean(x)
    spec = np.abs(np.fft.rfft(xc)) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0)
    psd = spec / (spec.sum() + EPS)
    spectral_entropy = float(-np.sum(psd * np.log(psd + EPS)) / np.log(len(psd) + EPS))
    if len(psd) > 1:
        k = int(np.argmax(psd[1:])) + 1
        dom_freq = float(freq[k])
        dom_period = float(1.0 / dom_freq) if dom_freq > 0 else 0.0
    else:
        dom_freq = dom_period = 0.0
    p = psd[1:]
    klow = max(1, int(len(p) * LOW_BAND_FRAC))
    low_ratio = float(p[:klow].sum() / (p.sum() + EPS))
    return {
        "spectral_entropy": spectral_entropy,
        "dominant_freq": dom_freq,
        "dominant_period": dom_period,
        "low_freq_ratio": low_ratio,
    }


def extract(name: str) -> dict:
    x = load_x(name)
    return {"file": name, **time_features(x), **freq_features(x)}


def plot_meta_space(df: pd.DataFrame, path) -> None:
    colors = {"point": "#d62728", "sequence": "#1f77b4", "mixed": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for t, c in colors.items():
        sub = df[df["anomaly_type"] == t]
        ax.scatter(
            sub["spectral_entropy"],
            sub["outlier_ratio"],
            s=12,
            alpha=0.5,
            c=c,
            label=f"{t} (n={len(sub)})",
        )
    ax.set_xlabel("spectral_entropy  (low = periodic/shape  ->  high = noisy)")
    ax.set_ylabel("outlier_ratio  (high = strong amplitude)")
    ax.set_title(
        "Figure 2: Meta-feature space of TSB-AD-U\n"
        "(amplitude vs shape, by anomaly type) -> motivates adaptive fusion"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    profile = pd.read_csv(EDA_TABLES_DIR / "eda_series_profile.csv")
    files = profile["file"].tolist()
    print(f"Extracting time/freq features for {len(files)} series ...")
    feats = pd.DataFrame(extract(f) for f in files)

    meta = profile.merge(feats, on="file", validate="one_to_one")
    ensure_output_dirs()
    out_path = EDA_TABLES_DIR / "eda_meta_features.csv"
    meta.to_csv(out_path, index=False)
    print(f"Wrote {len(meta)} rows x {meta.shape[1]} cols -> {out_path}")

    fig_path = EDA_FIGURES_DIR / "fig2_meta_feature_space.png"
    plot_meta_space(meta, fig_path)
    print(f"Saved figure -> {fig_path}")

    # ---- sanity + spot-check (ASCII) ----
    new_cols = [
        "acf_lag1",
        "trend_strength",
        "rolling_std_cv",
        "diff_std_ratio",
        "spectral_entropy",
        "dominant_period",
        "low_freq_ratio",
    ]
    print("\n== Range of new features ==")
    print(meta[new_cols].describe().loc[["min", "50%", "max"]].round(3).to_string())
    print("\n== Spot-check (periodic sine vs point) ==")
    for f in [
        "293_TODS_id_7_Synthetic_tr_500_1st_7.csv",
        "657_YAHOO_id_107_WebService_tr_500_1st_1260.csv",
    ]:
        r = meta[meta["file"] == f].iloc[0]
        print(
            f"  {f[:45]:<45} acf1={r['acf_lag1']:.3f} "
            f"entropy={r['spectral_entropy']:.3f} dom_period={r['dominant_period']:.1f}"
        )


if __name__ == "__main__":
    main()
