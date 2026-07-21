"""Figure 1: anomaly archetypes in TSB-AD-U.

Selects 4 representative series with DATA-DRIVEN RULES (from eda_series_profile.csv),
plots the raw signal + an overlay of the anomaly region (zoomed around the first
anomaly to make it visible). Read-only with respect to the raw data.

Run:    PYTHONPATH=src python -m aasp_ad.figures
Output: outputs/eda/figures/fig1_anomaly_archetypes.png
        outputs/eda/notes/fig1_exemplars.csv   (records the chosen series -> reproducible)
"""

import matplotlib

matplotlib.use("Agg")  # no GUI needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aasp_ad.config import (
    EDA_FIGURES_DIR,
    EDA_NOTES_DIR,
    EDA_TABLES_DIR,
    TSB_AD_U_DIR,
    ensure_output_dirs,
)

PLOT_MAX_LEN = 50000  # length cap when SELECTING an exemplar so the plot stays crisp


def select_exemplars(df: pd.DataFrame) -> dict:
    """4 archetypes, each chosen by one clear rule on existing columns."""
    plottable = df[df["length"] <= PLOT_MAX_LEN]
    point = df[df["anomaly_type"] == "point"]
    single = plottable[(plottable["anomaly_type"] == "sequence") & (plottable["n_segments"] == 1)]
    seq = plottable[plottable["anomaly_type"] == "sequence"]
    ctx = plottable[plottable["anomaly_type"].isin(["sequence", "mixed"])]
    return {
        "Point / spike (highest kurtosis)": point.loc[point["kurtosis"].idxmax()],
        "Single long subsequence (1 long segment)": single.loc[single["max_seg_len"].idxmax()],
        "Recurring sequences (most segments)": seq.loc[seq["n_segments"].idxmax()],
        "Low-amplitude / contextual (lowest outlier_ratio)": ctx.loc[ctx["outlier_ratio"].idxmin()],
    }


def load_xy(name: str):
    df = pd.read_csv(TSB_AD_U_DIR / name)
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y_bool = pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(dtype=float) == 1
    return x, y_bool


def _anomaly_spans(y_bool: np.ndarray):
    """Return (start, end_exclusive) pairs of anomaly segments."""
    d = np.diff(np.concatenate(([0], y_bool.astype(int), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=False))


def _crop_window(
    y_bool: np.ndarray,
    length: int,
    n_show: int = 6,
    pad: int = 200,
    max_width: int = 6000,
    min_width: int = 600,
):
    """Zoom window: cover up to n_show leading anomaly segments (to show
    recurrence), with context padding and a width cap for readability."""
    spans = _anomaly_spans(y_bool)
    first_s = spans[0][0]
    last_e = spans[min(len(spans), n_show) - 1][1]
    w0 = max(0, first_s - pad)
    w1 = min(length, max(last_e + pad, w0 + min_width))
    if w1 - w0 > max_width:
        w1 = min(length, w0 + max_width)
    return w0, w1


def plot_overlay(ax, x, y_bool, title: str):
    w0, w1 = _crop_window(y_bool, len(x))
    t = np.arange(w0, w1)
    ax.plot(t, x[w0:w1], lw=0.8, color="#1f77b4")
    for s, e in _anomaly_spans(y_bool[w0:w1]):
        ax.axvspan(w0 + s, w0 + e, color="red", alpha=0.25)
    ax.set_title(title, fontsize=10)
    ax.margins(x=0)


def main() -> None:
    profile = pd.read_csv(EDA_TABLES_DIR / "eda_series_profile.csv")
    exemplars = select_exemplars(profile)
    ensure_output_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    rows = []
    for ax, (label, row) in zip(axes.flat, exemplars.items(), strict=False):
        x, y_bool = load_xy(row["file"])
        sub = (
            f"{row['dataset']} | len={row['length']:d} | kurt={row['kurtosis']:.1f} | "
            f"n_seg={row['n_segments']:d} | seg_len_max={row['max_seg_len']:d} | "
            f"contam={row['contamination']:.3f}"
        )
        plot_overlay(ax, x, y_bool, f"{label}\n{sub}")
        rows.append({"archetype": label, **row.to_dict()})

    fig.suptitle("Figure 1: Anomaly archetypes in TSB-AD-U (red = labeled anomaly)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    fig_path = EDA_FIGURES_DIR / "fig1_anomaly_archetypes.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    notes_path = EDA_NOTES_DIR / "fig1_exemplars.csv"
    pd.DataFrame(rows).to_csv(notes_path, index=False)

    print("Selected exemplars:")
    for label, row in exemplars.items():
        print("  -", row["file"], "->", label.split(" (")[0])
    print("Saved figure ->", fig_path)
    print("Saved exemplars ->", notes_path)


if __name__ == "__main__":
    main()
