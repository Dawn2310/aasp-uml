"""AASP-UML: label-free ML patch scoring in a one-class setting.

Uses NO anomaly labels and NO neural networks. Rich hand-crafted patch features
are scored by Isolation Forest or kNN fitted exclusively on the TRAIN (normal)
portion of each series.

Design:
  1. Robust-scale the series by train-portion median/MAD.
  2. Extract patches at unit stride (same as PaAno).
  3. Compute 27 per-patch features capturing amplitude, shape, temporal and
     spectral properties.
  4. Fit an Isolation Forest (or kNN) on TRAIN patches only.
  5. Score ALL patches; back-project to time points by averaging.
  6. Multi-scale fusion over patch_lens = (64, 96, 128).

For a direct PaAno comparison, run AASP-UML and PaAno on the same local file
list and the same TSB-AD VUS protocol. The public TSB-AD-U-Eva list currently
contains 350 files; PaAno's paper also prints a conflicting 530-file count, so
reports should keep the exact split explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
FEATURE_CLIP = 50.0


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _train_end_or_raise(train_split: int | None, n: int, patch_len: int) -> int:
    """Return a valid train cutoff for patch fitting/scaling, or fail closed."""
    if train_split is None:
        raise ValueError("Missing train_split; refusing to use full-series statistics.")
    train_end = int(train_split)
    if not (2 * patch_len < train_end < n):
        raise ValueError(
            f"Invalid train_split={train_end} for n={n}, patch_len={patch_len}; "
            "refusing to use full-series statistics."
        )
    return train_end


def _robust_scale(x: np.ndarray, train_split: int | None, patch_len: int) -> np.ndarray:
    """Scale by train-portion median/MAD; preserves amplitude relative to normal."""
    n = len(x)
    end = _train_end_or_raise(train_split, n, patch_len)
    ref = np.asarray(x[:end], dtype=np.float64)
    med = float(np.nanmedian(ref))
    mad = float(np.nanmedian(np.abs(ref - med))) + EPS
    return ((np.asarray(x, dtype=np.float64) - med) / mad).astype(np.float32)


def _starts(n: int, patch_len: int) -> np.ndarray:
    if n < patch_len:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n - patch_len + 1, dtype=np.int64)


# ---------------------------------------------------------------------------
# Feature extraction - 27 features per patch
# ---------------------------------------------------------------------------


def _patch_features(x: np.ndarray, starts: np.ndarray, patch_len: int) -> np.ndarray:
    """Extract 27 rich features from robust-scaled patches.

    Covers amplitude, shape, temporal, and spectral properties so the ML model
    can detect point anomalies (YAHOO), shape anomalies (UCR), and context
    shifts (SMAP, MSL) in a single feature space.

    Output: (M, 27) float32 array.
    """
    wins = np.lib.stride_tricks.sliding_window_view(x, patch_len)[starts]  # (M, L)
    M, L = wins.shape

    # --- Amplitude statistics (11) ---
    mean_ = wins.mean(1)
    std_ = wins.std(1) + EPS
    min_ = wins.min(1)
    max_ = wins.max(1)
    range_ = max_ - min_
    q25 = np.percentile(wins, 25, axis=1)
    q75 = np.percentile(wins, 75, axis=1)
    iqr = q75 - q25 + EPS
    med_ = np.median(wins, axis=1)
    mad_ = np.median(np.abs(wins - med_[:, None]), axis=1) + EPS
    zn = (wins - med_[:, None]) / mad_[:, None]   # MAD-normalized
    zn = np.clip(zn, -100.0, 100.0)
    skew_ = (zn ** 3).mean(1)
    kurt_ = (zn ** 4).mean(1) - 3.0

    # --- Instance-normalized shape stats (2) ---
    ins = (wins - mean_[:, None]) / std_[:, None]  # instance-normalized
    ins = np.clip(ins, -100.0, 100.0)
    in_skew = (ins ** 3).mean(1)
    in_kurt = (ins ** 4).mean(1) - 3.0

    # --- Temporal / difference stats (4) ---
    diff = np.diff(wins, axis=1)                     # (M, L-1)
    net_slope = wins[:, -1] - wins[:, 0]
    mean_diff = diff.mean(1)
    std_diff = diff.std(1) + EPS
    max_abs_diff = np.abs(diff).max(1)

    # --- Crossing rates (2) ---
    zcr = (np.abs(np.diff(np.sign(wins), axis=1)) > 0).mean(1)
    centered = wins - mean_[:, None]
    mcr = (np.abs(np.diff(np.sign(centered), axis=1)) > 0).mean(1)

    # --- Energy (1) ---
    rms = np.sqrt((wins ** 2).mean(1))

    # --- Autocorrelation (3) ---
    def _acf(lag: int) -> np.ndarray:
        if lag >= L:
            return np.zeros(M, dtype=np.float32)
        m = wins.mean(1, keepdims=True)
        v = wins - m
        denom = (v ** 2).sum(1) + EPS
        return (v[:, :-lag] * v[:, lag:]).sum(1) / denom

    acf1 = _acf(1)
    acf2 = _acf(2)
    acf5 = _acf(min(5, L - 1))

    # --- Spectral (2) ---
    psd = np.abs(np.fft.rfft(wins - mean_[:, None], axis=1)) ** 2   # (M, K)
    psd_sum = psd.sum(1, keepdims=True) + EPS
    pn = psd / psd_sum
    spec_entropy = -(pn * np.log(pn + EPS)).sum(1) / np.log(psd.shape[1] + EPS)
    dom_ratio = psd.max(1) / psd_sum.squeeze(1)

    # --- Linear trend (1) ---
    t = np.linspace(-1.0, 1.0, L)
    t -= t.mean()
    denom_t = (t ** 2).sum() + EPS
    trend = ((wins - mean_[:, None]) * t[None, :]).sum(1) / denom_t

    # --- CUSUM max-deviation (1) ---
    # Detects sustained mean shifts: max |cumsum of mean-centered patch|
    cusum = np.abs(np.cumsum(centered, axis=1)).max(1) / (L * std_)

    feats = np.column_stack([
        mean_, std_, min_, max_, range_,       # 5
        q25, q75, iqr, mad_,                    # 4
        skew_, kurt_,                            # 2
        in_skew, in_kurt,                        # 2
        net_slope, mean_diff, std_diff, max_abs_diff,  # 4
        zcr, mcr,                                # 2
        rms,                                     # 1
        acf1, acf2, acf5,                        # 3
        spec_entropy, dom_ratio,                 # 2
        trend,                                   # 1
        cusum,                                   # 1
    ])                                           # total: 27
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# Column order must match the np.column_stack(...) call in _patch_features exactly.
FEATURE_NAMES = [
    "mean", "std", "min", "max", "range",
    "q25", "q75", "iqr", "mad",
    "skew", "kurt",
    "in_skew", "in_kurt",
    "net_slope", "mean_diff", "std_diff", "max_abs_diff",
    "zcr", "mcr",
    "rms",
    "acf1", "acf2", "acf5",
    "spec_entropy", "dom_ratio",
    "trend",
    "cusum",
]

# Groups for the --drop-group ablation (a1): which of the 27 engineered
# features encode amplitude vs. shape vs. temporal-ordering vs. spectral vs.
# trend information. rms and cusum are grouped under amplitude/temporal
# respectively since they are magnitude-of-energy and change-over-time
# measures, not distinct axes of their own.
FEATURE_GROUPS = {
    "amplitude": ["mean", "std", "min", "max", "range", "q25", "q75", "iqr", "mad", "skew", "kurt", "rms"],
    "shape": ["in_skew", "in_kurt"],
    "temporal": ["net_slope", "mean_diff", "std_diff", "max_abs_diff", "zcr", "mcr", "acf1", "acf2", "acf5", "cusum"],
    "spectral": ["spec_entropy", "dom_ratio"],
    "trend": ["trend"],
}


def _drop_group_mask(drop_group: str | None) -> np.ndarray:
    """Boolean keep-mask over the 27 patch-feature columns for the --drop-group ablation."""
    if drop_group is None:
        return np.ones(len(FEATURE_NAMES), dtype=bool)
    if drop_group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group: {drop_group!r}; choices={sorted(FEATURE_GROUPS)}")
    drop_names = set(FEATURE_GROUPS[drop_group])
    return np.array([name not in drop_names for name in FEATURE_NAMES], dtype=bool)


def _raw_patch_vector(x: np.ndarray, starts: np.ndarray, patch_len: int) -> np.ndarray:
    """Raw robust-scaled patch window as the feature vector (classic subsequence-kNN,
    amplitude-sensitive since only the series-level robust scaling was applied).

    Used only for the a0 ablation to isolate the effect of the 27D engineered
    representation from the rest of the pipeline (train-mask, back-projection,
    multi-scale fusion, calibration all stay identical).
    """
    wins = np.lib.stride_tricks.sliding_window_view(x, patch_len)[starts]  # (M, L)
    return np.nan_to_num(wins, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _raw_patch_vector_znorm(x: np.ndarray, starts: np.ndarray, patch_len: int) -> np.ndarray:
    """Per-patch instance-normalized raw window (classic UCR-style subsequence-kNN):
    each patch is centered/scaled by its OWN mean/std, discarding local amplitude and
    offset so only shape is compared -- the shape-only counterpart to _raw_patch_vector's
    amplitude-sensitive raw window. Run both for the a0 ablation (see run_aasp_eval.py).
    """
    wins = np.lib.stride_tricks.sliding_window_view(x, patch_len)[starts]  # (M, L)
    mu = wins.mean(1, keepdims=True)
    sig = wins.std(1, keepdims=True) + EPS
    znorm = (wins - mu) / sig
    return np.nan_to_num(znorm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _standardize_features(all_feats: np.ndarray, train_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score by train features and cap extremes to keep distance models finite."""
    mu = train_feats.mean(0, keepdims=True)
    sig = train_feats.std(0, keepdims=True) + EPS
    train_norm = (train_feats - mu) / sig
    all_norm = (all_feats - mu) / sig
    train_norm = np.nan_to_num(train_norm, nan=0.0, posinf=FEATURE_CLIP, neginf=-FEATURE_CLIP)
    all_norm = np.nan_to_num(all_norm, nan=0.0, posinf=FEATURE_CLIP, neginf=-FEATURE_CLIP)
    train_norm = np.clip(train_norm, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)
    all_norm = np.clip(all_norm, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)
    return all_norm, train_norm


# ---------------------------------------------------------------------------
# Back-projection
# ---------------------------------------------------------------------------


def _average_backproject(
    starts: np.ndarray, patch_scores: np.ndarray, n: int, patch_len: int
) -> np.ndarray:
    """Average patch scores back to per-point scores via cumsum trick (O(n))."""
    sum_d = np.zeros(n + 1, dtype=np.float64)
    cnt_d = np.zeros(n + 1, dtype=np.float64)
    ends = np.minimum(starts + patch_len, n)
    np.add.at(sum_d, starts, patch_scores)
    np.add.at(sum_d, ends, -patch_scores)
    np.add.at(cnt_d, starts, 1.0)
    np.add.at(cnt_d, ends, -1.0)
    sums = np.cumsum(sum_d[:-1])
    counts = np.cumsum(cnt_d[:-1])
    score = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    if np.any(counts <= 0):
        s = pd.Series(score)
        score = s.where(counts > 0).ffill().bfill().fillna(0.0).to_numpy()
    return score


# ---------------------------------------------------------------------------
# Per-series scoring - single patch length
# ---------------------------------------------------------------------------


def _scale_by_train(score: np.ndarray, train_split: int | None, patch_len: int) -> np.ndarray:
    """Scale anomaly scores strictly using train-portion median/MAD to prevent test leakage."""
    n = len(score)
    train_end = _train_end_or_raise(train_split, n, patch_len)
    # Boundary train points can be covered by patches that extend into test.
    # Use only the prefix whose covering patches are guaranteed train-only.
    safe_end = train_end - patch_len + 1
    train_scores = score[:safe_end]
    if len(train_scores) < 4:
        raise ValueError(f"Too few train scores ({len(train_scores)}) to scale safely.")
    med = np.median(train_scores)
    mad = np.median(np.abs(train_scores - med)) + EPS
    return np.clip((score - med) / mad, 0.0, None)


def score_series_iforest(
    x: np.ndarray,
    train_split: int | None,
    patch_len: int = 96,
    n_estimators: int = 200,
    random_state: int = 0,
) -> np.ndarray:
    """Isolation Forest score on rich patch features, train-portion fitted."""
    from sklearn.ensemble import IsolationForest

    n = len(x)
    if n < patch_len:
        return np.zeros(n, dtype=np.float32)

    x_s = _robust_scale(x, train_split, patch_len)
    all_starts = _starts(n, patch_len)
    if len(all_starts) == 0:
        return np.zeros(n, dtype=np.float32)

    train_end = _train_end_or_raise(train_split, n, patch_len)
    train_mask = all_starts + patch_len <= train_end

    all_feats = _patch_features(x_s, all_starts, patch_len)
    if train_mask.sum() < 4:
        raise ValueError(f"Too few training patches ({train_mask.sum()}) to fit Isolation Forest.")
    train_feats = all_feats[train_mask]

    # Normalise using train statistics so test patches are in the same space.
    all_norm, train_norm = _standardize_features(all_feats, train_feats)

    iforest = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iforest.fit(train_norm)
    # score_samples returns negative anomaly score; negate so higher = more anomalous.
    raw = -iforest.score_samples(all_norm).astype(np.float64)

    return _scale_by_train(_average_backproject(all_starts, raw, n, patch_len), train_split, patch_len)


def score_series_knn(
    x: np.ndarray,
    train_split: int | None,
    patch_len: int = 96,
    k_neighbors: int = 3,
    representation: str = "features",
    drop_group: str | None = None,
) -> np.ndarray:
    """kNN distance score on rich patch features (PaAno-style protocol).

    representation="raw"/"raw_znorm" replaces the 27D engineered features with
    the raw (resp. per-patch instance-normalized) robust-scaled patch window
    (a0 ablation: classic subsequence-kNN, same train-mask/back-projection/
    calibration). drop_group zeroes out one engineered feature group (a1
    ablation); only applies to representation="features".
    """
    from sklearn.neighbors import NearestNeighbors

    if representation not in {"features", "raw", "raw_znorm"}:
        raise ValueError(f"unknown representation: {representation!r}")
    if representation != "features" and drop_group is not None:
        raise ValueError("drop_group ablation only applies to representation='features'")

    n = len(x)
    if n < patch_len:
        return np.zeros(n, dtype=np.float32)

    x_s = _robust_scale(x, train_split, patch_len)
    all_starts = _starts(n, patch_len)
    if len(all_starts) == 0:
        return np.zeros(n, dtype=np.float32)

    train_end = _train_end_or_raise(train_split, n, patch_len)
    train_mask = all_starts + patch_len <= train_end

    if representation == "raw":
        all_feats = _raw_patch_vector(x_s, all_starts, patch_len)
    elif representation == "raw_znorm":
        all_feats = _raw_patch_vector_znorm(x_s, all_starts, patch_len)
    else:
        all_feats = _patch_features(x_s, all_starts, patch_len)
        if drop_group is not None:
            all_feats = all_feats[:, _drop_group_mask(drop_group)]
    if train_mask.sum() < 1:
        raise ValueError("No training patches available to fit kNN.")
    k_eff = max(1, min(k_neighbors, int(train_mask.sum())))
    train_feats = all_feats[train_mask]

    all_norm, train_norm = _standardize_features(all_feats, train_feats)

    nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1)
    nn.fit(train_norm)
    dist = nn.kneighbors(all_norm)[0].mean(1).astype(np.float64)

    return _scale_by_train(_average_backproject(all_starts, dist, n, patch_len), train_split, patch_len)


# ---------------------------------------------------------------------------
# Multi-scale fusion
# ---------------------------------------------------------------------------


def score_series_multiscale(
    x: np.ndarray,
    train_split: int | None,
    patch_lens: tuple[int, ...] = (64, 96, 128),
    method: str = "iforest",
    n_estimators: int = 200,
    k_neighbors: int = 3,
    representation: str = "features",
    drop_group: str | None = None,
) -> np.ndarray:
    """Multi-scale score: train independently per scale, then train-scale fusion.

    representation/drop_group are ablation-only knobs (see score_series_knn) and
    apply to method="knn"; they are rejected for method="iforest" since that
    path was not part of the requested ablation.
    """
    if method != "knn" and (representation != "features" or drop_group is not None):
        raise ValueError("representation/drop_group ablation is only supported for method='knn'")

    scores = []
    for i, pl in enumerate(patch_lens):
        if method == "knn":
            sc = score_series_knn(
                x, train_split, patch_len=pl, k_neighbors=k_neighbors,
                representation=representation, drop_group=drop_group,
            )
        else:
            sc = score_series_iforest(
                x, train_split, patch_len=pl, n_estimators=n_estimators, random_state=i
            )
        scores.append(sc)

    fused = np.stack(scores, axis=0).mean(axis=0)
    return _scale_by_train(fused, train_split, max(patch_lens))
