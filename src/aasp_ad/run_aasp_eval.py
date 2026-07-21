"""Strict AASP-UML evaluation on TSB-AD-U splits.

This runner is the paper-facing AASP-UML entry point. It uses only normal
training prefixes from each series, official TSB-AD VUS evaluation, and writes
an explicit protocol marker into every result CSV.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from TSB_AD.evaluation.metrics import generate_curve
from TSB_AD.utils.slidingWindows import find_length_rank

from aasp_ad.config import RESULTS_DIR, TSB_AD_U_DIR, eva_full_files, eva_repo_files, tuning_files
from aasp_ad.unsupervised_patch import (
    score_series_iforest,
    score_series_knn,
    score_series_multiscale,
)

EVAL_PROTOCOL = "TSB_AD_find_length_rank_opt_250"


def _load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).dropna()
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(dtype=np.int32)
    return x, y


def _train_split_from_name(name: str) -> int | None:
    match = re.search(r"_tr_(\d+)_", name)
    return int(match.group(1)) if match else None


def _split_files(split: str) -> tuple[list[Path], str, str]:
    if split == "eva350":
        return eva_repo_files(), "Eva350", "_eva350"
    if split == "eva_full822":
        return eva_full_files(), "Eva-Full822", "_eva"
    if split == "all870":
        return sorted(TSB_AD_U_DIR.glob("*.csv")), "All870", ""
    if split == "tuning":
        return tuning_files(), "Tuning48", "_tuning48"
    raise ValueError(f"unknown split: {split}")


def _vus_pr(score: np.ndarray, y: np.ndarray, sliding_window: int) -> float:
    _, _, _, _, _, _, _, vus_pr = generate_curve(
        y.astype(np.int32), np.asarray(score, dtype=float), sliding_window, "opt", 250
    )
    return float(vus_pr)


def _resume_metadata_matches(df: pd.DataFrame, expected: dict) -> pd.Series:
    """True for rows whose run-identifying columns match this invocation, so a
    resumed run only reuses rows produced under the same method/split/protocol/
    ablation settings (mirrors run_supervised_headroom.py's resume guard).

    drop_group is written as "" for no-drop rows, but pandas round-trips an
    empty CSV field as NaN, not "" -- fillna("") before comparing or every
    ablation-off row would spuriously fail to match and never resume.
    """
    mask = pd.Series(True, index=df.index)
    for col, value in expected.items():
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= df[col].fillna("").astype(str).eq(str(value))
    return mask


def run_aasp_eval(
    method: str,
    split: str,
    patch_lens: tuple[int, ...] = (64, 96, 128),
    n_estimators: int = 200,
    k_neighbors: int = 3,
    limit: int | None = None,
    representation: str = "features",
    drop_group: str | None = None,
) -> Path:
    if method not in {"iforest", "knn"}:
        raise ValueError(f"unknown method: {method}")
    is_ablation = representation != "features" or drop_group is not None
    if is_ablation and method != "knn":
        raise ValueError("representation/drop_group ablation is only supported for --method knn")

    files, split_label, suffix = _split_files(split)
    if limit:
        files = files[:limit]
        suffix = f"{suffix}_n{limit}" if suffix else f"_n{limit}"
    if representation != "features":
        suffix += f"_{representation}"
    if drop_group:
        suffix += f"_drop{drop_group}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"unsupervised_ml_{method}{suffix}.csv"
    latest_path = RESULTS_DIR / "unsupervised_ml.csv"

    print(
        f"=== AASP-UML | split={split_label} | n={len(files)} | "
        f"method={method} | patch_lens={patch_lens} | "
        f"representation={representation} | drop_group={drop_group} ===",
        flush=True,
    )

    expected_resume = {
        "method": method,
        "split": split,
        "patch_lens": ",".join(map(str, patch_lens)),
        "eval_protocol": EVAL_PROTOCOL,
        "representation": representation,
        "drop_group": drop_group or "",
    }

    rows: list[dict] = []
    multi_scores: list[float] = []
    single_scores: list[float] = []
    done: set[str] = set()

    if out_path.exists():
        try:
            old = pd.read_csv(out_path)
            matching = _resume_metadata_matches(old, expected_resume)
            if "status" not in old.columns:
                matching &= False
            resumable = old.loc[matching].copy()
            rows = resumable.to_dict("records")
            done = set(resumable.loc[resumable["status"] == "ok", "file"].astype(str))
            multi_scores = [
                float(r["vus_pr_multi"]) for r in rows
                if r.get("status") == "ok" and pd.notna(r.get("vus_pr_multi"))
            ]
            single_scores = [
                float(r["vus_pr_single96"]) for r in rows
                if r.get("status") == "ok" and pd.notna(r.get("vus_pr_single96"))
            ]
            print(f"Resuming from {out_path}: {len(done)} already finished.", flush=True)
        except Exception as e:
            print(f"Failed to resume from {out_path}: {e}. Starting fresh.", flush=True)
            rows = []
            multi_scores = []
            single_scores = []
            done = set()

    for i, path in enumerate(files, 1):
        if path.name in done:
            continue
        row = {
            "file": path.name,
            "method": method,
            "split": split,
            "patch_lens": ",".join(map(str, patch_lens)),
            "eval_protocol": EVAL_PROTOCOL,
            "representation": representation,
            "drop_group": drop_group or "",
            "status": "error",
            "error": "",
        }
        t0 = time.time()
        try:
            x, y = _load_xy(path)
            train_split = _train_split_from_name(path.name)
            if train_split is None:
                raise ValueError("missing train split in filename")

            sliding_window = int(find_length_rank(x.reshape(-1, 1), rank=1))

            if method == "knn":
                single_score = score_series_knn(
                    x, train_split, patch_len=96, k_neighbors=k_neighbors,
                    representation=representation, drop_group=drop_group,
                )
            else:
                single_score = score_series_iforest(
                    x, train_split, patch_len=96, n_estimators=n_estimators
                )

            multi_score = score_series_multiscale(
                x,
                train_split,
                patch_lens=patch_lens,
                method=method,
                n_estimators=n_estimators,
                k_neighbors=k_neighbors,
                representation=representation,
                drop_group=drop_group,
            )

            if not np.isfinite(single_score).all() or not np.isfinite(multi_score).all():
                raise ValueError("non-finite anomaly score")

            vus_single = _vus_pr(single_score, y, sliding_window)
            vus_multi = _vus_pr(multi_score, y, sliding_window)
            single_scores.append(vus_single)
            multi_scores.append(vus_multi)

            row.update(
                {
                    "status": "ok",
                    "length": len(y),
                    "train_split": int(train_split),
                    "sliding_window": sliding_window,
                    "vus_pr_multi": vus_multi,
                    "vus_pr_single96": vus_single,
                    "runtime_sec": time.time() - t0,
                }
            )
        except Exception as exc:
            print(f"  [{i}/{len(files)}] ERROR {path.name}: {exc}", flush=True)
            row.update({"error": repr(exc), "runtime_sec": time.time() - t0})

        rows.append(row)
        if i % 5 == 0 or i == len(files):
            mean_multi = float(np.mean(multi_scores)) if multi_scores else float("nan")
            mean_single = float(np.mean(single_scores)) if single_scores else float("nan")
            print(
                f"  [{i}/{len(files)}] multi={mean_multi:.4f} | single96={mean_single:.4f}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(out_path, index=False)
            if split.startswith("eva") and not limit and not is_ablation:
                pd.DataFrame(rows).to_csv(latest_path, index=False)

    print(f"Saved -> {out_path}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict AASP-UML evaluation.")
    parser.add_argument("--method", choices=["iforest", "knn"], default="iforest")
    parser.add_argument("--split", choices=["eva350", "eva_full822", "all870", "tuning"], default="eva350")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--representation", choices=["features", "raw", "raw_znorm"], default="features",
        help=(
            "a0 ablation: 'raw' scores on the raw robust-scaled patch window, "
            "'raw_znorm' on the per-patch instance-normalized window (classic "
            "UCR-style subsequence-kNN), instead of the 27D engineered features "
            "(--method knn only)"
        ),
    )
    parser.add_argument(
        "--drop-group", choices=["amplitude", "shape", "temporal", "spectral", "trend"], default=None,
        help="a1 ablation: drop one engineered feature group before fitting kNN (--method knn only, representation='features' only)",
    )
    args = parser.parse_args()
    if args.representation != "features" and args.drop_group is not None:
        parser.error("--representation raw/raw_znorm and --drop-group are mutually exclusive")
    run_aasp_eval(
        method=args.method,
        split=args.split,
        limit=args.limit,
        representation=args.representation,
        drop_group=args.drop_group,
    )


if __name__ == "__main__":
    main()
