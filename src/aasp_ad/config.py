"""Central path configuration for the project (reproducible).

Every module/notebook should import paths FROM HERE instead of hard-coding
absolute paths. Paths are derived relative to this file's location, so anyone
who clones the repo can run it immediately without editing paths.

Usage:
    from aasp_ad.config import TSB_AD_U_DIR, EDA_TABLES_DIR
    from aasp_ad.config import eva_full_files, tuning_files
"""

from pathlib import Path

# Project root = the directory two levels above this file:
#   src/aasp_ad/config.py -> parents[0]=aasp_ad, [1]=src, [2]=<project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----- Data -----
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TSB_AD_U_DIR = RAW_DATA_DIR / "TSB-AD-U"

# TSB-AD-U file lists mirrored from TheDatumOrg/TSB-AD.
# Eva      (350 files) matches the PaAno Appendix Table 15 subset counts.
# Eva-Full (822 files) is an extended non-tuning evaluation set.
# Tuning   ( 48 files) is for hyperparameter search only.
#
# Important: PaAno Table 1 prints TSB-AD-U Eval as 530 series, but Appendix
# Table 15 lists subset counts that sum to 350 and match TSB-AD-U-Eva.csv
# exactly. Keep this discrepancy visible in reports.
_EVA_LIST = DATA_DIR / "TSB-AD-U-Eva.csv"
_EVA_FULL_LIST = DATA_DIR / "TSB-AD-U-Eva-Full.csv"
_TUNING_LIST = DATA_DIR / "TSB-AD-U-Tuning.csv"
_PAANO_EVAL_530_LIST = DATA_DIR / "TSB-AD-U-Eval-530.csv"


def _load_file_list(csv_path: Path) -> set[str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    col = df.columns[0]
    return set(df[col].str.strip().tolist())


def eva_repo_files() -> list[Path]:
    """Return sorted Path list of the 350-file TSB-AD-U Eva split."""
    names = _load_file_list(_EVA_LIST)
    paths = sorted(p for p in TSB_AD_U_DIR.glob("*.csv") if p.name in names)
    return paths


def eva_full_files() -> list[Path]:
    """Return sorted Path list of the extended 822-file TSB-AD-U Eva-Full split."""
    names = _load_file_list(_EVA_FULL_LIST)
    paths = sorted(p for p in TSB_AD_U_DIR.glob("*.csv") if p.name in names)
    return paths


def eva_files() -> list[Path]:
    """Backward-compatible alias for the 350-file TSB-AD-U Eva split."""
    return eva_repo_files()


def paano_eval_530_files() -> list[Path]:
    """Return an optional PaAno Table-1 530-file Eval split, if supplied."""
    if not _PAANO_EVAL_530_LIST.exists():
        raise FileNotFoundError(
            "PaAno Table 1 prints TSB-AD-U Eval as 530 series, but Appendix "
            "Table 15 and TSB-AD-U-Eva.csv align on 350 series. "
            f"{_PAANO_EVAL_530_LIST} is not present; add it only if you obtain "
            "an author-provided or otherwise exact 530-file list."
        )
    names = _load_file_list(_PAANO_EVAL_530_LIST)
    paths = sorted(p for p in TSB_AD_U_DIR.glob("*.csv") if p.name in names)
    return paths


def tuning_files() -> list[Path]:
    """Return sorted Path list of the 48 TSB-AD-U Tuning files."""
    names = _load_file_list(_TUNING_LIST)
    paths = sorted(p for p in TSB_AD_U_DIR.glob("*.csv") if p.name in names)
    return paths

# ----- Documents -----
DOCS_DIR = PROJECT_ROOT / "docs"

# ----- EDA outputs -----
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
EDA_TABLES_DIR = EDA_DIR / "tables"
EDA_FIGURES_DIR = EDA_DIR / "figures"
EDA_NOTES_DIR = EDA_DIR / "notes"

# ----- Method results -----
RESULTS_DIR = OUTPUTS_DIR / "results"

# ----- Notebooks -----
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def ensure_output_dirs() -> None:
    """Create the output directories if missing (safe to call repeatedly)."""
    for d in (EDA_TABLES_DIR, EDA_FIGURES_DIR, EDA_NOTES_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Print ASCII only to avoid encoding errors on the Windows console (cp1252).
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("TSB_AD_U_DIR :", TSB_AD_U_DIR)
    print("  exists?    :", TSB_AD_U_DIR.exists())
    n_csv = len(list(TSB_AD_U_DIR.glob("*.csv"))) if TSB_AD_U_DIR.exists() else 0
    print("  n_csv      :", n_csv)
