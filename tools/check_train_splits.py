import numpy as np
import pandas as pd
from aasp_ad.config import eva_repo_files, eva_full_files, tuning_files

splits = {
    "eva350": eva_repo_files(),
    "eva_full822": eva_full_files(),
    "tuning": tuning_files(),
}

count = 0
for name, files in splits.items():
    print(f"Checking {name} split ({len(files)} files)...")
    for path in files:
        df = pd.read_csv(path).dropna()
        n = len(df)
        train_split = int(path.stem.split("_")[-3])
        for patch_len in (64, 96, 128):
            if not (2 * patch_len < train_split < n):
                print(
                    f"File {path.name} patch_len={patch_len} invalid split: "
                    f"train_split={train_split}, n={n}"
                )
                count += 1
                continue
            train_end = train_split
            all_starts = np.arange(0, n - patch_len + 1, dtype=np.int64)
            train_mask = all_starts + patch_len <= train_end
            num_train = train_mask.sum()
            if num_train < 10:
                print(
                    f"File {path.name} patch_len={patch_len} only has "
                    f"{num_train} train patches! train_split={train_split}, n={n}"
                )
                count += 1

print(f"Done. Found {count} issues.")
