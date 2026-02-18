# scripts/build_labels_incremental.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import shutil


DATA_DIR = Path("data")
LABEL_DIR = DATA_DIR / "labels"
FEAT_DIR = DATA_DIR / "features"

LABEL_PARQ = LABEL_DIR / "labels_model.parquet"
LABEL_CSV = LABEL_DIR / "labels_model.csv"

FEAT_PARQ = FEAT_DIR / "features_model.parquet"
FEAT_CSV = FEAT_DIR / "features_model.csv"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def save_table(df: pd.DataFrame, parq: Path, csv: Path) -> None:
    parq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parq, index=False)
    df.to_csv(csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-days", type=int, default=60, help="recompute labels for last N days")
    ap.add_argument("--labels-script", type=str, default="scripts/build_labels.py")
    ap.add_argument("--tmp-dir", type=str, default="data/_tmp_incremental_labels")
    args = ap.parse_args()

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    feats = read_table(FEAT_PARQ, FEAT_CSV)
    if feats.empty:
        raise FileNotFoundError("Missing features_model. Run build_features first.")
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise RuntimeError("features_model must have Date/Ticker")

    feats["Date"] = pd.to_datetime(feats["Date"], errors="coerce")
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    last_date = pd.to_datetime(feats["Date"].max())
    start = last_date - pd.Timedelta(days=int(args.buffer_days))

    old = read_table(LABEL_PARQ, LABEL_CSV)
    if not old.empty:
        old["Date"] = pd.to_datetime(old["Date"], errors="coerce")
        old["Ticker"] = old["Ticker"].astype(str).str.upper().str.strip()

    # build_labels.py는 전체 라벨 만들 거라 가정(현재 구조 기준)
    # 만든 다음 "최근구간만" 가져와 merge
    print(f"[INFO] running build_labels.py (will keep only recent slice since {start.date()})")
    subprocess.check_call(["python", args.labels_script])

    new = read_table(LABEL_PARQ, LABEL_CSV)
    if new.empty:
        raise RuntimeError("build_labels.py produced empty labels_model")
    if "Date" not in new.columns or "Ticker" not in new.columns:
        raise RuntimeError("labels_model must have Date/Ticker")

    new["Date"] = pd.to_datetime(new["Date"], errors="coerce")
    new["Ticker"] = new["Ticker"].astype(str).str.upper().str.strip()

    new_recent = new.loc[new["Date"] >= start].copy()

    if old.empty:
        merged = new
    else:
        old_keep = old.loc[old["Date"] < start].copy()
        merged = pd.concat([old_keep, new_recent], ignore_index=True)

    merged = (
        merged.sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    save_table(merged, LABEL_PARQ, LABEL_CSV)

    print(f"[DONE] merged labels_model rows={len(merged)} range={merged['Date'].min().date()}..{merged['Date'].max().date()}")
    print(f"[INFO] recent recomputed rows={len(new_recent)} start={start.date()}")


if __name__ == "__main__":
    main()