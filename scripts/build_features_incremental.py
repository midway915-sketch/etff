# scripts/build_features_incremental.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import shutil


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"

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
    ap.add_argument("--max-window", type=int, default=260, help="max rolling window used by features")
    ap.add_argument("--buffer-days", type=int, default=30, help="extra days to recompute beyond max-window")
    ap.add_argument("--tmp-dir", type=str, default="data/_tmp_incremental")
    ap.add_argument("--features-script", type=str, default="scripts/build_features.py")
    args = ap.parse_args()

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    # 1) 기존 features 로드
    old = read_table(FEAT_PARQ, FEAT_CSV)
    if not old.empty:
        # 표준 컬럼
        if "Date" in old.columns:
            old["Date"] = pd.to_datetime(old["Date"], errors="coerce")
        if "Ticker" in old.columns:
            old["Ticker"] = old["Ticker"].astype(str).str.upper().str.strip()

    # 2) 최근구간 재계산 시작점 결정
    # old가 없으면 그냥 build_features.py를 그대로 실행(풀)
    if old.empty or "Date" not in old.columns:
        print("[INFO] no existing features -> full build via build_features.py")
        subprocess.check_call(["python", args.features_script])
        return

    last_date = pd.to_datetime(old["Date"].max())
    # 최근 구간만 재계산: last_date 기준으로 max_window+buffer 만큼 뒤에서부터
    # (rolling 계산 안정성 확보)
    start = last_date - pd.Timedelta(days=int(args.max_window + args.buffer_days))

    print(f"[INFO] existing features last_date={last_date.date()} -> recompute from {start.date()}")

    # 3) 기존 파일을 tmp로 백업해두고, build_features.py가 새로 만든 결과를 tmp로 받기
    #    가장 안전한 방식: build_features.py를 실행하되, 결과물을 tmp로 옮겨서 recent slice만 가져온다.
    #    (build_features.py가 output 경로를 바꿀 수 없다면 이 방식이 제일 현실적)

    # 3-1) 현재 features 파일 백업
    backup_dir = tmp / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if FEAT_PARQ.exists():
        shutil.copy2(FEAT_PARQ, backup_dir / "features_model.parquet")
    if FEAT_CSV.exists():
        shutil.copy2(FEAT_CSV, backup_dir / "features_model.csv")

    # 3-2) build_features.py 실행 (현재 코드대로 전체 생성)
    #      여기서 “전체 생성”이지만, merge 시 recent slice만 쓰기 때문에 실질적으로는 안전/간단.
    #      (추후 build_features.py에 start-date 옵션 생기면 여기만 바꾸면 됨)
    print("[INFO] running build_features.py to refresh features (will keep only recent slice)")
    subprocess.check_call(["python", args.features_script])

    new = read_table(FEAT_PARQ, FEAT_CSV)
    if new.empty:
        raise RuntimeError("build_features.py produced empty features_model")

    # 표준화
    if "Date" not in new.columns or "Ticker" not in new.columns:
        raise RuntimeError(f"features_model must include Date/Ticker. got cols={list(new.columns)[:40]}")
    new["Date"] = pd.to_datetime(new["Date"], errors="coerce")
    new["Ticker"] = new["Ticker"].astype(str).str.upper().str.strip()

    # 4) recent slice만 추출
    new_recent = new.loc[new["Date"] >= start].copy()

    # 5) old에서 동일 구간 제거 후 append
    old_keep = old.loc[old["Date"] < start].copy()

    merged = pd.concat([old_keep, new_recent], ignore_index=True)

    merged = (
        merged.sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    save_table(merged, FEAT_PARQ, FEAT_CSV)

    print(f"[DONE] merged features_model rows={len(merged)} range={merged['Date'].min().date()}..{merged['Date'].max().date()}")
    print(f"[INFO] recent recomputed rows={len(new_recent)} start={start.date()}")
    print("[TIP] if this is still too slow, next step is adding start-date support directly into build_features.py")
    

if __name__ == "__main__":
    main()