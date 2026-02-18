# scripts/merge_incremental_table.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def _read(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


def _write(df: pd.DataFrame, parq: Path, csv: Path) -> None:
    parq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parq, index=False)
    df.to_csv(csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-parq", required=False, default="")
    ap.add_argument("--old-csv", required=False, default="")
    ap.add_argument("--new-parq", required=False, default="")
    ap.add_argument("--new-csv", required=False, default="")
    ap.add_argument("--out-parq", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--cut-date", required=False, default="", help="YYYY-MM-DD. old rows with Date < cut-date are kept.")
    args = ap.parse_args()

    old_parq = Path(args.old_parq) if args.old_parq else Path("_MISSING_OLD.parquet")
    old_csv = Path(args.old_csv) if args.old_csv else Path("_MISSING_OLD.csv")
    new_parq = Path(args.new_parq) if args.new_parq else Path("_MISSING_NEW.parquet")
    new_csv = Path(args.new_csv) if args.new_csv else Path("_MISSING_NEW.csv")

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)

    old = _read(old_parq, old_csv)
    new = _read(new_parq, new_csv)

    if new.empty and old.empty:
        raise SystemExit("[ERROR] both old and new are empty/missing")

    def norm(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if "Date" not in df.columns or "Ticker" not in df.columns:
            raise SystemExit(f"[ERROR] table must have Date/Ticker. cols={list(df.columns)[:40]}")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df = df.dropna(subset=["Date", "Ticker"])
        return df

    old = norm(old)
    new = norm(new)

    cut = None
    if args.cut_date:
        cut = pd.to_datetime(args.cut_date, errors="coerce")
        if pd.isna(cut):
            raise SystemExit(f"[ERROR] invalid --cut-date: {args.cut_date}")

    if old.empty:
        merged = new
    elif new.empty:
        merged = old
    else:
        if cut is not None:
            old_keep = old.loc[old["Date"] < cut].copy()
        else:
            old_keep = old.copy()
        merged = pd.concat([old_keep, new], ignore_index=True)

    merged = (
        merged.sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    _write(merged, out_parq, out_csv)

    print(f"[DONE] merged -> {out_parq} rows={len(merged)}")
    if len(merged):
        print(f"[INFO] range {merged['Date'].min().date()}..{merged['Date'].max().date()}")
        if cut is not None:
            print(f"[INFO] cut_date={cut.date()}")
        print(f"[INFO] old_rows={len(old)} new_rows={len(new)}")


if __name__ == "__main__":
    main()