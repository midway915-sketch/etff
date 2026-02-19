# scripts/build_success_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
LBL_DIR = DATA_DIR / "labels"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQ = LBL_DIR / "labels_success.parquet"
OUT_CSV = LBL_DIR / "labels_success.csv"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Training data not found: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build success labels (Future max >= Close*(1+PT) within H days).")
    ap.add_argument("--profit-target", required=True, type=float)  # e.g. 0.10
    ap.add_argument("--max-days", required=True, type=int)         # e.g. 40
    ap.add_argument("--start-date", default=None, type=str, help="optional YYYY-MM-DD (output rows >= start-date)")
    ap.add_argument("--prices-parq", default=str(PRICES_PARQ), type=str)
    ap.add_argument("--prices-csv", default=str(PRICES_CSV), type=str)
    ap.add_argument("--out-parq", default=str(OUT_PARQ), type=str)
    ap.add_argument("--out-csv", default=str(OUT_CSV), type=str)
    args = ap.parse_args()

    LBL_DIR.mkdir(parents=True, exist_ok=True)

    prices = read_table(Path(args.prices_parq), Path(args.prices_csv)).copy()

    # schema checks
    need_cols = ["Date", "Ticker", "High", "Close"]
    missing = [c for c in need_cols if c not in prices.columns]
    if missing:
        raise ValueError(f"prices missing required columns for success labels: {missing}")

    prices["Date"] = norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()

    prices["High"] = pd.to_numeric(prices["High"], errors="coerce")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")

    prices = (
        prices.dropna(subset=["Date", "Ticker", "High", "Close"])
        .sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )

    H = int(args.max_days)
    pt = float(args.profit_target)

    # Compute: future_max_high over next H days (t+1..t+H), exclude today
    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").copy()
        # shift(-1) to start from next day
        future_max = g["High"].shift(-1).rolling(H, min_periods=H).max()
        g["FutureMaxHigh"] = future_max
        g["Success"] = (g["FutureMaxHigh"] >= g["Close"] * (1.0 + pt)).astype("int64")
        return g

    out = prices.groupby("Ticker", group_keys=False).apply(_per_ticker)

    # drop rows where we can't know future (tail)
    out = out.dropna(subset=["FutureMaxHigh"]).copy()

    # optional output cut
    if args.start_date:
        sd = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(sd):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        out = out.loc[out["Date"] >= sd].copy()

    labels = out[["Date", "Ticker", "Success"]].copy()

    # metadata columns (optional but handy; doesn't break trainers)
    labels["profit_target"] = pt
    labels["max_days"] = H

    labels = labels.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    labels.to_parquet(out_parq, index=False)
    labels.to_csv(out_csv, index=False)

    dmin = pd.to_datetime(labels["Date"]).min().date() if len(labels) else None
    dmax = pd.to_datetime(labels["Date"]).max().date() if len(labels) else None
    print(f"[DONE] wrote: {out_parq} rows={len(labels)} range={dmin}..{dmax} pt={pt} H={H}")


if __name__ == "__main__":
    main()