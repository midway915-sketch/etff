# scripts/build_labels.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

FEAT_PARQ = FEAT_DIR / "features_model.parquet"
FEAT_CSV = FEAT_DIR / "features_model.csv"

OUT_PARQ = LABEL_DIR / "labels_model.parquet"
OUT_CSV = LABEL_DIR / "labels_model.csv"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def read_prices() -> pd.DataFrame:
    if PRICES_PARQ.exists():
        df = pd.read_parquet(PRICES_PARQ)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError(f"Missing prices: {PRICES_PARQ} (or {PRICES_CSV})")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise ValueError(f"prices missing column: {c}")
    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def compute_success_and_tau_for_ticker(
    g: pd.DataFrame,
    horizon_days: int,
    profit_target: float,
    stop_level: float,
) -> pd.DataFrame:
    """
    Conservative label:
      Success = (profit hit within horizon) AND (stop NOT hit within horizon)
    TauDays  = first day index (1..horizon) profit hit, else NaN

    Uses High/Low vs entry Close thresholds.
    """
    g = g.sort_values("Date").reset_index(drop=True).copy()

    close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(g["High"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(g["Low"], errors="coerce").to_numpy(dtype=float)

    n = len(g)
    success = np.zeros(n, dtype=np.int8)
    tau = np.full(n, np.nan, dtype=float)

    # precompute thresholds per row
    pt = 1.0 + float(profit_target)
    sl = 1.0 + float(stop_level)   # stop_level is negative typically

    # naive O(n*h) loop; with ~25 tickers and a few thousand days this is fine
    for i in range(n):
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            continue

        profit_px = entry * pt
        stop_px = entry * sl

        end = min(n, i + horizon_days + 1)  # include same-day? we start from i+1 for "future"
        hit_profit_day = None
        hit_stop = False

        # look forward day by day (i+1..end-1)
        for j in range(i + 1, end):
            if np.isfinite(low[j]) and low[j] <= stop_px:
                hit_stop = True
                # we still keep scanning to find earlier profit? order matters.
                # Conservative: if stop hit at any point, mark fail even if profit later.
                # (If you want order-aware: stop earlier than profit -> fail; profit earlier -> success)
                # For now conservative.
                break

            if np.isfinite(high[j]) and high[j] >= profit_px:
                hit_profit_day = j
                break

        if hit_profit_day is not None and (not hit_stop):
            success[i] = 1
            tau[i] = float(hit_profit_day - i)  # days until success

    out = pd.DataFrame({
        "Date": g["Date"].values,
        "Ticker": g["Ticker"].values,
        "Success": success,
        "TauDays": tau,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    # label parameters (default: env fallback -> hard default)
    ap.add_argument("--profit-target", type=float, default=None)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--stop-level", type=float, default=None)

    ap.add_argument("--start-date", type=str, default=None, help="only output rows with Date >= start-date (YYYY-MM-DD)")
    ap.add_argument("--buffer-days", type=int, default=120, help="extra past days to include for stable joins/labels")

    args = ap.parse_args()
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # defaults from env if not provided
    profit_target = args.profit_target if args.profit_target is not None else float(os.getenv("PROFIT_TARGET", "0.10"))
    max_days = args.max_days if args.max_days is not None else int(os.getenv("MAX_DAYS", "40"))
    stop_level = args.stop_level if args.stop_level is not None else float(os.getenv("STOP_LEVEL", "-0.10"))

    feats = read_table(FEAT_PARQ, FEAT_CSV).copy()
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must include Date and Ticker")

    feats["Date"] = pd.to_datetime(feats["Date"], errors="coerce")
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    prices = read_prices()

    # start-date handling: include buffer for joins/forward label calc
    start_date = None
    if args.start_date:
        start_date = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(start_date):
            raise ValueError(f"Invalid --start-date: {args.start_date}")

        compute_start = start_date - pd.Timedelta(days=int(args.buffer_days))
        feats = feats.loc[feats["Date"] >= compute_start].copy()
        prices = prices.loc[prices["Date"] >= compute_start].copy()

    # build labels from prices (per ticker)
    labels_list = []
    for t, g in prices.groupby("Ticker", sort=False):
        labels_list.append(compute_success_and_tau_for_ticker(g, max_days, profit_target, stop_level))
    labels = pd.concat(labels_list, ignore_index=True)

    labels["Date"] = pd.to_datetime(labels["Date"], errors="coerce")
    labels["Ticker"] = labels["Ticker"].astype(str).str.upper().str.strip()
    labels = labels.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # merge labels onto features dates
    merged = feats.merge(labels, on=["Date", "Ticker"], how="left")

    # define Target used by train_model.py
    merged["Target"] = pd.to_numeric(merged["Success"], errors="coerce")

    # strict keep: require Target exists (drop rows with no label)
    merged = merged.dropna(subset=["Target"]).copy()
    merged["Target"] = (merged["Target"] > 0).astype(int)

    # output filter to start-date (true output cut)
    if start_date is not None:
        merged = merged.loc[merged["Date"] >= start_date].copy()

    merged = merged.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    merged.to_parquet(OUT_PARQ, index=False)
    merged.to_csv(OUT_CSV, index=False)

    print(f"[DONE] wrote: {OUT_PARQ} rows={len(merged)}")
    if len(merged):
        print(f"[INFO] range: {merged['Date'].min().date()}..{merged['Date'].max().date()}")
        print(f"[INFO] label params: profit_target={profit_target} max_days={max_days} stop_level={stop_level}")


if __name__ == "__main__":
    main()