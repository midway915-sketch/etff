#!/usr/bin/env python3
# scripts/build_labels.py
from __future__ import annotations

# ✅ FIX: "python scripts/xxx.py"로 실행될 때도 scripts.* import가 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feature_spec import get_feature_cols


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


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def read_prices() -> pd.DataFrame:
    if PRICES_PARQ.exists():
        df = pd.read_parquet(PRICES_PARQ)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError(f"Missing prices: {PRICES_PARQ} (or {PRICES_CSV})")

    df = df.copy()
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("prices must include Date and Ticker")

    df["Date"] = _norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise ValueError(f"prices missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Ticker", "Close", "High", "Low"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )
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

    pt_mult = 1.0 + float(profit_target)
    sl_mult = 1.0 + float(stop_level)  # stop_level typically negative

    for i in range(n):
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            continue

        profit_px = entry * pt_mult
        stop_px = entry * sl_mult

        end = min(n, i + horizon_days + 1)  # scan i+1 .. end-1

        hit_profit_day: int | None = None
        hit_stop = False

        for j in range(i + 1, end):
            # conservative: if stop ever hit before profit, fail
            if np.isfinite(low[j]) and low[j] <= stop_px:
                hit_stop = True
                break
            if np.isfinite(high[j]) and high[j] >= profit_px:
                hit_profit_day = j
                break

        if hit_profit_day is not None and (not hit_stop):
            success[i] = 1
            tau[i] = float(hit_profit_day - i)

    out = pd.DataFrame(
        {
            "Date": _norm_date(g["Date"]).to_numpy(),
            "Ticker": g["Ticker"].astype(str).str.upper().str.strip().to_numpy(),
            "Success": success.astype(int),
            "TauDays": tau.astype(float),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--profit-target", type=float, default=None)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--stop-level", type=float, default=None)

    ap.add_argument("--start-date", type=str, default=None, help="output rows with Date >= start-date (YYYY-MM-DD)")
    ap.add_argument("--buffer-days", type=int, default=120, help="extra past days for stable joins/labels")
    args = ap.parse_args()

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # defaults from env if not provided
    profit_target = args.profit_target if args.profit_target is not None else float(os.getenv("PROFIT_TARGET", "0.10"))
    max_days = args.max_days if args.max_days is not None else int(os.getenv("MAX_DAYS", "40"))
    stop_level = args.stop_level if args.stop_level is not None else float(os.getenv("STOP_LEVEL", "-0.10"))

    # ✅ 18개 SSOT 강제(섹터 포함)
    feature_cols = get_feature_cols(sector_enabled=True)
    if len(feature_cols) != 18:
        raise RuntimeError(f"SSOT feature cols must be 18, got {len(feature_cols)}: {feature_cols}")

    feats = read_table(FEAT_PARQ, FEAT_CSV).copy()
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must include Date and Ticker")

    feats["Date"] = _norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = (
        feats.dropna(subset=["Date", "Ticker"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # ✅ FAIL FAST: features_model에 18개가 모두 있어야 함
    missing = [c for c in feature_cols if c not in feats.columns]
    if missing:
        raise ValueError(
            f"features_model missing SSOT feature cols (must have all 18): {missing}\n"
            f"-> Fix build_features.py / feature_spec.py consistency."
        )

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
    labels_list: list[pd.DataFrame] = []
    for _t, g in prices.groupby("Ticker", sort=False):
        labels_list.append(compute_success_and_tau_for_ticker(g, max_days, profit_target, stop_level))

    labels = pd.concat(labels_list, ignore_index=True) if labels_list else pd.DataFrame()
    if labels.empty:
        raise RuntimeError("No labels produced. Check prices input.")

    labels["Date"] = _norm_date(labels["Date"])
    labels["Ticker"] = labels["Ticker"].astype(str).str.upper().str.strip()
    labels = (
        labels.dropna(subset=["Date", "Ticker"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # merge labels onto features dates (✅ one-to-one expected)
    merged = feats[["Date", "Ticker"] + feature_cols].merge(
        labels[["Date", "Ticker", "Success", "TauDays"]],
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )

    # ✅ strict keep: require Success exists (drop rows with no label)
    merged = merged.dropna(subset=["Success"]).copy()
    merged["Success"] = pd.to_numeric(merged["Success"], errors="coerce").fillna(0).astype(int)

    # output filter to start-date (true output cut)
    if start_date is not None:
        merged = merged.loc[merged["Date"] >= start_date].copy()

    merged = (
        merged.sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    merged.to_parquet(OUT_PARQ, index=False)
    merged.to_csv(OUT_CSV, index=False)

    print(f"[DONE] wrote: {OUT_PARQ} rows={len(merged)}")
    if len(merged):
        print(f"[INFO] range: {merged['Date'].min().date()}..{merged['Date'].max().date()}")
        print(f"[INFO] label params: profit_target={profit_target} max_days={max_days} stop_level={stop_level}")
        print(f"[INFO] feature_cols(18, forced): {feature_cols}")


if __name__ == "__main__":
    main()