# scripts/patch_market_features.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

FEATURES_MODEL_PARQ = FEATURES_DIR / "features_model.parquet"
FEATURES_MODEL_CSV = FEATURES_DIR / "features_model.csv"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def save_table(df: pd.DataFrame, parq: Path, csv: Path) -> str:
    parq.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parq, index=False)
        return str(parq)
    except Exception as e:
        print(f"[WARN] parquet save failed: {e} -> saving csv {csv}")
        df.to_csv(csv, index=False)
        return str(csv)


def read_prices() -> pd.DataFrame:
    df = read_table(PRICES_PARQ, PRICES_CSV).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    need = {"Date", "Ticker", "High", "Low", "Close"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"prices missing columns: {miss}")
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def compute_market_frames(prices: pd.DataFrame, market_ticker: str = "SPY") -> pd.DataFrame:
    m = prices[prices["Ticker"] == market_ticker].sort_values("Date").copy()
    if m.empty:
        raise ValueError(f"Market ticker {market_ticker} not found. Run fetch_prices.py --include-extra")

    close = pd.to_numeric(m["Close"], errors="coerce")
    high = pd.to_numeric(m["High"], errors="coerce")
    low = pd.to_numeric(m["Low"], errors="coerce")

    roll_max_252 = close.rolling(252, min_periods=20).max()
    m["Market_Drawdown"] = (close / roll_max_252) - 1.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=10).mean()
    m["Market_ATR_ratio"] = atr14 / close

    return m[["Date", "Market_Drawdown", "Market_ATR_ratio"]].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch features_model with Market_Drawdown / Market_ATR_ratio computed from SPY.")
    ap.add_argument("--market-ticker", type=str, default="SPY")
    args = ap.parse_args()

    feats = read_table(FEATURES_MODEL_PARQ, FEATURES_MODEL_CSV).copy()

    # 최소 스키마 보정
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError(f"features_model must have Date and Ticker columns. cols={list(feats.columns)[:30]}")

    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    prices = read_prices()
    mfeat = compute_market_frames(prices, market_ticker=args.market_ticker)
    mfeat["Date"] = pd.to_datetime(mfeat["Date"])

    before_cols = set(feats.columns)

    # merge 후 combine_first로 기존 값 유지
    out = feats.merge(mfeat, on="Date", how="left", suffixes=("", "_mkt"))

    if "Market_Drawdown_mkt" in out.columns:
        if "Market_Drawdown" in out.columns:
            out["Market_Drawdown"] = pd.to_numeric(out["Market_Drawdown"], errors="coerce").combine_first(
                pd.to_numeric(out["Market_Drawdown_mkt"], errors="coerce")
            )
        else:
            out["Market_Drawdown"] = pd.to_numeric(out["Market_Drawdown_mkt"], errors="coerce")
        out.drop(columns=["Market_Drawdown_mkt"], inplace=True, errors="ignore")

    if "Market_ATR_ratio_mkt" in out.columns:
        if "Market_ATR_ratio" in out.columns:
            out["Market_ATR_ratio"] = pd.to_numeric(out["Market_ATR_ratio"], errors="coerce").combine_first(
                pd.to_numeric(out["Market_ATR_ratio_mkt"], errors="coerce")
            )
        else:
            out["Market_ATR_ratio"] = pd.to_numeric(out["Market_ATR_ratio_mkt"], errors="coerce")
        out.drop(columns=["Market_ATR_ratio_mkt"], inplace=True, errors="ignore")

    # 결과 검증
    miss = [c for c in ["Market_Drawdown", "Market_ATR_ratio"] if c not in out.columns]
    if miss:
        raise RuntimeError(f"[ERROR] still missing after patch: {miss}")

    after_cols = set(out.columns)
    added = sorted(list(after_cols - before_cols))
    print("[INFO] added cols:", added)

    saved = save_table(out, FEATURES_MODEL_PARQ, FEATURES_MODEL_CSV)
    print(f"[DONE] patched features_model saved to {saved}")
    print("[INFO] sample columns:", [c for c in ["Date","Ticker","Market_Drawdown","Market_ATR_ratio"] if c in out.columns])
    print("[INFO] rows:", len(out))


if __name__ == "__main__":
    main()