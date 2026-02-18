# scripts/build_features.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQ = FEAT_DIR / "features_model.parquet"
OUT_CSV = FEAT_DIR / "features_model.csv"

MARKET_TICKER = "SPY"   # market proxy
VIX_TICKER = "^VIX"     # optional (not required for these features)


def read_prices() -> pd.DataFrame:
    if PRICES_PARQ.exists():
        df = pd.read_parquet(PRICES_PARQ)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError(f"Missing prices: {PRICES_PARQ} (or {PRICES_CSV})")

    # normalize schema
    df = df.copy()
    if "Date" not in df.columns:
        raise ValueError("prices must include Date column")
    if "Ticker" not in df.columns:
        raise ValueError("prices must include Ticker column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    # Require essential OHLCV
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"prices missing required column: {c}")

    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def compute_atr_ratio(g: pd.DataFrame, n: int = 14) -> pd.Series:
    high = g["High"].astype(float).to_numpy()
    low = g["Low"].astype(float).to_numpy()
    close = g["Close"].astype(float).to_numpy()
    prev_close = np.r_[np.nan, close[:-1]]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)

    tr_s = pd.Series(tr, index=g.index)
    atr = tr_s.rolling(n, min_periods=n).mean()
    atr_ratio = atr / pd.Series(close, index=g.index)
    return atr_ratio


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    c = pd.to_numeric(g["Close"], errors="coerce")
    v = pd.to_numeric(g["Volume"], errors="coerce")

    # Drawdown windows
    roll_max_252 = c.rolling(252, min_periods=252).max()
    dd_252 = (c / roll_max_252) - 1.0

    roll_max_60 = c.rolling(60, min_periods=60).max()
    dd_60 = (c / roll_max_60) - 1.0

    # ATR ratio
    atr_ratio = compute_atr_ratio(g, n=14)

    # Z-score on 20d
    ma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std(ddof=0)
    z = (c - ma20) / std20

    # MACD histogram
    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    macd_hist = macd - signal

    # MA20 slope (5d change rate)
    ma20_slope = (ma20 / ma20.shift(5)) - 1.0

    # simple momentum score (ret_score) : 20d return
    ret_score = (c / c.shift(20)) - 1.0

    out = pd.DataFrame({
        "Date": g["Date"].values,
        "Ticker": g["Ticker"].values,
        "Drawdown_252": dd_252.values,
        "Drawdown_60": dd_60.values,
        "ATR_ratio": atr_ratio.values,
        "Z_score": z.values,
        "MACD_hist": macd_hist.values,
        "MA20_slope": ma20_slope.values,
        "ret_score": ret_score.values,
        "Volume": v.values,
        "Close": c.values,
    })

    return out


def compute_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    m = prices.loc[prices["Ticker"] == MARKET_TICKER].sort_values("Date").copy()
    if m.empty:
        raise ValueError(f"Market ticker {MARKET_TICKER} not found. Use fetch_prices.py --include-extra")

    c = pd.to_numeric(m["Close"], errors="coerce")

    roll_max_252 = c.rolling(252, min_periods=252).max()
    mdd = (c / roll_max_252) - 1.0

    atr_ratio = compute_atr_ratio(m, n=14)

    out = pd.DataFrame({
        "Date": m["Date"].values,
        "Market_Drawdown": mdd.values,
        "Market_ATR_ratio": atr_ratio.values,
    }).sort_values("Date").reset_index(drop=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", type=str, default=None, help="only output rows with Date >= start-date (YYYY-MM-DD)")
    ap.add_argument("--max-window", type=int, default=260, help="max rolling window used (controls lookback)")
    ap.add_argument("--buffer-days", type=int, default=40, help="extra days added to lookback for safety")
    ap.add_argument("--min-volume", type=float, default=0.0, help="optional: drop rows with Volume < min-volume")
    args = ap.parse_args()

    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    prices = read_prices()

    # start-date handling: include lookback
    start_date = None
    if args.start_date:
        start_date = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(start_date):
            raise ValueError(f"Invalid --start-date: {args.start_date}")

        lookback_days = int(args.max_window + args.buffer_days)
        compute_start = start_date - pd.Timedelta(days=lookback_days)
        prices = prices.loc[prices["Date"] >= compute_start].copy()

    # compute per-ticker features
    feats_list = []
    for t, g in prices.groupby("Ticker", sort=False):
        feats_list.append(compute_ticker_features(g))
    feats = pd.concat(feats_list, ignore_index=True)

    # market features merge
    market = compute_market_features(prices)

    feats = feats.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    market = market.sort_values("Date").reset_index(drop=True)

    feats = feats.merge(market, on="Date", how="left")

    # strict NaN drop (feature completeness)
    FEATURE_COLS = [
        "Drawdown_252", "Drawdown_60", "ATR_ratio", "Z_score",
        "MACD_hist", "MA20_slope", "Market_Drawdown", "Market_ATR_ratio",
        "ret_score",
    ]

    # optional volume filter
    if args.min_volume and args.min_volume > 0:
        feats = feats.loc[pd.to_numeric(feats["Volume"], errors="coerce") >= float(args.min_volume)].copy()

    feats = feats.dropna(subset=FEATURE_COLS + ["Date", "Ticker"]).copy()

    # output filter to start-date
    if start_date is not None:
        feats = feats.loc[pd.to_datetime(feats["Date"]) >= start_date].copy()

    feats = feats.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    # save
    feats.to_parquet(OUT_PARQ, index=False)
    feats.to_csv(OUT_CSV, index=False)

    print(f"[DONE] wrote: {OUT_PARQ} rows={len(feats)}")
    if len(feats):
        print(f"[INFO] range: {pd.to_datetime(feats['Date']).min().date()}..{pd.to_datetime(feats['Date']).max().date()}")


if __name__ == "__main__":
    main()