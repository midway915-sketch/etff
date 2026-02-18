# scripts/build_features.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
UNIVERSE_CSV = DATA_DIR / "universe.csv"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQ = FEAT_DIR / "features_model.parquet"
OUT_CSV = FEAT_DIR / "features_model.csv"

MARKET_TICKER = "SPY"   # market proxy (must exist in raw prices)
VIX_TICKER = "^VIX"     # optional (not required)


# -----------------------------
# IO helpers
# -----------------------------
def read_prices() -> pd.DataFrame:
    if PRICES_PARQ.exists():
        df = pd.read_parquet(PRICES_PARQ)
    elif PRICES_CSV.exists():
        df = pd.read_csv(PRICES_CSV)
    else:
        raise FileNotFoundError(f"Missing prices: {PRICES_PARQ} (or {PRICES_CSV})")

    df = df.copy()
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("prices must include Date and Ticker columns")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"prices missing required column: {c}")

    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def read_universe_groups() -> dict[str, str]:
    """
    Returns mapping: Ticker -> Group (sector-ish group)
    If universe.csv missing or lacks Group, returns empty dict.
    """
    if not UNIVERSE_CSV.exists():
        return {}
    uni = pd.read_csv(UNIVERSE_CSV)
    if "Ticker" not in uni.columns or "Group" not in uni.columns:
        return {}
    uni = uni.copy()
    uni["Ticker"] = uni["Ticker"].astype(str).str.upper().str.strip()
    uni["Group"] = uni["Group"].astype(str).str.strip()
    uni = uni.dropna(subset=["Ticker", "Group"])
    return dict(zip(uni["Ticker"].tolist(), uni["Group"].tolist()))


# -----------------------------
# feature building blocks
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_atr_ratio(g: pd.DataFrame, n: int = 14) -> pd.Series:
    high = pd.to_numeric(g["High"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(g["Low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(g["Close"], errors="coerce").to_numpy(dtype=float)
    prev_close = np.r_[np.nan, close[:-1]]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)

    tr_s = pd.Series(tr, index=g.index)
    atr = tr_s.rolling(n, min_periods=n).mean()
    atr_ratio = atr / pd.Series(close, index=g.index)
    return atr_ratio


def compute_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    m = prices.loc[prices["Ticker"] == MARKET_TICKER].sort_values("Date").copy()
    if m.empty:
        raise ValueError(f"Market ticker {MARKET_TICKER} not found. Use fetch_prices.py --include-extra")

    c = pd.to_numeric(m["Close"], errors="coerce")

    roll_max_252 = c.rolling(252, min_periods=252).max()
    mdd = (c / roll_max_252) - 1.0

    atr_ratio = compute_atr_ratio(m, n=14)

    # daily returns for beta
    mret = c.pct_change()

    out = pd.DataFrame({
        "Date": pd.to_datetime(m["Date"], errors="coerce").dt.tz_localize(None).values,
        "Market_Drawdown": mdd.values,
        "Market_ATR_ratio": atr_ratio.values,
        "Market_ret_1d": mret.values,
    }).sort_values("Date").reset_index(drop=True)

    return out


def compute_ticker_features(g: pd.DataFrame, market_ret_by_date: pd.Series) -> pd.DataFrame:
    """
    g: one ticker price frame (Date ascending)
    market_ret_by_date: Series indexed by Date with market 1d returns
    """
    g = g.sort_values("Date").copy()
    dt = pd.to_datetime(g["Date"], errors="coerce").dt.tz_localize(None)

    c = pd.to_numeric(g["Close"], errors="coerce")
    v = pd.to_numeric(g["Volume"], errors="coerce")

    # ----- base features (existing)
    roll_max_252 = c.rolling(252, min_periods=252).max()
    dd_252 = (c / roll_max_252) - 1.0

    roll_max_60 = c.rolling(60, min_periods=60).max()
    dd_60 = (c / roll_max_60) - 1.0

    atr_ratio = compute_atr_ratio(g, n=14)

    ma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std(ddof=0)
    z = (c - ma20) / std20

    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    macd_hist = macd - signal

    ma20_slope = (ma20 / ma20.shift(5)) - 1.0

    # existing "ret_score" kept as 20d return (used by utility)
    ret_20 = (c / c.shift(20)) - 1.0
    ret_score = ret_20.copy()

    # ----- NEW: TP1+trailing friendly features
    ret_5 = (c / c.shift(5)) - 1.0
    ret_10 = (c / c.shift(10)) - 1.0

    # breakout: close vs 20d rolling max (classic trend continuation proxy)
    roll_max_20 = c.rolling(20, min_periods=20).max()
    breakout_20 = (c / roll_max_20) - 1.0

    # volume surge: volume vs 20d avg volume
    vol_ma20 = v.rolling(20, min_periods=20).mean()
    vol_surge = v / vol_ma20

    # trend alignment: close vs EMA50
    ema50 = ema(c, 50)
    trend_align = (c / ema50) - 1.0

    # beta_60: cov(ret, market_ret) / var(market_ret)
    # align by Date
    r = c.pct_change()
    mret = market_ret_by_date.reindex(dt).astype(float)
    # rolling cov/var with min_periods=60
    cov = r.rolling(60, min_periods=60).cov(mret)
    var = mret.rolling(60, min_periods=60).var()
    beta_60 = cov / var

    out = pd.DataFrame({
        "Date": dt.values,
        "Ticker": g["Ticker"].values,

        # base
        "Drawdown_252": dd_252.values,
        "Drawdown_60": dd_60.values,
        "ATR_ratio": atr_ratio.values,
        "Z_score": z.values,
        "MACD_hist": macd_hist.values,
        "MA20_slope": ma20_slope.values,
        "ret_score": ret_score.values,

        # new
        "ret_5": ret_5.values,
        "ret_10": ret_10.values,
        "ret_20": ret_20.values,              # useful for sector strength & diagnostics
        "breakout_20": breakout_20.values,
        "vol_surge": vol_surge.values,
        "trend_align": trend_align.values,
        "beta_60": beta_60.values,

        # basics needed downstream
        "Volume": v.values,
        "Close": c.values,
    })

    return out


def add_sector_strength(feats: pd.DataFrame, ticker_to_group: dict[str, str]) -> pd.DataFrame:
    """
    Adds:
      - Sector_Ret_20: per-date per-group mean of ret_20
      - RelStrength: ret_20 - Sector_Ret_20
    Requires feats have Date,Ticker,ret_20
    """
    if not ticker_to_group:
        print("[WARN] universe.csv Group mapping not available -> sector strength skipped")
        return feats

    if "ret_20" not in feats.columns:
        print("[WARN] feats missing ret_20 -> sector strength skipped")
        return feats

    x = feats.copy()
    x["Group"] = x["Ticker"].map(ticker_to_group).fillna("UNKNOWN")

    # group mean per day
    sector_ret = (
        x.groupby(["Date", "Group"], as_index=False)["ret_20"]
        .mean()
        .rename(columns={"ret_20": "Sector_Ret_20"})
    )
    x = x.merge(sector_ret, on=["Date", "Group"], how="left")
    x["RelStrength"] = pd.to_numeric(x["ret_20"], errors="coerce") - pd.to_numeric(x["Sector_Ret_20"], errors="coerce")

    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", type=str, default=None, help="only output rows with Date >= start-date (YYYY-MM-DD)")
    ap.add_argument("--max-window", type=int, default=260, help="max rolling window used (controls lookback)")
    ap.add_argument("--buffer-days", type=int, default=40, help="extra days added to lookback for safety")
    ap.add_argument("--min-volume", type=float, default=0.0, help="optional: drop rows with Volume < min-volume")

    # final option: sector strength
    ap.add_argument("--enable-sector-strength", action="store_true", help="add sector strength features (needs universe.csv Group)")
    ap.add_argument("--disable-sector-strength", action="store_true", help="force disable sector strength")

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

    # market features + market returns index
    market = compute_market_features(prices)
    market = market.sort_values("Date").reset_index(drop=True)
    market_ret_by_date = pd.Series(market["Market_ret_1d"].values, index=pd.to_datetime(market["Date"]).dt.tz_localize(None))

    # per-ticker features
    feats_list = []
    for t, g in prices.groupby("Ticker", sort=False):
        feats_list.append(compute_ticker_features(g, market_ret_by_date=market_ret_by_date))
    feats = pd.concat(feats_list, ignore_index=True)

    # merge market regime features
    feats = feats.copy()
    feats["Date"] = pd.to_datetime(feats["Date"], errors="coerce").dt.tz_localize(None)
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()

    market_merge_cols = ["Date", "Market_Drawdown", "Market_ATR_ratio"]
    feats = feats.merge(market[market_merge_cols], on="Date", how="left")

    # optional volume filter
    if args.min_volume and args.min_volume > 0:
        feats = feats.loc[pd.to_numeric(feats["Volume"], errors="coerce") >= float(args.min_volume)].copy()

    # sector strength toggle:
    # default = ON (final recommended). can force off with --disable-sector-strength
    enable_sector = True
    if args.disable_sector_strength:
        enable_sector = False
    elif args.enable_sector_strength:
        enable_sector = True

    if enable_sector:
        ticker_to_group = read_universe_groups()
        feats = add_sector_strength(feats, ticker_to_group=ticker_to_group)

    # strict NaN drop (feature completeness)
    FEATURE_COLS = [
        # base
        "Drawdown_252", "Drawdown_60", "ATR_ratio", "Z_score",
        "MACD_hist", "MA20_slope", "Market_Drawdown", "Market_ATR_ratio",
        "ret_score",

        # new core
        "ret_5", "ret_10", "ret_20",
        "breakout_20", "vol_surge", "trend_align", "beta_60",
    ]

    # sector cols only if present
    if "Sector_Ret_20" in feats.columns:
        FEATURE_COLS += ["Sector_Ret_20", "RelStrength"]

    feats = feats.dropna(subset=FEATURE_COLS + ["Date", "Ticker"]).copy()

    # output filter to start-date
    if start_date is not None:
        feats = feats.loc[pd.to_datetime(feats["Date"], errors="coerce") >= start_date].copy()

    feats = feats.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    # save
    feats.to_parquet(OUT_PARQ, index=False)
    feats.to_csv(OUT_CSV, index=False)

    print(f"[DONE] wrote: {OUT_PARQ} rows={len(feats)}")
    if len(feats):
        dmin = pd.to_datetime(feats["Date"]).min().date()
        dmax = pd.to_datetime(feats["Date"]).max().date()
        print(f"[INFO] range: {dmin}..{dmax}")
        if "Sector_Ret_20" in feats.columns:
            print("[INFO] sector strength: ENABLED (Sector_Ret_20, RelStrength)")
        else:
            print("[INFO] sector strength: DISABLED")


if __name__ == "__main__":
    main()