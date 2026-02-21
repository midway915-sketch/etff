#!/usr/bin/env python3
# scripts/build_features.py
from __future__ import annotations

# ✅ FIX: "python scripts/xxx.py"로 실행될 때도 scripts.* import가 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feature_spec import (
    get_feature_cols,
    write_feature_cols_meta,
)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"

PRICES_PARQ = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

OUT_PARQ = FEAT_DIR / "features_model.parquet"
OUT_CSV = FEAT_DIR / "features_model.csv"

# 시장 피처(드로우다운/ATR/beta) 기준
MARKET_TICKER = "SPY"   # market proxy
# ✅ 상대강도(RelStrength) 기준 (요청: UPRO)
REL_BENCH_TICKER = "UPRO"

# ✅ 룩어헤드 방지: "당일 종가 진입(B)" 기준이면 무조건 1일 lag 피처만 사용
LOOKAHEAD_SHIFT_DAYS = 1


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

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Ticker", "Close", "High", "Low"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )
    return df


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
    """
    Market features based on MARKET_TICKER (default SPY):
      - Market_Drawdown
      - Market_ATR_ratio
      - Market_ret_1d (for beta)
      - Market_ret_20 ✅ NEW (SSOT)
    """
    m = prices.loc[prices["Ticker"] == MARKET_TICKER].sort_values("Date").copy()
    if m.empty:
        raise ValueError(f"Market ticker {MARKET_TICKER} not found. Use fetch_prices.py --include-extra")

    m["Date"] = pd.to_datetime(m["Date"], errors="coerce").dt.tz_localize(None)
    m = m.dropna(subset=["Date"]).reset_index(drop=True)

    c = pd.to_numeric(m["Close"], errors="coerce")

    roll_max_252 = c.rolling(252, min_periods=252).max()
    mdd = (c / roll_max_252) - 1.0

    atr_ratio = compute_atr_ratio(m, n=14)

    mret_1d = c.pct_change()
    mret_20 = (c / c.shift(20)) - 1.0  # ✅ NEW

    out = pd.DataFrame(
        {
            "Date": m["Date"].to_numpy(),
            "Market_Drawdown": mdd.to_numpy(),
            "Market_ATR_ratio": atr_ratio.to_numpy(),
            "Market_ret_1d": mret_1d.to_numpy(),
            "Market_ret_20": mret_20.to_numpy(),  # ✅ NEW
        }
    ).sort_values("Date").reset_index(drop=True)

    return out


def compute_benchmark_ret20(prices: pd.DataFrame, bench_ticker: str) -> pd.DataFrame:
    """
    Bench 20d return series for RelStrength:
      RelStrength = ret_20(ticker) - bench_ret_20(UPRO)
    """
    b = prices.loc[prices["Ticker"] == bench_ticker].sort_values("Date").copy()
    if b.empty:
        raise ValueError(f"Benchmark ticker {bench_ticker} not found in prices. (need in universe + fetch_prices include)")
    b["Date"] = pd.to_datetime(b["Date"], errors="coerce").dt.tz_localize(None)
    b = b.dropna(subset=["Date"]).reset_index(drop=True)
    c = pd.to_numeric(b["Close"], errors="coerce")
    bench_ret_20 = (c / c.shift(20)) - 1.0
    return pd.DataFrame({"Date": b["Date"].to_numpy(), "Bench_ret_20": bench_ret_20.to_numpy()})


def compute_ticker_features(g: pd.DataFrame, market_ret_by_date: pd.Series) -> pd.DataFrame:
    """
    ✅ beta_60 index 정렬로 길이 꼬임 방지
    """
    g = g.sort_values("Date").copy()
    dt = pd.to_datetime(g["Date"], errors="coerce").dt.tz_localize(None)

    c = pd.to_numeric(g["Close"], errors="coerce")
    v = pd.to_numeric(g["Volume"], errors="coerce")

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

    ret_20 = (c / c.shift(20)) - 1.0
    ret_score = ret_20.copy()

    ret_5 = (c / c.shift(5)) - 1.0
    ret_10 = (c / c.shift(10)) - 1.0

    roll_max_20 = c.rolling(20, min_periods=20).max()
    breakout_20 = (c / roll_max_20) - 1.0

    vol_ma20 = v.rolling(20, min_periods=20).mean()
    vol_surge = v / vol_ma20

    ema50 = ema(c, 50)
    trend_align = (c / ema50) - 1.0

    # --- beta_60 (✅ index 정렬)
    r = c.pct_change()  # index = g.index
    mret_np = market_ret_by_date.reindex(dt).to_numpy(dtype=float)  # length == len(g)
    mret = pd.Series(mret_np, index=g.index)  # ✅ align index with r

    cov = r.rolling(60, min_periods=60).cov(mret)
    var = mret.rolling(60, min_periods=60).var()
    beta_60 = cov / var

    n = len(g)
    cols_for_len = {
        "Date": len(dt),
        "Ticker": len(g["Ticker"]),
        "Drawdown_252": len(dd_252),
        "Drawdown_60": len(dd_60),
        "ATR_ratio": len(atr_ratio),
        "Z_score": len(z),
        "MACD_hist": len(macd_hist),
        "MA20_slope": len(ma20_slope),
        "ret_score": len(ret_score),
        "ret_5": len(ret_5),
        "ret_10": len(ret_10),
        "ret_20": len(ret_20),
        "breakout_20": len(breakout_20),
        "vol_surge": len(vol_surge),
        "trend_align": len(trend_align),
        "beta_60": len(beta_60),
        "Volume": len(v),
        "Close": len(c),
    }
    bad = {k: v for k, v in cols_for_len.items() if v != n}
    if bad:
        raise ValueError(f"Feature length mismatch for ticker={g['Ticker'].iloc[0]}: expected={n}, got={bad}")

    out = pd.DataFrame(
        {
            "Date": dt.to_numpy(),
            "Ticker": g["Ticker"].to_numpy(),
            "Drawdown_252": dd_252.to_numpy(),
            "Drawdown_60": dd_60.to_numpy(),
            "ATR_ratio": atr_ratio.to_numpy(),
            "Z_score": z.to_numpy(),
            "MACD_hist": macd_hist.to_numpy(),
            "MA20_slope": ma20_slope.to_numpy(),
            "ret_score": ret_score.to_numpy(),
            "ret_5": ret_5.to_numpy(),
            "ret_10": ret_10.to_numpy(),
            "ret_20": ret_20.to_numpy(),
            "breakout_20": breakout_20.to_numpy(),
            "vol_surge": vol_surge.to_numpy(),
            "trend_align": trend_align.to_numpy(),
            "beta_60": beta_60.to_numpy(),
            # debug columns
            "Volume": v.to_numpy(),
            "Close": c.to_numpy(),
        }
    )
    return out


def apply_lookahead_shift_per_ticker(df: pd.DataFrame, cols: list[str], shift_days: int) -> pd.DataFrame:
    """
    ✅ 룩어헤드 0 보장:
    Date=t 행의 cols 값은 't-1까지'로만 계산된 값이 되도록 shift(1).
    """
    if shift_days <= 0:
        return df

    out = df.sort_values(["Ticker", "Date"]).copy()
    out[cols] = out.groupby("Ticker", sort=False)[cols].shift(shift_days)
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

    start_date = None
    if args.start_date:
        start_date = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(start_date):
            raise ValueError(f"Invalid --start-date: {args.start_date}")

        lookback_days = int(args.max_window + args.buffer_days)
        compute_start = start_date - pd.Timedelta(days=lookback_days)
        prices = prices.loc[prices["Date"] >= compute_start].copy()

    # market (SPY) features
    market = compute_market_features(prices).sort_values("Date").reset_index(drop=True)

    market_ret_by_date = pd.Series(
        market["Market_ret_1d"].to_numpy(dtype=float),
        index=pd.to_datetime(market["Date"], errors="coerce").dt.tz_localize(None),
    )

    feats_list: list[pd.DataFrame] = []
    for _t, g in prices.groupby("Ticker", sort=False):
        feats_list.append(compute_ticker_features(g, market_ret_by_date=market_ret_by_date))

    feats = pd.concat(feats_list, ignore_index=True) if feats_list else pd.DataFrame()
    if feats.empty:
        raise RuntimeError("No features produced. Check prices input.")

    feats["Date"] = pd.to_datetime(feats["Date"], errors="coerce").dt.tz_localize(None)
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).reset_index(drop=True)

    # merge market columns needed by SSOT
    market_merge_cols = ["Date", "Market_Drawdown", "Market_ATR_ratio", "Market_ret_20"]
    feats = feats.merge(market[market_merge_cols], on="Date", how="left", validate="many_to_one")

    if args.min_volume and args.min_volume > 0:
        feats = feats.loc[pd.to_numeric(feats["Volume"], errors="coerce") >= float(args.min_volume)].copy()

    # ✅ RelStrength = ret_20 - ret_20(UPRO)
    bench = compute_benchmark_ret20(prices, bench_ticker=REL_BENCH_TICKER)
    feats = feats.merge(bench, on="Date", how="left", validate="many_to_one")

    feats["Bench_ret_20"] = pd.to_numeric(feats["Bench_ret_20"], errors="coerce")
    feats["ret_20"] = pd.to_numeric(feats["ret_20"], errors="coerce")
    feats["RelStrength"] = feats["ret_20"] - feats["Bench_ret_20"]

    # (debug용으로 남기고 싶으면 아래 drop 주석 처리)
    feats = feats.drop(columns=["Bench_ret_20"], errors="ignore")

    # ✅ SSOT 18개 강제
    FEATURE_COLS = get_feature_cols(sector_enabled=False)
    if len(FEATURE_COLS) != 18:
        raise RuntimeError(f"SSOT feature cols must be 18, got {len(FEATURE_COLS)}: {FEATURE_COLS}")

    missing_cols = [c for c in FEATURE_COLS if c not in feats.columns]
    if missing_cols:
        raise RuntimeError(f"Computed features missing SSOT columns: {missing_cols}")

    # ✅ LOOK-AHEAD GUARD: ticker별 1일 shift
    feats = apply_lookahead_shift_per_ticker(feats, FEATURE_COLS, LOOKAHEAD_SHIFT_DAYS)

    # shift 후 NaN 생기는 구간 제거(초반부)
    feats = feats.dropna(subset=FEATURE_COLS + ["Date", "Ticker"]).copy()

    # start-date cut
    if start_date is not None:
        feats = feats.loc[pd.to_datetime(feats["Date"], errors="coerce") >= start_date].copy()

    feats = (
        feats.sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # save
    try:
        feats.to_parquet(OUT_PARQ, index=False)
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}) -> writing csv only")
    feats.to_csv(OUT_CSV, index=False)

    # ✅ SSOT meta write
    meta_path = write_feature_cols_meta(cols=FEATURE_COLS, sector_enabled=False)

    print(f"[DONE] wrote: {OUT_PARQ} rows={len(feats)}")
    if len(feats):
        dmin = pd.to_datetime(feats["Date"]).min().date()
        dmax = pd.to_datetime(feats["Date"]).max().date()
        print(f"[INFO] range: {dmin}..{dmax}")
        print(f"[INFO] lookahead_shift_days: {LOOKAHEAD_SHIFT_DAYS} (B: enter at close)")
        print(f"[INFO] market proxy: {MARKET_TICKER} (Market_*)")
        print(f"[INFO] relstrength bench: {REL_BENCH_TICKER} (RelStrength = ret_20 - bench_ret_20)")
        print(f"[INFO] feature_cols_meta: {meta_path}")
        print(f"[INFO] feature_cols({len(FEATURE_COLS)}): {FEATURE_COLS}")


if __name__ == "__main__":
    main()