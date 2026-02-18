# scripts/build_strategy_labels.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
LABEL_DIR = DATA_DIR / "labels"

PRICES_PARQUET = RAW_DIR / "prices.parquet"
PRICES_CSV = RAW_DIR / "prices.csv"

FEATURES_PARQUET = FEAT_DIR / "features_model.parquet"
FEATURES_CSV = FEAT_DIR / "features_model.csv"

OUT_STRATEGY_RAW = DATA_DIR / "strategy_raw_data.csv"

MARKET_TICKER = "SPY"

DEFAULT_FEATURE_COLS = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_tag(pt: float, max_days: int, sl: float, ex: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{ex}"


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
    except Exception:
        df.to_csv(csv, index=False)
        return str(csv)


def auto_load_features(features_path: str | None) -> tuple[pd.DataFrame, str]:
    """
    1) --features-path 있으면 사용
    2) 기본 features_model.* 있으면 사용
    3) 없으면 data/features 내 Date/Ticker 있는 파일 자동탐색
    """
    if features_path:
        fp = Path(features_path)
        if not fp.exists():
            raise FileNotFoundError(f"features-path not found: {fp}")
        df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
        return df, str(fp)

    if FEATURES_PARQUET.exists():
        return pd.read_parquet(FEATURES_PARQUET), str(FEATURES_PARQUET)
    if FEATURES_CSV.exists():
        return pd.read_csv(FEATURES_CSV), str(FEATURES_CSV)

    if not FEAT_DIR.exists():
        raise FileNotFoundError(f"features dir not found: {FEAT_DIR}")

    candidates = []
    candidates += list(FEAT_DIR.glob("*.parquet"))
    candidates += list(FEAT_DIR.glob("*.csv"))
    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        raise FileNotFoundError(
            f"Missing file: {FEATURES_PARQUET} (or {FEATURES_CSV}) and no candidates in {FEAT_DIR}"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    last_err = None
    for p in candidates:
        try:
            df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
            if {"Date", "Ticker"}.issubset(df.columns):
                return df, str(p)
        except Exception as e:
            last_err = e

    raise FileNotFoundError(
        f"Could not find a usable features file in {FEAT_DIR}. "
        f"Need columns Date/Ticker. Last error: {last_err}"
    )


def compute_market_frame_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Market_Drawdown: SPY 기준 252일 롤링 고점 대비 드로우다운
    Market_ATR_ratio: SPY ATR(14) / Close
    """
    px = prices.copy()
    px["Date"] = pd.to_datetime(px["Date"])
    px["Ticker"] = px["Ticker"].astype(str).str.upper().str.strip()

    m = px[px["Ticker"] == MARKET_TICKER].sort_values("Date").copy()
    if m.empty:
        raise RuntimeError(
            f"Market ticker {MARKET_TICKER} not found in prices. "
            f"Run fetch_prices.py --include-extra"
        )

    for c in ["Open", "High", "Low", "Close"]:
        if c not in m.columns:
            raise ValueError(f"prices missing {c} for market ticker {MARKET_TICKER}")

    close = m["Close"].astype(float)
    high = m["High"].astype(float)
    low = m["Low"].astype(float)

    # Drawdown_252
    roll_max = close.rolling(252, min_periods=252).max()
    mdd = (close / roll_max) - 1.0

    # ATR(14) ratio
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atr_ratio = atr14 / close

    out = pd.DataFrame(
        {
            "Date": m["Date"].values,
            "Market_Drawdown": mdd.values,
            "Market_ATR_ratio": atr_ratio.values,
        }
    )
    return out


def ensure_market_features(feats: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    feats에 Market_Drawdown / Market_ATR_ratio가 없으면 SPY로 계산해서 붙인다.
    feats에 이미 있더라도(부분만 있거나 NaN이 많을 수 있음), market 계산값으로 NaN만 채운다.
    merge 시 컬럼 겹침으로 suffix가 붙는 문제를 방지하기 위해 suffixes=('', '_m') 사용.
    """
    need_any = ("Market_Drawdown" not in feats.columns) or ("Market_ATR_ratio" not in feats.columns)
    if not need_any:
        return feats

    market = compute_market_frame_from_prices(prices)

    # ✅ 겹치는 컬럼은 market 쪽에 _m suffix를 강제로 붙인다
    merged = feats.merge(market, on="Date", how="left", suffixes=("", "_m"))

    for col in ["Market_Drawdown", "Market_ATR_ratio"]:
        col_m = f"{col}_m"

        if col in merged.columns and col_m in merged.columns:
            # feats에 기존 col이 있었던 케이스: 기존값 우선, NaN만 market으로 채움
            merged[col] = merged[col].combine_first(merged[col_m])
            merged.drop(columns=[col_m], inplace=True)

        elif col not in merged.columns and col_m in merged.columns:
            # feats에 없어서 market 값이 _m로 들어온 케이스(드물지만 안전하게 처리)
            merged.rename(columns={col_m: col}, inplace=True)

    return merged



def forward_roll_max_excl_today(s: pd.Series, window: int) -> pd.Series:
    return s[::-1].rolling(window, min_periods=window).max()[::-1].shift(-1)


def forward_roll_min_excl_today(s: pd.Series, window: int) -> pd.Series:
    return s[::-1].rolling(window, min_periods=window).min()[::-1].shift(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build strategy labels (Success + Tail + ExtendNeeded).")
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument(
        "--tail-threshold",
        type=float,
        default=-0.30,
        help="Tail event threshold as return (e.g. -0.30 means -30% drawdown event).",
    )

    ap.add_argument("--features-path", type=str, default=None)
    ap.add_argument("--feature-cols", type=str, default="",
                    help="콤마로 feature 컬럼 지정(비우면 기본 8개)")
    args = ap.parse_args()

    pt = float(args.profit_target)
    max_days = int(args.max_days)
    sl = float(args.stop_level)
    ex = int(args.max_extend_days)
    tail_thr = float(args.tail_threshold)

    tag = fmt_tag(pt, max_days, sl, ex)

    # prices
    prices = read_table(PRICES_PARQUET, PRICES_CSV).copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = (
        prices.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    need_px = {"Date", "Ticker", "High", "Low", "Close"}
    miss_px = [c for c in need_px if c not in prices.columns]
    if miss_px:
        raise ValueError(f"prices missing columns: {miss_px}")

    # features (auto)
    feats, feats_src = auto_load_features(args.features_path)
    feats = feats.copy()
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = (
        feats.sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    # ✅ Market_* 없으면 SPY로 생성해서 붙임
    feats = ensure_market_features(feats, prices)

    feature_cols = DEFAULT_FEATURE_COLS
    if str(args.feature_cols).strip():
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    missing_feat = [c for c in feature_cols if c not in feats.columns]
    if missing_feat:
        raise ValueError(f"features missing feature columns: {missing_feat} (src={feats_src})")

    base = feats[["Date", "Ticker"] + feature_cols].merge(
        prices[["Date", "Ticker", "High", "Low", "Close"]],
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    base = base.dropna(subset=["Close"] + feature_cols).reset_index(drop=True)
    base = base.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # horizons
    success_h = max_days
    tail_h = max_days + ex  # max_days + extend 구간까지 대충 포함

    out = []
    for tkr, g in base.groupby("Ticker", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)

        close = g["Close"].astype(float)
        high = g["High"].astype(float)
        low = g["Low"].astype(float)

        fut_max_high_success = forward_roll_max_excl_today(high, success_h)
        fut_min_low_tail = forward_roll_min_excl_today(low, tail_h)

        success = (fut_max_high_success >= close * (1.0 + pt)).astype("float")

        fut_dd_tail = (fut_min_low_tail / close) - 1.0
        tail = (fut_dd_tail <= tail_thr).astype("float")

        # max_days 시점 close 수익률 (extend 필요 여부 근사용)
        offset = max_days - 1
        close_at_max = close.shift(-offset)
        ret_at_max = (close_at_max / close) - 1.0

        # success 못했고 max_days 시점 수익률이 stop_level보다 더 나쁘면 extend 필요한 케이스로 표시
        extend_needed = ((success == 0) & (ret_at_max < sl)).astype("float")

        g["Success"] = success
        g["Tail"] = tail
        g["FutureMinDD_TailH"] = fut_dd_tail
        g["RetAtMaxDays"] = ret_at_max
        g["ExtendNeeded"] = extend_needed

        g = g.dropna(
            subset=["Success", "Tail", "FutureMinDD_TailH", "RetAtMaxDays", "ExtendNeeded"]
        ).reset_index(drop=True)

        out.append(g)

    labeled = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if labeled.empty:
        raise RuntimeError("No strategy labeled rows produced. Check horizons and raw data.")

    labeled["Success"] = labeled["Success"].astype(int)
    labeled["Tail"] = labeled["Tail"].astype(int)
    labeled["ExtendNeeded"] = labeled["ExtendNeeded"].astype(int)

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    out_parq = LABEL_DIR / f"strategy_raw_data_{tag}.parquet"
    out_csv = LABEL_DIR / f"strategy_raw_data_{tag}.csv"
    saved_to = save_table(labeled, out_parq, out_csv)

    labeled.to_csv(OUT_STRATEGY_RAW, index=False)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "tag": tag,
        "profit_target": pt,
        "max_days": max_days,
        "stop_level": sl,
        "max_extend_days": ex,
        "tail_threshold": tail_thr,
        "rows": int(len(labeled)),
        "base_success_rate": float(labeled["Success"].mean()),
        "base_tail_rate": float(labeled["Tail"].mean()),
        "base_extend_needed_rate": float(labeled["ExtendNeeded"].mean()),
        "min_date": str(labeled["Date"].min().date()),
        "max_date": str(labeled["Date"].max().date()),
        "saved_to": saved_to,
        "also_written": str(OUT_STRATEGY_RAW),
        "feature_cols": feature_cols,
        "features_source": feats_src,
        "tail_horizon_days": int(tail_h),
        "market_ticker": MARKET_TICKER,
    }
    (LABEL_DIR / f"strategy_raw_data_{tag}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"[DONE] strategy labels saved -> {saved_to}")
    print(f"[DONE] also wrote -> {OUT_STRATEGY_RAW}")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
