#!/usr/bin/env python3
# scripts/fetch_prices.py
from __future__ import annotations

import argparse
import json
import os  # ✅ ADDED
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
UNIVERSE_CSV = DATA_DIR / "universe.csv"

OUT_PARQUET = RAW_DIR / "prices.parquet"
OUT_CSV_FALLBACK = RAW_DIR / "prices.csv"
META_JSON = RAW_DIR / "prices_meta.json"

DEFAULT_EXTRA_TICKERS = ["SPY", "^VIX"]

# ✅ ADDED: allow workflow env to control max-years without changing CLI calls
DEFAULT_MAX_YEARS = int(os.getenv("MAX_YEARS", "11"))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_today_normalized() -> pd.Timestamp:
    # pandas/버전별 tz 처리 차이 방어
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).strip() for c in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has columns:
    Date, Ticker, Open, High, Low, Close, AdjClose, Volume
    """
    if df is None:
        return pd.DataFrame()

    df = _flatten_columns(df)

    # Date normalize
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex) or df.index.name in ("Date", "Datetime"):
            idx_name = df.index.name or "Date"
            df = df.reset_index().rename(columns={idx_name: "Date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "Date"})

    # Ticker normalize
    if "Ticker" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})

    # Adj Close normalize
    if "AdjClose" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "AdjClose"})
        elif "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "AdjClose"})
        elif "adj_close" in df.columns:
            df = df.rename(columns={"adj_close": "AdjClose"})

    needed = {"Date", "Ticker"}
    if not needed.issubset(set(df.columns)):
        raise RuntimeError(f"[SCHEMA] Missing {needed - set(df.columns)}. Columns={list(df.columns)[:30]}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    # numeric normalize (build_features 쪽에서 다시 coerce하긴 하는데 여기서도 정리)
    for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _normalize_enabled_col(s: pd.Series) -> pd.Series:
    """
    Enabled가 bool/0/1/"true"/"TRUE"/"yes" 등 섞여 있어도 안전하게 bool로 정규화.
    """
    x = s.copy()
    if x.dtype == bool:
        return x
    # 문자열/숫자 혼합 대응
    x = x.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f", "", "nan", "none"}
    return x.apply(lambda v: True if v in truthy else (False if v in falsy else False))


def read_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path} (run scripts/universe.py first)")

    uni = pd.read_csv(path)
    if "Ticker" not in uni.columns:
        raise ValueError("universe.csv must contain 'Ticker' column")

    if "Enabled" in uni.columns:
        uni = uni.copy()
        uni["Enabled"] = _normalize_enabled_col(uni["Enabled"])
        uni = uni.loc[uni["Enabled"] == True].copy()  # noqa: E712

    tickers = uni["Ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
    tickers = [t for t in tickers if t]
    if not tickers:
        raise RuntimeError("Universe tickers became empty after Enabled filter. Check universe.csv Enabled values.")
    return tickers


def safe_download_one(
    ticker: str,
    start: str | None,
    end: str | None,
    retries: int = 3,
    sleep_base: float = 1.2,
) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=False,
            )

            if df is None or df.empty:
                raise ValueError(f"Empty data for {ticker} (start={start}, end={end})")

            df = df.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"

            df = _flatten_columns(df)

            expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            missing = [c for c in expected if c not in df.columns]
            if missing:
                raise ValueError(f"{ticker} missing columns: {missing} (cols={list(df.columns)})")

            df = df[expected].rename(columns={"Adj Close": "AdjClose"}).reset_index()
            df.insert(1, "Ticker", ticker)

            return normalize_schema(df)

        except Exception as e:
            last_err = e
            time.sleep(sleep_base * attempt)

    raise RuntimeError(f"Failed to download {ticker} after {retries} retries: {last_err}")


def load_existing_prices() -> pd.DataFrame:
    if OUT_PARQUET.exists():
        try:
            return normalize_schema(pd.read_parquet(OUT_PARQUET))
        except Exception as e:
            print(f"[WARN] read_parquet failed: {e}")

    if OUT_CSV_FALLBACK.exists():
        return normalize_schema(pd.read_csv(OUT_CSV_FALLBACK))

    return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"])


def save_prices(df: pd.DataFrame) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = normalize_schema(df)

    try:
        df.to_parquet(OUT_PARQUET, index=False)
        return f"parquet:{OUT_PARQUET}"
    except Exception as e:
        print(f"[WARN] Parquet save failed ({e}). Saving CSV fallback: {OUT_CSV_FALLBACK}")
        df.to_csv(OUT_CSV_FALLBACK, index=False)
        return f"csv:{OUT_CSV_FALLBACK}"


def clamp_start_by_max_years(start: str | None, max_years: int) -> str | None:
    """
    Clamp start to be no earlier than (today - max_years).
    If start is None or invalid, returns the clamp date string.
    """
    today = _utc_today_normalized()
    min_start = (today - pd.Timedelta(days=int(max_years) * 365)).date()
    min_start_str = str(min_start)

    if start is None:
        return min_start_str

    sdt = pd.to_datetime(start, errors="coerce")
    if pd.isna(sdt):
        return min_start_str
    if sdt.tzinfo is not None:
        sdt = sdt.tz_localize(None)
    if sdt.date() < min_start:
        return min_start_str
    return str(sdt.date())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLCV prices from Yahoo Finance (incremental).")
    parser.add_argument("--force-full", action="store_true", help="Ignore existing file and refetch (still clamped by --max-years).")
    parser.add_argument("--reset", action="store_true", help="Delete stored price files before fetching.")
    parser.add_argument("--start", type=str, default=None, help="Override start date (YYYY-MM-DD), still clamped by --max-years.")
    parser.add_argument("--end", type=str, default=None, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument("--lookback-days", type=int, default=7, help="For incremental fetch, refetch this many days overlap.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-base", type=float, default=1.2)
    parser.add_argument("--include-extra", action="store_true", help="Include SPY and ^VIX for market features.")
    # ✅ CHANGED: default now comes from env MAX_YEARS (fallback 11)
    parser.add_argument("--max-years", type=int, default=DEFAULT_MAX_YEARS, help="Limit raw price history to recent N years (default from env MAX_YEARS or 11).")
    args = parser.parse_args()

    tickers = read_universe(UNIVERSE_CSV)
    if args.include_extra:
        tickers = sorted(set(tickers + DEFAULT_EXTRA_TICKERS))
    else:
        tickers = sorted(set(tickers))

    today = _utc_today_normalized()
    min_start = (today - pd.Timedelta(days=int(args.max_years) * 365)).date()
    print(f"[INFO] max-years={args.max_years} -> clamp start >= {min_start}")
    print(f"[INFO] tickers={len(tickers)} -> {tickers}")

    if args.reset:
        for p in [OUT_PARQUET, OUT_CSV_FALLBACK, META_JSON]:
            if p.exists():
                p.unlink()
        print("[INFO] reset done")

    existing = pd.DataFrame()
    if not args.force_full:
        existing = load_existing_prices()

    last_dates: dict[str, pd.Timestamp] = {}
    if not existing.empty:
        last_dates = existing.groupby("Ticker")["Date"].max().to_dict()

    downloads: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []

    for t in tickers:
        try:
            if args.force_full or existing.empty or t not in last_dates:
                start = args.start
            else:
                ld = pd.to_datetime(last_dates[t], errors="coerce")
                if pd.isna(ld):
                    start = args.start
                else:
                    start_dt = (ld - pd.Timedelta(days=int(args.lookback_days))).date()
                    start = args.start or str(start_dt)

            start = clamp_start_by_max_years(start, int(args.max_years))

            df_new = safe_download_one(
                ticker=t,
                start=start,
                end=args.end,
                retries=int(args.retries),
                sleep_base=float(args.sleep_base),
            )
            downloads.append(df_new)
            print(f"[OK] {t}: {df_new['Date'].min().date()} -> {df_new['Date'].max().date()} rows={len(df_new)}")

        except Exception as e:
            failed.append((t, str(e)))
            print(f"[FAIL] {t}: {e}")

    if not downloads:
        raise RuntimeError("No data downloaded.")

    new_all = normalize_schema(pd.concat(downloads, ignore_index=True))

    # defensive clamp
    min_start_ts = pd.Timestamp(min_start)
    new_all = new_all[new_all["Date"] >= min_start_ts].reset_index(drop=True)

    if existing.empty:
        combined = new_all
    else:
        combined = normalize_schema(pd.concat([existing, new_all], ignore_index=True))
        combined = combined[combined["Date"] >= min_start_ts].reset_index(drop=True)

    combined = (
        combined.sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    saved_to = save_prices(combined)

    meta = {
        "updated_at_utc": now_utc_iso(),
        "saved_to": saved_to,
        "tickers_requested": tickers,
        "tickers_downloaded": sorted(set(new_all["Ticker"].unique().tolist())),
        "min_date": str(combined["Date"].min().date()) if not combined.empty else None,
        "max_date": str(combined["Date"].max().date()) if not combined.empty else None,
        "rows": int(len(combined)),
        "failed": [{"ticker": t, "error": err} for t, err in failed],
        "force_full": bool(args.force_full),
        "reset": bool(args.reset),
        "include_extra": bool(args.include_extra),
        "max_years": int(args.max_years),
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DONE] saved={saved_to} rows={len(combined)} range={meta['min_date']}..{meta['max_date']}")

    # NOTE:
    # In truncated backtests (e.g., 2008~2014), many tickers may not have existed yet,
    # and yfinance returns "Empty data". That's not a fatal error as long as we got
    # enough tickers to proceed.
    ok = len(downloads)
    total = len(tickers)
    
    # Fail only when "too few" tickers succeeded (hard floor).
    # (Before: failed >= half -> abort)
    min_ok = max(3, total // 4)  # require at least 25% (and at least 3 tickers)
    if ok < min_ok:
        raise RuntimeError(f"Too few tickers downloaded: ok={ok}/{total}, failed={len(failed)}")
if __name__ == "__main__":
    main()