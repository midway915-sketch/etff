# scripts/build_tau_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# IO helpers
# -----------------------------
def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}. cols={list(df.columns)[:50]}")


# -----------------------------
# Core: compute time-to-success
# -----------------------------
def compute_tau_days_for_ticker(
    df_t: pd.DataFrame,
    profit_target: float,
    max_days: int,
) -> pd.DataFrame:
    """
    For each row (Date), compute earliest day d in [1..max_days] such that
    future High (within d days ahead, inclusive of day d) >= entry_close*(1+profit_target).

    Returns columns: Date, Ticker, TauDays (float), SuccessWithinMaxDays (int)
    """
    df_t = df_t.sort_values("Date").reset_index(drop=True)
    n = len(df_t)
    if n == 0:
        return df_t.assign(TauDays=np.nan, SuccessWithinMaxDays=0)[["Date", "Ticker", "TauDays", "SuccessWithinMaxDays"]]

    close0 = pd.to_numeric(df_t["Close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df_t["High"], errors="coerce").to_numpy(dtype=float)

    tau = np.full(n, np.nan, dtype=float)
    success = np.zeros(n, dtype=int)

    # For each start i, find smallest d where max(high[i+1:i+d]) >= close0[i]*(1+pt)
    # Note: we start counting from "next day" as day 1. If you want same-day fill, change start offset to 0.
    pt = float(profit_target)
    target = close0 * (1.0 + pt)

    for i in range(n):
        if not np.isfinite(close0[i]) or close0[i] <= 0:
            continue
        thr = target[i]
        end = min(n - 1, i + max_days)
        # search forward days 1..max_days
        found = False
        for j in range(i + 1, end + 1):
            if np.isfinite(high[j]) and high[j] >= thr:
                tau[i] = float(j - i)  # days-to-hit
                success[i] = 1
                found = True
                break
        if not found:
            tau[i] = np.nan
            success[i] = 0

    out = pd.DataFrame({
        "Date": df_t["Date"].to_numpy(),
        "Ticker": df_t["Ticker"].to_numpy(),
        "TauDays": tau,
        "SuccessWithinMaxDays": success,
    })
    return out


def tau_class_from_tau_days(tau_days: float | None, success: int, k1: int, k2: int, max_days: int) -> int:
    """
    0=FAST, 1=MID, 2=SLOW
    - FAST: success and tau<=k1
    - MID : success and k1<tau<=k2
    - SLOW: everything else (including failures)
    """
    if success != 1 or tau_days is None or not np.isfinite(tau_days):
        return 2
    d = int(round(float(tau_days)))
    if d <= int(k1):
        return 0
    if d <= int(k2):
        return 1
    if d <= int(max_days):
        return 2
    return 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tau labels (FAST/MID/SLOW) for buy-sizing.")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)

    # keep interface consistent with your workflow even if tau labeling itself doesn't need stop/extend
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--k1", default=10, type=int, help="FAST cutoff (days)")
    ap.add_argument("--k2", default=20, type=int, help="MID cutoff (days)")

    ap.add_argument("--start-date", default="", type=str, help="optional: only label rows with Date >= start-date")
    ap.add_argument("--out-parq", default="data/labels/labels_tau.parquet", type=str)
    ap.add_argument("--out-csv", default="data/labels/labels_tau.csv", type=str)

    args = ap.parse_args()

    prices = read_table(args.prices_parq, args.prices_c_csv if hasattr(args, "prices_c_csv") else args.prices_c_csv if False else args.prices_csv)
    prices = prices.copy()

    ensure_cols(prices, ["Date", "Ticker", "High", "Close"], "prices")

    prices["Date"] = norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["Date", "Ticker", "Close", "High"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if args.start_date:
        sd = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(sd):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        # We still need history BEFORE start-date for forward scanning? Actually tau scans forward, so it's ok.
        prices = prices[prices["Date"] >= sd].copy()

    out_parts = []
    for t, df_t in prices.groupby("Ticker", sort=True):
        out_parts.append(compute_tau_days_for_ticker(df_t, args.profit_target, args.max_days))

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=["Date", "Ticker", "TauDays", "SuccessWithinMaxDays"])

    # TauClass
    k1 = int(args.k1)
    k2 = int(args.k2)
    md = int(args.max_days)
    out["TauClass"] = [
        tau_class_from_tau_days(td, int(s), k1, k2, md)
        for td, s in zip(out["TauDays"].tolist(), out["SuccessWithinMaxDays"].tolist())
    ]

    # store params for traceability
    out["ProfitTarget"] = float(args.profit_target)
    out["MaxDays"] = int(args.max_days)
    out["StopLevel"] = float(args.stop_level)
    out["MaxExtendDaysParam"] = int(args.max_extend_days)
    out["K1"] = int(k1)
    out["K2"] = int(k2)

    # save
    Path(args.out_parq).parent.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(args.out_parq, index=False)
        print(f"[DONE] wrote: {args.out_parq} rows={len(out)}")
    except Exception as e:
        print(f"[WARN] parquet write failed: {e}")

    out.to_csv(args.out_c_csv if hasattr(args, "out_c_csv") else args.out_csv, index=False)
    print(f"[DONE] wrote: {args.out_csv} rows={len(out)}")

    # quick stats
    vc = out["TauClass"].value_counts(dropna=False).to_dict() if "TauClass" in out.columns else {}
    print(f"[INFO] TauClass counts: {vc}")


if __name__ == "__main__":
    main()