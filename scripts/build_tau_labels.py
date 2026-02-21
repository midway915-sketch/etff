#!/usr/bin/env python3
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
    raise FileNotFoundError(f"Missing file: {p} (or {c})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}. cols(head)={list(df.columns)[:50]}")


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
    future High (within d days ahead) >= entry_close*(1+profit_target).

    Returns: Date, Ticker, TauDays(float), SuccessWithinMaxDays(int)
    Note: day 1 means "next day". (same-day fill not counted)
    """
    df_t = df_t.sort_values("Date").reset_index(drop=True)
    n = len(df_t)
    if n == 0:
        return pd.DataFrame(columns=["Date", "Ticker", "TauDays", "SuccessWithinMaxDays"])

    close0 = pd.to_numeric(df_t["Close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df_t["High"], errors="coerce").to_numpy(dtype=float)

    tau = np.full(n, np.nan, dtype=float)
    success = np.zeros(n, dtype=int)

    pt = float(profit_target)
    target = close0 * (1.0 + pt)

    for i in range(n):
        if not np.isfinite(close0[i]) or close0[i] <= 0:
            continue

        thr = target[i]
        end = min(n - 1, i + int(max_days))

        # scan forward days 1..max_days (i+1..end)
        for j in range(i + 1, end + 1):
            if np.isfinite(high[j]) and high[j] >= thr:
                tau[i] = float(j - i)
                success[i] = 1
                break

    out = pd.DataFrame(
        {
            "Date": df_t["Date"].to_numpy(),
            "Ticker": df_t["Ticker"].to_numpy(),
            "TauDays": tau,
            "SuccessWithinMaxDays": success,
        }
    )
    return out


def tau_class_fixed(tau_days: float, success: int, k1: int, k2: int) -> int:
    """
    0=FAST, 1=MID, 2=SLOW
    - FAST: success and tau<=k1
    - MID : success and k1<tau<=k2
    - SLOW: everything else (including failures)
    """
    if int(success) != 1 or (tau_days is None) or (not np.isfinite(tau_days)):
        return 2
    d = int(round(float(tau_days)))
    if d <= int(k1):
        return 0
    if d <= int(k2):
        return 1
    return 2


def _quantile_cutoffs_success(out: pd.DataFrame, q1: float, q2: float) -> tuple[int, int]:
    """
    Compute integer day cutoffs (k1,k2) using TauDays distribution among successes only.
    Returns (k1,k2) where:
      FAST: tau <= k1
      MID : k1 < tau <= k2
      SLOW: tau > k2 OR failure
    """
    suc = out.loc[(out["SuccessWithinMaxDays"] == 1) & np.isfinite(out["TauDays"])].copy()
    if suc.empty:
        # no success at all -> anything is slow
        return (0, 0)

    td = pd.to_numeric(suc["TauDays"], errors="coerce").dropna()
    if td.empty:
        return (0, 0)

    q1 = float(q1)
    q2 = float(q2)
    q1 = max(0.0, min(1.0, q1))
    q2 = max(0.0, min(1.0, q2))
    if q2 < q1:
        q1, q2 = q2, q1

    k1 = int(np.ceil(float(td.quantile(q1))))
    k2 = int(np.ceil(float(td.quantile(q2))))
    if k2 < k1:
        k2 = k1
    return (k1, k2)


def _current_success_distribution_from_fixed(out: pd.DataFrame, k1: int, k2: int) -> tuple[float, float]:
    """
    Using fixed k1/k2, compute FAST and MID proportions among successes only.
    Returns (p_fast, p_mid). p_slow = 1 - p_fast - p_mid.
    """
    suc = out.loc[(out["SuccessWithinMaxDays"] == 1) & np.isfinite(out["TauDays"])].copy()
    if suc.empty:
        return (0.0, 0.0)

    cls = [tau_class_fixed(td, 1, k1, k2) for td in suc["TauDays"].to_numpy()]
    cls = np.asarray(cls, dtype=int)

    n = float(len(cls))
    p_fast = float(np.sum(cls == 0) / n) if n > 0 else 0.0
    p_mid = float(np.sum(cls == 1) / n) if n > 0 else 0.0
    # (cls==2 is remainder)
    return (p_fast, p_mid)


def _assign_tau_class_by_cutoffs(out: pd.DataFrame, k1: int, k2: int) -> pd.Series:
    """
    Apply integer cutoffs to ALL rows:
      - failures -> 2
      - successes -> based on tau_days and cutoffs
    """
    tau = pd.to_numeric(out["TauDays"], errors="coerce")
    suc = pd.to_numeric(out["SuccessWithinMaxDays"], errors="coerce").fillna(0).astype(int)

    cls = np.full(len(out), 2, dtype=int)
    ok = (suc == 1) & np.isfinite(tau.to_numpy(dtype=float))

    td = np.round(tau.to_numpy(dtype=float)).astype(int)
    cls[ok & (td <= int(k1))] = 0
    cls[ok & (td > int(k1)) & (td <= int(k2))] = 1
    cls[ok & (td > int(k2))] = 2
    return pd.Series(cls, index=out.index, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tau labels (FAST/MID/SLOW) for buy-sizing.")
    ap.add_argument("--tag", required=True, type=str, help="e.g. pt10_h40_sl10_ex20")

    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)

    # keep interface consistent with workflow (even if tau calc doesn't need them)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    # legacy fixed split (for "current" baseline derivation)
    ap.add_argument("--k1", default=10, type=int, help="(legacy) FAST cutoff (days) for current split")
    ap.add_argument("--k2", default=20, type=int, help="(legacy) MID cutoff (days) for current split")

    # NEW: split mode
    ap.add_argument(
        "--split-mode",
        default="current",
        choices=["current", "quantile", "fixed"],
        help=(
            "current: match current distribution (derived from legacy k1/k2 among successes) via quantiles\n"
            "quantile: use q1/q2 among successes (e.g. 0.25/0.75 => 25/50/25)\n"
            "fixed: use legacy k1/k2 directly"
        ),
    )
    ap.add_argument("--q1", default=0.25, type=float, help="quantile for FAST cutoff among successes")
    ap.add_argument("--q2", default=0.75, type=float, help="quantile for MID cutoff among successes")

    ap.add_argument("--start-date", default="", type=str, help="optional: only SAVE rows with Date >= start-date")

    ap.add_argument("--out-parq", default="", type=str)
    ap.add_argument("--out-csv", default="", type=str)

    args = ap.parse_args()

    # ---- output paths (tagged)
    out_parq = args.out_parq.strip() or f"data/labels/labels_tau_{args.tag}.parquet"
    out_csv = args.out_csv.strip() or f"data/labels/labels_tau_{args.tag}.csv"

    # read
    prices = read_table(args.prices_parq, args.prices_csv).copy()
    ensure_cols(prices, ["Date", "Ticker", "High", "Close"], "prices")

    prices["Date"] = norm_date(prices["Date"])
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    prices["High"] = pd.to_numeric(prices["High"], errors="coerce")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")

    prices = (
        prices.dropna(subset=["Date", "Ticker", "Close", "High"])
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    if prices.empty:
        raise RuntimeError("prices is empty after cleaning. Check fetch_prices output.")

    # compute tau labels on FULL cleaned data
    out_parts: list[pd.DataFrame] = []
    for _t, df_t in prices.groupby("Ticker", sort=True):
        out_parts.append(compute_tau_days_for_ticker(df_t, args.profit_target, args.max_days))

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(
        columns=["Date", "Ticker", "TauDays", "SuccessWithinMaxDays"]
    )

    # ---- decide cutoffs
    split_mode = str(args.split_mode).lower().strip()
    k1_legacy = int(args.k1)
    k2_legacy = int(args.k2)

    if split_mode == "fixed":
        k1_used, k2_used = k1_legacy, k2_legacy
        split_note = f"fixed(k1={k1_used},k2={k2_used})"

    elif split_mode == "quantile":
        k1_used, k2_used = _quantile_cutoffs_success(out, float(args.q1), float(args.q2))
        split_note = f"quantile(q1={float(args.q1):.4f},q2={float(args.q2):.4f})=>k1={k1_used},k2={k2_used}"

    else:
        # current: reproduce "current distribution" derived from legacy fixed split (among successes)
        p_fast, p_mid = _current_success_distribution_from_fixed(out, k1_legacy, k2_legacy)
        q1 = float(p_fast)
        q2 = float(p_fast + p_mid)
        k1_used, k2_used = _quantile_cutoffs_success(out, q1, q2)
        split_note = (
            f"current(match legacy among successes: k1={k1_legacy},k2={k2_legacy} "
            f"-> p_fast={p_fast:.4f},p_mid={p_mid:.4f} -> q1={q1:.4f},q2={q2:.4f} "
            f"=> k1={k1_used},k2={k2_used})"
        )

    # ---- TauClass using chosen cutoffs
    out["TauClass"] = _assign_tau_class_by_cutoffs(out, k1_used, k2_used)

    # store params for traceability
    out["Tag"] = str(args.tag)
    out["SplitMode"] = split_mode
    out["SplitNote"] = split_note

    out["ProfitTarget"] = float(args.profit_target)
    out["MaxDays"] = int(args.max_days)
    out["StopLevel"] = float(args.stop_level)
    out["MaxExtendDaysParam"] = int(args.max_extend_days)

    out["K1_used"] = int(k1_used)
    out["K2_used"] = int(k2_used)
    out["K1_legacy"] = int(k1_legacy)
    out["K2_legacy"] = int(k2_legacy)
    out["Q1"] = float(args.q1)
    out["Q2"] = float(args.q2)

    # optional: only SAVE rows with Date >= start-date (but compute uses full data)
    if args.start_date:
        sd = pd.to_datetime(args.start_date, errors="coerce")
        if pd.isna(sd):
            raise ValueError(f"Invalid --start-date: {args.start_date}")
        out = out[out["Date"] >= sd].copy()

    out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # save
    Path(out_parq).parent.mkdir(parents=True, exist_ok=True)
    wrote_parq = False
    try:
        out.to_parquet(out_parq, index=False)
        wrote_parq = True
        print(f"[DONE] wrote: {out_parq} rows={len(out)}")
    except Exception as e:
        print(f"[WARN] parquet write failed: {e}")

    out.to_csv(out_csv, index=False)
    print(f"[DONE] wrote: {out_csv} rows={len(out)}")
    if not wrote_parq:
        print("[INFO] parquet failed -> csv is the source of truth for this run")

    vc_all = out["TauClass"].value_counts(dropna=False).to_dict() if "TauClass" in out.columns else {}
    print(f"[INFO] TauClass counts (ALL): {vc_all}")

    suc = out.loc[out["SuccessWithinMaxDays"] == 1]
    vc_suc = suc["TauClass"].value_counts(dropna=False).to_dict() if not suc.empty else {}
    print(f"[INFO] TauClass counts (SUCCESS only): {vc_suc}")
    print(f"[INFO] Split: {split_note}")


if __name__ == "__main__":
    main()