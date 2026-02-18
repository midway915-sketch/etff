# scripts/aggregate_gate_grid.py
from __future__ import annotations

import argparse
from pathlib import Path
import re
import math
import pandas as pd
import numpy as np


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_prices(prices_parq: Path, prices_csv: Path) -> pd.DataFrame:
    if prices_parq.exists():
        df = pd.read_parquet(prices_parq)
    elif prices_csv.exists():
        df = pd.read_csv(prices_csv)
    else:
        raise FileNotFoundError(f"Missing prices: {prices_parq} (or {prices_csv})")

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("prices must have Date,Ticker")
    for c in ["Close"]:
        if c not in df.columns:
            raise ValueError(f"prices missing {c}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def _find_suffix_from_summary_path(p: Path) -> str:
    # gate_summary_<TAG>_gate_<SUFFIX>.csv
    name = p.name
    m = re.match(r"gate_summary_(.+)_gate_(.+)\.csv$", name)
    if m:
        return m.group(2)
    # fallback
    return p.stem.replace("gate_summary_", "")


def _curve_path(signals_dir: Path, tag: str, suffix: str) -> Path:
    return signals_dir / f"sim_engine_curve_{tag}_gate_{suffix}.parquet"


def _trades_path(signals_dir: Path, tag: str, suffix: str) -> Path:
    return signals_dir / f"sim_engine_trades_{tag}_gate_{suffix}.parquet"


def _safe_to_dt(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


def _calc_cagr(start_equity: float, end_equity: float, days: float) -> float:
    if not np.isfinite(start_equity) or not np.isfinite(end_equity) or start_equity <= 0 or end_equity <= 0:
        return float("nan")
    if not np.isfinite(days) or days <= 0:
        return float("nan")
    years = days / 365.0
    if years <= 0:
        return float("nan")
    return float((end_equity / start_equity) ** (1.0 / years) - 1.0)


def _qqq_stats(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float, float]:
    """
    Return (qqq_seed_multiple, qqq_cagr, days)
    - Uses last available close on/after start and last available close on/before end
    """
    q = prices.loc[prices["Ticker"] == "QQQ"].copy()
    if q.empty:
        return (float("nan"), float("nan"), float("nan"))

    q = q.sort_values("Date")
    # pick start price: first close with Date >= start
    q_start = q.loc[q["Date"] >= start]
    if q_start.empty:
        return (float("nan"), float("nan"), float("nan"))
    p0 = float(q_start["Close"].iloc[0])

    # pick end price: last close with Date <= end
    q_end = q.loc[q["Date"] <= end]
    if q_end.empty:
        return (float("nan"), float("nan"), float("nan"))
    p1 = float(q_end["Close"].iloc[-1])

    if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(p1) or p1 <= 0:
        return (float("nan"), float("nan"), float("nan"))

    mult = p1 / p0
    days = float((end - start).days)
    cagr = _calc_cagr(1.0, mult, days)  # treat as seed multiple
    return (float(mult), float(cagr), float(days))


def enrich_one_summary(
    row: pd.Series,
    signals_dir: Path,
    prices: pd.DataFrame,
) -> dict:
    """
    Add:
      - WarmupEndDate (first EntryDate)
      - BacktestDaysAfterWarmup
      - IdleDaysAfterWarmup / IdlePctAfterWarmup
      - CAGR_AfterWarmup
      - QQQ_SeedMultiple_SamePeriod / QQQ_CAGR_SamePeriod
      - ActiveDaysAfterWarmup
    """
    tag = str(row.get("TAG", "run"))
    suffix = str(row.get("GateSuffix", ""))

    out = {}

    cpath = _curve_path(signals_dir, tag, suffix)
    tpath = _trades_path(signals_dir, tag, suffix)

    if (not cpath.exists()) or (not tpath.exists()):
        # if missing, just NaN fields
        out.update({
            "WarmupEndDate": "",
            "BacktestDaysAfterWarmup": np.nan,
            "ActiveDaysAfterWarmup": np.nan,
            "IdleDaysAfterWarmup": np.nan,
            "IdlePctAfterWarmup": np.nan,
            "CAGR_AfterWarmup": np.nan,
            "QQQ_SeedMultiple_SamePeriod": np.nan,
            "QQQ_CAGR_SamePeriod": np.nan,
        })
        return out

    trades = pd.read_parquet(tpath)
    if trades.empty or "EntryDate" not in trades.columns:
        out.update({
            "WarmupEndDate": "",
            "BacktestDaysAfterWarmup": np.nan,
            "ActiveDaysAfterWarmup": np.nan,
            "IdleDaysAfterWarmup": np.nan,
            "IdlePctAfterWarmup": np.nan,
            "CAGR_AfterWarmup": np.nan,
            "QQQ_SeedMultiple_SamePeriod": np.nan,
            "QQQ_CAGR_SamePeriod": np.nan,
        })
        return out

    entry = _safe_to_dt(trades["EntryDate"]).dropna()
    if entry.empty:
        out.update({
            "WarmupEndDate": "",
            "BacktestDaysAfterWarmup": np.nan,
            "ActiveDaysAfterWarmup": np.nan,
            "IdleDaysAfterWarmup": np.nan,
            "IdlePctAfterWarmup": np.nan,
            "CAGR_AfterWarmup": np.nan,
            "QQQ_SeedMultiple_SamePeriod": np.nan,
            "QQQ_CAGR_SamePeriod": np.nan,
        })
        return out

    warmup_end = entry.min()

    curve = pd.read_parquet(cpath)
    if curve.empty or "Date" not in curve.columns or "Equity" not in curve.columns:
        out.update({
            "WarmupEndDate": str(warmup_end.date()),
            "BacktestDaysAfterWarmup": np.nan,
            "ActiveDaysAfterWarmup": np.nan,
            "IdleDaysAfterWarmup": np.nan,
            "IdlePctAfterWarmup": np.nan,
            "CAGR_AfterWarmup": np.nan,
            "QQQ_SeedMultiple_SamePeriod": np.nan,
            "QQQ_CAGR_SamePeriod": np.nan,
        })
        return out

    curve = curve.copy()
    curve["Date"] = _safe_to_dt(curve["Date"])
    curve["Equity"] = pd.to_numeric(curve["Equity"], errors="coerce")
    if "InCycle" in curve.columns:
        inc = pd.to_numeric(curve["InCycle"], errors="coerce").fillna(0).astype(int)
    elif "InPosition" in curve.columns:
        inc = pd.to_numeric(curve["InPosition"], errors="coerce").fillna(0).astype(int)
    else:
        inc = pd.Series([0] * len(curve), index=curve.index)

    curve = curve.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)

    # only after warmup_end (include warmup_end day)
    cur2 = curve.loc[curve["Date"] >= warmup_end].copy()
    if cur2.empty:
        out.update({
            "WarmupEndDate": str(warmup_end.date()),
            "BacktestDaysAfterWarmup": np.nan,
            "ActiveDaysAfterWarmup": np.nan,
            "IdleDaysAfterWarmup": np.nan,
            "IdlePctAfterWarmup": np.nan,
            "CAGR_AfterWarmup": np.nan,
            "QQQ_SeedMultiple_SamePeriod": np.nan,
            "QQQ_CAGR_SamePeriod": np.nan,
        })
        return out

    inc2 = inc.loc[cur2.index] if inc.index.equals(curve.index) else pd.to_numeric(cur2.get("InCycle", 0), errors="coerce").fillna(0).astype(int)

    start_date = cur2["Date"].iloc[0]
    end_date = cur2["Date"].iloc[-1]
    days = float((end_date - start_date).days)

    start_eq = float(cur2["Equity"].iloc[0])
    end_eq = float(cur2["Equity"].iloc[-1])

    cagr = _calc_cagr(start_eq, end_eq, days)

    total_days = int(cur2["Date"].nunique())
    active_days = int((inc2 > 0).sum())
    idle_days = int(max(0, total_days - active_days))
    idle_pct = float(idle_days / total_days) if total_days > 0 else float("nan")

    qqq_mult, qqq_cagr, _ = _qqq_stats(prices, start_date, end_date)

    out.update({
        "WarmupEndDate": str(warmup_end.date()),
        "BacktestDaysAfterWarmup": days,
        "ActiveDaysAfterWarmup": active_days,
        "IdleDaysAfterWarmup": idle_days,
        "IdlePctAfterWarmup": idle_pct,
        "CAGR_AfterWarmup": cagr,
        "QQQ_SeedMultiple_SamePeriod": qqq_mult,
        "QQQ_CAGR_SamePeriod": qqq_cagr,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-dir", default="data/signals", type=str)
    ap.add_argument("--out-aggregate", default="data/signals/gate_grid_aggregate.csv", type=str)
    ap.add_argument("--out-top", default="data/signals/gate_grid_top_by_recent10y.csv", type=str)
    ap.add_argument("--pattern", default="gate_summary_*.csv", type=str)
    ap.add_argument("--topn", default=50, type=int)

    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    args = ap.parse_args()

    signals_dir = Path(args.signals_dir)
    if not signals_dir.exists():
        raise FileNotFoundError(f"signals dir not found: {signals_dir}")

    summaries = sorted(signals_dir.glob(args.pattern))
    if not summaries:
        raise FileNotFoundError(f"No summary files found: {signals_dir}/{args.pattern}")

    prices = _read_prices(Path(args.prices_parq), Path(args.prices_csv))

    rows = []
    for sp in summaries:
        df = _read_csv(sp)
        if df.empty:
            continue
        row = df.iloc[0].copy()

        # normalize expected fields from summarize_sim_trades.py
        # TAG and GateSuffix must exist
        if "TAG" not in row.index:
            row["TAG"] = "run"
        if "GateSuffix" not in row.index:
            row["GateSuffix"] = _find_suffix_from_summary_path(sp)

        enriched = enrich_one_summary(row, signals_dir=signals_dir, prices=prices)
        for k, v in enriched.items():
            row[k] = v

        rows.append(row)

    out = pd.DataFrame(rows)

    # ---- robust numeric conversions for scoring columns
    for c in [
        "Recent10Y_SeedMultiple", "SeedMultiple",
        "MaxHoldingDaysObserved", "MaxExtendDaysObserved",
        "CycleCount", "SuccessRate",
        "MaxLeveragePct",
        "CAGR_AfterWarmup", "QQQ_CAGR_SamePeriod",
        "IdlePctAfterWarmup",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # ---- add a simple "lever-adjusted" score (optional)
    # This keeps your existing idea: higher seed multiple, lower leverage peak
    if "SeedMultiple" in out.columns and "MaxLeveragePct" in out.columns:
        lev = out["MaxLeveragePct"].fillna(0.0)
        sm = out["SeedMultiple"].fillna(np.nan)
        # penalize leverage: divide by (1 + lev)
        out["SeedMultiple_LevAdj"] = sm / (1.0 + lev)

    # ---- sort aggregate: prefer SeedMultiple first, then LevAdj, then CAGR
    sort_cols = []
    if "SeedMultiple" in out.columns:
        sort_cols.append("SeedMultiple")
    if "SeedMultiple_LevAdj" in out.columns:
        sort_cols.append("SeedMultiple_LevAdj")
    if "CAGR_AfterWarmup" in out.columns:
        sort_cols.append("CAGR_AfterWarmup")

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    out_path = Path(args.out_aggregate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote aggregate: {out_path} rows={len(out)}")

    # ---- Top-by-recent10y (너가 계속 보던 지표 유지)
    top = out.copy()
    if "Recent10Y_SeedMultiple" in top.columns:
        top = top.sort_values(["Recent10Y_SeedMultiple"], ascending=[False])
    top = top.head(int(args.topn))
    top_path = Path(args.out_top)
    top.to_csv(top_path, index=False)
    print(f"[DONE] wrote top: {top_path} rows={len(top)}")

    # ---- quick headline
    if len(out):
        best = out.iloc[0].to_dict()
        print("=" * 60)
        print("[BEST] (by SeedMultiple / LevAdj / CAGR)")
        print(f"TAG={best.get('TAG')} suffix={best.get('GateSuffix')}")
        print(f"SeedMultiple={best.get('SeedMultiple')}  Recent10Y={best.get('Recent10Y_SeedMultiple')}")
        print(f"CAGR_AfterWarmup={best.get('CAGR_AfterWarmup')}  QQQ_CAGR={best.get('QQQ_CAGR_SamePeriod')}")
        print(f"IdlePctAfterWarmup={best.get('IdlePctAfterWarmup')}  MaxLevPct={best.get('MaxLeveragePct')}")
        print("=" * 60)


if __name__ == "__main__":
    main()