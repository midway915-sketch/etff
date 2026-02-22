# scripts/aggregate_gate_grid.py
from __future__ import annotations

import argparse
from pathlib import Path
import re
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
    if "Close" not in df.columns:
        raise ValueError("prices missing Close")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def _find_suffix_from_summary_path(p: Path) -> str:
    name = p.name
    m = re.match(r"gate_summary_(.+)_gate_(.+)\.csv$", name)
    if m:
        return m.group(2)
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
    q = prices.loc[prices["Ticker"] == "QQQ"].copy()
    if q.empty:
        return (float("nan"), float("nan"), float("nan"))

    q = q.sort_values("Date")

    q_start = q.loc[q["Date"] >= start]
    if q_start.empty:
        return (float("nan"), float("nan"), float("nan"))
    p0 = float(q_start["Close"].iloc[0])

    q_end = q.loc[q["Date"] <= end]
    if q_end.empty:
        return (float("nan"), float("nan"), float("nan"))
    p1 = float(q_end["Close"].iloc[-1])

    if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(p1) or p1 <= 0:
        return (float("nan"), float("nan"), float("nan"))

    mult = p1 / p0
    days = float((end - start).days)
    cagr = _calc_cagr(1.0, mult, days)
    return (float(mult), float(cagr), float(days))


def _clean_curve(curve: pd.DataFrame) -> pd.DataFrame:
    c = curve.copy()

    if "Date" not in c.columns or "Equity" not in c.columns:
        return pd.DataFrame()

    c["Date"] = _safe_to_dt(c["Date"])
    c["Equity"] = pd.to_numeric(c["Equity"], errors="coerce")

    if "InCycle" in c.columns:
        inc = pd.to_numeric(c["InCycle"], errors="coerce").fillna(0).astype(int)
    elif "InPosition" in c.columns:
        inc = pd.to_numeric(c["InPosition"], errors="coerce").fillna(0).astype(int)
    else:
        inc = pd.Series([0] * len(c), index=c.index, dtype=int)

    c["InCycle"] = (inc > 0).astype(int)

    c = c.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    c = c.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return c


def _parse_cap_from_suffix(suffix: str) -> tuple[str, str]:
    """
    GateSuffix가 ..._capnone / ..._caph2 / ..._captotal 로 끝나면 분리.
    returns: (base_suffix, cap_mode)
    """
    s = str(suffix or "")
    m = re.search(r"(.*)_cap(none|h2|total)$", s)
    if not m:
        return (s, "")
    return (m.group(1), m.group(2))


# ✅ NEW: yearly returns (warmup 이후 구간 기준)
def _calc_yearly_returns(cur2: pd.DataFrame) -> dict:
    """
    cur2: cleaned curve filtered to Date >= warmup_end (must have Date, Equity)
    Returns dict: {"YearReturn_YYYY": float, ...}
      - 연도별 수익률 = (해당연도 마지막 Equity / 직전 기준 Equity) - 1
      - 첫 해의 기준 Equity는 cur2의 첫 관측 Equity
      - 마지막 해는 부분연도(현재까지) 포함
    """
    if cur2 is None or cur2.empty:
        return {}

    c = cur2[["Date", "Equity"]].copy()
    c["Date"] = _safe_to_dt(c["Date"])
    c["Equity"] = pd.to_numeric(c["Equity"], errors="coerce")
    c = c.dropna(subset=["Date", "Equity"]).sort_values("Date").reset_index(drop=True)
    if c.empty:
        return {}

    # 연도별 마지막 equity
    c["Year"] = c["Date"].dt.year.astype(int)
    year_end = c.groupby("Year", as_index=False).tail(1).sort_values("Year").reset_index(drop=True)

    out: dict[str, float] = {}
    base = float(c["Equity"].iloc[0])  # 첫 해 기준: warmup 이후 첫 관측 equity
    if not np.isfinite(base) or base <= 0:
        return {}

    prev = base
    for _, r in year_end.iterrows():
        y = int(r["Year"])
        eq_end = float(r["Equity"])
        if np.isfinite(eq_end) and eq_end > 0 and np.isfinite(prev) and prev > 0:
            out[f"YearReturn_{y}"] = float(eq_end / prev - 1.0)
            prev = eq_end
        else:
            out[f"YearReturn_{y}"] = float("nan")
            # prev는 유지

    return out


def enrich_one_summary(row: pd.Series, signals_dir: Path, prices: pd.DataFrame) -> dict:
    tag = str(row.get("TAG", "run"))
    suffix = str(row.get("GateSuffix", ""))

    out = {}

    cpath = _curve_path(signals_dir, tag, suffix)
    tpath = _trades_path(signals_dir, tag, suffix)

    if (not cpath.exists()) or (not tpath.exists()):
        out.update({
            "WarmupEndDate": "",
            "BacktestDaysAfterWarmup": np.nan,
            "ActiveDaysAfterWarmup": np.nan,
            "IdleDaysAfterWarmup": np.nan,
            "IdlePctAfterWarmup": np.nan,
            "CAGR_AfterWarmup": np.nan,
            "QQQ_SeedMultiple_SamePeriod": np.nan,
            "QQQ_CAGR_SamePeriod": np.nan,
            "ExcessCAGR_AfterWarmup": np.nan,
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
            "ExcessCAGR_AfterWarmup": np.nan,
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
            "ExcessCAGR_AfterWarmup": np.nan,
        })
        return out

    warmup_end = entry.min()

    curve_raw = pd.read_parquet(cpath)
    curve = _clean_curve(curve_raw)
    if curve.empty:
        out.update({
            "WarmupEndDate": str(warmup_end.date()),
            "BacktestDaysAfterWarmup": np.nan,
            "ActiveDaysAfterWarmup": np.nan,
            "IdleDaysAfterWarmup": np.nan,
            "IdlePctAfterWarmup": np.nan,
            "CAGR_AfterWarmup": np.nan,
            "QQQ_SeedMultiple_SamePeriod": np.nan,
            "QQQ_CAGR_SamePeriod": np.nan,
            "ExcessCAGR_AfterWarmup": np.nan,
        })
        return out

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
            "ExcessCAGR_AfterWarmup": np.nan,
        })
        return out

    start_date = pd.Timestamp(cur2["Date"].iloc[0])
    end_date = pd.Timestamp(cur2["Date"].iloc[-1])
    days = float((end_date - start_date).days)

    start_eq = float(cur2["Equity"].iloc[0])
    end_eq = float(cur2["Equity"].iloc[-1])

    cagr = _calc_cagr(start_eq, end_eq, days)

    total_days = int(cur2["Date"].nunique())
    active_days = int((cur2["InCycle"] > 0).sum())
    idle_days = int(max(0, total_days - active_days))
    idle_pct = float(idle_days / total_days) if total_days > 0 else float("nan")

    qqq_mult, qqq_cagr, _ = _qqq_stats(prices, start_date, end_date)
    excess_cagr = cagr - qqq_cagr if np.isfinite(cagr) and np.isfinite(qqq_cagr) else float("nan")

    out.update({
        "WarmupEndDate": str(warmup_end.date()),
        "BacktestDaysAfterWarmup": days,
        "ActiveDaysAfterWarmup": active_days,
        "IdleDaysAfterWarmup": idle_days,
        "IdlePctAfterWarmup": idle_pct,
        "CAGR_AfterWarmup": cagr,
        "QQQ_SeedMultiple_SamePeriod": qqq_mult,
        "QQQ_CAGR_SamePeriod": qqq_cagr,
        "ExcessCAGR_AfterWarmup": excess_cagr,
    })

    # ✅ NEW: YearReturn_YYYY columns
    out.update(_calc_yearly_returns(cur2))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-dir", default="data/signals", type=str)
    ap.add_argument("--out-aggregate", default="data/signals/gate_grid_aggregate.csv", type=str)
    ap.add_argument("--out-top", default="data/signals/gate_grid_top_by_recent10y.csv", type=str)

    # ✅ NEW: base config별 best cap 1개만 모은 파일
    ap.add_argument("--out-top-bestcap", default="data/signals/gate_grid_top_by_recent10y_bestcap.csv", type=str)

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

        if "TAG" not in row.index:
            row["TAG"] = "run"
        if "GateSuffix" not in row.index:
            row["GateSuffix"] = _find_suffix_from_summary_path(sp)

        # ✅ parse cap from suffix
        base_suffix, cap_mode = _parse_cap_from_suffix(str(row["GateSuffix"]))
        row["BaseSuffix"] = base_suffix
        row["CapMode"] = cap_mode

        enriched = enrich_one_summary(row, signals_dir=signals_dir, prices=prices)
        for k, v in enriched.items():
            row[k] = v

        rows.append(row)

    out = pd.DataFrame(rows)

    # ---- robust numeric conversions for scoring columns
    num_cols = [
        "Recent10Y_SeedMultiple", "SeedMultiple",
        "MaxHoldingDaysObserved", "MaxExtendDaysObserved",
        "CycleCount", "SuccessRate",
        "MaxLeveragePct",
        "CAGR_AfterWarmup", "QQQ_CAGR_SamePeriod", "ExcessCAGR_AfterWarmup",
        "IdlePctAfterWarmup",
        "QQQ_SeedMultiple_SamePeriod",

        # summarize_sim_trades.py (existing)
        "TrailEntryCountTotal", "TrailEntryCountPerCycleAvg", "MaxCyclePeakReturn",
        "MaxCycleReturn", "P95CycleReturn", "MedianCycleReturn", "MeanCycleReturn",

        # ✅ summarize_sim_trades.py NEW (cap / tp1 / exits)
        "TP1HoldCapDays", "TotalHoldCapDays",
        "TP1CycleRate", "TP1CycleCount", "TP1FirstDay_Mean", "TP1FirstDay_Median",
        "Exit_TrailAll", "Exit_GraceEnd", "Exit_GraceRecovery", "Exit_RevalFail",
        "Exit_FinalClose", "Exit_TP1H2Cap", "Exit_TP1TotalCap", "Exit_Other",
    ]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # ---- leverage-adjusted score
    if "SeedMultiple" in out.columns and "MaxLeveragePct" in out.columns:
        lev = out["MaxLeveragePct"].fillna(0.0)
        sm = out["SeedMultiple"].astype(float)
        out["SeedMultiple_LevAdj"] = sm / (1.0 + lev)

    # ---- risk-ish score (optional)
    if "SeedMultiple" in out.columns:
        sm = out["SeedMultiple"].astype(float)

        lev = out["MaxLeveragePct"].fillna(0.0) if "MaxLeveragePct" in out.columns else 0.0
        idle = out["IdlePctAfterWarmup"].fillna(0.0) if "IdlePctAfterWarmup" in out.columns else 0.0

        trail = out["TrailEntryCountPerCycleAvg"].fillna(0.0) if "TrailEntryCountPerCycleAvg" in out.columns else 0.0
        peak = out["MaxCyclePeakReturn"].fillna(0.0) if "MaxCyclePeakReturn" in out.columns else 0.0

        pen = (0.25 * lev) + (0.15 * idle) + (0.05 * trail) + (0.03 * np.maximum(0.0, peak - 0.50))
        out["RiskAdjScore"] = sm / (1.0 + pen)

    # ---- sort aggregate
    sort_cols = []
    if "RiskAdjScore" in out.columns:
        sort_cols.append("RiskAdjScore")
    if "SeedMultiple" in out.columns:
        sort_cols.append("SeedMultiple")
    if "SeedMultiple_LevAdj" in out.columns:
        sort_cols.append("SeedMultiple_LevAdj")
    if "CAGR_AfterWarmup" in out.columns:
        sort_cols.append("CAGR_AfterWarmup")
    if "ExcessCAGR_AfterWarmup" in out.columns:
        sort_cols.append("ExcessCAGR_AfterWarmup")

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    out_path = Path(args.out_aggregate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[DONE] wrote aggregate: {out_path} rows={len(out)}")

    # ---- Top-by-recent10y (cap 포함해서 그냥 TopN)
    top = out.copy()
    if "Recent10Y_SeedMultiple" in top.columns:
        top = top.sort_values(["Recent10Y_SeedMultiple"], ascending=[False])
    top = top.head(int(args.topn))
    top_path = Path(args.out_top)
    top.to_csv(top_path, index=False)
    print(f"[DONE] wrote top: {top_path} rows={len(top)}")

    # ---- ✅ BaseSuffix별 “best cap 1개”만 뽑기
    bestcap = out.copy()
    if "Recent10Y_SeedMultiple" in bestcap.columns:
        bestcap = bestcap.sort_values(["BaseSuffix", "Recent10Y_SeedMultiple"], ascending=[True, False])
        bestcap = bestcap.drop_duplicates(subset=["BaseSuffix"], keep="first")
        bestcap = bestcap.sort_values(["Recent10Y_SeedMultiple"], ascending=[False]).head(int(args.topn))

    bestcap_path = Path(args.out_top_bestcap)
    bestcap.to_csv(bestcap_path, index=False)
    print(f"[DONE] wrote top-bestcap: {bestcap_path} rows={len(bestcap)}")

    # ---- quick headline
    if len(out):
        best = out.iloc[0].to_dict()
        print("=" * 60)
        print("[BEST] (by RiskAdjScore / SeedMultiple / LevAdj / CAGR / ExcessCAGR)")
        print(f"TAG={best.get('TAG')} suffix={best.get('GateSuffix')} base={best.get('BaseSuffix')} cap={best.get('CapMode')}")
        print(f"RiskAdjScore={best.get('RiskAdjScore')}")
        print(f"SeedMultiple={best.get('SeedMultiple')}  Recent10Y={best.get('Recent10Y_SeedMultiple')}")
        print(f"CAGR_AfterWarmup={best.get('CAGR_AfterWarmup')}  QQQ_CAGR={best.get('QQQ_CAGR_SamePeriod')}  Excess={best.get('ExcessCAGR_AfterWarmup')}")
        print(f"IdlePctAfterWarmup={best.get('IdlePctAfterWarmup')}  MaxLevPct={best.get('MaxLeveragePct')}")
        print(f"TrailAvg={best.get('TrailEntryCountPerCycleAvg')}  PeakCycleRet={best.get('MaxCyclePeakReturn')}")
        print(
            f"MaxCycleReturn={best.get('MaxCycleReturn')}  "
            f"P95={best.get('P95CycleReturn')}  "
            f"Median={best.get('MedianCycleReturn')}  "
            f"Mean={best.get('MeanCycleReturn')}"
        )
        print(f"TP1Rate={best.get('TP1CycleRate')} TP1FirstDay(med)={best.get('TP1FirstDay_Median')}")
        print(f"Exit_TP1H2Cap={best.get('Exit_TP1H2Cap')} Exit_TP1TotalCap={best.get('Exit_TP1TotalCap')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
