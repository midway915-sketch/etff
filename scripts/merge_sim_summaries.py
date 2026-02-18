from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SIGNALS_DIR = Path("data/signals")


def read_parquet_any(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def infer_variant_from_path(path: str) -> str:
    p = path.upper()
    if "_BASE" in p:
        return "BASE"
    if "_A" in p:
        return "A"
    if "_B" in p:
        return "B"
    return "BASE"


def summarize_one_method(
    tag: str,
    method: str,
    variant: str,
    curve: pd.DataFrame | None,
    trades: pd.DataFrame | None,
    max_days: int,
    recent_years: int,
) -> dict:
    out: dict = {"tag": tag, "label": method, "variant": variant}

    # ---- trades 기반
    if trades is not None and not trades.empty:
        t = trades.copy()
        t = t[t["Method"].astype(str) == str(method)]
        if "Variant" in t.columns:
            t = t[t["Variant"].astype(str) == str(variant)]
        t = t.sort_values("EntryDate")

        cycle_cnt = int(len(t))
        out["Cycle_Count_Closed"] = cycle_cnt
        out["Success_Rate_Closed"] = float((t["CycleReturn"] > 0).mean()) if cycle_cnt > 0 else 0.0

        if "LeveragePctMax" in t.columns and cycle_cnt > 0:
            out["Max_LeveragePct_Closed"] = float(pd.to_numeric(t["LeveragePctMax"], errors="coerce").max())
            out["Avg_LeveragePct_Closed"] = float(pd.to_numeric(t["LeveragePctMax"], errors="coerce").mean())
        else:
            out["Max_LeveragePct_Closed"] = float("nan")
            out["Avg_LeveragePct_Closed"] = float("nan")

        # max holding (닫힌 사이클 기준)
        max_hold = 0
        if "HoldingDays" in t.columns and cycle_cnt > 0:
            hd = pd.to_numeric(t["HoldingDays"], errors="coerce").max()
            if pd.notna(hd):
                max_hold = int(hd)
        out["Max_HoldingDays_Closed"] = int(max_hold)

    else:
        out["Cycle_Count_Closed"] = 0
        out["Success_Rate_Closed"] = 0.0
        out["Max_LeveragePct_Closed"] = float("nan")
        out["Avg_LeveragePct_Closed"] = float("nan")
        out["Max_HoldingDays_Closed"] = 0

    # ---- curve 기반
    if curve is not None and not curve.empty:
        c = curve.copy()
        c = c[c["Method"].astype(str) == str(method)]
        if "Variant" in c.columns:
            c = c[c["Variant"].astype(str) == str(variant)]
        c = c.sort_values("Date")

        last_date = pd.to_datetime(c["Date"].iloc[-1]).normalize()
        out["last_date"] = str(last_date.date())

        eq = pd.to_numeric(c["Equity"], errors="coerce").fillna(method="ffill").fillna(0).to_numpy(dtype=float)
        start_eq = float(eq[0]) if eq.size else float("nan")
        final_eq = float(eq[-1]) if eq.size else float("nan")
        out["Final_Equity"] = final_eq
        out["Seed_Multiple_All"] = float(final_eq / start_eq) if start_eq not in (0.0, float("nan")) else float("nan")
        out["Max_Drawdown"] = max_drawdown(eq)

        # max holding days (curve에 HoldingDays 있으면 사용, 없으면 trades closed로 fallback)
        max_hold_curve = 0
        if "HoldingDays" in c.columns:
            hd = pd.to_numeric(c["HoldingDays"], errors="coerce").max()
            if pd.notna(hd):
                max_hold_curve = int(hd)

        if max_hold_curve == 0:
            out["Max_HoldingDays_All"] = int(out.get("Max_HoldingDays_Closed", 0))
        else:
            out["Max_HoldingDays_All"] = int(max_hold_curve)

        # extend 최대치(= HoldingDays - max_days)
        if max_days > 0 and "HoldingDays" in c.columns:
            hd_arr = pd.to_numeric(c["HoldingDays"], errors="coerce").fillna(0).to_numpy(dtype=float)
            exceed = np.maximum(hd_arr - max_days, 0)
            out["Max_Extend_Over_MaxDays"] = int(np.nanmax(exceed)) if exceed.size else 0
        else:
            out["Max_Extend_Over_MaxDays"] = 0

        # 최근 N년 배수
        start_date = last_date - pd.DateOffset(years=recent_years)
        out["recent_start_date"] = str(start_date.date())

        dates = pd.to_datetime(c["Date"]).to_numpy()
        idx = int(np.searchsorted(dates, np.datetime64(start_date)))
        if idx >= len(c):
            idx = 0
        base_eq = float(pd.to_numeric(c["Equity"].iloc[idx], errors="coerce"))
        out["recent_seed_multiple"] = float(final_eq / base_eq) if base_eq else float("nan")

    else:
        out["last_date"] = ""
        out["recent_start_date"] = ""
        out["Final_Equity"] = float("nan")
        out["Seed_Multiple_All"] = float("nan")
        out["Max_Drawdown"] = float("nan")
        out["Max_HoldingDays_All"] = int(out.get("Max_HoldingDays_Closed", 0))
        out["Max_Extend_Over_MaxDays"] = 0
        out["recent_seed_multiple"] = float("nan")

    # 레버 교정 지표(너가 원한 “수익률 우선 + 레버 고려”)
    lev = out.get("Max_LeveragePct_Closed", float("nan"))
    mult = out.get("recent_seed_multiple", float("nan"))

    if isinstance(lev, (int, float)) and isinstance(mult, (int, float)) and np.isfinite(lev) and np.isfinite(mult):
        out["adj_recent_multiple_linear"] = float(mult / (1.0 + lev / 100.0))
        out["adj_recent_multiple_sq"] = float(mult / (1.0 + lev / 100.0) ** 2)
    else:
        out["adj_recent_multiple_linear"] = float("nan")
        out["adj_recent_multiple_sq"] = float("nan")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--recent-years", type=int, default=10)
    args = ap.parse_args()

    tag = args.tag
    max_days = int(args.max_days)
    recent_years = int(args.recent_years)

    curve_paths = sorted(glob.glob(f"{SIGNALS_DIR}/sim_engine_curve_*{tag}*.parquet"))
    trade_paths = sorted(glob.glob(f"{SIGNALS_DIR}/sim_engine_trades_*{tag}*.parquet"))

    if not curve_paths and not trade_paths:
        raise SystemExit(f"[ERROR] No curve/trades parquet found for tag={tag} in {SIGNALS_DIR}")

    curve_df = pd.DataFrame()
    trades_df = pd.DataFrame()

    if curve_paths:
        curve_df = pd.concat([read_parquet_any(p) for p in curve_paths], ignore_index=True)
        if "Variant" not in curve_df.columns:
            # 파일명에서 variant 추정해 넣기(혹시 엔진이 컬럼 안 넣었을 때 대비)
            # 단, 여러 파일 concat이므로 path별 처리 위해 최소한 method별/파일별로는 정확하지 않을 수 있음.
            curve_df["Variant"] = "BASE"

    if trade_paths:
        trades_df = pd.concat([read_parquet_any(p) for p in trade_paths], ignore_index=True)
        if "Variant" not in trades_df.columns:
            trades_df["Variant"] = "BASE"

    if "Method" not in curve_df.columns and "Method" not in trades_df.columns:
        raise SystemExit(f"[ERROR] No Method column found in curve/trades parquet for tag={tag}")

    methods = set()
    if not curve_df.empty and "Method" in curve_df.columns:
        methods |= set(curve_df["Method"].dropna().astype(str).unique().tolist())
    if not trades_df.empty and "Method" in trades_df.columns:
        methods |= set(trades_df["Method"].dropna().astype(str).unique().tolist())

    # variants는 parquet 컬럼이 있으면 그 기준, 없으면 BASE
    variants = set()
    if not curve_df.empty and "Variant" in curve_df.columns:
        variants |= set(curve_df["Variant"].dropna().astype(str).unique().tolist())
    if not trades_df.empty and "Variant" in trades_df.columns:
        variants |= set(trades_df["Variant"].dropna().astype(str).unique().tolist())
    if not variants:
        variants = {"BASE"}

    rows = []
    for m in sorted(methods):
        for v in sorted(variants):
            rows.append(
                summarize_one_method(
                    tag=tag,
                    method=m,
                    variant=v,
                    curve=curve_df if not curve_df.empty else None,
                    trades=trades_df if not trades_df.empty else None,
                    max_days=max_days,
                    recent_years=recent_years,
                )
            )

    out = pd.DataFrame(rows)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    gate_path = SIGNALS_DIR / f"gate_summary_{tag}.csv"
    out.to_csv(gate_path, index=False)

    print(f"[DONE] wrote {gate_path} rows={len(out)}")


if __name__ == "__main__":
    main()