# scripts/summarize_sim_trades.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce").dt.tz_localize(None)


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _infer_curve_from_trades(trades_path: Path) -> Path:
    name = trades_path.name.replace("sim_engine_trades_", "sim_engine_curve_")
    return trades_path.with_name(name)


def _safe_num(s: pd.Series, default=np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _seed_multiple_from_curve(curve: pd.DataFrame) -> float | None:
    if curve is None or curve.empty:
        return None

    for c in ["SeedMultiple", "seed_multiple"]:
        if c in curve.columns:
            v = _safe_num(curve[c]).dropna()
            if len(v):
                return float(v.iloc[-1])

    if "Equity" in curve.columns:
        eq = _safe_num(curve["Equity"]).dropna()
        if len(eq) >= 2 and float(eq.iloc[0]) != 0:
            return float(eq.iloc[-1] / eq.iloc[0])

    return None


def _recent10y_seed_multiple_from_curve(curve: pd.DataFrame) -> float | None:
    if curve is None or curve.empty or "Date" not in curve.columns:
        return None

    d = _to_dt(curve["Date"])
    if d.isna().all():
        return None

    last = d.max()
    start = last - pd.Timedelta(days=365 * 10)
    sub = curve.loc[d >= start].copy()
    if sub.empty:
        return None

    for c in ["SeedMultiple", "seed_multiple"]:
        if c in sub.columns:
            v = _safe_num(sub[c]).dropna()
            if len(v):
                first = float(v.iloc[0])
                lastv = float(v.iloc[-1])
                if first != 0:
                    return float(lastv / first)
                return float(lastv)

    if "Equity" in sub.columns:
        eq = _safe_num(sub["Equity"]).dropna()
        if len(eq) >= 2 and float(eq.iloc[0]) != 0:
            return float(eq.iloc[-1] / eq.iloc[0])

    return None


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _quantile_safe(x: pd.Series, q: float) -> float:
    x = _safe_num(x).dropna()
    if x.empty:
        return float("nan")
    try:
        return float(x.quantile(q))
    except Exception:
        return float("nan")


def _reason_counts(trades: pd.DataFrame) -> dict[str, int]:
    if trades is None or trades.empty or "Reason" not in trades.columns:
        return {
            "Exit_TrailAll": 0,
            "Exit_GraceEnd": 0,
            "Exit_GraceRecovery": 0,
            "Exit_RevalFail": 0,
            "Exit_FinalClose": 0,
            "Exit_TP1H2Cap": 0,
            "Exit_TP1TotalCap": 0,
            "Exit_Other": 0,
        }

    r = trades["Reason"].astype(str).fillna("")

    def _cnt(prefix: str) -> int:
        return int((r.str.startswith(prefix)).sum())

    c_trail = int((r == "TRAIL_EXIT_ALL").sum())
    c_grace_end = int((r == "GRACE_END_EXIT").sum())
    c_grace_rec = int((r == "GRACE_RECOVERY_EXIT").sum())
    c_reval_fail = _cnt("REVAL_FAIL")
    c_final = int((r == "FINAL_CLOSE").sum())
    c_h2 = int((r == "TP1_H2_CAP_EXIT").sum())
    c_total = int((r == "TP1_TOTAL_CAP_EXIT").sum())

    known = c_trail + c_grace_end + c_grace_rec + c_reval_fail + c_final + c_h2 + c_total
    other = int(max(0, len(trades) - known))

    return {
        "Exit_TrailAll": c_trail,
        "Exit_GraceEnd": c_grace_end,
        "Exit_GraceRecovery": c_grace_rec,
        "Exit_RevalFail": c_reval_fail,
        "Exit_FinalClose": c_final,
        "Exit_TP1H2Cap": c_h2,
        "Exit_TP1TotalCap": c_total,
        "Exit_Other": other,
    }


def _tp1_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "TP1CycleRate": 0.0,
            "TP1CycleCount": 0,
            "TP1FirstDay_Mean": np.nan,
            "TP1FirstDay_Median": np.nan,
        }

    col = _pick_first_existing_col(trades, ["TP1FirstHoldingDay", "tp1_first_holding_day"])
    if not col:
        return {
            "TP1CycleRate": 0.0,
            "TP1CycleCount": 0,
            "TP1FirstDay_Mean": np.nan,
            "TP1FirstDay_Median": np.nan,
        }

    v = _safe_num(trades[col], 0.0).fillna(0.0)
    hit = (v > 0).astype(int)
    cnt = int(hit.sum())
    rate = float(cnt / len(trades)) if len(trades) else 0.0

    vv = v[v > 0]
    mean_day = float(vv.mean()) if len(vv) else np.nan
    med_day = float(vv.median()) if len(vv) else np.nan

    return {
        "TP1CycleRate": rate,
        "TP1CycleCount": cnt,
        "TP1FirstDay_Mean": mean_day,
        "TP1FirstDay_Median": med_day,
    }


def _cap_meta(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "TP1HoldCapMode": "",
            "TP1HoldCapDays": np.nan,
            "TotalHoldCapDays": np.nan,
        }

    mode_col = _pick_first_existing_col(trades, ["TP1HoldCapMode"])
    tp1_days_col = _pick_first_existing_col(trades, ["TP1HoldCapDays"])
    total_days_col = _pick_first_existing_col(trades, ["TotalHoldCapDays"])

    mode = ""
    if mode_col:
        m = trades[mode_col].astype(str).replace("nan", "").fillna("")
        mode = str(m.iloc[0]) if len(m) else ""

    tp1_days = np.nan
    if tp1_days_col:
        x = _safe_num(trades[tp1_days_col]).dropna()
        tp1_days = float(x.iloc[0]) if len(x) else np.nan

    total_days = np.nan
    if total_days_col:
        x = _safe_num(trades[total_days_col]).dropna()
        total_days = float(x.iloc[0]) if len(x) else np.nan

    return {
        "TP1HoldCapMode": mode,
        "TP1HoldCapDays": tp1_days,
        "TotalHoldCapDays": total_days,
    }


def _cycle_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "CycleCount": 0,
            "SuccessRate": 0.0,
            "MaxHoldingDaysObserved": np.nan,
            "MaxLeveragePct": np.nan,
            "TrailEntryCountTotal": 0,
            "TrailEntryCountPerCycleAvg": 0.0,
            "MaxCyclePeakReturn": np.nan,
            "MaxCycleReturn": np.nan,
            "P95CycleReturn": np.nan,
            "MedianCycleReturn": np.nan,
            "MeanCycleReturn": np.nan,
        }

    cycle_cnt = int(len(trades))

    if "CycleReturn" in trades.columns:
        cr = _safe_num(trades["CycleReturn"]).replace([np.inf, -np.inf], np.nan)
    else:
        cr = pd.Series([np.nan] * cycle_cnt, index=trades.index, dtype=float)

    if "Win" in trades.columns:
        wins = (_safe_num(trades["Win"], 0.0) > 0).astype(int)
    elif "CycleReturn" in trades.columns:
        wins = (_safe_num(trades["CycleReturn"], 0.0) > 0).astype(int)
    else:
        wins = pd.Series([0] * cycle_cnt, index=trades.index, dtype=int)

    success_rate = float(wins.sum() / cycle_cnt) if cycle_cnt > 0 else 0.0

    max_hold_obs = np.nan
    if "HoldingDays" in trades.columns:
        mh = _safe_num(trades["HoldingDays"]).max()
        max_hold_obs = float(mh) if np.isfinite(mh) else np.nan

    max_lev = np.nan
    lev_col = _pick_first_existing_col(trades, ["MaxLeveragePct", "max_leverage_pct", "LeveragePct", "leverage_pct"])
    if lev_col:
        mv = _safe_num(trades[lev_col]).max()
        max_lev = float(mv) if np.isfinite(mv) else np.nan

    trail_col = _pick_first_existing_col(trades, ["TrailingEntries", "TrailEntryCount"])
    trail_total = 0
    trail_avg = 0.0
    if trail_col:
        te = _safe_num(trades[trail_col], 0.0).fillna(0).astype(int)
        trail_total = int(te.sum())
        trail_avg = float(te.mean()) if cycle_cnt > 0 else 0.0

    peak_col = _pick_first_existing_col(trades, ["CycleMaxReturn", "PeakCycleReturn"])
    max_peak_ret = np.nan
    if peak_col:
        mpr = _safe_num(trades[peak_col]).max()
        max_peak_ret = float(mpr) if np.isfinite(mpr) else np.nan

    max_cycle_ret = float(_safe_num(cr).max()) if _safe_num(cr).notna().any() else np.nan
    p95_cycle_ret = _quantile_safe(cr, 0.95)
    med_cycle_ret = _quantile_safe(cr, 0.50)
    mean_cycle_ret = float(_safe_num(cr).dropna().mean()) if _safe_num(cr).dropna().shape[0] else np.nan

    return {
        "CycleCount": cycle_cnt,
        "SuccessRate": float(success_rate),
        "MaxHoldingDaysObserved": max_hold_obs,
        "MaxLeveragePct": max_lev,
        "TrailEntryCountTotal": int(trail_total),
        "TrailEntryCountPerCycleAvg": float(trail_avg),
        "MaxCyclePeakReturn": max_peak_ret,
        "MaxCycleReturn": max_cycle_ret,
        "P95CycleReturn": p95_cycle_ret,
        "MedianCycleReturn": med_cycle_ret,
        "MeanCycleReturn": mean_cycle_ret,
    }


def _infer_tau_split(tag: str) -> str:
    """
    1) env TAU_SPLIT 있으면 그걸 우선 사용
    2) 없으면 TAG에서 패턴으로 추정
    """
    env_v = (os.environ.get("TAU_SPLIT", "") or "").strip()
    if env_v:
        return env_v

    t = (tag or "").lower()
    if "_taucur" in t or "_tau_cur" in t:
        return "cur"
    if "q255025" in t or "_tauq" in t:
        return "q255025"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-path", required=True, type=str)
    ap.add_argument("--curve-path", default="", type=str)
    ap.add_argument("--tag", required=True, type=str)
    ap.add_argument("--suffix", required=True, type=str)
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)
    ap.add_argument("--out-dir", default="data/signals", type=str)
    args = ap.parse_args()

    trades_path = Path(args.trades_path)
    trades = _read_any(trades_path)

    if args.curve_path:
        curve_path = Path(args.curve_path)
    else:
        curve_path = _infer_curve_from_trades(trades_path)

    curve = None
    if curve_path.exists():
        curve = _read_any(curve_path)

    st = _cycle_stats(trades)
    rc = _reason_counts(trades)
    tp1 = _tp1_stats(trades)
    capm = _cap_meta(trades)

    seed_mult = _seed_multiple_from_curve(curve) if isinstance(curve, pd.DataFrame) else None
    recent10y = _recent10y_seed_multiple_from_curve(curve) if isinstance(curve, pd.DataFrame) else None

    max_extend_obs = np.nan
    if np.isfinite(st["MaxHoldingDaysObserved"]):
        max_extend_obs = float(max(0.0, float(st["MaxHoldingDaysObserved"]) - float(args.max_days)))

    tau_split = _infer_tau_split(args.tag)

    out = {
        "TAG": args.tag,
        "TauSplit": tau_split,  # ✅ NEW: aggregate에서 필터/그룹 가능
        "GateSuffix": args.suffix,
        "ProfitTarget": float(args.profit_target),
        "MaxHoldingDays": int(args.max_days),
        "StopLevel": float(args.stop_level),
        "MaxExtendDaysParam": int(args.max_extend_days),

        "TP1HoldCapMode": capm["TP1HoldCapMode"],
        "TP1HoldCapDays": capm["TP1HoldCapDays"],
        "TotalHoldCapDays": capm["TotalHoldCapDays"],

        "SeedMultiple": seed_mult if seed_mult is not None else np.nan,
        "Recent10Y_SeedMultiple": recent10y if recent10y is not None else np.nan,

        "MaxHoldingDaysObserved": st["MaxHoldingDaysObserved"],
        "MaxExtendDaysObserved": max_extend_obs,

        "CycleCount": int(st["CycleCount"]),
        "SuccessRate": float(st["SuccessRate"]),
        "MaxLeveragePct": st["MaxLeveragePct"],

        "TrailEntryCountTotal": int(st["TrailEntryCountTotal"]),
        "TrailEntryCountPerCycleAvg": float(st["TrailEntryCountPerCycleAvg"]),
        "MaxCyclePeakReturn": st["MaxCyclePeakReturn"],

        "TP1CycleRate": tp1["TP1CycleRate"],
        "TP1CycleCount": int(tp1["TP1CycleCount"]),
        "TP1FirstDay_Mean": tp1["TP1FirstDay_Mean"],
        "TP1FirstDay_Median": tp1["TP1FirstDay_Median"],

        **rc,

        "MaxCycleReturn": st["MaxCycleReturn"],
        "P95CycleReturn": st["P95CycleReturn"],
        "MedianCycleReturn": st["MedianCycleReturn"],
        "MeanCycleReturn": st["MeanCycleReturn"],

        "TradesFile": str(trades_path),
        "CurveFile": str(curve_path) if curve_path.exists() else "",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gate_summary_{args.tag}_gate_{args.suffix}.csv"
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"[DONE] wrote gate summary: {out_path}")


if __name__ == "__main__":
    main()