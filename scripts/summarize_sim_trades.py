from __future__ import annotations

import argparse
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
    return pd.to_numeric(s, errors="coerce").fillna(default)


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


def _cycle_stats(trades: pd.DataFrame) -> tuple[int, float, float | None, float | None]:
    if trades is None or trades.empty:
        return 0, 0.0, None, None

    cycle_cnt = int(len(trades))

    # success
    if "Win" in trades.columns:
        wins = (_safe_num(trades["Win"], 0.0) > 0).astype(int)
    elif "CycleReturn" in trades.columns:
        wins = (_safe_num(trades["CycleReturn"], 0.0) > 0).astype(int)
    else:
        wins = pd.Series([0] * cycle_cnt)

    success_rate = float(wins.sum() / cycle_cnt) if cycle_cnt > 0 else 0.0

    # max holding observed
    max_hold = None
    if "HoldingDays" in trades.columns:
        mh = _safe_num(trades["HoldingDays"]).max()
        max_hold = float(mh) if np.isfinite(mh) else None

    # max leverage pct (cycle max among cycles)
    max_lev = None
    for c in ["MaxLeveragePct", "max_leverage_pct", "LeveragePct", "leverage_pct"]:
        if c in trades.columns:
            mv = _safe_num(trades[c]).max()
            max_lev = float(mv) if np.isfinite(mv) else None
            break

    return cycle_cnt, success_rate, max_hold, max_lev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-path", required=True, type=str)
    ap.add_argument("--curve-path", default="", type=str)  # optional
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

    cycle_cnt, success_rate, max_hold_obs, max_lev = _cycle_stats(trades)

    seed_mult = _seed_multiple_from_curve(curve) if isinstance(curve, pd.DataFrame) else None
    recent10y = _recent10y_seed_multiple_from_curve(curve) if isinstance(curve, pd.DataFrame) else None

    max_extend_obs = None
    if max_hold_obs is not None:
        max_extend_obs = float(max(0.0, max_hold_obs - float(args.max_days)))

    out = {
        "TAG": args.tag,
        "GateSuffix": args.suffix,
        "ProfitTarget": float(args.profit_target),
        "MaxHoldingDays": int(args.max_days),
        "StopLevel": float(args.stop_level),
        "MaxExtendDaysParam": int(args.max_extend_days),

        "SeedMultiple": seed_mult if seed_mult is not None else np.nan,
        "Recent10Y_SeedMultiple": recent10y if recent10y is not None else np.nan,

        "MaxHoldingDaysObserved": max_hold_obs if max_hold_obs is not None else np.nan,
        "MaxExtendDaysObserved": max_extend_obs if max_extend_obs is not None else np.nan,

        "CycleCount": int(cycle_cnt),
        "SuccessRate": float(success_rate),
        "MaxLeveragePct": max_lev if max_lev is not None else np.nan,

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