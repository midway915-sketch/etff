# scripts/aggregate_gate_grid.py
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np


def _read_csv_safe(p: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(p)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        return df
    except Exception as e:
        print(f"[WARN] failed to read {p}: {e}")
        return None


def _safe_float(x, default=np.nan) -> float:
    try:
        v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
        return float(v) if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def leverage_adjust(seed_mult: float, max_lev_pct: float, lam: float = 0.5) -> float:
    """
    교정된 수익(배수) 예시:
    - seed_mult: 시드 배수 (예: 3.2)
    - max_lev_pct: 최대 레버리지 비율 (예: 0.8 => 최대 대출이 entry_seed의 80%)
    - lam: 페널티 강도 (0~1 정도 권장)
    공식은 "원금 1배 초과분"에 대해서만 페널티를 줌:
      adj = 1 + (seed_mult - 1) / (1 + lam * max_lev_pct)

    직관:
    - seed_mult가 같으면 레버가 낮을수록 adj가 커짐
    - max_lev_pct=0이면 adj=seed_mult
    """
    if not np.isfinite(seed_mult):
        return np.nan
    base = float(seed_mult)
    if base <= 0:
        return base
    lev = max(0.0, float(max_lev_pct)) if np.isfinite(max_lev_pct) else 0.0
    lam = float(lam) if np.isfinite(lam) else 0.5
    excess = base - 1.0
    return float(1.0 + excess / (1.0 + lam * lev))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate gate_summary_*.csv and compute leverage-adjusted metrics.")
    ap.add_argument("--signals-dir", default="data/signals", type=str)
    ap.add_argument("--pattern", default="gate_summary_*.csv", type=str)

    ap.add_argument("--out-aggregate", default="data/signals/gate_grid_aggregate.csv", type=str)
    ap.add_argument("--out-top", default="data/signals/gate_grid_top_by_recent10y.csv", type=str)

    ap.add_argument("--topn", default=30, type=int)

    # leverage penalty strength (this is only for ranking; your hard cap is enforced by engine)
    ap.add_argument("--lambda-lev", default=0.5, type=float)

    args = ap.parse_args()

    sig_dir = Path(args.signals_dir)
    sig_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(str(sig_dir / args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No summary files matched: {sig_dir / args.pattern}")

    rows = []
    for p in paths:
        df = _read_csv_safe(p)
        if df is None:
            continue
        # gate_summary is 1-row csv
        r = df.iloc[0].to_dict()
        r["_summary_path"] = str(p)
        rows.append(r)

    if not rows:
        raise RuntimeError("No readable summary rows.")

    out = pd.DataFrame(rows)

    # Normalize important columns
    for col in [
        "SeedMultiple",
        "Recent10Y_SeedMultiple",
        "MaxLeveragePct",
        "MaxHoldingDaysObserved",
        "MaxExtendDaysObserved",
        "CycleCount",
        "SuccessRate",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Backward compatible column name variants (if your summaries used different names before)
    if "MaxLeveragePct" not in out.columns and "Max_LeveragePct_Closed" in out.columns:
        out["MaxLeveragePct"] = pd.to_numeric(out["Max_LeveragePct_Closed"], errors="coerce")

    if "MaxExtendDaysObserved" not in out.columns and "Max_Extend_Over_MaxDays" in out.columns:
        out["MaxExtendDaysObserved"] = pd.to_numeric(out["Max_Extend_Over_MaxDays"], errors="coerce")

    # Safety defaults
    if "MaxLeveragePct" not in out.columns:
        out["MaxLeveragePct"] = 0.0
    out["MaxLeveragePct"] = out["MaxLeveragePct"].fillna(0.0).clip(lower=0.0)

    if "SuccessRate" in out.columns:
        out["SuccessRate"] = out["SuccessRate"].fillna(0.0).clip(lower=0.0, upper=1.0)
    else:
        out["SuccessRate"] = 0.0

    if "CycleCount" in out.columns:
        out["CycleCount"] = out["CycleCount"].fillna(0).astype(int)
    else:
        out["CycleCount"] = 0

    # Compute leverage-adjusted seed multiples
    lam_lev = float(args.lambda_lev)

    out["Adj_SeedMultiple"] = [
        leverage_adjust(_safe_float(sm), _safe_float(lv, 0.0), lam=lam_lev)
        for sm, lv in zip(out.get("SeedMultiple", pd.Series([np.nan]*len(out))), out["MaxLeveragePct"])
    ]

    out["Adj_Recent10Y_SeedMultiple"] = [
        leverage_adjust(_safe_float(sm), _safe_float(lv, 0.0), lam=lam_lev)
        for sm, lv in zip(out.get("Recent10Y_SeedMultiple", pd.Series([np.nan]*len(out))), out["MaxLeveragePct"])
    ]

    # A simple composite score for ranking (you can change this anytime)
    # Priority: Adj_Recent10Y > Adj_AllTime, then leverage lower, then success higher
    # (NaN-safe)
    out["Score"] = (
        pd.to_numeric(out["Adj_Recent10Y_SeedMultiple"], errors="coerce").fillna(0.0) * 1.0
        + pd.to_numeric(out["Adj_SeedMultiple"], errors="coerce").fillna(0.0) * 0.3
        - out["MaxLeveragePct"].fillna(0.0) * 0.10
        + out["SuccessRate"].fillna(0.0) * 0.05
    )

    # Sort (best first)
    out_sorted = out.sort_values(
        by=["Adj_Recent10Y_SeedMultiple", "Adj_SeedMultiple", "MaxLeveragePct", "SuccessRate", "CycleCount"],
        ascending=[False, False, True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    out_agg_path = Path(args.out_aggregate)
    out_agg_path.parent.mkdir(parents=True, exist_ok=True)
    out_sorted.to_csv(out_agg_path, index=False)
    print(f"[DONE] wrote aggregate: {out_agg_path} rows={len(out_sorted)}")

    # TopN view by recent10y (adjusted)
    topn = int(args.topn)
    top = out_sorted.head(topn).copy()
    out_top_path = Path(args.out_top)
    out_top_path.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_top_path, index=False)
    print(f"[DONE] wrote top: {out_top_path} rows={len(top)}")

    # Quick console peek
    cols = [
        "TAG", "GateSuffix",
        "Recent10Y_SeedMultiple", "Adj_Recent10Y_SeedMultiple",
        "SeedMultiple", "Adj_SeedMultiple",
        "MaxLeveragePct", "MaxExtendDaysObserved", "CycleCount", "SuccessRate"
    ]
    cols = [c for c in cols if c in out_sorted.columns]
    print("\n[TOP 10]")
    print(out_sorted[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()