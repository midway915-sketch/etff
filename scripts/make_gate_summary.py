# scripts/make_gate_summary.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


DATA_DIR = Path("data")
SIG_DIR = DATA_DIR / "signals"


def extract_label_from_curve_filename(path: Path) -> str:
    """
    curve 파일명이 예:
      sim_engine_curve_pt10_h40_sl10_ex30_none_t0p20_q0p75_rutility_none_t0p20_q0p75_rutility.parquet
    같은 형태라서 label이 중복으로 들어간 경우가 많음.
    """
    stem = path.stem  # without .parquet

    m = re.match(r"sim_engine_curve_(pt\d+_h\d+_sl\d+_ex\d+)_(.+)$", stem)
    if not m:
        return stem

    rest = m.group(2)

    # rest = "{label}_{label}" 형태면 앞부분만 뽑기
    # 첫 '_' 기준으로 양쪽이 같으면 label로 인정
    for i in range(1, len(rest)):
        if rest[i] == "_" and rest[:i] == rest[i + 1 :]:
            return rest[:i]

    # 아니면 rest 전체를 label로
    return rest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. pt10_h40_sl10_ex30")
    ap.add_argument("--max-days", type=int, required=True, help="e.g. 40")
    ap.add_argument("--recent-years", type=int, default=10)
    ap.add_argument("--out", default="", help="output csv path (default: data/signals/gate_summary_{tag}.csv)")
    args = ap.parse_args()

    tag = args.tag
    max_days = int(args.max_days)
    recent_years = int(args.recent_years)

    summary_path = SIG_DIR / f"sim_engine_summary_{tag}_GATES_ALL.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    summary = pd.read_csv(summary_path)

    # curve 파일들 로드해서 recent10y / max_extend / max_holding 계산
    curve_files = sorted(SIG_DIR.glob(f"sim_engine_curve_{tag}_*.parquet"))
    if not curve_files:
        raise FileNotFoundError(f"No curve parquet files found for tag={tag} in {SIG_DIR}")

    rows = []
    for fp in curve_files:
        label = extract_label_from_curve_filename(fp)

        cdf = pd.read_parquet(fp)
        if "Date" not in cdf.columns or "Equity" not in cdf.columns:
            continue

        cdf["Date"] = pd.to_datetime(cdf["Date"])
        cdf = cdf.sort_values("Date").reset_index(drop=True)

        last_date = cdf["Date"].iloc[-1]
        end_equity = float(cdf["Equity"].iloc[-1])

        start_date = last_date - DateOffset(years=recent_years)
        if (cdf["Date"] >= start_date).any():
            start_row = cdf.loc[cdf["Date"] >= start_date].iloc[0]
        else:
            start_row = cdf.iloc[0]

        start_equity = float(start_row["Equity"])
        recent_mult = end_equity / start_equity if start_equity != 0 else np.nan

        max_hold = int(cdf["HoldingDays"].max()) if "HoldingDays" in cdf.columns else np.nan
        max_extend = int(max(0, max_hold - max_days)) if isinstance(max_hold, (int, np.integer)) else np.nan

        rows.append(
            {
                "label": label,
                "last_date": last_date.date().isoformat(),
                "recent_start_date": pd.to_datetime(start_row["Date"]).date().isoformat(),
                "recent_seed_multiple": recent_mult,
                "max_holding_days": max_hold,
                "max_extend_days_over_maxday": max_extend,
            }
        )

    curve_metrics = pd.DataFrame(rows)

    # merge
    out_df = summary.merge(curve_metrics, on="label", how="left")

    # 보기 좋은 컬럼명 정리(너 summary 컬럼명에 맞춰서 매핑)
    rename_map = {
        "seed_multiple": "Seed_Multiple_All",
        "final_equity": "Final_Equity",
        "max_drawdown": "Max_Drawdown",
        "closed_trades_count": "Cycle_Count_Closed",
        "win_rate_closed": "Success_Rate_Closed",
        "max_leverage_pct_closed": "Max_LeveragePct_Closed",
        "avg_leverage_pct_closed": "Avg_LeveragePct_Closed",
        "skipped_days": "Skipped_Days",
        "entered_days": "Entered_Days",
    }
    for k, v in rename_map.items():
        if k in out_df.columns:
            out_df = out_df.rename(columns={k: v})

    # 최종 컬럼 구성(없으면 자동으로 제외)
    preferred_cols = [
        "label",
        "tag",
        "Seed_Multiple_All",
        "recent_seed_multiple",
        "max_extend_days_over_maxday",
        "max_holding_days",
        "Cycle_Count_Closed",
        "Success_Rate_Closed",
        "Max_Drawdown",
        "Avg_LeveragePct_Closed",
        "Max_LeveragePct_Closed",
        "Skipped_Days",
        "Entered_Days",
        "last_date",
        "recent_start_date",
        "Final_Equity",
    ]
    cols = [c for c in preferred_cols if c in out_df.columns]
    out_df = out_df[cols].sort_values(cols[2] if len(cols) > 2 else cols[0], ascending=False)

    out_path = Path(args.out) if args.out else (SIG_DIR / f"gate_summary_{tag}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[DONE] wrote {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
