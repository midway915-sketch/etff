# scripts/analyze_sub_vs_main.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file: {p} (use .parquet or .csv)")


def norm_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def ensure_cycle_return(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)

    # if older file missing CycleType, assume MAIN
    if "CycleType" not in cols:
        df["CycleType"] = "MAIN"

    # If CycleReturn exists -> ok
    if "CycleReturn" in cols:
        df["CycleReturn"] = df["CycleReturn"].apply(safe_float)
        return df

    # If we can compute from Invested/Proceeds -> do it
    if ("Invested" in cols) and ("Proceeds" in cols):
        inv = df["Invested"].apply(safe_float)
        pro = df["Proceeds"].apply(safe_float)
        df["CycleReturn"] = np.where(inv > 0, (pro - inv) / inv, np.nan)
        return df

    # Otherwise, cannot analyze
    raise ValueError(
        "Missing CycleReturn (and cannot derive it). "
        f"cols={list(df.columns)[:80]}"
    )


def agg_stats(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"Group": label, "N": 0}

    r = df["CycleReturn"].astype(float)
    win = (r > 0).astype(int)

    mean_r = float(r.mean())
    med_r = float(r.median())
    std_r = float(r.std(ddof=1)) if len(r) > 1 else 0.0

    win_rate = float(win.mean())
    avg_win = float(r[r > 0].mean()) if (r > 0).any() else np.nan
    avg_loss = float(r[r <= 0].mean()) if (r <= 0).any() else np.nan
    profit_factor = (
        float(r[r > 0].sum() / abs(r[r <= 0].sum()))
        if (r > 0).any() and (r <= 0).any() and abs(r[r <= 0].sum()) > 1e-12
        else np.nan
    )

    gross = (1.0 + r).clip(lower=1e-12)
    geom_total = float(gross.prod())
    geom_mean = float(np.exp(np.log(gross).mean()))

    hd_col = "HoldingDays" if "HoldingDays" in df.columns else None
    hd_mean = float(df[hd_col].mean()) if hd_col else np.nan
    hd_med = float(df[hd_col].median()) if hd_col else np.nan

    inv_mean = float(df["Invested"].astype(float).mean()) if "Invested" in df.columns else np.nan

    return {
        "Group": label,
        "N": int(len(df)),
        "WinRate": win_rate,
        "MeanReturn": mean_r,
        "MedianReturn": med_r,
        "StdReturn": std_r,
        "AvgWin": avg_win,
        "AvgLoss": avg_loss,
        "ProfitFactor": profit_factor,
        "GeomTotalMultiple": geom_total,
        "GeomMeanGrossPerCycle": geom_mean,
        "MeanHoldingDays": hd_mean,
        "MedianHoldingDays": hd_med,
        "MeanInvested": inv_mean,
    }


def make_period_table(df: pd.DataFrame, period: str) -> pd.DataFrame:
    x = df.copy()
    x["EntryDate"] = norm_dt(x["EntryDate"]) if "EntryDate" in x.columns else pd.NaT
    x["Period"] = pd.to_datetime(x["EntryDate"], errors="coerce").dt.to_period(period).astype(str)

    out = []
    for (p, ctype), g in x.groupby(["Period", "CycleType"], dropna=False):
        out.append({**{"Period": p, "CycleType": ctype}, **agg_stats(g, label="")})
    per = pd.DataFrame(out)
    if not per.empty:
        per = per.drop(columns=["Group"], errors="ignore")
        per = per.sort_values(["Period", "CycleType"]).reset_index(drop=True)

    out2 = []
    for p, g in x.groupby(["Period"], dropna=False):
        d = agg_stats(g, label="")
        d["Period"] = p
        d["CycleType"] = "ALL"
        out2.append(d)
    per_all = pd.DataFrame(out2)
    if not per_all.empty:
        per_all = per_all.drop(columns=["Group"], errors="ignore")
        per_all = per_all.sort_values(["Period"]).reset_index(drop=True)

    return pd.concat([per_all, per], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze MAIN vs SUB performance from sim_engine_trades parquet/csv.")
    ap.add_argument("--trades-path", required=True, type=str)
    ap.add_argument("--out-dir", default="data/signals", type=str)
    ap.add_argument("--prefix", default="", type=str)
    ap.add_argument("--period", default="Y", type=str, choices=["Y", "M"])
    args = ap.parse_args()

    df = read_any(args.trades_path).copy()

    # Normalize datetimes if present
    for col in ["EntryDate", "ExitDate"]:
        if col in df.columns:
            df[col] = norm_dt(df[col])

    # Ensure CycleReturn exists or is derived
    df = ensure_cycle_return(df)
    df = df.dropna(subset=["CycleReturn"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after cleaning/deriving CycleReturn.")

    main_df = df[df["CycleType"].astype(str).str.upper() == "MAIN"].copy()
    sub_df = df[df["CycleType"].astype(str).str.upper() == "SUB"].copy()

    overall = pd.DataFrame([
        agg_stats(df, "ALL"),
        agg_stats(main_df, "MAIN"),
        agg_stats(sub_df, "SUB"),
    ])

    def gross_prod(x: pd.DataFrame) -> float:
        if x.empty:
            return 1.0
        g = (1.0 + x["CycleReturn"].astype(float)).clip(lower=1e-12)
        return float(g.prod())

    gross_all = gross_prod(df)
    gross_main = gross_prod(main_df)
    gross_sub = gross_prod(sub_df)
    lift_vs_main = (gross_all / gross_main) if gross_main > 0 else np.nan

    extra = pd.DataFrame([{
        "Metric": "GrossMultiple_ALL",
        "Value": gross_all
    }, {
        "Metric": "GrossMultiple_MAIN_ONLY",
        "Value": gross_main
    }, {
        "Metric": "GrossMultiple_SUB_ONLY",
        "Value": gross_sub
    }, {
        "Metric": "ALL_over_MAIN_multiple",
        "Value": lift_vs_main
    }])

    per = make_period_table(df, period=args.period)

    worst_sub = pd.DataFrame()
    if not sub_df.empty:
        tmp = make_period_table(sub_df, period=args.period)
        worst_sub = tmp[tmp["CycleType"] == "ALL"].copy()
        if not worst_sub.empty:
            worst_sub = worst_sub.sort_values("MeanReturn").head(10).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.strip()
    if prefix:
        prefix = prefix + "_"

    out_overall = out_dir / f"{prefix}sub_main_overall.csv"
    out_extra = out_dir / f"{prefix}sub_main_multiples.csv"
    out_period = out_dir / f"{prefix}sub_main_by_{'year' if args.period == 'Y' else 'month'}.csv"
    out_worst = out_dir / f"{prefix}sub_worst_{'year' if args.period == 'Y' else 'month'}_top10.csv"

    overall.to_csv(out_overall, index=False)
    extra.to_csv(out_extra, index=False)
    per.to_csv(out_period, index=False)
    if not worst_sub.empty:
        worst_sub.to_csv(out_worst, index=False)

    print("=== OVERALL (ALL / MAIN / SUB) ===")
    print(overall.to_string(index=False))
    print("\n=== MULTIPLES (diagnostic) ===")
    print(extra.to_string(index=False))

    print(f"\n[DONE] wrote: {out_overall}")
    print(f"[DONE] wrote: {out_extra}")
    print(f"[DONE] wrote: {out_period}")
    if not worst_sub.empty:
        print(f"[DONE] wrote: {out_worst}")
    else:
        print("[INFO] No SUB rows or worst-period table not created.")


if __name__ == "__main__":
    main()