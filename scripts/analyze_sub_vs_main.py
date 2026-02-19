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
        if np.isfinite(v):
            return v
        return np.nan
    except Exception:
        return np.nan


def agg_stats(df: pd.DataFrame, label: str) -> dict:
    """Return summary stats for a subset."""
    if df.empty:
        return {"Group": label, "N": 0}

    r = df["CycleReturn"].astype(float)
    win = (r > 0).astype(int)

    # basic moments
    mean_r = float(r.mean())
    med_r = float(r.median())
    std_r = float(r.std(ddof=1)) if len(r) > 1 else 0.0

    # win/lose
    win_rate = float(win.mean())
    avg_win = float(r[r > 0].mean()) if (r > 0).any() else np.nan
    avg_loss = float(r[r <= 0].mean()) if (r <= 0).any() else np.nan
    profit_factor = (
        float(r[r > 0].sum() / abs(r[r <= 0].sum()))
        if (r > 0).any() and (r <= 0).any() and abs(r[r <= 0].sum()) > 1e-12
        else np.nan
    )

    # geometric growth per cycle
    # NOTE: CycleReturn is (proceeds-invested)/invested, so gross = 1 + r
    gross = (1.0 + r).clip(lower=1e-12)
    geom_total = float(gross.prod())  # total gross multiple across cycles
    geom_mean = float(np.exp(np.log(gross).mean()))  # avg gross per cycle

    # holding days
    hd_col = "HoldingDays" if "HoldingDays" in df.columns else None
    hd_mean = float(df[hd_col].mean()) if hd_col else np.nan
    hd_med = float(df[hd_col].median()) if hd_col else np.nan

    # invested
    inv = df["Invested"].astype(float) if "Invested" in df.columns else pd.Series([np.nan] * len(df))
    inv_mean = float(inv.mean()) if inv.notna().any() else np.nan

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
    """
    period: 'Y' (year) or 'M' (month)
    Produces per-period stats by CycleType and total.
    """
    x = df.copy()
    x["EntryDate"] = norm_dt(x["EntryDate"])
    x["Period"] = x["EntryDate"].dt.to_period(period).astype(str)

    # per period per type
    out = []
    for (p, ctype), g in x.groupby(["Period", "CycleType"], dropna=False):
        out.append({**{"Period": p, "CycleType": ctype}, **agg_stats(g, label="")})
    per = pd.DataFrame(out)
    if not per.empty:
        per = per.drop(columns=["Group"], errors="ignore")
        per = per.sort_values(["Period", "CycleType"]).reset_index(drop=True)

    # also total per period (all types)
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

    # merge (stack)
    comb = pd.concat([per_all, per], ignore_index=True)
    return comb


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze MAIN vs SUB performance from sim_engine_trades parquet/csv.")
    ap.add_argument("--trades-path", required=True, type=str, help="sim_engine_trades_*.parquet (or csv)")
    ap.add_argument("--out-dir", default="data/signals", type=str)
    ap.add_argument("--prefix", default="", type=str, help="Optional output prefix")
    ap.add_argument("--period", default="Y", type=str, choices=["Y", "M"], help="Period granularity for breakdown")
    args = ap.parse_args()

    df = read_any(args.trades_path).copy()

    # normalize expected columns
    if "CycleType" not in df.columns:
        # if older file without CycleType, treat as MAIN
        df["CycleType"] = "MAIN"

    for col in ["EntryDate", "ExitDate"]:
        if col in df.columns:
            df[col] = norm_dt(df[col])

    # ensure CycleReturn numeric
    if "CycleReturn" not in df.columns:
        raise ValueError(f"Missing CycleReturn in trades. cols={list(df.columns)[:60]}")
    df["CycleReturn"] = df["CycleReturn"].apply(safe_float)

    # quick sanity
    df = df.dropna(subset=["CycleReturn"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid CycleReturn rows after cleaning.")

    # split
    main_df = df[df["CycleType"].astype(str).str.upper() == "MAIN"].copy()
    sub_df = df[df["CycleType"].astype(str).str.upper() == "SUB"].copy()

    # overall stats
    overall = pd.DataFrame([
        agg_stats(df, "ALL"),
        agg_stats(main_df, "MAIN"),
        agg_stats(sub_df, "SUB"),
    ])

    # compute: "ALL vs MAIN only" gross multiple difference
    # NOTE: This is cycle-level gross product; not portfolio-equity exact, but good diagnostic.
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

    # period breakdown
    per = make_period_table(df, period=args.period)

    # also: identify worst SUB periods quickly
    worst_sub = pd.DataFrame()
    if not sub_df.empty:
        tmp = make_period_table(sub_df, period=args.period)
        # keep only ALL row inside sub_df breakdown (which is effectively SUB)
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

    # print summary
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
        print("[INFO] No SUB rows or no worst-period table created.")


if __name__ == "__main__":
    main()