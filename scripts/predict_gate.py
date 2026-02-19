#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# IO helpers
# ----------------------------
def read_table(parq_path: Path, csv_path: Path) -> pd.DataFrame:
    if parq_path.exists():
        return pd.read_parquet(parq_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing file: {parq_path} (or {csv_path})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def coerce_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return x.astype(float)


def parse_require_files(s: str) -> List[Path]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    return [Path(p) for p in parts]


def ensure_required_files_exist(require_files: List[Path]) -> None:
    missing = [str(p) for p in require_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


# ----------------------------
# Column normalization
# ----------------------------
def ensure_date_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'Date' and 'Ticker' exist as columns.
    Robust to:
      - Date as index (named or unnamed)
      - date/datetime/timestamp column names
      - ticker/symbol column names
    """
    out = df.copy()

    # If Date is missing, try to pull it from index even if index is unnamed.
    if "Date" not in out.columns:
        if not isinstance(out.index, pd.RangeIndex):
            idx = out.index
            if isinstance(idx, pd.DatetimeIndex):
                out = out.reset_index()
                if "index" in out.columns and "Date" not in out.columns:
                    out = out.rename(columns={"index": "Date"})
            else:
                try:
                    tmp = pd.to_datetime(idx, errors="coerce")
                    if tmp.notna().any():
                        out = out.reset_index()
                        if "index" in out.columns and "Date" not in out.columns:
                            out = out.rename(columns={"index": "Date"})
                except Exception:
                    pass

    # Normalize known alternative column names
    colmap = {c.lower(): c for c in out.columns}

    if "date" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["date"]: "Date"})
    if "datetime" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["datetime"]: "Date"})
    if "timestamp" in colmap and "Date" not in out.columns:
        out = out.rename(columns={colmap["timestamp"]: "Date"})

    if "ticker" in colmap and "Ticker" not in out.columns:
        out = out.rename(columns={colmap["ticker"]: "Ticker"})
    if "symbol" in colmap and "Ticker" not in out.columns:
        out = out.rename(columns={colmap["symbol"]: "Ticker"})

    if "Date" not in out.columns:
        raise KeyError(
            f"Date column missing. cols(head)={list(out.columns)[:30]} index.name={out.index.name} "
            f"index.type={type(out.index).__name__}"
        )
    if "Ticker" not in out.columns:
        raise KeyError(
            f"Ticker column missing. cols(head)={list(out.columns)[:30]} index.name={out.index.name} "
            f"index.type={type(out.index).__name__}"
        )

    return out


def apply_exclude_tickers(df: pd.DataFrame, exclude_csv: str) -> pd.DataFrame:
    ex = [t.strip().upper() for t in (exclude_csv or "").split(",") if t.strip()]
    if not ex:
        return df
    return df[~df["Ticker"].isin(ex)].copy()


# ----------------------------
# Features source selection
# ----------------------------
def pick_features_source(require_files: List[Path]) -> Tuple[Path, Path]:
    """
    Prefer features_scored if present in require_files.
    Otherwise default to scored path; caller will fallback to features_model if not found.
    Returns (parq_candidate, csv_candidate)
    """
    for p in require_files:
        name = p.name
        if "features_scored" in name and (name.endswith(".parquet") or name.endswith(".csv")):
            if name.endswith(".parquet"):
                return p, p.with_suffix(".csv")
            return p.with_suffix(".parquet"), p

    return Path("data/features/features_scored.parquet"), Path("data/features/features_scored.csv")


# ----------------------------
# Utility / gating / ranking
# ----------------------------
def build_utility(df: pd.DataFrame, lambda_tail: float) -> pd.Series:
    # utility = ret_score - lambda * p_tail
    ret = coerce_num(df, "ret_score", 0.0)
    p_tail = coerce_num(df, "p_tail", 0.0)
    return ret - float(lambda_tail) * p_tail


def gate_filter(
    df: pd.DataFrame,
    mode: str,
    tail_threshold: float,
    utility_quantile: float,
) -> pd.DataFrame:
    """
    IMPORTANT: No groupby().apply() here.
    We use groupby().transform() so 'Date' column never disappears.
    """
    d = ensure_date_ticker_columns(df)
    mode = (mode or "none").strip().lower()

    if d.empty:
        return d

    if mode == "none":
        return d.copy()

    if mode == "tail":
        return d[d["p_tail"] <= float(tail_threshold)].copy()

    if mode == "utility":
        q = float(utility_quantile)
        thr = d.groupby("Date")["utility"].transform(lambda s: s.quantile(q))
        return d[d["utility"] >= thr].copy()

    if mode == "tail_utility":
        q = float(utility_quantile)
        d2 = d[d["p_tail"] <= float(tail_threshold)].copy()
        if d2.empty:
            return d2
        thr = d2.groupby("Date")["utility"].transform(lambda s: s.quantile(q))
        return d2[d2["utility"] >= thr].copy()

    raise ValueError(f"Unknown mode: {mode} (expected none|tail|utility|tail_utility)")


def rank_topk_per_day(df: pd.DataFrame, rank_by: str, topk: int) -> pd.DataFrame:
    d = ensure_date_ticker_columns(df)
    if d.empty:
        return d

    rb = (rank_by or "utility").strip().lower()
    if rb == "utility":
        metric = d["utility"]
    elif rb == "ret_score":
        metric = d["ret_score"]
    elif rb == "p_success":
        metric = d["p_success"]
    else:
        raise ValueError(f"Unknown rank_by: {rank_by} (expected utility|ret_score|p_success)")

    d["_metric"] = metric
    d = d.sort_values(["Date", "_metric"], ascending=[True, False])

    picks = (
        d.groupby("Date", as_index=False, group_keys=False)
        .head(int(topk))
        .drop(columns=["_metric"])
        .reset_index(drop=True)
    )
    return picks


# ----------------------------
# main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Gate picks generator (reads features_scored if available).")

    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--mode", type=str, required=True, help="none|tail|utility|tail_utility")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--suffix", type=str, required=True)

    ap.add_argument("--tail-threshold", type=float, required=True)
    ap.add_argument("--utility-quantile", type=float, required=True)
    ap.add_argument("--rank-by", type=str, required=True, help="utility|ret_score|p_success")
    ap.add_argument("--lambda-tail", type=float, required=True)

    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--ps-min", type=float, required=True)

    # ✅ SUB entry filters (anti-chasing)
    ap.add_argument("--sub-ps-min", type=float, required=True, help="SUB entry filter: p_success >= this")
    ap.add_argument("--sub-z-max", type=float, required=True, help="SUB entry filter: Z_score <= this")
    ap.add_argument("--sub-dd60-max", type=float, required=True, help="SUB entry filter: Drawdown_60 <= this (e.g. -0.05)")

    ap.add_argument("--exclude-tickers", type=str, default="")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--require-files", type=str, required=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    require_files = parse_require_files(args.require_files)
    ensure_required_files_exist(require_files)

    # choose features file (prefer scored)
    f_parq, f_csv = pick_features_source(require_files)

    # fallback if scored doesn't exist
    if (not f_parq.exists()) and (not f_csv.exists()):
        f_parq = Path("data/features/features_model.parquet")
        f_csv = Path("data/features/features_model.csv")

    feats = read_table(f_parq, f_csv).copy()
    features_src = str(f_parq) if f_parq.exists() else str(f_csv)

    feats = ensure_date_ticker_columns(feats)

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # core columns
    feats["p_success"] = coerce_num(feats, "p_success", 0.0)
    feats["p_tail"] = coerce_num(feats, "p_tail", 0.0)
    feats["ret_score"] = coerce_num(feats, "ret_score", 0.0)

    # ✅ sub filter feature columns (safe defaults)
    feats["Z_score"] = coerce_num(feats, "Z_score", 0.0)
    feats["Drawdown_60"] = coerce_num(feats, "Drawdown_60", 0.0)

    feats = apply_exclude_tickers(feats, args.exclude_tickers)

    # base gate: p_success min (MAIN gate)
    feats = feats[feats["p_success"] >= float(args.ps_min)].copy()

    feats["utility"] = build_utility(feats, float(args.lambda_tail))

    # ✅ compute SubOK on the *post main ps-min* universe (so SUB never uses stuff MAIN wouldn't consider)
    feats["SubOK"] = (
        (feats["p_success"] >= float(args.sub_ps_min)) &
        (feats["Z_score"] <= float(args.sub_z_max)) &
        (feats["Drawdown_60"] <= float(args.sub_dd60_max))
    ).astype(int)

    gated = gate_filter(
        feats,
        mode=args.mode,
        tail_threshold=float(args.tail_threshold),
        utility_quantile=float(args.utility_quantile),
    )

    picks = rank_topk_per_day(gated, rank_by=args.rank_by, topk=int(args.topk))

    picks_path = out_dir / f"picks_{args.tag}_gate_{args.suffix}.csv"
    meta_path = out_dir / f"picks_meta_{args.tag}_gate_{args.suffix}.json"

    keep_cols = [
        c for c in [
            "Date", "Ticker",
            "p_success", "p_tail", "ret_score", "utility",
            "Z_score", "Drawdown_60",
            "SubOK",
        ]
        if c in picks.columns
    ]
    picks_out = picks[keep_cols].copy()
    picks_out.to_csv(picks_path, index=False)

    meta = {
        "tag": args.tag,
        "suffix": args.suffix,
        "mode": args.mode,
        "rank_by": args.rank_by,
        "topk": int(args.topk),
        "ps_min": float(args.ps_min),
        "tail_threshold": float(args.tail_threshold),
        "utility_quantile": float(args.utility_quantile),
        "lambda_tail": float(args.lambda_tail),
        "exclude_tickers": args.exclude_tickers,
        "features_src": features_src,
        "rows_in_features": int(len(feats)),
        "rows_after_gate": int(len(gated)),
        "rows_picks": int(len(picks_out)),
        "profit_target": float(args.profit_target),
        "max_days": int(args.max_days),
        "stop_level": float(args.stop_level),
        "max_extend_days": int(args.max_extend_days),

        # SUB filter params (logged)
        "sub_ps_min": float(args.sub_ps_min),
        "sub_z_max": float(args.sub_z_max),
        "sub_dd60_max": float(args.sub_dd60_max),
        "subok_rate_in_features": float(feats["SubOK"].mean()) if len(feats) else 0.0,
        "subok_rate_in_gated": float(gated["SubOK"].mean()) if len(gated) and "SubOK" in gated.columns else 0.0,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote: {picks_path} rows={len(picks_out)}")
    print(f"[INFO] SubOK rate in feats(after ps-min)={meta['subok_rate_in_features']:.4f} gated={meta['subok_rate_in_gated']:.4f}")
    print(f"[DONE] wrote: {meta_path}")


if __name__ == "__main__":
    main()