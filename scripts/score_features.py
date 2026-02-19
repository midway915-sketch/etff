# scripts/score_features.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
FEATURE_DIR = DATA_DIR / "features"
META_DIR = DATA_DIR / "meta"


DEFAULT_FEATURES = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
]


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def ensure_features_exist(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = 0.0
    for c in feat_cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return out


def load_tail_feature_cols(tag: str) -> list[str] | None:
    # train_tail_model.py writes: data/meta/train_tail_report_{tag}.json
    p = META_DIR / f"train_tail_report_{tag}.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        cols = j.get("feature_cols")
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols]
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Score features_model with p_success and p_tail and save features_scored.")
    ap.add_argument("--tag", required=True, type=str, help="e.g. pt10_h40_sl10_ex30")

    ap.add_argument("--features-parq", default=str(FEATURE_DIR / "features_model.parquet"), type=str)
    ap.add_argument("--features-csv", default=str(FEATURE_DIR / "features_model.csv"), type=str)

    ap.add_argument("--out-parq", default=str(FEATURE_DIR / "features_scored.parquet"), type=str)
    ap.add_argument("--out-csv", default=str(FEATURE_DIR / "features_scored.csv"), type=str)

    ap.add_argument("--ps-model", default="app/model.pkl", type=str)
    ap.add_argument("--ps-scaler", default="app/scaler.pkl", type=str)
    ap.add_argument("--ps-features", default=",".join(DEFAULT_FEATURES), type=str)

    ap.add_argument("--tail-model", default="app/tail_model.pkl", type=str)
    ap.add_argument("--tail-scaler", default="app/tail_scaler.pkl", type=str)
    ap.add_argument("--tail-features", default="", type=str,
                    help="comma-separated override. default=read from train_tail_report_{tag}.json or fallback to ps-features")

    args = ap.parse_args()

    feats = read_table(Path(args.features_parq), Path(args.features_csv)).copy()
    print("[DEBUG] features_scored cols:", list(feats.columns)[:30], " index.name:", feats.index.name)
    if "Date" not in feats.columns or "Ticker" not in feats.columns:
        raise ValueError("features_model must contain Date,Ticker")

    feats["Date"] = norm_date(feats["Date"])
    feats["Ticker"] = feats["Ticker"].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # ---- p_success
    ps_model = joblib.load(args.ps_model)
    ps_scaler = joblib.load(args.ps_scaler)
    ps_cols = [c.strip() for c in str(args.ps_features).split(",") if c.strip()]
    feats_ps = ensure_features_exist(feats, ps_cols)
    X_ps = feats_ps[ps_cols].to_numpy(dtype=float)
    X_ps_s = ps_scaler.transform(X_ps)
    feats["p_success"] = ps_model.predict_proba(X_ps_s)[:, 1].astype(float)

    # ---- p_tail (optional)
    tail_model_path = Path(args.tail_model)
    tail_scaler_path = Path(args.tail_scaler)
    if tail_model_path.exists() and tail_scaler_path.exists():
        tail_model = joblib.load(tail_model_path)
        tail_scaler = joblib.load(tail_scaler_path)

        if args.tail_features.strip():
            tail_cols = [c.strip() for c in args.tail_features.split(",") if c.strip()]
        else:
            tail_cols = load_tail_feature_cols(args.tag) or ps_cols

        feats_tail = ensure_features_exist(feats, tail_cols)
        X_t = feats_tail[tail_cols].to_numpy(dtype=float)
        X_t_s = tail_scaler.transform(X_t)
        feats["p_tail"] = tail_model.predict_proba(X_t_s)[:, 1].astype(float)
    else:
        feats["p_tail"] = 0.0

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    try:
        feats.to_parquet(out_parq, index=False)
        print(f"[DONE] wrote: {out_parq} rows={len(feats)}")
    except Exception as e:
        print(f"[WARN] parquet save failed ({e}) -> writing csv")
        feats.to_csv(out_csv, index=False)
        print(f"[DONE] wrote: {out_csv} rows={len(feats)}")

if __name__ == "__main__":
    main()