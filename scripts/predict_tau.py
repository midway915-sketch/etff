# scripts/predict_tau.py
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


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


def tau_class_from_probs(p_fast: float, p_slow: float, thr: float = 0.45) -> int:
    # 0=FAST, 1=MID, 2=SLOW
    if np.isfinite(p_fast) and p_fast >= thr:
        return 0
    if np.isfinite(p_slow) and p_slow >= thr:
        return 2
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict tau probabilities and TauClass for all (Date,Ticker).")
    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)

    ap.add_argument("--model", default="app/tau_model.pkl", type=str)
    ap.add_argument("--scaler", default="app/tau_scaler.pkl", type=str)

    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), type=str)
    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)

    ap.add_argument("--thr", default=0.45, type=float, help="prob threshold for FAST/SLOW (else MID)")

    ap.add_argument("--out-parq", default="data/features/features_tau.parquet", type=str)
    ap.add_argument("--out-csv", default="data/features/features_tau.csv", type=str)

    args = ap.parse_args()

    if not Path(args.model).exists() or not Path(args.scaler).exists():
        raise FileNotFoundError(f"Missing tau model/scaler: {args.model}, {args.scaler}")

    feats = read_table(args.features_parq, args.features_csv).copy()
    if args.date_col not in feats.columns or args.ticker_col not in feats.columns:
        raise KeyError("features_model must have Date,Ticker")

    feats[args.date_col] = norm_date(feats[args.date_col])
    feats[args.ticker_col] = feats[args.ticker_col].astype(str).str.upper().str.strip()
    feats = feats.dropna(subset=[args.date_col, args.ticker_col]).sort_values([args.date_col, args.ticker_col]).reset_index(drop=True)

    feat_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    missing = [c for c in feat_cols if c not in feats.columns]
    if missing:
        raise KeyError(f"features_model missing feature columns: {missing}")

    X = feats[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)

    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)  # columns correspond to classes 0,1,2

    p_fast = probs[:, 0].astype(float)
    p_mid = probs[:, 1].astype(float)
    p_slow = probs[:, 2].astype(float)

    tau_class = [tau_class_from_probs(a, c, thr=float(args.thr)) for a, c in zip(p_fast.tolist(), p_slow.tolist())]

    out = pd.DataFrame({
        "Date": feats[args.date_col].to_numpy(),
        "Ticker": feats[args.ticker_col].to_numpy(),
        "p_tau_fast": p_fast,
        "p_tau_mid": p_mid,
        "p_tau_slow": p_slow,
        "TauClassPred": np.array(tau_class, dtype=int),
        "TauThr": float(args.thr),
    })

    Path(args.out_parq).parent.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(args.out_parq, index=False)
        print(f"[DONE] wrote: {args.out_parq} rows={len(out)}")
    except Exception as e:
        print(f"[WARN] parquet write failed: {e}")

    out.to_csv(args.out_csv, index=False)
    print(f"[DONE] wrote: {args.out_csv} rows={len(out)}")

    vc = out["TauClassPred"].value_counts().to_dict()
    print(f"[INFO] TauClassPred counts: {vc}")


if __name__ == "__main__":
    main()