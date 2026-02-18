# scripts/train_tau_model.py
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Train tau (FAST/MID/SLOW) multi-class model.")
    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)
    ap.add_argument("--labels-parq", default="data/labels/labels_tau.parquet", type=str)
    ap.add_argument("--labels-csv", default="data/labels/labels_tau.csv", type=str)

    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)
    ap.add_argument("--target-col", default="TauClass", type=str)

    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), type=str)
    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=800, type=int)

    ap.add_argument("--out-model", default="app/tau_model.pkl", type=str)
    ap.add_argument("--out-scaler", default="app/tau_scaler.pkl", type=str)

    args = ap.parse_args()

    feats = read_table(args.features_parq, args.features_csv).copy()
    labs = read_table(args.labels_parq, args.labels_csv).copy()

    for df, name in [(feats, "features_model"), (labs, "labels_tau")]:
        if args.date_col not in df.columns or args.ticker_col not in df.columns:
            raise KeyError(f"{name} must have {args.date_col},{args.ticker_col}")

    feats[args.date_col] = norm_date(feats[args.date_col])
    labs[args.date_col] = norm_date(labs[args.date_col])
    feats[args.ticker_col] = feats[args.ticker_col].astype(str).str.upper().str.strip()
    labs[args.ticker_col] = labs[args.ticker_col].astype(str).str.upper().str.strip()

    feat_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    missing = [c for c in feat_cols if c not in feats.columns]
    if missing:
        raise KeyError(f"features_model missing feature columns: {missing}")

    if args.target_col not in labs.columns:
        raise KeyError(f"labels_tau missing target column: {args.target_col}")

    # merge
    df = pd.merge(
        feats[[args.date_col, args.ticker_col] + feat_cols].copy(),
        labs[[args.date_col, args.ticker_col, args.target_col]].copy(),
        on=[args.date_col, args.ticker_col],
        how="inner",
    ).dropna(subset=feat_cols + [args.target_col])

    if df.empty:
        raise RuntimeError("No training rows after merge. Check labels_tau and features_model overlap.")

    # sort time
    df = df.sort_values(args.date_col).reset_index(drop=True)

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df[args.target_col], errors="coerce").fillna(2).astype(int).to_numpy()

    # split
    n = len(df)
    split = int(n * float(args.train_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # model: multinomial logistic
    model = LogisticRegression(
        max_iter=int(args.max_iter),
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=None,
        class_weight="balanced",
    )

    # optional time-series CV sanity (train only)
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))
    cv_losses = []
    for fold, (tr, va) in enumerate(tscv.split(X_train_s), start=1):
        m = LogisticRegression(
            max_iter=int(args.max_iter),
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
        )
        m.fit(X_train_s[tr], y_train[tr])
        p_va = m.predict_proba(X_train_s[va])
        ll = log_loss(y_train[va], p_va, labels=[0, 1, 2])
        cv_losses.append(ll)

    model.fit(X_train_s, y_train)

    # evaluate
    if len(X_test_s) > 0:
        p_test = model.predict_proba(X_test_s)
        ll_test = log_loss(y_test, p_test, labels=[0, 1, 2])
    else:
        ll_test = float("nan")

    print("=" * 60)
    print(f"[INFO] rows={n} train={len(X_train)} test={len(X_test)}")
    print(f"[INFO] CV logloss (train folds): {np.round(cv_losses, 4).tolist()} mean={float(np.mean(cv_losses)):.4f}")
    print(f"[INFO] Test logloss: {ll_test:.4f}")
    print(f"[INFO] Class distribution (all): {pd.Series(y).value_counts().to_dict()}")
    print("=" * 60)

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_model)
    joblib.dump(scaler, args.out_scaler)
    print(f"[DONE] saved model: {args.out_model}")
    print(f"[DONE] saved scaler: {args.out_scaler}")


if __name__ == "__main__":
    main()