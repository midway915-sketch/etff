# scripts/train_success_model.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path("data")
FEAT_PARQ = DATA_DIR / "features" / "features_model.parquet"
FEAT_CSV = DATA_DIR / "features" / "features_model.csv"

LBL_PARQ = DATA_DIR / "labels" / "labels_success.parquet"
LBL_CSV = DATA_DIR / "labels" / "labels_success.csv"

APP_DIR = Path("app")
MODEL_PATH = APP_DIR / "success_model.pkl"
SCALER_PATH = APP_DIR / "success_scaler.pkl"
META_PATH = APP_DIR / "success_model_meta.json"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def pick_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"Date", "Ticker"}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ratio", default=0.85, type=float)
    ap.add_argument("--max-iter", default=2000, type=int)
    args = ap.parse_args()

    APP_DIR.mkdir(parents=True, exist_ok=True)

    feats = read_table(FEAT_PARQ, FEAT_CSV).copy()
    lbl = read_table(LBL_PARQ, LBL_CSV).copy()

    for df in (feats, lbl):
        if "Date" not in df.columns or "Ticker" not in df.columns:
            raise ValueError("Both feats and labels must have Date,Ticker")
        df["Date"] = norm_date(df["Date"])
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    if "Target" not in lbl.columns:
        raise ValueError("labels_success must have Target")

    df = feats.merge(lbl[["Date", "Ticker", "Target"]], on=["Date", "Ticker"], how="inner")
    df = df.dropna(subset=["Target"]).copy()
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feature_cols = pick_feature_cols(df)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found for training.")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df["Target"], errors="coerce").fillna(0).astype(int).to_numpy()

    n = len(df)
    cut = int(n * float(args.train_ratio))
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte) if len(Xte) else Xte

    clf = LogisticRegression(max_iter=int(args.max_iter), n_jobs=1, class_weight="balanced")
    clf.fit(Xtr_s, ytr)

    if len(Xte):
        p = clf.predict_proba(Xte_s)[:, 1]
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(yte, p))
        except Exception:
            auc = float("nan")
        print(f"[INFO] success_model test AUC={auc:.4f} (may be NaN if degenerate)")

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "type": "success",
        "feature_cols": feature_cols,
        "rows": int(len(df)),
        "train_ratio": float(args.train_ratio),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] saved: {MODEL_PATH}, {SCALER_PATH}, {META_PATH}")


if __name__ == "__main__":
    main()