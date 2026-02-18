# scripts/train_model.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


DATA_DIR = Path("data")
LABELS_PARQ = DATA_DIR / "labels" / "labels_model.parquet"
LABELS_CSV = DATA_DIR / "labels" / "labels_model.csv"
APP_DIR = Path("app")


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
    raise FileNotFoundError(f"Training data not found: {parq} (or {csv})")


def parse_csv_list(s: str) -> list[str]:
    if s is None:
        return []
    items = [x.strip() for x in str(s).split(",")]
    return [x for x in items if x]


def ensure_features_exist(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in feat_cols:
        if c not in df.columns:
            # ✅ 방어: 누락 컬럼은 0으로 생성 (pipeline 안 죽게)
            df[c] = 0.0
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", type=str, help="optional tag suffix for saving model files")

    ap.add_argument("--target-col", default="Success", type=str)
    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)

    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), type=str)
    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)

    ap.add_argument("--out-model", default="", type=str)
    ap.add_argument("--out-scaler", default="", type=str)

    args = ap.parse_args()

    df = read_table(LABELS_PARQ, LABELS_CSV).copy()

    date_col = args.date_col
    target_col = args.target_col
    feat_cols = parse_csv_list(args.features) or DEFAULT_FEATURES

    for c in [date_col, target_col]:
        if c not in df.columns:
            raise ValueError(f"labels_model missing required column: {c}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    df = ensure_features_exist(df, feat_cols)

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    X = df[feat_cols].to_numpy(dtype=float)

    if len(y) < 200:
        raise RuntimeError(f"Not enough training rows: {len(y)}")

    split_idx = int(len(y) * float(args.train_ratio))
    split_idx = max(50, min(split_idx, len(y) - 50))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(max_iter=int(args.max_iter))
    tscv = TimeSeriesSplit(n_splits=int(args.n_splits))

    # ✅ sklearn 버전 차이 방어 (estimator vs base_estimator)
    try:
        model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=tscv)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)

    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = float("nan")
    if len(np.unique(y_test)) > 1:
        auc = float(roc_auc_score(y_test, probs))

    print("=" * 60)
    print("[TRAIN] p_success model")
    print("rows:", len(y), "train:", len(y_train), "test:", len(y_test))
    print("AUC:", (round(auc, 4) if np.isfinite(auc) else "nan"))
    print("base_rate:", round(float(np.mean(y_test)), 4) if len(y_test) else "nan")
    print("pred_mean:", round(float(np.mean(probs)), 4) if len(probs) else "nan")
    print("=" * 60)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    tag = (args.tag or "").strip()

    out_model = Path(args.out_model) if args.out_model else (APP_DIR / (f"model_{tag}.pkl" if tag else "model.pkl"))
    out_scaler = Path(args.out_scaler) if args.out_scaler else (APP_DIR / (f"scaler_{tag}.pkl" if tag else "scaler.pkl"))

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)

    print(f"[DONE] saved model -> {out_model}")
    print(f"[DONE] saved scaler -> {out_scaler}")


if __name__ == "__main__":
    main()