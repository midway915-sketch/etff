# scripts/train_strategy_models.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score


DATA_PATH = Path("data/strategy_raw_data.csv")
TAIL_MODEL_PATH = Path("app/tail_model.pkl")
TAIL_SCALER_PATH = Path("app/tail_scaler.pkl")

FEATURES = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
]


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train strategy models (tail).")
    ap.add_argument("--profit-target", type=float, required=True)  # 로그/재현용 (학습에는 직접 안 씀)
    ap.add_argument("--max-days", type=int, required=True)         # 로그/재현용
    ap.add_argument("--stop-level", type=float, required=True)     # 로그/재현용
    ap.add_argument("--max-extend-days", type=int, required=True)  # 로그/재현용
    ap.add_argument("--tail-threshold", type=float, default=-0.30, help="라벨 생성 시 사용한 값 기록용")

    ap.add_argument("--data", type=str, default=str(DATA_PATH))
    ap.add_argument("--target-col", type=str, default="Tail")
    ap.add_argument("--out-model", type=str, default=str(TAIL_MODEL_PATH))
    ap.add_argument("--out-scaler", type=str, default=str(TAIL_SCALER_PATH))
    ap.add_argument("--n-splits", type=int, default=3)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run scripts/build_strategy_labels.py first."
        )

    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    else:
        df = df.reset_index(drop=True)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in {data_path}")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in training data: {missing}")

    df = df.dropna(subset=FEATURES + [args.target_col]).reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df[args.target_col].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Tail target has only one class. tail_threshold/horizon might be off.")

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(max_iter=800)

    n_splits = min(int(args.n_splits), 5)
    if split_idx < 200:
        n_splits = min(n_splits, 2)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Logistic + Isotonic Calibration
    # TimeSeriesSplit 사용
    base = LogisticRegression(max_iter=800)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # ✅ sklearn 버전 호환: estimator= (new) / base_estimator= (old)
    try:
        model = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv=tscv,
        )
    except TypeError:
        model = CalibratedClassifierCV(
            base_estimator=base,
            method="isotonic",
            cv=tscv,
        )
    
    model.fit(X_train_s, y_train)


    probs = model.predict_proba(X_test_s)[:, 1]
    auc = safe_auc(y_test, probs)

    print("=" * 60)
    print("Tail Test ROC-AUC:", round(auc, 4) if auc == auc else "nan")
    print("Base Tail Rate:", round(float(y_test.mean()), 4))
    print("Predicted Mean Probability:", round(float(probs.mean()), 4))
    print("Rows:", len(df), "Train:", len(y_train), "Test:", len(y_test))
    print("Args:", {
        "profit_target": args.profit_target,
        "max_days": args.max_days,
        "stop_level": args.stop_level,
        "max_extend_days": args.max_extend_days,
        "tail_threshold": args.tail_threshold,
    })
    print("=" * 60)

    out_model = Path(args.out_model)
    out_scaler = Path(args.out_scaler)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)
    print(f"✅ saved: {out_model} / {out_scaler}")


if __name__ == "__main__":
    main()
