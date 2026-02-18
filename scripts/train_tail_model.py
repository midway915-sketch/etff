# scripts/train_tail_model.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


APP_DIR = Path("app")
META_DIR = Path("data/meta")
LABELS_DIR = Path("data/labels")

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


def choose_features(df: pd.DataFrame, requested: list[str]) -> list[str]:
    cols = set(df.columns)
    feats = [c for c in requested if c in cols]
    if len(feats) < 4:
        banned = {"TailTarget","Target","Success","Date","Ticker"}
        numeric = [c for c in df.columns if c not in banned and pd.api.types.is_numeric_dtype(df[c])]
        feats = feats + [c for c in numeric if c not in feats][:12]
    out, seen = [], set()
    for c in feats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fit_model(X_train, y_train, n_splits: int, max_iter: int):
    base = LogisticRegression(max_iter=max_iter)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    try:
        clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=tscv)
    except TypeError:
        clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=tscv)
    clf.fit(X_train, y_train)
    return clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)

    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES), type=str)
    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)
    args = ap.parse_args()

    pt_tag = int(round(args.profit_target * 100))
    sl_tag = int(round(abs(args.stop_level) * 100))
    tag = f"pt{pt_tag}_h{args.max_days}_sl{sl_tag}_ex{args.max_extend_days}"

    # preferred: labels_tail_tag
    parq = LABELS_DIR / f"labels_tail_{tag}.parquet"
    csv = LABELS_DIR / f"labels_tail_{tag}.csv"

    if parq.exists() or csv.exists():
        df = read_table(parq, csv).copy()
        src = str(parq if parq.exists() else csv)
    else:
        # fallback: labels_model, if it contains TailTarget
        parq2 = LABELS_DIR / "labels_model.parquet"
        csv2 = LABELS_DIR / "labels_model.csv"
        df = read_table(parq2, csv2).copy()
        src = str(parq2 if parq2.exists() else csv2)
        if "TailTarget" not in df.columns:
            raise FileNotFoundError(
                f"Missing tail labels. Provide {parq}/{csv} or include TailTarget in labels_model."
            )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")

    if "TailTarget" not in df.columns:
        raise ValueError(f"TailTarget column missing (src={src})")

    requested = [x.strip() for x in args.features.split(",") if x.strip()]
    feature_cols = choose_features(df, requested)

    use = df.dropna(subset=feature_cols + ["TailTarget"]).copy()
    for c in feature_cols:
        use[c] = pd.to_numeric(use[c], errors="coerce").fillna(0.0)
    y = pd.to_numeric(use["TailTarget"], errors="coerce").fillna(0).astype(int)

    X = use[feature_cols]
    split_idx = int(len(use) * float(args.train_ratio))
    split_idx = max(1, min(split_idx, len(use) - 1))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = _fit_model(X_train_s, y_train, n_splits=int(args.n_splits), max_iter=int(args.max_iter))

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else float("nan")

    APP_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, APP_DIR / "tail_model.pkl")
    joblib.dump(scaler, APP_DIR / "tail_scaler.pkl")

    report = {
        "tag": tag,
        "rows_used": int(len(use)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_cols": feature_cols,
        "auc": (round(auc, 6) if np.isfinite(auc) else None),
        "labels_src": src,
        "paths": {"model": str(APP_DIR / "tail_model.pkl"), "scaler": str(APP_DIR / "tail_scaler.pkl")},
    }
    (META_DIR / f"train_tail_report_{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("[DONE] train_tail_model.py")
    print("tag:", tag)
    print("AUC:", report["auc"])
    print("features:", feature_cols)
    print("=" * 60)


if __name__ == "__main__":
    main()