#!/usr/bin/env python3
from __future__ import annotations

# ✅ sys.path guard
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


DATA_DIR = Path("data")
LABELS_DIR = DATA_DIR / "labels"
FEATURES_DIR = DATA_DIR / "features"
META_DIR = DATA_DIR / "meta"
APP_DIR = Path("app")


def read_table(parq: Path, csv: Path) -> tuple[pd.DataFrame, str]:
    if parq.exists():
        return pd.read_parquet(parq).copy(), str(parq)
    if csv.exists():
        return pd.read_csv(csv).copy(), str(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def resolve_feature_cols_ssot() -> tuple[list[str], str]:
    cols, _sector_enabled = read_feature_cols_meta()
    if cols:
        return cols, "data/meta/feature_cols.json"
    cols = get_feature_cols(sector_enabled=False)
    return cols, "scripts/feature_spec.py:get_feature_cols(sector_enabled=False)"


def pick_tail_target_column(df: pd.DataFrame, src: str) -> str:
    """
    ✅ FIX: label 파일에 TailTarget이 아니라 p_tail이 있을 수 있음.
    우선순위:
      1) p_tail
      2) TailTarget
      3) tail_target
    """
    candidates = ["p_tail", "TailTarget", "tail_target"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Tail target column missing. expected one of {candidates} (labels_src={src}) "
        f"cols={list(df.columns)[:30]}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train tail model (p_tail classifier) aligned with SSOT features.")
    ap.add_argument("--profit-target", type=float, required=True)
    ap.add_argument("--max-days", type=int, required=True)
    ap.add_argument("--stop-level", type=float, required=True)
    ap.add_argument("--max-extend-days", type=int, required=True)

    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--max-iter", type=int, default=800)

    args = ap.parse_args()

    pt100 = int(round(float(args.profit_target) * 100))
    H = int(args.max_days)
    sl100 = int(round(abs(float(args.stop_level)) * 100))
    ex = int(args.max_extend_days)

    tag = f"pt{pt100}_h{H}_sl{sl100}_ex{ex}"

    labels_parq = LABELS_DIR / f"labels_tail_{tag}.parquet"
    labels_csv = LABELS_DIR / f"labels_tail_{tag}.csv"

    feats_parq = FEATURES_DIR / "features_model.parquet"
    feats_csv = FEATURES_DIR / "features_model.csv"

    labels_df, src = read_table(labels_parq, labels_csv)
    feats_df, feats_src = read_table(feats_parq, feats_csv)

    # normalize keys
    for df in (labels_df, feats_df):
        df["Date"] = norm_date(df["Date"])
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    labels_df = labels_df.dropna(subset=["Date", "Ticker"]).copy()
    feats_df = feats_df.dropna(subset=["Date", "Ticker"]).copy()

    # ✅ pick correct target
    target_col = pick_tail_target_column(labels_df, src)

    # SSOT feature cols
    feat_cols, feat_src = resolve_feature_cols_ssot()
    feat_cols = [c.strip() for c in feat_cols if c.strip()]
    if len(feat_cols) != 18:
        raise RuntimeError(f"SSOT feature cols must be 18, got {len(feat_cols)} from {feat_src}: {feat_cols}")
    if "RelStrength" not in feat_cols:
        raise RuntimeError(f"SSOT must include RelStrength (src={feat_src}). got={feat_cols}")

    missing = [c for c in feat_cols if c not in feats_df.columns]
    if missing:
        raise RuntimeError(
            f"features_model missing SSOT columns: {missing}\n"
            f"-> features_src={feats_src}\n"
            f"-> ssot_src={feat_src}"
        )

    # merge
    df = feats_df[["Date", "Ticker"] + feat_cols].merge(
        labels_df[["Date", "Ticker", target_col]],
        on=["Date", "Ticker"],
        how="inner",
    )
    if df.empty:
        raise RuntimeError("No overlap between features_model and labels_tail. Check date/ticker ranges.")

    # numeric coercion
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"Tail target has only one class after merge. target_col={target_col}")

    df = df.sort_values("Date").reset_index(drop=True)
    X = df[feat_cols].to_numpy(dtype=float)

    # time split (by row after sorting; acceptable since Date is sorted and join is stable)
    n = len(df)
    cut = int(n * float(args.train_ratio))
    cut = max(100, min(cut, n - 100))

    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # LogisticRegression (keep args minimal for maximum compatibility)
    model = LogisticRegression(solver="lbfgs", max_iter=int(args.max_iter))
    model.fit(X_train_s, y_train)

    proba = model.predict_proba(X_test_s)[:, 1]
    ll = float(log_loss(y_test, np.vstack([1 - proba, proba]).T, labels=[0, 1]))

    APP_DIR.mkdir(parents=True, exist_ok=True)
    model_path = APP_DIR / "tail_model.pkl"
    scaler_path = APP_DIR / "tail_scaler.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    report = {
        "tag": tag,
        "labels_source": src,
        "features_source": feats_src,
        "ssot_feature_cols_source": feat_src,
        "feature_cols": feat_cols,
        "target_col_used": target_col,
        "n_rows": int(n),
        "train_ratio": float(args.train_ratio),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_logloss": ll,
        "class_counts_train": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()},
        "class_counts_test": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().to_dict().items()},
    }

    META_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = META_DIR / f"train_tail_report_{tag}.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved model: {model_path}")
    print(f"[DONE] saved scaler: {scaler_path}")
    print(f"[DONE] wrote report: {rep_path}")
    print(f"[INFO] test logloss={ll:.6f} target_col={target_col}")


if __name__ == "__main__":
    main()
