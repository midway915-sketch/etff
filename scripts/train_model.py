#!/usr/bin/env python3
from __future__ import annotations

# ------------------------------------------------------------
# sys.path guard (avoid "No module named 'scripts'")
# ------------------------------------------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# ------------------------------------------------------------
# feature spec import (robust)
# ------------------------------------------------------------
try:
    from scripts.feature_spec import read_feature_cols_meta, get_feature_cols  # type: ignore
except Exception:
    try:
        from feature_spec import read_feature_cols_meta, get_feature_cols  # type: ignore
    except Exception:
        # last resort: minimal fallback (sector off)
        def read_feature_cols_meta():
            return ([], False)

        def get_feature_cols(sector_enabled: bool = False):
            base = [
                "Drawdown_252", "Drawdown_60", "ATR_ratio", "Z_score",
                "MACD_hist", "MA20_slope", "Market_Drawdown", "Market_ATR_ratio",
                "ret_score",
                "ret_5", "ret_10", "ret_20",
                "breakout_20", "vol_surge", "trend_align", "beta_60",
            ]
            if sector_enabled:
                base += ["Sector_Ret_20", "RelStrength"]
            return base

import argparse
import json

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
META_DIR = DATA_DIR / "meta"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Training data not found: {parq} (or {csv})")


def parse_csv_list(s: str) -> list[str]:
    items = [x.strip() for x in str(s or "").split(",")]
    return [x for x in items if x]


def resolve_feature_cols(args_features: str) -> tuple[list[str], str]:
    """
    Priority:
      1) --features (explicit override)
      2) data/meta/feature_cols.json (SSOT written by build_features.py)
      3) fallback SSOT default (sector disabled)
    """
    override = parse_csv_list(args_features)
    if override:
        return [str(c).strip() for c in override if str(c).strip()], "--features"

    cols_meta, _sector_enabled = read_feature_cols_meta()
    if cols_meta:
        return cols_meta, "data/meta/feature_cols.json"

    return get_feature_cols(sector_enabled=False), "feature_spec.py (fallback)"


def ensure_feature_columns_strict(df: pd.DataFrame, feat_cols: list[str], source_hint: str = "") -> None:
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        hint = f" (src={source_hint})" if source_hint else ""
        raise ValueError(f"Missing feature columns{hint}: {missing}")


def coerce_features_numeric(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feat_cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    return out


def write_train_report(tag: str, report: dict) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    p = META_DIR / (f"train_model_report_{tag}.json" if tag else "train_model_report.json")
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote train report -> {p}")


def _date_based_train_test_split(df: pd.DataFrame, date_col: str, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ✅ 룩어헤드/누수 방지용: row가 아니라 'Date' 단위로 split.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    uniq = pd.Series(dates.dropna().unique()).sort_values().to_list()
    if len(uniq) < 10:
        raise RuntimeError(f"Not enough unique dates for split: {len(uniq)}")

    cut_i = int(len(uniq) * float(train_ratio)) - 1
    cut_i = max(0, min(cut_i, len(uniq) - 2))  # ensure test has at least 1 date
    cut_date = pd.Timestamp(uniq[cut_i])

    train_df = df.loc[pd.to_datetime(df[date_col]) <= cut_date].copy()
    test_df = df.loc[pd.to_datetime(df[date_col]) > cut_date].copy()
    if len(train_df) < 50 or len(test_df) < 50:
        raise RuntimeError(f"Split too small. train={len(train_df)} test={len(test_df)} cut_date={cut_date.date()}")

    return train_df, test_df


def _make_date_cv_splits(df_train: pd.DataFrame, date_col: str, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    ✅ CV도 날짜 단위로만 진행 (같은 날짜가 train/val에 동시에 존재하지 않게).
    CalibratedClassifierCV(cv=...)에 넣을 수 있게 sample indices로 반환.
    """
    d = pd.to_datetime(df_train[date_col], errors="coerce").dt.tz_localize(None)
    uniq_dates = pd.Series(d.dropna().unique()).sort_values().to_list()
    if len(uniq_dates) < (n_splits + 2):
        raise RuntimeError(f"Not enough unique dates for TimeSeriesSplit: {len(uniq_dates)} (n_splits={n_splits})")

    # date -> ordinal id (0..U-1)
    date_to_id = {pd.Timestamp(x): i for i, x in enumerate(uniq_dates)}
    date_ids = d.map(lambda x: date_to_id.get(pd.Timestamp(x), -1)).to_numpy(dtype=int)

    tscv = TimeSeriesSplit(n_splits=int(n_splits))
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    # split on unique-date index space
    U = len(uniq_dates)
    dummy = np.arange(U)
    for tr_u, te_u in tscv.split(dummy):
        tr_mask = np.isin(date_ids, tr_u)
        te_mask = np.isin(date_ids, te_u)
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        splits.append((tr_idx, te_idx))

    if not splits:
        raise RuntimeError("Failed to build date-based CV splits.")
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", type=str, help="optional tag suffix for saving model files + meta report")

    ap.add_argument("--target-col", default="Success", type=str)
    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)

    ap.add_argument("--features", default="", type=str, help="comma-separated feature cols (override SSOT/meta)")

    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=500, type=int)

    ap.add_argument("--out-model", default="", type=str)
    ap.add_argument("--out-scaler", default="", type=str)

    args = ap.parse_args()

    df = read_table(LABELS_PARQ, LABELS_CSV).copy()

    date_col = args.date_col
    target_col = args.target_col
    ticker_col = args.ticker_col

    for c in [date_col, target_col]:
        if c not in df.columns:
            raise ValueError(f"labels_model missing required column: {c}")

    # normalize date + sort by Date then Ticker (deterministic)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    if ticker_col in df.columns:
        df[ticker_col] = df[ticker_col].astype(str).str.upper().str.strip()
        df = df.dropna(subset=[date_col]).sort_values([date_col, ticker_col]).reset_index(drop=True)
    else:
        df = df.dropna(subset=[date_col]).sort_values([date_col]).reset_index(drop=True)

    feat_cols, feat_src = resolve_feature_cols(args.features)
    feat_cols = [str(c).strip() for c in feat_cols if str(c).strip()]

    ensure_feature_columns_strict(
        df,
        feat_cols,
        source_hint=f"{feat_src}, labels_src={LABELS_PARQ if LABELS_PARQ.exists() else LABELS_CSV}",
    )

    df = coerce_features_numeric(df, feat_cols)

    # ✅ date-based split (no same-date mixing across train/test)
    train_df, test_df = _date_based_train_test_split(df, date_col=date_col, train_ratio=float(args.train_ratio))

    y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    X_train = train_df[feat_cols].to_numpy(dtype=float)

    y_test = pd.to_numeric(test_df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    X_test = test_df[feat_cols].to_numpy(dtype=float)

    if len(y_train) < 200:
        raise RuntimeError(f"Not enough training rows: {len(y_train)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(max_iter=int(args.max_iter))

    # ✅ date-based CV splits
    cv_splits = _make_date_cv_splits(train_df, date_col=date_col, n_splits=int(args.n_splits))

    # sklearn 버전 호환
    try:
        model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=cv_splits)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=cv_splits)

    model.fit(X_train_s, y_train)

    probs = model.predict_proba(X_test_s)[:, 1]
    auc = float("nan")
    if len(np.unique(y_test)) > 1:
        auc = float(roc_auc_score(y_test, probs))

    base_rate = float(np.mean(y_test)) if len(y_test) else float("nan")
    pred_mean = float(np.mean(probs)) if len(probs) else float("nan")

    print("=" * 60)
    print("[TRAIN] p_success model (date-split, date-CV)")
    print("rows_total:", len(df), "train:", len(train_df), "test:", len(test_df))
    print("AUC:", (round(auc, 6) if np.isfinite(auc) else "nan"))
    print("base_rate:", (round(base_rate, 6) if np.isfinite(base_rate) else "nan"))
    print("pred_mean:", (round(pred_mean, 6) if np.isfinite(pred_mean) else "nan"))
    print("feature_cols_source:", feat_src)
    print("feature_cols:", feat_cols)
    print("=" * 60)

    APP_DIR.mkdir(parents=True, exist_ok=True)
    tag = (args.tag or "").strip()

    out_model = Path(args.out_model) if args.out_model else (APP_DIR / (f"model_{tag}.pkl" if tag else "model.pkl"))
    out_scaler = Path(args.out_scaler) if args.out_scaler else (APP_DIR / (f"scaler_{tag}.pkl" if tag else "scaler.pkl"))

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)

    print(f"[DONE] saved model -> {out_model}")
    print(f"[DONE] saved scaler -> {out_scaler}")

    # train/test date range for traceability
    dtr0 = pd.to_datetime(train_df[date_col]).min()
    dtr1 = pd.to_datetime(train_df[date_col]).max()
    dte0 = pd.to_datetime(test_df[date_col]).min()
    dte1 = pd.to_datetime(test_df[date_col]).max()

    report = {
        "tag": tag,
        "target_col": target_col,
        "date_col": date_col,
        "ticker_col": ticker_col if ticker_col in df.columns else "",
        "feature_cols_source": feat_src,
        "feature_cols": feat_cols,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "train_date_range": [str(dtr0.date()) if pd.notna(dtr0) else "", str(dtr1.date()) if pd.notna(dtr1) else ""],
        "test_date_range": [str(dte0.date()) if pd.notna(dte0) else "", str(dte1.date()) if pd.notna(dte1) else ""],
        "auc": float(auc) if np.isfinite(auc) else None,
        "base_rate_test": float(base_rate) if np.isfinite(base_rate) else None,
        "pred_mean_test": float(pred_mean) if np.isfinite(pred_mean) else None,
        "out_model": str(out_model),
        "out_scaler": str(out_scaler),
    }
    write_train_report(tag, report)


if __name__ == "__main__":
    main()