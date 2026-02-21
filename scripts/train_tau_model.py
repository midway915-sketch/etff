#!/usr/bin/env python3
from __future__ import annotations

# ✅ FIX(A): "python scripts/xxx.py" 실행에서도 scripts.* import 되도록 repo root를 sys.path에 추가
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from scripts.feature_spec import read_feature_cols_meta, get_feature_cols


DATA_DIR = Path("data")
META_DIR = DATA_DIR / "meta"
APP_DIR = Path("app")
LABELS_DIR = DATA_DIR / "labels"
FEATURES_DIR = DATA_DIR / "features"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def parse_csv_list(s: str) -> list[str]:
    items = [x.strip() for x in str(s or "").split(",")]
    return [x for x in items if x]


def resolve_feature_cols(args_features: str) -> tuple[list[str], str]:
    """
    Priority:
      1) --features (explicit override)
      2) data/meta/feature_cols.json (SSOT written by build_features.py)
      3) fallback SSOT default (sector disabled)
    Returns: (feature_cols, source_string)
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
        raise ValueError(
            f"Missing feature columns{hint}: {missing}\n"
            f"-> features_model/build_features와 labels_tau/build_tau_labels가 같은 SSOT feature_cols를 쓰도록 맞춰야 함."
        )


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


def _date_based_train_test_split(df: pd.DataFrame, date_col: str, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    ✅ row split 금지: Date 단위 split
    Returns: (train_df, test_df, cut_date_str)
    """
    d = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    uniq = pd.Series(d.dropna().unique()).sort_values().to_list()
    if len(uniq) < 10:
        raise RuntimeError(f"Not enough unique dates for split: {len(uniq)}")

    cut_i = int(len(uniq) * float(train_ratio)) - 1
    cut_i = max(0, min(cut_i, len(uniq) - 2))  # test 최소 1일 확보
    cut_date = pd.Timestamp(uniq[cut_i])

    train_df = df.loc[pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None) <= cut_date].copy()
    test_df = df.loc[pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None) > cut_date].copy()

    if len(train_df) < 50 or len(test_df) < 50:
        raise RuntimeError(f"Split too small. train={len(train_df)} test={len(test_df)} cut_date={cut_date.date()}")

    return train_df, test_df, str(cut_date.date())


def _make_date_cv_splits(df_train: pd.DataFrame, date_col: str, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    ✅ CV도 Date 단위로만 진행 (같은 날짜 혼재 방지)
    Returns list[(train_idx, val_idx)] for our manual fold loop.
    """
    d = pd.to_datetime(df_train[date_col], errors="coerce").dt.tz_localize(None)
    uniq_dates = pd.Series(d.dropna().unique()).sort_values().to_list()

    if len(uniq_dates) < 4:
        raise RuntimeError(f"Not enough unique dates for CV: {len(uniq_dates)}")

    # 데이터 길이에 맞게 splits 축소
    n_splits = int(n_splits)
    n_splits = min(n_splits, 5)
    if len(uniq_dates) < (n_splits + 2):
        n_splits = max(2, min(n_splits, len(uniq_dates) - 2))
    if n_splits < 2:
        n_splits = 2

    date_to_id = {pd.Timestamp(x): i for i, x in enumerate(uniq_dates)}
    date_ids = d.map(lambda x: date_to_id.get(pd.Timestamp(x), -1)).to_numpy(dtype=int)

    tscv = TimeSeriesSplit(n_splits=int(n_splits))
    U = len(uniq_dates)
    dummy = np.arange(U)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_u, va_u in tscv.split(dummy):
        tr_idx = np.where(np.isin(date_ids, tr_u))[0]
        va_idx = np.where(np.isin(date_ids, va_u))[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        splits.append((tr_idx, va_idx))

    if not splits:
        raise RuntimeError("Failed to build date-based CV splits.")
    return splits


def _coerce_tau_class(y: pd.Series) -> np.ndarray:
    """
    TauClass는 {0,1,2}만 허용.
    - NaN/이상값은 2로 폴백
    """
    yy = pd.to_numeric(y, errors="coerce").fillna(2).astype(int)
    yy = yy.clip(lower=0, upper=2)
    return yy.to_numpy(dtype=int)


def _make_logreg(max_iter: int) -> LogisticRegression:
    """
    ✅ sklearn 버전/빌드 차이로 LogisticRegression 인자 호환이 깨지는 것을 방지.
    - 가능한 인자만 signature 기반으로 필터링해서 전달한다.
    """
    desired = {
        # 기본
        "solver": "lbfgs",
        "max_iter": int(max_iter),

        # 최신 sklearn에서 다중 클래스에서 multinomial로 잘 가도록 (가능하면)
        "multi_class": "auto",

        # 일부 빌드/버전에서만 지원되는 옵션들 (있으면 넣고 없으면 무시)
        "n_jobs": 1,
    }

    sig = inspect.signature(LogisticRegression)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in desired.items() if k in allowed}

    # 혹시 'multi_class'가 지원되면 auto 넣고,
    # 지원 안 하면(이번 에러) 그냥 빼고 진행
    return LogisticRegression(**filtered)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train tau (FAST/MID/SLOW) multi-class model. (A + date-split/CV)")

    # ✅ tag는 계속 받되, tau labels 파일은 tagged/legacy 둘 다 허용 (workflow 호환)
    ap.add_argument("--tag", required=True, type=str, help="e.g. pt10_h40_sl10_ex20")

    ap.add_argument("--features-parq", default=str(FEATURES_DIR / "features_model.parquet"), type=str)
    ap.add_argument("--features-csv", default=str(FEATURES_DIR / "features_model.csv"), type=str)

    ap.add_argument("--date-col", default="Date", type=str)
    ap.add_argument("--ticker-col", default="Ticker", type=str)
    ap.add_argument("--target-col", default="TauClass", type=str)

    # ✅ 기본은 SSOT(meta) 사용. 필요할 때만 override.
    ap.add_argument("--features", default="", type=str, help="comma-separated feature cols (override SSOT/meta)")

    ap.add_argument("--train-ratio", default=0.8, type=float)
    ap.add_argument("--n-splits", default=3, type=int)
    ap.add_argument("--max-iter", default=800, type=int)

    ap.add_argument("--out-model", default="", type=str)
    ap.add_argument("--out-scaler", default="", type=str)

    args = ap.parse_args()

    tag = (args.tag or "").strip()
    if not tag:
        raise ValueError("--tag is required.")

    # ✅ tau labels path resolution
    tagged_parq = LABELS_DIR / f"labels_tau_{tag}.parquet"
    tagged_csv = LABELS_DIR / f"labels_tau_{tag}.csv"
    legacy_parq = LABELS_DIR / "labels_tau.parquet"
    legacy_csv = LABELS_DIR / "labels_tau.csv"

    labels_parq = tagged_parq
    labels_csv = tagged_csv
    labels_name = f"labels_tau_{tag}"

    if labels_parq.exists() or labels_csv.exists():
        pass
    elif legacy_parq.exists() or legacy_csv.exists():
        print(
            f"[WARN] tagged tau labels not found for tag={tag}. "
            f"Falling back to legacy tau labels: {legacy_parq} (or {legacy_csv})"
        )
        labels_parq, labels_csv = legacy_parq, legacy_csv
        labels_name = "labels_tau (legacy)"
    else:
        raise FileNotFoundError(
            f"Missing tau labels for tag={tag}.\n"
            f"-> expected (tagged): {tagged_parq} (or {tagged_csv})\n"
            f"-> or (legacy): {legacy_parq} (or {legacy_csv})\n"
            f"-> run: python scripts/build_tau_labels.py ... (and make it write labels_tau*.parquet/csv)"
        )

    feats = read_table(Path(args.features_parq), Path(args.features_csv)).copy()
    labs = read_table(labels_parq, labels_csv).copy()

    # basic schema checks
    for df, name in [(feats, "features_model"), (labs, labels_name)]:
        if args.date_col not in df.columns or args.ticker_col not in df.columns:
            raise KeyError(f"{name} must have {args.date_col},{args.ticker_col}")

    if args.target_col not in labs.columns:
        raise KeyError(f"{labels_name} missing target column: {args.target_col}")

    # normalize keys
    feats[args.date_col] = norm_date(feats[args.date_col])
    labs[args.date_col] = norm_date(labs[args.date_col])
    feats[args.ticker_col] = feats[args.ticker_col].astype(str).str.upper().str.strip()
    labs[args.ticker_col] = labs[args.ticker_col].astype(str).str.upper().str.strip()

    feats = feats.dropna(subset=[args.date_col, args.ticker_col]).copy()
    labs = labs.dropna(subset=[args.date_col, args.ticker_col, args.target_col]).copy()

    # ✅ SSOT feature cols
    feat_cols, feat_src = resolve_feature_cols(args.features)
    feat_cols = [c.strip() for c in feat_cols if c.strip()]

    # ✅ STRICT: features_model에 피처가 없으면 바로 에러
    feats_src = str(Path(args.features_parq)) if Path(args.features_parq).exists() else str(Path(args.features_csv))
    ensure_feature_columns_strict(feats, feat_cols, source_hint=f"{feat_src}, feats_src={feats_src}")

    # numeric coercion
    feats = coerce_features_numeric(feats, feat_cols)

    # merge
    df = pd.merge(
        feats[[args.date_col, args.ticker_col] + feat_cols].copy(),
        labs[[args.date_col, args.ticker_col, args.target_col]].copy(),
        on=[args.date_col, args.ticker_col],
        how="inner",
    )

    df = df.dropna(subset=[args.date_col, args.ticker_col, args.target_col]).copy()
    if df.empty:
        raise RuntimeError("No training rows after merge. Check labels_tau and features_model overlap.")

    # sort time
    df = df.sort_values(args.date_col).reset_index(drop=True)

    # targets / X
    y_all = _coerce_tau_class(df[args.target_col])
    if len(np.unique(y_all)) < 2:
        raise RuntimeError("TauClass has only one class in data. Cannot train model.")

    n = len(df)
    if n < 200:
        raise RuntimeError(f"Not enough training rows: {n}")

    # ✅ Date-based train/test split
    train_df, test_df, cut_date = _date_based_train_test_split(df, args.date_col, float(args.train_ratio))
    y_train = _coerce_tau_class(train_df[args.target_col])
    y_test = _coerce_tau_class(test_df[args.target_col])

    X_train = train_df[feat_cols].to_numpy(dtype=float)
    X_test = test_df[feat_cols].to_numpy(dtype=float)

    # scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # CV (date-based)
    splits = _make_date_cv_splits(train_df, args.date_col, int(args.n_splits))

    # ✅ model factory (sklearn 호환 필터)
    model = _make_logreg(int(args.max_iter))

    # Fold eval
    fold_losses: list[float] = []
    for i, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr = X_train_s[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train_s[va_idx]
        y_va = y_train[va_idx]

        m = _make_logreg(int(args.max_iter))
        m.fit(X_tr, y_tr)

        proba = m.predict_proba(X_va)
        ll = float(log_loss(y_va, proba, labels=[0, 1, 2]))
        fold_losses.append(ll)
        print(f"[INFO] fold={i} logloss={ll:.6f} n_tr={len(tr_idx)} n_va={len(va_idx)}")

    # Fit full train
    model.fit(X_train_s, y_train)

    # Test eval
    test_proba = model.predict_proba(X_test_s)
    test_ll = float(log_loss(y_test, test_proba, labels=[0, 1, 2]))

    # outputs
    out_model = Path(args.out_model) if args.out_model else (APP_DIR / "tau_model.pkl")
    out_scaler = Path(args.out_scaler) if args.out_scaler else (APP_DIR / "tau_scaler.pkl")
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_scaler.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_model)
    joblib.dump(scaler, out_scaler)
    print(f"[DONE] saved model: {out_model}")
    print(f"[DONE] saved scaler: {out_scaler}")

    # report
    report = {
        "tag": tag,
        "labels_source": str(labels_parq if labels_parq.exists() else labels_csv),
        "labels_name": labels_name,
        "features_source": str(Path(args.features_parq)) if Path(args.features_parq).exists() else str(Path(args.features_csv)),
        "feature_cols_source": feat_src,
        "feature_cols": feat_cols,
        "n_rows_total": int(n),
        "train_ratio": float(args.train_ratio),
        "cut_date": cut_date,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_splits": int(len(splits)),
        "cv_logloss_mean": float(np.mean(fold_losses)) if fold_losses else None,
        "cv_logloss_std": float(np.std(fold_losses)) if fold_losses else None,
        "test_logloss": float(test_ll),
        "class_counts_train": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()},
        "class_counts_test": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().to_dict().items()},
    }

    META_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = META_DIR / f"train_tau_report_{tag}.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[DONE] wrote report: {rep_path}")
    print(f"[INFO] test logloss={test_ll:.6f} (cut_date={cut_date})")


if __name__ == "__main__":
    main()