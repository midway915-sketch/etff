# scripts/score_models.py
from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
FEAT_DIR = DATA_DIR / "features"
APP_DIR = Path("app")

IN_PARQ = FEAT_DIR / "features_model.parquet"
IN_CSV = FEAT_DIR / "features_model.csv"

OUT_PARQ = FEAT_DIR / "features_scored.parquet"
OUT_CSV = FEAT_DIR / "features_scored.csv"

TAIL_MODEL = APP_DIR / "tail_model.pkl"
TAIL_SCALER = APP_DIR / "tail_scaler.pkl"
TAIL_META = APP_DIR / "tail_model_meta.json"

SUC_MODEL = APP_DIR / "success_model.pkl"
SUC_SCALER = APP_DIR / "success_scaler.pkl"
SUC_META = APP_DIR / "success_model_meta.json"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def load_meta(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing model meta: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def score_one(df: pd.DataFrame, model_path: Path, scaler_path: Path, meta_path: Path, out_col: str) -> pd.DataFrame:
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Missing model/scaler for {out_col}: {model_path}, {scaler_path}")

    meta = load_meta(meta_path)
    cols = meta.get("feature_cols", [])
    if not cols:
        raise RuntimeError(f"{meta_path} has empty feature_cols")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"features missing required columns for {out_col}: {missing}")

    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    Xs = scaler.transform(X)
    p = model.predict_proba(Xs)[:, 1]
    df[out_col] = pd.to_numeric(p, errors="coerce").fillna(0.0).astype(float)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parq", default=str(IN_PARQ), type=str)
    ap.add_argument("--in-csv", default=str(IN_CSV), type=str)
    ap.add_argument("--out-parq", default=str(OUT_PARQ), type=str)
    ap.add_argument("--out-csv", default=str(OUT_CSV), type=str)
    args = ap.parse_args()

    df = read_table(Path(args.in_parq), Path(args.in_csv)).copy()
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("features_model must include Date,Ticker")

    df["Date"] = norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # score both
    df = score_one(df, TAIL_MODEL, TAIL_SCALER, TAIL_META, out_col="p_tail")
    df = score_one(df, SUC_MODEL, SUC_SCALER, SUC_META, out_col="p_success")

    out_parq = Path(args.out_parq)
    out_csv = Path(args.out_csv)
    out_parq.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_parq, index=False)
    df.to_csv(out_csv, index=False)

    print(f"[DONE] wrote: {out_parq} rows={len(df)}")
    print("[INFO] p_tail/p_success summary:")
    print(df[["p_tail", "p_success"]].describe(percentiles=[0.01,0.05,0.1,0.5,0.9,0.95,0.99]).to_string())


if __name__ == "__main__":
    main()