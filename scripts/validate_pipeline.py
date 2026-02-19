# scripts/validate_pipeline.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def must_have(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate pipeline files/columns before running grid.")
    ap.add_argument("--prices-parq", default="data/raw/prices.parquet", type=str)
    ap.add_argument("--prices-csv", default="data/raw/prices.csv", type=str)

    ap.add_argument("--features-parq", default="data/features/features_model.parquet", type=str)
    ap.add_argument("--features-csv", default="data/features/features_model.csv", type=str)

    ap.add_argument("--scored-parq", default="data/features/features_scored.parquet", type=str)
    ap.add_argument("--scored-csv", default="data/features/features_scored.csv", type=str)

    ap.add_argument("--require-scored", action="store_true")
    ap.add_argument("--require-success-model", action="store_true")
    ap.add_argument("--require-tail-model", action="store_true")

    ap.add_argument("--success-model", default="app/model.pkl", type=str)
    ap.add_argument("--success-scaler", default="app/scaler.pkl", type=str)
    ap.add_argument("--tail-model", default="app/tail_model.pkl", type=str)

    args = ap.parse_args()

    prices = read_table(args.prices_parq, args.prices_csv)
    must_have(prices, ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"], "prices")

    feats = read_table(args.features_parq, args.features_csv)
    must_have(feats, ["Date", "Ticker"], "features_model")
    must_have(feats, ["Close", "High", "Low"], "features_model (for engine/labels)")

    scored_path_ok = Path(args.scored_parq).exists() or Path(args.scored_csv).exists()
    if args.require_scored and not scored_path_ok:
        raise FileNotFoundError(f"features_scored missing: {args.scored_parq} (or {args.scored_csv})")

    if scored_path_ok:
        scored = read_table(args.scored_parq, args.scored_csv)
        must_have(scored, ["Date", "Ticker", "ret_score", "p_success", "p_tail", "utility"], "features_scored")

    if args.require_success_model:
        if not Path(args.success_model).exists() or not Path(args.success_scaler).exists():
            raise FileNotFoundError(f"Missing success model/scaler: {args.success_model}, {args.success_scaler}")

    if args.require_tail_model:
        if not Path(args.tail_model).exists():
            raise FileNotFoundError(f"Missing tail model: {args.tail_model}")

    print("[OK] validate_pipeline passed")
    print(f"[INFO] prices rows={len(prices)}")
    print(f"[INFO] features_model rows={len(feats)} scored_exists={scored_path_ok}")


if __name__ == "__main__":
    main()