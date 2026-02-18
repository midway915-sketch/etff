# scripts/get_start_date.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parq", default="data/features/features_model.parquet")
    ap.add_argument("--features-csv", default="data/features/features_model.csv")
    ap.add_argument("--recompute-days", type=int, default=20)
    args = ap.parse_args()

    p = Path(args.features_parq)
    c = Path(args.features_csv)

    df = None
    if p.exists():
        df = pd.read_parquet(p)
    elif c.exists():
        df = pd.read_csv(c)

    if df is None or len(df) == 0 or "Date" not in df.columns:
        print("")
        return

    d = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if len(d) == 0:
        print("")
        return

    last = d.max()
    start = last - pd.Timedelta(days=int(args.recompute_days))
    print(start.strftime("%Y-%m-%d"))

if __name__ == "__main__":
    main()
