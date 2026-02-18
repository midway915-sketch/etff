# scripts/build_tau_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


FEATURES_PARQ = "data/features/features_model.parquet"
FEATURES_CSV  = "data/features/features_model.csv"

PRICES_PARQ = "data/raw/prices.parquet"
PRICES_CSV  = "data/raw/prices.csv"

OUT_PARQ = "data/labels/labels_tau.parquet"
OUT_CSV  = "data/labels/labels_tau.csv"


def read_table(parq: str, csv: str) -> pd.DataFrame:
    p = Path(parq)
    c = Path(csv)
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def compute_tau_labels(prices: pd.DataFrame, profit_target: float, max_days: int, k1: int, k2: int) -> pd.DataFrame:
    """
    τ 라벨(구간 분류):
      - TauClass=0: k1일 이내 +PT 도달
      - TauClass=1: k2일 이내 +PT 도달 (단, k1은 실패)
      - TauClass=2: max_days 이내 +PT 도달 (단, k2는 실패)
      - TauClass=3: max_days 이내 미도달(실패/매우 느림)

    성공 판정은 "미래 High가 Close*(1+PT) 이상이었는가"로 단순화.
    (엔진의 intraday TP 가정과 일치)
    """

    df = prices.copy()
    df["Date"] = norm_date(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    need_cols = ["Date", "Ticker", "Close", "High"]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(f"prices missing required column: {c}")

    df = df.dropna(subset=["Date", "Ticker", "Close", "High"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # target price per row
    df["TargetPx"] = df["Close"] * (1.0 + float(profit_target))

    # groupwise rolling forward max of High for horizons
    def fwd_max_high(x: pd.Series, horizon: int) -> np.ndarray:
        # forward window: [t+1, ..., t+horizon]
        # implement via reverse rolling max
        a = x.to_numpy(dtype=float)
        n = len(a)
        out = np.full(n, np.nan, dtype=float)
        # reverse
        rev = a[::-1]
        # rolling max on reverse corresponds to forward max
        s = pd.Series(rev)
        # include current index in window, so we shift by 1 to exclude "today" if you want.
        # We'll include today as well (conservative? actually slightly optimistic).
        # To exclude today, use s.shift(1).rolling(h).max()
        rmax = s.shift(1).rolling(window=horizon, min_periods=1).max().to_numpy()
        out = rmax[::-1]
        return out

    k1 = int(k1)
    k2 = int(k2)
    max_days = int(max_days)

    g = df.groupby("Ticker", sort=False)

    df[f"FwdMaxHigh_{k1}"] = g["High"].transform(lambda s: fwd_max_high(s, k1))
    df[f"FwdMaxHigh_{k2}"] = g["High"].transform(lambda s: fwd_max_high(s, k2))
    df[f"FwdMaxHigh_{max_days}"] = g["High"].transform(lambda s: fwd_max_high(s, max_days))

    hit1 = (df[f"FwdMaxHigh_{k1}"] >= df["TargetPx"])
    hit2 = (df[f"FwdMaxHigh_{k2}"] >= df["TargetPx"])
    hitH = (df[f"FwdMaxHigh_{max_days}"] >= df["TargetPx"])

    tau = np.full(len(df), 3, dtype=int)
    tau[hitH.to_numpy()] = 2
    tau[hit2.to_numpy()] = 1
    tau[hit1.to_numpy()] = 0

    df["TauClass"] = tau
    df["TauHit_k1"] = hit1.astype(int)
    df["TauHit_k2"] = hit2.astype(int)
    df["TauHit_H"]  = hitH.astype(int)

    out = df[["Date", "Ticker", "TauClass", "TauHit_k1", "TauHit_k2", "TauHit_H"]].copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--k1", default=10, type=int)
    ap.add_argument("--k2", default=20, type=int)
    ap.add_argument("--start-date", default="", type=str)
    args = ap.parse_args()

    prices = read_table(PRICES_PARQ, PRICES_CSV)

    labels = compute_tau_labels(
        prices=prices,
        profit_target=float(args.profit_target),
        max_days=int(args.max_days),
        k1=int(args.k1),
        k2=int(args.k2),
    )

    if args.start_date:
        sd = pd.to_datetime(args.start_date, errors="coerce")
        if pd.notna(sd):
            labels = labels[labels["Date"] >= sd].copy()

    Path("data/labels").mkdir(parents=True, exist_ok=True)
    labels.to_parquet(OUT_PARQ, index=False)
    labels.to_csv(OUT_CSV, index=False)

    print(f"[DONE] wrote: {OUT_PARQ} rows={len(labels)}")
    vc = labels["TauClass"].value_counts(dropna=False).sort_index()
    print("[INFO] TauClass counts:", vc.to_dict())


if __name__ == "__main__":
    main()