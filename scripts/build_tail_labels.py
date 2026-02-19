# scripts/build_tail_labels.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

LABEL_DIR = Path("data/labels")


def fmt_tag(pt: float, max_days: int, sl: float, ex: int) -> str:
    pt_tag = int(round(pt * 100))
    sl_tag = int(round(abs(sl) * 100))
    return f"pt{pt_tag}_h{max_days}_sl{sl_tag}_ex{ex}"


def read_table(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing file: {parq} (or {csv})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build labels_tail_{tag} (TailTarget) from strategy_raw_data_{tag}.")
    ap.add_argument("--profit-target", required=True, type=float)
    ap.add_argument("--max-days", required=True, type=int)
    ap.add_argument("--stop-level", required=True, type=float)
    ap.add_argument("--max-extend-days", required=True, type=int)
    args = ap.parse_args()

    tag = fmt_tag(float(args.profit_target), int(args.max_days), float(args.stop_level), int(args.max_extend_days))

    src_parq = LABEL_DIR / f"strategy_raw_data_{tag}.parquet"
    src_csv = LABEL_DIR / f"strategy_raw_data_{tag}.csv"

    if not src_parq.exists() and not src_csv.exists():
        raise FileNotFoundError(
            f"Missing {src_parq} (or {src_csv}). "
            f"Run scripts/build_strategy_labels.py first."
        )

    df = read_table(src_parq, src_csv).copy()

    need = {"Date", "Ticker", "Tail"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"strategy_raw_data_{tag} missing columns: {missing}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()

    # Tail -> TailTarget (0/1)
    out["TailTarget"] = pd.to_numeric(out["Tail"], errors="coerce").fillna(0).astype(int)

    # train_tail_model.py의 choose_features()가 numeric feature도 고를 수 있게
    # Tail 컬럼은 제거하고 나머지는 그대로 둠
    out = out.drop(columns=["Tail"], errors="ignore")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    out_parq = LABEL_DIR / f"labels_tail_{tag}.parquet"
    out_csv = LABEL_DIR / f"labels_tail_{tag}.csv"

    try:
        out.to_parquet(out_parq, index=False)
        print(f"[DONE] wrote: {out_parq} rows={len(out)}")
    except Exception:
        out.to_csv(out_csv, index=False)
        print(f"[DONE] wrote: {out_csv} rows={len(out)}")


if __name__ == "__main__":
    main()