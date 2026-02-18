# scripts/merge_gate_summaries.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    sig = Path("data/signals")
    files = sorted(sig.glob(f"sim_engine_summary_{args.tag}_*.json"))
    if not files:
        raise RuntimeError(f"No summary json found for tag={args.tag}")

    rows = []
    for p in files:
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if "final_equity" in df.columns:
        df = df.sort_values("final_equity", ascending=False)

    out = Path(args.out) if args.out else sig / f"sim_engine_summary_{args.tag}_GATES_ALL.csv"
    df.to_csv(out, index=False)
    print(f"[DONE] merged -> {out}")
    print(df[["label", "seed_multiple", "max_drawdown", "win_rate_closed", "trades_closed", "max_leverage_pct_closed"]]
          .head(20).to_string(index=False))


if __name__ == "__main__":
    main()
