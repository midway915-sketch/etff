#!/usr/bin/env bash
set -euo pipefail

PRED="scripts/predict_gate.py"
SIM="scripts/simulate_single_position_engine.py"
SUM="scripts/summarize_sim_trades.py"

OUT_DIR="${OUT_DIR:-data/signals}"
mkdir -p "$OUT_DIR"

TAG="${LABEL_KEY:-run}"
echo "[INFO] TAG=$TAG"
echo "[INFO] OUT_DIR=$OUT_DIR"

# Required envs
: "${PROFIT_TARGET:?}"
: "${MAX_DAYS:?}"
: "${STOP_LEVEL:?}"
: "${MAX_EXTEND_DAYS:?}"
: "${P_TAIL_THRESHOLDS:?}"
: "${UTILITY_QUANTILES:?}"
: "${RANK_METRICS:?}"
: "${LAMBDA_TAIL:?}"
: "${GATE_MODES:?}"
: "${TRAIL_STOPS:?}"
: "${TP1_FRAC:?}"
: "${ENABLE_TRAILING:?}"
: "${TOPK_CONFIGS:?}"
: "${PS_MINS:?}"
: "${MAX_LEVERAGE_PCT:?}"
: "${EXCLUDE_TICKERS:?}"
: "${REQUIRE_FILES:?}"

# ----- helpers
split_csv() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(",") if p.strip()]
for p in parts:
  print(p)
PY
}

split_scsv() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(";") if p.strip()]
for p in parts:
  print(p)
PY
}

suffix_float() {
  python - <<PY
x=float("$1")
print(str(x).replace(".","p").replace("-","m"))
PY
}

# pick first item in a csv list (non-empty)
first_csv_item() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(",") if p.strip()]
print(parts[0] if parts else "")
PY
}

# compute stable hash for picks content (Date,Ticker only)
picks_hash() {
  local file="$1"
  python - <<PY
import hashlib, pandas as pd
from pathlib import Path

p = Path("$file")
if not p.exists():
    print("")
    raise SystemExit(0)

df = pd.read_csv(p)
# normalize
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).astype(str)
if "Ticker" in df.columns:
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
cols = [c for c in ["Date","Ticker"] if c in df.columns]
df = df[cols].dropna().sort_values(cols).reset_index(drop=True)

payload = df.to_csv(index=False).encode("utf-8")
h = hashlib.sha1(payload).hexdigest()
print(h)
PY
}

# ----- print config
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"
echo "[INFO] PS_MINS=$PS_MINS MAX_LEVERAGE_PCT=$MAX_LEVERAGE_PCT"

# representative defaults for "unused dims" per mode
DEFAULT_TMAX="$(first_csv_item "$P_TAIL_THRESHOLDS")"
DEFAULT_UQ="$(first_csv_item "$UTILITY_QUANTILES")"

if [ -z "$DEFAULT_TMAX" ] || [ -z "$DEFAULT_UQ" ]; then
  echo "[ERROR] P_TAIL_THRESHOLDS / UTILITY_QUANTILES must be non-empty CSV."
  exit 1
fi

echo "[INFO] DEFAULT_TMAX(for unused dim)=$DEFAULT_TMAX"
echo "[INFO] DEFAULT_UQ(for unused dim)=$DEFAULT_UQ"

# hash registry for dedupe
HASH_DIR="$OUT_DIR/_picks_hash"
mkdir -p "$HASH_DIR"

seen_hash_file="$HASH_DIR/seen_hashes.txt"
touch "$seen_hash_file"

is_hash_seen() {
  local h="$1"
  if [ -z "$h" ]; then
    return 1
  fi
  grep -q "^$h$" "$seen_hash_file"
}

mark_hash_seen() {
  local h="$1"
  if [ -n "$h" ]; then
    echo "$h" >> "$seen_hash_file"
  fi
}

# ----- main loops
while read -r mode; do
  mode="$(echo "$mode" | tr '[:upper:]' '[:lower:]' | xargs)"
  if [ -z "$mode" ]; then
    continue
  fi

  # mode별로 "의미 있는 차원"만 돌리기
  if [ "$mode" = "none" ]; then
    TMAX_LIST="$DEFAULT_TMAX"
    UQ_LIST="$DEFAULT_UQ"
  elif [ "$mode" = "tail" ]; then
    TMAX_LIST="$P_TAIL_THRESHOLDS"
    UQ_LIST="$DEFAULT_UQ"
  elif [ "$mode" = "utility" ]; then
    TMAX_LIST="$DEFAULT_TMAX"
    UQ_LIST="$UTILITY_QUANTILES"
  elif [ "$mode" = "tail_utility" ]; then
    TMAX_LIST="$P_TAIL_THRESHOLDS"
    UQ_LIST="$UTILITY_QUANTILES"
  else
    echo "[ERROR] Unknown mode: $mode"
    exit 1
  fi

  while read -r tmax; do
    while read -r uq; do
      while read -r rank_by; do
        while read -r psmin; do
          while read -r topk_line; do
            K="${topk_line%%|*}"
            W="${topk_line#*|}"

            while read -r trail; do
              t_s="$(suffix_float "$tmax")"
              uq_s="$(suffix_float "$uq")"
              lam_s="$(suffix_float "$LAMBDA_TAIL")"
              ps_s="$(suffix_float "$psmin")"
              tr_s="$(suffix_float "$trail")"
              tp_pct="$(python - <<PY
f=float("$TP1_FRAC")
print(int(round(f*100)))
PY
)"

              suffix="${mode}_t${t_s}_q${uq_s}_r${rank_by}_lam${lam_s}_ps${ps_s}_k${K}_w$(echo "$W" | tr ',' '_')_tp${tp_pct}_tr${tr_s}"

              echo "=============================="
              echo "[RUN] mode=$mode tail_max=$tmax u_q=$uq rank_by=$rank_by lambda=$LAMBDA_TAIL ps_min=$psmin topk=$K weights=$W trail=$trail suffix=$suffix"
              echo "=============================="

              # 1) predict picks
              python "$PRED" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --mode "$mode" \
                --tag "$TAG" \
                --suffix "$suffix" \
                --tail-threshold "$tmax" \
                --utility-quantile "$uq" \
                --rank-by "$rank_by" \
                --lambda-tail "$LAMBDA_TAIL" \
                --topk "$K" \
                --ps-min "$psmin" \
                --exclude-tickers "$EXCLUDE_TICKERS" \
                --out-dir "$OUT_DIR" \
                --require-files "$REQUIRE_FILES"

              picks_path="$OUT_DIR/picks_${TAG}_gate_${suffix}.csv"
              meta_path="$OUT_DIR/picks_meta_${TAG}_gate_${suffix}.json"

              # ---- hard skip: no picks rows
              if [ ! -f "$picks_path" ]; then
                echo "[WARN] picks missing -> skip simulate/summarize (suffix=$suffix)"
                continue
              fi

              # quick check: empty picks file
              rows="$(python - <<PY
import pandas as pd
try:
  df=pd.read_csv("$picks_path")
  print(len(df))
except Exception:
  print(0)
PY
)"
              if [ "${rows:-0}" = "0" ]; then
                echo "[INFO] picks rows=0 -> skip simulate/summarize (suffix=$suffix)"
                continue
              fi

              # ---- dedupe by picks content hash
              h="$(picks_hash "$picks_path")"
              if is_hash_seen "$h"; then
                echo "[INFO] duplicate picks hash=$h -> skip simulate/summarize (suffix=$suffix)"
                # 그래도 meta는 남겨둠(이미 생성됨)
                continue
              fi
              mark_hash_seen "$h"

              # 2) simulate
              python "$SIM" \
                --picks-path "$picks_path" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --max-leverage-pct "$MAX_LEVERAGE_PCT" \
                --enable-trailing "$ENABLE_TRAILING" \
                --tp1-frac "$TP1_FRAC" \
                --trail-stop "$trail" \
                --topk "$K" \
                --weights "$W" \
                --tag "$TAG" \
                --suffix "$suffix" \
                --out-dir "$OUT_DIR"

              trades_path="$OUT_DIR/sim_engine_trades_${TAG}_gate_${suffix}.parquet"

              # 3) summarize
              python "$SUM" \
                --trades-path "$trades_path" \
                --tag "$TAG" \
                --suffix "$suffix" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --out-dir "$OUT_DIR"

            done < <(split_csv "$TRAIL_STOPS")
          done < <(split_scsv "$TOPK_CONFIGS")
        done < <(split_csv "$PS_MINS")
      done < <(split_csv "$RANK_METRICS")
    done < <(split_csv "$UQ_LIST")
  done < <(split_csv "$TMAX_LIST")

done < <(split_csv "$GATE_MODES")

echo "[DONE] grid finished"
echo "[INFO] unique picks hashes: $(wc -l < "$seen_hash_file" | tr -d ' ')"
ls -la "$OUT_DIR" | sed -n '1,200p'
