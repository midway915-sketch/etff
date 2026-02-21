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

# ✅ dedupe on/off (default: true)
DEDUP_PICKS="${DEDUP_PICKS:-true}"
DEDUP_PICKS="$(echo "$DEDUP_PICKS" | tr '[:upper:]' '[:lower:]' | xargs)"
if [[ "$DEDUP_PICKS" != "true" && "$DEDUP_PICKS" != "false" ]]; then
  echo "[ERROR] DEDUP_PICKS must be true/false (got: $DEDUP_PICKS)"
  exit 1
fi
echo "[INFO] DEDUP_PICKS=$DEDUP_PICKS"

# ✅ MAX_EXTEND_DAYS는 "있으면 사용", 없으면 H//2 자동계산
if [ -z "${MAX_EXTEND_DAYS:-}" ]; then
  MAX_EXTEND_DAYS="$(python - <<PY
H=int("${MAX_DAYS}")
print(max(1, H//2))
PY
)"
  export MAX_EXTEND_DAYS
fi
echo "[INFO] MAX_EXTEND_DAYS=$MAX_EXTEND_DAYS (auto: H//2 when not provided)"

# ✅ 옵션B 제거: tp1-trail-unlimited는 true만 사용
TP1_TRAIL_UNLIMITED="true"

# ✅ cap 모드 제어
# - 초기 탐색: CAP_COMPARE=false, TP1_HOLD_CAP_SINGLE=none (기본)
# - 전체 비교: CAP_COMPARE=true, TP1_HOLD_CAP_MODES=none,h2,total
CAP_COMPARE="${CAP_COMPARE:-false}"
CAP_COMPARE="$(echo "$CAP_COMPARE" | tr '[:upper:]' '[:lower:]' | xargs)"
if [[ "$CAP_COMPARE" != "true" && "$CAP_COMPARE" != "false" ]]; then
  echo "[ERROR] CAP_COMPARE must be true/false (got: $CAP_COMPARE)"
  exit 1
fi

TP1_HOLD_CAP_SINGLE="${TP1_HOLD_CAP_SINGLE:-none}"
TP1_HOLD_CAP_SINGLE="$(echo "$TP1_HOLD_CAP_SINGLE" | tr '[:upper:]' '[:lower:]' | xargs)"
if [[ "$TP1_HOLD_CAP_SINGLE" != "none" && "$TP1_HOLD_CAP_SINGLE" != "h2" && "$TP1_HOLD_CAP_SINGLE" != "total" ]]; then
  echo "[ERROR] TP1_HOLD_CAP_SINGLE must be none|h2|total (got: $TP1_HOLD_CAP_SINGLE)"
  exit 1
fi

TP1_HOLD_CAP_MODES="${TP1_HOLD_CAP_MODES:-none,h2,total}"
echo "[INFO] CAP_COMPARE=$CAP_COMPARE"
echo "[INFO] TP1_HOLD_CAP_SINGLE=$TP1_HOLD_CAP_SINGLE"
echo "[INFO] TP1_HOLD_CAP_MODES=$TP1_HOLD_CAP_MODES"

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

first_csv_item() {
  local s="$1"
  python - <<PY
s = """$s"""
parts=[p.strip() for p in s.split(",") if p.strip()]
print(parts[0] if parts else "")
PY
}

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
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).astype(str)
if "Ticker" in df.columns:
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

cols = [c for c in ["Date","Ticker"] if c in df.columns]
df = df[cols].dropna().sort_values(cols).reset_index(drop=True)

payload = df.to_csv(index=False).encode("utf-8")
print(hashlib.sha1(payload).hexdigest())
PY
}

# ----- print config
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"
echo "[INFO] PS_MINS=$PS_MINS MAX_LEVERAGE_PCT=$MAX_LEVERAGE_PCT"
echo "[INFO] TP1_TRAIL_UNLIMITED=$TP1_TRAIL_UNLIMITED"

DEFAULT_TMAX="$(first_csv_item "$P_TAIL_THRESHOLDS")"
DEFAULT_UQ="$(first_csv_item "$UTILITY_QUANTILES")"
if [ -z "$DEFAULT_TMAX" ] || [ -z "$DEFAULT_UQ" ]; then
  echo "[ERROR] P_TAIL_THRESHOLDS / UTILITY_QUANTILES must be non-empty CSV."
  exit 1
fi
echo "[INFO] DEFAULT_TMAX(for unused dim)=$DEFAULT_TMAX"
echo "[INFO] DEFAULT_UQ(for unused dim)=$DEFAULT_UQ"

HASH_DIR="$OUT_DIR/_picks_hash"
mkdir -p "$HASH_DIR"
seen_hash_file="$HASH_DIR/seen_hashes.txt"
touch "$seen_hash_file"

is_hash_seen() { local h="$1"; [ -n "$h" ] && grep -q "^$h$" "$seen_hash_file"; }
mark_hash_seen() { local h="$1"; [ -n "$h" ] && echo "$h" >> "$seen_hash_file"; }

# cap modes iterator
iter_cap_modes() {
  if [ "$CAP_COMPARE" = "true" ]; then
    split_csv "$TP1_HOLD_CAP_MODES"
  else
    echo "$TP1_HOLD_CAP_SINGLE"
  fi
}

# ----- main loops
while read -r mode; do
  mode="$(echo "$mode" | tr '[:upper:]' '[:lower:]' | xargs)"
  [ -z "$mode" ] && continue

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
              tu_s="tu1"  # ✅ 옵션B 제거 -> 항상 tu1

              base_suffix="${mode}_${tu_s}_t${t_s}_q${uq_s}_r${rank_by}_lam${lam_s}_ps${ps_s}_k${K}_w$(echo "$W" | tr ',' '_')_tp${tp_pct}_tr${tr_s}"

              echo "=============================="
              echo "[RUN] mode=$mode tail_max=$tmax u_q=$uq rank_by=$rank_by lambda=$LAMBDA_TAIL ps_min=$psmin topk=$K weights=$W trail=$trail base_suffix=$base_suffix"
              echo "=============================="

              # ---- PREDICT
              python "$PRED" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --mode "$mode" \
                --tail-threshold "$tmax" \
                --utility-quantile "$uq" \
                --rank-by "$rank_by" \
                --lambda-tail "$LAMBDA_TAIL" \
                --topk "$K" \
                --ps-min "$psmin" \
                --tag "$TAG" \
                --suffix "$base_suffix" \
                --exclude-tickers "$EXCLUDE_TICKERS" \
                --out-dir "$OUT_DIR" \
                --require-files "$REQUIRE_FILES"

              picks_path="$OUT_DIR/picks_${TAG}_gate_${base_suffix}.csv"
              if [ ! -f "$picks_path" ]; then
                echo "[WARN] picks missing -> skip simulate/summarize (base_suffix=$base_suffix)"
                continue
              fi

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
                echo "[INFO] picks rows=0 -> skip simulate/summarize (base_suffix=$base_suffix)"
                continue
              fi

              # ---- DEDUPE (log only)
              if [ "$DEDUP_PICKS" = "true" ]; then
                h="$(picks_hash "$picks_path")"
                if is_hash_seen "$h"; then
                  echo "[INFO] duplicate picks hash=$h -> proceed (base_suffix=$base_suffix)"
                else
                  mark_hash_seen "$h"
                fi
              fi

              # ---- SIMULATE + SUMMARIZE (cap modes)
              while read -r cap_mode; do
                cap_mode="$(echo "$cap_mode" | tr '[:upper:]' '[:lower:]' | xargs)"
                [ -z "$cap_mode" ] && continue
                if [[ "$cap_mode" != "none" && "$cap_mode" != "h2" && "$cap_mode" != "total" ]]; then
                  echo "[ERROR] invalid cap_mode=$cap_mode (must be none|h2|total)"
                  exit 1
                fi

                cap_suffix="${base_suffix}_cap${cap_mode}"

                echo "------------------------------"
                echo "[SIM] cap_mode=$cap_mode cap_suffix=$cap_suffix"
                echo "------------------------------"

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
                  --tp1-trail-unlimited "$TP1_TRAIL_UNLIMITED" \
                  --tp1-hold-cap "$cap_mode" \
                  --topk "$K" \
                  --weights "$W" \
                  --tag "$TAG" \
                  --suffix "$cap_suffix" \
                  --out-dir "$OUT_DIR"

                trades_path="$OUT_DIR/sim_engine_trades_${TAG}_gate_${cap_suffix}.parquet"

                python "$SUM" \
                  --trades-path "$trades_path" \
                  --tag "$TAG" \
                  --suffix "$cap_suffix" \
                  --profit-target "$PROFIT_TARGET" \
                  --max-days "$MAX_DAYS" \
                  --stop-level "$STOP_LEVEL" \
                  --max-extend-days "$MAX_EXTEND_DAYS" \
                  --out-dir "$OUT_DIR"

              done < <(iter_cap_modes)

            done < <(split_csv "$TRAIL_STOPS")
          done < <(split_scsv "$TOPK_CONFIGS")
        done < <(split_csv "$PS_MINS")
      done < <(split_csv "$RANK_METRICS")
    done < <(split_csv "$UQ_LIST")
  done < <(split_csv "$TMAX_LIST")

done < <(split_csv "$GATE_MODES")

echo "[DONE] grid finished"
if [ "$DEDUP_PICKS" = "true" ]; then
  echo "[INFO] unique picks hashes: $(wc -l < "$seen_hash_file" | tr -d ' ')"
fi
ls -la "$OUT_DIR" | sed -n '1,200p'