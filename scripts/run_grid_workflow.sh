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

# ----- print config
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"
echo "[INFO] PS_MINS=$PS_MINS MAX_LEVERAGE_PCT=$MAX_LEVERAGE_PCT"

# ----- main loops
while read -r mode; do
  while read -r tmax; do
    while read -r uq; do
      while read -r rank_by; do
        while read -r psmin; do
          while read -r topk_line; do
            # topk_line format: "K|w1,w2"
            K="${topk_line%%|*}"
            W="${topk_line#*|}"

            while read -r trail; do
              # suffix compose
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

              # 1) predict picks (TopK rows per date)
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

              # 2) simulate (Top-1 or Top-2) + TP1 + trailing + leverage cap
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

              # 3) summarize -> gate_summary_*.csv
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
    done < <(split_csv "$UTILITY_QUANTILES")
  done < <(split_csv "$P_TAIL_THRESHOLDS")
done < <(split_csv "$GATE_MODES")

echo "[DONE] grid finished"
ls -la "$OUT_DIR" | sed -n '1,200p'