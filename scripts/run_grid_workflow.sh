#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Gate Grid Runner (robust)
# - loops: gate mode / tail_max / utility_q / rank_by / trail / topk
# - calls: predict_gate.py -> simulate_single_position_engine.py -> summarize_sim_trades.py
# - writes: picks_*.csv, sim_engine_trades_*.parquet, sim_engine_curve_*.parquet, gate_summary_*.csv
# ============================================================

# -------- helpers
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; echo "$s"; }

# Split comma-separated string into array (ignores empty tokens)
split_csv() {
  local s; s="$(trim "${1:-}")"
  local IFS=','; read -r -a _arr <<< "$s"
  local out=()
  for x in "${_arr[@]}"; do
    x="$(trim "$x")"
    [[ -n "$x" ]] && out+=("$x")
  done
  printf '%s\n' "${out[@]}"
}

# Replace '.' with 'p' and '-' with 'm' for suffix safety
fmt_num_for_suffix() {
  local x="$1"
  x="${x//- /-}"
  x="${x//-/m}"
  x="${x//./p}"
  echo "$x"
}

# Ensure a file exists (or stop)
require_path() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] required file missing: $p" >&2
    exit 1
  fi
}

# -------- env defaults (safe)
: "${OUT_DIR:=data/signals}"
: "${EXCLUDE_TICKERS:=SPY,^VIX}"

: "${LABEL_KEY:=pt10_h40_sl10_ex30}"
: "${PROFIT_TARGET:=0.10}"
: "${MAX_DAYS:=40}"
: "${STOP_LEVEL:=-0.10}"
: "${MAX_EXTEND_DAYS:=30}"

: "${P_TAIL_THRESHOLDS:=0.10,0.20,0.30}"
: "${UTILITY_QUANTILES:=0.60,0.75,0.90}"
: "${RANK_METRICS:=utility,ret_score}"
: "${LAMBDA_TAIL:=0.05}"
: "${GATE_MODES:=none,tail,utility,tail_utility}"

# 2-step exit + trailing
: "${ENABLE_TRAILING:=true}"
: "${TP1_FRAC:=0.50}"
: "${TRAIL_STOPS:=0.08,0.10,0.12}"

# top-k configs: "1|1.0;2|0.7,0.3;2|0.6,0.4"
: "${TOPK_CONFIGS:=1|1.0;2|0.7,0.3;2|0.6,0.4}"

# If you want strict pre-check of required files, set this env (single string)
# Example: REQUIRE_FILES="data/features/features_model.parquet,app/model.pkl,app/scaler.pkl"
: "${REQUIRE_FILES:=}"

mkdir -p "$OUT_DIR"

echo "[INFO] TAG=$LABEL_KEY"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOPK_CONFIGS=$TOPK_CONFIGS"
echo "[INFO] TRAIL_STOPS=$TRAIL_STOPS TP1_FRAC=$TP1_FRAC ENABLE_TRAILING=$ENABLE_TRAILING"
echo "[INFO] REQUIRE_FILES=${REQUIRE_FILES:-<empty>}"

# -------- pick script paths (supports ./scripts or repo root)
if [[ -f scripts/predict_gate.py ]]; then
  PRED="scripts/predict_gate.py"
elif [[ -f predict_gate.py ]]; then
  PRED="predict_gate.py"
else
  echo "[ERROR] predict_gate.py not found" >&2
  exit 1
fi

if [[ -f scripts/simulate_single_position_engine.py ]]; then
  SIM="scripts/simulate_single_position_engine.py"
elif [[ -f simulate_single_position_engine.py ]]; then
  SIM="simulate_single_position_engine.py"
else
  echo "[ERROR] simulate_single_position_engine.py not found" >&2
  exit 1
fi

if [[ -f scripts/summarize_sim_trades.py ]]; then
  SUM="scripts/summarize_sim_trades.py"
elif [[ -f summarize_sim_trades.py ]]; then
  SUM="summarize_sim_trades.py"
else
  echo "[ERROR] summarize_sim_trades.py not found" >&2
  exit 1
fi

# Optional strict checks (ONLY if REQUIRE_FILES is non-empty)
if [[ -n "${REQUIRE_FILES}" ]]; then
  # validate each path exists
  while IFS= read -r p; do
    require_path "$p"
  done < <(split_csv "$REQUIRE_FILES")
fi

# -------- parse lists
mapfile -t TAILS < <(split_csv "$P_TAIL_THRESHOLDS")
mapfile -t UQS   < <(split_csv "$UTILITY_QUANTILES")
mapfile -t RANKS < <(split_csv "$RANK_METRICS")
mapfile -t MODES < <(split_csv "$GATE_MODES")
mapfile -t TRAILS < <(split_csv "$TRAIL_STOPS")

# Parse topk configs separated by ';'
IFS=';' read -r -a TOPKS <<< "$(trim "$TOPK_CONFIGS")"

# -------- main loops
for mode in "${MODES[@]}"; do
  for tail in "${TAILS[@]}"; do
    for uq in "${UQS[@]}"; do
      for rank_by in "${RANKS[@]}"; do
        # gate thresholds apply depending on mode, but we still pass them consistently
        for topk_item in "${TOPKS[@]}"; do
          topk_item="$(trim "$topk_item")"
          [[ -z "$topk_item" ]] && continue
          k="${topk_item%%|*}"
          w="${topk_item#*|}"

          # trailing variants
          if [[ "$ENABLE_TRAILING" == "true" ]]; then
            for tr in "${TRAILS[@]}"; do
              suffix="${mode}_t0$(fmt_num_for_suffix "$tail")_q0$(fmt_num_for_suffix "$uq")_r${rank_by}_lam$(fmt_num_for_suffix "$LAMBDA_TAIL")_k${k}_w$(fmt_num_for_suffix "$w")_tp$(fmt_num_for_suffix "$TP1_FRAC")_tr0$(fmt_num_for_suffix "$tr")"
              echo "=============================="
              echo "[RUN] mode=$mode tail_max=$tail u_q=$uq rank_by=$rank_by lambda=$LAMBDA_TAIL topk=$k weights=$w trail=$tr suffix=$suffix"
              echo "=============================="

              # 1) predict picks
              python "$PRED" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --mode "$mode" \
                --tag "$LABEL_KEY" \
                --suffix "$suffix" \
                --out-dir "$OUT_DIR" \
                --exclude-tickers "$EXCLUDE_TICKERS" \
                --tail-threshold "$tail" \
                --utility-quantile "$uq" \
                --rank-by "$rank_by" \
                --lambda-tail "$LAMBDA_TAIL" \
                ${REQUIRE_FILES:+--require-files "$REQUIRE_FILES"}

              picks_path="$OUT_DIR/picks_${LABEL_KEY}_gate_${suffix}.csv"

              # 2) simulate
              python "$SIM" \
                --picks-path "$picks_path" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --tag "$LABEL_KEY" \
                --suffix "$suffix" \
                --out-dir "$OUT_DIR" \
                --topk "$k" \
                --topk-weights "$w" \
                --enable-trailing "$ENABLE_TRAILING" \
                --tp1-frac "$TP1_FRAC" \
                --trail-stop "$tr"

              # 3) summarize from trades parquet
              trades_path="$OUT_DIR/sim_engine_trades_${LABEL_KEY}_gate_${suffix}.parquet"
              python "$SUM" \
                --trades-path "$trades_path" \
                --tag "$LABEL_KEY" \
                --suffix "$suffix" \
                --profit-target "$PROFIT_TARGET" \
                --max-days "$MAX_DAYS" \
                --stop-level "$STOP_LEVEL" \
                --max-extend-days "$MAX_EXTEND_DAYS" \
                --out-dir "$OUT_DIR"
            done
          else
            suffix="${mode}_t0$(fmt_num_for_suffix "$tail")_q0$(fmt_num_for_suffix "$uq")_r${rank_by}_lam$(fmt_num_for_suffix "$LAMBDA_TAIL")_k${k}_w$(fmt_num_for_suffix "$w")_notrail"
            echo "=============================="
            echo "[RUN] mode=$mode tail_max=$tail u_q=$uq rank_by=$rank_by lambda=$LAMBDA_TAIL topk=$k weights=$w trailing=OFF suffix=$suffix"
            echo "=============================="

            python "$PRED" \
              --profit-target "$PROFIT_TARGET" \
              --max-days "$MAX_DAYS" \
              --stop-level "$STOP_LEVEL" \
              --max-extend-days "$MAX_EXTEND_DAYS" \
              --mode "$mode" \
              --tag "$LABEL_KEY" \
              --suffix "$suffix" \
              --out-dir "$OUT_DIR" \
              --exclude-tickers "$EXCLUDE_TICKERS" \
              --tail-threshold "$tail" \
              --utility-quantile "$uq" \
              --rank-by "$rank_by" \
              --lambda-tail "$LAMBDA_TAIL" \
              ${REQUIRE_FILES:+--require-files "$REQUIRE_FILES"}

            picks_path="$OUT_DIR/picks_${LABEL_KEY}_gate_${suffix}.csv"

            python "$SIM" \
              --picks-path "$picks_path" \
              --profit-target "$PROFIT_TARGET" \
              --max-days "$MAX_DAYS" \
              --stop-level "$STOP_LEVEL" \
              --max-extend-days "$MAX_EXTEND_DAYS" \
              --tag "$LABEL_KEY" \
              --suffix "$suffix" \
              --out-dir "$OUT_DIR" \
              --topk "$k" \
              --topk-weights "$w" \
              --enable-trailing "$ENABLE_TRAILING" \
              --tp1-frac "$TP1_FRAC"

            trades_path="$OUT_DIR/sim_engine_trades_${LABEL_KEY}_gate_${suffix}.parquet"
            python "$SUM" \
              --trades-path "$trades_path" \
              --tag "$LABEL_KEY" \
              --suffix "$suffix" \
              --profit-target "$PROFIT_TARGET" \
              --max-days "$MAX_DAYS" \
              --stop-level "$STOP_LEVEL" \
              --max-extend-days "$MAX_EXTEND_DAYS" \
              --out-dir "$OUT_DIR"
          fi

        done
      done
    done
  done
done

echo "[DONE] grid finished"