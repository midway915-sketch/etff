#!/usr/bin/env bash
set -euo pipefail

tok() {
  python - <<'PY' "$1"
import sys
x=float(sys.argv[1])
s=f"{x:.4f}".rstrip("0").rstrip(".")
s=s.replace(".","p").replace("-","m")
print(s)
PY
}

build_tag() {
  python - <<'PY' "$1" "$2" "$3" "$4"
import sys
pt=float(sys.argv[1]); h=int(float(sys.argv[2])); sl=float(sys.argv[3]); ex=int(float(sys.argv[4]))
pt_i=int(round(pt*100)); sl_i=int(round(abs(sl)*100))
print(f"pt{pt_i}_h{h}_sl{sl_i}_ex{ex}")
PY
}

run_one_gate() {
  local pt="$1"; local h="$2"; local sl="$3"; local ex="$4"
  local mode="$5"; local tail_max="$6"; local u_q="$7"; local rank_by="$8"
  local lambda_tail="$9"; local tau_gamma="${10}"

  local tag; tag="$(build_tag "$pt" "$h" "$sl" "$ex")"
  local suffix="${mode}_t$(tok "$tail_max")_q$(tok "$u_q")_r${rank_by}"

  local picks="data/signals/picks_${tag}_gate_${suffix}.csv"
  local trades="data/signals/sim_engine_trades_${tag}_gate_${suffix}.parquet"
  local curve="data/signals/sim_engine_curve_${tag}_gate_${suffix}.parquet"

  mkdir -p data/signals

  echo "=============================="
  echo "[RUN] tag=${tag} suffix=${suffix}"
  echo "[RUN] mode=${mode} tail_max=${tail_max} u_q=${u_q} rank_by=${rank_by}"
  echo "[RUN] picks=${picks}"
  echo "=============================="

  # 1) picks 생성
  python scripts/predict_gate.py \
    --profit-target "$pt" \
    --max-days "$h" \
    --stop-level "$sl" \
    --max-extend-days "$ex" \
    --mode "$mode" \
    --tail-threshold "$tail_max" \
    --utility-quantile "$u_q" \
    --rank-by "$rank_by" \
    --lambda-tail "$lambda_tail" \
    --tau-gamma "$tau_gamma" \
    --suffix "$suffix" \
    --out-csv "$picks"

  if [ ! -f "$picks" ]; then
    echo "[ERROR] picks not created: $picks"
    ls -la data/signals | sed -n '1,200p' || true
    exit 1
  fi

  # 2) ✅ 각 picks마다 시뮬(조합별 파일명 분리 강제)
  python scripts/simulate_single_position_engine.py \
    --profit-target "$pt" \
    --max-days "$h" \
    --stop-level "$sl" \
    --max-extend-days "$ex" \
    --picks-path "$picks" \
    --tag "$tag" \
    --suffix "$suffix"

  if [ ! -f "$trades" ]; then
    echo "[ERROR] trades parquet not created: $trades"
    echo "[DEBUG] signals dir:"
    ls -la data/signals | sed -n '1,200p' || true
    exit 1
  fi

  # 3) ✅ trades→curve 자동추론해서 요약 생성
  python scripts/summarize_sim_trades.py \
    --trades-path "$trades" \
    --tag "$tag" \
    --suffix "$suffix" \
    --profit-target "$pt" \
    --max-days "$h" \
    --stop-level "$sl" \
    --max-extend-days "$ex" \
    --out-dir "data/signals"

  echo "[OK] done: $suffix  trades=$(basename "$trades") curve=$(basename "$curve")"
}