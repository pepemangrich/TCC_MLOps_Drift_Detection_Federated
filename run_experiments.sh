#!/usr/bin/env bash
set -euo pipefail

# Pastas
mkdir -p logs results

# Configuração comum
export NUM_PARTITIONS=4
export DRIFT_WINDOW=20
export DRIFT_DEBUG=0
export SMOKE_DRIFT=0        # sem drift sintético nos experimentos
ROUNDS=5
BATCH=32
LR=0.01
EPOCHS=1

# Função auxiliar: roda um cenário e salva CSV/plots em pastas por rótulo
run_one() {
  local LABEL="$1"          # ex: IID_entropy_fixed
  local METHOD="$2"         # entropy_fixed | entropy_adaptive | kswin | adwin
  local ALPHA="$3"          # 0.0 (IID) ou ex. 0.3 (não-IID)
  local THRESH="${4:-}"     # threshold apenas para entropy_fixed (ex. 0.02)

  echo
  echo "=== [${LABEL}] DRIFT_METHOD=${METHOD} NON_IID_ALPHA=${ALPHA} ${THRESH:+DRIFT_THRESHOLD=$THRESH} ==="

  # Cenário: IID vs não-IID
  if [[ "$ALPHA" == "0" || "$ALPHA" == "0.0" ]]; then
    unset NON_IID_ALPHA
  else
    export NON_IID_ALPHA="$ALPHA"
  fi

  # Método de detecção
  export DRIFT_METHOD="$METHOD"
  # Threshold só faz sentido para entropy_fixed; limpe nos outros
  if [[ "$METHOD" == "entropy_fixed" ]]; then
    export DRIFT_THRESHOLD="${THRESH:-0.02}"
  else
    unset DRIFT_THRESHOLD
  fi

  if [[ "$METHOD" == "kswin" || "$METHOD" == "adwin" ]]; then
    python -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('river') else 1)" \
      || pip install -q river
  fi

  # Limpa logs da rodada anterior para não misturar
  rm -f logs/round_metrics.csv logs/round_metrics.jsonl

  # Arquivo de log desta execução
  TS="$(date +"%Y%m%d_%H%M%S")"
  LOGFILE="logs/${LABEL}_${TS}.txt"

  # Executa (salva stdout+stderr no LOGFILE)
  flwr run . \
    -c "num-server-rounds=${ROUNDS} fraction-evaluate=1.0 batch-size=${BATCH} local-epochs=${EPOCHS} learning-rate=${LR}" \
    --stream > "${LOGFILE}" 2>&1

  # Salva CSV/JSONL com rótulo do cenário em results/
  cp logs/round_metrics.csv   "results/round_metrics_${LABEL}.csv"
  cp logs/round_metrics.jsonl "results/round_metrics_${LABEL}.jsonl"

  # Plota para este cenário em subpasta dedicada de results/
  mkdir -p "results/${LABEL}"
  python plot_round_metrics.py --csv "results/round_metrics_${LABEL}.csv" --outdir "results/${LABEL}"

  echo "✓ Resultado do cenário '${LABEL}' salvo em:"
  echo "  - LOG:   ${LOGFILE}"
  echo "  - CSV:   results/round_metrics_${LABEL}.csv"
  echo "  - JSONL: results/round_metrics_${LABEL}.jsonl"
  echo "  - PNGs:  results/${LABEL}/global_accuracy.png, results/${LABEL}/drift_count.png"
}

# ========================
# Execuções (6 cenários)
# ========================

# 1) IID + threshold fixo
run_one "IID_entropy_fixed" "entropy_fixed" "0.0" "0.02"

# 2) IID + threshold adaptativo (média + 3σ)
run_one "IID_entropy_adaptive" "entropy_adaptive" "0.0"

# 3) IID + KSWIN
run_one "IID_kswin" "kswin" "0.0"

# 4) não-IID + threshold fixo
run_one "nonIID_entropy_fixed" "entropy_fixed" "0.3" "0.02"

# 5) não-IID + threshold adaptativo
run_one "nonIID_entropy_adaptive" "entropy_adaptive" "0.3"

# 6) não-IID + ADWIN
run_one "nonIID_adwin" "adwin" "0.3"

echo
echo "=== Experimentos concluídos! ==="