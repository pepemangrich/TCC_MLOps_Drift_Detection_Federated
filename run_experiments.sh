#!/usr/bin/env bash
set -euo pipefail

# Pastas
mkdir -p logs results

# ==========================
# Limpeza da batelada anterior
# ==========================
echo "🧹 Limpando artefatos de execuções anteriores..."
rm -f logs/*.txt logs/round_metrics.csv logs/round_metrics.jsonl logs/per_client_metrics.csv || true
rm -f results/round_metrics_*.csv results/round_metrics_*.jsonl results/per_client_metrics_*.csv || true
find results -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true

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
  local METHOD="$2"         # entropy_fixed | entropy_adaptive | kswin | adwin | wilbik_federated
  local ALPHA="$3"          # 0.0 (IID) ou ex. 0.3 (não-IID)
  local THRESH="${4:-}"     # threshold apenas p/ entropy_fixed (ex. 0.02)

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

  # Parâmetros do Wilbik federado (usa defaults se não estiverem no ambiente)
  if [[ "$METHOD" == "wilbik_federated" ]]; then
    export WILBIK_K="${WILBIK_K:-3}"
    export WILBIK_M="${WILBIK_M:-2.0}"
    export WILBIK_DELTA="${WILBIK_DELTA:-0.1}"
    export WILBIK_MAX_SAMPLES="${WILBIK_MAX_SAMPLES:-512}"
    echo "    WILBIK_K=${WILBIK_K} WILBIK_M=${WILBIK_M} WILBIK_DELTA=${WILBIK_DELTA} WILBIK_MAX_SAMPLES=${WILBIK_MAX_SAMPLES}"
  fi

  # Dependência opcional para KSWIN/ADWIN
  if [[ "$METHOD" == "kswin" || "$METHOD" == "adwin" ]]; then
    python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("river") else 1)
PY
    if [[ $? -ne 0 ]]; then
      pip install -q river
    fi
  fi

  # Limpa logs da rodada anterior para não misturar arquivos desta execução
  rm -f logs/round_metrics.csv logs/round_metrics.jsonl logs/per_client_metrics.csv

  # Arquivo de log desta execução
  TS="$(date +"%Y%m%d_%H%M%S")"
  LOGFILE="logs/${LABEL}_${TS}.txt"

  # Exporta label para o server gravar no CSV/JSONL
  export SCENARIO_LABEL="${LABEL}"

  # Executa (salva stdout+stderr no LOGFILE)
  flwr run . \
    -c "num-server-rounds=${ROUNDS} fraction-evaluate=1.0 batch-size=${BATCH} local-epochs=${EPOCHS} learning-rate=${LR}" \
    --stream > "${LOGFILE}" 2>&1

  # Salva CSV/JSONL com rótulo do cenário em results/ (cópia condicional)
  if [[ -f logs/round_metrics.csv ]]; then
    cp logs/round_metrics.csv   "results/round_metrics_${LABEL}.csv"
  else
    echo "⚠️  [${LABEL}] round_metrics.csv não foi gerado (verifique ${LOGFILE})"
  fi

  if [[ -f logs/round_metrics.jsonl ]]; then
    cp logs/round_metrics.jsonl "results/round_metrics_${LABEL}.jsonl"
  else
    echo "⚠️  [${LABEL}] round_metrics.jsonl não foi gerado (sem JSONL para este cenário)."
  fi

  if [[ -f logs/per_client_metrics.csv ]]; then
    cp logs/per_client_metrics.csv "results/per_client_metrics_${LABEL}.csv"
  fi

  # Plota para este cenário em subpasta dedicada de results/
  mkdir -p "results/${LABEL}"
  if [[ -f "results/per_client_metrics_${LABEL}.csv" ]]; then
    python plot_round_metrics.py \
      --csv "results/round_metrics_${LABEL}.csv" \
      --per-client-csv "results/per_client_metrics_${LABEL}.csv" \
      --outdir "results/${LABEL}"
  else
    python plot_round_metrics.py \
      --csv "results/round_metrics_${LABEL}.csv" \
      --outdir "results/${LABEL}"
  fi

  echo "✓ Resultado do cenário '${LABEL}' salvo em:"
  echo "  - LOG:   ${LOGFILE}"
  [[ -f "results/round_metrics_${LABEL}.csv" ]]   && echo "  - CSV:   results/round_metrics_${LABEL}.csv"
  [[ -f "results/round_metrics_${LABEL}.jsonl" ]] && echo "  - JSONL: results/round_metrics_${LABEL}.jsonl"
  [[ -f "results/per_client_metrics_${LABEL}.csv" ]] && echo "  - CSV (per-client): results/per_client_metrics_${LABEL}.csv"
  echo "  - PNGs globais:     results/${LABEL}/global_accuracy.png, results/${LABEL}/drift_count.png"
  [[ -f "results/per_client_metrics_${LABEL}.csv" ]] && echo "  - PNGs per-client:  results/${LABEL}/per_client_mean_accuracy.png, results/${LABEL}/per_client_heatmap.png"
}

# ========================
# Execuções (10 cenários)
# ========================

# 1) IID + threshold fixo
run_one "IID_entropy_fixed" "entropy_fixed" "0.0" "0.02"

# 2) IID + threshold adaptativo (média + 3σ)
run_one "IID_entropy_adaptive" "entropy_adaptive" "0.0"

# 3) IID + KSWIN
run_one "IID_kswin" "kswin" "0.0"

# 3b) IID + ADWIN
run_one "IID_adwin" "adwin" "0.0"

# 3c) IID + Wilbik (NOVO)
run_one "IID_wilbik" "wilbik_federated" "0.0"

# 4) não-IID + threshold fixo
run_one "nonIID_entropy_fixed" "entropy_fixed" "0.3" "0.02"

# 5) não-IID + threshold adaptativo
run_one "nonIID_entropy_adaptive" "entropy_adaptive" "0.3"

# 6) não-IID + ADWIN
run_one "nonIID_adwin" "adwin" "0.3"

# 6b) não-IID + KSWIN
run_one "nonIID_kswin" "kswin" "0.3"

# 6c) não-IID + Wilbik (NOVO)
run_one "nonIID_wilbik" "wilbik_federated" "0.3"

echo
echo "=== Experimentos concluídos! ==="