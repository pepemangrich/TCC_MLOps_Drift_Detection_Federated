#!/usr/bin/env bash
set -euo pipefail

# ================================
#   Execução paralela com isolamento
#   Requer: GNU parallel, rsync (opcional), Python + flwr
# ================================

# Checa GNU parallel
if ! command -v parallel >/dev/null 2>&1; then
  echo "[ERRO] 'parallel' não encontrado. Instale GNU Parallel (ex.: 'sudo apt-get install parallel' ou método equivalente)."
  exit 1
fi

# Pastas globais
mkdir -p logs results work

# Configuração comum (export para herdar no ambiente de cada job)
export NUM_PARTITIONS=4
export DRIFT_WINDOW=20
export DRIFT_DEBUG=0
export SMOKE_DRIFT=0    # sem drift sintético nos experimentos
export ROUNDS=5
export BATCH=32
export LR=0.01
export EPOCHS=1

# Instala river 1x, se necessário (usado por kswin/adwin)
python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("river") else 1)
PY
if [[ $? -ne 0 ]]; then
  pip install -q river
fi

# Função: roda UM cenário em diretório isolado (work/<LABEL>)
run_one_isolated() {
  local LABEL="$1"     # ex.: IID_entropy_fixed
  local METHOD="$2"    # entropy_fixed | entropy_adaptive | kswin | adwin
  local ALPHA="$3"     # 0.0 (IID) ou ex. 0.3 (não-IID)
  local THRESH="$4"    # "0.02" p/ fixed | "-" p/ demais

  local ROOT; ROOT="$(pwd)"
  local WDIR="work/${LABEL}"

  echo
  echo "=== [${LABEL}] DRIFT_METHOD=${METHOD} NON_IID_ALPHA=${ALPHA} ${THRESH:+DRIFT_THRESHOLD=$THRESH} ==="

  # Recria diretório isolado do cenário
  rm -rf "${WDIR}"
  mkdir -p "${WDIR}"

  # Copia o projeto para o diretório do cenário (exclui arquivos/pastas pesadas)
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
      --exclude 'work' --exclude 'results' \
      "${ROOT}/" "${WDIR}/"
  else
    # fallback simples (pode copiar um pouco mais do que o necessário)
    (shopt -s dotglob; cp -r "${ROOT}/"* "${WDIR}/" || true)
    rm -rf "${WDIR}/.git" "${WDIR}/.venv" "${WDIR}/work" "${WDIR}/results" || true
  fi

  pushd "${WDIR}" >/dev/null

  mkdir -p logs results

  # Variáveis de cenário
  export DRIFT_METHOD="${METHOD}"
  if [[ "${ALPHA}" == "0" || "${ALPHA}" == "0.0" ]]; then
    unset NON_IID_ALPHA
  else
    export NON_IID_ALPHA="${ALPHA}"
  fi

  if [[ "${METHOD}" == "entropy_fixed" ]]; then
    export DRIFT_THRESHOLD="$([[ "${THRESH}" == "-" || -z "${THRESH}" ]] && echo "0.02" || echo "${THRESH}")"
  else
    unset DRIFT_THRESHOLD
  fi

  # Limpa logs locais do cenário
  rm -f logs/round_metrics.csv logs/round_metrics.jsonl logs/per_client_metrics.csv

  # Executa Flower (stdout/stderr -> log do cenário)
  flwr run . \
    -c "num-server-rounds=${ROUNDS} fraction-evaluate=1.0 batch-size=${BATCH} local-epochs=${EPOCHS} learning-rate=${LR}" \
    --stream > "logs/${LABEL}.log" 2>&1

  # Copia artefatos do cenário para results/ local
  mkdir -p "results/${LABEL}"
  cp logs/round_metrics.csv   "results/round_metrics_${LABEL}.csv"
  cp logs/round_metrics.jsonl "results/round_metrics_${LABEL}.jsonl"
  [[ -f logs/per_client_metrics.csv ]] && cp logs/per_client_metrics.csv "results/per_client_metrics_${LABEL}.csv"

  # Plota (globais e per-client, se existir)
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

  # Consolida no diretório raiz do projeto
  mkdir -p "${ROOT}/results/${LABEL}"
  cp -a "results/${LABEL}/." "${ROOT}/results/${LABEL}/"
  cp "results/round_metrics_${LABEL}.csv"   "${ROOT}/results/round_metrics_${LABEL}.csv"
  cp "results/round_metrics_${LABEL}.jsonl" "${ROOT}/results/round_metrics_${LABEL}.jsonl"
  [[ -f "results/per_client_metrics_${LABEL}.csv" ]] && cp "results/per_client_metrics_${LABEL}.csv" "${ROOT}/results/per_client_metrics_${LABEL}.csv"

  # Log
  mkdir -p "${ROOT}/logs"
  cp "logs/${LABEL}.log" "${ROOT}/logs/${LABEL}_$(date +%Y%m%d_%H%M%S).txt"

  popd >/dev/null
  echo "✓ ${LABEL} finalizado."
}

export -f run_one_isolated
export ROUNDS BATCH LR EPOCHS DRIFT_WINDOW DRIFT_DEBUG SMOKE_DRIFT NUM_PARTITIONS

# Lista de cenários: LABEL METHOD ALPHA THRESH ("-" quando não aplicável)
read -r -d '' SCENARIOS <<'EOF'
IID_entropy_fixed        entropy_fixed       0.0 0.02
IID_entropy_adaptive     entropy_adaptive    0.0 -
IID_kswin                kswin               0.0 -
IID_adwin                adwin               0.0 -
nonIID_entropy_fixed     entropy_fixed       0.3 0.02
nonIID_entropy_adaptive  entropy_adaptive    0.3 -
nonIID_adwin             adwin               0.3 -
nonIID_kswin             kswin               0.3 -
EOF

# Nº de jobs paralelos (padrão = nº de CPUs)
JOBS="${JOBS:-$( (command -v nproc >/dev/null 2>&1 && nproc) || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
echo "[INFO] Rodando com ${JOBS} jobs paralelos."

# Dispara em paralelo
printf "%s\n" "${SCENARIOS}" | parallel --colsep ' ' --jobs "${JOBS}" run_one_isolated {1} {2} {3} {4}

echo
echo "=== Experimentos (paralelos) concluídos! Resultados em ./results ==="