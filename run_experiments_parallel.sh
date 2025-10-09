#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Execuções reprodutíveis em PARALELO com hiperparâmetros afinados por cenário
# Requer: GNU parallel, rsync (opcional), Python + flwr
# ============================================================

# --- checagens básicas ---
if ! command -v parallel >/dev/null 2>&1; then
  echo "[ERRO] 'parallel' não encontrado. Instale com 'sudo apt-get install parallel' (ou equivalente)."
  exit 1
fi

# Pastas raiz
mkdir -p logs results work

# ==========================
# Configuração comum
# ==========================
export NUM_PARTITIONS=4
export DRIFT_DEBUG=0
export SMOKE_DRIFT=0              # sem drift sintético

# Validade temporal/estatística
export ROUNDS=25
export BATCH=32
export LR=0.01
export EPOCHS=1

# Repetições (seeds)
SEEDS=(11 22 33)

# Instala river 1x, se necessário (usado por kswin/adwin)
python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("river") else 1)
PY
if [[ $? -ne 0 ]]; then
  pip install -q river
fi

# ==========================
# Funções auxiliares
# ==========================
reset_detector_env() {
  # Zera variáveis para evitar "vazar" de um cenário para outro
  unset DRIFT_WINDOW DRIFT_THRESHOLD ENTROPY_ADAPT_K
  unset DRIFT_KSWIN_STAT KSWIN_WINDOW KSWIN_ALPHA
  unset ADWIN_DELTA
  unset WILBIK_K WILBIK_M WILBIK_Q WILBIK_DELTA WILBIK_MAX_SAMPLES WILBIK_INIT_ITERS WILBIK_EPS
}

set_hparams() {
  local METHOD="$1"   # entropy_fixed | entropy_adaptive | kswin | adwin | wilbik_federated
  local ALPHA="$2"    # "0.0" (IID) ou e.g. "0.3" (não-IID)
  local REGIME="IID"; [[ "$ALPHA" != "0" && "$ALPHA" != "0.0" ]] && REGIME="nonIID"

  reset_detector_env

  case "$METHOD" in
    entropy_fixed)
      if [[ "$REGIME" == "IID" ]]; then
        export DRIFT_WINDOW=100
        export DRIFT_THRESHOLD=0.10
      else
        export DRIFT_WINDOW=50
        export DRIFT_THRESHOLD=0.08
      fi
      ;;

    entropy_adaptive)
      # baseline + k*sigma (se seu código usar ENTROPY_ADAPT_K)
      if [[ "$REGIME" == "IID" ]]; then
        export DRIFT_WINDOW=100
        export ENTROPY_ADAPT_K=3.0
      else
        export DRIFT_WINDOW=75
        export ENTROPY_ADAPT_K=2.5
      fi
      ;;

    kswin)
      # DRIFT_WINDOW -> window_size (n), DRIFT_KSWIN_STAT -> stat_size (r), DRIFT_THRESHOLD -> alpha
      if [[ "$REGIME" == "IID" ]]; then
        export DRIFT_WINDOW=200
        export DRIFT_KSWIN_STAT=50
        export DRIFT_THRESHOLD=0.01
      else
        export DRIFT_WINDOW=100
        export DRIFT_KSWIN_STAT=30
        export DRIFT_THRESHOLD=0.02
      fi
      ;;

    adwin)
      # δ do ADWIN
      if [[ "$REGIME" == "IID" ]]; then
        export ADWIN_DELTA=0.002
      else
        export ADWIN_DELTA=0.006
      fi
      ;;

    wilbik_federated)
      # Ajustes do capítulo; demais ficam fixos
      if [[ "$REGIME" == "IID" ]]; then
        export WILBIK_DELTA=0.20
      else
        export WILBIK_DELTA=0.25
      fi
      export WILBIK_K=3 WILBIK_M=2.0 WILBIK_Q=2.0
      export WILBIK_MAX_SAMPLES=512 WILBIK_INIT_ITERS=3 WILBIK_EPS=1e-4
      ;;
  esac
}

ensure_river_if_needed() {
  local METHOD="$1"
  if [[ "$METHOD" == "kswin" || "$METHOD" == "adwin" ]]; then
    python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("river") else 1)
PY
    if [[ $? -ne 0 ]]; then pip install -q river; fi
  fi
}

run_one_isolated() {
  local BASE_LABEL="$1"     # ex: IID_adwin
  local METHOD="$2"         # entropy_fixed | entropy_adaptive | kswin | adwin | wilbik_federated
  local ALPHA="$3"          # 0.0 (IID) ou ex. 0.3 (não-IID)
  local SEED="$4"           # 11 | 22 | 33

  local ROOT; ROOT="$(pwd)"
  local LABEL="${BASE_LABEL}_s${SEED}"
  local WDIR="work/${LABEL}"

  echo
  echo "=== [${LABEL}] DRIFT_METHOD=${METHOD} NON_IID_ALPHA=${ALPHA} PYTHONHASHSEED=${SEED} ==="

  # Recria diretório isolado do cenário
  rm -rf "${WDIR}"
  mkdir -p "${WDIR}"

  # Copia o projeto para o diretório do cenário (exclui pastas pesadas)
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
      --exclude 'work' --exclude 'results' \
      "${ROOT}/" "${WDIR}/"
  else
    (shopt -s dotglob; cp -r "${ROOT}/"* "${WDIR}/" || true)
    rm -rf "${WDIR}/.git" "${WDIR}/.venv" "${WDIR}/work" "${WDIR}/results" || true
  fi

  pushd "${WDIR}" >/dev/null

  mkdir -p logs results

  # Cenário: IID vs não-IID
  if [[ "$ALPHA" == "0" || "$ALPHA" == "0.0" ]]; then
    unset NON_IID_ALPHA
  else
    export NON_IID_ALPHA="$ALPHA"
  fi

  # Método e hiperparâmetros específicos
  export DRIFT_METHOD="$METHOD"
  set_hparams "$METHOD" "$ALPHA"
  ensure_river_if_needed "$METHOD"

  # Limpa logs locais do cenário
  rm -f logs/round_metrics.csv logs/round_metrics.jsonl logs/per_client_metrics.csv

  # Rótulo e seed
  export SCENARIO_LABEL="${LABEL}"
  export PYTHONHASHSEED="${SEED}"

  # Executa Flower (stdout/stderr -> log do cenário)
  flwr run . \
    -c "num-server-rounds=${ROUNDS} fraction-evaluate=1.0 batch-size=${BATCH} local-epochs=${EPOCHS} learning-rate=${LR}" \
    --stream > "logs/${LABEL}.log" 2>&1

  # Copia artefatos do cenário para results/ local
  mkdir -p "results/${LABEL}"
  [[ -f logs/round_metrics.csv   ]] && cp logs/round_metrics.csv   "results/round_metrics_${LABEL}.csv"
  [[ -f logs/round_metrics.jsonl ]] && cp logs/round_metrics.jsonl "results/round_metrics_${LABEL}.jsonl"
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
  [[ -f "results/round_metrics_${LABEL}.csv"   ]] && cp "results/round_metrics_${LABEL}.csv"   "${ROOT}/results/round_metrics_${LABEL}.csv"
  [[ -f "results/round_metrics_${LABEL}.jsonl" ]] && cp "results/round_metrics_${LABEL}.jsonl" "${ROOT}/results/round_metrics_${LABEL}.jsonl"
  [[ -f "results/per_client_metrics_${LABEL}.csv" ]] && cp "results/per_client_metrics_${LABEL}.csv" "${ROOT}/results/per_client_metrics_${LABEL}.csv"

  # Log
  mkdir -p "${ROOT}/logs"
  cp "logs/${LABEL}.log" "${ROOT}/logs/${LABEL}_$(date +%Y%m%d_%H%M%S).txt"

  popd >/dev/null
  echo "✓ ${LABEL} finalizado."
}

export -f reset_detector_env set_hparams ensure_river_if_needed run_one_isolated
export ROUNDS BATCH LR EPOCHS DRIFT_DEBUG SMOKE_DRIFT NUM_PARTITIONS

# ========================
# Grade de cenários (base) — igual ao seu run_experiments.sh
# ========================
SCENARIOS=(
  "IID_entropy_fixed        entropy_fixed       0.0"
  "IID_entropy_adaptive     entropy_adaptive    0.0"
  "IID_kswin                kswin               0.0"
  "IID_adwin                adwin               0.0"
  "IID_wilbik               wilbik_federated    0.0"
  "nonIID_entropy_fixed     entropy_fixed       0.3"
  "nonIID_entropy_adaptive  entropy_adaptive    0.3"
  "nonIID_adwin             adwin               0.3"
  "nonIID_kswin             kswin               0.3"
  "nonIID_wilbik            wilbik_federated    0.3"
)

# ========================
# Gera a lista (cenário x seed) e roda em paralelo
# ========================
JOBS="${JOBS:-$( (command -v nproc >/dev/null 2>&1 && nproc) || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
echo "[INFO] Rodando com ${JOBS} jobs paralelos."

# Monta linhas: "<BASE_LABEL> <METHOD> <ALPHA> <SEED>"
WORKLIST=$(mktemp)
for line in "${SCENARIOS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    printf "%s %s %s %s\n" $line "$SEED" >> "$WORKLIST"
  done
done

# Dispara em paralelo (cada job isola em work/<LABEL>)
parallel --colsep ' ' --jobs "${JOBS}" run_one_isolated {1} {2} {3} {4} :::: "$WORKLIST"
rm -f "$WORKLIST"

echo
echo "=== Experimentos paralelos concluídos! Resultados em ./results ==="