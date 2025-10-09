#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Execuções reprodutíveis com hiperparâmetros afinados por cenário
# ============================================================
# Perfis:
#  - Entropy (fixo):      IID  -> W=100,  θ=0.10   | nonIID -> W=50,  θ=0.08
#  - Entropy (adapt.):    IID  -> W=100,  k≈3.0    | nonIID -> W=75,  k≈2.5   (ENTROPY_ADAPT_K)
#  - KSWIN:               IID  -> n=200, r=50, α=0.01 | nonIID -> n=100, r=30, α=0.02
#  - ADWIN:               IID  -> δ=0.002         | nonIID -> δ=0.006
#  - Wilbik (federado):   IID  -> δ=0.20          | nonIID -> δ=0.25 (K=3, m=2.0, q=2.0, init=3, eps=1e-4)

mkdir -p logs results

echo "🧹 Limpando artefatos de execuções anteriores..."
rm -f logs/*.txt logs/round_metrics.csv logs/round_metrics.jsonl logs/per_client_metrics.csv || true
rm -f results/round_metrics_*.csv results/round_metrics_*.jsonl results/per_client_metrics_*.csv || true
find results -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true

# ==========================
# Configuração comum
# ==========================
export NUM_PARTITIONS=4
export DRIFT_DEBUG=0
export SMOKE_DRIFT=0

ROUNDS=25
BATCH=32
LR=0.01
EPOCHS=1

# Seeds disponíveis (usaremos 11 para runs "single")
SEEDS=(11 22 33)

# ==========================
# Helpers
# ==========================
have_rows() {  # retorna 0 (true) se CSV tem header + ≥1 linha
  local f="$1"
  [[ -f "$f" ]] && [[ "$(wc -l < "$f")" -gt 1 ]]
}

reset_detector_env() {
  unset DRIFT_WINDOW DRIFT_THRESHOLD ENTROPY_ADAPT_K
  unset DRIFT_KSWIN_STAT KSWIN_WINDOW KSWIN_ALPHA
  unset ADWIN_DELTA
  unset WILBIK_K WILBIK_M WILBIK_Q WILBIK_DELTA WILBIK_MAX_SAMPLES WILBIK_INIT_ITERS WILBIK_EPS
}

set_hparams() {
  local METHOD="$1"   # entropy_fixed | entropy_adaptive | kswin | adwin | wilbik_federated
  local ALPHA="$2"    # "0.0" (IID) ou "0.3" (nonIID)
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
      if [[ "$REGIME" == "IID" ]]; then
        export DRIFT_WINDOW=100
        export ENTROPY_ADAPT_K=3.0
      else
        export DRIFT_WINDOW=75
        export ENTROPY_ADAPT_K=2.5
      fi
      ;;

    kswin)
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
      if [[ "$REGIME" == "IID" ]]; then
        export ADWIN_DELTA=0.002
      else
        export ADWIN_DELTA=0.006
      fi
      ;;

    wilbik_federated)
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

# run_one BASE METHOD ALPHA [REPEATS_OVERRIDE]
run_one() {
  local BASE_LABEL="$1"
  local METHOD="$2"
  local ALPHA="$3"
  local REPEATS_OVERRIDE="${4:-1}"     # default: 1 execução
  local REPEATS="$REPEATS_OVERRIDE"

  for ((r=0; r<REPEATS; r++)); do
    # escolhe a seed r-esima; se passar do tamanho, usa a última
    local SEED_INDEX=$(( r < ${#SEEDS[@]} ? r : ${#SEEDS[@]}-1 ))
    local SEED="${SEEDS[$SEED_INDEX]}"

    local LABEL="${BASE_LABEL}_s${SEED}"
    local OUTDIR="results/${LABEL}"
    mkdir -p "${OUTDIR}"

    echo
    echo "=== [${LABEL}] DRIFT_METHOD=${METHOD} NON_IID_ALPHA=${ALPHA} ==="

    if [[ "$ALPHA" == "0" || "$ALPHA" == "0.0" ]]; then
      unset NON_IID_ALPHA
    else
      export NON_IID_ALPHA="$ALPHA"
    fi

    export DRIFT_METHOD="$METHOD"
    set_hparams "$METHOD" "$ALPHA"
    ensure_river_if_needed "$METHOD"

    # ---- Overrides temporários para Wilbik (usados no cenário _nodrift) ----
    [[ -n "${OVERRIDE_WILBIK_INIT_ITERS:-}" ]] && export WILBIK_INIT_ITERS="${OVERRIDE_WILBIK_INIT_ITERS}"
    [[ -n "${OVERRIDE_WILBIK_DELTA:-}"       ]] && export WILBIK_DELTA="${OVERRIDE_WILBIK_DELTA}"
    # ------------------------------------------------------------------------

    rm -f logs/round_metrics.csv logs/round_metrics.jsonl logs/per_client_metrics.csv

    local TS="$(date +"%Y%m%d_%H%M%S")"
    local LOGFILE="logs/${LABEL}_${TS}.txt"

    export SCENARIO_LABEL="${LABEL}"
    export PYTHONHASHSEED="${SEED}"

    flwr run . \
      -c "num-server-rounds=${ROUNDS} \
          fraction-evaluate=1.0 \
          batch-size=${BATCH} local-epochs=${EPOCHS} learning-rate=${LR}" \
      --stream > "${LOGFILE}" 2>&1

    if [[ -f logs/round_metrics.csv ]]; then
      cp logs/round_metrics.csv   "${OUTDIR}/round_metrics.csv"
    else
      echo "⚠️  [${LABEL}] round_metrics.csv não foi gerado (veja ${LOGFILE})"
    fi

    if [[ -f logs/round_metrics.jsonl ]]; then
      cp logs/round_metrics.jsonl "${OUTDIR}/round_metrics.jsonl"
    else
      echo "⚠️  [${LABEL}] round_metrics.jsonl não foi gerado."
    fi

    if [[ -f logs/per_client_metrics.csv ]]; then
      cp logs/per_client_metrics.csv "${OUTDIR}/per_client_metrics.csv"
    fi

    if [[ -f "${OUTDIR}/round_metrics.csv" ]]; then
      if have_rows "${OUTDIR}/per_client_metrics.csv"; then
        python plot_round_metrics.py \
          --csv "${OUTDIR}/round_metrics.csv" \
          --per-client-csv "${OUTDIR}/per_client_metrics.csv" \
          --outdir "${OUTDIR}"
      else
        echo "ℹ️  [${LABEL}] per_client_metrics vazio/ausente. Gerando só gráficos globais."
        python plot_round_metrics.py \
          --csv "${OUTDIR}/round_metrics.csv" \
          --outdir "${OUTDIR}"
      fi
    else
      echo "❌ [${LABEL}] round_metrics.csv é obrigatório para os plots. Últimas linhas do log:"
      tail -n 80 "${LOGFILE}" || true
      exit 1
    fi

    echo "✓ Resultado do cenário '${LABEL}' salvo em:"
    echo "  - LOG:   ${LOGFILE}"
    [[ -f "${OUTDIR}/round_metrics.csv" ]]     && echo "  - CSV:   ${OUTDIR}/round_metrics.csv"
    [[ -f "${OUTDIR}/round_metrics.jsonl" ]]   && echo "  - JSONL: ${OUTDIR}/round_metrics.jsonl"
    [[ -f "${OUTDIR}/per_client_metrics.csv" ]]&& echo "  - CSV (per-client): ${OUTDIR}/per_client_metrics.csv"
    echo "  - PNGs globais:     ${OUTDIR}/global_accuracy.png, ${OUTDIR}/drift_count.png"
    if have_rows "${OUTDIR}/per_client_metrics.csv"; then
      echo "  - PNGs per-client:  ${OUTDIR}/per_client_mean_accuracy.png, ${OUTDIR}/per_client_heatmap.png"
    fi
  done
}

# ========================
# Execuções
# ========================

# 1) IID + threshold fixo
run_one "IID_entropy_fixed" "entropy_fixed" "0.0" 1

# 2) IID + threshold adaptativo
run_one "IID_entropy_adaptive" "entropy_adaptive" "0.0" 1

# 3) IID + KSWIN
run_one "IID_kswin" "kswin" "0.0" 1

# 3b) IID + ADWIN
run_one "IID_adwin" "adwin" "0.0" 1

# 3c) IID + Wilbik  (ÚNICO com 3 seeds)
run_one "IID_wilbik" "wilbik_federated" "0.0" 3

# 3d) IID + Wilbik (sem drift: baseline atrasado)
OVERRIDE_WILBIK_INIT_ITERS=15 run_one "IID_wilbik_nodrift" "wilbik_federated" "0.0" 1

# 4) non-IID + threshold fixo
run_one "nonIID_entropy_fixed" "entropy_fixed" "0.3" 1

# 5) non-IID + threshold adaptativo
run_one "nonIID_entropy_adaptive" "entropy_adaptive" "0.3" 1

# 6) non-IID + ADWIN
run_one "nonIID_adwin" "adwin" "0.3" 1

# 6b) non-IID + KSWIN
run_one "nonIID_kswin" "kswin" "0.3" 1

# 6c) non-IID + Wilbik
run_one "nonIID_wilbik" "wilbik_federated" "0.3" 1

echo
echo "=== Experimentos concluídos! ==="