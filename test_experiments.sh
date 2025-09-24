#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Parâmetros "rápidos"
# ---------------------------
ROUNDS=${ROUNDS:-8}
BATCH=${BATCH:-64}
EPOCHS=${EPOCHS:-1}
LR=${LR:-0.01}
FRACTION_EVAL=${FRACTION_EVAL:-1.0}
NUM_PARTITIONS=${NUM_PARTITIONS:-4}

# Detector global (Wilbik federated / FCM+DB)
export DRIFT_METHOD=wilbik_federated
export WILBIK_K=${WILBIK_K:-10}          # comece com #classes (CIFAR-10)
export WILBIK_M=${WILBIK_M:-2.0}
export WILBIK_Q=${WILBIK_Q:-2.0}
export WILBIK_INIT_ITERS=${WILBIK_INIT_ITERS:-3}
export WILBIK_MAX_SAMPLES=${WILBIK_MAX_SAMPLES:-512}
export WILBIK_EPS=${WILBIK_EPS:-1e-3}
export WILBIK_DELTA=${WILBIK_DELTA:-0.20}  # banda ±20%

# Outros
export DRIFT_DEBUG=${DRIFT_DEBUG:-0}
export SMOKE_DRIFT=${SMOKE_DRIFT:-0}
export NUM_PARTITIONS  # garante que passe pro app

mkdir -p logs results

run_case () {
  local LABEL="$1"
  echo
  echo "=== [${LABEL}] Rounds=${ROUNDS} K=${WILBIK_K} m=${WILBIK_M} q=${WILBIK_Q} Δ=${WILBIK_DELTA} ==="

  # rótulo que cai nos CSV/JSONL
  export SCENARIO_LABEL="$LABEL"

  # limpa artefatos da rodada anterior
  rm -f logs/round_metrics.csv logs/round_metrics.jsonl logs/per_client_metrics.csv || true

  # roda o Flower no modo "projeto"
  flwr run . \
    -c "num-server-rounds=${ROUNDS} \
        fraction-evaluate=${FRACTION_EVAL} \
        batch-size=${BATCH} \
        local-epochs=${EPOCHS} \
        learning-rate=${LR}" \
    --stream

  # guarda resultados
  local OUTDIR="results/${LABEL}"
  mkdir -p "${OUTDIR}"
  [[ -f logs/round_metrics.csv ]]   && cp logs/round_metrics.csv   "${OUTDIR}/round_metrics.csv"
  [[ -f logs/round_metrics.jsonl ]] && cp logs/round_metrics.jsonl "${OUTDIR}/round_metrics.jsonl"
  [[ -f logs/per_client_metrics.csv ]] && cp logs/per_client_metrics.csv "${OUTDIR}/per_client_metrics.csv"

  # plota (global sempre; per-client se existir)
  if [[ -f "${OUTDIR}/round_metrics.csv" ]]; then
    if [[ -f "${OUTDIR}/per_client_metrics.csv" && $(wc -l < "${OUTDIR}/per_client_metrics.csv") -gt 1 ]]; then
      python plot_round_metrics.py \
        --csv "${OUTDIR}/round_metrics.csv" \
        --per-client-csv "${OUTDIR}/per_client_metrics.csv" \
        --outdir "${OUTDIR}"
    else
      python plot_round_metrics.py \
        --csv "${OUTDIR}/round_metrics.csv" \
        --outdir "${OUTDIR}"
    fi
  fi

  echo "✓ Artefatos salvos em ${OUTDIR}"
}

# --------- Caso 1: IID ---------
unset NON_IID_ALPHA
run_case "test_cifar10_iid_init${WILBIK_INIT_ITERS}_k${WILBIK_K}_q${WILBIK_Q}_delta${WILBIK_DELTA}"

# --------- Caso 2: não-IID (Dirichlet) ---------
export NON_IID_ALPHA=${NON_IID_ALPHA:-0.3}
run_case "test_cifar10_dirichlet${NON_IID_ALPHA}_init${WILBIK_INIT_ITERS}_k${WILBIK_K}_q${WILBIK_Q}_delta${WILBIK_DELTA}"

echo
echo "[OK] Testes concluídos. Veja ./results/* e ./logs"