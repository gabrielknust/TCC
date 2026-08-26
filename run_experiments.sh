#!/usr/bin/env bash
# Roda o pipeline completo (treino + calibração + inferência) para vários
# valores de N_BRANCHES sem intervenção manual.
#
# Uso:
#   ./run_experiments.sh
#
# Configurações (edite aqui):
#   N_BRANCHES_LIST → quais valores de nº de ramos rodar
#   LOSS_WEIGHTS_LIST → estratégias de pesos da loss (cada uma vira 1 treino)
#
# Cada treino salva DOIS checkpoints da mesma trajetória (melhor val loss e
# melhor val acc), então calibração e inferência rodam duas vezes: uma para a
# variante base e outra para a variante "-acc".
#
# A saída completa é gravada em logs/experiments_<timestamp>.log via tee.
set -euo pipefail

cd "$(dirname "$0")"

N_BRANCHES_LIST=(2 3 4 5)
export LOSS_WEIGHTS_LIST="equal,linspace"

BASE_VARIANT="${VARIANT:-}"
ACC_VARIANT="${BASE_VARIANT:+${BASE_VARIANT}-}acc"

mkdir -p logs
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
MASTER_LOG="logs/experiments_${TIMESTAMP}.log"

{
  echo "##############################################################"
  echo "# Pipeline iniciado: $(date)"
  echo "# N_BRANCHES_LIST: ${N_BRANCHES_LIST[*]}"
  echo "# LOSS_WEIGHTS_LIST: ${LOSS_WEIGHTS_LIST}"
  echo "# Variantes: '${BASE_VARIANT:-<padrão>}' e '${ACC_VARIANT}'"
  echo "##############################################################"

  for n in "${N_BRANCHES_LIST[@]}"; do
    export N_BRANCHES=$n
    echo
    echo "=============================================================="
    echo "## N_BRANCHES=$n :: TREINO ($(date +%H:%M:%S))"
    echo "=============================================================="
    VARIANT="$BASE_VARIANT" uv run python -m src.train

    for v in "$BASE_VARIANT" "$ACC_VARIANT"; do
      echo
      echo "=============================================================="
      echo "## N_BRANCHES=$n :: CALIBRAÇÃO [variante='${v:-<padrão>}'] ($(date +%H:%M:%S))"
      echo "=============================================================="
      VARIANT="$v" uv run python -m src.calibrate

      echo
      echo "=============================================================="
      echo "## N_BRANCHES=$n :: INFERÊNCIA [variante='${v:-<padrão>}'] ($(date +%H:%M:%S))"
      echo "=============================================================="
      VARIANT="$v" uv run python -m src.inference
    done
  done

  echo
  echo "=============================================================="
  echo "## PLOTS ($(date +%H:%M:%S))"
  echo "=============================================================="
  uv run python -m src.plots

  echo
  echo "##############################################################"
  echo "# Pipeline finalizado: $(date)"
  echo "##############################################################"
} 2>&1 | tee "$MASTER_LOG"

echo
echo "Log mestre salvo em: $MASTER_LOG"
