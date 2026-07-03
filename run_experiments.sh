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
# A saída completa é gravada em logs/experiments_<timestamp>.log via tee.
set -euo pipefail

cd "$(dirname "$0")"

N_BRANCHES_LIST=(2 3 4 5)
export LOSS_WEIGHTS_LIST="equal,linspace"

mkdir -p logs
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
MASTER_LOG="logs/experiments_${TIMESTAMP}.log"

{
  echo "##############################################################"
  echo "# Pipeline iniciado: $(date)"
  echo "# N_BRANCHES_LIST: ${N_BRANCHES_LIST[*]}"
  echo "# LOSS_WEIGHTS_LIST: ${LOSS_WEIGHTS_LIST}"
  echo "##############################################################"

  for n in "${N_BRANCHES_LIST[@]}"; do
    export N_BRANCHES=$n
    echo
    echo "=============================================================="
    echo "## N_BRANCHES=$n :: TREINO ($(date +%H:%M:%S))"
    echo "=============================================================="
    uv run python -m src.train

    echo
    echo "=============================================================="
    echo "## N_BRANCHES=$n :: CALIBRAÇÃO ($(date +%H:%M:%S))"
    echo "=============================================================="
    uv run python -m src.calibrate

    echo
    echo "=============================================================="
    echo "## N_BRANCHES=$n :: INFERÊNCIA ($(date +%H:%M:%S))"
    echo "=============================================================="
    uv run python -m src.inference
  done

  echo
  echo "##############################################################"
  echo "# Pipeline finalizado: $(date)"
  echo "##############################################################"
} 2>&1 | tee "$MASTER_LOG"

echo
echo "Log mestre salvo em: $MASTER_LOG"
