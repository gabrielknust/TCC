#!/usr/bin/env bash
# Retreino v4 completo, para rodar sem supervisão.
#
#   ./run_retrain_v4.sh              (em primeiro plano)
#   nohup ./run_retrain_v4.sh &      (sobrevive ao fechar o terminal)
#
# Faz, em sequência: grade b2-b5 x {equal,linspace}, variantes de arquitetura
# v3a/v3b (b3+equal) e, no fim, os gráficos. Cada treino gera DOIS checkpoints
# (melhor val loss e melhor val acc), e ambos passam por calibração+inferência.
#
# Não usa `set -e`: se uma etapa falhar, as demais continuam e o resumo final
# lista o que quebrou — melhor perder uma etapa do que a noite inteira.
set -uo pipefail

cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p logs
MASTER_LOG="logs/retrain_v4_${TIMESTAMP}.log"
FAILED=()
STEP=0
TOTAL=30   # 6 treinos + 12 calibrações + 12 inferências

run_step() {
  local desc="$1"; shift
  STEP=$((STEP + 1))
  echo
  echo "=============================================================="
  echo "## [$STEP/$TOTAL] $desc  ($(date '+%d/%m %H:%M:%S'))"
  echo "=============================================================="
  if "$@"; then
    echo "## OK: $desc"
  else
    echo "## !!! FALHOU: $desc"
    FAILED+=("$desc")
  fi
}

# $1=n_branches  $2=loss_weights  $3=variante  $4=extra_stride1 (opcional)
pipeline() {
  local n="$1" lw="$2" var="$3" es="${4:-}"
  local accvar="${var:+${var}-}acc"
  local name="${var:-padrão}"

  export N_BRANCHES="$n" LOSS_WEIGHTS_LIST="$lw" EXTRA_STRIDE1="$es"

  export VARIANT="$var"
  run_step "TREINO b$n [$lw] ($name)" uv run python -m src.train

  for v in "$var" "$accvar"; do
    export VARIANT="$v"
    run_step "CALIBRAÇÃO b$n [$lw] (${v:-padrão})" uv run python -m src.calibrate
    run_step "INFERÊNCIA b$n [$lw] (${v:-padrão})" uv run python -m src.inference
  done
}

{
  echo "##############################################################"
  echo "# RETREINO v4 — iniciado em $(date)"
  echo "# Correções: classificador final treina no warmup; dropout final;"
  echo "#            duplo checkpoint (loss/acc); val dividida 5k+5k; min_delta"
  echo "##############################################################"
  START=$(date +%s)

  # --- Grade principal: b2-b5 x {equal, linspace} ---
  for n in 2 3 4 5; do
    pipeline "$n" "equal,linspace" ""
  done

  # --- Variantes de arquitetura (b3 + equal) ---
  pipeline 3 "equal" "s14" "14"
  pipeline 3 "equal" "s7-14" "7,14"

  # --- Figuras ---
  unset VARIANT EXTRA_STRIDE1 N_BRANCHES LOSS_WEIGHTS_LIST
  run_step "GRÁFICOS" uv run python -m src.plots

  ELAPSED=$(( ($(date +%s) - START) / 60 ))
  echo
  echo "##############################################################"
  echo "# RETREINO v4 — finalizado em $(date)  (${ELAPSED} min)"
  if [ ${#FAILED[@]} -eq 0 ]; then
    echo "# TUDO OK — nenhuma etapa falhou."
  else
    echo "# ETAPAS QUE FALHARAM (${#FAILED[@]}):"
    printf '#   - %s\n' "${FAILED[@]}"
  fi
  echo "##############################################################"
} 2>&1 | tee "$MASTER_LOG"

echo
echo "Log mestre: $MASTER_LOG"
