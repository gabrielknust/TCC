#!/usr/bin/env bash
cd "$(dirname "$0")"

SAVE_DIR="${SAVE_DIR:-saved_model_v5}"
TOTAL=$(( 12 * $(echo "${SEEDS:-1 2 3 4 5}" | wc -w) ))

if pgrep -f run_retrain_v5.sh > /dev/null; then
  echo "Situação: rodando"
else
  echo "Situação: parado"
fi

DONE=$(ls "$SAVE_DIR" 2>/dev/null | grep -c -- '-last_')
echo "Treinos concluídos: $DONE de $TOTAL"

LOG=$(ls -t logs/retrain_v5_2*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  echo "Etapa atual: $(grep '^## ' "$LOG" | tail -1 | sed 's/^## //')"
  EPOCH=$(grep -oE 'Ep [0-9]+ \|.*Final: [0-9.]+%' "$LOG" | tail -1 | sed -E 's/ \| Train.*Final:/ | val final:/')
  [ -n "$EPOCH" ] && echo "Última época registrada: $EPOCH"
fi

FAILS=$(cat logs/retrain_v5_*.failed 2>/dev/null)
if [ -n "$FAILS" ]; then
  echo "Etapas que falharam:"
  echo "$FAILS"
else
  echo "Falhas: nenhuma"
fi

nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
  --format=csv,noheader 2>/dev/null | sed 's/^/GPU (uso, memória, temperatura): /'
