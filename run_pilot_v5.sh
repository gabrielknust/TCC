#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
STATE_DIR="logs/pilot_v5_state"
SAVE_DIR="saved_model_v5"
EPOCHS="${EPOCHS:-200}"
mkdir -p logs "$STATE_DIR" "$SAVE_DIR"
export TQDM_DISABLE=1
FAILED="logs/pilot_v5_${TIMESTAMP}.failed"

step() {
  local log="$1" desc="$2"; shift 2
  echo "## $(date '+%d/%m %H:%M:%S') INÍCIO: $desc" >> "$log"
  if "$@" >> "$log" 2>&1; then
    echo "## $(date '+%d/%m %H:%M:%S') OK: $desc" >> "$log"
    return 0
  fi
  echo "## $(date '+%d/%m %H:%M:%S') FALHOU: $desc" >> "$log"
  echo "$desc" >> "$FAILED"
  return 1
}

train() {
  local log="$1" n="$2" w="$3"
  local marker="$STATE_DIR/train_b${n}_${w}"
  [ -f "$marker" ] && return 0
  step "$log" "TREINO b$n $w ($EPOCHS épocas)" \
    env N_BRANCHES="$n" LOSS_WEIGHTS_LIST="$w" VARIANT=r1 TRAIN_SEED=1 SPLIT_SEED=42 \
        SAVE_DIR="$SAVE_DIR" EPOCHS="$EPOCHS" uv run python -m src.train \
    && touch "$marker"
}

START=$(date +%s)
echo "Piloto iniciado em $(date). Logs: logs/pilot_v5_${TIMESTAMP}_*.log"

if [ ! -f "$STATE_DIR/smoke" ]; then
  SMOKE_DIR=$(mktemp -d)
  echo "Teste rápido (6 épocas, cerca de 3 min)..."
  if step "logs/pilot_v5_${TIMESTAMP}_teste.log" "TESTE RÁPIDO" \
      env N_BRANCHES=3 LOSS_WEIGHTS_LIST=equal VARIANT=smoke SAVE_DIR="$SMOKE_DIR" EPOCHS=6 \
          uv run python -m src.train; then
    touch "$STATE_DIR/smoke"
    rm -rf "$SMOKE_DIR"
  else
    echo "O teste rápido falhou. Veja logs/pilot_v5_${TIMESTAMP}_teste.log"
    exit 1
  fi
fi

( train "logs/pilot_v5_${TIMESTAMP}_a.log" 3 equal; train "logs/pilot_v5_${TIMESTAMP}_a.log" 0 equal ) &
sleep 30
train "logs/pilot_v5_${TIMESTAMP}_b.log" 3 linspace &
wait

echo "Treinos finalizados em $(date) ($(( ($(date +%s) - START) / 60 )) min). Comparando com a receita atual."
step "logs/pilot_v5_${TIMESTAMP}_comparacao.log" "COMPARAÇÃO" uv run python -m src.pilot
cat "logs/pilot_v5_${TIMESTAMP}_comparacao.log"
if [ -s "$FAILED" ]; then
  echo "Etapas que falharam:"
  cat "$FAILED"
else
  echo "Nenhuma etapa falhou."
fi
