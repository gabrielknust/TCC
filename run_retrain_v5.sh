#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
export SAVE_DIR="${SAVE_DIR:-saved_model_v5}"
export RESULTS_DIR="${RESULTS_DIR:-results_v5}"
export EPOCHS="${EPOCHS:-200}"
export TQDM_DISABLE=1
SEEDS="${SEEDS:-1 2 3 4 5}"
SCRATCH="${SCRATCH:-1}"
INV_LINSPACE="${INV_LINSPACE:-1}"
STATE_DIR="${STATE_DIR:-logs/retrain_v5_state}"
LOG="logs/retrain_v5_${TIMESTAMP}.log"
FAILED="logs/retrain_v5_${TIMESTAMP}.failed"
mkdir -p logs "$STATE_DIR" "$SAVE_DIR" "$RESULTS_DIR"
exec 9>"logs/retrain_v5.lock"
if ! flock -n 9; then
  echo "O retreino já está rodando em outro terminal."
  exit 1
fi

step() {
  local desc="$1"; shift
  local start; start=$(date +%s)
  echo "## $(date '+%d/%m %H:%M:%S') INÍCIO: $desc" | tee -a "$LOG"
  if "$@" >> "$LOG" 2>&1; then
    echo "## $(date '+%d/%m %H:%M:%S') OK: $desc ($(( ($(date +%s) - start) / 60 )) min)" | tee -a "$LOG"
    return 0
  fi
  echo "## $(date '+%d/%m %H:%M:%S') FALHOU: $desc" | tee -a "$LOG"
  echo "$desc" >> "$FAILED"
  return 1
}

trained() {
  local hist="$SAVE_DIR/history/mobilenet_${1}_b${2}_${3}.csv"
  [ -f "$hist" ] && [ "$(wc -l < "$hist")" -eq $((EPOCHS + 1)) ] \
    && [ -f "$SAVE_DIR/mobilenet_${1}-last_b${2}_${3}_best.pth" ]
}

run_config() {
  local n="$1" w="$2" v="$3" es="$4" seed="$5" pretrained="${6:-1}"
  local split=$((41 + seed))
  if ! trained "$v" "$n" "$w"; then
    step "TREINO b$n $w $v (semente $seed, divisão $split)" \
      env N_BRANCHES="$n" LOSS_WEIGHTS_LIST="$w" VARIANT="$v" EXTRA_STRIDE1="$es" \
          TRAIN_SEED="$seed" SPLIT_SEED="$split" PRETRAINED="$pretrained" \
          uv run python -m src.train || return
  fi
  [ "$n" -gt 0 ] || return 0
  for cv in "$v" "${v}-acc" "${v}-last"; do
    local marker="$STATE_DIR/cascade_${cv}_b${n}_${w}"
    [ -f "$marker" ] && continue
    step "CALIBRAÇÃO $cv b$n $w" \
      env N_BRANCHES="$n" LOSS_WEIGHTS_LIST="$w" VARIANT="$cv" uv run python -m src.calibrate \
      && step "INFERÊNCIA $cv b$n $w" \
      env N_BRANCHES="$n" LOSS_WEIGHTS_LIST="$w" VARIANT="$cv" uv run python -m src.inference \
      && touch "$marker"
  done
}

START=$(date +%s)
echo "Retreino iniciado em $(date). Sementes: $SEEDS. Log completo: $LOG"
for seed in $SEEDS; do
  for n in 2 3 4 5; do
    for w in equal linspace; do
      run_config "$n" "$w" "r$seed" "" "$seed"
    done
  done
  run_config 0 equal "r$seed" "" "$seed"
  run_config 3 equal "s7-14-r$seed" "7,14" "$seed"
  [ "$SCRATCH" = "1" ] && run_config 0 equal "scratch-r$seed" "" "$seed" 0
  [ "$INV_LINSPACE" = "1" ] && run_config 3 inv_linspace "r$seed" "" "$seed"
  echo "== Semente $seed concluída em $(date) ($(( ($(date +%s) - START) / 60 )) min desde o início)" | tee -a "$LOG"
done

echo "Retreino finalizado em $(date) ($(( ($(date +%s) - START) / 60 )) min)"
if [ -s "$FAILED" ]; then
  echo "Etapas que falharam (rode o script de novo para refazê-las):"
  cat "$FAILED"
else
  echo "Nenhuma etapa falhou."
fi
