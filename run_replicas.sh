#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
STATE_DIR="logs/replicas_state"
mkdir -p logs "$STATE_DIR"
export TQDM_DISABLE=1

step() {
  local log="$1" desc="$2"; shift 2
  echo "## $(date '+%d/%m %H:%M:%S') INÍCIO: $desc" >> "$log"
  if "$@" >> "$log" 2>&1; then
    echo "## $(date '+%d/%m %H:%M:%S') OK: $desc" >> "$log"
    return 0
  fi
  echo "## $(date '+%d/%m %H:%M:%S') FALHOU: $desc" >> "$log"
  echo "$desc" >> "logs/replicas_${TIMESTAMP}.failed"
  return 1
}

tag_for() {
  local n="$1" w="$2" v="$3"
  echo "mobilenet_${v:+${v}_}b${n}_${w}_best"
}

run_config() {
  local log="$1" n="$2" w="$3" v="$4" es="$5" seed="$6" cascade="$7"
  local accv="${v:+${v}-}acc"
  local tag; tag=$(tag_for "$n" "$w" "$v")
  if [ ! -f "$STATE_DIR/train_${tag}" ]; then
    step "$log" "TREINO $tag (semente $seed)" \
      env N_BRANCHES="$n" LOSS_WEIGHTS_LIST="$w" VARIANT="$v" EXTRA_STRIDE1="$es" \
          TRAIN_SEED="$seed" uv run python -m src.train \
      && touch "$STATE_DIR/train_${tag}" || return
  fi
  [ "$cascade" = "yes" ] || return 0
  for cv in "$v" "$accv"; do
    local ctag; ctag=$(tag_for "$n" "$w" "$cv")
    [ -f "$STATE_DIR/cascade_${ctag}" ] && continue
    step "$log" "CALIBRAÇÃO $ctag" \
      env N_BRANCHES="$n" LOSS_WEIGHTS_LIST="$w" VARIANT="$cv" uv run python -m src.calibrate \
      && step "$log" "INFERÊNCIA $ctag" \
      env N_BRANCHES="$n" LOSS_WEIGHTS_LIST="$w" VARIANT="$cv" uv run python -m src.inference \
      && touch "$STATE_DIR/cascade_${ctag}"
  done
}

chain() {
  local r="$1"
  local log="logs/replicas_${TIMESTAMP}_r${r}.log"
  for n in 2 3 4 5; do
    for w in equal linspace; do
      run_config "$log" "$n" "$w" "r${r}" "" "$r" yes
    done
  done
  run_config "$log" 3 equal "s14-r${r}" 14 "$r" yes
  run_config "$log" 3 equal "s7-14-r${r}" 7,14 "$r" yes
  run_config "$log" 0 equal "r${r}" "" "$r" no
  if [ "$r" = "1" ]; then
    run_config "$log" 0 equal "" "" 0 no
  fi
}

START=$(date +%s)
echo "Réplicas iniciadas em $(date). Logs: logs/replicas_${TIMESTAMP}_r*.log"
chain 1 &
sleep 30
chain 2 &
wait
echo "Treinos finalizados em $(date) ($(( ($(date +%s) - START) / 60 )) min). Consolidando resultados."
CONSOLIDATION_LOG="logs/replicas_${TIMESTAMP}_consolidacao.log"
step "$CONSOLIDATION_LOG" "ESTATÍSTICA" uv run python -m src.stats \
  && step "$CONSOLIDATION_LOG" "RÉPLICAS" uv run python -m src.replicas \
  && step "$CONSOLIDATION_LOG" "POR CLASSE" uv run python -m src.per_class \
  && step "$CONSOLIDATION_LOG" "GRÁFICOS" uv run python -m src.plots
echo "Réplicas finalizadas em $(date) ($(( ($(date +%s) - START) / 60 )) min)"
if [ -s "logs/replicas_${TIMESTAMP}.failed" ]; then
  echo "Etapas que falharam:"
  cat "logs/replicas_${TIMESTAMP}.failed"
else
  echo "Nenhuma etapa falhou."
fi
