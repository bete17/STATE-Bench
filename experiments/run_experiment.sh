#!/usr/bin/env bash
# Run one agent setup on an experiment's fixed task lists, the three domains at once.
#
# usage: experiments/run_experiment.sh <experiment> <label> [agent-class]
#   <experiment>   a folder under experiments/ that holds tasks/<domain>.txt (for example pass1_75)
#   <label>        a name for this run's results (for example opencode_ag_train)
#   [agent-class]  an agent under agents/ (OpenCodeAgent, OpenCodeTrainAgent, OpenCodeAgTrainAgent, ...);
#                  leave it out for STATE-Bench's built-in agent
#
# Results go to experiments/<experiment>/<label>/<domain>/run1/..., which the tracker reads
# (python tools/tracker.py --outputs experiments/<experiment>).
#
# Optional settings, as environment variables:
#   MODEL      agent model                  (default gpt-6.1-sol)
#   REASONING  agent reasoning level        (default medium)
#   PROVIDER   agent provider               (default openai)
#   WORKERS    tasks at once, per domain    (default 3)
#   RUNS       attempts per task            (default 1)
#   PYTHON     how to run Python            (default "uv run python")
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
  sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
EXP=$1; LABEL=$2; CLASS=${3:-}
ROOT=experiments/$EXP
MODEL=${MODEL:-gpt-6.1-sol}; REASONING=${REASONING:-medium}; PROVIDER=${PROVIDER:-openai}
WORKERS=${WORKERS:-3}; RUNS=${RUNS:-1}
read -r -a PY <<< "${PYTHON:-uv run python}"
DOMAINS=(travel customer_support shopping_assistant)

# Every run of an experiment uses the same tasks: stop if the lists changed.
if [ -f "$ROOT/tasks/SHA256SUMS" ]; then
  (cd "$ROOT" && sha256sum -c --quiet tasks/SHA256SUMS) || { echo "task lists in $ROOT/tasks changed; refusing to run"; exit 1; }
fi
for D in "${DOMAINS[@]}"; do
  [ -s "$ROOT/tasks/$D.txt" ] || { echo "missing $ROOT/tasks/$D.txt (one task id per line)"; exit 1; }
  if ls -d "$ROOT/$LABEL/$D"/run* >/dev/null 2>&1; then
    echo "$ROOT/$LABEL/$D already has results; pick another label or move them away"; exit 1
  fi
done

export PYTHONUTF8=1            # Windows: save trajectories as UTF-8
start=$(date +%s)
pids=()
for D in "${DOMAINS[@]}"; do
  OUT=$ROOT/$LABEL/$D
  mkdir -p "$OUT"
  cp "$ROOT/tasks/$D.txt" "$OUT/tasks.txt"    # tells the tracker which tasks to expect
  ARGS=(--domain "$D" --tasks "$(paste -sd, "$ROOT/tasks/$D.txt")"
        --agent-provider "$PROVIDER" --agent-model-name "$MODEL" --agent-model-reasoning-level "$REASONING"
        --num-runs "$RUNS" --num-workers "$WORKERS" --output-dir "$OUT")
  [ -n "$CLASS" ] && ARGS+=(--agent-class "$CLASS")
  echo "starting $D -> $OUT (log: $ROOT/$LABEL/$D.log)"
  "${PY[@]}" -m state_bench.scripts.run_batch "${ARGS[@]}" > "$ROOT/$LABEL/$D.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "$LABEL finished in $(( $(date +%s) - start ))s$( [ $status -ne 0 ] && echo ' (a domain failed: see its .log)')"
exit $status
