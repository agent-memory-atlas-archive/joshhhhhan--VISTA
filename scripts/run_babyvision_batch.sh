#!/usr/bin/env bash
# Answer a BabyVision question set with bounded parallelism.
#
# The CLI runs its planned questions sequentially; this driver launches one
# CLI process per question so that -j questions run at once, each with its own
# output directory under a batch directory:
#
#   <runs>/<batch>/task<ID>/          one `vista-babyvision run` output
#   <runs>/<batch>/task<ID>.log       that process's stdout/stderr
#   <runs>/<batch>/driver.log         start/finish lines
#   <runs>/<batch>/batch.json         the driver's own settings
#   <runs>/<batch>/batch_summary.json accuracy at the end
#
# Every knob is an argument or environment variable; nothing is inferred from
# the shell profile. CODEX_HOME is unset on purpose: the runner prepares its own.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${BV_PYTHON:-"$ROOT/.venv/bin/python"}
CLI=${BV_CLI:-"$ROOT/.venv/bin/vista-babyvision"}
RUNS_DIR=${BV_RUNS_DIR:-"$ROOT/runs"}
RUNTIME_HOME=${BV_HOME:-"$HOME"}
CODEX_BIN=${BV_CODEX_BIN:-}
BABYVISION_ROOT=${BABYVISION_ROOT:-}

JOBS=2
BATCH_ID="bv"
BACKEND="codex"
MODEL="gpt-5.6-sol"
EFFORT="max"
PASSES=1
TASK_SET="tracking48"
TASK_IDS=()
SUBTYPES=()
AUTH_FILE=${CODEX_AUTH_FILE:-"$HOME/.codex/auth.json"}
EXTRA=()

usage() {
  cat <<'EOF2'
usage: run_babyvision_batch.sh [options] [-- extra CLI args]
  -j N                  parallel questions (default 2)
  --batch-id NAME       batch directory prefix (default bv)
  --babyvision-root DIR pinned checkout (default $BABYVISION_ROOT)
  --backend codex
  --model M --effort E
  --passes N            passes per question (default 1; official reporting uses 3)
  --task-set S          full | tracking48 (default tracking48)
  --task-ids "1 2 3"    explicit task ids (overrides the task set)
  --subtypes "A|B"      explicit subtypes, '|'-separated (overrides the task set)
  --auth-file PATH      Codex auth.json (default ~/.codex/auth.json)
EOF2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    --batch-id) BATCH_ID="$2"; shift 2 ;;
    --babyvision-root) BABYVISION_ROOT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --effort) EFFORT="$2"; shift 2 ;;
    --passes) PASSES="$2"; shift 2 ;;
    --task-set) TASK_SET="$2"; shift 2 ;;
    --task-ids) IFS=' ' read -ra TASK_IDS <<< "$2"; shift 2 ;;
    --subtypes) IFS='|' read -ra SUBTYPES <<< "$2"; shift 2 ;;
    --auth-file) AUTH_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ $BACKEND == codex ]] || { echo "--backend must be codex" >&2; exit 2; }
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || { echo "-j must be a positive integer" >&2; exit 2; }
[[ $PASSES =~ ^[1-9][0-9]*$ ]] || { echo "--passes must be a positive integer" >&2; exit 2; }
[[ -n $BABYVISION_ROOT && -f $BABYVISION_ROOT/data/babyvision_data/meta_data.jsonl ]] || {
  echo "--babyvision-root (or BABYVISION_ROOT) must name an unpacked checkout" >&2; exit 2; }
[[ -n $AUTH_FILE && -f $AUTH_FILE ]] || { echo "no Codex auth.json at $AUTH_FILE; run codex login or pass --auth-file" >&2; exit 2; }
[[ -x $CLI ]] || { echo "CLI not found: $CLI (pip install -e '.[babyvision]')" >&2; exit 2; }

# The question set is resolved once, by the CLI, so the driver and the runs
# agree on it exactly.
select_args=(--babyvision-root "$BABYVISION_ROOT" --task-set "$TASK_SET")
for id in "${TASK_IDS[@]:-}"; do [[ -n $id ]] && select_args+=(--task-id "$id"); done
for st in "${SUBTYPES[@]:-}"; do [[ -n $st ]] && select_args+=(--subtype "$st"); done
question_list=$("$CLI" list "${select_args[@]}" | "$PYTHON" -c 'import json,sys; [print(q["task_id"]) for q in json.load(sys.stdin)["questions"]]')
[[ -n $question_list ]] || { echo "No questions selected" >&2; exit 2; }
mapfile -t QUESTIONS <<< "$question_list"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BATCH_DIR="$RUNS_DIR/${BATCH_ID}_${STAMP}"
mkdir -p "$BATCH_DIR"
LOG="$BATCH_DIR/driver.log"

log() {
  printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

"$PYTHON" - "$BATCH_DIR/batch.json" "$BATCH_ID" "$BACKEND" "$MODEL" "$EFFORT" \
  "$PASSES" "$TASK_SET" "$JOBS" "${QUESTIONS[*]}" "$BABYVISION_ROOT" <<'PY'
import json, sys
from pathlib import Path
path, batch_id, backend, model, effort, passes, task_set, jobs, questions, root = sys.argv[1:]
Path(path).write_text(json.dumps({
    "benchmark": "BabyVision",
    "batch_id": batch_id,
    "backend": backend,
    "model": model,
    "effort": effort,
    "passes": int(passes),
    "task_set": task_set,
    "babyvision_root": root,
    "jobs": int(jobs),
    "questions": [int(q) for q in questions.split()],
}, indent=2) + "\n")
PY

log "BEGIN $BATCH_ID -- $BACKEND $MODEL/$EFFORT passes=$PASSES jobs=$JOBS questions=${#QUESTIONS[@]}"
log "batch dir $BATCH_DIR"

run_one() {
  local task="$1"
  local out="$BATCH_DIR/task${task}"
  local tlog="$BATCH_DIR/task${task}.log"
  log "start task$task"
  local args=(
    run --babyvision-root "$BABYVISION_ROOT" --task-id "$task"
    --backend "$BACKEND" --model "$MODEL" --effort "$EFFORT"
    --passes "$PASSES"
    --output "$out"
  )
  args+=(--auth-file "$AUTH_FILE")
  [[ -n $CODEX_BIN ]] && args+=(--binary "$CODEX_BIN")
  local code=0
  env -u CODEX_HOME HOME="$RUNTIME_HOME" \
    "$CLI" "${args[@]}" "${EXTRA[@]}" >"$tlog" 2>&1 || code=$?
  log "done  task$task (exit $code)"
  return "$code"
}

active=0
pids=()
for task in "${QUESTIONS[@]}"; do
  run_one "$task" &
  pids+=("$!")
  active=$((active + 1))
  if [[ $active -ge $JOBS ]]; then
    wait -n || true
    active=$((active - 1))
  fi
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

"$CLI" score --runs "$BATCH_DIR" > "$BATCH_DIR/batch_summary.json"
"$PYTHON" - "$BATCH_DIR/batch_summary.json" <<'PY' | tee -a "$LOG"
import json, sys
rows = json.load(open(sys.argv[1]))
for row in rows:
    print(f"{row['correct']}/{row['planned']} correct "
          f"({row['completed']} completed), mean accuracy {row['mean_accuracy']} [{row['score_source']}]")
if not rows or any(row["completed"] != row["planned"] for row in rows):
    raise SystemExit(1)
PY

log "END   $BATCH_ID (exit $failed)"
exit "$failed"
