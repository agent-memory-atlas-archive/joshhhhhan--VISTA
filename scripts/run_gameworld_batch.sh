#!/usr/bin/env bash
# Play the GameWorld catalog with bounded parallelism.
#
# The CLI runs its planned tasks sequentially; this driver launches one CLI
# process per task so that -j tasks play at once, each with its own output
# directory under a batch directory:
#
#   <runs>/<batch>/<game>__<task>/       one `vista-gameworld run` output
#   <runs>/<batch>/<game>__<task>.log    that process's stdout/stderr
#   <runs>/<batch>/driver.log            start/finish lines
#   <runs>/<batch>/batch.json            the driver's own settings
#
# Tasks are ordered breadth-first across games (every game's first task,
# then every game's second, ...) so a batch stopped early still covers the
# catalog. CODEX_HOME is unset on purpose: the runner prepares its own.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${GW_PYTHON:-"$ROOT/.venv/bin/python"}
CLI=${GW_CLI:-"$ROOT/.venv/bin/vista-gameworld"}
RUNS_DIR=${GW_RUNS_DIR:-"$ROOT/runs"}
RUNTIME_HOME=${GW_HOME:-"$HOME"}
BROWSERS=${PLAYWRIGHT_BROWSERS_PATH:-}
CODEX_BIN=${GW_CODEX_BIN:-}
GAMEWORLD_ROOT=${GAMEWORLD_ROOT:-}
GAMES_ROOT=${GAMES_ROOT:-}

JOBS=2
BATCH_ID="gw"
BACKEND="codex"
MODEL="gpt-5.6-sol"
EFFORT="max"
GAMES=()
AUTH_FILE=${CODEX_AUTH_FILE:-"$HOME/.codex/auth.json"}
EXTRA=()

usage() {
  cat <<'EOF2'
usage: run_gameworld_batch.sh [options] [-- extra CLI args]
  -j N                  parallel tasks (default 2)
  --batch-id NAME       batch directory prefix (default gw)
  --gameworld-root DIR  pinned GameWorld checkout (default $GAMEWORLD_ROOT)
  --games-root DIR      pinned GameWorld-Games checkout (default $GAMES_ROOT)
  --backend codex
  --model M --effort E
  --games "01_2048 05_breakout"   restrict to these games (default: whole catalog)
  --auth-file PATH      Codex auth.json (default ~/.codex/auth.json)
EOF2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    --batch-id) BATCH_ID="$2"; shift 2 ;;
    --gameworld-root) GAMEWORLD_ROOT="$2"; shift 2 ;;
    --games-root) GAMES_ROOT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --effort) EFFORT="$2"; shift 2 ;;
    --games) IFS=' ' read -ra GAMES <<< "$2"; shift 2 ;;
    --auth-file) AUTH_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ $BACKEND == codex ]] || { echo "--backend must be codex" >&2; exit 2; }
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || { echo "-j must be a positive integer" >&2; exit 2; }
[[ -n $GAMEWORLD_ROOT && -d $GAMEWORLD_ROOT/.git ]] || { echo "--gameworld-root (or GAMEWORLD_ROOT) must name the pinned checkout" >&2; exit 2; }
[[ -n $GAMES_ROOT && -d $GAMES_ROOT/.git ]] || { echo "--games-root (or GAMES_ROOT) must name the pinned checkout" >&2; exit 2; }
[[ -n $AUTH_FILE && -f $AUTH_FILE ]] || { echo "no Codex auth.json at $AUTH_FILE; run codex login or pass --auth-file" >&2; exit 2; }
[[ -x $CLI ]] || { echo "CLI not found: $CLI (pip install -e '.[gameworld]')" >&2; exit 2; }

# The task list comes from the CLI's own catalog reading, reordered
# breadth-first across games.
task_list=$("$CLI" list --gameworld-root "$GAMEWORLD_ROOT" --games-root "$GAMES_ROOT" \
  | "$PYTHON" -c '
import json, sys, collections
wanted = set(sys.argv[1:])
tasks = json.load(sys.stdin)["tasks"]
by_game = collections.OrderedDict()
for game, task in tasks:
    if wanted and game not in wanted:
        continue
    by_game.setdefault(game, []).append(task)
depth = max((len(v) for v in by_game.values()), default=0)
for i in range(depth):
    for game, items in by_game.items():
        if i < len(items):
            print(f"{game} {items[i]}")
' "${GAMES[@]}")
[[ -n $task_list ]] || { echo "No tasks selected" >&2; exit 2; }
mapfile -t TASKS <<< "$task_list"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BATCH_DIR="$RUNS_DIR/${BATCH_ID}_${STAMP}"
mkdir -p "$BATCH_DIR"
LOG="$BATCH_DIR/driver.log"

log() {
  printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

"$PYTHON" - "$BATCH_DIR/batch.json" "$BATCH_ID" "$BACKEND" "$MODEL" "$EFFORT" \
  "$JOBS" "$GAMEWORLD_ROOT" "$GAMES_ROOT" "${TASKS[@]}" <<'PY'
import json, sys
from pathlib import Path
path, batch_id, backend, model, effort, jobs, gw_root, games_root, *tasks = sys.argv[1:]
Path(path).write_text(json.dumps({
    "benchmark": "GameWorld",
    "batch_id": batch_id,
    "backend": backend,
    "model": model,
    "effort": effort,
    "gameworld_root": gw_root,
    "games_root": games_root,
    "jobs": int(jobs),
    "tasks": [t.split(" ") for t in tasks],
}, indent=2) + "\n")
PY

log "BEGIN $BATCH_ID -- $BACKEND $MODEL/$EFFORT jobs=$JOBS tasks=${#TASKS[@]}"
log "batch dir $BATCH_DIR"

run_one() {
  local game="$1" task="$2"
  local out="$BATCH_DIR/${game}__${task}"
  local tlog="$BATCH_DIR/${game}__${task}.log"
  log "start ${game}/${task}"
  local args=(
    run --gameworld-root "$GAMEWORLD_ROOT" --games-root "$GAMES_ROOT"
    --game "$game" --task "$task"
    --backend "$BACKEND" --model "$MODEL" --effort "$EFFORT"
    --output "$out"
  )
  args+=(--auth-file "$AUTH_FILE")
  [[ -n $CODEX_BIN ]] && args+=(--binary "$CODEX_BIN")
  local code=0
  env -u CODEX_HOME HOME="$RUNTIME_HOME" \
    ${BROWSERS:+PLAYWRIGHT_BROWSERS_PATH="$BROWSERS"} \
    "$CLI" "${args[@]}" "${EXTRA[@]}" >"$tlog" 2>&1 || code=$?
  log "done  ${game}/${task} (exit $code)"
  return "$code"
}

active=0
pids=()
for entry in "${TASKS[@]}"; do
  run_one ${entry} &
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

"$PYTHON" - "$BATCH_DIR" <<'PY' | tee -a "$LOG"
import json, sys, glob
from pathlib import Path
batch = Path(sys.argv[1])
rows = [json.loads(p.read_text()) for p in sorted(batch.glob("*__*/*__*/result.json"))]
planned = len(json.loads((batch / "batch.json").read_text())["tasks"])
scored = [r for r in rows if r.get("scored")]
summary = {
    "planned": planned,
    "completed": sum(1 for r in rows if r.get("completed")),
    "scored": len(scored),
    "successes": sum(1 for r in scored if r.get("success")),
    "success_rate_over_planned": (sum(1 for r in scored if r.get("success")) / planned) if planned else None,
    "mean_progress_over_scored": (sum(r["progress"] for r in scored) / len(scored)) if scored else None,
}
(batch / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"{summary['successes']} successes of {summary['scored']} scored / {planned} planned")
if summary["completed"] != planned or summary["scored"] != planned:
    raise SystemExit(1)
PY

log "END   $BATCH_ID (exit $failed)"
exit "$failed"
