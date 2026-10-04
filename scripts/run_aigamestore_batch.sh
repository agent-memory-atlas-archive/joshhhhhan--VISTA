#!/usr/bin/env bash
# Play several AI GameStore games with bounded parallelism.
#
# The CLI runs its planned games sequentially; this driver launches one CLI
# process per game so that -j games play at once, each with its own output
# directory under a batch directory:
#
#   <runs>/<batch>/game<N>/           one `vista-aigamestore run` output
#   <runs>/<batch>/game<N>.log        that process's stdout/stderr
#   <runs>/<batch>/driver.log         start/finish lines
#   <runs>/<batch>/batch.json         the driver's own settings
#
# Every knob is an argument or environment variable; nothing is inferred from
# the shell profile. CODEX_HOME is unset on purpose: the runner prepares its own.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${AIGS_PYTHON:-"$ROOT/.venv/bin/python"}
CLI=${AIGS_CLI:-"$ROOT/.venv/bin/vista-aigamestore"}
RUNS_DIR=${AIGS_RUNS_DIR:-"$ROOT/runs"}
RUNTIME_HOME=${AIGS_HOME:-"$HOME"}
BROWSERS=${PLAYWRIGHT_BROWSERS_PATH:-}
CODEX_BIN=${AIGS_CODEX_BIN:-}

JOBS=2
BATCH_ID="aigs"
BACKEND="codex"
MODEL="gpt-5.6-sol"
EFFORT="max"
REPEATS=1
GAMES_ROOT=${AIGAMESTORE_GAMES_ROOT:-}
AUTH_FILE=${CODEX_AUTH_FILE:-"$HOME/.codex/auth.json"}
GAMES=(1 2 3 4 5 6 7 8 9 10)
EXTRA=()

usage() {
  cat <<'EOF'
usage: run_aigamestore_batch.sh [options] [-- extra CLI args]
  -j N                parallel games (default 2)
  --batch-id NAME     batch directory prefix (default aigs)
  --games-root DIR    the harness checkout's games/ (default $AIGAMESTORE_GAMES_ROOT)
  --backend codex
  --model M --effort E
  --repeats N         independent runs per game (default 1)
  --auth-file PATH    Codex auth.json (default ~/.codex/auth.json)
  --games "1 2 3"     public game numbers (default all ten)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    --batch-id) BATCH_ID="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --effort) EFFORT="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --games-root) GAMES_ROOT="$2"; shift 2 ;;
    --auth-file) AUTH_FILE="$2"; shift 2 ;;
    --games) IFS=' ' read -ra GAMES <<< "$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ $BACKEND == codex ]] || { echo "--backend must be codex" >&2; exit 2; }
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || { echo "-j must be a positive integer" >&2; exit 2; }
[[ $REPEATS =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be a positive integer" >&2; exit 2; }
[[ ${#GAMES[@]} -gt 0 ]] || { echo "No games selected" >&2; exit 2; }
[[ -n $GAMES_ROOT && -d $GAMES_ROOT ]] || { echo "--games-root (or AIGAMESTORE_GAMES_ROOT) must name the harness checkout's games/ directory" >&2; exit 2; }
[[ -n $AUTH_FILE && -f $AUTH_FILE ]] || { echo "no Codex auth.json at $AUTH_FILE; run codex login or pass --auth-file" >&2; exit 2; }
[[ -x $CLI ]] || { echo "CLI not found: $CLI (pip install -e '.[aigamestore]')" >&2; exit 2; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BATCH_DIR="$RUNS_DIR/${BATCH_ID}_${STAMP}"
mkdir -p "$BATCH_DIR"
LOG="$BATCH_DIR/driver.log"

log() {
  printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

"$PYTHON" - "$BATCH_DIR/batch.json" "$BATCH_ID" "$BACKEND" "$MODEL" "$EFFORT" \
  "$REPEATS" "$GAMES_ROOT" "$JOBS" "${GAMES[*]}" <<'PY'
import json, sys
from pathlib import Path
path, batch_id, backend, model, effort, repeats, games_root, jobs, games = sys.argv[1:]
Path(path).write_text(json.dumps({
    "benchmark": "AI GameStore",
    "batch_id": batch_id,
    "backend": backend,
    "model": model,
    "effort": effort,
    "repeats": int(repeats),
    "games_root": games_root,
    "jobs": int(jobs),
    "games": [int(g) for g in games.split()],
}, indent=2) + "\n")
PY

log "BEGIN $BATCH_ID -- $BACKEND $MODEL/$EFFORT repeats=$REPEATS jobs=$JOBS games=${GAMES[*]}"
log "games root $GAMES_ROOT"
log "batch dir $BATCH_DIR"

run_one() {
  local game="$1"
  local out="$BATCH_DIR/game${game}"
  local glog="$BATCH_DIR/game${game}.log"
  log "start game$game"
  local args=(
    run --games-root "$GAMES_ROOT" --game "$game"
    --backend "$BACKEND" --model "$MODEL" --effort "$EFFORT"
    --repeats "$REPEATS"
    --output "$out"
  )
  args+=(--auth-file "$AUTH_FILE")
  [[ -n $CODEX_BIN ]] && args+=(--binary "$CODEX_BIN")
  local code=0
  env -u CODEX_HOME HOME="$RUNTIME_HOME" \
    ${BROWSERS:+PLAYWRIGHT_BROWSERS_PATH="$BROWSERS"} \
    "$CLI" "${args[@]}" "${EXTRA[@]}" >"$glog" 2>&1 || code=$?
  log "done  game$game (exit $code)"
  return "$code"
}

active=0
pids=()
for game in "${GAMES[@]}"; do
  run_one "$game" &
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

"$PYTHON" - "$BATCH_DIR" "$REPEATS" "${GAMES[*]}" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
from vista.benchmarks.aigamestore.run import aggregate
batch, repeats, games = Path(sys.argv[1]), int(sys.argv[2]), [int(g) for g in sys.argv[3].split()]
results = [json.loads(p.read_text()) for p in sorted(batch.glob("game*/game*__*/result.json"))]
summary = [aggregate(results, game_numbers=games, repeats=repeats)]
(batch / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for row in summary:
    print(f"{row['scored']}/{row['planned']} scored, "
          f"geometric mean {row['geometric_mean_normalized']}")
if not all(row["complete"] for row in summary):
    raise SystemExit(1)
PY

log "END   $BATCH_ID (exit $failed)"
exit "$failed"
