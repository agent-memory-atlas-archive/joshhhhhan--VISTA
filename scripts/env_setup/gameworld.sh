#!/usr/bin/env bash
# Prepare the GameWorld environment: the extras, a Chromium, and both halves of
# the benchmark at the revisions this runtime is pinned to.
#
#   ./scripts/env_setup/gameworld.sh [--dest DIR] [--python PATH] [--no-install]
set -euo pipefail
usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_common_args "$@"

install_extras gameworld
install_chromium

# Upstream splits the benchmark in two, and both halves are needed: the runtime,
# task catalogue and evaluator (~20 MB), and the games themselves -- HTML,
# JavaScript and art (~220 MB).
clone_pinned https://github.com/gameworld-project/GameWorld.git \
  "$DEST/GameWorld" "$(pin_from src/vista/benchmarks/gameworld/upstream.py GAMEWORLD_REVISION)"
clone_pinned https://github.com/gameworld-project/GameWorld-Games.git \
  "$DEST/GameWorld-Games" "$(pin_from src/vista/benchmarks/gameworld/upstream.py GAMES_REVISION)"

cat <<EOF

GameWorld is ready. Point the runtime at it:

  export GAMEWORLD_ROOT=$DEST/GameWorld GAMES_ROOT=$DEST/GameWorld-Games
EOF
