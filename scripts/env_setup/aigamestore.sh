#!/usr/bin/env bash
# Prepare the AI GameStore environment: the extras, a Chromium, and the
# benchmark's own repository, of which only games/ is used here.
#
#   ./scripts/env_setup/aigamestore.sh [--dest DIR] [--python PATH] [--no-install]
set -euo pipefail
usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_common_args "$@"

# Pinned upstream checkout supplying the game assets.
AIGAMESTORE_REVISION=5a7c7a2c536c3599cd2e5b7b1987a382392ca0f2

install_extras aigamestore
install_chromium
clone_pinned https://github.com/lance-ying/aigamestore_harness.git \
  "$DEST/aigamestore" "$AIGAMESTORE_REVISION"

cat <<EOF

AI GameStore is ready. Point the runtime at it:

  export AIGAMESTORE_GAMES_ROOT=$DEST/aigamestore/games
EOF
