#!/usr/bin/env bash
# Prepare the BabyVision environment: the extras and the benchmark's own
# repository, whose images and answer key ship as a zip inside data/.
#
#   ./scripts/env_setup/babyvision.sh [--dest DIR] [--python PATH] [--no-install]
set -euo pipefail
usage() { sed -n '2,/^[^#]/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_common_args "$@"

install_extras babyvision
clone_pinned https://github.com/UniPat-AI/BabyVision.git \
  "$DEST/BabyVision" "$(pin_from src/vista/benchmarks/babyvision/dataset.py BABYVISION_REVISION)"

say "unpacking the dataset in place"
python3 -m zipfile -e "$DEST/BabyVision/data/babyvision_data.zip" "$DEST/BabyVision/data"

cat <<EOF

BabyVision is ready. Point the runtime at it:

  export BABYVISION_ROOT=$DEST/BabyVision
EOF
