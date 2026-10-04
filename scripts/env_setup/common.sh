# Shared helpers for the per-benchmark environment scripts. Sourced, not run.
#
# Every setup script is idempotent: running it again on an existing checkout
# fetches the pinned revision and checks it out rather than failing.

ROOT=$(cd "$(dirname "${BASH_SOURCE[1]}")/../.." && pwd)
PYTHON=${VISTA_PYTHON:-"$ROOT/.venv/bin/python"}
DEST=${VISTA_LOCAL_DIR:-"$ROOT/local"}

parse_common_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dest) DEST="$2"; shift 2 ;;
      --python) PYTHON="$2"; shift 2 ;;
      --no-install) SKIP_INSTALL=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
  done
}

say() { printf '\n== %s\n' "$*"; }

install_extras() {
  [[ ${SKIP_INSTALL:-0} == 1 ]] && { say "skipping pip install (--no-install)"; return; }
  [[ -x $PYTHON ]] || {
    echo "No interpreter at $PYTHON. Create the venv first (see README Setup) or pass --python." >&2
    exit 2
  }
  say "installing $1"
  "$PYTHON" -m pip install -e "$ROOT[$1]"
}

install_chromium() {
  [[ ${SKIP_INSTALL:-0} == 1 ]] && return
  say "installing Chromium and its system dependencies (root or sudo required)"
  "$PYTHON" -m playwright install --with-deps chromium
  "$PYTHON" - <<'PY'
from playwright.sync_api import sync_playwright

with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    browser.close()
PY
}

# pin_from <file> <constant name> -- the revision as the code records it, so the
# checkout and the runtime can never disagree about which upstream this is.
pin_from() {
  local value
  value=$(sed -n "s/^$2 = \"\([0-9a-f]\{40\}\)\"$/\1/p" "$ROOT/$1")
  [[ -n $value ]] || { echo "Cannot read $2 from $1" >&2; exit 2; }
  printf '%s' "$value"
}

# clone_pinned <url> <directory> <revision>
clone_pinned() {
  local url="$1" dir="$2" revision="$3"
  mkdir -p "$(dirname "$dir")"
  if [[ -d $dir/.git ]]; then
    say "$dir exists; fetching ${revision:0:12}"
    git -C "$dir" fetch --quiet origin "$revision" 2>/dev/null || git -C "$dir" fetch --quiet origin
  else
    say "cloning $url into $dir"
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" checkout --quiet --detach "$revision"
  echo "   $dir is at $(git -C "$dir" rev-parse --short HEAD)"
}
