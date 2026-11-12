#!/usr/bin/env bash
# G3: from a clean copy of the repository (no .venv, node_modules, lab_data, build output),
# install, build and run one preset end to end. Starting the server is attempted only when local
# port binding is permitted (`lab serve` needs it); set SMOKE_SERVE=1 to require it.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/emmlab-smoke.XXXXXX")"
echo "clean copy -> $WORK"
rsync -a --exclude .venv --exclude node_modules --exclude lab_data --exclude 'src/emmlab/server/static' \
  --exclude .pytest_cache --exclude .ruff_cache --exclude __pycache__ --exclude .hypothesis \
  --exclude 'web/dist' --exclude 'web/test-results' "$SRC/" "$WORK/"
cd "$WORK"
uv sync --quiet
if [ -d web ]; then
  pnpm -C web install --store-dir "${TMPDIR:-/tmp}/pnpm-store" --frozen-lockfile --silent
  pnpm -C web run build
  test -f src/emmlab/server/static/index.html
fi
export EMMLAB_DATA="$WORK/lab_data"
uv run lab validate pisa_like
uv run lab run minimal --out "$WORK/lab_data/runs" | tee "$WORK/run.log"
grep -q "report:" "$WORK/run.log"
test -f "$WORK/lab_data/runs/minimal_recommended_s0.html"
if [ "${SMOKE_SERVE:-0}" = "1" ]; then
  uv run lab serve --port 8799 & PID=$!
  trap 'kill $PID 2>/dev/null || true' EXIT
  for _ in $(seq 1 30); do curl -sf http://127.0.0.1:8799/api/presets >/dev/null && break; sleep 1; done
  curl -sf http://127.0.0.1:8799/ | grep -q '<div id="root">'
  echo "server OK"
fi
echo "SMOKE OK"
