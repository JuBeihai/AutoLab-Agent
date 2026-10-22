#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/_python.sh
exec "$ETS_PYTHON" scripts/start_webui.py "$@"
