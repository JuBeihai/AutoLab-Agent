#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_python.sh
exec "$ETS_PYTHON" -u agent/orchestrator.py "$@"
