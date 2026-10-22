#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_python.sh
exec "$AUTOLAB_PYTHON" -u agent/orchestrator.py "$@"
