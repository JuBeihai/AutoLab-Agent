#!/usr/bin/env bash
# Portable interpreter selection; no captured environment from the original machine.
ETS_AGENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${ETS_PYTHON:-}" ]]; then
  if [[ -x "$ETS_AGENT_ROOT/.venv/bin/python" ]]; then
    ETS_PYTHON="$ETS_AGENT_ROOT/.venv/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    ETS_PYTHON="$VIRTUAL_ENV/bin/python"
  else
    ETS_PYTHON="$(command -v python3 || true)"
  fi
fi
if [[ ! -x "$ETS_PYTHON" ]]; then
  echo '请创建项目 .venv，或通过 ETS_PYTHON 指定可执行的 Python 绝对路径。' >&2
  return 1
fi
export ETS_AGENT_ROOT ETS_PYTHON
if [[ "$(uname -s)" == Darwin ]]; then
  _ets_omp_dir="$("$ETS_PYTHON" -c 'import importlib.util; from pathlib import Path; s=importlib.util.find_spec("sklearn"); print(Path(s.origin).parent/".dylibs" if s and s.origin else "")')"
  if [[ -d "$_ets_omp_dir" ]]; then
    export DYLD_LIBRARY_PATH="$_ets_omp_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  fi
fi
