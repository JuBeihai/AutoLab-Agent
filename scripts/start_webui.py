"""Portable UI entry point; load configuration without executing a shell file."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_environment():
    path = Path(os.environ.get('ETS_SECRETS_FILE', str(ROOT / '.env'))).expanduser()
    if not path.is_file():
        return
    if path.stat().st_mode & 0o777 != 0o600:
        raise ValueError('请先 chmod 600 .env（或 ETS_SECRETS_FILE 指向的文件）')
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:]
        name, sep, value = line.partition('=')
        if sep and name in {'ETS_LLM_API_KEY', 'ETS_GPT_API_KEY', 'ETS_DP_KEY', 'WEBUI_HOST', 'WEBUI_PORT'}:
            if not os.environ.get(name):
                os.environ[name] = value.strip().strip('"\'')


if __name__ == '__main__':
    os.chdir(ROOT)
    load_environment()
    os.environ.setdefault('ETS_PYTHON', sys.executable)
    extra = sys.argv[1:]
    sys.argv = ['webui.app', '--host', os.environ.get('WEBUI_HOST', '127.0.0.1'),
                '--port', os.environ.get('WEBUI_PORT', '5055'), *extra]
    print('ETS Agent：默认本机地址 http://127.0.0.1:' + os.environ.get('WEBUI_PORT', '5055'), flush=True)
    runpy.run_module('webui.app', run_name='__main__')
