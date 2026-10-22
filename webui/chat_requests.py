"""Cancellable LLM requests isolated from experiment processes."""

from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace


class ChatCancelled(Exception):
    pass


_lock = threading.Lock()
_active = {}
_recent = {}


def validate_id(request_id):
    if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{16,80}', request_id):
        raise ValueError('非法 request_id')


def _prune():
    cutoff = time.monotonic() - 600
    for request_id in list(_recent):
        if _recent[request_id][0] < cutoff:
            del _recent[request_id]
    while len(_recent) > 512:
        del _recent[next(iter(_recent))]


def cancel(request_id):
    validate_id(request_id)
    with _lock:
        _prune()
        if request_id in _active:
            _active[request_id].set()
            return 'cancelling'
        if request_id in _recent:
            return _recent[request_id][1]
        _recent[request_id] = (time.monotonic(), 'cancelled')
        return 'cancelled'


def is_cancelled(request_id):
    with _lock:
        return (request_id in _active and _active[request_id].is_set()) or _recent.get(request_id, (None, None))[1] == 'cancelled'


def _worker():
    try:
        payload = json.load(sys.stdin)
        from skills.llm_client import LLMClient
        with redirect_stdout(sys.stderr):
            if payload.get('research') is not None:
                from skills.research_agent import run as research_run
                def progress(value):
                    path = Path(payload['progress_path'])
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(value, ensure_ascii=False))
                    temporary.replace(path)
                response = research_run(Path(__file__).resolve().parents[1], payload['config'],
                    payload['messages'], payload['research'].get('session_id', ''), progress=progress)
            else:
                response = LLMClient(payload['config']).chat(payload['messages'])
        result = {'text': response.text, 'usage': response.usage, 'finish_reason': response.finish_reason}
        if payload.get('research') is not None:
            result['web_research'] = response.web_research
    except Exception as error:
        result = {'error': type(error).__name__}
    json.dump(result, sys.stdout, ensure_ascii=False)


def run(request_id, config, messages, command=None, research=None, on_progress=None):
    validate_id(request_id)
    cancelled = threading.Event()
    with _lock:
        _prune()
        if request_id in _recent and _recent[request_id][1] == 'cancelled':
            raise ChatCancelled()
        if request_id in _active or request_id in _recent:
            raise ValueError('request_id 已使用，请使用新的请求编号')
        if len(_active) >= 8:
            raise RuntimeError('同时咨询数量已达上限，请稍后重试')
        _active[request_id] = cancelled
    process = None
    try:
        if cancelled.is_set():
            raise ChatCancelled()
        with tempfile.TemporaryDirectory(prefix='ets-research-progress-') as progress_dir, \
             tempfile.TemporaryFile(mode='w+b') as request_file, tempfile.TemporaryFile(mode='w+b') as response_file:
            progress_path = Path(progress_dir) / 'progress.json'
            payload = {'config': config, 'messages': messages}
            if research is not None:
                payload.update(research=research, progress_path=str(progress_path))
            request_file.write(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
            request_file.seek(0)
            process = subprocess.Popen(command or [sys.executable, '-m', 'webui.chat_requests'],
                cwd=str(Path(__file__).resolve().parents[1]), stdin=request_file, stdout=response_file,
                stderr=subprocess.DEVNULL)
            timeout = max(30, (int(config.get('timeout_sec', 300)) + 30) *
                          (int(config.get('max_retries', 2)) + 1) + 60)
            if research is not None: timeout = 1200
            deadline = time.monotonic() + timeout
            last_progress = None
            while True:
                if cancelled.is_set():
                    raise ChatCancelled()
                if process.poll() is not None:
                    if process.returncode != 0:
                        raise RuntimeError('模型请求进程意外退出')
                    response_file.seek(0)
                    result = json.load(response_file)
                    if cancelled.is_set():
                        raise ChatCancelled()
                    if 'error' in result:
                        raise RuntimeError(f'模型请求失败（{result["error"]}），请重试')
                    return SimpleNamespace(**result)
                if on_progress and progress_path.exists():
                    try:
                        value = progress_path.read_text()
                        if value != last_progress:
                            on_progress(json.loads(value)); last_progress = value
                    except (OSError, ValueError):
                        pass
                if time.monotonic() > deadline:
                    raise TimeoutError('模型请求超时，请重试')
                cancelled.wait(.1)
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        with _lock:
            _active.pop(request_id, None)
            _recent[request_id] = (time.monotonic(), 'cancelled' if cancelled.is_set() else 'finished')
            _prune()


if __name__ == '__main__':
    _worker()
