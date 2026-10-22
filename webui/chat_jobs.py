"""Bounded in-memory chat jobs; retries of the same request never repeat work."""
import hashlib
import json
import threading
import time

from webui import chat_requests

_lock = threading.Lock()
_jobs = {}
_workers = set()


def _prune():
    completed = sorted((job['updated'], key) for key, job in _jobs.items() if job['status'] != 'running')
    for index, (updated, key) in enumerate(completed):
        if key not in _workers and (time.monotonic() - updated > 1800 or index < len(completed) - 64):
            del _jobs[key]


def get(request_id):
    chat_requests.validate_id(request_id)
    with _lock:
        _prune()
        job = _jobs.get(request_id)
        if job is None:
            return None
        return {key: value for key, value in job.items() if key != 'signature'}


def progress(request_id, stage):
    with _lock:
        job = _jobs.get(request_id)
        if job and job['status'] == 'running':
            if isinstance(stage, dict):
                job.update(stage=stage['stage'], research_trace=stage.get('trace', []), updated=time.monotonic())
            else:
                job.update(stage=stage, updated=time.monotonic())


def cancelled(request_id):
    with _lock:
        return _jobs.get(request_id, {}).get('status') == 'cancelled'


def session_busy(session_id):
    with _lock:
        return any(job['session_id'] == session_id and job['status'] == 'running' for job in _jobs.values())


def cancel(request_id):
    # Record cancellation even when the LLM subprocess has not started yet.
    status = chat_requests.cancel(request_id)
    with _lock:
        if request_id in _jobs:
            _jobs[request_id].update(status='cancelled', stage='已中断', updated=time.monotonic(),
                                     result={'ok': False, 'cancelled': True})
    return status


def start(request_id, body, work):
    chat_requests.validate_id(request_id)
    signature = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    with _lock:
        _prune()
        if request_id in _jobs:
            if _jobs[request_id]['signature'] != signature:
                raise ValueError('请求编号已用于其他内容')
        else:
            if len(_workers) >= 8:
                raise ValueError('同时咨询数量已达上限，请稍后重试')
            _jobs[request_id] = {'request_id': request_id, 'session_id': body.get('session_id'),
                'status': 'running', 'stage': '正在准备回复', 'updated': time.monotonic(), 'signature': signature}
            _workers.add(request_id)

            def run():
                try:
                    result = work()
                except Exception:
                    result = {'ok': False, 'error': '后台咨询失败，请重试'}
                with _lock:
                    _workers.discard(request_id)
                    job = _jobs.get(request_id)
                    if job and job['status'] == 'running':
                        job.update(status='complete', stage='已完成', result=result, updated=time.monotonic())

            threading.Thread(target=run, daemon=True, name='chat-' + request_id[:12]).start()
    return get(request_id)
