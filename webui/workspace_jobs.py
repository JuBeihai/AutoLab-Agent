"""Small bounded import queue; client retries reuse the same task ID."""
import re
import threading
import time
from skills import workspace, web_research

_lock = threading.Lock()
_jobs = {}


def validate_id(job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise ValueError('无效下载任务编号')


def _prune():
    done = sorted((v['updated'], k) for k, v in _jobs.items() if v['status'] != 'running')
    for i, (updated, key) in enumerate(done):
        if time.monotonic() - updated > 1800 or i < len(done) - 64:
            del _jobs[key]


def get(job_id):
    validate_id(job_id)
    with _lock:
        _prune()
        job = _jobs.get(job_id)
        return {k: v for k, v in job.items() if k != 'signature'} if job else None


def start(root, job_id, url, kind):
    validate_id(job_id)
    cfg = web_research.load_config(root)
    workspace.validate_import(url, kind, cfg)
    signature = (str(root), url, kind)
    with _lock:
        _prune()
        if job_id in _jobs:
            if _jobs[job_id]['signature'] != signature:
                raise ValueError('下载任务编号已用于其他网址')
        else:
            if sum(job['status'] == 'running' for job in _jobs.values()) >= 2:
                raise RuntimeError('已有两个下载任务，请稍后再试')
            _jobs[job_id] = {'request_id': job_id, 'status': 'running', 'url': url,
                             'updated': time.monotonic(), 'signature': signature}

            def run():
                try:
                    item = workspace.import_material(root, url, kind, config=cfg)
                    result = {'status': 'complete', 'item': item}
                except Exception as error:
                    result = {'status': 'failed', 'error': web_research.error_message(error)}
                with _lock:
                    _jobs[job_id].update(result, updated=time.monotonic())

            threading.Thread(target=run, daemon=True, name='workspace-' + job_id[:12]).start()
    return get(job_id)
