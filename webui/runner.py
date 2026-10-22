"""子进程管理：保存有序日志，每个 SSE 订阅者独立回放和续传。"""

from __future__ import annotations

import os
import sys
import re
import signal
import json
import subprocess
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional

from skills.runtime_memory import memory_snapshot, require_available_memory


_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = os.environ.get("ETS_PYTHON", sys.executable)
_MAX_HISTORY_BYTES = 2 * 1024 * 1024
_MAX_HISTORY_LINES = 2000
_MAX_FINISHED_JOBS = 12


# 日志落盘回调（由 app.py 注入），签名: (line: str) -> None
_LOG_SINK = None


def set_log_sink(fn) -> None:
    global _LOG_SINK
    _LOG_SINK = fn


def _sink(line: str) -> None:
    if _LOG_SINK is not None:
        try:
            _LOG_SINK(line)
        except Exception:
            pass


class Job:
    """一次 orchestrator 子进程及可重放的事件历史。"""

    __slots__ = ("job_id", "proc", "condition", "history", "started_at", "ended_at", "cmd",
                 "returncode", "stop_requested", "session_id", "secrets", "history_start", "history_bytes",
                 "memory_before", "error_type", "error_hint")

    def __init__(self, job_id: str, cmd: List[str]):
        self.job_id = job_id
        self.cmd = cmd
        self.proc: Optional[subprocess.Popen] = None
        self.condition = threading.Condition()
        self.history = deque()
        self.history_start = 0
        self.history_bytes = 0
        self.started_at = time.time()
        self.ended_at: Optional[float] = None
        self.returncode = None
        self.stop_requested = False
        self.session_id = None
        self.secrets = ()
        self.memory_before = None
        self.error_type = None
        self.error_hint = None

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def emit(self, line: str, event=None):
        if 'demo_decrypt' in line.lower():
            line = '[runner] 已隐藏包含解密表达式的 Spark 日志'
        for secret in self.secrets:
            line = line.replace(secret, '[REDACTED]')
        cached = line
        if event is None and len(cached) > 16000:
            cached = cached[:16000] + '…（完整日志见服务日志）'
        with self.condition:
            self.history.append((event, cached))
            self.history_bytes += len(cached) * 4
            while len(self.history) > 1 and (len(self.history) > _MAX_HISTORY_LINES or self.history_bytes > _MAX_HISTORY_BYTES):
                _, removed = self.history.popleft()
                self.history_bytes -= len(removed) * 4
                self.history_start += 1
            self.condition.notify_all()
        _sink(line)


_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _prune_finished_jobs():
    finished = sorted((item for item in _jobs.values() if item.ended_at is not None),
                      key=lambda item: item.ended_at, reverse=True)
    for expired in finished[_MAX_FINISHED_JOBS:]:
        del _jobs[expired.job_id]


def _reader_thread(job: Job, stream, tag: str):
    try:
        for raw in iter(stream.readline, ""):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if tag == "stdout" and line.startswith("[event] "):
                try:
                    payload = json.loads(line[len("[event] "):])
                    job.session_id = payload["session_id"]
                    job.emit(json.dumps(payload, ensure_ascii=False), event="experiment")
                    continue
                except (ValueError, KeyError, TypeError):
                    pass
            job.emit(f"[{tag}] {line}")
    except Exception as e:
        job.emit(f"[runner] {tag} 读取异常: {e}")
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _persist_job_result(job: Job, returncode):
    if job.stop_requested:
        job.error_hint = '用户停止了实验'
    elif returncode == -signal.SIGKILL:
        current_memory = memory_snapshot()
        previous_kills = (job.memory_before or {}).get('oom_kill_count')
        current_kills = (current_memory or {}).get('oom_kill_count')
        oom_recorded = (previous_kills is not None and current_kills is not None
                        and current_kills > previous_kills)
        job.error_type = 'out_of_memory' if oom_recorded else 'out_of_memory_or_sigkill'
        job.error_hint = ('实验进程被强制结束（SIGKILL），运行期间容器记录了内存耗尽杀进程事件。' if oom_recorded else
                          '实验进程被强制结束（SIGKILL），可能是容器内存不足或外部终止。')
        job.error_hint += '请检查内存占用，关闭闲置 Notebook 内核及其 Spark，或缩短取数日期窗口后重试。'
    elif returncode:
        job.error_hint = '实验进程退出但未完成，请查看任务日志'
    if not job.session_id or not re.fullmatch(r"[A-Za-z0-9_-]+", job.session_id):
        return
    path = _ROOT / "artifacts/feature_mining" / job.session_id / "session.json"
    try:
        manifest = json.loads(path.read_text())
        if manifest.get("status") in {"complete", "failed", "profile_complete", "cancelled"}:
            job.error_type = manifest.get('error_type') or job.error_type
            job.error_hint = manifest.get('error_hint') or job.error_hint
            return
        manifest.update(status="cancelled" if job.stop_requested else "failed", job_returncode=returncode,
                        error_hint=job.error_hint or '实验进程退出但未完成，请查看任务日志')
        if job.error_type:
            manifest['error_type'] = job.error_type
        targets = [path]
        current = _ROOT / "experiments/feature_mining/current.json"
        if current.exists() and json.loads(current.read_text()).get("session_id") == job.session_id:
            targets.append(current)
        for target in targets:
            temporary = target.with_suffix(f".{job.job_id}.tmp")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(target)
    except (OSError, ValueError) as error:
        _sink(f"[runner] 无法同步会话退出状态: {type(error).__name__}")


def _waiter_thread(job: Job, readers=()):
    if not job.proc:
        return
    rc = job.proc.wait()
    for reader in readers:
        reader.join()
    _persist_job_result(job, rc)
    if job.error_hint:
        job.emit(f'[runner] {job.error_hint}')
    job.emit(f"[runner] 进程结束 rc={rc}")
    with job.condition:
        job.returncode = rc
        job.ended_at = time.time()
        job.condition.notify_all()
    _sink(f"===== JOB END   {job.job_id}  rc={rc}  elapsed={job.ended_at - job.started_at:.1f}s")
    with _jobs_lock:
        _prune_finished_jobs()


def start_orchestrator(rounds: int, max_proposals: int, mock: bool,
                        no_agent: bool, resume_recipes: bool = False,
                        extra_env: Optional[dict] = None, sample_rows: int = 0,
                        profile_only: bool = False, continuous: bool = False,
                        search_timeout_minutes: int = 120, search_patience: int = 9,
                        research_skills: bool = False, llm_profile: Optional[str] = None,
                        web_research: bool = False, data_source: str = 'online',
                        snapshot_id: Optional[str] = None, training_mode: str = 'pooled', model_family: str = 'xgb',
                        model_study: bool = False) -> str:
    """启动 orchestrator，返回 job_id。同时最多允许一个 alive job（前端限制）。"""
    if sample_rows and not mock:
        raise ValueError('Notebook 一致模式不允许抽样，请缩短日期窗口')
    from skills.model_registry import validate
    validate(model_family)
    if model_study and (model_family != 'mlp' or no_agent or profile_only or rounds * max_proposals < 4):
        raise ValueError('自主模型实验需要MLP、至少4个候选预算；不能同时只跑baseline/画像')
    if training_mode not in ('pooled', 'by_city'):
        raise ValueError('training_mode 必须为 pooled 或 by_city')
    if data_source not in ('online', 'offline'):
        raise ValueError('data_source 必须为 online 或 offline')
    if not mock and data_source == 'offline':
        from skills.local_snapshot import validate
        validate(_ROOT, snapshot_id, json.loads((_ROOT / 'configs/data.json').read_text()))
    elif snapshot_id:
        raise ValueError('snapshot_id 仅用于离线真实数据模式')
    cmd = ([_PYTHON, '-u', 'agent/orchestrator.py'] if mock or data_source == 'offline'
           else ['bash', 'scripts/run_feature_mining.sh'])
    cmd += ['--rounds', str(rounds), '--max-proposals-per-round', str(max_proposals),
            '--sample-rows', str(sample_rows), '--training-mode', training_mode, '--model-family', model_family]
    if not mock:
        cmd += ['--data-source', data_source]
        if data_source == 'offline':
            cmd += ['--snapshot-id', snapshot_id]
    if continuous or model_study:
        cmd += ['--continuous', '--search-timeout-minutes', str(search_timeout_minutes),
                '--search-patience', str(search_patience)]
    if research_skills:
        cmd.append('--research-skills')
    if model_study:
        cmd.append('--model-study')
    if web_research:
        cmd.append('--web-research')
    if llm_profile is not None:
        from skills.llm_profiles import resolve
        resolve(json.loads((_ROOT / 'configs/llm.json').read_text()), llm_profile)
        cmd += ['--llm-profile', llm_profile]
    if profile_only:
        cmd.append('--profile-only')
    if mock:
        cmd.append("--mock")
    if resume_recipes:
        cmd.append("--resume-recipes")
    if no_agent:
        cmd.append("--no-agent")

    env = os.environ.copy()
    env['ETS_PYTHON'] = _PYTHON
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items() if v is not None})

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job = Job(job_id=job_id, cmd=cmd)
    job.secrets = tuple(env[name] for name in ('ETS_DP_KEY', 'ETS_LLM_API_KEY', 'ETS_GPT_API_KEY') if env.get(name))

    with _jobs_lock:
        for existing in _jobs.values():
            if existing.ended_at is None:
                raise RuntimeError(f"已有正在运行的实验 job={existing.job_id}，请先停止")
        job.memory_before = memory_snapshot() if mock else require_available_memory(env)
        _prune_finished_jobs()
        proc = subprocess.Popen(
            cmd, cwd=str(_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env, start_new_session=True,
        )
        job.proc = proc
        _jobs[job_id] = job
    _sink(f"===== JOB START {job_id}  cmd: {' '.join(cmd)}")
    job.emit(f"[runner] 启动: {' '.join(cmd)}")

    readers = [threading.Thread(target=_reader_thread, args=(job, stream, tag), daemon=True)
               for stream, tag in ((proc.stdout, "stdout"), (proc.stderr, "stderr"))]
    for reader in readers:
        reader.start()
    threading.Thread(target=_waiter_thread, args=(job, readers), daemon=True).start()
    return job_id


def stop_job(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or not job.is_alive():
        return False
    try:
        job.stop_requested = True
        os.killpg(job.proc.pid, signal.SIGTERM)
        try:
            job.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(job.proc.pid, signal.SIGKILL)
        return True
    except Exception:
        return False


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


@contextmanager
def idle_guard():
    with _jobs_lock:
        if any(job.ended_at is None for job in _jobs.values()):
            raise RuntimeError('实验正在运行，请结束后再删除会话')
        yield


def list_jobs() -> List[dict]:
    with _jobs_lock:
        items = list(_jobs.values())
    return [
        {"job_id": j.job_id, "alive": j.ended_at is None,
         "started_at": j.started_at, "ended_at": j.ended_at,
         "returncode": j.returncode, "stop_requested": j.stop_requested, "session_id": j.session_id,
         "error_type": j.error_type, "error_hint": j.error_hint,
         "cmd": " ".join(j.cmd)}
        for j in items
    ]


def stream_lines(job_id: str, from_head: bool = True, after=0):
    """每个订阅者使用独立游标，Last-Event-ID 对应已接收消息数。"""
    job = get_job(job_id)
    if not job:
        yield f"event: error\ndata: unknown job_id {job_id}\n\n"
        return

    with job.condition:
        end = job.history_start + len(job.history)
        cursor = min(max(after, job.history_start), end) if from_head else end
    while True:
        with job.condition:
            if cursor == job.history_start + len(job.history) and job.ended_at is None:
                job.condition.wait(timeout=15)
            cursor = max(cursor, job.history_start)
            messages = list(islice(job.history, cursor - job.history_start, None))
            finished = job.ended_at is not None
        for event, line in messages:
            cursor += 1
            event_header = f"event: {event}\n" if event else ""
            data = "\n".join(f"data: {part}" for part in line.split("\n"))
            yield f"id: {cursor}\n{event_header}{data}\n\n"
        if finished:
            payload = {"job_id": job_id, "returncode": job.returncode,
                       "stop_requested": job.stop_requested, "session_id": job.session_id,
                       "error_type": job.error_type, "error_hint": job.error_hint}
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"
            return
        if not messages:
            yield ": keepalive\n\n"
