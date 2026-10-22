"""Durable session-scoped chat turns. No credentials or system prompts are stored."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from webui.chat_requests import validate_id

_lock = threading.RLock()
BOOT_ID = uuid.uuid4().hex


@contextmanager
def locked():
    with _lock:
        yield


def _folder(artifacts, session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', session_id):
        raise ValueError('非法会话编号')
    base = Path(artifacts).resolve()
    session = base / 'feature_mining' / session_id
    folder = session / 'chat'
    if session.is_symlink() or folder.is_symlink() or not folder.resolve().is_relative_to(base):
        raise ValueError('非法聊天记录路径')
    if not (session / 'session.json').is_file():
        raise FileNotFoundError('会话不存在')
    return folder


def _read(path):
    if path.is_symlink():
        raise ValueError('非法聊天记录路径')
    record = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(record, dict) or record.get('request_id') != path.stem:
        raise ValueError('聊天记录损坏')
    return record


def _save(path, record):
    if path.is_symlink():
        raise ValueError('非法聊天记录路径')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    # Restrict local chat transcripts independently of the launcher's umask.
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _public(record):
    item = {key: value for key, value in record.items() if key not in ('fingerprint', 'owner')}
    if item['status'] == 'running' and record.get('owner') != BOOT_ID:
        item.update(status='interrupted', result={'ok': False,
                    'error': '服务在回复完成前重启；问题已保存，但没有完整回复。请手动重新发送。'})
    return item


def fingerprint(body):
    fields = {key: body.get(key) for key in ('question', 'session_id', 'history', 'llm_profile', 'include_history')}
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def find(artifacts, request_id):
    validate_id(request_id)
    with _lock:
        for path in (Path(artifacts) / 'feature_mining').glob('*/chat/' + request_id + '.json'):
            folder = _folder(artifacts, path.parent.parent.name)
            record = _read(folder / path.name)
            if record.get('session_id') != path.parent.parent.name:
                raise ValueError('聊天记录会话不一致')
            return _public(record)
    return None


def begin(artifacts, session_id, request_id, body):
    validate_id(request_id)
    with _lock:
        folder = _folder(artifacts, session_id)
        existing = find(artifacts, request_id)
        if existing:
            old = _read(_folder(artifacts, existing['session_id']) / (request_id + '.json'))
            if old.get('fingerprint') != fingerprint(body):
                raise ValueError('请求编号已用于其他内容或会话')
            return existing, False
        record = {'schema_version': 1, 'request_id': request_id, 'session_id': session_id,
                  'question': body['question'].strip(), 'status': 'running', 'owner': BOOT_ID,
                  'created_at': datetime.now(timezone.utc).isoformat(), 'order': str(time.time_ns()),
                  'fingerprint': fingerprint(body), 'result': None}
        _save(folder / (request_id + '.json'), record)
        return _public(record), True


def finish(artifacts, session_id, request_id, result):
    with _lock:
        path = _folder(artifacts, session_id) / (request_id + '.json')
        record = _read(path)
        # A late answer must not overwrite an explicitly cancelled turn.
        if record['status'] != 'running':
            return _public(record)
        record.update(result=result, status='cancelled' if result.get('cancelled') else 'complete' if result.get('ok') else 'failed',
                      finished_at=datetime.now(timezone.utc).isoformat())
        _save(path, record)
        return _public(record)


def cancel(artifacts, request_id):
    with _lock:
        item = find(artifacts, request_id)
        if item and item['status'] == 'running':
            finish(artifacts, item['session_id'], request_id,
                   {'ok': False, 'cancelled': True, 'error': '用户中断了回复；提问已保存。'})


def session_busy(artifacts, session_id):
    folder = _folder(artifacts, session_id)
    with _lock:
        return any(_public(_read(path))['status'] == 'running' for path in folder.glob('*.json'))


def transcript(artifacts, session_id, before=None, limit=30):
    if type(limit) is not int or not 1 <= limit <= 100 or (before is not None and not re.fullmatch(r'\d{1,30}', before)):
        raise ValueError('聊天分页参数无效')
    with _lock:
        folder = _folder(artifacts, session_id)
        records, skipped = [], 0
        for path in folder.glob('*.json'):
            try:
                record = _read(path)
                validate_id(record['request_id'])
                if record.get('session_id') != session_id or not re.fullmatch(r'\d{1,30}', record.get('order', '')):
                    raise ValueError('聊天记录元数据损坏')
                records.append(_public(record))
            except (OSError, ValueError, KeyError, TypeError):
                skipped += 1
    records.sort(key=lambda item: (int(item['order']), item['request_id']))
    available = [item for item in records if before is None or int(item['order']) < int(before)]
    page = available[-limit:]
    # Keep recent model context bounded separately from the complete transcript.
    history = []
    for item in records:
        if item['status'] != 'complete' or item.get('recovered'):
            continue
        result = item.get('result') or {}
        if item.get('question') and result.get('answer'):
            history.extend([{'role': 'user', 'content': item['question']}, {'role': 'assistant', 'content': result['answer']}])
    history = history[-12:]
    while history and sum(len(message['content']) for message in history) > 110000:
        history = history[2:]
    return {'session_id': session_id, 'turns': page, 'total': len(records), 'history': history,
            'next_before': page[0]['order'] if len(available) > len(page) else None, 'skipped': skipped}


def import_legacy(artifacts, session_id, request_id, text, role, provenance, order):
    """Only import explicitly session-tagged log text; never invent missing turns."""
    validate_id(request_id)
    with _lock:
        path = _folder(artifacts, session_id) / (request_id + '.json')
        if path.exists():
            return False
        record = {'schema_version': 1, 'request_id': request_id, 'session_id': session_id,
                  'question': text if role == 'user' else None, 'created_at': provenance['timestamp'],
                  'order': str(order), 'status': 'recovered', 'recovered': True, 'provenance': provenance,
                  'result': {'ok': role == 'assistant', 'answer': text if role == 'assistant' else None},
                  'note': '从旧日志恢复的片段；原日志可能截断，不代表完整对话，未推测缺失的问题或回复。'}
        _save(path, record)
        return True
