"""Reviewed data snapshots and isolated feature plugins. Never executes code on the host."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from skills.feature_program import fingerprint, NAME

REQUEST_ID = re.compile(r'R[0-9A-F]{7}')
FORBIDDEN_INPUTS = {'true_bin', 'true_ets', 'is_answer', 'match_time', 'cancel_time', 'predict_bin_44'}
RUNNER = '''import json, math, sys, unittest
sys.path.insert(0, '/work')
from feature import transform
if sys.argv[1] == 'test':
    suite = unittest.defaultTestLoader.discover('/work', pattern='test_feature.py')
    if suite.countTestCases() == 0:
        raise RuntimeError('No tests')
    result = unittest.TextTestRunner(stream=sys.stderr).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError('Tests failed or skipped')
else:
    with open('/work/input.jsonl') as stream:
        for line in stream:
            value = transform(json.loads(line))
            print(json.dumps(value, allow_nan=False), flush=True)
'''


def storage(root):
    return Path(root) / 'research_workspace'


def _document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def _names(value, maximum=16):
    if (not isinstance(value, list) or not 1 <= len(value) <= maximum
            or any(not isinstance(name, str) or not NAME.fullmatch(name) for name in value)
            or len(set(value)) != len(value)):
        raise ValueError('字段名称列表不合法')
    return value


def request_path(root, request_id):
    if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
        raise ValueError('申请ID不合法')
    path = storage(root) / 'requests' / request_id
    if path.is_symlink():
        raise ValueError('申请目录不能是符号链接')
    return path


def stage(root, session_id, proposal, catalog, mock=None):
    if not isinstance(proposal, dict) or proposal.get('kind') not in ('data_source', 'code_feature'):
        raise ValueError('申请类型只能是 data_source 或 code_feature')
    common = {'kind', 'reason', 'outputs'}
    expected = common | ({'source', 'keys', 'time_column', 'availability'} if proposal['kind'] == 'data_source'
                         else {'inputs', 'files'})
    if set(proposal) != expected:
        raise ValueError('申请字段不符合协议')
    if not isinstance(proposal['reason'], str) or not 1 <= len(proposal['reason']) <= 2000:
        raise ValueError('申请需包含简短理由')
    _names(proposal['outputs'], 4)
    if any(name in catalog or name in FORBIDDEN_INPUTS for name in proposal['outputs']):
        raise ValueError('扩展不能覆盖已存在字段或标签')
    if proposal['kind'] == 'code_feature':
        _names(proposal['inputs'])
        if any(name in FORBIDDEN_INPUTS or catalog.get(name, {}).get('role') != 'numeric'
               for name in proposal['inputs']):
            raise ValueError('代码扩展只能读取已探查的数值输入，禁止标签和结果字段')
        files = proposal['files']
        if not isinstance(files, dict) or set(files) != {'feature.py', 'test_feature.py'}:
            raise ValueError('代码工作区只允许 feature.py 和 test_feature.py')
        for filename, text in files.items():
            if not isinstance(text, str) or not 1 <= len(text.encode()) <= 24000:
                raise ValueError('代码文件为空或过大')
            ast.parse(text, filename=filename)
    else:
        _names(proposal['keys'], 3)
        if any(name not in catalog and name != 'order_id' for name in proposal['keys']):
            raise ValueError('数据源关联键未探查')
        for key in ('source', 'time_column', 'availability'):
            if not isinstance(proposal[key], str) or not 1 <= len(proposal[key]) <= 2000:
                raise ValueError('数据接入需声明来源、可用时间列与时点说明')
        if not NAME.fullmatch(proposal['time_column']):
            raise ValueError('时间列名称不合法')
        if len(set(proposal['keys'] + proposal['outputs'] + [proposal['time_column']])) != len(proposal['keys']) + len(proposal['outputs']) + 1:
            raise ValueError('关联键、输出和时间列不能重叠')
    digest = fingerprint(proposal)
    request_id = 'R' + digest[:7].upper()
    folder = request_path(root, request_id)
    path = folder / 'request.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['digest'] != digest:
            raise ValueError('申请短ID冲突')
        if proposal['kind'] == 'data_source' and previous.get('mock') is not mock:
            raise ValueError('数据模式不同，需独立的数据源申请')
        return {'id': request_id, 'status': previous['status'], 'kind': proposal['kind']}
    folder.mkdir(parents=True, mode=0o700)
    for filename, text in proposal.get('files', {}).items():
        (folder / filename).write_text(text)
    _document(path, {'id': request_id, 'session_id': session_id, 'digest': digest, 'mock': mock,
                     'status': 'pending_review', 'proposal': proposal})
    return {'id': request_id, 'status': 'pending_review', 'kind': proposal['kind']}


def requests(root, session_id=None):
    result = []
    for path in sorted((storage(root) / 'requests').glob('R*/request.json')):
        if path.is_symlink() or path.parent.is_symlink():
            continue
        try:
            record = json.loads(path.read_text())
            if session_id and record.get('session_id') != session_id:
                continue
            proposal = record['proposal']
            result.append({key: record.get(key) for key in ('id', 'session_id', 'status', 'validation', 'review_note')}
                          | {key: proposal.get(key) for key in ('kind', 'reason', 'outputs', 'source')})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def sandbox_info():
    image = os.environ.get('ETS_RESEARCH_SANDBOX_IMAGE', '')
    executable = shutil.which('docker')
    if not executable or not re.fullmatch(r'[a-zA-Z0-9./:_-]+@sha256:[a-f0-9]{64}', image):
        raise RuntimeError('代码执行关闭：需要 rootless Docker 和本地已准备、以 sha256 固定的 ETS_RESEARCH_SANDBOX_IMAGE')
    result = subprocess.run([executable, 'info', '--format', '{{json .}}'],
                            capture_output=True, text=True, timeout=10, check=True)
    info = json.loads(result.stdout)
    if not any('rootless' in option for option in info.get('SecurityOptions', [])):
        raise RuntimeError('代码执行关闭：只允许 rootless Docker')
    if info.get('CgroupVersion') != '2' or any(info.get(key) is not True for key in ('MemoryLimit', 'CpuCfsQuota', 'PidsLimit')):
        raise RuntimeError('代码执行关闭：需要 cgroup v2 和可用的内存、CPU、PID限制')
    return executable, image


def _sandbox(folder, mode, timeout=120):
    executable, image = sandbox_info()
    name = 'ets-research-' + os.urandom(8).hex()
    command = [executable, 'run', '--rm', '--pull', 'never', '--name', name,
               '--network', 'none', '--read-only', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges', '--user', '65534:65534',
               '--memory', '512m', '--memory-swap', '512m', '--cpus', '1', '--pids-limit', '64',
               '--log-driver', 'none', '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
               '--mount', f'type=bind,src={folder},dst=/work,readonly',
               '--workdir', '/work', '--entrypoint', 'python', image, '-B', '/work/runner.py', mode]
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        watchdog = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve().parents[1] / 'agent/sandbox_watchdog.py'),
                                     executable, name, str(timeout)], stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        process = None
        started = time.monotonic()
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors)
            while process.poll() is None:
                if time.monotonic() - started > timeout or os.fstat(output.fileno()).st_size + os.fstat(errors.fileno()).st_size > 32_000_000:
                    raise RuntimeError('代码沙箱超时或输出超出预算')
                time.sleep(0.1)
            if os.fstat(output.fileno()).st_size + os.fstat(errors.fileno()).st_size > 32_000_000:
                raise RuntimeError('代码沙箱输出超出预算')
            if process.returncode:
                errors.seek(0)
                raise ValueError('代码沙箱校验失败：' + errors.read(2000).decode(errors='replace'))
            output.seek(0)
            return output.read().decode()
        finally:
            try:
                subprocess.run([executable, 'rm', '-f', name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10, check=False)
            finally:
                watchdog.stdin.close()
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)


def _code(root, record, frame=None):
    proposal = record['proposal']
    with tempfile.TemporaryDirectory(prefix='ets-plugin-') as temporary:
        folder = Path(temporary) / 'work'
        folder.mkdir(mode=0o755)
        for filename, text in proposal['files'].items():
            (folder / filename).write_text(text)
            (folder / filename).chmod(0o644)
        (folder / 'runner.py').write_text(RUNNER)
        (folder / 'runner.py').chmod(0o644)
        if frame is None:
            _sandbox(folder, 'test')
            return
        inputs = frame[proposal['inputs']].apply(pd.to_numeric, errors='coerce')
        inputs = inputs.replace([float('inf'), float('-inf')], float('nan'))
        inputs.to_json(folder / 'input.jsonl', orient='records', lines=True)
        (folder / 'input.jsonl').chmod(0o644)
        output = _sandbox(folder, 'transform')
        rows = [json.loads(line) for line in output.splitlines()]
        if len(rows) != len(frame):
            raise ValueError('代码扩展改变了样本行数')
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(proposal['outputs']):
                raise ValueError('代码输出列不符合已审批契约')
            if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)) for value in row.values()):
                raise ValueError('代码输出必须为有限数值或 null')
        return pd.DataFrame(rows, index=frame.index, columns=proposal['outputs'])


def _load_source(path, proposal):
    source = pd.read_csv(path)
    expected = proposal['keys'] + [proposal['time_column']] + proposal['outputs']
    if set(source.columns) != set(expected):
        raise ValueError('快照只允许声明的键、可用时间和输出列')
    if source[proposal['keys']].isna().any().any():
        raise ValueError('来源关联键不能为空')
    source[proposal['time_column']] = pd.to_datetime(source[proposal['time_column']], errors='raise', utc=True)
    if source[proposal['time_column']].isna().any() or source.duplicated(proposal['keys'] + [proposal['time_column']]).any():
        raise ValueError('来源的关联键与可用时间必须非空且唯一')
    for name in proposal['outputs']:
        source[name] = pd.to_numeric(source[name], errors='raise')
        if source[name].isin([float('inf'), float('-inf')]).any():
            raise ValueError('来源不能包含无穷值')
    return source


def review(root, request_id, decision, note, snapshot=None):
    if decision not in ('approve', 'reject') or not isinstance(note, str) or not note.strip():
        raise ValueError('审批需要明确 approve/reject 和人工审核说明')
    folder = request_path(root, request_id)
    record = json.loads((folder / 'request.json').read_text())
    if fingerprint(record['proposal']) != record['digest']:
        raise ValueError('申请内容已被修改')
    if record['status'] != 'pending_review':
        raise ValueError('审批不可覆盖；修改需提交新申请')
    if decision == 'approve':
        if record['proposal']['kind'] == 'code_feature':
            for filename, text in record['proposal']['files'].items():
                if (folder / filename).is_symlink() or (folder / filename).read_text() != text:
                    raise ValueError('代码工作区已被修改，需提交新申请，不能审批旧摘要')
            _code(root, record)
            record['validation'] = 'sandbox_tests_passed; human_review_required_for_semantics'
        else:
            if snapshot is None or not Path(snapshot).is_file() or Path(snapshot).stat().st_size > 256_000_000:
                raise ValueError('审批数据源需要不超过256MB的本地 CSV 快照')
            _load_source(snapshot, record['proposal'])
            destination = folder / 'approved_snapshot.csv'
            shutil.copyfile(snapshot, destination)
            _load_source(destination, record['proposal'])
            record['snapshot_hash'] = hashlib.sha256(destination.read_bytes()).hexdigest()
            record['validation'] = 'schema_unique_keys_numeric_time_checked; availability_human_reviewed'
    record.update(status='approved' if decision == 'approve' else 'rejected', review_note=note)
    _document(folder / 'request.json', record)
    return record


def approved(root, mock=None):
    result = []
    for item in requests(root):
        if item['status'] != 'approved':
            continue
        record = json.loads((request_path(root, item['id']) / 'request.json').read_text())
        if mock is not None and record['proposal']['kind'] == 'data_source' and record.get('mock') is not mock:
            continue
        if fingerprint(record['proposal']) != record['digest']:
            continue
        result.append({'id': record['id'], 'kind': record['proposal']['kind'],
                       'outputs': record['proposal']['outputs'], 'digest': record['digest'],
                       'inputs': record['proposal'].get('inputs', []), 'mock': record.get('mock'),
                       'snapshot_hash': record.get('snapshot_hash')})
    return result


def materialize(root, frame, extensions):
    result = frame.copy()
    for extension in extensions:
        folder = request_path(root, extension['id'])
        record = json.loads((folder / 'request.json').read_text())
        proposal = record['proposal']
        if (record['status'] != 'approved' or fingerprint(proposal) != extension['digest']
                or proposal['outputs'] != extension['outputs'] or proposal['kind'] != extension['kind']):
            raise ValueError('扩展未审批或已改变，拒绝使用')
        if set(proposal['outputs']) & set(result.columns):
            raise ValueError('扩展不能覆盖现有字段')
        if proposal['kind'] == 'code_feature':
            generated = _code(root, record, result)
        else:
            snapshot = folder / 'approved_snapshot.csv'
            if snapshot.is_symlink() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != extension['snapshot_hash']:
                raise ValueError('来源快照与已审批版本不一致')
            source = _load_source(snapshot, proposal)
            left = result[proposal['keys']].copy()
            left['_prediction_time'] = pd.to_datetime(result['prediction_time'], errors='raise', utc=True)
            left['_row'] = range(len(left))
            for key in proposal['keys']:
                left[key] = left[key].astype('string').fillna('')
                source[key] = source[key].astype('string')
            joined = pd.merge_asof(left.sort_values('_prediction_time'),
                                   source.sort_values(proposal['time_column']),
                                   left_on='_prediction_time', right_on=proposal['time_column'],
                                   by=proposal['keys'], direction='backward', allow_exact_matches=False)
            generated = joined.sort_values('_row')[proposal['outputs']]
            generated.index = result.index
        for name in proposal['outputs']:
            result[name] = generated[name].to_numpy()
    return result
