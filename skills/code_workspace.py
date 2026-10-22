"""Session-scoped generated files and resource-bounded Linux namespace execution."""
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.parse import quote

STORAGE = 'research_workspace/generated'
EXTENSIONS = {'.py', '.ipynb', '.json', '.csv', '.md', '.txt'}
MAX_SOURCE_BYTES = 1_000_000
MAX_WORK_BYTES = 64 * 1024**2


def folder(root, session_id, create=False):
    if not isinstance(session_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', session_id):
        raise ValueError('文件工作区需要有效的会话ID')
    base = Path(root).resolve()
    for part in PurePosixPath(STORAGE, session_id).parts:
        base = base / part
        if base.is_symlink(): raise ValueError('工作区不能使用符号链接')
    if create: base.mkdir(parents=True, exist_ok=True, mode=0o700)
    return base


def safe_file(base, relative):
    if not isinstance(relative, str) or len(relative) > 240 or '\\' in relative:
        raise ValueError('非法文件路径')
    if base.is_symlink(): raise ValueError('不允许符号链接')
    parts = relative.split('/')
    if any(not p or p.startswith('.') or any(ord(c) < 32 for c in p) for p in parts):
        raise ValueError('只允许工作区内相对路径，禁止隐藏文件和路径穿越')
    target = base
    for part in parts:
        target /= part
        if target.is_symlink(): raise ValueError('不允许符号链接')
    if not target.resolve().is_relative_to(base.resolve()): raise ValueError('文件必须位于指定工作区')
    return target


def open_read(base, relative):
    """Open under directory FDs; concurrent generated symlink swaps cannot escape."""
    safe_file(base, relative)
    directory = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = relative.split('/')
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError('只能读取普通文件')
        return os.fdopen(descriptor, 'rb')
    except OSError as error:
        if error.errno in (40, 20): raise ValueError('文件路径包含符号链接或无效目录') from None
        raise
    finally:
        os.close(directory)


def download_url(session_id, path):
    return '/api/code-workspace/' + quote(session_id, safe='') + '/download?path=' + quote(path, safe='')


def info(base, session_id, path):
    relative = path.relative_to(base).as_posix()
    return {'path': relative, 'bytes': path.stat().st_size,
            'download_url': download_url(session_id, relative)}


def _process_start(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def _recover_interrupted(base):
    runs = base / 'runs'
    if not runs.exists() or runs.is_symlink(): return
    for entry in runs.iterdir():
        if entry.is_symlink() or not entry.is_dir(): continue
        record = entry / 'result.json'
        if not record.is_file() or record.is_symlink() or record.stat().st_size > 1_000_000: continue
        try:
            value = json.loads(record.read_text())
            if value.get('status') != 'running': continue
            owner = value.get('owner_start')
            if owner and _process_start(value.get('owner_pid')) == owner: continue
            value.update(status='interrupted', note='所属研究进程已退出，运行被中断；已写入的日志和产物保留。')
            temporary = record.with_suffix('.tmp'); temporary.write_text(json.dumps(value, ensure_ascii=False)); temporary.replace(record)
        except (OSError, ValueError, TypeError): pass


def list_files(root, session_id):
    base = folder(root, session_id)
    _recover_interrupted(base)
    items = []
    def walk(directory):
        if not directory.exists(): return
        for p in sorted(directory.iterdir(), reverse=directory.name == 'runs'):
            if p.is_symlink() or p.name.startswith('.'): continue
            if len(items) >= 300: return
            if p.is_dir(): walk(p)
            elif p.is_file(): items.append(info(base, session_id, p))
    walk(base)
    return {'directory': str(base), 'files': items, 'truncated': len(items) >= 300,
            'runtime': 'Linux user/mount/network/PID namespaces; Python + IPython notebook cells',
            'limits': {'timeout_seconds': 120, 'memory_mb': 2048, 'network': False},
            'note': 'files/为可编辑源文件；runs/保存独立运行快照、真实日志和输出。工作区验证不改变正式实验和门禁。'}


def read_file(root, session_id, path, offset=0):
    if type(offset) is not int or offset < 0: raise ValueError('offset必须为非负整数')
    base = folder(root, session_id)
    _recover_interrupted(base)
    target = safe_file(base, path)
    if not target.is_file(): raise ValueError('文件不存在')
    import io
    with open_read(base, path) as raw:
        if os.fstat(raw.fileno()).st_size > 16 * 1024**2: raise ValueError('文件过大，请下载查看')
        with io.TextIOWrapper(raw, encoding='utf-8') as stream:
            stream.read(offset)
            text = stream.read(16001)
    return {**info(base, session_id, target), 'text': text[:16000],
            'next_offset': offset + 16000 if len(text) > 16000 else None}


def validate_notebook(text):
    nb = json.loads(text)
    if nb.get('nbformat') != 4 or not isinstance(nb.get('cells'), list) or len(nb['cells']) > 100:
        raise ValueError('需要nbformat=4且最多100个单元的Python Notebook')
    language = nb.get('metadata', {}).get('kernelspec', {}).get('language', 'python')
    if str(language).lower() != 'python': raise ValueError('仅支持Python Notebook')
    nb.setdefault('nbformat_minor', 5)
    nb.setdefault('metadata', {})
    nb['metadata']['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3 (AutoLab)', 'language': 'python'}
    for c in nb['cells']:
        if not isinstance(c, dict) or c.get('cell_type') not in ('code', 'markdown', 'raw'):
            raise ValueError('Notebook单元类型无效')
        source = c.get('source', '')
        if not isinstance(source, str) and not (isinstance(source, list) and all(isinstance(s, str) for s in source)):
            raise ValueError('Notebook source必须为字符串或字符串列表')
        c.setdefault('source', ''); c.setdefault('metadata', {}); c.setdefault('id', uuid.uuid4().hex[:8])
        if c['cell_type'] == 'code': c.update(outputs=[], execution_count=None)
    return json.dumps(nb, ensure_ascii=False, indent=1)


def write_file(root, session_id, path, content):
    base = folder(root, session_id, create=True)
    target = safe_file(base / 'files', path)
    if target.suffix not in EXTENSIONS: raise ValueError('支持.py/.ipynb/.json/.csv/.md/.txt')
    if not isinstance(content, str) or len(content.encode()) > MAX_SOURCE_BYTES:
        raise ValueError('单个源文件须不超过1MB')
    if target.suffix == '.ipynb': content = validate_notebook(content)
    if target.suffix == '.json': json.loads(content)
    snapshot = list_files(root, session_id)['files']
    sources = [item for item in snapshot if item['path'].startswith('files/')]
    if len(sources) >= 50 and not target.exists(): raise ValueError('会话源文件最多50个')
    if sum(item['bytes'] for item in sources) + len(content.encode()) > 8_000_000:
        raise ValueError('会话源文件总量超过8MB')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.' + uuid.uuid4().hex)
    temporary.write_text(content, encoding='utf-8'); temporary.replace(target)
    return {**info(base, session_id, target), 'sha256': hashlib.sha256(content.encode()).hexdigest(),
            'status': 'saved', 'note': '已保存，尚未执行；run_file的path使用不带files/的相对路径。'}


def _size(work):
    total, count = 0, 0
    for parent, dirs, files in os.walk(work, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
        for name in files:
            p = Path(parent) / name
            if not p.is_symlink(): total += p.stat().st_size
            count += 1
            if total > MAX_WORK_BYTES or count > 1000: return False
    return True


@contextmanager
def execution_slot(root):
    import fcntl
    parent = folder(root, '_runtime', create=True)
    lock = parent / 'execution.lock'
    if lock.is_symlink(): raise ValueError('非法执行锁路径')
    with lock.open('a') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('另一个工作区程序正在运行，请稍后重试') from None
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def run_file(root, session_id, path, timeout_seconds=60):
    with execution_slot(root):
        return _run_file(root, session_id, path, timeout_seconds)


def _run_file(root, session_id, path, timeout_seconds=60):
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 120:
        raise ValueError('运行时限必须为1～120秒')
    if sys.platform != 'linux':
        raise RuntimeError('当前主机不支持Linux命名空间执行；未降级为宿主机直接运行')
    base = folder(root, session_id, create=True)
    source = safe_file(base / 'files', path)
    if not source.is_file() or source.suffix not in {'.py', '.ipynb'}:
        raise ValueError('先使用write_file创建.py或.ipynb文件')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    run_dir = base / 'runs' / run_id
    run_dir.mkdir(parents=True)
    work = run_dir / 'work'; work.mkdir()
    for item in list_files(root, session_id)['files']:
        if not item['path'].startswith('files/'): continue
        original = safe_file(base, item['path'])
        destination = work / original.relative_to(base / 'files')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    shutil.copyfile(work / path, run_dir / ('source' + source.suffix))
    worker = Path(__file__).with_name('code_worker.py')
    shutil.copyfile(worker, work / '.runner.py')
    env = {'HOME': '/work/.home', 'TMPDIR': '/tmp', 'PATH': sys.prefix + '/bin:/usr/bin:/bin',
           'LANG': 'C.UTF-8', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': '/work/.home/matplotlib', 'IPYTHONDIR': '/work/.home/ipython'}
    result = {'run_id': run_id, 'path': path, 'status': 'running', 'exit_code': None,
              'source_sha256': hashlib.sha256((work / path).read_bytes()).hexdigest(),
              'timeout_seconds': timeout_seconds, 'isolation': 'linux_namespaces',
              'owner_pid': os.getpid(), 'owner_start': _process_start(os.getpid()),
              'note': '工作区程序验证结果；成功退出不等于模型效果提升，不参与冠军门禁。'}
    started = time.monotonic(); proc = None
    result_path = run_dir / 'result.json'
    result_path.write_text(json.dumps(result, ensure_ascii=False))
    try:
        with tempfile.TemporaryDirectory(prefix='ets-code-jail-') as jail, \
             tempfile.TemporaryDirectory(prefix='ets-code-spec-') as specs, \
             (run_dir / 'stdout.txt').open('wb') as stdout, (run_dir / 'stderr.txt').open('wb') as stderr:
            spec = {'work': str(work), 'jail': jail, 'entry': path,
                    'timeout': timeout_seconds, 'memory_mb': 2048, 'env': env}
            spec_path = Path(specs) / 'spec.json'; spec_path.write_text(json.dumps(spec))
            proc = subprocess.Popen([sys.executable, '-I', '-B', str(worker), 'launch', str(os.getpid()), str(spec_path)],
                                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}, close_fds=True)
            reason = None
            while proc.poll() is None:
                if time.monotonic() - started > timeout_seconds: reason = 'timeout'
                elif stdout.tell() + stderr.tell() > 256_000: reason = 'output_limit'
                elif not _size(work): reason = 'disk_limit'
                if reason: break
                time.sleep(.1)
            if reason:
                proc.kill(); proc.wait(timeout=5)
            result.update(status=reason or ('success' if proc.returncode == 0 else 'failed'), exit_code=proc.returncode)
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill(); proc.wait(timeout=5)
        result['seconds'] = round(time.monotonic() - started, 2)
        for name in ('stdout', 'stderr'):
            log = run_dir / (name + '.txt')
            result[name] = log.read_bytes()[-16000:].decode(errors='replace') if log.exists() else ''
        result['files'] = [item for item in list_files(root, session_id)['files'] if item['path'].startswith(f'runs/{run_id}/')]
        result['result_url'] = download_url(session_id, f'runs/{run_id}/result.json')
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
