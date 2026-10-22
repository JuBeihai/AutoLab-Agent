"""Persistent, project-local eight-character session and experiment identifiers."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


@contextmanager
def activity_lock(root=ROOT, exclusive=False):
    directory = Path(root) / 'experiments'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.activity.lock').open('a') as handle:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(handle, operation | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('实验或 ID 迁移正在运行，请稍后重试') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_registry(root=ROOT):
    path = Path(root) / 'experiments/id_registry.json'
    if path.exists():
        return json.loads(path.read_text())
    return {'version': 1, 'next': {'S': 1, 'E': 1}, 'aliases': {}}


def write_registry(root, registry):
    path = Path(root) / 'experiments/id_registry.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def registry_lock(root=ROOT):
    directory = Path(root) / 'experiments'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.id_registry.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield read_registry(root)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def allocate(registry, kind):
    if kind not in {'S', 'E'}:
        raise ValueError('ID kind must be S or E')
    number = registry['next'][kind]
    if not 0 < number < 36 ** 7:
        raise ValueError('八位 ID 空间已耗尽')
    encoded = ''
    remaining = number
    while remaining:
        remaining, digit = divmod(remaining, 36)
        encoded = ALPHABET[digit] + encoded
    registry['next'][kind] = number + 1
    return kind + encoded.rjust(7, '0')


def new_id(kind, root=ROOT):
    with registry_lock(root) as registry:
        identifier = allocate(registry, kind)
        write_registry(root, registry)
        return identifier


def resolve_id(identifier, root=ROOT):
    return read_registry(root)['aliases'].get(identifier, identifier)
