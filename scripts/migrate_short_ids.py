"""Preview or migrate active experiment IDs, with metadata backups and rollback."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.identifiers import activity_lock, allocate, registry_lock, write_registry


OLD_ID = re.compile(r'(?:mining_(?:mock|real)_\d{8}_\d{6}_[a-f0-9]{6}(?:_r\d+_\d{8}_\d{6}_[a-f0-9]{6})?|r\d+_\d{8}_\d{6}_[a-f0-9]{6})')
SHORT_ID = re.compile(r'^[SE][0-9A-Z]{7}$')
ID_KEYS = {'session_id', 'experiment_id', 'champion_id', 'baseline_id', 'selected_id',
           'parent_experiment_id', 'resume_from_session_id'}


def check_idle(root):
    for process in Path('/proc').glob('[0-9]*'):
        try:
            arguments = (process / 'cmdline').read_bytes().split(b'\0')
            if not arguments or b'python' not in Path(arguments[0].decode()).name.encode():
                continue
            entrypoints = {b'agent/orchestrator.py', b'agent/feature_mining.py',
                           b'agent.orchestrator', b'agent.feature_mining', b'run_baseline.py'}
            if not any(argument in entrypoints or argument.endswith((b'/agent/orchestrator.py', b'/agent/feature_mining.py', b'/run_baseline.py'))
                       for argument in arguments[1:]):
                continue
            directory = (process / 'cwd').resolve()
            if directory == root or any(str(root).encode() in argument for argument in arguments):
                raise RuntimeError(f'实验进程 {process.name} 正在运行，拒绝迁移')
        except (OSError, UnicodeError):
            continue


def walk_identifiers(value, sessions, experiments):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ID_KEYS and isinstance(item, str) and item:
                (sessions if key in {'session_id', 'resume_from_session_id'} else experiments).add(item)
            walk_identifiers(item, sessions, experiments)
    elif isinstance(value, list):
        for item in value:
            walk_identifiers(item, sessions, experiments)


def read_files(root):
    paths = []
    contents = {}
    for name in ('artifacts', 'experiments', 'models'):
        for path in (root / name).rglob('*'):
            if path.is_symlink():
                raise ValueError(f'迁移目录中存在软链接，请先检查：{path}')
            paths.append(path)
            if not path.is_file() or path.name == 'id_registry.json':
                continue
            if path.suffix not in {'.json', '.jsonl', '.md', '.csv', '.txt', '.md5'}:
                continue
            if path.suffix == '.json' and Path(str(path) + '.md5').exists():
                continue
            contents[path] = path.read_text(encoding='utf-8')
    return paths, contents


def atomic_write(path, text):
    temporary = path.with_name(path.name + '.id-migration.tmp')
    temporary.write_text(text, encoding='utf-8')
    if path.exists():
        shutil.copymode(path, temporary)
    temporary.replace(path)


def migrate(root, apply=False):
    root = Path(root).resolve()
    with activity_lock(root, exclusive=True), registry_lock(root) as registry:
        check_idle(root)
        paths, contents = read_files(root)
        sessions, experiments = set(), set()
        documents = {}
        for path, text in contents.items():
            if path.suffix == '.json':
                documents[path] = json.loads(text)
            elif path.suffix == '.jsonl':
                documents[path] = [json.loads(line) for line in text.splitlines() if line.strip()]
            if path in documents:
                walk_identifiers(documents[path], sessions, experiments)
        for path in paths:
            for identifier in OLD_ID.findall(path.name):
                if re.fullmatch(r'mining_(?:mock|real)_\d{8}_\d{6}_[a-f0-9]{6}', identifier):
                    sessions.add(identifier)
                else:
                    experiments.add(identifier)
        for identifier in sessions | experiments:
            if SHORT_ID.fullmatch(identifier):
                kind = identifier[0]
                registry['next'][kind] = max(registry['next'][kind], int(identifier[1:], 36) + 1)
        for identifiers, kind in ((sessions, 'S'), (experiments, 'E')):
            for identifier in sorted(identifiers, key=lambda value: (re.findall(r'\d{8}_\d{6}', value)[-1:] or [value], value)):
                if len(identifier) > 8 and identifier not in registry['aliases']:
                    registry['aliases'][identifier] = allocate(registry, kind)
        aliases = registry['aliases']
        if not aliases:
            return {'sessions': 0, 'experiments': 0, 'files': 0, 'renames': 0, 'applied': False}
        pattern = re.compile('|'.join(re.escape(key) for key in sorted(aliases, key=len, reverse=True)))

        def replace(text):
            return pattern.sub(lambda match: aliases[match.group()], text)

        def add_created_at(value):
            if isinstance(value, dict):
                identifier = value.get('session_id', '')
                match = re.fullmatch(r'mining_(?:mock|real)_(\d{8})_(\d{6})_[a-f0-9]{6}', identifier)
                if match and 'status' in value and 'created_at' not in value:
                    value['created_at'] = datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S').isoformat()
                    return True
            return False

        changes = {}
        for path, text in contents.items():
            document = documents.get(path)
            if path.suffix == '.json' and add_created_at(document):
                text = json.dumps(document, ensure_ascii=False, indent=2) + '\n'
            updated = replace(text)
            if updated != contents[path]:
                changes[path] = updated
        renames = [(path, path.with_name(replace(path.name))) for path in paths if replace(path.name) != path.name]
        renames.sort(key=lambda pair: len(pair[0].parts), reverse=True)
        destinations = set()
        for source, destination in renames:
            if destination.exists() or destination in destinations:
                raise ValueError(f'目标路径已存在，拒绝覆盖：{destination}')
            destinations.add(destination)
        summary = {'sessions': sum(identifier in aliases for identifier in sessions),
                   'experiments': sum(identifier in aliases for identifier in experiments),
                   'files': len(changes), 'renames': len(renames), 'applied': False,
                   'aliases': aliases}
        if not apply or not changes and not renames:
            return summary
        backup = root / '.id_migration_backups' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        backup.mkdir(parents=True)
        registry_path = root / 'experiments/id_registry.json'
        originals = list(changes)
        if registry_path.exists():
            originals.append(registry_path)
        for path in originals:
            target = backup / 'originals' / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        journal = {**summary, 'status': 'prepared',
                   'renamed_paths': [[str(source.relative_to(root)), str(target.relative_to(root))] for source, target in renames],
                   'original_files': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}}
        atomic_write(backup / 'migration.json', json.dumps(journal, ensure_ascii=False, indent=2))
        moved = []
        try:
            for path, text in changes.items():
                atomic_write(path, text)
            for source, destination in renames:
                source.rename(destination)
                moved.append((source, destination))
            write_registry(root, registry)
            journal['status'] = 'complete'
            atomic_write(backup / 'migration.json', json.dumps(journal, ensure_ascii=False, indent=2))
        except Exception:
            for source, destination in reversed(moved):
                destination.rename(source)
            for path in originals:
                shutil.copy2(backup / 'originals' / path.relative_to(root), path)
            if registry_path not in originals and registry_path.exists():
                registry_path.unlink()
            journal['status'] = 'rolled_back'
            atomic_write(backup / 'migration.json', json.dumps(journal, ensure_ascii=False, indent=2))
            raise
        return {**summary, 'applied': True, 'backup': str(backup)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--apply', action='store_true', help='默认只预览；使用此开关才迁移并备份')
    arguments = parser.parse_args()
    print(json.dumps(migrate(arguments.root, apply=arguments.apply), ensure_ascii=False, indent=2))
