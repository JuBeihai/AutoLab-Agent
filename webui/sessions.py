"""Archive session-owned data and update workspace pointers."""

import json
import re
import uuid
from datetime import datetime

from agent.identifiers import activity_lock, resolve_id, new_id
from webui import runner


def create_chat_session(root, artifacts):
    with activity_lock(root):
        session_id = new_id('S', root)
        directory = artifacts / 'feature_mining' / session_id
        directory.mkdir(parents=True)
        session = {'session_id': session_id, 'mode': 'chat', 'status': 'complete',
                   'mock': None, 'created_at': datetime.now().isoformat(timespec='seconds'),
                   'title': '新建对话', 'champion_id': None, 'accepted_recipes': []}
        temporary = directory / 'session.tmp'
        temporary.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(directory / 'session.json')
        return session


def archive_session(session_id, root, artifacts, experiments):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session_id):
        raise ValueError('非法 session_id')
    with runner.idle_guard(), activity_lock(root, exclusive=True):
        session_id = resolve_id(session_id, root)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', session_id):
            raise ValueError('非法 session_id')
        artifact_path = artifacts / 'feature_mining' / session_id
        experiment_path = experiments / 'feature_mining' / session_id
        failed_path = experiments / 'feature_mining/failed' / f'{session_id}.json'
        owned = [artifact_path, experiment_path, failed_path]
        for path, base in ((artifact_path, artifacts), (experiment_path, experiments),
                           (failed_path, experiments)):
            if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
                raise ValueError('会话路径不安全')
        manifest_path = artifact_path / 'session.json'
        if not manifest_path.is_file():
            raise FileNotFoundError('会话不存在')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') not in {'complete', 'failed', 'profile_complete', 'cancelled'}:
            raise RuntimeError('该会话尚未结束，不能删除')
        remaining = []
        for path in (artifacts / 'feature_mining').glob('*/session.json'):
            if path.parent == artifact_path:
                continue
            candidate = json.loads(path.read_text())
            if candidate.get('mode') != 'chat':
                remaining.append(candidate)
        remaining.sort(key=lambda item: (item.get('created_at') or '', item['session_id']), reverse=True)
        successful = [item for item in remaining if item.get('status') == 'complete'
                      and (item.get('champion_id') != item.get('baseline_id') if item.get('baseline_id')
                           else len(item.get('accepted_recipes') or []) > len(item.get('inherited_recipes') or []))]
        pointers = {}
        from skills.city_training import CITIES
        from skills.city_training import latest_path
        pointer_candidates = [('current.json', remaining)]
        for family in ('xgb', 'mlp'):
            for population in ('queue_only', 'unified'):
                for city in (None, *CITIES):
                    pointer_candidates.append((latest_path(root, city, family, population).name,
                        [item for item in successful if item.get('training_city') == city
                         and item.get('model_family', 'xgb') == family
                         and item.get('training_population', 'queue_only') == population]))
        for name, candidates in pointer_candidates:
            path = experiments / 'feature_mining' / name
            if path.exists() and json.loads(path.read_text()).get('session_id') == session_id:
                pointers[path] = (path.read_bytes(), candidates[0] if candidates else None)
        backup = experiments / '.session_trash' / f'{session_id}_{uuid.uuid4().hex}'
        backup.mkdir(parents=True)
        moved = []
        try:
            for path, (original, replacement) in pointers.items():
                (backup / path.name).write_bytes(original)
            for source, name in zip(owned, ('artifacts', 'experiments', 'failed.json')):
                if source.exists():
                    destination = backup / name
                    source.rename(destination)
                    moved.append((source, destination))
            for path, (original, replacement) in pointers.items():
                if replacement is None:
                    path.unlink()
                else:
                    temporary = path.with_suffix('.delete.tmp')
                    temporary.write_text(json.dumps(replacement, ensure_ascii=False, indent=2))
                    temporary.replace(path)
        except Exception:
            for source, destination in reversed(moved):
                destination.rename(source)
            for path, (original, replacement) in pointers.items():
                path.write_bytes(original)
            raise
        return session_id
