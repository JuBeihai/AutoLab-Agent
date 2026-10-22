"""Recover explicitly session-labelled legacy log fragments, with provenance.

Run with --apply to persist; default is a read-only inventory. Untagged entries
are NOT assigned by proximity, timestamps, current session, or inferred meaning.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from webui import chat_store


def fragments(log_dir):
    for path in sorted(Path(log_dir).glob('session_*.log')):
        if path.is_symlink():
            continue
        date = re.search(r'session_(\d{8})_', path.name)
        if not date:
            continue
        current, pieces, start = None, [], 0
        with path.open(encoding='utf-8', errors='replace') as stream:
            for number, line in enumerate(stream, 1):
                header = re.match(r'^\[(\d\d:\d\d:\d\d)\] ', line)
                if header:
                    if current:
                        yield path, start, current, ''.join(pieces).rstrip()
                    current, pieces = None, []
                    tagged = re.match(r'^\[(\d\d:\d\d:\d\d)\] \[会话 ([A-Za-z0-9_-]+)\] \[(user|assistant)\] ?(.*)', line)
                    if tagged:
                        timestamp = datetime.strptime(date[1] + tagged[1], '%Y%m%d%H:%M:%S').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
                        current = {'session_id': tagged[2], 'role': tagged[3], 'timestamp': timestamp.isoformat(),
                                   'order': int(timestamp.timestamp()) * 1000000000 + number}
                        pieces = [tagged[4] + '\n']; start = number
                elif current and sum(map(len, pieces)) < 120000:
                    pieces.append(line)
            if current:
                yield path, start, current, ''.join(pieces).rstrip()


def recover(root, apply=False):
    counts, inserted, missing = {}, 0, 0
    for path, line, metadata, content in fragments(Path(root) / 'webui/logs'):
        session_id = metadata['session_id']
        if not (Path(root) / 'artifacts/feature_mining' / session_id / 'session.json').is_file():
            missing += 1
            continue
        counts[session_id] = counts.get(session_id, 0) + 1
        if apply:
            identifier = 'legacy_' + hashlib.sha256(f'{path.name}:{line}:{session_id}:{content}'.encode()).hexdigest()
            inserted += chat_store.import_legacy(Path(root) / 'artifacts', session_id, identifier, content,
                metadata['role'], {'file': path.name, 'line': line, 'timestamp': metadata['timestamp']}, metadata['order'])
    return {'sessions': counts, 'new_fragments': inserted, 'missing_session_fragments': missing,
            'applied': apply, 'note': '只恢复显式会话标签，所有旧日志片段均标记可能截断；不推测无标签提问的归属。'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(recover(Path(__file__).resolve().parents[1], args.apply), ensure_ascii=False, indent=2))
