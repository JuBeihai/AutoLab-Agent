"""Private raw Parquet snapshots: selection, compatibility and integrity checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from skills.evaluation_policy import CONTRACT

NAMES = ('sds', 'orders', 'queue')


def resolve(root, snapshot_id):
    if not isinstance(snapshot_id, str) or not re.fullmatch(r'raw_[A-Za-z0-9_-]+', snapshot_id):
        raise ValueError('请选择有效的本地快照')
    parent = Path(root).resolve() / 'data_snapshots'
    folder = parent / snapshot_id
    if parent.is_symlink() or folder.is_symlink() or not folder.is_dir():
        raise ValueError('本地快照目录不存在或不安全')
    return folder


def validate(root, snapshot_id, data_cfg, *, verify_hashes=False):
    folder = resolve(root, snapshot_id)
    manifest_path = folder / 'manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('快照清单不能是符号链接')
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        raise ValueError('无法读取本地快照清单') from None
    if not isinstance(manifest, dict) or manifest.get('status') != 'complete' or manifest.get('format') != 'parquet':
        raise ValueError('本地快照尚未完整下载')
    if manifest.get('data_contract') != CONTRACT:
        raise ValueError('本地快照与当前数据口径不一致，请重新下载')
    if manifest.get('data_config') != data_cfg:
        raise ValueError('本地快照的日期或筛选配置与当前配置不一致；请恢复下载时配置，或选择在线取数/重新下载')
    files = manifest.get('files')
    if not isinstance(files, dict):
        raise ValueError('快照文件清单不完整')
    for name in NAMES:
        item = files.get(name, {})
        path = folder / (name + '.parquet')
        if (not isinstance(item, dict) or item.get('file') != path.name
                or item.get('verified_round_trip') is not True or path.is_symlink() or not path.is_file()):
            raise ValueError(f'快照 {name} 文件缺失或未完成校验')
        if path.stat().st_size != item.get('bytes'):
            raise ValueError(f'快照 {name} 文件大小不符，请重新下载')
        if verify_hashes:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != item.get('sha256'):
                raise ValueError(f'快照 {name} 完整性校验失败，请重新下载')
    return folder, manifest


def list_snapshots(root, data_cfg):
    parent = Path(root) / 'data_snapshots'
    if not parent.is_dir() or parent.is_symlink():
        return []
    result = []
    for folder in sorted(parent.iterdir(), reverse=True):
        if not folder.name.startswith('raw_') or folder.is_symlink() or not folder.is_dir():
            continue
        entry = {'id': folder.name, 'compatible': False}
        try:
            _, manifest = validate(root, folder.name, data_cfg)
            entry.update(compatible=True, window=manifest['data_config']['window'],
                         bytes=sum(manifest['files'][name]['bytes'] for name in NAMES),
                         rows={name: manifest['files'][name]['rows'] for name in NAMES})
        except (OSError, ValueError, KeyError, TypeError):
            entry['reason'] = '快照未完成、文件不可用或与当前日期/筛选配置不匹配'
        result.append(entry)
    return result
