"""Small, dependency-free memory checks for containerized experiment workers."""
from __future__ import annotations

import math
import os
from pathlib import Path


class MemoryBudgetError(RuntimeError):
    pass


def memory_snapshot(root=Path('/sys/fs/cgroup')):
    root = Path(root)
    candidates = [root, root / 'memory']
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, relative = line.split(':', 2)
            if not controllers or 'memory' in controllers.split(','):
                relative = relative.lstrip('/')
                if '..' not in Path(relative).parts:
                    candidates.extend((root / relative, root / 'memory' / relative))
    except (OSError, ValueError):
        pass
    snapshots = []
    for directory in dict.fromkeys(candidates):
        for limit_name, used_name, events_name in (
                ('memory.max', 'memory.current', 'memory.events'),
                ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.oom_control')):
            try:
                limit = int((directory / limit_name).read_text().strip())
                used = int((directory / used_name).read_text().strip())
                if not 0 < limit < 2 ** 60 or used < 0:
                    continue
            except (OSError, ValueError):
                continue
            try:
                events = dict(line.split() for line in (directory / events_name).read_text().splitlines())
                killed = int(events.get('oom_kill', 0))
            except (OSError, ValueError):
                killed = None
            reclaimable = 0
            try:
                stats = {key: int(value) for key, value in
                         (line.split() for line in (directory / 'memory.stat').read_text().splitlines())}
                prefix = 'total_' if limit_name == 'memory.limit_in_bytes' else ''
                inactive = max(0, stats[prefix + 'inactive_file'])
                # Only count half of clean inactive file pages. Do not credit
                # shared, mapped, dirty or writeback pages; this is deliberately
                # more conservative than Docker's inactive_file subtraction.
                reserved = sum(max(0, stats.get(prefix + key, 0)) for key in
                               (('shmem', 'mapped_file', 'dirty', 'writeback') if prefix else
                                ('shmem', 'file_mapped', 'file_dirty', 'file_writeback')))
                reclaimable = min(used, max(0, inactive - reserved)) // 2
            except (OSError, ValueError, KeyError):
                pass  # Missing accounting falls back to raw free capacity.
            raw_available = max(0, limit - used)
            snapshots.append({'limit_bytes': limit, 'used_bytes': used,
                              'available_bytes': min(limit, raw_available + reclaimable),
                              'raw_available_bytes': raw_available,
                              'reclaimable_credit_bytes': reclaimable,
                              'oom_kill_count': killed, 'source': str(directory)})
    return min(snapshots, key=lambda item: item['available_bytes']) if snapshots else None


def require_available_memory(environment=None):
    environment = os.environ if environment is None else environment
    try:
        minimum = float(environment.get('AUTOLAB_MIN_AVAILABLE_MEMORY_GB', '6'))
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise RuntimeError('AUTOLAB_MIN_AVAILABLE_MEMORY_GB 必须是非负有限数字') from None
    snapshot = memory_snapshot()
    if snapshot and snapshot['available_bytes'] < minimum * 1024 ** 3:
        available = snapshot['available_bytes'] / 1024 ** 3
        raise MemoryBudgetError(f'容器剩余内存约 {available:.1f} GiB，低于真实实验启动要求 {minimum:g} GiB。'
                           '请关闭不再使用的 Notebook 内核及其 Spark，或增加容器内存后重试。')
    return snapshot
