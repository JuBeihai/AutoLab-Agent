import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from skills import data_loader, runtime_memory


class RuntimeMemoryTests(unittest.TestCase):
    def test_cgroup_v1_v2_and_unlimited(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                limit, usage, events = (('memory.max', 'memory.current', 'memory.events') if version == 2 else
                                        ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.oom_control'))
                (root / limit).write_text(str(20 * 1024 ** 3))
                (root / usage).write_text(str(15 * 1024 ** 3))
                (root / events).write_text('oom_kill 4\n')
                snapshot = runtime_memory.memory_snapshot(root)
                self.assertEqual(snapshot['available_bytes'], 5 * 1024 ** 3)
                self.assertEqual(snapshot['oom_kill_count'], 4)
                (root / limit).write_text('max' if version == 2 else str(2 ** 63 - 1))
                self.assertIsNone(runtime_memory.memory_snapshot(root))

    def test_guard_uses_container_headroom_and_configured_threshold(self):
        snapshot = {'available_bytes': 5 * 1024 ** 3}
        with patch.object(runtime_memory, 'memory_snapshot', return_value=snapshot):
            with self.assertRaisesRegex(runtime_memory.MemoryBudgetError, '5.0 GiB'):
                runtime_memory.require_available_memory({})
            self.assertIs(runtime_memory.require_available_memory({'AUTOLAB_MIN_AVAILABLE_MEMORY_GB': '4'}), snapshot)
            for invalid in ('nan', 'inf', '-1', 'invalid'):
                with self.assertRaises(RuntimeError):
                    runtime_memory.require_available_memory({'AUTOLAB_MIN_AVAILABLE_MEMORY_GB': invalid})
        with patch.object(runtime_memory, 'memory_snapshot', return_value=None):
            self.assertIsNone(runtime_memory.require_available_memory({}))

    def test_reclaimable_cache_credit_is_conservative_and_bad_stats_fall_back(self):
        for version in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                names = ('memory.limit_in_bytes', 'memory.usage_in_bytes') if version == 1 else ('memory.max', 'memory.current')
                (root / names[0]).write_text('20000'); (root / names[1]).write_text('16000')
                prefix = 'total_' if version == 1 else ''
                mapped = 'mapped_file' if version == 1 else 'file_mapped'
                (root / 'memory.stat').write_text(f'{prefix}inactive_file 10000\n{prefix}shmem 2000\n{prefix}{mapped} 1000\n')
                snapshot = runtime_memory.memory_snapshot(root)
                self.assertEqual(snapshot['raw_available_bytes'], 4000)
                self.assertEqual(snapshot['reclaimable_credit_bytes'], 3500)
                self.assertEqual(snapshot['available_bytes'], 7500)
                (root / 'memory.stat').write_text('invalid')
                self.assertEqual(runtime_memory.memory_snapshot(root)['available_bytes'], 4000)
