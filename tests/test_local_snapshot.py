import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from skills import local_snapshot, mining_data, data_inspection
from skills.evaluation_policy import CONTRACT
from webui import runner

ROOT = Path(__file__).resolve().parents[1]


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = json.loads((ROOT / 'configs/data.json').read_text())
        self.cfg = json.loads((ROOT / 'configs/mining.json').read_text())
        self.identifier = 'raw_test'
        self.folder = self.root / 'data_snapshots' / self.identifier
        self.folder.mkdir(parents=True)
        (self.root / 'configs').mkdir()
        (self.root / 'configs/data.json').write_text(json.dumps(self.data))
        self.manifest = {'status': 'complete', 'format': 'parquet', 'data_contract': CONTRACT,
                         'data_config': self.data, 'files': {}}
        self.frames = {name: pd.DataFrame({'order_id': ['a', 'a'], 'dt': ['2024-08-05'] * 2,
                                         'city_code': [100001] * 2}) for name in local_snapshot.NAMES}
        for name, frame in self.frames.items():
            path = self.folder / (name + '.parquet')
            frame.to_parquet(path, index=False)
            self.manifest['files'][name] = {'file': path.name, 'bytes': path.stat().st_size,
                'rows': len(frame), 'verified_round_trip': True, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        self.save_manifest()

    def save_manifest(self):
        (self.folder / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_round_trip_preserves_duplicates_without_spark_or_secret(self):
        captured = {}
        def prepare(frames, *args):
            captured.update(frames)
            return 'full', 'lineage', args[3]
        with patch('skills.runtime_memory.require_available_memory'), \
             patch.object(mining_data, 'prepare_raw', side_effect=prepare), \
             patch('skills.data_loader.build_spark_session') as spark, \
             patch('skills.env_secrets.get_dp_key') as key:
            full, _, audit = mining_data.load_local(self.data, self.cfg, '2024-09-09', self.identifier, root=self.root)
        self.assertEqual(full, 'full')
        self.assertEqual(audit['data_source'], 'offline')
        for name in local_snapshot.NAMES:
            pd.testing.assert_frame_equal(captured[name], self.frames[name])
        spark.assert_not_called()
        key.assert_not_called()

    def test_validation_rejects_missing_incomplete_mismatched_and_unsafe(self):
        for identifier in ('../raw_test', '/tmp/raw_test', None, [], 'raw_missing'):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                local_snapshot.validate(self.root, identifier, self.data)
        changed = copy.deepcopy(self.data)
        changed['window']['train_end'] = '2024-09-08'
        with self.assertRaisesRegex(ValueError, '不一致'):
            local_snapshot.validate(self.root, self.identifier, changed)
        self.manifest['status'] = 'downloading'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, '完整下载'):
            local_snapshot.validate(self.root, self.identifier, self.data)
        self.assertFalse(local_snapshot.list_snapshots(self.root, self.data)[0]['compatible'])

    def test_hash_and_row_budget_are_checked(self):
        self.manifest['files']['sds']['sha256'] = '0' * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, '完整性'):
            local_snapshot.validate(self.root, self.identifier, self.data, verify_hashes=True)
        self.manifest['files']['sds']['sha256'] = hashlib.sha256((self.folder / 'sds.parquet').read_bytes()).hexdigest()
        self.save_manifest()
        with patch('skills.runtime_memory.require_available_memory'), self.assertRaisesRegex(ValueError, '内存预算'):
            mining_data.load_local(self.data, {**self.cfg, 'max_rows': 1}, '2024-09-09', self.identifier, root=self.root)

    def test_symlink_rejected(self):
        link = self.folder.parent / 'raw_link'
        link.symlink_to(self.folder, target_is_directory=True)
        with self.assertRaises(ValueError):
            local_snapshot.validate(self.root, link.name, self.data)

    def test_offline_inspection_cannot_query_tables(self):
        audit = {'data_source': 'offline'}
        self.assertEqual(data_inspection.capabilities(audit, self.cfg)['operations'], ['fields'])
        with self.assertRaisesRegex(ValueError, '离线数据模式'):
            data_inspection.validate({'kind': 'table'}, {}, audit, self.cfg)

    def test_runner_forwards_offline_selection_without_spark_shell(self):
        with patch.object(runner, '_ROOT', self.root), patch.object(runner, '_jobs', {}), \
             patch.object(runner, 'require_available_memory', return_value={}), \
             patch.object(runner, '_LOG_SINK', None), patch.object(runner.subprocess, 'Popen') as spawn, \
             patch.object(runner.threading, 'Thread'):
            runner.start_orchestrator(1, 1, False, True, data_source='offline', snapshot_id=self.identifier)
        cmd = spawn.call_args.args[0]
        self.assertEqual(cmd[:3], [runner._PYTHON, '-u', 'agent/orchestrator.py'])
        self.assertEqual(cmd[cmd.index('--snapshot-id') + 1], self.identifier)
        self.assertEqual(cmd[cmd.index('--data-source') + 1], 'offline')

    def test_api_lists_and_validates_before_launch(self):
        with patch.dict(os.environ, {'ETS_WEBUI_STATE_DIR': str(self.root / 'state'),
                                    'ETS_WEBUI_LOG_DIR': str(self.root / 'logs')}):
            module = importlib.import_module('webui.app')
        with patch.object(module, '_ROOT', self.root), patch.object(module.runner, 'start_orchestrator', return_value='job_test') as launch:
            client = module.app.test_client()
            rows = client.get('/api/data-sources').get_json()['snapshots']
            self.assertEqual(rows[0]['id'], self.identifier)
            self.assertTrue(rows[0]['compatible'])
            for body in ({'data_source': 'bad'}, {'data_source': []}, {'data_source': 'offline'},
                         {'data_source': 'offline', 'snapshot_id': '../secret'},
                         {'data_source': 'online', 'snapshot_id': self.identifier}):
                self.assertEqual(client.post('/api/run', json=body).status_code, 400)
            launch.assert_not_called()
            response = client.post('/api/run', json={'data_source': 'offline', 'snapshot_id': self.identifier})
            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertEqual(launch.call_args.kwargs['snapshot_id'], self.identifier)
            self.assertEqual(launch.call_args.kwargs['data_source'], 'offline')
            self.assertEqual(client.post('/api/run', json={}).status_code, 200)
            self.assertEqual(launch.call_args.kwargs['data_source'], 'online')


if __name__ == '__main__':
    unittest.main()
