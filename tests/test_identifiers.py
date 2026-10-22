from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.experiment_log import new_experiment_id
from agent.identifiers import activity_lock, allocate, read_registry, resolve_id
from scripts.migrate_short_ids import migrate


class IdentifierTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_persistent_eight_character_ids_and_concurrent_allocation(self):
        self.assertEqual(new_experiment_id('session', self.root), 'S0000001')
        self.assertEqual(new_experiment_id('mining_real', self.root), 'S0000002')
        with ThreadPoolExecutor(max_workers=8) as workers:
            identifiers = list(workers.map(lambda _: new_experiment_id(root=self.root), range(100)))
        self.assertEqual(len(set(identifiers)), 100)
        for identifier in identifiers:
            self.assertRegex(identifier, r'^E[0-9A-Z]{7}$')
        self.assertEqual(read_registry(self.root)['next']['E'], 101)
        self.assertNotIn(new_experiment_id('r2', self.root), identifiers)

    def test_id_space_exhaustion_does_not_wrap(self):
        with self.assertRaises(ValueError):
            allocate({'next': {'S': 36 ** 7}}, 'S')

    def create_history(self):
        session = 'mining_mock_20240910_154139_1c8c8b'
        experiment = session + '_r0_20240910_154150_b4b91f'
        directory = self.root / 'artifacts/feature_mining' / session
        model_dir = directory / experiment
        model_dir.mkdir(parents=True)
        logs = self.root / 'experiments/feature_mining' / session
        logs.mkdir(parents=True)
        manifest = {'session_id': session, 'status': 'complete', 'champion_id': experiment,
                    'baseline_id': experiment, 'artifact_dir': str(directory), 'log_dir': str(logs)}
        (directory / 'session.json').write_text(json.dumps(manifest))
        for name in ('current', 'latest'):
            (logs.parent / f'{name}.json').write_text(json.dumps(manifest))
        record = {'experiment_id': experiment, 'timestamp': '2024-09-10T15:41:50',
                  'config_diff': {'session_id': session, 'parent_experiment_id': experiment},
                  'artifact_dir': str(model_dir), 'summary': {'narrative_zh': f'模型 {experiment}'}}
        (logs / 'experiments.jsonl').write_text(json.dumps(record) + '\n')
        (model_dir / 'validation_result.json').write_text(json.dumps(record))
        model = model_dir / 'model.json'
        model.write_bytes(b'{"model": "unchanged"}')
        digest = hashlib.md5(model.read_bytes()).hexdigest()
        model.with_suffix('.json.md5').write_text(digest + '  model.json\n')
        model.with_suffix('.json.meta.json').write_text(json.dumps({'experiment_id': experiment, 'md5': digest}))
        failed = logs.parent / 'failed'
        failed.mkdir()
        (failed / f'{session}.json').write_text(json.dumps(manifest))
        return session, experiment, digest

    def test_preview_migration_references_models_aliases_and_idempotence(self):
        session, experiment, digest = self.create_history()
        before = (self.root / 'experiments/feature_mining/current.json').read_bytes()
        preview = migrate(self.root)
        self.assertFalse(preview['applied'])
        self.assertEqual(preview['sessions'], 1)
        self.assertEqual(preview['experiments'], 1)
        self.assertEqual((self.root / 'experiments/feature_mining/current.json').read_bytes(), before)
        self.assertFalse((self.root / 'experiments/id_registry.json').exists())
        result = migrate(self.root, apply=True)
        self.assertTrue(result['applied'])
        self.assertEqual(resolve_id(session, self.root), 'S0000001')
        self.assertEqual(resolve_id(experiment, self.root), 'E0000001')
        self.assertEqual(resolve_id('E0000001', self.root), 'E0000001')
        pointer = json.loads((self.root / 'experiments/feature_mining/current.json').read_text())
        self.assertEqual(pointer['session_id'], 'S0000001')
        self.assertEqual(pointer['champion_id'], 'E0000001')
        self.assertEqual(pointer['created_at'], '2024-09-10T15:41:39')
        self.assertTrue(Path(pointer['artifact_dir']).is_dir())
        logs = Path(pointer['log_dir'])
        record = json.loads((logs / 'experiments.jsonl').read_text())
        self.assertEqual(record['config_diff']['session_id'], 'S0000001')
        self.assertEqual(record['config_diff']['parent_experiment_id'], 'E0000001')
        model = Path(record['artifact_dir']) / 'model.json'
        self.assertEqual(hashlib.md5(model.read_bytes()).hexdigest(), digest)
        self.assertEqual(json.loads(model.with_suffix('.json.meta.json').read_text())['experiment_id'], 'E0000001')
        self.assertTrue((logs.parent / 'failed/S0000001.json').exists())
        backup = Path(result['backup'])
        self.assertEqual((backup / 'originals/experiments/feature_mining/current.json').read_bytes(), before)
        self.assertEqual(json.loads((backup / 'migration.json').read_text())['status'], 'complete')
        self.assertFalse(migrate(self.root, apply=True)['applied'])
        self.assertEqual(new_experiment_id(root=self.root), 'E0000002')
        self.assertEqual(new_experiment_id('session', root=self.root), 'S0000002')

    def test_failed_migration_restores_original_paths_and_metadata(self):
        session, experiment, digest = self.create_history()
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        with patch('scripts.migrate_short_ids.write_registry', side_effect=OSError('simulated disk error')):
            with self.assertRaises(OSError):
                migrate(self.root, apply=True)
        for name, contents in before.items():
            self.assertEqual((self.root / name).read_bytes(), contents, name)
        self.assertFalse((self.root / 'artifacts/feature_mining/S0000001').exists())
        self.assertFalse((self.root / 'experiments/id_registry.json').exists())
        journal = next((self.root / '.id_migration_backups').glob('*/migration.json'))
        self.assertEqual(json.loads(journal.read_text())['status'], 'rolled_back')

    def test_migration_refuses_running_experiment(self):
        self.create_history()
        with activity_lock(self.root):
            with self.assertRaises(RuntimeError):
                migrate(self.root, apply=True)

    def test_migration_never_overwrites_existing_destination(self):
        self.create_history()
        (self.root / 'artifacts/feature_mining/S0000001').mkdir()
        with self.assertRaises(ValueError):
            migrate(self.root, apply=True)
