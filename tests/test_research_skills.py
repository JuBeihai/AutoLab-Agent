from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import pandas as pd

from agent import experiment_spec as specs
from agent import search_memory
from skills import research_extensions as extensions


ROOT = Path(__file__).resolve().parents[1]


def proposal(**changes):
    return {'id': 'trial', 'hypothesis': '检验改动', 'evidence': '训练画像', 'feature_recipes': [], **changes}


def recipe(name='derived', column='raw_first'):
    return {'name': name, 'kind': 'expression',
            'expression': {'op': 'log1p', 'args': [{'col': column}]}}


class SpecificationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'configs/mining.json').read_text())
        config = json.loads((ROOT / 'configs/train.json').read_text())
        config['training_method'] = 'full_train'
        self.parent = specs.baseline_spec(['raw_first', 'raw_second'], [recipe()], config)
        self.catalog = {'raw_first': {'role': 'numeric'}, 'raw_second': {'role': 'numeric'}}

    def derive(self, change):
        return specs.derive(change, self.parent, self.catalog, self.cfg)

    def test_delete_accepted_and_original_features_without_mutating_parent(self):
        before = deepcopy(self.parent)
        candidate = self.derive(proposal(remove_features=['derived', 'raw_second']))
        self.assertEqual(specs.feature_names(candidate), ['raw_first'])
        self.assertEqual(self.parent, before)
        self.assertEqual(specs.describe_change(self.parent, candidate)['removed'], ['derived', 'raw_second'])

    def test_replace_same_derived_name_and_validate_dependencies(self):
        self.parent['recipes'].append(recipe('dependent', 'derived'))
        with self.assertRaises(ValueError):
            self.derive(proposal(remove_features=['derived']))
        candidate = self.derive(proposal(remove_features=['derived', 'dependent'],
                                        feature_recipes=[recipe(column='raw_second')]))
        self.assertEqual(candidate['recipes'], [recipe(column='raw_second')])
        self.assertNotEqual(specs.signature(candidate), specs.signature(self.parent))

    def test_bounded_training_preserves_protocol(self):
        candidate = self.derive(proposal(training={'xgb_params': {'max_depth': 5, 'learning_rate': .04},
                                                  'class_weight_scheme': 'none'}))
        self.assertEqual(candidate['train_config']['xgboost']['max_depth'], 5)
        for key in ('training_method', 'training_filters', 'non_queue_strategy', 'split'):
            self.assertEqual(candidate['train_config'][key], self.parent['train_config'][key])
        self.assertEqual(candidate['train_config']['class_weight']['scheme'], 'none')

    def test_forbidden_and_out_of_range_parameters_rejected(self):
        for params in ({'early_stopping_rounds': 10}, {'objective': 'reg:squarederror'},
                       {'random_state': 8}, {'n_jobs': 32}, {'max_depth': 11},
                       {'max_depth': 3.5}, {'max_depth': True}, {'subsample': float('nan')}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.derive(proposal(training={'xgb_params': params}))
        for change in (proposal(training={'training_filters': {}}), proposal(sql='drop table'),
                       proposal(remove_features=['unknown']), proposal(remove_features=['derived', 'derived'])):
            with self.assertRaises(ValueError):
                self.derive(change)

    def test_noop_and_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.derive(proposal())
        with self.assertRaises(ValueError):
            self.derive(proposal(remove_features=specs.feature_names(self.parent)))

    def test_training_and_feature_identity_are_part_of_dedup(self):
        candidate = self.derive(proposal(training={'xgb_params': {'max_depth': 5}}))
        context = {'window': {}}
        entry = {'status': 'completed', 'context_hash': specs.program.fingerprint(context),
                 'parent_signature': 'parent', 'recipe_signature': search_memory.recipe_signature(candidate['recipes']),
                 'spec_signature': specs.signature(candidate)}
        self.assertIsNotNone(search_memory.duplicate_entry([entry], context, 'parent', candidate['recipes'], specs.signature(candidate)))
        self.assertIsNone(search_memory.duplicate_entry([entry], context, 'parent', self.parent['recipes'], specs.signature(self.parent)))
        self.assertIsNone(search_memory.duplicate_entry([entry], context, 'parent', candidate['recipes']))

    def test_unweighted_scheme_really_uses_uniform_sample_weights(self):
        from skills.trainer import _build_sample_weights
        weights, _ = _build_sample_weights(pd.Series([0, 0, 0, 1]), 'none')
        self.assertEqual(weights.tolist(), [1, 1, 1, 1])

    def test_extension_approval_and_atomic_output_removal(self):
        extension = {'id': 'R1234567', 'outputs': ['extra_first', 'extra_second'], 'digest': 'test'}
        with self.assertRaises(ValueError):
            self.derive(proposal(extensions=['R1234567']))
        candidate = specs.derive(proposal(extensions=['R1234567']), self.parent, self.catalog, self.cfg, [extension])
        with self.assertRaises(ValueError):
            specs.derive(proposal(remove_features=['extra_first']), candidate, self.catalog, self.cfg, [extension])
        restored = specs.derive(proposal(remove_features=extension['outputs']), candidate, self.catalog, self.cfg, [extension])
        self.assertEqual(restored, self.parent)


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = {'raw_input': {'role': 'numeric'}, 'city_code': {'groupable': True}}
        self.source = {'kind': 'data_source', 'reason': '实时供给变化', 'source': 'reviewed.supply',
                       'keys': ['order_id'], 'time_column': 'available_at',
                       'availability': '事件产生且已到达服务的时间', 'outputs': ['supply_signal']}
        self.code = {'kind': 'code_feature', 'reason': '新的受控转换', 'inputs': ['raw_input'],
                     'outputs': ['code_signal'], 'files': {
                         'feature.py': "def transform(row):\n    return {'code_signal': row['raw_input']}\n",
                         'test_feature.py': 'import unittest\nclass Tests(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(1, 1)\n'}}

    def stage(self, value):
        return extensions.stage(self.root, 'S0000001', value, self.catalog)['id']

    def snapshot(self, text=None):
        path = self.root / 'snapshot.csv'
        path.write_text(text or 'order_id,available_at,supply_signal\n10,2024-09-01 09:00:00,1\n10,2024-09-01 11:00:00,9\n')
        return path

    def test_request_never_approves_or_executes_itself(self):
        with patch.object(extensions, '_sandbox', side_effect=AssertionError('must not execute')):
            request_id = self.stage(self.code)
        self.assertEqual(extensions.requests(self.root)[0]['status'], 'pending_review')
        self.assertEqual(extensions.approved(self.root), [])
        self.assertEqual(self.stage(self.code), request_id)

    def test_code_rejects_paths_and_labels(self):
        bad = deepcopy(self.code)
        bad['files']['../../agent/feature_mining.py'] = 'pass'
        with self.assertRaises(ValueError):
            self.stage(bad)
        bad = deepcopy(self.code)
        bad['inputs'] = ['true_ets']
        with self.assertRaises(ValueError):
            self.stage(bad)

    def test_missing_sandbox_fails_closed_and_leaves_pending(self):
        request_id = self.stage(self.code)
        with patch.object(extensions.shutil, 'which', return_value=None), self.assertRaises(RuntimeError):
            extensions.review(self.root, request_id, 'approve', '人工确认转换语义')
        self.assertEqual(extensions.requests(self.root)[0]['status'], 'pending_review')

    def test_reject_and_review_note_required(self):
        request_id = self.stage(self.source)
        with self.assertRaises(ValueError):
            extensions.review(self.root, request_id, 'approve', '', self.snapshot())
        extensions.review(self.root, request_id, 'reject', '可用时点不可信')
        self.assertEqual(extensions.approved(self.root), [])
        with self.assertRaises(ValueError):
            extensions.review(self.root, request_id, 'approve', '再次审批', self.snapshot())

    def test_source_asof_preserves_duplicate_rows_order_and_excludes_future(self):
        request_id = self.stage(self.source)
        extensions.review(self.root, request_id, 'approve', '已核对时点与来源', self.snapshot())
        frame = pd.DataFrame({'order_id': [10, 10, 10, 10], 'prediction_time': [
            '2024-09-01 10:00:00', '2024-09-01 08:00:00', '2024-09-01 10:00:00', '2024-09-01 11:00:00'],
            'true_ets': [100, 200, 300, 400]}, index=[5, 3, 3, 7])
        result = extensions.materialize(self.root, frame, extensions.approved(self.root))
        pd.testing.assert_frame_equal(result[frame.columns], frame)
        self.assertEqual(result['supply_signal'].iloc[[0, 2, 3]].tolist(), [1, 1, 1])
        self.assertTrue(pd.isna(result['supply_signal'].iloc[1]))

    def test_source_duplicate_keys_schema_and_tamper_rejected(self):
        request_id = self.stage(self.source)
        duplicate = self.snapshot('order_id,available_at,supply_signal\n10,2024-09-01,1\n10,2024-09-01,2\n')
        with self.assertRaises(ValueError):
            extensions.review(self.root, request_id, 'approve', 'reviewed', duplicate)
        extensions.review(self.root, request_id, 'approve', 'reviewed', self.snapshot())
        registry = extensions.approved(self.root)
        (extensions.request_path(self.root, request_id) / 'approved_snapshot.csv').write_text('tampered')
        with self.assertRaises(ValueError):
            extensions.materialize(self.root, pd.DataFrame(), registry)

    def test_code_contract_enforced_after_review(self):
        request_id = self.stage(self.code)
        with patch.object(extensions, '_sandbox', return_value=''):
            extensions.review(self.root, request_id, 'approve', 'reviewed source and tests')
        frame = pd.DataFrame({'raw_input': [1, 2], 'true_ets': [30, 40]})
        def execute(folder, mode):
            self.assertEqual(mode, 'transform')
            self.assertNotIn('true_ets', (folder / 'input.jsonl').read_text())
            return '{"code_signal":1}\n{"code_signal":2}\n'
        with patch.object(extensions, '_sandbox', side_effect=execute):
            result = extensions.materialize(self.root, frame, extensions.approved(self.root))
        self.assertEqual(result['code_signal'].tolist(), [1, 2])
        for output in ('{}\n{}', '{"code_signal":1}', '{"code_signal":true}\n{"code_signal":2}'):
            with patch.object(extensions, '_sandbox', return_value=output), self.assertRaises(ValueError):
                extensions.materialize(self.root, frame, extensions.approved(self.root))

    def test_data_source_mode_is_not_mixed(self):
        request_id = extensions.stage(self.root, 'S0000001', self.source, self.catalog, mock=False)['id']
        extensions.review(self.root, request_id, 'approve', 'reviewed source', self.snapshot())
        self.assertEqual(len(extensions.approved(self.root, mock=False)), 1)
        self.assertEqual(extensions.approved(self.root, mock=True), [])

    def test_code_review_rejects_edited_workspace(self):
        request_id = self.stage(self.code)
        (extensions.request_path(self.root, request_id) / 'feature.py').write_text('raise RuntimeError()')
        with self.assertRaises(ValueError):
            extensions.review(self.root, request_id, 'approve', 'reviewed')

    def test_sandbox_requires_pinned_image_and_rootless_runtime(self):
        image = 'local/python@sha256:' + 'a' * 64
        with patch.object(extensions.shutil, 'which', return_value='/bin/docker'), \
             patch.dict(extensions.os.environ, {'AUTOLAB_RESEARCH_SANDBOX_IMAGE': 'python:latest'}), self.assertRaises(RuntimeError):
            extensions.sandbox_info()
        with patch.object(extensions.shutil, 'which', return_value='/bin/docker'), \
             patch.dict(extensions.os.environ, {'AUTOLAB_RESEARCH_SANDBOX_IMAGE': image}), \
             patch.object(extensions.subprocess, 'run', return_value=Mock(stdout='{"SecurityOptions":["seccomp"]}')), self.assertRaises(RuntimeError):
            extensions.sandbox_info()

    def test_sandbox_command_has_resource_and_access_boundaries(self):
        process = Mock(returncode=0)
        process.poll.return_value = 0
        with patch.object(extensions, 'sandbox_info', return_value=('/bin/docker', 'pinned-image')), \
             patch.object(extensions.subprocess, 'Popen', return_value=process) as launch, \
             patch.object(extensions.subprocess, 'run') as cleanup:
            self.assertEqual(extensions._sandbox(self.root, 'test'), '')
        command = launch.call_args.args[0]
        for flag, value in (('--network', 'none'), ('--pull', 'never'), ('--cap-drop', 'ALL'),
                            ('--security-opt', 'no-new-privileges'), ('--memory', '512m'),
                            ('--cpus', '1'), ('--pids-limit', '64'), ('--user', '65534:65534')):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn('--read-only', command)
        self.assertTrue(command[command.index('--mount') + 1].endswith('readonly'))
        self.assertNotIn('shell', launch.call_args.kwargs)
        self.assertNotIn('-e', command)
        self.assertTrue(launch.call_args_list[0].kwargs['start_new_session'])
        self.assertEqual(cleanup.call_args.args[0][1:3], ['rm', '-f'])

    def test_sandbox_timeout_cleans_up_container(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(extensions, 'sandbox_info', return_value=('/bin/docker', 'pinned-image')), \
             patch.object(extensions.subprocess, 'Popen', return_value=process), \
             patch.object(extensions.subprocess, 'run') as cleanup, \
             patch.object(extensions.time, 'monotonic', side_effect=[0, 121]), self.assertRaises(RuntimeError):
            extensions._sandbox(self.root, 'test')
        cleanup.assert_called_once()
        process.kill.assert_called_once()

    def test_sandbox_requires_enforceable_resource_limits(self):
        image = 'local/python@sha256:' + 'a' * 64
        info = {'SecurityOptions': ['name=rootless'], 'CgroupVersion': '2',
                'MemoryLimit': True, 'CpuCfsQuota': True, 'PidsLimit': False}
        with patch.object(extensions.shutil, 'which', return_value='/bin/docker'), \
             patch.dict(extensions.os.environ, {'AUTOLAB_RESEARCH_SANDBOX_IMAGE': image}), \
             patch.object(extensions.subprocess, 'run', return_value=Mock(stdout=json.dumps(info))), self.assertRaises(RuntimeError):
            extensions.sandbox_info()

    def test_watchdog_cleans_up_on_owner_exit_or_deadline(self):
        from agent import sandbox_watchdog
        with patch.object(sandbox_watchdog.select, 'select', return_value=([], [], [])) as monitor, \
             patch.object(sandbox_watchdog.subprocess, 'run') as cleanup, \
             patch.object(sandbox_watchdog.time, 'sleep'):
            sandbox_watchdog.watch('/bin/docker', 'ets-research-' + 'a' * 16, 120, 'pipe')
        monitor.assert_called_once_with(['pipe'], [], [], 120)
        self.assertEqual(cleanup.call_count, 3)


if __name__ == '__main__':
    unittest.main()
