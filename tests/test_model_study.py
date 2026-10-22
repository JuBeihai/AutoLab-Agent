from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import experiment_spec
from skills import model_study as study
from skills import mining_train

ROOT = Path(__file__).resolve().parents[1]


class AutonomousModelTests(unittest.TestCase):
    def setUp(self):
        self.config = {name: json.loads((ROOT / f'configs/{name}.json').read_text())
                       for name in ('train', 'data', 'features', 'mining')}
        self.config['train']['model_family'] = 'mlp'

    def test_api_and_runner_forward_and_validate_study(self):
        from webui.app import app
        from webui import runner
        with app.test_client() as client:
            self.assertEqual(client.post('/api/run', json={'model_study': 'yes'}).status_code, 400)
            with patch.object(runner, 'start_orchestrator', return_value='job_study') as start:
                response = client.post('/api/run', json={'model_study': True, 'model_family': 'mlp',
                    'rounds': 8, 'max_proposals': 3})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(start.call_args.kwargs['model_study'])
        with self.assertRaises(ValueError):
            runner.start_orchestrator(8, 3, True, True, model_family='mlp', model_study=True)
        with self.assertRaises(ValueError):
            runner.start_orchestrator(8, 3, True, False, model_family='xgb', model_study=True)

    def test_mlp_and_weight_protocol_is_bounded_and_inherited(self):
        parent = experiment_spec.baseline_spec(['a'], [], self.config['train'])
        proposal = {'id': 'p', 'hypothesis': '试验学习率', 'evidence': '训练曲线',
                    'feature_recipes': [], 'training': {'mlp_params': {'learning_rate': .0005},
                    'penalty_multipliers': {'3': 1.1, '4': 1.2, '5': 1.3}}}
        before = deepcopy(parent)
        result = experiment_spec.derive(proposal, parent, {'a': {'role': 'numeric'}}, self.config['mining'])
        self.assertEqual(parent, before)
        self.assertEqual(result['train_config']['mlp']['learning_rate'], .0005)
        self.assertEqual(result['train_config']['split'], parent['train_config']['split'])
        self.assertEqual(result['train_config']['class_weight']['penalty_multipliers']['5'], 1.3)
        self.assertEqual(experiment_spec.describe_change(parent, result)['mlp_params'], {'learning_rate': .0005})
        with self.assertRaisesRegex(ValueError, '没有可执行改动'):
            experiment_spec.derive({**proposal, 'training': {'mlp_params': {'learning_rate': .001}}},
                                   parent, {'a': {'role': 'numeric'}}, self.config['mining'])
        for changes in [{'mlp_params': {'hidden_layers': [1024]}}, {'mlp_params': {'learning_rate': True}},
                        {'mlp_params': {'max_epochs': 1000}}, {'xgb_params': {'max_depth': 5}},
                        {'model_seed': 456}, {'penalty_multipliers': {'6': 1}},
                        {'penalty_multipliers': {'3': float('nan')}}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                experiment_spec.derive({**proposal, 'training': changes}, parent, {'a': {'role': 'numeric'}}, self.config['mining'])

    def test_temporal_partition_and_seeds_preserve_order_groups(self):
        fold = study.folds(self.config['data']['window'])[0]
        self.assertEqual(fold['early_stop_start'], '2024-08-27')
        self.assertEqual(fold['validation_start'], '2024-09-03')
        df = pd.DataFrame({'dt': ['2024-08-26'] * 2 + ['2024-08-29'] * 2 + ['2024-09-04'] * 2,
            'order_id': ['fit', 'fit', 'stop', 'stop', 'val', 'val'], 'is_answer': 1,
            'match_time': pd.to_datetime(['2024-08-26'] * 2 + ['2024-08-29'] * 2 + ['2024-09-04'] * 2),
            'cancel_time': pd.NaT})
        study.partition(df, fold)
        cfg = {**self.config['train'], 'training_method': 'temporal',
               'temporal_early_stop_start': fold['early_stop_start'],
               'temporal_validation_start': fold['validation_start']}
        for seed in study.SEEDS:
            cfg['model_seed'] = seed
            fit, stop = mining_train.notebook_split(df, True, cfg)
            self.assertEqual(fit.index.tolist(), [0, 1])
            self.assertEqual(stop.index.tolist(), [2, 3])
        df.loc[4, 'order_id'] = 'fit'
        with self.assertRaisesRegex(ValueError, '跨越时间'):
            study.partition(df, fold)

    def test_probability_fusion_and_risk_policy(self):
        left, right = np.eye(6), np.flip(np.eye(6), axis=1)
        np.testing.assert_allclose(study.blend(left, right, .25), .25 * left + .75 * right)
        np.testing.assert_allclose(study.blend(left, right, .25).sum(axis=1), 1)
        ref = {'severe_under': .03, 'long_wait_severe_under': .3}
        self.assertFalse(study.acceptable({'severe_under': .04, 'long_wait_severe_under': .2}, ref))
        self.assertTrue(study.acceptable(ref, ref))

    def test_ensemble_is_not_reported_as_a_component_champion(self):
        from webui import app as web
        from skills import model_registry
        session = {'session_id': 'S0000001', 'model_study': True, 'champion_id': None}
        with patch.object(web, '_current_records', return_value=[
                {'experiment_id': 'E0000001', 'proposer': 'baseline'}]):
            self.assertIsNone(web._current_champion(session))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / 'artifacts/feature_mining/S0000001'
            bundle = directory / 'final_model/model_bundle.json'
            study.write_json(bundle, {'components': [{'model_family': 'mlp', 'weight': .5},
                                                     {'model_family': 'xgb', 'weight': .5}]})
            study.write_json(directory / 'session.json', {**session, 'model_bundle': str(bundle)})
            rows = model_registry.saved_models(root)
            self.assertEqual(rows[0]['model_family'], 'ensemble')
            self.assertEqual(rows[0]['model_path'], str(bundle))

    def test_real_training_reload_and_no_validation_label_leakage(self):
        from skills.mock_data import generate_mock_merged
        from skills.feature_builder import build_full_features
        data = generate_mock_merged('2024-08-05', '2024-09-16', n_per_day=300)
        data['prediction_time'] = pd.to_datetime(data['create_time'])
        data = build_full_features(data)
        data['dt'] = data['dt'].astype(str)
        self.config['train']['xgboost'].update(n_estimators=8, max_depth=2, early_stopping_rounds=3)
        self.config['train']['mlp'] = {'hidden_layers': [8], 'max_epochs': 2, 'patience': 2}
        args = SimpleNamespace(rounds=1, max_proposals_per_round=4, search_timeout_minutes=10, mock=True)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            directory = root / 'artifacts/feature_mining/S0000000'
            logs = root / 'experiments/feature_mining/S0000000'
            manifest = {'artifact_dir': str(directory), 'log_dir': str(logs), 'session_id': 'S0000000',
                        'window': self.config['data']['window'], 'title': '测试'}
            runner = study.Study(root, data, self.config, manifest, args, lambda *a, **k: None, lambda *a, **k: None)
            spec = experiment_spec.baseline_spec(['sp1_queue_count', 'hour'],
                [{'name': 'city_mean', 'kind': 'target_encoding', 'keys': ['city_code'],
                  'target': 'mean_bin', 'smoothing': 50}], self.config['train'])
            target, p1, _ = runner.fit(spec, runner.folds[0], 123, directory / 'first')
            # Test/selection labels cannot alter model training or probabilities.
            original = data['true_bin'].copy()
            data.loc[data.dt.ge(runner.folds[0]['validation_start']), 'true_bin'] = 5
            _, p2, _ = runner.fit(spec, runner.folds[0], 123, directory / 'second')
            np.testing.assert_array_equal(p1, p2)
            data['true_bin'] = original
            bundle = {'format': 'ets_model_study_bundle_v1', 'components': [
                {'directory': 'first', 'weight': 1, 'model_family': 'mlp'}],
                'training_population': 'unified', 'queue_rule': 'has_queue'}
            path = directory / 'bundle.json'
            study.write_json(path, bundle)
            loaded = study.predict_bundle(path, data.loc[target.index], self.config['mining'])
            np.testing.assert_allclose(p1, loaded.to_numpy(), rtol=1e-5, atol=1e-6)
            # XGB softmax artifacts also expose probabilities via margins.
            spec['train_config']['model_family'] = 'xgb'
            _, xp, _ = runner.fit(spec, runner.folds[0], 123, directory / 'xgb')
            np.testing.assert_allclose(xp.sum(axis=1), 1, atol=1e-5)
            bundle['components'] = [{'directory': 'first', 'weight': .25, 'model_family': 'mlp'},
                                    {'directory': 'xgb', 'weight': .75, 'model_family': 'xgb'}]
            study.write_json(path, bundle)
            np.testing.assert_allclose(study.predict_bundle(path, data.loc[target.index], self.config['mining']).to_numpy(),
                                       study.blend(p1, xp, .25), rtol=1e-5, atol=1e-6)

    def test_controller_repeats_only_finalists_and_locks_before_test(self):
        args = SimpleNamespace(rounds=8, max_proposals_per_round=3, search_timeout_minutes=10, mock=True)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {'artifact_dir': str(root / 'artifacts/feature_mining/S0000000'),
                        'log_dir': str(root / 'experiments/feature_mining/S0000000'),
                        'session_id': 'S0000000', 'window': self.config['data']['window'], 'title': '测试'}
            runner = study.Study(root, pd.DataFrame(index=[0]), self.config, manifest, args, lambda *a, **k: None, lambda *a, **k: None)
            calls = []
            def evaluate(name, spec, fold_index=0, seed=123, stage='screen'):
                key = name, fold_index, seed
                if key in runner.cache:
                    return runner.cache[key]
                calls.append((name, fold_index, seed, stage))
                neural = spec['train_config']['model_family'] == 'mlp'
                score = {'accuracy': .7 if neural else .6, 'severe_under': .02,
                         'long_wait_accuracy': .3, 'long_wait_severe_under': .2, 'legacy_accuracy': .65}
                result = {'id': f'E{len(calls):07d}', 'scores': score, 'index': np.array([0]),
                          'probabilities': np.ones((1, 6))/6,
                          'training': [{'best_epoch': 13, 'epochs_run': 23, 'mlp_config': DEFAULT_CONFIG}]}
                runner.cache[key] = result
                if stage == 'screen':
                    runner.screen[name] = result
                    runner.specs[name] = deepcopy(spec)
                return result
            def finalize(*args):
                saved = json.loads((runner.directory / 'model_study.json').read_text())
                self.assertEqual(saved['status'], 'selection_locked')
                self.assertIn('recommended', saved)
            fake_metrics = {study.STUDY_KEY: {'ALL': {'recall_ratio': .69, 'severe_under_ratio': .02}},
                            'answered_long_wait': {'ALL': {'recall_ratio': .3, 'severe_under_ratio': .2}},
                            study.SELECTION_KEY: {'ALL': {'recall_ratio': .65}}}
            with patch.object(runner, 'evaluate', side_effect=evaluate), patch.object(runner, 'finalize', side_effect=finalize), \
                 patch.object(study, 'measure', return_value=(fake_metrics, {})):
                runner.run()
            self.assertLessEqual(len(runner.screen), 24)
            for name in {c[0] for c in calls if c[3] == 'confirmation'}:
                self.assertTrue(all((name, fold, seed) in runner.cache for fold in range(2) for seed in study.SEEDS))
            self.assertTrue(any('learning_rate' in name for name in runner.specs))
            self.assertTrue(any('alpha' in name for name in runner.specs))
            self.assertTrue(any('hidden_layers' in name for name in runner.specs))
            self.assertTrue(any('batch_size' in name for name in runner.specs))
            self.assertFalse(any('增加epochs' in name for name in runner.specs))
            self.assertEqual(len([entry for entry in runner.report['confirmation'].values() if 'ensemble' in entry]), 3)


DEFAULT_CONFIG = {'max_epochs': 80}

if __name__ == '__main__':
    unittest.main()
