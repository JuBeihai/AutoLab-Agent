import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import shutil

import numpy as np
import pandas as pd

from skills import feature_builder, isotonic_h3, mining_data, mining_train
from skills.evaluation_policy import apply_training_population, CONTRACT, SELECTION_KEY, queue_mask, reference_baseline
from skills.evaluation_views import build_views, record_selection, selection_metrics

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT.parent / reference_baseline()['source_notebook']


class SixBucketBaselineTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            'order_id': list('abcd'), 'city_code': [100001] * 4, 'dt': ['2024-09-10'] * 4,
            'create_time': pd.to_datetime(['2024-09-10 10:00', '2024-09-10 10:01',
                                           '2024-09-10 10:02', '2024-09-10 10:03']),
            'match_time': pd.to_datetime(['2024-09-10 10:15', '2024-09-10 10:04', None, None]),
            'cancel_time': pd.to_datetime([None, None, '2024-09-10 10:04', '2024-09-10 10:20']),
            'has_queue': [True, False, True, True], 'base_pred_bin': [0, 4, 1, 1],
            'true_bin': [3, 0, 1, 3], 'predict_bin_44': [3, 0, 2, 2],
            'is_answer': [1, 1, 0, 0], 'true_ets': [900, 60, 120, 1020],
            'h3_res6': ['a'] * 4, 'h3_res7': ['b'] * 4, 'h3_res8': ['c'] * 4})

    def test_config_has_exact_notebook_features_and_time_windows(self):
        features = json.loads((ROOT / 'configs/features.json').read_text())
        columns = features['feature_set_27'] + features['feature_set_44_extra']
        self.assertEqual(len(columns), 43)
        self.assertNotIn('code_hash', columns)
        self.assertIn('is_peak', columns)
        self.assertEqual(features['num_bins'], 6)
        self.assertEqual(features['bin_edges_minutes'], [1, 5, 10, 20, 30])
        train = json.loads((ROOT / 'configs/train.json').read_text())
        self.assertEqual(train['queue_rule'], 'has_queue')
        self.assertEqual(train['training_method'], 'notebook_random')
        self.assertEqual(train['xgboost']['early_stopping_rounds'], 40)
        data = json.loads((ROOT / 'configs/data.json').read_text())
        for key, value in reference_baseline()['window'].items():
            self.assertEqual(data['window'][key], value)

    def test_boundaries_are_six_buckets_not_h3_resolution(self):
        for seconds, expected in [(0, 0), (60, 0), (61, 1), (300, 1), (301, 2),
                                  (600, 2), (601, 3), (1200, 3), (1201, 4),
                                  (1800, 4), (1801, 5), (99999, 5)]:
            self.assertEqual(feature_builder.ets_bin(seconds), expected)
            self.assertEqual(mining_data.notebook_bin(seconds), expected)
            self.assertEqual(isotonic_h3._remaining_to_bin(seconds / 60), expected)

    def test_routing_uses_flag_not_legacy_prediction(self):
        frame = self.frame()
        self.assertEqual(queue_mask(frame).tolist(), [True, False, True, True])
        cfg = json.loads((ROOT / 'configs/train.json').read_text())
        apply_training_population(cfg, 'queue_only')
        self.assertEqual(mining_train.fitting_mask(frame, True, cfg).tolist(), [True, False, False, False])
        frame.loc[1, 'has_queue'] = None
        self.assertFalse(queue_mask(frame).iloc[1])
        with self.assertRaisesRegex(ValueError, 'has_queue'):
            queue_mask(frame.drop(columns='has_queue'))

    def test_ordering_resets_nonqueue_and_selects_filtered_population(self):
        frame = self.frame()
        original = frame.copy(deep=True)
        ordered = isotonic_h3.apply_ordering(frame, spatial_col='h3_res7', pred_col='predict_bin_44')
        self.assertGreater(ordered[1], 0)
        views = build_views(frame)
        expected = frame.drop(index=2).copy()  # cancelled order overestimated before ordering
        expected['ordered'] = isotonic_h3.apply_ordering(expected, spatial_col='h3_res7', pred_col='predict_bin_44')
        expected.loc[~queue_mask(expected), 'ordered'] = 0
        self.assertEqual(selection_metrics(views)['ALL']['total'], 3)
        self.assertEqual(selection_metrics(views)['ALL']['recall'], int((expected.ordered == expected.true_bin).sum()))
        self.assertEqual(views['populations']['all']['h3_7']['metrics']['ALL']['total'], 4)
        pd.testing.assert_frame_equal(frame, original)

    def test_old_eleven_bucket_rows_cannot_enter_new_evaluation(self):
        frame = self.frame()
        frame.loc[0, 'true_bin'] = 10
        with self.assertRaisesRegex(ValueError, '6桶'):
            build_views(frame)
        old = {'config_diff': {'selection_metric': 'all_h3_7'},
               'metrics_by_city': {'all_h3_7': {'ALL': {'recall_ratio': .5}}}}
        self.assertEqual(record_selection(old)['ALL']['recall_ratio'], .5)

    def test_gate_uses_reference_floor_and_same_source_not_same_filtered_count(self):
        contract = {'schema': CONTRACT, 'source_sample_hash': 'fixed', 'mock': False,
                    'reference_baseline': reference_baseline()}
        def result(acc):
            return {'_evaluation_contract': deepcopy(contract), SELECTION_KEY: {
                city: {'total': 10, 'recall_ratio': acc, 'under_ratio': 0}
                for city in ('ALL', '100001', '100002', '100003')}}
        base, trial = result(.48), result(.49)
        self.assertFalse(mining_train.gate(base, trial)['pass'])
        trial[SELECTION_KEY]['ALL'].update(recall_ratio=.51, total=9, under_ratio=.9)
        self.assertTrue(mining_train.gate(base, trial)['pass'])
        trial['_evaluation_contract']['source_sample_hash'] = 'different'
        self.assertFalse(mining_train.gate(base, trial)['pass'])
        self.assertFalse(mining_train.gate(result(.63), result(.62))['pass'])
        self.assertFalse(mining_train.gate(result(.62), {'all_h3_7': {}})['pass'])
        for invalid in (None, float('nan'), float('inf'), -1, 2):
            self.assertFalse(mining_train.gate(result(.60), result(invalid))['pass'])


    def test_real_classifier_saves_six_classes_and_routes_nonqueue_to_zero(self):
        import xgboost as xgb
        cfg = json.loads((ROOT / 'configs/train.json').read_text())
        apply_training_population(cfg, 'queue_only')
        cfg['xgboost'].update(n_estimators=4, max_depth=2, early_stopping_rounds=2)
        frame = pd.DataFrame({'has_queue': [True] * 240 + [False] * 12,
                              'base_pred_bin': [0] * 240 + [4] * 12, 'is_answer': 1,
                              'true_bin': np.arange(252) % 6, 'signal': np.arange(252) % 19})
        with tempfile.TemporaryDirectory() as folder:
            models = mining_train.train_pair(frame, ['signal'], cfg, 6, Path(folder),
                                             {'experiment_id': 'six_bucket_test', 'window': {}})
            predicted = mining_train.predict(frame, models, ['signal'])
            self.assertTrue(predicted.predict_bin_44.between(0, 5).all())
            self.assertTrue(predicted.loc[~frame.has_queue, 'predict_bin_44'].eq(0).all())
            model_path = Path(folder) / 'ets_pred_model_queue_44.json'
            saved = json.loads(model_path.read_text())
            self.assertEqual(saved['learner']['learner_model_param']['num_class'], '6')
            restored = {'queue': xgb.Booster(model_file=str(model_path)), 'no_queue': None}
            np.testing.assert_array_equal(predicted.predict_bin_44, mining_train.predict(frame, restored, ['signal']).predict_bin_44)
            self.assertFalse((Path(folder) / 'ets_pred_model_no_queue_44.json').exists())

    def test_full_mock_session_trains_and_persists_new_contract_without_llm(self):
        from agent import feature_mining
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / 'configs', root / 'configs', ignore=shutil.ignore_patterns('*.sh'))
            # All writes are temporary test outputs; no production session or pointer changes.
            with patch.object(feature_mining, 'ROOT', root), patch.object(feature_mining, 'load_env'), \
                 patch.object(feature_mining, 'propose', side_effect=AssertionError('no LLM')), redirect_stdout(io.StringIO()):
                feature_mining.main(['--mock', '--smoke', '--no-agent', '--rounds', '0', '--training-population', 'queue_only'])
            manifest = json.loads((root / 'experiments/feature_mining/current.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['num_bins'], 6)
            self.assertEqual(manifest['selection_metric'], SELECTION_KEY)
            self.assertEqual(manifest['reference_baseline'], reference_baseline())
            record = json.loads((Path(manifest['artifact_dir']) / manifest['baseline_id'] / 'selection_result.json').read_text())
            self.assertEqual(len(record['config_diff']['experiment_spec']['base_features']), 43)
            self.assertEqual(record['evaluation_views']['num_bins'], 6)
            self.assertIn(SELECTION_KEY, record['metrics_by_city'])
            self.assertEqual(record['metrics_by_city']['_evaluation_contract']['schema'], CONTRACT)


if __name__ == '__main__':
    unittest.main()
