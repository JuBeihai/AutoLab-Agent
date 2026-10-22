import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from skills import mining_train
from skills import mining_data
from skills.data_loader import _SDS_FEATURE_FIELDS


class PersistenceTests(unittest.TestCase):
    def test_full_training_uses_all_eligible_rows_without_eval_set(self):
        progress = []
        import xgboost as xgb
        frame = pd.DataFrame({'base_pred_bin': [1] * 90 + [0] * 30,
                              'is_answer': [1] * 60 + [0] * 30 + [1] * 30,
                              'signal': np.arange(120), 'true_bin': np.arange(120) % 3})
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / 'configs/train.json').read_text())
        from skills.evaluation_policy import apply_training_population
        apply_training_population(cfg, 'queue_only')
        cfg['training_method'] = 'full_train'
        frame['has_queue'] = frame['base_pred_bin'] > 0
        cfg['xgboost'].update(n_estimators=4, max_depth=2, early_stopping_rounds=2)
        original_fit = xgb.XGBClassifier.fit
        def check_fit(model, **options):
            self.assertNotIn('eval_set', options)
            self.assertIsNone(model.get_params().get('early_stopping_rounds'))
            self.assertEqual(list(options['X'].index), list(range(60)))
            return original_fit(model, **options)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(mining_train, 'notebook_split', side_effect=AssertionError('no split')), \
             patch.object(xgb.XGBClassifier, 'fit', check_fit):
            models = mining_train.train_pair(frame, ['signal'], cfg, 3, Path(directory),
                                            {'experiment_id': 'full_train', 'window': {}}, on_progress=progress.append)
            self.assertEqual(models['queue'].num_boosted_rounds(), 4)
            self.assertIsNone(models['no_queue'])
            metadata = json.loads((Path(directory) / 'ets_pred_model_queue_44.json.meta.json').read_text())
            self.assertEqual(metadata['split_method'], 'full_train')
            self.assertEqual(metadata['fit_rows'], 60)
            self.assertEqual(metadata['early_stop_rows'], 0)
            self.assertIn('模型训练中：60 行', progress[0])
            self.assertIn('训练完成并保存：4 轮', progress[1])

    def test_notebook_classes_and_saved_booster_predictions(self):
        import xgboost as xgb
        df = pd.DataFrame({'dt': ['2024-08-01'] * 60 + ['2024-08-02'] * 60,
                           'base_pred_bin': [0, 1] * 60, 'is_answer': 1,
                           'x': np.arange(120) % 17, 'true_bin': [0, 1, 2] * 40})
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / 'configs/train.json').read_text())
        from skills.evaluation_policy import apply_training_population
        apply_training_population(cfg, 'queue_only')
        cfg['non_queue_strategy'] = 'model'
        df['has_queue'] = df['base_pred_bin'] > 0
        cfg['xgboost'].update(n_estimators=6, max_depth=2, early_stopping_rounds=2, eval_metric='mlogloss')
        with tempfile.TemporaryDirectory() as directory:
            pairs = mining_train.train_pair(df, ['x'], cfg, 3, Path(directory),
                      {'experiment_id': 'persistence_test', 'window': {}})
            expected = mining_train.predict(df, pairs, ['x'])['predict_bin_44']
            restored = {tag: xgb.Booster(model_file=str(Path(directory) / f'ets_pred_model_{tag}_44.json'))
                        for tag in ('queue', 'no_queue')}
            actual = mining_train.predict(df, restored, ['x'])['predict_bin_44']
            np.testing.assert_array_equal(expected, actual)
            self.assertTrue(actual.between(0, 10).all())

    def test_unregistered_json_field_reaches_numeric_catalog(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/mining.json').read_text())
        payloads = []
        for i in range(4):
            features = {key: 1 for key, _ in _SDS_FEATURE_FIELDS}
            features['newSupplyMeasurement'] = i + 1
            features['cancelTime'] = 99999
            if i == 3: features['testOnlyField'] = 10
            payloads.append(json.dumps(features))
        merged = pd.DataFrame({'order_id': ['1','2','3','4'], 'dt': ['2024-08-01'] * 3 + ['2024-08-02'],
                'city_code': [100001] * 4, 'create_time': ['2024-08-01 10:00:00'] * 4,
                'match_time': ['2024-08-01 10:02:00'] * 4, 'cancel_time': [None] * 4,
                'prediction_time': pd.to_datetime(['2024-08-01 10:00:00'] * 4),
                'start_lat': 39.9, 'start_lng': 116.4, 'tick_service_product_code': 'PRODUCT_A',
                'feature_map': payloads, 'json_param': ['{"originErtResult":60,"ertResult":60}'] * 4})
        full, lineage = mining_data.decode_features(merged, {'product_code_hash': {'PRODUCT_A': 2}}, cfg, '2024-08-01')
        profile = mining_data.profile_frame(full, '2024-08-01', '2024-08-01', lineage, cfg)
        self.assertIn('sds__newSupplyMeasurement', profile['catalog'])
        self.assertNotIn('sds__cancelTime', full)
        self.assertNotIn('sds__testOnlyField', full)
