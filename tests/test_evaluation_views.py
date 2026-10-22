import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from skills.evaluation_views import build_views, city_metrics, selection_metrics, record_selection
from skills.mining_train import metrics, gate
from skills.evaluation_policy import CONTRACT, SELECTION_KEY, reference_baseline


NOTEBOOK = Path(__file__).resolve().parents[2] / 'excluded_reference.ipynb'


class EvaluationViewsTests(unittest.TestCase):
    def test_gate_uses_h3_7_only_and_never_falls_back_to_raw(self):
        def rows(acc, under=0):
            return {city: {'total': 10, 'recall_ratio': acc, 'under_ratio': under}
                    for city in ('ALL', '100001', '100002', '100003')}
        contract = {'schema': CONTRACT, 'source_sample_hash': 'same', 'mock': True, 'reference_baseline': reference_baseline()}
        base = {'xgb_44': rows(.9), SELECTION_KEY: rows(.5), '_evaluation_contract': contract}
        trial = {'xgb_44': rows(.1), SELECTION_KEY: rows(.6, .4), '_evaluation_contract': contract}
        self.assertTrue(gate(base, trial)['pass'])
        self.assertFalse(gate(trial, base)['pass'])
        self.assertFalse(gate(base, {'xgb_44': rows(1)})['pass'])
        for invalid in (None, float('nan'), float('inf'), -1, 2):
            self.assertFalse(gate(base, {SELECTION_KEY: rows(invalid), '_evaluation_contract': contract})['pass'])
        for unavailable in ({}, {'populations': {'all': {'h3_7': {'unavailable_reason': 'missing grid'}}}}):
            with self.assertRaisesRegex(ValueError, 'H3-7'):
                selection_metrics(unavailable)

    def test_history_keeps_its_original_selection_policy(self):
        record = {'metrics_by_city': {'xgb_44': {'ALL': {'recall_ratio': .9}},
                                     'all_h3_7': {'ALL': {'recall_ratio': .6}}}}
        self.assertEqual(record_selection(record)['ALL']['recall_ratio'], .9)
        record['config_diff'] = {'selection_metric': 'all_h3_7'}
        self.assertEqual(record_selection(record)['ALL']['recall_ratio'], .6)
        del record['metrics_by_city']['all_h3_7']
        self.assertEqual(record_selection(record), {})

    def frame(self):
        return pd.DataFrame({
            'city_code': [100001] * 7 + [100002], 'dt': ['2024-08-31'] * 8,
            'create_time': ['2024-08-31 10:00:00', '2024-08-31 10:01:00',
                            '2024-08-31 10:01:00', '2024-08-31 10:02:00',
                            '2024-08-31 10:04:00', '2024-08-31 11:00:00',
                            '2024-08-31 11:01:00', '2024-08-31 10:01:00'],
            'match_time': [None, '2024-08-31 10:04:00', '2024-08-31 10:04:00',
                           '2024-08-31 10:08:00', None, None, None, None],
            'cancel_time': ['2024-08-31 10:10:00', None, None, None,
                            '2024-08-31 10:09:00', None, None, None],
            'base_pred_bin': [1, 0, 1, 1, 1, 1, 0, 1],
            'has_queue': [True, False, True, True, True, True, False, True],
            'predict_bin_44': [5, 0, 2, 0, 1, 5, 0, 1],
            'is_answer': [0, 1, 1, 1, 0, 1, 1, 1],
            'true_bin': [0, 0, 3, 1, 1, 5, 0, 1], 'true_ets': [60] * 8,
            'h3_res6': ['a'] * 8, 'h3_res7': ['b'] * 5 + ['c'] * 3,
            'h3_res8': ['d', 'd', 'd', 'e', 'e', 'f', 'f', 'd']})


    def test_raw_views_match_existing_metrics_and_gate_inputs(self):
        frame = self.frame()
        expected = metrics(frame)
        result = build_views(frame)['populations']
        for population, key in [('all', 'xgb_44'), ('answer_only', 'notebook_answer_only'),
                                ('answer_cancel', 'notebook_answer_cancel')]:
            for city, row in result[population]['raw']['metrics'].items():
                self.assertEqual(row, expected[key][city])

    def test_missing_grid_is_unavailable_not_fake_zero_prediction(self):
        frame = self.frame()
        frame.loc[0, 'h3_res6'] = None
        result = build_views(frame)['populations']['all']
        self.assertIn('unavailable_reason', result['h3_6'])
        self.assertIn('metrics', result['h3_7'])
        result = build_views(frame.drop(columns='match_time'))['populations']['all']
        self.assertIn('缺少保序输入字段', result['h3_7']['unavailable_reason'])

    def test_empty_populations_are_reported_as_zero_samples(self):
        frame = self.frame()
        frame['is_answer'] = 0
        result = build_views(frame)['populations']['answer_only']
        for view in result.values():
            self.assertEqual(view['metrics']['ALL']['total'], 0)
