import copy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from skills import feature_program as fp
from skills import mining_data as md
from skills import mining_train as mt
from agent.feature_mining import validate_proposal, windows
from skills.evaluation_policy import CONTRACT, SELECTION_KEY, reference_baseline

CFG = json.loads((Path(__file__).resolve().parents[1] / 'configs/mining.json').read_text())
CATALOG = {'x': {'role': 'numeric'}, 'y': {'role': 'numeric'},
           'city_code': {'role': 'numeric', 'groupable': True}}


class FeatureProgramTests(unittest.TestCase):
    def test_dynamic_formula_not_predefined(self):
        recipes = [{'name': 'invented_pressure', 'kind': 'expression',
                    'expression': {'op': 'div', 'args': [{'col': 'x'}, {'col': 'y'}]}}]
        df = pd.DataFrame({'x': [6., 2., np.nan], 'y': [3., 0., 1.]})
        out = fp.materialize(df, recipes, CATALOG, CFG, '2024-08-15')
        self.assertEqual(out.invented_pressure.iloc[0], 2)
        self.assertTrue(out.invented_pressure.iloc[1:].isna().all())

    def test_code_and_current_labels_rejected(self):
        for expression in ({'col': 'true_ets'}, {'col': 'cancel_time'},
                           {'op': '__import__', 'args': []}, {'const': float('inf')}):
            with self.assertRaises(ValueError):
                fp.validate_recipes([{'name': 'bad', 'kind': 'expression', 'expression': expression}], CATALOG, CFG)

    def test_hyperparameter_only_and_mixed_rejected(self):
        for p in ({'xgb_params': {'max_depth': 8}},
                  {'id': 'x', 'hypothesis': 'x', 'evidence': 'x', 'feature_recipes': [], 'xgb_params': {'max_depth': 8}}):
            with self.assertRaises(ValueError): validate_proposal(p, CATALOG, [], CFG)

    def test_historical_maturity_window_and_holdout_isolation(self):
        df = pd.DataFrame({'city_code': [1] * 5,
            'create_time': pd.to_datetime(['2024-08-01', '2024-08-02', '2024-08-03', '2024-08-04', '2024-08-05']),
            'match_time': pd.to_datetime(['2024-08-01 01:00', '2024-08-05', '2024-08-03 01:00', '2024-08-04 01:00', '2024-08-05 01:00'], format='mixed'),
            'cancel_time': pd.NaT, 'prediction_time': pd.to_datetime(['2024-08-01', '2024-08-02', '2024-08-03', '2024-08-04', '2024-08-05']),
            'true_ets': [10., 999., 30., 888., 777.]})
        recipe = {'name': 'hist', 'kind': 'history', 'keys': ['city_code'], 'value': 'true_ets', 'agg': 'mean', 'days': 7}
        a = fp.historical(recipe, df, df, '2024-08-03', min_count=1)
        self.assertTrue(np.isnan(a.iloc[0]))
        self.assertEqual(a.iloc[2], 10.)  # Aug 2's outcome not yet available, excludes current day
        self.assertEqual(a.iloc[3], 20.)
        self.assertEqual(a.iloc[4], 20.)  # cannot use validation/holdout targets
        changed = df.copy()
        changed.loc[3:, 'true_ets'] = 123456.
        b = fp.historical(recipe, changed, changed, '2024-08-03', min_count=1)
        np.testing.assert_allclose(a, b, equal_nan=True)

    def test_history_never_uses_current_day_and_missing_groups_remain_nan(self):
        df = pd.DataFrame({'city_code': [1, 1, 2], 'x': [4., 7., 8.],
            'create_time': pd.to_datetime(['2024-08-01 01:00', '2024-08-01 02:00', '2024-08-02'], format='mixed'),
            'prediction_time': pd.to_datetime(['2024-08-01 01:00', '2024-08-01 02:00', '2024-08-02'], format='mixed')})
        r = {'keys': ['city_code'], 'value': 'x', 'agg': 'mean', 'days': 1}
        self.assertTrue(fp.historical(r, df, df, '2024-08-01', 1).isna().all())

    def test_group_keys_and_windows_validated(self):
        r = {'name': 'h', 'kind': 'history', 'keys': ['order_id'], 'value': 'true_ets', 'agg': 'mean', 'days': 7}
        with self.assertRaises(ValueError): fp.validate_recipes([r], CATALOG, CFG)
        r['keys'] = ['city_code']; r['days'] = 999
        with self.assertRaises(ValueError): fp.validate_recipes([r], CATALOG, CFG)

    def test_duplicate_and_sparse_features_rejected(self):
        with self.assertRaises(ValueError):
            fp.validate_recipes([{'name': 'x', 'kind': 'expression', 'expression': {'col': 'y'}}], CATALOG, CFG)
        with self.assertRaises(ValueError): fp.quality_report(pd.DataFrame({'a': [np.nan, np.nan]}), ['a'])
        with self.assertRaises(ValueError): fp.quality_report(pd.DataFrame({'a': [1, 1]}), ['a'])


class DataTests(unittest.TestCase):
    def test_discover_numeric_json_and_arrays_without_identifiers(self):
        result = md.numeric_payload(json.dumps({'newSupplySignal': 3, 'orderId': '1234', 'userPin': '999',
                   'cancelTime': 123, 'vehicles': [{'vid': '123', 'remainingServiceTime': 12}, {'remainingServiceTime': 4}]}))
        self.assertEqual(result['newSupplySignal'], 3)
        self.assertEqual(result['vehicles_remainingServiceTime_mean'], 8)
        self.assertNotIn('orderId', result)
        self.assertNotIn('cancelTime', result)
        self.assertNotIn('userPin', result)
        self.assertNotIn('vehicles_vid_mean', result)

    def test_notebook_joins_keep_duplicates_and_late_logs(self):
        orders = pd.DataFrame({'dt': ['2024-08-01'], 'order_id': ['a'], 'city_code': [1], 'create_time': ['2024-08-01 10:00:00'],
                               'match_time': ['2024-08-01 10:05:00'], 'cancel_time': [None]})
        sds = pd.DataFrame({'dt': ['2024-08-01'] * 2, 'order_id': ['a', 'a'], 'city_code': [1, 1],
            'event_time': ['2024-08-01 10:00:10', '2024-08-01 10:02:00'], 'feature_map': ['{"x":1}', '{"x":2}']})
        queue = pd.DataFrame({'dt': ['2024-08-01'] * 2, 'order_id': ['a', 'a'], 'event_time': ['2024-08-01 10:00:09', '2024-08-01 10:00:11'],
                              'json_param': ['{"x":3}', '{"x":999}']})
        out = md.merge_notebook(orders, sds, queue, CFG)
        self.assertEqual(len(out), 4)
        self.assertEqual(set(out.feature_map), {'{"x":1}', '{"x":2}'})
        self.assertEqual(set(out.json_param), {'{"x":3}', '{"x":999}'})

    def test_profile_does_not_read_holdout_labels_or_values(self):
        full = pd.DataFrame({'dt': ['2024-08-01','2024-08-01','2024-08-02'],
                              'new_feature': [1.,2.,999.], 'true_bin': [0,1,10]})
        lineage = {'new_feature': {'source': 'sds'}}
        before = md.profile_frame(full, '2024-08-01', '2024-08-01', lineage, CFG)
        full.loc[2, ['new_feature', 'true_bin']] = [-999, -999]
        after = md.profile_frame(full, '2024-08-01', '2024-08-01', lineage, CFG)
        self.assertEqual(before, after)

    def test_epoch_timestamps(self):
        s = md.timestamp(pd.Series(['1725148800000', '1725148800', '2024-09-01 00:00:00']))
        self.assertEqual(s.nunique(), 1)

    def test_only_train_and_test_windows_do_not_overlap(self):
        w = windows({'window': {'train_start':'2024-08-05','train_end':'2024-08-17',
                 'test_start':'2024-08-18','test_end':'2024-08-19','end_date':'2024-08-19'}})
        self.assertEqual(w['train_end'], '2024-08-17')
        self.assertEqual(set(w), {'train_start', 'train_end', 'test_start', 'test_end'})
        self.assertLess(w['train_end'], w['test_start'])

    def test_gate_fails_on_missing_cities(self):
        self.assertFalse(mt.gate({SELECTION_KEY: {}}, {SELECTION_KEY: {}})['pass'])

    def test_gate_accepts_equal_or_better_overall_acc_despite_slice_regressions(self):
        base = {SELECTION_KEY: {
            city: {'total': 100 if city != 'ALL' else 300, 'recall_ratio': .5,
                   'under_ratio': .2, 'severe_under_ratio': .1}
            for city in ('100001', '100002', '100003', 'ALL', 'LONG_WAIT')}}
        base['_evaluation_contract'] = {'schema': CONTRACT, 'source_sample_hash': 'same', 'mock': True, 'reference_baseline': reference_baseline()}
        for overall_acc in (.5, .6):
            with self.subTest(overall_acc=overall_acc):
                trial = copy.deepcopy(base)
                trial[SELECTION_KEY]['100001'].update(recall_ratio=.3, under_ratio=.5)
                trial[SELECTION_KEY]['100002']['recall_ratio'] = .7
                trial[SELECTION_KEY]['100003']['recall_ratio'] = overall_acc * 3 - 1
                trial[SELECTION_KEY]['ALL']['recall_ratio'] = overall_acc
                trial[SELECTION_KEY]['LONG_WAIT']['severe_under_ratio'] = .4
                self.assertEqual(mt.gate(base, trial), {'pass': True, 'reasons': []})

    def test_gate_does_not_require_long_wait_slice(self):
        base = {SELECTION_KEY: {city: {'total': 10, 'recall_ratio': .5}
                          for city in ('100001', '100002', '100003', 'ALL')}}
        base['_evaluation_contract'] = {'schema': CONTRACT, 'source_sample_hash': 'same', 'mock': True, 'reference_baseline': reference_baseline()}
        for long_wait in (None, {'total': 0, 'severe_under_ratio': 0}):
            with self.subTest(long_wait=long_wait):
                trial = copy.deepcopy(base)
                if long_wait is not None:
                    trial[SELECTION_KEY]['LONG_WAIT'] = long_wait
                self.assertTrue(mt.gate(base, trial)['pass'])

    def test_gate_still_rejects_overall_acc_drop(self):
        base = {SELECTION_KEY: {city: {'total': 10, 'recall_ratio': .5}
                          for city in ('100001', '100002', '100003', 'ALL')}}
        base['_evaluation_contract'] = {'schema': CONTRACT, 'source_sample_hash': 'same', 'mock': True, 'reference_baseline': reference_baseline()}
        trial = copy.deepcopy(base)
        trial[SELECTION_KEY]['ALL']['recall_ratio'] = .499
        self.assertEqual(mt.gate(base, trial), {'pass': False, 'reasons': ['6桶 H3-7 保序后整体召回低于当前对照']})

    def test_gate_still_rejects_empty_or_mismatched_populations(self):
        base = {SELECTION_KEY: {city: {'total': 10, 'recall_ratio': .5}
                          for city in ('100001', '100002', '100003', 'ALL')}}
        base['_evaluation_contract'] = {'schema': CONTRACT, 'source_sample_hash': 'same', 'mock': True, 'reference_baseline': reference_baseline()}
        for city in base[SELECTION_KEY]:
            for count in (0,):
                with self.subTest(city=city, count=count):
                    trial = copy.deepcopy(base)
                    trial[SELECTION_KEY][city]['total'] = count
                    self.assertEqual(mt.gate(base, trial),
                                     {'pass': False, 'reasons': [f'{city} 缺失或评估样本为空']})


if __name__ == '__main__': unittest.main()

class RollingTests(unittest.TestCase):
    def test_excludes_self_equal_timestamps_and_future(self):
        frame = pd.DataFrame({'city_code': [1, 1, 1, 1, 1], 'x': [2., 4., 900., 999., 8.],
            'prediction_time': pd.to_datetime(['2024-08-01 10:00','2024-08-01 10:01','2024-08-01 10:02','2024-08-01 10:02','2024-08-01 10:10'])})
        r = {'name': 'recent_x', 'kind': 'rolling', 'keys': ['city_code'], 'value': 'x', 'agg': 'mean', 'minutes': 5}
        out = fp.rolling(r, frame, frame, min_count=1)
        np.testing.assert_allclose(out, [np.nan, 2., 3., 3., np.nan], equal_nan=True)
        altered = frame.copy(); altered.loc[2:, 'x'] = -1000.
        before_future = fp.rolling(r, altered, altered, min_count=1)
        np.testing.assert_allclose(out.iloc[:4], before_future.iloc[:4], equal_nan=True)

    def test_rolling_labels_are_forbidden(self):
        r = {'name': 'bad', 'kind': 'rolling', 'keys': ['city_code'], 'value': 'true_ets', 'agg': 'mean', 'minutes': 5}
        with self.assertRaises(ValueError): fp.validate_recipes([r], CATALOG, CFG)

    def test_boolean_profile(self):
        frame = pd.DataFrame({'dt': ['2024-08-01'] * 3, 'has_queue': [True, False, True], 'true_bin': [2, 0, 3]})
        profile = md.profile_frame(frame, '2024-08-01', '2024-08-01', {}, CFG)
        self.assertEqual(profile['catalog']['has_queue']['quantiles']['0.5'], 1.)
