from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from agent import experiment_spec, search_memory
from agent.research_controller import ResearchController
from skills import data_inspection, feature_program, mining_diagnostics, mining_train


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        days = ['2024-08-01', '2024-08-01', '2024-08-02', '2024-08-03',
                '2024-08-03', '2024-08-04', '2024-08-04', '2024-08-05']
        self.frame = pd.DataFrame({'prediction_time': pd.to_datetime(days),
            'match_time': pd.to_datetime([day + ' 12:00' for day in days]),
            'cancel_time': pd.NaT, 'true_bin': [0, 2, 4, 6, 8, 999, 999, 999],
            'city': ['A', 'A', 'A', 'B', 'B', 'A', 'Z', 'B'],
            'hour': [1, 2, 1, 2, 2, 1, 2, 2], 'value': [1., np.nan, 3., 5., np.nan, 10000., np.nan, 20000.]})
        self.frame.loc[4, 'match_time'] = pd.Timestamp('2024-08-05 12:00')
        self.catalog = {'city': {'role': 'group', 'groupable': True},
                        'hour': {'role': 'numeric', 'groupable': True}, 'value': {'role': 'numeric'}}
        self.cfg = {'max_features_per_proposal': 4, 'max_total_features': 24}
        self.fitting = self.frame.index[:5]

    def materialize(self, recipes, frame=None):
        state = {}
        result = feature_program.materialize(self.frame if frame is None else frame, recipes,
            self.catalog, self.cfg, '2024-08-03', fitting_index=self.fitting, preprocessing_state=state)
        json.dumps(state, allow_nan=False)
        return result, state

    def test_median_and_indicator_use_only_actual_training_rows(self):
        recipes = [{'name': 'filled', 'kind': 'impute', 'column': 'value', 'strategy': 'median', 'value': None},
                   {'name': 'missing', 'kind': 'missing_indicator', 'column': 'value'}]
        original = self.frame.copy(deep=True)
        result, state = self.materialize(recipes)
        self.assertEqual(result.loc[6, 'filled'], 3.)
        self.assertEqual(state['filled']['fill_value'], 3.)
        self.assertEqual(result.loc[1, 'missing'], 1.)
        pd.testing.assert_frame_equal(original, self.frame)
        with self.assertRaises(ValueError):
            feature_program.materialize(self.frame, recipes, self.catalog, self.cfg, '2024-08-03')

    def test_constant_imputation_and_all_missing_median(self):
        recipe = {'name': 'filled', 'kind': 'impute', 'column': 'value', 'strategy': 'constant', 'value': 0}
        result, _ = self.materialize([recipe])
        self.assertEqual(result.loc[1, 'filled'], 0.)
        frame = self.frame.copy()
        frame.loc[self.fitting, 'value'] = np.nan
        with self.assertRaises(ValueError):
            self.materialize([{**recipe, 'strategy': 'median', 'value': None}], frame)

    def test_joint_one_hot_unknowns_and_chained_numeric_interaction(self):
        recipe = {'name': 'cross', 'kind': 'one_hot', 'keys': ['city', 'hour'], 'values': ['B', 2]}
        interaction = {'name': 'interaction', 'kind': 'expression', 'expression': {
            'op': 'mul', 'args': [{'col': 'cross'}, {'col': 'value'}]}}
        result, _ = self.materialize([recipe, interaction])
        self.assertEqual(result['cross'].tolist(), [0., 0., 0., 1., 1., 0., 0., 1.])
        self.assertEqual(result.loc[3, 'interaction'], 5.)
        with self.assertRaises(ValueError):
            self.materialize([{**recipe, 'values': ['B', 1]}])
        with self.assertRaises(ValueError):
            self.materialize([{**recipe, 'values': ['Z', 2]}])

    def test_target_encoding_excludes_same_day_unmatured_and_test_labels(self):
        recipe = {'name': 'encoded', 'kind': 'target_encoding', 'keys': ['city'], 'target': 'mean_bin', 'smoothing': 2}
        result, state = self.materialize([recipe])
        self.assertTrue(result.loc[:1, 'encoded'].isna().all())
        self.assertEqual(result.loc[2, 'encoded'], 1.)
        self.assertEqual(result.loc[3, 'encoded'], 2.)
        self.assertEqual(result.loc[4, 'encoded'], 2.)
        self.assertAlmostEqual(result.loc[5, 'encoded'], 2.4, places=5)
        self.assertEqual(result.loc[6, 'encoded'], 3.)
        self.assertEqual(result.loc[7, 'encoded'], 4.)
        changed = self.frame.copy()
        changed.loc[4:, 'true_bin'] = -500
        new, _ = self.materialize([recipe], changed)
        pd.testing.assert_series_equal(result['encoded'], new['encoded'])
        self.assertEqual(state['encoded']['daily_pools'][-1]['eligible_rows'], 4)
        self.assertTrue(state['encoded']['mapping'])

    def test_target_encoding_uses_cancel_maturity_and_class_probability(self):
        recipe = {'name': 'encoded', 'kind': 'target_encoding', 'keys': ['city'], 'target': 2, 'smoothing': 2}
        frame = self.frame.copy()
        frame.loc[1, 'match_time'] = pd.NaT
        frame.loc[1, 'cancel_time'] = pd.Timestamp('2024-08-01 22:00')
        result, _ = self.materialize([recipe], frame)
        self.assertEqual(result.loc[2, 'encoded'], .5)
        self.assertTrue(result['encoded'].dropna().between(0, 1).all())

    def test_own_label_never_changes_own_day_encoding(self):
        recipe = {'name': 'encoded', 'kind': 'target_encoding', 'keys': ['city'], 'target': 'mean_bin', 'smoothing': 20}
        original, _ = self.materialize([recipe])
        frame = self.frame.copy()
        frame.loc[2, 'true_bin'] = 10
        changed, _ = self.materialize([recipe], frame)
        self.assertEqual(original.loc[2, 'encoded'], changed.loc[2, 'encoded'])
        self.assertNotEqual(original.loc[3, 'encoded'], changed.loc[3, 'encoded'])

    def test_supervised_derivatives_cannot_leak_through_global_median_fill(self):
        encoded = {'name': 'encoded', 'kind': 'target_encoding', 'keys': ['city'], 'target': 'mean_bin', 'smoothing': 20}
        derived = {'name': 'scaled', 'kind': 'expression', 'expression': {
            'op': 'mul', 'args': [{'col': 'encoded'}, {'const': 2}]}}
        fill = {'name': 'filled', 'kind': 'impute', 'column': 'scaled', 'strategy': 'median', 'value': None}
        with self.assertRaisesRegex(ValueError, '间接泄漏'):
            self.materialize([encoded, derived, fill])
        result, _ = self.materialize([encoded, derived, {**fill, 'strategy': 'constant', 'value': 0}])
        self.assertEqual(result.loc[0, 'filled'], 0.)

    def test_encoding_schema_rejects_labels_infinite_values_and_unbounded_keys(self):
        recipe = {'name': 'encoded', 'kind': 'target_encoding', 'keys': ['city'], 'target': 'mean_bin', 'smoothing': 2}
        for change in ({'keys': ['true_bin']}, {'keys': ['city'] * 4}, {'smoothing': float('inf')},
                       {'smoothing': 0}, {'target': True}, {'target': 11}, {'sql': 'select 1'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.materialize([{**recipe, **change}])

    def test_recipe_signature_handles_preprocessing_and_joint_key_order(self):
        first = {'name': 'encoded', 'kind': 'one_hot', 'keys': ['city', 'hour'], 'values': ['B', 2]}
        swapped = {**first, 'name': 'renamed', 'keys': ['hour', 'city'], 'values': [2, 'B']}
        self.assertEqual(search_memory.recipe_signature([first]), search_memory.recipe_signature([swapped]))
        self.assertNotEqual(search_memory.recipe_signature([first]),
                            search_memory.recipe_signature([{**first, 'values': [2, 'B']}]))
        impute = {'name': 'filled', 'kind': 'impute', 'column': 'value', 'strategy': 'median', 'value': None}
        indicator = {'name': 'missing', 'kind': 'missing_indicator', 'column': 'filled'}
        renamed = [{**impute, 'name': 'other'}, {**indicator, 'column': 'other'}]
        self.assertEqual(search_memory.recipe_signature([impute, indicator]), search_memory.recipe_signature(renamed))

    def test_encoded_inputs_train_and_predict_with_real_xgb_without_stopping_split(self):
        dates = pd.date_range('2024-08-01', periods=6).repeat(22)
        frame = pd.DataFrame({'prediction_time': dates, 'match_time': dates + pd.Timedelta(hours=1),
                              'cancel_time': pd.NaT, 'true_bin': list(range(11)) * 12,
                              'city': ['A', 'B'] * 66, 'base_pred_bin': 1, 'has_queue': True, 'is_answer': 1})
        recipes = [{'name': 'encoded', 'kind': 'target_encoding', 'keys': ['city'], 'target': 'mean_bin', 'smoothing': 20},
                   {'name': 'city_a', 'kind': 'one_hot', 'keys': ['city'], 'values': ['A']}]
        expanded = feature_program.materialize(frame, recipes, self.catalog, self.cfg, '2024-08-04',
                                               fitting_index=frame.index[:88])
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/train.json').read_text())
        from skills.evaluation_policy import apply_training_population
        apply_training_population(config, 'queue_only')
        config['training_method'] = 'full_train'
        config['xgboost'].update(n_estimators=3, max_depth=2, n_jobs=1)
        with tempfile.TemporaryDirectory() as directory:
            models = mining_train.train_pair(expanded.iloc[:88], ['encoded', 'city_a'], config, 11,
                Path(directory), {'experiment_id': 'E0000001', 'window': {'train_end': '2024-08-04'}})
            predictions = mining_train.predict(expanded.iloc[88:], models, ['encoded', 'city_a'])
            self.assertEqual(len(predictions), 44)
            self.assertTrue(predictions['predict_bin_44'].between(0, 10).all())
            self.assertEqual(models['queue'].num_boosted_rounds(), 3)
            self.assertIsNone(models['no_queue'])


class DataInspectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'reference_tables': ['app.approved'], 'inspection_rows': 20}
        self.audit = {'sources': [{'table': 'app.approved', 'status': 'schema_read', 'fields': [
            {'name': 'dt', 'type': 'string'}, {'name': 'supply', 'type': 'double'},
            {'name': 'order_id', 'type': 'bigint'}, {'name': 'phone', 'type': 'bigint'},
            {'name': 'true_bin', 'type': 'int'}, {'name': 'blob', 'type': 'string'}]}]}
        self.catalog = {'supply': {'role': 'numeric'}}
        self.window = {'train_start': '2024-08-01', 'train_end': '2024-08-02',
                       'test_start': '2024-08-03', 'test_end': '2024-08-04'}
        self.frame = pd.DataFrame({'dt': ['2024-08-01', '2024-08-02', '2024-08-03'], 'supply': [1., 3., 999.]})
        self.request = {'kind': 'table', 'table': 'app.approved', 'columns': ['supply']}

    def test_only_allowlisted_numeric_non_sensitive_columns(self):
        self.assertEqual(data_inspection.capabilities(self.audit, self.cfg)['tables'][0]['numeric_columns'], ['supply'])
        for change in ({'columns': ['order_id']}, {'columns': ['true_bin']}, {'columns': ['phone']},
                       {'columns': ['blob']}, {'table': 'app.unapproved'}, {'table': 'app.approved; DROP TABLE t'},
                       {'columns': ['supply`']}, {'sql': 'SELECT * FROM app.approved'}, {'start': '2020-01-01'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                data_inspection.validate({**self.request, **change}, self.catalog, self.audit, self.cfg)
        for name in ('orderId', 'phoneNumber', 'deviceId', 'email', 'latitude', 'userAccount'):
            self.assertFalse(data_inspection.inspectable({'name': name, 'type': 'bigint'}), name)

    def test_fields_are_training_only_and_return_no_raw_rows(self):
        with patch.object(data_inspection.data_loader, 'build_spark_session', side_effect=AssertionError('no database')):
            result = data_inspection.execute({'kind': 'fields', 'columns': ['supply']}, self.frame,
                self.catalog, self.audit, self.cfg, self.window, mock=True)
        self.assertEqual(result['fields']['supply']['rows'], 2)
        self.assertEqual(result['fields']['supply']['quantiles']['1.0'], 3.)
        self.assertNotIn('999', json.dumps(result))
        self.assertNotIn('records', result)

    def test_real_table_request_is_bounded_readonly_and_closes_session(self):
        spark = Mock()
        spark.sql.return_value.toPandas.return_value = self.frame[['supply']].iloc[:2]
        with patch.object(data_inspection.data_loader, 'build_spark_session', return_value=spark):
            result = data_inspection.execute(self.request, self.frame, self.catalog, self.audit, self.cfg, self.window)
        query = spark.sql.call_args.args[0]
        self.assertEqual(query, "SELECT `supply` FROM app.approved WHERE dt >= '2024-08-01' AND dt <= '2024-08-02' LIMIT 20")
        self.assertEqual(result['rows_read'], 2)
        self.assertEqual(result['scope'], 'bounded_prefix_not_full_population')
        spark.stop.assert_called_once()

    def test_mock_and_missing_partition_never_connect(self):
        with patch.object(data_inspection.data_loader, 'build_spark_session', side_effect=AssertionError('no database')):
            with self.assertRaises(ValueError):
                data_inspection.execute(self.request, self.frame, self.catalog, self.audit, self.cfg, self.window, mock=True)
            audit = deepcopy(self.audit)
            audit['sources'][0]['fields'] = audit['sources'][0]['fields'][1:]
            with self.assertRaises(ValueError):
                data_inspection.execute(self.request, self.frame, self.catalog, audit, self.cfg, self.window)

    def test_database_errors_do_not_expose_details_and_always_stop(self):
        spark = Mock()
        spark.sql.side_effect = RuntimeError('credential=secret')
        with patch.object(data_inspection.data_loader, 'build_spark_session', return_value=spark):
            with self.assertRaisesRegex(RuntimeError, '^只读表探查未完成：RuntimeError$'):
                data_inspection.execute(self.request, self.frame, self.catalog, self.audit, self.cfg, self.window)
        spark.stop.assert_called_once()

    def test_timeout_cancels_only_own_job_group(self):
        spark = Mock()
        spark.sql.return_value.toPandas.return_value = self.frame[['supply']]
        def timer_factory(seconds, cancel):
            timer = Mock()
            timer.start.side_effect = cancel
            return timer
        with patch.object(data_inspection.data_loader, 'build_spark_session', return_value=spark), \
             patch.object(data_inspection.threading, 'Timer', side_effect=timer_factory):
            with self.assertRaisesRegex(RuntimeError, 'TimeoutError'):
                data_inspection.execute(self.request, self.frame, self.catalog, self.audit, self.cfg, self.window)
        group = spark.sparkContext.setJobGroup.call_args.args[0]
        spark.sparkContext.cancelJobGroup.assert_called_once_with(group)
        spark.stop.assert_called_once()


class SelectionAndPlanningTests(unittest.TestCase):
    def test_inherited_encoding_or_empty_training_change_does_not_bypass_diversity(self):
        parent = {'base_features': ['value'], 'train_config': {'xgboost': {'max_depth': 6}}, 'extensions': [],
                  'recipes': [{'name': 'encoded', 'kind': 'target_encoding', 'keys': ['city'], 'target': 'mean_bin', 'smoothing': 20}]}
        candidate = deepcopy(parent)
        candidate['recipes'].append({'name': 'extra', 'kind': 'expression', 'expression': {'col': 'value'}})
        self.assertFalse(experiment_spec.has_non_additive_change(parent, candidate))
        candidate['train_config']['xgboost']['max_depth'] = 5
        self.assertTrue(experiment_spec.has_non_additive_change(parent, candidate))
        candidate = deepcopy(parent)
        candidate['recipes'][0]['smoothing'] = 40
        self.assertTrue(experiment_spec.has_non_additive_change(parent, candidate))

    def test_all_original_inputs_ranked_and_correlations_do_not_mutate_training(self):
        class Model:
            def get_score(self, importance_type):
                return {'signal': 8}
            def predict(self, matrix):
                return matrix.get_data().toarray()[:, 0]
        frame = pd.DataFrame({'signal': [0., 1.] * 16, 'duplicate': [0., 1.] * 16, 'constant': 2.,
                              'true_bin': [0, 1] * 16, 'base_pred_bin': 1, 'has_queue': True, 'is_answer': 1})
        original = frame.copy(deep=True)
        config = {'training_filters': {'queue': {'require_is_answer': True}}, 'non_queue_strategy': 'zero'}
        report = mining_diagnostics.feature_selection_report(frame, {'queue': Model(), 'no_queue': None},
            ['signal', 'duplicate', 'constant'], config, {'importance_rows': 16, 'importance_features': 3})
        model = report['models']['queue']
        self.assertEqual(len(model['ranking']), 3)
        self.assertEqual(model['ranking'][0]['name'], 'signal')
        self.assertEqual(model['high_correlation_pairs'][0]['spearman'], 1.)
        self.assertIn('constant', model['ablation_candidates'])
        self.assertEqual(len(model['permutation']), 3)
        pd.testing.assert_frame_equal(frame, original)
        json.dumps(report, allow_nan=False)

    def test_three_small_gains_require_actual_change_not_direction_rename(self):
        controller = ResearchController(15, research_skills=True)
        plan = {'action': 'explore', 'direction': 'supply_demand', 'reason': 'test', 'next_if_failed': 'test'}
        for delta in (.02, .01, -.01):
            controller.record(plan, 'completed', improved=delta > 0, delta_pp=delta)
        self.assertTrue(controller.snapshot()['require_alternative_to_add_only'])
        self.assertEqual(controller.snapshot()['recommended_direction'], 'feature_selection')
        proposal = {'feature_recipes': [{'kind': 'expression'}]}
        with self.assertRaises(ValueError):
            controller.validate({'plan': {**plan, 'direction': 'feature_selection'}, 'proposals': [proposal]})
        controller.validate({'plan': plan, 'proposals': [{**proposal, 'remove_features': ['noise']}]})
        controller.validate({'plan': plan, 'proposals': [{'feature_recipes': [{'kind': 'target_encoding'}]}]})
        before = controller.no_improvement
        controller.record(plan, 'inspected')
        self.assertEqual(controller.no_improvement, before)
        controller.record(plan, 'completed', improved=True, delta_pp=.2)
        self.assertFalse(controller.snapshot()['require_alternative_to_add_only'])


if __name__ == '__main__':
    unittest.main()
