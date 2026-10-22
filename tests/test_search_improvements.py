import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from agent import search_memory as memory
from skills import mining_diagnostics as diagnostics
from skills.feature_program import fingerprint


class SearchMemoryTests(unittest.TestCase):
    def setUp(self):
        self.context = {'mock': True, 'data_contract': 'notebook_xgb44_v1', 'evaluation_split': 'test',
                        'training_method': 'full_train', 'window': {'train_end': '2024-08-30'},
                        'data_fingerprint': 'data', 'config_fingerprint': 'config'}
        self.recipes = [{'name': 'ratio', 'kind': 'expression', 'expression': {
            'op': 'add', 'args': [{'col': 'queue'}, {'const': 1}]}}]

    def test_formula_aliases_and_dependencies_are_canonical(self):
        renamed = copy.deepcopy(self.recipes)
        renamed[0]['name'] = 'new_name'
        renamed[0]['expression']['args'].reverse()
        self.assertEqual(memory.recipe_signature(self.recipes), memory.recipe_signature(renamed))
        self.recipes.append({'name': 'derived', 'kind': 'expression', 'expression': {'col': 'ratio'}})
        renamed.append({'name': 'different', 'kind': 'expression', 'expression': {'col': 'new_name'}})
        self.assertEqual(memory.recipe_signature(self.recipes), memory.recipe_signature(renamed))
        renamed[-1]['expression']['col'] = 'other_input'
        self.assertNotEqual(memory.recipe_signature(self.recipes), memory.recipe_signature(renamed))

    def test_noncommutative_formulas_remain_distinct(self):
        for operation in ('div', 'sub'):
            self.recipes[0]['expression']['op'] = operation
            swapped = copy.deepcopy(self.recipes)
            swapped[0]['expression']['args'].reverse()
            self.assertNotEqual(memory.recipe_signature(self.recipes), memory.recipe_signature(swapped))

    def test_snapshot_includes_values_order_duplicates_and_history(self):
        frame = pd.DataFrame({'order_id': ['same', 'same', 'older'], 'value': [1., 2., 3.]})
        original = memory.frame_fingerprint(frame)
        changed = frame.copy()
        changed.loc[2, 'value'] = 4
        self.assertNotEqual(original, memory.frame_fingerprint(changed))
        self.assertNotEqual(original, memory.frame_fingerprint(frame.iloc[::-1]))
        self.assertNotEqual(original, memory.frame_fingerprint(frame.iloc[:2]))
        self.assertEqual(original, memory.frame_fingerprint(frame.copy()))

    def test_only_completed_exact_context_and_parent_block_retry(self):
        entry = {'status': 'completed', 'context_hash': fingerprint(self.context),
                 'parent_signature': 'parent', 'recipe_signature': memory.recipe_signature(self.recipes)}
        self.assertIs(memory.duplicate_entry([entry], self.context, 'parent', self.recipes), entry)
        for field in ('data_fingerprint', 'config_fingerprint', 'window', 'mock'):
            changed = {**self.context, field: 'different'}
            self.assertIsNone(memory.duplicate_entry([entry], changed, 'parent', self.recipes))
        self.assertIsNone(memory.duplicate_entry([entry], self.context, 'another_parent', self.recipes))
        for status in ('execution_failed', 'legacy_reference'):
            entry['status'] = status
            self.assertIsNone(memory.duplicate_entry([entry], self.context, 'parent', self.recipes))

    def test_persistent_memory_isolated_from_mock_and_archived_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mock in [('S0000001', True), ('S0000002', False)]:
                folder = root / 'artifacts/feature_mining' / name
                folder.mkdir(parents=True)
                (folder / 'session.json').write_text(json.dumps({**self.context, 'mock': mock}))
                memory.append_memory(folder, {'status': 'completed', 'session_id': name})
                with (folder / 'search_memory.jsonl').open('a') as stream:
                    stream.write('incomplete-json\n')
            loaded = memory.load_memory(root, 'S0000003', self.context)
            self.assertEqual([entry['session_id'] for entry in loaded], ['S0000001'])
            self.assertEqual(memory.load_memory(root, 'S0000001', self.context), [])
            (root / 'artifacts/feature_mining/S0000001').rename(root / 'archived')
            self.assertEqual(memory.load_memory(root, 'S0000003', self.context), [])

    def test_legacy_feedback_remains_reference_only(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'artifacts/feature_mining/S0000001'
            folder.mkdir(parents=True)
            (folder / 'session.json').write_text(json.dumps(self.context))
            (folder / 'search_feedback.json').write_text(json.dumps([
                {'recipes': self.recipes, 'gate_pass': False, 'reasons': ['旧门禁']}]))
            entries = memory.load_memory(directory, 'S0000002', self.context)
            prompt = memory.memory_for_prompt(entries, self.context, 'parent')
            self.assertEqual(prompt[0]['status'], 'legacy_reference')
            self.assertTrue(prompt[0]['retry_allowed'])
            self.assertFalse(prompt[0]['historical_gate_pass'])

    def test_prompt_memory_budget(self):
        entries = [{'status': 'legacy_reference', 'recipes': self.recipes}] * 20
        self.assertLessEqual(len(memory.memory_for_prompt(entries, self.context, 'parent')), 16)
        self.assertEqual(memory.memory_for_prompt(entries, self.context, 'parent', max_chars=1), [])

    def test_ties_do_not_replace_simpler_champion(self):
        self.assertFalse(memory.prefer_candidate(.5, 0, .5, 1))
        self.assertFalse(memory.prefer_candidate(.5, 2, .5, 2))
        self.assertTrue(memory.prefer_candidate(.5, 2, .5, 1))
        self.assertTrue(memory.prefer_candidate(.5, 0, .501, 4))
        self.assertFalse(memory.prefer_candidate(.5, 4, .499, 0))


class MiningDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.config = {'non_queue_strategy': 'zero', 'training_filters': {'queue': {'require_is_answer': True}}}
        self.frame = pd.DataFrame({'city_code': [100001] * 4, 'dt': ['2024-08-31'] * 2 + ['2024-09-01'] * 2,
            'order_id': ['duplicate', 'duplicate', 'other', 'last'], 'is_answer': [1, 0, 1, 1],
            'base_pred_bin': [1, 1, 0, 1], 'true_bin': [1, 2, 0, 3],
            'has_queue': [True, True, False, True],
            'predict_bin_44': [1, 1, 0, 2], 'feature': [7., 2., 3., 7.]})

    def test_actual_fit_population_not_all_training_rows(self):
        report = diagnostics.fitting_profiles(self.frame, ['feature'], self.config)
        self.assertEqual(report['queue']['rows'], 2)
        self.assertEqual(report['queue']['fields']['feature']['unique'], 1)
        self.assertNotIn('no_queue', report)
        self.assertEqual(diagnostics.field_stats(self.frame['feature'])['unique'], 3)

    def test_coverage_detects_fit_constant_and_test_date_missing_without_mutation(self):
        test = self.frame.copy()
        test.loc[test['dt'] == '2024-09-01', 'feature'] = np.nan
        before = test.copy()
        result = diagnostics.feature_coverage(self.frame, test, ['feature'], self.config)['feature']
        self.assertEqual(result['fitting']['queue']['unique'], 1)
        self.assertEqual(result['test_by_date']['2024-09-01']['missing_rate'], 1.)
        self.assertEqual(result['test_by_city']['100001']['missing_rate'], .5)
        self.assertIn('测试期存在整日缺失', result['warnings'])
        pd.testing.assert_frame_equal(test, before)

    def test_coverage_does_not_depend_on_test_labels(self):
        changed = self.frame.copy()
        changed['true_bin'] = 1000
        self.assertEqual(diagnostics.feature_coverage(self.frame, self.frame, ['feature'], self.config),
                         diagnostics.feature_coverage(self.frame, changed, ['feature'], self.config))

    def test_model_usage_reports_actual_counts_and_unavailable_models(self):
        class Booster:
            def get_score(self, importance_type):
                return {'feature': 3} if importance_type == 'weight' else {'feature': 1.5}
        result = diagnostics.model_usage({'queue': Booster(), 'no_queue': None}, ['feature', 'unused'])
        self.assertEqual(result['queue']['features']['feature']['split_count'], 3)
        self.assertEqual(result['queue']['features']['unused']['split_count'], 0)
        self.assertFalse(result['no_queue']['available'])

    def test_corrected_and_introduced_errors_preserve_duplicate_rows(self):
        candidate = self.frame.copy()
        candidate['predict_bin_44'] = [0, 2, 0, 3]
        result = diagnostics.prediction_changes(self.frame, candidate)
        self.assertEqual((result['corrected'], result['introduced'], result['net_corrected']), (2, 1, 1))
        self.assertEqual(result['delta_pp'], 25.)
        with self.assertRaises(ValueError):
            diagnostics.prediction_changes(self.frame, candidate.iloc[::-1])

    def test_accuracy_feedback_ranks_error_counts_not_under_rate(self):
        result = diagnostics.accuracy_feedback(self.frame)
        self.assertEqual(result['errors'], 2)
        self.assertEqual(result['queue_errors'], 2)
        self.assertEqual(result['non_queue_errors'], 0)
        self.assertEqual(sum(row['errors'] for row in result['confusions']), 2)
        counts = [row['errors'] for row in result['slices']]
        self.assertEqual(counts, sorted(counts, reverse=True))
        json.dumps(result, allow_nan=False)

    def test_empty_queue_feedback_is_valid(self):
        frame = self.frame.copy()
        frame['base_pred_bin'] = 0
        frame['has_queue'] = False
        self.assertEqual(diagnostics.accuracy_feedback(frame)['queue_slices'], [])


if __name__ == '__main__':
    unittest.main()
