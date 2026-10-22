from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_mining_session as session_tests
from agent import feature_mining


class ResearchSkillSessionTests(unittest.TestCase):
    setUp = session_tests.MiningSessionTests.setUp
    predict = session_tests.MiningSessionTests.predict
    run_session = session_tests.MiningSessionTests.run_session

    def run_plans(self, proposals):
        plans = [{'plan': {'action': 'explore', 'direction': 'feature_interaction',
                          'reason': '验证新能力', 'next_if_failed': '保留冠军并换方向'},
                  'proposals': [proposal]} for proposal in proposals]
        with patch.object(feature_mining, 'propose', side_effect=plans):
            return self.run_session(['--research-skills', '--continuous', '--rounds', '1',
                                     '--max-proposals-per-round', str(len(proposals))])

    def test_skill_training_only_promotes_and_updates_latest(self):
        trial = {**self.proposals[0], 'feature_recipes': [], 'training': {'xgb_params': {'max_depth': 5}}}
        configs = []
        def train(frame, columns, config, *arguments, **keywords):
            configs.append(deepcopy(config))
            return {'columns': list(columns)}
        def predict(frame, models, columns):
            result = self.predict(frame, models, columns)
            if len(configs) == 1:
                result.iloc[0, result.columns.get_loc('predict_bin_44')] = (result['true_bin'].iloc[0] + 1) % 6
            return result
        with patch.object(feature_mining.mining_train, 'train_pair', side_effect=train), \
             patch.object(feature_mining.mining_train, 'predict', side_effect=predict):
            manifest, records, _ = self.run_plans([trial])
        self.assertEqual(configs[0]['xgboost']['max_depth'], 3)
        self.assertEqual(configs[1]['xgboost']['max_depth'], 5)
        self.assertEqual(manifest['accepted_recipes'], [])
        self.assertNotEqual(manifest['baseline_id'], manifest['champion_id'])
        self.assertEqual(records[1]['config_diff']['xgb_params'], {'max_depth': 5})
        latest = json.loads((self.root / 'experiments/feature_mining/latest.json').read_text())
        self.assertEqual(latest['champion_id'], manifest['champion_id'])
        self.assertTrue(manifest['holdout_gate']['pass'])

    def test_skill_remove_original_feature_on_tie_promotes_simpler_model(self):
        base = json.loads((self.root / 'configs/features.json').read_text())['feature_set_27'][0]
        trial = {**self.proposals[0], 'feature_recipes': [], 'remove_features': [base]}
        manifest, records, _ = self.run_plans([trial])
        self.assertEqual(records[1]['config_diff']['candidate_changes']['removed'], [base])
        self.assertEqual(len(manifest['champion_spec']['base_features']), 42)
        self.assertEqual(manifest['champion_id'], records[1]['experiment_id'])
        self.assertTrue((self.root / 'experiments/feature_mining/latest.json').exists())

    def test_skill_branch_improves_parent_but_not_champion_is_rejected(self):
        calls = []
        def predict(frame, models, columns):
            result = self.predict(frame, models, columns)
            count = [3, 1, 2][len(calls)]
            calls.append(1)
            result.iloc[:count, result.columns.get_loc('predict_bin_44')] = (result['true_bin'].iloc[:count] + 1) % 6
            return result
        def propose(*arguments, **keywords):
            profile = arguments[0]
            trial = deepcopy(self.proposals[0 if len(calls) == 1 else 1])
            if len(calls) > 1:
                trial['parent_experiment_id'] = profile['available_parents'][0]['experiment_id']
            return {'plan': {'action': 'explore', 'direction': 'feature_interaction',
                             'reason': '测试分支', 'next_if_failed': '继续冠军'}, 'proposals': [trial]}
        with patch.object(feature_mining, 'propose', side_effect=propose), \
             patch.object(feature_mining.mining_train, 'predict', side_effect=predict):
            manifest, records, _ = self.run_session(['--research-skills', '--continuous', '--rounds', '1', '--max-proposals-per-round', '2'])
        self.assertEqual(records[2]['config_diff']['parent_experiment_id'], records[0]['experiment_id'])
        self.assertEqual(records[2]['config_diff']['compared_champion_id'], records[1]['experiment_id'])
        changes = records[2]['summary']['search_diagnostics']['prediction_changes']
        self.assertGreater(changes['parent']['delta_pp'], 0)
        self.assertLess(changes['champion']['delta_pp'], 0)
        self.assertFalse(records[2]['gate_pass'])
        self.assertEqual(manifest['champion_id'], records[1]['experiment_id'])

    def test_skill_pending_request_does_not_train_or_approve(self):
        request = {'kind': 'data_source', 'reason': '补充供给信息', 'source': 'reviewed.supply',
                   'keys': ['order_id'], 'time_column': 'available_at', 'availability': '产生时间',
                   'outputs': ['supply_signal']}
        trial = {**self.proposals[0], 'feature_recipes': [], 'requests': [request]}
        manifest, records, _ = self.run_plans([trial])
        self.assertEqual(len(records), 1)
        self.assertEqual(manifest['extension_requests'][0]['status'], 'pending_review')
        state = json.loads((Path(manifest['artifact_dir']) / 'research_state.json').read_text())
        self.assertEqual(state['steps'][0]['status'], 'pending_review')

    def test_skill_inspection_feedback_reaches_next_plan_without_training(self):
        calls = []
        def propose(profile, *args, **kwargs):
            calls.append(deepcopy(profile))
            trial = {**self.proposals[0], 'feature_recipes': []}
            if len(calls) == 1:
                trial['inspect'] = {'kind': 'fields', 'columns': ['sp2_queue_count', 'sp1_queue_count']}
            else:
                trial['remove_features'] = ['sp2_queue_count']
            return {'plan': {'action': 'explore', 'direction': 'feature_selection',
                             'reason': '先核对冗余再消融', 'next_if_failed': '保留原模型'}, 'proposals': [trial]}
        with patch.object(feature_mining, 'propose', side_effect=propose):
            manifest, records, _ = self.run_session(['--research-skills', '--continuous', '--rounds', '1',
                                                    '--max-proposals-per-round', '2'])
        self.assertEqual(len(records), 2)
        self.assertEqual(manifest['inspection_calls'], 1)
        self.assertEqual(len(calls[1]['inspection_results']), 1)
        published = json.loads((Path(manifest['artifact_dir']) / 'data_profile.json').read_text())
        self.assertEqual(len(published['inspection_results']), 1)
        self.assertNotIn('available_parents', published)
        self.assertEqual(calls[1]['data_inspection']['remaining_calls'], 3)
        self.assertIn('sp2_queue_count', records[1]['config_diff']['candidate_changes']['removed'])
        state = json.loads((Path(manifest['artifact_dir']) / 'research_state.json').read_text())
        self.assertEqual(state['steps'][0]['status'], 'inspected')
        self.assertIsNone(state['steps'][0]['experiment_id'])

    def test_skill_duplicate_or_mixed_inspection_is_rejected(self):
        trial = {**self.proposals[0], 'feature_recipes': [],
                 'inspect': {'kind': 'fields', 'columns': ['sp2_queue_count']}}
        manifest, records, _ = self.run_plans([trial, trial, {**trial, 'remove_features': ['sp2_queue_count']}])
        self.assertEqual(len(records), 1)
        self.assertEqual(manifest['inspection_calls'], 1)
        feedback = json.loads((Path(manifest['artifact_dir']) / 'search_feedback.json').read_text())
        self.assertIn('复用', feedback[1]['rejected'])
        self.assertIn('分开提交', feedback[2]['rejected'])

    def test_skill_inspection_session_budget(self):
        path = self.root / 'configs/mining.json'
        cfg = json.loads(path.read_text())
        cfg['max_inspection_calls'] = 1
        path.write_text(json.dumps(cfg))
        first = {**self.proposals[0], 'feature_recipes': [],
                 'inspect': {'kind': 'fields', 'columns': ['sp2_queue_count']}}
        second = {**first, 'inspect': {'kind': 'fields', 'columns': ['sp1_queue_count']}}
        manifest, records, _ = self.run_plans([first, second])
        self.assertEqual(len(records), 1)
        self.assertEqual(manifest['inspection_calls'], 1)
        feedback = json.loads((Path(manifest['artifact_dir']) / 'search_feedback.json').read_text())
        self.assertIn('预算已用尽', feedback[-1]['rejected'])

    def test_skill_target_encoding_and_one_hot_preserve_test_population(self):
        first = {**self.proposals[0], 'feature_recipes': [{'name': 'city_target', 'kind': 'target_encoding',
                 'keys': ['city_code'], 'target': 'mean_bin', 'smoothing': 20}]}
        second = {**self.proposals[0], 'feature_recipes': [{'name': 'city_region_a', 'kind': 'one_hot',
                  'keys': ['city_code'], 'values': [100001]}]}
        manifest, records, _ = self.run_plans([first, second])
        self.assertEqual(len(records), 3)
        self.assertEqual(len({record['config_diff']['selection_sample_hash'] for record in records}), 1)
        self.assertEqual(len({record['metrics_by_city']['xgb_44']['ALL']['total'] for record in records}), 1)
        doc = json.loads((Path(records[1]['artifact_dir']) / 'feature_manifest.json').read_text())
        self.assertIn('city_target', doc['preprocessing_state'])
        self.assertEqual(doc['fixed_train_config']['training_method'], 'notebook_random')
        self.assertEqual(records[0]['config_diff']['feature_recipes'], [])

    def test_skill_off_rejects_training_mutation(self):
        trial = {**self.proposals[0], 'training': {'xgb_params': {'max_depth': 5}}}
        with patch.object(feature_mining, 'propose', return_value=[trial]):
            manifest, records, _ = self.run_session(['--rounds', '1'])
        self.assertEqual(len(records), 1)
        self.assertEqual(manifest['champion_id'], manifest['baseline_id'])

    def test_skill_approved_source_runs_without_changing_baseline_population(self):
        from skills import research_extensions
        source = {'kind': 'data_source', 'reason': '城市历史供给', 'source': 'reviewed.supply',
                  'keys': ['city_code'], 'time_column': 'available_at', 'availability': '历史已发布数据',
                  'outputs': ['city_supply_signal']}
        created = research_extensions.stage(self.root, 'S0000000', source, {'city_code': {'groupable': True}}, mock=True)
        snapshot = self.root / 'supply.csv'
        snapshot.write_text('city_code,available_at,city_supply_signal\n100001,2024-08-01,7\n100002,2024-08-01,23\n100003,2024-08-01,41\n')
        research_extensions.review(self.root, created['id'], 'approve', 'reviewed synthetic snapshot', snapshot)
        trial = {**self.proposals[0], 'feature_recipes': [], 'extensions': [created['id']]}
        manifest, records, _ = self.run_plans([trial])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['config_diff']['experiment_spec']['extensions'], [])
        self.assertEqual(records[1]['config_diff']['add_features'], ['city_supply_signal'])
        self.assertEqual(records[0]['config_diff']['selection_sample_hash'], records[1]['config_diff']['selection_sample_hash'])
        self.assertEqual(records[0]['metrics_by_city']['xgb_44']['ALL']['total'], records[1]['metrics_by_city']['xgb_44']['ALL']['total'])

    def test_skill_remove_accepted_recipe_inherits_training_configuration(self):
        calls = []
        def predict(frame, models, columns):
            result = self.predict(frame, models, columns)
            count = 3 if not calls else 1
            calls.append(1)
            result.iloc[:count, result.columns.get_loc('predict_bin_44')] = (result['true_bin'].iloc[:count] + 1) % 6
            return result
        first = {**self.proposals[0], 'training': {'xgb_params': {'max_depth': 5}}}
        second = {**self.proposals[0], 'feature_recipes': [],
                  'remove_features': [self.proposals[0]['feature_recipes'][0]['name']]}
        with patch.object(feature_mining.mining_train, 'predict', side_effect=predict):
            manifest, records, _ = self.run_plans([first, second])
        self.assertEqual(len(records), 3)
        self.assertEqual(manifest['champion_id'], records[2]['experiment_id'])
        self.assertEqual(manifest['accepted_recipes'], [])
        self.assertEqual(manifest['champion_spec']['train_config']['xgboost']['max_depth'], 5)
        self.assertEqual(records[2]['config_diff']['parent_experiment_id'], records[1]['experiment_id'])
