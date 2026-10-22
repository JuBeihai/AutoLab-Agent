from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from agent import feature_mining
from skills.evaluation_policy import CONTRACT, SELECTION_KEY


class MiningSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = Path(__file__).resolve().parents[1]
        shutil.copytree(source / 'configs', self.root / 'configs', ignore=shutil.ignore_patterns('*.sh'))
        from skills.evaluation_policy import apply_training_population
        train_path = self.root / 'configs/train.json'
        train_cfg = json.loads(train_path.read_text())
        apply_training_population(train_cfg, 'queue_only')
        train_path.write_text(json.dumps(train_cfg))
        data_path = self.root / 'configs/data.json'
        data_config = json.loads(data_path.read_text())
        data_config['window'].update(start_date='2024-08-24', train_start='2024-08-24',
                                     train_end='2024-08-30', test_start='2024-08-31',
                                     test_end='2024-09-02', end_date='2024-09-02')
        data_path.write_text(json.dumps(data_config))
        self.proposals = json.loads((source / 'tests/smoke_proposals.json').read_text())['proposals']
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(feature_mining, 'ROOT', self.root))
        self.stack.enter_context(patch.object(feature_mining, 'load_env'))
        self.stack.enter_context(patch.object(feature_mining.mining_train, 'train_pair',
            side_effect=lambda frame, columns, *args, **kwargs: {'columns': list(columns)}))
        self.stack.enter_context(patch.object(feature_mining.mining_train, 'predict', side_effect=self.predict))
        # These orchestration fixtures mock training and never write model files.
        # Real artifact capture and replay are covered by test_reproduction.py.
        self.stack.enter_context(patch('skills.reproduction.capture'))
        self.stack.enter_context(patch('skills.reproduction.finalize'))

    def predict(self, frame, models, columns):
        self.assertEqual(list(columns), models['columns'])
        self.assertTrue(set(columns).issubset(frame.columns))
        result = frame.copy()
        result['predict_bin_44'] = result['true_bin']
        # Search-ranking fixtures use a fixed answered/queue population; routing
        # and model-dependent cancellation filtering have separate parity tests.
        result['has_queue'] = True
        result['is_answer'] = 1
        result['h3_res7'] = result.index.astype(str)
        return result

    def run_session(self, arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            feature_mining.main(['--mock', '--smoke'] + arguments)
        manifest = json.loads((self.root / 'experiments/feature_mining/current.json').read_text())
        self.assertRegex(manifest['session_id'], r'^S[0-9A-Z]{7}$')
        self.assertIn('created_at', manifest)
        records_path = self.root / 'experiments/feature_mining' / manifest['session_id'] / 'experiments.jsonl'
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        for record in records:
            self.assertRegex(record['experiment_id'], r'^E[0-9A-Z]{7}$')
        events = [json.loads(line[len('[event] '):]) for line in output.getvalue().splitlines() if line.startswith('[event] ')]
        return manifest, records, events

    def test_h3_7_controls_gate_champion_feedback_and_persisted_results(self):
        original_build = feature_mining.build_views
        scores = iter((.5, .6, .55))

        def controlled_views(frame, **kwargs):
            views = original_build(frame, **kwargs)
            # Opposing version scores ensure neither raw nor H3-6/8 drives selection.
            accuracy = next(scores)
            for population, versions in views['populations'].items():
                for version, view in versions.items():
                    for row in view.get('metrics', {}).values():
                        row['recall_ratio'] = accuracy if (population, version) == ('answer_cancel', 'h3_7') else 1 - accuracy
            return views

        def declining_raw(frame, models, columns):
            result = self.predict(frame, models, columns)
            if len(columns) > 43:
                result['predict_bin_44'] = (result['true_bin'] + 1) % 6
            return result

        plans = [{'plan': {'action': 'explore', 'direction': 'supply_demand',
                           'reason': '检验保序后收益', 'next_if_failed': '换特征'}, 'proposals': [p]}
                 for p in self.proposals[:2]]
        with patch.object(feature_mining, 'build_views', side_effect=controlled_views), \
             patch.object(feature_mining, 'propose', side_effect=plans), \
             patch.object(feature_mining.mining_train, 'predict', side_effect=declining_raw):
            manifest, records, _ = self.run_session(['--continuous', '--rounds', '1', '--max-proposals-per-round', '2'])
        self.assertEqual([r['gate_pass'] for r in records], [None, True, False])
        self.assertEqual(manifest['champion_id'], records[1]['experiment_id'])
        self.assertEqual(manifest['selection_metric'], SELECTION_KEY)
        self.assertEqual([r['summary']['verdict'] for r in records], ['baseline', 'improvement', 'regression'])
        directory = Path(manifest['artifact_dir'])
        research = json.loads((directory / 'research_state.json').read_text())
        self.assertAlmostEqual(research['steps'][0]['delta_pp'], 10)
        self.assertAlmostEqual(research['steps'][1]['delta_pp'], -5)
        journal = [json.loads(line) for line in (directory / 'search_memory.jsonl').read_text().splitlines()]
        self.assertEqual(journal[0]['metrics']['recall_ratio'], .6)
        self.assertEqual(journal[0]['selection_metric'], SELECTION_KEY)
        result = json.loads((directory / 'holdout_result.json').read_text())
        self.assertTrue(result['gate']['pass'])
        self.assertEqual(result['selected'][SELECTION_KEY]['ALL']['recall_ratio'], .6)
        self.assertEqual(result['selected']['xgb_44']['ALL']['recall_ratio'], 0)

    def test_selected_model_is_persisted_without_calling_api_for_baseline(self):
        with patch.object(feature_mining, 'propose', side_effect=AssertionError('baseline cannot call LLM')):
            manifest, records, _ = self.run_session(['--no-agent', '--llm-profile', 'responses'])
        self.assertEqual(manifest['llm'], {'profile_id': 'responses', 'model': 'YOUR_RESPONSES_MODEL', 'protocol': 'responses'})
        self.assertEqual(len(records), 1)

    def test_resume_baseline_has_no_gate_and_uses_inherited_holdout_columns(self):
        previous = {'session_id': 'previous_success', 'mock': True, 'status': 'complete',
                    'data_contract': CONTRACT, 'selection_metric': SELECTION_KEY,
                    'evaluation_split': 'test', 'training_method': 'notebook_random',
                    'accepted_recipes': self.proposals[0]['feature_recipes']}
        latest = self.root / 'experiments/feature_mining/latest.json'
        latest.parent.mkdir(parents=True)
        latest.write_text(json.dumps(previous))
        with patch.object(feature_mining.mining_train, 'gate', side_effect=AssertionError('baseline must not be gated')):
            manifest, records, events = self.run_session(['--resume-recipes', '--no-agent'])
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['resume_from_session_id'], 'previous_success')
        self.assertEqual(records[0]['proposer'], 'baseline')
        self.assertIsNone(records[0]['gate_pass'])
        self.assertIsNone(records[0]['config_diff']['parent_experiment_id'])
        self.assertEqual(records[0]['config_diff']['add_features'], ['queue_idle_pressure'])
        self.assertEqual(json.loads(latest.read_text()), previous)
        self.assertEqual(events[-1]['status'], 'complete')

    def test_rounds_record_actual_parent_and_emit_completed_events(self):
        attempts = []

        def improving_predict(frame, models, columns):
            result = self.predict(frame, models, columns)
            attempts.append(len(columns))
            count = 4 - len(attempts)
            result.iloc[:count, result.columns.get_loc('predict_bin_44')] = (result['true_bin'].iloc[:count] + 1) % 6
            return result

        with patch.object(feature_mining, 'propose', side_effect=[[self.proposals[0]], [self.proposals[1]]]), \
             patch.object(feature_mining.mining_train, 'predict', side_effect=improving_predict), \
             patch.object(feature_mining.mining_train, 'gate', return_value={'pass': True, 'reasons': []}):
            manifest, records, events = self.run_session(['--rounds', '2', '--max-proposals-per-round', '1'])
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1]['config_diff']['parent_experiment_id'], records[0]['experiment_id'])
        self.assertEqual(records[2]['config_diff']['parent_experiment_id'], records[1]['experiment_id'])
        self.assertEqual(manifest['champion_id'], records[2]['experiment_id'])
        completed = [event['experiment_id'] for event in events if event['type'] == 'experiment_completed']
        self.assertEqual(completed, [record['experiment_id'] for record in records])

    def test_equal_accuracy_keeps_champion_and_preserves_exploration_branch(self):
        with patch.object(feature_mining, 'propose', return_value=[self.proposals[0]]):
            manifest, records, _ = self.run_session(['--rounds', '1'])
        self.assertTrue(records[1]['gate_pass'])
        self.assertEqual(records[1]['summary']['verdict'], 'neutral')
        self.assertFalse(records[1]['summary']['champion_eligible'])
        self.assertEqual(manifest['champion_id'], records[0]['experiment_id'])
        self.assertEqual(manifest['accepted_recipes'], [])
        self.assertEqual(manifest['exploration_branches'][0]['experiment_id'], records[1]['experiment_id'])
        self.assertFalse((self.root / 'experiments/feature_mining/latest.json').exists())
        diagnostic = json.loads((Path(records[1]['artifact_dir']) / 'diagnostics.json').read_text())
        self.assertEqual(diagnostic['prediction_changes']['parent']['net_corrected'], 0)
        self.assertIn('queue_idle_pressure', diagnostic['feature_coverage'])

    def test_cross_session_renamed_duplicate_is_not_retrained(self):
        with patch.object(feature_mining, 'propose', return_value=[self.proposals[0]]):
            first, first_records, _ = self.run_session(['--rounds', '1'])
        renamed = json.loads(json.dumps(self.proposals[0]))
        renamed['feature_recipes'][0]['name'] = 'same_formula_new_name'

        def propose(*arguments):
            recalled = arguments[9]
            self.assertTrue(any(entry['session_id'] == first['session_id'] and entry['exact_context']
                                and not entry['retry_allowed'] for entry in recalled))
            return [renamed]

        with patch.object(feature_mining, 'propose', side_effect=propose):
            second, records, _ = self.run_session(['--rounds', '1'])
        self.assertEqual(len(records), 1)
        feedback = json.loads((Path(second['artifact_dir']) / 'search_feedback.json').read_text())
        self.assertEqual(feedback[0]['status'], 'duplicate_skipped')
        self.assertEqual(feedback[0]['source_experiment_id'], first_records[1]['experiment_id'])

    def test_execution_failure_can_retry_same_recipe(self):
        attempts = []

        def train(frame, columns, *arguments, **keywords):
            if len(columns) > 43:
                attempts.append(len(columns))
                if len(attempts) == 1:
                    raise RuntimeError('simulated temporary failure')
            return {'columns': list(columns)}

        with patch.object(feature_mining, 'propose', return_value=[self.proposals[0]]), \
             patch.object(feature_mining.mining_train, 'train_pair', side_effect=train):
            manifest, records, _ = self.run_session(['--rounds', '2'])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(records), 2)
        entries = [json.loads(line) for line in (Path(manifest['artifact_dir']) / 'search_memory.jsonl').read_text().splitlines()]
        self.assertEqual([entry['status'] for entry in entries], ['execution_failed', 'completed'])

    def test_continuous_replans_after_actual_failed_candidate(self):
        calls = []

        def propose(*arguments, **keywords):
            state = keywords['research_state']
            self.assertEqual(arguments[6], 1)
            self.assertEqual(state['attempted_steps'], len(calls))
            if calls:
                self.assertEqual(arguments[3][-1]['status'], 'completed')
                self.assertEqual(arguments[3][-1]['outcome'], 'regression')
                self.assertEqual(state['no_improvement_steps'], 1)
                self.assertIn('prediction_changes', arguments[3][-1]['diagnostics'])
            proposal = self.proposals[len(calls)]
            calls.append(state)
            return {'plan': {'action': 'explore', 'direction': 'history_coverage' if len(calls) == 2 else 'supply_demand',
                             'reason': '根据上一次实际结果更新假设', 'next_if_failed': '分析覆盖后转向时间变化'},
                    'proposals': [proposal]}

        def predict(frame, models, columns):
            result = self.predict(frame, models, columns)
            if len(columns) > 43:
                result.iloc[0, result.columns.get_loc('predict_bin_44')] = (result['true_bin'].iloc[0] + 1) % 6
            return result

        with patch.object(feature_mining, 'propose', side_effect=propose), \
             patch.object(feature_mining.mining_train, 'predict', side_effect=predict):
            manifest, records, _ = self.run_session(['--continuous', '--rounds', '1', '--max-proposals-per-round', '2'])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(records), 3)
        self.assertFalse(records[1]['gate_pass'])
        self.assertEqual(manifest['champion_id'], records[0]['experiment_id'])
        self.assertEqual(manifest['research_state']['attempted_steps'], 2)
        self.assertIn('预算', manifest['research_state']['stop_reason'])
        self.assertEqual(records[2]['summary']['research_plan']['direction'], 'history_coverage')

    def test_continuous_planning_failure_does_not_abort_session(self):
        stop = {'plan': {'action': 'stop', 'direction': 'supply_demand', 'reason': '没有新证据',
                         'next_if_failed': '补充可用字段后重试'}, 'proposals': []}
        with patch.object(feature_mining, 'propose', side_effect=[ValueError('invalid JSON'), stop]):
            manifest, records, _ = self.run_session(['--continuous', '--rounds', '1', '--max-proposals-per-round', '3'])
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(len(records), 1)
        state = json.loads((Path(manifest['artifact_dir']) / 'research_state.json').read_text())
        self.assertEqual([step['status'] for step in state['steps']], ['planning_failed', 'stopped'])
        self.assertIn('没有新证据', state['stop_reason'])

    def test_exhausted_planning_is_failed_but_keeps_baseline(self):
        with patch.object(feature_mining, 'propose', side_effect=feature_mining.LLMEmptyContentError('模型未返回正式回答')) as propose, \
             redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            feature_mining.main(['--mock', '--smoke', '--continuous', '--rounds', '2', '--max-proposals-per-round', '3'])
        self.assertEqual(stopped.exception.code, 1)
        self.assertEqual(propose.call_count, 3)
        manifest = json.loads((self.root / 'experiments/feature_mining/current.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['error_type'], 'planning_failed')
        self.assertIn('模型未返回正式回答', manifest['error_hint'])
        self.assertEqual(manifest['baseline_id'], manifest['champion_id'])
        self.assertTrue((Path(manifest['artifact_dir']) / 'holdout_result.json').exists())
        records = self.root / 'experiments/feature_mining' / manifest['session_id'] / 'experiments.jsonl'
        self.assertEqual(len(records.read_text().splitlines()), 1)

    def test_continuous_patience_stops_before_full_budget(self):
        planned = {'plan': {'action': 'explore', 'direction': 'supply_demand', 'reason': '检验供需比值',
                            'next_if_failed': '改查历史覆盖'}, 'proposals': [self.proposals[0]]}
        with patch.object(feature_mining, 'propose', return_value=planned) as propose:
            manifest, records, _ = self.run_session(['--continuous', '--rounds', '2',
                                                    '--max-proposals-per-round', '3', '--search-patience', '1'])
        self.assertEqual(propose.call_count, 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(manifest['research_state']['attempted_steps'], 1)
        self.assertIn('连续 1', manifest['research_state']['stop_reason'])

    def test_resume_rejects_cross_data_mode(self):
        latest = self.root / 'experiments/feature_mining/latest.json'
        latest.parent.mkdir(parents=True)
        latest.write_text(json.dumps({'session_id': 'real_success', 'mock': False, 'accepted_recipes': []}))
        with self.assertRaises(SystemExit), patch('sys.stderr', new_callable=io.StringIO):
            feature_mining.main(['--mock', '--resume-recipes', '--no-agent'])
        self.assertFalse((self.root / 'artifacts').exists())

    def test_resume_rejects_old_contract_before_creating_session(self):
        latest = self.root / 'experiments/feature_mining/latest.json'
        latest.parent.mkdir(parents=True)
        latest.write_text(json.dumps({'session_id': 'old_success', 'mock': True, 'accepted_recipes': []}))
        with self.assertRaises(SystemExit), patch('sys.stderr', new_callable=io.StringIO):
            feature_mining.main(['--mock', '--resume-recipes', '--no-agent'])
        self.assertFalse((self.root / 'artifacts').exists())

    def test_real_sampling_rejected_before_creating_session(self):
        with self.assertRaises(SystemExit), patch('sys.stderr', new_callable=io.StringIO):
            feature_mining.main(['--sample-rows', '8000', '--profile-only'])
        self.assertFalse((self.root / 'artifacts').exists())

    def test_all_candidates_use_the_same_full_test_window_without_early_stop(self):
        observed = []
        trained = []
        config = json.loads((self.root / 'configs/data.json').read_text())['window']

        def train(frame, columns, train_cfg, *args, **kwargs):
            self.assertTrue(frame['dt'].between(config['train_start'], config['train_end']).all())
            self.assertEqual(train_cfg['training_method'], 'notebook_random')
            self.assertEqual(train_cfg['xgboost']['early_stopping_rounds'], 40)
            trained.append(tuple(frame.index))
            return {'columns': list(columns)}

        def predict(frame, models, columns):
            is_test = frame['dt'].between(config['test_start'], config['test_end']).all()
            self.assertTrue(is_test, 'all metrics must use the test window')
            self.assertTrue((frame['base_pred_bin'] == 0).any())
            self.assertTrue((frame['is_answer'] == 0).any())
            observed.append(tuple(frame.index))
            return self.predict(frame, models, columns)

        def propose(*args):
            self.assertEqual(len(observed), 1, 'baseline test metrics must exist before proposal')
            self.assertEqual(args[2]['xgb_44']['ALL']['total'], len(observed[0]))
            return [self.proposals[0]]

        with patch.object(feature_mining.mining_train, 'predict', side_effect=predict), \
             patch.object(feature_mining.mining_train, 'train_pair', side_effect=train), \
             patch.object(feature_mining.mining_train, 'early_stop_rows', side_effect=AssertionError('no early stop split')), \
             patch.object(feature_mining, 'propose', side_effect=propose):
            manifest, records, _ = self.run_session(['--rounds', '1', '--max-proposals-per-round', '1'])
        self.assertEqual(manifest['window']['train_end'], config['train_end'])
        self.assertNotIn('validation_start', manifest['window'])
        self.assertEqual(manifest['evaluation_split'], 'test')
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(trained[0], trained[1])
        self.assertEqual(len(trained[0]), manifest['rows_by_split']['train'])
        self.assertEqual(len(observed[0]), manifest['rows_by_split']['test'])
        self.assertFalse(set(trained[0]) & set(observed[0]))
        for record in records:
            self.assertEqual(record['config_diff']['evaluation_split'], 'test')
            self.assertEqual(record['config_diff']['selection_sample_hash'], manifest['selection_sample_hash'])
            self.assertTrue((Path(record['artifact_dir']) / 'selection_result.json').exists())
            self.assertFalse((Path(record['artifact_dir']) / 'validation_result.json').exists())
        result = json.loads((Path(manifest['artifact_dir']) / 'holdout_result.json').read_text())
        self.assertTrue(result['used_for_search'])
        self.assertEqual(result['baseline'], records[0]['metrics_by_city'])
        selected = next(record for record in records if record['experiment_id'] == manifest['champion_id'])
        self.assertEqual(result['selected'], selected['metrics_by_city'])
        self.assertEqual(result['baseline_views'], records[0]['evaluation_views'])
        self.assertEqual(result['selected_views'], selected['evaluation_views'])
        for record in records:
            self.assertEqual(record['evaluation_views']['protocol'], 'notebook_xgb43_6_zero_after_ordering_v1')
            raw = record['evaluation_views']['populations']['all']['raw']['metrics']['ALL']
            self.assertEqual(raw, record['metrics_by_city']['xgb_44']['ALL'])

    def test_resume_rejects_old_evaluation_protocol(self):
        latest = self.root / 'experiments/feature_mining/latest.json'
        latest.parent.mkdir(parents=True)
        latest.write_text(json.dumps({'session_id': 'old_validation', 'mock': True,
            'data_contract': 'notebook_xgb44_v1', 'evaluation_split': 'validation', 'accepted_recipes': []}))
        with self.assertRaises(SystemExit), patch('sys.stderr', new_callable=io.StringIO):
            feature_mining.main(['--mock', '--resume-recipes', '--no-agent'])
        self.assertFalse((self.root / 'artifacts').exists())

    def test_progress_events_report_real_stages_without_fake_percentages(self):
        manifest, records, events = self.run_session(['--no-agent'])
        progress = [event for event in events if event['type'] == 'progress']
        stages = [event['stage'] for event in progress]
        self.assertEqual(list(dict.fromkeys(stages)), ['loading', 'profiling', 'features', 'evaluating', 'evaluated', 'finishing'])
        self.assertTrue(all(event['session_id'] == manifest['session_id'] for event in progress))
        evaluation = next(event for event in progress if event['stage'] == 'evaluating')
        self.assertEqual(evaluation['experiment_id'], records[0]['experiment_id'])
        self.assertIn(f"{manifest['rows_by_split']['test']:,}", evaluation['message'])
        self.assertTrue(all('percent' not in event for event in progress))

    def test_resume_rejects_internal_early_stop_session(self):
        latest = self.root / 'experiments/feature_mining/latest.json'
        latest.parent.mkdir(parents=True)
        latest.write_text(json.dumps({'session_id': 'old_internal', 'mock': True,
            'data_contract': 'notebook_xgb44_v1', 'evaluation_split': 'train_internal_early_stop',
            'accepted_recipes': []}))
        with self.assertRaises(SystemExit), patch('sys.stderr', new_callable=io.StringIO):
            feature_mining.main(['--mock', '--resume-recipes', '--no-agent'])
        self.assertFalse((self.root / 'artifacts').exists())

    def test_three_city_champions_and_success_pointers_update_independently(self):
        from skills import city_training
        original_build = feature_mining.build_views
        scores = {'100001': iter((.6, .7)), '100002': iter((.8, .7)), '100003': iter((.5, .6))}
        seen = []
        def train(frame, columns, cfg, *args, **kwargs):
            city = cfg['training_city']
            self.assertTrue(frame.city_code.astype(str).eq(city).all())
            seen.append(city)
            return {'columns': list(columns)}
        def controlled_views(frame, **kwargs):
            self.assertEqual(frame.city_code.nunique(), 1)
            city = str(frame.city_code.iloc[0])
            views = original_build(frame, **kwargs)
            accuracy = next(scores[city])
            for row in views['populations']['answer_cancel']['h3_7']['metrics'].values():
                row['recall_ratio'] = accuracy
            return views
        plans = [{'plan': {'action':'explore', 'direction':'supply_demand',
                           'reason':'按城市验证独立更新', 'next_if_failed':'换特征'},
                  'proposals':[self.proposals[0]]} for _ in range(3)]
        with patch.object(feature_mining, 'build_views', side_effect=controlled_views), \
             patch.object(feature_mining, 'propose', side_effect=plans), \
             patch.object(feature_mining.mining_train, 'train_pair', side_effect=train), \
             redirect_stdout(io.StringIO()):
            feature_mining.main(['--mock','--smoke','--training-mode','by_city',
                                 '--continuous','--rounds','1','--max-proposals-per-round','1'])
        group_path = next((self.root/'artifacts/city_training').glob('*.json'))
        group = city_training.group_summary(self.root, group_path.stem)
        self.assertEqual(group['status'], 'complete')
        self.assertEqual({city:entry['updated'] for city,entry in group['cities'].items()},
                         {'100001':True,'100002':False,'100003':True})
        self.assertEqual(seen, ['100001','100001','100002','100002','100003','100003'])
        for city in ('100001','100003'):
            self.assertTrue(city_training.latest_path(self.root,city).exists())
        self.assertFalse(city_training.latest_path(self.root,'100002').exists())
        self.assertFalse(city_training.latest_path(self.root).exists())
