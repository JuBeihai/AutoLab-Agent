import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from webui import runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.jobs_patch = patch.object(runner, '_jobs', {})
        self.sink_patch = patch.object(runner, '_LOG_SINK', None)
        self.jobs_patch.start()
        self.sink_patch.start()
        memory_patch = patch.object(runner, 'require_available_memory', return_value={'oom_kill_count': 0})
        memory_patch.start()
        self.addCleanup(memory_patch.stop)
        self.addCleanup(self.jobs_patch.stop)
        self.addCleanup(self.sink_patch.stop)
        self.job = runner.Job('test_job', [])
        runner._jobs[self.job.job_id] = self.job

    def finish(self):
        with self.job.condition:
            self.job.returncode = 0
            self.job.ended_at = 1
            self.job.condition.notify_all()

    def test_sse_no_duplicates_and_completed_reconnection(self):
        self.job.emit('first')
        self.job.emit('second')
        self.finish()
        events = list(runner.stream_lines('test_job'))
        self.assertEqual(events[0], 'id: 1\ndata: first\n\n')
        self.assertEqual(events[1], 'id: 2\ndata: second\n\n')
        self.assertEqual(len(events), 3)
        resumed = list(runner.stream_lines('test_job', after=1))
        self.assertEqual(resumed, events[1:])
        self.assertEqual(list(runner.stream_lines('test_job', after=2)), events[-1:])

    def test_subscribers_each_receive_every_event(self):
        first = runner.stream_lines('test_job')
        second = runner.stream_lines('test_job')
        self.job.emit('shared')
        self.assertEqual(next(first), next(second))
        self.job.emit('new')
        self.assertEqual(next(first), next(second))
        self.finish()
        self.assertEqual(next(first), next(second))

    def test_bounded_logs_keep_absolute_sse_ids_for_lagging_subscriber(self):
        with patch.object(runner, '_MAX_HISTORY_LINES', 2):
            self.job.emit('first')
            subscriber = runner.stream_lines('test_job')
            self.assertIn('id: 1', next(subscriber))
            for value in ('second', 'third', 'fourth'):
                self.job.emit(value)
            self.finish()
        self.assertEqual(len(self.job.history), 2)
        events = list(subscriber)
        self.assertIn('id: 3\ndata: third', events[0])
        self.assertIn('id: 4\ndata: fourth', events[1])
        self.assertEqual(list(runner.stream_lines('test_job', after=3)), events[1:])

    def test_low_memory_rejects_real_run_without_spawning(self):
        runner._jobs.clear()
        with patch.object(runner, 'require_available_memory', side_effect=RuntimeError('内存不足')), \
             patch.object(runner.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, '内存不足'):
                runner.start_orchestrator(1, 1, False, False)
        spawn.assert_not_called()
        self.assertEqual(runner._jobs, {})

    def test_long_logs_are_bounded_but_disk_sink_keeps_full_text(self):
        long_line = 'long log ' * 5000
        with patch.object(runner, '_LOG_SINK') as sink:
            self.job.emit(long_line)
        self.assertLess(len(self.job.history[0][1]), 17000)
        sink.assert_called_once_with(long_line)

    def test_finished_job_cache_prunes_only_old_completed_jobs(self):
        for position in range(20):
            job = runner.Job(f'finished_{position}', [])
            job.ended_at = position + 1
            runner._jobs[job.job_id] = job
        runner._prune_finished_jobs()
        self.assertEqual(len(runner._jobs), runner._MAX_FINISHED_JOBS + 1)
        self.assertIs(runner.get_job('test_job'), self.job)
        self.assertIsNone(runner.get_job('finished_0'))
        self.assertIsNotNone(runner.get_job('finished_19'))

    def test_sigkill_reason_persists_and_reaches_sse(self):
        for killed in (4, 5):
            with self.subTest(killed=killed), tempfile.TemporaryDirectory() as directory, \
                 patch.object(runner, '_ROOT', Path(directory)), \
                 patch.object(runner, 'memory_snapshot', return_value={'oom_kill_count': killed}):
                self.job.session_id = 'S0000001'
                self.job.memory_before = {'oom_kill_count': 4}
                artifact = Path(directory) / 'artifacts/feature_mining/S0000001/session.json'
                artifact.parent.mkdir(parents=True)
                artifact.write_text(json.dumps({'status': 'loading'}))
                runner._persist_job_result(self.job, -9)
                manifest = json.loads(artifact.read_text())
                self.assertEqual(manifest['status'], 'failed')
                self.assertEqual(manifest['job_returncode'], -9)
                self.assertEqual(manifest['error_type'], 'out_of_memory' if killed == 5 else 'out_of_memory_or_sigkill')
                self.assertIn('SIGKILL', manifest['error_hint'])
                self.job.returncode = -9
                self.job.ended_at = 1
                event = list(runner.stream_lines('test_job'))[-1]
                self.assertIn(manifest['error_type'], event)

    def test_real_logs_redact_credentials_before_replay_and_persistence(self):
        self.job.secrets = ('example-private-token',)
        with patch.object(runner, '_LOG_SINK') as sink:
            self.job.emit('failure example-private-token')
            self.job.emit("Project demo_decrypt('hidden-sql-token', start_point)")
        persisted = str(sink.call_args_list)
        replay = str(self.job.history)
        for text in (persisted, replay):
            self.assertNotIn('example-private-token', text)
            self.assertNotIn('hidden-sql-token', text)
            self.assertIn('[REDACTED]', text)

    def test_structured_events_and_reader_drain(self):
        payload = {'type': 'experiment_completed', 'session_id': 'session_test', 'experiment_id': 'experiment_test'}
        runner._reader_thread(self.job, io.StringIO('[event] ' + json.dumps(payload) + '\n'), 'stdout')
        self.assertEqual(self.job.session_id, 'session_test')
        self.assertEqual(self.job.history[0][0], 'experiment')
        self.job.proc = Mock()
        self.job.proc.wait.return_value = 0
        reader = Mock()
        reader.join.side_effect = lambda: self.job.emit('last line')
        with patch.object(runner, '_persist_job_result'):
            runner._waiter_thread(self.job, [reader])
        events = list(runner.stream_lines('test_job'))
        self.assertIn('event: experiment', events[0])
        self.assertIn('last line', events[1])
        self.assertIn('event: done', events[-1])

    def test_concurrent_start_only_creates_one_process(self):
        runner._jobs.clear()
        barrier = threading.Barrier(2)
        results = []

        def start():
            barrier.wait()
            try:
                results.append(runner.start_orchestrator(1, 1, True, False))
            except RuntimeError:
                results.append('busy')

        workers = [threading.Thread(target=start) for _ in range(2)]
        with patch.object(runner.subprocess, 'Popen') as spawn, patch.object(runner.threading, 'Thread'):
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=5)
            self.assertEqual(spawn.call_count, 1)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        self.assertEqual(results.count('busy'), 1)

    def test_stop_targets_the_isolated_process_group(self):
        self.job.proc = Mock(pid=12345)
        self.job.proc.poll.return_value = None
        with patch.object(runner.os, 'killpg') as kill:
            self.assertTrue(runner.stop_job('test_job'))
            kill.assert_called_once_with(12345, runner.signal.SIGTERM)
        self.assertTrue(self.job.stop_requested)

    def test_runner_passes_model_selection_and_redacts_both_keys(self):
        runner._jobs.clear()
        with patch.object(runner.subprocess, 'Popen') as spawn, patch.object(runner.threading, 'Thread'), \
             patch.dict(os.environ, {'ETS_LLM_API_KEY': 'shared-test-key', 'ETS_GPT_API_KEY': 'gpt-test-key'}):
            job_id = runner.start_orchestrator(1, 1, True, False, llm_profile='responses')
        command = spawn.call_args.args[0]
        self.assertEqual(command[command.index('--llm-profile') + 1], 'responses')
        job = runner.get_job(job_id)
        job.emit('shared-test-key gpt-test-key')
        self.assertNotIn('shared-test-key', str(job.history))
        self.assertNotIn('gpt-test-key', str(job.history))

    def test_real_run_loads_spark_environment_and_forwards_profile_options(self):
        runner._jobs.clear()
        with patch.object(runner.subprocess, 'Popen') as spawn, patch.object(runner.threading, 'Thread'):
            runner.start_orchestrator(1, 2, False, True, sample_rows=0, profile_only=True)
        command = spawn.call_args.args[0]
        self.assertEqual(command[:2], ['bash', 'scripts/run_feature_mining.sh'])
        self.assertEqual(command[command.index('--sample-rows') + 1], '0')
        self.assertIn('--profile-only', command)
        self.assertIn('--no-agent', command)
        self.assertNotIn('--mock', command)
        self.assertEqual(spawn.call_args.kwargs['env']['ETS_PYTHON'], runner._PYTHON)

    def test_stopped_session_persists_without_changing_latest(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, '_ROOT', Path(directory)):
            self.job.session_id = 'session_test'
            self.job.stop_requested = True
            artifact = Path(directory) / 'artifacts/feature_mining/session_test/session.json'
            pointer = Path(directory) / 'experiments/feature_mining/current.json'
            for path in (artifact, pointer):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'session_id': 'session_test', 'status': 'searching'}))
            latest = pointer.with_name('latest.json')
            latest.write_text('{"session_id": "previous_success"}')
            runner._persist_job_result(self.job, -15)
            self.assertEqual(json.loads(artifact.read_text())['status'], 'cancelled')
            self.assertEqual(json.loads(pointer.read_text())['status'], 'cancelled')
            self.assertEqual(json.loads(latest.read_text())['session_id'], 'previous_success')


class WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.runtime.cleanup)
        with patch.dict(os.environ, {'ETS_WEBUI_STATE_DIR': cls.runtime.name + '/state',
                                     'ETS_WEBUI_LOG_DIR': cls.runtime.name + '/logs'}):
            cls.module = importlib.import_module('webui.app')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        log = patch.object(self.module, 'SESSION_LOG_PATH', self.root / 'webui-test.log')
        log.start(); self.addCleanup(log.stop)
        self.artifacts = self.root / 'artifacts'
        self.experiments = self.root / 'experiments'
        for name, value in (('_ROOT', self.root), ('ARTIFACTS_DIR', self.artifacts),
                            ('EXPERIMENTS_DIR', self.experiments)):
            replacement = patch.object(self.module, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.client = self.module.app.test_client()
        self.old_id = 'mining_mock_20240913_100000_aaaaaa'
        self.current_id = 'mining_real_20240914_100000_bbbbbb'
        self.old = self.make_session(self.old_id, True, 0.99)
        self.current = self.make_session(self.current_id, False, 0.6)
        pointer = self.experiments / 'feature_mining/current.json'
        pointer.write_text(json.dumps({**self.current, 'status': 'loading'}))
        legacy = self.experiments / 'experiments.jsonl'
        legacy.write_text(json.dumps({'experiment_id': 'legacy_wrong_champion'}) + '\n')

    def make_session(self, session_id, mock, accuracy):
        directory = self.artifacts / 'feature_mining' / session_id
        directory.mkdir(parents=True)
        logs = self.experiments / 'feature_mining' / session_id
        logs.mkdir(parents=True)
        records = []
        for round_id in range(3):
            experiment_id = f'{session_id}_r{round_id}_20240914_10000{round_id}_cccccc'
            parent = records[-1]['experiment_id'] if records else None
            records.append({'experiment_id': experiment_id, 'round_id': round_id,
                            'timestamp': f'2024-09-14T10:00:0{round_id}',
                            'proposer': 'baseline' if round_id == 0 else 'agent',
                            'gate_pass': None if round_id == 0 else True,
                            'config_diff': {'session_id': session_id, 'mock': mock,
                                            'parent_experiment_id': parent, 'evaluation_split': 'validation',
                                            'add_features': ['unselected_feature']},
                            'metrics_by_city': {'xgb_44': {'ALL': {'recall_ratio': accuracy + round_id * 0.001}}},
                            'summary': {'verdict': 'baseline' if round_id == 0 else 'improvement'}})
        manifest = {'session_id': session_id, 'mock': mock, 'status': 'complete',
                    'champion_id': records[-1]['experiment_id'], 'accepted_recipes': [{'name': 'selected_feature'}],
                    'window': {'test_start': '2024-08-24'}, 'production_approved': False}
        (directory / 'session.json').write_text(json.dumps(manifest))
        (logs / 'experiments.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
        (directory / 'search_feedback.json').write_text(json.dumps([{'rejected': 'sparse feature'}]))
        return manifest

    def test_champion_discovery_and_memory_share_session(self):
        champion = self.client.get('/api/champion').get_json()
        discovery = self.client.get('/api/discovery').get_json()
        memory = self.client.get('/api/memory').get_json()
        self.assertEqual(champion['champion']['experiment_id'], self.current['champion_id'])
        self.assertEqual(champion['session'], discovery['session'])
        self.assertEqual(champion['session']['status'], 'complete')
        self.assertEqual(memory['champion'], champion['champion'])
        self.assertEqual(memory['accepted_features'], ['selected_feature'])
        historical = self.client.get('/api/champion', query_string={'session_id': self.old_id}).get_json()
        self.assertEqual(historical['champion']['experiment_id'], self.old['champion_id'])
        self.assertTrue(historical['session']['mock'])

    def test_history_and_lineage_use_selected_session_and_real_parent(self):
        query = {'session_id': self.current_id}
        records = self.client.get('/api/experiments', query_string=query).get_json()['records']
        self.assertEqual(len(records), 3)
        lineage = self.client.get('/api/lineage', query_string=query).get_json()['lineages']
        self.assertEqual(len(lineage), 1)
        nodes = lineage[0]['nodes']
        self.assertEqual(nodes[2]['parent_id'], nodes[1]['id'])
        self.assertAlmostEqual(nodes[2]['acc_delta_vs_parent'], 0.001)

    def test_new_policy_history_and_lineage_use_h3_7_without_relabelling_old_records(self):
        path = self.experiments / 'feature_mining' / self.current_id / 'experiments.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for index, record in enumerate(records):
            record['config_diff']['selection_metric'] = 'all_h3_7'
            record['metrics_by_city']['all_h3_7'] = {'ALL': {'recall_ratio': .6 + index * .01}}
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        query = {'session_id': self.current_id}
        rows = self.client.get('/api/experiments', query_string=query).get_json()['records']
        self.assertTrue(all(row['selection_metric'] == 'all_h3_7' for row in rows))
        self.assertAlmostEqual(rows[0]['all_acc'], .6)
        nodes = self.client.get('/api/lineage', query_string=query).get_json()['lineages'][0]['nodes']
        self.assertAlmostEqual(nodes[2]['all_acc'], .62)
        self.assertAlmostEqual(nodes[2]['acc_delta_vs_parent'], .01)
        old = self.client.get('/api/experiments', query_string={'session_id': self.old_id}).get_json()['records']
        self.assertTrue(all(row['selection_metric'] == 'xgb_44' for row in old))

    def test_unknown_session_and_path_traversal_are_rejected(self):
        self.assertEqual(self.client.get('/api/champion?session_id=missing').status_code, 404)
        self.assertEqual(self.client.get('/api/champion?session_id=../escape').status_code, 400)

    def test_delete_session_archives_owned_data_and_updates_pointers(self):
        mining = self.experiments / 'feature_mining'
        (mining / 'latest.json').write_text(json.dumps(self.current))
        (mining / 'failed').mkdir()
        (mining / 'failed' / f'{self.current_id}.json').write_text(json.dumps(self.current))
        response = self.client.delete(f'/api/sessions/{self.current_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['deleted_session_id'], self.current_id)
        self.assertFalse((self.artifacts / 'feature_mining' / self.current_id).exists())
        self.assertFalse((mining / self.current_id).exists())
        self.assertFalse((mining / 'failed' / f'{self.current_id}.json').exists())
        self.assertEqual(self.client.get('/api/sessions').get_json()['current_session_id'], self.old_id)
        self.assertTrue((self.artifacts / 'feature_mining' / self.old_id).exists())
        archived = list((self.experiments / '.session_trash').glob('*/artifacts/session.json'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(json.loads(archived[0].read_text())['session_id'], self.current_id)
        records = self.client.get('/api/experiments').get_json()['records']
        self.assertNotIn(self.current['champion_id'], [record['experiment_id'] for record in records])
        self.assertEqual(self.client.delete(f'/api/sessions/{self.current_id}').status_code, 404)

    def test_delete_last_session_clears_pointers(self):
        mining = self.experiments / 'feature_mining'
        (mining / 'latest.json').write_text(json.dumps(self.current))
        self.assertEqual(self.client.delete(f'/api/sessions/{self.old_id}').status_code, 200)
        self.assertEqual(self.client.delete(f'/api/sessions/{self.current_id}').status_code, 200)
        self.assertEqual(self.client.get('/api/sessions').get_json(), {'sessions': [], 'current_session_id': None})
        self.assertFalse((mining / 'current.json').exists())
        self.assertFalse((mining / 'latest.json').exists())

    def test_delete_protects_running_jobs_and_cli_activity(self):
        from agent.identifiers import activity_lock
        with patch.object(runner, '_jobs', {'active': runner.Job('active', [])}):
            self.assertEqual(self.client.delete(f'/api/sessions/{self.current_id}').status_code, 409)
        with activity_lock(self.root):
            self.assertEqual(self.client.delete(f'/api/sessions/{self.current_id}').status_code, 409)
        manifest = self.artifacts / 'feature_mining' / self.current_id / 'session.json'
        manifest.write_text(json.dumps({**self.current, 'status': 'searching'}))
        self.assertEqual(self.client.delete(f'/api/sessions/{self.current_id}').status_code, 409)
        self.assertTrue(manifest.exists())

    def test_delete_rejects_symlinks_and_invalid_ids(self):
        alias = self.artifacts / 'feature_mining' / 'symlink'
        alias.symlink_to(self.artifacts / 'feature_mining' / self.current_id, target_is_directory=True)
        self.assertEqual(self.client.delete('/api/sessions/symlink').status_code, 400)
        self.assertEqual(self.client.delete('/api/sessions/bad%20id').status_code, 400)
        self.assertEqual(self.client.delete('/api/sessions/missing').status_code, 404)
        self.assertTrue((alias / 'session.json').exists())

    def test_delete_rolls_back_on_move_failure(self):
        original_rename = Path.rename
        def fail_experiments(path, destination):
            if path == self.experiments / 'feature_mining' / self.current_id:
                raise OSError('simulated failure')
            return original_rename(path, destination)
        with patch.object(Path, 'rename', fail_experiments):
            self.assertEqual(self.client.delete(f'/api/sessions/{self.current_id}').status_code, 500)
        self.assertEqual(self.client.get('/api/sessions').get_json()['current_session_id'], self.current_id)
        self.assertTrue((self.artifacts / 'feature_mining' / self.current_id / 'session.json').exists())
        self.assertTrue((self.experiments / 'feature_mining' / self.current_id / 'experiments.jsonl').exists())

    def test_delete_latest_falls_back_only_to_successful_sessions(self):
        mining = self.experiments / 'feature_mining'
        successful = {**self.old, 'accepted_recipes': ['inherited', 'new'], 'inherited_recipes': ['inherited']}
        (self.artifacts / 'feature_mining' / self.old_id / 'session.json').write_text(json.dumps(successful))
        (mining / 'latest.json').write_text(json.dumps(self.current))
        self.assertEqual(self.client.delete(f'/api/sessions/{self.current_id}').status_code, 200)
        self.assertEqual(json.loads((mining / 'latest.json').read_text()), successful)

    def test_delete_legacy_alias_preserves_id_registry(self):
        from scripts.migrate_short_ids import migrate
        result = migrate(self.root, apply=True)
        registry = self.experiments / 'id_registry.json'
        original = registry.read_bytes()
        response = self.client.delete(f'/api/sessions/{self.current_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['deleted_session_id'], result['aliases'][self.current_id])
        self.assertEqual(registry.read_bytes(), original)

    def test_delete_latest_keeps_training_only_success(self):
        successful = {**self.old, 'accepted_recipes': [], 'inherited_recipes': [],
                      'baseline_id': 'E0000001', 'champion_id': 'E0000002'}
        (self.artifacts / 'feature_mining' / self.old_id / 'session.json').write_text(json.dumps(successful))
        pointer = self.experiments / 'feature_mining/latest.json'
        pointer.write_text(json.dumps(self.current))
        self.assertEqual(self.client.delete(f'/api/sessions/{self.current_id}').status_code, 200)
        self.assertEqual(json.loads(pointer.read_text()), successful)

    def test_research_requests_are_read_only_and_session_scoped(self):
        from skills import research_extensions
        request = {'kind': 'data_source', 'reason': '研究供需', 'source': 'reviewed.supply',
                   'keys': ['order_id'], 'time_column': 'available_at',
                   'availability': '事件产生时间', 'outputs': ['supply_signal']}
        created = research_extensions.stage(self.root, self.current_id, request, {})
        response = self.client.get('/api/research-requests').get_json()
        self.assertEqual(response['requests'][0]['id'], created['id'])
        self.assertEqual(response['requests'][0]['status'], 'pending_review')
        self.assertEqual(self.client.get('/api/research-requests', query_string={'session_id': self.old_id}).get_json()['requests'], [])
        self.assertEqual(self.client.post('/api/research-requests', json={'decision': 'approve'}).status_code, 405)
        self.assertEqual(self.client.get('/api/discovery').get_json()['extension_requests'], response['requests'])

    def test_migrated_ids_keep_old_links_and_actual_lineage(self):
        from scripts.migrate_short_ids import migrate
        result = migrate(self.root, apply=True)
        short_session = result['aliases'][self.current_id]
        short_champion = result['aliases'][self.current['champion_id']]
        sessions = self.client.get('/api/sessions').get_json()
        self.assertEqual(sessions['current_session_id'], short_session)
        self.assertTrue(all(len(session['session_id']) <= 8 for session in sessions['sessions']))
        for identifier in (self.current_id, short_session):
            response = self.client.get('/api/champion', query_string={'session_id': identifier})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['champion']['experiment_id'], short_champion)
        for identifier in (self.current['champion_id'], short_champion):
            response = self.client.get('/api/experiments/' + identifier)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['record']['experiment_id'], short_champion)
        lineages = self.client.get('/api/lineage', query_string={'session_id': short_session}).get_json()['lineages']
        self.assertEqual(len(lineages), 1)
        self.assertEqual(lineages[0]['session_id'], short_session)
        nodes = lineages[0]['nodes']
        self.assertTrue(all(node['short_label'] == node['id'] and len(node['id']) <= 8 for node in nodes))
        self.assertEqual(nodes[2]['parent_id'], nodes[1]['id'])

    def test_chat_includes_current_context_and_bounded_history(self):
        with patch.dict(os.environ, {'ETS_LLM_API_KEY': 'test-only'}), patch('webui.app.chat_requests.run') as client:
            client.return_value = SimpleNamespace(text='回答', usage={}, finish_reason='stop')
            response = self.client.post('/api/chat', json={'question': '为什么被拒绝？',
                'session_id': self.current_id, 'history': [{'role': 'user', 'content': '上一问'}]})
            self.assertEqual(response.status_code, 200)
            messages = client.call_args.args[2]
            context = json.loads(messages[0]['content'].split('当前会话上下文：\n')[1])
            self.assertEqual(context['champion']['experiment_id'], self.current['champion_id'])
            self.assertEqual(context['search_feedback'][0]['rejected'], 'sparse feature')
            self.assertFalse(context['mock'])
            self.assertEqual(messages[1], {'role': 'user', 'content': '上一问'})
            self.assertNotIn('legacy_wrong_champion', messages[0]['content'])
        invalid = self.client.post('/api/chat', json={'question': 'test', 'history': [{'role': 'system', 'content': 'bad'}]})
        self.assertEqual(invalid.status_code, 400)

    def test_create_chat_session_is_empty_and_does_not_change_experiment(self):
        pointer = self.experiments / 'feature_mining/current.json'
        original = pointer.read_bytes()
        with patch.object(runner, 'start_orchestrator') as start:
            response = self.client.post('/api/sessions', json={})
        self.assertEqual(response.status_code, 201)
        session = response.get_json()['session']
        self.assertRegex(session['session_id'], r'^S[0-9A-Z]{7}$')
        self.assertEqual(session['mode'], 'chat')
        self.assertEqual(pointer.read_bytes(), original)
        self.assertFalse((self.experiments / 'feature_mining' / session['session_id']).exists())
        start.assert_not_called()
        self.assertIn(session['session_id'], [item['session_id'] for item in self.client.get('/api/sessions').get_json()['sessions']])
        query = {'session_id': session['session_id']}
        self.assertIsNone(self.client.get('/api/champion', query_string=query).get_json()['champion'])
        self.assertEqual(self.client.get('/api/experiments', query_string=query).get_json()['records'], [])
        self.assertEqual(self.client.delete(f'/api/sessions/{session["session_id"]}').status_code, 200)
        self.assertTrue((self.artifacts / 'feature_mining' / self.current_id).exists())

    def test_chat_sessions_never_replace_experiment_current_pointer(self):
        session = self.client.post('/api/sessions').get_json()['session']
        self.assertEqual(self.client.delete(f'/api/sessions/{self.current_id}').status_code, 200)
        self.assertEqual(self.client.get('/api/sessions').get_json()['current_session_id'], self.old_id)
        self.assertTrue((self.artifacts / 'feature_mining' / session['session_id']).exists())

    def test_cancellable_chat_and_empty_chat_context(self):
        from webui import chat_requests
        session = self.client.post('/api/sessions').get_json()['session']
        with patch.dict(os.environ, {'ETS_LLM_API_KEY': 'test-only'}), \
             patch.object(chat_requests, 'run', return_value=SimpleNamespace(text='你好', usage={}, finish_reason='stop')) as run:
            response = self.client.post('/api/chat', json={'question': '你好', 'session_id': session['session_id'],
                                                         'request_id': 'request_frontend_001'})
            self.assertTrue(response.get_json()['ok'])
            self.assertEqual(run.call_args.args[0], 'request_frontend_001')
            context = json.loads(run.call_args.args[2][0]['content'].split('当前会话上下文：\n')[1])
            self.assertEqual(context['mode'], 'chat')
            self.assertIsNone(context['champion'])
            self.assertIn('research', run.call_args.kwargs)
            run.side_effect = chat_requests.ChatCancelled()
            response = self.client.post('/api/chat', json={'question': '你好', 'request_id': 'request_frontend_002'})
            self.assertTrue(response.get_json()['cancelled'])
        with patch.object(chat_requests, 'cancel', return_value='cancelling') as cancel:
            response = self.client.post('/api/chat/request_frontend_001/cancel')
            self.assertTrue(response.get_json()['ok'])
            cancel.assert_called_once_with('request_frontend_001')
        self.assertEqual(self.client.post('/api/chat/bad/cancel').status_code, 400)
        self.assertEqual(self.client.post('/api/chat', json={'question': '你好', 'request_id': 'bad'}).status_code, 400)

    def test_chat_uses_configured_limit_and_reports_truncation(self):
        previous = '历史内容' * 2000 + '结尾不能丢失'
        answer = '| LONG_WAIT | 206 | 0.0000'
        with patch.dict(os.environ, {'ETS_LLM_API_KEY': 'test-only'}), \
             patch('webui.app.chat_requests.run') as client:
            for finish in ('length', 'stop', None):
                client.return_value = SimpleNamespace(text=answer, usage={}, finish_reason=finish)
                response = self.client.post('/api/chat', json={'question': '继续回答', 'session_id': self.current_id,
                    'history': [{'role': 'assistant', 'content': previous}]})
                self.assertEqual(response.status_code, 200)
                result = response.get_json()
                self.assertEqual(result['answer'], answer)
                self.assertEqual(result['truncated'], finish == 'length')
                self.assertEqual(result['finish_reason'], finish)
                self.assertEqual(result['max_tokens'], client.call_args.args[1]['max_tokens'])
                self.assertNotIn('max_tokens', client.call_args.kwargs)
                self.assertEqual(client.call_args.args[2][1]['content'], previous)

    def test_chat_rejects_oversized_history_without_silent_truncation(self):
        response = self.client.post('/api/chat', json={'question': '继续',
            'history': [{'role': 'assistant', 'content': 'a' * 120001}]})
        self.assertEqual(response.status_code, 400)
        response = self.client.post('/api/chat', json={'question': '继续',
            'history': [{'role': 'assistant', 'content': 'a' * 100000}] * 3})
        self.assertEqual(response.status_code, 400)

    def test_run_input_validation_and_parameter_forwarding(self):
        for body in ([], {'rounds': 'bad'}, {'rounds': 1.5}, {'rounds': True}, {'mock': 'false'},
                     {'profile_only': 'true'}, {'sample_rows': -1}, {'sample_rows': 250001},
                     {'sample_rows': True}, {'sample_rows': 1.5}, {'sample_rows': 8000}):
            self.assertEqual(self.client.post('/api/run', json=body).status_code, 400)
        with patch.object(runner, 'start_orchestrator', return_value='test_job') as start:
            response = self.client.post('/api/run', json={'rounds': 2, 'max_proposals': 3,
                'mock': False, 'no_agent': True, 'resume_recipes': True})
            self.assertEqual(response.get_json()['job_id'], 'test_job')
            start.assert_called_once_with(rounds=2, max_proposals=3, mock=False, no_agent=True, resume_recipes=True,
                                          sample_rows=0, profile_only=False, continuous=False,
                                          search_timeout_minutes=120, search_patience=9, research_skills=False, model_study=False, web_research=False, llm_profile='chat',
                                          data_source='online', snapshot_id=None, training_mode='pooled', model_family='xgb')

    def test_continuous_run_api_validates_and_forwards_budget(self):
        with patch.object(runner, 'start_orchestrator', return_value='continuous_job') as start:
            response = self.client.post('/api/run', json={'rounds': 5, 'max_proposals': 3,
                'continuous': True, 'search_timeout_minutes': 60, 'search_patience': 8, 'research_skills': True})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(start.call_args.kwargs['continuous'])
            self.assertTrue(start.call_args.kwargs['research_skills'])
            self.assertEqual(start.call_args.kwargs['search_timeout_minutes'], 60)
            self.assertEqual(start.call_args.kwargs['search_patience'], 8)
        for body in ({'continuous': 'yes'}, {'research_skills': 'yes'}, {'search_timeout_minutes': 0}, {'search_patience': True}, {'search_patience': 161}):
            self.assertEqual(self.client.post('/api/run', json=body).status_code, 400)

    def test_model_selection_for_chat_and_run_is_request_scoped(self):
        before = (self.module.CONFIG_DIR / 'llm.json').read_bytes()
        options = self.client.get('/api/llm-models').get_json()
        self.assertEqual({item['profile_id'] for item in options['profiles']}, {'chat', 'responses'})
        with patch.object(runner, 'start_orchestrator', return_value='selected_job') as start:
            result = self.client.post('/api/run', json={'llm_profile': 'responses'})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(start.call_args.kwargs['llm_profile'], 'responses')
        with patch.dict(os.environ, {'ETS_LLM_API_KEY': '', 'ETS_GPT_API_KEY': 'test-gpt'}), \
             patch('webui.app.chat_requests.run') as client:
            client.return_value = SimpleNamespace(text='GPT回答', usage={}, finish_reason='stop')
            result = self.client.post('/api/chat', json={'question': '你好', 'llm_profile': 'responses'})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(client.call_args.args[1]['protocol'], 'responses')
            self.assertEqual(result.get_json()['llm']['model'], 'YOUR_RESPONSES_MODEL')
            self.assertEqual(self.client.post('/api/chat', json={'question': '你好', 'llm_profile': 'chat'}).status_code, 400)
        for value in ('unknown', {}, True):
            self.assertEqual(self.client.post('/api/run', json={'llm_profile': value}).status_code, 400)
            self.assertEqual(self.client.post('/api/chat', json={'question': '你好', 'llm_profile': value}).status_code, 400)
        self.assertEqual((self.module.CONFIG_DIR / 'llm.json').read_bytes(), before)

    def test_cross_session_history_is_read_only_labelled_and_optional(self):
        historical = self.make_session('S0000010', False, .55)
        pointer = (self.experiments / 'feature_mining/current.json').read_bytes()
        with patch.dict(os.environ, {'ETS_LLM_API_KEY': 'test-only'}), patch('webui.app.chat_requests.run') as client:
            client.return_value = SimpleNamespace(text='历史分析', usage={}, finish_reason='stop')
            response = self.client.post('/api/chat', json={'question': '看看 S0000010', 'session_id': self.current_id})
            context = json.loads(client.call_args.args[2][0]['content'].split('当前会话上下文：\n')[1])
            self.assertEqual(context['champion']['experiment_id'], self.current['champion_id'])
            self.assertEqual({record['session_id'] for record in context['cross_session_history']['records']}, {'S0000010'})
            self.assertTrue(response.get_json()['history_sources'])
            self.assertEqual((self.experiments / 'feature_mining/current.json').read_bytes(), pointer)
            response = self.client.post('/api/chat', json={'question': '看看 S0000010', 'session_id': self.current_id, 'include_history': False})
            self.assertEqual(response.get_json()['history_sources'], [])
        response = self.client.get('/api/research-history', query_string={'q': 'S0000010', 'session_id': self.current_id})
        self.assertEqual(len(response.get_json()['records']), 3)
        self.assertEqual(self.client.get('/api/research-history', query_string={'q': 'E0000999'}).get_json()['records'], [])
        self.assertEqual(self.client.post('/api/chat', json={'question': '历史', 'include_history': 'yes'}).status_code, 400)

    def test_real_profile_api_forwards_explicit_sample(self):
        with patch.object(runner, 'start_orchestrator', return_value='profile_job') as start:
            response = self.client.post('/api/run', json={'mock': False, 'sample_rows': 0,
                                                         'profile_only': True})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(start.call_args.kwargs['mock'])
        self.assertEqual(start.call_args.kwargs['sample_rows'], 0)
        self.assertTrue(start.call_args.kwargs['profile_only'])

    def test_champion_reports_new_internal_split_and_keeps_old_split(self):
        response = self.client.get('/api/champion').get_json()
        self.assertEqual(response['evaluation_split'], 'validation')
        with patch.object(self.module, '_current_champion', return_value={
                'config_diff': {'evaluation_split': 'train_internal_early_stop'}}):
            response = self.client.get('/api/champion').get_json()
        self.assertEqual(response['evaluation_split'], 'train_internal_early_stop')

    def test_sse_endpoint_honors_last_event_id(self):
        job = runner.Job('test_stream', [])
        job.emit('first')
        job.emit('second')
        job.ended_at = 1
        job.returncode = 0
        with patch.object(runner, '_jobs', {'test_stream': job}):
            response = self.client.get('/api/run/test_stream/stream', headers={'Last-Event-ID': '1'})
            text = response.get_data(as_text=True)
            self.assertNotIn('data: first', text)
            self.assertIn('data: second', text)
            self.assertIn('event: done', text)
            self.assertEqual(self.client.get('/api/run/missing/stream').status_code, 404)


class FrontendTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for frontend regression tests')
    def test_frontend_controls(self):
        path = Path(__file__).with_name('test_webui_controls.js')
        result = subprocess.run(['node', str(path)], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
