import json
from pathlib import Path
import tempfile
import unittest

from agent.experiment_history import query_experiments
from agent.research_controller import ResearchController


def plan(action='explore'):
    return {'action': action, 'direction': 'supply_demand', 'reason': '验证供需假设', 'next_if_failed': '改查历史覆盖'}


class ResearchControllerTests(unittest.TestCase):
    def test_budget_and_stagnation_stop_are_bounded(self):
        controller = ResearchController(3, patience=9)
        for _ in range(3):
            controller.record(plan(), 'completed')
        self.assertTrue(controller.stopped())
        self.assertIn('尝试预算', controller.stop_reason)
        controller = ResearchController(20, patience=2)
        controller.record(plan(), 'completed')
        controller.record(plan(), 'completed', improved=True)
        self.assertEqual(controller.no_improvement, 0)
        controller.record(plan(), 'completed')
        controller.record(plan(), 'completed')
        self.assertIn('连续 2', controller.stop_reason)

    def test_timeout_prevents_new_work_without_killing_running_work(self):
        now = [0]
        controller = ResearchController(10, timeout_minutes=1, clock=lambda: now[0])
        self.assertFalse(controller.stopped())
        now[0] = 61
        self.assertTrue(controller.stopped())
        self.assertEqual(controller.snapshot()['remaining_seconds'], 0)

    def test_failure_repair_is_limited_then_changes_direction(self):
        controller = ResearchController(20)
        with self.assertRaises(ValueError):
            controller.validate({'plan': plan('repair'), 'proposals': [{}]})
        controller.record(plan(), 'execution_failed')
        self.assertTrue(controller.snapshot()['repair_allowed'])
        controller.record(plan('repair'), 'execution_failed')
        self.assertTrue(controller.snapshot()['repair_allowed'])
        controller.record(plan('repair'), 'execution_failed')
        self.assertFalse(controller.snapshot()['repair_allowed'])
        self.assertEqual(controller.snapshot()['recommended_action'], 'explore')

    def test_completed_no_gain_is_not_execution_failure(self):
        controller = ResearchController(20)
        controller.record(plan(), 'completed')
        controller.record(plan(), 'completed')
        self.assertFalse(controller.snapshot()['repair_allowed'])
        self.assertNotEqual(controller.snapshot()['recommended_direction'], 'supply_demand')

    def test_single_proposal_and_explicit_stop_contract(self):
        controller = ResearchController(10)
        for proposals in ([], [{}, {}]):
            with self.assertRaises(ValueError):
                controller.validate({'plan': plan(), 'proposals': proposals})
        self.assertEqual(controller.validate({'plan': plan('stop'), 'proposals': []})[1], [])
        for _ in range(3):
            controller.record({}, 'planning_failed')
        self.assertIn('规划失败', controller.stop_reason)


class ExperimentHistoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifacts, self.experiments = self.root / 'artifacts', self.root / 'experiments'
        for session_id, experiment_id, mock in [('S0000001', 'E0000001', False), ('S0000002', 'E0000002', True)]:
            folder = self.artifacts / 'feature_mining' / session_id
            folder.mkdir(parents=True)
            (folder / 'session.json').write_text(json.dumps({'session_id': session_id, 'mock': mock,
                'window': {'test_start': '2024-09-01'}, 'evaluation_split': 'test'}))
            logs = self.experiments / 'feature_mining' / session_id
            logs.mkdir(parents=True)
            (logs / 'experiments.jsonl').write_text(json.dumps({'experiment_id': experiment_id,
                'hypothesis': 'history_coverage', 'metrics_by_city': {'xgb_44': {'ALL': {'recall_ratio': .5}}},
                'config_diff': {'feature_recipes': []}, 'gate_pass': False}) + '\nmalformed\n')

    def query(self, question='', session=None, **kwargs):
        return query_experiments(self.artifacts, self.experiments,
                                 session or {'session_id': 'S0000003', 'mock': False}, question, **kwargs)

    def test_default_same_mode_and_explicit_id_with_provenance(self):
        self.assertEqual([entry['session_id'] for entry in self.query()['records']], ['S0000001'])
        results = self.query('看看 S0000002')['records']
        self.assertEqual(results[0]['experiment_id'], 'E0000002')
        self.assertTrue(results[0]['mock'])
        self.assertIn('不是当前会话', results[0]['scope_note'])
        self.assertEqual(self.query('看看S0000002这个会话')['requested_ids'], ['S0000002'])
        self.assertEqual(self.query('看看E0000002的结果')['records'][0]['session_id'], 'S0000002')
        self.assertEqual(self.query('E0000999')['records'], [])

    def test_archived_sessions_not_returned_and_budget_enforced(self):
        self.assertEqual(self.query(max_chars=1)['records'], [])
        (self.artifacts / 'feature_mining/S0000001').rename(self.root / 'archived')
        self.assertEqual(self.query()['records'], [])

    def test_chat_session_can_read_both_modes_without_adopting_champion(self):
        results = self.query(session={'session_id': 'S0000003', 'mock': None})
        self.assertEqual(len(results['records']), 2)
        self.assertNotIn('champion', results)


if __name__ == '__main__':
    unittest.main()
