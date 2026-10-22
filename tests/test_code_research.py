import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from skills import research_agent as ra, experiment_research as er
from test_autonomous_research import FakeModel, reply


class CodeResearchTests(unittest.TestCase):
    def test_chat_repairs_same_path_and_can_rerun_identical_tool_arguments(self):
        with tempfile.TemporaryDirectory() as root:
            box = ra.ResearchTools(root, 'S1', time.monotonic() + 60)
            model = FakeModel([
                reply([('write_file', {'path': 'calc.py', 'content': 'print(missing)'})]),
                reply([('run_file', {'path': 'calc.py', 'timeout_seconds': 10})]),
                reply([('write_file', {'path': 'calc.py', 'content': 'print(42)'})]),
                reply([('run_file', {'path': 'calc.py', 'timeout_seconds': 10})]), reply(text='修复后输出42')])
            with patch('skills.code_workspace.run_file', side_effect=[
                    {'status': 'failed', 'exit_code': 1, 'stderr': 'NameError', 'run_id': 'first'},
                    {'status': 'success', 'exit_code': 0, 'stdout': '42', 'run_id': 'second'}]):
                response = ra.run(root, {}, [], session_id='S1', toolbox=box, model=model)
            report = response.web_research
            self.assertEqual(len(report['code_runs']), 2)
            self.assertEqual(report['trace'][1]['status'], 'failed')
            self.assertEqual(report['trace'][3]['status'], 'ok')
            self.assertNotIn('content', report['trace'][0]['arguments'])
            self.assertIn('NameError', str(model.inputs[2]))
            self.assertEqual((Path(root)/'research_workspace/generated/S1/files/calc.py').read_text(), 'print(42)')

    def test_experiment_research_can_create_verify_and_persist_code_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            model = FakeModel([
                reply([('write_file', {'path': 'method.py', 'content': 'assert 1 + 1 == 2'})]),
                reply([('run_file', {'path': 'method.py', 'timeout_seconds': 10})]),
                reply(text='小型计算已验证；不代表等待时间真实效果提升')])
            session = er.ExperimentResearchSession(root, Path(root) / 'reports/S1', {})
            with patch.object(ra, 'ToolModel', return_value=model), patch('skills.code_workspace.run_file', return_value={
                    'status': 'success', 'exit_code': 0, 'stdout': '', 'run_id': 'check1'}):
                report = session.update(1, {'task': 'verify method'})
            self.assertEqual(report['decision'], 'verified')
            self.assertEqual(report['network_calls'], 0)
            self.assertEqual(report['code_runs'][0]['run_id'], 'check1')
            saved = json.loads((Path(root)/'reports/S1/web_research_01.json').read_text())
            self.assertEqual(saved['generated_files'][0]['path'], 'files/method.py')
            self.assertEqual(saved['summary'], report['summary'])

    def test_tool_schema_contains_file_execution_for_both_paths(self):
        self.assertTrue({'write_file', 'run_file', 'list_workspace', 'read_workspace_file'} <= {t['name'] for t in ra.TOOLS})
        self.assertTrue(issubclass(er.ExperimentTools, ra.ResearchTools))
