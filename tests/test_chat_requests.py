import io
import json
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from webui import chat_requests


class ChatRequestTests(unittest.TestCase):
    def setUp(self):
        self.active = patch.object(chat_requests, '_active', {})
        self.recent = patch.object(chat_requests, '_recent', {})
        self.active.start()
        self.recent.start()
        self.addCleanup(self.active.stop)
        self.addCleanup(self.recent.stop)

    def test_completed_worker_returns_large_result_and_releases_slot(self):
        script = 'import json; print(json.dumps({"text": "回答" * 50000, "usage": {}, "finish_reason": "length"}))'
        response = chat_requests.run('request_completed_001', {}, [], command=[sys.executable, '-c', script])
        self.assertEqual(len(response.text), 100000)
        self.assertEqual(response.finish_reason, 'length')
        self.assertFalse(chat_requests._active)
        self.assertEqual(chat_requests.cancel('request_completed_001'), 'finished')

    def test_cancel_terminates_only_target_worker_and_cleans_up(self):
        started = threading.Event()
        processes = []
        errors = []
        original = subprocess.Popen
        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            started.set()
            return process
        def work():
            try:
                chat_requests.run('request_cancelled_001', {}, [],
                                  command=[sys.executable, '-c', 'import time; time.sleep(60)'])
            except Exception as error:
                errors.append(error)
        untouched = threading.Event()
        chat_requests._active['another_request_001'] = untouched
        with patch.object(chat_requests.subprocess, 'Popen', side_effect=launch):
            thread = threading.Thread(target=work)
            thread.start()
            self.assertTrue(started.wait(timeout=5))
            try:
                self.assertEqual(chat_requests.cancel('request_cancelled_001'), 'cancelling')
            finally:
                chat_requests.cancel('request_cancelled_001')
                thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(processes[0].poll())
        self.assertIsInstance(errors[0], chat_requests.ChatCancelled)
        self.assertNotIn('request_cancelled_001', chat_requests._active)
        self.assertFalse(untouched.is_set())
        self.assertEqual(chat_requests.cancel('request_cancelled_001'), 'cancelled')

    def test_cancel_before_registration_prevents_upstream_request(self):
        chat_requests.cancel('request_pre_cancelled_001')
        with patch.object(chat_requests.subprocess, 'Popen') as launch:
            with self.assertRaises(chat_requests.ChatCancelled):
                chat_requests.run('request_pre_cancelled_001', {}, [])
        launch.assert_not_called()

    def test_worker_preserves_response_without_printed_logs(self):
        output = io.StringIO()
        def answer(*args):
            print('retry diagnostic')
            return SimpleNamespace(text='完成', usage={'total_tokens': 5}, finish_reason='stop')
        with patch.object(sys, 'stdin', io.StringIO(json.dumps({'config': {}, 'messages': []}))), \
             patch.object(sys, 'stdout', output), patch.object(sys, 'stderr', io.StringIO()), \
             patch('skills.llm_client.LLMClient') as client:
            client.return_value.chat.side_effect = answer
            chat_requests._worker()
        self.assertEqual(json.loads(output.getvalue())['text'], '完成')

    def test_spawn_error_invalid_id_and_duplicate_are_safe(self):
        with self.assertRaises(ValueError):
            chat_requests.cancel('../bad')
        with patch.object(chat_requests.subprocess, 'Popen', side_effect=OSError('launch failed')):
            with self.assertRaises(OSError):
                chat_requests.run('request_spawn_failed_001', {}, [])
        self.assertFalse(chat_requests._active)
        with self.assertRaisesRegex(ValueError, '已使用'):
            chat_requests.run('request_spawn_failed_001', {}, [])
