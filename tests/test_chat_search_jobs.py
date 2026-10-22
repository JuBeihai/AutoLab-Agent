"""Chat entry point delegates search decisions to the model-owned tool loop."""
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from webui import chat_jobs, chat_requests
from tests import test_webui as fixtures


class ChatSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.WebApiTests.setUpClass.__func__(cls)
    def setUp(self):
        fixtures.WebApiTests.setUp(self)
        log = patch('webui.app.SESSION_LOG_PATH', self.root / 'chat-test.log')
        log.start(); self.addCleanup(log.stop)
    make_session = fixtures.WebApiTests.make_session

    def wait_result(self, request_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.client.get('/api/chat/' + request_id).get_json()
            if job['status'] != 'running': return job
            time.sleep(.01)
        self.fail('job did not finish')

    def test_model_owns_search_and_same_id_never_repeats_work(self):
        report = {'operation': 'agent', 'status': 'ok', 'sources': [],
                  'trace': [{'step': 1, 'tool': 'search_web', 'arguments': {'query': 'model chose this'}, 'status': 'ok'}]}
        request_id = 'autonomous_chat_search_001'
        body = {'question': '那篇论文再深入看看', 'session_id': self.current_id,
                'request_id': request_id, 'background': True}
        with patch.object(chat_jobs, '_jobs', {}), patch.object(chat_jobs, '_workers', set()), \
             patch.dict(os.environ, {'AUTOLAB_LLM_API_KEY': 'test-only'}), \
             patch('webui.app.chat_research.plan') as old_plan, \
             patch('webui.app.chat_requests.run', return_value=SimpleNamespace(text='依据正文', usage={}, finish_reason='stop', web_research=report)) as model:
            self.assertEqual(self.client.post('/api/chat', json=body).status_code, 202)
            result = self.wait_result(request_id)
            self.assertTrue(result['result']['ok'])
            self.assertEqual(result['result']['web_research'], report)
            self.assertEqual(model.call_args.kwargs['research']['session_id'], self.current_id)
            self.assertNotIn('没有独立工具调用接口', model.call_args.args[2][0]['content'])
            old_plan.assert_not_called()
            self.client.post('/api/chat', json=body)
            self.assertEqual(model.call_count, 1)
            self.assertEqual(self.client.post('/api/chat', json={**body, 'question': '不同问题'}).status_code, 409)

    def test_cancellation_applies_to_entire_research_job(self):
        request_id = 'autonomous_chat_cancel_001'
        started = threading.Event(); release = threading.Event(); self.addCleanup(release.set)
        def worker(*args, **kwargs):
            started.set(); release.wait(3)
            if chat_requests.is_cancelled(request_id): raise chat_requests.ChatCancelled()
            return SimpleNamespace(text='late', usage={}, finish_reason='stop')
        with patch.object(chat_jobs, '_jobs', {}), patch.object(chat_jobs, '_workers', set()), \
             patch.dict(os.environ, {'AUTOLAB_LLM_API_KEY': 'test-only'}), \
             patch('webui.app.chat_requests.run', side_effect=worker):
            self.client.post('/api/chat', json={'question': '研究公开论文', 'session_id': self.current_id,
                'request_id': request_id, 'background': True})
            self.assertTrue(started.wait(2))
            self.assertEqual(self.client.post('/api/chat/' + request_id + '/cancel').status_code, 200)
            release.set()
            self.assertEqual(self.wait_result(request_id)['status'], 'cancelled')
            deadline=time.monotonic()+3
            while chat_jobs._workers and time.monotonic()<deadline: time.sleep(.01)

    def test_plain_chat_still_enters_model_and_does_not_invent_research(self):
        with patch.dict(os.environ, {'AUTOLAB_LLM_API_KEY': 'test-only'}), \
             patch('webui.app.chat_research.plan') as old_plan, \
             patch('webui.app.chat_requests.run', return_value=SimpleNamespace(text='解释', usage={}, finish_reason='stop', web_research=None)) as model:
            result = self.client.post('/api/chat', json={'question': '解释准确率，不要联网', 'session_id': self.current_id}).get_json()
            self.assertTrue(result['ok']); self.assertIsNone(result['web_research'])
            old_plan.assert_not_called(); model.assert_called_once()

    def test_worker_error_is_explicit_and_not_rule_fallback(self):
        with patch.dict(os.environ, {'AUTOLAB_LLM_API_KEY': 'test-only'}), \
             patch('webui.app.chat_research.collect') as old_collect, \
             patch('webui.app.chat_requests.run', side_effect=RuntimeError('工具调用接口失败')):
            result = self.client.post('/api/chat', json={'question': '研究论文', 'session_id': self.current_id}).get_json()
            self.assertFalse(result['ok']); old_collect.assert_not_called()
