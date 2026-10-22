import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from webui import chat_store, chat_jobs
from tests import test_webui as fixtures


class PersistenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.WebApiTests.setUpClass.__func__(cls)
    make_session = fixtures.WebApiTests.make_session

    def setUp(self):
        fixtures.WebApiTests.setUp(self)
        log = patch.object(self.module, 'SESSION_LOG_PATH', self.root / 'test.log'); log.start(); self.addCleanup(log.stop)

    def wait_result(self, request_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.client.get('/api/chat/' + request_id).get_json()
            if job['status'] != 'running': return job
            time.sleep(.01)
        self.fail('background request did not finish')

    def test_chat_reply_is_durable_and_retry_after_restart_does_not_call_model(self):
        body = {'question': '完整问题' * 1000, 'session_id': self.current_id,
                'request_id': 'persist_request_00001', 'background': True, 'history': []}
        answer = '完整回复' * 5000
        with patch.dict(os.environ, {'AUTOLAB_LLM_API_KEY': 'test-only'}), \
             patch.object(chat_jobs, '_jobs', {}), patch.object(chat_jobs, '_workers', set()), \
             patch('webui.app.chat_requests.run', return_value=SimpleNamespace(text=answer, usage={}, finish_reason='length')) as model:
            self.assertEqual(self.client.post('/api/chat', json=body).status_code, 202)
            result = self.wait_result(body['request_id'])
            self.assertEqual(result['result']['answer'], answer)
            # Clearing process-memory jobs simulates restart; disk is authoritative.
            chat_jobs._jobs.clear()
            self.assertEqual(self.client.get('/api/chat/' + body['request_id']).get_json()['result']['answer'], answer)
            with patch.dict(os.environ, {'AUTOLAB_LLM_API_KEY': ''}):
                retry = self.client.post('/api/chat', json=body)
            self.assertEqual(retry.status_code, 202); self.assertEqual(model.call_count, 1)
            transcript = self.client.get('/api/sessions/' + self.current_id + '/chat').get_json()
            self.assertEqual(transcript['turns'][0]['question'], body['question'])
            self.assertEqual(transcript['turns'][0]['result']['answer'], answer)
            self.assertTrue(transcript['turns'][0]['result']['truncated'])
            self.assertEqual(self.client.get('/api/sessions/' + self.old_id + '/chat').get_json()['total'], 0)
            self.assertEqual(self.client.post('/api/chat', json={**body, 'question': 'changed'}).status_code, 409)

    def test_storage_failure_prevents_model_and_delete_archives_transcript(self):
        with patch.dict(os.environ, {'AUTOLAB_LLM_API_KEY': 'test-only'}), \
             patch.object(chat_store, '_save', side_effect=OSError('disk full')), \
             patch('skills.llm_client.LLMClient') as model:
            response = self.client.post('/api/chat', json={'question': 'test', 'session_id': self.current_id})
            self.assertEqual(response.status_code, 503); model.assert_not_called()
        item, _ = chat_store.begin(self.artifacts, self.current_id, 'persist_request_delete',
                                   {'question': 'preserve', 'session_id': self.current_id})
        self.assertEqual(self.client.delete('/api/sessions/' + self.current_id).status_code, 409)
        chat_store.finish(self.artifacts, self.current_id, item['request_id'], {'ok': True, 'answer': 'saved'})
        self.assertEqual(self.client.delete('/api/sessions/' + self.current_id).status_code, 200)
        archived = list((self.experiments / '.session_trash').glob('*/artifacts/chat/*.json'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(json.loads(archived[0].read_text())['question'], 'preserve')
        self.assertEqual(self.client.get('/api/sessions/' + self.current_id + '/chat').status_code, 404)

    def test_restart_pending_request_is_visible_and_never_restarts_automatically(self):
        body = {'question': '未完成的问题', 'session_id': self.current_id, 'history': [],
                'request_id': 'persist_request_pending', 'background': True}
        chat_store.begin(self.artifacts, self.current_id, body['request_id'], body)
        with patch.object(chat_store, 'BOOT_ID', 'different-process'), patch('webui.app.chat_requests.run') as model:
            response = self.client.post('/api/chat', json=body)
            self.assertEqual(response.status_code, 202)
            self.assertIn('重启', response.get_json()['result']['error'])
            transcript = self.client.get('/api/sessions/' + self.current_id + '/chat').get_json()
            self.assertEqual(transcript['turns'][0]['status'], 'interrupted')
            model.assert_not_called()


if __name__ == '__main__': unittest.main()
