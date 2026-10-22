import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from webui import chat_store, recover_chat_logs


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.artifacts = self.root / 'artifacts'
        for sid in ('S0000001', 'S0000002'):
            directory = self.artifacts / 'feature_mining' / sid
            directory.mkdir(parents=True); (directory / 'session.json').write_text(json.dumps({'session_id': sid}))

    def begin(self, identifier='request_id_0000001', sid='S0000001', question='问题' * 2000):
        body = {'question': question, 'session_id': sid, 'history': []}
        return chat_store.begin(self.artifacts, sid, identifier, body)

    def test_full_answer_survives_real_new_process_and_sessions_stay_isolated(self):
        item, created = self.begin(); self.assertTrue(created)
        answer = '完整回答：\n' * 12000
        result = {'ok': True, 'answer': answer, 'llm': {'model': 'test-model'},
                  'web_research': {'sources': [{'title': '论文', 'url': 'https://example.com/paper'}]},
                  'truncated': True, 'history_sources': [{'session_id': 'S0000002', 'experiment_id': 'E0000001'}]}
        chat_store.finish(self.artifacts, 'S0000001', item['request_id'], result)
        output = subprocess.check_output([sys.executable, '-c',
            'import json,sys; from webui.chat_store import transcript; print(json.dumps(transcript(sys.argv[1], "S0000001")))', str(self.artifacts)], text=True)
        saved = json.loads(output)
        self.assertEqual(saved['turns'][0]['result'], result)
        self.assertEqual(saved['turns'][0]['question'], '问题' * 2000)
        self.assertEqual(chat_store.transcript(self.artifacts, 'S0000002')['turns'], [])
        path = self.artifacts / 'feature_mining/S0000001/chat' / (item['request_id'] + '.json')
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn('owner', saved['turns'][0]); self.assertNotIn('fingerprint', saved['turns'][0])

    def test_pending_restart_is_interrupted_not_reexecuted_and_cancellation_wins(self):
        item, _ = self.begin()
        self.assertTrue(chat_store.session_busy(self.artifacts, 'S0000001'))
        with patch.object(chat_store, 'BOOT_ID', 'new-service'):
            saved = chat_store.find(self.artifacts, item['request_id'])
            self.assertEqual(saved['status'], 'interrupted')
            self.assertIn('重启', saved['result']['error'])
            self.assertFalse(self.begin()[1])
            self.assertFalse(chat_store.session_busy(self.artifacts, 'S0000001'))
        chat_store.cancel(self.artifacts, item['request_id'])
        chat_store.finish(self.artifacts, 'S0000001', item['request_id'], {'ok': True, 'answer': 'late answer'})
        self.assertEqual(chat_store.find(self.artifacts, item['request_id'])['status'], 'cancelled')
        self.assertNotIn('answer', chat_store.find(self.artifacts, item['request_id'])['result'])

    def test_concurrent_retries_create_only_one_turn_and_conflicts_fail(self):
        results = []
        workers = [threading.Thread(target=lambda: results.append(self.begin()[1])) for _ in range(8)]
        for worker in workers: worker.start()
        for worker in workers: worker.join()
        self.assertEqual(sum(results), 1)
        with self.assertRaises(ValueError): self.begin(sid='S0000002')
        with self.assertRaises(ValueError): self.begin(question='different')

    def test_pagination_context_limit_and_bad_file_do_not_destroy_history(self):
        for i in range(35):
            item, _ = self.begin(identifier=f'request_id_{i:08d}', question=f'Q{i}')
            chat_store.finish(self.artifacts, 'S0000001', item['request_id'], {'ok': True, 'answer': f'A{i}'})
        page = chat_store.transcript(self.artifacts, 'S0000001')
        self.assertEqual(len(page['turns']), 30); self.assertEqual(len(page['history']), 12)
        older = chat_store.transcript(self.artifacts, 'S0000001', before=page['next_before'])
        self.assertEqual(len(older['turns']), 5); self.assertIsNone(older['next_before'])
        (self.artifacts / 'feature_mining/S0000001/chat/bad_record.json').write_text('{')
        page = chat_store.transcript(self.artifacts, 'S0000001')
        self.assertEqual(page['skipped'], 1); self.assertEqual(page['total'], 35)
        with self.assertRaises(ValueError): chat_store.transcript(self.artifacts, '../secret')
        with self.assertRaises(ValueError): chat_store.transcript(self.artifacts, 'S0000001', before='../secret')

    def test_symlink_rejection_and_save_failure_do_not_claim_success(self):
        folder = self.artifacts / 'feature_mining/S0000001/chat'; folder.symlink_to(self.root)
        with self.assertRaises(ValueError): self.begin()
        with patch.object(chat_store, '_save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.begin(sid='S0000002')

    def test_legacy_import_is_labelled_idempotent_and_never_guesses_unscoped_questions(self):
        folder = self.root / 'webui/logs'; folder.mkdir(parents=True)
        (folder / 'session_20241009_175535.log').write_text(
            '[17:58:13] [user] 无会话标签的问题\n'
            '[17:58:17] [会话 S0000001] [assistant] 回答第一行\n第二行\n'
            '[17:58:18] [会话 S0000002] [user] 另一会话提问\n'
            '[17:58:19] [assistant] 没有标签的回答\n', encoding='utf-8')
        self.assertEqual(recover_chat_logs.recover(self.root)['new_fragments'], 0)
        self.assertEqual(recover_chat_logs.recover(self.root, True)['new_fragments'], 2)
        self.assertEqual(recover_chat_logs.recover(self.root, True)['new_fragments'], 0)
        first = chat_store.transcript(self.artifacts, 'S0000001')
        self.assertEqual(first['turns'][0]['result']['answer'], '回答第一行\n第二行')
        self.assertIsNone(first['turns'][0]['question']); self.assertTrue(first['turns'][0]['recovered'])
        self.assertEqual(first['history'], [])


if __name__ == '__main__': unittest.main()
