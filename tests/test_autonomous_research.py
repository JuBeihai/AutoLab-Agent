import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from skills.tool_model import ToolModel
from skills.research_agent import ResearchTools, run


def reply(calls=(), text=''):
    return SimpleNamespace(calls=[{'id': str(i), 'name': n, 'arguments': json.dumps(a)}
                                 for i, (n, a) in enumerate(calls)], text=text,
                           replay=[{'role': 'assistant', 'content': text}], usage={'total_tokens': 2})


class FakeModel:
    def __init__(self, turns): self.turns = iter(turns); self.inputs = []
    def turn(self, messages, tools, tool_choice):
        self.inputs.append((list(messages), tool_choice)); return next(self.turns)
    def output(self, call_id, result): return {'role': 'tool', 'content': json.dumps(result)}


class LoopTests(unittest.TestCase):
    def test_native_research_history_does_not_need_trigger_keywords(self):
        from webui.chat_research import recall
        report = {'mode': 'autonomous_tools', 'operation': 'agent', 'sources': [{'source_id': 'W1', 'url': 'https://example.com/paper'}]}
        with patch('webui.chat_store.transcript', return_value={'turns': [
                {'status': 'complete', 'request_id': 'request123', 'question': '这篇再深入看看', 'result': {'web_research': report}}]}):
            result = recall('/unused', 'S1')
            self.assertEqual(result['sources'], report['sources'])

    def test_model_failure_preserves_actual_tool_trace(self):
        from skills.llm_client import LLMError
        with tempfile.TemporaryDirectory() as root:
            box = ResearchTools(root, '', time.monotonic() + 60)
            box.execute = lambda *args: {'results': []}
            model = FakeModel([])
            with patch.object(model, 'turn', side_effect=[reply([('search_web', {'query': 'paper', 'engine': 'bing'})]), LLMError('连接失败')]):
                result = run(root, {}, [], model=model, toolbox=box)
            self.assertEqual(len(result.web_research['trace']), 1)
            self.assertEqual(result.web_research['stop_reason'], 'model_error')
            self.assertIn('中断', result.text)

    def test_model_rewrites_irrelevant_query_reads_source_then_answers(self):
        with tempfile.TemporaryDirectory() as root:
            box = ResearchTools(root, 'S1', time.monotonic() + 60)
            actions = []
            def execute(name, args):
                actions.append((name, args))
                if name == 'search_web': return {'results': [{'title': 'unrelated' if len(actions) == 1 else 'Official paper', 'url': 'https://example.com/paper'}]}
                return {'text': 'Evidence from the actual paper'}
            box.execute = execute
            model = FakeModel([reply([('search_web', {'query': '中文查询', 'engine': 'bing'})]),
                               reply([('search_web', {'query': 'precise English technical report', 'engine': 'arxiv'})]),
                               reply([('read_url', {'url': 'https://example.com/paper'})]), reply(text='根据原文回答')])
            result = run(root, {}, [{'role': 'user', 'content': '研究这个版本'}], model=model, toolbox=box)
            self.assertEqual(len(actions), 3)
            self.assertEqual(actions[1][1]['engine'], 'arxiv')
            self.assertIn('unrelated', model.inputs[1][0][-1]['content'])
            self.assertIn('actual paper', model.inputs[3][0][-1]['content'])
            self.assertEqual(result.text, '根据原文回答')
            self.assertEqual(len(result.web_research['trace']), 3)
            self.assertEqual(result.usage['total_tokens'], 8)

    def test_failed_tool_is_returned_to_model_for_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            box = ResearchTools(root, '', time.monotonic() + 60)
            box.execute = lambda *args: (_ for _ in ()).throw(TimeoutError())
            model = FakeModel([reply([('read_url', {'url': 'https://example.com'})]), reply(text='已尝试但连接超时')])
            result = run(root, {}, [], model=model, toolbox=box)
            self.assertIn('error', model.inputs[-1][0][-1]['content'])
            self.assertEqual(result.web_research['trace'][0]['status'], 'failed')

    def test_no_tool_for_general_answer_and_budget_forces_final(self):
        with tempfile.TemporaryDirectory() as root:
            result = run(root, {}, [], model=FakeModel([reply(text='普通解释')]))
            self.assertIsNone(result.web_research)
            box = ResearchTools(root, '', time.monotonic() + 60)
            box.execute = lambda *args: {'results': []}
            model = FakeModel([reply([('search_web', {'query': 'one', 'engine': 'bing'})]), reply(text='预算结束')])
            result = run(root, {}, [], model=model, toolbox=box, max_calls=1)
            self.assertEqual(model.inputs[-1][1], 'none')
            self.assertEqual(result.web_research['stop_reason'], 'budget')

    def test_duplicates_and_invalid_arguments_do_not_execute(self):
        with tempfile.TemporaryDirectory() as root:
            box = ResearchTools(root, '', time.monotonic() + 60)
            count = []
            box.execute = lambda *args: count.append(args) or {'results': []}
            call = ('search_web', {'query': 'same', 'engine': 'bing'})
            model = FakeModel([reply([call]), reply([call]), reply(text='完成')])
            result = run(root, {}, [], model=model, toolbox=box)
            self.assertEqual(len(count), 1)
            self.assertEqual(result.web_research['trace'][1]['status'], 'failed')
            with self.assertRaises(ValueError): box._document('0' * 64)


class ToolModelTests(unittest.TestCase):
    def model(self, protocol):
        with patch.dict(os.environ, {'AUTOLAB_LLM_API_KEY': 'test'}):
            return ToolModel({'model': 'test', 'endpoint': 'https://example.com', 'protocol': protocol, 'max_retries': 0})

    def test_responses_preserves_calls_and_reasoning_for_replay(self):
        model = self.model('responses')
        output = [{'type': 'reasoning', 'id': 'rs1', 'summary': []},
                  {'type': 'function_call', 'call_id': 'c1', 'name': 'search_web', 'arguments': '{}'}]
        result = model.parse({'status': 'completed', 'output': output})
        self.assertEqual(result.replay, output)
        self.assertEqual(result.calls[0]['id'], 'c1')
        self.assertEqual(model.output('c1', {'ok': True})['type'], 'function_call_output')

    def test_chat_accepts_empty_content_tool_calls(self):
        model = self.model('chat_completions')
        result = model.parse({'choices': [{'finish_reason': 'tool_calls', 'message': {
            'role': 'assistant', 'content': None, 'reasoning_content': 'provider state',
            'tool_calls': [{'id': 'c1', 'function': {'name': 'search_web', 'arguments': '{}'}}]}}]})
        self.assertEqual(result.calls[0]['name'], 'search_web')
        self.assertEqual(result.replay[0]['reasoning_content'], 'provider state')

    def test_stream_missing_terminal_is_not_success(self):
        model = self.model('responses')
        stream = io.BytesIO(b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n')
        stream.headers = {'Content-Type': 'text/event-stream'}
        with patch('urllib.request.urlopen', return_value=stream), self.assertRaisesRegex(Exception, '中断'):
            model.turn([], [])

    def test_https_and_private_addresses_are_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            box = ResearchTools(root, '', time.monotonic() + 60)
            for url in ('http://example.com', 'https://127.0.0.1', 'https://user:pass@example.com'):
                with self.assertRaises(Exception): box.read_url(url)
            with self.assertRaises(ValueError): box.search_web('api_key=private', 'bing')

    def test_pdf_pagination_requires_actual_import(self):
        with tempfile.TemporaryDirectory() as root:
            box = ResearchTools(root, '', time.monotonic() + 60)
            box.fetcher.get_bytes = lambda *a, **k: (b'<html><title>Report</title><p>' + b'evidence ' * 3000 + b'</p><a href="/paper.pdf">PDF</a></html>', 'https://example.com/report', 'text/html', 'utf-8')
            result = box.read_url('https://example.com/report')
            self.assertEqual(len(result['text']), 12000)
            self.assertEqual(result['links'][0]['url'], 'https://example.com/paper.pdf')
            next_page = box.read_document(result['document_id'], result['next_offset'])
            self.assertTrue(next_page['text'])
            self.assertTrue(box.find_in_document(result['document_id'], 'evidence')['matches'])


if __name__ == '__main__': unittest.main()
