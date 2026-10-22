import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error

from skills import llm_profiles
from skills.llm_client import LLMClient, LLMError, LLMEmptyContentError

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/llm.json').read_text())
MESSAGES = [{'role': 'system', 'content': '系统约束'}, {'role': 'user', 'content': '之前的问题'},
            {'role': 'assistant', 'content': '之前的回答'}, {'role': 'user', 'content': '现在的问题'}]


def response(events, json_body=None):
    raw = (json.dumps(json_body) if json_body is not None else ''.join(
        'event: ' + event.get('type', 'message') + '\ndata: ' + json.dumps(event) + '\n\n' for event in events))
    stream = io.BytesIO(raw.encode())
    stream.headers = {'Content-Type': 'application/json' if json_body is not None else 'text/event-stream'}
    return stream


def terminal(text='完整回答', status='completed', reason=None):
    return {'status': status, 'output': [{'type': 'reasoning', 'summary': []},
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30},
            'incomplete_details': {'reason': reason} if reason else None}


class LLMModelTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'ETS_LLM_API_KEY': 'test-shared', 'ETS_GPT_API_KEY': 'test-gpt'})
        environment.start()
        self.addCleanup(environment.stop)
        self.config = {**llm_profiles.resolve(CONFIG, 'responses'), 'max_retries': 0}

    def test_profiles_default_and_allowlist(self):
        self.assertEqual(llm_profiles.resolve(CONFIG)['model'], 'YOUR_CHAT_MODEL')
        self.assertEqual(self.config['model'], 'YOUR_RESPONSES_MODEL')
        for value in ('bad', 'http://attacker', {}, [], True):
            with self.assertRaises(ValueError):
                llm_profiles.resolve(CONFIG, value)
        public = json.dumps(llm_profiles.public(CONFIG))
        self.assertNotIn('endpoint', public)
        self.assertNotIn('api_key', public)

    def test_gpt_key_precedence_and_shared_fallback(self):
        self.assertEqual(LLMClient(self.config)._api_key, 'test-gpt')
        with patch.dict(os.environ, {'ETS_GPT_API_KEY': ''}):
            self.assertEqual(LLMClient(self.config)._api_key, 'test-shared')
        self.assertEqual(LLMClient(llm_profiles.resolve(CONFIG, 'chat'))._api_key, 'test-shared')

    def test_responses_payload_history_streaming_and_final_text(self):
        events = [{'type': 'response.output_text.delta', 'delta': '完整'},
                  {'type': 'response.output_text.delta', 'delta': '回答'},
                  {'type': 'response.completed', 'response': terminal()}]
        pieces = []
        with patch('urllib.request.urlopen', return_value=response(events)) as request:
            result = LLMClient(self.config).chat(MESSAGES, stream=True, on_delta=lambda *part: pieces.append(part))
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload['instructions'], '系统约束')
        self.assertEqual(payload['input'], MESSAGES[1:])
        self.assertEqual(payload['model'], 'YOUR_RESPONSES_MODEL')
        self.assertEqual(payload['max_output_tokens'], 8000)
        self.assertTrue(payload['stream'])
        for key in ('messages', 'max_tokens', 'temperature', 'top_p'):
            self.assertNotIn(key, payload)
        self.assertEqual(request.call_args.args[0].full_url, CONFIG['profiles']['responses']['endpoint'])
        self.assertEqual(result.text, '完整回答')
        self.assertEqual(pieces, [('content', '完整'), ('content', '回答')])
        self.assertEqual(result.finish_reason, 'stop')
        self.assertEqual(result.usage['total_tokens'], 30)

    def test_responses_terminal_only_and_json_gateway(self):
        for stream in (response([{'type': 'response.completed', 'response': terminal()}]), response([], terminal())):
            with patch('urllib.request.urlopen', return_value=stream):
                result = LLMClient(self.config).chat(MESSAGES)
            self.assertEqual(result.text, '完整回答')

    def test_incomplete_maps_to_existing_length_warning(self):
        events = [{'type': 'response.incomplete', 'response': terminal('部分回答', 'incomplete', 'max_output_tokens')}]
        with patch('urllib.request.urlopen', return_value=response(events)):
            result = LLMClient(self.config).chat(MESSAGES)
        self.assertEqual(result.finish_reason, 'length')
        self.assertEqual(result.text, '部分回答')

    def test_disconnect_and_failure_never_report_partial_success(self):
        for events in ([{'type': 'response.output_text.delta', 'delta': '一部分'}],
                       [{'type': 'response.failed', 'response': {'status': 'failed'}}],
                       [{'type': 'error', 'message': 'private upstream body'}],
                       [{'type': 'response.incomplete', 'response': terminal('partial', 'incomplete', 'content_filter')}]):
            with patch('urllib.request.urlopen', return_value=response(events)), self.assertRaises(LLMError):
                LLMClient(self.config).chat(MESSAGES)

    def test_reasoning_only_rejected_and_refusal_supported(self):
        with patch('urllib.request.urlopen', return_value=response([], terminal(''))), self.assertRaises(LLMEmptyContentError):
            LLMClient(self.config).chat(MESSAGES)
        body = terminal()
        body['output'][1]['content'] = [{'type': 'refusal', 'refusal': '无法回答'}]
        with patch('urllib.request.urlopen', return_value=response([], body)):
            self.assertEqual(LLMClient(self.config).chat(MESSAGES).text, '无法回答')

    def test_transient_http_retry_and_unauthorized_no_retry(self):
        client = LLMClient({**self.config, 'max_retries': 1})
        transient = urllib.error.HTTPError('gateway', 503, 'unavailable', {}, None)
        with patch('urllib.request.urlopen', side_effect=[transient, response([], terminal())]) as request, patch('time.sleep'):
            self.assertEqual(client.chat(MESSAGES).text, '完整回答')
        self.assertEqual(request.call_count, 2)
        denied = urllib.error.HTTPError('gateway', 401, 'denied', {}, None)
        with patch('urllib.request.urlopen', side_effect=denied) as request, self.assertRaises(LLMError):
            client.chat(MESSAGES)
        self.assertEqual(request.call_count, 1)

    def test_chat_json_and_stream_remain_chat_completions(self):
        client = LLMClient(llm_profiles.resolve(CONFIG, 'chat'))
        body = {'choices': [{'message': {'content': 'GLM回答', 'reasoning_content': 'summary'}, 'finish_reason': 'stop'}]}
        with patch('urllib.request.urlopen', return_value=response([], body)) as request:
            self.assertEqual(client.chat(MESSAGES).text, 'GLM回答')
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload['messages'], MESSAGES)
        self.assertIn('max_tokens', payload)
        events = [{'choices': [{'delta': {'content': 'GLM回答'}, 'finish_reason': 'stop'}]}]
        with patch('urllib.request.urlopen', return_value=response(events)):
            self.assertEqual(client.chat(MESSAGES, stream=True).text, 'GLM回答')

    def test_chat_empty_stream_retries_with_bounded_budget(self):
        client = LLMClient({**llm_profiles.resolve(CONFIG, 'chat'), 'max_retries': 2, 'max_tokens': 12000})
        empty = [{'choices': [{'delta': {'reasoning_content': 'internal'}, 'finish_reason': 'length'}]}]
        valid = [{'choices': [{'delta': {'content': '有效方案'}, 'finish_reason': 'stop'}]}]
        pieces = []
        with patch('urllib.request.urlopen', side_effect=[response(empty), response(empty), response(valid)]) as request, \
             patch('time.sleep'):
            result = client.chat(MESSAGES, stream=True, on_delta=lambda *part: pieces.append(part))
        self.assertEqual(result.text, '有效方案')
        budgets = [json.loads(call.args[0].data)['max_tokens'] for call in request.call_args_list]
        self.assertEqual(budgets, [12000, 24000, 24000])
        self.assertEqual([part for part in pieces if part[0] == 'content'], [('content', '有效方案')])
        self.assertEqual({json.loads(call.args[0].data)['model'] for call in request.call_args_list}, {'YOUR_CHAT_MODEL'})

    def test_empty_retries_exhaust_with_safe_actionable_error(self):
        client = LLMClient({**llm_profiles.resolve(CONFIG, 'chat'), 'max_retries': 1})
        empty = [{'choices': [{'delta': {}, 'finish_reason': 'stop'}]}]
        with patch('urllib.request.urlopen', side_effect=[response(empty), response(empty)]) as request, \
             patch('time.sleep'), self.assertRaisesRegex(LLMEmptyContentError, '未返回正式回答') as failure:
            client.chat(MESSAGES, stream=True)
        self.assertEqual(request.call_count, 2)
        self.assertNotIn('test-shared', str(failure.exception))

    def test_chat_json_response_to_stream_request_is_supported(self):
        client = LLMClient(llm_profiles.resolve(CONFIG, 'chat'))
        body = {'choices': [{'message': {'content': '完整方案'}, 'finish_reason': 'stop'}]}
        pieces = []
        with patch('urllib.request.urlopen', return_value=response([], body)):
            result = client.chat(MESSAGES, stream=True, on_delta=lambda *part: pieces.append(part))
        self.assertEqual(result.text, '完整方案')
        self.assertEqual(pieces, [('content', '完整方案')])

    def test_chat_nonstream_empty_response_retries(self):
        client = LLMClient({**llm_profiles.resolve(CONFIG, 'chat'), 'max_retries': 1})
        empty = {'choices': [{'message': {'content': ''}, 'finish_reason': 'length'}]}
        valid = {'choices': [{'message': {'content': '完整方案'}, 'finish_reason': 'stop'}]}
        with patch('urllib.request.urlopen', side_effect=[response([], empty), response([], valid)]) as request, \
             patch('time.sleep'):
            self.assertEqual(client.chat(MESSAGES).text, '完整方案')
        self.assertEqual(request.call_count, 2)

    def test_stream_with_delivered_content_is_not_replayed_on_failure(self):
        client = LLMClient({**llm_profiles.resolve(CONFIG, 'chat'), 'max_retries': 2})
        def broken_stream(messages, budget, temperature, deliver):
            deliver('content', '已显示的部分回答')
            raise LLMError('流式网络错误')
        pieces = []
        with patch.object(client, '_chat_stream', side_effect=broken_stream) as request, \
             patch('time.sleep') as wait, self.assertRaises(LLMError):
            client.chat(MESSAGES, stream=True, on_delta=lambda *part: pieces.append(part))
        self.assertEqual(request.call_count, 1)
        wait.assert_not_called()
        self.assertEqual(pieces, [('content', '已显示的部分回答')])
