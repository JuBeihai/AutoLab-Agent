"""Native tool-call turns for both configured gateways; no intent routing."""
import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

from skills.llm_client import LLMClient, LLMError


class ToolModel:
    def __init__(self, config):
        self.client = LLMClient(config)
        self.protocol = self.client.protocol

    def turn(self, messages, tools, tool_choice='auto'):
        c = self.client
        if self.protocol == 'responses':
            payload = {'model': c.model, 'stream': True, 'store': False,
                       'max_output_tokens': c.max_tokens,
                       'instructions': '\n\n'.join(m['content'] for m in messages if m.get('role') in ('system', 'developer')),
                       'input': [m for m in messages if m.get('role') not in ('system', 'developer')],
                       'tools': [dict(type='function', **t) for t in tools], 'tool_choice': tool_choice}
        else:
            payload = {'model': c.model, 'messages': messages, 'stream': False,
                       'max_tokens': c.max_tokens, 'temperature': c.temperature,
                       'tools': [{'type': 'function', 'function': t} for t in tools], 'tool_choice': tool_choice}
        for attempt in range(c.max_retries + 1):
            try:
                req = urllib.request.Request(c.endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                    headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream, application/json',
                             'Authorization': 'Bearer ' + c._api_key})
                with urllib.request.urlopen(req, timeout=c.timeout_sec) as response:
                    if 'event-stream' not in response.headers.get('Content-Type', ''):
                        raw = json.load(response)
                    else:
                        raw, lines = None, []
                        def event(data):
                            if data == '[DONE]': return None
                            item = json.loads(data)
                            if item.get('type') in ('response.failed', 'error'):
                                raise LLMError('工具调用模型返回失败事件')
                            if item.get('type') in ('response.completed', 'response.incomplete'):
                                return item['response']
                        for line in response:
                            line = line.decode('utf-8').rstrip('\r\n')
                            if line.startswith('data:'): lines.append(line[5:].lstrip())
                            elif not line and lines:
                                raw = event('\n'.join(lines)); lines = []
                                if raw is not None: break
                        if raw is None and lines: raw = event('\n'.join(lines))
                        if raw is None: raise LLMError('工具调用流中断，未收到完整结果')
                return self.parse(raw)
            except urllib.error.HTTPError as error:
                error.close()
                if error.code not in (429, 500, 502, 503, 504) or attempt == c.max_retries:
                    raise LLMError(f'工具调用接口 HTTP {error.code}；未降级为规则搜索') from None
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempt == c.max_retries:
                    raise LLMError('工具调用接口连接失败：' + type(error).__name__) from None
            time.sleep(c.retry_backoff_sec * (attempt + 1))

    def parse(self, raw):
        calls, text = [], ''
        if self.protocol == 'responses':
            if raw.get('status') != 'completed':
                raise LLMError('模型工具回合未完整结束，请提高输出预算或重试')
            replay = raw.get('output') or []
            for item in replay:
                if item.get('type') == 'function_call':
                    calls.append({'id': item['call_id'], 'name': item['name'], 'arguments': item['arguments']})
                elif item.get('type') == 'message':
                    text += ''.join(x.get('text', x.get('refusal', '')) for x in item.get('content', [])
                                    if x.get('type') in ('output_text', 'refusal'))
        else:
            choice = raw['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise LLMError('模型工具回合被长度截断，未执行不完整指令')
            message = choice['message']
            # Keep provider reasoning fields required for subsequent tool turns.
            replay = [message]
            text = message.get('content') or ''
            for item in message.get('tool_calls') or []:
                calls.append({'id': item['id'], 'name': item['function']['name'],
                              'arguments': item['function']['arguments']})
        if not calls and not text.strip(): raise LLMError('模型未返回回答或工具调用')
        return SimpleNamespace(text=text, calls=calls, replay=replay, usage=raw.get('usage') or {})

    def output(self, call_id, result):
        content = json.dumps(result, ensure_ascii=False)
        if self.protocol == 'responses':
            return {'type': 'function_call_output', 'call_id': call_id, 'output': content}
        return {'role': 'tool', 'tool_call_id': call_id, 'content': content}
