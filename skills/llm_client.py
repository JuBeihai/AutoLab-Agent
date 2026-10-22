"""内部 LLM 网关客户端。

GLM Chat Completions 与 GPT Responses 两种内部网关协议，
使用 stdlib urllib，不依赖第三方 SDK（服务器无 pypi 外网）。

处理点：
- YOUR_CHAT_MODEL 是推理模型，返回 message.content + message.reasoning_content 两部分
  - content 是最终答案（我们用这个）
  - reasoning_content 是思维链（只做日志，不入实验记录）
- 若 content 为空但 reasoning 有内容，说明 max_tokens 太小被截在思维链里，会告警并抛错
- 网络瞬时错误（超时 / 5xx）自动重试；4xx 直接抛
- 返回结构化 dict：{text, reasoning, usage, raw}
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from skills.env_secrets import get_llm_api_key
from skills.llm_profiles import key_env


class LLMError(RuntimeError):
    pass


class LLMEmptyContentError(LLMError):
    """content 为空（通常是 max_tokens 太小，全部消耗在 reasoning 里）"""

    def __init__(self, message, finish_reason=None):
        super().__init__(message)
        self.finish_reason = finish_reason


@dataclass
class LLMResponse:
    text: str                      # message.content
    reasoning: str                 # message.reasoning_content（可能为空）
    usage: dict                    # {prompt_tokens, completion_tokens, total_tokens}
    finish_reason: Optional[str]
    raw: dict                      # 完整的原始响应


class LLMClient:
    def __init__(self, cfg: dict):
        self.endpoint = cfg["endpoint"]
        self.model = cfg["model"]
        self.protocol = cfg.get('protocol', 'chat_completions')
        if self.protocol not in ('chat_completions', 'responses'):
            raise ValueError('未知 LLM 协议')
        self.max_tokens = int(cfg.get("max_tokens", 4000))
        self.temperature = float(cfg.get("temperature", 0.3))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.timeout_sec = int(cfg.get("timeout_sec", 120))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.retry_backoff_sec = int(cfg.get("retry_backoff_sec", 3))
        self.empty_retry_max_tokens = int(cfg.get('empty_retry_max_tokens', 24000))
        self._api_key = get_llm_api_key(key_env(cfg))

    def _retry_budget(self, budget, error):
        if isinstance(error, LLMEmptyContentError) and error.finish_reason == 'length':
            return max(budget, min(budget * 2, self.empty_retry_max_tokens))
        return budget

    def chat(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        on_delta: Optional[Any] = None,
    ) -> LLMResponse:
        """发起一次 chat completion 请求。

        Parameters
        ----------
        messages : list of {"role": "system"/"user"/"assistant", "content": str}
        max_tokens, temperature : 单次覆盖 cfg 默认值
        stream : True 时走 SSE，边收边通过 on_delta 回调交付
        on_delta : callable(kind: str, text: str) -> None
            kind ∈ {"reasoning", "content"}；text 是本次增量（可能只有几个字）
            仅在 stream=True 时被调用；返回值忽略。

        Returns
        -------
        LLMResponse（stream 模式下 text/reasoning 是全部拼接后的完整字符串）
        """
        budget = self.max_tokens if max_tokens is None else max_tokens
        if stream or self.protocol == 'responses':
            # 流式路径：一断难续，遇到 503/429/超时时整段重发（限流场景常用）
            last_err = None
            for attempt in range(self.max_retries + 1):
                content_started = False
                def deliver(kind, piece):
                    nonlocal content_started
                    if kind == 'content' and piece.strip():
                        content_started = True
                    if stream and on_delta:
                        on_delta(kind, piece)
                try:
                    if self.protocol == 'responses':
                        return self._responses_stream(messages, budget, deliver)
                    return self._chat_stream(messages, budget, temperature, deliver)
                except LLMError as e:
                    msg = str(e)
                    # 4xx 且非 429 直接抛出；429/503/其他 5xx/网络类都重试
                    retriable = (isinstance(e, LLMEmptyContentError) or "HTTP 429" in msg or "HTTP 5" in msg or
                                 "网络" in msg or "流式" in msg or "max_concurrency" in msg)
                    last_err = e
                    if content_started or not retriable or attempt >= self.max_retries:
                        raise
                    budget = self._retry_budget(budget, e)
                    wait = self.retry_backoff_sec * (attempt + 1)
                    print(f"[llm] 流式失败可重试：{msg[:120]}. {wait}s 后重试，输出预算 {budget}", flush=True)
                    time.sleep(wait)
            raise last_err or LLMError("stream retries 用尽")

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": budget,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        last_err = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                if attempt > 0:
                    print(f"[llm] 第 {attempt+1}/{self.max_retries+1} 次尝试 "
                          f"(timeout={self.timeout_sec}s)", flush=True)
                req = urllib.request.Request(
                    self.endpoint, data=body, headers=headers, method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw_text = resp.read().decode("utf-8")
                    raw = json.loads(raw_text)
                    return self._parse_response(raw)
            except urllib.error.HTTPError as e:
                elapsed = time.time() - t0
                err_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
                if 400 <= e.code < 500:
                    raise LLMError(f"HTTP {e.code}: {err_body[:400]}") from e
                last_err = LLMError(f"HTTP {e.code} (attempt {attempt+1}, {elapsed:.0f}s): {err_body[:200]}")
            except (urllib.error.URLError, TimeoutError) as e:
                elapsed = time.time() - t0
                last_err = LLMError(f"网络错误 (attempt {attempt+1}, 等待 {elapsed:.0f}s 后超时/失败): {e}")
            except json.JSONDecodeError as e:
                elapsed = time.time() - t0
                last_err = LLMError(f"响应不是合法 JSON (attempt {attempt+1}, {elapsed:.0f}s): {e}")
            except LLMEmptyContentError as error:
                last_err = error
                budget = self._retry_budget(budget, error)
                payload['max_tokens'] = budget
                body = json.dumps(payload, ensure_ascii=False).encode('utf-8')

            if attempt < self.max_retries:
                wait = self.retry_backoff_sec * (attempt + 1)
                print(f"[llm] 失败：{last_err}. {wait}s 后重试", flush=True)
                time.sleep(wait)

        raise last_err or LLMError("未知错误")

    def _responses_stream(self, messages, max_tokens, on_delta):
        instructions = '\n\n'.join(message['content'] for message in messages if message['role'] in ('system', 'developer'))
        inputs = [{'role': message['role'], 'content': message['content']} for message in messages
                  if message['role'] not in ('system', 'developer')]
        payload = {'model': self.model, 'instructions': instructions, 'input': inputs,
                   'max_output_tokens': self.max_tokens if max_tokens is None else max_tokens,
                   'stream': True}
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                    headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream',
                             'Authorization': f'Bearer {self._api_key}'}, method='POST')
        parts = []
        terminal = None

        def consume(event):
            nonlocal terminal
            if not isinstance(event, dict):
                raise LLMError('Responses 事件必须为对象')
            kind = event.get('type')
            if kind in ('response.output_text.delta', 'response.refusal.delta'):
                piece = event.get('delta', '')
                if not isinstance(piece, str):
                    raise LLMError('Responses 文本增量格式错误')
                parts.append(piece)
                if piece and on_delta:
                    try:
                        on_delta('content', piece)
                    except Exception:
                        pass
            elif kind in ('response.completed', 'response.incomplete'):
                terminal = event.get('response')
                if not isinstance(terminal, dict):
                    raise LLMError('Responses 终止事件缺少 response')
                terminal = {**terminal, 'status': 'completed' if kind == 'response.completed' else 'incomplete'}
            elif kind in ('response.failed', 'error'):
                raise LLMError('Responses 返回失败事件，未将部分输出作为成功答案')

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    terminal = json.loads(response.read().decode('utf-8'))
                else:
                    data_lines = []
                    for raw_line in response:
                        line = raw_line.decode('utf-8').rstrip('\r\n')
                        if line.startswith('data:'):
                            data_lines.append(line[5:].lstrip())
                        elif not line and data_lines:
                            data = '\n'.join(data_lines)
                            data_lines = []
                            if data == '[DONE]':
                                break
                            consume(json.loads(data))
                            if terminal is not None:
                                break
                    if data_lines:
                        data = '\n'.join(data_lines)
                        if data != '[DONE]':
                            consume(json.loads(data))
        except urllib.error.HTTPError as error:
            raise LLMError(f'HTTP {error.code}: Responses 请求失败') from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LLMError(f'Responses 网络错误：{type(error).__name__}') from error
        except (ValueError, UnicodeError, TypeError) as error:
            raise LLMError('Responses 事件格式错误') from error
        if terminal is None:
            raise LLMError('Responses 流式响应提前结束，未收到完成事件')
        if not isinstance(terminal, dict):
            raise LLMError('Responses 返回格式错误')
        status = terminal.get('status')
        if status not in ('completed', 'incomplete'):
            raise LLMError('Responses 未成功完成')
        reason = (terminal.get('incomplete_details') or {}).get('reason')
        if status == 'incomplete' and reason != 'max_output_tokens':
            raise LLMError('Responses 未完成，原因不是输出长度限制')
        final_parts = []
        for item in terminal.get('output') or []:
            if item.get('type') != 'message':
                continue
            for content in item.get('content', []):
                if content.get('type') == 'output_text':
                    final_parts.append(content.get('text', ''))
                elif content.get('type') == 'refusal':
                    final_parts.append(content.get('refusal', ''))
        text = (''.join(final_parts) or ''.join(parts)).strip()
        if not text:
            raise LLMEmptyContentError('Responses 正式回答为空，请检查输出预算和服务响应',
                                       'length' if reason == 'max_output_tokens' else 'stop')
        if not parts and on_delta:
            try:
                on_delta('content', text)
            except Exception:
                pass
        return LLMResponse(text=text, reasoning='', usage=terminal.get('usage') or {},
                           finish_reason='length' if status == 'incomplete' else 'stop', raw=terminal)

    def _chat_stream(
        self,
        messages: List[dict],
        max_tokens: Optional[int],
        temperature: Optional[float],
        on_delta: Optional[Callable[[str, str], None]],
    ) -> LLMResponse:
        """SSE 流式版本。边收边通过 on_delta(kind, text) 交付。

        协议：OpenAI 兼容 SSE，每行 `data: {...}`；`data: [DONE]` 结束。
        - delta.reasoning_content → kind='reasoning'（思维链，可选展示）
        - delta.content            → kind='content'（正式答案）
        网关瞬断/超时不重试（流已经开始就断不能续），直接抛给上层。
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "stream": True,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream",
        }

        content_buf: list = []
        reasoning_buf: list = []
        finish = None
        usage: dict = {}

        try:
            req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                if 'application/json' in resp.headers.get('Content-Type', ''):
                    result = self._parse_response(json.loads(resp.read().decode('utf-8')))
                    if on_delta:
                        on_delta('content', result.text)
                    return result
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "usage" in chunk and isinstance(chunk["usage"], dict):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    ch0 = choices[0]
                    if ch0.get("finish_reason"):
                        finish = ch0["finish_reason"]
                    delta = ch0.get("delta") or {}
                    r_piece = delta.get("reasoning_content") or ""
                    c_piece = delta.get("content") or ""
                    if r_piece:
                        reasoning_buf.append(r_piece)
                        if on_delta:
                            try:
                                on_delta("reasoning", r_piece)
                            except Exception:
                                pass
                    if c_piece:
                        content_buf.append(c_piece)
                        if on_delta:
                            try:
                                on_delta("content", c_piece)
                            except Exception:
                                pass
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise LLMError(f"HTTP {e.code}: {err_body[:400]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"流式网络错误：{e}") from e

        content = "".join(content_buf).strip()
        reasoning = "".join(reasoning_buf).strip()
        if not content:
            hint = ""
            if reasoning and finish == "length":
                hint = " (提示：max_tokens 太小被思维链吃光，调大 max_tokens 试试)"
            raise LLMEmptyContentError(
                f"模型未返回正式回答{hint}；finish_reason={finish}, reasoning_len={len(reasoning)}", finish
            )
        return LLMResponse(
            text=content, reasoning=reasoning,
            usage=usage, finish_reason=finish, raw={"streamed": True},
        )

    @staticmethod
    def _parse_response(raw: dict) -> LLMResponse:
        try:
            choice = raw["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"响应结构不合预期：{raw}") from e

        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        finish = choice.get("finish_reason")
        usage = raw.get("usage", {})

        if not content:
            hint = ""
            if reasoning and finish == "length":
                hint = (" (提示：finish_reason=length 且 reasoning 非空，"
                        "可能是 max_tokens 太小被思维链吃光，调大 max_tokens 试试)")
            raise LLMEmptyContentError(
                f"模型未返回正式回答{hint}；finish_reason={finish}, reasoning_len={len(reasoning)}", finish
            )

        return LLMResponse(
            text=content, reasoning=reasoning,
            usage=usage, finish_reason=finish, raw=raw,
        )


# ------------------------------------------------------------------
# JSON 提取辅助
# ------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象。

    尝试顺序：
    1. 直接 json.loads
    2. 剥掉 ```json ... ``` 围栏
    3. 找第一个 { 到最后一个 } 的子串

    如果都失败抛 LLMError。
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise LLMError(f"无法从 LLM 输出中解析出 JSON 对象。前 200 字符：{text[:200]}")
