import re
import json
import requests
import time
import base64

from .utils import load_plugin_config, mie_log, resolve_token

MY_CATEGORY = "🐑 MieNodes/🐑 LLM Service Config"


# 引入 time 模块用于在重试间增加延迟

def _drop_image_detail_auto(messages):
    """Drop `detail: "auto"` from any image_url content part.

    The OpenAI image_url spec allows `auto` / `low` / `high`, but
    MiniMax M-series vision models reject "auto" with HTTP 400
    (`invalid params, invalid image detail: auto`). OpenAI treats
    a missing `detail` field as "auto" internally, so removing the
    key is safe for OpenAI-compat services too. Gemini uses a
    different content shape (inline_data) and never sees this.

    Returns the input unchanged when no `detail: "auto"` is present
    (cheap fast path). When a change is needed, returns a new list
    with selectively-copied dicts - never mutates the caller's
    message structure, so the same list can be reused across calls
    (e.g. retry loops, or sending the same prompt to multiple
    providers in sequence).
    """
    if not messages:
        return messages
    new_messages = None
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content = None
        for pi, part in enumerate(content):
            if not (
                isinstance(part, dict)
                and part.get("type") == "image_url"
                and isinstance(part.get("image_url"), dict)
                and part["image_url"].get("detail") == "auto"
            ):
                continue
            if new_content is None:
                new_content = list(content)
            new_part = dict(part)
            new_part["image_url"] = dict(part["image_url"])
            del new_part["image_url"]["detail"]
            new_content[pi] = new_part
        if new_content is not None:
            if new_messages is None:
                new_messages = list(messages)
            new_messages[mi] = dict(msg)
            new_messages[mi]["content"] = new_content
    return new_messages if new_messages is not None else messages


def _minimax_reasoning_effort(model: str, kwargs: dict):
    """MiniMax-only request hint. Other providers are not sent this field.

    MiniMax-M3.1 thinks before ``content`` is written, and
    ``reasoning_effort=low`` leaves the answer in ``content``.
    ``thinking.type=disabled`` is HTTP 400 on this API. DeepSeek, Qwen,
    GLM, Kimi, Gemini, and Claude each use a different switch; sending
    this one to them is how a MiniMax patch breaks those calls. The
    cross-provider guarantee is ``_select_answer``: the answer channel
    is what callers receive, and a structured caller can refuse the
    thought channel.
    """
    explicit = kwargs.get("reasoning_effort")
    if explicit:
        return str(explicit)
    name = (model or "").lower().replace("_", ".")
    if "m3.1" in name:
        return "low"
    return None


class GeneralLLMServiceConnector:
    def __init__(self, api_url, manual_token, model, timeout=30, max_retries=3, retry_delay=5, 
                 config_file="mie_llm_keys.json", config_key=None, prefer_local_config=True):
        self.api_url = api_url
        self.manual_token = manual_token
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.config_file = config_file
        self.config_key = config_key
        self.prefer_local_config = prefer_local_config
        # finish_reason of the LAST 200 response (None = absent/unknown).
        # Informational plumbing for callers: 'length' on an empty reply
        # means their max_tokens budget starved the answer — the caller
        # (which owns the budget) may escalate and retry.
        self.last_finish_reason = None

    @property
    def api_token(self):
        return resolve_token(
            self.manual_token, 
            default_key=self.config_key, 
            config_file=self.config_file, 
            config_key=self.config_key, 
            prefer_local=self.prefer_local_config
        )

    def _provider_messages(self, messages):
        """Hook for converting OpenAI-style multimodal messages into the
        provider-native shape. Default is identity: OpenAI-compatible services
        already understand `image_url` content parts, so the default no-op is
        correct for them. `GeminiConnectorGeneral` overrides this to map
        `image_url` -> `inline_data`.

        Returning a fresh list is recommended so callers can safely mutate.
        """
        out = list(messages) if messages is not None else messages
        return self._sanitize_image_detail(out)

    def _sanitize_image_detail(self, messages):
        """Hook for provider-specific image_url `detail` value sanitization.

        Default is identity (preserves whatever the caller set). Some
        providers reject the OpenAI-default `detail: "auto"` value;
        override this in those connectors. Called from
        `_provider_messages` so subclasses that fully override
        `_provider_messages` (e.g. Gemini) opt out automatically.
        """
        return messages

    # Strips reasoning / chain-of-thought blocks that some models (DeepSeek R1,
    # GLM-Z, MiniMax M-series, etc.) emit before the final answer. Matches
    # both `<think>...</think>` and `<thinking>...</thinking>`.
    _THINK_BLOCK_RE = re.compile(
        r"<think>[\s\S]*?</think>"
        r"|<thinking>[\s\S]*?</thinking>",
        re.IGNORECASE,
    )
    # Some providers strip the opening tag server-side but leave the closing
    # one (seen live: SiliconFlow DeepSeek-V4-Flash returning `OK</think>OK`),
    # so orphan closers are removed as a second pass after paired blocks.
    _ORPHAN_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
    # An opening tag with no closer means the whole field is still the
    # thought. Do not treat that as the answer. A closer anywhere in the
    # text is left for ``_sanitize_response`` (the SiliconFlow shape above
    # keeps the text on both sides of the orphan tag).
    _UNCLOSED_THINK_RE = re.compile(
        r"^\s*<think(?:ing)?>([\s\S]*)$",
        re.IGNORECASE,
    )
    # Answer-channel vs thought-channel. These names show up across
    # OpenAI-compatible gateways; none of them is MiniMax-specific.
    #   reasoning_content: DeepSeek, Qwen, GLM, Kimi, MiniMax, MiMo
    #   reasoning / reasoning_text: OpenRouter and some OpenAI gateways
    #   reasoning_details: OpenRouter list of reasoning parts
    #   content parts typed reasoning/thinking, or Gemini ``thought: true``
    _REASONING_FIELDS = (
        "reasoning_content",
        "reasoning",
        "reasoning_text",
        "reasoning_details",
    )
    _REASONING_PART_TYPES = {
        "reasoning",
        "thinking",
        "thought",
        "reasoning_text",
    }

    def _sanitize_response(self, text, preserve_thinking=False):
        """Strip `<think>` / `<thinking>` reasoning blocks from `text`.

        Several reasoning models emit their chain-of-thought inside the content
        field before the actual answer. For prompt-rewriter use cases the
        thinking is noise that pollutes downstream models' input, so we strip
        it by default. Pass `preserve_thinking=True` to keep it (useful for
        debug / `CheckLLMServiceConnectivity` style diagnostics).
        """
        if text is None or preserve_thinking:
            return text
        if not isinstance(text, str):
            return ""
        cleaned = self._THINK_BLOCK_RE.sub("", text)
        cleaned = self._ORPHAN_THINK_CLOSE_RE.sub("", cleaned)
        return cleaned.strip()

    def _pop_answer_policy(self, kwargs):
        """Response-side switches. Pop them so they never enter the payload.

        ``allow_reasoning_fallback`` defaults True, which preserves the
        older prompt-rewriter behavior. Structured callers set it False
        on the connector (or pass the kwarg) so an empty answer stays
        empty on every provider, not only MiniMax.
        """
        preserve_thinking = bool(kwargs.pop("preserve_thinking", False))
        if "allow_reasoning_fallback" in kwargs:
            allow_reasoning_fallback = bool(kwargs.pop("allow_reasoning_fallback"))
        else:
            allow_reasoning_fallback = bool(
                getattr(self, "allow_reasoning_fallback", True)
            )
        return preserve_thinking, allow_reasoning_fallback

    @classmethod
    def _part_is_reasoning(cls, part):
        if part.get("thought") is True or part.get("thoughtsContent"):
            return True
        ptype = str(part.get("type") or "").strip().lower()
        if ptype in cls._REASONING_PART_TYPES:
            return True
        return "reason" in ptype or ptype == "thinking"

    @classmethod
    def _part_text(cls, part):
        for key in ("text", "thinking", "reasoning", "thoughtsContent", "content"):
            value = part.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    @classmethod
    def _reasoning_value_text(cls, value):
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return cls._part_text(value)
        if isinstance(value, list):
            chunks = []
            for item in value:
                if isinstance(item, str) and item:
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = cls._part_text(item)
                    if text:
                        chunks.append(text)
            return "\n".join(chunks)
        return ""

    @classmethod
    def _split_content(cls, content):
        """Split a string or a part list into ``(answer, reasoning)``.

        Reasoning parts stay out of the answer. A plain string is the
        answer channel; inline ``<think>`` blocks are stripped later.
        """
        if content is None:
            return "", ""
        if isinstance(content, str):
            return content, ""
        if not isinstance(content, list):
            return str(content), ""
        answer_chunks = []
        reasoning_chunks = []
        for part in content:
            if isinstance(part, str):
                if part:
                    answer_chunks.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = cls._part_text(part)
            if not text:
                continue
            if cls._part_is_reasoning(part):
                reasoning_chunks.append(text)
            else:
                answer_chunks.append(text)
        return "\n".join(answer_chunks), "\n".join(reasoning_chunks)

    def _message_channels(self, message):
        """Answer text and thought text from one chat ``message`` object."""
        if not isinstance(message, dict):
            return "", ""
        answer, reasoning = self._split_content(message.get("content"))
        extra = [reasoning] if reasoning else []
        for key in self._REASONING_FIELDS:
            text = self._reasoning_value_text(message.get(key))
            if text:
                extra.append(text)
        return answer, "\n".join(extra)

    def _peel_unclosed_think(self, answer, reasoning):
        if not isinstance(answer, str) or not answer:
            return answer or "", reasoning or ""
        if self._ORPHAN_THINK_CLOSE_RE.search(answer):
            return answer, reasoning or ""
        match = self._UNCLOSED_THINK_RE.match(answer)
        if not match:
            return answer, reasoning or ""
        peeled = match.group(1).strip()
        merged = "\n".join(bit for bit in (reasoning, peeled) if bit)
        return "", merged

    def _select_answer(
        self,
        answer,
        reasoning,
        *,
        preserve_thinking,
        allow_reasoning_fallback,
        tag,
    ):
        """Return ``(text, saw_reasoning)``.

        The answer channel wins whenever it survives sanitizing. The
        thought channel is used only when the caller still allows that
        fallback. The log phrase stays ``reasoning_content`` so existing
        operators and tests match every provider's thought field.
        """
        answer, reasoning = self._peel_unclosed_think(answer, reasoning)
        cleaned = self._sanitize_response(
            answer, preserve_thinking=preserve_thinking
        )
        if cleaned:
            return cleaned, bool(reasoning)
        if reasoning and allow_reasoning_fallback:
            mie_log(
                f"{tag} content empty after sanitize; "
                f"falling back to reasoning_content "
                f"({len(reasoning)} chars)"
            )
            cleaned = self._sanitize_response(
                reasoning, preserve_thinking=preserve_thinking
            )
            return cleaned or "", True
        if reasoning:
            mie_log(
                f"{tag} content empty; ignoring "
                f"reasoning_content ({len(reasoning)} chars)"
            )
            return "", True
        return "", False

    @staticmethod
    def _payload_chars(payload):
        """Return ``(total_chars, last_user_chars)`` for an LLM request payload.

        Used by the connector's POST / retry log lines so a slow or failing
        call carries the payload size alongside the elapsed seconds — without
        this a 300s ReadTimeout leaves no clue whether the request was 8KB or
        80KB. ``messages[*].content`` is summed; non-message fields (model,
        response_format, stream) are not counted (constant per connector).
        """
        total = 0
        last_user_chars = 0
        try:
            for m in payload.get("messages") or []:
                content = m.get("content") or ""
                if isinstance(content, list):
                    # Multi-modal payloads: count only text parts.
                    content = "".join(
                        str(p.get("text", ""))
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                n = len(str(content))
                total += n
                if m.get("role") == "user":
                    last_user_chars = n
        except Exception:
            pass
        return total, last_user_chars

    def generate_payload(self, messages, **kwargs):
        """
        生成标准的 OpenAI 兼容服务的 Payload。
        子类如果需要不同的默认参数或结构，可以重写此方法。
        """
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "response_format": {"type": "text"},
        }

    def invoke(self, messages, **kwargs):
        """
        调用 LLM 服务，并实现针对瞬时错误的重试机制。
        重试包括：Timeout, ConnectionError, 和 5xx 状态码。

        日志约定（按出现顺序）:
          - 每次 attempt 开头: ``[<model>] attempt N/M: POST <url> timeout=Ts``
          - 5xx 失败: 同前缀 + HTTP 状态码 + 耗时 + 响应体前 200 字符
          - Timeout/ConnectionError 失败: 同前缀 + 异常类型 + 耗时
          - 成功: ``[<model>] attempt N/M ok in Xs response_chars=N``
        外部 caller（如 Bernini 的 ``_chat``）会再记一次合并耗时，但不会覆盖
        单次 attempt 的真实耗时——两边互补。
        """
        # Response-side switches. Pop before generate_payload so they
        # cannot leak into a provider that rejects unknown fields.
        preserve_thinking, allow_reasoning_fallback = self._pop_answer_policy(kwargs)
        payload = self.generate_payload(messages, **kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        for attempt in range(self.max_retries):
            is_last_attempt = (attempt == self.max_retries - 1)
            attempt_idx = attempt + 1
            tag = f"[{self.model}] attempt {attempt_idx}/{self.max_retries}"
            req_chars, last_user_chars = self._payload_chars(payload)
            mie_log(
                f"{tag}: POST {self.api_url} timeout={self.timeout}s "
                f"request_chars={req_chars} last_user_chars={last_user_chars}"
            )
            attempt_t0 = time.perf_counter()

            try:
                response = requests.post(
                    self.api_url, json=payload, headers=headers, timeout=self.timeout
                )
                attempt_elapsed = time.perf_counter() - attempt_t0

                if response.status_code == 200:
                    response_data = response.json()
                    # Expose the finish_reason to callers on EVERY 200
                    # (so a later success never leaves a stale 'length'
                    # behind). 'length' on an empty reply is the caller's
                    # signal to raise ITS max_tokens budget and retry —
                    # only the caller knows its own budget doctrine.
                    try:
                        self.last_finish_reason = response_data["choices"][
                            0
                        ].get("finish_reason")
                    except (KeyError, IndexError, TypeError):
                        self.last_finish_reason = None
                    try:
                        message = response_data["choices"][0]["message"]
                    except (KeyError, IndexError) as e:
                        raise ValueError(
                            f"Unexpected response format: {type(e).__name__}. "
                            f"Response: {response.text[:200]}...")
                    answer, reasoning = self._message_channels(message)
                    cleaned, saw_reasoning = self._select_answer(
                        answer,
                        reasoning,
                        preserve_thinking=preserve_thinking,
                        allow_reasoning_fallback=allow_reasoning_fallback,
                        tag=tag,
                    )
                    if not cleaned and not saw_reasoning:
                        # HTTP 200 but zero usable text. finish_reason
                        # distinguishes the two causes we have seen live:
                        # "length" = the reasoning chain consumed the
                        # whole max_tokens budget before the answer
                        # started (raise the caller's budget); anything
                        # else = provider-side empty reply.
                        finish = self.last_finish_reason
                        if finish is None:
                            finish = "<unknown>"
                        mie_log(
                            f"{tag} empty reply; finish_reason={finish!r} "
                            f"(length => raise the caller's max_tokens "
                            f"budget so reasoning cannot starve the "
                            f"answer)"
                        )
                    mie_log(
                        f"{tag} ok in {attempt_elapsed:.2f}s "
                        f"response_chars={len(cleaned or '')}"
                    )
                    return cleaned

                # 5xx: 瞬时错误，包含响应体前 200 字符方便诊断
                if 500 <= response.status_code < 600:
                    body_snip = (response.text or "").replace("\n", " ")[:200]
                    detail = (
                        f"{tag} got HTTP {response.status_code} in {attempt_elapsed:.2f}s "
                        f"body={body_snip!r}"
                    )
                    if is_last_attempt:
                        raise Exception(
                            f"{detail}. Max retries ({self.max_retries}) exceeded."
                        )
                    mie_log(
                        f"{detail}. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                    continue

                # 4xx 等非重试错误：仍把响应体前 200 字符带出来
                body_snip = (response.text or "").replace("\n", " ")[:200]
                raise Exception(
                    f"{tag} failed with HTTP {response.status_code} in {attempt_elapsed:.2f}s "
                    f"body={body_snip!r}"
                )

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                attempt_elapsed = time.perf_counter() - attempt_t0
                error_type = type(e).__name__
                detail = (
                    f"{tag} {error_type} after {attempt_elapsed:.2f}s "
                    f"(request_chars={req_chars}, last_user_chars={last_user_chars})"
                )
                if is_last_attempt:
                    raise Exception(
                        f"{detail}. Max retries ({self.max_retries}) exceeded."
                    )
                mie_log(
                    f"{detail}. Retrying in {self.retry_delay} seconds... "
                    f"(Attempt {attempt_idx}/{self.max_retries})"
                )
                time.sleep(self.retry_delay)
                continue

            except requests.exceptions.RequestException as e:
                raise Exception(f"A non-retryable request error occurred: {e}")

        # 理论上不会执行到这里，但以防万一
        raise Exception(
            f"LLM Service failed after {self.max_retries} attempts due to an unknown error."
        )

    def get_state(self):
        """返回用于比较状态的字符串表示"""
        # 恢复为原先的无分隔符形式，保证与历史行为一致（避免回归）
        return f"{self.api_url}{self.api_token}{self.model}"
class StandardOpenAICompatibleConnector(GeneralLLMServiceConnector):
    """
    针对 SiliconFlow, ZhiPu, Kimi 等具有标准 OpenAI 兼容参数的 API。
    """

    def generate_payload(self, messages, **kwargs):
        # 封装 OpenAI 兼容服务的通用参数
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "n": kwargs.get("n", 1),
            "response_format": {"type": "text"},
        }
class SiliconFlowConnectorGeneral(StandardOpenAICompatibleConnector):
    api_url = "https://api.siliconflow.cn/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
class ZhiPuConnectorGeneral(StandardOpenAICompatibleConnector):
    """Standard ZhiPu BigModel connector (NOT the Coding / Token Plan tier).

    Targets the public ZhiPu BigModel API at the standard
    `/api/paas/v4/chat/completions` endpoint. ZhiPu API keys are 49-char
    hex `{id}.{secret}` strings (NOT `eyJ...` JWTs), and the same key
    format is accepted by both the standard and the `/api/coding/...`
    endpoints — the endpoint (i.e. which subscription you bought) is
    what selects the billing tier, not the key format. For the Coding /
    Token Plan subscription endpoint use `ZhiPuCodeConnectorGeneral`
    and `SetZhiPuCodeLLMServiceConnector`.
    """
    api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
class ZhiPuCodeConnectorGeneral(StandardOpenAICompatibleConnector):
    """ZhiPu Coding / Token Plan connector.

    Targets the ZhiPu Coding Plan endpoint at `/api/coding/...`
    with a Coding / Token Plan subscription. Distinct from
    the standard ZhiPu BigModel API in URL and billing (the GLM-5.3
    series is served here too — both endpoints expose the same model
    list; verified live 2026-09-29). Keys share the 49-char hex format
    with the standard tier. Pair with `SetZhiPuCodeLLMServiceConnector`.
    """
    api_url = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
class KimiConnectorGeneral(StandardOpenAICompatibleConnector):
    """Moonshot Kimi connector (api.moonshot.cn, pay-as-you-go only).

    The official Kimi API platform has NO subscription tier (docs:
    "Kimi API 开放平台是按量计费模式、无订阅制方案"); the Kimi for Coding
    subscription is a separate OAuth product for CLI tools, not an API-key
    chat endpoint, so no token-plan sibling connector exists.

    Payload note: per the official model parameter reference the current
    lineup locks sampling params server-side (kimi-k3 / kimi-k2.7-code /
    kimi-k2.6-thinking: temperature=1.0; every model: top_p=0.95, n=1,
    presence/frequency_penalty=0) and recommends NOT sending them
    explicitly — sending other values fails live with HTTP 400
    (`invalid temperature: only 1 is allowed for this model`). This
    payload therefore forwards only `max_tokens` and lets the server
    apply its own sampling defaults.
    """

    api_url = "https://api.moonshot.cn/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
            "response_format": {"type": "text"},
        }
class BailianLLMServiceConnector(GeneralLLMServiceConnector):
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        # 阿里百炼（通义千问）的 Payload 可能有所不同，这里保留其特殊性
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            # 可以根据需要添加其他参数
        }
class DeepSeekConnectorGeneral(GeneralLLMServiceConnector):
    api_url = "https://api.deepseek.com/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
class MiniMaxConnectorGeneral(StandardOpenAICompatibleConnector):
    """Standard MiniMax Open Platform connector.

    Targets the public MiniMax Open Platform API. Use with the
    standard `eyJ...` (JWT) API key issued from the MiniMax
    developer console. For the newer Token Plan / Coding Plan
    (`sk-cp-...` prefixed) keys, use `MiniMaxTokenPlanConnectorGeneral`
    and `SetMiniMaxTokenPlanLLMServiceConnector` instead.
    """
    api_url = "https://api.minimaxi.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        payload = super().generate_payload(messages, **kwargs)
        effort = _minimax_reasoning_effort(self.model, kwargs)
        if effort:
            payload["reasoning_effort"] = effort
        return payload

    def _sanitize_image_detail(self, messages):
        """Drop `detail: "auto"` from image_url parts. MiniMax rejects
        the OpenAI default with HTTP 400 (`invalid image detail: auto`);
        only `low` and `high` are accepted. OpenAI treats a missing
        field as "auto" internally, so stripping is a no-op there.
        """
        return _drop_image_detail_auto(messages)
class MiniMaxTokenPlanConnectorGeneral(StandardOpenAICompatibleConnector):
    """MiniMax Token Plan / Coding Plan connector.

    Targets the MiniMax Token Plan endpoint with `sk-cp-...` prefixed
    API keys (the Token Plan / Coding Plan subscription). The endpoint
    URL is shared with the Open Platform for now; the key prefix is
    what distinguishes the two billing tracks. The node default is
    MiniMax-M3.1-Flash-Preview.
    """
    api_url = "https://api.minimaxi.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        payload = super().generate_payload(messages, **kwargs)
        effort = _minimax_reasoning_effort(self.model, kwargs)
        if effort:
            payload["reasoning_effort"] = effort
        return payload

    def _sanitize_image_detail(self, messages):
        """Same as MiniMaxConnectorGeneral: drop `detail: "auto"`.
        See `_drop_image_detail_auto` for the rationale.
        """
        return _drop_image_detail_auto(messages)
class MiMoConnectorGeneral(StandardOpenAICompatibleConnector):
    """Standard Xiaomi MiMo Open Platform connector.

    Targets the public Xiaomi MiMo API at
    `https://api.xiaomimimo.com/v1/chat/completions` with `sk-xxxxx`
    API keys. For the Token Plan / Coding Plan (`tp-xxxxx` keys), use
    `MiMoTokenPlanConnectorGeneral` and `SetMiMoTokenPlanLLMServiceConnector`
    instead.

    Key differences from the generic OpenAI-compat shape:
      - Uses `max_completion_tokens` (newer OpenAI standard) instead of
        `max_tokens`; the MiMo docs only show the `max_completion_tokens`
        spelling.
      - Drops `top_k`, `n`, and `response_format` from the payload; the
        MiMo docs never show these and they are likely to 400.
      - Uses MiMo-friendly defaults: `temperature=1.0`, `top_p=0.95`.
      - Drops `detail: "auto"` from `image_url` parts (the MiMo image
        understanding docs do not include a `detail` field; the OpenAI
        default `"auto"` is known to be rejected by some providers).
    """

    api_url = "https://api.xiaomimimo.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_completion_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 0.95),
        }

    def _sanitize_image_detail(self, messages):
        """Drop `detail: "auto"` from image_url parts. The MiMo image
        understanding docs never include a `detail` field; only `low` /
        `high` are forwarded when the caller sets them explicitly. See
        `_drop_image_detail_auto` for the rationale.
        """
        return _drop_image_detail_auto(messages)
class MiMoTokenPlanConnectorGeneral(StandardOpenAICompatibleConnector):
    """Xiaomi MiMo Token Plan / Coding Plan connector.

    Targets the MiMo Token Plan endpoint at
    `https://token-plan-cn.xiaomimimo.com/v1/chat/completions` with
    `tp-xxxxx` API keys. Distinct from the standard tier in base URL,
    billing model, and API key format. The model lineup is shared with
    the standard tier. Pair with `SetMiMoTokenPlanLLMServiceConnector`.
    """

    api_url = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_completion_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 0.95),
        }

    def _sanitize_image_detail(self, messages):
        """Same as `MiMoConnectorGeneral`: drop `detail: "auto"`.
        See `_drop_image_detail_auto` for the rationale.
        """
        return _drop_image_detail_auto(messages)
class DoubaoConnectorGeneral(StandardOpenAICompatibleConnector):
    """ByteDance Volcano Ark (火山方舟) Doubao connector.

    Targets the Ark OpenAI-compatible endpoint at
    `https://ark.cn-beijing.volces.com/api/v3/chat/completions` with an
    Ark API Key (or IAM API Key) from the Volcano console. `model`
    accepts either the versioned doubao-seed ids (see the Set-node
    dropdown) or an inference Endpoint ID (`ep-xxxx`) created in the
    console. Pair with `SetDoubaoLLMServiceConnector`.
    """
    api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
class QianfanConnectorGeneral(StandardOpenAICompatibleConnector):
    """Baidu Qianfan (百度千帆) ERNIE connector.

    Targets the Qianfan ModelBuilder v2 OpenAI-compatible endpoint at
    `https://qianfan.baidubce.com/v2/chat/completions` with a Bearer API
    key from the Qianfan console. Serves the ERNIE family plus hosted
    third-party models (glm / deepseek, see the Set-node dropdown).
    Pair with `SetQianfanLLMServiceConnector`.
    """
    api_url = "https://qianfan.baidubce.com/v2/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
class SparkConnectorGeneral(GeneralLLMServiceConnector):
    """iFLYTEK Spark (讯飞星火) X2 connector.

    Targets the Spark-X2 OpenAI-compatible endpoint at
    `https://spark-api-open.xf-yun.com/x2/chat/completions`, authorized
    with the console-issued APIPassword as a Bearer token. Per the
    official Spark-X2 HTTP protocol doc the model value is `spark-x`
    and the generation is selected by the URL path (`/x2/` here;
    X1.5 lives under `/v2/`, the retired lite/generalv3 classics under
    `/v1/` — point `SetGeneralLLMServiceConnector` at those paths for
    legacy models). Slim payload: the doc only shows
    messages/stream/temperature-style params, so sampling extras are
    omitted. Pair with `SetSparkLLMServiceConnector`.
    """
    api_url = "https://spark-api-open.xf-yun.com/x2/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
        }
class OpenAIConnectorGeneral(GeneralLLMServiceConnector):
    """OpenAI connector.

    Targets `https://api.openai.com/v1/chat/completions`. Uses
    `max_completion_tokens` (the newer OpenAI spelling) instead of
    `max_tokens` — the gpt-5+/o-series models reject `max_tokens`
    server-side. Sampling params are omitted so the server applies its
    per-model defaults (reasoning-tier models pin temperature anyway).
    Pair with `SetOpenAILLMServiceConnector`.
    """
    api_url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_completion_tokens": kwargs.get("max_tokens", 512),
        }
class GrokConnectorGeneral(StandardOpenAICompatibleConnector):
    """xAI Grok connector.

    Targets `https://api.x.ai/v1/chat/completions` — an
    OpenAI-compatible endpoint that accepts the standard parameter set
    (verified against the xAI docs; `/v1/completions` is legacy).
    Pair with `SetGrokLLMServiceConnector`.
    """
    api_url = "https://api.x.ai/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
class OpenRouterConnectorGeneral(StandardOpenAICompatibleConnector):
    """OpenRouter aggregator connector.

    Targets `https://openrouter.ai/api/v1/chat/completions` — one API
    key routes to 400+ models across OpenAI / Anthropic / Google / xAI /
    DeepSeek / GLM / Qwen / Kimi / MiniMax / MiMo etc. Model ids are
    `<vendor>/<model>` slugs (live list at GET /api/v1/models; the
    Set-node dropdown carries a verified cross-vendor selection).
    Pair with `SetOpenRouterLLMServiceConnector`.
    """
    api_url = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
class ClaudeConnectorGeneral(GeneralLLMServiceConnector):
    """Anthropic Claude connector (official OpenAI-compat layer).

    Targets `https://api.anthropic.com/v1/chat/completions` — Anthropic's
    official OpenAI SDK compatibility layer, authorized with a Bearer
    Claude API key. Per the compat docs many OpenAI fields
    (response_format, penalties, seed, logprobs) are silently ignored
    and `n` must be 1, so this payload stays slim: model / messages /
    stream / max_tokens. Model ids use Anthropic's dashed spelling
    (e.g. `claude-opus-5-5`). For full-feature access (prompt caching,
    structured outputs, thinking) use the native Anthropic API instead.
    Pair with `SetClaudeLLMServiceConnector`.
    """
    api_url = "https://api.anthropic.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
        }
class SiliconFlowJevConnectorGeneral(GeneralLLMServiceConnector):
    """SiliconFlow Jev / System One decision-model connector.

    Jev models (TypeSafe AI's Jev + the open replicas SiliconFlow hosts:
    Kev-4B / SemIf / diffusiongemma) are NOT chat models: the standard
    `/v1/chat/completions` endpoint rejects them with 400. They are
    called on the bare `/v1/systemone` endpoint with a
    `state + questions` payload and answer with structured decisions
    instead of prose — three question types:
      - ``noul``   -> yes/no probability
      - ``choice`` -> picked option + full probability distribution
      - ``score``  -> position on an ordered scale + per-level probs
    Verified live against SiliconFlow's `/v1/systemone` on 2026-09-29
    (same API key as the standard SiliconFlow chat tier; open-replica
    models free until the 2026-10-08 promo ends).

    `ask()` is the real API. `invoke()` exists only so the connector can
    flow through `LLMServiceConnector` sockets — it requires a
    `questions` kwarg and otherwise raises a pointer to the
    `CallJevDecision` node.
    """

    api_url = "https://api.siliconflow.cn/v1/systemone"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def ask(self, state, questions):
        """POST one `state + questions` decision request, return the raw
        response dict (``{model, answers, usage}``). Retry policy mirrors
        `invoke()`: 5xx / Timeout / ConnectionError retried, 4xx raised.
        """
        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries):
            is_last_attempt = attempt == self.max_retries - 1
            attempt_idx = attempt + 1
            tag = f"[{self.model}] ask {attempt_idx}/{self.max_retries}"
            mie_log(f"{tag}: POST {self.api_url} timeout={self.timeout}s")
            attempt_t0 = time.perf_counter()
            try:
                response = requests.post(
                    self.api_url, json=payload, headers=headers, timeout=self.timeout
                )
                elapsed = time.perf_counter() - attempt_t0
                if response.status_code == 200:
                    data = response.json()
                    n_answers = len((data or {}).get("answers") or {})
                    mie_log(
                        f"{tag} ok in {elapsed:.2f}s questions_answered={n_answers}"
                    )
                    return data
                body_snip = (response.text or "").replace("\n", " ")[:200]
                if 500 <= response.status_code < 600 and not is_last_attempt:
                    mie_log(
                        f"{tag} got HTTP {response.status_code} in {elapsed:.2f}s "
                        f"body={body_snip!r}. Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                    continue
                raise Exception(
                    f"{tag} failed with HTTP {response.status_code} in {elapsed:.2f}s "
                    f"body={body_snip!r}"
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                elapsed = time.perf_counter() - attempt_t0
                detail = f"{tag} {type(e).__name__} after {elapsed:.2f}s"
                if is_last_attempt:
                    raise Exception(
                        f"{detail}. Max retries ({self.max_retries}) exceeded."
                    )
                mie_log(f"{detail}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        raise Exception(
            f"Jev decision request failed after {self.max_retries} attempts."
        )

    def invoke(self, messages, **kwargs):
        questions = kwargs.pop("questions", None)
        if not questions:
            raise Exception(
                "Jev decision models answer structured questions, not chat "
                "prompts — CallLLMService cannot drive them. Use the "
                "CallJevDecision node (state + questions JSON) instead."
            )
        state = "\n".join(
            str(m.get("content", ""))
            for m in (messages or [])
            if isinstance(m, dict)
        )
        data = self.ask(state, questions)
        return json.dumps(data.get("answers", data), ensure_ascii=False)
class GeminiConnectorGeneral(GeneralLLMServiceConnector):
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_token, model, **kwargs):
        self.model = model
        # self.api_token = api_token  # Removed, using base class dynamic property
        api_url = f"{self.base_url}/{model}:generateContent"
        # 继承基类的 timeout, max_retries, retry_delay
        super().__init__(api_url, api_token, model, **kwargs)

    # OpenAI `data:<mime>;base64,<data>` -> mime / payload groups
    _DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.*)$", re.DOTALL)

    def _provider_messages(self, messages):
        """Convert OpenAI-style image_url data URLs into Gemini inline_data parts.

        Gemini expects a different content shape from OpenAI - it uses
        `parts` with `inline_data` for images and `text` for prose,
        not the `type: image_url` part shape. We rewrite the user (and
        model) messages in-place so the rest of `generate_payload` can
        iterate the rewritten structure uniformly.

        Non-data-URL `image_url` (e.g. `https://`) is left as-is and the
        Gemini API will resolve it; if the Gemini endpoint rejects it, the
        caller should pre-encode via `image_tensor_batch_to_data_urls`.
        """
        if not messages:
            return messages
        out = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue
            new_parts = []
            for p in content:
                if not isinstance(p, dict):
                    new_parts.append(p)
                    continue
                ptype = p.get("type")
                if ptype == "text" and "text" in p:
                    new_parts.append({"text": p["text"]})
                elif ptype == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    m = self._DATA_URL_RE.match(url)
                    if m:
                        mime_type, b64 = m.group(1), m.group(2)
                        new_parts.append({"inline_data": {"mime_type": mime_type, "data": b64}})
                    elif url:
                        # Remote URL: Gemini can fetch it directly via file_data
                        new_parts.append({"file_data": {"mime_type": "image/jpeg", "file_uri": url}})
                else:
                    # Unknown part type - pass through unchanged so the API
                    # surfaces a clear error rather than us silently losing it.
                    new_parts.append(p)
            new_msg = dict(msg)
            new_msg["content"] = new_parts
            out.append(new_msg)
        return out

    def generate_payload(self, messages, **kwargs):
        contents = []
        for msg in self._provider_messages(messages):
            role = "user" if msg.get("role") == "user" else "model"
            parts = msg.get("content") or []
            if not isinstance(parts, list):
                parts = [{"text": str(parts)}]
            # Drop any empty parts so Gemini does not error
            parts = [p for p in parts if p]
            if not parts:
                parts = [{"text": ""}]
            contents.append({"role": role, "parts": parts})
        return {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 10240),
                "temperature": kwargs.get("temperature", 0.7),
                "topP": kwargs.get("top_p", 0.9),
                "topK": kwargs.get("top_k", 50)
            }
        }

    def invoke(self, messages, **kwargs):
        """
        重写 invoke 方法以处理 Gemini 特有的认证方式 (URL 参数) 和响应解析。
        日志格式与基类 ``GeneralLLMServiceConnector.invoke()`` 对齐，方便排查。
        """
        # Same answer/thought policy as the OpenAI-compatible connectors.
        preserve_thinking, allow_reasoning_fallback = self._pop_answer_policy(kwargs)
        payload = self.generate_payload(messages, **kwargs)
        headers = {"Content-Type": "application/json"}
        # Gemini 认证方式：Token 作为 URL 参数
        url = f"{self.api_url}?key={self.api_token}"

        for attempt in range(self.max_retries):
            is_last_attempt = (attempt == self.max_retries - 1)
            attempt_idx = attempt + 1
            tag = f"[{self.model}] attempt {attempt_idx}/{self.max_retries}"
            req_chars, last_user_chars = self._payload_chars(payload)
            mie_log(
                f"{tag}: POST {url} timeout={self.timeout}s "
                f"request_chars={req_chars} last_user_chars={last_user_chars}"
            )
            attempt_t0 = time.perf_counter()

            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                attempt_elapsed = time.perf_counter() - attempt_t0

                if response.status_code == 200:
                    response_data = response.json()
                    # 适配 Gemini 响应解析: candidates -> content -> parts -> text
                    if not response_data.get("candidates"):
                        raise ValueError(f"No candidates in response. Response: {response.text}")
                    parts = response_data["candidates"][0]["content"].get("parts") or []
                    # Thinking models put the chain in parts flagged
                    # ``thought: true``. Those parts are not the answer, even
                    # when they are first. The fallback policy matches the
                    # OpenAI-compatible connectors.
                    answer, reasoning = self._split_content(parts)
                    cleaned, _saw_reasoning = self._select_answer(
                        answer,
                        reasoning,
                        preserve_thinking=preserve_thinking,
                        allow_reasoning_fallback=allow_reasoning_fallback,
                        tag=tag,
                    )
                    mie_log(
                        f"{tag} ok in {attempt_elapsed:.2f}s "
                        f"response_chars={len(cleaned or '')}"
                    )
                    return cleaned

                if 500 <= response.status_code < 600:
                    body_snip = (response.text or "").replace("\n", " ")[:200]
                    detail = (
                        f"{tag} got HTTP {response.status_code} in {attempt_elapsed:.2f}s "
                        f"body={body_snip!r}"
                    )
                    if is_last_attempt:
                        raise Exception(
                            f"{detail}. Max retries ({self.max_retries}) exceeded."
                        )
                    mie_log(
                        f"{detail}. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                    continue

                body_snip = (response.text or "").replace("\n", " ")[:200]
                raise Exception(
                    f"{tag} failed with HTTP {response.status_code} in {attempt_elapsed:.2f}s "
                    f"body={body_snip!r}"
                )

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                attempt_elapsed = time.perf_counter() - attempt_t0
                error_type = type(e).__name__
                detail = (
                    f"{tag} {error_type} after {attempt_elapsed:.2f}s "
                    f"(request_chars={req_chars}, last_user_chars={last_user_chars})"
                )
                if is_last_attempt:
                    raise Exception(
                        f"{detail}. Max retries ({self.max_retries}) exceeded."
                    )
                mie_log(
                    f"{detail}. Retrying in {self.retry_delay} seconds... "
                    f"(Attempt {attempt_idx}/{self.max_retries})"
                )
                time.sleep(self.retry_delay)
                continue

            except requests.exceptions.RequestException as e:
                raise Exception(f"[{self.model}] A non-retryable request error occurred: {e}")

            except Exception as e:
                # 捕获其他非网络错误，例如 ValueError（如 No candidates in response）
                raise Exception(f"[{self.model}] Unknown error during API call: {e}")

        raise Exception(
            f"[{self.model}] LLM Service failed after {self.max_retries} attempts due to an unknown error."
        )
class OllamaConnectorGeneral(StandardOpenAICompatibleConnector):
    """Ollama (local) OpenAI-compatible connector.

    Posts to ``{host}/v1/chat/completions``. Ollama ships an OpenAI-compat
    layer from 0.3 onwards that supports vision (``image_url`` content
    parts), JSON mode, tools, streaming, and reasoning models. The
    ``api_key`` header is required by the OpenAI-compat shape but is
    ignored by Ollama, so we pass through whatever the caller provided
    (or fall back to a placeholder so the Authorization header stays
    non-empty).

    ``num_ctx`` / ``keep_alive`` / Modelfile-level knobs are NOT exposed
    on the OpenAI-compat endpoint; configure those via Modelfile instead.

    ``host`` should be the bare Ollama server URL without a trailing
    slash and without ``/v1/chat/completions`` -- those are appended
    here. Examples:

      - ``http://127.0.0.1:11434`` (default, local install)
      - ``http://192.168.1.10:11434`` (LAN host running Ollama)
      - ``http://host.docker.internal:11434`` (Ollama in Docker, reached
        from another container on the same host)
    """

    def __init__(self, host, model, api_token=None, **kwargs):
        api_url = host.rstrip("/") + "/v1/chat/completions"
        # Ollama ignores the api_key but the Authorization header must
        # still be non-empty so requests does not choke on a bare Bearer.
        super().__init__(
            api_url,
            api_token or "ollama",
            model,
            **kwargs,
        )
class BailianTokenPlanConnectorGeneral(StandardOpenAICompatibleConnector):
    """Alibaba Bailian Token Plan connector (subscription tier, Beijing).

    Targets the Bailian Token Plan (个人版, Credits-based) endpoint at
    `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions`
    with `sk-sp-...` prefixed subscription API keys issued from
    https://bailian.console.aliyun.com/ (key rotates when the plan is
    re-purchased). Covers the Qwen3.8 / Qwen3.7 flagships plus DeepSeek,
    GLM and multimodal families. NOT interchangeable with the PAYG `sk-`
    key or with the Coding Plan key scoped to
    `coding.dashscope.aliyuncs.com`. Pair with
    `SetBailianTokenPlanLLMServiceConnector`.
    """
    api_url = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
class BailianCodingPlanConnectorGeneral(StandardOpenAICompatibleConnector):
    """Alibaba Bailian Coding Plan connector (subscription tier).

    Targets the Bailian Coding Plan endpoint at
    `https://coding.dashscope.aliyuncs.com/v1/chat/completions` with
    `sk-sp-...` prefixed subscription API keys (per the official Coding
    Plan docs — NOT `sk-cp-`). The Coding Plan model list now includes
    general chat models (qwen3.7-plus, glm-5, MiniMax-M2.5, kimi-k2.5)
    alongside the qwen3-coder family; keys are NOT interchangeable with
    the PAYG `sk-` key or with the Token Plan `sk-sp-` key scoped to the
    `token-plan.cn-beijing.maas.aliyuncs.com` host. Pair with
    `SetBailianCodingPlanLLMServiceConnector`.
    """
    api_url = "https://coding.dashscope.aliyuncs.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)
