"""LiteLLM CustomLLM adapter for the protocol-neutral Cagentrix director."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any

import litellm
from litellm import CustomLLM
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import GenericStreamingChunk, ModelResponse, ModelResponseStream

from cagentrix.director.base import DirectorDecision, TurnRequest
from cagentrix.director.readonly import ReadonlyDirector, normalize_tools, tool_arguments_json
from cagentrix.profiles.loader import load_profile, load_rules
from cagentrix.profiles.models import Profile, ReadonlyRules

PROVIDER_NAME = "cagentrix"
_HISTORY_HEAD_ITEMS = 4
_HISTORY_TAIL_ITEMS = 16
_COMPACTION_ITEM_TYPES = {"compaction", "compaction_trigger", "context_compaction"}
_HANDLER_ARGUMENTS = (
    "model",
    "messages",
    "api_base",
    "custom_prompt_dict",
    "model_response",
    "print_verbose",
    "encoding",
    "api_key",
    "logging_obj",
    "optional_params",
    "acompletion",
    "litellm_params",
    "logger_fn",
    "headers",
    "timeout",
    "client",
)


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_content_text(item) for item in value)
    if isinstance(value, Mapping):
        for key in ("text", "content", "input", "output", "input_text", "output_text"):
            if key in value:
                return _content_text(value[key])
    return ""


def _bounded_value_size(value: Any, remaining: int) -> int:
    """Estimate serialized size while stopping at the reported-token ceiling."""

    if remaining <= 0 or value is None:
        return 0
    if isinstance(value, str):
        return min(len(value) + 2, remaining)
    if isinstance(value, (bytes, bytearray)):
        return min(len(value) + 2, remaining)
    if isinstance(value, Mapping):
        size = 2
        for key, item in value.items():
            if size >= remaining:
                return remaining
            size += _bounded_value_size(key, remaining - size)
            size += _bounded_value_size(item, remaining - size)
        return min(size, remaining)
    if isinstance(value, (list, tuple, set, frozenset)):
        size = 2
        for item in value:
            if size >= remaining:
                return remaining
            size += _bounded_value_size(item, remaining - size)
        return min(size, remaining)
    return min(len(str(value)) + 1, remaining)


def _bounded_history(items: list[Any] | tuple[Any, ...]) -> tuple[Mapping[str, Any], ...]:
    """Keep the initial prompt and recent protocol items needed by the director."""

    if len(items) <= _HISTORY_HEAD_ITEMS + _HISTORY_TAIL_ITEMS:
        selected = items
    else:
        selected = (*items[:_HISTORY_HEAD_ITEMS], *items[-_HISTORY_TAIL_ITEMS:])
    return tuple(item for item in selected if isinstance(item, Mapping))


def _tool_trace(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"python_type": type(value).__name__}
    function = value.get("function")
    function_keys = sorted(function) if isinstance(function, Mapping) else []
    return {
        "type": value.get("type"),
        "name": value.get("name"),
        "function_name": function.get("name") if isinstance(function, Mapping) else None,
        "keys": sorted(value),
        "function_keys": function_keys,
    }


class CagentrixHandler(CustomLLM):
    """A deterministic handler that never calls an upstream model."""

    def __init__(
        self,
        profile: Profile,
        rules: ReadonlyRules,
        root: Path,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.rules = rules
        self.root = root
        self.director = ReadonlyDirector(profile, rules, root)
        self._response_sessions: OrderedDict[str, str] = OrderedDict()

    def completion(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Return a LiteLLM ModelResponse for sync SDK/proxy calls."""

        values = self._call_values(args, kwargs)
        session_id, decision = self._decide(values)
        self._simulate_inference_sync()
        response = self._model_response(values.get("model", self.profile.model), decision, values)
        self._remember_response_session(response.id, session_id)
        return response

    async def acompletion(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Return a LiteLLM ModelResponse for async SDK/proxy calls."""

        values = self._call_values(args, kwargs)
        session_id, decision = self._decide(values)
        await self._simulate_inference_async()
        response = self._model_response(values.get("model", self.profile.model), decision, values)
        self._remember_response_session(response.id, session_id)
        return response

    def streaming(
        self, *args: Any, **kwargs: Any
    ) -> Iterator[GenericStreamingChunk | ModelResponseStream]:
        """Yield the smallest useful generic streaming sequence."""

        values = self._call_values(args, kwargs)
        session_id, decision = self._decide(values)
        response_id = self._response_id()
        self._remember_response_session(response_id, session_id)
        self._simulate_inference_sync()
        yield from self._stream_chunks(decision, response_id, self._prompt_tokens(values))

    async def astreaming(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk | ModelResponseStream]:
        """Yield generic streaming chunks for LiteLLM's async path."""

        values = self._call_values(args, kwargs)
        session_id, decision = self._decide(values)
        response_id = self._response_id()
        self._remember_response_session(response_id, session_id)
        await self._simulate_inference_async()
        for chunk in self._stream_chunks(decision, response_id, self._prompt_tokens(values)):
            yield chunk

    def _simulate_inference_sync(self) -> None:
        delay = self.profile.session_limits.inference_delay_seconds
        if delay > 0:
            time.sleep(delay)

    async def _simulate_inference_async(self) -> None:
        delay = self.profile.session_limits.inference_delay_seconds
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _call_values(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(kwargs)
        values.update(
            {
                name: value
                for name, value in zip(_HANDLER_ARGUMENTS, args, strict=False)
                if name not in values
            }
        )
        optional_params = values.get("optional_params")
        if isinstance(optional_params, Mapping):
            for key in ("tools", "metadata", "input", "previous_response_id"):
                if key not in values and key in optional_params:
                    values[key] = optional_params[key]
        return values

    def _decide(self, values: Mapping[str, Any]) -> tuple[str, DirectorDecision]:
        raw_tools = values.get("tools")
        history = self._history(values)
        if os.environ.get("CAGENTRIX_TRACE_REQUESTS") == "1":
            trace_path = Path(
                os.environ.get("CAGENTRIX_TRACE_PATH", "/tmp/cagentrix-request-trace.jsonl")
            )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "model": values.get("model"),
                            "tool_count": len(raw_tools) if isinstance(raw_tools, list) else None,
                            "tools": [_tool_trace(item) for item in raw_tools]
                            if isinstance(raw_tools, list)
                            else [],
                            "history_types": [item.get("type") for item in history],
                            "history_roles": [item.get("role") for item in history],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        session_id = self._session_id(values, history)
        if _is_compaction_request(values, history):
            return session_id, DirectorDecision(
                tool_call=None,
                stop_reason="Context compacted; continue the read-only inspection loop.",
            )
        request = TurnRequest(
            model=str(values.get("model", self.profile.model)),
            tools=normalize_tools(raw_tools),
            history=history,
            session_id=session_id,
        )
        return session_id, self.director.next_action(request)

    @staticmethod
    def _history(kwargs: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        messages = kwargs.get("messages")
        if isinstance(messages, (list, tuple)):
            return _bounded_history(messages)
        input_items = kwargs.get("input")
        if isinstance(input_items, list):
            return _bounded_history(input_items)
        if isinstance(input_items, str):
            return ({"role": "user", "content": input_items},)
        return ()

    def _session_id(
        self,
        kwargs: Mapping[str, Any],
        history: tuple[Mapping[str, Any], ...],
    ) -> str:
        metadata = kwargs.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("cagentrix_session_id", "session_id", "conversation_id"):
                value = metadata.get(key)
                if isinstance(value, str) and value:
                    return value
        previous_response_id = kwargs.get("previous_response_id")
        if isinstance(previous_response_id, str) and previous_response_id:
            remembered = self._response_sessions.get(previous_response_id)
            if remembered is not None:
                self._response_sessions.move_to_end(previous_response_id)
                return remembered
        for item in history:
            if item.get("role") in {"user", "system"}:
                text = _content_text(item.get("content", item.get("input", "")))
                if text:
                    return "prompt-" + hashlib.sha256(text.encode()).hexdigest()[:24]
        if isinstance(previous_response_id, str) and previous_response_id:
            return "response-" + previous_response_id
        return "default"

    def _model_response(
        self,
        model: str,
        decision: DirectorDecision,
        values: Mapping[str, Any],
    ) -> ModelResponse:
        if decision.tool_call is None:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": decision.stop_reason or "No read-only action is available.",
            }
            finish_reason = "stop"
        else:
            call = decision.tool_call
            message = {
                "role": "assistant",
                "content": decision.preamble,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": tool_arguments_json(decision),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        prompt_tokens = self._prompt_tokens(values)
        return ModelResponse(
            id=self._response_id(),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 8,
                "total_tokens": prompt_tokens + 8,
            },
        )

    def _stream_chunks(
        self,
        decision: DirectorDecision,
        response_id: str,
        prompt_tokens: int,
    ) -> Iterator[GenericStreamingChunk | ModelResponseStream]:
        if decision.tool_call is None:
            yield {
                "text": decision.stop_reason or "No read-only action is available.",
                "tool_use": None,
                "finish_reason": "stop",
                "is_finished": True,
                "index": 0,
                "id": response_id,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 8,
                    "total_tokens": prompt_tokens + 8,
                },
            }
            return
        call = decision.tool_call
        arguments = json.dumps(call.arguments, separators=(",", ":"), sort_keys=True)
        if decision.preamble:
            yield ModelResponseStream(
                id=response_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=self.profile.model,
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": decision.preamble,
                        },
                        "finish_reason": None,
                    }
                ],
            )
        delta: dict[str, Any] = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                }
            ],
        }
        yield ModelResponseStream(
            id=response_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=self.profile.model,
            choices=[
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "tool_calls",
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 8,
                "total_tokens": prompt_tokens + 8,
            },
        )

    def _remember_response_session(self, response_id: str, session_id: str) -> None:
        self._response_sessions[response_id] = session_id
        self._response_sessions.move_to_end(response_id)
        while len(self._response_sessions) > self.profile.session_limits.max_sessions:
            self._response_sessions.popitem(last=False)

    @staticmethod
    def _response_id() -> str:
        return f"cagentrix-{int(time.time() * 1000000)}"

    def _prompt_tokens(self, values: Mapping[str, Any]) -> int:
        """Estimate usage without serializing an unbounded conversation again."""

        max_prompt_tokens = self.profile.context_window
        character_limit = max_prompt_tokens * 4
        size = 2
        for key in ("messages", "input", "tools"):
            size += _bounded_value_size(values.get(key), character_limit - size)
            if size >= character_limit:
                return max_prompt_tokens
        return max(1, min(max_prompt_tokens, (size + 3) // 4))


def _is_compaction_request(
    values: Mapping[str, Any],
    history: tuple[Mapping[str, Any], ...],
) -> bool:
    """Recognize Codex's compact turn without matching unrelated system instructions."""

    for item in reversed(history[-_HISTORY_TAIL_ITEMS:]):
        if item.get("type") in _COMPACTION_ITEM_TYPES:
            return True
    for key in ("compaction", "context_compaction"):
        marker = values.get(key)
        if marker is True or (
            isinstance(marker, Mapping) and marker.get("type") in _COMPACTION_ITEM_TYPES
        ):
            return True
    if normalize_tools(values.get("tools")):
        return False
    if not history:
        return False
    last = history[-1]
    if last.get("role") not in {"user", "developer"}:
        return False
    last_text = _content_text(last.get("content", last.get("input", ""))).lower()
    return any(
        marker in last_text
        for marker in (
            "summarize the conversation",
            "summarise the conversation",
            "context compaction",
            "compact the conversation",
        )
    )


def _environment_handler() -> CagentrixHandler:
    root = Path(os.environ.get("CAGENTRIX_ROOT", Path.cwd())).resolve()
    profile_name = os.environ.get("CAGENTRIX_PROFILE", "codex")
    profile = load_profile(profile_name, root)
    return CagentrixHandler(profile, load_rules(profile, root), root)


cagentrix_handler = _environment_handler()


def register_custom_provider(handler: CagentrixHandler | None = None) -> CagentrixHandler:
    """Register the handler using LiteLLM's documented custom provider map."""

    selected = handler or cagentrix_handler
    current = list(getattr(litellm, "custom_provider_map", []) or [])
    current = [entry for entry in current if entry.get("provider") != PROVIDER_NAME]
    current.append({"provider": PROVIDER_NAME, "custom_handler": selected})
    litellm.custom_provider_map = current
    return selected


register_custom_provider()


class CagentrixResponseNormalizer(CustomLogger):
    """Remove only the empty message emitted beside a Responses tool call.

    LiteLLM owns the Responses conversion. This hook handles the current bridge's
    harmless placeholder message so the wire response contains one function_call item.
    """

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: Any, response: Any
    ) -> Any:
        return self._normalize(response)

    async def async_post_call_success_deployment_hook(
        self, request_data: dict, response: Any, call_type: Any
    ) -> Any:
        return self._normalize(response)

    async def async_post_call_streaming_deployment_hook(
        self, request_data: dict, response_chunk: Any, call_type: Any
    ) -> Any:
        return self._normalize(response_chunk)

    @classmethod
    def _normalize(cls, response: Any) -> Any:
        target = response
        if cls._get(response, "type") == "response.completed":
            target = cls._get(response, "response")

        output = cls._get(target, "output")
        if not isinstance(output, list):
            return response
        if not any(cls._get(item, "type") == "function_call" for item in output):
            return response
        filtered = [item for item in output if not cls._is_empty_message(item)]
        function_calls = [item for item in filtered if cls._get(item, "type") == "function_call"]
        non_function_items = [
            item for item in filtered if cls._get(item, "type") != "function_call"
        ]
        normalized = function_calls + non_function_items
        if normalized == output:
            return response
        cls._set(target, "output", normalized)
        return response

    @staticmethod
    def _get(value: Any, key: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(key)
        return getattr(value, key, None)

    @classmethod
    def _is_empty_message(cls, item: Any) -> bool:
        if cls._get(item, "type") != "message":
            return False
        content = cls._get(item, "content")
        if not isinstance(content, list) or len(content) != 1:
            return False
        content_item = content[0]
        return cls._get(content_item, "type") == "output_text" and not cls._get(
            content_item, "text"
        )

    @staticmethod
    def _set(value: Any, key: str, replacement: Any) -> None:
        if isinstance(value, Mapping):
            value[key] = replacement
        else:
            setattr(value, key, replacement)


cagentrix_response_normalizer = CagentrixResponseNormalizer()
