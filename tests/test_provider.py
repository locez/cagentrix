import asyncio
import json
from dataclasses import replace
from pathlib import Path

import cagentrix.provider.litellm_handler as handler_module
from cagentrix.profiles.loader import load_profile, load_rules
from cagentrix.provider.litellm_handler import CagentrixHandler

ROOT = Path(__file__).parents[1]


def _handler(*, delay: float = 0.0) -> CagentrixHandler:
    profile = load_profile("codex", ROOT)
    profile = replace(
        profile,
        session_limits=replace(
            profile.session_limits,
            inference_delay_seconds=delay,
        ),
    )
    return CagentrixHandler(profile, load_rules(profile, ROOT), ROOT)


def _tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            },
        }
    ]


def test_handler_returns_model_response_with_one_tool_call() -> None:
    response = _handler().completion(
        model="cagentrix-codex",
        messages=[{"role": "user", "content": "inspect"}],
        tools=_tools(),
        metadata={"cagentrix_session_id": "provider-test"},
    )
    data = response.model_dump()
    tool_calls = data["choices"][0]["message"]["tool_calls"]

    assert data["object"] == "chat.completion"
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "exec_command"


def test_handler_uses_chinese_preamble_for_a_chinese_request() -> None:
    response = _handler().completion(
        model="cagentrix-codex",
        messages=[{"role": "user", "content": "请检查项目结构"}],
        tools=_tools(),
    )

    content = response.model_dump()["choices"][0]["message"]["content"]
    assert content and content.startswith("我")


def test_handler_simulates_configured_inference_delay(monkeypatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr(handler_module.time, "sleep", delays.append)

    _handler(delay=0.75).completion(
        model="cagentrix-codex",
        messages=[{"role": "user", "content": "inspect"}],
        tools=_tools(),
    )

    assert delays == [0.75]


def test_sync_async_and_streaming_handler_paths() -> None:
    handler = _handler()
    kwargs = {
        "model": "cagentrix-codex",
        "messages": [{"role": "user", "content": "inspect"}],
        "tools": _tools(),
    }

    async_response = asyncio.run(handler.acompletion(**kwargs))
    async_chunks = asyncio.run(_collect(handler.astreaming(**kwargs)))
    sync_chunks = list(handler.streaming(**kwargs))

    assert len(async_response.model_dump()["choices"][0]["message"]["tool_calls"]) == 1
    assert len(async_chunks) == 1
    assert len(sync_chunks) == 1
    async_data = async_chunks[0].model_dump()
    sync_data = sync_chunks[0].model_dump()
    assert async_data["choices"][0]["delta"]["content"]
    assert sync_data["choices"][0]["delta"]["content"]
    async_tool_use = async_data["choices"][0]["delta"]["tool_calls"][0]
    sync_tool_use = sync_data["choices"][0]["delta"]["tool_calls"][0]
    assert async_tool_use["function"]["name"] == "exec_command"
    assert sync_tool_use["function"]["name"] == "exec_command"
    command = json.loads(async_tool_use["function"]["arguments"])["command"]
    assert command.startswith(("find ", "rg ", "grep ", "sed "))
    assert " | " in command
    assert async_data["usage"]["prompt_tokens"] > 0


def test_provider_tool_result_advances_generator_and_usage_is_nonzero() -> None:
    handler = _handler()
    tools = _tools()
    first_messages = [{"role": "user", "content": "inspect"}]
    first = handler.completion(
        model="cagentrix-codex",
        messages=first_messages,
        tools=tools,
        metadata={"cagentrix_session_id": "provider-loop"},
    )
    first_call = first.model_dump()["choices"][0]["message"]["tool_calls"][0]
    replay = handler.completion(
        model="cagentrix-codex",
        messages=first_messages,
        tools=tools,
        metadata={"cagentrix_session_id": "provider-loop"},
    )
    next_response = handler.completion(
        model="cagentrix-codex",
        messages=[
            *first_messages,
            {"role": "assistant", "tool_calls": [first_call]},
            {"role": "tool", "tool_call_id": first_call["id"], "content": "file output"},
        ],
        tools=tools,
        metadata={"cagentrix_session_id": "provider-loop"},
    )

    replay_call = replay.model_dump()["choices"][0]["message"]["tool_calls"][0]
    next_call = next_response.model_dump()["choices"][0]["message"]["tool_calls"][0]
    assert replay_call == first_call
    assert next_call["id"] != first_call["id"]
    assert first.usage.prompt_tokens > 0


def test_handler_answers_codex_compaction_turn_without_a_tool_call() -> None:
    response = _handler().completion(
        model="cagentrix-codex",
        input=[
            {"role": "user", "content": "summarize the conversation"},
            {"type": "compaction"},
        ],
    )

    message = response.model_dump()["choices"][0]["message"]
    assert message["content"].startswith("Context compacted;")
    assert message.get("tool_calls") is None


def test_handler_does_not_misclassify_compaction_text_in_old_history() -> None:
    response = _handler().completion(
        model="cagentrix-codex",
        messages=[
            {"role": "system", "content": "The runtime supports context compaction."},
            {"role": "user", "content": "hi"},
        ],
        tools=_tools(),
    )

    assert response.model_dump()["choices"][0]["message"]["tool_calls"]


def test_prompt_usage_is_bounded_for_large_histories() -> None:
    response = _handler().completion(
        model="cagentrix-codex",
        messages=[{"role": "user", "content": "x" * 1_200_000}],
        tools=_tools(),
    )

    assert response.usage.prompt_tokens == 258_400


async def _collect(iterator: object) -> list[object]:
    return [item async for item in iterator]  # type: ignore[union-attr]
