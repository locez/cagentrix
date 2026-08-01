import json
import socket
from collections.abc import Iterator
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from cagentrix.profiles.loader import load_profile
from cagentrix.runtime.launcher import ProxyLauncher

ROOT = Path(__file__).parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer sk-cagentrix",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "sk-cagentrix",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _post_stream(url: str, payload: dict[str, object]) -> list[dict[str, object]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Authorization": "Bearer sk-cagentrix",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        events = []
        for line in response.read().decode("utf-8").splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))
        return events


@pytest.fixture
def running_proxy(request: pytest.FixtureRequest) -> Iterator[tuple[str, str]]:
    profile_name = str(request.param)
    profile = load_profile(profile_name, ROOT)
    launcher = ProxyLauncher(profile, ROOT, "127.0.0.1", _free_port())
    launcher.start()
    try:
        launcher.wait_until_ready(timeout=30)
        assert launcher.info is not None
        yield launcher.info.api_base, profile_name
    finally:
        launcher.stop()


@pytest.mark.parametrize("running_proxy", ["codex", "opencode", "claude"], indirect=True)
def test_protocols_return_one_tool_call(running_proxy: tuple[str, str]) -> None:
    api_base, profile_name = running_proxy
    schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
    }
    if profile_name == "codex":
        response = _post_json(
            f"{api_base}/responses",
            {
                "model": "cagentrix-codex",
                "input": [{"role": "user", "content": "inspect"}],
                "tools": [
                    {"type": "function", "name": "exec_command", "parameters": schema}
                ],
            },
        )
        output = response["output"]
        assert isinstance(output, list)
        function_calls = [item for item in output if item["type"] == "function_call"]
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "exec_command"
    elif profile_name == "opencode":
        response = _post_json(
            f"{api_base}/chat/completions",
            {
                "model": "cagentrix-opencode",
                "messages": [{"role": "user", "content": "inspect"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "exec_command", "parameters": schema},
                    }
                ],
            },
        )
        tool_calls = response["choices"][0]["message"]["tool_calls"]
        assert isinstance(tool_calls, list)
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "exec_command"
    else:
        response = _post_json(
            f"{api_base}/messages",
            {
                "model": "cagentrix-claude",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "inspect"}],
                "tools": [{"name": "exec_command", "input_schema": schema}],
            },
        )
        content = response["content"]
        assert isinstance(content, list)
        tool_uses = [item for item in content if item["type"] == "tool_use"]
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "exec_command"


@pytest.mark.parametrize("running_proxy", ["codex"], indirect=True)
def test_responses_stream_emits_tool_completion(running_proxy: tuple[str, str]) -> None:
    api_base, _ = running_proxy
    schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
    }
    events = _post_stream(
        f"{api_base}/responses",
        {
            "model": "cagentrix-codex",
            "input": "inspect",
            "stream": True,
            "tools": [
                {"type": "function", "name": "exec_command", "parameters": schema}
            ],
        },
    )

    event_types = [event["type"] for event in events]
    assert "response.function_call_arguments.done" in event_types
    assert event_types[-1] == "response.completed"
    completed = events[-1]["response"]
    assert isinstance(completed, dict)
    function_calls = [item for item in completed["output"] if item["type"] == "function_call"]
    assert len(function_calls) == 1
    assert function_calls[0]["name"] == "exec_command"
