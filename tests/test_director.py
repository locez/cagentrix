from pathlib import Path

from cagentrix.director.base import TurnRequest
from cagentrix.director.readonly import ReadonlyDirector, normalize_tools
from cagentrix.profiles.loader import load_profile, load_rules

ROOT = Path(__file__).parents[1]


def _director() -> ReadonlyDirector:
    profile = load_profile("codex", ROOT)
    return ReadonlyDirector(profile, load_rules(profile, ROOT), ROOT)


def _shell_tool(name: str = "exec_command") -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }


def test_director_returns_one_declared_readonly_tool() -> None:
    director = _director()
    tools = normalize_tools(
        [
            _shell_tool("write_file"),
            _shell_tool(),
            {"type": "custom", "name": "arbitrary_code"},
        ]
    )

    decision = director.next_action(
        TurnRequest(model="cagentrix-codex", tools=tools, session_id="one")
    )

    assert decision.tool_call is not None
    assert decision.tool_call.name == "exec_command"
    assert len([decision.tool_call]) == 1
    command = decision.tool_call.arguments["command"]
    assert isinstance(command, str)
    assert command.startswith(("find ", "rg ", "grep "))
    assert decision.preamble is not None
    assert command in decision.preamble
    assert not any(marker in command for marker in (";", "&&", "||", "`", "$", ">", "<"))


def test_director_prefers_shell_over_native_search_tools() -> None:
    director = _director()
    tools = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "Grep",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                    },
                },
            },
            _shell_tool(),
        ]
    )

    decision = director.next_action(
        TurnRequest(model="cagentrix-codex", tools=tools, session_id="shell-preferred")
    )

    assert decision.tool_call is not None
    assert decision.tool_call.name == "exec_command"
    assert decision.tool_call.arguments["command"]
    assert decision.preamble is not None


def test_native_shell_without_name_is_normalized_and_allowed() -> None:
    tools = normalize_tools([{"type": "shell", "parameters": {"type": "object"}}])

    assert [tool.name for tool in tools] == ["shell"]
    decision = _director().next_action(
        TurnRequest(model="cagentrix-codex", tools=tools, session_id="native-shell")
    )
    assert decision.tool_call is not None
    assert decision.tool_call.name == "shell"


def test_disallowed_or_missing_tools_stop_without_inventing_a_call() -> None:
    director = _director()
    decision = director.next_action(
        TurnRequest(
            model="cagentrix-codex",
            tools=normalize_tools([_shell_tool("write_file")]),
            session_id="none",
        )
    )

    assert decision.tool_call is None
    assert decision.stop_reason


def test_tool_result_history_advances_bounded_session() -> None:
    director = _director()
    tools = normalize_tools([_shell_tool()])
    first = director.next_action(
        TurnRequest(model="cagentrix-codex", tools=tools, session_id="loop")
    )
    second = director.next_action(
        TurnRequest(
            model="cagentrix-codex",
            tools=tools,
            history=(
                {"role": "assistant", "tool_calls": [{"id": "cagentrix_loop_0"}]},
                {"role": "tool", "tool_call_id": "cagentrix_loop_0", "content": "files"},
            ),
            session_id="loop",
        )
    )

    assert first.tool_call is not None
    assert second.tool_call is not None
    assert first.tool_call.id != second.tool_call.id
    assert first.tool_call.arguments != second.tool_call.arguments
    assert director.sessions.size == 1


def test_tool_ids_stay_unique_after_bounded_turn_counter_wraps() -> None:
    director = _director()
    tools = normalize_tools([_shell_tool()])
    history: tuple[dict[str, object], ...] = ()
    call_ids: list[str] = []

    for _ in range(director.sessions.max_events + 8):
        decision = director.next_action(
            TurnRequest(
                model="cagentrix-codex",
                tools=tools,
                history=history,
                session_id="long-loop",
            )
        )
        assert decision.tool_call is not None
        call_id = decision.tool_call.id
        call_ids.append(call_id)
        history = (
            {"role": "assistant", "tool_calls": [{"id": call_id}]},
            {"role": "tool", "tool_call_id": call_id, "content": "files"},
        )

    assert len(call_ids) == len(set(call_ids))
    assert call_ids[director.sessions.max_events] == "cagentrix_long-loop_32"


def test_pending_call_is_replayed_until_a_tool_result_arrives() -> None:
    director = _director()
    tools = normalize_tools([_shell_tool()])
    first = director.next_action(
        TurnRequest(model="cagentrix-codex", tools=tools, session_id="pending")
    )
    replay = director.next_action(
        TurnRequest(model="cagentrix-codex", tools=tools, session_id="pending")
    )
    next_action = director.next_action(
        TurnRequest(
            model="cagentrix-codex",
            tools=tools,
            history=({"type": "function_call_output", "call_id": first.tool_call.id},),  # type: ignore[union-attr]
            session_id="pending",
        )
    )

    assert first.tool_call is not None
    assert replay.tool_call == first.tool_call
    assert next_action.tool_call is not None
    assert next_action.tool_call.id != first.tool_call.id
    assert next_action.preamble is None or isinstance(next_action.preamble, str)


def test_function_tool_arguments_only_fill_known_schema_fields() -> None:
    profile = load_profile("codex", ROOT)
    director = ReadonlyDirector(profile, load_rules(profile, ROOT), ROOT)
    tool = normalize_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "unknown": {"type": "string"},
                        },
                    },
                },
            }
        ]
    )

    decision = director.next_action(
        TurnRequest(model="cagentrix-codex", tools=tool, session_id="read")
    )

    assert decision.tool_call is not None
    assert decision.tool_call.arguments == {"path": "."}
