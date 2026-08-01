"""Deterministic read-only tool selection and command generation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from cagentrix.director.base import DirectorDecision, ToolCall, ToolSpec, TurnRequest
from cagentrix.director.command_generator import ReadonlyCommandGenerator
from cagentrix.director.session import SessionStore
from cagentrix.profiles.models import Profile, ReadonlyRules


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        dumped = to_dict()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    return None


def normalize_tools(raw_tools: Iterable[Any] | None) -> tuple[ToolSpec, ...]:
    """Keep only usable function tools and normalize Responses/Anthropic shapes.

    Native Responses ``shell`` tools are represented as function candidates because
    their input still has to be constrained by the profile's command template. Native
    ``custom`` tools are intentionally dropped: LiteLLM cannot safely bridge them to
    the function shape consumed by this server.
    """

    result: list[ToolSpec] = []
    for raw_tool in raw_tools or ():
        tool = _as_mapping(raw_tool)
        if tool is None:
            continue
        tool_type = tool.get("type")
        if tool_type == "custom":
            continue
        if tool_type not in (None, "function", "shell"):
            continue
        function = _as_mapping(tool.get("function")) or tool
        name = function.get("name") or tool.get("name")
        if tool_type == "shell" and not name:
            name = "shell"
        if not isinstance(name, str) or not name:
            continue
        parameters = (
            _as_mapping(function.get("parameters"))
            or _as_mapping(function.get("input_schema"))
            or {"type": "object", "properties": {}}
        )
        description = function.get("description")
        result.append(
            ToolSpec(
                name=name,
                parameters=parameters,
                original_type=str(tool_type or "function"),
                description=description if isinstance(description, str) else None,
            )
        )
    return tuple(result)


class ReadonlyDirector:
    """Select exactly one client-declared read-only tool per turn."""

    def __init__(self, profile: Profile, rules: ReadonlyRules, root: Path) -> None:
        self.profile = profile
        self.rules = rules
        self.root = root
        self.sessions = SessionStore(
            max_sessions=profile.session_limits.max_sessions,
            max_events=profile.session_limits.max_events,
        )
        self.command_generator = ReadonlyCommandGenerator(rules, root)
        self._allowed = [re.compile(pattern) for pattern in profile.tool_rules.allowed_patterns]
        self._shell = [re.compile(pattern) for pattern in profile.tool_rules.shell_patterns]

    def next_action(self, request: TurnRequest) -> DirectorDecision:
        state = self.sessions.state_for(request.session_id)
        if state.pending_call_id is not None and _has_tool_result(
            request.history, state.pending_call_id
        ):
            self.sessions.clear_pending(request.session_id)
            state = self.sessions.state_for(request.session_id)

        if state.pending_call_id is not None and state.pending_tool_name is not None:
            return DirectorDecision(
                tool_call=ToolCall(
                    state.pending_call_id,
                    state.pending_tool_name,
                    dict(state.pending_arguments),
                ),
                preamble=state.pending_preamble,
            )

        turn = state.turns
        generated = self.command_generator.generate(turn)
        candidate = next((tool for tool in request.tools if self._is_allowed(tool.name)), None)
        if candidate is None:
            return DirectorDecision(
                tool_call=None,
                stop_reason="No client-declared read-only function tool is available.",
            )
        if self._is_shell(candidate.name):
            arguments = self._shell_arguments(candidate, generated.command)
        else:
            arguments = self._function_arguments(candidate, turn, generated.command)
        call_id = self._call_id(request.session_id, turn)
        self.sessions.record_call(
            request.session_id,
            call_id=call_id,
            tool_name=candidate.name,
            arguments=arguments,
            preamble=generated.preamble,
        )
        return DirectorDecision(
            tool_call=ToolCall(call_id, candidate.name, arguments),
            preamble=generated.preamble,
        )

    def _is_allowed(self, name: str) -> bool:
        return any(pattern.search(name) for pattern in self._allowed)

    def _is_shell(self, name: str) -> bool:
        return any(pattern.search(name) for pattern in self._shell)

    def _call_id(self, session_id: str, turn: int) -> str:
        safe_session = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)[:24] or "default"
        return f"cagentrix_{safe_session}_{turn}"

    def _shell_arguments(self, tool: ToolSpec, command: str) -> dict[str, str]:
        return {self._field_for(tool, "command", fallback="command"): command}

    def _function_arguments(self, tool: ToolSpec, turn: int, command: str) -> dict[str, Any]:
        schema = tool.parameters
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        arguments: dict[str, Any] = {}
        for field_name, field_schema in properties.items():
            if not isinstance(field_name, str):
                continue
            semantic = self._semantic_field(field_name)
            if semantic is None:
                continue
            arguments[field_name] = self._value_for(semantic, field_schema, turn, command)
        return arguments

    def _field_for(self, tool: ToolSpec, semantic: str, *, fallback: str) -> str:
        properties = tool.parameters.get("properties", {})
        if isinstance(properties, Mapping):
            aliases = self.profile.tool_rules.parameter_fields.get(semantic, ())
            for alias in aliases:
                if alias in properties:
                    return alias
        return fallback

    def _semantic_field(self, field_name: str) -> str | None:
        lowered = field_name.lower()
        for semantic, aliases in self.profile.tool_rules.parameter_fields.items():
            if lowered in {alias.lower() for alias in aliases}:
                return semantic
        return None

    def _value_for(self, semantic: str, field_schema: Any, turn: int, command: str) -> Any:
        schema = field_schema if isinstance(field_schema, Mapping) else {}
        if "default" in schema:
            return schema["default"]
        if semantic == "command":
            return command
        if semantic == "pattern":
            return self.rules.patterns[turn % len(self.rules.patterns)]
        if semantic == "path":
            return "."
        if semantic == "limit":
            return self.rules.max_results
        kind = schema.get("type")
        if kind == "boolean":
            return False
        if kind == "integer" or kind == "number":
            return 0
        if kind == "array":
            return []
        return "Cagentrix"


def _has_tool_result(history: tuple[Mapping[str, Any], ...], call_id: str) -> bool:
    """Find the pending result near the end without rescanning old transcripts."""

    for item in reversed(history[-16:]):
        if _contains_tool_result(item, call_id):
            return True
    return False


def _contains_tool_result(value: Any, call_id: str) -> bool:
    if isinstance(value, Mapping):
        item_type = value.get("type")
        if value.get("role") == "tool" or item_type in {"function_call_output", "tool_result"}:
            if any(value.get(key) == call_id for key in ("tool_call_id", "tool_use_id", "call_id")):
                return True
        return any(
            _contains_tool_result(nested, call_id)
            for nested in value.values()
            if isinstance(nested, (Mapping, list, tuple))
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_tool_result(item, call_id) for item in reversed(value[-16:]))
    return False


def tool_arguments_json(decision: DirectorDecision) -> str:
    """Serialize a call's arguments in the form required by OpenAI tools."""

    if decision.tool_call is None:
        return "{}"
    return json.dumps(decision.tool_call.arguments, separators=(",", ":"), sort_keys=True)
