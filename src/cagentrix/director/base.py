"""Small domain objects shared by directors and provider adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """A normalized function tool supplied by the client."""

    name: str
    parameters: dict[str, Any]
    original_type: str = "function"
    description: str | None = None


@dataclass(frozen=True)
class TurnRequest:
    """The protocol-independent input to a director turn."""

    model: str
    tools: tuple[ToolSpec, ...]
    history: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    session_id: str = "default"


@dataclass(frozen=True)
class ToolCall:
    """One tool call emitted by a director."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class DirectorDecision:
    """Either one tool call or a protocol adapter-friendly stop reason."""

    tool_call: ToolCall | None
    stop_reason: str | None = None
    preamble: str | None = None
