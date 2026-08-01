"""Typed profile and read-only rule models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandTemplate:
    """A validated argv template for a read-only repository command."""

    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ReadonlyRules:
    """The auditable command-generation data loaded from TOML."""

    name: str
    patterns: tuple[str, ...]
    templates: tuple[CommandTemplate, ...]
    max_results: int
    preambles: tuple[str, ...] = ()
    preamble_interval: int = 4


@dataclass(frozen=True)
class ToolRules:
    """Profile-specific tool matching and argument field mappings."""

    allowed_patterns: tuple[str, ...]
    shell_patterns: tuple[str, ...]
    parameter_fields: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class SessionLimits:
    """Bounds preventing an unattended client from growing memory forever."""

    max_sessions: int = 128
    max_events: int = 32
    inference_delay_seconds: float = 0.75


@dataclass(frozen=True)
class ClientConfig:
    """How the selected coding-agent UI is started after the proxy is ready."""

    command: str
    args: tuple[str, ...]
    client_model: str | None = None
    base_url_env: str | None = None
    base_url_template: str = "{api_base}"
    api_key_env: str | None = None
    generated_config: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Profile:
    """A protocol-neutral agent profile loaded from TOML."""

    name: str
    protocol: str
    model: str
    use_chat_completions_api: bool
    context_window: int
    auto_compact_token_limit: int
    default_port: int
    rule_set: str
    rules_file: str | None
    tool_rules: ToolRules
    session_limits: SessionLimits
    client: ClientConfig
    source: Path


def _as_string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be an array of strings")
    return tuple(value)
