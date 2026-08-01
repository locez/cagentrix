"""Typed profile and read-only rule models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandStage:
    """One validated argv stage in a read-only command pipeline."""

    argv: tuple[str, ...]


@dataclass(frozen=True)
class LanguageDescriptor:
    """Data-driven language and project conventions used by the explorer."""

    name: str
    extensions: tuple[str, ...]
    manifests: tuple[str, ...] = ()
    source_dirs: tuple[str, ...] = ()
    test_dirs: tuple[str, ...] = ()
    definition_patterns: tuple[str, ...] = ()
    entrypoint_paths: tuple[str, ...] = ()
    test_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class NarrativeTemplate:
    """A localized, data-driven explanation for one exploration moment."""

    locale: str
    trigger: str
    text: str


@dataclass(frozen=True)
class CommandTemplate:
    """A validated read-only command or pipeline recipe."""

    name: str
    argv: tuple[str, ...]
    pipeline: tuple[CommandStage, ...] | None = None
    kind: str = "inventory"
    requires: tuple[str, ...] = ()
    path_kind: str = "scope"
    keyword_kind: str = "none"
    weight: int = 1

    @property
    def stages(self) -> tuple[CommandStage, ...]:
        """Return the structured stages while retaining the legacy argv field."""

        return self.pipeline or (CommandStage(self.argv),)


@dataclass(frozen=True)
class ReadonlyRules:
    """The auditable command-generation data loaded from TOML."""

    name: str
    patterns: tuple[str, ...]
    templates: tuple[CommandTemplate, ...]
    max_results: int
    preambles: tuple[str, ...] = ()
    preamble_interval: int = 4
    languages: tuple[LanguageDescriptor, ...] = ()
    max_scan_files: int = 512
    max_scan_bytes: int = 2_000_000
    max_file_bytes: int = 128_000
    max_observation_chars: int = 8_000
    max_observation_lines: int = 32
    max_pipeline_stages: int = 3
    pipeline_bonus: int = 18
    ignored_dirs: tuple[str, ...] = (
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "node_modules",
        "target",
        "dist",
        "build",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    )
    narratives: tuple[NarrativeTemplate, ...] = ()
    default_locale: str = "en"


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
    inference_delay_seconds: float = 1.0


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
