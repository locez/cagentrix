"""Load built-in profiles and optional per-project TOML overrides."""

from __future__ import annotations

import re
import tomllib
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

from cagentrix.config import project_config_dir
from cagentrix.profiles.models import (
    ClientConfig,
    CommandTemplate,
    Profile,
    ReadonlyRules,
    SessionLimits,
    ToolRules,
    _as_string_tuple,
)

BUILTIN_DIR = Path(__file__).with_name("builtin")
_AGENT_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_PROTOCOLS = {"responses", "chat_completions", "messages"}
_READONLY_COMMANDS = {"find", "grep", "rg", "sed"}
_PREAMBLE_PLACEHOLDERS = {"command", "template"}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _positive_int(value: Any, *, field: str, default: int) -> int:
    selected = default if value is None else value
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
        raise ValueError(f"{field} must be a positive integer")
    return selected


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cagentrix configuration file not found: {path}") from exc


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _parse_tool_rules(data: dict[str, Any]) -> ToolRules:
    tools = data.get("tools")
    if not isinstance(tools, dict):
        raise ValueError("profile must define [tools]")
    fields = tools.get("parameter_fields", {})
    if not isinstance(fields, dict):
        raise ValueError("[tools.parameter_fields] must be a table")
    mappings = {
        key: _as_string_tuple(value, field=f"tools.parameter_fields.{key}")
        for key, value in fields.items()
    }
    return ToolRules(
        allowed_patterns=_as_string_tuple(
            tools.get("allowed_patterns"), field="tools.allowed_patterns"
        ),
        shell_patterns=_as_string_tuple(tools.get("shell_patterns"), field="tools.shell_patterns"),
        parameter_fields=mappings,
    )


def _parse_client(data: dict[str, Any]) -> ClientConfig:
    client = data.get("client")
    if not isinstance(client, dict):
        raise ValueError("profile must define [client]")
    command = client.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("client.command must be a non-empty string")
    args = _as_string_tuple(client.get("args", []), field="client.args")
    client_model = client.get("model")
    base_url_env = client.get("base_url_env")
    base_url_template = client.get("base_url_template", "{api_base}")
    api_key_env = client.get("api_key_env")
    generated_config = client.get("generated_config")
    raw_env = client.get("env", {})
    if not isinstance(raw_env, dict):
        raise ValueError("client.env must be a table")
    client_env: dict[str, str] = {}
    for key, value in raw_env.items():
        if not isinstance(key, str) or not _ENV_NAME.fullmatch(key):
            raise ValueError(f"client.env has an invalid environment variable: {key!r}")
        if not isinstance(value, str):
            raise ValueError(f"client.env.{key} must be a string")
        client_env[key] = value
    string_fields = (
        ("client.model", client_model),
        ("client.base_url_env", base_url_env),
        ("client.base_url_template", base_url_template),
        ("client.api_key_env", api_key_env),
        ("client.generated_config", generated_config),
    )
    for field, value in string_fields:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field} must be a non-empty string when set")
    return ClientConfig(
        command=command.strip(),
        args=args,
        client_model=client_model.strip() if isinstance(client_model, str) else None,
        base_url_env=base_url_env.strip() if isinstance(base_url_env, str) else None,
        base_url_template=base_url_template.strip()
        if isinstance(base_url_template, str)
        else "{api_base}",
        api_key_env=api_key_env.strip() if isinstance(api_key_env, str) else None,
        generated_config=(
            generated_config.strip() if isinstance(generated_config, str) else None
        ),
        env=client_env,
    )


def load_profile(name: str, root: Path) -> Profile:
    """Load a built-in profile, then merge ``.cagentrix/agents/<name>.toml``."""

    if not _AGENT_NAME.fullmatch(name):
        raise ValueError(f"invalid agent name: {name!r}")
    builtin_path = BUILTIN_DIR / f"{name}.toml"
    override_path = project_config_dir(root) / "agents" / f"{name}.toml"
    if builtin_path.exists():
        data = _read_toml(builtin_path)
        if override_path.exists():
            data = _deep_merge(data, _read_toml(override_path))
    elif override_path.exists():
        data = _read_toml(override_path)
    else:
        raise FileNotFoundError(f"Cagentrix profile not found: {name}")

    agent = data.get("agent")
    if not isinstance(agent, dict):
        raise ValueError(f"{builtin_path} must define [agent]")
    profile_name = agent.get("name")
    protocol = agent.get("protocol")
    model = agent.get("model")
    if profile_name != name or not isinstance(profile_name, str):
        raise ValueError(f"profile name must match requested agent {name!r}")
    if protocol not in _PROTOCOLS:
        raise ValueError(f"unsupported protocol in {builtin_path}: {protocol!r}")
    if not isinstance(model, str) or not model:
        raise ValueError("agent.model must be a non-empty string")
    context_window = _positive_int(
        agent.get("context_window"), field="agent.context_window", default=131_072
    )
    auto_compact_token_limit = _positive_int(
        agent.get("auto_compact_token_limit"),
        field="agent.auto_compact_token_limit",
        default=max(1, int(context_window * 0.85)),
    )
    if auto_compact_token_limit >= context_window:
        raise ValueError("agent.auto_compact_token_limit must be smaller than context_window")
    default_port = agent.get("default_port")
    if not isinstance(default_port, int) or not 1 <= default_port <= 65535:
        raise ValueError("agent.default_port must be a TCP port")
    use_bridge = agent.get("use_chat_completions_api", False)
    if not isinstance(use_bridge, bool):
        raise ValueError("agent.use_chat_completions_api must be boolean")

    session = data.get("session", {})
    if not isinstance(session, dict):
        raise ValueError("[session] must be a table")
    limits = SessionLimits(
        max_sessions=int(session.get("max_sessions", 128)),
        max_events=int(session.get("max_events", 32)),
        inference_delay_seconds=float(session.get("inference_delay_seconds", 0.75)),
    )
    if limits.max_sessions < 1 or limits.max_events < 1:
        raise ValueError("session limits must be positive")
    if (
        not isfinite(limits.inference_delay_seconds)
        or limits.inference_delay_seconds < 0
    ):
        raise ValueError("session.inference_delay_seconds must be a finite non-negative number")
    rules_file = agent.get("rules_file")
    if rules_file is not None and (
        not isinstance(rules_file, str) or Path(rules_file).is_absolute()
    ):
        raise ValueError("agent.rules_file must be a relative path")
    return Profile(
        name=name,
        protocol=protocol,
        model=model,
        use_chat_completions_api=use_bridge,
        context_window=context_window,
        auto_compact_token_limit=auto_compact_token_limit,
        default_port=default_port,
        rule_set=str(agent.get("rule_set", "default")),
        rules_file=rules_file,
        tool_rules=_parse_tool_rules(data),
        session_limits=limits,
        client=_parse_client(data),
        source=override_path if override_path.exists() else builtin_path,
    )


def _rules_path(profile: Profile, root: Path) -> Path:
    if profile.rules_file:
        return root / ".cagentrix" / profile.rules_file
    return root / ".cagentrix" / "readonly_rules.toml"


def load_rules(profile: Profile, root: Path) -> ReadonlyRules:
    """Load built-in command rules with an optional project-level replacement."""

    builtin_path = BUILTIN_DIR / "readonly_rules.toml"
    data = _read_toml(builtin_path)
    custom_path = _rules_path(profile, root)
    if custom_path.exists():
        data = _deep_merge(data, _read_toml(custom_path))
    rule_sets = data.get("rule_sets")
    if not isinstance(rule_sets, dict) or not isinstance(rule_sets.get(profile.rule_set), dict):
        raise ValueError(f"unknown read-only rule set: {profile.rule_set!r}")
    rule_data = rule_sets[profile.rule_set]
    patterns = _as_string_tuple(rule_data.get("patterns"), field="rule_set.patterns")
    raw_templates = rule_data.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("rule_set.templates must be a non-empty array")
    templates: list[CommandTemplate] = []
    for index, raw in enumerate(raw_templates):
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError(f"rule_set.templates[{index}] must have a name")
        argv = _as_string_tuple(raw.get("argv"), field=f"rule_set.templates[{index}].argv")
        if not argv or argv[0] not in _READONLY_COMMANDS:
            raise ValueError("read-only command templates must start with find, grep, rg, or sed")
        has_shell_operator = any(
            any(marker in part for marker in (";", "&&", "||", ">", "<", "`", "$", "\n"))
            for part in argv
        )
        if has_shell_operator:
            raise ValueError("shell operators are not allowed in read-only command templates")
        if argv[0] == "find" and any(
            part in {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
            or part.startswith(("-fprint", "-fls"))
            for part in argv[1:]
        ):
            raise ValueError(
                "find write and execution actions are not allowed in read-only command templates"
            )
        if argv[0] == "sed" and any(
            part in {"-i", "--in-place"} or part.startswith(("-i", "--in-place="))
            for part in argv[1:]
        ):
            raise ValueError("sed in-place edits are not allowed in read-only command templates")
        if any(".." in Path(part).parts for part in argv[1:] if "{" not in part):
            raise ValueError("read-only command templates cannot escape the project directory")
        templates.append(CommandTemplate(name=raw["name"], argv=argv))
    max_results = int(rule_data.get("max_results", 40))
    if max_results < 1 or max_results > 1000:
        raise ValueError("rule_set.max_results must be between 1 and 1000")
    preambles = _as_string_tuple(rule_data.get("preambles", []), field="rule_set.preambles")
    for preamble in preambles:
        placeholders = set(re.findall(r"\{([a-z_]+)\}", preamble))
        if not placeholders <= _PREAMBLE_PLACEHOLDERS:
            raise ValueError("rule_set.preambles may only use {command} and {template}")
    preamble_interval = int(rule_data.get("preamble_interval", 4))
    if preamble_interval < 1:
        raise ValueError("rule_set.preamble_interval must be positive")
    return ReadonlyRules(
        name=profile.rule_set,
        patterns=patterns,
        templates=tuple(templates),
        max_results=max_results,
        preambles=preambles,
        preamble_interval=preamble_interval,
    )
