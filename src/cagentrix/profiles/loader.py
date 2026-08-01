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
    CommandStage,
    CommandTemplate,
    LanguageDescriptor,
    NarrativeTemplate,
    Profile,
    ReadonlyRules,
    SessionLimits,
    ToolRules,
    _as_string_tuple,
)

BUILTIN_DIR = Path(__file__).with_name("builtin")
_AGENT_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_PROTOCOLS = {"responses", "chat_completions", "messages"}
_READONLY_COMMANDS = {"find", "grep", "rg", "sed", "head", "sort", "git", "ps"}
_GIT_GLOBAL_FLAGS = {"--no-pager", "--no-optional-locks"}
_GIT_READONLY_SUBCOMMANDS = {
    "diff",
    "grep",
    "log",
    "ls-files",
    "rev-parse",
    "status",
}
_GIT_BLOCKED_FLAGS = {
    "-C",
    "--exec-path",
    "--git-dir",
    "--work-tree",
    "--upload-pack",
    "--receive-pack",
    "--config-env",
}
_TEMPLATE_KINDS = {
    "inventory",
    "metadata",
    "entrypoint",
    "definition",
    "reference",
    "source_test_pair",
    "config_boundary",
    "focused_inspection",
    "cross_check",
    "fallback",
}
_PATH_KINDS = {
    "scope",
    "all_scope",
    "source_scope",
    "test_scope",
    "manifest_scope",
    "manifest",
    "entrypoint",
    "source_file",
    "test_file",
    "doc_file",
    "hit_file",
    "any_file",
}
_KEYWORD_KINDS = {"none", "identifier", "definition", "reference", "focused", "process"}
_TEMPLATE_PLACEHOLDERS = {
    "glob",
    "keyword",
    "line_range",
    "max_results",
    "path",
    "pattern",
    "root",
    "scope",
}
_PREAMBLE_PLACEHOLDERS = {"command", "keyword", "path", "reason", "template"}
_LOCALE = re.compile(r"^[a-z]{2}(?:-[A-Z]{2})?$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_SED_FILE_COMMAND = re.compile(
    r"(?:^|[;{}\n])\s*(?:[0-9,$*?/'\"\\]+)?[eErRwW](?=\s|$|[;}]|[0-9])"
)


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
        inference_delay_seconds=float(session.get("inference_delay_seconds", 1.0)),
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


def _parse_languages(data: Any) -> tuple[LanguageDescriptor, ...]:
    if data is None:
        return ()
    if not isinstance(data, list):
        raise ValueError("languages must be an array of tables")
    descriptors: list[LanguageDescriptor] = []
    for index, raw in enumerate(data):
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError(f"languages[{index}] must have a name")
        name = raw["name"].strip()
        if not name or len(name) > 40:
            raise ValueError(f"languages[{index}].name must be a short non-empty string")
        definition_patterns = _as_string_tuple(
            raw.get("definition_patterns", []),
            field=f"languages[{index}].definition_patterns",
        )
        for pattern in definition_patterns:
            if len(pattern) > 400:
                raise ValueError(f"languages[{index}].definition_patterns contains a long pattern")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"languages[{index}] has an invalid definition pattern") from exc
        descriptors.append(
            LanguageDescriptor(
                name=name,
                extensions=_as_string_tuple(
                    raw.get("extensions", []), field=f"languages[{index}].extensions"
                ),
                manifests=_as_string_tuple(
                    raw.get("manifests", []), field=f"languages[{index}].manifests"
                ),
                source_dirs=_as_string_tuple(
                    raw.get("source_dirs", []), field=f"languages[{index}].source_dirs"
                ),
                test_dirs=_as_string_tuple(
                    raw.get("test_dirs", []), field=f"languages[{index}].test_dirs"
                ),
                definition_patterns=definition_patterns,
                entrypoint_paths=_as_string_tuple(
                    raw.get("entrypoint_paths", []),
                    field=f"languages[{index}].entrypoint_paths",
                ),
                test_patterns=_as_string_tuple(
                    raw.get("test_patterns", []), field=f"languages[{index}].test_patterns"
                ),
            )
        )
    return tuple(descriptors)


def _parse_narratives(data: Any) -> tuple[NarrativeTemplate, ...]:
    if data is None:
        return ()
    if not isinstance(data, list):
        raise ValueError("rule_set.narratives must be an array of tables")
    narratives: list[NarrativeTemplate] = []
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"rule_set.narratives[{index}] must be a table")
        locale = raw.get("locale")
        trigger = raw.get("trigger", "any")
        text = raw.get("text")
        if (
            not isinstance(locale, str)
            or not _LOCALE.fullmatch(locale)
            or not isinstance(trigger, str)
            or not re.fullmatch(r"[a-z_]{1,32}", trigger)
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(f"rule_set.narratives[{index}] has invalid locale, trigger, or text")
        placeholders = set(re.findall(r"\{([a-z_]+)\}", text))
        if not placeholders <= _PREAMBLE_PLACEHOLDERS:
            raise ValueError(
                "rule_set.narratives may only use {command}, {keyword}, {path}, "
                "{reason}, and {template}"
            )
        narratives.append(NarrativeTemplate(locale=locale, trigger=trigger, text=text))
    return tuple(narratives)


def _parse_template_stages(raw: dict[str, Any], *, field: str) -> tuple[CommandStage, ...]:
    if "pipeline" in raw:
        pipeline = raw["pipeline"]
        if not isinstance(pipeline, list) or not pipeline:
            raise ValueError(f"{field}.pipeline must be a non-empty array")
        stages: list[CommandStage] = []
        for stage_index, stage in enumerate(pipeline):
            stages.append(
                CommandStage(
                    _as_string_tuple(stage, field=f"{field}.pipeline[{stage_index}]")
                )
            )
        return tuple(stages)
    return (
        CommandStage(_as_string_tuple(raw.get("argv"), field=f"{field}.argv")),
    )


def _validate_template_stage(stage: CommandStage, *, field: str) -> None:
    argv = stage.argv
    if not argv or argv[0] not in _READONLY_COMMANDS:
        raise ValueError(f"{field} must start with a validated read-only command")
    for part in argv:
        placeholders = set(re.findall(r"\{([a-z_]+)\}", part))
        if "{" in part or "}" in part:
            if part not in {f"{{{name}}}" for name in _TEMPLATE_PLACEHOLDERS}:
                raise ValueError(f"{field} has an unsupported placeholder")
        if any(marker in part for marker in (";", "&&", "||", ">", "<", "`", "$", "\n")):
            raise ValueError("shell operators are not allowed in read-only command templates")
        if placeholders and not placeholders <= _TEMPLATE_PLACEHOLDERS:
            raise ValueError(f"{field} has an unsupported placeholder")
        if (
            not placeholders
            and (Path(part).is_absolute() or _WINDOWS_ABSOLUTE_PATH.match(part))
        ):
            raise ValueError("read-only command templates cannot use absolute paths")
        if ".." in Path(part).parts and not placeholders:
            raise ValueError("read-only command templates cannot escape the project directory")
    if argv[0] == "find" and any(
        part in {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
        or part.startswith(("-fprint", "-fls"))
        for part in argv[1:]
    ):
        raise ValueError(
            f"{field} contains a find write or execution action"
        )
    if argv[0] == "sed" and any(
        part in {"-i", "--in-place"} or part.startswith(("-i", "--in-place="))
        for part in argv[1:]
    ):
        raise ValueError(f"{field} contains a sed in-place edit")
    if argv[0] == "sort" and any(
        part in {"-o", "--output"} or part.startswith("--output=")
        for part in argv[1:]
    ):
        raise ValueError(f"{field} contains a sort output file")
    if argv[0] == "git":
        _validate_git_stage(argv, field=field)
    if argv[0] == "ps" and any(
        part not in {"-e", "-f", "-ef", "--no-headers"} for part in argv[1:]
    ):
        raise ValueError(f"{field} contains an unsupported ps option")
    if argv[0] == "sed":
        if any(
            part in {"-f", "--file"} or part.startswith("--file=")
            for part in argv[1:]
        ):
            raise ValueError(f"{field} cannot load an external sed program")
        for program in _sed_programs(argv):
            if _SED_FILE_COMMAND.search(program) or _sed_substitution_executes(program):
                raise ValueError(f"{field} contains a sed file or execution command")
    if argv[0] == "find" and any(part in {"-H", "-L"} for part in argv[1:]):
        raise ValueError(f"{field} cannot follow symbolic links outside the project")
    if argv[0] == "rg" and any(part in {"-L", "--follow"} for part in argv[1:]):
        raise ValueError(f"{field} cannot follow symbolic links outside the project")
    if argv[0] == "grep" and any(
        part in {"-R", "--dereference-recursive"} for part in argv[1:]
    ):
        raise ValueError(f"{field} cannot follow symbolic links outside the project")
    if argv[0] in {"rg", "grep"} and any(
        part in {"--pre", "--pre-glob", "--hostname-bin", "--replace"}
        or part.startswith(("--pre=", "--hostname-bin="))
        for part in argv[1:]
    ):
        raise ValueError(f"{field} contains an execution-capable search option")


def _validate_git_stage(argv: tuple[str, ...], *, field: str) -> None:
    index = 1
    while index < len(argv) and argv[index] in _GIT_GLOBAL_FLAGS:
        index += 1
    if index >= len(argv) or argv[index] not in _GIT_READONLY_SUBCOMMANDS:
        raise ValueError(f"{field} must use a validated read-only git subcommand")
    subcommand = argv[index]
    args = argv[index + 1 :]
    if any(
        part in _GIT_BLOCKED_FLAGS
        or part.startswith(("--git-dir=", "--work-tree=", "--upload-pack="))
        for part in args
    ):
        raise ValueError(f"{field} contains a git path or execution override")
    if subcommand == "diff":
        allowed = {"--no-ext-diff", "--stat", "--name-only", "--", "{path}"}
    elif subcommand == "grep":
        allowed = {"-n", "--line-number", "-I", "-e", "--", "{keyword}", "{scope}"}
    elif subcommand == "log":
        allowed = {"--oneline", "--decorate=no", "-n", "--max-count=12", "12", "--", "{path}"}
    elif subcommand == "ls-files":
        allowed = {"--cached", "--", "{scope}"}
    elif subcommand == "status":
        allowed = {
            "--short",
            "--porcelain=v1",
            "--branch",
            "--untracked-files=no",
            "--no-renames",
        }
    else:
        allowed = {"--show-toplevel", "--show-prefix"}
    if any(
        part not in allowed and not part.isdigit()
        for part in args
    ):
        raise ValueError(f"{field} contains an unsupported git {subcommand} option")


def _sed_programs(argv: tuple[str, ...]) -> tuple[str, ...]:
    programs: list[str] = []
    index = 1
    while index < len(argv):
        part = argv[index]
        if part in {"-e", "--expression"}:
            if index + 1 < len(argv):
                programs.append(argv[index + 1])
                index += 2
                continue
            break
        if part.startswith("--expression="):
            programs.append(part.split("=", 1)[1])
            index += 1
            continue
        if part == "--":
            if index + 1 < len(argv):
                programs.append(argv[index + 1])
            break
        if part.startswith("-"):
            index += 1
            continue
        programs.append(part)
        break
    return tuple(programs)


def _sed_substitution_executes(program: str) -> bool:
    """Reject the GNU sed ``s///e`` execution flag without parsing full sed."""

    for index, value in enumerate(program):
        if value != "s" or (
            index > 0 and program[index - 1] not in ";\n{}0123456789,$/'\""
        ):
            continue
        if index + 1 >= len(program) or program[index + 1].isalnum():
            continue
        delimiter = program[index + 1]
        first = _next_sed_delimiter(program, index + 2, delimiter)
        if first is None:
            continue
        second = _next_sed_delimiter(program, first + 1, delimiter)
        if second is None:
            continue
        flags = program[second + 1 :].split(";", 1)[0].split("}", 1)[0]
        if "e" in flags.lower():
            return True
    return False


def _next_sed_delimiter(program: str, start: int, delimiter: str) -> int | None:
    escaped = False
    for index in range(start, len(program)):
        value = program[index]
        if escaped:
            escaped = False
        elif value == "\\":
            escaped = True
        elif value == delimiter:
            return index
    return None


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
    patterns = _as_string_tuple(rule_data.get("patterns", []), field="rule_set.patterns")
    raw_templates = rule_data.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("rule_set.templates must be a non-empty array")
    max_pipeline_stages = int(rule_data.get("max_pipeline_stages", 3))
    if not 1 <= max_pipeline_stages <= 4:
        raise ValueError("rule_set.max_pipeline_stages must be between 1 and 4")
    pipeline_bonus = int(rule_data.get("pipeline_bonus", 18))
    if not 0 <= pipeline_bonus <= 100:
        raise ValueError("rule_set.pipeline_bonus must be between 0 and 100")
    templates: list[CommandTemplate] = []
    for index, raw in enumerate(raw_templates):
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError(f"rule_set.templates[{index}] must have a name")
        field = f"rule_set.templates[{index}]"
        stages = _parse_template_stages(raw, field=field)
        if len(stages) > max_pipeline_stages:
            raise ValueError(f"{field} exceeds max_pipeline_stages")
        for stage_index, stage in enumerate(stages):
            _validate_template_stage(stage, field=f"{field}.pipeline[{stage_index}]")
        kind = raw.get("kind", "inventory")
        path_kind = raw.get("path_kind", "scope")
        keyword_kind = raw.get("keyword_kind", "none")
        if kind not in _TEMPLATE_KINDS:
            raise ValueError(f"{field}.kind is not a supported exploration category")
        if path_kind not in _PATH_KINDS:
            raise ValueError(f"{field}.path_kind is not a supported path slot")
        if keyword_kind not in _KEYWORD_KINDS:
            raise ValueError(f"{field}.keyword_kind is not a supported keyword slot")
        requires = _as_string_tuple(raw.get("requires", []), field=f"{field}.requires")
        weight = raw.get("weight", 1)
        if isinstance(weight, bool) or not isinstance(weight, int) or not 1 <= weight <= 100:
            raise ValueError(f"{field}.weight must be between 1 and 100")
        templates.append(
            CommandTemplate(
                name=raw["name"],
                argv=stages[0].argv,
                pipeline=stages,
                kind=kind,
                requires=requires,
                path_kind=path_kind,
                keyword_kind=keyword_kind,
                weight=weight,
            )
        )
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
    narratives = _parse_narratives(rule_data.get("narratives", []))
    default_locale = rule_data.get("default_locale", "en")
    if not isinstance(default_locale, str) or not _LOCALE.fullmatch(default_locale):
        raise ValueError("rule_set.default_locale must be a locale such as en or zh")
    languages = _parse_languages(data.get("languages", []))
    bounded_fields = {
        "max_scan_files": (32, 4096, 512),
        "max_scan_bytes": (16_384, 16_000_000, 2_000_000),
        "max_file_bytes": (4_096, 1_000_000, 128_000),
        "max_observation_chars": (512, 64_000, 8_000),
        "max_observation_lines": (4, 256, 32),
    }
    limits: dict[str, int] = {}
    for field, (minimum, maximum, default) in bounded_fields.items():
        value = int(rule_data.get(field, default))
        if not minimum <= value <= maximum:
            raise ValueError(f"rule_set.{field} must be between {minimum} and {maximum}")
        limits[field] = value
    ignored_dirs = _as_string_tuple(
        rule_data.get("ignored_dirs", [
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
        ]),
        field="rule_set.ignored_dirs",
    )
    return ReadonlyRules(
        name=profile.rule_set,
        patterns=patterns,
        templates=tuple(templates),
        max_results=max_results,
        preambles=preambles,
        preamble_interval=preamble_interval,
        languages=languages,
        max_scan_files=limits["max_scan_files"],
        max_scan_bytes=limits["max_scan_bytes"],
        max_file_bytes=limits["max_file_bytes"],
        max_observation_chars=limits["max_observation_chars"],
        max_observation_lines=limits["max_observation_lines"],
        max_pipeline_stages=max_pipeline_stages,
        pipeline_bonus=pipeline_bonus,
        ignored_dirs=ignored_dirs,
        narratives=narratives,
        default_locale=default_locale,
    )
