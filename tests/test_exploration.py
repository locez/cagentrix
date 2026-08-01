import shlex
from pathlib import Path

from cagentrix.director.base import TurnRequest
from cagentrix.director.command_generator import ReadonlyCommandGenerator
from cagentrix.director.exploration import (
    ExplorationState,
    RepositoryProbe,
    detect_text_locale,
    first_message_locale,
    observation_for_call,
)
from cagentrix.director.readonly import ReadonlyDirector, normalize_tools
from cagentrix.profiles.loader import load_profile, load_rules

ROOT = Path(__file__).parents[1]


def _rules(root: Path):
    profile = load_profile("codex", root)
    return load_rules(profile, root)


def _shell_tools() -> tuple:
    return normalize_tools(
        [
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
    )


def test_locale_switches_to_chinese_when_any_human_message_contains_han_text() -> None:
    assert first_message_locale(
        (
            {"role": "system", "content": "中文系统说明"},
            {"role": "user", "content": "Inspect the project"},
        )
    ) == "en"
    assert first_message_locale(({"role": "user", "content": "请检查项目结构"},)) == "zh"
    assert first_message_locale(
        (
            {"role": "user", "content": "Inspect the project"},
            {"role": "user", "content": "请检查项目结构"},
        )
    ) == "zh"
    assert detect_text_locale("Please inspect `中文`") == "zh"
    assert detect_text_locale("Please inspect 项目") == "zh"

    generator = ReadonlyCommandGenerator(_rules(ROOT), ROOT)
    state = ExplorationState()
    english = generator.generate(
        0,
        history=({"role": "user", "content": "Inspect the project"},),
        state=state,
    )
    chinese = generator.generate(
        1,
        history=(
            {"role": "user", "content": "Inspect the project"},
            {"role": "user", "content": "请检查项目"},
        ),
        state=state,
    )

    assert state.locale == "zh"
    assert english.preamble and "I'll" in english.preamble
    assert chinese.preamble and "我" in chinese.preamble

    chinese_state = ExplorationState()
    chinese_first = generator.generate(
        0,
        history=({"role": "user", "content": "请检查项目"},),
        state=chinese_state,
    )
    assert chinese_state.locale == "zh"
    assert chinese_first.preamble and "我" in chinese_first.preamble


def test_locale_reads_responses_text_blocks() -> None:
    assert first_message_locale(
        (
            {
                "role": "user",
                "content": [{"type": "input_text", "input_text": "请检查项目"}],
            },
        )
    ) == "zh"


def test_probe_uses_language_descriptors_without_language_specific_code(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname = \"signal\"\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name":"web-signal"}\n', encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "lib.rs").write_text(
        "pub struct Signal\npub fn emit_signal() {}\n", encoding="utf-8"
    )
    (tmp_path / "src" / "index.ts").write_text(
        "export class SignalPanel {}\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "signal_test.rs").write_text(
        "use signal::Signal;\n", encoding="utf-8"
    )

    snapshot = RepositoryProbe(_rules(tmp_path)).scan(tmp_path)

    assert {"rust", "node"} <= set(snapshot.languages)
    assert {"Cargo.toml", "package.json"} <= set(snapshot.manifests)
    assert "src/lib.rs" in snapshot.entrypoints or snapshot.entrypoints == ()
    assert any(symbol.name == "Signal" for symbol in snapshot.symbols)
    assert any(
        fact.path == "tests/signal_test.rs" and fact.role == "test"
        for fact in snapshot.files
    )


def test_generator_binds_real_paths_keywords_and_pipelines(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"signal\"\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "class SignalService:\n    pass\n\ndef emit_signal():\n    return SignalService()\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "from src.service import SignalService\n", encoding="utf-8"
    )

    generator = ReadonlyCommandGenerator(_rules(tmp_path), tmp_path)
    state = ExplorationState()
    history = ({"role": "user", "content": "Inspect the signal service"},)
    generated = []
    for turn in range(12):
        item = generator.generate(turn, history=history, state=state)
        generated.append(item)
        state.remember_action(item.signature, template_name=item.template_name)
        state.last_kind = item.action_kind

    assert generated[0].command.startswith("find ")
    assert " | sort | head " in generated[0].command
    assert any(" | " in item.command for item in generated)
    assert any(" | " in item.command for item in generated[:6])
    assert any(
        item.command.startswith("sed ") and " | rg -n " in item.command
        for item in generated[:6]
    )
    assert any(item.command.startswith(("rg ", "grep ")) for item in generated)
    assert all("README.md" not in item.command for item in generated)
    assert any(item.keyword == "SignalService" for item in generated)
    for item in generated:
        if item.path is not None:
            assert (tmp_path / item.path).exists()
        if item.keyword is not None:
            evidence = "\n".join(
                f"{path.relative_to(tmp_path)}\n{path.read_text(encoding='utf-8')}"
                for path in tmp_path.rglob("*")
                if path.is_file()
            )
            assert item.keyword.casefold() in evidence.casefold()
        argv = item.command.replace(" | ", " ").split()
        assert ".." not in argv
        shlex.split(item.command)


def test_generator_rotates_git_and_process_observations_for_this_repository() -> None:
    generator = ReadonlyCommandGenerator(_rules(ROOT), ROOT)
    state = ExplorationState()
    generated = []
    for turn in range(24):
        item = generator.generate(
            turn,
            history=({"role": "user", "content": "Inspect the project"},),
            state=state,
        )
        generated.append(item)
        state.remember_action(item.signature, template_name=item.template_name)
        state.last_kind = item.action_kind

    assert any(item.command.startswith("git ") for item in generated)
    assert any(item.command.startswith("ps -ef | rg ") for item in generated)
    assert sum(item.command.startswith(("git ", "ps ")) for item in generated) >= 5


def test_tool_observation_drives_the_next_focused_command(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"signal\"\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "service.py"
    source.write_text("class SignalService:\n    pass\n", encoding="utf-8")
    director = ReadonlyDirector(
        load_profile("codex", tmp_path),
        _rules(tmp_path),
        tmp_path,
    )
    tools = _shell_tools()
    first = director.next_action(
        TurnRequest(
            model="cagentrix-codex",
            tools=tools,
            history=({"role": "user", "content": "Inspect SignalService"},),
            session_id="observe",
        )
    )
    assert first.tool_call is not None
    observation_history = (
        {
            "type": "function_call_output",
            "call_id": first.tool_call.id,
            "output": "src/service.py:1:class SignalService:",
        },
    )
    observation = observation_for_call(
        observation_history,
        first.tool_call.id,
        tmp_path,
        max_chars=8000,
        max_lines=32,
    )
    assert observation is not None
    assert observation.hits[0].path == "src/service.py"
    second = director.next_action(
        TurnRequest(
            model="cagentrix-codex",
            tools=tools,
            history=observation_history,
            session_id="observe",
        )
    )

    assert second.tool_call is not None
    assert "src/service.py" in second.tool_call.arguments["command"]
