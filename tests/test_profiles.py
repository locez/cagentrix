from pathlib import Path

from cagentrix.profiles.loader import load_profile, load_rules

ROOT = Path(__file__).parents[1]


def test_builtin_profiles_and_rule_data_load() -> None:
    codex = load_profile("codex", ROOT)
    rules = load_rules(codex, ROOT)

    assert codex.protocol == "responses"
    assert codex.client.command == "codex"
    assert codex.client.client_model == "gpt-5.6-luna"
    assert codex.context_window == 258400
    assert codex.auto_compact_token_limit == 220000
    assert codex.session_limits.inference_delay_seconds == 0.75
    assert any("{api_base}" in argument for argument in codex.client.args)
    assert rules.templates[0].argv[0] == "rg"
    assert {template.argv[0] for template in rules.templates} >= {"find", "grep", "rg", "sed"}
    assert rules.preambles


def test_project_profile_overlay_preserves_data_driven_client_config(tmp_path: Path) -> None:
    agent_dir = tmp_path / ".cagentrix" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "codex.toml").write_text(
        """
[agent]
default_port = 4051

[client]
args = ["--no-alt-screen", "-m", "{model}", "-c", "openai_base_url=\\\"{api_base}\\\""]
""",
        encoding="utf-8",
    )

    profile = load_profile("codex", tmp_path)

    assert profile.default_port == 4051
    assert profile.client.args[0] == "--no-alt-screen"
    assert profile.client.base_url_env == "OPENAI_BASE_URL"
    assert profile.source == agent_dir / "codex.toml"


def test_project_can_define_a_new_profile_without_python_changes(tmp_path: Path) -> None:
    agent_dir = tmp_path / ".cagentrix" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "reader.toml").write_text(
        """
[agent]
name = "reader"
protocol = "chat_completions"
model = "cagentrix-reader"
default_port = 4055
rule_set = "default"

[client]
command = "reader-ui"
args = ["--base-url", "{api_base}"]

[tools]
allowed_patterns = ["(?i)^read_file$"]
shell_patterns = []

[tools.parameter_fields]
path = ["path"]

[session]
max_sessions = 4
max_events = 2
""",
        encoding="utf-8",
    )

    profile = load_profile("reader", tmp_path)

    assert profile.name == "reader"
    assert profile.client.command == "reader-ui"
    assert profile.session_limits.max_events == 2
