import json
from pathlib import Path

from cagentrix.profiles.loader import load_profile
from cagentrix.runtime.launcher import (
    ProxyLauncher,
    render_client_argv,
    render_client_env,
    resolve_client_executable,
)
from cagentrix.runtime.litellm_config import (
    build_litellm_config,
    supports_responses_chat_bridge,
    write_codex_model_catalog,
    write_opencode_config,
)

ROOT = Path(__file__).parents[1]


def test_litellm_config_registers_custom_provider_and_bridge() -> None:
    profile = load_profile("codex", ROOT)
    config = build_litellm_config(profile)

    assert supports_responses_chat_bridge()
    assert config["model_list"][0]["litellm_params"]["model"] == "cagentrix/cagentrix-codex"
    assert config["model_list"][0]["litellm_params"]["use_chat_completions_api"] is True
    assert config["model_list"][1]["model_name"] == "gpt-5.6-luna"
    assert config["model_list"][1]["litellm_params"]["model"] == "cagentrix/cagentrix-codex"
    assert config["litellm_settings"]["drop_params"] is True
    assert config["litellm_settings"]["custom_provider_map"][0]["provider"] == "cagentrix"
    assert config["litellm_settings"]["callbacks"]
    assert "general_settings" not in config


def test_codex_client_command_points_to_the_local_proxy() -> None:
    profile = load_profile("codex", ROOT)

    argv = render_client_argv(profile, ROOT, "http://127.0.0.1:4051/v1")

    assert argv == [
        "codex",
        "-m",
        "gpt-5.6-luna",
        "-c",
        'model_provider="cagentrix"',
        "-c",
        'model_providers.cagentrix.name="Cagentrix"',
        "-c",
        'model_providers.cagentrix.base_url="http://127.0.0.1:4051/v1"',
        "-c",
        'model_providers.cagentrix.wire_api="responses"',
        "-c",
        "model_context_window=258400",
        "-c",
        "model_auto_compact_token_limit=220000",
        "-c",
        "features.remote_compaction_v2=false",
        "-c",
        'model_catalog_json="<generated-by-cagentrix>"',
    ]


def test_codex_catalog_keeps_client_model_tool_capable(tmp_path: Path) -> None:
    profile = load_profile("codex", ROOT)
    path = write_codex_model_catalog(tmp_path, profile)
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data["models"][0]

    assert model["slug"] == "gpt-5.6-luna"
    assert model["context_window"] == 258400
    assert model["truncation_policy"]["limit"] == 220000
    assert model["supports_parallel_tool_calls"] is False
    assert "use_responses_lite" not in model
    assert "tool_mode" not in model


def test_launcher_can_render_client_command_before_start() -> None:
    profile = load_profile("codex", ROOT)
    launcher = ProxyLauncher(profile, ROOT, "127.0.0.1", 4051)

    assert launcher.info is None
    assert launcher.profile.client.command == "codex"


def test_installed_nix_clients_are_resolvable_without_path_aliases() -> None:
    opencode = resolve_client_executable("opencode")
    claude = resolve_client_executable("claude")

    assert opencode is not None
    assert claude is not None


def test_opencode_argv_and_generated_provider_are_local() -> None:
    profile = load_profile("opencode", ROOT)
    argv = render_client_argv(profile, ROOT, "http://127.0.0.1:4011/v1")

    assert argv == ["opencode", "--model", "cagentrix/cagentrix-opencode"]

    path = write_opencode_config(ROOT / ".tmp-test-opencode", profile, "http://127.0.0.1:4011/v1")
    try:
        data = json.loads((path / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    finally:
        import shutil

        shutil.rmtree(ROOT / ".tmp-test-opencode", ignore_errors=True)

    assert data["model"] == "cagentrix/cagentrix-opencode"
    assert data["provider"]["cagentrix"]["options"]["baseURL"] == (
        "http://127.0.0.1:4011/v1"
    )


def test_claude_custom_model_environment_avoids_upstream_model_validation() -> None:
    profile = load_profile("claude", ROOT)

    assert render_client_argv(profile, ROOT, "http://127.0.0.1:4012/v1") == [
        "claude",
        "--bare",
        "--model",
        "cagentrix-claude",
        "--tools",
        "Bash",
        "--permission-mode",
        "dontAsk",
    ]
    assert render_client_env(profile, ROOT, "http://127.0.0.1:4012/v1") == {
        "ANTHROPIC_CUSTOM_MODEL_OPTION": "cagentrix-claude",
        "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Cagentrix",
        "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "Local read-only simulator",
    }
    assert profile.client.base_url_template == "{server_url}"
