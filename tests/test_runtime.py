import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

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
        "-s",
        "read-only",
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


def test_launcher_rejects_an_occupied_port() -> None:
    profile = load_profile("claude", ROOT)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = int(occupied.getsockname()[1])
        launcher = ProxyLauncher(profile, ROOT, "127.0.0.1", port)

        with pytest.raises(RuntimeError, match="already in use"):
            launcher.start()


def test_cli_sigterm_releases_proxy_process_and_port() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cagentrix.cli",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "claude",
            "--server-only",
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("Cagentrix exited before its proxy became ready")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health/liveliness", timeout=1) as response:
                    if 200 <= response.status < 300:
                        break
            except (OSError, URLError):
                time.sleep(0.1)
        else:
            raise AssertionError("Cagentrix proxy did not become ready")

        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 128 + signal.SIGTERM
        with socket.socket() as rebound:
            rebound.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rebound.bind(("127.0.0.1", port))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)


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
    assert data["permission"]["edit"] == "deny"
    assert data["permission"]["external_directory"] == "deny"
    assert data["permission"]["bash"]["*"] == "deny"
    assert data["permission"]["bash"]["rg *"] == "allow"
    assert data["permission"]["bash"]["git --no-pager log *"] == "allow"
    assert data["permission"]["bash"]["git --no-optional-locks status *"] == "allow"
    assert data["permission"]["bash"]["ps -ef*"] == "allow"


def test_claude_custom_model_environment_avoids_upstream_model_validation() -> None:
    profile = load_profile("claude", ROOT)

    assert render_client_argv(profile, ROOT, "http://127.0.0.1:4012/v1") == [
        "claude",
        "--bare",
        "--model",
        "cagentrix-claude",
        "--tools",
        "Bash",
        "--allowedTools",
        "Bash(rg *)",
        "Bash(grep *)",
        "Bash(sed -n *)",
        "Bash(find *)",
        "Bash(sort *)",
        "Bash(head *)",
        "Bash(ps -ef*)",
        "Bash(git --no-optional-locks status *)",
        "Bash(git --no-pager log *)",
        "Bash(git --no-pager diff *)",
        "Bash(git ls-files *)",
        "Bash(git grep *)",
        "--permission-mode",
        "plan",
    ]
    assert render_client_env(profile, ROOT, "http://127.0.0.1:4012/v1") == {
        "ANTHROPIC_CUSTOM_MODEL_OPTION": "cagentrix-claude",
        "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Cagentrix",
        "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": "Local read-only simulator",
    }
    assert profile.client.base_url_template == "{server_url}"
