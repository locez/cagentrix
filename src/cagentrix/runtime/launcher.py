"""Start and cleanly stop a LiteLLM proxy subprocess."""

from __future__ import annotations

import atexit
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self
from urllib.error import URLError
from urllib.request import urlopen

from cagentrix.profiles.models import Profile
from cagentrix.runtime.litellm_config import (
    PROXY_KEY,
    write_codex_model_catalog,
    write_litellm_config,
    write_opencode_config,
)

_EXECUTABLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


@dataclass(frozen=True)
class LaunchInfo:
    host: str
    port: int
    profile: Profile
    config_path: Path

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def render_client_argv(
    profile: Profile,
    root: Path,
    api_base: str,
    model_catalog_json: Path | None = None,
) -> list[str]:
    """Render a profile's executable and argument templates without a shell."""

    replacements = _client_replacements(
        profile,
        root,
        api_base,
        model_catalog_json,
    )
    args = [profile.client.command]
    for argument in profile.client.args:
        args.append(_render_client_value(argument, replacements))
    return args


def render_client_env(profile: Profile, root: Path, api_base: str) -> dict[str, str]:
    """Render profile-declared child environment values."""

    replacements = _client_replacements(profile, root, api_base)
    return {
        key: _render_client_value(value, replacements) for key, value in profile.client.env.items()
    }


def _client_replacements(
    profile: Profile,
    root: Path,
    api_base: str,
    model_catalog_json: Path | None = None,
) -> dict[str, str]:
    return {
        "{api_base}": api_base,
        "{server_url}": api_base.removesuffix("/v1"),
        "{model}": profile.model,
        "{client_model}": profile.client.client_model or profile.model,
        "{context_window}": str(profile.context_window),
        "{auto_compact_token_limit}": str(profile.auto_compact_token_limit),
        "{root}": str(root),
        "{model_catalog_json}": str(model_catalog_json or "<generated-by-cagentrix>"),
    }


def _render_client_value(value: str, replacements: dict[str, str]) -> str:
    rendered = value
    for placeholder, replacement in replacements.items():
        rendered = rendered.replace(placeholder, replacement)
    return rendered


def resolve_client_executable(command: str) -> str | None:
    """Resolve a client from PATH and common Nix profile locations."""

    resolved = shutil.which(command)
    if resolved is not None:
        return resolved
    if not _EXECUTABLE_NAME.fullmatch(command):
        return None

    locations: list[Path] = []
    for raw_profile in os.environ.get("NIX_PROFILES", "").split(":"):
        if raw_profile:
            locations.append(Path(raw_profile) / "bin" / command)
    locations.extend(
        [
            Path.home() / ".nix-profile" / "bin" / command,
            Path("/nix/var/nix/profiles/default/bin") / command,
        ]
    )
    store = Path("/nix/store")
    if store.is_dir():
        locations.extend(sorted(store.glob(f"*-profile/bin/{command}")))
    for candidate in locations:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


class ProxyLauncher:
    """Own the temporary LiteLLM config and child process lifetime."""

    def __init__(self, profile: Profile, root: Path, host: str, port: int) -> None:
        self.profile = profile
        self.root = root
        self.host = host
        self.port = port
        self._temp_dir: Path | None = None
        self._model_catalog_path: Path | None = None
        self._client_config_dir: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._process_group: int | None = None
        self._client_process: subprocess.Popen[bytes] | None = None
        self._client_process_group: int | None = None
        self._exit_cleanup: Callable[[], None] | None = None
        self.info: LaunchInfo | None = None

    def start(self) -> LaunchInfo:
        if self._process is not None:
            raise RuntimeError("Cagentrix proxy is already running")
        self._assert_port_available()
        self._temp_dir = Path(tempfile.mkdtemp(prefix="cagentrix-"))
        config_path = write_litellm_config(self._temp_dir, self.profile)
        api_base = f"http://{self.host}:{self.port}/v1"
        if any("{model_catalog_json}" in argument for argument in self.profile.client.args):
            self._model_catalog_path = write_codex_model_catalog(self._temp_dir, self.profile)
        if self.profile.client.generated_config == "opencode":
            self._client_config_dir = write_opencode_config(
                self._temp_dir, self.profile, api_base
            )
        litellm_executable = Path(sys.executable).with_name("litellm")
        if not litellm_executable.exists():
            resolved = shutil.which("litellm")
            if resolved is None:
                raise RuntimeError("LiteLLM console script is not installed; run `uv sync`")
            litellm_executable = Path(resolved)
        environment = os.environ.copy()
        environment.update(
            {
                "CAGENTRIX_PROFILE": self.profile.name,
                "CAGENTRIX_ROOT": str(self.root),
            }
        )
        self._process = subprocess.Popen(
            [
                str(litellm_executable),
                "--config",
                str(config_path),
                "--host",
                self.host,
                "--port",
                str(self.port),
            ],
            cwd=self.root,
            env=environment,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        self._process_group = self._process.pid
        self.info = LaunchInfo(self.host, self.port, self.profile, config_path)
        self._register_exit_cleanup()
        return self.info

    def _assert_port_available(self) -> None:
        """Reject an occupied port before a stale proxy can satisfy readiness checks."""

        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((self.host, self.port))
        except OSError as exc:
            raise RuntimeError(
                f"port {self.port} is already in use; stop the existing proxy or choose --port"
            ) from exc

    def wait_until_ready(self, timeout: float = 20.0) -> None:
        """Wait until LiteLLM accepts requests, or fail with a useful error."""

        process = self._process
        info = self.info
        if process is None or info is None:
            raise RuntimeError("Cagentrix proxy is not running")
        check_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        health_url = f"http://{check_host}:{self.port}/health/liveliness"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("LiteLLM proxy exited before becoming ready")
            try:
                with urlopen(health_url, timeout=1) as response:
                    ready = 200 <= response.status < 300
            except (OSError, URLError):
                time.sleep(0.1)
                continue
            if process.poll() is not None:
                raise RuntimeError("LiteLLM proxy exited before becoming ready")
            if ready:
                return
        raise RuntimeError(f"LiteLLM proxy did not become ready within {timeout:g} seconds")

    def _register_exit_cleanup(self) -> None:
        if self._exit_cleanup is not None:
            return

        def cleanup() -> None:
            self.stop()

        self._exit_cleanup = cleanup
        atexit.register(cleanup)

    def client_argv(self) -> list[str]:
        if self.info is None:
            raise RuntimeError("Cagentrix proxy is not running")
        return render_client_argv(
            self.profile,
            self.root,
            self.info.api_base,
            self._model_catalog_path,
        )

    def run_client(self) -> int:
        """Run the profile's interactive client in the user's terminal."""

        info = self.info
        if info is None:
            raise RuntimeError("Cagentrix proxy is not running")
        argv = self.client_argv()
        executable = resolve_client_executable(argv[0])
        if executable is None:
            raise FileNotFoundError(
                f"agent executable not found: {argv[0]!r}; install it or use --server-only"
            )
        environment = os.environ.copy()
        environment.update(render_client_env(self.profile, self.root, info.api_base))
        if self.profile.client.base_url_env:
            replacements = _client_replacements(self.profile, self.root, info.api_base)
            environment[self.profile.client.base_url_env] = _render_client_value(
                self.profile.client.base_url_template,
                replacements,
            )
        if self.profile.client.api_key_env:
            environment[self.profile.client.api_key_env] = PROXY_KEY
        if self._client_config_dir is not None:
            environment["XDG_CONFIG_HOME"] = str(self._client_config_dir)
        self._client_process = subprocess.Popen(
            [executable, *argv[1:]],
            cwd=self.root,
            env=environment,
            start_new_session=True,
        )
        self._client_process_group = self._client_process.pid
        client_process = self._client_process
        return client_process.wait()

    def wait(self) -> int:
        if self._process is None:
            raise RuntimeError("Cagentrix proxy is not running")
        return self._process.wait()

    def stop(self) -> None:
        cleanup = self._exit_cleanup
        self._exit_cleanup = None
        if cleanup is not None:
            atexit.unregister(cleanup)
        client_process = self._client_process
        self._client_process = None
        client_process_group = self._client_process_group
        self._client_process_group = None
        self._terminate_process(client_process, client_process_group)
        process = self._process
        self._process = None
        process_group = self._process_group
        self._process_group = None
        self._terminate_process(process, process_group)
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        self._model_catalog_path = None
        self._client_config_dir = None

    @staticmethod
    def _terminate_process(
        process: subprocess.Popen[bytes] | None,
        process_group: int | None,
    ) -> None:
        if process is None:
            return
        if process_group is None:
            try:
                candidate = os.getpgid(process.pid)
            except ProcessLookupError:
                candidate = None
            process_group = candidate if candidate == process.pid else None
        try:
            if process_group is not None:
                os.killpg(process_group, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        except ProcessLookupError:
            pass
        if process_group is not None:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
