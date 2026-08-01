"""Command-line entry point for Cagentrix."""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import sys
from pathlib import Path

from cagentrix import __version__
from cagentrix.config import AppConfig
from cagentrix.profiles.loader import load_profile, load_rules
from cagentrix.runtime.launcher import ProxyLauncher, render_client_argv
from cagentrix.runtime.litellm_config import build_litellm_config

_ENDPOINTS = {
    "responses": "/responses",
    "chat_completions": "/chat/completions",
    "messages": "/messages",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cagentrix",
        description="Run a deterministic, read-only fake LLM API for a coding agent.",
    )
    parser.add_argument("--version", action="version", version=f"Cagentrix {__version__}")
    parser.add_argument("agent", nargs="?", help="profile name, such as codex, opencode, or claude")
    parser.add_argument("--host", default="127.0.0.1", help="local bind address")
    parser.add_argument("--port", type=int, help="local TCP port (defaults to the profile port)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load the profile and print the generated LiteLLM config without starting it",
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="start only the local proxy and do not launch the profile's coding-agent UI",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.agent:
        parser.print_help()
        return 2
    root = Path.cwd().resolve()
    try:
        profile = load_profile(args.agent, root)
        load_rules(profile, root)
        port = args.port or profile.default_port
        if not 1 <= port <= 65535:
            parser.error("--port must be between 1 and 65535")
        config = AppConfig(root, profile.name, args.host, port, args.dry_run)
        api_base = f"http://{config.host}:{config.port}/v1"
        endpoint = api_base + _ENDPOINTS[profile.protocol]
        print(f"Cagentrix profile: {profile.name}")
        print(f"Cagentrix protocol: {profile.protocol}")
        print(f"Cagentrix API base: {api_base}")
        print(f"Cagentrix endpoint: {endpoint}")
        print(f"Cagentrix model: {profile.model}")
        client_argv = render_client_argv(profile, config.root, api_base)
        print(f"Cagentrix client: {shlex.join(client_argv)}")
        if config.dry_run:
            print(json.dumps(build_litellm_config(profile), indent=2))
            return 0
        launcher = ProxyLauncher(profile, config.root, config.host, config.port)
        launcher.start()
        termination_signals = [signal.SIGTERM]
        sighup = getattr(signal, "SIGHUP", None)
        if sighup is not None:
            termination_signals.append(sighup)
        previous_signals = {
            signum: signal.getsignal(signum) for signum in termination_signals
        }

        def handle_sigterm(signum: int, _frame: object) -> None:
            launcher.stop()
            raise SystemExit(128 + signum)

        for signum in termination_signals:
            signal.signal(signum, handle_sigterm)
        try:
            launcher.wait_until_ready()
            if args.server_only:
                print("Cagentrix proxy is running. Press Ctrl-C to stop.", flush=True)
                return launcher.wait()
            print(f"Launching {profile.client.command} UI. Press Ctrl-C to stop.", flush=True)
            return launcher.run_client()
        except KeyboardInterrupt:
            return 0
        finally:
            for signum, handler in previous_signals.items():
                signal.signal(signum, handler)
            launcher.stop()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"cagentrix: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
