"""Generate the temporary LiteLLM proxy configuration."""

from __future__ import annotations

import inspect
import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import litellm

from cagentrix.profiles.models import Profile

CUSTOM_HANDLER = "cagentrix.provider.litellm_handler.cagentrix_handler"
RESPONSE_NORMALIZER = "cagentrix.provider.litellm_handler.cagentrix_response_normalizer"
PROXY_KEY = "sk-cagentrix"


def _version_tuple(raw_version: str) -> tuple[int, int, int]:
    numbers = [int(item) for item in re.findall(r"\d+", raw_version)[:3]]
    padded = (numbers + [0, 0, 0])[:3]
    return padded[0], padded[1], padded[2]


def installed_litellm_version() -> str:
    try:
        return version("litellm")
    except PackageNotFoundError as exc:
        raise RuntimeError("LiteLLM is not installed; run `uv sync` first") from exc


def supports_responses_chat_bridge() -> bool:
    """Check both the minimum release and the installed bridge implementation."""

    if _version_tuple(installed_litellm_version()) < (1, 80, 0):
        return False
    try:
        source = inspect.getsource(litellm.responses)
    except (OSError, TypeError):
        return False
    return "use_chat_completions_api" in source and "_pop_use_chat_completions_api_kw" in source


def build_litellm_config(profile: Profile) -> dict[str, Any]:
    """Build a JSON-compatible config accepted by ``litellm --config``."""

    if profile.use_chat_completions_api and not supports_responses_chat_bridge():
        current = installed_litellm_version()
        raise RuntimeError(
            "Cagentrix profile requires LiteLLM use_chat_completions_api support; "
            f"installed version is {current}. Upgrade with `uv lock --upgrade-package litellm`."
        )
    params: dict[str, Any] = {
        "model": f"{CUSTOM_HANDLER.split('.')[0]}/{profile.model}",
        "api_key": PROXY_KEY,
    }
    if profile.use_chat_completions_api:
        params["use_chat_completions_api"] = True
    model_list = [
        {
            "model_name": profile.model,
            "litellm_params": params,
        }
    ]
    client_model = profile.client.client_model
    if client_model and client_model != profile.model:
        model_list.append(
            {
                "model_name": client_model,
                "litellm_params": dict(params),
            }
        )
    return {
        "model_list": model_list,
        "litellm_settings": {
            "drop_params": True,
            "custom_provider_map": [
                {
                    "provider": "cagentrix",
                    "custom_handler": CUSTOM_HANDLER,
                }
            ],
            "callbacks": [RESPONSE_NORMALIZER],
        },
    }


def write_litellm_config(directory: Path, profile: Profile) -> Path:
    """Write a temporary JSON config (JSON is also valid LiteLLM config input)."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "litellm.json"
    path.write_text(json.dumps(build_litellm_config(profile), indent=2) + "\n", encoding="utf-8")
    return path


def write_codex_model_catalog(directory: Path, profile: Profile) -> Path:
    """Write a tool-capable model catalog for Codex's local model metadata.

    Codex uses local metadata to decide which tools to send before it makes the
    first request.  Some model entries opt into Responses Lite/code-mode and
    consequently omit the normal function tools.  The generated entry keeps
    the selected client model name visible in the UI while deliberately
    omitting those opt-in flags.
    """

    directory.mkdir(parents=True, exist_ok=True)
    model = profile.client.client_model or profile.model
    catalog = {
        "models": [
            {
                "slug": model,
                "display_name": model,
                "description": "Cagentrix local read-only simulation.",
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "Fast responses"},
                    {"effort": "medium", "description": "Balanced responses"},
                    {"effort": "high", "description": "Deeper responses"},
                ],
                "base_instructions": (
                    "You are Codex working in a local read-only inspection session."
                ),
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 1,
                "support_verbosity": True,
                "default_verbosity": "low",
                "default_reasoning_summary": "none",
                "include_skills_usage_instructions": False,
                "context_window": profile.context_window,
                "max_context_window": profile.context_window,
                "supports_parallel_tool_calls": False,
                "experimental_supported_tools": [],
                "supports_search_tool": False,
                "supports_image_detail_original": False,
                "multi_agent_version": "v1",
                "truncation_policy": {
                    "mode": "tokens",
                    "limit": profile.auto_compact_token_limit,
                },
            }
        ]
    }
    path = directory / "codex-models.json"
    path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return path


def write_opencode_config(directory: Path, profile: Profile, api_base: str) -> Path:
    """Write an isolated OpenCode config containing Cagentrix's local provider."""

    if profile.client.generated_config != "opencode":
        raise ValueError(
            "unsupported generated client config: "
            f"{profile.client.generated_config!r}"
        )
    config_root = directory / "xdg-config"
    config_path = config_root / "opencode" / "opencode.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    model = profile.model
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"cagentrix/{model}",
        "provider": {
            "cagentrix": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Cagentrix",
                "options": {
                    "baseURL": api_base,
                    "apiKey": PROXY_KEY,
                },
                "models": {
                    model: {
                        "name": "Cagentrix read-only simulator",
                        "limit": {"context": profile.context_window, "output": 1024},
                    }
                },
            }
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_root
