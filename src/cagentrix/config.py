"""Application configuration and project path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Runtime values selected by the CLI."""

    root: Path
    agent: str
    host: str
    port: int
    dry_run: bool = False


def project_config_dir(root: Path) -> Path:
    """Return the optional per-project Cagentrix configuration directory."""

    return root / ".cagentrix"
