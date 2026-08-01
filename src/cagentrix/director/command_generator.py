"""Auditable, read-only command generation for a repository context."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from cagentrix.profiles.models import ReadonlyRules


@dataclass(frozen=True)
class GeneratedCommand:
    """One safe command and optional human-like planning text."""

    command: str
    preamble: str | None
    template_name: str


class ReadonlyCommandGenerator:
    """Render only validated rule templates against the selected project root."""

    def __init__(self, rules: ReadonlyRules, root: Path) -> None:
        self.rules = rules
        self.root = root

    def generate(self, turn: int) -> GeneratedCommand:
        """Return a deterministic command for a bounded, repeatable busy loop."""

        template_count = len(self.rules.templates)
        pattern_count = len(self.rules.patterns)
        template = self.rules.templates[turn % template_count]
        pattern = self.rules.patterns[(turn // template_count) % pattern_count]
        rendered: list[str] = []
        for part in template.argv:
            if part == "{pattern}":
                rendered.append(pattern)
            elif part == "{root}":
                rendered.append(".")
            elif part == "{max_results}":
                rendered.append(str(self.rules.max_results))
            elif "{" in part or "}" in part:
                raise ValueError(f"unsupported placeholder in rule template {template.name!r}")
            else:
                rendered.append(part)
        preamble = None
        if self.rules.preambles and turn % self.rules.preamble_interval == 0:
            preamble_index = (turn // self.rules.preamble_interval) % len(self.rules.preambles)
            preamble = self._render_preamble(
                self.rules.preambles[preamble_index],
                command=shlex.join(rendered),
                template_name=template.name,
            )
        return GeneratedCommand(
            command=shlex.join(rendered),
            preamble=preamble,
            template_name=template.name,
        )

    @staticmethod
    def _render_preamble(template: str, *, command: str, template_name: str) -> str:
        replacements = {"{command}": command, "{template}": template_name}
        rendered = template
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)
        if "{" in rendered or "}" in rendered:
            raise ValueError("unsupported placeholder in read-only preamble")
        return rendered
