"""Deterministic, evidence-driven generation of safe repository commands."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cagentrix.director.exploration import (
    ExplorationState,
    ProjectSnapshot,
    RepositoryProbe,
    focus_terms_from_history,
    resolve_locale,
)
from cagentrix.profiles.models import CommandTemplate, NarrativeTemplate, ReadonlyRules

_PHASES = (
    "inventory",
    "metadata",
    "entrypoint",
    "definition",
    "reference",
    "focused_inspection",
    "source_test_pair",
    "config_boundary",
    "cross_check",
)


@dataclass(frozen=True)
class GeneratedCommand:
    """One safe command, its evidence target, and occasional planning text."""

    command: str
    preamble: str | None
    template_name: str
    action_kind: str = "inventory"
    path: str | None = None
    keyword: str | None = None
    signature: str = ""


class ReadonlyCommandGenerator:
    """Plan read-only exploration actions from repository and session evidence."""

    def __init__(self, rules: ReadonlyRules, root: Path) -> None:
        self.rules = rules
        self.root = root.resolve()
        self.snapshot: ProjectSnapshot = RepositoryProbe(rules).scan(self.root)

    def generate(
        self,
        turn: int,
        *,
        history: tuple[Mapping[str, Any], ...] = (),
        state: ExplorationState | None = None,
    ) -> GeneratedCommand:
        """Choose the highest-value unvisited action for the current evidence."""

        exploration = state or ExplorationState()
        resolve_locale(
            exploration,
            history,
            default_locale=self.rules.default_locale,
            supported_locales=self._supported_locales(),
        )
        focus_terms = focus_terms_from_history(history, self.snapshot)
        candidates: list[tuple[int, int, GeneratedCommand]] = []
        for index, template in enumerate(self.rules.templates):
            if not self._requirements_met(template, exploration):
                continue
            generated = self._bind(template, turn, exploration, focus_terms)
            if generated is None:
                continue
            score = self._score(template, turn, exploration, focus_terms)
            if generated.signature in exploration.visited_actions:
                score -= 1_000
            candidates.append((score, -index, generated))
        if not candidates:
            raise ValueError("read-only rules contain no applicable command template")
        fresh_templates = [
            candidate
            for candidate in candidates
            if candidate[2].template_name not in exploration.visited_templates
        ]
        unvisited_actions = [
            candidate
            for candidate in (fresh_templates or candidates)
            if candidate[2].signature not in exploration.visited_actions
        ]
        selected = max(
            unvisited_actions or fresh_templates or candidates,
            key=lambda item: (item[0], item[1]),
        )[2]
        return self._with_preamble(selected, turn, exploration)

    def _bind(
        self,
        template: CommandTemplate,
        turn: int,
        state: ExplorationState,
        focus_terms: tuple[str, ...],
    ) -> GeneratedCommand | None:
        required_placeholders = {
            value
            for stage in template.stages
            for argv in stage.argv
            for value in (
                "glob",
                "keyword",
                "line_range",
                "max_results",
                "path",
                "pattern",
                "root",
                "scope",
            )
            if f"{{{value}}}" in argv
        }
        replacements: dict[str, str] = {}
        rotation = 0 if state.last_kind != template.kind else turn
        if "root" in required_placeholders:
            replacements["root"] = "."
        if "max_results" in required_placeholders:
            replacements["max_results"] = str(self.rules.max_results)
        if "scope" in required_placeholders:
            scopes = self.snapshot.scopes_for(
                template.path_kind,
                observed_paths=tuple(state.observed_paths),
            )
            replacements["scope"] = scopes[rotation % len(scopes)]
        path: str | None = None
        if "path" in required_placeholders:
            paths = self.snapshot.files_for(
                template.path_kind,
                observed_paths=tuple(state.observed_paths),
            )
            if not paths:
                return None
            path = paths[rotation % len(paths)]
            replacements["path"] = path
        scope = replacements.get("scope", path or ".")
        keyword: str | None = None
        if "keyword" in required_placeholders or template.keyword_kind != "none":
            keywords = self.snapshot.keyword_candidates(
                template.keyword_kind,
                focus_terms=focus_terms,
                observed_symbols=tuple(state.observed_symbols),
                scope=None if template.keyword_kind == "process" else scope,
            )
            if not keywords:
                return None
            keyword = keywords[rotation % len(keywords)]
            replacements["keyword"] = keyword
        if "pattern" in required_placeholders:
            legacy_patterns = self.rules.patterns or ((keyword or "project"),)
            replacements["pattern"] = legacy_patterns[turn % len(legacy_patterns)]
        if "glob" in required_placeholders:
            replacements["glob"] = self.snapshot.choose_glob(scope)
        if "line_range" in required_placeholders:
            replacements["line_range"] = self.snapshot.choose_line_range(
                path or scope,
                observed_hits=tuple(state.observed_hits),
                turn=turn,
            )
        rendered_stages: list[str] = []
        for stage in template.stages:
            rendered_argv: list[str] = []
            for part in stage.argv:
                if part.startswith("{") and part.endswith("}"):
                    name = part[1:-1]
                    if name not in replacements:
                        raise ValueError(f"missing command slot {name!r} in {template.name!r}")
                    rendered_argv.append(replacements[name])
                elif "{" in part or "}" in part:
                    raise ValueError(f"unsupported placeholder in rule template {template.name!r}")
                else:
                    rendered_argv.append(part)
            rendered_stages.append(shlex.join(rendered_argv))
        command = " | ".join(rendered_stages)
        bound_path = path or (scope if scope != "." else None)
        return GeneratedCommand(
            command=command,
            preamble=None,
            template_name=template.name,
            action_kind=template.kind,
            path=bound_path,
            keyword=keyword,
            signature=command,
        )

    def _requirements_met(self, template: CommandTemplate, state: ExplorationState) -> bool:
        features = {
            "git": (self.snapshot.root / ".git").exists(),
            "manifest": bool(self.snapshot.manifests),
            "manifests": bool(self.snapshot.manifests),
            "entrypoints": bool(self.snapshot.entrypoints),
            "source": any(fact.role == "source" for fact in self.snapshot.files),
            "source_files": any(fact.role == "source" for fact in self.snapshot.files),
            "test": any(fact.role == "test" for fact in self.snapshot.files),
            "test_files": any(fact.role == "test" for fact in self.snapshot.files),
            "language": bool(self.snapshot.languages),
            "identifiers": bool(self.snapshot.identifiers),
            "symbols": bool(self.snapshot.symbols),
            "observation": bool(state.observed_paths or state.observed_symbols),
            "hits": bool(state.observed_hits),
        }
        return all(features.get(requirement, False) for requirement in template.requires)

    def _score(
        self,
        template: CommandTemplate,
        turn: int,
        state: ExplorationState,
        focus_terms: tuple[str, ...],
    ) -> int:
        preferred = self._preferred_phase(turn, state)
        score = template.weight * 5
        if template.kind == preferred:
            score += 100
        elif template.kind in _nearby_phases(preferred):
            score += 25
        if focus_terms and template.keyword_kind != "none":
            score += 20
        if state.observed_hits and template.kind in {
            "focused_inspection",
            "reference",
            "cross_check",
        }:
            score += 20
        if state.observed_paths and template.kind in {"reference", "source_test_pair"}:
            score += 12
        if len(template.stages) > 1:
            score += self.rules.pipeline_bonus
        if state.last_kind == template.kind:
            score -= 18
        return score - min(turn, 12)

    @staticmethod
    def _preferred_phase(turn: int, state: ExplorationState) -> str:
        if turn == 0:
            return "inventory"
        if state.observed_hits and state.last_kind != "focused_inspection":
            return "focused_inspection"
        if state.last_kind in _PHASES:
            index = _PHASES.index(state.last_kind)
            return _PHASES[min(index + 1, len(_PHASES) - 1)]
        return _PHASES[min(turn, len(_PHASES) - 1)]

    def _with_preamble(
        self,
        generated: GeneratedCommand,
        turn: int,
        state: ExplorationState,
    ) -> GeneratedCommand:
        should_narrate = bool(self.rules.narratives or self.rules.preambles) and (
            turn == 0
            or turn % self.rules.preamble_interval == 0
            or state.last_kind != generated.action_kind
        )
        if not should_narrate:
            return generated
        reason = _reason_for(
            generated.action_kind,
            generated.path,
            generated.keyword,
            locale=state.locale or self.rules.default_locale,
        )
        replacements = {
            "command": generated.command,
            "keyword": generated.keyword or "",
            "path": generated.path or ".",
            "reason": reason,
            "template": generated.template_name,
        }
        narrative = self._narrative_for(generated.action_kind, state, turn)
        template = narrative.text if narrative is not None else None
        if template is None and self.rules.preambles:
            template = self.rules.preambles[
                (turn // max(1, self.rules.preamble_interval)) % len(self.rules.preambles)
            ]
        if template is None:
            return generated
        rendered = template
        for placeholder, value in replacements.items():
            rendered = rendered.replace(f"{{{placeholder}}}", value)
        if "{" in rendered or "}" in rendered:
            raise ValueError("unsupported placeholder in read-only preamble")
        return GeneratedCommand(
            command=generated.command,
            preamble=rendered,
            template_name=generated.template_name,
            action_kind=generated.action_kind,
            path=generated.path,
            keyword=generated.keyword,
            signature=generated.signature,
        )

    def _supported_locales(self) -> tuple[str, ...]:
        locales = {narrative.locale for narrative in self.rules.narratives}
        locales.add(self.rules.default_locale)
        return tuple(sorted(locales))

    def _narrative_for(
        self,
        kind: str,
        state: ExplorationState,
        turn: int,
    ) -> NarrativeTemplate | None:
        locale = state.locale or self.rules.default_locale
        localized = [
            narrative
            for narrative in self.rules.narratives
            if narrative.locale == locale
        ]
        phase_changed = state.last_kind != kind
        trigger_order = [kind]
        if phase_changed:
            trigger_order.append("phase_change")
        if turn % max(1, self.rules.preamble_interval) == 0:
            trigger_order.append("interval")
        trigger_order.append("any")
        for trigger in trigger_order:
            candidates = [item for item in localized if item.trigger == trigger]
            if candidates:
                return candidates[turn % len(candidates)]
        if locale != self.rules.default_locale:
            fallback = [
                narrative
                for narrative in self.rules.narratives
                if narrative.locale == self.rules.default_locale
            ]
            if fallback:
                return fallback[turn % len(fallback)]
        return None


def _nearby_phases(phase: str) -> set[str]:
    try:
        index = _PHASES.index(phase)
    except ValueError:
        return set()
    return {
        value
        for value in _PHASES[max(0, index - 1) : min(len(_PHASES), index + 2)]
        if value != phase
    }


def _reason_for(
    kind: str,
    path: str | None,
    keyword: str | None,
    *,
    locale: str = "en",
) -> str:
    if locale == "zh":
        reasons = {
            "inventory": "先确认项目的文件和目录布局",
            "metadata": f"先核对项目元数据 {path or '文件'}",
            "definition": f"先定位真实符号 {keyword or '目标'} 的定义",
            "reference": f"继续追踪 {keyword or '这个符号'} 的引用",
            "source_test_pair": "把源码和测试目录中的对应线索对齐",
            "focused_inspection": f"查看 {path or '命中文件'} 附近的具体实现",
            "config_boundary": f"核对 {keyword or '配置线索'} 在项目中的边界",
            "cross_check": "在源码、测试和配置之间做一次交叉核对",
            "fallback": "继续收集下一条项目证据",
        }
    else:
        reasons = {
            "inventory": "First I will confirm the project's file and directory layout",
            "metadata": f"First I will inspect the project metadata in {path or 'the manifest'}",
            "definition": f"First I will locate the definition of {keyword or 'the target symbol'}",
            "reference": f"I will follow references to {keyword or 'this symbol'}",
            "source_test_pair": "I will align the source and test evidence",
            "focused_inspection": (
                f"I will inspect the relevant part of {path or 'the matched file'}"
            ),
            "config_boundary": (
                f"I will check where {keyword or 'the configuration signal'} "
                "crosses project boundaries"
            ),
            "cross_check": "I will cross-check the signal across source, tests, and configuration",
            "fallback": "I will collect the next piece of repository evidence",
        }
    return reasons.get(kind, reasons["fallback"])
