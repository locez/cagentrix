"""Bounded repository evidence and observation helpers for read-only exploration."""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from cagentrix.profiles.models import LanguageDescriptor, ReadonlyRules

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,63}")
_DEFINITION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,63}")
_PATH_LINE = re.compile(r"^(?P<path>[^:\n]+):(?P<line>[0-9]+):(?P<text>.*)$")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_STOP_WORDS = frozenset(
    {
        "and",
        "any",
        "are",
        "async",
        "await",
        "class",
        "check",
        "code",
        "command",
        "const",
        "def",
        "else",
        "false",
        "fn",
        "for",
        "from",
        "function",
        "if",
        "impl",
        "import",
        "in",
        "inspect",
        "its",
        "let",
        "match",
        "module",
        "new",
        "none",
        "pub",
        "return",
        "review",
        "repository",
        "repo",
        "read",
        "self",
        "show",
        "source",
        "struct",
        "test",
        "tests",
        "the",
        "this",
        "true",
        "type",
        "use",
        "var",
        "where",
        "while",
        "with",
    }
)
_DOC_SUFFIXES = {".md", ".mdx", ".rst", ".txt"}
_CONFIG_SUFFIXES = {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
_LOCK_NAMES = {
    "Cargo.lock",
    "Gemfile.lock",
    "go.sum",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}


@dataclass(frozen=True)
class FileFact:
    """Small bounded facts about one repository file."""

    path: str
    size: int
    line_count: int
    language: str | None
    role: str


@dataclass(frozen=True)
class SymbolEvidence:
    """A symbol-like identifier and the file evidence that produced it."""

    name: str
    path: str
    line: int
    role: str
    language: str | None = None


@dataclass(frozen=True)
class ObservedHit:
    """A bounded path/line match extracted from a client tool result."""

    path: str
    line: int
    text: str


@dataclass(frozen=True)
class ToolObservation:
    """Safe structured evidence extracted from one tool result."""

    paths: tuple[str, ...] = ()
    hits: tuple[ObservedHit, ...] = ()
    symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectSnapshot:
    """A deterministic, bounded view of the current repository."""

    root: Path
    files: tuple[FileFact, ...]
    directories: tuple[str, ...]
    languages: tuple[str, ...]
    manifests: tuple[str, ...]
    entrypoints: tuple[str, ...]
    symbols: tuple[SymbolEvidence, ...]
    identifiers: tuple[str, ...]
    identifier_paths: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def files_for(self, kind: str, *, observed_paths: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Return existing paths suitable for a typed command slot."""

        facts = {fact.path: fact for fact in self.files}
        observed = tuple(path for path in observed_paths if path in facts)
        if kind == "hit_file":
            return observed or tuple(symbol.path for symbol in self.symbols)
        if kind == "manifest":
            return self.manifests
        if kind == "entrypoint":
            return self.entrypoints
        if kind == "source_file":
            return tuple(fact.path for fact in self.files if fact.role == "source")
        if kind == "test_file":
            return tuple(fact.path for fact in self.files if fact.role == "test")
        if kind == "doc_file":
            return tuple(fact.path for fact in self.files if fact.role == "docs")
        return tuple(fact.path for fact in self.files)

    def scopes_for(
        self,
        kind: str,
        *,
        observed_paths: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return stable directory scopes derived from actual files."""

        source_files = [fact.path for fact in self.files if fact.role == "source"]
        test_files = [fact.path for fact in self.files if fact.role == "test"]
        if kind == "source_scope":
            paths = source_files
        elif kind == "test_scope":
            paths = test_files
        elif kind == "manifest_scope":
            paths = list(self.manifests)
        else:
            paths = [fact.path for fact in self.files]
        observed_facts = {fact.path: fact for fact in self.files}
        observed_scopes = {
            _parent_scope(path)
            for path in observed_paths
            if path in observed_facts
            and (
                kind not in {"source_scope", "test_scope"}
                or observed_facts[path].role == kind.removesuffix("_scope")
            )
        }
        scopes = {_top_level_scope(path) for path in paths}
        scopes.update(observed_scopes)
        if kind in {"scope", "all_scope"}:
            scopes.add(".")
        if not scopes:
            return (".",)
        return tuple(sorted(scopes, key=lambda value: (value != ".", value.count("/"), value)))

    def keyword_candidates(
        self,
        kind: str,
        *,
        focus_terms: tuple[str, ...] = (),
        observed_symbols: tuple[str, ...] = (),
        scope: str | None = None,
    ) -> tuple[str, ...]:
        """Rank evidence-backed identifiers for a rule's keyword slot."""

        ordered: list[str] = []
        related_symbols = [
            symbol.name
            for symbol in self.symbols
            if any(term.casefold() in symbol.name.casefold() for term in focus_terms)
        ]
        if kind == "process":
            root_name = self.root.name
            if any(root_name.casefold() == value.casefold() for value in self.identifiers):
                ordered.append(root_name)
        if kind in {"reference", "definition", "focused"}:
            ordered.extend(observed_symbols)
            ordered.extend(related_symbols)
        if kind in {"definition", "focused"}:
            ordered.extend(symbol.name for symbol in self.symbols)
        else:
            ordered.extend(related_symbols)
            ordered.extend(focus_terms)
        ordered.extend(focus_terms)
        if kind not in {"definition", "focused"}:
            ordered.extend(symbol.name for symbol in self.symbols)
        ordered.extend(self.identifiers)
        result: list[str] = []
        seen: set[str] = set()
        known = {value.casefold() for value in self.identifiers}
        known.update(symbol.name.casefold() for symbol in self.symbols)
        identifier_path_index = {
            name.casefold(): paths for name, paths in self.identifier_paths
        }
        symbol_path_lists: dict[str, list[str]] = {}
        for symbol in self.symbols:
            symbol_path_lists.setdefault(symbol.name.casefold(), []).append(symbol.path)
        symbol_path_index = {
            name: tuple(paths) for name, paths in symbol_path_lists.items()
        }
        for value in ordered:
            if not _DEFINITION_NAME.fullmatch(value) or value.casefold() not in known:
                continue
            if scope is not None and not self._keyword_in_scope(
                value,
                scope,
                identifier_path_index,
                symbol_path_index,
            ):
                continue
            lowered = value.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            result.append(value)
        return tuple(result)

    def _keyword_in_scope(
        self,
        value: str,
        scope: str,
        identifier_path_index: Mapping[str, tuple[str, ...]],
        symbol_path_index: Mapping[str, tuple[str, ...]],
    ) -> bool:
        lowered = value.casefold()
        paths = (
            *identifier_path_index.get(lowered, ()),
            *symbol_path_index.get(lowered, ()),
        )
        return any(_path_in_scope(path, scope) for path in paths)

    def choose_line_range(
        self,
        path: str,
        *,
        observed_hits: tuple[ObservedHit, ...] = (),
        turn: int = 0,
    ) -> str:
        """Choose a bounded sed print range around known evidence."""

        line = next((hit.line for hit in observed_hits if hit.path == path), None)
        if line is None:
            line = next((symbol.line for symbol in self.symbols if symbol.path == path), None)
        fact = next((item for item in self.files if item.path == path), None)
        if line is None:
            start = 1 + (turn % 3) * 20
        else:
            start = max(1, line - 30)
        end = start + 100
        if fact is not None and fact.line_count > 0:
            end = min(end, fact.line_count)
        return f"{start},{max(start, end)}p"

    def choose_glob(self, scope: str) -> str:
        """Return an extension glob supported by files in the selected scope."""

        scoped = [fact for fact in self.files if _path_in_scope(fact.path, scope)]
        counts = Counter(fact.path.rsplit(".", 1)[-1] for fact in scoped if "." in fact.path)
        if not counts:
            return "*"
        suffix, _ = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        return f"*.{suffix}"

    def matching_terms(self, text: str) -> tuple[str, ...]:
        """Keep only task words that are actual snapshot identifiers."""

        known = {value.casefold(): value for value in self.identifiers}
        for symbol in self.symbols:
            known.setdefault(symbol.name.casefold(), symbol.name)
        candidates = tuple(
            dict.fromkeys((*self.identifiers, *(symbol.name for symbol in self.symbols)))
        )
        result: list[str] = []
        seen: set[str] = set()
        for token in _IDENTIFIER.findall(text):
            if token.casefold() in _STOP_WORDS:
                continue
            value = known.get(token.casefold())
            if value is None and len(token) >= 4:
                matches = sorted(
                    (
                        candidate
                        for candidate in candidates
                        if token.casefold() in candidate.casefold()
                    ),
                    key=lambda candidate: (len(candidate), candidate.casefold()),
                )
                value = matches[0] if matches else None
            if value is not None and value.casefold() not in seen:
                seen.add(value.casefold())
                result.append(value)
        return tuple(result[:12])


class RepositoryProbe:
    """Scan only bounded metadata and text facts from a project root."""

    def __init__(self, rules: ReadonlyRules) -> None:
        self.rules = rules
        self._descriptors = tuple(rules.languages)

    def scan(self, root: Path) -> ProjectSnapshot:
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"repository root is not a directory: {root}")

        files: list[FileFact] = []
        directories: set[str] = {"."}
        languages: set[str] = set()
        manifests: list[str] = []
        entrypoints: list[str] = []
        symbols: list[SymbolEvidence] = []
        identifier_counts: Counter[str] = Counter()
        identifier_case: dict[str, str] = {}
        identifier_paths: dict[str, set[str]] = {}
        read_budget = self.rules.max_scan_bytes

        def visit(directory: Path, depth: int) -> None:
            nonlocal read_budget
            if len(files) >= self.rules.max_scan_files or depth > 8:
                return
            try:
                entries = sorted(os.scandir(directory), key=lambda entry: entry.name.casefold())
            except OSError:
                return
            for entry in entries:
                if len(files) >= self.rules.max_scan_files:
                    return
                try:
                    if entry.is_symlink():
                        continue
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name in self.rules.ignored_dirs:
                            continue
                        relative_dir = _relative_path(path, root)
                        if relative_dir != ".":
                            directories.add(relative_dir)
                        visit(path, depth + 1)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    relative = _relative_path(path, root)
                    descriptor = self._descriptor_for(path)
                    text = ""
                    if (
                        read_budget > 0
                        and path.name not in {".env", ".env.local"}
                        and path.name not in _LOCK_NAMES
                    ):
                        text, consumed = _read_text(
                            path,
                            min(self.rules.max_file_bytes, read_budget),
                        )
                        read_budget -= consumed
                    role = self._role_for(path, descriptor)
                    size = entry.stat(follow_symlinks=False).st_size
                    files.append(
                        FileFact(
                            path=relative,
                            size=size,
                            line_count=text.count("\n") + 1 if text else 0,
                            language=descriptor.name if descriptor else None,
                            role=role,
                        )
                    )
                    if descriptor is not None:
                        languages.add(descriptor.name)
                    if role == "manifest":
                        manifests.append(relative)
                    if descriptor is not None and _is_entrypoint(relative, descriptor):
                        entrypoints.append(relative)
                    _add_parent_directories(relative, directories)
                    for value in _IDENTIFIER.findall(relative.replace("/", " ")):
                        _record_identifier(
                            value,
                            identifier_counts,
                            identifier_case,
                            identifier_paths,
                            relative,
                        )
                    if text:
                        for value in _IDENTIFIER.findall(text):
                            _record_identifier(
                                value,
                                identifier_counts,
                                identifier_case,
                                identifier_paths,
                                relative,
                            )
                        if descriptor is not None:
                            symbols.extend(_extract_symbols(text, relative, descriptor))
                except OSError:
                    continue

        visit(root, 0)
        ranked_identifiers = tuple(
            identifier_case[key]
            for key, _ in sorted(
                identifier_counts.items(), key=lambda item: (-item[1], item[0])
            )
            if key not in _STOP_WORDS
        )
        symbols.sort(key=lambda item: (item.path, item.line, item.name.casefold()))
        return ProjectSnapshot(
            root=root,
            files=tuple(sorted(files, key=lambda item: item.path)),
            directories=tuple(sorted(directories)),
            languages=tuple(sorted(languages)),
            manifests=tuple(sorted(manifests)),
            entrypoints=tuple(sorted(set(entrypoints))),
            symbols=tuple(symbols[: self.rules.max_scan_files]),
            identifiers=ranked_identifiers[: self.rules.max_scan_files],
            identifier_paths=tuple(
                (
                    identifier_case[key],
                    tuple(sorted(identifier_paths.get(key, set()))),
                )
                for key in sorted(identifier_paths)
                if key in identifier_case
            ),
        )

    def _descriptor_for(self, path: Path) -> LanguageDescriptor | None:
        for descriptor in self._descriptors:
            if path.name in descriptor.manifests:
                return descriptor
        suffix = path.suffix.lower().lstrip(".")
        for descriptor in self._descriptors:
            if suffix in {item.lower().lstrip(".") for item in descriptor.extensions}:
                return descriptor
        return None

    def _role_for(self, path: Path, descriptor: LanguageDescriptor | None) -> str:
        relative_parts = path.parts
        if descriptor is not None and path.name in descriptor.manifests:
            return "manifest"
        if descriptor is not None:
            if any(part in descriptor.test_dirs for part in relative_parts[:-1]):
                return "test"
            if any(fnmatch(path.name, pattern) for pattern in descriptor.test_patterns):
                return "test"
            if path.suffix.lower().lstrip(".") in {
                item.lower().lstrip(".") for item in descriptor.extensions
            }:
                return "source"
        if path.suffix.lower() in _DOC_SUFFIXES:
            return "docs"
        if path.suffix.lower() in _CONFIG_SUFFIXES:
            return "config"
        return "other"


@dataclass
class ExplorationState:
    """Bounded per-session evidence used by the heuristic planner."""

    visited_actions: list[str] = field(default_factory=list)
    visited_templates: list[str] = field(default_factory=list)
    observed_paths: list[str] = field(default_factory=list)
    observed_symbols: list[str] = field(default_factory=list)
    observed_hits: list[ObservedHit] = field(default_factory=list)
    phase: str = "inventory"
    last_kind: str | None = None
    locale: str | None = None

    def remember_action(self, signature: str, *, template_name: str | None = None) -> None:
        if signature in self.visited_actions:
            if template_name is None or template_name in self.visited_templates:
                return
        else:
            self.visited_actions.append(signature)
        del self.visited_actions[:-32]
        if template_name is not None and template_name not in self.visited_templates:
            self.visited_templates.append(template_name)
        del self.visited_templates[:-32]

    def absorb(self, observation: ToolObservation) -> None:
        _append_unique(self.observed_paths, observation.paths, limit=24)
        _append_unique(self.observed_symbols, observation.symbols, limit=24)
        for hit in observation.hits:
            if hit not in self.observed_hits:
                self.observed_hits.append(hit)
        del self.observed_hits[:-32]


def focus_terms_from_history(
    history: tuple[Mapping[str, Any], ...], snapshot: ProjectSnapshot
) -> tuple[str, ...]:
    """Extract only repository-backed terms from the latest user/developer text."""

    for item in reversed(history[-16:]):
        if item.get("role") not in {"user", "developer"}:
            continue
        text = _content_text(item.get("content", item.get("input", "")))
        terms = snapshot.matching_terms(text[-4_000:])
        if terms:
            return terms
    return ()


def resolve_locale(
    state: ExplorationState,
    history: tuple[Mapping[str, Any], ...],
    *,
    default_locale: str,
    supported_locales: tuple[str, ...],
) -> str:
    """Keep English by default, but switch permanently when Chinese appears."""

    detected = first_message_locale(history)
    if detected == "zh" and "zh" in supported_locales:
        state.locale = "zh"
    elif state.locale is None:
        selected = detected or default_locale
        state.locale = selected if selected in supported_locales else default_locale
    return state.locale or default_locale


def first_message_locale(history: tuple[Mapping[str, Any], ...]) -> str | None:
    """Return Chinese when any human message contains Han text, otherwise English."""

    saw_message = False
    for item in history:
        if item.get("role") != "user" or _is_tool_result_item(item):
            continue
        text = _content_text(item.get("content", item.get("input", "")))
        if not text.strip():
            continue
        saw_message = True
        if detect_text_locale(text) == "zh":
            return "zh"
    return "en" if saw_message else None


def detect_text_locale(text: str) -> str:
    """Return ``zh`` when the supplied text contains any Han text."""

    if _CJK.search(text):
        return "zh"
    return "en"


def _is_tool_result_item(item: Mapping[str, Any]) -> bool:
    content = item.get("content")
    if not isinstance(content, (list, tuple)):
        return False
    blocks = [block for block in content if isinstance(block, Mapping)]
    return bool(blocks) and all(block.get("type") == "tool_result" for block in blocks)


def observation_for_call(
    history: tuple[Mapping[str, Any], ...],
    call_id: str,
    root: Path,
    *,
    max_chars: int,
    max_lines: int,
) -> ToolObservation | None:
    """Find and parse one bounded tool result across supported protocol shapes."""

    text = _result_text(history, call_id, max_chars=max_chars)
    if text is None:
        return None
    paths: list[str] = []
    hits: list[ObservedHit] = []
    symbols: list[str] = []
    for raw_line in text.splitlines()[:max_lines]:
        line = raw_line.strip()
        match = _PATH_LINE.match(line)
        if match is not None and match.group("line").isdigit():
            path = _safe_observed_path(match.group("path"), root)
            if path is not None:
                hit = ObservedHit(
                    path=path,
                    line=int(match.group("line")),
                    text=match.group("text")[:240],
                )
                if hit not in hits:
                    hits.append(hit)
                if path not in paths:
                    paths.append(path)
                _append_unique(symbols, _identifier_values(hit.text), limit=24)
            continue
        path = _safe_observed_path(line, root)
        if path is not None and path not in paths:
            paths.append(path)
    _append_unique(symbols, _identifier_values(text), limit=24)
    return ToolObservation(tuple(paths[:24]), tuple(hits[:32]), tuple(symbols[:24]))


def _extract_symbols(
    text: str, path: str, descriptor: LanguageDescriptor
) -> list[SymbolEvidence]:
    result: list[SymbolEvidence] = []
    for pattern in descriptor.definition_patterns:
        try:
            matcher = re.compile(pattern)
        except re.error:
            continue
        for match in matcher.finditer(text):
            name = match.groupdict().get("name") if match.groupdict() else None
            if not name and match.lastindex:
                name = match.group(1)
            if not isinstance(name, str) or not _DEFINITION_NAME.fullmatch(name):
                continue
            result.append(
                SymbolEvidence(
                    name=name,
                    path=path,
                    line=text.count("\n", 0, match.start()) + 1,
                    role="definition",
                    language=descriptor.name,
                )
            )
    return result


def _read_text(path: Path, limit: int) -> tuple[str, int]:
    try:
        data = path.read_bytes()[:limit]
    except OSError:
        return "", 0
    if b"\x00" in data:
        return "", len(data)
    return data.decode("utf-8", errors="ignore"), len(data)


def _record_identifier(
    value: str,
    counts: Counter[str],
    canonical: dict[str, str],
    locations: dict[str, set[str]],
    path: str,
) -> None:
    lowered = value.casefold()
    if lowered in _STOP_WORDS or len(value) < 3:
        return
    counts[lowered] += 1
    canonical.setdefault(lowered, value)
    locations.setdefault(lowered, set()).add(path)


def _identifier_values(text: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in _IDENTIFIER.findall(text):
        if value.casefold() in _STOP_WORDS or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        result.append(value)
    return tuple(result)


def _result_text(
    history: tuple[Mapping[str, Any], ...], call_id: str, *, max_chars: int
) -> str | None:
    for item in reversed(history[-16:]):
        result = _find_result(item, call_id)
        if result is not None:
            return result[:max_chars]
    return None


def _find_result(value: Any, call_id: str) -> str | None:
    if isinstance(value, Mapping):
        item_type = value.get("type")
        is_result = value.get("role") == "tool" or item_type in {
            "function_call_output",
            "tool_result",
        }
        if is_result and any(
            value.get(key) == call_id for key in ("tool_call_id", "tool_use_id", "call_id")
        ):
            return _content_text(value.get("output", value.get("content", "")))
        for nested in reversed(tuple(value.values())):
            if isinstance(nested, (Mapping, list, tuple)):
                result = _find_result(nested, call_id)
                if result is not None:
                    return result
    elif isinstance(value, (list, tuple)):
        for nested in reversed(value[-16:]):
            result = _find_result(nested, call_id)
            if result is not None:
                return result
    return None


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in (
            "text",
            "content",
            "output",
            "value",
            "input_text",
            "output_text",
        ):
            if key in value:
                return _content_text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_content_text(item) for item in value)
    return ""


def _safe_observed_path(value: str, root: Path) -> str | None:
    candidate = value.strip().strip("`'\"")
    if candidate.startswith("./"):
        candidate = candidate[2:]
    if not candidate or Path(candidate).is_absolute() or ".." in Path(candidate).parts:
        return None
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.exists():
        return None
    return _relative_path(path, root)


def _relative_path(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    return relative.as_posix() or "."


def _parent_scope(path: str) -> str:
    parent = Path(path).parent.as_posix()
    return "." if parent in {"", "."} else parent


def _top_level_scope(path: str) -> str:
    parts = Path(path).parts
    return parts[0] if len(parts) > 1 else "."


def _path_in_scope(path: str, scope: str) -> bool:
    if scope == ".":
        return True
    return path == scope or path.startswith(f"{scope.rstrip('/')}/")


def _is_entrypoint(path: str, descriptor: LanguageDescriptor) -> bool:
    return any(path == pattern or Path(path).name == pattern or fnmatch(path, pattern)
               for pattern in descriptor.entrypoint_paths)


def _add_parent_directories(path: str, directories: set[str]) -> None:
    parent = Path(path).parent
    while str(parent) not in {"", "."}:
        directories.add(parent.as_posix())
        parent = parent.parent


def _append_unique(target: list[str], values: tuple[str, ...], *, limit: int) -> None:
    for value in values:
        if value not in target:
            target.append(value)
        if len(target) >= limit:
            break
