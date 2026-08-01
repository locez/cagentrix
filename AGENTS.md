# AGENTS.md

## Project

Cagentrix is a deterministic fake LLM API server for coding agents. It must never
call a real model or an upstream API. LiteLLM is the protocol adapter and custom
provider layer; do not replace it with an unrelated HTTP implementation.

These instructions apply to the entire repository unless a more specific
`AGENTS.md` is added in a subdirectory.

## Runtime and Tooling

- Use Python `3.14` only. The supported range is declared in `pyproject.toml`.
- Use `uv` for environments, dependencies, commands, and lockfile updates.
- Use the `src/` layout and import the package through its installed/project path.
- Use type annotations for public functions, methods, and data structures.
- Keep formatting and lint expectations compatible with Ruff's configuration:
  `uv run ruff check .`.
- Run commands through `uv run`; do not install project dependencies with `pip`.
- Keep `uv.lock` reproducible. Update it only when dependency metadata changes.

## Python Practices

- Prefer small, focused functions and typed dataclasses over unstructured dictionaries
  at internal boundaries. Validate external data when it enters the system.
- Use standard-library parsers and structured APIs for TOML, JSON, and protocol data.
  Do not parse structured data with ad hoc string operations.
- Preserve the existing separation between profiles, director logic, provider
  adapters, and runtime/CLI code. Protocol-specific behavior belongs in profiles or
  adapters, not in a growing `if agent == ...` chain in the director.
- Keep configuration and command-generation policy data-driven in TOML when a change
  is policy rather than algorithm. Document new fields and validate user overrides.
- Keep async code non-blocking. Use `asyncio.sleep` in async paths and do not put
  blocking subprocess, filesystem, or sleep work on an event loop without an explicit
  reason.
- Bound session counts, retained history, request-derived caches, and diagnostic
  output. Do not repeatedly serialize or recursively scan an unbounded conversation
  when a bounded recent view is sufficient.
- Preserve Cagentrix's read-only contract. Generated commands may only come from
  validated profile rule templates and must not introduce writes, arbitrary shell
  text, command substitution, redirection, or execution flags.

## Tests

- Tests must verify intended behavior and public protocol contracts, not implementation
  trivia. Include regression coverage for bug fixes and keep the three protocol paths
  consistent where their behavior is shared.
- Tests must not require real API keys, real model calls, upstream network access, or
  user authentication state.
- **Never write, weaken, skip, delete, or rewrite a test merely to make the current
  implementation pass.** Do not hide a failure with broad mocks, unconditional
  `xfail`, relaxed assertions, or test-only branches. If the intended contract changes,
  change the implementation and documentation first, then update the affected test
  with a clear rationale and add coverage for the new contract.
- For a bug, first reproduce the behavior, then add a focused regression test that
  would fail without the fix. Keep performance and lifecycle tests deterministic.
- Before handing off code, run:

  ```bash
  uv sync
  uv run ruff check .
  uv run pytest
  ```

- For changes to LiteLLM integration or client launch behavior, also run the relevant
  protocol smoke tests and, when the installed clients are available, a local UI or
  HTTP smoke test without an upstream service.

## Repository Safety

- Inspect the current worktree before editing and preserve existing user changes.
- Keep changes scoped to the request. Do not use destructive Git commands such as
  `git reset --hard` or `git checkout --` to discard work.
- Do not commit generated caches, credentials, temporary proxy files, or client state.
- Update README/documentation when behavior, configuration, profiles, or commands
  change.

## Commits

When creating a commit, use Conventional Commits:

```text
<type>(<scope>): <imperative summary>
```

Use a focused type such as `feat`, `fix`, `docs`, `test`, `refactor`, `perf`,
`chore`, `build`, or `ci`. Keep the subject concise and imperative, explain relevant
behavioral trade-offs in the body, and use a `BREAKING CHANGE:` footer when required.
Do not mix unrelated changes in one commit. Do not create a commit unless the user
asks for one.
