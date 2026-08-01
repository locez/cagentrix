"""Bounded per-session state for deterministic multi-turn behavior."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from cagentrix.director.exploration import ExplorationState


@dataclass
class SessionState:
    turns: int = 0
    events: int = 0
    pending_call_id: str | None = None
    pending_tool_name: str | None = None
    pending_arguments: dict[str, Any] = field(default_factory=dict)
    pending_preamble: str | None = None
    exploration: ExplorationState = field(default_factory=ExplorationState)


class SessionStore:
    """An LRU store with bounded session count and bounded counters."""

    def __init__(self, *, max_sessions: int = 128, max_events: int = 32) -> None:
        if max_sessions < 1 or max_events < 1:
            raise ValueError("session limits must be positive")
        self.max_sessions = max_sessions
        self.max_events = max_events
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()

    def state_for(self, session_id: str) -> SessionState:
        """Return a session and refresh its LRU position."""

        state = self._sessions.pop(session_id, None)
        if state is None:
            state = SessionState()
        self._sessions[session_id] = state
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        return state

    def next_turn(self, session_id: str) -> int:
        """Return the current turn index and advance bounded state."""

        state = self.state_for(session_id)
        current = state.turns
        state.turns = (state.turns + 1) % self.max_events
        state.events = min(state.events + 1, self.max_events)
        return current

    def record_call(
        self,
        session_id: str,
        *,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        preamble: str | None = None,
    ) -> None:
        """Remember one pending call while keeping counters bounded."""

        state = self.state_for(session_id)
        state.pending_call_id = call_id
        state.pending_tool_name = tool_name
        state.pending_arguments = dict(arguments)
        state.pending_preamble = preamble
        state.turns = (state.turns + 1) % self.max_events
        state.events = min(state.events + 1, self.max_events)

    def clear_pending(self, session_id: str) -> None:
        """Mark the current call complete after a client tool result arrives."""

        state = self.state_for(session_id)
        state.pending_call_id = None
        state.pending_tool_name = None
        state.pending_arguments = {}
        state.pending_preamble = None

    @property
    def size(self) -> int:
        return len(self._sessions)
