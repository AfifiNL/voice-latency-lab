"""Optional bridge into a LiveKit Agents session (written against livekit-agents 1.8.x).

It imports nothing from LiveKit: it relies only on `session.on(event_name, callback)`, the
event attributes listed below, and `agent.update_instructions(text)`, all read from the
livekit-agents 1.8.11 source. It was tested with a fake session and with LiveKit's own event
classes, never against a running LiveKit server.

  user_input_transcribed   .transcript, .is_final, .created_at
  user_state_changed       .old_state, .new_state, .created_at
  agent_state_changed      .old_state, .new_state, .created_at

The STT emits a final transcript per *segment*, and a user turn can contain several. The
bridge therefore collects the final segments of a user turn and gives the language policy one
decision per turn, when the turn is committed: by `commit_turn()` (call it from your agent's
`on_user_turn_completed` hook so a switch applies to this turn's reply), or at the latest when
the agent starts thinking. Final segments may arrive before or after the end of speech; times
are computed once the turn ends, relative to the last end of user speech (negative = before).

When the policy switches language, the bridge calls `agent.update_instructions` with the text
`instructions_for(language)` returns; a failure is logged and kept in `errors`, never lost.
Changing STT/TTS voices per language is provider-specific and left to `on_switch`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from voice_lab.language import Decision, Language, LanguagePolicy

log = logging.getLogger(__name__)


class _Session(Protocol):
    def on(self, event: Any, callback: Callable[..., Any] | None = None) -> Callable[..., Any]: ...


class _Agent(Protocol):
    def update_instructions(self, instructions: str) -> Coroutine[Any, Any, None]: ...


@dataclass
class _OpenTurn:
    segments: list[str] = field(default_factory=list)
    speech_end_at: float | None = None
    last_final_at: float | None = None
    thinking_at: float | None = None
    first_audio_at: float | None = None
    decision: Decision | None = None


@dataclass(frozen=True)
class LiveTurn:
    """Milliseconds after the end of user speech; None when unknown or it did not happen."""

    language: str
    transcript: str
    segments: int
    final_transcript_ms: float | None
    agent_thinking_ms: float | None
    first_audio_ms: float | None
    turn_end_ms: float | None


class LiveKitBridge:
    def __init__(
        self,
        policy: LanguagePolicy,
        agent: _Agent,
        instructions_for: Callable[[Language], str],
        *,
        on_switch: Callable[[Decision], None] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.policy = policy
        self.agent = agent
        self.instructions_for = instructions_for
        self.on_switch = on_switch
        self.log_path = log_path
        self.turns: list[LiveTurn] = []
        self.decisions: list[Decision] = []
        self.errors: list[str] = []
        self._turn: _OpenTurn | None = None
        self._tasks: set[asyncio.Future[None]] = set()

    def attach(self, session: _Session) -> None:
        session.on("user_input_transcribed", self._on_transcribed)
        session.on("user_state_changed", self._on_user_state)
        session.on("agent_state_changed", self._on_agent_state)

    # -- turn bookkeeping ------------------------------------------------------------------

    def _open(self) -> _OpenTurn:
        if self._turn is None:
            self._turn = _OpenTurn()
        return self._turn

    def _on_user_state(self, ev: Any) -> None:
        if ev.new_state == "speaking":
            turn = self._turn
            if turn is not None and turn.decision is not None:
                self._finish(None)  # the previous turn was answered (or interrupted)
            self._open()
        elif ev.old_state == "speaking":
            self._open().speech_end_at = float(ev.created_at)

    def _on_transcribed(self, ev: Any) -> None:
        if not ev.is_final:
            return
        text = str(ev.transcript).strip()
        if not text:
            return
        turn = self._open()
        turn.segments.append(text)
        turn.last_final_at = float(ev.created_at)

    def commit_turn(self) -> Decision | None:
        """Decide the reply language for the current user turn (once). Returns the decision,
        or None when there is no turn or no final transcript yet."""
        turn = self._turn
        if turn is None or turn.decision is not None or not turn.segments:
            return None if turn is None else turn.decision
        decision = self.policy.observe(" ".join(turn.segments))
        turn.decision = decision
        self.decisions.append(decision)
        if decision.switched:
            future = asyncio.ensure_future(
                self.agent.update_instructions(self.instructions_for(decision.reply_language))
            )
            self._tasks.add(future)
            future.add_done_callback(self._task_done)
            if self.on_switch is not None:
                self.on_switch(decision)
        return decision

    def _task_done(self, future: asyncio.Future[None]) -> None:
        self._tasks.discard(future)
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            message = f"update_instructions failed: {type(exc).__name__}"
            self.errors.append(message)
            log.warning(message)

    def _on_agent_state(self, ev: Any) -> None:
        turn = self._turn
        if turn is None:
            return
        at = float(ev.created_at)
        if ev.new_state == "thinking":
            self.commit_turn()
            if turn.thinking_at is None:
                turn.thinking_at = at
        elif ev.new_state == "speaking" and turn.first_audio_at is None:
            turn.first_audio_at = at
        elif ev.old_state == "speaking" and ev.new_state in {"listening", "idle"}:
            self._finish(at)

    def _finish(self, end_at: float | None) -> None:
        turn, self._turn = self._turn, None
        if turn is None:
            return
        if turn.decision is None:
            self._turn = turn  # not committed yet: commit now so no turn goes undecided
            self.commit_turn()
            self._turn = None
        base = turn.speech_end_at

        def rel(t: float | None) -> float | None:
            return None if t is None or base is None else round((t - base) * 1000, 1)

        record = LiveTurn(
            language=turn.decision.reply_language if turn.decision else self.policy.current,
            transcript=" ".join(turn.segments),
            segments=len(turn.segments),
            final_transcript_ms=rel(turn.last_final_at),
            agent_thinking_ms=rel(turn.thinking_at),
            first_audio_ms=rel(turn.first_audio_at),
            turn_end_ms=rel(end_at),
        )
        self.turns.append(record)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")

    async def aclose(self) -> None:
        """Wait for pending instruction updates (their errors land in `errors`)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
