"""Contract check against the real livekit-agents event classes (no server, not run in CI).

Drives `LiveKitBridge` with LiveKit's own event objects through a minimal fake session. Run it
after upgrading livekit-agents, for example:

    uv run --with "livekit-agents==1.8.11" python scripts/check_livekit_contract.py
"""

from __future__ import annotations

import asyncio
import typing
from collections.abc import Callable
from typing import Any

from livekit.agents.voice import events as ev

from voice_lab.language import LanguagePolicy
from voice_lab.livekit_bridge import LiveKitBridge


class Session:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Callable[[Any], None]]] = {}

    def on(self, event: str, callback: Callable[[Any], None] | None = None) -> Any:
        assert callback is not None
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def emit(self, event: str, arg: Any) -> None:
        for cb in self.handlers.get(event, []):
            cb(arg)


class Agent:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    async def update_instructions(self, instructions: str) -> None:
        self.instructions.append(instructions)


async def main() -> None:
    for name in ("user_input_transcribed", "user_state_changed", "agent_state_changed"):
        assert name in typing.get_args(ev.EventTypes), name
    session, agent = Session(), Agent()
    bridge = LiveKitBridge(LanguagePolicy(), agent, lambda lang: f"Reply in {lang}.")
    bridge.attach(session)
    t = 1000.0
    # Turn 1 has two final STT segments (one user turn); turn 2 has one.
    for segments in (
        ["Kunt u mij helpen?", "Ik zoek mijn bestelling."],
        ["Ik wil graag weten waar mijn pakket is."],
    ):
        session.emit(
            "user_state_changed",
            ev.UserStateChangedEvent(old_state="listening", new_state="speaking", created_at=t - 2),
        )
        for i, text in enumerate(segments):
            session.emit(
                "user_input_transcribed",
                ev.UserInputTranscribedEvent(transcript=text, is_final=True, created_at=t - 1 + i),
            )
        session.emit(
            "user_state_changed",
            ev.UserStateChangedEvent(old_state="speaking", new_state="listening", created_at=t),
        )
        for old, new, dt in [
            ("listening", "thinking", 0.45),
            ("thinking", "speaking", 1.2),
            ("speaking", "listening", 4.0),
        ]:
            session.emit(
                "agent_state_changed",
                ev.AgentStateChangedEvent(old_state=old, new_state=new, created_at=t + dt),
            )
        t += 10
    await bridge.aclose()
    assert [d.reply_language for d in bridge.decisions] == ["en", "nl"], bridge.decisions
    assert agent.instructions == ["Reply in nl."], agent.instructions
    assert [x.segments for x in bridge.turns] == [2, 1]
    assert [x.first_audio_ms for x in bridge.turns] == [1200.0, 1200.0]
    print("ok: event names and fields match; one decision per turn; 2 turns timed")


if __name__ == "__main__":
    asyncio.run(main())
