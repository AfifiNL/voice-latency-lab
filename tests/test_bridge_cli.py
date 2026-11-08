from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_lab.cli import main
from voice_lab.language import LanguagePolicy
from voice_lab.livekit_bridge import LiveKitBridge

ROOT = Path(__file__).resolve().parents[1]


class Session:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}

    def on(self, event: str, callback: Any = None) -> Any:
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def emit(self, event: str, **fields: Any) -> None:
        for cb in self.handlers.get(event, []):
            cb(SimpleNamespace(**fields))


class Agent:
    def __init__(self, fail: bool = False) -> None:
        self.instructions: list[str] = []
        self.fail = fail

    async def update_instructions(self, instructions: str) -> None:
        if self.fail:
            raise RuntimeError("not running")
        self.instructions.append(instructions)


def user_turn(s: Session, segments: list[str], t: float, *, final_after_end: bool = False) -> None:
    s.emit("user_state_changed", old_state="listening", new_state="speaking", created_at=t - 2)
    if not final_after_end:
        for i, text in enumerate(segments):
            s.emit("user_input_transcribed", transcript=text[:4], is_final=False, created_at=t - 1)
            s.emit(
                "user_input_transcribed", transcript=text, is_final=True, created_at=t - 1 + i * 0.5
            )
    s.emit("user_state_changed", old_state="speaking", new_state="listening", created_at=t)
    if final_after_end:
        for i, text in enumerate(segments):
            s.emit(
                "user_input_transcribed",
                transcript=text,
                is_final=True,
                created_at=t + 0.2 + i * 0.1,
            )


def agent_reply(s: Session, t: float) -> None:
    s.emit("agent_state_changed", old_state="listening", new_state="thinking", created_at=t + 0.5)
    s.emit("agent_state_changed", old_state="thinking", new_state="speaking", created_at=t + 0.9)
    s.emit("agent_state_changed", old_state="speaking", new_state="listening", created_at=t + 3.0)


def bridge(agent: Agent, **kw: Any) -> tuple[Session, LiveKitBridge]:
    s = Session()
    b = LiveKitBridge(LanguagePolicy(), agent, lambda lang: f"Reply in {lang}.", **kw)
    b.attach(s)
    return s, b


def test_two_final_segments_in_one_turn_are_one_decision(tmp_path: Path) -> None:
    async def go() -> tuple[Agent, LiveKitBridge]:
        a = Agent()
        s, b = bridge(a, log_path=tmp_path / "turns.jsonl")
        user_turn(s, ["Kunt u mij helpen met mijn bestelling?", "Ik wil weten waar het is."], 100)
        agent_reply(s, 100)
        await b.aclose()
        return a, b

    a, b = asyncio.run(go())
    assert len(b.decisions) == 1  # not two Dutch "replies" from one turn
    assert a.instructions == []  # one Dutch turn is not enough to switch
    assert b.turns[0].segments == 2
    assert b.turns[0].final_transcript_ms == pytest.approx(-500.0)  # final before end of speech
    assert b.turns[0].first_audio_ms == pytest.approx(900.0)
    assert len((tmp_path / "turns.jsonl").read_text().splitlines()) == 1


def test_switch_after_two_turns_and_finals_after_end_of_speech() -> None:
    async def go() -> tuple[Agent, LiveKitBridge]:
        a = Agent()
        s, b = bridge(a)
        for t in (100.0, 110.0):
            user_turn(s, ["Ik wil graag weten waar mijn pakket is."], t, final_after_end=True)
            agent_reply(s, t)
        await b.aclose()
        return a, b

    a, b = asyncio.run(go())
    assert a.instructions == ["Reply in nl."]
    assert [t.language for t in b.turns] == ["en", "nl"]
    assert b.turns[1].final_transcript_ms == pytest.approx(200.0)


def test_commit_turn_from_the_agent_hook_decides_before_thinking() -> None:
    async def go() -> tuple[Agent, LiveKitBridge]:
        a = Agent()
        s, b = bridge(a)
        user_turn(s, ["Ik wil graag weten waar mijn pakket is."], 100)
        agent_reply(s, 100)
        user_turn(s, ["Kunt u mij helpen met mijn bestelling?"], 110)
        decision = b.commit_turn()  # e.g. from Agent.on_user_turn_completed
        assert decision is not None
        assert decision.switched
        assert b.commit_turn() is decision  # idempotent
        agent_reply(s, 110)
        await b.aclose()
        return a, b

    a, b = asyncio.run(go())
    assert a.instructions == ["Reply in nl."]
    assert len(b.decisions) == 2


def test_failed_instruction_update_is_recorded() -> None:
    async def go() -> LiveKitBridge:
        s, b = bridge(Agent(fail=True))
        for t in (100.0, 110.0):
            user_turn(s, ["Ik wil graag weten waar mijn pakket is."], t)
            agent_reply(s, t)
        await b.aclose()
        return b

    b = asyncio.run(go())
    assert b.errors == ["update_instructions failed: RuntimeError"]


def test_barge_in_closes_the_previous_turn_and_interims_are_ignored() -> None:
    s, b = bridge(Agent())
    s.emit("agent_state_changed", old_state="listening", new_state="speaking", created_at=1.0)
    s.emit("user_input_transcribed", transcript="Kunt u", is_final=False, created_at=1.0)
    assert b.decisions == []
    user_turn(s, ["Can you help me with my order please?"], 10)
    s.emit("agent_state_changed", old_state="listening", new_state="thinking", created_at=10.5)
    user_turn(s, ["Actually, can you check my invoice too?"], 20)  # user barges in
    assert len(b.turns) == 1
    assert b.turns[0].turn_end_ms is None


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["detect", "ok", "Kunt u mij helpen met mijn bestelling?"]) == 0
    assert main(["converse", str(ROOT / "examples" / "conversation.txt")]) == 0
    assert "SWITCH switch en -> nl" in capsys.readouterr().out
    base, cur = tmp_path / "base.jsonl", tmp_path / "cur.jsonl"
    assert main(["simulate", "--turns", "300", "--out", str(base)]) == 0
    assert main(["simulate", "--turns", "300", "--no-speculation", "--out", str(cur)]) == 0
    capsys.readouterr()
    assert main(["compare", str(base), str(cur)]) == 1  # no speculation is slower
    assert "first_audio_ms p50" in capsys.readouterr().out
    # Turning speculation on is faster, but adds more wasted calls per turn than allowed.
    assert main(["compare", str(cur), str(base)]) == 1
    shown = capsys.readouterr().out
    assert "wasted calls per turn" in shown
    assert "first_audio_ms p50" not in shown.split("REGRESSION")[1]
    assert main(["compare", str(base), str(base)]) == 0
    assert main(["summary", str(base)]) == 0
    fixture = ROOT / "fixtures" / "live-ttft.jsonl"
    out = tmp_path / "x.jsonl"
    assert main(["simulate", "--turns", "50", "--ttft-from", str(fixture), "--out", str(out)]) == 0
    warm = {
        float(r["ttft_ms"])
        for r in map(json.loads, fixture.read_text().splitlines())
        if not r["cold"] and r["ttft_ms"] is not None
    }
    sampled = {
        round(json.loads(x)["first_token_ms"] - json.loads(x)["llm_start_ms"], 1)
        for x in out.read_text().splitlines()
    }
    assert sampled <= {round(v, 1) for v in warm}  # cold calls are not bootstrapped
    capsys.readouterr()
    assert main(["demo"]) == 0
    assert "speculation kept" in capsys.readouterr().out


def test_ttft_report_check_detects_a_stale_summary(tmp_path: Path) -> None:
    fixture = ROOT / "fixtures" / "live-ttft.jsonl"
    stale = tmp_path / "s.json"
    stale.write_text(json.dumps({"x": 1}))
    assert main(["ttft-report", str(fixture), "--check", str(stale)]) == 1
