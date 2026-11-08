from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_lab import ttft
from voice_lab.llm import GeminiStreamingLLM, LLMError, ReplayLLM, ScriptedLLM, system_prompt

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "live-ttft.jsonl"
REVIEW = ROOT / "fixtures" / "live-ttft-review.json"
SUMMARY = ROOT / "fixtures" / "live-ttft.summary.json"


class FakeAioModels:
    def __init__(self, chunks: list[str] | Exception) -> None:
        self.chunks = chunks
        self.calls = 0

    async def generate_content_stream(self, **_: Any) -> Any:
        self.calls += 1
        if isinstance(self.chunks, Exception):
            raise self.chunks

        async def gen() -> Any:
            for c in self.chunks:  # type: ignore[union-attr]
                yield SimpleNamespace(text=c)

        return gen()


def gemini(
    chunks: list[str] | Exception, max_calls: int = 3
) -> tuple[GeminiStreamingLLM, FakeAioModels]:
    models = FakeAioModels(chunks)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return GeminiStreamingLLM("m", max_calls=max_calls, client=client), models


async def collect(llm: Any, text: str = "hi", lang: str = "en") -> str:
    return "".join([c async for c in llm.stream(text, lang)])


def test_gemini_streams_chunks() -> None:
    llm, _ = gemini(["Hel", "lo", None])  # type: ignore[list-item]
    assert asyncio.run(collect(llm)) == "Hello"


def test_gemini_budget_zero_makes_no_call() -> None:
    llm, models = gemini(["x"], max_calls=0)
    with pytest.raises(LLMError, match="budget exhausted"):
        asyncio.run(collect(llm))
    assert models.calls == 0


def test_gemini_errors_hide_vendor_messages() -> None:
    class ClientError(Exception):
        code = 429

    llm, _ = gemini(ClientError("quota for project secret-project-123"))
    with pytest.raises(LLMError) as info:
        asyncio.run(collect(llm))
    assert "secret-project-123" not in str(info.value)
    assert "HTTP 429" in str(info.value)


def test_system_prompt_names_the_language_and_has_no_placeholders() -> None:
    p = system_prompt("nl")
    assert "Dutch only" in p
    assert "fictional" in p
    assert "Never name real companies" in p
    assert "[" not in p
    assert "{" not in p


def test_measure_and_record_with_a_fake() -> None:
    rows = asyncio.run(
        ttft.record(
            ScriptedLLM("Sure.", first_token_s=0.01),
            [
                {"id": "a", "language": "en", "text": "hi"},
                {"id": "b", "language": "en", "text": "yo"},
            ],
            "fake",
        )
    )
    assert [r["cold"] for r in rows] == [True, False]
    assert rows[0]["ttft_ms"] >= 10
    assert rows[1]["reply"] == "Sure."


class Failing:
    async def stream(self, text: str, language: str) -> Any:
        raise LLMError("Gemini call failed: ServerError (HTTP 503)")
        yield ""  # pragma: no cover - makes this an async generator


def test_a_failed_call_is_recorded_not_retried_or_dropped() -> None:
    rows = asyncio.run(ttft.record(Failing(), [{"id": "a", "language": "en", "text": "hi"}], "x"))
    assert rows[0]["ttft_ms"] is None
    assert "HTTP 503" in rows[0]["error"]
    ok = {
        **rows[0],
        "id": "b",
        "cold": False,
        "error": "",
        "ttft_ms": 5.0,
        "reply": "Can you help me with my order please?",
    }
    s = ttft.summarize([*rows, ok])
    assert s["x/en"]["failed_calls"] == 1
    assert s["x/en"]["ttft_p50_ms"] == 5.0


def test_sdk_retries_are_off_by_default() -> None:
    assert inspect.signature(GeminiStreamingLLM).parameters["attempts"].default == 1


def test_replay_serves_recordings_and_misses_loudly() -> None:
    llm = ReplayLLM([{"language": "en", "text": "hi", "reply": "Hello."}])
    assert asyncio.run(collect(llm)) == "Hello."
    with pytest.raises(LLMError, match="re-record"):
        asyncio.run(collect(llm, "other"))


def test_recorded_fixture_matches_its_committed_summary() -> None:
    rows = ttft.read_rows(FIXTURE)
    assert ttft.summarize(rows, ttft.load_review(REVIEW)) == json.loads(SUMMARY.read_text())


def test_review_covers_every_recorded_call_once() -> None:
    rows = ttft.read_rows(FIXTURE)
    review = json.loads(REVIEW.read_text())["labels"]
    assert sorted((r["config"], r["id"]) for r in rows) == sorted(
        (x["config"], x["id"]) for x in review
    )


def test_every_recorded_reply_is_in_the_requested_language() -> None:
    summary = ttft.summarize(ttft.read_rows(FIXTURE))
    for name, s in summary.items():
        assert s["reply_in_requested_language"] == s["calls"], name
        assert s["failed_calls"] == 0, name


def test_recorded_replies_stay_fictional() -> None:
    review = json.loads(REVIEW.read_text())
    assert "_rubric" in review
    assert not any(x["names_real_entity"] for x in review["labels"])
