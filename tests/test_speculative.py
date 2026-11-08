from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator, Coroutine
from typing import Any

import pytest

from voice_lab.language import Language
from voice_lab.llm import LLMError, ScriptedLLM
from voice_lab.speculative import Outcome, SpeculationConfig, SpeculativeResponder, same_words


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine and fail on any 'exception was never retrieved' report."""
    reports: list[dict[str, Any]] = []

    async def main() -> T:
        asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: reports.append(ctx))
        result = await coro
        gc.collect()
        await asyncio.sleep(0)
        return result

    result = asyncio.run(main())
    assert reports == []
    return result


async def speculate(
    interim: str,
    final: str,
    llm: ScriptedLLM | None = None,
    *,
    language: Language = "en",
    final_language: Language | None = None,
) -> tuple[ScriptedLLM, Outcome]:
    llm = llm or ScriptedLLM("Ok.", first_token_s=0.01)
    r = SpeculativeResponder(llm, language)
    await r.on_interim(interim, stable=True)
    await asyncio.sleep(0.02)  # the endpointing delay: the speculative request goes out
    return llm, await r.on_final(final, final_language or language)


def test_same_words_ignores_case_punctuation_and_diacritics() -> None:
    assert same_words("Where is my parcel?", "where is my parcel")
    assert same_words("هَلْ يمكنني الدفع؟", "هل يمكنني الدفع")
    assert not same_words("cancel my order", "don't cancel my order")


def test_kept_when_final_has_the_same_words() -> None:
    llm, out = run(speculate("where is my parcel now", "Where is my parcel now?"))
    assert out.mode == "kept"
    assert out.wasted_calls == 0
    assert out.saved_s > 0
    assert llm.calls == ["where is my parcel now"]


@pytest.mark.parametrize(
    ("interim", "final"),
    [
        ("please cancel my order", "Please don't cancel my order."),  # negation
        ("a table for two people", "A table for four people."),  # number
        ("ik wil de bestelling annuleren", "Ik wil de bestelling niet annuleren."),  # Dutch niet
        ("i want to book a table", "I want to book a table for four tonight."),  # appended
    ],
)
def test_any_word_difference_restarts(interim: str, final: str) -> None:
    llm, out = run(speculate(interim, final))
    assert out.mode == "restarted"
    assert out.wasted_calls == 1
    assert out.generated_from == final
    assert llm.calls == [interim, final]


def test_language_switch_on_this_turn_restarts_in_the_new_language() -> None:
    _, out = run(
        speculate(
            "ik wil graag mijn pakket volgen",
            "Ik wil graag mijn pakket volgen.",
            language="en",
            final_language="nl",
        )
    )
    assert out.mode == "restarted"
    assert out.language == "nl"
    assert "language changed" in out.reason


def test_unstable_or_short_interims_do_not_start() -> None:
    async def go() -> tuple[ScriptedLLM, Outcome]:
        llm = ScriptedLLM("Hi.")
        r = SpeculativeResponder(llm, "en")
        await r.on_interim("where is my parcel", stable=False)
        await r.on_interim("hi there", stable=True)
        return llm, await r.on_final("Where is my parcel?", "en")

    llm, out = run(go())
    assert out.mode == "direct"
    assert llm.calls == ["Where is my parcel?"]


def test_cancelled_before_the_request_went_out_is_not_wasted() -> None:
    async def go() -> tuple[ScriptedLLM, Outcome]:
        llm = ScriptedLLM("Ok.", first_token_s=0.01)
        r = SpeculativeResponder(llm, "en")
        await r.on_interim("i want to book a table", stable=True)
        # No await in between: the first speculation never ran.
        await r.on_interim("i want to book a table for two tonight", stable=True)
        await asyncio.sleep(0.02)
        return llm, await r.on_final("I want to book a table for two tonight.", "en")

    llm, out = run(go())
    assert out.mode == "kept"
    assert out.wasted_calls == 0
    assert llm.calls == ["i want to book a table for two tonight"]


def test_replaced_after_the_request_went_out_is_wasted() -> None:
    async def go() -> Outcome:
        llm = ScriptedLLM("Ok.", first_token_s=0.05)
        r = SpeculativeResponder(llm, "en")
        await r.on_interim("i want to book a table", stable=True)
        await asyncio.sleep(0.01)
        await r.on_interim("i want to book a table for two tonight", stable=True)
        await asyncio.sleep(0.01)
        return await r.on_final("I want to book a table for two tonight.", "en")

    out = run(go())
    assert out.mode == "kept"
    assert out.wasted_calls == 1


class FailingOnce:
    """Fails the first request, answers the rest."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def stream(self, text: str, language: Language) -> AsyncIterator[str]:
        self.calls.append(text)
        await asyncio.sleep(0)
        if len(self.calls) == 1:
            raise LLMError("speculative call failed")
        yield "Answer."


def test_failed_speculation_restarts_on_the_final() -> None:
    async def go() -> tuple[FailingOnce, Outcome]:
        llm = FailingOnce()
        r = SpeculativeResponder(llm, "en")
        await r.on_interim("where is my parcel now", stable=True)
        await asyncio.sleep(0.01)
        return llm, await r.on_final("Where is my parcel now?", "en")

    llm, out = run(go())
    assert out.mode == "restarted"
    assert out.reply == "Answer."
    assert out.wasted_calls == 1
    assert out.reason == "speculative call failed"
    assert len(llm.calls) == 2


def test_failed_speculation_that_is_replaced_leaves_no_unretrieved_error() -> None:
    async def go() -> None:
        r = SpeculativeResponder(FailingOnce(), "en")
        await r.on_interim("where is my parcel now", stable=True)
        await asyncio.sleep(0.01)  # the speculative call has failed by now
        await r.on_interim("where is my second parcel now", stable=True)
        await r.close()

    run(go())


def test_close_cancels_an_abandoned_turn() -> None:
    async def go() -> tuple[SpeculativeResponder, ScriptedLLM]:
        llm = ScriptedLLM("Ok.", first_token_s=10.0)
        r = SpeculativeResponder(llm, "en")
        await r.on_interim("where is my parcel now", stable=True)
        await asyncio.sleep(0.01)
        await r.close()
        return r, llm

    r, llm = run(go())
    assert r.wasted == 1
    assert llm.calls == ["where is my parcel now"]


def test_disabled_never_speculates() -> None:
    async def go() -> Outcome:
        r = SpeculativeResponder(ScriptedLLM("Ok."), "en", SpeculationConfig(enabled=False))
        await r.on_interim("where is my parcel now", stable=True)
        return await r.on_final("where is my parcel now", "en")

    assert run(go()).mode == "direct"
