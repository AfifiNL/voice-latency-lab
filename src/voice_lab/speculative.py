"""Speculative generation on interim transcripts.

Speech-to-text engines emit interim transcripts while the user talks and a final transcript
after end-of-turn detection. Waiting for the final transcript before calling the LLM adds the
whole endpointing delay to the response time. The responder here starts the LLM as soon as an
interim transcript is *stable* (the STT marks it so) and has enough words. When the final
transcript arrives:

  * the speculation is kept only if the final transcript is the same words as the interim it
    started from (ignoring case, punctuation, whitespace and Arabic diacritics) AND the reply
    language is still the one it was generated in;
  * otherwise it is cancelled and the LLM starts again on the final text.

"Same words" is deliberately strict. A similarity score would keep a reply generated for
"cancel my order" when the final says "don't cancel my order", or "two" when it says "four";
an appended word ("... for four people") can change the request just as much. Speed is never
bought with a reply to a different question.

Every speculative call that was actually issued and then discarded counts as a wasted call.
A speculation cancelled before its request was sent costs nothing and is not counted. A
speculative call that fails is discarded and the final text is answered normally. `close()`
cancels anything still running when a turn is abandoned (the caller hung up, barge-in).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from voice_lab.language import Language, words


class StreamingLLM(Protocol):
    def stream(self, text: str, language: Language) -> AsyncIterator[str]: ...


def same_words(a: str, b: str) -> bool:
    """True when both texts are the same word sequence after normalisation."""
    return words(a) == words(b)


@dataclass(frozen=True)
class SpeculationConfig:
    enabled: bool = True
    min_words: int = 3


@dataclass
class Generation:
    text: str  # the transcript it was generated from
    language: Language
    started: float
    requested: bool = False  # the LLM request was issued
    first_token: float | None = None
    finished: float | None = None
    reply: str = ""


@dataclass(frozen=True)
class Outcome:
    reply: str
    mode: Literal["kept", "restarted", "direct"]
    generated_from: str
    language: Language
    wasted_calls: int
    final_at: float
    llm_started: float
    first_token: float | None
    saved_s: float  # final_at - llm_started when kept, else 0
    reason: str


class SpeculativeResponder:
    """One instance per user turn. Always end a turn with `on_final` or `close`."""

    def __init__(
        self,
        llm: StreamingLLM,
        language: Language,
        cfg: SpeculationConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.llm = llm
        self.language = language  # reply language when the turn started
        self.cfg = cfg or SpeculationConfig()
        self.clock = clock
        self._task: asyncio.Task[Generation] | None = None
        self._spec: Generation | None = None
        self.wasted = 0
        self.issued = 0  # LLM requests issued in this turn, kept or not

    async def _run(self, gen: Generation) -> Generation:
        gen.requested = True
        self.issued += 1
        parts: list[str] = []
        async for chunk in self.llm.stream(gen.text, gen.language):
            if chunk and gen.first_token is None:
                gen.first_token = self.clock()
            parts.append(chunk)
        gen.reply = "".join(parts)
        gen.finished = self.clock()
        return gen

    async def _discard(self) -> None:
        """Cancel the speculation (if any), retrieve its result or error, count the waste."""
        task, spec = self._task, self._spec
        self._task, self._spec = None, None
        if task is None or spec is None:
            return
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        if spec.requested:
            self.wasted += 1

    async def on_interim(self, text: str, stable: bool) -> None:
        if not (self.cfg.enabled and stable and len(words(text)) >= self.cfg.min_words):
            return
        if self._spec is not None and same_words(self._spec.text, text):
            return  # already generating from this text
        await self._discard()
        self._spec = Generation(text=text, language=self.language, started=self.clock())
        self._task = asyncio.create_task(self._run(self._spec))

    async def on_final(self, text: str, language: Language | None = None) -> Outcome:
        """`language` is the reply language decided for this turn (it may have just switched)."""
        final_at = self.clock()
        lang = language or self.language
        spec, task = self._spec, self._task
        reason = "no speculation"
        if spec is not None and task is not None:
            if spec.language != lang:
                reason = f"reply language changed {spec.language} -> {lang}"
            elif not same_words(spec.text, text):
                reason = "final transcript differs from the interim"
            else:
                try:
                    gen = await task
                except asyncio.CancelledError:
                    raise
                except Exception:
                    reason = "speculative call failed"
                else:
                    self._task, self._spec = None, None
                    return Outcome(
                        gen.reply,
                        "kept",
                        gen.text,
                        gen.language,
                        self.wasted,
                        final_at,
                        gen.started,
                        gen.first_token,
                        max(0.0, final_at - gen.started),
                        "kept",
                    )
            await self._discard()
            mode: Literal["restarted", "direct"] = "restarted"
        else:
            mode = "direct"
        gen = await self._run(Generation(text=text, language=lang, started=self.clock()))
        return Outcome(
            gen.reply,
            mode,
            gen.text,
            lang,
            self.wasted,
            final_at,
            gen.started,
            gen.first_token,
            0.0,
            reason,
        )

    async def close(self) -> None:
        """Abandon the turn: cancel any running speculation."""
        await self._discard()
