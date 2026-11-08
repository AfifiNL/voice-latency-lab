"""Streaming LLM adapters: Gemini (optional extra), a scripted fake, and a recorded replay.

Gemini authentication is left to the google-genai SDK and the environment (GEMINI_API_KEY, or
Vertex AI with GOOGLE_CLOUD_PROJECT and Application Default Credentials). This module never
reads, logs or returns credentials. Every Gemini instance has a hard call budget; the default
of 0 refuses all live calls, so nothing spends money unless a caller sets a number on purpose.

A "call" is one request. The SDK's own retries are off by default (`attempts=1`): a retried
request would hide its first failure inside a longer time to first token, so for latency
measurement a failure is recorded as a failure.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Sequence
from typing import Any

from voice_lab.language import Language

LANGUAGE_NAMES: dict[Language, str] = {"en": "English", "nl": "Dutch", "ar": "Arabic"}
RETRY_STATUS = (408, 429, 500, 502, 503, 504)


def system_prompt(language: Language) -> str:
    return (
        "You answer the phone for a small fictional shop. "
        f"Reply in {LANGUAGE_NAMES[language]} only, in at most two short sentences of plain "
        "spoken text: no lists, no markdown, no emoji. Never name real companies, "
        "organisations, brands, websites, streets, cities or countries. You know nothing about "
        "the shop's opening hours, prices, stock, orders, bookings or policies: do not state "
        "any of them, and say that you will check instead."
    )


class LLMError(RuntimeError):
    """A model call failed. Never carries vendor messages (they can echo request details)."""


class CallBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise LLMError(
                    f"live call budget exhausted ({self.used}/{self.limit}); "
                    "set VOICE_LAB_MAX_LIVE_CALLS deliberately to allow more"
                )
            self.used += 1


class GeminiStreamingLLM:
    def __init__(
        self,
        model: str,
        *,
        max_calls: int,
        vertex: bool = False,
        location: str | None = None,
        thinking_level: str | None = None,
        timeout_s: float = 30.0,
        attempts: int = 1,
        client: Any = None,
    ) -> None:
        self.model = model
        self.thinking_level = thinking_level.upper() if thinking_level else None
        self.name = f"gemini:{model}" + (
            f":thinking-{thinking_level.lower()}" if thinking_level else ""
        )
        self.budget = CallBudget(max_calls)
        self._client = (
            client if client is not None else self._make(vertex, location, timeout_s, attempts)
        )

    @staticmethod
    def _make(vertex: bool, location: str | None, timeout_s: float, attempts: int) -> Any:
        from google import genai
        from google.genai import types

        http = types.HttpOptions(
            timeout=int(timeout_s * 1000),
            retry_options=types.HttpRetryOptions(
                attempts=attempts,
                initial_delay=1.0,
                max_delay=10.0,
                http_status_codes=list(RETRY_STATUS),
            ),
        )
        if vertex:
            return genai.Client(vertexai=True, location=location or None, http_options=http)
        return genai.Client(http_options=http)

    async def stream(self, text: str, language: Language) -> AsyncIterator[str]:
        from google.genai import types

        self.budget.take()
        config = types.GenerateContentConfig(
            system_instruction=system_prompt(language),
            thinking_config=(
                types.ThinkingConfig(thinking_level=types.ThinkingLevel(self.thinking_level))
                if self.thinking_level
                else None
            ),
        )
        try:
            chunks = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=text, config=config
            )
            async for chunk in chunks:
                yield chunk.text or ""
        except Exception as exc:
            code = getattr(exc, "code", None)
            raise LLMError(
                f"Gemini call failed: {type(exc).__name__}" + (f" (HTTP {code})" if code else "")
            ) from None


class ScriptedLLM:
    """Offline fake: replies with fixed chunks after fixed delays (seconds)."""

    name = "scripted"

    def __init__(
        self,
        reply: str = "Sure, one moment.",
        first_token_s: float = 0.0,
        chunk_s: float = 0.0,
        chunks: int = 3,
    ) -> None:
        self.reply = reply
        self.first_token_s = first_token_s
        self.chunk_s = chunk_s
        self.chunks = max(1, chunks)
        self.calls: list[str] = []

    async def stream(self, text: str, language: Language) -> AsyncIterator[str]:
        self.calls.append(text)
        await asyncio.sleep(self.first_token_s)
        size = -(-len(self.reply) // self.chunks)
        for i in range(0, len(self.reply), size):
            if i:
                await asyncio.sleep(self.chunk_s)
            yield self.reply[i : i + size]


class ReplayLLM:
    """Serves recorded replies by (language, user text); a miss is an error, never a guess."""

    name = "replay"

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._by: dict[tuple[str, str], str] = {
            (r["language"], r["text"]): r["reply"] for r in rows
        }

    async def stream(self, text: str, language: Language) -> AsyncIterator[str]:
        reply = self._by.get((language, text))
        if reply is None:
            raise LLMError("no recorded reply for this request; re-record the fixture")
        yield reply
