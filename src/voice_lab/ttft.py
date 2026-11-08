"""Measure time-to-first-token of a streaming LLM, record it, and summarise a recording.

`record` times each call with a monotonic clock: first non-empty chunk (TTFT) and last chunk
(total). The recording keeps the reply text, so the replay can check offline that each reply is
in the requested language, using the same detector as the language policy.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voice_lab.harness import percentile
from voice_lab.language import Language, detect
from voice_lab.llm import LLMError
from voice_lab.speculative import StreamingLLM


def load_utterances(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def measure(llm: StreamingLLM, text: str, language: Language) -> dict[str, Any]:
    """One call. A failed call is recorded (with its error class), not retried or dropped."""
    start = time.perf_counter()
    first: float | None = None
    parts: list[str] = []
    error = ""
    try:
        async for chunk in llm.stream(text, language):
            if chunk and first is None:
                first = time.perf_counter()
            parts.append(chunk)
    except LLMError as exc:
        error = str(exc)
    end = time.perf_counter()
    return {
        "ttft_ms": None if first is None else round((first - start) * 1000, 1),
        "total_ms": round((end - start) * 1000, 1),
        "reply": "".join(parts).strip(),
        "error": error,
    }


async def record(
    llm: StreamingLLM, utterances: Sequence[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, u in enumerate(utterances):  # sequential: parallel calls would distort latency
        m = await measure(llm, u["text"], u["language"])
        # The first call of a process also pays for connection setup (DNS, TLS, auth token);
        # it is kept in the file but left out of the percentiles.
        rows.append(
            {
                "id": u["id"],
                "language": u["language"],
                "config": label,
                "text": u["text"],
                "cold": i == 0,
                **m,
            }
        )
    return rows


def summarize(
    rows: Sequence[dict[str, Any]], review: dict[tuple[str, str], bool] | None = None
) -> dict[str, Any]:
    """Per config and language. `review` maps (config, id) to a person's reading of whether the
    reply states a fact about the business the model cannot know."""
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(f"{r['config']}/{r['language']}", []).append(r)
        groups.setdefault(f"{r['config']}/all", []).append(r)
    for name in sorted(groups):
        g = groups[name]
        ok = [r for r in g if not r.get("error") and r.get("ttft_ms") is not None]
        warm = [r for r in ok if not r.get("cold")] or ok
        ttft = [float(r["ttft_ms"]) for r in warm]
        total = [float(r["total_ms"]) for r in warm]
        right = sum(1 for r in g if detect(r["reply"]).language == r["language"])
        undetermined = sum(1 for r in g if detect(r["reply"]).language is None)
        out[name] = {
            "calls": len(g),
            "failed_calls": len(g) - len(ok),
            "cold_calls_excluded": len(ok) - len(warm),
            "ttft_p50_ms": percentile(ttft, 50),
            "ttft_p95_ms": percentile(ttft, 95),
            "total_p50_ms": percentile(total, 50),
            "reply_in_requested_language": right,
            "reply_language_undetermined": undetermined,
        }
        if review is not None:
            out[name]["flagged_in_review"] = sum(1 for r in g if review[(r["config"], r["id"])])
    return out


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)
    path.write_text(text + "\n", encoding="utf-8")


def load_review(path: Path) -> dict[tuple[str, str], bool]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {
        (x["config"], x["id"]): bool(x["states_business_fact"] or x.get("names_real_entity"))
        for x in doc["labels"]
    }
