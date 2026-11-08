"""Offline demo, no network, no keys: the language policy over a scripted conversation, one
speculative turn with a fake LLM, and the latency effect of speculation in the simulator."""

from __future__ import annotations

import asyncio

from voice_lab import harness
from voice_lab.language import LanguagePolicy
from voice_lab.llm import ScriptedLLM
from voice_lab.speculative import SpeculativeResponder

CONVERSATION = [
    "Hello, I would like to ask about my order please.",
    "ok",
    "Eigenlijk spreek ik liever Nederlands, als dat kan.",
    "ja",
    "Kunt u kijken waar mijn pakket nu is?",
    "Thanks!",
    "Ik wil graag weten wanneer het pakket komt.",
]


async def _speculative_turns() -> None:
    llm = ScriptedLLM("One moment, I will check that.", first_token_s=0.05)
    for interim, final in [
        ("where is my parcel right now", "Where is my parcel right now?"),
        ("please cancel my order", "Please don't cancel my order."),
        ("i want to book a table for two", "I want to book a table for two tonight."),
    ]:
        r = SpeculativeResponder(llm, "en")
        await r.on_interim(interim, stable=True)
        await asyncio.sleep(0.03)  # the STT's endpointing delay
        out = await r.on_final(final, "en")
        print(
            f"   {interim!r} -> {final!r}: {out.mode}, "
            f"wasted calls {out.wasted_calls}, saved {out.saved_s * 1000:.0f} ms"
        )


def run_demo() -> int:
    print("== 1. reply language over a conversation (switch after 2 substantial replies)")
    policy = LanguagePolicy()
    for line in CONVERSATION:
        d = policy.observe(line)
        print(
            f"   reply in {d.reply_language}{' (switched)' if d.switched else ''}: "
            f"{d.reason}  | {line}"
        )
    print("== 2. speculative generation on a stable interim transcript (fake LLM)")
    asyncio.run(_speculative_turns())
    print("== 3. simulated latency, 300 turns, same seed, speculation off vs on")
    print("   (the real policy and responder decide each turn; stage delays are assumptions)")
    off = harness.summarize(harness.simulate(harness.SimConfig(speculation=False), 300, 7))
    on = harness.summarize(harness.simulate(harness.SimConfig(speculation=True), 300, 7))
    for m in ("first_token_ms", "first_audio_ms"):
        print(
            f"   {m}: p50 {off[m]['p50']} -> {on[m]['p50']} ms, "
            f"p95 {off[m]['p95']} -> {on[m]['p95']} ms"
        )
    print(
        f"   speculation kept {on['speculation_kept']}, restarted {on['speculation_restarted']}"
        f" (wasted calls per turn {on['wasted_rate']}), not started {on['speculation_direct']}"
    )
    return 0
