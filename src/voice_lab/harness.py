"""Per-turn latency records, summaries, a regression check, and a seeded simulator.

Every turn is measured from the moment the user stops speaking (t = 0 ms):

    final_transcript_ms   the STT final transcript arrives
    llm_start_ms          the LLM call that produced the spoken reply started
                          (negative when it started on a stable interim transcript)
    first_token_ms        its first token arrived
    first_audio_ms        the first reply audio plays: what the caller experiences as delay
    turn_end_ms           the reply finished playing

Percentiles use the nearest-rank method in integer arithmetic, so a summary is reproducible and
every reported value is one that was actually observed.

`simulate` is the bench path: it runs the real `LanguagePolicy` and `SpeculativeResponder` on
every turn (which language to answer in, whether the speculation is kept, which calls are
wasted) and draws only the stage delays from stated distributions.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from voice_lab.language import LANGUAGES, Language, LanguagePolicy, words
from voice_lab.speculative import SpeculationConfig, SpeculativeResponder

METRICS = ("final_transcript_ms", "ttft_ms", "first_token_ms", "first_audio_ms", "turn_end_ms")
Mode = Literal["kept", "restarted", "direct"]


@dataclass(frozen=True)
class TurnRecord:
    turn: int
    language: str
    final_transcript_ms: float
    llm_start_ms: float
    first_token_ms: float
    first_audio_ms: float
    turn_end_ms: float
    speculation: Mode
    wasted_calls: int = 0

    @property
    def ttft_ms(self) -> float:
        return self.first_token_ms - self.llm_start_ms

    def value(self, metric: str) -> float:
        return float(getattr(self, metric))


def write_jsonl(path: Path, records: Iterable[TurnRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(asdict(r), sort_keys=True) for r in records]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[TurnRecord]:
    return [
        TurnRecord(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: Sequence[float], p: int) -> float:
    """Nearest-rank percentile; `p` is an integer in 1..100 so the rank is exact."""
    if not values:
        raise ValueError("no values")
    if not isinstance(p, int) or not 0 < p <= 100:
        raise ValueError("p must be an integer in 1..100")
    ordered = sorted(values)
    rank = -(-p * len(ordered) // 100)  # ceil(p * n / 100) without floating point
    return ordered[rank - 1]


def summarize(records: Sequence[TurnRecord]) -> dict[str, Any]:
    out: dict[str, Any] = {"turns": len(records)}
    for m in METRICS:
        vals = [r.value(m) for r in records]
        out[m] = {
            "p50": round(percentile(vals, 50), 1),
            "p95": round(percentile(vals, 95), 1),
            "mean": round(sum(vals) / len(vals), 1),
        }
    for mode in ("kept", "restarted", "direct"):
        out[f"speculation_{mode}"] = sum(1 for r in records if r.speculation == mode)
    out["wasted_calls"] = sum(r.wasted_calls for r in records)
    out["wasted_rate"] = round(out["wasted_calls"] / len(records), 4)
    return out


@dataclass(frozen=True)
class Thresholds:
    """A latency metric regresses when it grows by more than max(abs, pct * baseline); the
    wasted-call rate (wasted calls per turn) when it grows by more than `wasted_rate`.
    Defaults sit just above the largest seed-to-seed spread of the 1,000-turn bench with
    identical code (docs/DESIGN.md). Turn end is reported but not gated: it is dominated by
    how long the reply is, not by latency."""

    p50_ms: float = 40.0
    p95_ms: float = 120.0
    pct: float = 0.0
    wasted_rate: float = 0.07
    metrics: tuple[str, ...] = ("first_audio_ms", "first_token_ms")


def compare(
    baseline: dict[str, Any], current: dict[str, Any], t: Thresholds | None = None
) -> list[str]:
    t = t or Thresholds()
    problems: list[str] = []
    for m in t.metrics:
        for stat, abs_ms in (("p50", t.p50_ms), ("p95", t.p95_ms)):
            old, new = float(baseline[m][stat]), float(current[m][stat])
            allowed = max(abs_ms, t.pct * old)
            if new - old > allowed:
                problems.append(
                    f"{m} {stat}: {old:.1f} -> {new:.1f} ms (+{new - old:.1f}, "
                    f"allowed +{allowed:.1f})"
                )
    old_rate, new_rate = float(baseline["wasted_rate"]), float(current["wasted_rate"])
    if new_rate - old_rate > t.wasted_rate:
        problems.append(
            f"wasted calls per turn: {old_rate:.3f} -> {new_rate:.3f} "
            f"(allowed +{t.wasted_rate:.3f})"
        )
    return problems


# --- Bench simulator -----------------------------------------------------------------------
# Stage delays below are illustrative assumptions for a demo, not measurements. Replace them
# with your own recorded turns (or pass measured TTFT samples) before drawing conclusions.


@dataclass(frozen=True)
class Lognormal:
    median_ms: float
    sigma: float

    def sample(self, rng: random.Random) -> float:
        return self.median_ms * math.exp(rng.gauss(0.0, self.sigma))


@dataclass(frozen=True)
class SimConfig:
    stt_final: Lognormal = Lognormal(450.0, 0.35)  # end of speech -> final transcript
    interim_stable_ms: tuple[float, float] = (-150.0, 100.0)  # uniform, relative to speech end
    # How the last stable interim relates to the final transcript (probabilities).
    interim_same: float = 0.8  # identical words
    interim_truncated: float = 0.12  # the final adds the last word(s)
    # the rest: one word differs (a misheard word, a number, a negation)
    language_change: float = 0.1  # chance per turn that the caller changes language
    llm_ttft: Lognormal = Lognormal(600.0, 0.3)
    tts_first_audio: Lognormal = Lognormal(200.0, 0.3)
    reply_audio: tuple[float, float] = (2_000.0, 6_000.0)  # uniform playback duration
    speculation: bool = True
    ttft_samples_ms: tuple[float, ...] = field(default=())  # bootstrap these instead of llm_ttft


DEFAULT_UTTERANCES: dict[Language, tuple[str, ...]] = {
    "en": (
        "Can I change my appointment to next week?",
        "Do you have the blue backpack in stock?",
        "Can someone call me back later today?",
    ),
    "nl": (
        "Kan ik mijn afspraak naar volgende week verzetten?",
        "Heeft u de blauwe rugzak op voorraad?",
        "Kan iemand mij later vandaag terugbellen?",
    ),
    "ar": (
        "هل يمكنني تغيير موعدي إلى الأسبوع القادم؟",
        "هل لديكم الحقيبة الزرقاء في المخزن؟",
        "هل يمكن أن يتصل بي أحد لاحقا اليوم؟",
    ),
}


class _InstantLLM:
    """The bench measures decisions with real components; delays come from the config."""

    async def stream(self, text: str, language: Language) -> AsyncIterator[str]:
        yield "ok"


def _interim_for(final: str, rng: random.Random, cfg: SimConfig) -> str:
    ws = words(final)
    roll = rng.random()
    if roll < cfg.interim_same or len(ws) < 4:
        return " ".join(ws)  # interim transcripts usually come without punctuation or case
    if roll < cfg.interim_same + cfg.interim_truncated:
        return " ".join(ws[: -rng.choice((1, 2))])
    i = rng.randrange(len(ws))
    return " ".join([*ws[:i], "uh" if ws[i] != "uh" else "um", *ws[i + 1 :]])


async def _decide(
    cfg: SimConfig, plan: list[tuple[str, str, bool]]
) -> list[tuple[Language, Mode, int]]:
    """Run the real policy and responder over (interim, final, interim_before_final) turns."""
    policy, llm = LanguagePolicy(), _InstantLLM()
    out: list[tuple[Language, Mode, int]] = []
    for interim, final, early in plan:
        r = SpeculativeResponder(llm, policy.current, SpeculationConfig(enabled=cfg.speculation))
        if early:
            await r.on_interim(interim, stable=True)
            await asyncio.sleep(0)  # time passes before the final: the request goes out
        decision = policy.observe(final)
        outcome = await r.on_final(final, decision.reply_language)
        await r.close()
        out.append((decision.reply_language, outcome.mode, outcome.wasted_calls))
    return out


def simulate(
    cfg: SimConfig,
    turns: int,
    seed: int,
    utterances: dict[Language, tuple[str, ...]] | None = None,
) -> list[TurnRecord]:
    """Deterministic for a given (cfg, turns, seed, utterances). Audio never plays before the
    final transcript, and TTS starts only after it (no speculative synthesis)."""
    corpus = utterances or DEFAULT_UTTERANCES
    rng = random.Random(seed)  # noqa: S311 - seeded simulation, not cryptography
    spoken: Language = "en"
    plan: list[tuple[str, str, bool]] = []
    delays: list[tuple[float, float, float, float, float]] = []
    for _ in range(turns):
        if rng.random() < cfg.language_change:
            spoken = rng.choice([x for x in LANGUAGES if x != spoken])
        final_text = rng.choice(corpus[spoken])
        interim_text = _interim_for(final_text, rng, cfg)
        final = cfg.stt_final.sample(rng)
        interim_at = rng.uniform(*cfg.interim_stable_ms)
        ttft = rng.choice(cfg.ttft_samples_ms) if cfg.ttft_samples_ms else cfg.llm_ttft.sample(rng)
        delays.append(
            (
                final,
                interim_at,
                ttft,
                cfg.tts_first_audio.sample(rng),
                rng.uniform(*cfg.reply_audio),
            )
        )
        plan.append((interim_text, final_text, interim_at < final))
    decided = asyncio.run(_decide(cfg, plan))
    out: list[TurnRecord] = []
    for i, ((final, interim_at, ttft, tts, audio), (lang, mode, wasted)) in enumerate(
        zip(delays, decided, strict=True)
    ):
        start = interim_at if mode == "kept" else final
        first_token = start + ttft
        first_audio = max(first_token, final) + tts
        out.append(
            TurnRecord(
                turn=i,
                language=lang,
                final_transcript_ms=round(final, 1),
                llm_start_ms=round(start, 1),
                first_token_ms=round(first_token, 1),
                first_audio_ms=round(first_audio, 1),
                turn_end_ms=round(first_audio + audio, 1),
                speculation=mode,
                wasted_calls=wasted,
            )
        )
    return out
