from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from voice_lab import harness
from voice_lab.harness import SimConfig, Thresholds, compare, percentile, simulate, summarize


def test_percentile_nearest_rank_in_integer_arithmetic() -> None:
    vals = list(range(1, 21))
    assert percentile(vals, 50) == 10
    assert percentile(vals, 95) == 19
    assert percentile(vals, 100) == 20
    assert percentile([5.0], 95) == 5.0
    # 7 * 100 / 100 is 7.000000000000001 in floating point; the rank must still be 7.
    assert percentile(list(range(1, 101)), 7) == 7
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)
    with pytest.raises(ValueError, match="integer"):
        percentile([1.0], 0)
    with pytest.raises(ValueError, match="integer"):
        percentile([1.0], 95.5)  # type: ignore[arg-type]


def test_simulation_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    a = simulate(SimConfig(), 50, seed=3)
    assert a == simulate(SimConfig(), 50, seed=3)
    assert a != simulate(SimConfig(), 50, seed=4)
    path = tmp_path / "t.jsonl"
    harness.write_jsonl(path, a)
    assert harness.read_jsonl(path) == a


def test_simulation_invariants() -> None:
    for r in simulate(SimConfig(), 300, seed=1):
        assert r.first_audio_ms >= r.final_transcript_ms  # never speak before the final
        assert r.first_audio_ms >= r.first_token_ms
        assert r.turn_end_ms > r.first_audio_ms
        assert (r.wasted_calls == 1) == (r.speculation == "restarted")
        if r.speculation != "kept":
            assert r.llm_start_ms == r.final_transcript_ms


def test_real_responder_decides_each_turn() -> None:
    always_same = replace(SimConfig(), interim_same=1.0, language_change=0.0)
    assert {r.speculation for r in simulate(always_same, 100, 1)} == {"kept"}
    never_same = replace(SimConfig(), interim_same=0.0, interim_truncated=1.0, language_change=0.0)
    assert {r.speculation for r in simulate(never_same, 100, 1)} == {"restarted"}


def test_real_policy_switches_language_and_the_switch_turn_restarts() -> None:
    cfg = replace(SimConfig(), interim_same=1.0, language_change=0.2)
    records = simulate(cfg, 200, 2)
    langs = [r.language for r in records]
    assert len(set(langs)) == 3
    switch_turns = [i for i in range(1, len(langs)) if langs[i] != langs[i - 1]]
    assert switch_turns
    # With identical interims, only a language switch can make the responder restart.
    for i, r in enumerate(records):
        assert (r.speculation == "restarted") == (i in switch_turns)


def test_speculation_helps_on_the_same_seed() -> None:
    off = summarize(simulate(SimConfig(speculation=False), 300, 7))
    on = summarize(simulate(SimConfig(speculation=True), 300, 7))
    assert on["first_audio_ms"]["p50"] < off["first_audio_ms"]["p50"]
    assert off["wasted_rate"] == 0
    assert on["wasted_calls"] == on["speculation_restarted"] > 0


def test_ttft_samples_are_bootstrapped() -> None:
    cfg = replace(SimConfig(), ttft_samples_ms=(100.0, 200.0))
    assert {round(r.ttft_ms, 1) for r in simulate(cfg, 50, 1)} <= {100.0, 200.0}


def test_compare_flags_regressions_only() -> None:
    base = summarize(simulate(SimConfig(), 1000, 7))
    assert compare(base, summarize(simulate(SimConfig(), 1000, 7))) == []
    slower_tts = replace(SimConfig(), tts_first_audio=harness.Lognormal(300.0, 0.3))
    problems = compare(base, summarize(simulate(slower_tts, 1000, 7)))
    assert any(p.startswith("first_audio_ms p50") for p in problems)
    faster = replace(SimConfig(), llm_ttft=harness.Lognormal(300.0, 0.3))
    assert compare(base, summarize(simulate(faster, 1000, 7))) == []


def test_identical_code_with_other_seeds_passes_the_default_tolerances() -> None:
    base = summarize(simulate(SimConfig(), 1000, 7))
    for seed in (1, 11, 23):
        assert compare(base, summarize(simulate(SimConfig(), 1000, seed))) == []


def test_wasted_calls_are_compared_as_rates() -> None:
    base = summarize(simulate(SimConfig(), 1000, 7))
    # Twice the turns, same behaviour: more wasted calls in total, but not per turn.
    assert compare(base, summarize(simulate(SimConfig(), 2000, 7))) == []
    worse = summarize(simulate(replace(SimConfig(), interim_same=0.6), 1000, 7))
    assert any(
        "wasted calls per turn" in p
        for p in compare(base, worse, Thresholds(p50_ms=1e9, p95_ms=1e9))
    )
