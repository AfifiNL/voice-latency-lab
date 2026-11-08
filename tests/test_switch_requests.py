from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_lab import evaluate
from voice_lab.language import LanguagePolicy, PolicyConfig
from voice_lab.switch_requests import find_request

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ROOT / "data" / "requests_heldout.jsonl"

REQUESTS = [
    # English
    ("Can we speak Dutch?", "nl"),
    ("Can we keep speaking English?", "en"),
    ("Can we stay in English?", "en"),
    ("Would you mind speaking Dutch?", "nl"),
    ("Is it possible to speak English?", "en"),
    ("Do you speak any English?", "en"),
    ("Can we go back to English?", "en"),
    ("Could you switch back to Dutch?", "nl"),
    ("I'm not sure, can we speak Dutch?", "nl"),  # negation in an earlier clause
    ("No, in Dutch please.", "nl"),
    ("no in dutch please", "nl"),  # unpunctuated: "no" + "in <language>"
    ("can we speak dutch my english is not great", "nl"),  # target right after the verb
    ("Can we speak Dutch instead?", "nl"),
    # Dutch
    ("Kan het ook in het Engels?", "en"),
    ("Kunt u Engels?", "en"),
    ("Mag ik Engels spreken?", "en"),
    ("Kunnen we verder in het Engels?", "en"),
    ("Kunnen we overstappen op het Engels?", "en"),
    ("Kunnen we teruggaan naar het Engels?", "en"),
    ("Ik spreek liever Nederlands.", "nl"),
    # Arabic (standard, dialect, vowelled, Persian keyboard)
    ("هل يمكنك التحدث باللغة الإنجليزية؟", "en"),
    ("هل يمكننا التحدث باللغة الهولندية؟", "nl"),
    ("تتكلم عربي؟", "ar"),
    ("بتحكي عربي؟", "ar"),
    ("ممكن عربي؟", "ar"),
    ("أبغى أتكلم عربي", "ar"),
    ("عاوز أتكلم عربي", "ar"),
    ("لا بالعربي لو سمحت", "ar"),
    ("ما رأيك نتكلم عربي؟", "ar"),
    ("ممکن نتکلم عربی؟", "ar"),
    ("ممكن نرجع للعربي؟", "ar"),
    ("أُرِيدُ التَّحَدُّثَ بِالعَرَبِيَّةِ", "ar"),
]

NEAR_MISSES = [
    # stop / avoid / past switching
    "I don't want to switch to Dutch.",
    "There's no need to switch to English.",
    "Why did you switch to Dutch?",
    "No Dutch please.",
    "Stop speaking Dutch, please.",
    "Could you not speak Dutch?",
    "Hou op met Nederlands alsjeblieft.",
    "Waarom spreekt u Engels?",
    "بطل تحكي عربي لو سمحت",
    "خلاص بلاش عربي لو سمحت",
    "ما بدي نحكي انجليزي",
    # documents and sending
    "Can you send me the manual in Dutch, please?",
    "Can you translate this into Dutch?",
    "Ik wil de factuur graag in het Engels ontvangen.",
    "Kun je de brief in het Engels sturen?",
    "أرسل لي الفاتورة بالإنجليزية من فضلك",
    # other people, reported speech, statements
    "Mijn vrouw spreekt liever Engels.",
    "Could you speak Dutch to my mother when she calls?",
    "Speak English to my son, please.",
    'He said "can we speak Dutch?"',
    "قال لي: ممكن نتكلم عربي؟",
    "إحنا بنتكلم عربي في البيت",
    "We can speak Dutch at home.",
    "Wij spreken thuis Arabisch.",
    "هو يتكلم الإنجليزية",
    # alternatives and bare names
    "Can we speak Dutch or English?",
    "ممكن نتكلم عربي ولا انجليزي؟",
    "Kunt u Engels of Nederlands praten?",
    "English.",
    "عربي",
]


@pytest.mark.parametrize(("text", "target"), REQUESTS)
def test_requests_are_found(text: str, target: str) -> None:
    hit = find_request(text)
    assert hit is not None
    assert hit.language == target


@pytest.mark.parametrize("text", NEAR_MISSES)
def test_near_misses_are_not_requests(text: str) -> None:
    assert find_request(text) is None


def test_unit_tests_do_not_quote_the_held_out_set() -> None:
    held = {json.loads(line)["text"] for line in HELDOUT.read_text().splitlines()}
    quoted = {t for t, _ in REQUESTS} | set(NEAR_MISSES)
    assert not held & quoted


def test_explicit_request_switches_at_once() -> None:
    p = LanguagePolicy()
    d = p.observe("Hello, can we speak Dutch please?")
    assert (d.reply_language, d.switched, d.explicit) == ("nl", True, True)
    assert not p.observe("Ik wil graag weten waar mijn pakket is.").switched
    d = p.observe("Sorry, in English please")
    assert (d.reply_language, d.switched) == ("en", True)


def test_explicit_request_keeps_the_detection_view_of_substantial() -> None:
    d = LanguagePolicy().observe("Dutch please.")
    assert d.switched
    assert d.explicit
    assert not d.substantial  # two words: a request, but not substantial evidence


def test_request_for_the_current_language_resets_a_pending_switch() -> None:
    p = LanguagePolicy()
    p.observe("Kunt u mij helpen met mijn bestelling?")  # nl 1/2
    d = p.observe("Can we keep speaking English?")
    assert (d.reply_language, d.switched, d.explicit, d.streak) == ("en", False, True, 0)
    assert not p.observe("Ik wil graag weten waar mijn pakket is.").switched


def test_explicit_requests_can_be_turned_off() -> None:
    p = LanguagePolicy(PolicyConfig(explicit_requests=False))
    assert not p.observe("Can we speak Dutch please?").switched


def _floors() -> dict[str, object]:
    return json.loads((ROOT / "evals" / "detector_floors.json").read_text())  # type: ignore[no-any-return]


def test_committed_corpora_meet_their_floors() -> None:
    det = evaluate.detector_scores(evaluate.load(ROOT / "data" / "detector_corpus.jsonl"))
    req = evaluate.request_scores(evaluate.load(ROOT / "data" / "requests.jsonl"))
    assert evaluate.check(det, req, _floors()) == []
    assert det["en"]["items"] >= 60
    assert det["nl"]["items"] >= 60
    assert det["ar"]["items"] >= 60
    assert det["none"]["items"] >= 30


def test_the_gate_fails_a_detector_that_guesses(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_lab.language import Detection

    monkeypatch.setattr(evaluate, "detect", lambda text: Detection("en", 1.0, 3, "guess"))
    det = evaluate.detector_scores(evaluate.load(ROOT / "data" / "detector_corpus.jsonl"))
    req = evaluate.request_scores(evaluate.load(ROOT / "data" / "requests.jsonl"))
    problems = evaluate.check(det, req, _floors())
    assert any(p.startswith("nl: wrong") for p in problems)
    assert any(p.startswith("none: undetermined") for p in problems)


def test_the_gate_fails_patterns_that_fire_on_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_lab.switch_requests import SwitchRequest

    monkeypatch.setattr(evaluate, "find_request", lambda text: SwitchRequest("nl", text))
    req = evaluate.request_scores(evaluate.load(ROOT / "data" / "requests.jsonl"))
    det = evaluate.detector_scores(evaluate.load(ROOT / "data" / "detector_corpus.jsonl"))
    problems = evaluate.check(det, req, _floors())
    assert any("false positives" in p for p in problems)
    assert any("wrong target" in p for p in problems)
