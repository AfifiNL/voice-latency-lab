from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_lab import evaluate
from voice_lab.language import LanguagePolicy, PolicyConfig
from voice_lab.switch_requests import find_request

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("text", "target"),
    [
        # English
        ("Can we speak Dutch?", "nl"),
        ("Could you please continue in English?", "en"),
        ("In English please.", "en"),
        ("Do you speak Arabic?", "ar"),
        ("Please switch to Dutch.", "nl"),
        ("I'd prefer English if that's ok.", "en"),
        ("Thanks so far, but can we continue in Dutch? My English is not great.", "nl"),
        ("I'm not sure, can we speak Dutch?", "nl"),  # a negation in an earlier clause
        ("No, in Dutch please.", "nl"),
        # Dutch
        ("Kunnen we in het Engels verder?", "en"),
        ("Kunt u Nederlands praten?", "nl"),
        ("Spreekt u Engels?", "en"),
        ("In het Engels graag.", "en"),
        ("Ik spreek liever Nederlands.", "nl"),
        ("Mag het ook in het Arabisch?", "ar"),
        ("Kunnen we overschakelen naar het Engels?", "en"),
        # Arabic (with and without hamza and diacritics)
        ("هل يمكن أن نتحدث بالعربية؟", "ar"),
        ("ممكن نتكلم انجليزي؟", "en"),
        ("هل تتحدث الإنجليزية؟", "en"),
        ("بالعربي لو سمحت", "ar"),
        ("تكلم معي بالهولندية من فضلك", "nl"),
        ("أُرِيدُ التَّحَدُّثَ بِالعَرَبِيَّةِ", "ar"),
        ("أفضل الإنجليزية", "en"),
    ],
)
def test_requests_are_found(text: str, target: str) -> None:
    hit = find_request(text)
    assert hit is not None
    assert hit.language == target


@pytest.mark.parametrize(
    "text",
    [
        "My Dutch is bad.",
        "I'm learning Dutch at school.",
        "Is the manual in Dutch?",
        "Can you send me the manual in Dutch?",
        "Please don't switch to Dutch.",
        "My children speak Dutch at school.",
        "My Dutch, please excuse it, is rusty.",
        "Can we speak Dutch or English?",  # two targets: ambiguous
        "Mijn Engels is niet zo goed.",
        "De handleiding is in het Engels.",
        "Ik leer graag Nederlands.",
        "Niet in het Engels alstublieft.",
        "Mijn kinderen spreken Nederlands.",
        "عربيتي ضعيفة",
        "أدرس اللغة الهولندية في الجامعة",
        "الكتيب باللغة الإنجليزية فقط",
        "لا تتكلم الإنجليزية من فضلك",
        "أختي تتكلم الهولندية",
    ],
)
def test_near_misses_are_not_requests(text: str) -> None:
    assert find_request(text) is None


def test_explicit_request_switches_at_once() -> None:
    p = LanguagePolicy()
    d = p.observe("Hello, can we speak Dutch please?")
    assert (d.reply_language, d.switched, d.explicit) == ("nl", True, True)
    # The next Dutch turn simply confirms; an English request switches straight back.
    assert not p.observe("Ik wil graag weten waar mijn pakket is.").switched
    d = p.observe("Sorry, in English please")
    assert (d.reply_language, d.switched) == ("en", True)


def test_request_for_the_current_language_resets_a_pending_switch() -> None:
    p = LanguagePolicy()
    p.observe("Kunt u mij helpen met mijn bestelling?")  # nl 1/2
    d = p.observe("Can we keep speaking English? I'd prefer English.")
    assert (d.reply_language, d.switched, d.explicit, d.streak) == ("en", False, True, 0)
    assert not p.observe("Ik wil graag weten waar mijn pakket is.").switched


def test_explicit_requests_can_be_turned_off() -> None:
    p = LanguagePolicy(PolicyConfig(explicit_requests=False))
    assert not p.observe("Can we speak Dutch please?").switched


def test_committed_corpora_meet_their_floors() -> None:
    det = evaluate.detector_scores(evaluate.load(ROOT / "data" / "detector_corpus.jsonl"))
    req = evaluate.request_scores(evaluate.load(ROOT / "data" / "requests.jsonl"))
    floors = json.loads((ROOT / "evals" / "detector_floors.json").read_text())
    assert evaluate.check(det, req, floors) == []
    assert det["en"]["items"] >= 60
    assert det["nl"]["items"] >= 60
    assert det["ar"]["items"] >= 60
    assert det["none"]["items"] >= 30


def test_the_gate_fails_a_detector_that_guesses(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_lab.language import Detection

    monkeypatch.setattr(evaluate, "detect", lambda text: Detection("en", 1.0, 3, "guess"))
    det = evaluate.detector_scores(evaluate.load(ROOT / "data" / "detector_corpus.jsonl"))
    req = evaluate.request_scores(evaluate.load(ROOT / "data" / "requests.jsonl"))
    floors = json.loads((ROOT / "evals" / "detector_floors.json").read_text())
    problems = evaluate.check(det, req, floors)
    assert any(p.startswith("nl: wrong") for p in problems)
    assert any(p.startswith("none: undetermined") for p in problems)


def test_the_gate_fails_patterns_that_fire_on_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_lab.switch_requests import SwitchRequest

    monkeypatch.setattr(evaluate, "find_request", lambda text: SwitchRequest("nl", text))
    req = evaluate.request_scores(evaluate.load(ROOT / "data" / "requests.jsonl"))
    det = evaluate.detector_scores(evaluate.load(ROOT / "data" / "detector_corpus.jsonl"))
    floors = json.loads((ROOT / "evals" / "detector_floors.json").read_text())
    problems = evaluate.check(det, req, floors)
    assert any("false positives" in p for p in problems)
    assert any("wrong target" in p for p in problems)
