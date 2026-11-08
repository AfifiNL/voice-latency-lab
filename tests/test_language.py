from __future__ import annotations

import pytest

from voice_lab.language import (
    EN_WORDS,
    NL_WORDS,
    OTHER_WORDS,
    LanguagePolicy,
    PolicyConfig,
    detect,
    words,
)


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Can you help me with my order please?", "en"),
        ("What time do you open tomorrow morning?", "en"),
        ("Kunt u mij helpen met mijn bestelling?", "nl"),
        ("Ik heb een vraag over de rekening", "nl"),
        ("Hello, ik wil graag een afspraak maken", "nl"),  # English greeting, Dutch sentence
        ("مرحبا، أريد أن أغير موعدي", "ar"),
        ("أريد تغيير الموعد please", "ar"),  # mostly Arabic script
    ],
)
def test_detects(text: str, lang: str) -> None:
    assert detect(text).language == lang


@pytest.mark.parametrize(
    "text",
    ["ok", "ja", "yes", "12345", "", "?!", "Westerbrook", "مر", "is in", "Hello Jan"],
)
def test_short_or_ambiguous_is_undetermined(text: str) -> None:
    assert detect(text).language is None


@pytest.mark.parametrize(
    "text",
    [
        "Je voudrais changer mon rendez-vous pour la semaine prochaine",
        "Quiero hablar de la entrega de mi pedido en casa",
        "Ich möchte meinen Termin auf nächste Woche verschieben",
        "Vorrei cambiare il mio appuntamento",
        "Eu queria mudar a minha consulta para a semana que vem",
    ],
)
def test_other_latin_languages_are_undetermined(text: str) -> None:
    d = detect(text)
    assert d.language is None
    assert d.reason.startswith("other language")


@pytest.mark.parametrize(
    "text",
    ["می\u200cخواهم وقتم را تغییر بدهم", "میں اپنا وقت بدلنا چاہتا ہوں"],
)
def test_persian_and_urdu_are_not_arabic(text: str) -> None:
    d = detect(text)
    assert d.language is None
    assert "Persian/Urdu" in d.reason


def test_arabic_diacritics_do_not_split_words() -> None:
    assert words("مَرْحَبًا") == ["مرحبا"]
    assert words("شكرًا جزيلًا") == ["شكرا", "جزيلا"]
    assert detect("شكرًا جزيلًا").words == 1  # "shukran" is a filler


@pytest.mark.parametrize("filler", ["ja ja ja", "nee nee nee", "no no no", "ok ok, yes yes"])
def test_repeated_fillers_are_not_substantial(filler: str) -> None:
    d = LanguagePolicy(PolicyConfig(initial="ar")).observe(filler)
    assert not d.substantial
    assert d.streak == 0


def test_one_arabic_word_is_detected_but_not_substantial() -> None:
    d = LanguagePolicy().observe("مرحبا")
    assert d.detection.language == "ar"
    assert not d.substantial


def test_mixed_evidence_is_undetermined() -> None:
    d = detect("the order, de bestelling")
    assert d.language is None
    assert d.reason.startswith("mixed")


def test_shared_words_count_for_neither_language() -> None:
    assert not EN_WORDS & NL_WORDS
    assert not (EN_WORDS | NL_WORDS) & OTHER_WORDS
    for shared in ("in", "is", "was", "we", "me", "over", "of", "die", "door", "met", "want"):
        assert shared not in EN_WORDS
        assert shared not in NL_WORDS


def test_switches_only_after_two_substantial_replies() -> None:
    p = LanguagePolicy()
    first = p.observe("Kunt u mij helpen met mijn bestelling?")
    assert (first.reply_language, first.switched, first.streak) == ("en", False, 1)
    second = p.observe("Ik wil graag weten waar mijn pakket is.")
    assert (second.reply_language, second.switched) == ("nl", True)


@pytest.mark.parametrize("filler", ["ok", "ja", "12 34", "Thanks!", "مرحبا"])
def test_short_replies_neither_count_nor_reset(filler: str) -> None:
    p = LanguagePolicy()
    p.observe("Kunt u mij helpen met mijn bestelling?")
    d = p.observe(filler)
    assert not d.substantial
    assert (d.candidate, d.streak) == ("nl", 1)
    assert p.observe("Ik wil graag weten waar mijn pakket is.").switched


def test_reply_in_current_language_resets_the_streak() -> None:
    p = LanguagePolicy()
    p.observe("Kunt u mij helpen met mijn bestelling?")
    d = p.observe("Sorry, can we keep this in English please?")
    assert (d.candidate, d.streak) == (None, 0)
    assert not p.observe("Ik wil graag weten waar mijn pakket is.").switched


def test_alternating_candidates_never_switch() -> None:
    p = LanguagePolicy()
    for _ in range(3):
        assert not p.observe("Kunt u mij helpen met mijn bestelling?").switched
        assert not p.observe("هل يمكنني تغيير موعدي إلى الأسبوع القادم؟").switched
    assert p.current == "en"


def test_arabic_switch_and_back() -> None:
    p = LanguagePolicy()
    p.observe("هل يمكنني تغيير موعدي إلى الأسبوع القادم؟")
    assert p.observe("لم يصل طردي بعد، هل يمكنك التحقق من ذلك؟").reply_language == "ar"
    p.observe("Can you tell me where my parcel is now?")
    assert p.observe("And can you send me the tracking link too?").reply_language == "en"


def test_configurable_threshold_and_initial_language() -> None:
    p = LanguagePolicy(PolicyConfig(initial="nl", switch_after=1))
    assert p.observe("Can you help me with my order please?").switched
    with pytest.raises(ValueError, match="at least 1"):
        LanguagePolicy(PolicyConfig(switch_after=0))


def test_min_words_makes_short_detected_replies_insubstantial() -> None:
    p = LanguagePolicy(PolicyConfig(min_words=5))
    d = p.observe("Could you please help me")  # English, but only 5 distinct words
    assert d.detection.language == "en"
    assert d.detection.words == 5
    assert d.substantial
    d = p.observe("Yes please, thank you")  # "yes" and "thank" are fillers: 2 content words
    assert d.detection.language == "en"
    assert not d.substantial
