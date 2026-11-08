"""Reply-language detection and a switching policy with hysteresis.

Detection is deliberately small and explainable:
  * Arabic is recognised by script (share of letters in the Arabic Unicode blocks), after
    removing Arabic diacritics and the tatweel. Text that uses letters specific to Persian or
    Urdu (پ چ ژ گ ک ی ے ٹ ڈ ڑ) is "undetermined", not Arabic.
  * English vs Dutch is decided by counting short function words from two lists. Words that are
    common in both languages ("in", "is", "was", "we", "me", "over", "of", "die", "door", "met",
    "want", "even", "had") are in neither list, so they never count.
  * A third list holds function words of other Latin-script languages (French, Spanish,
    Portuguese, Italian, German). When they outweigh the English/Dutch evidence, or there are two
    or more of them, the text is "undetermined": a French or Spanish sentence that happens to
    contain "de" or "en" is not Dutch.
  * A reply with too little evidence is "undetermined" rather than guessed.

The policy switches the reply language only after N consecutive *substantial* replies in a new
language (default 2). A reply is substantial when its language is determined and it has at
least `min_words` distinct words that are not fillers ("ok", "ja ja ja", "no no no" are not
substantial). Other replies neither count towards a switch nor reset one in progress, so a
one-word answer or a quoted name never flips the conversation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

Language = Literal["en", "nl", "ar"]
LANGUAGES: tuple[Language, ...] = ("en", "nl", "ar")

EN_WORDS = frozenset(
    """
    the a an and to i you it not that this what with for on are there but my how where when
    please thanks thank can could would should will have has do does did your they he she be at
    from by just about which who why today tomorrow yes no need like get know think if or so all
    some any more than then up out our us been were am my mine yours their them its these those
    very really much many again here's i'm it's don't can't won't let's
    """.split()
)
NL_WORDS = frozenset(
    """
    de het een en van ik je jij u niet dat wat voor op zijn er maar om ook als bij nog naar kan
    wil mijn hoe waar wanneer graag alstublieft alsjeblieft bedankt dank goed heb hebt heeft
    moet morgen vandaag wij ze zij hier daar deze dit geen wel dus omdat te uit hij haar hem ons
    onze jullie waarom welke kunt kun zou zal gaat ga gaan mag mogen alles iets niets veel meer
    al toch nu dan toen tegen zonder tussen na sinds tot nee ja hoeveel misschien kunnen willen
    mij jou jouw uw ben bent wordt worden werd zich
    """.split()
)
# Function words of other Latin-script languages, none of which is in the two lists above.
OTHER_WORDS = frozenset(
    """
    la el los las le les des du est une pas que qui por para con del una es y mi mis tu su sus
    lo como pero muy nous vous avec dans sur pour mais ce cette mon ma mes il elle ils sont
    ou très uma não com os da das dos em ao à é ich und nicht ist das ein eine mit auf für sie
    wir sich auch noch nach mir dich bitte danke gli della che non sono
    """.split()
)
FILLERS = frozenset(
    """
    ok okay oké ja jawel nee no yes yeah yep nope uh uhm um umm hmm hm eh ah oh aha nou sure
    sorry thanks thank bedankt dank
    نعم لا ايوه أيوه اه آه طيب شكرا
    """.split()
)
for _a, _b in ((EN_WORDS, NL_WORDS), (EN_WORDS, OTHER_WORDS), (NL_WORDS, OTHER_WORDS)):
    if _a & _b:  # pragma: no cover - guards future edits of the lists
        raise RuntimeError(f"word lists overlap: {sorted(_a & _b)}")

_WORD = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)?", re.UNICODE)
# Arabic harakat, Quranic marks, superscript alef and tatweel: not letters for our purpose.
_ARABIC_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
# Letters used for Persian/Urdu (and others) but not in standard Arabic.
_NON_ARABIC_LETTERS = frozenset("پچژگکیےٹڈڑ")


def normalize(text: str) -> str:
    return _ARABIC_MARKS.sub("", unicodedata.normalize("NFC", text))


def _is_arabic(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x0600 <= cp <= 0x06FF
        or 0x0750 <= cp <= 0x077F
        or 0x08A0 <= cp <= 0x08FF
        or 0xFB50 <= cp <= 0xFDFF
        or 0xFE70 <= cp <= 0xFEFF
    )


def words(text: str) -> list[str]:
    """Lower-cased words, Arabic diacritics removed, typographic apostrophes unified."""
    return [w.lower().replace("\u2019", "'") for w in _WORD.findall(normalize(text))]


def content_words(text: str) -> set[str]:
    """Distinct words that are not fillers."""
    return {w for w in words(text) if w not in FILLERS}


@dataclass(frozen=True)
class Detection:
    language: Language | None  # None = undetermined
    confidence: float  # 0..1
    words: int  # distinct non-filler words
    reason: str


@dataclass(frozen=True)
class DetectorConfig:
    min_letters_arabic: int = 4
    min_function_words: int = 2  # EN/NL evidence needed before deciding
    min_margin: float = 0.5  # (winner - loser) / (winner + loser)


DEFAULT_DETECTOR = DetectorConfig()


def detect(text: str, cfg: DetectorConfig = DEFAULT_DETECTOR) -> Detection:
    clean = normalize(text)
    letters = [ch for ch in clean if unicodedata.category(ch).startswith("L")]
    ws = words(clean)
    distinct = len(content_words(clean))
    if not letters:
        return Detection(None, 0.0, distinct, "no letters")
    arabic = sum(1 for ch in letters if _is_arabic(ch))
    if arabic / len(letters) >= 0.5:
        if any(ch in _NON_ARABIC_LETTERS for ch in letters):
            return Detection(
                None, 0.0, distinct, "Arabic script, not Arabic (Persian/Urdu letters)"
            )
        if arabic < cfg.min_letters_arabic:
            return Detection(None, 0.0, distinct, "too short to tell")
        return Detection("ar", round(arabic / len(letters), 3), distinct, "Arabic script")
    en = sum(1 for w in ws if w in EN_WORDS)
    nl = sum(1 for w in ws if w in NL_WORDS)
    other = sum(1 for w in ws if w in OTHER_WORDS)
    if other >= 2 or (other and other >= max(en, nl)):
        return Detection(None, 0.0, distinct, f"other language ({other} non-EN/NL function words)")
    if en + nl < cfg.min_function_words:
        return Detection(None, 0.0, distinct, "too short to tell")
    margin = abs(en - nl) / (en + nl)
    if margin < cfg.min_margin:
        return Detection(None, round(margin, 3), distinct, f"mixed ({en} en / {nl} nl)")
    lang: Language = "en" if en > nl else "nl"
    return Detection(lang, round(margin, 3), distinct, f"{en} en / {nl} nl function words")


@dataclass(frozen=True)
class PolicyConfig:
    initial: Language = "en"
    switch_after: int = 2  # consecutive substantial replies in the new language
    min_words: int = 3  # distinct non-filler words a substantial reply needs
    detector: DetectorConfig = field(default_factory=DetectorConfig)


@dataclass(frozen=True)
class Decision:
    reply_language: Language
    switched: bool
    detection: Detection
    substantial: bool
    candidate: Language | None
    streak: int
    reason: str


class LanguagePolicy:
    """Stateful per conversation. Call `observe` once per user turn (the whole final
    transcript of the turn, not each STT segment)."""

    def __init__(self, cfg: PolicyConfig | None = None) -> None:
        cfg = cfg or PolicyConfig()
        if cfg.switch_after < 1:
            raise ValueError("switch_after must be at least 1")
        self.cfg = cfg
        self.current: Language = cfg.initial
        self.candidate: Language | None = None
        self.streak = 0

    def observe(self, text: str) -> Decision:
        d = detect(text, self.cfg.detector)
        lang = d.language
        substantial = lang is not None and d.words >= self.cfg.min_words
        switched = False
        if lang is None or not substantial:
            reason = "ignored: " + (d.reason if lang is None else "too few distinct words")
        elif lang == self.current:
            self.candidate, self.streak = None, 0
            reason = f"confirms {self.current}"
        else:
            if lang == self.candidate:
                self.streak += 1
            else:
                self.candidate, self.streak = lang, 1
            if self.streak >= self.cfg.switch_after:
                reason = f"switch {self.current} -> {lang} after {self.streak} replies"
                self.current, self.candidate, self.streak = lang, None, 0
                switched = True
            else:
                reason = (
                    f"{lang} reply {self.streak}/{self.cfg.switch_after}; "
                    f"still replying in {self.current}"
                )
        return Decision(self.current, switched, d, substantial, self.candidate, self.streak, reason)
