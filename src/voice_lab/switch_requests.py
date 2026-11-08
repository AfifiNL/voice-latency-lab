"""Explicit requests to change the reply language ("can we speak Dutch?").

A caller who asks for a language should get it on that turn, without the two-turn hysteresis
that protects against accidental switches. Requests are found with deterministic patterns in
English, Dutch and Arabic, for any of the three target languages ("can we continue in Dutch?",
"kunnen we in het Engels verder?", "ممكن نتكلم عربي؟", "in English please").

What counts as a request is deliberately narrow: a language name together with a request frame
(a modal question with a speaking verb, an imperative, "please", "I'd prefer ...", "switch to
..."). A mention that is not a request ("my Dutch is bad", "the manual is in English",
"أدرس اللغة الهولندية") does not match. A request is ignored when a negation stands right before
it in the same clause ("please don't switch to Dutch", "niet in het Engels", "لا تتكلم
الإنجليزية"), and when one turn asks for two different languages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voice_lab.language import Language, normalize

_AR_PREFIX = r"(?:باللغه |اللغه )?(?:بال|ال|ب)?"
LANGUAGE_NAMES: dict[Language, str] = {
    "en": rf"(?:english|engels|{_AR_PREFIX}(?:انجليزيه|انجليزي|انكليزيه|انكليزي))",
    "nl": rf"(?:dutch|nederlands|hollands|{_AR_PREFIX}(?:هولنديه|هولندي))",
    "ar": rf"(?:arabic|arabisch|{_AR_PREFIX}(?:عربيه|عربي))",
}

_EN = [
    r"\b(?:can|could|may|shall)\s+(?:we|you|i)\s+(?:please\s+|just\s+|maybe\s+)?"
    r"(?:speak|talk|continue|carry on|go on|go back|switch|change|do this|do it|have this|reply|"
    r"answer)"
    r"(?:\s+\w+){0,2}?\s+(?:in\s+|to\s+|over to\s+|back to\s+)?{L}\b",
    r"\bdo\s+you\s+(?:also\s+)?speak\s+{L}\b",
    r"(?:^|[.!?,;]\s*)(?:please\s+)?(?:speak|talk|continue|switch|reply|answer|respond)\s+"
    r"(?:to\s+|in\s+)?{L}\b",
    r"\b(?:let's|let us)\s+(?:speak|talk|continue|switch|go on)\s+(?:in\s+|to\s+)?{L}\b",
    r"\bi(?:'d| would)?\s+(?:prefer|rather|like to)\s+(?:to\s+)?(?:speak\s+|talk\s+|continue\s+)?"
    r"(?:in\s+)?{L}\b",
    r"\b(?:in\s+)?{L}\s*,?\s*please\b(?!\s+(?:excuse|forgive|ignore|bear))",
    r"\bplease\s*,?\s*(?:in\s+)?{L}\b",
    r"\b(?:switch|go)(?:\s+over|\s+back)?\s+to\s+{L}\b",
]
_NL = [
    r"\b(?:kunnen|kunt|kun|kan|mogen|zullen|zal)\s+(?:we|wij|u|je|jij|ik)\s+(?:\w+\s+){0,3}?"
    r"(?:in\s+het\s+)?{L}\s+(?:praten|spreken|verder|verdergaan|doen|antwoorden|houden)\b",
    r"\b(?:overschakelen|overstappen|wisselen|switchen)\s+naar\s+(?:het\s+)?{L}\b",
    r"\b(?:spreekt|praat)\s+u\s+(?:ook\s+)?{L}\b",
    r"\b(?:spreek|praat)\s+(?:je|jij)\s+(?:ook\s+)?{L}\b",
    r"\b(?:in\s+het\s+)?{L}\s*,?\s*(?:alstublieft|alsjeblieft|graag|aub)\b",
    r"\bgraag\s+in\s+het\s+{L}\b",
    r"\bliever\s+(?:in\s+het\s+)?{L}\b",
    r"\bmag\s+(?:het|dat)\s+(?:ook\s+)?in\s+het\s+{L}\b",
    r"(?:^|[.!?,;]\s*)(?:spreek|praat)\s+(?:alstublieft\s+|aub\s+|even\s+)?(?:in\s+het\s+)?{L}\b",
]
_AR = [
    r"(?:نتحدث|نتكلم|نكمل|نواصل|نحكي)\s+(?:معا\s+|سويا\s+)?{L}",
    r"(?:هل|ممكن|يمكنك|تقدر|بتقدر)\s+(?:\S+\s+){0,2}?"
    r"(?:تتحدث|تتكلم|تحدثني|تكلمني|ترد|تحكي)\s+(?:معي\s+|علي\s+)?{L}",
    r"(?:^|[.!?,;،؟]\s*)(?:من فضلك\s+|لو سمحت\s+|رجاء\s+)?"
    r"(?:تكلم|تحدث|كلمني|حدثني|رد|اكمل|كمل|احكي)\s+(?:معي\s+)?{L}",
    r"{L}\s*[،,]?\s*(?:من فضلك|لو سمحت|رجاء|ارجوك|اذا سمحت)",
    r"(?:من فضلك|لو سمحت|رجاء|ارجوك|اذا سمحت)\s*[،,]?\s*{L}",
    r"(?:اريد|اود|بدي|عايز|حاب)\s+(?:ان\s+)?(?:التحدث|الكلام|اتحدث|اتكلم|نتحدث|نتكلم|احكي|نحكي)\s+{L}",
    r"(?:افضل|بفضل)\s+(?:التحدث\s+|الكلام\s+)?{L}",
]
_PATTERNS: list[tuple[Language, re.Pattern[str]]] = [
    (lang, re.compile(template.replace("{L}", name)))
    for template in (*_EN, *_NL, *_AR)
    for lang, name in LANGUAGE_NAMES.items()
]
_NEGATIONS = frozenset(
    {
        "not",
        "don't",
        "dont",
        "never",
        "won't",
        "doesn't",  # English
        "niet",
        "geen",
        "nooit",  # Dutch
        "لا",
        "ما",
        "لن",
        "مش",
        "ليس",
        "مو",  # Arabic
    }
)
_CLAUSE_END = re.compile(r"[.!?,;،؟]")


@dataclass(frozen=True)
class SwitchRequest:
    language: Language
    phrase: str


def _norm(text: str) -> str:
    t = normalize(text).lower().replace(chr(0x2019), "'")
    t = re.sub("[أإآ]", "ا", t).replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", t).strip()


def _negated(text: str, start: int, end: int) -> bool:
    before = _CLAUSE_END.split(text[:start])[-1].split()[-2:]
    inside = text[start:end].split()
    return any(w.strip(".,!?;،؟") in _NEGATIONS for w in (*before, *inside))


def find_request(text: str) -> SwitchRequest | None:
    """The language the caller explicitly asks for, or None (no request, negated request, or
    requests for two different languages in one turn)."""
    t = _norm(text)
    hits: list[tuple[Language, str]] = []
    for lang, rx in _PATTERNS:
        for m in rx.finditer(t):
            if not _negated(t, m.start(), m.end()):
                hits.append((lang, m.group(0).strip(" .!?,;،؟")))
    if len({lang for lang, _ in hits}) != 1:
        return None
    return SwitchRequest(hits[0][0], hits[0][1])
