"""Explicit requests to change the reply language ("can we speak Dutch?").

A caller who asks for a language should get it on that turn, without the two-turn hysteresis
that protects against accidental switches. Because a false request switches the call at once,
recognition is precision-first and works on tokens, clause by clause:

1. The text is normalised (case, Arabic diacritics, alef/taa marbuta/alef maqsura, Persian
   keyboard ی and ک), split into sentences and clauses (at , ; : and the Arabic comma).
2. A request needs a *frame* with the target language right after its verb (only filler words
   such as "please", "just", "ook", "معي" in between): a modal question ("can we speak Dutch",
   "kunnen we in het Engels verder", "هل يمكنك التحدث باللغة الإنجليزية"), "would you mind",
   "is it possible to", "do you speak", an imperative at the start of a clause, "let's",
   "I'd prefer / liever / أفضل", "switch (back) to / overstappen op", a short clause that is only
   the language and "please" ("In English please", "Engels graag", "بالعربي لو سمحت"), or, in
   Arabic, a spoken-verb question ("تتكلم عربي؟").
3. A frame is vetoed by: a negation, stop or "why/did" word earlier in its clause ("I don't want
   to switch to Dutch", "No Dutch please", "Why did you switch to Dutch?", "Hou op met
   Nederlands", "بلاش عربي"; Arabic لا/ما only directly before the verb, so "لا بالعربي لو سمحت"
   is still a request); reported speech or quotation marks earlier in the sentence ("He said ...",
   "قال لي: ..."); another person as subject or addressee ("my wife prefers", "speak Dutch to my
   mother"); a document or sending word in the clause ("send me the manual in Dutch, please",
   "الفاتورة"); an alternative ("Dutch or English?").
4. A turn that asks for two different languages is ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voice_lab.language import Language, normalize

# -- vocabulary -------------------------------------------------------------------------------

_EN_NAMES: dict[str, Language] = {"english": "en", "dutch": "nl", "arabic": "ar"}
_NL_NAMES: dict[str, Language] = {
    "engels": "en",
    "nederlands": "nl",
    "hollands": "nl",
    "arabisch": "ar",
}
_AR_ROOTS: dict[str, Language] = {
    "انجليزي": "en",
    "انجليزيه": "en",
    "انكليزي": "en",
    "انكليزيه": "en",
    "هولندي": "nl",
    "هولنديه": "nl",
    "عربي": "ar",
    "عربيه": "ar",
}
_AR_PREFIXES = (("بال", True), ("لل", True), ("ال", False), ("ب", True), ("ل", True))

_SKIP = frozenset(
    "please just only also maybe now ook even maar gewoon alstublieft alsjeblieft graag "
    "misschien nu معي معاي معايا معانا معنا سويا شوي شويه الان الحين علي عليا".split()
)
_EN_VERBS = frozenset("speak talk continue switch change reply answer respond stay proceed".split())
_EN_MULTI = (
    ("keep", "speaking"),
    ("keep", "talking"),
    ("carry", "on"),
    ("go", "on"),
    ("go", "back"),
    ("switch", "back"),
    ("switch", "over"),
    ("change", "over"),
    ("do", "this"),
    ("do", "it"),
)
_EN_ING = frozenset("speaking talking continuing switching answering replying changing".split())
_NL_VERBS_AFTER = frozenset(
    "praten spreken verder verdergaan doorgaan antwoorden houden doen communiceren".split()
)
_NL_MOVE = frozenset("overstappen overschakelen switchen wisselen teruggaan terug".split())
_AR_V_WE = frozenset("نتحدث نتكلم نحكي نكمل نواصل نرجع".split())
_AR_V_YOU = frozenset("تتحدث تتكلم تحكي بتحكي بتتكلم تحدثني تكلمني تكلمنا ترد".split())
_AR_V_IMP = frozenset("تكلم تحدث كلمني حدثني احكي احكيلي كمل اكمل رد ارجع".split())
_AR_V_I = frozenset("اتكلم اتحدث احكي".split())
_AR_V_NOUN = frozenset("التحدث الكلام الحديث".split())
_AR_VERBS = _AR_V_WE | _AR_V_YOU | _AR_V_IMP | _AR_V_I | _AR_V_NOUN
_AR_MODAL = frozenset(
    "هل ممكن يمكن يمكنك يمكننا يمكنني نقدر تقدر بتقدر تستطيع نستطيع "
    "خلينا خلنا دعنا يلا رايك".split()
)
_AR_WANT = frozenset("اريد اود بدي عايز عاوز ابغى ابغي ابي حاب حابب نريد نبغى بدنا".split())
_AR_PREFER = frozenset("افضل بفضل نفضل".split())
_POLITE = frozenset("please graag alstublieft alsjeblieft aub رجاء ارجوك".split())
_POLITE_PAIRS = (("لو", "سمحت"), ("من", "فضلك"), ("اذا", "سمحت"))
_LEAD = frozenset("sorry ok okay yes so well um uh but and nee no لا اوكي طيب".split())

_NEGATIONS = frozenset(
    "not don't dont never no stop quit without didn't did doesn't won't wouldn't why "
    "niet geen nooit stop stoppen ophouden zonder waarom "
    "لن مش ليس مو بلاش بطل بدون توقف كفى كفايه ليش ليه لماذا مافي".split()
)
_NEG_PAIRS = (("hou", "op"), ("houd", "op"))
_AR_VERB_NEG = frozenset("لا ما".split())  # negations only directly before a verb
_REPORTED = frozenset(
    "said says asked asks told tells saying asking zei zegt zeiden vroeg vraagt vroegen "
    "قال قالت قالي قاللي قالوا قالو سال سالت سالني يقول بيقول".split()
)
_OTHERS = frozenset(
    "he she they wife husband mother father mom mum dad son daughter colleague friend children "
    "kids child partner boss someone anyone everyone who people "
    "hij zij vrouw man moeder vader zoon dochter collega vriend vriendin kinderen kind baas "
    "iemand mensen "
    "زوجتي زوجي امي ابوي والدي والدتي اختي اخي ابني بنتي ابنتي هو هي هم صديقي زميلي حدا احد".split()
)
_ADDRESSEE = (
    (frozenset("to with for".split()), frozenset("my him her them his their".split())),
    (frozenset("tegen met voor".split()), frozenset("mijn hem haar zijn hun".split())),
    (frozenset(["مع"]), frozenset("امي ابوي والدي والدتي زوجتي زوجي اختي اخي ابني بنتي".split())),
)
_WHEN = (frozenset("when als wanneer".split()), frozenset("he she they hij zij".split()))
_DOCUMENTS = frozenset(
    "manual invoice bill document documents contract letter form receipt email e mail website "
    "menu brochure instructions subtitles copy version file report quote send receive mail "
    "handleiding factuur rekening documenten brief formulier bon bonnetje versie bestand "
    "offerte sturen ontvangen opsturen toesturen "
    "فاتوره الفاتوره دليل الدليل عقد العقد مستند المستند المستندات رساله الرساله نموذج النموذج "
    "ايصال الايصال الموقع كتيب الكتيب نسخه النسخه ملف الملف ارسل ارسلي ابعت ابعتلي ترسل".split()
)
_OR = frozenset("or of ولا او والا".split())
_QUOTES = '"“”«»„'
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


@dataclass(frozen=True)
class SwitchRequest:
    language: Language
    phrase: str


@dataclass(frozen=True)
class _Clause:
    tokens: list[str]
    question: bool
    sentence_before: list[str]  # tokens earlier in the same sentence (other clauses)
    quoted: bool


def _norm(text: str) -> str:
    t = normalize(text).lower().replace(chr(0x2019), "'")
    t = re.sub("[أإآ]", "ا", t).replace("ة", "ه").replace("ى", "ي")
    return t.replace("ی", "ي").replace("ک", "ك")


def _clauses(text: str) -> list[_Clause]:
    out: list[_Clause] = []
    for sentence, end in re.findall(r"([^.!?؟]*)([.!?؟]*)", text):
        if not sentence.strip():
            continue
        before: list[str] = []
        consumed = 0
        for part in re.split(r"[,;:،]", sentence):
            start = sentence.find(part, consumed)
            consumed = start + len(part)
            quoted = any(q in sentence[:start] for q in _QUOTES) or any(q in part for q in _QUOTES)
            tokens = _TOKEN.findall(part)
            if tokens:
                out.append(_Clause(tokens, "?" in end or "؟" in end, list(before), quoted))
            before.extend(tokens)
    return out


def _language_token(tok: str) -> tuple[Language, bool] | None:
    """(language, prepositional) for a single token naming a language."""
    if tok in _EN_NAMES:
        return _EN_NAMES[tok], False
    if tok in _NL_NAMES:
        return _NL_NAMES[tok], False
    if tok in _AR_ROOTS:
        return _AR_ROOTS[tok], False
    for prefix, prepositional in _AR_PREFIXES:
        if tok.startswith(prefix) and tok[len(prefix) :] in _AR_ROOTS:
            return _AR_ROOTS[tok[len(prefix) :]], prepositional
    return None


def _language_phrase(t: list[str], i: int) -> tuple[Language, int, bool] | None:
    """A language phrase starting at t[i]: (language, index after it, prepositional)."""
    j, prep = i, False
    if j < len(t) and t[j] in {"in", "into", "to", "naar", "op"}:
        j, prep = j + 1, True
    elif j + 1 < len(t) and t[j] in {"back", "over"} and t[j + 1] == "to":
        j, prep = j + 2, True
    if j < len(t) and t[j] == "het":
        j += 1
    if j < len(t) and t[j] in {"باللغه", "اللغه", "لغه"}:
        prep = prep or t[j] == "باللغه"
        j += 1
    if j < len(t):
        hit = _language_token(t[j])
        if hit is not None:
            return hit[0], j + 1, prep or hit[1]
    return None


def _skip(t: list[str], i: int) -> int:
    while i < len(t) and t[i] in _SKIP:
        i += 1
    return i


def _en_verb(t: list[str], i: int) -> int | None:
    for a, b in _EN_MULTI:
        if i + 1 < len(t) and t[i] == a and t[i + 1] == b:
            return i + 2
    return i + 1 if i < len(t) and t[i] in _EN_VERBS else None


def _politeness_end(t: list[str], i: int) -> int:
    """Index after any politeness words starting at i."""
    while i < len(t):
        if t[i] in _POLITE:
            i += 1
        elif any(i + 1 < len(t) and (t[i], t[i + 1]) == pair for pair in _POLITE_PAIRS):
            i += 2
        else:
            break
    return i


def _verb_then_language(
    t: list[str], k: int, verbs: frozenset[str] | None = None
) -> tuple[Language, int, int] | None:
    """Verb at t[k] (English verbs if `verbs` is None), then the language: (lang, start, end)."""
    k = _skip(t, k)
    j = _en_verb(t, k) if verbs is None else (k + 1 if k < len(t) and t[k] in verbs else None)
    if j is None:
        return None
    hit = _language_phrase(t, _skip(t, j))
    return (hit[0], k, hit[1]) if hit else None


def _frames(c: _Clause) -> list[tuple[Language, int, int]]:
    """Every (language, start, end) a request frame matches in the clause."""
    t, n, out = c.tokens, len(c.tokens), []
    for i in range(n):
        w = t[i]
        nxt = t[i + 1] if i + 1 < n else ""
        # English: modal question, "would you mind", "is it possible", "do you speak", "let's".
        if w in {"can", "could", "may", "shall", "would", "will"} and nxt in {"we", "you", "i"}:
            if (hit := _verb_then_language(t, i + 2)) is not None:
                out.append((hit[0], i, hit[2]))
        if (
            w == "would"
            and t[i + 1 : i + 3] == ["you", "mind"]
            and i + 3 < n
            and t[i + 3] in _EN_ING
        ):
            if (lp := _language_phrase(t, _skip(t, i + 4))) is not None:
                out.append((lp[0], i, lp[1]))
        if (
            w == "is"
            and t[i + 1 : i + 2] == ["it"]
            and i + 2 < n
            and t[i + 2]
            in {
                "possible",
                "ok",
                "okay",
                "alright",
                "fine",
            }
        ):
            k = i + 3
            k += 1 if k < n and t[k] == "to" else (2 if t[k : k + 1] == ["if"] else 0)
            if (hit := _verb_then_language(t, k)) is not None:
                out.append((hit[0], i, hit[2]))
        if w == "do" and t[i + 1 : i + 2] == ["you"]:
            k = i + 2 + (1 if t[i + 2 : i + 3] == ["also"] else 0)
            if k < n and t[k] == "speak":
                k += 2 if t[k + 1 : k + 2] == ["any"] else 1
                if (lp := _language_phrase(t, k)) is not None:
                    out.append((lp[0], i, lp[1]))
        if w in {"let's"} or (w == "let" and nxt == "us"):
            if (hit := _verb_then_language(t, i + (1 if w == "let's" else 2))) is not None:
                out.append((hit[0], i, hit[2]))
        if w in {"i", "i'd"} or (w == "i" and nxt == "would"):
            k = i + 1 + (1 if nxt == "would" else 0)
            if k < n and t[k] in {"prefer", "rather"}:
                k += 1
                k += 1 if k < n and t[k] == "to" else 0
                k += 1 if k < n and (t[k] in _EN_VERBS) else 0
                if (lp := _language_phrase(t, k)) is not None:
                    out.append((lp[0], i, lp[1]))
            elif t[k : k + 2] == ["like", "to"] and (hit := _verb_then_language(t, k + 2)):
                out.append((hit[0], i, hit[2]))
        if w in {"switch", "go", "change"}:
            k = i + 1 + (1 if nxt in {"back", "over"} else 0)
            if t[k : k + 1] == ["to"] and (lp := _language_phrase(t, k)) is not None:
                out.append((lp[0], i, lp[1]))
        # Dutch: modal + pronoun ... language + verb; move verbs; "spreekt u"; "liever".
        if w in {
            "kunnen",
            "kunt",
            "kun",
            "kan",
            "mogen",
            "mag",
            "zullen",
            "zal",
            "zou",
            "zouden",
        } and (nxt in {"we", "wij", "u", "je", "jij", "ik", "het"}):
            k = _skip(t, i + 2)
            moved = k < n and t[k] in {"verder", "door", *_NL_MOVE}
            if moved:
                k += 1
            if (lp := _language_phrase(t, k)) is not None:
                after = lp[1]
                ends = after >= n or all(x in _SKIP for x in t[after:])
                if (
                    moved
                    or (after < n and t[after] in _NL_VERBS_AFTER)
                    or (ends and nxt in {"u", "je", "jij", "het"})
                ):
                    out.append((lp[0], i, after + (1 if after < n else 0)))
        if w in _NL_MOVE and nxt in {"naar", "op"} and (lp := _language_phrase(t, i + 1)):
            out.append((lp[0], i, lp[1]))
        if w in {"spreekt", "praat", "spreek"} and nxt in {"u", "je", "jij"}:
            if (lp := _language_phrase(t, _skip(t, i + 2))) is not None:
                out.append((lp[0], i, lp[1]))
        if w == "liever" and (lp := _language_phrase(t, i + 1 + (1 if nxt == "in" else 0))):
            out.append((lp[0], i, lp[1]))
        if w == "graag" and nxt == "in" and (lp := _language_phrase(t, i + 1)):
            out.append((lp[0], i, lp[1]))
        if w == "laten" and nxt == "we" and (lp := _language_phrase(t, i + 2)) is not None:
            if lp[1] < n and t[lp[1]] in _NL_VERBS_AFTER:
                out.append((lp[0], i, lp[1] + 1))
        # Arabic: modal ... verb + language; desire/prefer + verb + language.
        if w in _AR_MODAL or w in _AR_WANT:
            k = _skip(t, i + 1)
            while k < n and t[k] in _AR_MODAL | {"ان"}:
                k += 1
            if (hit := _verb_then_language(t, k, _AR_VERBS)) is not None:
                out.append((hit[0], i, hit[2]))
            if w == "ممكن" and (lp := _language_phrase(t, i + 1)) and c.question:
                if lp[1] == n:
                    out.append((lp[0], i, lp[1]))
        if w in _AR_PREFER:
            k = i + 1 + (1 if nxt == "ان" else 0)
            k += 1 if k < n and t[k] in _AR_VERBS else 0
            if (lp := _language_phrase(t, k)) is not None:
                out.append((lp[0], i, lp[1]))
    # Clause-initial forms: imperative, Arabic spoken-verb question or proposal, short
    # "language please".
    k0 = 0
    while k0 < n and t[k0] in _LEAD:
        k0 += 1
    k1 = _politeness_end(t, k0)
    imperative = (
        frozenset(
            {
                "speak",
                "talk",
                "continue",
                "switch",
                "reply",
                "answer",
                "respond",
                "stay",
                "spreek",
                "praat",
            }
        )
        | _AR_V_IMP
    )
    if (hit := _verb_then_language(t, k1, imperative)) is not None:
        out.append(hit)
    if (hit := _verb_then_language(t, k1, _AR_V_WE | _AR_V_YOU)) is not None:
        tail = t[hit[2] :]
        if c.question or (
            hit[1] == k1
            and t[k1] in _AR_V_WE
            and all(x in _POLITE | {"اذا", "لو", "ممكن", "سمحت", "من", "فضلك"} for x in tail)
        ):
            out.append(hit)
    lp = _language_phrase(t, k1)
    if lp is not None and _politeness_end(t, lp[1]) == n:
        polite = k1 > k0 or n > lp[1]
        leading_no = any(x in {"no", "nee", "لا"} for x in t[:k0])
        if (polite or lp[2]) and (lp[2] or not leading_no):
            out.append((lp[0], k1, n))
    return out


def _vetoed(c: _Clause, start: int, end: int) -> bool:
    t = c.tokens
    earlier = t[:start]
    here = _language_phrase(t, start)
    lead_ok = bool(earlier) and all(x in _LEAD for x in earlier) and here is not None and here[2]
    if not lead_ok and any(x in _NEGATIONS for x in earlier):
        return True
    if any(earlier[k : k + 2] == list(pair) for pair in _NEG_PAIRS for k in range(len(earlier))):
        return True
    if start > 0 and t[start - 1] in _AR_VERB_NEG and t[start] in _AR_VERBS | _AR_WANT | _AR_PREFER:
        return True
    if c.quoted or any(x in _REPORTED for x in (*c.sentence_before, *earlier)):
        return True
    if any(x in _OTHERS for x in earlier):
        return True
    if any(x in _DOCUMENTS for x in t):
        return True
    rest = t[end:]
    for heads, objs in (*_ADDRESSEE, _WHEN):
        if any(rest[k] in heads and rest[k + 1] in objs for k in range(len(rest) - 1)):
            return True
    return bool(rest) and rest[0] in _OR and _language_phrase(t, end + 1) is not None


def find_request(text: str) -> SwitchRequest | None:
    """The language the caller explicitly asks for, or None (no request, a vetoed request, or
    requests for two different languages in one turn)."""
    hits: list[tuple[Language, str]] = []
    for clause in _clauses(_norm(text)):
        for lang, start, end in _frames(clause):
            if not _vetoed(clause, start, end):
                hits.append((lang, " ".join(clause.tokens[start:end])))
    if len({lang for lang, _ in hits}) != 1:
        return None
    return SwitchRequest(hits[0][0], hits[0][1])
