# Design notes

## Goals, in priority order

1. Never surprise the caller: the reply language changes only on clear, repeated evidence, and
   a reply is never generated for words the caller did not say.
2. Measure latency from the caller's side (end of speech to first audio), per turn, and make a
   slower or more wasteful change fail CI.
3. Keep every decision explainable and deterministic so it can be tested without audio.
4. Measure for real what can be measured cheaply (LLM time to first token), and label
   everything else as simulated.

## Decisions

**Function-word detection instead of a language-ID model.** A voice turn is often two to six
words, where general language-ID models are unreliable, and the agent only has to choose among
three known languages. Two disjoint function-word lists (shared words count for neither) and
Unicode script for Arabic give a decision a reader can follow ("3 nl / 0 en function words").
A third list (French, Spanish, Portuguese, Italian, German function words) keeps those languages
from being read as Dutch because they contain "de" or "en": two or more of them, or as many as
the English/Dutch evidence, make the text undetermined. Arabic diacritics and the tatweel are
removed before tokenising (otherwise "مَرْحَبًا" splits into five "words"), and letters used for
Persian or Urdu but not Arabic make the text undetermined. Too little evidence returns
*undetermined*; the policy then keeps the current language. Known gap: other Latin-script
languages that share short words with English (Polish "a", "i", "to") can be read as English.

**Substantial = determined language and three distinct non-filler words.** Repeated fillers
("ja ja ja", "no no no") and politeness words ("ok", "thanks", "شكرا") are not evidence of a
language change.

**Explicit requests bypass the hysteresis.** Hysteresis protects against switching on a quote
or a stray sentence; it should not make a caller who asks "can we speak Dutch?" wait another
turn. Requests are recognised by patterns, not by a model: a target language name inside a
request frame, in any of the three languages (`switch_requests.py`). The frames are narrow on
purpose, because a false request switches the call at once with no second chance: a mention
("my Dutch is bad", "the manual is in English"), a third-person statement ("my children speak
Dutch"), a negated request in the same clause ("please don't switch to Dutch", "لا تتكلم
الإنجليزية") and a turn asking for two languages are all ignored. Arabic is matched after
removing diacritics and unifying alef, taa marbuta and alef maqsura, so "بالعربية", "بالعربي" and
"بِالعَرَبِيَّةِ" are the same. A request for the current language cancels a pending switch.

**Hysteresis with "ignore, don't reset".** Non-substantial turns neither count towards a switch
nor cancel one in progress. Otherwise "ja" between two Dutch sentences would either switch too
early (if it counted) or block a switch the caller clearly wants (if it reset). A substantial
turn in the current language does reset the streak.

**A labelled corpus and a gate for the detector.** `data/detector_corpus.jsonl` (220 items) and
`data/requests.jsonl` (57 items) were written and split into `dev` and `test` before the first
run, and `evals/detector_floors.json` was written before it too: per language at least 85 %
correct (95 % for Arabic, which is recognised by script) and at most 3 % wrong, since a wrong
language costs more than an undetermined one; at least 80 % undetermined where that is the right
answer; request recall at least 85 %, near misses read as requests at most 10 %, and no request
with the wrong target. History, so the numbers can be judged:

1. First run, `dev` only: English 29/31 correct (2 undetermined: "The password reset email never
   arrived.", "Is the blue model still available?"), Dutch 31/31, Arabic 33/33, undetermined
   13/15 (wrong: "Can I pay met pin?" as English, "no no no" as English), requests 17/17, near
   misses 0/12. Floors met.
2. Two general changes on `dev`: text made only of fillers is undetermined, and 22 common
   function words were added (14 English, 8 Dutch: "never", "still", "only", "also",
   "because", ...; "nooit", "alleen", "terwijl", ...). Not added: anything aimed at the remaining
   mixed-sentence miss.
3. `test`, run once: English 31/31, Dutch 31/31, Arabic 33/33, undetermined 14/15 (wrong:
   "مرحبا" as Arabic), requests 17/17, near misses 0/11.
4. Afterwards the demo showed that "can we go back to English?" was not recognised; "go back
   (to)" and "switch back to" were added to the English request frames. Both splits were re-run
   with unchanged results.

The corpus is the author's own clean text, so these numbers mostly show consistency and guard
against regressions; they do not predict accuracy on speech-to-text output.

**Speculation keeps a reply only for the same words in the same language.** An earlier version
kept a speculation when the final transcript was 90 % similar to the interim; a review showed
that this keeps the reply to "cancel my order" for "don't cancel my order", and "two" for "four".
No similarity threshold separates those from harmless differences, and an appended word ("...
for four people tonight") can change the request as much. So the rule is equality of the word
sequence after removing case, punctuation and Arabic diacritics, and the reply language decided
for the final turn must be the one the speculation was generated in (on the turn where the
language switches, the speculation is discarded). Audio still waits for the final transcript;
the gain is that the LLM has already been working.

**Count waste honestly.** A speculation replaced before its request was sent cost nothing and
is not counted. One that was sent and then discarded, failed, or was cancelled by `close()` (an
abandoned turn) is. Failed speculative calls never fail the turn: the final text is answered.
Every task's result or error is retrieved, so nothing is left as "exception never retrieved".

**Measure from the end of user speech.** STT, LLM and TTS each report their own latencies, but
the caller experiences the silence after they stop talking. All turn metrics are relative to that
moment, so stages that overlap (speculation) are measured honestly.

**Nearest-rank percentiles in integer arithmetic.** Reproducible, no interpolation, every
reported value occurred, and `ceil(p * n / 100)` is computed with integers so p = 7 of 100 values
is the 7th, not the 8th.

**The bench runs the real decision code.** `harness.simulate` feeds synthetic interim and final
transcripts through the real `LanguagePolicy` and `SpeculativeResponder` (with an instant fake
LLM), so kept/restarted/wasted and the reply language come from the code under test; only stage
delays are drawn from stated distributions, and LLM TTFT can be bootstrapped from the warm,
successful calls of a real recording.

**The regression check compares rates and percentiles, with tolerances from measured noise.**
p50 and p95 of first token and first audio, and wasted calls *per turn* (so runs of different
length compare fairly; the turn counts are printed). Turn end is reported but not gated: it is
dominated by how long the reply is. Tolerances were set from the spread across 40 seeds with
identical code (TTFT bootstrapped from the warm `minimal` recordings):

| metric | range over 40 seeds, 300 turns | range over 40 seeds, 1,000 turns | tolerance (1,000 turns) |
|---|---|---|---|
| first audio p50 | 54.3 ms | 34.0 ms | 40 ms |
| first audio p95 | 179.8 ms | 89.3 ms | 120 ms |
| first token p50 | 44.7 ms | 27.9 ms | 40 ms |
| first token p95 | 157.3 ms | 99.8 ms | 120 ms |
| wasted calls per turn | 0.097 | 0.056 | 0.07 |

With these tolerances and 1,000 turns, all 40 seeds pass against the seed-7 baseline. The same
gate fails on: speculation turned off (+404 ms first-audio p50), a TTS whose median is 50 ms
slower (+57 ms), and interims matching the final 70 % instead of 80 % of the time (wasted calls
per turn 0.260 → 0.359). It does not detect a 25 ms slower TTS median at this sample size.
Hence the bench runs 1,000 turns (about 60 ms per run), not 300.

**Record the real model once; replay in CI.** The TTFT recording is streamed, sequential (so
calls do not compete), has the SDK's retries off (a retried request would hide a failure inside
a slower TTFT), records failures instead of dropping them, keeps the first (cold) call but
excludes it from percentiles and from the bench, and stores the reply text so the replay can
also check reply language. The hand review of the replies is a separate file with its rubric,
so the judgement is visible and versioned. A hard call budget in the adapter (default 0) means
nothing can spend money by accident.

**A fictional setting for live calls.** The live prompt puts the model in a small fictional
shop and forbids real companies, places and websites, and the questions do not ask for
addresses or real-world services. An earlier recording without that framing produced real
company names and street addresses in its replies, and was replaced.

**LiveKit via a structural bridge, one decision per user turn.** livekit-agents emits a final
transcript per STT segment, so the bridge collects a turn's segments and decides once, when the
turn is committed (`commit_turn()` from the agent's `on_user_turn_completed` hook, or at the
latest when the agent starts thinking). Final segments may come before or after the end-of-speech
event; times are computed when the turn ends. The bridge depends only on `session.on(...)`, three
event types' attributes and `agent.update_instructions`, read from the livekit-agents 1.8.11
source, so the core installs in seconds and CI stays small; `scripts/check_livekit_contract.py`
checks it against the real event classes after an upgrade.

## Calibration

| setting | value | why |
|---|---|---|
| `switch_after` | 2 | one turn can be a quote, a name or a polite word in another language |
| explicit requests | on, narrow frames | a false request switches at once, so precision before recall |
| detector floors | `evals/detector_floors.json` | written before the first corpus run |
| `min_words` | 3 distinct non-filler words | below this, function-word evidence is too thin to act on |
| `min_function_words`, `min_margin` | 2, 0.5 | a single function word, or a near tie, is not evidence |
| speculation keep rule | same words, same language | see above; no threshold to tune |
| regression tolerances | table above | set from measured seed-to-seed spread, not chosen by feel |

The language settings were not tuned on the recorded live data; they are starting points.
