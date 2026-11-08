# voice-latency-lab

A small, testable starter for the parts of a multilingual voice agent that decide how it
*feels* on a call: which language to answer in, when to start generating, and how long the
caller waits. Three pieces, each usable on its own:

- **Reply-language policy (EN / NL / AR).** An explicit request ("can we speak Dutch?",
  "kunnen we in het Engels verder?", "بالعربي لو سمحت") switches the reply language on that
  turn. Otherwise it switches only after two consecutive substantial turns in a new language;
  "ok", "ja ja ja", a number or a single word never flips the call. Measured on a labelled
  corpus of 220 sentences and 166 request/near-miss items (66 held out), gated in CI.
- **Speculative generation on interim transcripts.** Starts the LLM on a stable interim
  transcript and keeps the result only if the final transcript is the same words and the reply
  language did not change; otherwise it cancels, restarts on the final text, and counts the
  wasted call.
- **A per-turn latency harness.** JSONL turn records (end of speech → final transcript → first
  token → first audio → end of turn), nearest-rank p50/p95, and a `compare` command that exits 1
  when a change makes the caller wait longer or wastes more calls per turn.

It also includes a measured time-to-first-token (TTFT) recording of a real model (Gemini Flash,
streamed, EN/NL/AR) that CI replays offline, and an optional bridge into a LiveKit Agents session.
Speech-to-text and text-to-speech are **simulated** throughout; see the limits.

## What it does

`make demo` (offline, no keys):

```text
== 1. reply language: 2 substantial turns switch it, an explicit request at once
   reply in en: confirms en  | Hello, I would like to ask about my order please.
   reply in en: ignored: only fillers  | ok
   reply in en: nl reply 1/2; still replying in en  | Ik heb een vraag over mijn bestelling van vorige week.
   reply in en: ignored: only fillers  | ja
   reply in nl (switched): switch en -> nl after 2 replies  | Kunt u kijken waar mijn pakket nu is?
   reply in nl: ignored: only fillers  | Thanks!
   reply in en (switched): explicit request ("can we go back to english"): switch nl -> en  | Sorry, can we go back to English? My colleague is listening too.
   reply in en: confirms en  | My Dutch is not very good anyway.
== 2. speculative generation on a stable interim transcript (fake LLM)
   'where is my parcel right now' -> 'Where is my parcel right now?': kept, wasted calls 0, saved 30 ms
   'please cancel my order' -> "Please don't cancel my order.": restarted, wasted calls 1, saved 0 ms
   'i want to book a table for two' -> 'I want to book a table for two tonight.': restarted, wasted calls 1, saved 0 ms
== 3. simulated latency, 300 turns, same seed, speculation off vs on
   (the real policy and responder decide each turn; stage delays are assumptions)
   first_token_ms: p50 1089.1 -> 666.7 ms, p95 1601.1 -> 1424.4 ms
   first_audio_ms: p50 1302.5 -> 937.6 ms, p95 1815.0 -> 1602.0 ms
   speculation kept 213, restarted 87 (wasted calls per turn 0.29), not started 0
```

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
make install    # .venv with extras and dev tools (locked)
make demo       # the output above
make check      # lint, strict mypy on src/, 156 tests, latency, TTFT and detector gates (what CI runs)
```

```bash
uv run voice-lab detect "Kunt u mij helpen met mijn bestelling?" "ok"
uv run voice-lab converse examples/conversation.txt
uv run voice-lab simulate --turns 1000 --out reports/a.jsonl
uv run voice-lab simulate --turns 1000 --no-speculation --out reports/b.jsonl
uv run voice-lab compare reports/a.jsonl reports/b.jsonl    # exit 1: b is slower
```

## How it works

```text
user stops speaking (t = 0)
   │  interim transcripts ──stable, ≥ 3 words──► speculative LLM call ─┐
   ▼                                                                    │
final transcript(s) of the turn ──► language policy ──► reply language │
   │                                                                    │
   ├── same words as the interim AND same reply language ──► keep ◄─────┘
   └── otherwise ──► cancel (a wasted call if it was sent) ──► LLM on the final text
   ▼
first token ──► TTS ──► first audio (never before the final transcript)
   │
   ▼
turn record: final_transcript_ms, llm_start_ms, first_token_ms, first_audio_ms, turn_end_ms
```

**Language detection** is deliberately small and explainable. Arabic is recognised by script,
after removing Arabic diacritics and the tatweel; text with letters specific to Persian or Urdu
(پ چ ژ گ ک ی ے ٹ ڈ ڑ) is undetermined. English and Dutch are separated by counting function
words from two disjoint lists; words common to both ("in", "is", "was", "we", "over", "die",
"door", "met", "want") count for neither. A third list of French, Spanish, Portuguese, Italian
and German function words makes such sentences undetermined instead of "Dutch" because they
contain "de" or "en". Fewer than two function words, or a margin under 0.5, is undetermined, not
a guess.

**Explicit requests** are found by deterministic, clause-scoped frames in English, Dutch and
Arabic, for any of the three target languages, with the target language right after the verb:
"can we speak / keep speaking / go back to Dutch", "would you mind speaking", "is it possible to
speak", "do you speak (any)", an imperative, "let's", "I'd prefer", "kunnen we in het Engels
verder", "kunt u Engels?", "overstappen op het Engels", "هل يمكنك التحدث باللغة الإنجليزية؟",
"تتكلم عربي؟", "أبغى أتكلم عربي", or a short clause that is only the language and "please"
("In English please", "Engels graag", "بالعربي لو سمحت"). A frame is vetoed by a negation, stop
or "why/did" word earlier in its clause ("I don't want to switch to Dutch", "No Dutch please",
"Why did you switch to Dutch?", "Hou op met Nederlands", "بلاش عربي"), reported speech or
quotation marks ("He said ...", "قال لي: ..."), another person as subject or addressee ("my wife
prefers English", "speak Dutch to my mother"), a document or sending word ("send me the manual
in Dutch, please", "الفاتورة"), or an alternative ("Dutch or English?"). A turn asking for two
languages is ignored. A request switches at once, or keeps the current language and cancels a
pending switch.

**The policy** otherwise counts a turn as substantial when its language is determined and it has
at least three distinct words that are not fillers ("ok", "ja", "no", "yes", "uh", "شكرا"...).
Other turns neither count towards a switch nor reset one in progress. A substantial turn in the
current language resets the streak. N (default 2) and the minimum are settings.

**Speculation** is an asyncio component (`SpeculativeResponder`): `on_interim(text, stable)`
starts or replaces a generation, `on_final(text, language)` keeps it or cancels and restarts,
`close()` cancels it when a turn is abandoned. "Same words" ignores case, punctuation and Arabic
diacritics and nothing else: "cancel my order" vs "don't cancel my order", "two" vs "four", or a
word added at the end all restart. A speculation cancelled before its request was sent is not
counted as wasted; a speculative call that fails is discarded and the final text is answered.

**The harness** measures from the end of user speech, because that is the silence the caller
hears. `voice-lab simulate` is the bench: for every turn it runs the real `LanguagePolicy` and
`SpeculativeResponder` on synthetic interim/final transcripts (which language, kept or restarted,
which calls are wasted) and draws only the stage delays from stated distributions, with LLM TTFT
bootstrapped from the real recording (`--ttft-from`; failed and cold calls are left out). `make
bench` runs 1,000 turns and fails CI if p50 of first token or first audio grows by more than
40 ms, p95 by more than 120 ms, or wasted calls per turn by more than 0.07. Those tolerances sit
just above the largest spread seen across 40 seeds with identical code (docs/DESIGN.md).

## Measured: language detection and explicit requests

**Detector.** `data/detector_corpus.jsonl` holds 220 labelled items: 60 English, 60 Dutch and 65
Arabic caller sentences (Modern Standard Arabic plus four colloquial and one fully vowelled), and
35 others: 12 sentences in other languages (French, Spanish, German, Italian, Portuguese, Polish,
Turkish, Persian, Urdu), 16 short items (fillers, numbers, an invented place name) and 7 mixes of
English, Dutch and Arabic (5 labelled with their dominant language, 2 balanced ones labelled
"undetermined"). It was split into `dev` and `test` and the floors in `evals/detector_floors.json`
were written before the first run; `make check` fails below them.

| expected | items | correct | undetermined | wrong |
|---|---|---|---|---|
| English | 62 | 62 | 0 | 0 |
| Dutch | 62 | 62 | 0 | 0 |
| Arabic | 66 | 66 | 0 | 0 |
| undetermined | 30 | 28 | - | 2 |

The two misses: "Can I pay met pin?" (labelled a balanced mix, detected as English) and "مرحبا"
(one word, labelled too short, detected as Arabic; the policy still ignores it as one word).
Two general detector changes were made after the first `dev` run (docs/DESIGN.md). Of the 110
`test` items, 6 short ones turned out to be quoted in unit tests and were moved to `dev`; on the
remaining 104 the detector got every item right, and both misses above are in `dev`.

**Explicit requests.** Two sets, kept apart:

| set | requests found (right target) | wrong target | near misses read as requests |
|---|---|---|---|
| `data/requests.jsonl` (dev: written with the patterns, includes a reviewer's counterexamples) | 60 / 60 | 0 | 0 / 40 |
| `data/requests_heldout.jsonl` (written before the code, run once) | **24 / 35** | 0 | **0 / 31** |

The held-out set was committed before the request code was rewritten (commit `b5d73e7`), the
code was frozen in the next commit (`366c733`), and the set was then run once
(`evals/requests_heldout_run.txt`). Precision held: no stop request, document request, other
person's preference, reported speech or "why did you switch" was read as a request. Recall did
not: 24 of 35 (0.686) is below the 0.85 floor. The misses are phrasings the frames do not cover,
for example "Can you talk to me in English, please?", "Would it be possible to continue in
Dutch?", "Do you happen to speak Arabic?", "Zou u Engels kunnen praten?", "Is Engels ook goed?",
"هل تجيد اللغة الهولندية؟" and "يا ريت نحكي عربي". They were not fixed: fixing them now would be
tuning on the held-out set. `make check` holds that set to its single-run result (recall at least
0.686, no near miss read as a request), so a change cannot quietly make it worse; a better
recall needs a new held-out set.

Read all of this with care. Every item was written by the author of the rules, as clean typed
text, minutes to hours before the code was run on it; transcripts from a real speech-to-text
engine (no punctuation, disfluencies, recognition errors) will score lower. The useful parts are
the gates and the lists of near misses.

## Measured: time to first token (Gemini Flash)

`fixtures/live-ttft.jsonl` holds 48 streamed calls to `gemini-3.6-flash` on Vertex AI (EU
location), made on 2026-11-08 from one machine, one call at a time: the same 24 caller questions
(8 English, 8 Dutch, 8 Arabic, `data/utterances.jsonl`) at thinking level `minimal` and at
`low`. The system prompt puts the model on the phone for a small fictional shop, forbids naming
real companies, places or websites, and tells it that it knows nothing about the shop's hours,
prices, stock, orders, bookings or policies. A "call" is one HTTP request: the SDK's automatic
retries were off, so a failure would have been recorded as a failure (none occurred). The first
call of each run also paid for connection setup; it is kept in the file and left out of the
percentiles.

| thinking | calls (warm) | TTFT p50 | TTFT p95 | full reply p50 | reply in the requested language | flagged in review |
|---|---|---|---|---|---|---|
| minimal | 24 (23) | 602 ms | 715 ms | 770 ms | 24 / 24 | 1 / 24 |
| low | 24 (23) | 3,342 ms | 4,445 ms | 3,558 ms | 24 / 24 | 0 / 24 |

- **Thinking level is the biggest latency lever here**: `minimal` cut median TTFT by about
  2.7 s, far more than speculation saves in the bench below.
- **Review.** Every reply was read against one written rubric (in
  `fixtures/live-ttft-review.json`): flag a reply that asserts anything about the shop the model
  cannot know (hours, prices, stock, order status, whether a booking, change or call-back is
  possible, what a policy allows), or that names a real company, place or website. One reply was
  flagged: with `minimal`, "dat is zeker mogelijk" ("that is certainly possible") to a request
  for a call-back. No reply named a real entity.
- "Reply in the requested language" uses this repo's own detector on the model's replies.
- 8 calls per language and level is a small sample from one network on one evening. An earlier
  recording the same evening, with a different prompt and questions, measured `low` at a 1,988 ms
  median; it was replaced because its replies named real places. Re-measure on your own network.

CI replays the recording: `make ttft-check` recomputes the summary from the fixture and the review
and fails if it no longer matches `fixtures/live-ttft.summary.json`. Re-recording (`make
record-ttft`) needs credentials, costs money, and is capped in code (`VOICE_LAB_MAX_LIVE_CALLS`,
default 0 = no live calls).

### Speculation in the bench

1,000 turns, seed 7, real policy and responder, LLM TTFT drawn from the 23 warm `minimal`
recordings. Endpointing, STT and TTS delays, and how often the last stable interim already has
the final words (80 %), are the simulator's assumptions (`harness.SimConfig`), not measurements:

| | first token p50 / p95 | first audio p50 / p95 | kept / restarted | wasted calls per turn |
|---|---|---|---|---|
| no speculation | 1,071.0 / 1,406.2 ms | 1,276.6 / 1,671.0 ms | 0 / 0 | 0.000 |
| speculation | 632.5 / 1,264.5 ms | 872.8 / 1,501.2 ms | 740 / 260 | 0.260 |

The gain is roughly the assumed endpointing delay. The price is one extra LLM call in every turn
where the final words differ from the interim, or where the reply language switches on that turn.
`voice-lab compare` reports both, so turning speculation on fails the default gate on wasted calls
even though it is faster: that trade-off is a decision, not a regression to wave through.

## LiveKit

`voice_lab.livekit_bridge.LiveKitBridge` attaches to a LiveKit Agents `AgentSession` through
`session.on(...)`. It collects the final transcript segments of a user turn
(`user_input_transcribed`), gives the language policy one decision per turn, and on a switch
calls `agent.update_instructions(...)`. Call `bridge.commit_turn()` from your agent's
`on_user_turn_completed` hook so a switch applies to the reply of that same turn; otherwise the
bridge commits when the agent starts thinking. `user_state_changed` and `agent_state_changed`
give per-turn timings (end of speech, last final transcript, thinking, first audio, end of
turn). A failed instruction update is logged and kept in `bridge.errors`.

It imports nothing from LiveKit. It was written against the livekit-agents 1.8.11 source, unit
tested with a fake session, and checked with LiveKit's own event classes
(`scripts/check_livekit_contract.py`). **It has not been run against a LiveKit server**, and
switching STT/TTS voices per language is left to the caller (`on_switch`).

## Limits

- **STT and TTS are simulated.** Only the LLM latency is measured. The speculation gain depends
  on your STT's endpointing delay and how often its final transcript has the same words as the
  last stable interim; measure both with your provider.
- **The detector corpus is small and written by the author of the rules** (see above).
- **Language detection is a heuristic for three languages.** French, Spanish, Portuguese,
  Italian and German sentences come out undetermined through a short function-word list; other
  Latin-script languages (Polish, for example, shares "a", "i" and "to" with English) can be
  misread as English or Dutch. Arabic is recognised by script whatever the dialect, and dialects
  are not told apart; Persian and Urdu are undetermined only when they use one of their own
  letters. Transliterated Arabic ("Arabizi"), names and very short turns are undetermined.
- **Explicit requests are patterns, not understanding.** They are tuned for precision: on the
  held-out set about a third of the requests were missed (see above), as are negated questions
  ("can't we just speak English?"). Only English, Dutch and Arabic can be requested; a request in
  a fourth language ("parlez-vous français?") is ignored.
- **Small TTFT sample** (8 calls per language and level, one network, one evening).
- **The LiveKit bridge is untested against a real session** (see above).
- **mypy is strict on the package only**; tests are linted, not type-checked.
- Synthetic sentences only; no caller data.

## License

Apache-2.0. See [LICENSE](LICENSE). Copyright 2026 Hossam Afifi.
