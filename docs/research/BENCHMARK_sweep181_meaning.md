# Sweep 181 — finding by meaning, learning from the model end, and learning HOW an answer was found

Owner direction (Moose, 2026-09-24): *"if an LLM is attached to the back side of leCore, it should be able to
resolve unknown or new phrasing of things, which should then allow us to learn on the go … Learning the meaning
behind not just the question and the answer, but also how we find the answer is probably the most important
thing."*

Harness: `tools/bench_meaning.py`. Code: `holographic/agents_and_reasoning/holographic_meaning.py` (the index),
`holographic/unified/holographic_unified_p28_meaning.py` (the wiring), the ladder's T0m/T4m rungs in
`holographic_zoo.py`. Tests: `tests/test_meaning.py`.

## Data: real human wording, not templates

- **CLINC150** (Larson et al. 2019, CC BY 3.0). It has 150 intents, 100 crowd-written training phrasings per
  intent, and 30 test phrasings per intent (4,500 total). It also has 1,000 out-of-scope test questions that
  match no intent.
- **Banking77** (Casanueva et al. 2020, CC BY 4.0). It has 77 fine-grained banking intents with deliberately
  confusable neighbours, and 3,080 test phrasings. CLINC's out-of-scope set stands in as its off-domain set.
- **Crypto price questions**, hand-written for this sweep (`tests/data/meaning_crypto.json`). There are 40
  price questions, written both the way people type in chat and the way LLM agents phrase tool requests. There
  are also 12 questions that are a coin and nothing else ("SOL", "eth?", "$BTC"): these must be clarified.
- **CLINC150's `weather` and `exchange_rate` phrasings.** These are real people asking live questions, used to
  learn methods.

Neither public dataset ships in the repo. Pass their folder with `--data`.

## The model end is a scripted stand-in: read these numbers as that

No model is reachable from the sandbox this was built in. The stand-in reads the typed prompt exactly as a model
would (the `QUESTION` line and the `CANDIDATES` JSON) and answers from the dataset's labels:

- If a candidate has the right intent, it answers `same`.
- Otherwise it answers `new`.
- `noisy:r` gives a **wrong** verdict with probability r.

This measures the **learning loop** and its robustness to a fallible teacher. It says nothing about any real
model's judgement.

## 1. Before: the ladder alone (one taught wording per intent, held-out wordings)

| | served correct | served WRONG | refused |
|---|---|---|---|
| CLINC150 before (reflex only) | 6 / 4,500 (0.1%) | 0 | 99.9% |
| Banking77 before | 1 / 3,080 (0.0%) | 0 | 100% |
| CLINC150 after, **no model**, cold start | 68 (1.5%) | 2 (0.0%) | 98.4% |
| Banking77 after, **no model**, cold start | 35 (1.1%) | 2 (0.1%) | 98.8% |

Out-of-scope questions served: 0.1% on CLINC and 0.0% on Banking.

Cold start is deliberately near-exact. The prior gate (score ≥ 0.80, lead ≥ 0.15) was chosen on CLINC150's
**validation** split for precision ≥ 0.97. The first guess (0.45 / 0.10) served 6.9% WRONG on Banking77. With
one wording per intent, a pure-lexical matcher cannot tell "my card hasn't arrived" from "when will my card be
delivered". That is the model's job, and memory learns from it.

## 2. Learning on the go: the model end attached (default gate)

The setup:

1. Teach one wording per intent.
2. Stream every other training phrasing through `ask()` with the stand-in attached. That is 14,950 CLINC
   questions, including the 100 out-of-scope training questions, or 9,926 Banking questions.
3. Detach the model and ask the test set.

The per-1,000 figures are counted during the stream.

| dataset / teacher | model calls per 1,000, start → end | memory WRONG per 1,000 | after, model detached: correct / WRONG | out-of-scope served |
|---|---|---|---|---|
| CLINC150 / perfect | 898 → 458 | 1 – 15 | **48.3% / 1.5%** (precision 0.971) | 2.0% |
| Banking77 / perfect | 832 → 527 | 8 – 43 | **37.2% / 2.1%** (precision 0.947) | 1.0% |
| CLINC150 / 10% wrong verdicts | 954 → 853 | 0 – 5 | 9.8% / 0.2% | 0.1% |

What memory learned on CLINC150 with the perfect teacher:

- 592 rows (150 taught; the rest are the out-of-scope and duplicate-intent rows the teacher created with `new`).
- 7,466 wordings.
- 58,670 learned word associations.
- 1,024 calibration labels, and the gate calibrated.

### The trade-off, kept loud: the serve bar

The gate serves when the calibrated P(the model would agree) is at or above the bar.

An optional **teacher-relative** bar lowers it to P_SERVE × (the agreement ceiling on the most confident fifth
of verdicts), floored at `P_FLOOR`. Its purpose is so that a noisy teacher cannot hold memory to a standard the
teacher itself cannot meet.

It is **off by default** (`P_FLOOR = P_SERVE = 0.95`). The ceiling cannot tell a noisy teacher from confusable
rows:

| P_FLOOR | CLINC / perfect | Banking / perfect | CLINC / 10% noise | CLINC / 20% noise |
|---|---|---|---|---|
| 0.95 (default) | 48.3% / 1.5% | 37.2% / 2.1% | 9.8% / 0.2% | not run |
| 0.90 | 60.4% / 3.4% | 59.7% / **6.8%** | 39.8% / 1.3% | 5.1% / 0.2% |
| 0.75 | 61.0% / 3.4% | 64.6% / **8.6%** | 63.8% / 5.4% | 50.8% / 2.8% |

Each cell is correct / WRONG after the stream, with the model detached. The 0.75 noisy runs predate the near-miss guard (section 4). Raw outputs: `docs/research/evidence/bench_meaning.json`.

**The negative.** With a noisy teacher and the safe default, the gate barely opens: memory keeps escalating,
which is safe and costs model calls. Lowering the floor buys coverage and pays in wrong serves on confusable
domains. The knob is per mind (`mind.meaning.P_FLOOR`).

## 3. Learning HOW: live questions become methods, never cached values

Setup:

- 148 static CLINC intents are taught as rivals.
- 959 questions are streamed through with a stand-in that reports **how** it answered: verb, args, and the
  words in the question that gave each arg.
- Then the held-out questions are asked with the model detached (default gate; 691 of the 959 stream questions went to the model).

Every answer is checked against a tool whose output changes on every call, so a cached value would be caught
as stale.

| domain (held-out) | right call | WRONG call | escalated | stale answers | bare value → clarified |
|---|---|---|---|---|---|
| crypto price (hand-written, 20 + 6 bare coins) | 11 | 0 | 9 | **0** | **6 / 6** (0 acted on) |
| exchange rate (CLINC150, 26) | 16 | 1 | 9 | **0** | — |
| weather (CLINC150, 30) | 9 | 0 | 21 | **0** | — |

What the methods learned:

- Slot words map to tool values: "solana" → SOL, "yen" → JPY. These are pooled per slot name across methods.
- A question that gives more than the learned call uses ("20 yen" when the method has no amount) is escalated.
  The method then grows an optional slot.
- An argument that comes from the question only sometimes ("weather" versus "weather in tampa") gets a default.

**The negative.** The one WRONG exchange call is direction. Slots of the same kind (from/to currency) are filled
in the order the method's first example named them. Direction words ("to", "for", "in") are not parsed.

**Correction, 2026-09-26 (found by the CLM panel swarm, w2-hd; kept loud).** The exchange-rate row above does NOT
measure direction. The stand-in teacher's truth (`tools/bench_meaning.py` `method_truth`) is itself POSITIONAL
(the first currency named = FROM), which is semantically wrong on 29 of 74 directional training items and 5 of 17
directional test items (all 34 checked by eye). Scoring checks the currency SET and the amount, so "16 right" means
the right currencies and amount, not the right direction. A direction-labelled set is backlog item E0.1.

**Update, sweep 182 (2026-09-26): direction is now MEASURED, on a labelled set.** `tests/data/exchange_direction.json`
holds 179 directional exchange items whose FROM / TO were labelled by the words that name them (CLINC150 phrasings +
hand-written; 54 more skipped with reasons) plus symmetric sentences ("the rate between X and Y"), and the methods
mode now scores a directional item right only when FROM **and** TO are right. The old "16 right" is superseded, not
re-counted. What the labelled set says (`docs/research/BENCHMARK_sweep182_contrastive.md`, E3.1):

- the context-role direction reader, held out: **49/51 vs 35/51** for the positional rule (`tools/bench_rolecall.py`,
  `docs/research/evidence/bench_rolecall.json`);
- end to end through `ask()`, the same learned memory asked twice: **27 right / 2 wrong direction** with the reader vs
  **18 / 11** positional (21 escalated either way); a sentence with no direction is left open 13 of 14 times instead
  of guessed (`tools/bench_meaning.py direction` -> `bench_meaning_direction.json`);
- the methods mode re-run on the labelled items (29 exchange questions): 11 right call / 3 wrong / 15 escalated with
  the reader vs 15 / 5 / 9 positional on the same memory, and 0 symmetric sentences acted on vs 6 -- fewer right
  calls, fewer wrong ones, more escalations (`bench_meaning_methods.json`). KEPT LOUD: on this mode the reader buys
  safety, not coverage.

Found on the way: "20,000" was read as TWO numbers ("20", "000"), so a model's amount 20000 matched neither and the
learned fx method froze 20000 as a constant; `numbers_in` now reads thousands separators.

## 4. Near-miss safety (an external benchmark's probe)

`tests/test_external_abstention.py` asks questions one word away from a taught fact, such as "what colour is my
**sister's** bike" against a taught "what colour is my bike". The first cut of the meaning rung served one of the
four (score 0.80 against the 0.80 gate).

The fix: a rare word of the question that belongs to **another** row, is absent from this row's wordings, and is
not tied to them by a learned association blocks the serve. The probe is back to 4 of 4 declined.

## 5. Bugs found on the way

- **Negation was a stopword.** "that isn't right" was served the answer to "that is right" at high confidence.
  Contractions now expand to an explicit "not".
- **Numbers.** "20 dollars" and "50 dollars" were different questions to the matcher. A number is now a value
  (`<num>`) on this rung. The reflex keeps digits as identifiers, on purpose.
- **Credentials typed in a sentence.** "my password is Hunter2-…" passed the learning guard, which only matched
  `password: …` / `password=…`. It is now refused (`_DISCLOSE`), without flagging "my password is too short".
- **The workspace revision read the whole `.lews` on every leStudio request.** This was found while painting in
  the same session: 3.2 s per paint stroke became 0.012 s.

## 6. What is not solved

- **Lexical ceiling.** With many wordings per intent, top-1 accuracy is ~0.83–0.84 on CLINC150's test split
  (0.832 on the index saved by the first oracle run; 0.8413 on `final/default/idx_clinc.json`, the default-gate
  run -- the swarm re-measured it). Fine-tuned encoders reach ~0.95. The model end covers the gap, one verdict at a time.
- **The answer key was not persisted (fixed 2026-09-26).** `MeaningIndex.state()` omitted `akey`, so after a
  reload every row became its own answer: the saved CLINC index had 602 rows and 602 keys, and 8.2% of test
  runner-ups were the SAME intent -- a non-rival shrinking the lead. Fixed in `state()`/`from_state()`, with a
  save/reload test. The online numbers above were measured in one process and are unaffected.
- **Duplicate rows.** When the right row is not among the eight candidates shown to the model, a perfect teacher
  says `new`, and the intent gets a second row. Rows with the same answer are merged for ranking (the answer
  key), but they still exist.
- **Unknown values.** An unknown city ("weather in boise") binds a weather method's default location. Only known
  slot words are checked for being left unused.
