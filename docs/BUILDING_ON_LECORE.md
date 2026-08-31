# Building on leCore

Everything here comes from a real integration — leStudio, a layer-and-node image editor on
this engine, whose author wrote down every seam they had to paper over. The first thing
they needed did not exist: **a substrate scoped to one app and one user.**

## Start here

```python
from holographic.agents_and_reasoning.holographic_appkit import App
app = App("lestudio", user="ana", root="~/.lecore")     # or mind.app_substrate(...)

app.remember("what brush for skin", "soft round at 12% flow")
app.recall("what brush for skin")        # {answer, provenance: "taught", tier}
app.observe("retouch a portrait",        # a sequence that WORKED
            ["duplicate layer", "frequency separation", "soft round 12%", "curves lift"])
app.suggest("retouch a portrait")        # what this user did last time
app.habits()                             # subsequences they repeat, with support counts
app.save()                               # on quit
```

## Isolation is physical, not polite

Each `(app, user)` is its **own partition directory**. One user's memory cannot appear in
another's by any code path — not a similarity hit, not a shared fallback, not a bug in a
salt. A salt is a convention; a directory is a fact. Two people on one laptop, two apps on
one machine, and two tenants of a hosted service are all the same case.

The hosted service takes the same argument: `zoo_ask` / `zoo_teach` accept **`user`**, which
routes to `<root>/users/<id>`. Users of a public service have very different goals, so
their learning has to be theirs — otherwise one person's preference is served to everybody
and every self-improvement curve is averaged into a blur.

## Why apps learn procedures better than chat does

`plan_warm` keys on the goal's content words, so prose paraphrase misses: *"roll back a bad
deploy"* warms, *"what to do when a deployment goes bad"* does not. An app's goals are
**structured** — tool names, node kinds, menu paths — and structured goals repeat verbatim.
The weakest part of the chain path for conversation is the strongest part for an app.

Measured on a two-user leStudio simulation: after three retouching sessions, Ana's mined
habit was `duplicate layer → frequency separation` at **support 3**; Bo, doing motion
graphics in the same app, mined `cosine palette → shader pipeline → export png` at support
2. Nobody wrote a rule. The app noticed.

## What you get for free

* **Provenance on every answer** — `taught` (the user established it) vs `model-cached`
  (a guess). Pass `established_only=True` to be told to ask rather than be handed a guess
  dressed as the user's own preference.
* **A veto** — `forget(question)` stops an answer serving in the same breath. The audit
  record keeps what happened; leCore does not rewrite history.
* **A capability preflight** — `app.capabilities([...])` reports what *this* build has, so a
  missing faculty is a disabled button rather than a traceback. leStudio hand-rolled this;
  now it ships.
* **Restart survival** — `save()` / automatic load, one container per user.

## Growing with the user, concretely

1. Log what worked: `observe(goal, steps)` after any multi-step action the user completed.
2. Offer it back: `suggest(goal)` on the next visit to the same goal.
3. Notice style: `habits()` for subsequences repeated across *different* goals — promote one
   to a default only on your own evidence gate, never automatically.
4. Ask before assuming: `recall(q, established_only=True)` when the answer would change what
   the app does on the user's behalf.

## Honest limits

* `suggest` warms on near-exact goals. Structured goal strings (stable tool names) are the
  fix; free-text goals will miss on paraphrase until a paraphrase-robust goal key lands.
* `habits()` returns **proposals with support counts**, not decisions. Two occurrences is
  evidence, not a mandate.
* Nothing here calls a model. If you attach one (`llm=`), it is the last rung, and anything
  it answers is marked `model-cached` until a human establishes it.
