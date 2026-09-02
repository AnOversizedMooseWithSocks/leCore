"""MEMCURATE -- curation, decay and reflection over the memory partition (NOOA section 6, item 5).

docs/COMPETITIVE_NOOA.md rates leCore PARTIAL here: "`recall` exists; the curation/decay/
reflection subsystem does not", against NOOA's memory subsystem measured at +11.8 RHAE over the
same agent with markdown notes -- the strongest published evidence for any single item on that
list. This module is the missing subsystem, built from parts already in the engine.

WHAT WAS ALREADY HERE, and is REUSED rather than rebuilt:
  * holographic_actr -- ACT-R base-level activation A = ln(sum_j age_j^-d), the ladder's own
    power law, plus rank() with a retrieval threshold that ABSTAINS. That is the decay maths and
    it was already correct; nothing here reimplements it.
  * tiered_memory -- the short/long conductor with promotion and demotion. The shape of the
    answer (move between tiers) is its shape, applied to the taught partition instead of a
    key/value dict.

WHAT WAS MISSING, and it is not the maths. `holographic_actr.rank` takes {name: [use_times]} from
its CALLER, and the ladder records no access data at all -- no hit counts, no timestamps, no last
-used. Activation ranking existed as an instrument with nothing to point at. What this module adds
is the MISSING HALF: a use journal over the real partition, and a LOGICAL clock to timestamp it.

THE ARGUMENT THIS DESIGN HAD TO ANSWER, on record in unicron_turn_memory:
    "ACT-R eviction made that survivable by overwriting the least active slot, but EVICTION IS A
     LOSS, AND A FLAT FILE IS THE REASON IT WAS NEEDED."
That sweep found the eviction was an artefact of a flat layout, and replaced it with nested bases
-- four times the capacity at 100% recall, forgetting nothing. Walking into a "decay-based
forgetting" feature after that finding would be repeating the mistake the engine already paid to
learn. SO THERE IS NO DELETE PATH IN THIS MODULE. Curation DEMOTES to a restorable archive and
journals every action; `restore()` is exact; the default is plan-only and mutates nothing. A
curation pass that silently drops a fact a user taught is a data-loss bug, not a feature.

TIME IS LOGICAL, NEVER WALL-CLOCK. A decay that reads the system clock gives a different answer
on Tuesday, which makes every downstream decision irreproducible -- and this engine's
constitutional rule is that the same inputs give the same bytes. The clock here is an integer:
a row's birth is its INDEX in the taught log, and each observed use takes the next tick. Fact
identity is sha256 over the normalised question (hashlib, never hash()).

HONESTY ABOUT THE MEASUREMENT, stated here because it must not be lost downstream: there is no
widely adopted public benchmark that scores forgetting, decay or consolidation directly -- LoCoMo,
LongMemEval and BEAM score retrieval and reasoning, and only BEAM's "updating information over
time" comes close. So `curation_benchmark` below is SELF-MEASURED against baselines constructed
here, and the number is not comparable to Mem0's LoCoMo 92.5 / LongMemEval 94.4 / BEAM-1M 64.1.
The baselines are the honest ones: keep-everything (the unbounded upper bound), keep-newest (a
recency window), keep-most-used (frequency), and a no-information random subset.

MEASURED, 20 seeds, paired bootstrap 95% CI, hit rate of a 40-of-200 hot set on future queries:

    drift   keep_all  random  recency  frequency   ACTIVATION   winner
    0.00      1.0000  0.2000   0.6915     0.7437       0.7397   frequency
    0.25      1.0000  0.2000   0.6080     0.6584       0.6538   frequency
    0.50      1.0000  0.2000   0.5939     0.6246       0.6340   activation
    0.75      1.0000  0.2000   0.6020     0.6100       0.6399   activation
    1.00      1.0000  0.2000   0.6915     0.6224       0.6610   recency

AND THE RESULT IS MIXED, WHICH IS THE POINT OF REPORTING IT THIS WAY. Paired against the recency
window activation wins at drift 0.00-0.75 (+0.048/+0.046/+0.040/+0.038, every interval clear of
zero) and LOSES at drift 1.00 by -0.031 [-0.041, -0.020] -- under a complete topic switch the
history's frequency signal is worthless and the simplest possible baseline wins. Against frequency
it is a genuine TIE on a stationary process (the interval straddles zero at drift 0.00 and 0.25).
So: activation is the ROBUST MIDDLE -- the only policy never significantly worse than both -- and
it is not a dominant policy. On a stationary workload, count hits and skip this.
"""

import hashlib

import numpy as np

from holographic.agents_and_reasoning.holographic_actr import DECAY_D, base_level, rank

#: Policies the curator can act with. There is deliberately no "drop": see the module docstring.
POLICIES = ("archive", "plan")


def normalize_question(q):
    """The canonical form of a question -- lowercased, whitespace-collapsed.

    Identity has to survive re-typing, or a fact taught twice with different spacing becomes two
    facts and consolidation never fires. This is the same normalisation the ladder's own `_exact`
    key uses, deliberately, so a curated id and a reflex key agree about what "the same question"
    means."""
    return " ".join(str(q).lower().split())


def fact_id(question):
    """Content-addressed id for a taught row: sha256 of the normalised question, 16 hex chars.

    hashlib and never hash(): the id has to be identical across processes and PYTHONHASHSEED
    values, or an audit journal cannot be compared with the one from the run that produced it."""
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()[:16]


def rows_from_mind(mind):
    """The taught partition as a list of rows, read-only. Empty list when nothing was taught.

    Reads `zoo["ladder"].taught_log`, whose rows are [question, answer, session, provenance]. A
    row shorter than that is a pre-provenance record and still counts -- refusing to read old rows
    would make the curator useless on exactly the long-lived partitions it exists for."""
    lad = (getattr(mind, "zoo", {}) or {}).get("ladder")
    log = list(getattr(lad, "taught_log", []) or [])
    out = []
    for i, r in enumerate(log):
        r = list(r) + [None] * (4 - len(r))
        out.append({"born": i, "question": str(r[0]), "answer": str(r[1]),
                    "session": r[2], "provenance": r[3], "id": fact_id(r[0])})
    return out


class MemoryCurator:
    """Curation, decay and reflection over a taught memory partition.

    Three legs, and they are one object because they read one index:
      * DECAY      -- ACT-R activation per fact, from a use journal on a LOGICAL clock.
      * CURATION   -- a bounded hot set; everything else DEMOTED to a restorable archive.
      * REFLECTION -- consolidation of re-taught questions: the newest answer is current, the
                      superseded ones are archived with a pointer, never dropped.

    Nothing mutates until you ask. `plan()` is pure; `apply()` moves rows into the curator's
    archive; `apply(write_through=True)` is the only call that touches the mind's own log, and
    `restore()` undoes it exactly."""

    def __init__(self, mind=None, rows=None, d=DECAY_D):
        """Build a curator over a mind's taught partition, or over explicit `rows`.

        Rows may be given directly (a list of dicts, or of [q, a] pairs) so the curator can be
        tested and measured without booting a partition -- the measurement harness needs that and
        so does every regression trap."""
        self.mind = mind
        self.d = float(d)
        if rows is None:
            rows = rows_from_mind(mind) if mind is not None else []
        self.rows = [self._row(i, r) for i, r in enumerate(rows)]
        self.archive = {}
        self.journal = []
        # THE LOGICAL CLOCK. A row's birth is its index, so the tick after the last row is "now"
        # for a partition nobody has queried yet. No wall clock anywhere in this file.
        self.clock = len(self.rows)
        self.uses = {}
        for r in self.rows:
            # A RE-TEACH IS A USE, and this is the only access signal an uninstrumented partition
            # carries. It is real: re-establishing a fact is somebody reaching for it. Seeding
            # from it means the curator says something useful on a partition that was never
            # instrumented, which is the partition it will actually meet.
            self.uses.setdefault(r["id"], []).append(float(r["born"]))

    @staticmethod
    def _row(i, r):
        """Normalise one input row into the curator's record shape."""
        if isinstance(r, dict):
            out = dict(r)
            out.setdefault("born", i)
            out.setdefault("answer", "")
            out.setdefault("session", None)
            out.setdefault("provenance", None)
            out["question"] = str(out.get("question", ""))
            out["id"] = out.get("id") or fact_id(out["question"])
            return out
        q, a = (list(r) + [""])[:2]
        return {"born": i, "question": str(q), "answer": str(a), "session": None,
                "provenance": None, "id": fact_id(q)}

    # -- the use journal -------------------------------------------------------------------
    def observe(self, question, t=None):
        """Record that a fact was USED, at logical time `t` (default: the next tick).

        This is the half the engine did not have. Call it wherever a recall actually happens; the
        curator does not hook the ladder itself, because a faculty that silently instruments
        someone else's object is a faculty that breaks when that object changes."""
        fid = fact_id(question)
        t = self.clock if t is None else float(t)
        self.clock = max(self.clock, float(t)) + 1
        self.uses.setdefault(fid, []).append(float(t))
        return {"id": fid, "at": float(t), "uses": len(self.uses[fid])}

    def activation(self, now=None):
        """{fact_id: ACT-R activation} at logical time `now` (default: the current tick).

        Delegates to holographic_actr.base_level -- recency and frequency in ONE number, which is
        the whole reason this is worth doing rather than counting hits."""
        now = self.clock if now is None else float(now)
        live = {r["id"] for r in self.rows}
        return {k: base_level(v, now, d=self.d) for k, v in self.uses.items() if k in live}

    def ranked(self, now=None, threshold=None):
        """Facts ordered by activation, highest first; below `threshold` they are NOT retrieved.

        The threshold is ACT-R's own retrieval threshold and it is the same discipline as
        decide_or_abstain: a confident wrong memory costs more than a missing one."""
        now = self.clock if now is None else float(now)
        live = {r["id"] for r in self.rows}
        return rank({k: v for k, v in self.uses.items() if k in live}, now,
                    threshold=threshold, d=self.d)

    # -- the plan --------------------------------------------------------------------------
    def plan(self, keep=None, now=None, threshold=None, consolidate=True):
        """What curation WOULD do. Pure: reads everything, changes nothing.

        `keep` bounds the hot set (None = unbounded, so only consolidation and the threshold act).
        Returns {hot, archive, superseded, ranked, ...} where `archive` and `superseded` are the
        rows that would be DEMOTED -- never deleted, and each one restorable by id.

        WHY A PLAN OBJECT AND NOT A DIRECT APPLY: the sweep-125 lesson in a second costume. A
        curation pass you cannot inspect before it runs is one you find out about afterwards, and
        the thing it touched is the user's own taught facts."""
        now = self.clock if now is None else float(now)
        scored = dict(self.ranked(now=now))
        superseded = []
        if consolidate:
            # REFLECTION. The same question taught more than once is one fact with a history:
            # the NEWEST answer is current and the older rows are superseded. Ordered by birth so
            # the choice is deterministic, and the pointer is kept so the history is readable.
            newest = {}
            for r in self.rows:
                prev = newest.get(r["id"])
                if prev is None or r["born"] > prev["born"]:
                    newest[r["id"]] = r
            for r in self.rows:
                cur = newest[r["id"]]
                if r["born"] != cur["born"]:
                    superseded.append({"row": r, "superseded_by": cur["born"],
                                       "reason": "re-taught at t=%d" % cur["born"]})
        sup_born = {s["row"]["born"] for s in superseded}
        live = [r for r in self.rows if r["born"] not in sup_born]
        live.sort(key=lambda r: (-scored.get(r["id"], float("-inf")), r["id"]))
        cold = []
        if threshold is not None:
            below = [r for r in live if scored.get(r["id"], float("-inf")) < float(threshold)]
            cold += [{"row": r, "reason": "activation %.4f below threshold %.4f"
                      % (scored.get(r["id"], float("-inf")), float(threshold))} for r in below]
            live = [r for r in live if r not in below]
        if keep is not None and len(live) > int(keep):
            cold += [{"row": r, "reason": "outside the hot set of %d by activation" % int(keep)}
                     for r in live[int(keep):]]
            live = live[:int(keep)]
        return {"now": float(now), "keep": keep, "threshold": threshold,
                "hot": live, "archive": cold, "superseded": superseded,
                "counts": {"rows": len(self.rows), "hot": len(live), "archive": len(cold),
                           "superseded": len(superseded),
                           "distinct_facts": len({r["id"] for r in self.rows})},
                "ranked": [(k, round(v, 6)) for k, v in
                           sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))]}

    def apply(self, plan=None, write_through=False, **kw):
        """Carry out a plan by DEMOTING, never deleting. Returns the audit journal.

        Every demoted row lands in `self.archive` keyed by (id, born) with the reason it moved and
        the logical time it moved at; `restore()` puts it back byte for byte. `write_through=True`
        additionally rewrites the mind's taught log to the hot set -- the ONLY call in this module
        that touches the partition, off by default, and still fully reversible because the archive
        holds the originals.

        KEPT NEGATIVE, said out loud: write_through rewrites a list the ladder also caches keys
        for. The archive makes the ROWS recoverable; it does not roll back any derived index the
        ladder built from them. Restore, then re-teach, if you need the reflex keys back too."""
        plan = self.plan(**kw) if plan is None else plan
        moved = []
        for entry in plan["archive"] + plan["superseded"]:
            r = entry["row"]
            key = (r["id"], r["born"])
            self.archive[key] = dict(r)
            rec = {"id": r["id"], "born": r["born"], "question": r["question"],
                   "action": "superseded" if "superseded_by" in entry else "archived",
                   "reason": entry.get("reason"), "at": float(plan["now"]),
                   "restorable_as": list(key)}
            self.journal.append(rec)
            moved.append(rec)
        keep_born = {r["born"] for r in plan["hot"]}
        self.rows = [r for r in self.rows if r["born"] in keep_born]
        wrote = None
        if write_through and self.mind is not None:
            lad = (getattr(self.mind, "zoo", {}) or {}).get("ladder")
            if lad is not None:
                lad.taught_log = [[r["question"], r["answer"], r["session"], r["provenance"]]
                                  for r in sorted(self.rows, key=lambda r: r["born"])]
                wrote = len(lad.taught_log)
        return {"moved": len(moved), "hot": len(self.rows), "archived": len(self.archive),
                "wrote_through": wrote, "journal": moved}

    def restore(self, fact_id_=None, born=None, write_through=False):
        """Put archived rows back, exactly. The reason forgetting here is reversible.

        With no arguments it restores EVERYTHING -- the undo button for a curation pass that
        turned out to be wrong."""
        back = []
        for key in sorted(self.archive):
            fid, b = key
            if (fact_id_ is None or fid == fact_id_) and (born is None or b == born):
                back.append(self.archive.pop(key))
        self.rows = sorted(self.rows + back, key=lambda r: r["born"])
        for r in back:
            self.journal.append({"id": r["id"], "born": r["born"], "question": r["question"],
                                 "action": "restored", "reason": "restore()",
                                 "at": float(self.clock), "restorable_as": None})
        if write_through and self.mind is not None:
            lad = (getattr(self.mind, "zoo", {}) or {}).get("ladder")
            if lad is not None:
                lad.taught_log = [[r["question"], r["answer"], r["session"], r["provenance"]]
                                  for r in self.rows]
        return {"restored": len(back), "rows": len(self.rows), "archived": len(self.archive)}

    def reflect(self, now=None, k=5):
        """A readable digest of what the memory currently is: what is hot, what was superseded,
        what is archived and why.

        REFLECTION IN THIS ENGINE'S SENSE IS NOT AN LLM SUMMARY. It is the deterministic report
        that makes a curation decision auditable after the fact -- the same reason goal_close
        records the reason text rather than just closing the goal."""
        now = self.clock if now is None else float(now)
        scored = dict(self.ranked(now=now))
        byid = {}
        for r in self.rows:
            byid.setdefault(r["id"], r)
        top = sorted(byid.values(), key=lambda r: (-scored.get(r["id"], float("-inf")), r["id"]))
        actions = {}
        for j in self.journal:
            actions[j["action"]] = actions.get(j["action"], 0) + 1
        return {"now": float(now), "rows": len(self.rows), "archived": len(self.archive),
                "distinct_facts": len(byid), "actions": actions,
                "top": [{"question": r["question"], "activation": round(scored.get(r["id"], 0.0), 6),
                         "uses": len(self.uses.get(r["id"], ()))} for r in top[:int(k)]],
                "coldest": [{"question": r["question"],
                             "activation": round(scored.get(r["id"], 0.0), 6)}
                            for r in top[-int(k):][::-1]],
                "journal_tail": self.journal[-int(k):]}


# ---------------------------------------------------------------------------------------
# THE MEASUREMENT. Self-measured on purpose -- see the module docstring.
# ---------------------------------------------------------------------------------------

def _stream(n_facts, n_queries, seed, zipf, drift, shift_at=0.6):
    """A deterministic access stream whose TOPIC SET shifts partway through the history.

    `drift` is the fraction of PROBABILITY MASS that moves onto previously-cold facts after the
    shift -- not a fraction of rank slots. That distinction is the whole harness:

    KEPT NEGATIVE, and the first version of this function was the bug. Drift was a fraction of
    RANK SLOTS, and under Zipf(1.1) the top 25 of 200 ranks carry 71% of the traffic -- so
    "drift=0.25" silently moved 71% of all future queries onto the facts that had been COLDEST in
    the history. History was not merely uninformative, it was ANTI-predictive, and every policy
    scored BELOW the 0.20 random-subset baseline (0.086-0.13). Three policies all losing to chance
    is not a result about the policies, it is a broken generator, and it read as a finding for one
    whole measurement cycle. The diagnostic that caught it was comparing against random, which is
    why the random baseline is now reported in every row.

    THE SHIFT HAPPENS DURING THE HISTORY (at `shift_at` of it), not after it. That is the honest
    setting for curation: a curator runs continuously, so the recent past has already seen the
    beginning of the new regime. Putting the shift after the history instead would leave NO policy
    able to know about the new topics, which measures nothing about curation."""
    rng = np.random.default_rng(int(seed))
    w = 1.0 / np.arange(1, n_facts + 1) ** float(zipf)
    w = w / w.sum()
    order_old = rng.permutation(n_facts)
    # the NEW regime's popular head is drawn from what was cold before, so the two regimes share
    # as little mass as the drift says and no more
    order_new = np.concatenate([order_old[::-1][: n_facts // 2], order_old[: n_facts - n_facts // 2]])

    def draw(n, mix):
        """n facts from (1-mix) * the old popularity and mix * the new one."""
        pick_new = rng.random(n) < float(mix)
        ranks = rng.choice(n_facts, size=n, p=w)
        return [int(order_new[r] if nw else order_old[r]) for r, nw in zip(ranks, pick_new)]

    half = n_queries // 2
    cut = int(round(float(shift_at) * half))
    hist = draw(cut, 0.0) + draw(half - cut, drift)
    fut = draw(n_queries - half, drift)
    return hist, fut


def curation_benchmark(n_facts=200, n_queries=800, hot=40, seed=0, zipf=1.1, drift=0.5,
                       shift_at=0.6, d=DECAY_D):
    """SELF-MEASURED curation quality: of a bounded hot set of `hot` facts chosen from a history
    of uses, what fraction of FUTURE queries land in it?

    NOT COMPARABLE TO A PUBLIC NUMBER, and this is the honest caveat that has to travel with the
    result: no widely adopted benchmark scores forgetting, decay or consolidation directly, so
    Mem0's LoCoMo 92.5 / LongMemEval 94.4 / BEAM-1M 64.1 measure something else. The baselines
    here are constructed in this file:
      * keep_all   -- no bound at all. The upper bound, and the measure of what the bound costs.
      * recency    -- the last `hot` DISTINCT facts used. The baseline that matters, because it is
                      what everyone writes first and it is nearly free.
      * frequency  -- the `hot` most-used facts. Pure counting, no decay.
      * actr       -- ACT-R activation, recency and frequency in one number.
      * random     -- hot/n_facts, the expected hit rate of a subset chosen with NO information.
                      Reported in every row because it is the diagnostic that caught this
                      harness's own generator bug: three history-informed policies all scoring
                      BELOW chance is a broken stream, not a result about the policies.
    Returns a row per policy with `hit_rate` on the future stream."""
    hist, fut = _stream(int(n_facts), int(n_queries), seed, zipf, drift, shift_at=shift_at)
    uses = {}
    for t, f in enumerate(hist):
        uses.setdefault(f, []).append(float(t))
    now = float(len(hist))
    hot = int(hot)
    last_seen = {f: max(ts) for f, ts in uses.items()}

    def _fill(ordered):
        """Every bounded policy spends the SAME budget, padding with facts it has no information
        about (lowest id first). Otherwise a policy that names fewer facts than the budget looks
        good for having declined to guess, which is not the property under test."""
        kept = list(dict.fromkeys(ordered))[:hot]
        for f in range(n_facts):
            if len(kept) >= hot:
                break
            if f not in kept:
                kept.append(f)
        return set(kept)

    pol = {
        # the partition holds EVERY fact -- keep_all is the unbounded upper bound, and its gap to
        # the rest is the price of the bound. (The first version scored only facts seen in the
        # history, which made the "upper bound" 0.78 and not an upper bound at all.)
        "keep_all": set(range(n_facts)),
        "recency": _fill(sorted(last_seen, key=lambda f: (-last_seen[f], f))),
        "frequency": _fill(sorted(uses, key=lambda f: (-len(uses[f]), f))),
        "actr": _fill([f for f, _a in rank({f: ts for f, ts in uses.items()}, now, d=d)]),
    }
    out = []
    for name in ("keep_all", "recency", "frequency", "actr"):
        kept = pol[name]
        hits = sum(1 for f in fut if f in kept)
        out.append({"policy": name, "kept": len(kept), "hits": hits, "queries": len(fut),
                    "hit_rate": round(hits / len(fut), 4)})
    out.append({"policy": "random", "kept": hot, "hits": None, "queries": len(fut),
                "hit_rate": round(hot / float(n_facts), 4)})
    bounded = [r for r in out if r["policy"] not in ("keep_all", "random")]
    best = max(r["hit_rate"] for r in bounded)
    winners = sorted(r["policy"] for r in bounded if r["hit_rate"] == best)
    return {"params": {"n_facts": n_facts, "n_queries": n_queries, "hot": hot, "seed": seed,
                       "zipf": zipf, "drift": drift},
            "rows": out, "best_bounded": winners,
            "actr_beats_recency": bool(
                [r for r in out if r["policy"] == "actr"][0]["hit_rate"] >
                [r for r in out if r["policy"] == "recency"][0]["hit_rate"]),
            "self_measured": True,
            "note": ("self-measured; no public benchmark scores forgetting/decay/consolidation "
                     "directly, so this is not comparable to LoCoMo/LongMemEval/BEAM numbers")}


def _bootstrap_ci(deltas, seed=0, n=4000, alpha=0.05):
    """Percentile bootstrap CI for a mean, with a SEEDED generator so the interval is reproducible.

    Reported because "actr won" over five seeds is not a result until the interval is shown: on
    this harness two of the three activation wins turned out to straddle zero, and a table without
    an interval would have read as three wins."""
    a = np.asarray(deltas, float)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(a), size=(int(n), len(a)))
    means = a[idx].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean()), float(lo), float(hi)


def drift_sweep(drifts=(0.0, 0.25, 0.5, 0.75, 1.0), seeds=tuple(range(20)), **kw):
    """The honest presentation: hit rate per policy ACROSS the drift range, with PAIRED deltas.

    A single drift value can be chosen to make any policy look good; the sweep stops that. The
    paired bootstrap stops the second mistake, which is reading a 0.003 mean difference over five
    seeds as a win. Deltas are paired per seed because the seed IS the stream -- comparing
    unpaired means throws away the only variance reduction available here.

    THIS IS WHERE THE NEGATIVE LIVES. Activation does not dominate: frequency wins the stationary
    end, recency wins the full-switch end, and in between the activation margin is small enough
    that its interval has to be read before the claim is made."""
    rows = []
    for dr in drifts:
        acc = {}
        for s in seeds:
            for r in curation_benchmark(seed=s, drift=dr, **kw)["rows"]:
                acc.setdefault(r["policy"], []).append(r["hit_rate"])
        row = {"drift": dr, "seeds": len(seeds)}
        row.update({p: round(float(np.mean(v)), 4) for p, v in acc.items()})
        row["sd_actr"] = round(float(np.std(acc["actr"], ddof=1)), 4)
        for other in ("recency", "frequency"):
            d = np.asarray(acc["actr"]) - np.asarray(acc[other])
            m, lo, hi = _bootstrap_ci(d, seed=int(dr * 1000))
            row["actr_minus_" + other] = round(m, 4)
            row["ci_" + other] = [round(lo, 4), round(hi, 4)]
            # a CI that straddles zero is NOT a win, and the field is named so nobody has to
            # squint at the interval to find that out
            row["beats_" + other] = bool(lo > 0.0)
        row["winner"] = max((p for p in acc if p not in ("keep_all", "random")),
                            key=lambda p: np.mean(acc[p]))
        rows.append(row)
    sig = [r["drift"] for r in rows if r["beats_recency"]]
    return {"rows": rows, "seeds": list(seeds), "self_measured": True,
            "drifts_where_actr_significantly_beats_recency": sig,
            "verdict": ("activation beats a recency window at %s and nowhere else" % (sig,)
                        if sig else
                        "activation does NOT significantly beat a recency window at any drift"),
            "note": ("self-measured; no public benchmark scores forgetting/decay/consolidation "
                     "directly, so this is not comparable to LoCoMo/LongMemEval/BEAM numbers")}

def _selftest():
    """Regression trap for item F. Every assertion is a NUMBER or an exact structural contract --
    including THE NEGATIVE, which is pinned as hard as the wins: activation LOSES to a recency
    window under a complete topic switch, and a future sweep that quietly makes that pass would
    be hiding the one result this module is most tempted to overstate."""
    import json

    # ---- 1. THE PARTITION IS READ, CONSOLIDATED AND RANKED -------------------------------
    rows = [("what is the capital of france", "paris"),
            ("who wrote dune", "frank herbert"),
            ("what is the boiling point of water", "100 C"),
            ("what is the capital of france", "paris, france")]   # a RE-TEACH: same fact
    c = MemoryCurator(rows=rows)
    assert len(c.rows) == 4 and len({r["id"] for r in c.rows}) == 3, "the re-teach is one fact"
    assert c.clock == 4, "the logical clock starts after the last row's birth index"
    c.observe("who wrote dune")
    c.observe("who wrote dune")
    c.observe("what is the capital of france")
    p = c.plan(keep=2)
    assert p["counts"] == {"rows": 4, "hot": 2, "archive": 1, "superseded": 1,
                           "distinct_facts": 3}, p["counts"]
    assert p["superseded"][0]["row"]["born"] == 0, "the OLDER row is the superseded one"
    assert p["superseded"][0]["superseded_by"] == 3
    assert [r["question"] for r in p["hot"]] == ["what is the capital of france",
                                                 "who wrote dune"]
    assert p["archive"][0]["row"]["question"] == "what is the boiling point of water"

    # ---- 2. PLAN IS PURE. Reading the plan must not move a single row --------------------
    assert len(c.rows) == 4 and c.archive == {}, "plan() mutated the store"
    assert json.dumps(c.plan(keep=2), default=str) == json.dumps(p, default=str), "not deterministic"

    # ---- 3. NOTHING IS EVER DESTROYED. This is the answer to unicron_turn_memory's kept
    #         negative -- "eviction is a loss" -- and it is a conservation law, so it is
    #         asserted as one: rows_after + archived == rows_before, always.
    before = [dict(r) for r in c.rows]
    j = c.apply(p)
    assert j["moved"] == 2 and j["hot"] == 2 and j["archived"] == 2
    assert len(c.rows) + len(c.archive) == len(before), "a row went missing: that is data loss"
    assert all(x["action"] in ("archived", "superseded") for x in j["journal"])
    assert all(x["reason"] for x in j["journal"]), "a silent demotion is worse than a loud one"

    # ---- 4. RESTORE IS EXACT, or 'reversible' is a word rather than a property -----------
    r = c.restore()
    assert r["restored"] == 2 and r["archived"] == 0
    assert [dict(x) for x in c.rows] == before, "restore() did not return the store byte for byte"

    # ---- 5. IDENTITY IS CONTENT-ADDRESSED AND PROCESS-STABLE (hashlib, never hash()) -----
    assert fact_id("What Is  The Capital Of FRANCE") == fact_id("what is the capital of france")
    assert fact_id("a") == "ca978112ca1bbdca", fact_id("a")   # sha256('a')[:16], fixed forever

    # ---- 6. THE MEASUREMENT. Self-measured; see the module docstring for why there is no
    #         public number to compare against. 20 seeds, paired bootstrap 95% CI.
    sw = drift_sweep()
    by = {r["drift"]: r for r in sw["rows"]}
    assert set(by) == {0.0, 0.25, 0.5, 0.75, 1.0}
    for d_, row in by.items():
        assert row["keep_all"] == 1.0, "keep-everything must be the unbounded upper bound"
        assert row["random"] == 0.2, "40 hot of 200 facts is a 0.2 no-information baseline"
        # THE DIAGNOSTIC THAT CAUGHT THE GENERATOR BUG: every history-informed policy must beat
        # chance. The first stream made all three score 0.086-0.13 and it read as a finding.
        for pol in ("recency", "frequency", "actr"):
            assert row[pol] > 0.55, ("%s at drift %s scored %.4f -- below the 0.2 random "
                                     "baseline means the STREAM is broken, not the policy"
                                     % (pol, d_, row[pol]))
    # the wins, with their intervals
    assert by[0.5]["beats_recency"] and by[0.75]["beats_recency"]
    assert by[0.5]["actr_minus_recency"] > 0.03 and by[0.75]["actr_minus_recency"] > 0.03
    # THE NEGATIVE, PINNED: under a COMPLETE topic switch a plain recency window WINS, and the
    # interval is entirely below zero. Activation is the robust middle, not the dominant policy.
    assert by[1.0]["beats_recency"] is False, "the drift=1.0 loss to recency has been hidden"
    assert by[1.0]["actr_minus_recency"] < -0.02
    assert by[1.0]["ci_recency"][1] < 0.0, "the loss must be significant, not a wobble"
    assert by[1.0]["winner"] == "recency"
    # and against FREQUENCY it is a genuine tie on a stationary process -- also not a win
    assert by[0.0]["beats_frequency"] is False and by[0.25]["beats_frequency"] is False
    assert by[0.0]["ci_frequency"][0] < 0.0 < by[0.0]["ci_frequency"][1], "a tie, not a loss"
    assert by[0.75]["beats_frequency"] and by[1.0]["beats_frequency"]
    assert sw["self_measured"] is True and "LoCoMo" in sw["note"]

    print("OK: holographic_memcurate self-test passed -- %d rows curate to %d hot with %d "
          "archived and NOTHING deleted (restore() is exact); ids are sha256-stable; and the "
          "measurement is honest in both directions: activation beats a recency window by "
          "%+.4f at drift 0.50 %s and LOSES to it by %+.4f at drift 1.00 %s, while tying "
          "frequency on a stationary process %s"
          % (len(before), j["hot"], j["archived"],
             by[0.5]["actr_minus_recency"], by[0.5]["ci_recency"],
             by[1.0]["actr_minus_recency"], by[1.0]["ci_recency"],
             by[0.0]["ci_frequency"]))


if __name__ == "__main__":
    _selftest()
