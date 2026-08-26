"""LEVERS -- the six moves that turn a measured wall into a boundary you can cross.

WHY THIS MODULE EXISTS AND WHY IT IS NOT A DOCUMENT. The six levers are the most
reused idea in this engine and they lived only as PRACTICE: named in NOTES
entries, applied correctly a hundred times by whoever had read them, and
findable by nobody else. Asked five ways a stranger would ask -- "what do I do
when I hit a wall", "ways to beat a capacity limit", "the six levers", "I am
blocked, what are my options" -- find_capability returned advise_scale,
crystal_habit and time_of_impact. THE MOST GENERALISABLE THING IN THE ENGINE WAS
THE LEAST DISCOVERABLE.

An LLM driving leCore has exactly the problem the levers solve and no way to
learn them: it hits a limit, concludes "this is impossible", and stops. That is
the difference between a tool an agent gives up on and one it works around.

THE DOCTRINE, in one line: A MEASURED LIMIT IS A COMPOSABILITY BOUNDARY, NOT A
WALL -- and the levers are ordered by cost, so you walk them in order and stop
at the first that applies.

EACH LEVER CARRIES ITS OWN EVIDENCE. Every entry below names a measurement from
this repo, because a lever recommended without a case where it worked is advice,
and advice is what this project replaces with numbers.
"""

LEVERS = (
    {
        "n": 1,
        "name": "cache locality -- bake once, sample O(1)",
        "when": "the same expensive value is recomputed per query",
        "do": "precompute into a table or field, then sample it; pay once, "
              "read forever",
        "evidence": "the prefix cache: a full 400-token recompute is 0.1257 s "
                    "and one resumed step is 0.0021 s -- 61x, and it is the "
                    "dominant win in generation",
        "costs": "memory for the bake, and staleness if the input moves",
    },
    {
        "n": 2,
        "name": "partition into a commutative monoid",
        "when": "the work is too big for one pass but the combine is "
                "associative and order-free",
        "do": "split, compute independently, merge with `distribute` -- the "
              "merge must not care about order or grouping",
        "evidence": "bundling IS a commutative monoid, which is why superposed "
                    "memory partitions at all; the same shape is what lets a "
                    "tiled reduce match a single-pass one exactly",
        "costs": "nothing, IF the combine is genuinely commutative -- and a "
                 "combine that ALMOST is will pass a small test and diverge "
                 "at scale",
    },
    {
        "n": 3,
        "name": "determinism instead of storage -- regenerate from seeds",
        "when": "you are storing something you could recompute exactly",
        "do": "keep the seed and the rule, not the output",
        "evidence": "registers regenerate from a seed rather than shipping "
                    "keys (16/16 and 128/128 recalled from disk); KV is "
                    "bit-identical on re-prefill, so 819 KB of cache is a memo "
                    "of work whose INPUT costs 3.2 KB",
        "costs": "CPU at read time, and an absolute dependence on determinism "
                 "-- one salted hash() and the whole lever is a corruption bug",
    },
    {
        "n": 4,
        "name": "more dimensions -- extra roles, accumulators, or a lift",
        "when": "two things are interfering, or the problem is non-linear "
                "where you are standing",
        "do": "add a role to bind against, an accumulator to separate the "
              "streams, or lift into a space where the problem is linear",
        "evidence": "a data-dependent BRANCH cannot fuse into one operator -- "
                    "install BOTH arms and gate the output, and 128/128 "
                    "decisive cases match the hard branch exactly",
        "costs": "capacity per added dimension, and lever 4 is the one most "
                 "often REFUTED by measurement -- see lever 6",
    },
    {
        "n": 5,
        "name": "tile the domain under an orchestrator",
        "when": "the whole will not fit but a piece will, and pieces are "
                "independent",
        "do": "process tiles, let an orchestrator hold only the seams",
        "evidence": "memory bounded by the TILE rather than the input: the "
                    "tiled fold streams from disk with np.memmap, and lazy "
                    "tensor loading turned a full-model copy into a rename",
        "costs": "seam handling, and a tiling whose seams interact is not "
                 "actually tiled",
    },
    {
        "n": 6,
        "name": "a measured limit is a TILE SIZE -- group, coordinate, "
                "clean up between levels",
        "when": "you measured a hard cliff and are about to call it structural",
        "do": "make the cliff number the group size, add a coordinator level "
              "above it, and CLEAN UP BETWEEN LEVELS; when the coordinator "
              "hits its own limit, split again",
        "evidence": "a NOTES entry declared the bundled-fact capacity cliff "
                    "STRUCTURAL after lever 4 was refuted -- and it was FALSE. "
                    "The nested register file reached 4,096 facts at 100% "
                    "recall (128 turns x 32) where a flat file evicted after "
                    "one turn",
        "costs": "a level of indirection per split, and the recall path -- not "
                 "the packing -- is where the win lives",
    },
)


def levers(problem=None):
    """The six levers, in cost order. Pass a problem description to rank them.

    NO SCORING CLEVERNESS: the ranking is keyword overlap against each lever's
    `when` and `name`, and it returns ALL SIX either way. A ranker that hid the
    other five would be worse than the list, because the whole value of the
    doctrine is walking it IN ORDER until one applies -- the cheapest lever that
    works beats the best-matching one."""
    out = [dict(x) for x in LEVERS]
    if not problem:
        return out
    toks = {w for w in str(problem).lower().replace("-", " ").split()
            if len(w) > 3}
    for lv in out:
        hay = (lv["when"] + " " + lv["name"] + " " + lv["do"]).lower()
        lv["match"] = sum(1 for t in toks if t in hay)
    out.sort(key=lambda lv: (-lv["match"], lv["n"]))
    return out


def wall_report(what, measured=None):
    """Turn "I hit a wall" into the ordered questions that get past it.

    THIS IS THE FACULTY AN AGENT ACTUALLY NEEDS. A limit reached is a decision
    point, and the failure mode is stopping at it -- so this returns the levers
    ranked, plus the one question each asks about YOUR problem, plus the
    standing rule that a measured number is a tile size before it is a wall."""
    ranked = levers(what)
    return {
        "problem": str(what),
        "measured": measured,
        "rule": "a measured limit is a composability boundary, not a wall -- "
                "walk the levers in cost order and stop at the first that "
                "applies",
        "ask": [{"lever": lv["n"], "name": lv["name"],
                 "question": "does this apply? %s" % lv["when"],
                 "evidence": lv["evidence"]} for lv in ranked],
        "last_resort": "if all six are genuinely refuted, record it as a KEPT "
                       "NEGATIVE with the measurement -- never as an opinion, "
                       "and never silently",
    }


def _selftest():
    assert len(LEVERS) == 6
    assert [lv["n"] for lv in LEVERS] == [1, 2, 3, 4, 5, 6]

    # ---- EVERY LEVER CARRIES EVIDENCE AND A COST. A lever without a case
    #      where it worked is advice, and one without a cost is a sales pitch.
    for lv in LEVERS:
        for k in ("name", "when", "do", "evidence", "costs"):
            assert lv.get(k) and len(lv[k]) > 20, (lv["n"], k)

    # ---- RANKING NEVER HIDES A LEVER, because the doctrine is to walk all six
    #      in order; a filter would defeat the thing it is filtering for.
    for probe in ("I ran out of memory", "recomputing the same value",
                  "two signals interfere", "", None):
        got = levers(probe)
        assert len(got) == 6, (probe, len(got))
        assert {lv["n"] for lv in got} == {1, 2, 3, 4, 5, 6}

    # ---- AND IT ACTUALLY RANKS: a storage problem should not lead with
    #      lever 4, and a recompute problem should surface lever 1.
    assert levers("the same expensive value is recomputed every query")[0]["n"] == 1
    r = wall_report("bundled facts hit a hard capacity cliff at 32",
                    measured={"cliff": 32})
    assert len(r["ask"]) == 6 and r["measured"]["cliff"] == 32
    assert "kept negative" in r["last_resort"].lower()

    print("levers selftest OK -- the SIX levers are a callable faculty now, not "
          "an oral tradition: each carries its own MEASUREMENT from this repo "
          "(61x prefix cache, 4,096 facts at 100%% recall, 128/128 decisive "
          "branches) and its own COST, ranking never hides one because the "
          "doctrine is to walk them in cost order, and the last resort is a "
          "KEPT NEGATIVE with a number rather than an opinion")


if __name__ == "__main__":
    _selftest()
