"""holographic_seedpack.py -- THE DISTILLED DOCTRINE, shipped with the engine (cp33).

What nomic text is to the embedding space, this pack is to the OPERATING KNOWLEDGE: the
lessons a long working session taught the substrate, distilled to their timeless form and
shipped as data, so a FRESH mind can boot already knowing how to be driven well. Each
entry carries the checkpoint that minted it -- these are measured lessons, not opinions.

The pack is OPT-IN (`mind.doctrine_load()`): booting every mind pre-taught would change
default behavior and hide the substrate's true cold state from benchmarks. Loading it
teaches through the NORMAL gate (same calibrated reflex machinery as any taught answer),
so seeded doctrine is indistinguishable in kind from lived doctrine -- and can be
overwritten by re-teaching, exactly like anything else learned.

Session-specific memories (what happened in round 1 of loop-X) stay in partitions where
they belong. The test for admission here: would this answer help a DIFFERENT deployment
of leCore, driven by a DIFFERENT model, next year? If not, it is memory, not doctrine.
"""

# (question, answer, provenance) -- the answer text is the deliverable; keep each one
# self-contained, imperative where possible, and honest about numbers being local.
DOCTRINE = [
    ("how should I answer questions cheaply with lecore",
     "walk the ladder: T0 reflex first (calibrated, verify-on-hit), T1 substrate recall "
     "and bound corpora, T2 deterministic dispatch, and only then the model rung -- "
     "teach every escalated answer back so the same question never costs tokens again",
     "cp15"),
    ("when should the reflex refuse to answer",
     "on any gate failure: cleanup margin, calibrated null, context coherence, veto set, "
     "or verify-on-hit token overlap below threshold -- near-exact repeats belong to "
     "reflex, paraphrases belong to recall or synthesis; a wrong T0 answer costs trust "
     "that hit rates never buy back",
     "cp24"),
    ("how do I run a long agent task with lecore",
     "agent_loop: gather from the substrate before any model call, resume-or-create the "
     "goal by objective, work steps under the trajectory tool cache, remember each "
     "round, checkpoint each round -- and stop on IDLE rounds, never wall clocks: "
     "deadlines kill legitimate slow work",
     "cp28"),
    ("how do I work with a large codebase in lecore",
     "codebase_map the tree once (full docstrings -- the map is what code_write sees), "
     "then query the archive instead of re-reading files; write code through code_write "
     "so nothing returns without passing ast and its test gate",
     "cp28-cp32"),
    ("how should research be stored in lecore",
     "research_archive: verbatim texts into the container corpus (lossless, bm25-"
     "queryable), notes as manifests; the container is the ONLY store -- loose json on "
     "disk is storage and forbidden, json over the wire is protocol and fine",
     "cp25-cp31"),
    ("what is the right size policy for memory",
     "compress, never cull: deliverables, cache values, certificates, feedback history "
     "all roundtrip FULL -- a clipped cache value is a wrong cache value; bounds are "
     "for compute budgets and screens, never for the record",
     "cp32"),
    ("how do I keep the goal book honest",
     "goals converge by semantic overlap, drift pauses work (wandering is stopped, not "
     "funded), pauses need deliberate resume, and work delivered outside executors is "
     "closed with goal_close carrying the receipt -- a book of ghosts is worse than no "
     "book",
     "cp19-cp30"),
    ("how should regression be run on this engine",
     "run the REAL selftests, not a subset: a regression that constructs a server but "
     "never runs its _selftest hid six checkpoints of red; every tool added moves its "
     "pin in the same commit; the checkpoint zips are the merge base a zip-level merge "
     "needs",
     "cp27-cp29"),
    ("what makes a benchmark of this engine honest",
     "a CORRECT column next to every hit rate (a confident wrong answer is not a win), "
     "the baseline is the expensive path replaced (never an array read), refused "
     "verdicts are results (do-not-trade beats a fake edge), and negative results stay "
     "pinned in docstrings where the next reader trips over them",
     "cp22-cp24"),
    ("when does the experience lever pay and when does it not",
     "it pays on self-similar workloads where the expensive path repeats up to "
     "similarity -- routing, solves, corpus answers; it does NOT pay on structureless "
     "streams (measured ~1x), volatile regions, or walls an exact algebra already "
     "dissolved -- there break-even is infinity by design",
     "cp15"),
    ("how are model calls kept rare",
     "the caller is the model rung: gather first, plan warm (warm plans propose, the "
     "cross-exam disposes), cache tool trajectories, teach answers back -- measured "
     "floor around one to two calls per novel objective and zero on reruns",
     "cp19-cp23"),
    ("how is learned state kept trustworthy over time",
     "calibrate from outcomes (isotonic reflex error), veto payloads never keys, "
     "regenerate polluted floors from durable texts, migrate by replay under the "
     "current key function, and fingerprint partitions so drift is a number, not a "
     "feeling",
     "cp21-cp26"),
    ("what storage format does lecore use for everything durable",
     "typed holographic containers (.lecore): one container per partition for learned "
     "state, one for the knowledge journal and scopes; legacy loose json migrates by "
     "replay on first touch and is renamed -- the selftest asserts no loose json "
     "survives a save",
     "cp20-cp31"),
    ("what is the doctrine on flagged debt",
     "a flagged debt left unmigrated is a scheduled outage -- the flag is the plan; "
     "run it before it detonates, and when it detonates anyway, fix the organ at the "
     "choke point, not the symptom at the call site",
     "cp28-cp31"),
]


def register_doctrine(mind, force=False):
    """Teach the pack through the normal gate. Returns the count taught.

    IDEMPOTENT, because boot() calls this and boot() is called more than once.
    _remember APPENDS unconditionally, so every boot re-taught the same 14 facts:
        boot 1 -> taught  14      boot 4 -> taught  56
        boot 2 -> taught  28      boot 5 -> taught  70
    A LONG-RUNNING SERVICE THAT RE-BOOTS GREW ITS TAUGHT STORE WITHOUT BOUND, 14
    rows a time, all identical. Recall was unaffected (measured: the same answer
    after 1 boot and after 10), so this is pure bloat rather than corruption --
    which is exactly why nothing caught it: no test asked, and the wrong answer
    never appeared.
    A marker on the mind is enough; `force=True` re-teaches after a reset."""
    # CHECK THE STORE, NOT JUST A MARKER. The marker stops a second boot on ONE
    # mind; it does NOT stop a boot that MOUNTED a partition already containing
    # doctrine -- and boot()'s order is POST -> mount -> doctrine, so the facts
    # arrive and are then taught again on top. Measured across save/reboot
    # cycles: taught 30 -> 100 -> 240 -> 520, all duplicates of the same 14.
    # THE SAME BUG AS THE LAST SWEEP, ONE LAYER OUT: fixing it in memory did not
    # fix it through the partition, because the second path never touched the
    # marker. Ask the store whether it already knows the first fact.
    if getattr(mind, "_doctrine_registered", False) and not force:
        return 0
    if not force and DOCTRINE:
        try:
            # `answer` is the ladder's read path (_recall does not exist -- I
            # guessed the name once and it cost a round trip). A doctrine fact
            # already present comes back at T0 with its [doctrine ...] tag.
            got = mind.zoo["ladder"].answer(DOCTRINE[0][0])
            txt = str(got.get("answer") if isinstance(got, dict) else got or "")
            if "[doctrine" in txt:
                mind._doctrine_registered = True
                return 0
        except Exception:
            pass                       # no recall path -> fall through and teach
    lad = mind.zoo["ladder"]
    n = 0
    for q, a, prov in DOCTRINE:
        lad._remember(lad._qkey(q), "%s  [doctrine %s]" % (a, prov), q)
        n += 1
    mind._doctrine_registered = True
    return n


def _selftest():
    import lecore
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "MODEL")
    n = register_doctrine(m)
    assert n == len(DOCTRINE) >= 14
    a = m.ask("how do I run a long agent task with lecore")
    assert a["tier"] == "T0" and "idle" in str(a["answer"]).lower()
    b = m.ask("what is the right size policy for memory")
    assert b["tier"] == "T0" and "cull" in str(b["answer"]).lower()
    # the pack must NOT auto-load: a virgin mind stays virgin
    v = lecore.UnifiedMind()
    v.zoo_attach(lambda p: "MODEL")
    c = v.ask("how do I run a long agent task with lecore")
    assert c["tier"] not in ("T0",), "doctrine must be opt-in; cold state stays honest"
    return "OK: seedpack self-test passed (%d doctrine entries teach at T0 through the " \
           "normal gate; virgin minds stay virgin -- opt-in like the nomic text)" % n


if __name__ == "__main__":
    print(_selftest())
