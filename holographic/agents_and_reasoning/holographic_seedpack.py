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
    # -- THE CODE-TOOLS PACK (sweep 64): the integrated code tooling, taught at
    # boot so a booted mind answers these at T0 and routes to the faculties
    # without a model call. Grounded in the SOTA survey (aider repo maps,
    # Agentless hierarchical localization, the field-wide convergence on exact
    # string replacement, edit-commit checkpointing) and this repo's sweeps.
    ("how do i edit a file safely",
     "view the region first, grep the TRUE anchor (never recall it), file_replace "
     "with count=1, then file_python_check IMMEDIATELY -- every edit, no batching; "
     "exact string replacement is the field-wide agent standard, and the editor "
     "keeps an undo stack (file editor undo) so a bad edit reverts instead of "
     "being patched on top",
     "sweeps 59-63; SWE-agent/OpenHands convergence"),
    ("how do i find where to make a change in a codebase",
     "hierarchical localization, cheapest first: repo_map for the ranked skeleton, "
     "file_grep to the file, file_view for the line range, only then edit -- never "
     "read whole files when a budgeted map answers; codebase_diagram when structure "
     "itself is the question",
     "aider repomap; Agentless; sweep 61"),
    ("how do i audit a codebase",
     "repo_map for the spine, spec_conformance for claims (mechanical file:line "
     "evidence, unverifiable is an honest abstain, prose never convicts), "
     "docs_generate for the reference; for leCore itself add the three wiring "
     "audits (reachability, catalog gaps, skill lint) and demand 0/0/0",
     "sweeps 61-63"),
    ("how should the model and the substrate divide code work",
     "the model PLANS -- writes the sop, sop_check until clean; the substrate "
     "EXECUTES -- sop_run invokes faculties, sandboxes code, verifies each step; "
     "the model is consulted only at guidance and escalation, and llm_calls in "
     "the result proves the count; whole-code generation is the last resort "
     "after templates and emitters",
     "sweep 62; leOS Director pattern"),
    ("how do i run tests or experiments in parallel",
     "the worker pool gives each worker its own interpreter (real parallelism "
     "under the GIL) with big read-only state shared once; select affected tests "
     "rather than everything; sandbox_run isolates a single experiment with "
     "rlimits; determinism is per-process -- PYTHONHASHSEED=0 in every worker",
     "pool + select_tests machinery"),
    ("how do i keep a long edit session safe",
     "checkpoint discipline: a state that passes its checks is the FLOOR -- later "
     "edits build on it, and consecutive regressions mean revert to the floor, "
     "never stack fixes on red; the editor undo stack and the delivery zip are "
     "the revert paths",
     "edit-commit checkpointing; sweeps 60-63"),
    ("what is rule zero before building anything",
     "ask the engine first: find_capability with five stranger phrasings of the "
     "need; reuse or extend whatever surfaces; a capability that cannot be "
     "surfaced and invoked does not exist, and only fallback hits license a build",
     "session doctrine; every sweep"),
    ("how do i verify a change before shipping",
     "four rungs, in order: static check on every touched file, the module "
     "selftest, end-to-end through the mind, and the wiring audits -- plus the "
     "http round trip for anything agent-facing; keep the evidence (command and "
     "output), assertion is not verification",
     "sweeps 61-63; ECO verification pipeline"),
    # -- imported from the branch memory (sweep 76): general lessons only; the
    #    selftest below GATES every row against absolute paths and key-shaped
    #    strings, so the seed constraint is enforced, never merely remembered --
    ("how do sessions prevent context bleed",
     "session_open salts the question key with the session name before keying, so "
     "each conversation lives in its own key space -- another session's reflexes "
     "are unreachable by the vector algebra itself; shared knowledge serves via "
     "the unsalted fallback; session_search crosses sessions explicitly and "
     "reopening resumes exactly; replay-on-load re-salts so isolation survives "
     "save cycles",
     "branch import, sweep 76"),
    ("how does a model boot on this substrate",
     "call boot() (or autoboot/agent_boot) FIRST: POST runs measured checks, the "
     "partition mounts -- and the memory rollover consolidates every prior state "
     "file into one fresh generation, so whatever memory exists arrives in "
     "current context -- doctrine loads through the normal gate, and os_prompt() "
     "hands the model its generated operating screen: syscall table, rules, "
     "escalation contract",
     "branch import + sweep 75 rollover, sweep 76"),
    ("can a warm plan be trusted without review",
     "no: warm plans PROPOSE, the cross-exam DISPOSES. Measured live: plan_warm "
     "matched one goal's steps to a different goal on text similarity alone; the "
     "cross-exam caught it by flagging every step needs_think. A warm plan with "
     "no grounded step is a suggestion wearing a plan's clothes",
     "branch import, sweep 76"),
    ("what makes a memory partition balloon",
     "three compounding causes, each measured: full-precision context vectors "
     "bloated by one-shot noise tokens; audit vectors serialized as text meta "
     "instead of binary arrays; taught replay re-writing already-served pairs "
     "every load/save cycle. The diet: quantized contexts with norm-pruning "
     "protected by the taught text, arrays as arrays, and replay that skips what "
     "the floor already serves",
     "branch import, sweep 76"),
    ("how do I research something online with lecore",
     "synthesize a bridge tool: the attached model IS the I/O rung -- it fetches "
     "results into a scratch file both sides agree on; the synthesized tool reads "
     "it; corpus_bind the findings, corpus_ask to ground claims, semantic_ingest "
     "the text, and teach the conclusions back as reflex answers so the research "
     "never costs tokens twice",
     "branch import (sanitized: no fixed paths), sweep 76"),
    ("how does the engine make an attached model self improving",
     "the loop: ask serves what is known; escalations go to the attached model "
     "(any text->text callable); feedback grades outcomes -- ok strengthens, "
     "not-ok vetoes and calibrates; reflect_failures turns failures into taught "
     "rules; successes teach back; workflow_distill turns done goals into plans "
     "that warm new objectives. No weight updates -- measured: a weak local "
     "model went 100% to 0% errors in one round while a frozen control stayed "
     "flat",
     "branch import, sweep 76"),
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
            # Probe FIRST and LAST: a partition seeded before the code-tools
            # pack (sweep 64) knows the first fact but not the last -- probing
            # only [0] would mark it registered and the new lessons would
            # never arrive. Same bug class as the marker-vs-store lesson
            # above, one layer further out: the guard must ask about the
            # WHOLE pack, and first+last brackets it.
            got0 = mind.zoo["ladder"].answer(DOCTRINE[0][0])
            gotN = mind.zoo["ladder"].answer(DOCTRINE[-1][0])
            t0 = str(got0.get("answer") if isinstance(got0, dict) else got0 or "")
            tN = str(gotN.get("answer") if isinstance(gotN, dict) else gotN or "")
            if "[doctrine" in t0 and "[doctrine" in tN:
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
    install_code_sops(mind)
    return n


_SELF_REVIEW_SOP = (
    "# lecore self review\n"
    "## step: wiring audit reachability\n"
    "shell: audit_reachability\n"
    "## step: wiring audit catalog gaps\n"
    "shell: audit_catalog_gaps\n"
    "## step: sandbox sanity\n"
    "python: print(6*7)\n"
    "verify: \"42\" in result[\"stdout\"]\n")


def install_code_sops(mind):
    """Install the boot SOP library: named, runnable orders a model can
    sop_run by name the moment the mind exists. Owner-directed DEFAULT-ON
    (sweep 64): boot BESTOWS the integrated code tools rather than hiding
    them behind a flag -- the recorded exception to extensions-default-off,
    by explicit direction. Idempotent (an already-saved SOP is not re-saved);
    a mind without the sop faculties is skipped silently, not crashed."""
    try:
        # OPERATOR-TIME command registration (the p11 doctrine: configure the
        # mind, THEN serve it -- boot IS the operator). Fixed argv, so the
        # SOP's shell steps carry only allowlisted NAMES, never command lines.
        import os
        for cname, script in (("audit_reachability", "tools/reachability_audit.py"),
                              ("audit_catalog_gaps", "tools/catalog_gaps.py")):
            if os.path.exists(script):
                try:
                    mind._register_command(cname, ["python3", script])
                except Exception:
                    pass       # already registered, or a stripped build
        if mind.sop_load("lecore_self_review").get("found"):
            return 0
        mind.sop_save("lecore_self_review", _SELF_REVIEW_SOP)
        return 1
    except Exception:
        return 0


def _selftest():
    import lecore
    # THE SEED CONSTRAINT AS A GATE (owner-directed, sweep 76): every new instance
    # starts from this pack, so no row may carry sensitive data or absolute file
    # path references -- the branch import surfaced one answer with a literal
    # scratch path, sanitized on entry. A constraint that lives only in review
    # is one distracted sweep from being violated; here it fails the build.
    import re
    _path = re.compile(r"(/home/\w+|/mnt/|/tmp/|/Users/|/var/|[A-Za-z]:\\\\)")
    _key = re.compile(r"(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{12,}"
                      r"|Bearer\s+[A-Za-z0-9._-]{16,}|eyJ[A-Za-z0-9_-]{20,}|[a-f0-9]{48,})")
    for _row in DOCTRINE:
        _txt = str(_row[0]) + " " + str(_row[1])
        assert not _path.search(_txt), "seed row carries an absolute path: %r" % _row[0]
        assert not _key.search(_txt), "seed row carries a key-shaped string: %r" % _row[0]
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "MODEL")
    n = register_doctrine(m)
    assert n == len(DOCTRINE) >= 14
    # the six branch-imported lessons answer at T0 through the normal gate
    for _q, _frag in [("how do sessions prevent context bleed", "salts"),
                      ("can a warm plan be trusted without review", "cross-exam"),
                      ("what makes a memory partition balloon", "replay"),
                      ("how do I research something online with lecore", "bridge"),
                      ("how does the engine make an attached model self improving", "veto"),
                      ("how does a model boot on this substrate", "rollover")]:
        _a = m.ask(_q)
        assert _a["tier"] == "T0" and _frag in str(_a["answer"]).lower(), (_q, _a)
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
