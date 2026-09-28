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
    # -- THE DECISION-LOOP PACK (sweeps 171-182, distilled 2026-09-27): what typed decisions, the meaning rung, the
    #    CLM panel and the learning-loop audit taught about driving and teaching this engine. Admission test as
    #    above -- each row would help a DIFFERENT deployment driven by a DIFFERENT model; the numbers are local
    #    measurements (docs/research/BENCHMARK_sweep182_contrastive.md and the sweep-171..181 benchmarks), kept so
    #    the reader knows the lesson was measured, not assumed.
    ("how do I make a decision with a fixed set of answers in lecore",
     "ask a TYPED question instead of prompting a model: typed(state, options) or systemone_decide with a "
     "schema returns value, ranked, margin, p_correct, p_null, an honest abstention and an id; lint the schema "
     "first (systemone_lint: balanced example budgets, at least 3 examples per option, split multi-clause "
     "states) -- a fixed-answer question answered in free text is a guess nothing can learn from",
     "sweeps 171-176"),
    ("how does a lecore decision learn from what actually happened",
     "report the outcome by id: decision_outcome(id, truth) for every id you acted on -- typed, route, rank, "
     "serve, meaning, compose -- is the ONLY outcome path and it trains that door's prototypes, its calibrator "
     "and the reflex; ids survive a restart once the partition is booted; report 'failed' only when no truth is "
     "known; never let an unjudged self-report train a door (measured: agents that believed themselves learned "
     "14 wrong answers where judged agents learned 0)",
     "sweeps 176-182"),
    ("how do I read the confidence of a lecore answer",
     "p_correct is the calibrated probability the answer is right (HIGH = confident) and is None until that "
     "door has calibrated on reported outcomes of both kinds -- an uncalibrated door says so instead of "
     "inventing a number; p_null is a significance p-value against the door's null (LOW = significant); the "
     "bare p is deprecated because it meant opposite things at different doors",
     "sweep 182"),
    ("should different decision doors share one calibration or one learned store",
     "share the algorithm, never the state: one update rule and one store class, but every door keeps its own "
     "prototypes, codebook, label stream and calibrator -- a shared calibration let one door's outcomes move "
     "another door's veto, and a correction written for one key erased a neighbour's answer at cosine 0.8",
     "sweep 182"),
    ("how should a correction be written into an associative memory trace",
     "subtract the wrong answer as well as adding the truth, and project the correction off the nearest keys "
     "that own a different answer (LMS plus an affine projection): a truth-only write left the wrong answer at "
     "full strength (the truth read back 45% of the time after one correction); the projected correction reads "
     "100% and leaves a neighbour at cosine 0.8 untouched; corrections are raw deltas and must be replayed "
     "verbatim on every rebuild, split and reload",
     "sweep 182"),
    ("how can I tell a noisy teacher from genuinely confusable data",
     "measure the teacher against ITSELF: re-ask a few percent of questions with the candidates permuted and "
     "solve P(agree) = (1-e)^2 + e^2/m for the flip rate e -- agreement with your own most confident rows "
     "cannot tell the two apart (it read 0.94-0.99 for a PERFECT teacher); with e measured, correct the serve "
     "gate and train on noise-corrected targets, using a conservative bound on e, because a small re-ask "
     "sample misreads it and swings coverage",
     "sweep 182"),
    ("how do I choose a serve or abstain threshold honestly",
     "calibrate on validation plus half of a held-out off-topic set, report on test plus the other half, under "
     "the deployment's real mix of in-scope and off-topic questions; report risk-coverage (AURC) and several "
     "operating points with paired-bootstrap intervals over seeds, and count the configurations you tried -- "
     "one coverage-at-exact-precision number swung from 37.7% to 3.3% between two calibration samples of the "
     "same model",
     "sweep 182"),
    ("what confidence signal should an abstaining door use",
     "the door's own score, calibrated isotonically on its reported outcomes, with an absolute floor for "
     "refusals; a softmax over the candidates is a feature, never the confidence -- it cannot say 'none of "
     "these' and abstained worse (AURC 0.094 vs 0.078 for the absolute cosine); measure top score against "
     "margin per door (ranking a caller's own candidates, the margin won at every set size)",
     "sweep 182"),
    ("how do I know learned state survives a restart",
     "test the whole life cycle, door by door: decide, roll over and save, cold-reload in a FRESH mind, report "
     "the outcome by id, check the door learned, reload again and check it kept it, and scan the saved files "
     "for secrets; live closures die with the process, so a record must name WHAT to call (a hook spec) for "
     "the reloaded mind to re-derive -- an audit of 14 doors found 10 that silently stopped learning or leaked "
     "across a restart",
     "sweep 182"),
    ("when must a learned threshold be re-derived",
     "whenever the population it was calibrated on changes: a threshold on a z-score against a null built from "
     "the catalog's own vocabulary moved (6.87 to 7.17) when the catalog grew, and a benchmark's answer/menu "
     "split moved with it; re-run the calibration, and freeze benchmark workloads as fixtures so the questions "
     "stay put while the code under test is compared",
     "sweeps 179-182"),
    ("what must lecore never learn or save",
     "secrets -- keys, passwords, seed phrases, in every wording people paste them, including 'X is my "
     "password' -- and static answers to live questions (prices, weather): remember the METHOD that fetches a "
     "live value, never the value; a question made only of a value (a bare ticker) is clarified, never acted "
     "on; and redact before remembering anything you refused -- a guard that learned from its refusals once "
     "saved the refused secret into its own examples",
     "sweeps 179-182"),
    ("how does lecore find a taught answer when the question is reworded",
     "near-exact repeats belong to the reflex; rewordings belong to the meaning rung, which serves a taught row "
     "only at a calibrated probability the model would agree, and hands an unsure question to the model as a "
     "TYPED choice (same row / new answer plus how it was found / unclear) whose verdict is learned at once -- "
     "each new wording costs one model call, once",
     "sweep 181"),
    ("what is a good hard negative for a learned prototype",
     "the candidate that actually beat the truth, labelled by the teacher or an outcome -- never a generated "
     "sentence; and check the teacher's margin before trusting it: a wrong 'not A' usually points at a far "
     "candidate, so cos(q, pick) - cos(q, A) flags planted teacher errors (AUROC 0.936) where a hardness-based "
     "quality score did worse than chance",
     "sweep 182"),
    ("how should a swarm of agents share one lecore memory",
     "one service process per memory partition and N agents on one URL, never in-process minds on the same "
     "partition; every step goes through swarm_step with done_when, evidence and a verify that runs before the "
     "step is accepted; only a verified outcome teaches, and a failed verify is a labelled failure too; teach "
     "measurements and ruled-out approaches as you go, so the next agent's first question finds them",
     "sweeps 176-182"),
    # -- THE CORE-MEMORY PACK (2026-09-27, owner-directed: the seed and core memory exist to cut model calls and
    #    pick tools; what a mind learns about wording, methods and APIs must reach it). Numbers are local
    #    measurements (tools/bench_core_memory.py -> docs/research/evidence/bench_core_memory.json).
    ("how does lecore remember how to use an api that needs a key",
     "as a named environment placeholder, never the key: api_learn reads the spec's security scheme, and a key "
     "passed once by hand to api_use teaches which header or query parameter carries it -- both are stored as "
     "${SERVICE_PARAM} (toyair + X-Api-Key -> ${TOYAIR_API_KEY}); tool reflex params and method constants too. The "
     "value is read from the environment only at call time, and an unset variable fails loudly naming it, before "
     "any request is made -- a learned call never goes out without its auth",
     "core memory, 2026-09-27"),
    ("what should travel from a working memory into the core memory",
     "whatever saves a model call or picks a tool, once it is confirmed: the confirmed wordings of shipped rows "
     "(associations re-learned from shipped wordings only), method rows (verb, slots, words-to-value maps, never a "
     "live value), api specs and tool reflexes with placeholders, and door stores the commons rule calls "
     "shareable; a row the model created ships only after a second event confirmed it. Measured on CLINC150: "
     "rows alone carried 19 of 800 held-out rewordings a fresh mind could serve, the extended distillate with its "
     "calibration 308 (the mind that learned them: 312); question vectors without their text, one user's "
     "decisions and a deployment's thresholds stay home",
     "core memory, 2026-09-27"),
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
        if not force:
            # TEACH ONLY WHAT IS MISSING (2026-09-27). The first+last probe above answers "is the whole pack
            # here?"; when a pack GROWS, a partition seeded earlier fails it and the old code re-taught EVERY row,
            # duplicating the rows the store already served (the bloat the two fixes above exist to stop). Ask
            # the store about each row and teach only the rows it cannot answer with their doctrine tag.
            try:
                got = lad.answer(q)
                txt = str(got.get("answer") if isinstance(got, dict) else got or "")
                if "[doctrine" in txt and txt.startswith(a[:40]):
                    continue
            except Exception:
                pass
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
    # the decision-loop pack (sweeps 171-182) answers at T0 too
    for _q, _frag in [("how do I read the confidence of a lecore answer", "p_correct"),
                      ("how does a lecore decision learn from what actually happened", "decision_outcome"),
                      ("what must lecore never learn or save", "method"),
                      ("how can I tell a noisy teacher from genuinely confusable data", "re-ask")]:
        _a = m.ask(_q)
        assert _a["tier"] == "T0" and _frag in str(_a["answer"]).lower(), (_q, _a)
    # a GROWN pack teaches only the new rows into a store that already holds the old ones
    g = lecore.UnifiedMind()
    g.zoo_attach(lambda p: "MODEL")
    lad = g.zoo["ladder"]
    for _q, _a2, _p in DOCTRINE[:-3]:
        lad._remember(lad._qkey(_q), "%s  [doctrine %s]" % (_a2, _p), _q)
    before = len(lad.taught_log)
    assert register_doctrine(g) == 3 and len(lad.taught_log) == before + 3
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
