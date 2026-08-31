"""holographic_bios.py -- THE BIOS AND BOOT SEQUENCE for a mind on this substrate (cp34).

leOS gave the lineage its OS instincts (substrate_gather, activity_monitor, the tool
loop); Unicron gave it introspection (spectral reading of its own memory). This module
makes the boot explicit, the way firmware does:

  POST      power-on self test: MEASURED checks -- levers present, ladder serving,
            container round-trip, calibration machinery, Unicron spectral health when
            learned state exists. Every check returns (name, ok, detail); a red POST is
            reported, never papered over.
  INVENTORY the machine table an operator reads: lever count, catalog size, faculty
            count, partitions found, doctrine state, journal format.
  BOOT      boot order: mount the named partition if given (learning_load), else stay
            virgin; opt-in doctrine (the seedpack); ensure services (the zoo tenant).
            Returns the full boot report.
  OS PROMPT the hand-off: a DETERMINISTIC text the attached LLM receives as its
            operating system -- identity, POST summary, the syscall table (core
            faculties with one-line contracts), the operating rules (distilled
            doctrine), memory status, and the escalation contract. This is the BIOS
            screen an LLM reads instead of a human.

The prompt is generated FROM the live mind (catalog + doctrine + state), so it cannot
drift from the engine the way a hand-written primer would -- the same law that keeps
CAPABILITIES.md regenerated, applied to the model's own boot screen.
"""

import numpy as np

# The SYSCALL TABLE: the curated core an attached model needs on screen at boot.
# Everything else is reachable through the find syscall -- the table teaches the pattern,
# the catalog carries the long tail. (name, signature, one-line contract)
SYSCALLS = [
    ("ask", "ask(question)", "walk the ladder: reflex -> recall -> dispatch -> YOU; "
     "answers carry {tier, via, why}; cheap rungs refuse rather than guess"),
    ("teach", "zoo['ladder'].teach / answer_feedback(q, ok)", "close the loop: teach "
     "escalated answers back; report outcomes so calibration stays honest"),
    ("find", "find_capability(need, k)", "the long tail: 3,500+ cataloged capabilities "
     "ranked for a plain-english need; returns names + how to call"),
    ("loop", "agent_loop(objective, executors, ...)", "long-running work: gather first, "
     "resume-or-create the goal, cache tool trajectories, checkpoint each round, stop "
     "on IDLE rounds never wall clocks"),
    ("map", "codebase_map(root, topic)", "index a codebase into the archive once; query "
     "it instead of re-reading files"),
    ("write", "code_write(name, task, topic, test)", "code returns only after ast + its "
     "test gate pass; failing drafts are refused with the failure"),
    ("archive", "research_archive(topic, texts) / archive_query(topic, q)", "verbatim "
     "corpus in the container, bm25-answerable; notes are manifests"),
    ("goal", "goal_create / goal_work / goal_close", "plans converge by overlap, drift "
     "pauses work, receipts close what executors did not run"),
    ("save", "learning_save(root) / learning_load(root)", "the partition IS the mind's "
     "continuity: one typed container, full record, compression not culling"),
    ("health", "learning_spectrum() / partition_drift(a, b)", "Unicron inward: memory "
     "health as measured spectra and drift numbers, not feelings"),
]


def post(mind):
    """Power-on self test. Returns [(check, ok, detail)] -- measured, never assumed."""
    checks = []
    try:
        lv = mind.levers()
        checks.append(("levers", len(lv) == 7, "%d/7 levers" % len(lv)))
    except Exception as exc:
        checks.append(("levers", False, str(exc)[:80]))
    try:
        a = mind.ask("__post__ probe question never taught")
        ok = a.get("tier") not in ("T0",)
        checks.append(("ladder-honesty", ok,
                       "unknown question escalates (tier %s)" % a.get("tier")))
    except Exception as exc:
        checks.append(("ladder-honesty", False, str(exc)[:80]))
    try:
        from holographic.io_and_interop.holographic_container import (save_container,
                                                                      load_container)
        blob = save_container([{"kind": "bios.post", "id": "v1",
                                "meta": {"n": 3}, "arrays":
                                {"x": np.arange(4, dtype=np.float32)}}])
        got = load_container(blob)
        ok = got["sections"][0]["meta"]["n"] == 3
        checks.append(("container-roundtrip", ok, "%d bytes" % len(blob)))
    except Exception as exc:
        checks.append(("container-roundtrip", False, str(exc)[:80]))
    try:
        lad = mind.zoo["ladder"]
        ok = hasattr(lad, "read_error_prob") or hasattr(mind, "reflex_error_prob") \
            or hasattr(lad, "trace")
        checks.append(("calibration-machinery", ok, "reflex gate present"))
    except Exception as exc:
        checks.append(("calibration-machinery", False, str(exc)[:80]))
    # Unicron inward -- only meaningful once learned state exists
    try:
        te = getattr(mind, "_lever7_text", None)
        vocab = len(getattr(te, "context", {}) or {}) if te else 0
        if vocab >= 32:
            sp = mind.learning_spectrum()
            verdict = str(sp.get("verdict", ""))[:60] if isinstance(sp, dict) else "read"
            checks.append(("unicron-spectral", True,
                           "vocab %d, %s" % (vocab, verdict or "spectra read")))
        else:
            # "virgin mind" IS WRONG WHENEVER MEMORY LOADED AND WAS THIN. A
            # partition carrying 116 logged queries reported exactly the same
            # line as an empty one, because this check reads the LEVER-7 TEXT
            # VOCABULARY and nothing else -- true about vocabulary, misleading
            # about the boot. Say which it is.
            _from = getattr(mind, "_learning_loaded_from", None)
            checks.append(("unicron-spectral", True,
                           "skipped: vocabulary %d < 32 (%s)"
                           % (vocab,
                              "no memory loaded -- virgin mind" if not _from
                              else "memory loaded from %r, but too few DISTINCT "
                                   "texts to read a spectrum" % str(_from))))
    except Exception as exc:
        checks.append(("unicron-spectral", False, str(exc)[:80]))
    return checks


def inventory(mind):
    """The machine table."""
    inv = {}
    try:
        inv["levers"] = len(mind.levers())
    except Exception:
        inv["levers"] = 0
    try:
        inv["catalog"] = len(mind.catalog_names()) if hasattr(mind, "catalog_names") \
            else sum(1 for _ in getattr(mind, "_catalog", {"x": 1}))
    except Exception:
        inv["catalog"] = "unknown"
    inv["faculties"] = len([a for a in dir(mind) if not a.startswith("_")])
    lad = mind.zoo.get("ladder") if hasattr(mind, "zoo") else None
    inv["taught"] = len(getattr(lad, "taught_log", []) or []) if lad else 0
    inv["goals"] = len(mind.goal_book.goals) if hasattr(mind, "goal_book") else 0
    inv["archives"] = sorted(getattr(mind, "_archive_corpora", {}) or {})
    return inv


def boot(mind, partition=None, doctrine=True, llm=None):
    """The boot order: POST -> mount -> doctrine -> services -> report."""
    if llm is not None and not mind.zoo.get("llm"):
        mind.zoo_attach(llm)
    report = {"post": post(mind), "mounted": None, "doctrine": 0}
    if partition:
        mind.learning_load(str(partition))
        mind._archive_root = str(partition)
        report["mounted"] = str(partition)
        report["post_after_mount"] = post(mind)       # spectral check now has state
    if doctrine:
        import holographic.agents_and_reasoning.holographic_seedpack as \
            holographic_seedpack
        report["doctrine"] = holographic_seedpack.register_doctrine(mind)
    report["inventory"] = inventory(mind)
    report["ok"] = all(ok for _, ok, _ in report["post"])
    return report


def os_prompt(mind, report=None):
    """The BIOS screen an LLM reads: deterministic, generated from the live mind."""
    import holographic.agents_and_reasoning.holographic_seedpack as holographic_seedpack
    report = report or {"post": post(mind), "inventory": inventory(mind)}
    posted_src = report.get("post_after_mount") or report["post"]   # the screen shows
    lines = []                                                      # the MOUNTED state
    lines.append("=== leCore BIOS v1 -- you are the model rung of this substrate ===")
    lines.append("LINEAGE: leOS (gather-first, idle-based loops) + Unicron (spectral "
                 "introspection) + the seven levers.")
    lines.append("POST: " + "; ".join("%s %s (%s)" %
                 (n, "OK" if ok else "FAIL", d) for n, ok, d in posted_src))
    inv = report["inventory"]
    lines.append("MACHINE: %s levers | %s faculties | %s taught | %s goals | "
                 "archives: %s" % (inv.get("levers"), inv.get("faculties"),
                                   inv.get("taught"), inv.get("goals"),
                                   ", ".join(inv.get("archives") or []) or "none"))
    lines.append("--- SYSCALLS (the core; everything else via find) ---")
    for name, sig, contract in SYSCALLS:
        lines.append("  %-8s %s\n           %s" % (name, sig, contract))
    lines.append("--- OPERATING RULES (distilled doctrine, provenance-tagged) ---")
    for q, a, prov in holographic_seedpack.DOCTRINE:
        lines.append("  * [%s] %s" % (prov, a))
    lines.append("--- CONTRACT ---")
    lines.append("  Gather from the substrate BEFORE answering. Teach back what you "
                 "answer. Report outcomes so calibration stays honest. Refuse rather "
                 "than guess; a refusal with a why is a good result. The partition is "
                 "your continuity: save on milestones, and everything durable rides the "
                 "container at FULL length.")
    return "\n".join(lines)


def _selftest():
    import lecore
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "MODEL")
    rep = boot(m, partition=None, doctrine=True)
    assert rep["ok"], "virgin POST must pass: %r" % rep["post"]
    assert rep["doctrine"] >= 14
    txt = os_prompt(m, rep)
    for needle in ("BIOS", "POST", "SYSCALLS", "OPERATING RULES", "CONTRACT",
                   "agent_loop", "idle", "refuse"):
        assert needle.lower() in txt.lower(), needle
    a = m.ask("how do I run a long agent task with lecore")
    assert a["tier"] == "T0", "doctrine must serve at T0 after boot"
    # determinism: the screen is a function of the mind, not the clock
    assert os_prompt(m, rep) == txt
    return ("OK: bios self-test passed (POST %d checks green on a virgin mind; boot "
            "teaches %d doctrine entries; the OS prompt is deterministic and carries "
            "syscalls + rules + contract)" % (len(rep["post"]), rep["doctrine"]))


if __name__ == "__main__":
    print(_selftest())
