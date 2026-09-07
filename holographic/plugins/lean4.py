"""LEAN 4 BRIDGE -- the first bundled plugin, and the worked example of the migration.

WHAT MOVED, AND WHAT DELIBERATELY DID NOT
-----------------------------------------
Moved here: the four faculties that exist to talk to an EXTERNAL Lean 4 install --
`lean_export`, `lean_verify`, `lean_status`, `lean_fuzz`. These are the honest
optional-dependency surface: `lean_verify` is useless without a `lean` binary,
`lean_status` exists only to report whether that binary is there, and the other two
are the emit/differential-test path around it.

NOT moved, on purpose: `logic_prove`, `logic_check_proof`, `logic_consequences`,
`logic_induce`, `logic_query`, `proof_store`, `proof_recall` and the rest of the
`logic_*` family. Those are the pure-stdlib Horn kernel in holographic_lean -- 1,318
lines that need NO external anything and that other core faculties build on. Moving
them would be moving load-bearing core out of core and calling it de-bloating. The
line is "does this need something outside the wheel", not "does the word Lean appear
in it".

THE VERBS CLOSE OVER THE MODULE, NOT THE MIND. None of the four touched `self` beyond
being methods, so each is a plain function here and the plugin is testable without a
mind at all. A verb that genuinely needs mind state would close over the `mind`
argument `register` receives -- explicit, which is the point.

BEHAVIOUR IS UNCHANGED, byte for byte: the bodies below are the former method bodies
with `self` dropped, and the docstrings are carried over verbatim so `GET /tools`,
find_capability and REFERENCE.md say exactly what they said before. This is a MOVE,
not a rewrite -- a rewrite hidden inside a migration is how a refactor silently
changes results.
"""

PLUGIN = {
    "name": "lean4",
    "version": "1.0",
    "does": "Lean 4 bridge: emit Lean source, verify it through an installed lean binary, "
            "report the dependency tier, and fuzz the export path.",
}


def lean_export(goal, rules, theorem_name="derived", check=True):
    """Prove a goal and emit self-contained Lean 4 source (axioms + term-mode theorem).
    HONEST SCOPE: Lean verifies the proof FOLLOWS from the rules; it does NOT verify the
    rules are consistent -- an inconsistent rule set proves anything and typechecks doing
    it (see logic_consequences' absurdity smoke). check levels: True/"internal" runs the
    independent in-process checker before emitting; "external" ALSO round-trips through an
    installed lean binary and refuses ok=True unless BOTH agree (the de Bruijn criterion:
    two independent checkers, agreement as the deliverable) -- with no binary, ok is False
    and external.available says why, never faked. Returns {"ok","lean","proof"[,"external"]};
    ok=False, lean=None when underivable. See holographic_lean.to_lean."""
    from holographic.agents_and_reasoning import holographic_lean as _L
    rs = _L.rules_from_wire(rules)
    p = _L.prove(_L.atom_from_wire(goal), rs)
    if p is None:
        return {"ok": False, "lean": None, "proof": None}
    if check:
        _L.check_proof(p, rs)
    src = _L.to_lean(p, rs, theorem_name=theorem_name)
    out = {"ok": True, "lean": src, "proof": _L.proof_to_wire(p)}
    if check == "external":
        res = _L.lean_check(src)
        out["external"] = res
        # agreement is the deliverable: internal passed above; ok stands only if the
        # external kernel ALSO said proved (available and ok) -- absence is a loud False
        out["ok"] = bool(res.get("available")) and bool(res.get("ok"))
    return out


def lean_verify(source, timeout=60):
    """Round-trip Lean 4 source through an installed `lean` binary (opt-in bridge,
    numba-style; the engine never requires it). Returns {"available", "ok", ...} --
    {"available": False} when no binary exists, stated honestly rather than pretended.
    See holographic_lean.lean_check."""
    from holographic.agents_and_reasoning import holographic_lean as _L
    return _L.lean_check(source, timeout=timeout)


def lean_status():
    """Report the Lean 4 dependency tier without downloading or requiring anything.
    TIER 0 (always on, NumPy+stdlib): kernel, independent checker, Lean-source EMITTER,
    induction, fuzz oracle's non-Lean stages, proof memory at provenance 'checked'.
    TIER 1 (opt-in, ~1.3 GB installed): an external Lean binary -- buys exactly the
    'lean_verified' provenance tier. Install/remove via tools/install_lean.py (version
    and sha256 PINNED; a verifier downloaded unverified would be a joke at our own
    expense). Returns {"tier", "on_path", "local_install", "version", "pinned_version",
    "path_hint", "install_hint"}."""
    import importlib.util
    import os
    # The repo root, four levels up from holographic/plugins/lean4.py. Computed rather
    # than hardcoded because this file moved one directory deeper than the method it
    # replaces -- an off-by-one here is a silent "Lean not installed" on a box where it is.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location(
        "lecore_install_lean", os.path.join(root, "tools", "install_lean.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    st = mod.status()
    st["tier"] = 1 if st["version"] else 0
    st["install_hint"] = None if st["version"] else "python3 tools/install_lean.py"
    return st


def lean_fuzz(n=30, seed=0):
    """Differential oracle over the whole logic chain: n random HOSTILE theories (Lean
    keywords, collision pairs, digit-led names) through prove-both-strategies -> check
    -> export -> external Lean when installed (which is itself probed with a corrupted
    term each run). Failures return with their seed for pinning as Lean-free regression
    tests -- the distillation contract: Lean finds a bug once, the repo keeps the pin,
    the binary stays optional. Standing result on record: 300 theories, 793 exports,
    0 failures. An empty list is a measured statement about n seeds, not a proof.
    See holographic_lean.fuzz_export."""
    from holographic.agents_and_reasoning import holographic_lean as _L
    return _L.fuzz_export(n=int(n), seed=int(seed))


def register(mind, config=None):
    """Bind the four Lean 4 bridge verbs. `config` is accepted and unused -- the bridge
    has nothing to configure; the binary is found by tools/install_lean.py's search."""
    return [
        {"name": "lean_export", "fn": lean_export,
         "does": "Prove a goal and emit self-contained Lean 4 source (axioms + term-mode theorem).",
         "example": "mind.lean_export({'pred':'p','args':['a']}, {'facts':[{'pred':'p','args':['a']}],'rules':[]})",
         "aliases": ("export a proof to lean", "emit lean 4 source", "write a lean theorem",
                     "formal proof export", "turn a derivation into lean")},
        {"name": "lean_verify", "fn": lean_verify,
         "does": "Round-trip Lean 4 source through an installed lean binary; honest {'available': False} without one.",
         "example": "mind.lean_verify('theorem t : True := trivial')",
         "aliases": ("check a proof with lean", "typecheck lean source", "run the lean binary",
                     "verify externally", "does lean accept this")},
        {"name": "lean_status", "fn": lean_status,
         "does": "Report the Lean 4 dependency tier (0 = always on, 1 = external binary installed).",
         "example": "mind.lean_status()",
         "aliases": ("is lean installed", "lean dependency tier", "do i have the lean binary",
                     "check lean availability", "lean install status")},
        {"name": "lean_fuzz", "fn": lean_fuzz,
         "does": "Differential fuzz over the prove -> check -> export -> external-Lean chain; failures carry their seed.",
         "example": "mind.lean_fuzz(n=5, seed=0)",
         "aliases": ("fuzz the lean export", "differential test the prover",
                     "random theories against lean", "stress the proof exporter")},
    ]


def _selftest():
    """Contracts:
    1. The four verbs bind and are discoverable through a mind that loads this plugin.
    2. Behaviour is IDENTICAL to the pre-migration methods: a known derivable goal still
       exports Lean source, and lean_status still answers without a binary present.
    3. A slim mind (plugins=()) does NOT carry them -- which is the whole point of the move.
    """
    import lecore

    m = lecore.UnifiedMind(dim=64, seed=0)          # default: bundled plugins auto-load
    for verb in ("lean_export", "lean_verify", "lean_status", "lean_fuzz"):
        assert callable(getattr(m, verb, None)), "bundled lean4 did not bind %r" % verb

    rules = [{"head": ["human", ["socrates"]], "name": "h"},
             {"head": ["mortal", ["?x"]], "body": [["human", ["?x"]]], "name": "m"}]
    out = m.lean_export(["mortal", ["socrates"]], rules, theorem_name="soc")
    assert out["ok"] and "theorem soc : mortal socrates :=" in out["lean"], out
    # the pure Horn kernel stayed in core -- prove that the split did not sever them
    assert m.logic_prove(["mortal", ["socrates"]], rules) is not None
    assert m.lean_fuzz(n=2, seed=0)["failures"] == []

    st = m.lean_status()
    assert st["tier"] in (0, 1) and "install_hint" in st, st
    # No binary in CI: the honest answer, never a pretend one.
    assert m.lean_verify("theorem t : True := trivial")["available"] in (True, False)

    assert "lean_export" in [getattr(c, "name", "")
                             for c in m.find_capability("export a proof to lean")[:3]], \
        "lean4 verbs bound but UNDISCOVERABLE"

    slim = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    assert not hasattr(slim, "lean_verify"), "plugins=() still carried the lean4 verbs"
    print("holographic.plugins.lean4 selftest OK")


if __name__ == "__main__":
    _selftest()
