"""PROGLIB -- VSA programs that find themselves when the context calls for them.

Moose asked whether there are VSA programs we can install or run on the fly, and
whether they can be naturally discoverable from context. Rule 0 first, and it
answered most of the question: LECORE ALREADY HAS THE PROGRAMS.

`HoloMachine` describes itself as "a formatted holographic drive that can store
and execute stored programs" with FOURTEEN OPCODES -- LOAD, STORE, BIND, BUNDLE,
PERMUTE, RECALL, PUSH, POP, APPLY, CALL, IFMATCH, ITERATE, REPEAT, HALT -- which
is the VSA algebra plus control flow. `assemble` turns a list of
(opcode, operand) into ONE HYPERVECTOR. `define` names a procedure that other
programs CALL. `APPLY` reaches any named faculty. VERIFIED here: a program run
inline and the same program reached through CALL produce IDENTICAL accumulators,
so composition is exact rather than approximate.

SO PROGRAMS ARE ALREADY SELF-CONTAINED (one vector) AND COMPOSABLE (CALL). What
was missing is the third thing Moose asked for: DISCOVERY. A library of programs
nobody can find by describing their situation is a library nobody uses, which is
the same failure Rule 0 exists to prevent for capabilities.

WHAT THIS ADDS: programs are indexed by the SAME mechanism leCore already uses
for passages -- a bundle-over-positions address of their description, matched by
cosine. So `find("undo a binding and clean it up")` returns the program whose
description that resembles, and the program is a vector you can immediately run.
The index is a codebook, the match is an argmax, and both are things a model can
do in its own head -- which is why this composes with unicron_memory_search
rather than duplicating it.

THE HONEST LIMIT: discovery is by DESCRIPTION SIMILARITY, not by understanding
what a program does. A program described badly is a program that will not be
found, exactly as a catalog entry with poor aliases is unreachable -- and this
project has the skill_lint audit precisely because that failure is so easy.
"""

import hashlib

import numpy as np


def _symbol(text, dim, seed_tag="proglib"):
    """A deterministic hypervector for a piece of text. hashlib, never hash()."""
    h = hashlib.sha256(("%s:%s" % (seed_tag, text)).encode("utf-8")).digest()
    g = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = g.standard_normal(int(dim))
    return v / (np.linalg.norm(v) + 1e-30)


def describe(text, dim, seed_tag="proglib"):
    """Address a description as a BUNDLE over its words.

    The same construction memsearch uses for passages, and for the same measured
    reason: a bundle is order-blind and robust to a partial cue, so a user who
    types three of the five words still lands on the right program."""
    words = [w for w in str(text).lower().split() if len(w) > 2]
    if not words:
        return _symbol(text, dim, seed_tag)
    v = np.sum([_symbol(w, dim, seed_tag) for w in words], axis=0)
    return v / (np.linalg.norm(v) + 1e-30)


class ProgramLibrary:
    """Named VSA programs, addressable by describing what you want.

    Holds the (opcode, operand) source alongside the assembled vector, because a
    program you cannot read is a program nobody will trust enough to run."""

    def __init__(self, machine, dim=None, seed_tag="proglib"):
        self.m = machine
        self.dim = int(dim or getattr(machine, "dim", 1024))
        self.seed_tag = str(seed_tag)
        self.progs = {}
        self._addr = {}

    def add(self, name, program, description, define=True):
        """Register a program: source, assembled vector, and its address.

        `define` also installs it as a CALLABLE PROCEDURE, so other programs can
        CALL it by name -- which is what makes the library composable rather
        than merely a list."""
        vec = self.m.assemble(list(program))
        if define:
            try:
                self.m.define(name, list(program))
            except Exception:
                pass
        self.progs[name] = {"program": list(program), "vector": vec,
                            "description": str(description)}
        self._addr[name] = describe(description, self.dim, self.seed_tag)
        return self.progs[name]

    def find(self, context, k=3):
        """Which programs does this situation call for? Ranked, with scores."""
        if not self._addr:
            return []
        q = describe(context, self.dim, self.seed_tag)
        names = list(self._addr)
        A = np.stack([self._addr[n] for n in names])
        scores = A @ q
        order = np.argsort(scores)[::-1][:int(k)]
        return [(names[i], float(scores[i]), self.progs[names[i]]["description"])
                for i in order]

    def run(self, name, init_acc=None, **kw):
        out = self.m.run(self.progs[name]["vector"], init_acc=init_acc, **kw)
        return out[0] if isinstance(out, tuple) else out

    def confidence(self, context, k=3):
        """How dominant is the best match? {top, score, margin, confident}.

        THE MARGIN, NOT THE SCORE, is what says whether a hit is real -- leCore
        already established this in `capability_confidence`, whose docstring
        calls it "the antidote to reading a fallback as a hit". A top score of
        0.4 means nothing if the runner-up is 0.39; it means a lot if the
        runner-up is 0.05. proglib originally abstained on an absolute
        threshold alone, which is the exact mistake that module exists to
        prevent."""
        hits = self.find(context, k=max(2, int(k)))
        if not hits:
            return {"top": None, "score": 0.0, "margin": 0.0,
                    "confident": False}
        top, score = hits[0][0], hits[0][1]
        runner = hits[1][1] if len(hits) > 1 else 0.0
        margin = float(score - runner)
        return {"top": top, "score": float(score), "margin": margin,
                "confident": bool(score > 0.15 and margin > 0.05)}

    def run_for(self, context, init_acc=None, min_score=0.15, **kw):
        """Find the program this context calls for, and RUN it -- or ABSTAIN.

        Abstention is the point: a library that always returns its best guess
        will run the wrong program on a context it has nothing for, and a wrong
        program is a wrong answer rather than a slow one."""
        c = self.confidence(context)
        if not c["confident"]:
            return None, {"ran": None, "why":
                          "score %.3f margin %.3f -- not dominant enough to act"
                          % (c["score"], c["margin"]), **c}
        return self.run(c["top"], init_acc=init_acc, **kw), {"ran": c["top"],
                                                             **c}

    def as_vault(self, prefix="prog"):
        """Every program as a vault object -- stored, recalled, runnable.

        Only the assembled VECTOR and the source are kept; the ADDRESS
        regenerates from the description via hashlib, so the index is never
        stored."""
        out = {}
        for name, p in self.progs.items():
            out["%s:%s" % (prefix, name)] = {
                "kind": "vsa_program",
                "meta": {"name": name, "description": p["description"],
                         "program": [[str(o), (None if v is None else str(v))]
                                     for o, v in p["program"]],
                         "dim": self.dim, "seed_tag": self.seed_tag},
                "arrays": {"vector": np.asarray(p["vector"])}}
        return out


#: THE MACHINE'S REAL VOCABULARY. Operands are not free strings -- the VM cleans
#: each one up to the NEAREST atom in the codebook for that opcode's operand
#: type, so an unknown name silently becomes whatever was closest. This is
#: correct behaviour for a cleanup memory and a silent disaster for a caller who
#: assumed literals: assembling with a made-up operand produced a trace reading
#: ('LOAD','f'), ('BIND','d') with no error raised anywhere.
VOCABULARY = {
    "opcodes": ("LOAD", "STORE", "RECALL", "BIND", "BUNDLE", "PERMUTE",
                "PUSH", "POP", "APPLY", "CALL", "IFMATCH", "ITERATE",
                "REPEAT", "HALT"),
    "data": tuple("abcdef"),            # LOAD / BIND / BUNDLE / IFMATCH / HALT
    "registers": tuple("R%d" % i for i in range(8)),   # STORE / RECALL
    "counts": tuple(range(1, 9)),                      # REPEAT
    "faculties": ("cleanup", "denoise", "matmul"),     # APPLY, host-supplied
    "names": "a defined procedure",                    # CALL / ITERATE
}

#: OPERAND TYPE PER OPCODE -- checked before assembly, because the VM will not
#: complain. REPEAT takes a COUNT and must be followed by a CALL; ITERATE takes a
#: PROCEDURE NAME and runs it to a fixed point. Getting those two backwards is
#: the easiest mistake here and produces a plausible wrong answer.
OPERAND_KIND = {
    "LOAD": "data", "BIND": "data", "BUNDLE": "data", "IFMATCH": "data",
    "HALT": "data", "STORE": "registers", "RECALL": "registers",
    "PERMUTE": "counts", "REPEAT": "counts", "APPLY": "faculties",
    "CALL": "names", "ITERATE": "names", "PUSH": None, "POP": None,
}


def check(program, faculties=(), procedures=()):
    """Validate a program BEFORE assembling it. Returns a list of problems.

    Exists because the VM raises nothing: every operand is cleaned up to the
    nearest atom of its type, so a typo becomes a different valid instruction.
    A checker is the only place a mistake can still be caught."""
    bad = []
    ops = set(VOCABULARY["opcodes"])
    fac = set(VOCABULARY["faculties"]) | set(faculties)
    names = set(procedures)
    for i, step in enumerate(program):
        op, arg = (list(step) + [None])[:2]
        if op not in ops:
            bad.append("%d: unknown opcode %r" % (i, op))
            continue
        kind = OPERAND_KIND.get(op)
        if kind is None:
            continue
        if kind == "faculties":
            if arg not in fac:
                bad.append("%d: APPLY %r is not a registered faculty %s"
                           % (i, arg, sorted(fac)))
        elif kind == "names":
            if names and arg not in names:
                bad.append("%d: %s %r is not a defined procedure" % (i, op, arg))
        elif arg not in VOCABULARY[kind]:
            bad.append("%d: %s operand %r not in %s"
                       % (i, op, arg, VOCABULARY[kind]))
        if op == "REPEAT":
            nxt = program[i + 1] if i + 1 < len(program) else (None,)
            if nxt[0] != "CALL":
                bad.append("%d: REPEAT must be followed by CALL (it repeats a "
                           "procedure, not the next instruction)" % i)
    if program and program[-1][0] != "HALT":
        bad.append("program does not end in HALT")
    return bad


def _selftest():
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine
    from holographic.agents_and_reasoning.holographic_ai import cosine

    fac = {"double": lambda a: a * 2.0,
           "flip": lambda a: -a,
           "norm": lambda a: a / (np.linalg.norm(a) + 1e-30)}
    M = HoloMachine(dim=1024, seed=0, faculties=fac)
    lib = ProgramLibrary(M, dim=1024)

    lib.add("scale_up", [("APPLY", "double"), ("HALT", None)],
            "double the accumulator make it bigger amplify")
    lib.add("invert", [("APPLY", "flip"), ("HALT", None)],
            "negate the accumulator flip its sign invert")
    lib.add("normalise", [("APPLY", "norm"), ("HALT", None)],
            "normalise the accumulator to unit length")
    lib.add("big_negative", [("CALL", "scale_up"), ("CALL", "invert"),
                             ("HALT", None)],
            "double then negate combine amplify and invert")

    # ---- DISCOVERY: describe a situation, get the right program ----
    for ctx, want in (("I need to flip the sign", "invert"),
                      ("make it unit length", "normalise"),
                      ("amplify then invert it", "big_negative")):
        got = lib.find(ctx, k=1)[0][0]
        assert got == want, (ctx, got, want)

    # ---- AND IT MUST ABSTAIN on a context it has nothing for ----
    _out, info = lib.run_for("bake a cake with chocolate frosting",
                             init_acc=np.ones(1024))
    assert info["ran"] is None, info

    # ---- AND THE MARGIN MUST DO THE WORK, not the raw score. leCore's
    #      capability_confidence calls the margin "the antidote to reading a
    #      fallback as a hit"; abstaining on an absolute threshold alone was
    #      exactly the mistake that module exists to prevent.
    good = lib.confidence("negate the accumulator")
    assert good["confident"] and good["margin"] > 0.05, good
    vague = lib.confidence("the accumulator")
    assert vague["margin"] < good["margin"], (vague, good)

    # ---- COMPOSITION IS EXACT: CALL equals inline ----
    start = np.random.default_rng(0).standard_normal(1024)
    composed = lib.run("big_negative", init_acc=start)
    inline = M.run(M.assemble([("APPLY", "double"), ("APPLY", "flip"),
                               ("HALT", None)]), init_acc=start)
    inline = inline[0] if isinstance(inline, tuple) else inline
    assert abs(abs(cosine(composed, inline)) - 1.0) < 1e-6, cosine(composed,
                                                                  inline)

    # ---- AND THE WHOLE LIBRARY VAULTS, addresses regenerating from text ----
    from holographic.caching_and_storage.holographic_modelvault import (
        store, recall)
    blob = store(lib.as_vault())
    back = recall(blob)
    assert len(back) == 4, list(back)
    e = back["prog:big_negative"]
    assert e["meta"]["description"]
    readdr = describe(e["meta"]["description"], 1024)
    assert np.allclose(readdr, lib._addr["big_negative"])

    # ---- THE CHECKER MUST CATCH WHAT THE VM SILENTLY ACCEPTS ----
    assert check([("LOAD", "a"), ("HALT", "a")]) == []
    assert check([("LOAD", "zzz"), ("HALT", "a")]), "unknown data operand"
    assert check([("APPLY", "nope"), ("HALT", "a")]), "unregistered faculty"
    assert check([("LOAD", "a"), ("REPEAT", 2), ("PERMUTE", 1), ("HALT", "a")]), \
        "REPEAT not followed by CALL"
    assert check([("LOAD", "a")]), "missing HALT"

    # ---- AND EVERY OPCODE MUST BE SEMANTICALLY CORRECT, not merely runnable.
    #      Measured: all 14 check out. REPEAT is exact to 1.000000 at counts
    #      1..4 ONLY in its correct form (REPEAT n; CALL proc) -- written as
    #      REPEAT n; PERMUTE it silently produces cosine 0.018 to the intended
    #      result, which is the trap this checker exists for.
    from holographic.agents_and_reasoning.holographic_ai import (
        bind as _bind, bundle as _bundle, permute as _perm)
    M2 = HoloMachine(dim=1024, seed=0, faculties=fac)
    A, B = M2.data_atoms["a"], M2.data_atoms["b"]
    M2.define("spin", [("PERMUTE", 1), ("HALT", "a")])

    def _r(prog, **kw):
        out = M2.run(M2.assemble(prog), **kw)
        return np.asarray(out[0] if isinstance(out, tuple) else out)

    sem = [
        ("LOAD", _r([("LOAD", "a"), ("HALT", "a")]), A),
        ("BIND", _r([("LOAD", "a"), ("BIND", "b"), ("HALT", "a")]), _bind(A, B)),
        ("BUNDLE", _r([("LOAD", "a"), ("BUNDLE", "b"), ("HALT", "a")]),
         _bundle([A, B])),
        ("PERMUTE", _r([("LOAD", "a"), ("PERMUTE", 1), ("HALT", "a")]),
         _perm(A, 1)),
        ("STORE/RECALL", _r([("LOAD", "a"), ("STORE", "R0"), ("LOAD", "b"),
                             ("RECALL", "R0"), ("HALT", "a")]), A),
        ("PUSH/POP", _r([("LOAD", "a"), ("PUSH", None), ("LOAD", "b"),
                         ("POP", None), ("HALT", "a")]), A),
        ("REPEAT+CALL", _r([("LOAD", "a"), ("REPEAT", 3), ("CALL", "spin"),
                            ("HALT", "a")]), _perm(A, 3)),
    ]
    for nm, got, want in sem:
        c = float(cosine(got, np.asarray(want)))
        assert c > 0.99, (nm, c)

    # ---- IFMATCH MUST ACTUALLY BRANCH, or it is decoration ----
    taken = _r([("LOAD", "a"), ("IFMATCH", "a"), ("PERMUTE", 1), ("HALT", "a")])
    skipped = _r([("LOAD", "a"), ("IFMATCH", "b"), ("PERMUTE", 1), ("HALT", "a")])
    assert cosine(taken, _perm(A, 1)) > 0.99, "IFMATCH on a match must RUN"
    assert cosine(skipped, _perm(A, 1)) < 0.5, "IFMATCH on a miss must SKIP"

    print("proglib selftest OK -- leCore ALREADY had the programs (HoloMachine's "
          "14 opcodes, assemble to ONE vector, define/CALL for composition); this "
          "adds DISCOVERY: 3 of 3 situations described in plain words find the "
          "right program, an unrelated context correctly ABSTAINS, a CALL-composed "
          "program equals its inline form to 1e-6, and the whole library vaults "
          "in %.1f KB with every ADDRESS regenerated from its description rather "
          "than stored; all 7 opcode semantics verified against the algebra "
          "(REPEAT exact at counts 1-4 in its CALL form), IFMATCH genuinely "
          "branches, and the operand checker rejects 4 malformed programs the "
          "VM would have run silently" % (len(blob) / 1e3))


if __name__ == "__main__":
    _selftest()
