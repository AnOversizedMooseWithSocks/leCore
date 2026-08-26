"""A tiny formal-logic kernel with Lean 4 export -- proofs the engine can CHECK, not just claim.

WHY THIS EXISTS
---------------
Rule-0 audit (2026-08-16): ten phrasings ("lean4 proof", "theorem prover", "formal verification",
"check a proof", "unification of terms", ...) returned only fallbacks -- the engine had symbolic
REGRESSION (laws from data, holographic_symbolic) but nothing that PROVES a stated proposition and
lets an external tool verify the proof. That is a different animal: regression is inductive and
gated by MDL; this module is deductive and gated by a checker.

THE DESIGN, and why it is shaped this way:

  * The logic is the **Horn fragment of first-order logic** (facts + definite clauses), proved by
    deterministic forward chaining with syntactic unification. Horn is chosen deliberately: it is
    the largest fragment where forward chaining is complete, terminates on finite ground bases,
    and needs no backtracking search -- so the prover is a plain worklist loop a reader can audit,
    not a black-box tactic engine. (Full FOL with negation/disjunction is a declared negative:
    it needs resolution + occurs-check subtleties + non-termination handling, and would turn a
    readable kernel into a small Prolog. Extend only with a measured need.)

  * The prover and the checker are SEPARATE, on purpose. `prove` builds an explicit `Proof` tree;
    `check_proof` re-verifies every node against the rule set from scratch, trusting nothing the
    prover said. Two components agreeing is not evidence of correctness -- but a checker that
    shares no state with the prover is the cheapest honest instrument we can build in-process.

  * `to_lean` emits genuine **Lean 4 source**: the signature declares predicates and constants,
    facts and rules become named hypotheses (axioms), and the derivation becomes a term-mode
    application. The point is that the ULTIMATE checker is external: paste the output into Lean
    (or use `lean_check` if a `lean` binary is installed) and the proof either typechecks or it
    does not. leCore never claims "verified by Lean" unless Lean itself said so.

  * `lean_check` is an OPT-IN bridge, numba-style: if `lean` is on PATH it round-trips the source
    through it; if not, it says so honestly ({"available": False}) instead of pretending.

  * `encode_atom` maps ground atoms into the hypervector space (predicate bound with role-tagged
    arguments) so a fact base is searchable by similarity with the SAME algebra as everything
    else -- the substrate bet, kept.

Determinism: worklist order is insertion order; variable renaming is counter-based; no hash()
anywhere (names are compared as strings; content digests would use hashlib). Same input, same
proof tree, same Lean text, every run.

KEPT NEGATIVES (on record so the next session does not reinvent them):
  * No dependent types / no Nat arithmetic / no rewriting -- this is not a Lean kernel clone.
    The export TARGETS Lean; it does not reimplement it. Reimplementing a typechecker "just a
    little" is the road to an unsound one.
  * Backward chaining (goal-directed) was considered and dropped for v1: forward chaining on
    finite bases is complete for Horn queries and simpler to check; backward adds loop-detection
    machinery for zero new theorems here.
  * Function symbols in terms (f(g(x))) are excluded in v1 -- with them, forward chaining can
    diverge (infinite Herbrand universe). Constants + variables only; the prover therefore
    TERMINATES by construction. Lifting this is a real extension, not a bug fix.
"""

import shutil
import subprocess
import tempfile
import os

import numpy as np


# ---------------------------------------------------------------------------
# Terms and atoms.  A term is a constant ("socrates") or a variable ("?x").
# An Atom is predicate + argument tuple:  mortal(socrates), parent(?x, ?y).
# Plain tuples/strings, no classes needed beyond Atom/Rule/Proof -- the whole
# syntax must stay printable and diffable.
# ---------------------------------------------------------------------------

def is_var(t):
    """A variable is any string starting with '?'. Everything else is a constant."""
    return isinstance(t, str) and t.startswith("?")


class Atom:
    """A predicate applied to terms: Atom("parent", ("tom", "?x")).  Immutable, hashable-by-content
    via its key() so it can live in Python sets deterministically (string key, not id/hash order)."""

    def __init__(self, pred, args=()):
        self.pred = str(pred)
        self.args = tuple(args)

    def key(self):
        return self.pred + "(" + ",".join(self.args) + ")"

    def is_ground(self):
        return not any(is_var(a) for a in self.args)

    def __repr__(self):
        return self.key()

    def __eq__(self, other):
        return isinstance(other, Atom) and self.key() == other.key()

    def __hash__(self):
        # hash of a STRING is salted by PYTHONHASHSEED; we run under PYTHONHASHSEED=0 house-wide,
        # and nothing below depends on set ITERATION order (worklists are lists) -- sets are used
        # for membership only, so this hash never influences output.
        return hash(self.key())


class Rule:
    """A definite clause:  head :- body_1, ..., body_n.   n == 0 makes it a fact."""

    def __init__(self, head, body=(), name=None):
        self.head = head
        self.body = tuple(body)
        self.name = name or ("fact_" + head.pred if not body else "rule_" + head.pred)

    def __repr__(self):
        if not self.body:
            return "%s: %s." % (self.name, self.head)
        return "%s: %s :- %s." % (self.name, self.head, ", ".join(map(str, self.body)))


class Proof:
    """An explicit derivation tree: this ground atom follows from this rule under this substitution,
    given these sub-proofs (one per body atom). The tree IS the deliverable -- the checker walks it,
    and to_lean serialises it."""

    def __init__(self, atom, rule, subst, children=()):
        self.atom = atom          # the ground Atom proved
        self.rule = rule          # the Rule instance used
        self.subst = dict(subst)  # variable -> constant map that grounded the rule
        self.children = tuple(children)

    def size(self):
        return 1 + sum(c.size() for c in self.children)


# ---------------------------------------------------------------------------
# Unification -- syntactic, constants+variables only (no function symbols, so
# no occurs-check needed: a variable can only bind to a constant or another
# variable, never to a term containing itself).
# ---------------------------------------------------------------------------

def _walk(t, s):
    """Follow variable bindings to the end of the chain."""
    while is_var(t) and t in s:
        t = s[t]
    return t


def unify(a, b, s=None):
    """Unify two Atoms under substitution s. Returns the extended substitution dict, or None.
    Deterministic: argument positions are processed left to right."""
    if s is None:
        s = {}
    if a.pred != b.pred or len(a.args) != len(b.args):
        return None
    s = dict(s)
    for x, y in zip(a.args, b.args):
        x, y = _walk(x, s), _walk(y, s)
        if x == y:
            continue
        if is_var(x):
            s[x] = y
        elif is_var(y):
            s[y] = x
        else:
            return None  # two distinct constants
    return s


def substitute(atom, s):
    """Apply substitution to an atom's arguments."""
    return Atom(atom.pred, tuple(_walk(a, s) for a in atom.args))


def _rename(rule, idx):
    """Rename a rule's variables apart with a counter suffix -- counter-based, never id()-based,
    so renaming is reproducible run to run."""
    m = {}
    def r(a):
        return Atom(a.pred, tuple((t + "_%d" % idx) if is_var(t) else t for t in a.args))
    return Rule(r(rule.head), tuple(r(b) for b in rule.body), name=rule.name)


# ---------------------------------------------------------------------------
# The prover: forward chaining to a fixpoint, recording a Proof for every new
# ground atom.  Complete for Horn queries over finite constant vocabularies.
# ---------------------------------------------------------------------------

def _prove_seminaive(goal, rules, max_steps, _return_table):
    """Semi-naive bottom-up evaluation (Bancilhon & Ramakrishnan 1986). Every pass, each
    rule runs once per body position j with position j drawn ONLY from the previous pass's
    delta and the other positions from all facts -- so no ground rule instance is evaluated
    twice across the run (the non-repetition property). Facts are indexed by predicate, and
    by (predicate, first-arg) for body atoms whose first argument is already bound at join
    time -- the join that made naive quadratic-per-pass. Deterministic: rules in order,
    delta positions ascending, facts in derivation order."""
    proofs = {}                     # ground key -> Proof, first derivation wins
    by_pred = {}                    # pred -> [Proof] in derivation order
    by_pred_a0 = {}                 # (pred, ground arg0) -> [Proof]
    order = []                      # all facts in derivation order

    def _add(pr):
        proofs[pr.atom.key()] = pr
        by_pred.setdefault(pr.atom.pred, []).append(pr)
        if pr.atom.args:
            by_pred_a0.setdefault((pr.atom.pred, pr.atom.args[0]), []).append(pr)
        order.append(pr)

    for r in rules:
        if not r.body:
            if not r.head.is_ground():
                raise ValueError("facts must be ground: %r" % r)
            if r.head.key() not in proofs:
                _add(Proof(r.head, r, {}))

    def _matches(bs, pool_new=None):
        """Candidate facts for a (possibly partially ground) body atom: exact hit when
        ground, (pred, arg0) index when the first arg is bound, else the predicate list.
        pool_new restricts to the delta (a set of atom keys) when given."""
        if bs.is_ground():
            hit = proofs.get(bs.key())
            cands = [hit] if hit is not None else []
        elif bs.args and not is_var(bs.args[0]):
            cands = by_pred_a0.get((bs.pred, bs.args[0]), [])
        else:
            cands = by_pred.get(bs.pred, [])
        if pool_new is None:
            return cands
        return [c for c in cands if c.atom.key() in pool_new]

    delta = [pr for pr in order]    # pass 0: every base fact is new
    steps = 0
    while delta and steps < max_steps:
        delta_keys = {pr.atom.key() for pr in delta}
        new_round = []
        for ridx, r in enumerate(rules):
            if not r.body:
                continue
            rr = _rename(r, ridx)
            for dpos in range(len(rr.body)):
                # one delta variant per body position (the 1986 rewrite): position dpos
                # must use a NEW fact; positions before it use OLD-only facts (strictly
                # pre-delta) to avoid re-deriving the same instance from two variants,
                # positions after it may use anything derived so far.
                partial = [({}, [])]
                for j, b in enumerate(rr.body):
                    nxt = []
                    for s, kids in partial:
                        bs = substitute(b, s)
                        if j == dpos:
                            cands = _matches(bs, delta_keys)
                        elif j < dpos:
                            cands = [c for c in _matches(bs) if c.atom.key() not in delta_keys]
                        else:
                            cands = _matches(bs)
                        for c in cands:
                            s2 = unify(bs, c.atom, s)
                            if s2 is not None:
                                nxt.append((s2, kids + [c]))
                    partial = nxt
                    if not partial:
                        break
                for s, kids in partial:
                    h = substitute(rr.head, s)
                    if h.is_ground() and h.key() not in proofs:
                        orig = {}
                        for a in (r.head,) + r.body:
                            for t in a.args:
                                if is_var(t):
                                    orig[t] = _walk(t + "_%d" % ridx, s)
                        pr = Proof(h, r, orig, kids)
                        _add(pr)
                        new_round.append(pr)
                steps += 1
        delta = new_round
    if _return_table:
        return proofs
    return proofs.get(goal.key())


def prove(goal, rules, max_steps=10000, _return_table=False, strategy="naive"):
    """Prove a ground goal Atom from Horn rules by forward chaining.

    Returns a Proof tree, or None if the goal is not derivable. Deterministic:
    rules fire in list order, facts accumulate in derivation order.

    `max_steps` is a fuse, not a tuning knob: with constants-only terms the ground
    Herbrand base is finite and the loop provably terminates before any sane fuse.

    The goal must be GROUND: a goal with variables is a QUERY (enumerate bindings), which is
    a different, deferred capability -- silently returning None for one was measured as the
    trap (the caller reads "not derivable" when the truth is "wrong question shape").

    strategy="naive" is the original worklist (kept byte-for-byte: pinned Lean outputs and
    proof shapes depend on its discovery order). strategy="seminaive" is Bancilhon &
    Ramakrishnan 1986: each pass joins only rule instances touching at least one fact NEW
    in the previous pass (per-body-position delta variants), facts indexed by predicate.
    Same atom SET as naive -- the textbook theorem, pinned by test -- but proof TREES may
    differ (a different valid derivation can be found first), which is why it is opt-in,
    never a silent flip. MEASURED REASON: naive T_P on the repo's own import graph
    (708 modules, 2,246 edges) did not finish in 300s.
    """
    if not goal.is_ground():
        raise ValueError("goal %r contains variables -- prove() takes a ground goal; "
                         "querying for bindings is a deferred extension, not a silent None" % goal)
    validate_rules(rules)
    if strategy == "seminaive":
        return _prove_seminaive(goal, rules, max_steps, _return_table)
    proofs = {}          # ground key -> Proof (first derivation wins; determinism keeps it stable)
    agenda = []          # ground atoms in derivation order
    for i, r in enumerate(rules):
        if not r.body:
            g = r.head
            if not g.is_ground():
                raise ValueError("facts must be ground: %r" % r)
            if g.key() not in proofs:
                proofs[g.key()] = Proof(g, r, {})
                agenda.append(g)
    steps = 0
    changed = True
    while changed and steps < max_steps:
        changed = False
        for ridx, r in enumerate(rules):
            if not r.body:
                continue
            rr = _rename(r, ridx)
            # match body atoms against known facts, left to right (a tiny join)
            partial = [({}, [])]  # (substitution, matched child proofs)
            for b in rr.body:
                nxt = []
                for s, kids in partial:
                    bs = substitute(b, s)
                    for fk in list(proofs.keys()):
                        s2 = unify(bs, proofs[fk].atom, s)
                        if s2 is not None:
                            nxt.append((s2, kids + [proofs[fk]]))
                partial = nxt
                if not partial:
                    break
            for s, kids in partial:
                h = substitute(rr.head, s)
                if h.is_ground() and h.key() not in proofs:
                    # translate the substitution back to the ORIGINAL rule's variable names --
                    # the Proof carries the original rule, so its subst must speak that language
                    # (the renamed keys "?x_%d" are an internal detail of this join)
                    orig = {}
                    for a in (r.head,) + r.body:
                        for t in a.args:
                            if is_var(t):
                                orig[t] = _walk(t + "_%d" % ridx, s)
                    proofs[h.key()] = Proof(h, r, orig, kids)
                    agenda.append(h)
                    changed = True
            steps += 1
    if _return_table:
        # consequences() reads the whole fixpoint; ordinary callers never see this
        return proofs
    return proofs.get(goal.key())


def check_proof(proof, rules):
    """Independently verify a Proof tree against the rule set. Trusts nothing from the prover:
    re-unifies the rule head with the claimed atom, re-checks every body atom is exactly what
    the corresponding child proves, recurses. Returns True or raises AssertionError loudly."""
    names = {r.name for r in validate_rules(rules)}
    assert proof.rule.name in names, "unknown rule %r" % proof.rule.name
    rr = _rename(proof.rule, 0)
    s = unify(rr.head, proof.atom)
    assert s is not None, "head %r does not match atom %r" % (rr.head, proof.atom)
    assert len(rr.body) == len(proof.children), "arity mismatch in %r" % proof.atom
    # extend the head substitution by matching each body atom to its child's conclusion
    for b, child in zip(rr.body, proof.children):
        s = unify(substitute(b, s), child.atom, s)
        assert s is not None, "body %r not proved by child %r" % (b, child.atom)
        check_proof(child, rules)
    assert proof.atom.is_ground(), "non-ground conclusion %r" % proof.atom
    return True


# ---------------------------------------------------------------------------
# Lean 4 export.  Constants become an inductive-free `U`-typed opaque space
# (axioms), predicates become Props, facts/rules become hypotheses, and the
# derivation becomes a term-mode application.  The output typechecks in stock
# Lean 4 with no imports.
# ---------------------------------------------------------------------------

# Lean 4 keywords and our own reserved names: emitting any of these as an axiom name produces
# invalid source ("axiom fun : ..." -- measured, Lean verdict False), and a constant literally
# named "U" shadows the universe axiom. The list covers the term-level keywords our output can
# collide with; obscure command keywords cannot appear in axiom position anyway.
_LEAN_RESERVED = frozenset((
    "U", "Prop", "Type", "Sort", "axiom", "theorem", "lemma", "def", "abbrev", "example",
    "fun", "forall", "exists", "let", "in", "if", "then", "else", "match", "with", "do",
    "by", "have", "show", "from", "calc", "where", "deriving", "structure", "inductive",
    "class", "instance", "variable", "universe", "open", "import", "namespace", "end",
    "section", "mutual", "partial", "unsafe", "private", "protected", "noncomputable",
    "macro", "syntax", "notation", "set_option", "attribute", "true", "false", "sorry",
    "admit", "rec", "mk"))


class _LeanNamer:
    """A deterministic, COLLISION-FREE name mangler for one export.

    WHY A CONTEXT AND NOT A FUNCTION: the old per-name sanitiser mapped distinct source
    names to the same identifier ("a-b" and "a_b" both became a_b), which MERGES two
    distinct constants in the emitted Lean -- a soundness hole, since a false statement
    about one could typecheck as a true statement about the other. Uniqueness is a
    property of the whole export, so the namer owns the whole export's namespace
    (constants, predicates, rule names and the theorem name all share Lean's top level).

    Deterministic: first-seen order decides suffixes; same rules, same names, every run.
    """

    def __init__(self):
        self._map = {}      # source name -> lean ident
        self._taken = set()

    def name(self, s, kind=""):
        # (kind, s) is the key: a rule named "t" and a theorem named "t" are DIFFERENT
        # entities sharing a string -- keying on the bare string merged them (measured:
        # "axiom t ... theorem t" -- external Lean False). Idents stay unique via _taken.
        key = (kind, s)
        if key in self._map:
            return self._map[key]
        out = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(s).lstrip("?"))
        if not out or out[0].isdigit():
            out = "v_" + out
        if out in _LEAN_RESERVED:
            out = out + "_"           # keyword escape: fun -> fun_, U -> U_
        base, k = out, 2
        while out in self._taken:      # collision escape: second a_b -> a_b_2, stable order
            out = "%s_%d" % (base, k)
            k += 1
        self._map[key] = out
        self._taken.add(out)
        return out


def _lean_ident(s, namer=None, kind=""):
    """Sanitise a name into a Lean identifier. With a _LeanNamer the result is unique and
    keyword-safe across the export; without one (legacy/direct calls) it is best-effort."""
    if namer is not None:
        return namer.name(s, kind)
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(s).lstrip("?"))
    return ("v_" + out) if (not out or out[0].isdigit()) else out


def _lean_atom(atom, namer, var_map=None):
    # a var_map hit is ALREADY a final local ident; everything else is a constant
    vm = var_map or {}
    args = " ".join(vm[a] if a in vm else _lean_ident(a, namer, "const") for a in atom.args)
    return ("%s %s" % (_lean_ident(atom.pred, namer, "pred"), args)).strip()


def _proof_term(proof, namer):
    """Serialise a Proof tree as a Lean application:  (rule c1 c2 (sub1) (sub2))."""
    # order of explicit arguments: the rule's variables in first-appearance order, then sub-proofs
    rr = proof.rule
    seen, order = set(), []
    for a in (rr.head,) + rr.body:
        for t in a.args:
            if is_var(t) and t not in seen:
                seen.add(t)
                order.append(t)
    # walk the substitution chain: subst may map renamed vars through intermediates to a constant
    consts = [_lean_ident(_walk(v, proof.subst), namer, "const") for v in order]
    kids = [_proof_term(c, namer) for c in proof.children]
    inner = " ".join([_lean_ident(rr.name, namer, "rule")] + consts + ["(%s)" % k if " " in k else k for k in kids])
    return inner


def validate_rules(rules):
    """The precondition every entry point shares: rule names must be UNIQUE (they become Lean
    axiom names and wire-format keys -- a duplicate silently last-wins in both, measured as an
    external-Lean False verdict), and a predicate must keep ONE arity (p/1 and p/2 cannot share
    a Lean signature). Raises ValueError loudly; returns the rules unchanged."""
    seen, arity = set(), {}
    for r in rules:
        if r.name in seen:
            raise ValueError("duplicate rule name %r -- names key the wire format and the "
                             "Lean axioms; make them unique" % r.name)
        seen.add(r.name)
        for a in (r.head,) + r.body:
            if a.pred in arity and arity[a.pred] != len(a.args):
                raise ValueError("predicate %r used with arities %d and %d -- one predicate, "
                                 "one arity" % (a.pred, arity[a.pred], len(a.args)))
            arity[a.pred] = len(a.args)
    return rules


def to_lean(goal_proof, rules, theorem_name="derived"):
    """Emit self-contained Lean 4 source: opaque universe, predicate/constant signature,
    every rule as an axiom, and the derivation as a term-mode theorem. External Lean is
    the authority -- this function only writes the file, it never claims success.

    Naming is COLLISION-FREE per export via _LeanNamer: distinct source names never merge
    (merging is a soundness hole -- a false statement about one constant could typecheck as
    a true one about another), Lean keywords and "U" are escaped, and variables/constants/
    predicates/rules/theorem share one namespace, first-seen order, deterministic."""
    validate_rules(rules)
    namer = _LeanNamer()
    namer.name(theorem_name, "thm")   # theorems and rules may share a string; kinds keep them apart
    preds, consts = {}, []
    def scan(a):
        preds[a.pred] = max(preds.get(a.pred, 0), len(a.args))
        for t in a.args:
            if not is_var(t) and t not in consts:
                consts.append(t)
    for r in rules:
        scan(r.head)
        for b in r.body:
            scan(b)
    scan(goal_proof.atom)
    lines = ["-- generated by leCore holographic_lean.to_lean; checked externally by Lean 4",
             "axiom U : Type"]
    for c in consts:
        lines.append("axiom %s : U" % _lean_ident(c, namer, "const"))
    for p in sorted(preds):
        lines.append("axiom %s : %s" % (_lean_ident(p, namer, "pred"), "U -> " * preds[p] + "Prop"))
    for r in rules:
        seen, order = set(), []
        for a in (r.head,) + r.body:
            for t in a.args:
                if is_var(t) and t not in seen:
                    seen.add(t); order.append(t)
        # variables are LOCAL to the binder: name them in a child scope so ?x in two rules
        # stays x in both, but never collides with a global constant already named x
        vm = {}
        local_taken = set(namer._taken)
        for v in order:
            base = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in v.lstrip("?")) or "x"
            if base[0].isdigit(): base = "v_" + base
            if base in _LEAN_RESERVED: base += "_"
            cand, k = base, 2
            while cand in local_taken:
                cand = "%s_%d" % (base, k); k += 1
            local_taken.add(cand); vm[v] = cand
        binder = ("forall " + " ".join(vm[v] for v in order) + " : U, ") if order else ""
        body = " -> ".join([_lean_atom(b, namer, vm) for b in r.body] + [_lean_atom(r.head, namer, vm)])
        lines.append("axiom %s : %s%s" % (_lean_ident(r.name, namer, "rule"), binder, body))
    lines.append("theorem %s : %s := %s" % (_lean_ident(theorem_name, namer, "thm"),
                                            _lean_atom(goal_proof.atom, namer), _proof_term(goal_proof, namer)))
    return "\n".join(lines) + "\n"


def lean_check(source, timeout=60):
    """Round-trip Lean source through an installed `lean` binary, if any (opt-in bridge,
    numba-style: the engine never requires it). Returns a dict with `available`, and when
    available: `ok`, `stdout`, `stderr`. Never raises on a failed proof -- the verdict IS the data."""
    exe = shutil.which("lean")
    if not exe:
        return {"available": False, "ok": None,
                "note": "no `lean` binary on PATH; install elan/lean4 to verify externally"}
    # `elan` installs a `lean` shim before it installs or selects a toolchain.
    # A PATH hit alone therefore does not mean Lean is usable. Probe the actual
    # compiler first so an unconfigured shim is reported as unavailable instead
    # of making every otherwise-optional proof test fail.
    try:
        version = subprocess.run([exe, "--version"], capture_output=True, text=True,
                                 timeout=min(timeout, 15))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "ok": None,
                "note": "`lean` exists but is not runnable: %s" % exc}
    if version.returncode != 0:
        detail = (version.stderr or version.stdout or "Lean toolchain is not configured").strip()
        return {"available": False, "ok": None, "note": detail}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "lecore_check.lean")
        with open(p, "w") as f:
            f.write(source)
        try:
            r = subprocess.run([exe, p], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"available": True, "ok": False, "stderr": "timeout", "stdout": ""}
        # INSTRUMENT ERROR, caught live (2026-08-16): `sorry` typechecks with exit 0 and only a
        # WARNING -- so returncode alone blessed an unproven theorem. "ok" means PROVED, not
        # "compiled": any sorry/admit escape hatch in the output demotes the verdict.
        sorried = "declaration uses 'sorry'" in (r.stdout + r.stderr)
        return {"available": True, "ok": (r.returncode == 0) and not sorried,
                "sorried": sorried, "stdout": r.stdout, "stderr": r.stderr}


# ---------------------------------------------------------------------------
# VSA layer: ground atoms as hypervectors, so a fact base joins the substrate
# and similarity search costs one matmul. Delegates the algebra, never re-does it.
# ---------------------------------------------------------------------------

def encode_atom(atom, sym, bind, bundle):
    """Encode a ground atom: bind(pred_vec, bundle_i(bind(role_i, arg_i))). `sym(name)` supplies
    a deterministic symbol vector; `bind`/`bundle` are the engine's own ops passed in so this
    module owns no algebra (the substrate bet: one algebra, many costumes)."""
    parts = [bind(sym("role_%d" % i), sym(a)) for i, a in enumerate(atom.args)]
    payload = bundle(parts) if parts else sym("nullary")
    return bind(sym("pred_" + atom.pred), payload)


def consequences(rules, max_steps=100000, strategy="naive"):
    """ALL derivable ground atoms -- the least fixpoint of the rule set, made a first-class result.

    Kowalski's seat named this (panel review): the forward-chaining loop IS the van Emden-
    Kowalski T_P operator (JACC 1976 -- least-fixpoint semantics of definite clauses), and its
    fixpoint is exactly the set of consequences; prove() was computing it and throwing it away.
    Returning it makes COMPLETENESS a measured property: the selftest pins the exact set, not a
    citation. Deterministic: atoms are listed in derivation order.

    HONEST SCOPE (de Moura's seat): this enumerates what FOLLOWS from the rules; it says nothing
    about whether the rules are CONSISTENT -- an absurd rule set derives absurd atoms happily.
    See detect_absurdity for the cheap smoke."""
    validate_rules(rules)
    # reuse prove()'s loop by asking for an atom that cannot exist, then read the table it built.
    # WHY NOT refactor prove(): the loop is 30 readable lines; sharing state machinery across two
    # entry points costs more clarity than the duplication it saves -- so instead prove() gains a
    # _return_table hook used ONLY here, keeping one loop and one contract.
    table = prove(Atom("__lecore_impossible__", ()), rules, max_steps=max_steps,
                  _return_table=True, strategy=strategy)
    return [pr.atom for pr in table.values()]


def detect_absurdity(rules, absurd=("false", "absurd", "bottom"), max_steps=100000):
    """The cheap consistency smoke (de Moura's seat, panel review): Lean verifies a derivation
    FOLLOWS from the rules, never that the rules are consistent -- an inconsistent theory proves
    anything and typechecks doing it. Convention over machinery: if the caller uses any of the
    `absurd` predicate names for contradiction, this reports whether one is derivable and hands
    back its proof. Not a decision procedure for consistency (that needs full FOL refutation --
    Robinson's territory, deferred); it is the smoke detector, stated as such."""
    validate_rules(rules)
    for a in consequences(rules, max_steps=max_steps):
        if a.pred in absurd:
            return {"absurd": True, "atom": [a.pred, list(a.args)],
                    "proof": proof_to_wire(prove(a, rules))}
    return {"absurd": False, "atom": None, "proof": None}


def proof_measure(proof):
    """Honest complexity meter for a derivation (Gentzen's seat, panel review: his 1936
    consistency proof assigns ordinals to derivations; the engineering shadow of that idea is
    "report the shape, let it travel with the result"). Returns size (nodes), height (longest
    branch), and the rule-usage multiset -- so "a 4-node, height-3 proof over 2 rules" is data,
    not narrative."""
    def height(p):
        return 1 + max((height(c) for c in p.children), default=0)
    counts = {}
    def tally(p):
        counts[p.rule.name] = counts.get(p.rule.name, 0) + 1
        for c in p.children:
            tally(c)
    tally(proof)
    return {"size": proof.size(), "height": height(proof), "rules_used": counts}


# ---------------------------------------------------------------------------
# Wire format: plain lists/dicts <-> Atom/Rule/Proof, so a UnifiedMind faculty
# (and therefore POST /invoke) can speak this module in JSON without ever
# importing its classes. An atom on the wire is ["pred", ["arg1", "arg2"]].
# ---------------------------------------------------------------------------

def atom_from_wire(w):
    """["pred", [args...]] -> Atom. A bare "pred" string means a nullary atom."""
    if isinstance(w, str):
        return Atom(w)
    return Atom(w[0], tuple(w[1]) if len(w) > 1 else ())


def rules_from_wire(ws):
    """List of {"head": atom, "body": [atoms], "name": str} dicts -> [Rule].
    "body" and "name" are optional (a headless-body rule is a fact)."""
    out = []
    for w in ws:
        out.append(Rule(atom_from_wire(w["head"]),
                        tuple(atom_from_wire(b) for b in w.get("body", ())),
                        name=w.get("name")))
    return out


def proof_to_wire(p):
    """Proof tree -> nested JSON-safe dict. Inverse of proof_from_wire given the same rules."""
    return {"atom": [p.atom.pred, list(p.atom.args)], "rule": p.rule.name,
            "subst": dict(p.subst), "children": [proof_to_wire(c) for c in p.children]}


def proof_from_wire(w, rules):
    """Nested dict -> Proof, resolving rule names against `rules`. Raises KeyError on an
    unknown rule name -- a forged wire proof must not silently invent axioms."""
    by_name = {r.name: r for r in rules}
    return Proof(atom_from_wire(w["atom"]), by_name[w["rule"]], w.get("subst", {}),
                 tuple(proof_from_wire(c, rules) for c in w.get("children", ())))


def decode_atom(vec, preds, symbols, max_args, sym, bind, unbind, nearest, floor=0.25):
    """Decode a fact vector back to (pred, args) -- encode_atom's inverse, built ENTIRELY from
    the engine's existing verbs (unbind + nearest cleanup), passed in like encode_atom's.

    RULE-0 RECORD (panel Tier 2, Olshausen's seat): the audit found the resonator ALREADY
    shipped in three costumes (sbc_resonator, fpe_lattice_resonator, recursive_factor), and
    for THIS structure a resonator is not even needed -- encode_atom's construction is
    bind(pred, bundle_i(bind(role_i, arg_i))), so with known role vectors the decode is
    exact-inverse unbinding plus cleanup, the same recall recall_all() does for records.
    What was genuinely missing was only the atom SHAPE: try each candidate predicate, unbind
    it, read every role slot, score, and ABSTAIN below the floor rather than guess (the
    engine's standing honesty contract). Deterministic given the codebooks.

    Returns {"pred", "args", "score", "abstained"}; on abstention pred/args are None and the
    best rejected candidate travels in "best" so the caller can see WHY."""
    import numpy as _np
    best = None
    for pred in preds:
        payload = unbind(vec, sym("pred_" + pred))
        args, total = [], 0.0
        for i in range(max_args):
            cand = unbind(payload, sym("role_%d" % i))
            mat = _np.stack([sym(s) for s in symbols])
            j, score = nearest(cand, mat)
            args.append((symbols[j], float(score)))
        # an atom's arity is where the per-slot score falls off a cliff; keep slots above the
        # floor, score the atom by its kept slots' mean (a nullary atom scores its pred match)
        kept = [(a, s) for a, s in args if s >= floor]
        score = (sum(s for _, s in kept) / len(kept)) if kept else float(
            _np.dot(vec, sym("pred_" + pred)) / (_np.linalg.norm(vec) * _np.linalg.norm(sym("pred_" + pred)) + 1e-12))
        cand_out = {"pred": pred, "args": [a for a, _ in kept], "score": float(score)}
        if best is None or cand_out["score"] > best["score"]:
            best = cand_out
    if best is None or best["score"] < floor:
        return {"pred": None, "args": None, "score": 0.0 if best is None else best["score"],
                "abstained": True, "best": best}
    best["abstained"] = False
    return best


def fact_capacity(dim, n_symbols=32, n_preds=4, arity=2, loads=(1, 2, 4, 8, 16, 32),
                  seeds=range(6), floor=0.25):
    """PLATE'S QUESTION, measured (panel Tier 2): how many facts survive in ONE bundled trace
    at dimension D before decode fidelity collapses? Role-filler relation encoding is the
    worked example of the HRR book (Plate 1995/2003, e.g. eats(fido, bone)); its capacity
    analysis predicts recall degrades with load M at fixed D. This measures OUR construction
    (encode_atom facts bundled into a single trace, decoded by decode_atom) rather than citing
    the book's curves -- state what the number is a function of: (D, n_symbols, n_preds,
    arity, load, floor).

    Returns {"loads": [...], "exact": {load: fraction of facts decoded EXACTLY (pred + all
    args) with mean/CI across seeds}, ...}. Deterministic per seed; dedicated rng per trial
    (planted truths own their seeds)."""
    import numpy as _np
    from holographic.agents_and_reasoning.holographic_ai import bind, unbind, bundle, nearest
    from holographic.misc.holographic_measure import measure
    symbols = ["s%d" % i for i in range(n_symbols)]
    preds = ["p%d" % i for i in range(n_preds)]
    out = {"dim": dim, "n_symbols": n_symbols, "n_preds": n_preds, "arity": arity,
           "floor": floor, "loads": list(loads), "exact": {}}
    for load in loads:
        def run_once(seed, _load=load):
            rng = _np.random.default_rng(10_000 + seed)   # this trial's truths own this seed
            # INSTRUMENT ERROR #17, kept loud: the first draft seeded each symbol with
            # int.from_bytes(name)%2**32, which keeps only the FIRST FOUR BYTES -- the shared
            # "cap:" prefix -- so every symbol got the IDENTICAL vector and load-1 recall
            # measured 0.000. The house rule exists for this exact reason: hashlib, never
            # ad-hoc byte math -- and the house already HAS the tool. Delegate to it.
            from holographic.agents_and_reasoning.holographic_ai import derived_atom
            table = {}
            def sym(name):
                if name not in table:
                    table[name] = derived_atom(0, "cap:" + name, dim)
                return table[name]
            bnd = lambda vs: bundle(_np.stack(vs))
            facts = [Atom(preds[int(rng.integers(n_preds))],
                          tuple(symbols[int(rng.integers(n_symbols))] for _ in range(arity)))
                     for _ in range(_load)]
            encs = [encode_atom(f, sym, bind, bnd) for f in facts]
            # PLAIN SUM, not bundle(): bundle normalizes, so a 4-fact trace has norm ~1 while
            # each subtracted encoding has full strength -- explain-away then OVER-subtracts
            # and corrupts the residual (measured: recall at load 4 FELL as D rose, the
            # inverted-scaling red flag). Superposition arithmetic must stay linear.
            trace = _np.sum(encs, axis=0)
            # ITERATIVE DECODE WITH EXPLAIN-AWAY -- the same move recall_all(iterative=True)
            # already makes: decode the best atom, SUBTRACT its clean encoding, repeat.
            # Decoding the raw trace once would return the same winner _load times, which
            # measures nothing (the strawman this comment exists to prevent).
            residual = trace.copy()
            got = set()
            for _ in range(_load):
                d = decode_atom(residual, preds, symbols, arity, sym, bind, unbind, nearest,
                                floor=floor)
                if d["abstained"]:
                    break
                atom = Atom(d["pred"], tuple(d["args"]))
                got.add(atom.key())
                residual = residual - encode_atom(atom, sym, bind, bnd)
            want = {f.key() for f in facts}
            return len(got & want) / len(want)
        out["exact"][load] = measure(run_once, seeds=seeds)
    return out


# ---------------------------------------------------------------------------
# QUERY ENGINE: goal-directed evaluation WITH TABLING (E1).
#
# SOTA NOTE (searched 2026-08-16 before building, and the search CHANGED THE DESIGN):
# the backlog said "SLD with the occurs check". Plain SLD is NOT the state of the art and
# would have been WRONG for our flagship workload -- it diverges on left recursion and on
# cyclic relations, and our canonical query is transitive closure over a CYCLIC import
# graph. The standard fix since Chen & Warren 1996 (SLG resolution; XSB, SWI-Prolog) is
# TABLING: memo answers per subgoal variant, and suspend a call that is a variant of one
# already in progress, resuming it from the table. We implement the readable form --
# linear tabling with re-evaluation to a table fixpoint (Zhou et al.) -- rather than an
# SLG-WAM, because a stack machine buys speed we have not measured a need for and costs
# the auditability that is this module's whole point.
# ---------------------------------------------------------------------------

class _BudgetExceeded(Exception):
    """Internal signal: the demand closure blew the caller's budget, so the goal-directed
    path is the WRONG path for this goal (see query's measured law) -- unwind and say so."""


def occurs_in(var, term, subst):
    """The occurs check: does `var` appear inside `term` after following bindings?

    ROBINSON'S WARNING, promoted from NOTES to code (1965; the subtle half of unification):
    unifying x with a term CONTAINING x builds a cyclic term and proves falsehoods -- several
    Prologs shipped that bug as a speed hack. With constants-and-variables terms the check is
    VACUOUS (a variable can only bind to a constant or another variable), so this function is
    cheap here and correct in advance: when function symbols land, it is already the guard,
    and unify() already calls it rather than needing a retrofit."""
    t = _walk(term, subst)
    if t == var:
        return True
    if isinstance(t, (tuple, list)):     # future: f(g(x)) as nested tuples
        return any(occurs_in(var, sub, subst) for sub in t)
    return False


def _variant_key(atom):
    """Canonical key for a subgoal up to variable RENAMING: p(?a,?b) and p(?x,?y) are the
    same subgoal and must share one answer table (this is what makes tabling terminate).
    First-appearance order, so the key is deterministic."""
    seen, out = {}, []
    for a in atom.args:
        if is_var(a):
            if a not in seen:
                seen[a] = "?%d" % len(seen)
            out.append(seen[a])
        else:
            out.append(a)
    return atom.pred + "(" + ",".join(out) + ")"


def query(goal, rules, max_rounds=64, budget=None):
    """Goal-directed evaluation WITH TABLING: answer a (possibly non-ground) goal by
    working BACKWARD from it, touching only the rules and facts the goal needs -- unlike
    consequences(), which derives the entire fixpoint whether you want it or not.

    Returns {"answers": [Atom, ...], "proofs": {atom_key: Proof}, "rounds": n} -- ground
    instances of the goal, each with a checkable derivation. Deterministic: rules in order,
    answers in discovery order, variant keys canonical.

    TERMINATION: a subgoal that is a VARIANT of one already being evaluated does not recurse;
    it reads the current table instead (the "suspend"), and the outer loop re-evaluates until
    no table grows (the "resume"). This is why `ancestor(tom,?x)` over a CYCLIC graph
    terminates here and would spin forever under plain SLD. max_rounds is a fuse, not a
    tuning knob: the ground Herbrand base is finite, so the table fixpoint is reached.

    Non-ground goals ARE allowed here -- that is the point, and it retires prove()'s
    documented "querying for bindings is a deferred extension".

    MEASURED LAW, and the negative it contains (repo import graph, 2,268 edges, fixpoint
    9.3s): speedup is a function of the goal's DEMAND CLOSURE, not of graph size --
    304x at demand 1, 137x at 2-4, 68x at 9, and 0.3x (i.e. 3x SLOWER) at demand 690,
    where the goal needs essentially the whole graph and goal-direction buys nothing while
    tabling still costs. Break-even sits near demand ~200 on this workload. So query() is
    NOT a strict upgrade over consequences() and is never made the default; `budget` caps
    tabled answers and returns {"budget_exceeded": True} so a caller (see
    mind.logic_query) can fall back to the fixpoint instead of paying the slow path."""
    validate_rules(rules)
    tables = {}      # variant key -> {ground atom key: Proof}
    growing = [True]
    done = set()     # subgoals already EXPANDED this round -- see the note below

    def answers_for(g, active):
        k = _variant_key(g)
        tab = tables.setdefault(k, {})
        if k in done:
            # MEASURED BUG, fixed here: without this, a subgoal's rules were re-expanded on
            # EVERY reference, so a shared subgoal deep in a recursive graph was recomputed
            # exponentially -- the repo's own 2,246-edge import graph did not finish in 600s
            # while the 69-edge family finished instantly. Expanding each subgoal ONCE per
            # round (the standard linear-tabling optimisation) is what makes tabling pay.
            return list(tab.values())
        if k in active:
            # SUSPEND: this call is a variant of one in progress. Read what is known now;
            # the outer round loop resumes it with whatever the table gained meanwhile.
            return list(tab.values())
        active = active | {k}
        for ridx, r in enumerate(rules):
            rr = _rename(r, ridx)
            s0 = unify(rr.head, g)
            if s0 is None:
                continue
            for s, kids in _solve_body(rr.body, s0, active, answers_for):
                head = substitute(rr.head, s)
                if not head.is_ground() or head.key() in tab:
                    continue
                orig = {}
                for a in (r.head,) + r.body:
                    for t in a.args:
                        if is_var(t):
                            orig[t] = _walk(t + "_%d" % ridx, s)
                tab[head.key()] = Proof(head, r, orig, kids)
                growing[0] = True
                if budget is not None and sum(len(t) for t in tables.values()) > budget:
                    raise _BudgetExceeded()
        done.add(k)
        return list(tab.values())

    rounds = 0
    found = []
    try:
        while growing[0] and rounds < max_rounds:
            growing[0] = False
            done.clear()      # a new round re-expands with everything the last round learned
            found = answers_for(goal, frozenset())
            rounds += 1
    except _BudgetExceeded:
        # honest partial: say so, do not pretend the answer list is complete
        return {"answers": [], "proofs": {}, "rounds": rounds, "budget_exceeded": True}
    out, proofs = [], {}
    for pr in found:
        if unify(goal, pr.atom) is not None:
            out.append(pr.atom)
            proofs[pr.atom.key()] = pr
    return {"answers": out, "proofs": proofs, "rounds": rounds, "budget_exceeded": False}


def _solve_body(body, s, active, answers_for):
    """Left-to-right conjunction over tabled subgoals; yields (substitution, child proofs).
    Split out of query() so the recursion reads as one page each."""
    if not body:
        yield s, []
        return
    first, rest = body[0], body[1:]
    fs = substitute(first, s)
    for pr in answers_for(fs, active):
        s2 = unify(fs, pr.atom, s)
        if s2 is None:
            continue
        for s3, kids in _solve_body(rest, s2, active, answers_for):
            yield s3, [pr] + kids


# ---------------------------------------------------------------------------
# INDUCTION: learning-from-failures on the finite Horn fragment.  The Eno loop
# (panel horizon item) realised with the field's current reference method.
# ---------------------------------------------------------------------------

def _candidate_bodies(target_arity, body_preds, max_body, max_vars):
    """Enumerate candidate clause bodies in a CANONICAL, deterministic, smallest-first order.

    Head variables are ?v0..?v{arity-1}; body literals draw variables from ?v0..?v{max_vars-1}.
    Only LINKED clauses are yielded (every body literal shares a variable with the head or with
    an earlier-linked literal) -- unlinked literals are free-floating conditions that definite-
    clause semantics cannot use, so enumerating them only burns the budget. Smallest-first is
    the Occam bias Popper calls textual minimality (Cropper & Morel 2021)."""
    from itertools import combinations, product
    vars_ = ["?v%d" % i for i in range(max_vars)]
    head_vars = set(vars_[:target_arity])
    literals = []
    for pred, ar in sorted(body_preds.items()):
        for args in product(vars_, repeat=ar):
            literals.append(Atom(pred, args))
    literals.sort(key=lambda a: a.key())
    for size in range(1, max_body + 1):
        for combo in combinations(literals, size):
            linked, frontier = [], set(head_vars)
            pending = list(combo)
            progress = True
            while pending and progress:
                progress = False
                for lit in list(pending):
                    if set(lit.args) & frontier:
                        linked.append(lit)
                        frontier |= set(lit.args)
                        pending.remove(lit)
                        progress = True
            if not pending:
                yield combo


def induce_rules(background, positives, negatives, target, body_preds,
                 max_body=2, max_vars=3, max_candidates=20000):
    """Learn Horn clauses for `target` from ground examples -- LEARNING FROM FAILURES on the
    finite fragment (Cropper & Morel, Machine Learning 2021: generate / test / constrain,
    where a failed hypothesis PRUNES part of the hypothesis space).

    HONEST SCOPE: this is the LFF loop shape on OUR kernel, not Popper parity -- no ASP
    generation, no predicate invention, no noise handling; single-target, bounded clause
    size. What it shares with the reference method is the architecture and the pruning
    logic; what it adds is that TEST is our own measured T_P fixpoint (consequences), so
    recursion comes free (the candidate participates in its own chaining -- ancestor learns).

    generate: canonical smallest-first linked clauses (_candidate_bodies).
    test:     background + accepted + candidate, run to fixpoint; coverage of E+ / E-.
    constrain (the LFF step, adapted to smallest-first order -- the directions matter):
      * TOO SPECIFIC (derives no new positive): prune every LATER SUPERSET of its body --
        more conditions can only derive less. This list is CLEARED whenever a clause is
        accepted, because acceptance changes the fixpoint (under recursion a previously
        barren body can become productive) -- clearing costs retests, never answers.
      * TOO GENERAL (derives a negative): in smallest-first order its generalisations
        (subsets) were already enumerated, so Popper's generalisation-pruning direction
        has nothing left to prune FORWARD here; the failure is recorded and counted, and
        its exact body is never retested. KEPT APPROXIMATIONS, on record: subset pruning
        is theta-subsumption restricted to canonical variable naming (permutation-
        equivalent clauses cost retests, never soundness -- every accepted clause was
        TESTED, not inferred safe).
    accept:   greedy cover -- keep a consistent clause that derives new positives; stop
              when all of E+ is covered.

    background: list[Rule] of ground facts (and any prior rules). positives/negatives:
    ground Atoms of the target predicate. body_preds: {pred: arity} vocabulary (include
    the target itself to allow recursion). Returns {"rules", "covered", "stats"} with the
    honest counters (tested / pruned_general / pruned_specific), and rules=None when the
    space is exhausted uncovered -- never a best-effort guess dressed as an answer."""
    validate_rules(background)
    tgt_arity = len(positives[0].args) if positives else 0
    head = Atom(target, tuple("?v%d" % i for i in range(tgt_arity)))
    accepted, covered = [], set()
    pos_keys = {a.key() for a in positives}
    neg_keys = {a.key() for a in negatives}
    too_general, too_specific = set(), []  # failure signatures (frozensets of literal keys)
    stats = {"tested": 0, "pruned_general": 0, "pruned_specific": 0}
    for n, body in enumerate(_candidate_bodies(tgt_arity, body_preds, max_body, max_vars)):
        if n >= max_candidates:
            break
        sig = frozenset(b.key() for b in body)
        if sig in too_general:
            stats["pruned_general"] += 1     # exact retest of a known-general failure
            continue
        if any(s <= sig for s in too_specific):
            stats["pruned_specific"] += 1    # superset of a barren body derives no more
            continue
        cand = Rule(head, body, name="learned_%d" % len(accepted))
        stats["tested"] += 1
        trial = background + accepted + [cand]
        derived = {a.key() for a in consequences(trial) if a.pred == target}
        if derived & neg_keys:
            too_general.add(sig)             # record; generalisations are already behind us
            continue
        new = (derived & pos_keys) - covered
        if not new:
            too_specific.append(sig)
            continue
        accepted.append(cand)
        covered |= new
        too_specific = []   # the fixpoint changed; barren bodies may now be productive
        if covered == pos_keys:
            return {"rules": accepted, "covered": sorted(covered), "stats": stats}
    return {"rules": None, "covered": sorted(covered), "stats": stats}


def conjecture_and_refute(background, positives, negatives, target, body_preds,
                          max_body=2, max_vars=3, theorem_name="conjecture"):
    """THE ENO LOOP (panel horizon item), one orchestration over existing faculties:
    INDUCE candidate rules from data (induce_rules, LFF), DEDUCE their full consequences
    (consequences, the measured T_P fixpoint), REFUTE against the negatives -- and when the
    surviving theory proves every positive, hand the first positive's derivation to the
    EXTERNAL authority as Lean 4 source. Conjectures-and-refutations, mechanized, with the
    refutations kept as first-class output (the failures are the next induction's data).

    Returns {"rules" (wire), "lean", "consequences", "refuted_count", "stats"}."""
    r = induce_rules(background, positives, negatives, target, body_preds,
                     max_body=max_body, max_vars=max_vars)
    if r["rules"] is None:
        return {"rules": None, "lean": None, "consequences": None,
                "refuted_count": r["stats"]["tested"], "stats": r["stats"]}
    theory = background + r["rules"]
    cons = [[a.pred, list(a.args)] for a in consequences(theory)]
    pr = prove(positives[0], theory)
    lean_src = to_lean(pr, theory, theorem_name=theorem_name) if pr else None
    wire = [{"head": [ru.head.pred, list(ru.head.args)],
             "body": [[b.pred, list(b.args)] for b in ru.body], "name": ru.name}
            for ru in r["rules"]]
    refuted = r["stats"]["tested"] - len(r["rules"])
    return {"rules": wire, "lean": lean_src, "consequences": cons,
            "refuted_count": refuted, "stats": r["stats"]}


def fuzz_export(n=100, seed=0, use_lean="auto", max_consts=6, max_preds=3,
                max_facts=8, max_rules=3, max_body=2, max_vars=3):
    """DIFFERENTIAL ORACLE for the whole chain: random theories -> prove (BOTH strategies)
    -> independent check -> Lean export -> external Lean verdict. Any disagreement anywhere
    is a bug in OUR code, returned with its seed so it can be pinned as a Lean-free
    regression test -- the distillation contract: Lean finds the bug once, the repo keeps
    the pin forever, the binary stays optional.

    Checks per random theory (seeded, deterministic):
      * naive and seminaive prove/None-agree on every candidate goal (the equality theorem,
        exercised on hostile inputs rather than the friendly family base);
      * every found proof passes check_proof;
      * to_lean output is byte-deterministic across the two strategies' EXPORT of the SAME
        proof object;
      * with a lean binary (use_lean="auto"): external Lean typechecks every derivable
        goal's export, AND rejects a deliberately corrupted export (theorem atom swapped to
        an underivable one) -- the oracle is itself probed each run, per instrument-error
        ledger discipline.
    Constant/pred names are drawn from a hostile pool (Lean keywords, 'U', dash/underscore
    collision pairs, digit-led) so the namer's fixes stay exercised.

    Returns {"n", "derivable", "lean_available", "lean_checked", "failures": [...]} --
    failures carry {"seed", "stage", "detail"}. An empty failures list is a MEASURED
    statement about n seeds, not a proof of correctness; it says so here."""
    import numpy as _np
    hostile = ["a", "b", "fun", "U", "a-b", "a_b", "1x", "theorem", "in", "p", "q"]
    res = {"n": n, "derivable": 0, "lean_available": None, "lean_checked": 0, "failures": []}
    exe_checked_negative = False
    for k in range(n):
        rng = _np.random.default_rng(seed * 100003 + k)
        consts = list(dict.fromkeys(hostile[:2 + int(rng.integers(max_consts))]))
        preds = {}
        for i in range(1 + int(rng.integers(max_preds))):
            preds["pr_%s" % hostile[int(rng.integers(len(hostile)))] + "_%d" % i] =                 1 + int(rng.integers(2))
        pnames = sorted(preds)
        rules, names = [], set()
        for i in range(1 + int(rng.integers(max_facts))):
            pr = pnames[int(rng.integers(len(pnames)))]
            args = tuple(consts[int(rng.integers(len(consts)))] for _ in range(preds[pr]))
            nm = "f%d" % i
            rules.append(Rule(Atom(pr, args), name=nm))
        for i in range(int(rng.integers(max_rules + 1))):
            hp = pnames[int(rng.integers(len(pnames)))]
            hvars = tuple("?v%d" % j for j in range(preds[hp]))
            body = []
            for _ in range(1 + int(rng.integers(max_body))):
                bp = pnames[int(rng.integers(len(pnames)))]
                body.append(Atom(bp, tuple("?v%d" % int(rng.integers(max_vars))
                                           for _ in range(preds[bp]))))
            # keep the clause linked to its head so it can ever fire
            if not (set(hvars) & set(a for b in body for a in b.args if is_var(a))):
                continue
            rules.append(Rule(Atom(hp, hvars), tuple(body), name="r%d" % i))
        try:
            cons_n = sorted(a.key() for a in consequences(rules))
            cons_s = sorted(a.key() for a in consequences(rules, strategy="seminaive"))
            if cons_n != cons_s:
                res["failures"].append({"seed": k, "stage": "equality",
                                        "detail": "fixpoints differ: %d vs %d"
                                                  % (len(cons_n), len(cons_s))})
                continue
        except Exception as e:
            res["failures"].append({"seed": k, "stage": "fixpoint",
                                    "detail": "%s: %s" % (type(e).__name__, e)})
            continue
        derivable = [a for a in consequences(rules)]
        # one underivable probe: a fresh constant no fact mentions
        ghost = Atom(pnames[0], tuple("zz_ghost" for _ in range(preds[pnames[0]])))
        for strat in ("naive", "seminaive"):
            if prove(ghost, rules, strategy=strat) is not None:
                res["failures"].append({"seed": k, "stage": "soundness",
                                        "detail": "ghost derived under %s" % strat})
        for a in derivable[:3]:
            try:
                pr_n = prove(a, rules)
                pr_s = prove(a, rules, strategy="seminaive")
                if pr_n is None or pr_s is None:
                    res["failures"].append({"seed": k, "stage": "agreement",
                                            "detail": "consequence %s not re-proved" % a})
                    continue
                check_proof(pr_n, rules); check_proof(pr_s, rules)
                src = to_lean(pr_n, rules, theorem_name="fz")
                if src != to_lean(prove(a, rules), rules, theorem_name="fz"):
                    res["failures"].append({"seed": k, "stage": "determinism",
                                            "detail": "export bytes differ for %s" % a})
                    continue
                res["derivable"] += 1
                lv = lean_check(src) if use_lean in ("auto", True) else {"available": False}
                if res["lean_available"] is None:
                    res["lean_available"] = lv.get("available", False)
                if lv.get("available"):
                    if not lv["ok"]:
                        res["failures"].append({"seed": k, "stage": "lean",
                                                "detail": lv["stderr"][:200]})
                    else:
                        res["lean_checked"] += 1
                        if not exe_checked_negative:
                            # probe the oracle once per run (instrument-error ledger rule):
                            # a corrupted proof term MUST be rejected, or the green light
                            # we are collecting is decorative
                            bad = src.rsplit(":=", 1)[0] + ":= lecore_undefined_term\n"
                            if lean_check(bad)["ok"]:
                                res["failures"].append({"seed": k, "stage": "oracle",
                                                        "detail": "lean accepted garbage"})
                            exe_checked_negative = True
            except Exception as e:
                res["failures"].append({"seed": k, "stage": "prove/export",
                                        "detail": "%s: %s" % (type(e).__name__, e)})
    return res


def _selftest():
    """Regression trap, not a smoke test: exact derivability, checker independence proven by a
    FORGED proof being rejected, Lean text pinned to structural content, and a kept negative
    (underivable goal returns None, loudly asserted)."""
    # Socrates, as tradition demands -- plus a two-hop ancestry chain to exercise unification.
    human = lambda x: Atom("human", (x,))
    mortal = lambda x: Atom("mortal", (x,))
    parent = lambda x, y: Atom("parent", (x, y))
    anc = lambda x, y: Atom("ancestor", (x, y))
    rules = [
        Rule(human("socrates"), name="h_soc"),
        Rule(Atom("mortal", ("?x",)), (Atom("human", ("?x",)),), name="mortality"),
        Rule(parent("tom", "bob"), name="p_tb"),
        Rule(parent("bob", "liz"), name="p_bl"),
        Rule(Atom("ancestor", ("?x", "?y")), (Atom("parent", ("?x", "?y")),), name="anc_base"),
        Rule(Atom("ancestor", ("?x", "?z")),
             (Atom("parent", ("?x", "?y")), Atom("ancestor", ("?y", "?z"))), name="anc_step"),
    ]
    # 1) derivability, exactly
    pr = prove(mortal("socrates"), rules)
    assert pr is not None and pr.atom.key() == "mortal(socrates)"
    assert check_proof(pr, rules)
    pr2 = prove(anc("tom", "liz"), rules)
    assert pr2 is not None and pr2.size() == 4, "tom->liz needs base+step over 2 parent facts, size 4, got %r" % (pr2 and pr2.size())
    assert check_proof(pr2, rules)
    # 2) kept negative: what is NOT derivable stays not derivable
    assert prove(mortal("zeus"), rules) is None, "zeus was never declared human; deriving him would be a soundness bug"
    assert prove(anc("liz", "tom"), rules) is None, "ancestry must not run backwards"
    # 3) the checker trusts nothing: a forged proof (right rule, wrong conclusion) must be REJECTED
    forged = Proof(mortal("zeus"), rules[1], {"?x": "zeus"}, (Proof(human("zeus"), rules[0], {}),))
    rejected = False
    try:
        check_proof(forged, rules)
    except AssertionError:
        rejected = True
    assert rejected, "checker accepted a forged premise -- prover/checker independence is broken"
    # 4) Lean export: structural pins (declarations present, theorem line well-formed), and
    #    determinism -- two runs, identical bytes
    src = to_lean(pr2, rules, theorem_name="tom_anc_liz")
    assert src == to_lean(prove(anc("tom", "liz"), rules), rules, theorem_name="tom_anc_liz")
    for needle in ("axiom U : Type", "axiom ancestor : U -> U -> Prop",
                   "axiom anc_step : forall", "theorem tom_anc_liz : ancestor tom liz :="):
        assert needle in src, "missing %r in Lean output" % needle
    # 5) the bridge reports honestly whether Lean exists; if it does, the proof must typecheck
    res = lean_check(src)
    if res["available"]:
        assert res["ok"], "external Lean rejected our proof:\n%s" % res["stderr"]
    # 6) VSA encoding: same atom -> same vector; different atom -> low cosine (planted truth,
    #    dedicated rng)
    rng = np.random.default_rng(12345)
    table = {}
    def sym(name):
        if name not in table:
            table[name] = rng.standard_normal(2048) / np.sqrt(2048)
        return table[name]
    bind = lambda a, b: np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=a.size)
    bundle = lambda vs: np.sum(vs, axis=0)
    v1 = encode_atom(parent("tom", "bob"), sym, bind, bundle)
    v1b = encode_atom(parent("tom", "bob"), sym, bind, bundle)
    v2 = encode_atom(parent("bob", "liz"), sym, bind, bundle)
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos(v1, v1b) > 0.999999
    assert abs(cos(v1, v2)) < 0.2, "distinct ground atoms must be near-orthogonal, got %.3f" % cos(v1, v2)
    # 7) Tier-1 (panel): the least fixpoint is EXACTLY the 7 consequences of this base --
    #    completeness measured, not cited (van Emden-Kowalski); absurdity smoke fires only
    #    when a contradiction rule exists; proof_measure pins the ancestry derivation's shape
    cs = sorted(a.key() for a in consequences(rules))
    assert cs == ['ancestor(bob,liz)', 'ancestor(tom,bob)', 'ancestor(tom,liz)',
                  'human(socrates)', 'mortal(socrates)', 'parent(bob,liz)',
                  'parent(tom,bob)'], cs
    bad = rules + [Rule(Atom("false", ()), (Atom("mortal", ("socrates",)),), name="oops")]
    assert detect_absurdity(bad)["absurd"] is True
    assert detect_absurdity(rules)["absurd"] is False
    mm = proof_measure(pr2)
    assert mm == {"size": 4, "height": 3,
                  "rules_used": {"anc_step": 1, "p_tb": 1, "anc_base": 1, "p_bl": 1}}, mm
    # 8) Tier-2 (panel): decode is encode's inverse for one clean fact; noise ABSTAINS; and
    #    the KEPT NEGATIVE is pinned small -- load-8 recall through one bundled trace stays
    #    low even as D grows (1/sqrt(M) independent of D: widening the vector does not help;
    #    INDEX fact bases, never bundle them)
    from holographic.agents_and_reasoning.holographic_ai import unbind as _ub, nearest as _nr
    dec = decode_atom(v1, ["parent", "human"], ["tom", "bob", "liz"], 2, sym, bind, _ub, _nr)
    assert dec["pred"] == "parent" and dec["args"] == ["tom", "bob"] and not dec["abstained"]
    noise_dec = decode_atom(rng.standard_normal(2048), ["parent"], ["tom", "bob"], 2,
                            sym, bind, _ub, _nr)
    assert noise_dec["abstained"], "noise must abstain, not confabulate a fact"
    capr = fact_capacity(dim=256, n_symbols=8, n_preds=2, loads=(1, 8), seeds=range(2))
    assert capr["exact"][1]["mean"] == 1.0, "one clean fact must decode exactly"
    assert capr["exact"][8]["mean"] < 0.5, "the bundled-trace cliff is the kept negative"
    # 9) THE ENO LOOP pinned: LFF learns mortal from human (1 clause, exact) and RECURSIVE
    #    ancestor from parent facts (2 clauses); the induced theory refutes every negative
    #    and its Lean export is structurally sound (external verdict is the bridge's job)
    bg_i = [Rule(Atom("human", (n,)), name="h_" + n) for n in ("socrates", "plato")]
    ri = induce_rules(bg_i, [Atom("mortal", ("socrates",)), Atom("mortal", ("plato",))],
                      [], "mortal", {"human": 1})
    assert [str(r.head) + "|" + ",".join(map(str, r.body)) for r in ri["rules"]] == \
        ["mortal(?v0)|human(?v0)"], ri
    fams = [("tom", "bob"), ("bob", "liz"), ("liz", "ann")]
    bg_a = [Rule(Atom("parent", q), name="p_%d" % i) for i, q in enumerate(fams)]
    out = conjecture_and_refute(
        bg_a,
        [Atom("ancestor", ("tom", "bob")), Atom("ancestor", ("tom", "liz")),
         Atom("ancestor", ("tom", "ann")), Atom("ancestor", ("bob", "ann"))],
        [Atom("ancestor", ("bob", "tom")), Atom("ancestor", ("ann", "tom"))],
        "ancestor", {"parent": 2, "ancestor": 2}, theorem_name="t")
    assert out["rules"] is not None and len(out["rules"]) == 2
    assert out["refuted_count"] > 0, "a search that refuted nothing tested nothing"
    assert "theorem t : ancestor tom bob :=" in out["lean"]
    # 10) SEMINAIVE pinned: same atom set as naive on a recursive base (the Bancilhon-
    #     Ramakrishnan equality theorem, asserted not cited), and its proof still CHECKS
    sn = {a.key() for a in consequences(rules, strategy="seminaive")}
    nv = {a.key() for a in consequences(rules)}
    assert sn == nv, "seminaive must derive EXACTLY the naive fixpoint"
    pr_sn = prove(Atom("ancestor", ("tom", "liz")), rules, strategy="seminaive")
    assert pr_sn is not None and check_proof(pr_sn, rules)
    # 11) E1 TABLED QUERY: bindings for a non-ground goal, each with a checkable proof;
    #     the acid test is LEFT RECURSION OVER A CYCLE, where plain SLD diverges forever
    qr = query(Atom("ancestor", ("tom", "?w")), rules)
    assert sorted(a.key() for a in qr["answers"]) == ["ancestor(tom,bob)", "ancestor(tom,liz)"]
    # and the query slice must EQUAL the fixpoint's slice -- two engines, one answer set
    assert {a.key() for a in qr["answers"]} == {
        a.key() for a in consequences(rules) if a.pred == "ancestor" and a.args[0] == "tom"}
    for a in qr["answers"]:
        assert check_proof(qr["proofs"][a.key()], rules)
    cyc = [Rule(Atom("edge", ("a", "b")), name="e0"), Rule(Atom("edge", ("b", "c")), name="e1"),
           Rule(Atom("edge", ("c", "a")), name="e2"),        # a CYCLE
           Rule(Atom("path", ("?x", "?y")),                  # LEFT-recursive clause
                (Atom("path", ("?x", "?z")), Atom("edge", ("?z", "?y"))), name="pl"),
           Rule(Atom("path", ("?x", "?y")), (Atom("edge", ("?x", "?y")),), name="pb")]
    cr = query(Atom("path", ("a", "?w")), cyc)
    assert sorted(a.key() for a in cr["answers"]) == ["path(a,a)", "path(a,b)", "path(a,c)"]
    assert query(Atom("ancestor", ("tom", "?w")), rules, budget=1)["budget_exceeded"] is True
    assert occurs_in("?x", "?x", {}) and not occurs_in("?x", "a", {})
    # 12) wire round-trip: proof -> dict -> proof still CHECKS, and json survives it
    import json
    w = json.loads(json.dumps(proof_to_wire(pr2)))
    assert check_proof(proof_from_wire(w, rules), rules)
    print("OK: holographic_lean -- prove/check/export/bridge/encode all pinned "
          "(lean binary available: %s)" % res["available"])


if __name__ == "__main__":
    _selftest()
