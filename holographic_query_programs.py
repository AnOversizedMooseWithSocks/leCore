"""holographic_query_programs.py -- VSA PROGRAMS AS DATABASE OBJECTS (query backlog PR1-PR6).

SUPERSEDED BY holographic_queryprog -- the wired version (it has the curated catalog home). This earlier implementation of the same
backlog item is kept for its tests but is intentionally NOT wired into any pipeline; use holographic_queryprog instead.

Stored procedures, VSA-native: a catalog of installed programs you can query, an "install" that registers a program
with the semantic metadata it needs to be self-discoverable, and the ability to RUN a program over data -- all from
the database interface. In DB terms this is pg_proc-style stored procedures / UDFs, where the "procedure" is a
hypervector the VSA machine executes.

Most of it is PROMOTE, not build -- listing, explaining, and running already ship (capability_registry, the machine's
run, explain_program, the store's fuzzy WHERE). Only the install path and the execute-on-query bridge are new here.

The safety story is a genuine selling point: a VSA "program" is a hypervector over a fixed, tiny opcode set
(LOAD/BIND/BUNDLE/PERMUTE/APPLY/IFMATCH/ITERATE/REPEAT/HALT), NOT arbitrary code -- it can't do I/O or run host code,
and it can only call the WHITELISTED handlers it was installed with (an APPLY to a faculty not in the whitelist is a
safe no-op, which the machine already does). We add step limits so a program must halt.

KEPT NEGATIVES (loud): a program working on decoded vectors is on the FUZZY side, so results carry a confidence and
can be uncertain -- a garbage program shows LOW confidence, not a confident wrong answer (exact stored values still
round-trip losslessly). Find-by-meaning is a bag-of-words descriptor (shared word atoms), not a learned embedding --
good for keyword-semantic recall, not deep paraphrase. System programs are read-only from SQL (the wall). NumPy +
stdlib; deterministic.
"""
import re

import numpy as np

from holographic_ai import bind, unbind, bundle, cosine, Vocabulary
from holographic_machine import HoloMachine
from holographic_query import UserTable, QueryError, run_sql, explain_program, capability_registry


def _tokens(text):
    """The distinct lowercase word tokens of a string (for the bag-of-words descriptor)."""
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class ProgramCatalog:
    """A catalog of installed VSA programs, queryable as a table, findable by meaning, explainable, and runnable over
    data -- with system programs read-only and user programs add/remove-at-will."""

    def __init__(self, dim=1024, seed=0, faculties=()):
        self.dim = dim
        self.seed = seed
        # the machine's faculty codebook is the UNIVERSE of names a program may APPLY; handlers are supplied per run.
        self.machine = HoloMachine(dim=dim, seed=seed, faculties=list(faculties) or ["identity"])
        self.programs = {}                                    # name -> dict(program_vec, program, doc, inputs, ...)
        self._words = Vocabulary(dim, seed + 9)               # a SHARED bag-of-words space for docs & queries

    # ---- PR5 (BUILD): install / uninstall ----
    def install(self, name, program, doc, inputs=(), outputs=(), allowed_handlers=(), tier="user"):
        """Register a program so the system can DISCOVER, DESCRIBE, and RUN it. `program` is a list of (opcode,
        operand) instructions (assembled to a vector here). The load-bearing bit is the semantic metadata: a
        bag-of-words DESCRIPTOR (so find-by-meaning works) plus the declared input/output roles and the SANDBOX
        whitelist of handlers it may call. A user program lives in the 'user' tier (add/remove at will)."""
        if name in self.programs and self.programs[name]["tier"] == "system":
            raise QueryError("%r is a system program (read-only); cannot overwrite from here" % name)
        pv = self.machine.assemble(program)
        self.programs[name] = {
            "program": list(program), "program_vec": pv, "doc": doc,
            "inputs": list(inputs), "outputs": list(outputs),
            "handlers": list(allowed_handlers), "tier": tier,
            "descriptor": self._embed(doc),                  # its MEANING, for PR4
        }
        return self

    def uninstall(self, name):
        """Remove a USER program (add/remove at will). A system program is refused -- read-only, changeable only
        through the management methods, never a raw delete."""
        if self.programs.get(name, {}).get("tier") == "system":
            raise QueryError("%r is a system program (read-only); it cannot be uninstalled from SQL" % name)
        self.programs.pop(name, None)
        return self

    def _embed(self, text):
        """A bag-of-words descriptor: the superposition of the doc's word atoms in the shared word space."""
        vecs = [self._words.get(w) for w in _tokens(text)]
        return bundle(vecs) if vecs else np.zeros(self.dim)

    # ---- PR1 (PROMOTE): the catalog as a queryable table ----
    def catalog_table(self, mind=None):
        """`system.programs` as a query Table: user-installed programs, plus (if a `mind` is given) the built-in
        faculties from capability_registry, each tagged with its tier (system vs user). Query it with ordinary SQL."""
        t = UserTable("programs", ["name", "tier", "domain", "doc"], dim=self.dim, seed=self.seed)
        if mind is not None:
            reg = capability_registry(mind, dim=self.dim, seed=self.seed)
            for r in reg.rows:
                t.insert({"name": r["name"], "tier": "system",
                          "domain": r.get("domain", "general"), "doc": r.get("doc", "")})
        for name, p in self.programs.items():
            t.insert({"name": name, "tier": p["tier"], "domain": "user", "doc": p["doc"]})
        return t

    # ---- PR4 (PROMOTE): find a program by meaning ----
    def find(self, query_text, k=5):
        """Find installed programs whose DOC is semantically close to `query_text` -- a bag-of-words cosine over the
        shared word space (the store's fuzzy-match idea, applied to the program catalog). Ranked with a score. This
        is a genuine superpower over a SQL function catalog, which is exact-match only."""
        q = self._embed(query_text)
        scored = sorted(((n, float(cosine(p["descriptor"], q))) for n, p in self.programs.items()),
                        key=lambda t: (-t[1], t[0]))
        return [{"name": n, "score": s, "doc": self.programs[n]["doc"]} for n, s in scored[:k]]

    # ---- PR3 (PROMOTE): EXPLAIN (dry run) ----
    def explain(self, name):
        """A handler-less DRY RUN: which faculties the program WOULD call and how many steps it takes, without
        executing them. Reuses explain_program."""
        if name not in self.programs:
            raise QueryError("no such program %r" % name)
        return explain_program(self.machine, self.programs[name]["program_vec"])

    # ---- PR6 (BUILD): the execute-on-query bridge, sandboxed ----
    def execute(self, name, accumulator, handlers, max_steps=1000):
        """Run an installed program over an accumulator (e.g. a bundle of the query rows' record vectors), IN THE
        VECTOR DOMAIN -- no per-row Python round-trip. SANDBOX: only the program's own whitelisted handlers are
        callable (an APPLY to anything else is a safe no-op); a step limit bounds a runaway. Returns (result_vector,
        trace)."""
        if name not in self.programs:
            raise QueryError("no such program %r" % name)
        p = self.programs[name]
        allowed = {h: handlers[h] for h in p["handlers"] if h in handlers}   # the whitelist, intersected with offered
        return self.machine.run(p["program_vec"], init_acc=np.asarray(accumulator, float),
                                handlers=allowed, max_steps=max_steps)


def encode_rows_accumulator(table, indices=None):
    """Bundle a set of a table's record vectors into ONE accumulator -- the 'feed the query rows in' step for
    execute(). indices=None uses every row."""
    idx = range(len(table)) if indices is None else indices
    vecs = [table.records[i] for i in idx]
    return bundle(vecs) if vecs else np.zeros(table.dim)


def _selftest():
    """Install a couple of user programs; list them (with a mind's system faculties), find one by meaning, explain a
    dry run, execute one over rows through the sandbox (a whitelisted handler transforms the accumulator; a
    non-whitelisted one is a no-op), and confirm system programs are read-only. Deterministic."""
    cat = ProgramCatalog(dim=1024, seed=0, faculties=["tag", "wipe"])

    # a program that APPLYs the 'tag' faculty then halts
    cat.install("tagger", [("APPLY", "tag"), ("HALT", "a")], doc="tag or label a record vector",
                inputs=["value"], outputs=["tagged"], allowed_handlers=["tag"])
    cat.install("cluster_series", [("APPLY", "wipe"), ("HALT", "a")],
                doc="cluster a noisy time series into groups", allowed_handlers=["wipe"])

    # PR1: the catalog lists user programs (and would list system faculties given a mind)
    tbl = cat.catalog_table()
    names = {r["name"] for r in run_sql("SELECT name FROM programs", tbl)}
    assert {"tagger", "cluster_series"} <= names

    # PR4: find by meaning -- 'group a signal over time' should surface the clustering program, not the tagger
    top = cat.find("group a signal over time", k=1)[0]
    assert top["name"] == "cluster_series"

    # PR3: EXPLAIN is a dry run naming the faculties it would call
    ex = cat.explain("tagger")
    assert "tag" in ex["faculties_called"] and ex["n_steps"] >= 1

    # PR6: execute the tagger over a record vector; the whitelisted 'tag' handler binds a TAG atom
    TAG = cat.machine._atom("__TAG__", unitary=True)          # unitary -> unbind is near-exact
    handlers = {"tag": (lambda acc: bind(acc, TAG)), "wipe": (lambda acc: np.zeros_like(acc))}
    value = cat._words.get("hello")
    out, trace = cat.execute("tagger", value, handlers)
    assert cosine(out, value) < 0.2                           # the transform clearly ran (output != input)
    assert cosine(unbind(out, TAG), value) > 0.9              # and unbinds back to the input (the tag round-trips)

    # SANDBOX: a program may only call handlers it was installed with. Re-install the tagger WITHOUT whitelisting
    # 'wipe', point its APPLY at 'wipe', and confirm the (non-whitelisted) handler is NOT called -> a no-op.
    cat.install("sneaky", [("APPLY", "wipe"), ("HALT", "a")], doc="tries to wipe", allowed_handlers=[])  # empty whitelist
    out2, _ = cat.execute("sneaky", value, handlers)
    assert cosine(out2, value) > 0.99                         # 'wipe' was refused -> accumulator unchanged

    # system programs are read-only
    cat.install("sys_builtin", [("HALT", "a")], doc="a system program", tier="system")
    try:
        cat.uninstall("sys_builtin")
        raise AssertionError("system program was uninstalled")
    except QueryError:
        pass

    print("holographic_query_programs selftest OK: installed user programs are catalogued and found by meaning "
          "('group a signal over time' -> cluster_series), EXPLAIN dry-runs the tagger (%d steps, calls %s), execute "
          "runs it over a record in the vector domain (tag unbinds back at cosine>0.9), the sandbox refuses a "
          "non-whitelisted handler (no-op), and system programs are read-only; deterministic"
          % (ex["n_steps"], ex["faculties_called"]))


if __name__ == "__main__":
    _selftest()
