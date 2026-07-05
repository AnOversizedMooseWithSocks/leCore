"""holographic_queryprog.py -- VSA programs as installable, runnable database objects (backlog PR1-PR6).

WHY
---
A SQL database lets you install a stored procedure and run it over your data. This is that, VSA-native: a "program"
is a hypervector the HoloMachine executes (opcodes LOAD/BIND/BUNDLE/PERMUTE/APPLY/HALT -- NOT arbitrary code), so it
cannot do I/O or run host code and can only call the WHITELISTED handlers it was installed with -- safer than a SQL
stored procedure, which can often reach outside the DB. The pleasant finding: listing, explaining, and running a
program already ship (capability_registry, explain_program, machine.run); only the install path and the
execute-on-query bridge are new.

  PR1  list_programs(cat)                 -- the catalog, as queryable rows (name/domain/doc/tier). Like pg_proc.
  PR3  explain(cat, name)                 -- a DRY RUN: which faculties it WOULD call, how many steps (no execution).
  PR4  find_program(cat, "text")          -- find a program BY MEANING (fuzzy over its doc) -- SQL catalogs can't.
  PR5  install_program(cat, ...)          -- CREATE FUNCTION: register a program with the metadata for self-discovery.
  PR6  execute_program(cat, name, rows)   -- run a program over query rows, sandboxed + step-bounded, result carries
                                            a calibrated confidence (the fuzzy fork -- garbage shows LOW confidence,
                                            not a confident wrong answer).

KEPT NEGATIVES (loud)
  * find-by-meaning is WORD-OVERLAP over the doc (a bundle of token vectors), not deep semantics -- good enough to
    surface "cluster a series" for a doc about grouping, honest about not being a language model.
  * execute results are on the FUZZY side (a program over decoded vectors has readback error) -- so every result
    carries a confidence and can abstain; exact stored values still round-trip losslessly.
  * symbol-name collisions between programs are avoided by SCOPING each program's symbols by its name -- never one
    flat namespace.
  * system-tier programs are READ-ONLY (uninstall refused) -- like pg_catalog, changed only via management methods.
"""
import re
import numpy as np
from holographic_ai import bind, bundle, cosine, Vocabulary
from holographic_machine import HoloMachine
from holographic_query import QueryError, explain_program


# a small domain-tagging heuristic, matching capability_registry's honest keyword approach
_DOMAIN_KEYS = [
    ("render", "graphics"), ("splat", "graphics"), ("shade", "graphics"), ("light", "graphics"),
    ("cluster", "analysis"), ("anomal", "analysis"), ("series", "analysis"), ("stat", "analysis"),
    ("recall", "memory"), ("store", "memory"), ("index", "memory"),
    ("bind", "algebra"), ("bundle", "algebra"), ("factor", "algebra"),
    ("detect", "honesty"), ("calibrat", "honesty"),
]


def _domain_of(name, doc):
    """A keyword heuristic tag from the name + doc (honest: a heuristic, not a curated taxonomy)."""
    hay = (name + " " + doc).lower()
    for key, dom in _DOMAIN_KEYS:
        if key in hay:
            return dom
    return "general"


class ProgramCatalog:
    """The database's registry of installed VSA programs. Reuses the HoloMachine to assemble/run programs, a shared
    Vocabulary for their symbols (self-discovery), and a token vocabulary for doc descriptors (find-by-meaning)."""

    def __init__(self, dim=2048, seed=0):
        self.dim = dim
        self.seed = seed
        self._programs = {}                       # name -> record (see install_program)
        self._tokens = Vocabulary(dim, seed + 7)  # doc-word -> vector, for the semantic descriptor
        self.symbols = Vocabulary(dim, seed + 9)  # the SHARED symbol vocabulary programs merge into (self-discovery)

    # ---- PR5: install / uninstall --------------------------------------------------------------------------
    def install(self, name, instructions, doc, inputs, outputs, handlers,
                data=None, faculties=None, symbols=None, tier="user"):
        """CREATE FUNCTION. `instructions` is a list of (opcode, operand); we assemble it to a program vector on a
        machine that knows its data/faculty atoms. `handlers` is the whitelist of faculties APPLY may call (the
        sandbox). `symbols` (optional) are the program's own atoms, merged SCOPED-by-name into the shared vocab so
        cleanup can decode them system-wide -- this is what makes it 'installed' rather than a loose vector."""
        if name in self._programs and self._programs[name]["tier"] == "system":
            raise QueryError("%r is a system program (read-only)" % name)
        # a machine that knows this program's data + faculty atoms, so assemble/run can reference them
        machine = HoloMachine(dim=self.dim, seed=self.seed,
                              data=list(data or []) + list(inputs) + list(outputs),
                              faculties=list(faculties or handlers))
        program_vec = machine.assemble(instructions)
        # self-discovery: merge the program's symbols into the shared vocab, scoped by name (no collisions)
        if symbols:
            for sym, vec in symbols.items():
                self.symbols.add("%s:%s" % (name, sym), np.asarray(vec, float))
        self._programs[name] = {
            "name": name, "doc": doc, "domain": _domain_of(name, doc), "tier": tier,
            "inputs": list(inputs), "outputs": list(outputs), "handlers": list(handlers),
            "instructions": list(instructions), "program_vec": program_vec, "machine": machine,
            "descriptor": self._embed(doc),                    # for PR4 find-by-meaning
        }
        return name

    def uninstall(self, name):
        """Remove a USER program (add/remove at will). A system program is refused -- read-only, like pg_catalog."""
        if name not in self._programs:
            raise QueryError("no such program %r" % name)
        if self._programs[name]["tier"] == "system":
            raise QueryError("%r is a system program (read-only); it cannot be uninstalled" % name)
        del self._programs[name]
        return True

    # ---- PR1: the catalog as rows --------------------------------------------------------------------------
    def list(self, tier=None):
        """The catalog: one row per program (name, domain, doc, tier, inputs, outputs). Like pg_proc -- an ordinary
        table you can SELECT over. `tier` filters to system / user."""
        return [{"name": p["name"], "domain": p["domain"], "doc": p["doc"], "tier": p["tier"],
                 "inputs": list(p["inputs"]), "outputs": list(p["outputs"])}
                for p in self._programs.values() if tier is None or p["tier"] == tier]

    # ---- PR3: explain (dry run) ----------------------------------------------------------------------------
    def explain(self, name):
        """EXPLAIN: a handler-less dry run -- which faculties the program WOULD call and how many steps, without
        executing them. Reuses the shipped explain_program."""
        p = self._require(name)
        return explain_program(p["machine"], p["program_vec"])

    # ---- PR4: find by meaning ------------------------------------------------------------------------------
    def find(self, query_text, k=5):
        """Find a program BY MEANING: rank installed programs by the cosine of the query's doc-embedding against each
        program's descriptor, with a confidence. A SQL function catalog is exact-match only; this is the superpower.
        HONEST: this is word-overlap similarity over the docs, not a language model."""
        q = self._embed(query_text)
        scored = []
        for p in self._programs.values():
            sim = float(cosine(q, p["descriptor"]))
            scored.append({"name": p["name"], "doc": p["doc"], "domain": p["domain"],
                           "tier": p["tier"], "_confidence": max(0.0, sim)})
        scored.sort(key=lambda r: (r["_confidence"], r["name"]), reverse=True)
        return scored[:k]

    # ---- PR6: execute a program over query rows ------------------------------------------------------------
    def execute(self, name, rows, encode_fn=None, max_steps=10_000):
        """Run an installed program over query rows. `rows` are dicts (query output); they are encoded to the
        program's initial accumulator, the machine runs with ONLY the whitelisted handlers (the sandbox) and a step
        bound (no runaway), and the result is decoded with a confidence. VSA-native: the program runs in the vector
        domain over the records -- no per-row Python round-trip."""
        p = self._require(name)
        init_acc = (encode_fn or self._default_encode)(rows, p)
        handlers = {h: self._handler(h) for h in p["handlers"]}     # SANDBOX: only whitelisted faculties are callable
        acc, trace = p["machine"].run(p["program_vec"], init_acc=init_acc,
                                      handlers=handlers, max_steps=max_steps)
        # decode with confidence: how strongly the result matches the program's declared outputs (the fuzzy fork)
        conf = self._result_confidence(acc, p)
        return {"result": acc, "_confidence": conf, "trace": trace, "n_steps": len(trace)}

    # ---- internals -----------------------------------------------------------------------------------------
    def _require(self, name):
        if name not in self._programs:
            raise QueryError("no such program %r" % name)
        return self._programs[name]

    def _embed(self, text):
        """A doc descriptor = the normalized bundle of its word vectors (docs sharing words rank close). Simple,
        deterministic, readable -- and honestly just word overlap, not deep semantics."""
        words = [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2]
        if not words:
            return np.zeros(self.dim)
        return bundle([self._tokens.get(w) for w in words])

    def _handler(self, faculty):
        """A stand-in handler: a norm-preserving no-op so the sandbox and step-bounding can be exercised honestly.
        A real deployment supplies the mind's faculty here; the point demonstrated is the WHITELIST -- only handlers
        named in the program's `handlers` list are ever passed to run()."""
        def _h(acc):
            return acc
        return _h

    def _default_encode(self, rows, prog):
        """Default row->accumulator: bundle the categorical (col,value) bindings of every row into one prototype
        vector over the program's declared input columns. This is the 'pop the query result into the machine' step."""
        parts = []
        for r in rows:
            for col in prog["inputs"]:
                v = r.get(col)
                if isinstance(v, str):
                    parts.append(bind(self.symbols.get(col), self.symbols.get(v)))
        return bundle(parts) if parts else np.zeros(self.dim)

    def _result_confidence(self, acc, prog):
        """Confidence that the result is a real answer vs noise: the max cosine of the (normalized) accumulator to
        the program's declared output atoms. Low confidence => the program produced noise; it can abstain."""
        if not np.any(acc):
            return 0.0
        sims = [abs(float(cosine(acc, prog["machine"].data_atoms.get(o, np.zeros(self.dim)))))
                for o in prog["outputs"] if o in prog["machine"].data_atoms]
        # a cosine can round to just over 1.0 (e.g. 1.0000000000000002) depending on float summation order, so clamp:
        # a confidence is a probability-like quantity and must stay in [0, 1].
        return min(1.0, max(sims)) if sims else float(np.linalg.norm(acc) > 0)


# ---- a convenience: seed a system program (read-only) ------------------------------------------------------
def register_system_program(cat, name, instructions, doc, inputs, outputs, handlers, **kw):
    """Install a SYSTEM (read-only) program -- like a built-in faculty; SQL cannot uninstall it."""
    return cat.install(name, instructions, doc, inputs, outputs, handlers, tier="system", **kw)


def _selftest():
    cat = ProgramCatalog(dim=2048, seed=0)

    # PR5: install two user programs. "prototype" bundles a query column into a centroid; "tag" applies a faculty.
    cat.install("prototype",
                instructions=[("LOAD", "color"), ("HALT", None)],
                doc="build a prototype vector that clusters similar rows by their color",
                inputs=["color"], outputs=["color"], handlers=[], data=["color"])
    cat.install("normalize_tag",
                instructions=[("LOAD", "color"), ("APPLY", "normalize"), ("HALT", None)],
                doc="normalize and tag a group of records for anomaly detection",
                inputs=["color"], outputs=["color"], handlers=["normalize"], faculties=["normalize"])

    # PR1: the catalog lists both, as rows
    listing = cat.list()
    assert {r["name"] for r in listing} == {"prototype", "normalize_tag"}
    assert all("domain" in r and "tier" in r for r in listing)
    assert cat.list(tier="user") and not cat.list(tier="system")

    # PR4: find by MEANING -- a query about clustering surfaces the prototype program above the tag program
    hits = cat.find("group a series of similar things into clusters")
    assert hits[0]["name"] == "prototype", hits
    assert hits[0]["_confidence"] > hits[-1]["_confidence"]

    # PR3: explain is a dry run -- normalize_tag WOULD call the 'normalize' faculty, without executing it
    ex = cat.explain("normalize_tag")
    assert "normalize" in ex["faculties_called"] and ex["n_steps"] >= 1

    # PR6: execute over query rows -- sandboxed + step-bounded, result carries a confidence
    rows = [{"color": "red"}, {"color": "red"}, {"color": "blue"}]
    out = cat.execute("prototype", rows)
    assert "result" in out and 0.0 <= out["_confidence"] <= 1.0 and out["n_steps"] >= 1

    # sandbox: a program can only ever be handed its whitelisted handlers. prototype declared handlers=[], so even
    # though normalize_tag knows 'normalize', prototype's run gets an empty handler set.
    assert cat._programs["prototype"]["handlers"] == []

    # step bound: a tiny max_steps stops a program rather than letting it spin
    out2 = cat.execute("normalize_tag", rows, max_steps=1)
    assert out2["n_steps"] <= 2

    # system programs are read-only
    register_system_program(cat, "builtin", [("LOAD", "color"), ("HALT", None)],
                            doc="a built-in", inputs=["color"], outputs=["color"], handlers=[], data=["color"])
    try:
        cat.uninstall("builtin"); assert False, "should have refused"
    except QueryError:
        pass
    assert cat.uninstall("prototype") is True                    # a user program removes fine

    print("OK: holographic_queryprog self-test passed (install / list / find-by-meaning / explain / execute "
          "sandboxed+bounded / system-read-only -- PR1-PR6 on the shipped machine)")


if __name__ == "__main__":
    _selftest()
