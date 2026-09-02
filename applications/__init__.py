"""applications -- THE LIBRARY: named, end-to-end programs you can run, not snippets you can read.

WHY THIS EXISTS (sweep 124, backlog A, verdict verbatim): "THE MACHINERY IS ALIVE; THE LIBRARY DOES NOT
EXIST." That sweep checked and found vsarun / vsabake / query_programs / signalprogram all selftesting
green, and six application-shaped asks routing to real faculties -- and then found no curated collection
of end-to-end runnable programs anywhere: no applications/, no examples/, GALLERY at five sections, and
writing_vsa_programs.md an ISA manual rather than a cookbook. Eight sweeps later the audit returned the
same fallbacks (Program & machine (VM), Material library, decoded-instruction cache) for every phrasing
of "run a named example". A reader could not answer "show me what this does" without writing the program
themselves.

WHAT AN APPLICATION IS HERE, and every clause is enforced somewhere
------------------------------------------------------------------
  * NAMED and DISCOVERABLE -- `mind.apps()` lists it, `mind.app_run(name)` runs it. A folder of scripts
    nobody can find through the engine is the exact failure this repo is organised against; by its own
    governing rule a capability find_capability cannot surface does not exist.
  * END-TO-END -- it starts from data it makes itself and finishes at a result or an artefact. No fixture
    downloads, no model directory, no network.
  * IT PROVES A NUMBER. `run()` returns a `proved` dict, and the number is asserted, not printed. "It ran"
    is not a demonstration; tools/showcase.py established that pattern for engine claims and this borrows
    it for applications.
  * IT GOES THROUGH FACULTIES. An application calls `mind.<verb>()` and imports nothing from
    `holographic.*`. This is the load-bearing rule: a program that reaches into modules directly is a
    script, it bypasses the surface every other audit protects, and it rots silently the first time a
    faculty changes. tests/test_applications.py parses each file's AST and fails on a `holographic`
    import, so the rule cannot decay into a comment.
  * IT IS AFFORDABLE. An application nobody can afford to run is a rotting example. Every one here is
    budgeted in single-digit seconds and the total is asserted in the tests.

KEPT NEGATIVE -- THE COMPARISON, STATED HONESTLY. Torchhd (JMLR 24) is the reference open-source VSA/HDC
library and it ships an examples/ directory; leCore shipped none, and that gap is the whole case for this
item. It is NOT a like-for-like win and must never be written as one: Torchhd's examples are machine
learning tasks benchmarked on PUBLIC DATASETS (MNIST, UCI, language ID), which is a claim about accuracy
against a shared yardstick. These are end-to-end programs over leCore's own machinery, which is a claim
about reach and reproducibility and nothing else. Anyone reading both repos must find that sentence true.

KEPT NEGATIVE -- SCOPE. This is a FIRST TRANCHE, four domains of the six sweep 124 listed. 3-D
(points_to_mesh -> decimate -> unwrap -> render) and advanced-algorithms (resonator factoring at scale,
Physarum) are NOT here. Four that work end to end beat a dozen that nearly do.

SOURCE CHECKOUT ONLY, said out loud: this package sits beside the engine rather than inside it, so a
wheel install does not carry it and `mind.apps()` raises a legible ImportError instead of reporting an
empty library. An empty list would read as "there are no applications", which is a different and wrong
answer.
"""
import importlib
import time

#: name -> (module path, domain). The ONE place a name is bound to code; `apps()` and `run()` both read
#: it, so a library entry cannot exist in the listing and be unrunnable, or vice versa.
REGISTRY = {
    "spectral_heat": ("applications.math.spectral_heat", "math"),
    "interleaved_sources": ("applications.demux.interleaved_sources", "demux"),
    "request_to_record": ("applications.parse.request_to_record", "parse"),
    "texture_composite": ("applications.art.texture_composite", "art"),
    "infinite_zoom": ("applications.demoscene.infinite_zoom", "demoscene"),
}


def _load(name):
    """Import one application module by registry name, with an error that names the library.

    ValueError, NOT KeyError, and the reason is the HTTP boundary rather than Python taste: the service
    maps ValueError from a faculty to {ok: false, error: ...} and lets anything else become a 500. Asking
    for an application that does not exist is a CALLER error -- the same rule ObjectRefs already follows
    for a bad handle -- and a 500 tells an agent the engine broke rather than that it mistyped a name.
    MEASURED before this was fixed: POST /invoke app_run {"name": "nope"} returned HTTP 500."""
    if name not in REGISTRY:
        raise ValueError("no application %r; mind.apps() lists %s" % (name, sorted(REGISTRY)))
    return importlib.import_module(REGISTRY[name][0])


def apps():
    """The library, as records: [{name, domain, proves, artefact}] -- what mind.apps() returns.

    Reads each application's own declared metadata rather than a second copy kept here, because two
    descriptions of one program drift and the one in the listing is the one nobody runs."""
    out = []
    for name, (path, domain) in sorted(REGISTRY.items()):
        mod = importlib.import_module(path)
        out.append({"name": name, "domain": domain, "proves": mod.PROVES,
                    "artefact": getattr(mod, "ARTEFACT", None), "module": path})
    return out


def run(mind, name, **kw):
    """Run one application end to end and return {name, domain, proves, seconds, proved, ...}.

    `seconds` is measured here rather than trusted: an application's cost is part of whether it is a
    usable example, and the number belongs beside the result rather than in a comment that ages."""
    mod = _load(name)
    t0 = time.time()
    result = mod.run(mind, **kw)
    result.update({"name": name, "domain": REGISTRY[name][1], "proves": mod.PROVES,
                   "seconds": round(time.time() - t0, 3)})
    return result


def _selftest():
    """Every registered name must import, declare its metadata, and expose run() -- the listing and the
    library cannot disagree. Pins the COUNT so a silently dropped application fails loudly."""
    names = sorted(REGISTRY)
    assert len(names) == 5, names
    listed = apps()
    assert [a["name"] for a in listed] == names, "apps() must list exactly the registry"
    for a in listed:
        mod = importlib.import_module(a["module"])
        assert callable(mod.run) and callable(mod._selftest), a["name"]
        assert isinstance(mod.PROVES, str) and len(mod.PROVES) > 20, "%s: say what it proves" % a["name"]
        assert mod.NAME == a["name"] and mod.DOMAIN == a["domain"], "%s: metadata disagrees" % a["name"]
    assert len({a["domain"] for a in listed}) == 5, "one application per domain in this tranche"
    print("applications registry selftest OK: %d applications across %d domains -- %s"
          % (len(names), len({a["domain"] for a in listed}), ", ".join(names)))


if __name__ == "__main__":
    _selftest()
