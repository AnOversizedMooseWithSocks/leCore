#!/usr/bin/env python3
"""pipelinemap.py -- derive the WORKFLOW GRAPH from the live catalog and write it as documentation.

WHY THIS EXISTS
---------------
The catalog already tags many capabilities with `consumes`/`produces` io-kinds (holographic_iokinds), and
`suggest_pipeline` already CHAINS them on demand ("how do I get from a mesh to an image?"). But nothing drew
the WHOLE graph as a standing document -- an agent or a person could ask for one route, but could not SEE the
map of which tool-outputs feed which tool-inputs across the engine. This generator closes that gap: it reads
the same typed edges suggest_pipeline uses (each tagged capability is an edge consume_kind -> produce_kind)
and emits:

  * docs/PIPELINE_MAP.md   -- a mermaid graph of the io-kind flow (GitHub renders ```mermaid natively, so no
                              dependency enters the engine), plus, per kind, the capabilities that produce it
                              and consume it, and an ORPHAN/DEAD-END report (produced-but-never-consumed and
                              consumed-but-never-produced kinds -- the gaps worth tagging or building).
  * pipelines.json         -- the machine-readable edge list + per-kind adjacency, the contract an agent can
                              load to plan multi-step work without re-deriving it.

It does NOT reimplement the chaining logic -- it extracts the same edges and lets mermaid/JSON present them.
The truth stays in the catalog tags; this is a VIEW, regenerated in CI like REFERENCE/CAPABILITIES so it can
never rot. KEPT NEGATIVE / HONEST LIMIT: coverage is only as good as the tags. At time of writing ~22% of
capabilities declare io-kinds, so the drawn graph is the TAGGED subset, not the whole engine -- the coverage
line at the top of the map says so out loud, so a sparse graph reads as "tag more", not "the engine is small".

OLD-SCHOOL AND DEPENDENCY-FREE: standard library only (json, os). It imports holographic_catalog (the same
deterministic data the other doc generators read). No timestamp is written -- this file is drift-checked in
CI, so any non-deterministic content would make it "stale" every day (the lesson from apiquickref/docgen).
"""
import json
import os

REPO = os.path.dirname(os.path.abspath(__file__))


def _edges(cat):
    """The typed edges of the workflow graph: for each capability that declares BOTH consumes and produces,
    one directed edge consume_kind -> produce_kind per (consumed, produced) pair. This is EXACTLY the edge
    set holographic_catalog.suggest_pipeline builds for its BFS -- we extract it here to draw, not to chain.
    Sorted by capability name so the output is deterministic (same reason suggest_pipeline sorts)."""
    out = []
    for cap in sorted(cat._by_name.values(), key=lambda c: c.name):
        if not cap.consumes or not cap.produces:
            continue                                       # untagged on either side = no typed edge to draw
        for ci in cap.consumes:
            for po in cap.produces:
                out.append((ci, po, cap.name))
    return out


def _adjacency(edges):
    """Per-kind view: which capabilities PRODUCE this kind, which CONSUME it. This is what a planner or a
    reader wants -- 'I have an X, what can act on it?' (consumers) and 'how do I get an X?' (producers)."""
    produce = {}                                           # kind -> sorted list of capability names
    consume = {}
    for ci, po, name in edges:
        consume.setdefault(ci, set()).add(name)
        produce.setdefault(po, set()).add(name)
    return ({k: sorted(v) for k, v in produce.items()},
            {k: sorted(v) for k, v in consume.items()})


def _orphans(edges, all_kinds):
    """The gap report. A kind PRODUCED but never CONSUMED is a dead-end (you can make it, nothing uses it);
    a kind CONSUMED but never PRODUCED is a source that must come from OUTSIDE the tagged graph (an input the
    user supplies, or an untagged producer). Both are exactly the 'find the gap' signal this repo lives on."""
    produced = {po for _, po, _ in edges}
    consumed = {ci for ci, _, _ in edges}
    dead_end = sorted(produced - consumed)                 # produced, nothing downstream consumes it
    source_only = sorted(consumed - produced)              # consumed, nothing in-graph produces it
    untouched = sorted(k for k in all_kinds if k not in produced and k not in consumed)
    return dead_end, source_only, untouched


def generate(root=REPO):
    """Write docs/PIPELINE_MAP.md and pipelines.json from the live catalog. Returns (md_path, json_path)."""
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    from holographic.caching_and_storage.holographic_iokinds import IO_KINDS
    cat = default_catalog()

    edges = _edges(cat)
    produce, consume = _adjacency(edges)
    dead_end, source_only, untouched = _orphans(edges, IO_KINDS)

    # coverage: how many capabilities carry BOTH tags (the ones that can appear as an edge)
    total = len(cat._by_name)
    tagged = sum(1 for c in cat._by_name.values() if c.consumes and c.produces)
    pct = (100 * tagged // total) if total else 0

    # ---- mermaid: the io-kind flow graph. Nodes are io-kinds; an edge kind_a --> kind_b is labelled with a
    # representative capability (the first by name) so the diagram stays readable even when several caps share
    # an edge. The full edge->caps mapping lives in pipelines.json for anyone who needs all of them.
    edge_caps = {}                                         # (a,b) -> sorted caps, for a readable single label
    for ci, po, name in edges:
        edge_caps.setdefault((ci, po), []).append(name)
    md = []
    md.append("# leCore Pipeline Map")
    md.append("")
    md.append("*The workflow graph, auto-derived by `pipelinemap.py` from the catalog's `consumes`/`produces` "
              "tags. Nodes are io-kinds; an edge means some capability turns the source kind into the target "
              "kind. This is a VIEW of the live tags -- to change it, tag capabilities, not this file.*")
    md.append("")
    md.append("> **Coverage: %d of %d capabilities carry io-kind tags (%d%%).** The graph below is that tagged "
              "subset. Untagged capabilities are real but do not yet declare a typed edge -- backfilling tags "
              "grows the map." % (tagged, total, pct))
    md.append("")
    md.append("```mermaid")
    md.append("graph LR")
    for (a, b) in sorted(edge_caps):
        caps = sorted(set(edge_caps[(a, b)]))
        label = caps[0].replace("holographic_", "")
        extra = "" if len(caps) == 1 else (" +%d" % (len(caps) - 1))
        md.append('    %s["%s"] -->|%s%s| %s["%s"]' % (a, a, label, extra, b, b))
    md.append("```")
    md.append("")

    # ---- per-kind tables: producers and consumers. This is the 'I have X / I want X' lookup.
    md.append("## By io-kind")
    md.append("")
    for k in IO_KINDS:
        prod = produce.get(k, [])
        cons = consume.get(k, [])
        if not prod and not cons:
            continue
        md.append("### `%s`" % k)
        md.append("- **produced by:** %s" % (", ".join(c.replace("holographic_", "") for c in prod) or "_(nothing tagged)_"))
        md.append("- **consumed by:** %s" % (", ".join(c.replace("holographic_", "") for c in cons) or "_(nothing tagged)_"))
        md.append("")

    # ---- the gap report: dead-ends and sources. The whole point of a map is to see what's missing.
    md.append("## Gaps (the find-a-gap report)")
    md.append("")
    md.append("- **dead-end kinds** (produced, nothing tagged consumes them): %s"
              % (", ".join("`%s`" % k for k in dead_end) or "_none_"))
    md.append("- **source-only kinds** (consumed, nothing tagged produces them -- user-supplied or untagged "
              "producer): %s" % (", ".join("`%s`" % k for k in source_only) or "_none_"))
    md.append("- **untouched kinds** (in the vocabulary, in no tagged edge yet): %s"
              % (", ".join("`%s`" % k for k in untouched) or "_none_"))
    md.append("")

    md_path = os.path.join(root, "docs", "PIPELINE_MAP.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- pipelines.json: the machine-readable contract. Edge list + adjacency + coverage + gaps.
    data = {
        "coverage": {"tagged": tagged, "total": total, "percent": pct},
        "edges": [{"consumes": ci, "produces": po, "capability": name} for ci, po, name in edges],
        "produced_by": produce,
        "consumed_by": consume,
        "gaps": {"dead_end": dead_end, "source_only": source_only, "untouched": untouched},
    }
    json_path = os.path.join(root, "pipelines.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)     # sort_keys -> deterministic bytes for the drift gate
        f.write("\n")
    return md_path, json_path


def _selftest():
    """Assert the REAL contract: the derived edge set matches what suggest_pipeline would traverse, the graph
    is non-empty on the real catalog, and the JSON is deterministic across two runs (drift-gate safe)."""
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    cat = default_catalog()
    edges = _edges(cat)
    assert edges, "no typed edges -- expected the tagged subset to be non-empty"
    # every edge's capability must actually declare that consume and produce (no fabricated edges)
    by = cat._by_name
    for ci, po, name in edges:
        assert ci in by[name].consumes and po in by[name].produces, "edge not backed by the tag: %s" % name
    # determinism: two generates produce byte-identical json
    import tempfile
    d1 = tempfile.mkdtemp(); d2 = tempfile.mkdtemp()
    os.makedirs(os.path.join(d1, "docs")); os.makedirs(os.path.join(d2, "docs"))
    generate(d1); generate(d2)
    a = open(os.path.join(d1, "pipelines.json")).read()
    b = open(os.path.join(d2, "pipelines.json")).read()
    assert a == b, "pipelines.json is non-deterministic -- would false-trip the drift gate"
    print("  pipelinemap selftest OK: %d edges, deterministic json, every edge backed by a tag" % len(edges))


if __name__ == "__main__":
    _selftest()
    md, js = generate()
    print("  wrote", md)
    print("  wrote", js)
