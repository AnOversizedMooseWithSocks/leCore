"""Regression trap: the 13 UnifiedMind mixin parts must count as ONE referencing module.

Pure stdlib text analysis -- runs without the embedding model, which is the whole point: the defect this
pins was invisible to every model-free test and cost three blind CI pushes to find.

WHAT BROKE. Splitting UnifiedMind into 13 files turned one referencing module into thirteen. Each part
carries the same boilerplate import header, so mind/organizer/creature each gained +12 indegree. Edge weight
is raw_count * idf(dst) with idf = log(1 + N/(1+indeg)), so higher indegree means LOWER weight: the graph
quietly stopped treating those modules as specific.
"""
import os
import re

import pytest

import holographic.semantic_router.holographic_workflowgraph as wg
from holographic.semantic_router.holographic_workflowgraph import build_workflow_graph, _module_texts

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_mixin_parts_are_not_graph_nodes():
    g = build_workflow_graph(_REPO)
    nodes = set(g["out"]) | set(g["in"])
    parts = [n for n in nodes if re.match(r"^unified_p\d\d_", n)]
    assert not parts, "mixin parts are back as independent graph nodes: %s" % sorted(parts)[:5]


def test_the_facade_counts_once_not_thirteen_times():
    """THE LOAD-BEARING NUMBER. creature is imported by the boilerplate header in all 13 parts. If the merge
    regresses, its indegree jumps by ~12 and every edge into it is devalued by the idf term."""
    merged = _module_texts(_REPO, merge_parts=True)
    inflated = _module_texts(_REPO, merge_parts=False)
    assert len(inflated) - len(merged) >= 13, \
        "expected at least 13 part files to fold away, got %d" % (len(inflated) - len(merged))
    assert "unified" in merged and not any(k.startswith("unified_p") for k in merged)


def test_merging_removes_the_duplicated_header_edges():
    """Measured on the live repo: 1732 -> 1213 edges. Gate the SIGN and rough scale, not the exact figure,
    so ordinary development does not trip it."""
    g_on = build_workflow_graph(_REPO)
    orig = wg._module_texts
    wg._module_texts = lambda root, **kw: orig(root, merge_parts=False)
    try:
        g_off = build_workflow_graph(_REPO)
    finally:
        wg._module_texts = orig
    assert len(g_on["edges"]) < len(g_off["edges"]), "merging did not reduce the edge count"
    assert len(g_off["edges"]) - len(g_on["edges"]) > 200, \
        "expected the duplicated import header to account for hundreds of edges, got %d" \
        % (len(g_off["edges"]) - len(g_on["edges"]))


def test_no_mixin_part_is_ever_dropped_as_a_hub():
    """Two parts had crossed the 15% hub threshold and were being dropped from the graph while the facade
    they belong to was not -- a strictly wrong outcome, since the parts carry the facade's references."""
    g = build_workflow_graph(_REPO)
    bad = [h for h in g["dropped_hubs"] if re.match(r"^unified_p\d\d_", h)]
    assert not bad, "a mixin part was dropped as a hub: %s" % bad


def test_the_escape_hatch_reproduces_the_inflated_graph():
    """Keep the A/B runnable so the measurement can be re-checked instead of re-argued."""
    inflated = _module_texts(_REPO, merge_parts=False)
    assert any(k.startswith("unified_p") for k in inflated), "merge_parts=False no longer reproduces the old graph"
