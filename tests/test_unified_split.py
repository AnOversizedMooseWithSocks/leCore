"""Regression traps for the UnifiedMind mixin split.

UnifiedMind was one class of 17,613 lines in a single 1.31 MB file -- 131% of the cap an agent can read in
one pass, so the engine could no longer read its own central nervous system. It is now a ~320-line shim
assembling 13 mixin parts from holographic/unified/.

These tests pin the properties that made that split SAFE, because every one of them is a property a future
edit can quietly destroy while all the ordinary tests stay green.
"""
import ast
import glob
import os
import sys

import pytest

from holographic.misc.holographic_unified import (
    UnifiedMind, unified_sources, unified_source_text)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARTS_GLOB = os.path.join(_REPO, "holographic", "unified", "holographic_unified_p*.py")

# The shim held 17,621 lines before the split. This is not a style preference -- it is the gate that stops
# the file growing back into something the engine cannot read. Generous enough for real shim work.
SHIM_MAX_LOC = 1000
# Matches structure_audit's GIANT_LOC: a part that crosses this is a part that needs splitting again.
PART_MAX_LOC = 2000


def _part_classes():
    return [b for b in UnifiedMind.__bases__ if b.__name__.startswith("_UnifiedPart")]


def test_every_part_on_disk_is_actually_a_base_of_unified_mind():
    """THE ORPHAN TRAP. A part file that exists but is not in the bases list contributes nothing: every
    method in it silently disappears from the mind, and no import error is raised. Compare the filesystem
    against the LIVE class rather than trusting either one alone."""
    on_disk = {os.path.basename(p)[:-3] for p in glob.glob(_PARTS_GLOB)}
    in_bases = {b.__module__.rsplit(".", 1)[-1] for b in _part_classes()}
    assert on_disk, "no part files found -- has the layout moved?"
    assert on_disk == in_bases, (
        "part files and UnifiedMind's bases disagree; orphaned on disk: %s, missing from disk: %s"
        % (sorted(on_disk - in_bases), sorted(in_bases - on_disk)))


def test_no_method_name_is_defined_in_two_parts():
    """THE MRO TRAP, and the reason the split is safe at all.

    In a single class body a redefined name means THE LAST ONE WINS. Across mixin bases it means THE FIRST
    BASE WINS -- the opposite. So a duplicate name spread across two parts does not raise; it silently
    resolves to a different body than the same code would have in one file. (The original class had exactly
    one such duplicate, resolve_capability_uri, and it was removed during the split for precisely this
    reason.) While no name is duplicated, base ORDER is irrelevant and can never become load-bearing."""
    owner = {}
    dupes = []
    for path in sorted(glob.glob(_PARTS_GLOB)):
        mod = os.path.basename(path)[:-3]
        tree = ast.parse(open(path, encoding="utf-8").read())
        cls = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        assert len(cls) == 1, "%s should hold exactly one part class, found %d" % (mod, len(cls))
        for n in cls[0].body:
            if isinstance(n, ast.FunctionDef):
                if n.name in owner:
                    dupes.append((n.name, owner[n.name], mod))
                owner[n.name] = mod
    assert not dupes, "method name(s) defined in two parts -- MRO would pick one silently: %s" % dupes


def test_unified_sources_covers_the_whole_surface():
    """unified_source_text() is what the audits and text-searching tests read now. If it ever stops covering
    a part, those consumers go quietly blind rather than failing -- which is exactly how the split broke
    structure_audit, reachability_audit and test_queryembed_artifact in the first place."""
    srcs = unified_sources()
    assert len(srcs) == 1 + len(_part_classes()), "unified_sources() lost a file"
    assert all(os.path.isfile(p) for p in srcs), "unified_sources() names a path that does not exist"
    text = unified_source_text()
    # a string that only ever existed deep in the faculty bodies, never in the shim
    assert "_query_embedder" in text, "the assembled text is missing faculty bodies"
    assert len(text) > 1_000_000, "the assembled text is implausibly short (%d chars)" % len(text)


def test_the_shim_stays_a_shim_and_no_part_becomes_a_giant():
    """The whole point was file size. Pin it, or it grows back one faculty at a time."""
    shim = os.path.join(_REPO, "holographic", "misc", "holographic_unified.py")
    shim_loc = sum(1 for _ in open(shim, encoding="utf-8"))
    assert shim_loc < SHIM_MAX_LOC, (
        "the shim is back up to %d lines (max %d) -- new faculties belong in a PART, not here" % (shim_loc, SHIM_MAX_LOC))
    for path in sorted(glob.glob(_PARTS_GLOB)):
        loc = sum(1 for _ in open(path, encoding="utf-8"))
        assert loc < PART_MAX_LOC, "%s reached %d lines -- split it again" % (os.path.basename(path), loc)


def test_no_part_is_readable_as_a_standalone_module():
    """The parts are NOT a public API. They carry no __init__ and assume the state UnifiedMind.__init__
    builds, so instantiating one alone is a bug, not a shortcut. Pinned as a declared negative so nobody
    'promotes' a part to a real class."""
    for b in _part_classes():
        assert "__init__" not in vars(b), (
            "%s grew its own __init__ -- a part must never be independently constructible" % b.__name__)
        assert b.__name__.startswith("_"), "part classes stay underscore-private"


def test_the_faculty_surface_is_intact():
    """A blunt count gate. The service auto-introspects public mind methods into GET /tools, so a lost
    method is a lost HTTP tool -- and a split that drops 40 methods would otherwise pass every other test
    in this file."""
    m = UnifiedMind(dim=128, seed=0)
    public = [n for n in dir(m) if not n.startswith("_")]
    assert len(public) > 1400, "public faculty count fell to %d" % len(public)
    # spot-check one faculty from the first part and one from the last, resolved through the MRO
    assert callable(getattr(m, "read", None)) and callable(getattr(m, "mantis_falsecolor", None))


def test_resolve_capability_uri_kept_the_live_body_not_the_dead_one():
    """The one deliberate deletion. Two definitions existed; the LATER one was live and documents the
    URI-ONLY negative that a downstream integrator got wrong. If a future edit ever restores the earlier
    body, that hard-won docstring silently disappears."""
    doc = UnifiedMind.resolve_capability_uri.__doc__ or ""
    assert "URI-ONLY" in doc, "resolve_capability_uri lost the live body's URI-ONLY warning"
    m = UnifiedMind(dim=128, seed=0)
    assert m.resolve_capability_uri("render_mesh") == [], "the URI-ONLY negative no longer holds"
