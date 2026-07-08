"""Tests for G5 procedural grammar (holographic_grammar): context-free L-system parallel rewriting, a 3D
turtle that emits a skeleton, scenegraph assembly (each segment instanced through a transform -- a recursive
bundle), and productions carried as a holographic record. Recursive composition, not a growth simulation."""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import unbind, cosine
from holographic.agents_and_reasoning.holographic_grammar import LSystem, productions_record, turtle_to_segments, grow_plant, greeble_panel, _selftest
from holographic.scene_and_pipeline.holographic_scenegraph import flatten_scene
from holographic.misc.holographic_fractal import box_counting_dimension


def test_algae_lengths_are_fibonacci():
    algae = LSystem("A", {"A": "AB", "B": "A"})
    assert [len(algae.expand(n)) for n in range(7)] == [1, 2, 3, 5, 8, 13, 21]


def test_expansion_is_deterministic():
    algae = LSystem("A", {"A": "AB", "B": "A"})
    assert algae.expand(5) == algae.expand(5)


def test_turtle_skeleton_grows_with_depth():
    plant = LSystem("X", {"X": "F[+X][-X]FX", "F": "FF"})
    assert len(turtle_to_segments(plant.expand(4))) > len(turtle_to_segments(plant.expand(2))) > 0


def test_branching_fills_the_plane():
    plant = LSystem("X", {"X": "F[+X][-X]FX", "F": "FF"})
    def dim(symbols):
        segs = turtle_to_segments(symbols)
        pts = np.array([p for seg in segs for p in seg])[:, [0, 2]]
        return box_counting_dimension(pts)
    assert dim(plant.expand(4)) >= dim(plant.expand(2)) - 0.05


def test_productions_record_recovers_a_rule():
    plant = LSystem("X", {"X": "F[+X][-X]FX", "F": "FF"})
    rec, atoms = productions_record(plant.productions)
    recovered = unbind(rec, atoms["X"])
    s = cosine(recovered, atoms["exp:X"])
    assert s > 0.45 and s > 3 * abs(cosine(recovered, atoms["exp:F"]))


def test_grow_plant_and_greeble_determinism():
    plant = LSystem("X", {"X": "F[+X][-X]FX", "F": "FF"})
    mesh, _, _ = grow_plant(plant, 3, angle_deg=25, step=0.5)
    assert mesh.n_vertices > 0 and len(mesh.faces) > 0
    a = flatten_scene(greeble_panel(2.0, 1.0, seed=1, max_depth=3))
    b = flatten_scene(greeble_panel(2.0, 1.0, seed=1, max_depth=3))
    assert np.allclose(a.vertices, b.vertices)


def test_selftest_runs():
    _selftest()
