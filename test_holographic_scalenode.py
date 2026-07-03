"""Fluids/matter item 5 (part 2): ScaleNode -- recursive monoid rollup over a scene hierarchy."""
import numpy as np
from holographic_ai import Vocabulary
from holographic_scene_doc import Scene
from holographic_scalenode import ScaleNode


def _galaxy():
    voc = Vocabulary(256, seed=0)
    scene = Scene(dim=256, seed=0)
    galaxy = scene.add(name="galaxy", params={"mass": 0.0, "look": voc.get("galaxy")})
    total = 0.0
    for si in range(3):
        system = scene.add(name="system%d" % si, params={"mass": 0.0}, parent=galaxy)
        for pi in range(4):
            m = 1.0 + si + pi
            total += m
            scene.add(name="p%d_%d" % (si, pi), params={"mass": m, "look": voc.get("p%d_%d" % (si, pi))}, parent=system)
    return scene, galaxy, total


def test_mass_rolls_up_exactly():
    scene, galaxy, total = _galaxy()
    assert abs(ScaleNode(scene).summary(galaxy)["mass"] - total) < 1e-9


def test_leaf_count():
    scene, galaxy, _ = _galaxy()
    assert ScaleNode(scene).summary(galaxy)["leaves"] == 12


def test_look_bundles():
    scene, galaxy, _ = _galaxy()
    look = ScaleNode(scene).summary(galaxy)["look"]
    assert look is not None and look.shape == (256,)


def test_adding_child_updates_summary_by_its_mass():
    scene, galaxy, _ = _galaxy()
    sn = ScaleNode(scene)
    before = sn.summary(galaxy)["mass"]
    sys0 = scene.children_of(galaxy)[0]
    scene.add(name="new", params={"mass": 7.0}, parent=sys0)
    assert abs((sn.summary(galaxy)["mass"] - before) - 7.0) < 1e-9    # the monoid: one associative update


def test_draw_summary_when_small_descend_when_big():
    scene, galaxy, _ = _galaxy()
    sn = ScaleNode(scene, lod_px=8.0)
    small = sn.draw(galaxy, apparent_px=2.0)
    assert "summary" in small and "children" not in small
    big = sn.draw(galaxy, apparent_px=1000.0, apparent_of=lambda h: 1000.0)
    assert "children" in big and len(big["children"]) == 3


def test_leaf_returns_own():
    scene, galaxy, _ = _galaxy()
    leaf = scene.children_of(scene.children_of(galaxy)[0])[0]
    s = ScaleNode(scene).summary(leaf)
    assert s["leaves"] == 1 and s["mass"] == 1.0                # first planet: si=0,pi=0 -> mass 1.0
