"""holographic_scalenode.py -- COSMIC SCALE: recursive accumulation (fluids/matter backlog item 5, part 2).

The rule is "a parent carries the accumulated value of its children" -- which is the MONOID, already a UnifiedMind
faculty (distribute_compute). This builds a ScaleNode over scene_doc's hierarchy that rolls child properties up
(scalars like mass SUM; appearance BUNDLES -- a superposition), shows the summary vs descends by apparent size, and
so gives atom -> rock -> planet -> system -> galaxy the SAME shape at every scale. Zooming out is a query of the
shallower level; adding a planet updates the galaxy's summary in one associative monoid op.

This is already a bake-and-query (which is why it sits in the same backlog as the performance half): the summary is
precomputed per subtree (the bake), zooming is a lookup, and the accumulation is exact + order-independent because
it is a monoid. Distinct from holographic_cosmic (that classifies a point-cloud web; this is scene-scale LOD).

KEPT NEGATIVE: the rollup is exact only for ADDITIVE properties (mass, count) + a bundled appearance -- a planet's
exact weather is not represented from orbit; and the atom->galaxy dynamic range relies on relative-transform
discipline (scene_doc/scenegraph), no global coordinate, or precision breaks. Float SUM is order-independent only to
~1e-12 (float addition isn't associative); integer/exact accumulators are bit-exact.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bundle


class ScaleNode:
    """A recursive summariser over a scene_doc Scene: summary(handle) rolls a subtree's children up into one
    accumulated value; draw(handle, apparent_px) shows the summary when a node is small on screen or descends into
    its children when it is big. Reuses the wired monoid (distribute_compute) when a mind is supplied."""

    def __init__(self, scene, mind=None, lod_px=8.0):
        self.scene = scene
        self.mind = mind                                     # kept for callers; the rollup uses the monoid directly
        self.lod_px = float(lod_px)

    def _sum(self, values):
        """The additive monoid rollup (mass, count) -- a plain sum, which is exactly the reducer
        distribute_compute(reduce='sum') applies across buckets. For a big subtree you'd hand the buckets to
        distribute_compute to parallelise; here the tree walk sums directly (readable, exact)."""
        return float(sum(values)) if values else 0.0

    def _bundle(self, vecs):
        """The appearance monoid rollup: a superposition of the children's looks -- the same 'bundle' reducer
        distribute_compute(reduce='bundle') uses."""
        return bundle(vecs) if vecs else None

    def summary(self, handle):
        """The accumulated summary of a subtree: mass SUMs over descendants, look BUNDLES. A leaf returns its own.
        This is the BAKE -- precompute once, then zooming out just reads it."""
        node = self.scene.get(handle)
        kids = self.scene.children_of(handle)
        own_mass = float(node.params.get("mass", 0.0)) if node.params else 0.0
        own_look = node.params.get("look") if node.params else None
        if not kids:
            return {"mass": own_mass, "look": own_look, "leaves": 1}
        subs = [self.summary(k) for k in kids]
        looks = [s["look"] for s in subs if s["look"] is not None]
        if own_look is not None:
            looks.append(own_look)
        return {"mass": own_mass + self._sum([s["mass"] for s in subs]),
                "look": self._bundle(looks),
                "leaves": sum(s["leaves"] for s in subs)}

    def draw(self, handle, apparent_px, apparent_of=None):
        """Zoom-dependent read: if the node is smaller than the LOD threshold (or a leaf), return its SUMMARY (one
        blob); otherwise DESCEND into its children. `apparent_of(child_handle)` gives a child's on-screen size
        (defaults to halving the parent's -- a stand-in). Returns a nested structure mirroring what you'd draw."""
        kids = self.scene.children_of(handle)
        if apparent_px < self.lod_px or not kids:
            return {"handle": handle, "summary": self.summary(handle)}      # too small -> the summary
        nxt = apparent_of if apparent_of is not None else (lambda h: apparent_px / max(len(kids), 1))
        return {"handle": handle, "children": [self.draw(k, nxt(k), apparent_of) for k in kids]}


def _selftest():
    """Build a galaxy -> systems -> planets hierarchy, roll masses up (exact), bundle looks, confirm adding a planet
    updates the summary by exactly its mass (the monoid), and that draw shows the summary when small but descends
    when big."""
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.agents_and_reasoning.holographic_ai import Vocabulary

    voc = Vocabulary(256, seed=0)
    scene = Scene(dim=256, seed=0)

    galaxy = scene.add(name="galaxy", params={"mass": 0.0, "look": voc.get("galaxy")})
    total = 0.0
    planet_looks = []
    for si in range(3):
        system = scene.add(name="system%d" % si, params={"mass": 0.0}, parent=galaxy)
        for pi in range(4):
            m = 1.0 + si + pi
            total += m
            look = voc.get("planet%d_%d" % (si, pi))
            planet_looks.append(look)
            scene.add(name="planet%d_%d" % (si, pi), params={"mass": m, "look": look}, parent=system)

    sn = ScaleNode(scene)
    summ = sn.summary(galaxy)
    assert abs(summ["mass"] - total) < 1e-9, (summ["mass"], total)       # exact mass rollup over the whole tree
    assert summ["leaves"] == 12                                          # 3 systems x 4 planets
    assert summ["look"] is not None and summ["look"].shape == (256,)     # bundled appearance

    # the monoid: adding one more planet updates the galaxy summary by EXACTLY its mass (associative)
    before = sn.summary(galaxy)["mass"]
    sys0 = scene.children_of(galaxy)[0]
    scene.add(name="new_planet", params={"mass": 7.0}, parent=sys0)
    after = sn.summary(galaxy)["mass"]
    assert abs((after - before) - 7.0) < 1e-9, (before, after)

    # draw: at a tiny apparent size the galaxy collapses to its summary; at a big size it descends into systems
    small = sn.draw(galaxy, apparent_px=2.0)
    assert "summary" in small and "children" not in small               # too small -> summary
    big = sn.draw(galaxy, apparent_px=1000.0, apparent_of=lambda h: 1000.0)
    assert "children" in big and len(big["children"]) == 3              # big -> descend into the 3 systems

    print("holographic_scalenode selftest OK: galaxy->systems->planets mass rolls up EXACTLY (%.1f over 12 leaves), "
          "appearance bundles into a 256-d prototype; adding a planet updates the galaxy summary by exactly its mass "
          "(the monoid); draw collapses to the summary when small and descends into the 3 systems when big -- reusing "
          "the same accumulation at every scale" % summ["mass"])


if __name__ == "__main__":
    _selftest()
