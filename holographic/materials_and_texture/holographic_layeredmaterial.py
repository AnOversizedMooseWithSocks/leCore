"""holographic_layeredmaterial.py -- CMP2: an ORDERED stack of material layers, with a layer-ORDER schema.

A layered material is what real surfaces are: a base under a diffuse under a specular/reflection under a
coat/clearcoat. The ORDER matters -- a clearcoat sits ON TOP of the paint, never under it -- so this module makes the
order a SCHEMA that is checked when you build the stack: you cannot put a reflection layer below a diffuse one. That
is the "with respect to hierarchy" discipline for materials, enforced at COMPOSE time (a clear error up front) rather
than discovered as a wrong-looking render.

Each layer composites OVER the accumulated result below it: from the bottom up,

    value = alpha_layer(uv) * layer.value(uv) + (1 - alpha_layer(uv)) * value_below

-- the same "over" mix Material.blend seeds, lifted to a stack, where a layer's coverage `alpha` may itself vary over
the surface (a constant, a field, or a CMP1 texture graph -- so a coat can cover only part of a surface). Sampling
walks the stack bottom-to-top.

KEPT NEGATIVE (loud): ORDERING IS NOT ENERGY CONSERVATION. This fixes the STACKING (which layer is above which, and a
simple over-composite of their values); it does NOT do the RADIOMETRY of a true layered BRDF, where a coat physically
darkens and tints what is under it (Fresnel at the interface, absorption through the coat). We ship correct ordering +
an honest over-composite; a physically-correct layered BRDF (energy conserved across the interface) is a separate,
harder thing and is NOT claimed here.

Reuses (no new machinery): holographic_material.Material (each layer is a Material; the per-channel sample and the
2-way "over" it seeds), holographic_texturegraph (CMP1 -- a coverage alpha is a texture graph), holographic_typed
(the optional vector form, for caching/searching a stack by its structure). Plain NumPy; the stack stays a readable
list you can print.
"""
import numpy as np

from holographic.materials_and_texture.holographic_texturegraph import _coerce, _s              # a coverage alpha is a CMP1 node; _s scalarises it

# The canonical layer order, bottom (0) to top. Same RANK == same tier (may coexist / be adjacent); a layer may only
# sit ABOVE one of equal-or-lower rank. This one table is the whole "order schema" -- readable, and easy to extend.
LAYER_RANK = {
    "base": 0,          # the substrate
    "diffuse": 1,       # the matte body colour
    "specular": 2,      # glossy highlight tier ...
    "reflection": 2,    # ... same tier as specular
    "coat": 3,          # a clear coat on top ...
    "clearcoat": 3,     # ... same tier
}


class Layer:
    """One layer of a stack: a KIND (which fixes its place in the order), a Material carrying its channels, and a
    coverage `alpha` -- how much of what is below shows through, a number in [0,1], a field, or a CMP1 texture graph
    (so coverage can vary over the surface). The bottom layer's alpha is ignored (there is nothing beneath it)."""

    def __init__(self, kind, material, alpha=1.0):
        if kind not in LAYER_RANK:
            raise ValueError("unknown layer kind %r -- known: %s" % (kind, ", ".join(LAYER_RANK)))
        if not (hasattr(material, "channels") and hasattr(material, "sample")):
            raise TypeError("layer %r needs a Material (with .channels and .sample), got %r"
                            % (kind, type(material).__name__))
        self.kind = kind
        self.material = material
        self._alpha = _coerce(alpha)                          # reuse CMP1: number | field | texture graph
        self.rank = LAYER_RANK[kind]

    def has(self, channel):
        return channel in self.material.channels

    def value(self, channel, uv):
        return self.material.sample(channel, uv)

    def alpha_at(self, uv):
        a = _s(self._alpha.sample(uv))
        return float(np.clip(a, 0.0, 1.0))                    # coverage is a fraction

    def __repr__(self):
        return "Layer(%s, %d channels)" % (self.kind, len(self.material.channels))


class LayeredMaterial:
    """An ORDERED stack of layers, bottom-to-top, with the order enforced by LAYER_RANK. Sample a channel with
    sample(channel, uv): it composites the layers that carry that channel from the bottom up, each OVER the one below
    by its coverage alpha."""

    def __init__(self, layers=None):
        self.layers = []
        for layer in (layers or []):
            self.add(layer)

    def add(self, layer):
        """Append a layer ON TOP of the stack. Refused at COMPOSE time if it would sit below a higher-ranked layer
        (e.g. a diffuse placed above a reflection) -- that is the order schema doing its job."""
        if not isinstance(layer, Layer):
            raise TypeError("add() takes a Layer, got %r" % type(layer))
        if self.layers and layer.rank < self.layers[-1].rank:
            raise ValueError("layer %r (tier %d) cannot sit above %r (tier %d): higher tiers -- "
                             "specular/reflection, then coat/clearcoat -- must be ABOVE base/diffuse, not below"
                             % (layer.kind, layer.rank, self.layers[-1].kind, self.layers[-1].rank))
        self.layers.append(layer)
        return self

    def channel_names(self):
        """Every channel any layer carries (the union)."""
        names = set()
        for layer in self.layers:
            names |= set(layer.material.channels)
        return sorted(names)

    def sample(self, channel, uv):
        """Composite `channel` bottom-to-top: start at the lowest layer that has it, then for each higher layer that
        has it, value = alpha*layer + (1-alpha)*value_below. Layers without the channel are skipped (they don't
        occlude a channel they don't define). Returns 0.0 if no layer carries the channel."""
        value = None
        for layer in self.layers:                             # self.layers is bottom-to-top
            if not layer.has(channel):
                continue
            v = layer.value(channel, uv)
            if value is None:
                value = v                                     # the lowest layer with this channel is the base for it
            else:
                a = layer.alpha_at(uv)
                value = a * v + (1.0 - a) * value             # "over": this layer composited onto what's below
        return 0.0 if value is None else value

    def sample_all(self, uv):
        """Every channel at `uv` -> {name: value}."""
        return {name: self.sample(name, uv) for name in self.channel_names()}

    def to_expr(self):
        """The typed-tree form: ('stack', ('layer:base', ...channels...), ('layer:coat', ...)) bottom-to-top -- a
        structural signature of the stack (its layer kinds + channels, in order), for the optional encode path."""
        parts = []
        for layer in self.layers:
            chans = tuple(sorted(layer.material.channels))
            parts.append(tuple(["layer:" + layer.kind] + list(chans)) if chans else "layer:" + layer.kind)
        return tuple(["stack"] + parts)

    def encode(self, dim, seed=0):
        """The stack as ONE hypervector (via holographic_typed.encode_tree) -- to cache a baked stack by its identity
        or find similar stacks. Encodes the ORDERED structure (kinds + channels), not per-point values. Two stacks
        with the same layers in the same order encode identically; reordering them changes the code."""
        from holographic.misc.holographic_typed import encode_tree
        return encode_tree(dim, seed, self.to_expr())

    def __repr__(self):
        return "LayeredMaterial([%s])" % ", ".join(l.kind for l in self.layers)


def _selftest():
    import numpy as np
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    from holographic.materials_and_texture.holographic_material import Material, texture_field

    # Material.sample is a COSINE (direction) readout, so layers must differ in PATTERN. Two opposite ramps again.
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 7) for v in np.linspace(0.05, 0.95, 7)]
    base_mat = Material(enc, {"albedo": texture_field(enc, grid, [u for (u, v) in grid])})         # ramps up
    coat_mat = Material(enc, {"albedo": texture_field(enc, grid, [1.0 - u for (u, v) in grid])})   # ramps down

    base = Layer("base", base_mat)                            # substrate
    coat = Layer("coat", coat_mat, alpha=0.4)                # 40%-covering clear coat on top
    stack = LayeredMaterial([base, coat])
    assert [l.kind for l in stack.layers] == ["base", "coat"]

    # the composite is exactly over(coat, base, 0.4) = 0.4*coat + 0.6*base, per point
    for uv in ([0.2, 0.5], [0.5, 0.5], [0.8, 0.5]):
        expect = 0.4 * coat_mat.sample("albedo", uv) + 0.6 * base_mat.sample("albedo", uv)
        assert abs(stack.sample("albedo", uv) - expect) < 1e-9, uv

    # ORDER SCHEMA: a coat is fine ABOVE a base; a base is REFUSED above a coat (out of order) -- at compose time.
    try:
        LayeredMaterial([coat, base])                        # base (tier 0) above coat (tier 3) -> refused
        raise AssertionError("order schema should refuse a base above a coat")
    except ValueError as e:
        assert "cannot sit above" in str(e)
    # equally, adding a diffuse on top of a reflection is refused
    diff = Layer("diffuse", base_mat)
    refl = Layer("reflection", coat_mat, alpha=0.5)
    try:
        LayeredMaterial([refl, diff])                        # diffuse (1) above reflection (2) -> refused
        raise AssertionError("cannot put diffuse above reflection")
    except ValueError:
        pass
    # a valid ascending stack builds fine
    ok = LayeredMaterial([Layer("base", base_mat), Layer("diffuse", base_mat),
                          Layer("specular", coat_mat, alpha=0.3), Layer("clearcoat", coat_mat, alpha=0.2)])
    assert len(ok.layers) == 4

    # a VARYING coverage alpha (a CMP1 field): coat shows more on the right than the left
    from holographic.materials_and_texture.holographic_texturegraph import FieldLeaf
    grad = FieldLeaf(lambda pts: np.asarray(pts, float)[:, 0])   # alpha = u
    varcoat = LayeredMaterial([Layer("base", base_mat), Layer("coat", coat_mat, alpha=grad)])
    left = varcoat.sample("albedo", [0.1, 0.5])              # alpha~0.1 -> mostly base
    right = varcoat.sample("albedo", [0.9, 0.5])            # alpha~0.9 -> mostly coat
    assert abs(left - base_mat.sample("albedo", [0.1, 0.5])) < abs(left - coat_mat.sample("albedo", [0.1, 0.5]))
    assert abs(right - coat_mat.sample("albedo", [0.9, 0.5])) < abs(right - base_mat.sample("albedo", [0.9, 0.5]))

    # optional structural encode: same stack -> same code; a different order -> different code
    v_ok = ok.encode(1024)
    v_same = LayeredMaterial([Layer("base", base_mat), Layer("diffuse", base_mat),
                              Layer("specular", coat_mat, alpha=0.3), Layer("clearcoat", coat_mat, alpha=0.2)]).encode(1024)
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos(v_ok, v_same) > 0.99

    print("OK: holographic_layeredmaterial self-test passed (base+coat composites to 0.4*coat+0.6*base exactly; the "
          "order schema REFUSES base-above-coat and diffuse-above-reflection at compose time; a valid 4-tier stack "
          "builds; a varying-alpha coat shows more where the mask is higher; structural encode matches)")


if __name__ == "__main__":
    _selftest()
