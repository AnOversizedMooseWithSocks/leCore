"""holographic_multimaterial.py -- CMP3: N materials selected/blended per-point by weight MASKS.

Material.blend mixes TWO materials by one scalar t -- the same everywhere. CMP3 generalises that to N materials
mixed by a set of MASKS, one weight per material, where each mask VARIES over the surface (a CMP1 texture graph, a
field, or a constant). At a point the value of a channel is

    sum_i  w_i(uv) * material_i.sample(channel, uv)

-- a superposition of the materials WEIGHTED BY A FIELD, which is exactly a bundle whose coefficients are the mask.
That is how you paint rust into metal, moss onto stone, or a decal onto a surface: the mask says how much of each
material shows where.

KEPT NEGATIVE (loud): the weights must PARTITION -- sum to ~1 at every point -- or total brightness drifts (a point
where the masks sum to 1.6 comes out ~60% too bright). So by default we NORMALISE the weights to a partition of unity
at each point, w_i / sum_j w_j, with a uniform fallback wherever every mask is ~0 (so nothing goes black). Pass
normalize=False only when your masks already partition and you want the raw sum.

Two modes: 'blend' (soft weighted sum -- the default, smooth transitions) and 'select' (hard pick -- the single
highest-weight material at each point, for a material-ID / splat map where you want crisp boundaries, no cross-fade).

Reuses (no new machinery): holographic_material.Material (the per-channel exact sampling, and the 2-way blend this
generalises), holographic_texturegraph (CMP1 -- a mask IS a texture graph), holographic_fieldhome (a mask may be a raw
field). Plain NumPy; the object stays readable and evaluates directly.
"""
import numpy as np

# reuse CMP1's helpers: _coerce turns a number/color/field/callable into a graph node; _s scalarises a sampled weight
from holographic.materials_and_texture.holographic_texturegraph import _coerce, _s


class MultiMaterial:
    """N materials combined per-point by weight masks. Sample a channel with sample(name, uv); every mask is a CMP1
    node (a constant, a field, or a full texture graph), coerced on construction so the object is uniform."""

    def __init__(self, materials, weights, mode="blend", normalize=True):
        if len(materials) < 1:
            raise ValueError("MultiMaterial needs at least one material")
        if len(materials) != len(weights):
            raise ValueError("one weight mask per material required (%d materials, %d masks)"
                             % (len(materials), len(weights)))
        if mode not in ("blend", "select"):
            raise ValueError("mode must be 'blend' or 'select', got %r" % (mode,))
        for i, mat in enumerate(materials):                     # validate up front, not deep in sample() later
            if not (hasattr(mat, "channels") and hasattr(mat, "sample")):
                raise TypeError("material %d is not a Material (needs .channels and .sample), got %r"
                                % (i, type(mat).__name__))
        self.materials = list(materials)
        self.weights = [_coerce(w) for w in weights]            # each mask -> a CMP1 node with .sample(uv)
        self.mode = mode
        self.normalize = normalize

    def channel_names(self):
        """Every channel any component material has (the union) -- what sample_all will return."""
        names = set()
        for m in self.materials:
            names |= set(m.channels)
        return sorted(names)

    def weights_at(self, uv):
        """The weight of each material at `uv`. In 'blend' mode a partition of unity (normalised, non-negative); in
        'select' mode a one-hot pick of the single highest-weight material. This is the whole mask logic in one place."""
        raw = np.array([_s(w.sample(uv)) for w in self.weights], dtype=float)
        raw = np.maximum(raw, 0.0)                              # a negative mask weight is meaningless -> clamp to 0
        if self.mode == "select":
            one_hot = np.zeros_like(raw)
            one_hot[int(np.argmax(raw))] = 1.0                 # crisp pick: the dominant material wins outright
            return one_hot
        if not self.normalize:
            return raw                                         # caller promises the masks already partition
        total = raw.sum()
        if total <= 1e-9:
            return np.full(len(raw), 1.0 / len(raw))           # all masks ~0 -> uniform, so the point isn't black
        return raw / total                                     # partition of unity: keeps total brightness put

    def sample(self, name, uv):
        """A channel's value at `uv`: the weighted sum over the materials that HAVE that channel. A material missing
        the channel contributes nothing (matching Material.blend's 'present on only one side blends toward zero')."""
        w = self.weights_at(uv)
        total = 0.0
        for wi, m in zip(w, self.materials):
            if wi != 0.0 and name in m.channels:
                total = total + wi * m.sample(name, uv)
        return total

    def sample_all(self, uv):
        """Every channel at `uv` -> {name: value}."""
        return {name: self.sample(name, uv) for name in self.channel_names()}

    def __repr__(self):
        return "MultiMaterial(%d materials, mode=%r%s)" % (
            len(self.materials), self.mode, "" if self.normalize else ", normalize=False")


def _selftest():
    import numpy as np
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    from holographic.materials_and_texture.holographic_material import Material, texture_field

    # NOTE: Material.sample is a COSINE readout (direction, scale-normalised), so two materials must differ in
    # PATTERN, not just a constant. Build two with OPPOSITE albedo ramps so their readouts differ at every point,
    # then check the blend is exactly the weighted sum of those readouts (encoder-agnostic -- we test the formula).
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 7) for v in np.linspace(0.05, 0.95, 7)]
    matA = Material(enc, {"albedo": texture_field(enc, grid, [u for (u, v) in grid])})          # ramps up in u
    matB = Material(enc, {"albedo": texture_field(enc, grid, [1.0 - u for (u, v) in grid])})    # ramps down in u

    def A(uv):
        return matA.sample("albedo", uv)

    def B(uv):
        return matB.sample("albedo", uv)

    # the two materials genuinely differ (opposite ramps) -- otherwise the blend test would be vacuous
    assert abs(A([0.2, 0.5]) - B([0.2, 0.5])) > 0.1

    # BLEND with masks that ramp left->right: matA gets (1-u), matB gets (u). Raw callables -> CMP1 field leaves.
    left = lambda pts: 1.0 - np.asarray(pts, float)[:, 0]
    right = lambda pts: np.asarray(pts, float)[:, 0]
    mm = MultiMaterial([matA, matB], [left, right], mode="blend")

    for uv in ([0.15, 0.5], [0.5, 0.5], [0.85, 0.5]):
        w = mm.weights_at(uv)
        expect = w[0] * A(uv) + w[1] * B(uv)                    # the exact formula this is supposed to compute
        assert abs(mm.sample("albedo", uv) - expect) < 1e-9, uv
    # at the left the (1-u) mask dominates -> reads like matA; at the right the (u) mask dominates -> like matB
    assert abs(mm.sample("albedo", [0.1, 0.5]) - A([0.1, 0.5])) < abs(mm.sample("albedo", [0.1, 0.5]) - B([0.1, 0.5]))
    assert abs(mm.sample("albedo", [0.9, 0.5]) - B([0.9, 0.5])) < abs(mm.sample("albedo", [0.9, 0.5]) - A([0.9, 0.5]))

    # NORMALISE keeps brightness put -- the kept negative, shown both ways against the explicit formula.
    norm = MultiMaterial([matA, matB], [1.0, 1.0], mode="blend", normalize=True)    # 0.5*A + 0.5*B (a proper mix)
    drift = MultiMaterial([matA, matB], [1.0, 1.0], mode="blend", normalize=False)  # 1*A + 1*B (sum -> ~2x, too bright)
    uv = [0.4, 0.6]
    assert abs(norm.sample("albedo", uv) - 0.5 * (A(uv) + B(uv))) < 1e-9
    assert abs(drift.sample("albedo", uv) - (A(uv) + B(uv))) < 1e-9
    assert abs(drift.sample("albedo", uv)) > 1.8 * abs(norm.sample("albedo", uv)) - 1e-9   # unnormalised ~2x brighter

    # SELECT mode: hard pick the dominant material (a material-ID / splat map), no cross-fade.
    picker = MultiMaterial([matA, matB], [0.3, 0.7], mode="select")
    assert abs(picker.sample("albedo", uv) - B(uv)) < 1e-9      # matB (weight 0.7) wins outright

    # all-zero masks -> uniform fallback, so the point isn't black
    zero = MultiMaterial([matA, matB], [0.0, 0.0], mode="blend")
    assert abs(zero.sample("albedo", uv) - 0.5 * (A(uv) + B(uv))) < 1e-9

    print("OK: holographic_multimaterial self-test passed (mask-weighted blend equals w0*A+w1*B exactly at every "
          "point; left mask reads like A, right like B; normalise gives 0.5*(A+B) while unnormalised sums to ~2x "
          "(brightness drift, the kept negative); select picks the dominant material; zero masks fall back to uniform)")


if __name__ == "__main__":
    _selftest()
