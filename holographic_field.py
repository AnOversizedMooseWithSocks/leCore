"""
holographic_field.py
=====================

One primitive, many jobs. Step back from the specific machines we built and a
pattern repeats at every level: a scalar field over the space -- a function that
gives each point a number. The SDF region, the brain's value function, the
memory density / void map, the success compass: every one of them is a field
you sample, compose, and follow uphill. This file is that primitive on its own,
with the operators to build worlds out of it.

The lineage is the demoscene, not task software: a tiny deterministic seed
unfolds into a whole landscape; rich structure is composed from a few operators
(add, blend, smooth-union, warp); the SAME field read one way is geometry and
read another way is a signal in time. Nothing is stored that can be generated.

One honest high-dimensional twist, and it's a feature, not a bug: by
concentration of measure, a field built from scattered landmarks is FLAT almost
everywhere and has structure only near the landmarks. The void is genuinely
empty -- which is precisely why void detection works. The seeds still grow a
world; the world is the neighborhoods of the seeds, not the empty sphere.

Needs: numpy and holographic_ai.py beside it.
"""

import numpy as np
from holographic_ai import random_vector, cosine, slerp


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# THE PRIMITIVE
# ---------------------------------------------------------------------------

class Field:
    """A scalar field over the vector space: any function point -> number, with
    operators to combine fields into new fields. Closed under composition -- the
    same operators apply whether the field is over the whole space, a single
    neighborhood, or (sampled along a path) over time."""

    def __init__(self, fn):
        self.fn = fn

    def sample(self, x):
        return self.fn(_unit(np.asarray(x, dtype=float)))

    # --- combine fields into fields (self-similar: results are Fields too) ---
    def __add__(self, other):
        g = other.fn if isinstance(other, Field) else (lambda x: other)
        return Field(lambda x: self.fn(x) + g(x))

    def __mul__(self, other):
        g = other.fn if isinstance(other, Field) else (lambda x: other)
        return Field(lambda x: self.fn(x) * g(x))

    def __neg__(self):
        return Field(lambda x: -self.fn(x))

    def blend(self, other, t):
        """Linear cross-fade between two fields."""
        return Field(lambda x: (1 - t) * self.fn(x) + t * other.fn(x))

    def smooth_union(self, other, k=0.25):
        """The demoscene smooth-maximum: a union of two fields with no hard
        crease at the seam. k sets how wide the rounding is."""
        def f(x):
            a, b = self.fn(x), other.fn(x)
            h = max(0.0, min(1.0, 0.5 + 0.5 * (a - b) / k))
            return a * h + b * (1 - h) + k * h * (1 - h)
        return Field(f)

    def warp_toward(self, modulator, anchor, amount=1.0):
        """Domain warping: bend the field by displacing each sample point toward
        an anchor by an amount set by another field. (In high-D, warping toward
        structure is what reshapes the field; warping along empty directions does
        nothing.)"""
        anchor = _unit(anchor)
        return Field(lambda x: self.fn(_unit(x + amount * modulator.fn(x) * (anchor - x))))

    def above(self, level=0.0):
        """Turn a field into a region: a new field positive where this one
        exceeds `level`. contains(x) is then just sample(x) > 0."""
        return Field(lambda x: self.fn(x) - level)

    def contains(self, x, level=0.0):
        return self.sample(x) > level


# ---------------------------------------------------------------------------
# WAYS TO MAKE A FIELD
# ---------------------------------------------------------------------------

def bump(center):
    """A peak at `center`, falling off with cosine distance. The atom of every
    landmark-based field."""
    center = _unit(center)
    return Field(lambda x: float(cosine(x, center)))


def ball(center, radius):
    """A signed region: positive inside `radius` (in radians), negative outside.
    The SDF, as a field."""
    center = _unit(center)
    return Field(lambda x: radius - float(np.arccos(np.clip(cosine(x, center), -1, 1))))


def landscape(points, weights):
    """A weighted sum of bumps -- the workhorse. Read as terrain, as a value
    function (weights = rewards), as a density map (weights = 1), or as an
    attention map (weights = relevances). One constructor, all of those."""
    points = [_unit(p) for p in points]
    norm = float(np.sum(np.abs(weights))) or 1.0
    return Field(lambda x: float(sum(w * cosine(x, p) for p, w in zip(points, weights)) / norm))


def seeded_landscape(dim, seed, n=8, spread=1.0):
    """A whole landscape grown from one integer seed: scatter n landmarks, give
    them random weights. Returns (field, landmarks). The demoscene move --
    tiny seed in, structured world out."""
    rng = np.random.default_rng(seed)
    points = [random_vector(dim, rng) for _ in range(n)]
    weights = rng.normal(size=n) * spread
    return landscape(points, weights), points


# ---------------------------------------------------------------------------
# READING A FIELD  (sampling, gradient, climbing -- the same ops for every use)
# ---------------------------------------------------------------------------

def direction(field, x, k=16, sigma=0.3, seed=0):
    """Estimate the uphill direction at x by probing tangent perturbations.
    This single operation is the success compass, the SDF push-inside, and the
    surface normal in raymarching -- all 'which way does the field rise.'"""
    rng = np.random.default_rng(seed)
    x = _unit(np.asarray(x, dtype=float))
    base = field.sample(x)
    grad = np.zeros_like(x)
    for _ in range(k):
        d = rng.normal(size=len(x))
        d = d - (d @ x) * x                  # project onto the tangent plane
        d = _unit(d)
        grad += (field.sample(_unit(x + sigma * d)) - base) * d
    return _unit(grad)


def ascend(field, x, steps=50, sigma=0.5, k=12, seed=0):
    """Climb to a local peak of any field. The brain finding its best state, the
    compass heading to success, a ray reaching a surface -- the same loop."""
    rng = np.random.default_rng(seed)
    x = _unit(np.asarray(x, dtype=float))
    for _ in range(steps):
        best_x, best_v = x, field.sample(x)
        for _ in range(k):
            cand = _unit(x + sigma * rng.normal(size=len(x)))
            v = field.sample(cand)
            if v > best_v:
                best_x, best_v = cand, v
        x = best_x
        sigma *= 0.96
    return x


# ---------------------------------------------------------------------------
# DEMOS
# ---------------------------------------------------------------------------

def demo_one_field_many_roles():
    print("=" * 70)
    print("DEMO 1 -- One primitive wearing every hat")
    print("=" * 70)
    dim = 256
    field, marks = seeded_landscape(dim, seed=1, n=6)
    print("\nA landscape grown from a single seed (6 landmarks). The SAME object")
    print("is read as terrain, a value function, a density map, or an attention")
    print("map. Threshold it and it's a region:\n")
    region = field.above(0.10)
    near = marks[int(np.argmax([field.sample(m) for m in marks]))]
    far = -near
    print(f"  field at its highest landmark : {field.sample(near):+.2f}  in region: {region.contains(near)}")
    print(f"  field at the opposite point   : {field.sample(far):+.2f}  in region: {region.contains(far)}")
    print("\n  SDF region, brain value, memory density, success compass -- in this")
    print("  system they are not analogies. They are this one type, composed.\n")


def demo_smooth_composition():
    print("=" * 70)
    print("DEMO 2 -- Composition: the demoscene smooth-union")
    print("=" * 70)
    dim = 256
    rng = np.random.default_rng(2)
    a, b = random_vector(dim, rng), random_vector(dim, rng)
    peak_a, peak_b = bump(a), bump(b)
    hard = Field(lambda x: max(peak_a.fn(x), peak_b.fn(x)))   # plain union: a crease
    soft = peak_a.smooth_union(peak_b, k=0.30)                # rounded seam

    path = [slerp(a, b, t) for t in np.linspace(0, 1, 41)]
    hv = [hard.sample(p) for p in path]
    sv = [soft.sample(p) for p in path]
    kink_hard = float(np.max(np.abs(np.diff(hv, 2))))
    kink_soft = float(np.max(np.abs(np.diff(sv, 2))))
    print("\nMerging two peaks. Plain union leaves a sharp crease at the seam;")
    print("smooth-union rounds it -- the move every raymarched demo leans on.\n")
    print(f"  seam sharpness (max curvature)  hard: {kink_hard:.3f}   smooth: {kink_soft:.3f}")
    print(f"  -> smooth-union is {kink_hard / max(kink_soft, 1e-6):.0f}x less creased\n")


def demo_universal_gradient():
    print("=" * 70)
    print("DEMO 3 -- One gradient, every purpose")
    print("=" * 70)
    dim = 256
    rng = np.random.default_rng(3)
    good, bad = random_vector(dim, rng), random_vector(dim, rng)
    value = landscape([good, bad], [1.0, -1.0])     # reward landscape
    start = random_vector(dim, rng)
    end = ascend(value, start, seed=3)
    print("\nClimbing a value landscape (good = +1, bad = -1) from a cold start:\n")
    print(f"  field value:        {value.sample(start):+.2f} -> {value.sample(end):+.2f}")
    print(f"  similarity to good: {cosine(start, good):+.2f} -> {cosine(end, good):+.2f}")
    print(f"  similarity to bad:  {cosine(start, bad):+.2f} -> {cosine(end, bad):+.2f}")
    print("\n  This climb is the brain seeking its best state, the compass heading")
    print("  to success, and a ray reaching a surface -- the same ascend().\n")


def demo_space_becomes_time():
    print("=" * 70)
    print("DEMO 4 -- A path through a field is a signal in time")
    print("=" * 70)
    dim = 256
    field, marks = seeded_landscape(dim, seed=5, n=6)
    tour = [marks[0], marks[1], marks[2], marks[0]]    # a closed loop
    signal = []
    for i in range(3):
        for t in np.linspace(0, 1, 24, endpoint=False):
            signal.append(field.sample(slerp(tour[i], tour[i + 1], t)))
    print("\nWalk a closed loop through the landscape and the field readings become")
    print("a periodic signal -- an oscillator. The same field is geometry when you")
    print("place it and rhythm when you travel it.\n")
    print(f"  signal length: {len(signal)} steps   range: ({min(signal):+.2f}, {max(signal):+.2f})")
    print(f"  loops cleanly: start {signal[0]:+.2f} ~ wrap {signal[-1]:+.2f}")
    print("\n  Use it as an LFO to modulate anything -- exploration rate, a blend")
    print("  weight, a gate -- driven by the very landscape it lives in.\n")


def demo_concentration_note():
    print("=" * 70)
    print("DEMO 5 -- The void is real (an honest high-D note)")
    print("=" * 70)
    dim = 256
    field, marks = seeded_landscape(dim, seed=8, n=6)
    rng = np.random.default_rng(99)
    at_marks = [abs(field.sample(m)) for m in marks]
    in_void = [abs(field.sample(random_vector(dim, rng))) for _ in range(200)]
    print("\nField strength near a landmark vs. at random empty points:\n")
    print(f"  typical |field| at a landmark : {np.mean(at_marks):.3f}")
    print(f"  typical |field| in empty space: {np.mean(in_void):.3f}")
    print("\n  In high dimensions a field is flat almost everywhere and structured")
    print("  only near its seeds. That is why global procedural noise doesn't")
    print("  translate -- but it is also exactly why the void is detectable. The")
    print("  seed grows a world; the world is the neighborhoods of the seeds.\n")


if __name__ == "__main__":
    demo_one_field_many_roles()
    demo_smooth_composition()
    demo_universal_gradient()
    demo_space_becomes_time()
    demo_concentration_note()
