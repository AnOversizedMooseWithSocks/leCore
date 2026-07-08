"""
holographic_extras.py
=====================

Three more leOS concepts, each boiled down to pure numpy/VSA (no models, no
Ollama, no dependencies beyond the engine):

  * ResidueSystem    -- exact integer arithmetic carried out as vector binding,
                        via the Chinese Remainder Theorem. Add and subtract
                        numbers by binding their vectors; decode with the
                        resonator. No ALU, no if-statements -- just convolution.
  * Region           -- named regions of vector space with a signed distance and
                        Boolean composition (union / intersection / subtraction).
                        A richer, composable version of the engine's partition
                        routing: build arbitrary semantic filters from simple
                        balls, and get a free "which way is the boundary" vector.
  * PredictiveFilter -- "only record what surprises you." Predicts the next
                        observation, passes only the prediction errors, and
                        re-calibrates after a real change. A stable stream goes
                        silent; a genuine shift fires once. (leOS's bots.)

Needs: numpy, holographic_ai.py, and holographic_reasoning.py beside it.
"""

import numpy as np
from math import prod
from holographic.agents_and_reasoning.holographic_ai import random_vector, cosine, bind, involution, geodesic, log_map, exp_map, slerp
from holographic.agents_and_reasoning.holographic_reasoning import ResonatorNetwork


# ---------------------------------------------------------------------------
# 1. RESIDUE ARITHMETIC  (exact integer math as vector binding, via CRT)
#
#    The Chinese Remainder Theorem says an integer below M = m1*m2*...*mk is
#    pinned down uniquely by its remainders against a set of coprime moduli.
#    We give each modulus a special "generator" vector whose powers cycle every
#    m steps -- raise it to the r-th power to stand for remainder r, and because
#    it cycles, binding two of them adds the remainders MODULO m automatically.
#    Bind one generator-power per modulus together and you have a single vector
#    that is the integer. Then:
#        add        = bind            (remainders add, mod each m -> mod M)
#        subtract   = bind with inverse
#        scale by c = bind a number with itself c times
#    Decoding reads each remainder back with the resonator (the number is just a
#    product of generator-powers) and reassembles the integer by CRT. The whole
#    arithmetic unit is circular convolution.
# ---------------------------------------------------------------------------

class ResidueSystem:
    """Exact integer arithmetic on integers in [0, M) done with vector ops.

    moduli must be pairwise coprime; M is their product and sets the range.
    Defaults (7, 11, 13) give M = 1001, i.e. 0..1000.
    """

    def __init__(self, moduli=(7, 11, 13), dim=2048, seed=0):
        self.moduli = list(moduli)
        self.dim = dim
        self.M = prod(self.moduli)
        # One generator per modulus, plus the codebook of its powers 0..m-1.
        self.phases = [self._phases(m, seed + i + 1) for i, m in enumerate(self.moduli)]
        self.codebooks = [np.array([self._power(self.phases[j], r) for r in range(m)])
                          for j, m in enumerate(self.moduli)]
        self._resonator = ResonatorNetwork(self.codebooks)

    def _phases(self, m, seed):
        # Phases are multiples of 2*pi/m, so the generator's m-th power is the
        # identity (the cycle wraps). Kept conjugate-symmetric for real vectors.
        rng = np.random.default_rng(seed)
        k = rng.integers(0, m, size=self.dim)
        k[0] = 0
        for d in range(1, self.dim // 2 + 1):
            k[self.dim - d] = (m - k[d]) % m
        if self.dim % 2 == 0:
            k[self.dim // 2] = 0
        return 2 * np.pi * k / m

    def _power(self, phases, r):
        # generator^r as a real unit vector (rotating the phases by r).
        return np.real(np.fft.ifft(np.exp(1j * r * phases)))

    def encode(self, n):
        """Integer -> one vector (bind a generator-power per modulus)."""
        n %= self.M
        v = self.codebooks[0][n % self.moduli[0]]
        for j in range(1, len(self.moduli)):
            v = bind(v, self.codebooks[j][n % self.moduli[j]])
        return v

    def decode(self, vec):
        """Vector -> integer (resonator recovers the remainders, CRT rebuilds n)."""
        residues = self._resonator.factor(vec)
        return self._crt(residues)

    def _crt(self, residues):
        n = 0
        for m, r in zip(self.moduli, residues):
            Mj = self.M // m
            n += r * Mj * pow(Mj, -1, m)     # modular inverse of Mj mod m
        return n % self.M

    # --- arithmetic, all carried out on the vectors themselves ---
    def add(self, va, vb):
        return bind(va, vb)

    def subtract(self, va, vb):
        return bind(va, involution(vb))

    def scale(self, v, c):
        """Multiply the encoded integer by a non-negative integer constant c."""
        out = self.encode(0)                 # identity element
        for _ in range(c):
            out = bind(out, v)
        return out


# ---------------------------------------------------------------------------
# 2. SDF REGIONS  (named regions of space with Boolean algebra)
#
#    The engine routes by nearest anchor. A region is the richer idea: a named
#    patch of the sphere with a soft edge, reporting a SIGNED distance -- negative
#    inside, positive outside, zero on the boundary. The payoff is composition:
#    borrowed from computer-graphics signed-distance fields, complex regions are
#    built from simple balls with three trivial rules,
#        union(A, B)        = min(sdf_A, sdf_B)
#        intersection(A, B) = max(sdf_A, sdf_B)
#        A minus B          = max(sdf_A, -sdf_B)
#    so "in the animals region but not the pets region" is one expression. The
#    signed distance also gives a free gradient: the way toward a region's centre
#    is the direction that pulls a point inside.
# ---------------------------------------------------------------------------

class Region:
    """A region of the unit sphere defined by a signed-distance function.

    Build base regions with ball(); combine them with .union/.intersect/.subtract.
    sdf(q) < 0 means q is inside. centers carries the defining centres so steer()
    can point a query toward the region.
    """

    def __init__(self, sdf_fn, centers):
        self._sdf = sdf_fn
        self.centers = centers

    def sdf(self, q):
        return self._sdf(q)

    def contains(self, q):
        return self.sdf(q) < 0

    def union(self, other):
        return Region(lambda q: min(self.sdf(q), other.sdf(q)),
                      self.centers + other.centers)

    def intersect(self, other):
        return Region(lambda q: max(self.sdf(q), other.sdf(q)),
                      self.centers + other.centers)

    def subtract(self, other):
        return Region(lambda q: max(self.sdf(q), -other.sdf(q)), self.centers)

    def complement(self):
        return Region(lambda q: -self.sdf(q), self.centers)

    def _nearest_center(self, q):
        return min(self.centers, key=lambda c: geodesic(q, c))

    def steer(self, q, step=0.5):
        """Move q along the geodesic toward the region (toward the nearest
        defining centre) -- the SDF gradient used as a pull."""
        target = self._nearest_center(q)
        return exp_map(q, step * log_map(q, target))


def ball(center, radius):
    """A spherical region: everything within geodesic distance `radius` (in
    radians) of `center`."""
    return Region(lambda q: geodesic(q, center) - radius, [center])


def route(regions, q):
    """Index of the region q sits deepest inside (most negative signed distance).
    Falls back to the nearest region even if q is outside all of them."""
    return int(np.argmin([r.sdf(q) for r in regions]))


# ---------------------------------------------------------------------------
# 3. PREDICTIVE-CODING NOVELTY FILTER  ("only record what surprises you")
#
#    leOS's bots don't log everything they see -- they predict the next
#    observation and record only the prediction error. A monitor on a quiet
#    source produces almost nothing; a monitor on a changing source produces
#    exactly the moments that changed. This is the cheap front-line filter that
#    keeps a system from drowning in its own routine observations, and it pairs
#    naturally with the drift detector (which judges geometry) by judging time:
#    is this reading surprising given what came just before?
#
#    The prediction is a smoothed running expectation; surprise is distance from
#    it. A reading counts as novel only if its surprise stands out against the
#    recent surprise level (a moving baseline), so the filter auto-tunes to
#    whatever noise the source happens to have, and re-settles after a real shift.
# ---------------------------------------------------------------------------

class PredictiveFilter:
    """Pass only surprising observations; stay quiet on predictable ones.

    observe(vec) returns (is_novel, surprise). Slow drift is absorbed by the
    moving prediction; an abrupt change fires once and then the baseline
    re-calibrates to the new regime.
    """

    def __init__(self, momentum=0.5, base_decay=0.9, k=4.0, warmup=6):
        self.pred = None
        self.momentum = momentum        # how smoothly the prediction tracks
        self.base_decay = base_decay     # how slowly the surprise baseline moves
        self.k = k                        # how many baseline std-devs counts as novel
        self.warmup = warmup
        self.t = 0
        self.base_mean = 0.0
        self.base_var = 0.0

    def observe(self, vec):
        vec = np.asarray(vec, dtype=float)
        if self.pred is None:             # first reading: nothing to compare to
            self.pred = vec.copy()
            return True, 1.0

        surprise = 1.0 - cosine(vec, self.pred)
        self.t += 1
        threshold = self.base_mean + self.k * (self.base_var ** 0.5)
        novel = self.t > self.warmup and surprise > threshold

        # Move the surprise baseline (always), so it re-settles after a shift.
        self.base_var = self.base_decay * self.base_var \
            + (1 - self.base_decay) * (surprise - self.base_mean) ** 2
        self.base_mean = self.base_decay * self.base_mean \
            + (1 - self.base_decay) * surprise

        # Move the prediction toward what we just saw.
        self.pred = self.momentum * self.pred + (1 - self.momentum) * vec
        n = np.linalg.norm(self.pred)
        if n > 0:
            self.pred = self.pred / n
        return novel, surprise


# ---------------------------------------------------------------------------
# 4. DEMOS
# ---------------------------------------------------------------------------

def demo_residue():
    print("=" * 70)
    print("DEMO P -- Residue arithmetic: exact integer math as vector binding")
    print("=" * 70)
    rs = ResidueSystem(moduli=(7, 11, 13), dim=2048, seed=0)
    print(f"\nModuli {tuple(rs.moduli)} -> exact integers 0..{rs.M - 1}, each one a vector.\n")

    print("Round-trip encode/decode:")
    for n in [0, 1, 358, 1000]:
        print(f"  {n:4d} -> vector -> {rs.decode(rs.encode(n))}")

    print("\nArithmetic done ENTIRELY on the vectors, then decoded:")
    a, b = 247, 358
    print(f"  {a} + {b}  = {rs.decode(rs.add(rs.encode(a), rs.encode(b)))}   (expect {a + b})")
    print(f"  {b} - {a}  = {rs.decode(rs.subtract(rs.encode(b), rs.encode(a)))}   (expect {b - a})")
    print(f"  {a} * 3   = {rs.decode(rs.scale(rs.encode(a), 3))}   (expect {a * 3})")
    print("\n  No arithmetic logic unit, no carry handling -- the answers fall out")
    print("  of circular convolution because the generators cycle modulo m.\n")


def demo_region():
    print("=" * 70)
    print("DEMO Q -- SDF regions: Boolean algebra on patches of the space")
    print("=" * 70)
    rng = np.random.default_rng(0)
    dim = 1024
    a = random_vector(dim, rng)
    far = random_vector(dim, rng)
    b = slerp(a, far, 0.30)                 # a neighbouring centre, overlapping
    region_a, region_b = ball(a, 0.40), ball(b, 0.40)

    points = {"in A only": a, "in both": slerp(a, b, 0.5),
              "in B only": b, "outside both": far}
    print(f"\nTwo overlapping regions (centres {geodesic(a, b):.2f} rad apart, "
          f"radius 0.40).\n")
    print("  point          A      B    A|B    A&B    A-B")
    for name, q in points.items():
        row = (region_a.contains(q), region_b.contains(q),
               region_a.union(region_b).contains(q),
               region_a.intersect(region_b).contains(q),
               region_a.subtract(region_b).contains(q))
        print("  {:13s} {}".format(name, "  ".join(f"{str(x):5s}" for x in row)))

    q = slerp(a, far, 0.55)                 # just outside A
    print("\nSteering a point toward region A pulls it inside:")
    print(f"  signed distance {region_a.sdf(q):+.2f} -> {region_a.sdf(region_a.steer(q, step=0.7)):+.2f}"
          f"   (negative = inside)\n")


def demo_predictive():
    print("=" * 70)
    print("DEMO R -- Predictive filter: record only what surprises you")
    print("=" * 70)
    rng = np.random.default_rng(0)
    dim = 1024
    state_p = random_vector(dim, rng)
    state_q = random_vector(dim, rng)

    pf = PredictiveFilter()
    stable_flags, switch_step, settled_flags = 0, None, 0
    print("\nA stream sits near one state for 30 steps, then jumps to another:\n")
    for i in range(60):
        base = state_p if i < 30 else state_q
        novel, _ = pf.observe(base + 0.25 * random_vector(dim, rng))
        if 0 < i < 30 and novel:
            stable_flags += 1
        if i >= 30 and novel:
            if switch_step is None:
                switch_step = i
            else:
                settled_flags += 1
    print(f"  flags during the stable first phase : {stable_flags} / 29")
    print(f"  first flag after the jump (step 30) : step {switch_step}")
    print(f"  flags once settled into the new state: {settled_flags}")
    print("\n  Routine readings are predicted away and produce nothing; the one")
    print("  moment that mattered -- the jump -- is exactly what gets recorded.\n")


if __name__ == "__main__":
    demo_residue()
    demo_region()
    demo_predictive()
