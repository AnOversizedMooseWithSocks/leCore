"""holographic_samplinghome.py -- the SAMPLING home (consolidation backlog R4): one place for the Monte-Carlo
sampling machinery a renderer reaches for.

WHY THIS EXISTS
---------------
The sampling pieces are shipped but scattered: quasi-random / low-discrepancy points (holographic_lowdiscrepancy),
blue-noise / Poisson-disk points (holographic_sampling), MIS weighting (holographic_mis), robust accumulation with
a firefly clamp (holographic_accumulate) -- and the COSINE-HEMISPHERE direction sampler was re-implemented in three
modules (brdf, globalillum, lights). A caller that wants "give me n good sample directions / offsets" shouldn't have
to know which module each lives in, and the hemisphere sampler shouldn't exist three times.

`Sampling` is that one home. It is THIN -- it ROUTES to the shipped modules for the pattern generators, MIS, and
accumulation (route, don't rewrite). The one thing it OWNS is the vectorised cosine-hemisphere sampler, promoted
here so the gather paths (globalillum, lightcache, ...) share a single implementation instead of copies.

    Sampling.low_discrepancy(n, d, seed)      # quasi-random points (AA offsets, stratification)
    Sampling.poisson_disk(radius, bounds)     # blue-noise points
    Sampling.cosine_hemisphere(N, n, seed)    # n cosine-weighted dirs per normal -> (M,n,3)   [OWNED]
    Sampling.mis_weight(pairs, power)         # multiple-importance-sampling combine
    Sampling.accumulate(samples, ...)         # firefly-clamped robust mean of sample passes
"""
import numpy as np


class Sampling:
    """A namespace of staticmethods over the engine's sampling machinery. Patterns, directions, MIS, accumulation."""

    @staticmethod
    def low_discrepancy(n, d=2, seed=0):
        """`n` low-discrepancy (quasi-random) points in [0,1)^d -- even coverage with less clumping than white
        noise. Used for anti-aliasing offsets and stratified sampling. Routes to holographic_lowdiscrepancy."""
        from holographic_lowdiscrepancy import low_discrepancy
        return low_discrepancy(n, d=d, seed=seed)

    @staticmethod
    def poisson_disk(radius, bounds, k=30, seed=0):
        """Blue-noise (Poisson-disk) points at least `radius` apart within `bounds`. Routes to holographic_sampling.
        """
        from holographic_sampling import poisson_disk_sample
        return poisson_disk_sample(radius, bounds, k=k, seed=seed)

    @staticmethod
    def cosine_hemisphere(N, n, seed=0):
        """`n` cosine-weighted sample directions around each unit normal in N:(M,3). Returns (M, n, 3). Vectorised.

        OWNED here (consolidation R4): this was copied into brdf/globalillum/lights. The cosine weighting means the
        directions are already importance-sampled for a diffuse (Lambertian) surface, so a plain average of what
        they hit IS the irradiance estimate -- which is why the gather paths use it."""
        rng = np.random.default_rng(seed)
        M = len(N)
        u1 = rng.random((M, n)); u2 = rng.random((M, n))
        r = np.sqrt(u1); th = 2 * np.pi * u2
        x = r * np.cos(th); y = r * np.sin(th); z = np.sqrt(np.clip(1 - u1, 0, 1))   # local cosine-weighted dir
        up = np.where(np.abs(N[:, 1:2]) < 0.99, np.array([0., 1, 0]), np.array([1., 0, 0]))  # a stable 'up' per normal
        T = np.cross(up, N); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
        B = np.cross(N, T)
        return (x[..., None] * T[:, None, :] + y[..., None] * B[:, None, :] + z[..., None] * N[:, None, :])

    @staticmethod
    def mis_weight(pairs, power=1.0):
        """Combine estimators from different sampling strategies with the balance (power=1) / power heuristic --
        Veach's multiple importance sampling. Routes to holographic_mis."""
        from holographic_mis import combine_estimators
        return combine_estimators(pairs, power=power)

    @staticmethod
    def accumulate(samples, schedule="harmonic", clamp_k=None, alpha=0.2):
        """Robustly average a list of sample passes, winsorising a whole outlier pass (a firefly frame) so one bad
        pass can't blow out a pixel. Routes to holographic_accumulate.robust_accumulate."""
        from holographic_accumulate import robust_accumulate
        return robust_accumulate(samples, schedule=schedule, alpha=alpha, clamp_k=clamp_k)


def sampling_backends():
    """The sampling facilities the home exposes (for the catalog / discovery)."""
    return ("low_discrepancy", "poisson_disk", "cosine_hemisphere", "mis_weight", "accumulate")


def _selftest():
    # low-discrepancy: routes and matches the module directly
    from holographic_lowdiscrepancy import low_discrepancy as _ld
    assert np.array_equal(Sampling.low_discrepancy(16, d=2, seed=3), _ld(16, d=2, seed=3))

    # cosine hemisphere: right shape, unit-length dirs, in the upper hemisphere of each normal
    N = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    dirs = Sampling.cosine_hemisphere(N, 64, seed=0)
    assert dirs.shape == (3, 64, 3)
    lens = np.linalg.norm(dirs, axis=2)
    assert np.allclose(lens, 1.0, atol=1e-6)
    ndotl = np.einsum("mnk,mk->mn", dirs, N)
    assert (ndotl >= -1e-6).all()                                # never below the surface

    # deterministic
    assert np.array_equal(dirs, Sampling.cosine_hemisphere(N, 64, seed=0))

    # accumulate clamps a firefly pass
    clean = np.ones((4, 4, 3)) * 0.5
    passes = [clean, clean, clean + 5.0, clean]                  # one blown pass
    acc = Sampling.accumulate(passes, schedule="mean", clamp_k=2.5)
    assert abs(float(acc.mean()) - 0.5) < 0.3                    # not dragged to ~1.6 (the naive mean)
    print("OK: holographic_samplinghome self-test passed (low-discrepancy routes exactly; cosine-hemisphere unit & "
          "upper-hemisphere & deterministic; accumulate clamps a firefly; over %s)" % ", ".join(sampling_backends()))


if __name__ == "__main__":
    _selftest()
