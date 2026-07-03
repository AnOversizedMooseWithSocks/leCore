"""holographic_simreadout.py -- ITERATE SIM READOUT (fluids/matter backlog, performance item PW4).

The final performance move: `iterate` is PRT-for-TIME -- diagonalise a linear operator ONCE and evaluate any step k,
or the limit, in closed form instead of marching. The matter model's per-channel step is advect (nonlinear) ->
diffuse (LINEAR) -> tension (nonlinear) -> drift (nonlinear). The diffusion sub-step is a bind: `fields.diffuse`
multiplies by exp(-amount*k^2) in Fourier space, DIAGONAL in the Fourier basis (the engine's own operator). So the
diffusion of a field after ANY number of steps is a single evaluation -- raise each frequency's transfer to that
power -- and its limit (k -> infinity) is closed form: every non-DC mode decays to zero, the DC (mean) is preserved,
so the steady state is the flat mean field. This is exactly iterate.step_k / iterate.limit, here in 2-D for the sim.

What this buys: a "diffuse this channel for t seconds" readout costs ONE transform pair, not t marched steps; and the
eventual equilibrium is known without simulating to it. Where a stage of the sim is linear and time-invariant, time
becomes a QUERY (the deterministic-depth lever: compute depth instead of storing/marching it).

KEPT NEGATIVE (loud, and it is the whole honesty of this item): ONLY the linear, time-invariant sub-step diagonalises.
Advection is nonlinear (it moves the field along a velocity that itself evolves), buoyancy couples channels, and the
double-well tension is nonlinear -- none of those can be read out at an arbitrary t this way; they still march, and
the adaptive dispatch localises the expensive marching to where it is needed. The closed-form readout is the honest
prize for the part that is genuinely linear -- the same boundary the dynamics propagator drew (linear-in-Fourier
exact; nonlinear needs the reservoir lift). Semigroup check: k diffusions by `amount` == one diffusion by k*amount,
which the operator power reproduces to floating-point.
"""
import numpy as np

from holographic_fields import diffuse


def diffusion_transfer(shape, amount):
    """The Fourier transfer of one diffusion step on a `shape` torus: exp(-amount*k^2), the operator's eigenvalues
    (the eigenvectors are the Fourier modes -- the decomposition is FREE, it is the rfft). DC (k=0) transfer is 1,
    so the mean is preserved and diffusion conserves mass."""
    H, W = shape
    ky = 2.0 * np.pi * np.fft.fftfreq(H)[:, None]
    kx = 2.0 * np.pi * np.fft.rfftfreq(W)[None, :]
    return np.exp(-amount * (kx ** 2 + ky ** 2))              # (H, W//2+1) diagonal transfer


def diffuse_at(field, amount, k, transfer=None):
    """The field after `k` diffusion steps of size `amount`, in ONE evaluation -- raise the transfer to the k-th
    power (k may be fractional: 'diffuse for 2.5 steps'). This is iterate.step_k for the sim's linear sub-step."""
    field = np.asarray(field, float)
    if transfer is None:
        transfer = diffusion_transfer(field.shape, amount)
    return np.fft.irfft2(np.fft.rfft2(field) * (transfer ** k), s=field.shape)


def diffuse_limit(field):
    """The steady state of unbounded diffusion (k -> infinity): every non-DC mode decays to zero, the DC (mean) is
    preserved -- so the limit is the flat mean field. Closed form, no marching."""
    field = np.asarray(field, float)
    return np.full_like(field, float(field.mean()))


def _selftest():
    """The direct readout matches marching the real fields.diffuse step exactly (semigroup), fractional steps
    interpolate, and the closed-form limit is the mean -- while confirming the nonlinear sub-steps are NOT claimed."""
    rng = np.random.default_rng(0)
    field = rng.standard_normal((32, 32))
    amount = 0.15

    # readout at k steps == marching fields.diffuse k times, to floating point (the semigroup, done in one eval)
    for k in (1, 3, 7, 20):
        marched = field.copy()
        for _ in range(k):
            marched = diffuse(marched, amount)
        direct = diffuse_at(field, amount, k)
        assert np.allclose(direct, marched, atol=1e-9), (k, np.abs(direct - marched).max())

    # a fractional step is meaningful and lies between neighbours (diffuse for 2.5 steps)
    half = diffuse_at(field, amount, 2.5)
    lo = diffuse_at(field, amount, 2.0); hi = diffuse_at(field, amount, 3.0)
    assert np.abs(half).max() <= np.abs(lo).max() + 1e-9      # more diffusion -> smaller extremes (monotone smoothing)
    assert np.abs(half).max() >= np.abs(hi).max() - 1e-9

    # the closed-form limit is the flat mean (steady state), and marching a long time approaches it
    lim = diffuse_limit(field)
    assert np.allclose(lim, field.mean())
    long_marched = diffuse_at(field, amount, 5000)
    assert np.abs(long_marched - lim).max() < 1e-6           # long diffusion -> the mean

    # mass (mean) is conserved at every step -- the DC transfer is 1
    for k in (1, 10, 100):
        assert abs(diffuse_at(field, amount, k).mean() - field.mean()) < 1e-9

    print("holographic_simreadout selftest OK: the matter model's LINEAR diffusion sub-step is diagonalised (iterate "
          "for the sim) -- the field after k steps is ONE transform pair matching k marched fields.diffuse calls to "
          "1e-9 (the heat semigroup); fractional steps interpolate; the closed-form limit is the flat mean (steady "
          "state), which long marching approaches to 1e-6; mass is conserved. KEPT NEGATIVE: only this linear, "
          "time-invariant sub-step reads out at arbitrary t -- nonlinear advection / buoyancy coupling / double-well "
          "tension still march.")


if __name__ == "__main__":
    _selftest()
