"""spectral_heat -- solve the heat equation to ANY time in one step, exactly.

THE POINT, and it is a property rather than a benchmark: for a linear constant-coefficient PDE on a
periodic grid, every Fourier mode evolves independently as exp(rate(k) * t). So the solution at t=20 is
not twenty thousand small steps -- it is one multiply. There is no CFL limit to respect, no accumulated
per-step error, and the cost does not depend on the horizon at all.

MEASURED, and this is the number the application asserts: against the closed form u(x,t) = e^(-nu t)
sin(x), max|error| is 4.4e-16 at t=0.5 and 1.7e-16 at t=20 -- machine precision, and it does NOT grow
with the horizon. The explicit finite-difference stepper this compares against is the honest baseline:
the SAME problem, the strongest ordinary method, run at a stable step size. MEASURED here it reaches
5.0e-04 -- small in absolute terms, and that is the point of quoting it rather than a strawman: the
difference worth showing is not that the baseline is bad, it is that the baseline needs 231 steps to
reach t=20 on this grid while the spectral form needs ONE, and its error grows with the horizon while
ours does not. (231, measured -- an earlier draft of this docstring guessed 66,000 and was wrong by two
orders of magnitude. The step count scales as N^2 with grid refinement, so quote the grid or say nothing.)

THE TRAP THIS APPLICATION EXISTS TO DOCUMENT (it cost the author three wrong probes): `rate` is a
CALLABLE of |k|, not a number, and `dx` must be the REAL grid spacing. Passing rate=0.1 raises
"'float' object is not callable"; leaving dx at its 1.0 default while sampling 2*pi over 64 points
silently scales every wavenumber and the error grows to 6.2e-01 -- a wrong answer that looks like a
solver limitation rather than a units mistake. Both are asserted below so the guidance cannot rot.
"""
import math

import numpy as np

NAME = "spectral_heat"
DOMAIN = "math"
PROVES = ("a linear PDE advanced to any horizon in ONE step, exact to machine precision (4.4e-16), "
          "against an explicit finite-difference baseline whose error grows with the horizon")
ARTEFACT = None


def run(mind, n=64, nu=0.05, horizons=(0.5, 2.0, 5.0, 20.0), mode=1):
    """Advance u_t = nu u_xx spectrally and compare with both the closed form and an FD baseline.

    Returns {proved: {max_error, worst_horizon, fd_max_error, ratio}, rows: [...]} -- `rows` carries one
    record per horizon so a reader can see that the spectral error is FLAT in t while the baseline's is
    not, which is the claim. Everything goes through mind.spectral_pde; nothing imports the solver."""
    x = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    dx = 2.0 * math.pi / n                       # the REAL spacing: see the module docstring's trap
    u0 = np.sin(mode * x)
    rows = []
    for t in horizons:
        field = mind.spectral_pde(u0, order="parabolic", rate=lambda k: -nu * k ** 2, dx=dx)
        field.advance(t)
        got = np.asarray(field.field, dtype=float)
        exact = math.exp(-nu * mode ** 2 * t) * np.sin(mode * x)
        fd, fd_steps = _fd_reference(u0, nu, dx, t)
        rows.append({"t": t, "spectral_error": float(np.max(np.abs(got - exact))),
                     "fd_error": float(np.max(np.abs(fd - exact))),
                     "spectral_steps": 1, "fd_steps": fd_steps})
    worst = max(rows, key=lambda r: r["spectral_error"])
    fd_worst = max(r["fd_error"] for r in rows)
    return {"rows": rows,
            "proved": {"max_error": worst["spectral_error"], "worst_horizon": worst["t"],
                       "fd_max_error": fd_worst, "fd_steps_at_longest": rows[-1]["fd_steps"],
                       "spectral_steps_at_longest": 1,
                       "ratio": (fd_worst / worst["spectral_error"]) if worst["spectral_error"] else None}}


def _fd_reference(u0, nu, dx, t):
    """THE HONEST BASELINE: explicit central-difference heat stepping at 90% of the stability limit.

    Deliberately the strongest ordinary method for this problem rather than a strawman -- same grid, same
    initial condition, a stable step size. Its error is not a bug in it; accumulating per-step truncation
    is what marching does, and showing that is the point. Local to this application on purpose: it is a
    reference implementation, not a capability, and it must never become something the engine offers."""
    dt = 0.9 * dx ** 2 / (2.0 * nu)
    steps = max(1, int(math.ceil(t / dt)))
    dt = t / steps
    u = np.array(u0, dtype=float)
    for _ in range(steps):
        u = u + nu * dt * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx ** 2
    return u, steps


def _selftest():
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)
    r = run(mind)
    p = r["proved"]
    # 1. EXACT, at machine precision -- the property, not a tolerance chosen to pass.
    assert p["max_error"] < 1e-13, p
    # 2. AND FLAT IN t: the error at the longest horizon must not exceed the shortest by any real factor.
    #    A stepping solver cannot do this, and it is the whole reason to reach for the spectral form.
    first, last = r["rows"][0]["spectral_error"], r["rows"][-1]["spectral_error"]
    assert last <= max(first, 1e-15) * 10, (first, last)
    # 3. The baseline must be genuinely worse, or the comparison is theatre.
    assert p["fd_max_error"] > 1e-4 and p["ratio"] > 1e8, p
    # 3b. THE COST CLAIM, which matters more than the error ratio: the horizon costs the baseline steps
    #     and costs the spectral form nothing.
    assert p["fd_steps_at_longest"] > 200 and p["spectral_steps_at_longest"] == 1, p
    # 4. THE UNITS TRAP, pinned: the default dx=1.0 on a 2*pi/64 grid is a WRONG ANSWER, not a warning.
    x = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    bad = mind.spectral_pde(np.sin(x), order="parabolic", rate=lambda k: -0.05 * k ** 2)
    bad.advance(20.0)
    err = float(np.max(np.abs(np.asarray(bad.field) - math.exp(-0.05 * 20.0) * np.sin(x))))
    assert err > 0.1, "the wrong-dx path must stay visibly wrong, or the docstring's warning is stale"
    print("spectral_heat OK: max|err| %.2e (flat in t); FD baseline %.2e needing %d steps to reach "
          "t=%.0f where the spectral form needs 1"
          % (p["max_error"], p["fd_max_error"], p["fd_steps_at_longest"], r["rows"][-1]["t"]))


if __name__ == "__main__":
    _selftest()
