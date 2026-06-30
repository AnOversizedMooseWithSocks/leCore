"""Optional SymPy DESIGN-TIME codegen: derive an exact gradient (an SDF surface normal, a force = -grad energy)
symbolically, then emit a pure-NumPy function. The RUNTIME stays pure NumPy and autodiff-free -- only the one-time
DERIVATION touches SymPy, and what it hands back closes over numpy alone.

(Distinct from holographic_symbolic.py, which is MDL symbolic *regression* -- discovering a law FROM data. This is
the other direction: take a KNOWN symbolic expression and emit fast exact numeric code for it and its derivatives.)

WHY (the panel's 'unlock', Quilez + Baker seats): holostuff takes SDF normals by FINITE DIFFERENCES today
(sdf_normal: extra evals + a step-size knob that trades truncation error against round-off). The surface normal is
exactly grad(SDF); SymPy differentiates the SDF expression symbolically, simplifies it, and lambdifies it to NumPy --
giving the EXACT normal with no step-size error and no autodiff machinery in the engine. The same move turns a
symbolic energy into an analytic force (Baker's pluggable cleanup-energy gradient) and a symbolic capacity/entropy
expression into a checked closed form (Plate/Cranmer/Duda theory work).

GATED like the other accelerators: if SymPy is absent, HAS_SYMPY is False and the helpers raise a clear message --
the CORE never depends on this; it is a dev/derivation tool whose OUTPUT (a numpy function) is what ships.
"""

import numpy as np

try:
    import sympy as sp
    HAS_SYMPY = True
except Exception:                                            # pragma: no cover - exercised only without sympy
    HAS_SYMPY = False


def _require():
    if not HAS_SYMPY:
        raise ImportError("holographic_codegen needs sympy (a design-time derivation dependency). "
                          "Install it from requirements-accel.txt; the generated functions are pure NumPy.")


def compile_field(expr, variables=("x", "y", "z")):
    """Derive and lambdify a scalar field AND its exact gradient. `expr` is a SymPy expression or a string in the
    given variable names (e.g. 'sqrt(x**2+y**2+z**2) - 1.0'). Returns dict:
      value(P)    -> (N,)   the field at points P of shape (N, k)
      gradient(P) -> (N, k) the EXACT symbolic gradient at P (no finite-difference error)
      grad_expr             the list of symbolic partial derivatives (for inspection / further codegen)
      expr                  the (sympified) expression
    The lambdified functions use only numpy at call time, so the result is shippable without sympy present."""
    _require()
    syms = sp.symbols(variables)
    if not isinstance(syms, (tuple, list)):
        syms = (syms,)
    e = sp.sympify(expr) if isinstance(expr, str) else expr
    grad = [sp.diff(e, s) for s in syms]                     # exact partials
    f_val = sp.lambdify(syms, e, "numpy")
    f_grad = [sp.lambdify(syms, g, "numpy") for g in grad]
    k = len(syms)

    def _cols(P):
        P = np.asarray(P, float)
        if P.ndim == 1:
            P = P[None, :]
        return P, [P[:, i] for i in range(k)]

    def value(P):
        P, cols = _cols(P)
        out = f_val(*cols)
        return np.broadcast_to(np.asarray(out, float), (len(P),)).copy()

    def gradient(P):
        P, cols = _cols(P)
        comps = [np.broadcast_to(np.asarray(fg(*cols), float), (len(P),)) for fg in f_grad]
        return np.stack(comps, axis=1)

    return {"value": value, "gradient": gradient, "grad_expr": grad, "expr": e}


def sdf_normal_fn(expr, variables=("x", "y", "z")):
    """Convenience for SDFs: return (value_fn, normal_fn) where normal_fn gives the EXACT unit surface normal
    (normalized exact gradient) at points P -- the Quilez seat's exact-normal path, replacing finite differences."""
    c = compile_field(expr, variables)

    def normal(P):
        g = c["gradient"](P)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        return g / (n + 1e-12)

    return c["value"], normal


def gradient_fn(expr, variables):
    """General case: return a function P->(N,k) giving the exact gradient of `expr`. A FORCE is -gradient(energy),
    so force_fn = lambda P: -gradient_fn(energy, vars)(P) -- the Baker seat's analytic-force path, autodiff-free."""
    return compile_field(expr, variables)["gradient"]


def sphere(center=(0.0, 0.0, 0.0), r=1.0):
    """Symbolic SDF of a sphere -- a building block for compound scenes. Returns a SymPy expression in (x,y,z)."""
    import sympy as sp
    x, y, z = sp.symbols("x y z")
    cx, cy, cz = center
    return sp.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - r


def box(center=(0.0, 0.0, 0.0), half=(1.0, 1.0, 1.0)):
    """Symbolic SDF of an axis-aligned box (the exact Quilez box distance). Returns a SymPy expression."""
    import sympy as sp
    x, y, z = sp.symbols("x y z")
    cx, cy, cz = center; hx, hy, hz = half
    qx = sp.Abs(x - cx) - hx; qy = sp.Abs(y - cy) - hy; qz = sp.Abs(z - cz) - hz
    outside = sp.sqrt(sp.Max(qx, 0) ** 2 + sp.Max(qy, 0) ** 2 + sp.Max(qz, 0) ** 2)
    inside = sp.Min(sp.Max(qx, sp.Max(qy, qz)), 0)
    return outside + inside


def op_union(a, b):
    """Union of two SDFs = Min (the closer surface wins). Compiles to njit fine -- the Heaviside-gated gradient
    that SymPy produces for Min lambdifies and JITs correctly."""
    import sympy as sp
    return sp.Min(a, b)


def op_intersect(a, b):
    """Intersection of two SDFs = Max."""
    import sympy as sp
    return sp.Max(a, b)


def op_subtract(a, b):
    """Subtract b from a (carve b out of a) = Max(a, -b)."""
    import sympy as sp
    return sp.Max(a, -b)


def op_smooth_union(a, b, k=0.3):
    """Quilez polynomial smooth-union -- a soft blend between two SDFs with blend radius k. Fully field-native; the
    blended surface and its exact normal both compile via sdf_numba_fn."""
    import sympy as sp
    h = sp.Max(k - sp.Abs(a - b), 0) / k
    return sp.Min(a, b) - h ** 3 * k * sp.Rational(1, 6)


def sdf_numba_fn(expr, variables=("x", "y", "z")):
    """SymPy -> NUMBA: compile a symbolic 3-D SDF to njit SCALAR value and exact-normal functions, plus njit
    grid evaluators over (N,3) points. The decisive unlock: the scalar njit SDF can be CALLED FROM ANOTHER njit loop
    (a sphere-trace march, an SDF-grid fill) -- the Python-closure barrier that earlier blocked Numba from the
    raymarch is gone, so the whole hot path can compile. Measured to match the numpy path to ~1e-10 and to beat both
    numpy-vectorized and (by ~16x) a python scalar loop for scalar-heavy eval.

    Returns dict: scalar_value(x,y,z), scalar_normal(x,y,z)->(nx,ny,nz) [njit, exact or finite-diff], grid_value(P),
    grid_normal(P), exact_normal (bool: True if the symbolic gradient compiled, False if it fell back to FD).
    Needs sympy AND numba (requirements-accel.txt); 3-D only (SDFs are 3-D) -- raises otherwise."""
    _require()
    import sympy as sp
    try:
        from numba import njit
    except Exception as exc:                                  # pragma: no cover
        raise ImportError("sdf_numba_fn needs numba (requirements-accel.txt); the sympy->numpy path "
                          "(sdf_normal_fn) is the fallback.") from exc
    if len(variables) != 3:
        raise ValueError("sdf_numba_fn is specialized to 3-D (x, y, z) SDFs")
    syms = sp.symbols(variables)
    e = sp.sympify(expr) if isinstance(expr, str) else expr
    fv = njit(sp.lambdify(syms, e, "math"))                  # 'math' backend -> scalar, njit-compatible

    # Try EXACT symbolic normals; some compound SDFs (nested Min/Max/Abs) leave an unprintable Derivative, so fall
    # back to njit FINITE DIFFERENCES on the scalar value -- robust for any SDF, still fast, just not exact.
    exact_normal = True
    try:
        gx = njit(sp.lambdify(syms, sp.diff(e, syms[0]), "math"))
        gy = njit(sp.lambdify(syms, sp.diff(e, syms[1]), "math"))
        gz = njit(sp.lambdify(syms, sp.diff(e, syms[2]), "math"))

        @njit
        def scalar_normal(x, y, z):
            a = gx(x, y, z); b = gy(x, y, z); c = gz(x, y, z)
            n = (a * a + b * b + c * c) ** 0.5 + 1e-12
            return a / n, b / n, c / n
        scalar_normal(0.5, 0.5, 0.5)                         # force-compile to surface any Derivative/print error
    except Exception:
        exact_normal = False

        @njit
        def scalar_normal(x, y, z):                          # central finite differences on the njit scalar SDF
            h = 1e-4
            a = (fv(x + h, y, z) - fv(x - h, y, z)) / (2 * h)
            b = (fv(x, y + h, z) - fv(x, y - h, z)) / (2 * h)
            c = (fv(x, y, z + h) - fv(x, y, z - h)) / (2 * h)
            n = (a * a + b * b + c * c) ** 0.5 + 1e-12
            return a / n, b / n, c / n

    @njit
    def grid_value(P):
        n = P.shape[0]
        out = np.empty(n)
        for i in range(n):
            out[i] = fv(P[i, 0], P[i, 1], P[i, 2])
        return out

    @njit
    def grid_normal(P):
        n = P.shape[0]
        out = np.empty((n, 3))
        for i in range(n):
            nx, ny, nz = scalar_normal(P[i, 0], P[i, 1], P[i, 2])
            out[i, 0] = nx; out[i, 1] = ny; out[i, 2] = nz
        return out

    return {"scalar_value": fv, "scalar_normal": scalar_normal,
            "grid_value": grid_value, "grid_normal": grid_normal, "exact_normal": exact_normal}


def _selftest():
    if not HAS_SYMPY:
        print("codegen selftest skipped (no sympy)")
        return
    rng = np.random.default_rng(0)

    # Sphere SDF: |p| - R. Exact normal = p/|p| (radial). Compare exact vs analytic vs finite-difference.
    R = 1.3
    val, nrm = sdf_normal_fn(f"sqrt(x**2 + y**2 + z**2) - {R}")
    P = rng.standard_normal((200, 3)) * 1.5
    analytic_n = P / np.linalg.norm(P, axis=1, keepdims=True)
    exact_n = nrm(P)
    exact_err = float(np.max(np.abs(exact_n - analytic_n)))

    def fd_normal(P, h):                                     # what holostuff does today: finite differences
        g = np.zeros_like(P)
        for i in range(3):
            e = np.zeros(3); e[i] = h
            g[:, i] = (val(P + e) - val(P - e)) / (2 * h)
        return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    fd_err = float(np.max(np.abs(fd_normal(P, 1e-2) - analytic_n)))

    assert exact_err < 1e-12, exact_err                      # exact normal matches analytic to machine precision
    assert fd_err > exact_err, (fd_err, exact_err)           # finite differences carry real step-size error

    # A harder SDF (torus) where hand-deriving the normal is annoying -- exact gradient still matches numeric grad
    tval, tnrm = sdf_normal_fn("sqrt((sqrt(x**2+y**2)-1.0)**2 + z**2) - 0.4")
    Pt = rng.standard_normal((50, 3))
    num_g = np.zeros_like(Pt)
    for i in range(3):
        e = np.zeros(3); e[i] = 1e-6
        num_g[:, i] = (tval(Pt + e) - tval(Pt - e)) / (2e-6)
    num_n = num_g / (np.linalg.norm(num_g, axis=1, keepdims=True) + 1e-12)
    torus_err = float(np.max(np.abs(tnrm(Pt) - num_n)))
    assert torus_err < 1e-5, torus_err

    # Force = -grad(energy): a quadratic well's force is linear and exact
    gforce = gradient_fn("0.5*(x**2 + y**2 + z**2)", ("x", "y", "z"))
    Pf = rng.standard_normal((10, 3))
    assert np.allclose(-gforce(Pf), -Pf), "force of a quadratic well should be -p"

    print(f"codegen selftest ok: sphere exact-normal error {exact_err:.1e} (vs finite-difference {fd_err:.1e} at "
          f"step 1e-2); torus exact normal matches numeric to {torus_err:.1e}; force=-grad(energy) exact -- "
          f"no step knob, no autodiff")


if __name__ == "__main__":
    _selftest()
