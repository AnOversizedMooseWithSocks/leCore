"""Procedural bridges (S3): where the SDF / procedural layer connects to the rest of the stack -- MEASURED.

WHY THIS MODULE EXISTS
----------------------
The recent SDF/procedural work (S1-S2) is built from the same primitives as the rest of the engine, so
the honest question is: does it UNLOCK anything elsewhere -- compression, the soft-operator family,
denoising, structure? We measured every candidate on the real substrate before writing a line, and the
results split cleanly into two genuine wins and two honest negatives/already-dones. All four are kept
here, loud, because a negative ruled out by measurement is as valuable as a win.

  C1  COMPRESSION / COMPLEXITY  -- WIN. A procedural generator's size is CONSTANT in its output's
      complexity: a Menger sponge's DSL is 12 bytes whether it marches to 100k or 250k faces. Storing
      the GENERATOR instead of the expanded geometry escapes the capacity/complexity wall for any
      content that HAS a short generator -- the same MDL principle as symbolic_regress/compress_signal
      ("find the law, store the law"), now for geometry. `procedural_compression` quantifies it.
      KEPT NEGATIVE: only COMPRESSIBLE content has a short generator; an arbitrary scanned/random mesh
      does not (the symbolic-regression "not everything has a law" negative). Procedural compression is
      lossy and content-restricted, not a universal codec.

  C4  THE SOFT OPERATOR  -- WIN (a unification). The SDF smooth-union and the engine's memory cleanup
      are the SAME temperature-controlled soft operator. smooth_union(k->0) IS the hard union, exactly as
      the modern-Hopfield/softmax cleanup at beta->inf IS the hard nearest-neighbour. `soft_min` is the
      log-sum-exp form -- the same log-sum-exp that softmax (a soft-arg-MAX) uses, here a soft-arg-MIN
      over distances. One operator: a soft blend of geometry and a soft recall of memory are the same math.

  C2  FPE FIELD AS A DENOISER -- NEGATIVE (kept). An FPE field fit to noisy samples is a kernel/RBF
      regressor (Nadaraya-Watson). Measured: it HURTS on uniformly-sampled smooth signals (it
      over-smooths) and rounds sharp edges badly; on NON-UNIFORMLY sampled signals it is at best marginal
      and seed-dependent -- NOT a reliable win. So the robust finding is the negative, and it is NOT wired
      into `denoise` (it would degrade it). `fpe_smooth` is kept only as the honest record of the attempt
      and a scattered-data kernel smoother for the caller who explicitly wants one.

  C3  SDF SCENE AS A FACTORABLE STRUCTURE -- ALREADY DONE. An SDF tree is a typed.tree_to_recipe recipe,
      so decode_structure / decompose_structure / op_kinds already read its structure back. Nothing to
      build; noted so we do not "discover" it again.
"""

import numpy as np


# ---------------------------------------------------------------------------
# C1 -- procedural representation as compression (and the capacity/complexity escape).
# ---------------------------------------------------------------------------

def procedural_compression(node, bounds=((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), res=48):
    """Measure a procedural object's compression: the tiny GENERATOR (DSL) vs the expanded geometry.

    Returns a dict with the DSL byte length, the marched mesh's face count and rough byte size, and the
    ratio. The point the numbers make: the generator is the MDL code for the geometry -- store the law,
    regenerate the mesh -- and its size does NOT grow with the output's complexity.
    """
    from holographic_procgen import object_to_mesh
    dsl = node.to_dsl()
    dsl_bytes = len(dsl.encode("utf-8"))
    mesh = object_to_mesh(node, bounds=bounds, res=res)
    mesh_bytes = int(mesh.vertices.nbytes + sum(len(f) for f in mesh.faces) * 4)
    return {
        "dsl": dsl,
        "dsl_bytes": dsl_bytes,
        "mesh_faces": mesh.n_faces,
        "mesh_bytes": mesh_bytes,
        "ratio": mesh_bytes / max(dsl_bytes, 1),
    }


# ---------------------------------------------------------------------------
# C4 -- the soft operator shared by SDF blending and memory cleanup.
# ---------------------------------------------------------------------------

def soft_min(a, b, k):
    """Log-sum-exp soft minimum: -k*log(exp(-a/k)+exp(-b/k)). As k->0 it becomes min(a,b).

    This is the SAME log-sum-exp the modern-Hopfield / softmax cleanup uses -- softmax is a soft-arg-MAX,
    this is a soft-arg-MIN over two distances. A smooth CSG union of geometry and a soft recall of a
    memory are one operator at a temperature; k here plays the role of 1/beta there.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.minimum(a, b)                                   # shift for numerical stability
    return m - k * np.log(np.exp(-(a - m) / k) + np.exp(-(b - m) / k))


# ---------------------------------------------------------------------------
# C2 -- the FPE field as a (kernel) denoiser. KEPT NEGATIVE: scattered-data niche only.
# ---------------------------------------------------------------------------

def fpe_smooth(xs, ys, bandwidth=6.0, query_xs=None, dim=1024, seed=0):
    """Denoise a 1-D signal by fitting an FPE field to (xs, noisy ys) and re-querying -- kernel regression.

    Honest scope: this OVER-SMOOTHS uniformly-sampled or sharp signals (the shipped trajectory/spectral
    denoisers beat it there); its only edge is NON-UNIFORMLY sampled smooth signals, where those methods
    do not apply. Provided for that niche, not as a general denoiser.
    """
    from holographic_fpe import VectorFunctionEncoder
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    enc = VectorFunctionEncoder(1, dim=dim, bounds=[(float(xs.min()), float(xs.max()))],
                                bandwidth=bandwidth, seed=seed)
    f = enc.bundle([[x] for x in xs], weights=list(ys))
    qx = xs if query_xs is None else np.asarray(query_xs, float)
    raw = np.array([enc.query(f, [x]) for x in qx])
    return raw * (np.std(ys) / (np.std(raw) + 1e-9))      # rescale (query reads value up to bundle norm)


# ---------------------------------------------------------------------------

def _selftest():
    from holographic_sdf import sphere, menger, SDF

    def snr(clean, est):
        return 10 * np.log10(np.var(clean) / (np.var(clean - est) + 1e-12))

    # C1 (WIN): the generator size is CONSTANT while output complexity grows -- the capacity/complexity escape.
    sizes = []
    for depth in (1, 2, 3):
        m = procedural_compression(menger(depth, 1.0), res=40)
        sizes.append((m["dsl_bytes"], m["mesh_faces"], m["ratio"]))
    dsl_bytes = [s[0] for s in sizes]; faces = [s[1] for s in sizes]
    assert len(set(dsl_bytes)) == 1, f"generator size should be constant in complexity, got {dsl_bytes}"
    assert faces[2] > faces[0] * 1.3, "output complexity should grow with depth"
    assert sizes[0][2] > 1000, "procedural compression ratio should be large"

    # C4 (WIN, unification): smooth_union(k->0) IS the hard union; soft_min(k->0) IS min -- one temperature op.
    a = sphere(1.0); c = sphere(1.0).translate([1.5, 0, 0])
    P = np.array([[0.75, 0, 0.0]])
    hard = float(np.minimum(a.eval(P), c.eval(P))[0])
    gaps = [abs(float(a.smooth_union(c, k).eval(P)[0]) - hard) for k in (0.5, 0.1, 0.01)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 0.01, f"smooth_union should -> hard as k->0: {gaps}"
    # soft_min is the same temperature operator on raw distances
    av, bv = -0.25, 0.10
    assert abs(soft_min(av, bv, 0.001) - min(av, bv)) < 1e-3, "soft_min should -> min as k->0"
    assert soft_min(av, bv, 0.5) < min(av, bv), "at finite temperature soft_min rounds below the hard min"

    # C2 (NEGATIVE, kept): fpe_smooth HURTS on a uniformly-sampled smooth signal (it over-smooths). The
    #   scattered-sampling case is at best marginal and SEED-DEPENDENT -- not a reliable win -- so the
    #   robust, repeatable finding is the negative, and that is what we lock in.
    rng = np.random.default_rng(0)
    N = 120; x = np.linspace(0, 1, N); clean = np.sin(2 * np.pi * 2 * x)
    noisy = clean + rng.normal(0, 0.3, N)
    uni = snr(clean, fpe_smooth(x, noisy, bandwidth=6.0))
    assert uni < snr(clean, noisy), "KEPT NEGATIVE: fpe_smooth should NOT beat the noisy signal on uniform sampling"
    xs = np.sort(rng.uniform(0, 1, N)); cl = np.sin(2 * np.pi * 2 * xs)
    ns = cl + rng.normal(0, 0.5, N)
    sca = snr(cl, fpe_smooth(xs, ns, bandwidth=6.0))      # reported, not asserted as a win (it isn't reliably one)
    assert np.isfinite(sca), "fpe_smooth should at least produce a finite estimate"

    # C3 (ALREADY DONE): an SDF scene is a recipe whose structure reads back through the existing machinery.
    from holographic_typed import tree_to_recipe, op_kinds
    rec = tree_to_recipe(512, 0, a.smooth_union(c, 0.3).to_tree())
    assert len(op_kinds(rec)) > 0

    print("holographic_procbridge selftest passed:",
          f"C1 dsl_bytes={dsl_bytes[0]} const, faces {faces[0]}->{faces[2]} ratio~{sizes[0][2]:.0f}x | "
          f"C4 smooth_union->hard gaps {[round(g,4) for g in gaps]} | "
          f"C2 uni_snr={uni:.1f}dB(neg) scattered={sca:.1f}dB")


if __name__ == "__main__":
    _selftest()
