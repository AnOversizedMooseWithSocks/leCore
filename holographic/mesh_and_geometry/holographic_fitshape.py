"""Fit a target to the closest procedural formula and return it as Shadertoy code (holographic_fitshape).

WHY THIS MODULE EXISTS
----------------------
The pieces to answer "give me the closest procedural formula for this shape/texture, as a Shadertoy" are all here now:
fold_fit (recover a fractal recipe from a point cloud), the fractal/SDF library, and sdf_shader (emit a complete
raymarch shader). This module is the DISPATCHER that ties them into one call and -- critically -- reports the fit
quality HONESTLY, with a baseline and its kept negatives, so a caller knows how close the match actually is.

It follows the shape of fit_deterministic: for a target, try the APPLICABLE fitters, keep the best by a measured
score, and return the winner plus its emitted code. It does NOT claim to fit anything ("a creature") -- it fits from a
named library of procedural models and tells you how well.

THE TWO HONEST PATHS (dispatched on the target)
  * 3-D POINT CLOUD -> a FRACTAL SDF recipe (via fold_fit) -> Shadertoy. This is the strong path: fold_fit recovers a
    fold recipe with a measured baseline-improvement, and the recipe reconstructs an SDF that emits a complete raymarch
    shader. Good for self-similar / fractal-ish 3-D structure (the fern/tree/coral family that IS fractal). Returns the
    recipe, the reconstructed SDF, the fit quality (baseline-improvement ratio), and the Shadertoy string.
  * 2-D IMAGE / HEIGHT / TEXTURE -> a PROCEDURAL NOISE formula (fBm) matched to the target's STATISTICAL SIGNATURE
    (spectral slope + detail energy), plus a GLSL noise snippet. KEPT NEGATIVE, loud: this is a STATISTICAL-SIMILARITY
    fit, NOT parameter recovery and NOT a pixel match -- fBm parameters are not uniquely identifiable from low-order
    statistics (measured: several parameter sets score as well as the truth), so we return "a procedural texture in the
    same family (matched roughness + detail)" and say so. It is useful for "give me a procedural stand-in for this
    texture", not "recover the exact generator".

WHAT IT DOES NOT DO (scoped, honest)
  * It does not fit arbitrary meshes to a minimal SDF-primitive tree (that is a separate CSG-search project).
  * It does not fit L-systems / Barnsley affine IFS (a fern's true generator) -- flagged as the next fitter to add;
    fold_fit's Mandelbox recipe is a fractal STAND-IN, not a botanical model.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_sdf import fold_fractal


def _texture_features(img):
    """A compact statistical signature of a 2-D field: (spectral slope, gradient energy). The spectral slope is the
    log-log fall-off of the radial power spectrum -- the 'roughness' / fractal-dimension signature that fBm's
    gain/lacunarity control; the gradient energy is how much fine detail is present. Normalised so it compares across
    scales. This is a SIGNATURE for matching a family, NOT a fingerprint that uniquely identifies parameters."""
    img = np.asarray(img, float)
    img = (img - img.mean()) / (img.std() + 1e-9)
    n = min(img.shape)
    f = np.fft.fftshift(np.abs(np.fft.fft2(img)))
    cy, cx = np.array(f.shape) // 2
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    prof = np.bincount(r.ravel(), f.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    prof = prof[1:n // 2]
    k = np.arange(1, len(prof) + 1)
    slope = float(np.polyfit(np.log(k), np.log(prof + 1e-9), 1)[0])
    gx, gy = np.gradient(img)
    grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    return np.array([slope, grad])


def _noise_glsl():
    """A small, self-contained value-noise + fBm GLSL snippet (the standard Shadertoy fBm helper) parameterised by
    OCTAVES / LACUNARITY / GAIN. Independent of leCore's internal noise -- a caller drops this into a shader."""
    return (
        "float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }\n"
        "float vnoise(vec2 p){ vec2 i=floor(p),f=fract(p); f=f*f*(3.0-2.0*f);\n"
        "  float a=hash(i),b=hash(i+vec2(1,0)),c=hash(i+vec2(0,1)),d=hash(i+vec2(1,1));\n"
        "  return mix(mix(a,b,f.x),mix(c,d,f.x),f.y); }\n"
        "float fbm(vec2 p,int oct,float lac,float gain){ float s=0.0,a=0.5,f=1.0;\n"
        "  for(int i=0;i<oct;i++){ s+=a*vnoise(p*f); f*=lac; a*=gain; } return s; }\n")


def fit_texture(target, octaves_range=(3, 5), lac_range=(1.8, 2.2),
                gain_range=(0.4, 0.6), bw_range=(2.0, 4.0), mind=None, res=24):
    """Fit a PROCEDURAL fBm formula to a 2-D `target` (image / height map / texture) by matching its STATISTICAL
    SIGNATURE (spectral slope + detail energy), and return a GLSL fbm snippet realising it. Grid-searches a small
    (octaves, lacunarity, gain, base_bandwidth) bank, scoring each by feature distance to the target, keeps the best.
    Uses the noise field's VECTORISED sample_grid_fast (not a per-point Python loop), so the search is a handful of
    grid renders.

    Returns {params, quality, baseline, glsl, note}. `quality` is 1/(1+loss) in [0,1] (higher = closer signature);
    `baseline` is the same for the bank's centre guess. KEPT NEGATIVE (in `note` too): this is a family match, not
    parameter recovery -- fBm parameters are not uniquely identifiable from low-order statistics, so the returned
    params are ONE setting whose texture shares the target's roughness + detail, not 'the' generator. Needs a `mind`
    for procedural_noise (pass the UnifiedMind)."""
    if mind is None:
        raise ValueError("fit_texture needs mind=<UnifiedMind> for procedural_noise")
    target = np.asarray(target, float)
    if target.ndim == 3:
        target = target.mean(axis=2)                            # collapse RGB to luminance
    N = min(res, min(target.shape))
    ti = np.linspace(0, target.shape[0] - 1, N).astype(int)
    tj = np.linspace(0, target.shape[1] - 1, N).astype(int)
    tf = _texture_features(target[np.ix_(ti, tj)])

    def render(o, l, g, bw):
        pn = mind.procedural_noise(n_dims=2, octaves=int(o), lacunarity=l, gain=g, base_bandwidth=bw, seed=1)
        return np.asarray(pn.sample_grid_fast(N))               # VECTORISED grid render (no per-point loop)

    def loss(o, l, g, bw):
        return float(np.sum((_texture_features(render(o, l, g, bw)) - tf) ** 2))

    best = None
    for o in octaves_range:
        for l in lac_range:
            for g in gain_range:
                for bw in bw_range:
                    lo = loss(o, l, g, bw)
                    if best is None or lo < best[1]:
                        best = ({"octaves": int(o), "lacunarity": float(l), "gain": float(g),
                                 "base_bandwidth": float(bw)}, lo)
    params, loss0 = best
    baseline = loss(octaves_range[len(octaves_range) // 2], lac_range[len(lac_range) // 2],
                    gain_range[len(gain_range) // 2], bw_range[len(bw_range) // 2])
    glsl = _noise_glsl() + (
        "// fbm(uv*%.3g, %d, %.3g, %.3g)  -- matched to the target's roughness + detail\n"
        % (params["base_bandwidth"], params["octaves"], params["lacunarity"], params["gain"]))
    return {
        "params": params,
        "quality": 1.0 / (1.0 + loss0),
        "baseline": 1.0 / (1.0 + baseline),
        "glsl": glsl,
        "note": "STATISTICAL-SIMILARITY fit (matched spectral slope + detail energy), NOT parameter recovery or a "
                "pixel match -- fBm params are not uniquely identifiable from these statistics; this is a procedural "
                "texture in the same family.",
    }


def fit_pointcloud(target, iterations=10, coarse=6, refine_steps=30, mind=None):
    """Fit a FRACTAL SDF recipe to a 3-D `target` (M,3) point cloud via fold_fit, reconstruct the SDF, and emit its
    Shadertoy. Returns {recipe, sdf, quality, baseline, shadertoy, note}. `quality` is the fold_fit baseline-
    improvement RATIO (the honest discriminative signal -- how much better than a default guess; >~3x is a real fit,
    ~1x means the cloud is not fractal-like). Good for self-similar 3-D structure; a smooth blob will score ~1x and
    say so."""
    if mind is None:
        raise ValueError("fit_pointcloud needs mind=<UnifiedMind> for fold_fit + sdf_shader")
    from holographic.mesh_and_geometry.holographic_foldfit import fold_fit
    res = fold_fit(target, iterations=iterations, coarse=coarse, refine_steps=refine_steps, mind=mind)
    scale, minr, L = res["recipe"]
    sdf = fold_fractal(iterations=iterations, scale=scale, min_radius=minr, fold_limit=L)
    ratio = res["baseline"] / max(res["loss"], 1e-9)
    shadertoy = mind.sdf_shader(sdf)
    return {
        "recipe": res["recipe"],
        "sdf": sdf,
        "quality": float(ratio),
        "baseline": 1.0,
        "shadertoy": shadertoy,
        "note": "fractal-SDF fit via fold_fit; quality is the baseline-improvement RATIO (>~3x = a real fractal fit, "
                "~1x = the cloud is not fractal-like). The recipe is a Mandelbox stand-in, not a botanical/L-system "
                "model.",
    }


def fit_shape(target, mind=None, **kw):
    """CLOSEST-FIT DISPATCHER: given a `target`, fit the closest procedural formula and return it with its Shadertoy /
    GLSL / recipe. Dispatches on the target shape:
      * a 3-D (M,3) POINT CLOUD -> `method='fractal'` (default): a fold recipe via fold_fit -> Shadertoy; OR
        `method='primitives'`: a union of spheres via fit_primitives (for a hard-surface/non-fractal 'creature'). There
        is NO reliable auto between them (a fractal's distance estimate reads ~0 near ANY cloud, so a residual compare
        always flatters it) -- the caller picks the family.
      * a 2-D (M,2) POINT CLOUD -> the CLOSEST NAMED AFFINE-IFS system (via ifs_fit) -- a fern/tree/sierpinski/...
      * a 2-D IMAGE / HEIGHT / TEXTURE grid -> a PROCEDURAL fBm matched to the statistical signature + a GLSL snippet.
    Returns the fitter's dict with an added `kind`. Honest by construction: each path reports a measured quality vs a
    baseline and carries its kept negative in `note`."""
    arr = np.asarray(target, float)
    if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 8:
        # 3-D point cloud. method='fractal' (default) fits a fold recipe; method='primitives' fits a sphere set.
        # There is NO reliable AUTO between them: a fold fractal's DE is a LOWER BOUND, so it reads ~0 near ANY bounded
        # cloud's surface (it 'contains' the points) -- measured, so a residual comparison always flatters the fractal.
        # The caller picks the family; both are offered because they suit different shapes (self-similar vs blobby).
        method = kw.pop("method", "fractal")
        if method == "primitives":
            return _fit_primitives_path(arr, **kw)
        out = fit_pointcloud(arr, mind=mind, **{k: v for k, v in kw.items()
                                                if k in ("iterations", "coarse", "refine_steps")})
        out["kind"] = "pointcloud->fractal_sdf"
        return out
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 64:
        from holographic.mesh_and_geometry.holographic_ifs import ifs_fit as _ifs_fit
        out = _ifs_fit(arr)                                    # (M,2) point cloud -> named affine IFS (fern/tree/...)
        out["kind"] = "pointcloud2d->affine_ifs"
        return out
    if arr.ndim in (2, 3):
        out = fit_texture(arr, mind=mind, **kw)                # 2-D image / height / texture -> fBm
        out["kind"] = "texture->fbm"
        return out
    raise ValueError("fit_shape: target must be a 2-D image/height field, a 2-D (M,2) or 3-D (M,3) point cloud")


def _fit_primitives_path(arr, **kw):
    """The 3-D primitive-set path of fit_shape: a union of spheres + its Shadertoy shader."""
    from holographic.mesh_and_geometry.holographic_primfit import fit_primitives
    from holographic.mesh_and_geometry.holographic_sdf import _emit_shader
    out = fit_primitives(arr, auto_k=kw.pop("auto_k", True), **{k: v for k, v in kw.items()
                                                                if k in ("k", "k_max", "tol")})
    out["shadertoy"] = _emit_shader(out["sdf"])
    out["kind"] = "pointcloud->primitive_sdf"
    out["note"] = ("union of %d spheres approximating the surface; quality = improvement over one bounding sphere. "
                   "Spheres only -- a blocky shape fits coarsely; not a minimal CSG tree." % out["k"])
    return out


def _selftest():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    # --- 3-D PATH: a fractal point cloud fits well (high ratio) and emits a shader. ---
    from holographic.mesh_and_geometry.holographic_foldfit import surface_points
    cloud = surface_points((2.1, 0.5, 1.0), n=250, iterations=10, seed=0)
    res = fit_shape(cloud, mind=m)
    assert res["kind"] == "pointcloud->fractal_sdf"
    assert "mainImage" in res["shadertoy"], "the 3-D fit emits a complete Shadertoy raymarch shader"
    assert res["quality"] > 3.0, "a real fractal cloud fits well (baseline-improvement %.1fx)" % res["quality"]
    # (the sphere-is-worse discrimination is already pinned in holographic_foldfit's selftest; not re-run here)

    # --- 2-D PATH: fits a texture's family and emits a GLSL fbm snippet; the KEPT NEGATIVE is reported. ---
    pn = m.procedural_noise(n_dims=2, octaves=5, lacunarity=2.0, gain=0.5, base_bandwidth=3.0, seed=7)
    tex = np.asarray(pn.sample_grid_fast(20))                  # vectorised render (no per-point loop)
    tres = fit_texture(tex, mind=m, res=16)                    # small res keeps the selftest quick
    assert "fbm(" in tres["glsl"] and "vnoise" in tres["glsl"], "the 2-D fit emits a GLSL fbm snippet"
    assert 0.0 <= tres["quality"] <= 1.0
    assert "NOT parameter recovery" in tres["note"], "the kept negative is reported loudly"
    assert fit_texture(tex, mind=m, res=16)["params"] == tres["params"], "fit_texture is deterministic"
    # the dispatcher routes a 2-D array to the texture path
    assert fit_shape(tex, mind=m)["kind"] == "texture->fbm"

    print("holographic_fitshape selftest: ok (3-D point cloud -> fold_fit recipe -> Shadertoy raymarch shader, real "
          "fractal fits %.1fx over baseline; 2-D texture -> fBm family match + GLSL fbm snippet, quality %.2f, KEPT "
          "NEGATIVE reported: statistical-similarity not parameter recovery; deterministic)"
          % (res["quality"], tres["quality"]))


if __name__ == "__main__":
    _selftest()
