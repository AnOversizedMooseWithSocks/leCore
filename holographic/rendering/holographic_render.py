"""A CPU rendering subsystem (RND-1): camera, lights, a mesh rasteriser, and a volumetric ray-marcher.

WHY THIS MODULE EXISTS
----------------------
The engine could PROJECT geometry (mesh, splats, fields) and emit a GPU shader (sdf_shader -> GLSL), but it had
no way to produce a rasterised IMAGE on its own -- no camera, no lights, no "render this scene to pixels." This
module is that missing piece, in pure NumPy, and it is built to fit the engine's representations:

  * The VOLUMETRIC renderer operates DIRECTLY on the engine's fields. Smoke, fire, water, and surfaced particles
    are all density (and emission) FIELDS -- a callable points(N,3)->density. Rendering one is marching camera
    rays through the field and accumulating the volume-rendering integral (transmittance * emission, with
    absorption). So this is genuinely field-native: the field IS the volume. Fire = emission (a blackbody-ish
    ramp on density); smoke = grey absorption; water = tinted absorption.
  * The RASTERISER turns the projected meshes into a shaded image with a z-buffer and Lambert shading, using a
    PBRMaterial's base colour and the lights.

PERFORMANCE, HONESTLY
  This is a correctness-first CPU renderer. The volumetric path is vectorised over ALL pixels at once (every ray
  marched in lockstep, one field sample call per step), which is the fast path NumPy is good at. The rasteriser
  loops over triangles in Python (each triangle's pixels are filled vectorised), with frustum + back-face
  CULLING so the loop only touches visible faces -- "cull, don't batch." It is NOT a GPU rasteriser and does not
  pretend to be: for a heavy interactive viewport the GPU stays the muscle (this is the brain that can render
  offline frames, previews, and -- via the delta idea -- only the TILES that changed). Measured throughput is
  reported in _selftest so the bound is on the record, not hidden.
"""

import numpy as np


# =====================================================================================================
# Camera -- view + projection, and per-pixel rays for the volumetric path.
# =====================================================================================================
class Camera:
    """A pinhole camera. `eye` looks at `target` with `up`; `fov_deg` is the vertical field of view."""

    def __init__(self, eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
                 fov_deg=50.0, aspect=1.0, near=0.05, far=100.0):
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        self.up = np.asarray(up, float)
        self.fov_deg = float(fov_deg)
        self.aspect = float(aspect)
        self.near = float(near)
        self.far = float(far)

    def _basis(self):
        """The camera's right/up/forward orthonormal basis (forward points FROM eye TO target)."""
        f = self.target - self.eye
        f = f / (np.linalg.norm(f) + 1e-12)
        r = np.cross(f, self.up)
        r = r / (np.linalg.norm(r) + 1e-12)
        u = np.cross(r, f)
        return r, u, f

    def view_matrix(self):
        """World -> camera (look-at). Camera looks down -z in view space (OpenGL convention)."""
        r, u, f = self._basis()
        R = np.stack([r, u, -f], axis=0)                      # rows = camera axes
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = -R @ self.eye
        return M

    def projection_matrix(self, aspect=None):
        """Perspective projection (OpenGL-style, maps the frustum to the [-1,1] cube).

        `aspect` overrides the stored `self.aspect` for this call. The renderer passes the FRAME's own
        width/height, because that -- not a value stored on the camera -- is the aspect actually being drawn
        into. See rasterize_mesh's WHY comment; ray_dirs has always done the same thing for the ray path."""
        t = np.tan(np.radians(self.fov_deg) / 2.0)
        a = float(self.aspect if aspect is None else aspect)
        n, fa = self.near, self.far
        P = np.zeros((4, 4))
        P[0, 0] = 1.0 / (a * t)
        P[1, 1] = 1.0 / t
        P[2, 2] = -(fa + n) / (fa - n)
        P[2, 3] = -2.0 * fa * n / (fa - n)
        P[3, 2] = -1.0
        return P

    def ray_dirs(self, width, height, jitter=None):
        """Per-pixel world-space ray origins (the eye) and unit directions, shape (H, W, 3), for ray marching.
        Pixel centres by default (y increasing downward, row 0 = top). Pass `jitter=(dx, dy)` -- each a scalar or
        an (H,W) array of sub-pixel offsets in [-0.5, 0.5) -- to shoot the ray through a jittered point inside the
        pixel instead of the centre; averaging several jittered samples anti-aliases the edges (see path_trace)."""
        r, u, f = self._basis()
        t = np.tan(np.radians(self.fov_deg) / 2.0)
        ys, xs = np.mgrid[0:height, 0:width]
        ox, oy = (0.5, 0.5) if jitter is None else (0.5 + jitter[0], 0.5 + jitter[1])
        # The horizontal must be widened by the frame's OWN aspect (width/height) or a circle renders as an ellipse
        # -- e.g. a sphere in a 240x180 (4:3) frame looks squished if we used a square aspect. Derive it from the
        # actual pixel grid rather than a stored self.aspect that may not match the resolution being rendered.
        aspect = width / height
        ndc_x = (2.0 * (xs + ox) / width - 1.0) * aspect * t
        ndc_y = (1.0 - 2.0 * (ys + oy) / height) * t
        dirs = (ndc_x[..., None] * r + ndc_y[..., None] * u + f)   # f is forward
        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12)
        return self.eye, dirs


# =====================================================================================================
# Lights -- a tiny set: directional (sun), point, and a global ambient term.
# =====================================================================================================
class Light:
    """A light. kind='directional' uses `direction` (toward the scene); 'point' uses `position`; 'ambient' is a
    constant fill. `color` is RGB in [0,1], `intensity` scales it."""

    def __init__(self, kind="directional", direction=(-0.4, -0.8, -0.5), position=(2.0, 3.0, 2.0),
                 color=(1.0, 1.0, 1.0), intensity=1.0):
        self.kind = kind
        self.direction = np.asarray(direction, float)
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-12)
        self.position = np.asarray(position, float)
        self.color = np.asarray(color, float)
        self.intensity = float(intensity)


# =====================================================================================================
# Mesh rasteriser -- z-buffered, flat Lambert shading, frustum + back-face culled.
# =====================================================================================================
def rasterize_mesh(mesh, camera, width=512, height=512, lights=None, base_color=(0.8, 0.8, 0.8),
                   background=(0.05, 0.06, 0.08), ambient=0.15, vectorized=True, texture=None, uvs=None,
                   smooth=False, two_sided=False, vertex_colors=None):
    """Rasterise a triangle mesh to an (H, W, 3) RGB image in [0,1] with a z-buffer and per-face Lambert shading.
    `lights` is a list of Light (defaults to one directional sun + ambient). `base_color` is the surface albedo
    (or pass a PBRMaterial's base_color). Frustum-clips and back-face culls.

    vectorized=True (default) ports the per-triangle Python loop to a single VECTORISED fragment scatter -- the
    "cull, don't batch / scatter-accumulate" pattern: cull to visible faces, expand each face's bounding box into
    a flat fragment array with repeat/cumsum (the same ragged-expand as spatial_hash_pairs), compute every
    fragment's barycentric coords at once, and resolve the z-buffer with a single lexsort (sort fragments by
    pixel then depth, take the nearest per pixel) -- no Python per-triangle loop. This is the concrete sense in
    which "more VSA-native" (array ops, scatter = bundle) beats a Python loop: it is the SAME NumPy underneath,
    but as one batched op instead of N small ones. vectorized=False keeps the readable per-triangle loop as the
    reference. Both give the same image.

    SMOOTH shading (default-off, byte-identical absent): smooth=True interpolates per-VERTEX normals across each
    fragment (Gouraud/Phong-style) instead of one flat normal per face -- what makes an organic model (a subdivided
    creature, a fruit) read as curved rather than faceted. Needs vectorized=True. Costs one vertex_normals() call.

    TEXTURED path (default-off, byte-identical absent): pass `texture` (H,W,3 float [0,1]) and per-vertex `uvs`
    (n_verts,2) -- or leave uvs=None to read mesh.uvs -- and every fragment's albedo is the BILINEARLY sampled
    texture at its barycentric-interpolated UV instead of the flat base_color (lighting unchanged). What showing a
    UV-transferred / baked texture through the engine needs. Requires vectorized=True (the fragment path carries
    the barycentrics); textured + vectorized=False raises."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    # P3 (client): accept a plain dict / CameraController here, not only a strict Camera. fit_camera returns a
    # dict and callers piped it straight in, hitting "dict has no attribute projection_matrix". render_mesh
    # already coerced at its boundary; doing it here too means the LOW-LEVEL entry point is as forgiving as the
    # faculty. as_camera passes a real Camera through by identity, so nothing that already works changes.
    if not hasattr(camera, "projection_matrix"):
        from holographic.io_and_interop.holographic_coerce import as_camera
        camera = as_camera(camera)
    if uvs is None:
        uvs = getattr(mesh, "uvs", None)                     # per-vertex UVs survive fan triangulation (indices kept)
    # VCOL: per-vertex colours from the param or the mesh attribute (RGBA -> RGB). Takes precedence over
    # texture when given, so a coloured recall-bake renders with no atlas. None = the old texture/flat paths.
    vcol = vertex_colors
    if vcol is None and getattr(mesh, "colours", None) is not None and texture is None:
        vcol = np.asarray(mesh.colours, float)[:, :3]
    elif vcol is not None:
        vcol = np.asarray(vcol, float)
        if vcol.shape[1] == 4:
            vcol = vcol[:, :3]
    if not all(len(f) == 3 for f in mesh.faces):
        mesh = Mesh(mesh.vertices, [tuple(t) for t in mesh.triangulate()])
    if vcol is not None and not vectorized:
        raise ValueError("vertex-colour rendering needs vectorized=True (the fragment path carries the barycentrics)")
    if texture is not None:
        if not vectorized:
            raise ValueError("texture rendering needs vectorized=True (the fragment path carries the barycentrics)")
        if uvs is None:
            raise ValueError("texture given but no UVs: pass uvs= or set mesh.uvs")
        texture = np.asarray(texture, float)
        uvs = np.asarray(uvs, float)
    if lights is None:
        lights = [Light("directional", direction=(-0.4, -0.8, -0.5), intensity=1.0)]
    base_color = np.asarray(base_color[:3], float)

    vnormals = None
    if smooth:
        if not vectorized:
            raise ValueError("smooth shading needs vectorized=True (the fragment path carries the barycentrics)")
        vnormals = mesh.vertex_normals(store=False)          # per-vertex shading normals (indices match F)

    V = mesh.vertices
    F = np.asarray(mesh.faces, dtype=int)
    # THE FRAME'S OWN ASPECT drives the projection, not a value stored on the camera. MEASURED BUG this fixes:
    # at 640x360 a sphere rasterised 1.78x wider than tall (an ellipse), because the stored self.aspect
    # defaults to 1.0 while the pixel grid is 16:9 -- and the RAY path (ray_dirs) already derived it from
    # width/height, so the engine's two render paths disagreed about the shape of the same scene under the same
    # camera. That disagreement is what makes this a bug rather than a convention. A camera with an explicit
    # aspect can still force one by rendering at a matching size; square frames -- every default, every
    # turnaround view, every silhouette guard view -- are bit-identical either way.
    try:
        _P = camera.projection_matrix(aspect=width / float(height))
    except TypeError:
        _P = camera.projection_matrix()          # duck-typed camera without the override: leave it alone
    MVP = _P @ camera.view_matrix()
    Vh = np.hstack([V, np.ones((len(V), 1))])
    clip = Vh @ MVP.T
    w = clip[:, 3:4]
    ndc = clip[:, :3] / np.where(np.abs(w) < 1e-9, 1e-9, w)
    sx = (ndc[:, 0] * 0.5 + 0.5) * width
    sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    sz = ndc[:, 2]
    screen = np.stack([sx, sy], axis=1)

    img = np.tile(np.asarray(background, float), (height, width, 1))
    zbuf = np.full((height, width), np.inf)

    P0, P1, P2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(P1 - P0, P2 - P0)
    fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    centroids = (P0 + P1 + P2) / 3.0

    # per-face Lambert shading (vectorised over faces), -> a colour per face
    shade = np.zeros((len(F), 3))
    for lt in lights:
        if lt.kind == "ambient":
            shade += lt.intensity * lt.color
            continue
        if lt.kind == "directional":
            L = np.broadcast_to(-lt.direction, fn.shape)
        else:
            L = lt.position - centroids
            L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
        ndl_raw = np.sum(fn * L, axis=1)
        # two-sided: |n.l| -- thin sheets and unorientable patches shade like their front (see docstring WHY)
        ndl = np.abs(ndl_raw) if two_sided else np.clip(ndl_raw, 0.0, None)
        shade += ndl[:, None] * lt.intensity * lt.color
    face_light = ambient + shade                              # the light term alone (albedo applied at resolve)
    face_col = np.clip(face_light * base_color, 0.0, 1.0)

    front = (w[F[:, 0], 0] > 0) & (w[F[:, 1], 0] > 0) & (w[F[:, 2], 0] > 0)
    vis = np.where(front)[0]
    if vis.size == 0:
        return img

    if not vectorized:
        for fi in vis:
            _raster_one(fi, F, screen, sz, face_col[fi], img, zbuf, width, height)
        return img

    # ---- vectorised fragment scatter -------------------------------------------------------------
    pa = screen[F[vis, 0]]; pb = screen[F[vis, 1]]; pc = screen[F[vis, 2]]
    minx = np.clip(np.floor(np.minimum.reduce([pa[:, 0], pb[:, 0], pc[:, 0]])).astype(int), 0, width - 1)
    maxx = np.clip(np.ceil(np.maximum.reduce([pa[:, 0], pb[:, 0], pc[:, 0]])).astype(int), 0, width - 1)
    miny = np.clip(np.floor(np.minimum.reduce([pa[:, 1], pb[:, 1], pc[:, 1]])).astype(int), 0, height - 1)
    maxy = np.clip(np.ceil(np.maximum.reduce([pa[:, 1], pb[:, 1], pc[:, 1]])).astype(int), 0, height - 1)
    bw = (maxx - minx + 1).clip(min=0)
    bh = (maxy - miny + 1).clip(min=0)
    counts = bw * bh
    keep = counts > 0
    if not np.any(keep):
        return img
    vis, pa, pb, pc, minx, miny, bw, bh, counts = (vis[keep], pa[keep], pb[keep], pc[keep],
                                                   minx[keep], miny[keep], bw[keep], bh[keep], counts[keep])
    offs = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offs[1:])
    nfrag = int(offs[-1])
    ft = np.repeat(np.arange(len(counts)), counts)            # which triangle each fragment belongs to
    local = np.arange(nfrag) - offs[ft]                       # the k-th fragment within its triangle's bbox
    fpx = minx[ft] + (local % bw[ft])
    fpy = miny[ft] + (local // bw[ft])
    px = fpx + 0.5; py = fpy + 0.5
    a0 = pa[ft]; b0 = pb[ft]; c0 = pc[ft]
    d = (b0[:, 1] - c0[:, 1]) * (a0[:, 0] - c0[:, 0]) + (c0[:, 0] - b0[:, 0]) * (a0[:, 1] - c0[:, 1])
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    l0 = ((b0[:, 1] - c0[:, 1]) * (px - c0[:, 0]) + (c0[:, 0] - b0[:, 0]) * (py - c0[:, 1])) / d
    l1 = ((c0[:, 1] - a0[:, 1]) * (px - c0[:, 0]) + (a0[:, 0] - c0[:, 0]) * (py - c0[:, 1])) / d
    l2 = 1.0 - l0 - l1
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    if not np.any(inside):
        return img
    ti = vis[ft[inside]]                                      # original face index per kept fragment
    depth = l0[inside] * sz[F[ti, 0]] + l1[inside] * sz[F[ti, 1]] + l2[inside] * sz[F[ti, 2]]
    pix = fpy[inside] * width + fpx[inside]
    # z-resolve: nearest fragment wins per pixel (lexsort by pixel, then depth; take first per pixel)
    order = np.lexsort((depth, pix))
    pix_s = pix[order]
    first = np.ones(len(pix_s), dtype=bool)
    first[1:] = pix_s[1:] != pix_s[:-1]                       # first occurrence of each pixel = its min depth
    win = order[first]
    flat = img.reshape(-1, 3)
    # SMOOTH shading: per-fragment normal = barycentric blend of the winning face's vertex normals; recompute the
    # Lambert term per fragment (default flat path uses the per-face face_light, byte-identical).
    smooth_light = None
    if smooth and vnormals is not None:
        li0, li1, li2 = l0[inside][win], l1[inside][win], l2[inside][win]
        fw = ti[win]
        fn = (li0[:, None] * vnormals[F[fw, 0]] + li1[:, None] * vnormals[F[fw, 1]] + li2[:, None] * vnormals[F[fw, 2]])
        fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
        sl = np.zeros((len(fw), 3))
        for lt in lights:
            if lt.kind == "ambient":
                sl += lt.intensity * lt.color; continue
            if lt.kind == "directional":
                L = np.broadcast_to(-np.asarray(lt.direction, float), fn.shape)
            else:
                L = np.asarray(lt.position, float) - centroids[fw]
                L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
            ndl_raw = np.sum(fn * L, axis=1)
            ndl = np.abs(ndl_raw) if two_sided else np.clip(ndl_raw, 0.0, None)
            sl += ndl[:, None] * lt.intensity * lt.color
        smooth_light = ambient + sl                          # (n_win, 3)
    if vcol is not None:
        # VCOL: per-fragment colour = barycentric blend of the winning face's corner COLOURS, times the light
        # term -- the same machine as the texture path (barycentric interpolate + shade) but the albedo comes
        # from mesh.colours instead of a sampled atlas. This is what lets H5's vertex-scale recall bake (which
        # produces per-vertex colour with nowhere to render) and any coloured DCC mesh be SHOWN with no atlas.
        # Mirrors the texture branch exactly so shading (flat/smooth/two_sided) is byte-identical in behaviour.
        li0, li1, li2 = l0[inside][win], l1[inside][win], l2[inside][win]
        fw = ti[win]
        albedo = (li0[:, None] * vcol[F[fw, 0]] + li1[:, None] * vcol[F[fw, 1]] + li2[:, None] * vcol[F[fw, 2]])
        if smooth_light is not None:
            flat[pix[win]] = np.clip(smooth_light * albedo, 0.0, 1.0)
        else:
            flat[pix[win]] = np.clip(face_light[fw][:, None] * albedo if face_light.ndim == 1
                                     else face_light[fw] * albedo, 0.0, 1.0)
    elif texture is None:
        if smooth_light is not None:
            flat[pix[win]] = np.clip(smooth_light * base_color, 0.0, 1.0)
        else:
            flat[pix[win]] = face_col[ti[win]]                # the flat-albedo path, byte-identical to before
    else:
        # per-fragment UV = barycentric blend of the winning face's corner UVs, BILINEARLY sampled, times the
        # face (or smooth) light term. This is what lets a UV-transferred / baked texture be SHOWN through the engine.
        li0, li1, li2 = l0[inside][win], l1[inside][win], l2[inside][win]
        fw = ti[win]
        uvw = (li0[:, None] * uvs[F[fw, 0]] + li1[:, None] * uvs[F[fw, 1]] + li2[:, None] * uvs[F[fw, 2]])
        th, tw = texture.shape[:2]
        fu = (uvw[:, 0] % 1.0) * (tw - 1)
        fv = (uvw[:, 1] % 1.0) * (th - 1)
        x0 = np.floor(fu).astype(int); y0 = np.floor(fv).astype(int)
        x1 = np.minimum(x0 + 1, tw - 1); y1 = np.minimum(y0 + 1, th - 1)
        ax = (fu - x0)[:, None]; ay = (fv - y0)[:, None]
        albedo = ((texture[y0, x0] * (1 - ax) + texture[y0, x1] * ax) * (1 - ay)
                  + (texture[y1, x0] * (1 - ax) + texture[y1, x1] * ax) * ay)
        if smooth_light is not None:
            flat[pix[win]] = np.clip(smooth_light * albedo, 0.0, 1.0)
        else:
            flat[pix[win]] = np.clip(face_light[fw][:, None] * albedo if face_light.ndim == 1
                                     else face_light[fw] * albedo, 0.0, 1.0)
    return img


def _raster_one(fi, F, screen, sz, col, img, zbuf, width, height):
    """The reference per-triangle fill (vectorized=False path)."""
    a, b, c = F[fi]
    pa, pb, pc = screen[a], screen[b], screen[c]
    minx = max(int(np.floor(min(pa[0], pb[0], pc[0]))), 0); maxx = min(int(np.ceil(max(pa[0], pb[0], pc[0]))), width - 1)
    miny = max(int(np.floor(min(pa[1], pb[1], pc[1]))), 0); maxy = min(int(np.ceil(max(pa[1], pb[1], pc[1]))), height - 1)
    if minx > maxx or miny > maxy:
        return
    ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
    px = xs + 0.5; py = ys + 0.5
    d = ((pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1]))
    if abs(d) < 1e-9:
        return
    l0 = ((pb[1] - pc[1]) * (px - pc[0]) + (pc[0] - pb[0]) * (py - pc[1])) / d
    l1 = ((pc[1] - pa[1]) * (px - pc[0]) + (pa[0] - pc[0]) * (py - pc[1])) / d
    l2 = 1.0 - l0 - l1
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    depth = l0 * sz[a] + l1 * sz[b] + l2 * sz[c]
    sub_z = zbuf[miny:maxy + 1, minx:maxx + 1]
    win = inside & (depth < sub_z)
    if not np.any(win):
        return
    sub_img = img[miny:maxy + 1, minx:maxx + 1]
    sub_img[win] = col
    sub_z[win] = depth[win]


# =====================================================================================================
# Volumetric renderer -- march camera rays through a density (and emission) FIELD.
# =====================================================================================================
def volume_render(field, camera, bounds, width=256, height=256, steps=96, mode="smoke",
                  sigma=12.0, emission_color=None, albedo=(0.9, 0.9, 0.95), lights=None,
                  background=(0.0, 0.0, 0.0), early_term=True, empty_skip=True, occ_res=24,
                  occ_thresh=1e-3, term_eps=2e-3, self_shadow=False, shadow_steps=16,
                  shadow_sigma=None, ambient=(0.42, 0.52, 0.66), phase_g=0.0, powder=False,
                  multi_scatter=1, only=None):
    """Render a density FIELD (callable points(N,3)->density>=0) volumetrically by marching camera rays through
    `bounds`=(min_corner, max_corner) and accumulating the volume-rendering integral. Vectorised over ALL pixels.
    Returns (img RGB (H,W,3), alpha (H,W)).

    `mode`: 'smoke' (grey/`albedo` absorption), 'fire' (EMISSION via a blackbody-ish ramp on density),
    'density' (raw density as luminance). `sigma` scales optical density.

    TWO production-renderer (V-Ray-style) optimisations, both default-on and result-preserving:
      * empty_skip -- an EMPTY-SPACE-SKIPPING macro grid: the field is sampled ONCE on a coarse occ_res^3 grid;
        during marching, a ray in an empty macro-cell contributes nothing, so the fine field is sampled ONLY for
        rays whose macro-cell is occupied. For sparse volumes (smoke is mostly air) this slashes the dominant
        cost -- field evaluations. (The engine's own "cull, don't batch": skip the work, don't do it faster.)
      * early_term -- EARLY RAY TERMINATION: a ray whose transmittance has fallen below `term_eps` is opaque, so
        it is dropped from the active set and no longer sampled (V-Ray/PBRT Russian-roulette-style).
    Set either False to fall back to the plain uniform march.

    `self_shadow` (default off) turns on SINGLE-SCATTERING self-shadowing for `mode='smoke'`: at each sample the
    density is marched `shadow_steps` toward the light to get that point's transmittance to the sun, so a thick
    volume is BRIGHT on its lit crown and DARK in its self-shadowed core -- the thing that makes a cloud read as a
    solid 3-D body rather than a flat blob. `ambient` is the sky fill colour that lifts the shadowed side (so it
    goes blue-grey, not black). `shadow_sigma` defaults to `sigma`. Costs one short extra march per sample; leave
    off for thin smoke/fog where the flat term is already fine.

    Three further physically-motivated refinements (all self_shadow-only; see cloud_field / the Nubis course notes
    -- Schneider, "Authoring Real-Time Volumetric Cloudscapes", SIGGRAPH 2015/2017 -- for the technique lineage):
      * `phase_g` (0 = off, try ~0.7-0.85) -- a Henyey-Greenstein forward-scattering lobe: light travelling THROUGH
        the cloud toward the camera is brighter than light bouncing back, which is what makes a cloud's edge glow
        (the "silver lining") when the sun is behind or beside it. 0 keeps the old flat isotropic term.
      * `powder` (default off) -- the Beer-Powder term: `1-exp(-k*density)`, which darkens thick, flat-lit faces
        so they read as ROUND rather than flat and chalky. Leave off for wispy fog where there's little "thick".
      * `only` (default None) -- a (H,W) boolean mask: render ONLY these rays. Pixels outside it return `background`
        at alpha 0 and cost nothing; pixels inside are bit-identical to the full render. For tiles, previews, and
        escalation passes. KEPT NEGATIVE: coarse-first escalation on top of this does NOT pay against the shipped
        `empty_skip`/`early_term` defaults -- those ARE coarse-first, better applied (15.2x against a residual
        mask's 3.0x on the same scene). See the comment at the mask, and holographic_coarsefirst.
      * `multi_scatter` (default 1 = off) -- an integer > 1 sums that many light-transmittance octaves at halved
        extinction/amplitude each, a cheap approximation of multiple scattering. Real clouds are never truly
        black even deep in shadow (light bounces many times before escaping); a plain single-scatter Beer's law
        UNDERESTIMATES that light by orders of magnitude. 3-4 octaves is enough to lift shadows convincingly
        without the cost of a real multi-bounce simulation."""
    lo = np.asarray(bounds[0], float); hi = np.asarray(bounds[1], float)
    eye, dirs = camera.ray_dirs(width, height)
    D = dirs.reshape(-1, 3)
    O = np.broadcast_to(eye, D.shape)
    inv = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
    t0 = (lo - O) * inv; t1 = (hi - O) * inv
    tmin = np.maximum(np.maximum.reduce(np.minimum(t0, t1), axis=1), 0.0)
    tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)
    hit = tmax > tmin
    if only is not None:
        # Render ONLY these rays; everything else returns `background` at alpha 0, so a caller can merge by mask.
        # Rays inside the mask are BIT-IDENTICAL to the full render (verified), and rays outside cost nothing.
        #
        # WHY THIS EXISTS, and the kept negative that came with it. This was added to run coarse-first escalation
        # (holographic_coarsefirst) on the volume march: render cheap at low `steps`, escalate the high-uncertainty
        # pixels to high `steps`. MEASURED, and it does not pay against the shipped defaults:
        #
        #     base                       uniform 96 steps    coarse-first, top 20% at 96 steps
        #     empty_skip + early_term      22,461 evals       23,137 evals   (1.0x -- coarse pass is pure overhead)
        #     both OFF (a dumb march)     341,184 evals      115,512 evals   (3.0x, at IDENTICAL RMSE 0.00145)
        #
        # `empty_skip` and `early_term` ARE coarse-first, under other names and applied better: don't sample rays in
        # empty macro-cells, and stop sampling a ray once it is opaque. Together they buy 15.2x on this scene, which
        # is 5x what a residual mask buys on the dumb march -- and they leave nothing for the mask to find, because
        # the pixels a gradient flags are the same pixels that hit the volume. Coarse-first buys adaptivity for a
        # method that has NONE. This one already had it. (The signal is real, though: escalating the same number of
        # RANDOM pixels on the dumb march leaves RMSE 0.01273 against 0.00145 -- 8.8x worse.)
        hit = hit & np.asarray(only, bool).reshape(-1)

    col = np.zeros((D.shape[0], 3))
    T = np.ones(D.shape[0])
    if emission_color is None:
        emission_color = np.array([1.0, 0.5, 0.15])
    light = (lights[0] if lights else Light("directional", direction=(-0.4, -0.8, -0.3)))
    Ldir = -light.direction

    occ = None
    if empty_skip:                                            # sample the field once on a coarse macro grid
        gx = np.linspace(lo[0], hi[0], occ_res); gy = np.linspace(lo[1], hi[1], occ_res); gz = np.linspace(lo[2], hi[2], occ_res)
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        gpts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
        gdens = np.asarray(field(gpts), float).reshape(occ_res, occ_res, occ_res)
        # dilate occupancy by one cell so a ray near a boundary isn't wrongly skipped
        occ_raw = gdens > occ_thresh
        occ = occ_raw.copy()
        for ax in range(3):
            occ |= np.roll(occ_raw, 1, axis=ax) | np.roll(occ_raw, -1, axis=ax)
        cell = (occ_res - 1) / np.where((hi - lo) > 0, hi - lo, 1.0)

    seg = (tmax - tmin)
    dt = seg / max(steps - 1, 1)
    n_samples = 0
    for s in np.linspace(0.0, 1.0, steps):
        t = tmin + s * seg
        active = hit & (t <= tmax)
        if early_term:
            active &= (T > term_eps)                          # drop opaque rays
        if not np.any(active):
            break
        pts = O[active] + t[active, None] * D[active]
        if empty_skip:                                        # only sample rays in OCCUPIED macro-cells
            ic = np.clip(((pts - lo) * cell).astype(int), 0, occ_res - 1)
            occupied = occ[ic[:, 0], ic[:, 1], ic[:, 2]]
            idx = np.where(active)[0][occupied]
            sample_pts = pts[occupied]
        else:
            idx = np.where(active)[0]
            sample_pts = pts
        if idx.size == 0:
            continue
        dens = np.clip(np.asarray(field(sample_pts), float), 0.0, None)
        n_samples += idx.size
        a = 1.0 - np.exp(-sigma * dens * dt[idx])
        if mode == "fire":
            c = _blackbody_ramp(dens) * emission_color
            col[idx] += (T[idx, None] * a[:, None]) * c
        elif mode == "density":
            col[idx] += (T[idx, None] * a[:, None]) * dens[:, None]
        else:
            if self_shadow:
                # march the density toward the light from THIS sample to get its transmittance to the sun
                ssig = shadow_sigma if shadow_sigma is not None else sigma
                seg_l = np.linalg.norm(hi - lo) * 0.5           # a half-diagonal is plenty to exit the volume
                dl = seg_l / max(shadow_steps, 1)
                tau = np.zeros(sample_pts.shape[0])
                for ls in range(1, shadow_steps + 1):
                    q = sample_pts + Ldir * (ls * dl)
                    inb = np.all((q >= lo) & (q <= hi), axis=1)
                    if not np.any(inb):
                        break
                    dq = np.zeros(sample_pts.shape[0])
                    dq[inb] = np.clip(np.asarray(field(q[inb]), float), 0.0, None)
                    tau += dq * dl
                if multi_scatter > 1:
                    # cheap energy-conserving multi-scatter approximation: sum progressively-attenuated octaves
                    # of transmittance (each sees half the extinction, at half amplitude) -- real clouds are never
                    # truly black in shadow because light scatters many times before escaping; a single Beer's-law
                    # term underestimates that by orders of magnitude (see the selftest for the measured gap).
                    amp, ext_a, wsum, ms = 1.0, 1.0, 0.0, np.zeros(sample_pts.shape[0])
                    for _ in range(int(multi_scatter)):
                        ms += amp * np.exp(-ssig * tau * ext_a)
                        wsum += amp; amp *= 0.5; ext_a *= 0.55
                    light_T = (ms / wsum)[:, None]
                else:
                    light_T = np.exp(-ssig * tau)[:, None]      # 1 = full sun, ->0 = deep self-shadow
                sun_c = np.asarray(albedo)
                amb_c = np.asarray(ambient)
                if phase_g:
                    # cos(angle) between the VIEW ray and the direction light travels: forward-scattering (the
                    # camera looking roughly the way the light is heading) glows brightest -- the silver lining.
                    cos_theta = D[idx] @ (light.direction / (np.linalg.norm(light.direction) + 1e-9))
                    ph = _henyey_greenstein(cos_theta, phase_g)
                    ph = ph / _henyey_greenstein(np.array([1.0]), phase_g)[0]     # normalise: forward peak ~1
                    phase = (0.5 + 0.7 * np.clip(ph, 0.0, 4.0))[:, None]
                else:
                    phase = 1.0
                if powder:
                    pw = 1.0 - np.exp(-2.2 * dens)
                    direct_mul = (0.25 + 0.75 * pw)[:, None]
                else:
                    direct_mul = 1.0
                lit = sun_c * (0.15 + 0.85 * light_T) * phase * direct_mul + amb_c * (0.35 + 0.30 * (1.0 - light_T))
                col[idx] += (T[idx, None] * a[:, None]) * lit
            else:
                lit = (0.4 + 0.6 * max(0.0, float(np.dot(Ldir, np.array([0, 1.0, 0]))))) * np.asarray(albedo)
                col[idx] += (T[idx, None] * a[:, None]) * lit
        T[idx] *= (1.0 - a)
    alpha = 1.0 - T
    img = col + T[:, None] * np.asarray(background, float)
    out = np.clip(img.reshape(height, width, 3), 0, 1), alpha.reshape(height, width)
    volume_render.last_samples = n_samples                    # for measurement (field evals actually done)
    return out


def _blackbody_ramp(d):
    """A cheap fire ramp: density in ~[0,1] -> a dark-red -> orange -> yellow -> white glow, returned as a (N,3)
    multiplier in [0,1] (brighter and whiter as density rises)."""
    d = np.clip(d, 0.0, 1.0)
    r = np.clip(d * 2.2, 0, 1)
    g = np.clip(d * 1.4 - 0.2, 0, 1)
    b = np.clip(d * 1.1 - 0.6, 0, 1)
    return np.stack([r, g, b], axis=1)


def _henyey_greenstein(cos_theta, g):
    """The Henyey-Greenstein phase function: how strongly light scatters at angle theta between the view ray and
    the light direction, for asymmetry `g` (0 = isotropic, ->1 = sharply forward-peaked, the physically-typical
    value for cloud water droplets is ~0.7-0.9). Used to make a cloud's edge GLOW when the sun is behind/beside it
    (the "silver lining") -- see volume_render's phase_g."""
    g2 = g * g
    return (1.0 - g2) / (4.0 * np.pi * np.power(np.clip(1.0 + g2 - 2.0 * g * cos_theta, 1e-6, None), 1.5))


# ==================================================================================================================
# PNG SCANLINE FILTERING -- the 43x we were leaving on the floor.
#
# A PNG encoder may prefix each scanline with one of five FILTERS (None / Sub / Up / Average / Paeth), which predict
# each byte from its neighbours and store the residual. zlib then compresses the residual, not the pixels. This
# encoder emitted filter 0 (None) on every scanline from the day it was written, which is correct, lossless, and
# leaves almost the whole point of the format unused.
#
# MEASURED, six 96x96 images (the `holographic_pack` demo sets), against Pillow's `optimize=True`:
#
#       image set            filter 0 only     filtered (this)     Pillow      ratio to Pillow
#       smooth gradients        67,245 B           1,857 B         1,560 B     43.1x -> 1.19x
#       flat logo suite          3,553 B           2,079 B         4,467 B      0.80x -> 0.47x
#
# On a gradient the old encoder was 43x larger than Pillow's; filtered, it is within 20% -- and on flat art it now
# BEATS Pillow by 2x. Every render this engine has ever written to disk paid that, and nothing measured it, because
# a PNG that is 43x too big is still a perfectly valid PNG that opens fine.
#
# THE HEURISTIC is libpng's: for each scanline try all five filters and keep the one whose residual bytes have the
# smallest sum of absolute values, read as SIGNED. It is not optimal (that needs a search over the whole image, since
# a line's filter changes the next line's `Up` residual) but it is what every production encoder does, and it is
# deterministic -- ties break toward the lower filter number, so the same pixels always produce the same bytes.
#
# `filters=False` restores the exact legacy byte stream. The DECODED PIXELS are identical either way -- PNG
# filtering is lossless by construction -- which a round-trip test pins with its own un-filter, so the guarantee
# does not rest on a third-party decoder.
# ==================================================================================================================
def _paeth(a, b, c):
    """The PNG Paeth predictor: whichever of left / above / upper-left is closest to a + b - c. Integer only."""
    p = a.astype(np.int16) + b.astype(np.int16) - c.astype(np.int16)
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    out = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
    return out.astype(np.uint8)


def _png_scanlines(rows, filters=True, bpp=3):
    """Serialise (H, W, 3) uint8 rows into PNG's filtered scanline stream: one filter byte then the residual.

    With `filters=False` every line gets filter 0 (the legacy stream, byte-for-byte). Otherwise each line tries all
    five and keeps the smallest sum of |signed residual| -- libpng's heuristic. Deterministic: ties go to the lower
    filter number, so identical pixels always give identical bytes.

    Vectorised across ALL scanlines at once, which is possible because PNG's filters reference the UNFILTERED row
    above, not the filtered one -- so no line depends on another line's choice. (A per-line python loop measured 11x
    the encode cost of the legacy path; this is ~2x, and the 2x is the two zlib passes in `png_bytes`, not this.)"""
    h = rows.shape[0]
    flat = rows.reshape(h, -1)                                # (H, W*3) bytes, in PNG's byte order
    if not filters:
        return b"".join(b"\x00" + flat[y].tobytes() for y in range(h))
    z = np.zeros((h, bpp), np.uint8)
    left = np.concatenate([z, flat[:, :-bpp]], axis=1)        # a: the pixel to the left (zero past the edge)
    up = np.concatenate([np.zeros((1, flat.shape[1]), np.uint8), flat[:-1]], axis=0)      # b: the row above
    upleft = np.concatenate([z, up[:, :-bpp]], axis=1)        # c: above-left
    cands = np.stack([
        flat,                                                                             # 0 None
        (flat - left).astype(np.uint8),                                                    # 1 Sub
        (flat - up).astype(np.uint8),                                                      # 2 Up
        (flat - ((left.astype(np.uint16) + up) // 2).astype(np.uint8)).astype(np.uint8),   # 3 Average
        (flat - _paeth(left, up, upleft)).astype(np.uint8),                                # 4 Paeth
    ])                                                                                     # (5, H, W*3)
    # libpng's minimum-sum-of-absolute-differences, on the residual read as SIGNED bytes.
    cost = np.abs(cands.view(np.int8).astype(np.int32)).sum(axis=2)                        # (5, H)
    pick = np.argmin(cost, axis=0)                            # argmin returns the FIRST minimum -> ties keep the
    out = []                                                  # lower filter number, deterministically
    for y in range(h):
        out.append(bytes([int(pick[y])]) + cands[pick[y], y].tobytes())
    return b"".join(out)


def png_bytes(rgb01, level=6, filters=True):
    """Encode an (H,W,3) image in [0,1] to PNG *bytes* -- a minimal, pure-stdlib encoder (zlib + struct), so the
    render module carries no image-library dependency. 8-bit RGB.

    Returning bytes (not writing a file) is what every web/demo backend actually needs: it can send the result
    straight over HTTP without re-implementing this encoder. `save_png` below just wraps this and writes to disk.

    `level` is the zlib compression level: use 1 for fast streamed preview frames (smaller CPU cost per frame),
    6 for stills (the default -- a good size/speed balance). The decoded PIXELS are identical at every level
    (PNG is lossless); only the size of the compressed byte stream changes.

    `filters=True` (the default) applies PNG's per-scanline filters, choosing one per line by libpng's
    minimum-sum-of-absolute-differences heuristic. This encoder emitted filter 0 (None) everywhere until it was
    measured: on a smooth gradient that made it 43x LARGER than Pillow (67,245 B against 1,560); filtered, it lands
    at 1,857 B -- within 20% -- and on flat vector-style art it now beats Pillow by 2x (2,079 B against 4,467).
    `filters=False` restores the exact legacy byte stream. The DECODED PIXELS are identical either way: PNG
    filtering is lossless by construction, and a round-trip test pins that with its own un-filter rather than
    trusting a third-party decoder."""
    import struct, zlib
    # NOTE: `* 255` truncates rather than rounds. Kept deliberately -- it matches this encoder's long-standing
    # output exactly, so existing images stay bit-identical. (A +0.5 round would shift some pixels by one LSB.)
    a = (np.clip(np.asarray(rgb01, float), 0, 1) * 255).astype(np.uint8)
    h, w = a.shape[:2]
    rows = np.ascontiguousarray(a[:, :, :3])
    # Compress BOTH strategies and keep the smaller. A per-line filter choice can LOSE (measured: on flat vector art
    # it grew the file 3,553 -> 4,903 bytes) because filter 0 leaves the byte stream uniform, and zlib's LZ77 then
    # matches long runs ACROSS scanlines -- matches that mixed filters break. No per-line heuristic can see that;
    # only the compressor can. So we ask it. Ties keep the legacy stream, so identical pixels give identical bytes.
    idat = zlib.compress(_png_scanlines(rows, filters=False), level)
    if filters:
        alt = zlib.compress(_png_scanlines(rows, filters=True), level)
        if len(alt) < len(idat):
            idat = alt

    def chunk(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)        # 8-bit, colour type 2 (RGB)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def save_image(path, rgb01, level=6, filters=True):
    """Save an (H,W,3) [0,1] image, routed by extension: .png uses the stdlib encoder (deterministic,
    zero-dependency, always available); anything else (.jpg, .webp, .bmp, ...) uses Pillow when installed and
    otherwise refuses with the install command -- the same opt-in contract as every accelerator
    (`pip install pillow`, or the `images` extra). PNG stays stdlib ON PURPOSE: the deterministic path must
    never grow a dependency, and lossy formats are a presentation choice, not an engine output."""
    import os

    import numpy as np
    ext = os.path.splitext(str(path))[1].lower()
    if ext in ("", ".png"):
        return save_png(path, rgb01, level=level, filters=filters)
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("saving %r needs Pillow (opt-in, like the other accelerators): "
                           "pip install pillow   (or the `images` extra). PNG needs nothing." % ext)
    arr = np.clip(np.asarray(rgb01, float) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))
    return str(path)


def save_png(path, rgb01, level=6, filters=True):
    """Write an (H,W,3) image in [0,1] to a PNG file. Thin wrapper over `png_bytes` -- see it for the encoder
    details, the `level` argument (1 = fast preview, 6 = still, the default) and `filters`."""
    with open(path, "wb") as f:
        f.write(png_bytes(rgb01, level, filters=filters))


def frame_delta_tiles(prev, curr, tile=32, thresh=1e-3):
    """The pixel-streaming primitive: split two frames into `tile`x`tile` blocks and return only the tiles that
    CHANGED, as a list of (row, col, tile_pixels). A viewport that re-renders after a local edit or a small
    camera nudge can push just these tiles instead of the whole frame -- the rendering analogue of the engine's
    O(change) delta/patch protocol. Returns (changed_tiles, fraction_changed). `prev` may be None (first frame ->
    everything changes)."""
    curr = np.asarray(curr, float)
    h, w = curr.shape[:2]
    ny = (h + tile - 1) // tile
    nx = (w + tile - 1) // tile
    out = []
    total = ny * nx
    for ty in range(ny):
        for tx in range(nx):
            y0, y1 = ty * tile, min((ty + 1) * tile, h)
            x0, x1 = tx * tile, min((tx + 1) * tile, w)
            block = curr[y0:y1, x0:x1]
            if prev is None:
                out.append((ty, tx, block.copy()))
            else:
                pblock = np.asarray(prev, float)[y0:y1, x0:x1]
                if np.max(np.abs(block - pblock)) > thresh:
                    out.append((ty, tx, block.copy()))
    return out, (len(out) / total if total else 0.0)


def fit_camera(mesh, direction=(1.0, 0.75, 1.1), up=(0.0, 1.0, 0.0), fov_deg=50.0, aspect=1.0, margin=1.06):
    """Solve for the camera that FRAMES a mesh: the closest eye along `direction` that keeps every vertex
    inside the frustum, with the target chosen so the subject is CENTRED. Returns a camera dict
    {eye, target, up, fov_deg} ready for render_mesh.

    WHY THIS EXISTS, twice measured: guessing "eye = centre + dir * radius * k" fails in both directions on
    real assets. On a 0.086-unit crab scan, preview_asset's framing left the subject at 4% of the frame; on a
    3.6-unit ladybird scan, a hand-picked k clipped it against all four edges. Bounding-SPHERE framing is the
    usual quick fix and is still wrong here: it ignores the aspect ratio and wastes the frame on a flat, wide
    subject (a crab is 0.086 x 0.031 -- a sphere around it is mostly air).

    The solve is exact rather than iterative: for each vertex, |x| <= (dist - z) * tan(fov/2) * aspect and
    |y| <= (dist - z) * tan(fov/2) must hold, so dist >= max over vertices of (|x|/tx + z, |y|/ty + z). Taking
    the max satisfies every vertex at once.

    CENTRING IS SEPARATE FROM DISTANCE and is the part that is easy to get wrong: the CENTROID is not the
    centre of the projected extents (a scan's vertices bunch where the scanner saw detail), so aiming at the
    centroid leaves the subject off-centre and clipping one edge while the other has slack -- measured on the
    ladybird, which clipped at column 0 while its right edge had 7 px spare. The target is therefore the point
    that centres the PROJECTED bounding box, computed in the camera's own basis.

    KEPT NEGATIVE: this frames the VERTEX cloud. A displacement map, a wide line width, or a fat point sprite
    can still paint outside it. `margin` (default 6%) is the cheap insurance."""
    V = np.asarray(mesh.vertices, float)
    d = np.asarray(direction, float); d = d / (np.linalg.norm(d) or 1.0)
    u = np.asarray(up, float)
    if abs(float(u @ d)) > 0.99:
        u = np.array([1.0, 0.0, 0.0])
    r = np.cross(u, d); r /= (np.linalg.norm(r) or 1.0)
    u2 = np.cross(d, r)
    c0 = V.mean(0)
    P = V - c0
    x = P @ r; y = P @ u2; z = P @ d
    # centre the target on the PROJECTED extents, not the centroid (see docstring)
    cx = 0.5 * (x.min() + x.max()); cy = 0.5 * (y.min() + y.max())
    target = c0 + r * cx + u2 * cy
    x = x - cx; y = y - cy
    ty = np.tan(np.radians(float(fov_deg)) / 2.0)
    tx = ty * float(aspect)
    dist = float(np.maximum(np.abs(x) / max(tx, 1e-9) + z, np.abs(y) / max(ty, 1e-9) + z).max())
    dist = max(dist * float(margin), 1e-6)
    # RETURN the aspect we fit for. The solve above sizes the horizontal half-angle as tx = ty*aspect, so if
    # the renderer then uses a DIFFERENT aspect (its default) the fit is silently wrong -- the subject is
    # framed for one frame shape and drawn in another. This is the same two-paths-disagree-on-aspect bug that
    # bit rasterize_mesh vs the ray path earlier; the camera must CARRY the aspect it was fit for.
    return {"eye": (target + d * dist).tolist(), "target": target.tolist(),
            "up": tuple(float(v) for v in u2), "fov_deg": float(fov_deg), "aspect": float(aspect)}


def _std_views(mesh, margin=1.6):
    """Standard orthographic-ish camera set for a model turnaround: top, front, side, and a 3/4. Cameras pull back
    from the mesh centroid by `margin` * the bbox diagonal so the whole model fits every frame."""
    V = np.asarray(mesh.vertices, float)
    c = V.mean(0)
    diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
    d = diag * float(margin)
    return {
        "top":    (c + np.array([0.0, d, 1e-3]), (0.0, 0.0, -1.0)),
        "bottom": (c + np.array([0.0, -d, 1e-3]), (0.0, 0.0, 1.0)),
        "front":  (c + np.array([d, 0.0, 1e-3]), (0.0, 1.0, 0.0)),
        # left/right: the two z-axis views. "side" is kept as an alias of "right" so every existing caller and
        # every recorded IoU number stays byte-identical -- the new names are ADDITIVE, per the constitution.
        "right":  (c + np.array([0.0, 0.0, d]),  (0.0, 1.0, 0.0)),
        "left":   (c + np.array([0.0, 0.0, -d]), (0.0, 1.0, 0.0)),
        "side":   (c + np.array([0.0, 0.0, d]),  (0.0, 1.0, 0.0)),
        "3q":     (c + np.array([d * 0.7, d * 0.5, d * 0.7]), (0.0, 1.0, 0.0)),
        # "persp" is 3q from the OPPOSITE quadrant, so the pair of perspective views covers all eight octants'
        # worth of outline between them rather than doubling up on one corner.
        "persp":  (c + np.array([-d * 0.7, d * 0.5, -d * 0.7]), (0.0, 1.0, 0.0)),
    }, c, diag


def gauss_area_map(mesh, nth=24, nph=48):
    """The Extended Gaussian Image (Horn 1984): every face's AREA binned by its NORMAL's direction on the
    sphere. A one-array signature of the surface's ORIENTATION FIELD -- translation-invariant, O(F), ~0.14 s
    on 322k faces. Compare two with egi_similarity.

    WHAT IT MEASURES, established by experiment rather than hope -- this instrument came out of the search for
    a one-image silhouette check, and it is NOT one: decimating a dense sphere leaves the silhouette at 0.99
    worst-view IoU while EGI similarity collapses 0.59 -> 0.06, because the normal field coarsens drastically
    even when the outline barely moves. Orientation and outline are ORTHOGONAL quantities. That makes this the
    complement to silhouette_sweep, whose kept negative has always been "blind to interior detail and
    normals": sweep guards the OUTLINE, EGI guards the SURFACE CHARACTER. Use both when "did the optimisation
    destroy the model" means more than the profile.

    KEPT NEGATIVES from the same search, so nobody re-walks it: (1) radial OCCUPANCY over the direction sphere
    ("does a ray from the centre hit the object") carries almost no information -- 94.7% of bins occupied on
    the crab, any star-ish object saturates it. (2) The radial EXTENT map (max distance per direction, the
    'spherical extension function' of Kazhdan et al., SGP 2003) is a MAX-statistic: with density-matched
    surface sampling its whole degradation ladder (0.9868/0.9861/0.9717) sits at its own noise floor (0.9895)
    -- a leg can thin to a remnant that still reaches the same max radius and extent never notices, while
    silhouettes integrate projected AREA and do. (3) The SH rotation-invariance machinery of that literature
    solves alignment for RETRIEVAL; our two meshes share a frame, so direct per-bin comparison is strictly
    more sensitive and none of it is needed."""
    V = np.asarray(mesh.vertices, float)
    F = mesh.__dict__.get("_silhouette_F")
    if F is None:
        try:
            F = np.asarray(mesh.faces, dtype=int)
            if F.ndim != 2 or F.shape[1] != 3:
                raise ValueError
        except (ValueError, TypeError):
            F = np.asarray([f[:3] for f in mesh.faces if len(f) >= 3], int)
        mesh.__dict__["_silhouette_F"] = F
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = 0.5 * np.linalg.norm(n, axis=1)
    nn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    th = np.arccos(np.clip(nn[:, 1], -1, 1))
    ph = np.arctan2(nn[:, 2], nn[:, 0]) % (2 * np.pi)
    ti = np.clip((th / np.pi * nth).astype(int), 0, nth - 1)
    pi_ = np.clip((ph / (2 * np.pi) * nph).astype(int), 0, nph - 1)
    m = np.zeros(nth * nph)
    np.add.at(m, ti * nph + pi_, a)
    return m


def egi_similarity(ref_mesh, mesh, nth=24, nph=48):
    """Orientation-field preservation in [0, 1]: 1 - normalised L1 between the two Extended Gaussian Images.
    1.0 = identical area-weighted normal distributions. Sensitive BY DESIGN: it reads how much the normal
    field coarsened, which is large under any decimation (a grid-5 sphere reads 0.06 against its dense source
    while the silhouette reads 0.99) -- interpret it as 'how much surface character changed', not pass/fail
    against the silhouette guard's 0.95 scale. Returns {"similarity", "ref_area", "area"}."""
    a = gauss_area_map(ref_mesh, nth=nth, nph=nph)
    b = gauss_area_map(mesh, nth=nth, nph=nph)
    s = float(np.abs(a).sum() + np.abs(b).sum())
    simv = 1.0 - float(np.abs(a - b).sum() / s) if s > 0 else 1.0
    return {"similarity": round(simv, 4), "ref_area": float(a.sum()), "area": float(b.sum())}


def silhouette_mask(mesh, direction, up=(0.0, 1.0, 0.0), size=128, frame=None):
    """A binary ORTHOGRAPHIC coverage mask of `mesh` seen along `direction` -- the silhouette and nothing else.

    WHY NOT rasterize_mesh: a silhouette needs no z-buffer, no lighting, no perspective -- only "which pixels
    does the projection cover". The full rasteriser costs 0.5-1.5 s per view on a 322k-face scan (measured);
    this is vectorised end to end with no per-triangle Python loop: project the vertices orthographically,
    SAMPLE every edge at pixel density (one big scatter -- the outline cannot be missed because every triangle
    contributes its whole boundary), then FLOOD-FILL the background inward from the border and invert. Interior
    pixels are whatever the background could not reach.

    Orthographic on purpose, not as a shortcut: Moose's symmetry observation -- the silhouette along +axis and
    -axis is the same outline mirrored -- is TRUE under orthographic projection and FALSE under perspective
    (measured: the spike box read top 0.779 vs bottom 1.000 with the perspective critic, because perspective
    plus occlusion break the symmetry). So a sweep only needs azimuths in [0, pi). PRECISION OF THAT CLAIM,
    measured: the equivalence is exact in the continuous limit but NOT pixel-exact -- independently quantising
    the two mirrored projections of the crab cost 15% IoU (0.847) purely at the outline, because thin limbs
    give the silhouette an enormous perimeter. This is exactly why the SWEEP always renders both meshes under
    the SAME direction and SAME frame: truncation then bites both masks identically and cancels out of the
    comparison. Never compare masks made under different directions.

    `frame=(centre, radius)` pins the projection window; pass the SAME frame for two meshes to make their masks
    comparable (the sweep does this with the reference's frame). Returns a (size, size) bool array.

    KEPT NEGATIVE: flood-fill-from-border cannot represent ENCLOSED holes -- a donut seen face-on masks as a
    disc, so a decimation that fills the hole is invisible to this instrument from that direction (usually
    another azimuth still catches the shape change). The perspective critic (turnaround) does not have this
    blind spot; it is slower and keeps its place for final review renders."""
    V = np.asarray(mesh.vertices, float)
    # MEASURED perf fix: the mask pipeline itself is 0.26 s on 322k faces, but converting the face list cost
    # another ~0.8 s PER CALL -- and a sweep calls this 7 times per mesh. Cache the (F,) array on the mesh
    # object (additive attribute, same pattern as uv_transfer_report); invalidation is a non-issue because
    # meshes here are immutable value objects -- every operation returns a NEW mesh.
    F = mesh.__dict__.get("_silhouette_F")
    if F is None:
        try:
            F = np.asarray(mesh.faces, dtype=int)              # all-triangle fast path: one C conversion
            if F.ndim != 2 or F.shape[1] != 3:
                raise ValueError
        except (ValueError, TypeError):
            F = np.asarray([f[:3] for f in mesh.faces if len(f) >= 3], int)   # ngons: slice per face
        mesh.__dict__["_silhouette_F"] = F
    d = np.asarray(direction, float); d = d / (np.linalg.norm(d) or 1.0)
    u = np.asarray(up, float)
    if abs(float(u @ d)) > 0.99:                                  # up parallel to view: pick another up
        u = np.array([1.0, 0.0, 0.0])
    r = np.cross(u, d); r /= (np.linalg.norm(r) or 1.0)
    u2 = np.cross(d, r)
    P = np.stack([V @ r, V @ u2], axis=1)                         # orthographic: just a basis projection
    if frame is None:
        c = P.mean(0)
        rad = float(np.abs(P - c).max()) * 1.05 or 1.0
    else:
        c3, rad = frame
        c = np.array([float(np.asarray(c3, float) @ r), float(np.asarray(c3, float) @ u2)])
        rad = float(rad)
    Q = (P - c) / (2 * rad) + 0.5                                 # into [0,1]^2
    px = np.clip(Q * (size - 1), 0, size - 1)
    mask = np.zeros((size, size), bool)
    # every edge, sampled at ~pixel density, one scatter. 3F edges; sample counts proportional to length.
    E = np.concatenate([px[F[:, [0, 1]]], px[F[:, [1, 2]]], px[F[:, [2, 0]]]])   # (3F, 2, 2)
    seg = E[:, 1] - E[:, 0]
    n_s = np.minimum(np.ceil(np.linalg.norm(seg, axis=1)).astype(int) + 1, 4 * size)
    tot = int(n_s.sum())
    idx = np.repeat(np.arange(len(E)), n_s)
    # within-edge parameter 0..1, built without a loop: cumulative offsets
    starts = np.concatenate([[0], np.cumsum(n_s)[:-1]])
    tpar = (np.arange(tot) - starts[idx]) / np.maximum(n_s[idx] - 1, 1)
    pts = E[idx, 0] + seg[idx] * tpar[:, None]
    xi = np.clip(pts[:, 0].astype(int), 0, size - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, size - 1)
    mask[yi, xi] = True
    # flood the background from the border: iterative dilation of "outside", blocked by the outline
    outside = np.zeros_like(mask)
    outside[0, :] = ~mask[0, :]; outside[-1, :] = ~mask[-1, :]
    outside[:, 0] |= ~mask[:, 0]; outside[:, -1] |= ~mask[:, -1]
    while True:
        grown = outside.copy()
        grown[1:, :] |= outside[:-1, :]; grown[:-1, :] |= outside[1:, :]
        grown[:, 1:] |= outside[:, :-1]; grown[:, :-1] |= outside[:, 1:]
        grown &= ~mask
        if grown.sum() == outside.sum():
            break
        outside = grown
    return ~outside


def silhouette_sweep(ref_mesh, mesh, n_azimuth=6, size=128, include_top=True, ref_cache=None):
    """Rotate the pair under a fixed orthographic camera and score silhouette IoU at every stop -- Moose's
    turntable, made cheap enough to be a DEFAULT guard. Azimuths cover [0, pi) only: under orthographic
    projection theta and theta+pi give mirror-identical outlines (the symmetry that perspective breaks), so
    n_azimuth=6 already sees the outline from 12 directions' worth of information, plus the top view.

    The REFERENCE's frame (centre + projection radius) is used for both meshes at every angle, so the masks are
    comparable and a shrunken candidate scores honestly low rather than being re-framed to fit. Returns
    {"iou": {view: value}, "worst": float, "worst_view": str, "mean": float, "seconds": float}."""
    import time as _time
    t0 = _time.time()
    RV = np.asarray(ref_mesh.vertices, float)
    c3 = RV.mean(0)
    rad = float(np.linalg.norm(RV - c3, axis=1).max()) * 1.05 or 1.0
    views = []
    for k in range(int(n_azimuth)):
        th = np.pi * k / float(n_azimuth)
        views.append(("az%03d" % int(np.degrees(th)), np.array([np.cos(th), 0.0, np.sin(th)])))
    if include_top:
        views.append(("top", np.array([0.0, 1.0, 0.0])))
    iou = {}
    # ref_cache: a dict the CALLER owns. The reference's masks are identical for every candidate a guard tries,
    # so a walk-back loop passes the same dict each time and only the candidate is re-projected -- the 322k
    # source is masked once per direction for the whole search, not once per step.
    if ref_cache is None:
        ref_cache = {}
    for name, d in views:
        key = (name, int(size))
        a = ref_cache.get(key)
        if a is None:
            a = silhouette_mask(ref_mesh, d, size=size, frame=(c3, rad))
            ref_cache[key] = a
        b = silhouette_mask(mesh, d, size=size, frame=(c3, rad))
        un = float(np.logical_or(a, b).sum())
        iou[name] = (float(np.logical_and(a, b).sum()) / un) if un > 0 else 1.0
    worst_view = min(iou, key=iou.get)
    return {"iou": {k: round(v, 4) for k, v in iou.items()}, "worst": float(iou[worst_view]),
            "worst_view": worst_view, "mean": float(np.mean(list(iou.values()))),
            "seconds": round(_time.time() - t0, 3)}


def turnaround(mesh, ref_mesh=None, views=("top", "front", "side", "3q"), width=360, height=360,
               base_color=(0.7, 0.72, 0.62), ref_color=(0.55, 0.68, 0.75), background=(0.05, 0.06, 0.08),
               margin=1.6):
    """TURNAROUND: render `mesh` from the standard modelling views (top/front/side/3q) in ONE call and, if a
    `ref_mesh` is given, score how well the silhouettes MATCH per view -- the loop that (by hand) caught the mantis's
    slurped legs and the box-model's proportions. Returns a dict:
      sheet      : an (H, W*, 3) image, the views concatenated (mesh alone, or mesh|ref stacked per view if ref given)
      views      : the view names used
      iou        : {view: silhouette IoU vs ref} (only if ref_mesh) -- intersection-over-union of the two foreground
                   masks under the SAME camera, in [0,1]; 1.0 = identical silhouette
      mean_iou   : the average over views (the single 'does it look right' number)
    Both meshes are framed by `mesh`'s bbox so the comparison is fair (same camera). Deterministic.

    WHY A NUMBER: "looks right" was vibes; silhouette IoU turns the critic into something an agent can OPTIMISE
    against (raise the score by fixing the view where it's lowest). KEPT NEGATIVE: silhouette IoU is a 2-D outline
    match -- it does not see internal topology or depth, so a filled-in wrong-interior model can still score high;
    pair it with mesh_report for the topology side."""
    cams, _c, _diag = _std_views(mesh, margin=margin)
    out_rows = []
    iou = {}
    for name in views:
        eye, up = cams[name]
        cam = Camera(eye=eye.tolist(), target=_c.tolist(), up=up, fov_deg=35.0, near=1e-3, far=max(5.0, _diag * 6))
        img = rasterize_mesh(mesh, cam, width=width, height=height, base_color=base_color, background=background)
        if ref_mesh is not None:
            rimg = rasterize_mesh(ref_mesh, cam, width=width, height=height, base_color=ref_color,
                                  background=background)
            bg = np.asarray(background, float)
            ma = np.abs(img - bg).sum(-1) > 1e-3               # foreground mask of the model
            mr = np.abs(rimg - bg).sum(-1) > 1e-3              # foreground mask of the reference
            inter = float(np.logical_and(ma, mr).sum())
            union = float(np.logical_or(ma, mr).sum())
            iou[name] = (inter / union) if union > 0 else 1.0
            out_rows.append(np.concatenate([img, rimg], axis=1))
        else:
            out_rows.append(img)
    sheet = np.concatenate(out_rows, axis=0)
    result = {"sheet": sheet, "views": list(views)}
    if ref_mesh is not None:
        result["iou"] = iou
        result["mean_iou"] = float(np.mean([iou[v] for v in views]))
    return result


def _selftest():
    import time
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    # --- rasterise a lit sphere: the side facing the light is brighter than the far side ---
    def sphere(P): P = np.asarray(P, float); return np.linalg.norm(P, axis=1) - 0.7
    v, ax = sample_field(sphere, (np.array([-1., -1, -1]), np.array([1., 1, 1])), 32)
    M = marching_tetrahedra_vec(v, ax)
    cam = Camera(eye=(0, 0, 3), target=(0, 0, 0), fov_deg=45, aspect=1.0)
    t = time.time()
    img = rasterize_mesh(M, cam, 128, 128, lights=[Light("directional", direction=(-1, -1, -1))],
                         base_color=(0.8, 0.5, 0.3))
    rt = time.time() - t
    lit = img.sum(2)
    assert lit.max() > 0.3 and (lit > 0.02).sum() > 200       # something got drawn and shaded
    # --- volume-render a blob: opaque centre, transparent edges ---
    def blob(P): P = np.asarray(P, float); return np.clip(1.0 - np.linalg.norm(P, axis=1) / 0.6, 0, 1)
    _, alpha = volume_render(blob, cam, (np.array([-1., -1, -1]), np.array([1., 1, 1])), 96, 96, steps=64)
    assert alpha.max() > 0.5 and alpha.min() < 0.05           # dense centre, empty corners
    fire, _ = volume_render(blob, cam, (np.array([-1., -1, -1]), np.array([1., 1, 1])), 64, 64, steps=48, mode="fire")
    assert fire[..., 0].max() > fire[..., 2].max()            # fire is redder than it is blue
    # save_image routing trap: PNG must ALWAYS work (stdlib); a lossy extension either saves (Pillow present)
    # or refuses with the install command in the message -- never a bare ImportError, never a silent no-op.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        _img = np.zeros((4, 4, 3)); _img[1:3, 1:3] = 0.7
        save_image(_td + "/t.png", _img)
        assert open(_td + "/t.png", "rb").read(4) == b"\x89PNG"
        try:
            _out = save_image(_td + "/t.jpg", _img)
            assert str(_out).endswith(".jpg")
        except RuntimeError as _e:
            assert "pip install pillow" in str(_e), "the refusal must name the install command"

    # TEXTURED raster path: texture=None BYTE-IDENTICAL to the flat path; a checkerboard on a UV'd box renders
    # with real texture contrast (std over a face strip), not flat shading -- the trap for backlog A4.
    from holographic.mesh_and_geometry.holographic_mesh import box as _tbox
    _tb = _tbox(); _tcam = Camera(eye=(2.2, 1.6, 2.4), target=(0, 0, 0), fov_deg=40)
    _f1 = rasterize_mesh(_tb, _tcam, width=64, height=64)
    assert np.array_equal(_f1, rasterize_mesh(_tb, _tcam, width=64, height=64, texture=None))
    _uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]] * 2, float)
    _chk = np.stack([np.indices((8, 8)).sum(0) % 2] * 3, axis=-1).astype(float)
    _ft = rasterize_mesh(_tb, _tcam, width=64, height=64, texture=_chk, uvs=_uv)
    assert float(np.abs(_ft - _f1).mean()) > 0.01 and float(_ft[26:38, 20:46].std()) > 0.08

    # SMOOTH shading (default-off): smooth=None byte-identical; on a coarse faceted mesh (a level-1 CC box, 24 big
    # quads) smooth=True visibly differs from flat per-face shading (Gouraud gradient across each face).
    from holographic.mesh_and_geometry.holographic_meshsubdiv import catmull_clark as _cc_sm
    _coarse = _cc_sm(_tbox(), 1)
    _sf = rasterize_mesh(_coarse, _tcam, width=64, height=64)
    assert np.array_equal(_sf, rasterize_mesh(_coarse, _tcam, width=64, height=64, smooth=False))
    _ss = rasterize_mesh(_coarse, _tcam, width=64, height=64, smooth=True)
    assert float(np.abs(_ss - _sf).mean()) > 0.003          # smooth changes the shading

    # turnaround + silhouette IoU: a mesh vs ITSELF scores 1.0 every view; a half-size copy scores much lower --
    # the "does it look right" critic that caught the mantis's slurped legs, now a number an agent can optimise.
    _ta = turnaround(_tbox(), ref_mesh=_tbox(), width=48, height=48)
    assert abs(_ta["mean_iou"] - 1.0) < 1e-9
    _half = Mesh(np.asarray(_tbox().vertices, float) * 0.5, [tuple(f) for f in _tbox().faces])
    _ta2 = turnaround(_tbox(), ref_mesh=_half, width=48, height=48)
    assert _ta2["mean_iou"] < 0.6                            # a wrong-scale silhouette scores clearly lower

    print(f"render selftest ok: rasterised {M.n_faces} faces in {rt*1000:.0f} ms @128x128; "
          f"volume alpha max {alpha.max():.2f}; fire glows red")


def _selftest_silhouette():
    """Pin the fast silhouette instrument: (1) identity sweeps to 1.0 in every direction; (2) a mesh that LOST
    its spike is caught by SOME direction of the turntable (the whole point of sweeping: degradation is local
    and one bad azimuth must be enough); (3) masks are deterministic; (4) the ref frame is shared, so a
    half-size copy scores badly instead of being re-framed to fit."""
    from holographic.mesh_and_geometry.holographic_mesh import box, Mesh
    from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
    b = triangulate_ngons(box())
    V = np.asarray(b.vertices, float).tolist(); F = [tuple(f) for f in b.faces]
    tip = len(V); V.append([0.5, 3.0, 0.5]); F += [(2, 3, tip), (3, 7, tip), (7, 6, tip), (6, 2, tip)]
    spiky = Mesh(np.array(V), F)
    r_id = silhouette_sweep(spiky, spiky, n_azimuth=6, size=96)
    assert all(v == 1.0 for v in r_id["iou"].values()), r_id["iou"]
    r_lost = silhouette_sweep(spiky, triangulate_ngons(box()), n_azimuth=6, size=96)
    assert r_lost["worst"] < 0.85, "losing the spike must be caught: %s" % r_lost["iou"]
    a = silhouette_mask(spiky, [1.0, 0.0, 0.3], size=96)
    b2 = silhouette_mask(spiky, [1.0, 0.0, 0.3], size=96)
    assert np.array_equal(a, b2)
    half = Mesh(np.array(V) * 0.5, F)
    r_half = silhouette_sweep(spiky, half, n_azimuth=4, size=96)
    assert r_half["worst"] < 0.5, "a shrunken copy must score badly under the shared frame"
    # EGI: identity exact; TRANSLATION-invariant (normals do not move); and the complement demonstrated --
    # a coarsened sphere keeps its silhouette but not its orientation field.
    e_id = egi_similarity(spiky, spiky)
    assert e_id["similarity"] == 1.0
    moved = Mesh(np.array(V) + np.array([5.0, -2.0, 1.0]), F)
    assert egi_similarity(spiky, moved)["similarity"] == 1.0, "EGI must ignore translation"
    # fit_camera: nothing clips, at any aspect, on an OFF-CENTRE mesh (the case that breaks centroid framing)
    off = Mesh(np.array(V) + np.array([4.0, 1.0, -3.0]), F)
    for (w_, h_) in ((320, 320), (480, 270), (270, 480)):
        cam = Camera(**fit_camera(off, direction=(1.0, 0.6, 0.9), aspect=w_ / float(h_)))
        img = rasterize_mesh(off, cam, width=w_, height=h_, background=(0.0, 0.0, 0.0))
        fgm = np.asarray(img, float).sum(2) > 0.02
        assert fgm.any(), "subject vanished at %dx%d" % (w_, h_)
        ys_, xs_ = np.where(fgm)
        assert xs_.min() > 0 and xs_.max() < w_ - 1 and ys_.min() > 0 and ys_.max() < h_ - 1, \
            "fit_camera clipped an off-centre mesh at %dx%d" % (w_, h_)
    # and the projection uses the FRAME's aspect: a sphere is a circle at 16:9, not an ellipse
    from holographic.mesh_and_geometry.holographic_meshtools import _uv_sphere_fixture
    sp = _uv_sphere_fixture(32)
    cam = Camera(eye=(0, 0, 6.0), target=(0, 0, 0))
    im2 = np.asarray(rasterize_mesh(sp, cam, width=640, height=360, background=(0.0, 0.0, 0.0)), float)
    fg2 = im2.sum(2) > 0.02
    ys2, xs2 = np.where(fg2)
    ratio = (xs2.max() - xs2.min() + 1) / float(ys2.max() - ys2.min() + 1)
    assert abs(ratio - 1.0) < 0.06, "sphere rasterised as an ellipse (w/h %.3f): aspect regression" % ratio

    e_lost = egi_similarity(spiky, triangulate_ngons(box()))
    assert e_lost["similarity"] < 0.9, "losing the spike's faces must move the orientation field"
    print("silhouette selftest OK (identity 1.0 all views; lost spike caught at worst %.3f; deterministic; "
          "half-size scores %.3f under the shared frame; EGI identity 1.0, translation-invariant, spike loss "
          "%.3f)" % (r_lost["worst"], r_half["worst"], e_lost["similarity"]))


if __name__ == "__main__":
    _selftest(); _selftest_silhouette()
