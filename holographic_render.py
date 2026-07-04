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

    def projection_matrix(self):
        """Perspective projection (OpenGL-style, maps the frustum to the [-1,1] cube)."""
        t = np.tan(np.radians(self.fov_deg) / 2.0)
        n, fa = self.near, self.far
        P = np.zeros((4, 4))
        P[0, 0] = 1.0 / (self.aspect * t)
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
                   background=(0.05, 0.06, 0.08), ambient=0.15, vectorized=True):
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
    reference. Both give the same image."""
    from holographic_mesh import Mesh
    if not all(len(f) == 3 for f in mesh.faces):
        mesh = Mesh(mesh.vertices, [tuple(t) for t in mesh.triangulate()])
    if lights is None:
        lights = [Light("directional", direction=(-0.4, -0.8, -0.5), intensity=1.0)]
    base_color = np.asarray(base_color[:3], float)

    V = mesh.vertices
    F = np.asarray(mesh.faces, dtype=int)
    MVP = camera.projection_matrix() @ camera.view_matrix()
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
        ndl = np.clip(np.sum(fn * L, axis=1), 0.0, None)
        shade += ndl[:, None] * lt.intensity * lt.color
    face_col = np.clip((ambient + shade) * base_color, 0.0, 1.0)

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
    flat[pix[win]] = face_col[ti[win]]
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
                  occ_thresh=1e-3, term_eps=2e-3):
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
    Set either False to fall back to the plain uniform march."""
    lo = np.asarray(bounds[0], float); hi = np.asarray(bounds[1], float)
    eye, dirs = camera.ray_dirs(width, height)
    D = dirs.reshape(-1, 3)
    O = np.broadcast_to(eye, D.shape)
    inv = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
    t0 = (lo - O) * inv; t1 = (hi - O) * inv
    tmin = np.maximum(np.maximum.reduce(np.minimum(t0, t1), axis=1), 0.0)
    tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)
    hit = tmax > tmin

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


def png_bytes(rgb01, level=6):
    """Encode an (H,W,3) image in [0,1] to PNG *bytes* -- a minimal, pure-stdlib encoder (zlib + struct), so the
    render module carries no image-library dependency. 8-bit RGB.

    Returning bytes (not writing a file) is what every web/demo backend actually needs: it can send the result
    straight over HTTP without re-implementing this encoder. `save_png` below just wraps this and writes to disk.

    `level` is the zlib compression level: use 1 for fast streamed preview frames (smaller CPU cost per frame),
    6 for stills (the default -- a good size/speed balance). The decoded PIXELS are identical at every level
    (PNG is lossless); only the size of the compressed byte stream changes."""
    import struct, zlib
    # NOTE: `* 255` truncates rather than rounds. Kept deliberately -- it matches this encoder's long-standing
    # output exactly, so existing images stay bit-identical. (A +0.5 round would shift some pixels by one LSB.)
    a = (np.clip(np.asarray(rgb01, float), 0, 1) * 255).astype(np.uint8)
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y, :, :3].tobytes() for y in range(h))   # filter byte 0 per scanline

    def chunk(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)        # 8-bit, colour type 2 (RGB)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, level)) + chunk(b"IEND", b"")


def save_png(path, rgb01, level=6):
    """Write an (H,W,3) image in [0,1] to a PNG file. Thin wrapper over `png_bytes` -- see it for the encoder
    details and the `level` argument (1 = fast preview, 6 = still, the default)."""
    with open(path, "wb") as f:
        f.write(png_bytes(rgb01, level))


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


def _selftest():
    import time
    from holographic_mesh import Mesh
    from holographic_meshbridge import sample_field, marching_tetrahedra_vec
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
    print(f"render selftest ok: rasterised {M.n_faces} faces in {rt*1000:.0f} ms @128x128; "
          f"volume alpha max {alpha.max():.2f}; fire glows red")


if __name__ == "__main__":
    _selftest()
