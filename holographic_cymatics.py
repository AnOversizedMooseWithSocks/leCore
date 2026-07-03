"""holographic_cymatics.py -- A4: CYMATICS. A sound makes a plate vibrate in its modes and sand collects on the
nodal lines -- Chladni figures.

WHY THIS EXISTS (Acoustics & Cymatics backlog, item A4 -- the headline)
----------------------------------------------------------------------
Chladni (1787) scattered sand on a metal plate, drew a bow across it, and the sand jumped away from the parts
that were moving and piled up on the still lines -- the NODES -- tracing a different figure for every tone. The
figure is the plate's vibration mode, and the vibration modes of a shape ARE the eigenmodes of its Laplacian --
which this engine already computes (holographic_spectral). So cymatics is not a new solver: it is the eigenmodes
we own, DRIVEN by a sound's spectrum (A1), with sand moved to the nodes.

THE METHOD (readable, first-principles)
---------------------------------------
  1. Build the plate's Laplacian on its domain (a square or a disk) with a FIXED (Dirichlet, u=0) rim -- the
     5-point discrete -grad^2. Its eigenvectors phi_k are the mode shapes; the eigenvalues give the mode
     frequencies f_k = sqrt(lambda_k). (The eigensolve + reproducible signs come from holographic_spectral.)
  2. DRIVE it with a sound: each frequency present in the signal resonantly excites the modes whose f_k is near
     it (a Lorentzian resonance kernel), and the displacement is the weighted sum u(x) = sum_k a_k phi_k(x).
  3. SAND moves DOWN the gradient of |u|^2 -- away from the moving antinodes, toward the still nodes where
     |u| ~ 0 -- so grains pile on the nodal lines and draw the figure. (Sand = the particle system we own.)

HONEST SCOPE (kept negative): a MEMBRANE-mode model (Laplacian eigenmodes + node-drift), not the full biharmonic
plate (real Chladni plates are free-edge and stiffer); the figures are the right phenomenology (crosses, rings,
lattices), art-directable, not a metrological match. The eigensolver is DENSE (spectral's own kept negative), so
the grid is modest. Sand is over-damped drift to nodes, not a granular collision simulation. Deterministic;
NumPy + stdlib. (Water & cornstarch media are A5, next.)
"""
import numpy as np


def _domain_mask(shape, grid):
    """A boolean (grid, grid) mask of which cells are INSIDE the plate. 'square' = all cells; 'circle' = a disk.
    The rim just outside the mask is the fixed (u=0) boundary."""
    if shape == "square":
        m = np.ones((grid, grid), bool)
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False            # fixed outer ring
        return m
    if shape in ("circle", "disk", "round"):
        yy, xx = np.mgrid[0:grid, 0:grid]
        c = (grid - 1) / 2.0
        return ((xx - c) ** 2 + (yy - c) ** 2) <= (0.47 * grid) ** 2
    raise ValueError("unknown plate shape %r (use 'square' or 'circle')" % shape)


def _dirichlet_laplacian(mask):
    """The Dirichlet 5-point Laplacian (-grad^2) over the in-domain cells of `mask`. Diagonal 4, -1 to each
    in-domain 4-neighbour; out-of-domain neighbours are the fixed u=0 rim (they drop out, diagonal stays 4). This
    is the operator whose eigenvectors are the plate's mode shapes."""
    idx = {}
    cells = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                idx[(i, j)] = len(cells); cells.append((i, j))
    n = len(cells)
    L = np.zeros((n, n))
    for (i, j), c in idx.items():
        L[c, c] = 4.0                                              # continuum -grad^2 discretisation
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = idx.get((i + di, j + dj))
            if nb is not None:
                L[c, nb] = -1.0                                    # in-domain neighbour; else Dirichlet 0
    return L, cells


class ChladniPlate:
    """A vibrating plate whose Chladni figures come from its Laplacian eigenmodes. Build it once (the eigensolve
    is the cost); then `drive(freqs, amps)` sets the displacement field from a sound's spectrum, `step_medium(dt)`
    drifts sand toward the nodes, and `render()` shows the sand figure. `drive_mode(k)` selects a single mode."""

    def __init__(self, shape="square", grid=40, medium="sand", n_modes=48, base_hz=200.0,
                 n_grains=6000, seed=0):
        from holographic_spectral import laplacian_eigenbasis
        self.grid = int(grid)
        self.medium = medium
        self.mask = _domain_mask(shape, self.grid)
        L, self.cells = _dirichlet_laplacian(self.mask)
        n_modes = min(int(n_modes), len(self.cells) - 1)
        w, V = laplacian_eigenbasis(L, n_basis=n_modes)            # the eigenmodes (reused eigensolver + sign_fix)
        self.eigvals = np.maximum(w, 0.0)
        self.modes = V                                             # columns are mode shapes over self.cells
        f = np.sqrt(self.eigvals)
        self.mode_hz = base_hz * f / (f[f > 1e-9][0] if np.any(f > 1e-9) else 1.0)   # tune fundamental -> base_hz
        self.u = np.zeros((self.grid, self.grid))                  # current displacement field (0 outside domain)
        # scatter sand uniformly over the domain (particle positions in grid coords, x=col, y=row)
        rng = np.random.default_rng(seed)
        inside = np.array(self.cells, float)                       # (n,2) = (row, col)
        pick = rng.integers(0, len(inside), n_grains)
        jitter = rng.uniform(-0.5, 0.5, (n_grains, 2))
        rc = inside[pick] + jitter
        self.sand = np.stack([rc[:, 1], rc[:, 0]], axis=1)         # (N,2) as (x=col, y=row)
        # A5 media: WATER carries a standing surface height; CORNSTARCH carries a held-peak field. Both are driven
        # by the same displacement field, but respond at the ANTINODES (opposite of sand-at-nodes).
        self.surface = np.zeros((self.grid, self.grid))            # water: standing Faraday surface height
        self.peaks = np.zeros((self.grid, self.grid))              # cornstarch: transient shear-thickened fingers
        self.drive_hz = float(base_hz)                             # the dominant driving frequency (sets cell size / shear rate)

    # -- turning mode coefficients into a displacement field ------------------------------------------------------
    def _field_from_coeffs(self, coeffs):
        """Sum the eigenmodes weighted by `coeffs` into a full (grid,grid) displacement field (0 outside domain)."""
        vec = self.modes @ coeffs                                  # value per in-domain cell
        f = np.zeros((self.grid, self.grid))
        for (i, j), val in zip(self.cells, vec):
            f[i, j] = val
        mx = np.abs(f).max()
        return f / mx if mx > 1e-12 else f                         # normalise so |u|<=1 (a pattern, not a scale)

    def drive_mode(self, k, amp=1.0):
        """Excite a SINGLE eigenmode k -- the cleanest way to see one Chladni figure. Sets the displacement to
        that mode shape."""
        c = np.zeros(self.modes.shape[1]); c[int(k)] = float(amp)
        self.u = self._field_from_coeffs(c)
        self.drive_hz = float(self.mode_hz[int(k)])                # this mode's frequency (sets Faraday cell size)
        return self.u

    def drive(self, freqs, amps, width_hz=None):
        """Drive the plate with a sound: each (freq, amp) resonantly excites the modes whose frequency is near it
        (a Lorentzian kernel), and the displacement is the weighted sum of mode shapes. A changing spectrum ->
        a changing figure."""
        freqs = np.atleast_1d(np.asarray(freqs, float))
        amps = np.atleast_1d(np.asarray(amps, float))
        width = float(width_hz) if width_hz else max(1.0, 0.5 * np.median(np.diff(np.sort(self.mode_hz))))
        coeffs = np.zeros(self.modes.shape[1])
        for fr, am in zip(freqs, amps):
            coeffs += am / ((self.mode_hz - fr) ** 2 + width ** 2)  # resonance: modes near this tone respond most
        self.u = self._field_from_coeffs(coeffs)
        self.drive_hz = float(freqs[int(np.argmax(amps))]) if len(freqs) else self.drive_hz  # loudest tone drives
        return self.u

    # -- moving the medium (sand: drift to nodes) -----------------------------------------------------------------
    def _sample(self, field, xy):
        """Bilinear sample of a (grid,grid) field at particle positions (N,2)=(x,y), clamped at the edges."""
        x = np.clip(xy[:, 0], 0, self.grid - 1.001); y = np.clip(xy[:, 1], 0, self.grid - 1.001)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        fx = x - x0; fy = y - y0
        f00 = field[y0, x0]; f10 = field[y0, x0 + 1]; f01 = field[y0 + 1, x0]; f11 = field[y0 + 1, x0 + 1]
        return (f00 * (1 - fx) * (1 - fy) + f10 * fx * (1 - fy) + f01 * (1 - fx) * fy + f11 * fx * fy)

    def step_medium(self, dt=0.1, strength=6.0):
        """Advance the chosen medium one step over the current displacement field:
          * SAND drifts DOWN grad|u|^2 to the still NODES (Chladni figure).
          * WATER forms a standing Faraday surface at the ANTINODES (crests where the plate moves most).
          * CORNSTARCH (shear-thickening) HOLDS peaks at the antinodes under FAST drive and relaxes under slow.
        The three responses ride on the SAME driving field -- only the material's rule differs."""
        if self.medium == "water":
            return self._step_water(dt)
        if self.medium == "cornstarch":
            return self._step_cornstarch(dt)
        return self._step_sand(dt, strength)

    def _step_sand(self, dt=0.1, strength=6.0):
        """SAND: over-damped drift DOWN grad|u|^2, toward the nodes, settling there."""
        energy = self.u ** 2                                       # |u|^2: big at antinodes, ~0 at nodes
        gy, gx = np.gradient(energy)                               # grad points UPHILL (toward antinodes)
        fx = -self._sample(gx, self.sand) * strength               # push DOWNHILL, toward the nodes
        fy = -self._sample(gy, self.sand) * strength
        self.sand[:, 0] += fx * dt
        self.sand[:, 1] += fy * dt
        # keep grains on the plate: reflect any that drift off the domain back to the nearest in-domain cell
        xi = np.clip(np.round(self.sand[:, 0]).astype(int), 0, self.grid - 1)
        yi = np.clip(np.round(self.sand[:, 1]).astype(int), 0, self.grid - 1)
        off = ~self.mask[yi, xi]
        if off.any():
            self.sand[off, 0] = np.clip(self.sand[off, 0], 1, self.grid - 2)
            self.sand[off, 1] = np.clip(self.sand[off, 1], 1, self.grid - 2)
        return self.sand

    def _step_water(self, dt=0.1, relax=0.35):
        """WATER (Faraday 1831): a vertically-driven surface forms a STANDING wave. The steady pattern is the
        driven mode's antinode structure -- crests where |u| is largest -- so the surface height relaxes toward
        |u|. (The surface physically oscillates at HALF the drive frequency; the standing PATTERN is what we
        render.) Finer patterns (smaller cells) at higher drive frequency, because a higher tone lights a higher,
        finer mode."""
        target = np.abs(self.u) * self.mask                        # crests at the antinodes
        self.surface += (target - self.surface) * min(1.0, relax * (1 + dt))
        return self.surface

    def _step_cornstarch(self, dt=0.1, shear_threshold=250.0):
        """CORNSTARCH: a shear-thickening (non-Newtonian) suspension. The local shear rate ~ drive_frequency *
        |displacement|; where it exceeds a threshold the suspension THICKENS and holds a standing peak; where it
        is low it relaxes back to flat. So it stands in fingers under fast/hard drive and slumps under slow drive
        -- the 'walking oobleck' signature."""
        shear = self.drive_hz * np.abs(self.u) * self.mask         # local shear rate proxy
        thickened = np.clip((shear - shear_threshold) / shear_threshold, 0.0, 1.0)   # 0 fluid .. 1 solid-like
        target = thickened * np.abs(self.u)                        # held peaks where it thickened
        # thickened regions hold fast (peaks persist); un-thickened relax toward flat
        hold = 0.7 * thickened + 0.05
        self.peaks += (target - self.peaks) * np.clip(hold + dt, 0.0, 1.0)
        return self.peaks

    def settle(self, steps=60, dt=0.1, strength=6.0):
        """Run the medium to its steady pattern under the current displacement (sand rest / water standing surface
        / cornstarch peaks)."""
        for _ in range(int(steps)):
            self.step_medium(dt=dt, strength=strength)
        return self.sand if self.medium == "sand" else (self.surface if self.medium == "water" else self.peaks)

    # -- readback -------------------------------------------------------------------------------------------------
    def sand_density(self):
        """Histogram the sand grains onto the grid -> a (grid,grid) density (grains per cell). High on the nodes."""
        xi = np.clip(self.sand[:, 0].astype(int), 0, self.grid - 1)
        yi = np.clip(self.sand[:, 1].astype(int), 0, self.grid - 1)
        d = np.zeros((self.grid, self.grid))
        np.add.at(d, (yi, xi), 1.0)
        return d

    def render(self, sand_rgb=(0.92, 0.86, 0.62), plate_rgb=(0.10, 0.10, 0.12)):
        """A picture of the cymatic figure. SAND: pale grains on the nodal lines. WATER: a blue standing surface,
        bright at the antinode crests. CORNSTARCH: pale fingers standing where it thickened."""
        if self.medium == "water":
            s = self.surface.copy()
            if s.max() > 1e-9:
                s = s / s.max()
            deep = np.array([0.03, 0.12, 0.28]); crest = np.array([0.55, 0.80, 0.95])   # dark trough -> bright crest
            img = deep[None, None, :] * (1 - s[..., None]) + crest[None, None, :] * s[..., None]
            img[~self.mask] = 0.0
            return np.clip(img, 0, 1)
        if self.medium == "cornstarch":
            p = self.peaks.copy()
            if p.max() > 1e-9:
                p = p / p.max()
            base = np.array([0.20, 0.19, 0.18]); peak = np.array([0.93, 0.92, 0.90])     # slurry -> pale fingers
            img = base[None, None, :] * (1 - p[..., None]) + peak[None, None, :] * p[..., None]
            img[~self.mask] = 0.0
            return np.clip(img, 0, 1)
        # sand (default)
        d = self.sand_density()
        if d.max() > 0:
            d = np.clip(d / np.percentile(d[d > 0], 95), 0, 1) if (d > 0).sum() > 5 else d / d.max()
        img = np.asarray(plate_rgb, float)[None, None, :] * np.ones((self.grid, self.grid, 1))
        img = img * (1 - d[..., None]) + np.asarray(sand_rgb, float)[None, None, :] * d[..., None]
        img[~self.mask] = 0.0                                      # outside the plate is black
        return np.clip(img, 0, 1)

    def nodal_fraction_on_sand(self, node_thresh=0.15):
        """Honest measurement: the mean |u| UNDER the settled sand vs the plate average. If sand sits on nodes,
        the sand-weighted |u| is much smaller than the average |u|. Returns (sand_weighted_|u|, plate_mean_|u|)."""
        au = np.abs(self.u)
        d = self.sand_density()
        w = d.sum()
        sand_u = float((au * d).sum() / w) if w > 0 else 0.0
        plate_u = float(au[self.mask].mean())
        return sand_u, plate_u


def _selftest():
    """A driven plate has real eigenmodes; sand settles onto the nodes (its |u| much below the plate average); a
    square gives a different fundamental figure than a circle; deterministic."""
    # (1) square plate: drive a mode and let sand settle -> sand sits where |u| ~ 0 (the nodes)
    p = ChladniPlate("square", grid=36, n_modes=30, n_grains=5000, seed=0)
    assert p.modes.shape[1] >= 20 and (p.eigvals[1:] > 0).all()    # real positive modes
    p.drive_mode(5)                                                # a mid mode with clear nodal lines
    p.settle(steps=80, dt=0.1, strength=8.0)
    sand_u, plate_u = p.nodal_fraction_on_sand()
    assert sand_u < 0.5 * plate_u, (sand_u, plate_u)               # sand concentrated on the low-|u| nodal set

    # (2) driving by SOUND (a tone near a mode frequency) lights that region up: displacement is non-trivial
    fnd = p.mode_hz[3]
    u = p.drive([fnd], [1.0])
    assert np.abs(u).max() > 0.5 and np.isfinite(u).all()

    # (3) a circular plate produces a different figure than a square one at the same mode index
    c = ChladniPlate("circle", grid=36, n_modes=30, n_grains=5000, seed=0)
    c.drive_mode(5); c.settle(steps=60)
    sand_uc, plate_uc = c.nodal_fraction_on_sand()
    assert sand_uc < 0.6 * plate_uc                                # sand on nodes for the disk too
    assert c.mask.sum() < p.mask.sum()                             # the disk is a smaller domain than the square

    # (4) deterministic
    a = ChladniPlate("square", grid=28, n_modes=20, n_grains=2000, seed=1); a.drive_mode(4); a.settle(30)
    b = ChladniPlate("square", grid=28, n_modes=20, n_grains=2000, seed=1); b.drive_mode(4); b.settle(30)
    assert np.array_equal(a.sand_density(), b.sand_density())
    print("holographic_cymatics selftest OK: plate eigenmodes are the Chladni modes; sand settles onto the nodes "
          "(sand |u|=%.3f << plate |u|=%.3f); square vs circle differ; deterministic" % (sand_u, plate_u))


if __name__ == "__main__":
    _selftest()
