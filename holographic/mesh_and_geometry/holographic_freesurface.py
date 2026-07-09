"""holographic_freesurface.py -- the OVERTURNING FREE SURFACE (Physics & FX backlog, item #8, rung 4).

This is the top rung of the adaptive ladder, and the honest reason it exists: every method below it represents the
water as a HEIGHT FIELD h(x) -- one height per horizontal position. The spectral ocean, the wave packets, shallow
water: all single-valued. But a BREAKING wave's crest curls FORWARD over its own base, so above one x there are
suddenly TWO surfaces (the falling jet, and the wave face beneath it). A height field literally cannot express
that. Neither can a grid velocity field (advection is single-valued too). Overturning needs PARTICLES.

So when the AdaptiveSolver's trigger flags a tile as breaking, we hand its surface to THIS solver: seed particles
along the crest, carrying the wave's orbital velocity (at a steep crest the tip is thrown FORWARD faster than the
wave itself moves), then let them fly under gravity. The crest tip arcs forward and PLUNGES past the wave face --
the surface folds over -- and then the particles land and the jet collapses back into a height the spectral field
can resume from. Seeded from the field, returned to the field: the cheap read/write handoff the backlog describes.

HONEST SCOPE (kept loud, the VFX-vs-physics line): this is a BALLISTIC-particle model of the plunging crest -- the
standard visual-effects approach -- not a full incompressible Navier-Stokes solver. Once the jet is airborne it IS
in free fall, so ballistics is physically right for the throw and the plunge; what it does NOT model is the
pressure, incompressibility, and turbulent air entrainment AFTER the jet hits the water (the whitewater). So it
captures the OVERTURNING TOPOLOGY -- the multi-valued surface that is the whole reason for this rung -- and leaves
the post-impact mixing as the documented gap. A production solver (FLIP/PIC/SPH/MPM with a pressure projection) is
the research-heavy extension; the AdaptiveSolver localizes WHEN this runs, not how hard the full version is to
write. Deterministic; NumPy + stdlib only.
"""
import numpy as np


class FreeSurface:
    """A set of water-surface particles under gravity -- the solver that can OVERTURN. Each particle has a position
    and a velocity; `advance` flies them ballistically and rests them on the ground. The surface is whatever the
    particles trace out, so it can fold over itself (multi-valued), which is exactly what a height field can't do."""

    def __init__(self, g=9.81, ground=0.0, damping=0.3):
        self.g = float(g)
        self.ground = float(ground)
        self.damping = float(damping)                          # energy lost when a particle lands (splash -> rest)
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.x0 = np.zeros((0,))                               # each particle's STARTING x (to detect folding)

    def seed(self, pos, vel):
        """Add particles at `pos` (N,2) with velocities `vel` (N,2)."""
        pos = np.atleast_2d(np.asarray(pos, float))
        vel = np.atleast_2d(np.asarray(vel, float))
        self.pos = np.vstack([self.pos, pos])
        self.vel = np.vstack([self.vel, vel])
        self.x0 = np.concatenate([self.x0, pos[:, 0]])
        return self

    def advance(self, dt, steps=1):
        """Fly every particle ballistically: gravity pulls y down, position integrates velocity. A particle that
        reaches the ground rests there (its vertical motion stops, horizontal motion is damped) -- the jet landing
        and the water coming to rest."""
        for _ in range(int(steps)):
            self.vel[:, 1] -= self.g * dt
            self.pos = self.pos + self.vel * dt
            landed = self.pos[:, 1] <= self.ground
            self.pos[landed, 1] = self.ground
            self.vel[landed, 1] = 0.0
            self.vel[landed, 0] *= (1.0 - self.damping)        # friction/splash on landing
        return self

    def is_overturning(self):
        """True if the surface has FOLDED: some particle that STARTED behind another has ended up AHEAD of it. When
        a crest tip overtakes the wave face beneath it, the surface becomes multi-valued -- the definition of a
        breaking (overturning) wave, and the thing a height field cannot represent."""
        # sort by starting x; a fold means the current x is NOT in the same order
        order = np.argsort(self.x0, kind="stable")
        cur_x = self.pos[order, 0]
        return bool(np.any(np.diff(cur_x) < -1e-9))            # a later-starting particle is now behind an earlier

    def is_multivalued(self, tol=0.5):
        """True if two particles sit at (nearly) the same x but a real gap in height -- an explicit check that the
        surface has two sheets at one location (the falling jet above the wave face)."""
        idx = np.argsort(self.pos[:, 0], kind="stable")
        xs = self.pos[idx, 0]; ys = self.pos[idx, 1]
        close = np.abs(np.diff(xs)) < tol
        gap = np.abs(np.diff(ys)) > tol
        return bool(np.any(close & gap))

    def settle_height(self, x_bins):
        """Collapse the particles back to a HEIGHT profile (the max particle height in each x-bin) -- how the jet,
        once landed, is handed BACK to the spectral field as a single-valued surface it can resume. `x_bins` is the
        bin edges; returns the height per bin (0 where empty)."""
        h = np.zeros(len(x_bins) - 1)
        which = np.digitize(self.pos[:, 0], x_bins) - 1
        for b in range(len(h)):
            in_bin = self.pos[which == b, 1]
            if len(in_bin):
                h[b] = in_bin.max()
        return h


def seed_breaking_crest(fs, x_start=0.0, length=10.0, n=40, crest_speed=6.0, phase_speed=3.0, height=4.0):
    """Set up a PLUNGING BREAKER on a FreeSurface: a crest whose TIP is thrown forward faster than the wave itself
    travels (crest_speed > phase_speed is the breaking condition -- when orbital velocity beats phase velocity, the
    crest outruns its base). The tip gets a strong forward+up velocity; the base moves slowly. Returns fs."""
    xs = np.linspace(x_start, x_start + length, n)
    # a crest profile: a bump peaking near the front, tallest at the tip
    prof = height * np.exp(-((xs - (x_start + 0.7 * length)) ** 2) / (2 * (0.18 * length) ** 2))
    pos = np.column_stack([xs, prof])
    # velocity: the tip (tall crest) is thrown forward and up (orbital velocity at breaking); the base barely moves
    frac = prof / (prof.max() + 1e-9)                          # 1 at the tip, ~0 at the base
    vx = phase_speed + (crest_speed - phase_speed) * frac      # tip moves at crest_speed, base at phase_speed
    vy = 0.6 * crest_speed * frac                              # the tip is also thrown upward
    fs.seed(pos, np.column_stack([vx, vy]))
    return fs


def free_surface_step(region, dt, g=9.81):
    """The AdaptiveSolver's REAL free_surface stepper (replacing the honest placeholder): take a steep height tile,
    seed particles from its surface with a forward velocity proportional to the local slope (a steep face throws
    water forward), fly them one step under gravity, and settle back to a height. A minimal but genuine breaking
    solve on the tile -- the handoff, closing the loop between item #5 (dispatch) and item #8 (the grid rung)."""
    region = np.asarray(region, float)
    H, W = region.shape
    fs = FreeSurface(g=g, ground=float(region.min()))
    ys, xs = np.mgrid[0:H, 0:W]
    gy, gx = np.gradient(region)
    slope = np.sqrt(gx * gx + gy * gy)
    pos = np.column_stack([xs.ravel().astype(float), region.ravel()])
    # a steep cell throws water forward (in +x) and up, proportional to its slope
    vel = np.column_stack([slope.ravel(), 0.5 * slope.ravel()])
    fs.seed(pos, vel)
    fs.advance(dt, steps=2)
    # settle back to a height field of the same shape (max particle height per x-column, broadcast over rows)
    x_bins = np.arange(W + 1) - 0.5
    col_h = fs.settle_height(x_bins)
    out = np.where(col_h[None, :] > 0, col_h[None, :], region)  # keep the original where no particle landed
    return np.broadcast_to(out, (H, W)).copy()


def _selftest():
    """A steep crest (orbital velocity > phase velocity) OVERTURNS -- the surface folds and becomes multi-valued;
    a gentle crest does NOT; the jet settles back to a height for the handoff; free_surface_step runs on a tile;
    deterministic."""
    # (1) a BREAKING crest overturns: the tip is thrown forward and plunges past the base
    fs = FreeSurface(g=9.81, ground=0.0)
    seed_breaking_crest(fs, length=10.0, n=40, crest_speed=8.0, phase_speed=3.0, height=4.0)
    assert not fs.is_overturning()                             # not folded yet, at t=0
    fs.advance(0.05, steps=20)                                 # catch it MID-PLUNGE (the jet is airborne over the face)
    assert fs.is_overturning(), "a steep breaking crest must fold over (overturn)"
    assert fs.is_multivalued(), "the folded surface is multi-valued -- a height field can't represent it"

    # (2) a GENTLE crest (tip barely faster than the wave) does NOT overturn
    calm = FreeSurface(g=9.81, ground=0.0)
    seed_breaking_crest(calm, length=10.0, n=40, crest_speed=3.2, phase_speed=3.0, height=1.0)
    calm.advance(0.05, steps=40)
    assert not calm.is_overturning(), "a gentle wave stays single-valued (does not break)"

    # (3) settle_height collapses the folded (multi-valued) surface back to a single-valued HEIGHT -- how the jet
    # is handed BACK to the spectral field. At the fold moment the top sheet gives a real height profile.
    x_bins = np.linspace(0, 20, 41)
    h = fs.settle_height(x_bins)
    assert h.shape == (40,) and h.max() > 0                    # single-valued, with height -> the field can resume
    assert np.all(np.isfinite(h))

    # (4) free_surface_step runs a genuine breaking solve on a steep tile (the AdaptiveSolver handoff)
    tile = np.zeros((8, 8))
    tile[:, 4:] = np.linspace(0, 6, 4)[None, :]                # a steep face in the tile
    out = free_surface_step(tile, dt=0.1)
    assert out.shape == (8, 8) and np.isfinite(out).all()

    # (5) deterministic
    a = FreeSurface(); seed_breaking_crest(a, crest_speed=8.0); a.advance(0.05, steps=20)
    b = FreeSurface(); seed_breaking_crest(b, crest_speed=8.0); b.advance(0.05, steps=20)
    assert np.array_equal(a.pos, b.pos)

    n_folded = int(np.sum(np.diff(fs.pos[np.argsort(fs.x0, kind="stable"), 0]) < 0))
    print("holographic_freesurface selftest OK: a steep crest (orbital speed 8 > phase speed 3) OVERTURNS -- the "
          "tip plunges forward and the surface folds into a multi-valued sheet (%d order inversions) that no height "
          "field can hold; a gentle wave stays single-valued; the jet settles back to a height for the handoff; "
          "free_surface_step runs on a tile; deterministic" % n_folded)


if __name__ == "__main__":
    _selftest()
