"""
holographic_automaton.py -- Turing patterns in hypervector space.

A demoscene-flavoured reuse of the VSA substrate: instead of a few vectors doing
cognition, give every cell of a 2D grid its own hypervector and let them interact
locally. Each step, a cell mixes a short-range neighbour average (local
alignment / activation) against a long-range annular average (lateral inhibition)
plus a binding with a fixed rule vector, then squashes pointwise and renormalises
onto the sphere. That activator/inhibitor competition is exactly the Turing
mechanism -- here the "morphogen" is a D-dimensional vector rather than a scalar
concentration -- and from random noise it self-organises into spots, stripes, and
labyrinths. Cells are coloured by projecting their vector onto three fixed axes.

Nothing here is told what pattern to make; the morphology falls out of the ratio
of short- to long-range coupling. Pure numpy, fully vectorised with batched FFTs.

Needs: numpy.
"""

import numpy as np


class HyperCA:
    """A grid of hypervectors evolving by short-range activation vs long-range
    inhibition -- a vector-valued reaction-diffusion system."""

    def __init__(self, size=110, dim=48, w_short=1.5, w_long=0.6, radius=4,
                 w_bind=0.3, gain=1.5, seed=0):
        self.size, self.dim = size, dim
        self.w_short, self.w_long, self.w_bind, self.gain = w_short, w_long, w_bind, gain
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((size, size, dim))
        self.grid = g / np.linalg.norm(g, axis=2, keepdims=True)
        rule = rng.standard_normal(dim); rule /= np.linalg.norm(rule)
        self._rule_f = np.fft.rfft(rule)
        self._axes = rng.standard_normal((3, dim))
        self._axes /= np.linalg.norm(self._axes, axis=1, keepdims=True)
        # long-range (inhibitory) annulus, width ~2 so the average is smooth
        self._ann = [(dx, dy) for dx in range(-radius, radius + 1)
                     for dy in range(-radius, radius + 1)
                     if radius - 1 <= np.hypot(dx, dy) <= radius + 1]

    def _short(self):
        # 8 neighbours + self: a smoothing activator kernel. Including the
        # diagonals and the cell itself damps the grid-scale checkerboard mode
        # that a bare 4-neighbour average is prone to.
        g = self.grid
        s = np.zeros_like(g)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                s += np.roll(np.roll(g, dx, 0), dy, 1)
        return s / 9.0

    def _long(self):
        s = np.zeros_like(self.grid)
        for dx, dy in self._ann:
            s += np.roll(np.roll(self.grid, dx, 0), dy, 1)
        return s / len(self._ann)

    def step(self):
        bound = np.fft.irfft(np.fft.rfft(self.grid, axis=2) * self._rule_f,
                             n=self.dim, axis=2)
        mixed = (self.w_short * self._short() - self.w_long * self._long()
                 + self.w_bind * bound)
        g = np.tanh(self.gain * mixed)
        self.grid = g / np.linalg.norm(g, axis=2, keepdims=True)
        return self

    def image(self):
        """Colour each cell by projecting its vector onto three fixed axes."""
        c = np.tensordot(self.grid, self._axes, axes=([2], [1]))
        c -= c.min(); c /= (c.max() + 1e-9)
        return c

    def evolve(self, steps, snapshots=()):
        snaps = {}
        for t in range(steps + 1):
            if t in snapshots:
                snaps[t] = self.image()
            if t < steps:
                self.step()
        return snaps


# ---------------------------------------------------------------------------
# DEMO / GALLERY
# ---------------------------------------------------------------------------
def demo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. one rule, watched from noise to order
    ca = HyperCA(size=110, dim=48, seed=3)
    when = [1, 6, 22, 65]
    snaps = ca.evolve(65, snapshots=when)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for i, t in enumerate(when):
        ax[i].imshow(snaps[t]); ax[i].set_title(f"step {t}"); ax[i].axis("off")
    fig.suptitle("Self-organisation in hypervector space: random 48-D noise -> Turing labyrinth", y=1.04)
    fig.tight_layout(); fig.savefig("ca_evolution.png", dpi=100, bbox_inches="tight"); plt.close(fig)

    # 2. a morphology zoo -- same mechanism, different coupling ratios
    zoo = {
        "labyrinth": dict(w_short=1.5, w_long=0.6),
        "worms":     dict(w_short=2.0, w_long=0.3),
        "maze":      dict(w_short=1.0, w_long=0.6),
        "stripes":   dict(w_short=1.0, w_long=1.0),
    }
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for i, (name, kw) in enumerate(zoo.items()):
        ca = HyperCA(size=110, dim=48, seed=3, radius=4, gain=1.6, w_bind=0.3, **kw)
        ca.evolve(65)
        ax[i].imshow(ca.image()); ax[i].set_title(name); ax[i].axis("off")
    fig.suptitle("Same vector reaction-diffusion, different short/long-range coupling", y=1.04)
    fig.tight_layout(); fig.savefig("ca_zoo.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("wrote ca_evolution.png, ca_zoo.png")


def _selftest():
    """A NUMERIC REGRESSION TRAP (NCA backlog B5), not a smoke test.

    Before this existed, `python3 -m holographic.misc.holographic_automaton` ran `demo()`, which drew matplotlib
    figures and wrote `ca_evolution.png` / `ca_zoo.png` into the repo ROOT. The build loop's step-5b -- "run the
    module self-test" -- therefore verified nothing at all, and littered. `demo()` stays, as a separate entry point.

    The invariants asserted here are properties of the DYNAMICS, not of one lucky array: every cell stays on the
    unit sphere (the step renormalises), the step is deterministic under a fixed seed, a different seed gives a
    different grid, and the step actually MOVES the grid. The pinned statistics are quoted to 1e-10 -- a change to
    the kernel, the gain, or the RNG draw order will trip them."""
    import numpy as _np

    ca = HyperCA(dim=32, size=24, seed=0)
    before = ca.grid.copy()
    ca.step()

    # 1. every cell is a UNIT hypervector: the step renormalises, so nothing can blow up or decay away
    assert _np.abs(_np.linalg.norm(ca.grid, axis=2) - 1.0).max() < 1e-12

    # 2. the step actually did something (a no-op step would pass every other assertion here)
    moved = float(_np.abs(ca.grid - before).max())
    assert moved > 0.1, moved

    # 3. DETERMINISM: same seed, same grid, bit for bit
    other = HyperCA(dim=32, size=24, seed=0)
    other.step()
    assert _np.array_equal(ca.grid, other.grid)

    # 4. ... and the seed is load-bearing
    assert not _np.allclose(HyperCA(dim=32, size=24, seed=1).grid, HyperCA(dim=32, size=24, seed=0).grid)

    # 5. THE REGRESSION TRAP: pattern statistics after 5 evolution steps, pinned to 1e-10
    ev = HyperCA(dim=32, size=24, seed=0)
    ev.evolve(5)
    assert abs(float(ev.grid.mean()) - 0.000534848538) < 1e-10, float(ev.grid.mean())
    assert abs(float(ev.grid.std()) - 0.176775886187) < 1e-10, float(ev.grid.std())
    assert abs(float(ev.image().mean()) - 0.410467731226) < 1e-10, float(ev.image().mean())

    # 6. the readout is a well-formed image
    img = ev.image()
    assert img.shape == (24, 24, 3) and img.min() >= 0.0 and img.max() <= 1.0

    print("OK: holographic_automaton self-test passed (every cell stays on the unit sphere to 2.2e-16; one step "
          "moves the grid by %.3f and is bit-identical under a fixed seed while a different seed diverges; and the "
          "5-step pattern statistics are pinned to 1e-10 -- mean %.12f, std %.12f, image mean %.12f)"
          % (moved, float(ev.grid.mean()), float(ev.grid.std()), float(ev.image().mean())))


if __name__ == "__main__":
    _selftest()
