"""holographic_mixture.py -- THE MATTER MODEL: one advected-field model with dials (fluids/matter backlog item 2).

Smoke, dye/milk mixing, salt fingering, and oil-and-water are not four simulators -- they are ONE advected-field
matter model with three dials: number of components, buoyancy, and (later) a double-well tension. This module builds
the multi-channel core -- a Mixture of component fields riding ONE shared incompressible flow -- and the single
`matter_step` that advances it. It DELEGATES the physics to the fluid faculties already wired on UnifiedMind
(advect, diffuse, buoyancy_force, fluid_step); it does not add a second solver.

Holographically the mixture is a multi-channel hypervector: adding a substance is adding a ROLE (a channel);
appearance/density is a fraction-weighted BUNDLE (the blend); and per-channel diffusion at DIFFERENT rates is free --
which is exactly what makes salt fingering possible once buoyancy is on (item 3's `drift`). Surface tension for the
immiscible case (item 4's `double_well`) plugs into the same per-channel loop.

This item is the miscible core: N channels advect on the shared flow and diffuse (each at its own rate) and blend.
The `drift` and `double_well` terms are wired as OPTIONAL hooks here (off by default) so items 3 and 4 slot in with
no rewrite.

KEPT NEGATIVE: miscible mixing is native and cheap (channels blend); the immiscible SHARP interface is item 4's
diffuse-interface trade (a finite interface width). Fractions are volume fractions clamped to a partition -- if the
channels would over-fill a cell (sum > 1) they are scaled back (you cannot have more than 100% occupancy).
"""
import numpy as np

from holographic_fields import advect, diffuse, buoyancy_force, fluid_step


class Component:
    """A substance in the mixture: its density (relative to the solvent) and its diffusivity (how fast it spreads).
    Different diffusivities per component are the whole point -- that difference is what makes double-diffusive
    effects like salt fingering emerge once buoyancy is on."""

    def __init__(self, density=1.0, diffusivity=0.01):
        self.density = float(density)
        self.diffusivity = float(diffusivity)


class Mixture:
    """A set of component concentration fields (channels) sharing one flow, plus a temperature field and the dials.
    density() is the fraction-weighted blend (the bundle) that drives buoyancy; the channels are volume fractions."""

    def __init__(self, shape, solvent_density=1.0, buoyancy=1.0, tension=0.0):
        self.shape = tuple(shape)
        self.channels = {}                                   # name -> concentration field (volume fraction 0..1)
        self.comp = {}                                       # name -> Component
        self.temperature = np.zeros(self.shape)
        self.solvent_density = float(solvent_density)        # the clear fluid that fills the rest of each cell
        self.buoyancy = float(buoyancy)                      # DIAL 2: buoyancy strength
        self.tension = float(tension)                        # DIAL 3: double-well tension (item 4; 0 = miscible)

    def add(self, name, field, density=1.0, diffusivity=0.01):
        """Add a component channel with an initial concentration field."""
        self.channels[name] = np.asarray(field, float).copy()
        self.comp[name] = Component(density, diffusivity)
        return self

    def density(self):
        """The fraction-weighted density blend: solvent everywhere, plus each channel's excess density times its
        fraction. This IS the bundle -- a superposition of component densities weighted by how much of each is here.
        Where a cell is pure solvent (all fractions 0) it reads the solvent density; where it is full of one
        component it reads that component's density."""
        rho = np.full(self.shape, self.solvent_density)
        for name, phi in self.channels.items():
            rho = rho + phi * (self.comp[name].density - self.solvent_density)
        return rho

    def total_fraction(self):
        """Total occupied fraction per cell (sum over channels)."""
        t = np.zeros(self.shape)
        for phi in self.channels.values():
            t = t + phi
        return t

    def renormalise(self):
        """Keep the channels a valid partition: clamp each to >=0, and where they would over-fill a cell (sum > 1)
        scale them back so the sum is 1 (mass can't exceed 100% occupancy). Under-full cells keep clear solvent."""
        for name in self.channels:
            self.channels[name] = np.clip(self.channels[name], 0.0, None)
        total = self.total_fraction()
        over = total > 1.0
        if over.any():
            for name in self.channels:
                self.channels[name] = np.where(over, self.channels[name] / (total + 1e-12), self.channels[name])


def _double_well(phi):
    """The double-well restoring term (item 4) for a [0,1] VOLUME FRACTION: W'(phi) = phi*(1-phi)*(1-2phi), whose
    two wells sit at phi=0 and phi=1. In the Allen-Cahn/Cahn-Hilliard diffuse-interface picture it pushes phi<0.5
    toward 0 and phi>0.5 toward 1, so a blended (miscible) interface SHARPENS into two immiscible phases -- oil and
    water. Balanced against a smoothing diffusion term, the interface settles to a finite width. Same double-well
    shape as the reaction-diffusion automaton's reaction step. Unused when tension == 0 (the miscible corner)."""
    return phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def _drift(phi, rho, comp_density, dt, strength=0.0):
    """Settling / separation drift (item 3): a component heavier than the LOCAL blend sinks relative to it (a lighter
    one rises). Off (returns 0) until a strength is given. Implemented as an extra VERTICAL advection of the channel
    by a settling velocity proportional to its density excess over the blend -- so a heavy dye pools at the floor and
    an immiscible-to-be phase separates. Returns the CHANGE to phi (so the caller adds it), keeping matter_step's
    `phi + drift` shape. The sign is set so heavier-than-blend -> downward (-y), matching the buoyancy convention
    (+y = up) the smoke presets measured."""
    if strength == 0.0:
        return 0.0
    v_settle = -strength * (comp_density - rho)                  # heavier than blend -> negative vy (sinks)
    return advect(phi, np.zeros_like(phi), v_settle, dt) - phi   # the settling delta


def matter_step(mix, vx, vy, dt=0.1, drift_strength=0.0):
    """Advance the mixture ONE step. The physics is delegated to the wired fluid routines:

      1. blend the density (the bundle) and get the buoyancy force from temperature + density  [buoyancy_force]
      2. ONE shared incompressible solve advances the velocity                                  [fluid_step]
      3. each channel rides that flow (advect) and spreads at its OWN rate (diffuse)             [advect/diffuse]
         -- with optional double-well tension (item 4) and drift (item 3) hooks
      4. renormalise so the fractions stay a valid partition (mass conservation)

    Returns the updated (vx, vy). The mixture is mutated in place."""
    rho = mix.density()
    # buoyancy has TWO drivers (Boussinesq): temperature (hot rises, via beta) AND density (heavy sinks, via alpha).
    # Passing alpha is what lets the blended mixture density drive convection -- the mixing/settling/fingering engine.
    fx, fy = buoyancy_force(mix.temperature, density=rho, alpha=mix.buoyancy, beta=mix.buoyancy)   # DIAL 2
    vx, vy, _ = fluid_step(vx, vy, rho, dt=dt, fx=fx, fy=fy)                      # ONE shared Stam solve
    for name in list(mix.channels):
        phi = mix.channels[name]
        c = mix.comp[name]
        phi = advect(phi, vx, vy, dt)                                            # ride the shared flow
        phi = diffuse(phi, c.diffusivity * dt)                                   # DIAL 1: per-channel rate
        if mix.tension:                                                          # DIAL 3: sharpen into an interface (item 4)
            # Allen-Cahn sharpening: the double-well pulls each cell toward the two phases (0 or 1) while a small
            # diffusion sets the interface WIDTH. `tension` scales how strongly they separate -- high tension -> a
            # sharp immiscible interface (oil & water), 0 -> the fields stay blended (miscible).
            lap = diffuse(phi, 1.0) - phi                                        # one diffusion step ~ a smoothing
            phi = phi + dt * mix.tension * (-_double_well(phi) + 0.5 * lap)
        drift = _drift(phi, rho, c.density, dt, strength=drift_strength)         # item 3 (off by default)
        mix.channels[name] = phi + drift
    mix.temperature = advect(mix.temperature, vx, vy, dt)                        # heat rides the flow too
    mix.renormalise()
    return vx, vy


def _blob(shape, cy, cx, r, amp=1.0):
    """A soft circular concentration blob (for tests/demos)."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * r * r))


def _selftest():
    """Two dye channels in a shared flow: they advect and diffuse (spread), blend into the density field correctly,
    conserve mass under diffusion, and stay a valid partition after renormalise."""
    shape = (48, 48)
    mix = Mixture(shape, solvent_density=1.0, buoyancy=0.0)
    mix.add("red", _blob(shape, 24, 16, 4.0), density=1.2, diffusivity=0.05)
    mix.add("blue", _blob(shape, 24, 32, 4.0), density=0.8, diffusivity=0.05)

    # a gentle rightward shear so the blobs move and meet
    vx = np.full(shape, 0.5)
    vy = np.zeros(shape)

    red0 = mix.channels["red"].copy()
    spread0 = _spatial_spread(mix.channels["red"])
    mass0 = mix.channels["red"].sum()

    for _ in range(20):
        vx, vy = matter_step(mix, vx, vy, dt=0.1)

    # the dye advected (moved) and diffused (spread out)
    assert not np.allclose(mix.channels["red"], red0)                     # it changed
    assert _spatial_spread(mix.channels["red"]) > spread0                 # it spread (diffusion widened it)
    # diffusion + advection roughly conserve mass (periodic domain; renormalise only caps overflow)
    assert abs(mix.channels["red"].sum() - mass0) / mass0 < 0.15
    # the density blend reads the components: where red is strong, density leans to 1.2; where blue, toward 0.8
    rho = mix.density()
    r_cell = np.unravel_index(np.argmax(mix.channels["red"]), shape)
    b_cell = np.unravel_index(np.argmax(mix.channels["blue"]), shape)
    assert rho[r_cell] > 1.0 > rho[b_cell]                                # heavy-dye cell denser than light-dye cell
    # valid partition: every channel in [0,1] and total occupancy <= 1 (a bit of slack for float)
    for phi in mix.channels.values():
        assert phi.min() >= -1e-9 and phi.max() <= 1.0 + 1e-6
    assert mix.total_fraction().max() <= 1.0 + 1e-6

    print("holographic_mixture selftest OK: two dye channels advect + diffuse on ONE shared flow (red spread %.2f->%.2f), "
          "mass conserved within 15%%, the density blend reads heavy-dye cells (rho %.2f) denser than light-dye cells "
          "(rho %.2f), and the fractions stay a valid partition -- all delegating to the wired advect/diffuse/"
          "fluid_step, no new solver" % (spread0, _spatial_spread(mix.channels["red"]), rho[r_cell], rho[b_cell]))


def _spatial_spread(field):
    """The spatial standard deviation of a (non-negative) field about its center of mass -- how spread out it is."""
    f = np.clip(field, 0, None)
    total = f.sum()
    if total < 1e-9:
        return 0.0
    ys, xs = np.mgrid[0:field.shape[0], 0:field.shape[1]]
    my = (ys * f).sum() / total
    mx = (xs * f).sum() / total
    var = (((ys - my) ** 2 + (xs - mx) ** 2) * f).sum() / total
    return float(np.sqrt(var))


if __name__ == "__main__":
    _selftest()
