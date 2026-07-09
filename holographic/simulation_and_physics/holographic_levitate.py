"""holographic_levitate.py -- A7: ACOUSTIC LEVITATION. A standing sound wave holds beads in mid-air.

WHY THIS EXISTS (Acoustics & Cymatics backlog, item A7 -- the "sound moves objects" showpiece)
---------------------------------------------------------------------------------------------
Point two loudspeakers at each other and you get a STANDING wave -- a pattern of pressure that does not travel,
with still PRESSURE NODES every half wavelength. A small bead placed in it feels a steady push toward the nearest
node (the acoustic radiation force) strong enough to hold it against gravity: ultrasonic levitation. This is the
last acoustics item, and it rides on the standing field the wave solver (A3) provides, the particle system we
already push with forces, and the gravity we already have.

THE PHYSICS (Gor'kov 1962, readable)
------------------------------------
In a standing wave p(y) = A*cos(k*y) (k = 2*pi/lambda), the time-averaged force on a small particle is the
gradient of the GOR'KOV POTENTIAL:

    U = V * [ f1 * <p^2>/(2*rho0*c0^2)  -  f2 * (3/4)*rho0*<v^2> ]

where <p^2> and <v^2> are the mean-square pressure and velocity of the wave, V the particle volume, and f1, f2
the monopole/dipole scattering factors set by the density/compressibility CONTRAST between the particle and the
medium (f1 = 1 - kappa_p/kappa0, f2 = 2(rho_p-rho0)/(2*rho_p+rho0)). For a dense particle in air both are ~1, and
U is smallest at the PRESSURE NODES (where cos(k*y)=0) -- so the force F = -grad U pushes beads to the nodes and
pins them there, spaced lambda/2 apart. Turn the field off and gravity wins: they fall.

HONEST SCOPE (kept negative): the Gor'kov approximation holds for particles MUCH SMALLER than the wavelength
(the Rayleigh limit) -- big objects, and acoustic streaming, are out of scope; a 1-D vertical standing wave
(the classic levitator geometry); the medium is inviscid. Deterministic; NumPy + stdlib. Reuses
holographic_fields.ParticleSystem for the integration.
"""
import numpy as np

# medium defaults: air at room temperature
_RHO_AIR = 1.2       # kg/m^3
_C_AIR = 343.0       # m/s


def _scattering_factors(rho_p, c_p, rho0=_RHO_AIR, c0=_C_AIR):
    """The Gor'kov monopole (f1) and dipole (f2) factors from the particle/medium contrast. For a dense, stiff
    particle in air both approach 1, which is what makes it collect at the pressure nodes."""
    kappa0 = 1.0 / (rho0 * c0 ** 2)                                # medium compressibility
    kappa_p = 1.0 / (rho_p * c_p ** 2)                             # particle compressibility
    f1 = 1.0 - kappa_p / kappa0
    f2 = 2.0 * (rho_p - rho0) / (2.0 * rho_p + rho0)
    return f1, f2


def gorkov_potential(y, wavelength, amplitude, radius=1e-3, rho_p=25.0, c_p=2500.0,
                     rho0=_RHO_AIR, c0=_C_AIR):
    """The Gor'kov potential U(y) of a small particle in a vertical standing wave p=A*cos(k*y). Its MINIMA are the
    stable trapping points. For a dense bead these sit at the pressure nodes (cos(k*y)=0)."""
    k = 2.0 * np.pi / wavelength
    V = (4.0 / 3.0) * np.pi * radius ** 3
    f1, f2 = _scattering_factors(rho_p, c_p, rho0, c0)
    p2 = 0.5 * amplitude ** 2 * np.cos(k * y) ** 2                 # <p^2> in a standing wave
    v2 = 0.5 * (amplitude / (rho0 * c0)) ** 2 * np.sin(k * y) ** 2  # <v^2> (velocity antinode = pressure node)
    return V * (f1 * p2 / (2.0 * rho0 * c0 ** 2) - f2 * (3.0 / 4.0) * rho0 * v2)


def gorkov_force_y(y, wavelength, amplitude, radius=1e-3, rho_p=25.0, c_p=2500.0,
                   rho0=_RHO_AIR, c0=_C_AIR, h=1e-6):
    """The vertical acoustic radiation force F = -dU/dy on a particle at height y (central difference of the
    Gor'kov potential -- readable, no hand-differentiated trig to get wrong). Points toward the nearest node."""
    kw = dict(wavelength=wavelength, amplitude=amplitude, radius=radius, rho_p=rho_p, c_p=c_p, rho0=rho0, c0=c0)
    up = gorkov_potential(y + h, **kw)
    dn = gorkov_potential(y - h, **kw)
    return -(up - dn) / (2.0 * h)


def pressure_nodes(wavelength, height):
    """The heights of the pressure nodes in a column of the given height -- where beads get trapped, spaced
    lambda/2 apart (cos(k*y)=0 -> y = lambda/4 + n*lambda/2)."""
    lam = wavelength
    nodes = []
    y = lam / 4.0
    while y <= height:
        nodes.append(y); y += lam / 2.0
    return np.array(nodes)


class LevitationChamber:
    """Beads in a vertical standing wave, feeling the acoustic radiation force plus gravity. With the field ON, a
    dense bead is trapped at a pressure node against gravity; with it OFF, gravity wins and it falls. Reuses the
    engine's ParticleSystem for the integration (beads spread in x for viewing; the physics acts along y)."""

    def __init__(self, height=0.10, wavelength=0.0086, amplitude=4000.0, n_beads=40,
                 gravity=9.81, bead_radius=1e-3, bead_density=25.0, mass_scale=3e-7, seed=0):
        # defaults: a ~40 kHz ultrasonic levitator in air -> lambda ~ 8.6 mm, nodes ~4.3 mm apart
        from holographic.misc.holographic_fields import ParticleSystem
        self.height = float(height)
        self.wavelength = float(wavelength)
        self.amplitude = float(amplitude)
        self.g = float(gravity)
        self.radius = float(bead_radius)
        self.rho_p = float(bead_density)
        # a small effective mass so the acoustic force (per unit mass) can overcome gravity -- levitators work
        # exactly because the bead is light; mass_scale rolls the tiny particle mass into the force normalisation
        self.mass_scale = float(mass_scale)
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, self.wavelength * 3, n_beads)         # spread across a few wavelengths (for viewing)
        y = rng.uniform(0.05 * height, height, n_beads)            # released across the column
        self.ps = ParticleSystem(np.stack([x, y], axis=1))

    def _accel(self, field_on):
        """Acceleration on each bead: gravity down, plus (if the field is on) the acoustic radiation force along y.
        Returns an (N,2) array for the ParticleSystem."""
        y = self.ps.pos[:, 1]
        ay = -self.g * np.ones_like(y)                             # gravity
        if field_on:
            fy = gorkov_force_y(y, self.wavelength, self.amplitude, radius=self.radius, rho_p=self.rho_p)
            ay = ay + fy / self.mass_scale                         # radiation force per unit (effective) mass
        return np.stack([np.zeros_like(y), ay], axis=1)

    def step(self, field_on=True, dt=2e-4, damping=0.06):
        """Advance the beads one step. Light damping models air drag so beads settle rather than ring forever.
        Beads cannot fall through the floor (y=0)."""
        self.ps.step(force=self._accel(field_on), dt=dt, damping=damping)
        self.ps.pos[:, 1] = np.clip(self.ps.pos[:, 1], 0.0, self.height)
        landed = self.ps.pos[:, 1] <= 0.0                          # hit the floor -> stop bouncing
        self.ps.vel[landed, 1] = 0.0
        return self.ps.pos

    def settle(self, steps=4000, field_on=True, dt=2e-4, damping=0.06):
        for _ in range(int(steps)):
            self.step(field_on=field_on, dt=dt, damping=damping)
        return self.ps.pos

    def heights(self):
        return self.ps.pos[:, 1].copy()


def _selftest():
    """The radiation force vanishes at nodes/antinodes and points toward the nodes between; with the field ON a
    dense bead is held aloft at a pressure node against gravity; with it OFF it falls to the floor; node spacing
    is lambda/2. Deterministic."""
    lam = 0.0086
    nodes = pressure_nodes(lam, 0.10)

    # (1) node spacing is lambda/2
    assert np.allclose(np.diff(nodes), lam / 2.0, atol=1e-9)

    # (2) the force is ~zero AT a node and pushes TOWARD it from either side (a stable trap)
    node = nodes[2]
    assert abs(gorkov_force_y(node, lam, 4000.0)) < abs(gorkov_force_y(node + lam / 8, lam, 4000.0))
    f_below = gorkov_force_y(node - lam / 8, lam, 4000.0)
    f_above = gorkov_force_y(node + lam / 8, lam, 4000.0)
    assert f_below > 0 and f_above < 0                             # pushes up from below, down from above -> toward node

    # (3) FIELD ON: dense beads are held aloft (not on the floor), each near a pressure node
    chamber = LevitationChamber(height=0.05, wavelength=lam, amplitude=5000.0, n_beads=30, seed=0)
    chamber.settle(steps=6000, field_on=True)
    h_on = chamber.heights()
    assert (h_on > 0.002).mean() > 0.8                             # most beads levitating, off the floor
    nd = pressure_nodes(lam, 0.05)
    nearest = np.min(np.abs(h_on[:, None] - nd[None, :]), axis=1)  # distance from each bead to its nearest node
    assert np.median(nearest) < lam / 8                            # trapped close to the nodes

    # (4) FIELD OFF: gravity wins, beads fall to the floor
    chamber_off = LevitationChamber(height=0.05, wavelength=lam, amplitude=5000.0, n_beads=30, seed=0)
    chamber_off.settle(steps=8000, field_on=False)
    assert (chamber_off.heights() < 0.005).mean() > 0.8          # nearly all fell to (near) the floor
    assert chamber_off.heights().mean() < h_on.mean() * 0.5       # far lower than when the field held them up

    # (5) deterministic
    a = LevitationChamber(wavelength=lam, n_beads=10, seed=1); a.settle(500)
    b = LevitationChamber(wavelength=lam, n_beads=10, seed=1); b.settle(500)
    assert np.array_equal(a.heights(), b.heights())
    print("holographic_levitate selftest OK: Gor'kov force pushes beads to pressure nodes (spaced lambda/2); field "
          "ON holds %d%% of dense beads aloft against gravity near nodes; field OFF they fall; deterministic"
          % int((h_on > 0.002).mean() * 100))


if __name__ == "__main__":
    _selftest()
