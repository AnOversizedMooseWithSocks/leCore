"""holographic_oxidation.py -- M4: the OXIDIZATION / CORROSION front. Rust and patina that SPREAD over time.

WHY THIS EXISTS (Material Structure & Process backlog, item M4)
--------------------------------------------------------------
Metals do not corrode uniformly -- rust starts at an exposed edge or a wet spot and CREEPS inward, and a bright
copper roof greens from its weathered patches outward. That spreading is a reaction-diffusion FRONT: a cell
corrodes faster when it is exposed/wet (pitting starts there) and faster still when its neighbours are already
corroded (the front is autocatalytic -- rust feeds rust). This module is that front, as a readable scalar field
(the same reaction-diffusion family the hypervector CA in holographic_automaton implements, kept here as a plain
oxidation fraction because that is clearer to read and to render), plus the base->oxide colour BLEND that turns
the fraction into an appearance -- steel -> orange rust, copper -> green patina, silver -> dark tarnish.

THE MODEL (readable)
--------------------
An oxidation field `ox` in [0,1] over a surface grid (0 = pristine, 1 = fully corroded). Each step:
    d_ox/dt = rate * moisture * (seed_rate * exposure  +  spread * neighbour_oxidation) * (1 - ox)
  * `exposure`   -- where corrosion can NUCLEATE (exposed faces / scratches); by default the grid border.
  * `neighbour_oxidation` -- the mean corrosion of the 4 neighbours; this is the diffusion term that makes it a
    FRONT advancing from already-rusted cells, not uniform bleaching.
  * `(1 - ox)`   -- corrosion slows as a cell saturates (it can only rust so far).
  * moisture / rate -- wet rusts faster than dry; steel rusts faster than copper patinas.
Oxidation is monotonic (drive >= 0), so rust never spontaneously un-rusts -- honest for a decay process.

HONEST SCOPE (kept negative): a phenomenological reaction-diffusion FRONT with per-material rate and oxide colour,
NOT electrochemistry (no galvanic cells, pH, or passivation-layer growth). Aluminium's real self-limiting oxide is
approximated by a low rate, not a passivation model. The numbers are plausible and art-directable. Deterministic;
NumPy + stdlib. Oxide colour / rate are the corrosion data columns; they live here beside the physics that uses
them and could migrate to the material definition data layer.
"""
import numpy as np

# The corrosion data columns, per material: the colour it corrodes TO, its relative rate, and whether it needs
# moisture (most do; some tarnish in dry air). base colour comes from the material's own albedo (matlib), reused.
OXIDATION = {
    # material       oxide_color (rgb)        rate   needs_moisture   oxide_name
    "steel":     dict(oxide_color=(0.55, 0.27, 0.12), rate=1.00, needs_moisture=True,  oxide="rust"),
    "iron":      dict(oxide_color=(0.52, 0.25, 0.11), rate=1.10, needs_moisture=True,  oxide="rust"),
    "copper":    dict(oxide_color=(0.36, 0.66, 0.55), rate=0.30, needs_moisture=True,  oxide="patina"),
    "bronze":    dict(oxide_color=(0.34, 0.60, 0.50), rate=0.28, needs_moisture=True,  oxide="patina"),
    "silver":    dict(oxide_color=(0.20, 0.18, 0.15), rate=0.45, needs_moisture=False, oxide="tarnish"),
    "aluminum":  dict(oxide_color=(0.70, 0.70, 0.72), rate=0.08, needs_moisture=False, oxide="dull_oxide"),
}


def _neighbour_mean(field):
    """Mean of the 4 nearest neighbours of each cell, with edge replication (so a border cell just sees fewer
    distinct neighbours, not a wrap-around). This is the diffusion term that carries the corrosion front along."""
    P = np.pad(field, 1, mode="edge")
    return 0.25 * (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:])


def _base_albedo(material):
    """The pristine colour of a material (reused from the matlib catalog if it is there, else a neutral grey)."""
    try:
        import holographic.materials_and_texture.holographic_matlib as _ml
        if material in _ml.RENDER_MATERIALS:
            return _ml.albedo(material)
    except Exception:
        pass
    return np.array([0.6, 0.6, 0.62])


class OxidationField:
    """A corrosion front over a surface grid. `exposure` marks where rust can nucleate (default: the border, i.e.
    exposed edges); `moisture` (scalar or field) scales the rate. Call `step(material, dt)` to advance the front,
    and `albedo(material)` to get the per-cell base->oxide colour blend for rendering."""

    def __init__(self, shape, exposure=None, moisture=1.0, seed=None):
        self.ox = np.zeros(shape, float)
        if exposure is None:                                        # default: the grid border is the exposed face
            e = np.zeros(shape, float); e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = 1.0
            exposure = e
        self.exposure = np.asarray(exposure, float)
        self.moisture = moisture
        if seed is not None:                                        # optional initial corrosion seed (a scratch)
            self.ox = np.maximum(self.ox, np.asarray(seed, float))

    def step(self, material, dt=1.0, spread=0.6, seed_rate=0.15):
        """Advance corrosion by dt. Rust nucleates at exposed/wet cells and SPREADS from corroded neighbours; each
        cell saturates toward fully corroded. Monotonic (rust does not reverse). Returns the oxidation field."""
        c = OXIDATION[material]
        moist = np.asarray(self.moisture, float)
        if c["needs_moisture"]:
            drive_moist = moist
        else:
            drive_moist = np.maximum(moist, 0.4)                    # tarnish proceeds even fairly dry
        neigh = _neighbour_mean(self.ox)
        drive = seed_rate * self.exposure + spread * neigh          # nucleate at faces + advance the front
        d_ox = c["rate"] * drive_moist * dt * drive * (1.0 - self.ox)
        self.ox = np.clip(self.ox + d_ox, 0.0, 1.0)
        return self.ox

    def fraction(self):
        """Overall corroded fraction (mean oxidation), a single honest progress number."""
        return float(self.ox.mean())

    def albedo(self, material):
        """Per-cell colour: the base material BLENDED toward its oxide by the local oxidation fraction. This is the
        blend/interpolation capability applied to weathering -- pristine where ox=0, full oxide where ox=1."""
        base = _base_albedo(material)
        oxide = np.asarray(OXIDATION[material]["oxide_color"], float)
        t = self.ox[..., None]                                      # (H,W,1) blend weight
        return (1.0 - t) * base + t * oxide


def oxide_color(material, ox_fraction):
    """The blended colour of `material` at a scalar oxidation fraction 0..1 -- pristine base to full oxide. The
    same lerp the field uses per cell, for a single sample (e.g. a whole-object weathering slider)."""
    base = _base_albedo(material)
    oxide = np.asarray(OXIDATION[material]["oxide_color"], float)
    t = float(np.clip(ox_fraction, 0.0, 1.0))
    return (1.0 - t) * base + t * oxide


def _selftest():
    """Corrosion NUCLEATES at exposed faces and SPREADS inward (a front, not uniform), is monotonic, respects
    per-material rate and moisture, and blends base->oxide. Deterministic."""
    # (1) the front spreads from the border inward: edges corrode before the centre
    f = OxidationField((21, 21))                                    # border exposed by default
    centre_hist, edge_hist = [], []
    for _ in range(30):
        f.step("steel", dt=1.0)
        edge_hist.append(f.ox[0, 10]); centre_hist.append(f.ox[10, 10])
    assert edge_hist[-1] > centre_hist[-1]                          # edge more corroded than centre (a front)
    assert centre_hist[-1] > 0.0                                    # but it did reach the centre over time
    assert all(centre_hist[i + 1] >= centre_hist[i] - 1e-12 for i in range(len(centre_hist) - 1))  # monotonic

    # (2) steel rusts faster than copper patinas, given the same exposure and time
    s = OxidationField((15, 15)); cu = OxidationField((15, 15))
    for _ in range(20):
        s.step("steel"); cu.step("copper")
    assert s.fraction() > cu.fraction()

    # (3) moisture matters: a dry steel plate corrodes slower than a wet one
    wet = OxidationField((15, 15), moisture=1.0); dry = OxidationField((15, 15), moisture=0.1)
    for _ in range(20):
        wet.step("steel"); dry.step("steel")
    assert wet.fraction() > dry.fraction()

    # (4) the blend: steel goes orange-ish rust, copper goes green-ish patina
    rust = oxide_color("steel", 1.0); patina = oxide_color("copper", 1.0)
    assert rust[0] > rust[2]                                        # rust: red > blue (orange-brown)
    assert patina[1] > patina[0]                                    # patina: green dominant
    assert np.allclose(oxide_color("steel", 0.0), _base_albedo("steel"))   # ox=0 -> pristine base

    # (5) deterministic
    a = OxidationField((10, 10)); b = OxidationField((10, 10))
    for _ in range(10):
        a.step("iron"); b.step("iron")
    assert np.array_equal(a.ox, b.ox)
    print("holographic_oxidation selftest OK: corrosion nucleates at exposed faces and spreads inward as a front "
          "(monotonic); steel rusts faster than copper; wet > dry; base->oxide blend (rust orange, patina green)")


if __name__ == "__main__":
    _selftest()
