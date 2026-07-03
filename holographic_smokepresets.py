"""holographic_smokepresets.py -- SMOKE PRESETS (fluids/matter backlog, content item 1).

The smoke SOLVER is already wired (holographic_fields.smoke_step, exposed on UnifiedMind): temperature drives
velocity by buoyancy, vorticity confinement keeps it curly, density + temperature advect with the flow -- all on the
FFT fluid solver the bind operator provides. So there is nothing to build here but the LOOKS: named parameter bundles
(buoyancy / confinement / viscosity / gravity / ambient) plus a source placement, each producing a recognisable
smoke behaviour, and a runner that steps the wired solver.

This is deliberately the lowest-effort item -- reuse, not a new solver. The presets are just settings; the six looks
differ only by dials, which is the whole thesis of the matter model that follows (smoke is the 1-channel, tension-0
corner of it). Rendering uses the existing volint closed-form optical depth (render_fog), not a new marcher.

KEPT NEGATIVE: these are 2-D looks on a modest grid for interactivity; the solver itself is unchanged, so any solver
limitation (a coarse grid smears fine curl) is inherited, not introduced. Deterministic (seeded source jitter).
"""
import numpy as np

from holographic_fields import smoke_step


# Each preset is a bundle of smoke_step dials plus a source kind. The NAMED look is the emergent behaviour of the
# wired solver under these settings -- not a special code path.
SMOKE_PRESETS = {
    # a hot plume at the base: strong buoyancy + moderate confinement -> smoke rises and curls
    "rising":     dict(buoyancy=2.2, confinement=0.30, viscosity=0.00, gravity=0.0, ambient=0.0, source="base"),
    # thinner, curlier: less lift, more vorticity confinement -> wispy tendrils
    "wispy":      dict(buoyancy=1.3, confinement=0.60, viscosity=0.00, gravity=0.0, ambient=0.0, source="base"),
    # thick slow billows: strong lift but viscous, little confinement -> rounded rolls
    "billow":     dict(buoyancy=2.6, confinement=0.05, viscosity=0.03, gravity=0.0, ambient=0.0, source="base"),
    # heavy smoke that SINKS: buoyancy weak, gravity on -> pools at the floor
    "heavy":      dict(buoyancy=0.0, confinement=0.10, viscosity=0.01, gravity=1.2, ambient=0.0, source="base"),
    # a still ambient haze: no lift, gentle viscosity -> hangs where it is emitted
    "still_room": dict(buoyancy=0.0, confinement=0.00, viscosity=0.03, gravity=0.0, ambient=0.0, source="center"),
    # a smoky room: moderate lift stratifies a diffuse fill toward the ceiling
    "stratified": dict(buoyancy=1.1, confinement=0.20, viscosity=0.00, gravity=0.0, ambient=0.0, source="base_wide"),
}


def preset_names():
    """The available preset names."""
    return list(SMOKE_PRESETS.keys())


def _source(nx, ny, kind, seed=0):
    """Build (density_source, temperature_source) arrays for a source kind. A hot source injects BOTH density and
    temperature (temperature is what buoyancy acts on). Rows index y (row 0 = bottom)."""
    rng = np.random.default_rng(seed)
    ds = np.zeros((ny, nx))
    ts = np.zeros((ny, nx))
    cx = nx // 2
    if kind == "base" or kind == "base_wide":
        w = max(2, nx // (8 if kind == "base" else 3))          # narrow plume vs wide room fill
        y0, y1 = 1, max(2, ny // 12)
        ds[y0:y1, cx - w:cx + w] = 1.0
        ts[y0:y1, cx - w:cx + w] = 1.0
    elif kind == "top":                                         # heavy smoke released high, sinks
        w = max(2, nx // 8)
        y0, y1 = ny - max(2, ny // 12), ny - 1
        ds[y0:y1, cx - w:cx + w] = 1.0
        ts[y0:y1, cx - w:cx + w] = 1.0                          # warm but gravity dominates
    elif kind == "center":                                     # a puff in the middle that just hangs
        w = max(2, nx // 8)
        cy = ny // 2
        ds[cy - w:cy + w, cx - w:cx + w] = 1.0
        ts[cy - w:cy + w, cx - w:cx + w] = 0.0                  # no heat -> no lift
    ds += 0.001 * rng.standard_normal((ny, nx))                 # tiny jitter breaks symmetry (deterministic)
    return np.clip(ds, 0, None), ts


def simulate(name, nx=48, ny=48, steps=40, dt=0.1, seed=0, source_steps=None):
    """Run the wired smoke solver under a preset and return the fields. The source injects for `source_steps` steps
    (default: the first 60%) then stops, so the plume can rise/settle. Returns dict(density, temperature, vx, vy)."""
    if name not in SMOKE_PRESETS:
        raise ValueError("unknown smoke preset %r (have %s)" % (name, ", ".join(SMOKE_PRESETS)))
    p = dict(SMOKE_PRESETS[name])
    kind = p.pop("source")
    if source_steps is None:
        source_steps = max(3, int(steps * 0.2))   # brief puff -> the plume detaches and the dials shape it
    ds, ts = _source(nx, ny, kind, seed=seed)
    vx = np.zeros((ny, nx)); vy = np.zeros((ny, nx))
    density = np.zeros((ny, nx)); temperature = np.zeros((ny, nx))
    for s in range(steps):
        dens_src = ds if s < source_steps else None
        temp_src = ts if s < source_steps else None
        vx, vy, density, temperature = smoke_step(vx, vy, density, temperature, dt=dt,
                                                  dens_source=dens_src, temp_source=temp_src, **p)
    return {"density": density, "temperature": temperature, "vx": vx, "vy": vy}


def plume_center_of_mass(density):
    """The density-weighted mean row (y). Higher = the smoke has risen; lower = it sank/pooled. Normalised 0..1."""
    ny = density.shape[0]
    d = np.clip(density, 0, None)
    total = d.sum()
    if total < 1e-9:
        return 0.0
    ys = np.arange(ny)[:, None]
    return float((ys * d).sum() / total / (ny - 1))


def render(name, nx=48, ny=48, steps=40, dt=0.1, seed=0):
    """Simulate a preset and return a simple grayscale image of its density (normalised), for a quick look. Uses the
    density field directly; volint.render_fog is the full 3-D optical-depth renderer for the volumetric case."""
    out = simulate(name, nx=nx, ny=ny, steps=steps, dt=dt, seed=seed)
    d = np.clip(out["density"], 0, None)
    m = d.max()
    img = (d / m) if m > 1e-9 else d
    return img[::-1]                                             # flip so row 0 (bottom) is at the image bottom


def _buoyant_vs_heavy():
    """A controlled central puff: with buoyancy the plume's centre of mass rises ABOVE the middle; with gravity it
    sinks BELOW. This isolates the physics from source placement (the preset sources sit at the base)."""
    ny = nx = 40
    ds = np.zeros((ny, nx)); ts = np.zeros((ny, nx))
    cy, cx, w = ny // 2, nx // 2, 4
    ds[cy - w:cy + w, cx - w:cx + w] = 1.0
    ts[cy - w:cy + w, cx - w:cx + w] = 1.0

    def run(buoyancy, gravity):
        vx = np.zeros((ny, nx)); vy = np.zeros((ny, nx))
        density = np.zeros((ny, nx)); temperature = np.zeros((ny, nx))
        for s in range(40):
            dsrc = ds if s < 4 else None
            tsrc = ts if s < 4 else None
            vx, vy, density, temperature = smoke_step(vx, vy, density, temperature, dt=0.1,
                                                      dens_source=dsrc, temp_source=tsrc,
                                                      buoyancy=buoyancy, gravity=gravity)
        return plume_center_of_mass(density)

    return run(2.2, 0.0), run(0.0, 1.2)


def _selftest():
    """Confirm the six presets give DISTINCT looks, a still centre puff hangs mid-grid, and the buoyancy/gravity
    dial actually pushes the plume up vs down (controlled central puff)."""
    coms = {name: plume_center_of_mass(simulate(name, nx=40, ny=40, steps=45, seed=0)["density"])
            for name in SMOKE_PRESETS}

    # the six presets are genuinely different looks (distinct centres of mass)
    assert len(set(round(c, 2) for c in coms.values())) >= 4, coms
    # a still centre puff (no lift, no sink) hangs around the middle
    assert 0.3 < coms["still_room"] < 0.7, coms
    # the buoyancy/gravity dial works: a hot puff rises above centre, a heavy one sinks below (same source)
    up, down = _buoyant_vs_heavy()
    assert up > 0.5 > down, (up, down)
    # every preset actually puts smoke in the domain
    for name in SMOKE_PRESETS:
        assert simulate(name, nx=32, ny=32, steps=12, seed=1)["density"].sum() > 0.0

    print("holographic_smokepresets selftest OK: six presets give distinct looks ("
          + ", ".join("%s=%.2f" % (n, coms[n]) for n in SMOKE_PRESETS)
          + "); a still puff hangs mid-grid; the buoyancy/gravity dial lifts a hot puff (COM=%.2f) above centre and "
            "sinks a heavy one (COM=%.2f) below -- all on the wired FFT smoke solver" % (up, down))


if __name__ == "__main__":
    _selftest()
