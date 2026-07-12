"""Cross-faculty integration for the merged scene generators reaching into geometry + signal (backlog items 8/9):
lombscargle reads a REAL in-engine producer (an nbody orbit), and a nebula volume isosurfaces into a mesh."""
import numpy as np
import lecore


def test_lombscargle_recovers_nbody_orbital_period():
    """ITEM 8: the period finder consumes a real in-engine timeseries -- the orbital radius of an nbody body, which
    oscillates once per orbit. Closes the loop sim -> timeseries -> period."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    vc = m.circular_orbit_velocity(1000.0, 1, 1.0)
    # a mildly elliptical orbit so the radius genuinely oscillates (a pure circle has constant radius)
    r = m.nbody_simulate(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.0], [0.0, vc * 1.2]]),
                         np.array([1000.0, 1.0]), 0.0008, 3000, G=1.0, softening=1e-4, record_every=3)
    traj = np.asarray(r["trajectory"])
    radius = np.linalg.norm(traj[:, 1, :], axis=1)
    t = np.arange(len(radius)) * 0.0008 * 3
    per = m.best_period(t, radius, min_period=0.05, max_period=1.0)
    assert per["power"] > 10.0, "a clear periodic radius should give a strong periodogram peak (%.1f)" % per["power"]
    assert 0.1 < per["period"] < 0.9, "recovered orbital period out of expected band (%.3f)" % per["period"]


def test_nebula_isosurfaces_to_a_mesh():
    """ITEM 9: a nebula density volume isosurfaces into a real mesh via surface_mesh -- the generator reaches the
    polygon half of the modeler. (Down: surface_mesh works on the nebula's own field callable.)"""
    m = lecore.UnifiedMind(dim=256, seed=0)
    v = m.nebula_volume(res=16, seed=0)
    fn = m.nebula_field_fn(v, bounds=((0.0, 0.0, 0.0), (15.0, 15.0, 15.0)))
    surf = lambda p: 0.5 - fn(p)                              # implicit surface at density = 0.5
    mesh = m.surface_mesh(surf, bounds=((0.0, 0.0, 0.0), (15.0, 15.0, 15.0)), resolution=16, level=0.0)
    assert mesh.n_faces > 100, "the nebula isosurface should be a non-trivial mesh (%d faces)" % mesh.n_faces


def test_nebula_occupancy_to_mesh():
    """ITEM 9 (alt): thresholded nebula density -> occupancy -> mesh (verts, faces) via occupancy_to_mesh."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    v = m.nebula_volume(res=16, seed=0)
    occ = (v > 0.5).astype(float)
    verts, faces = m.occupancy_to_mesh(occ, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    assert np.asarray(verts).shape[0] > 0 and np.asarray(faces).shape[0] > 0, "occupancy mesh should be non-empty"


def test_star_cluster_positions_are_a_point_field():
    """ITEM 9: a star cluster is a deterministic point field -- the raw material fit_primitives/sdf_from_points
    consume. Here we assert the field is well-formed and deterministic (the geometry bridge's input contract)."""
    m = lecore.UnifiedMind(dim=256, seed=0)
    c = m.star_cluster(40, seed=0, extent=2.0)
    pos = np.array([s["position"] for s in c["systems"]])
    assert pos.shape == (40, 2) and np.all(np.abs(pos) <= 2.0 + 1e-6), "cluster positions within the extent"
    c2 = m.star_cluster(40, seed=0, extent=2.0)
    pos2 = np.array([s["position"] for s in c2["systems"]])
    assert np.array_equal(pos, pos2), "star cluster must be deterministic"
