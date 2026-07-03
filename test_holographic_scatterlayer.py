"""Fluids/matter item 5 (part 1): ScatterLayer -- emit geometry onto any surface."""
import numpy as np
from holographic_ai import Vocabulary
from holographic_sdf import sphere, box
from holographic_scatterlayer import ScatterLayer, _cell_code


def _grass(dim=512):
    return Vocabulary(dim, seed=0).get("grass")


def test_scatters_on_sphere():
    layer = ScatterLayer(_grass(), count=60, seed=0)
    res = layer.apply(sphere(1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)))
    assert res["count"] > 20
    d = np.abs([sphere(1.0).eval(p[None, :])[0] for p in res["points"]])
    assert d.max() < 0.05                                       # all on the surface


def test_scatters_on_box_too():
    layer = ScatterLayer(_grass(), count=60, seed=0)
    res = layer.apply(box(1.0, 0.6, 0.8), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)))
    assert res["count"] > 20                                    # any surface, not just terrain


def test_unit_normals():
    res = ScatterLayer(_grass(), count=40, seed=0).apply(sphere(1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)))
    assert np.allclose(np.linalg.norm(res["normals"], axis=1), 1.0, atol=1e-5)


def test_density_map_steers_placement():
    top = lambda P: (np.asarray(P)[:, 1] > 0).astype(float)
    res = ScatterLayer(_grass(), count=120, density=top, seed=1).apply(sphere(1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)))
    assert (res["points"][:, 1] > 0).mean() > 0.9              # nearly all up top


def test_layer_is_region_queryable():
    layer = ScatterLayer(_grass(), count=50, seed=0)
    res = layer.apply(sphere(1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)))
    near = layer.recall_region(res["layer"], res["points"][0])
    far = layer.recall_region(res["layer"], np.array([100.0, 100.0, 100.0]))
    assert near > far + 0.03                                    # a scattered cell reads above an empty one


def test_placements_are_binds():
    layer = ScatterLayer(_grass(), count=10, seed=0)
    res = layer.apply(sphere(1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)))
    assert all(p["vec"].shape == (512,) for p in res["placements"])   # each placement is a bound vector


def test_cell_code_deterministic():
    a = _cell_code(np.array([0.1, 0.2, 0.3]), 256, 0.25, seed=0)
    b = _cell_code(np.array([0.1, 0.2, 0.3]), 256, 0.25, seed=0)
    assert np.array_equal(a, b) and abs(np.linalg.norm(a) - 1.0) < 1e-6
