"""Tests for holographic_regionfield: composable labelled regions -> classify / slice / cull / behaviour."""
import numpy as np
from holographic.misc.holographic_regionfield import RegionField, Region, layered_sphere
from holographic.simulation_and_physics.holographic_semantic import _SphereSDF, _BoxSDF


def test_priority_resolves_overlap_to_inner_layer():
    field = layered_sphere()
    labels = [field.regions[i].label if i >= 0 else None for i in field.classify(
        np.array([[0, 0, 0], [0.5, 0, 0], [0.9, 0, 0], [2.0, 0, 0]]))]
    assert labels == ["core", "mantle", "crust", None]         # inner (high priority) wins on overlap


def test_slice_reveals_all_layers():
    field = layered_sphere()
    img, idx = field.slice((0, 0, 0), (1, 0, 0), (0, 1, 0), extent=1.5, res=80)
    assert len(set(idx[idx >= 0].tolist())) == 3               # the cut shows every layer
    assert img.shape == (80, 80, 3)


def test_culling_is_precise():
    field = layered_sphere()
    pts = np.array([[0, 0, 0], [0.9, 0, 0], [1.5, 0, 0], [5, 5, 5]])
    assert field.cull(pts).tolist() == [True, True, False, False]   # inside kept, outside known-empty


def test_material_and_behavior_by_region():
    rf = RegionField([
        Region(_BoxSDF((0, 0, 0), (1, 0.1, 1)), "ground", 1, material=(0.3, 0.5, 0.2), behavior="static"),
        Region(_SphereSDF((0, 0.5, 0), 0.3), "flame", 2, material=(1.0, 0.4, 0.0), behavior="fire"),
    ])
    mat = rf.material_at(np.array([[0, 0, 0], [0, 0.5, 0]]))
    assert np.allclose(mat[1], (1.0, 0.4, 0.0))                # the flame region's material at its centre
    beh = rf.behavior_at(np.array([[0, 0.5, 0], [3, 3, 3]]))
    assert beh == ["fire", None]                               # one field also picks the simulation


def test_region_drives_reflectivity_and_roughness():
    """A region can set per-point reflectivity and roughness -- one body, many materials."""
    import numpy as np
    from holographic.misc.holographic_regionfield import RegionField, Region
    from holographic.simulation_and_physics.holographic_semantic import _SphereSDF
    rf = RegionField([
        Region(_SphereSDF((0, 0, 0), 2.0), "body", 0, material=(0.3, 0.3, 0.3), reflect=0.05, roughness=0.0),
        Region(_SphereSDF((1.0, 0, 0), 0.6), "mirror", 2, material=(0.9, 0.9, 0.9), reflect=0.85, roughness=0.0),
        Region(_SphereSDF((-1.0, 0, 0), 0.6), "brushed", 2, material=(0.7, 0.6, 0.4), reflect=0.5, roughness=0.16),
    ])
    pts = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.5, 0]])   # mirror / brushed / body
    assert np.allclose(rf.reflect_at(pts), [0.85, 0.5, 0.05])
    assert np.allclose(rf.roughness_at(pts), [0.0, 0.16, 0.0])


def test_region_roughness_can_be_a_field():
    """A region's roughness can be a map/field (a socket), not just a number -- roughness varies across the region."""
    import numpy as np
    from holographic.misc.holographic_regionfield import RegionField, Region
    from holographic.simulation_and_physics.holographic_semantic import _SphereSDF
    from holographic.misc.holographic_param import Param
    rf = RegionField([Region(_SphereSDF((0, 0, 0), 2.0), "body", 0, material=(0.5, 0.5, 0.5),
                             reflect=0.6, roughness=Param(field=lambda P: 0.05 + 0.15 * np.clip(P[:, 1], 0, None)))])
    pts = np.array([[0, -1.5, 0.5], [0, 1.5, 0.5]])                       # low vs high on the sphere
    rough = rf.roughness_at(pts)
    assert rough[1] > rough[0]                                            # roughness rises with height (a field)
    assert np.allclose(rf.reflect_at(pts), 0.6)                           # a constant param still works


def test_region_albedo_can_be_a_texture():
    """A region's ALBEDO can be a field/callable (a texture), consistent with reflect/roughness taking maps."""
    import numpy as np
    from holographic.misc.holographic_regionfield import RegionField, Region
    from holographic.simulation_and_physics.holographic_semantic import _SphereSDF
    grad = lambda P: np.stack([np.clip(0.5 + 0.4 * P[:, 1], 0, 1), 0.3 + 0 * P[:, 1],
                               np.clip(0.5 - 0.4 * P[:, 1], 0, 1)], axis=1)
    rf = RegionField([Region(_SphereSDF((0, 0, 0), 2.0), "body", 0, material=grad)])
    pts = np.array([[0, -1.5, 0.5], [0, 1.5, 0.5]])
    cols = rf.material_at(pts)
    assert cols[0, 0] < cols[1, 0] and cols[0, 2] > cols[1, 2]        # colour varies with height (a texture)
    rf2 = RegionField([Region(_SphereSDF((0, 0, 0), 2.0), "b", 0, material=(0.8, 0.2, 0.2))])
    assert np.allclose(rf2.material_at(pts[:1]), [0.8, 0.2, 0.2])     # constant colour still works
