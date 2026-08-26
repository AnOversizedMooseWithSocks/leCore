"""Tests for S2 procedural generation (holographic_procgen): seed-to-object SDFs, the Menger fractal model,
greeble-a-mesh, and vegetation scattered across a fBm terrain -- compositions of S1 + G1-G6."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import box as mesh_box
from holographic.scene_and_pipeline.holographic_scenegraph import flatten_scene
from holographic.io_and_interop.holographic_procgen import procedural_object, object_to_mesh, greeble_mesh, scatter_on_terrain, vegetated_terrain, _selftest
from holographic.mesh_and_geometry.holographic_sdf import menger


def test_procedural_object_deterministic_and_varies():
    assert procedural_object(7, 3).to_dsl() == procedural_object(7, 3).to_dsl()
    assert procedural_object(7, 3).to_dsl() != procedural_object(8, 3).to_dsl()


def test_procedural_object_renders_and_emits():
    obj = procedural_object(7, 3)
    assert object_to_mesh(obj, res=28).n_faces > 0
    assert "mainImage" in obj.to_glsl()


def test_menger_model_renders():
    mesh = object_to_mesh(menger(2, 1.0), bounds=((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2)), res=40)
    assert mesh.n_faces > 0


def test_greeble_adds_geometry_deterministically():
    base = mesh_box(1, 1, 1)
    g = greeble_mesh(base, seed=3, density=1.0)
    assert g.n_vertices > base.n_vertices
    assert np.allclose(g.vertices, greeble_mesh(base, seed=3, density=1.0).vertices)


def test_scatter_places_at_terrain_height():
    from holographic.mesh_and_geometry.holographic_terrain import Terrain
    terr = Terrain(bounds=[(0, 4), (0, 4)], octaves=3, dim=512, seed=2)
    scene, placements = scatter_on_terrain(terr, lambda rng: mesh_box(0.1, 0.1, 0.3), count=8, seed=1)
    assert len(placements) == 8
    for (x, y, z) in placements:
        assert abs(z - terr.height([x, y])) < 1e-9


def test_vegetated_terrain_builds_a_scene():
    scene, terr = vegetated_terrain(seed=5, n_plants=4, plant_iterations=2)
    assert flatten_scene(scene).n_faces > 0


def test_selftest_runs():
    _selftest()
