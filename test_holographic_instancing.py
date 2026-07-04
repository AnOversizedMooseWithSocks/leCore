"""Tests for holographic_instancing.py (CMP4) -- type-correct binding + shared-definition (edit-once) instancing."""
import pytest
from holographic_mesh import box
from holographic_scenegraph import translation
from holographic_instancing import (Definition, Instance, InstancedScene, material_kind, geometry_kind_of,
                                     SURFACE, VOLUME)


def test_material_and_geometry_kinds():
    assert material_kind("metal") == SURFACE and material_kind("glass") == SURFACE
    assert material_kind("smoke") == VOLUME and material_kind("fog") == VOLUME
    assert geometry_kind_of(box(1, 1, 1)) == SURFACE
    assert geometry_kind_of(object()) == VOLUME


def test_valid_binding_builds():
    d = Definition("chair", box(1, 1, 1), "metal")
    assert d.material_kind == SURFACE and d.geometry_kind == SURFACE


def test_volumetric_on_mesh_refused():
    with pytest.raises(TypeError) as e:
        Definition("bad", box(1, 1, 1), "smoke")
    assert "volumetric material" in str(e.value)


def test_surface_on_volume_refused():
    with pytest.raises(TypeError):
        Definition("bad", object(), "metal", geometry_kind=VOLUME)


def test_volumetric_on_volume_ok():
    d = Definition("haze", object(), "fog", geometry_kind=VOLUME)
    assert d.material_kind == VOLUME and d.geometry_kind == VOLUME


def test_edit_once_updates_all_instances():
    chair = Definition("chair", box(1, 1, 1), "metal")
    scene = InstancedScene()
    a = scene.place(chair, translation([-2, 0, 0]))
    b = scene.place(chair, translation([2, 0, 0]))
    assert a.material == "metal" and b.material == "metal"
    chair.set_material("glass")                       # one edit
    assert a.material == "glass" and b.material == "glass"   # ... changes every instance


def test_invalid_repaint_refused_and_unchanged():
    chair = Definition("chair", box(1, 1, 1), "metal")
    with pytest.raises(TypeError):
        chair.set_material("smoke")                   # can't put a volumetric material on surface geometry
    assert chair.material == "metal"                  # unchanged after the refused edit


def test_instances_share_and_group():
    chair = Definition("chair", box(1, 1, 1), "metal")
    table = Definition("table", box(2, 1, 2), "wood") if material_kind("wood") == SURFACE else Definition("table", box(2, 1, 2), "metal")
    scene = InstancedScene()
    scene.place(chair); scene.place(chair); scene.place(table)
    assert len(scene.instances) == 3
    assert len(scene.definitions()) == 2
    assert len(scene.instances_of(chair)) == 2


def test_flatten_materializes_surface_instances():
    chair = Definition("chair", box(1, 1, 1), "metal")
    scene = InstancedScene()
    scene.place(chair, translation([-2, 0, 0]))
    scene.place(chair, translation([2, 0, 0]))
    scene.place(Definition("haze", object(), "fog", geometry_kind=VOLUME))   # a volume -- not triangles
    merged = scene.flatten_surface()
    assert merged.n_vertices == 2 * box(1, 1, 1).n_vertices     # 2 surface instances merged
    assert len(scene.surface_instances()) == 2 and len(scene.volume_instances()) == 1


def test_instance_requires_a_definition():
    with pytest.raises(TypeError):
        Instance("not a definition")
