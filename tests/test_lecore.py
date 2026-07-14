"""Modeling-app backlog item H: the curated public API façade."""
import numpy as np
import lecore


def test_facade_namespaces_exist():
    for area in ("product", "scene", "model", "render", "sim", "transform"):
        assert hasattr(lecore, area)
    assert hasattr(lecore.product, "LocalAgentCore")
    assert hasattr(lecore.scene, "Scene") and hasattr(lecore.model, "ModifierStack")
    assert hasattr(lecore.render, "CancelToken") and hasattr(lecore.sim, "MPMSnow")
    assert hasattr(lecore.transform, "look_at")


def test_facade_operations_work_end_to_end():
    doc = lecore.scene.Scene(dim=64, seed=0)
    h = doc.add(name="part", geometry=np.zeros((4, 3)))
    assert h in doc.objects
    st = lecore.model.ModifierStack(1.0); st.add("x", lambda v, k=2: v * k, {"k": 3})
    assert st.evaluate() == 3
    assert lecore.transform.look_at((0, 0, 5), (0, 0, 0)).shape == (4, 4)


def test_areas_map():
    a = lecore.areas()
    assert set(a) == {"product", "scene", "model", "render", "sim", "transform"} and all(len(v) for v in a.values())
