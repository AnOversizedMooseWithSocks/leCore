"""Tests for holographic_rendergraph.py (CMP5) -- the pipeline orchestrating the CMP1-CMP4 graphs (bake vs live)."""
import numpy as np
import pytest
from holographic_texturegraph import Map, Const, field_leaf
from holographic_instancing import Definition, InstancedScene, VOLUME
from holographic_mesh import box
from holographic_scenegraph import translation
from holographic_rendergraph import BakedTexture, bake_texture, resolve_texture, RenderGraph, PreparedScene


def _graph():
    return Map("mix", a=Const([1.0, 0, 0]), b=Const([0, 0, 1.0]), t=field_leaf("fbm", n_dims=2, seed=0))


def test_baked_texture_matches_live_within_interp_error():
    g = _graph()
    baked = bake_texture(g, res=128)
    for uv in [(0.2, 0.3), (0.5, 0.5), (0.8, 0.1), (0.33, 0.77)]:
        assert np.max(np.abs(np.asarray(g.sample(uv)) - np.asarray(baked.sample(uv)))) < 0.05


def test_baked_texture_same_sample_interface():
    baked = bake_texture(_graph(), res=32)
    out = baked.sample([0.4, 0.6])
    assert np.asarray(out).shape == (3,)          # a colour graph -> colour out, like the live graph


def test_adaptive_resolve():
    g = _graph()
    assert isinstance(resolve_texture(g, static=True), BakedTexture)     # static -> bake
    assert resolve_texture(g, static=False) is g                         # dynamic -> live
    assert isinstance(resolve_texture(g, bake=True), BakedTexture)       # forced bake
    assert resolve_texture(g, bake=False) is g                           # forced live


def test_plan_reports_bake_vs_live_and_bind():
    g = _graph()
    scene = InstancedScene()
    chair = Definition("chair", box(1, 1, 1), "metal")
    scene.place(chair)
    rg = RenderGraph(res=32).add_texture("rust", g, static=True).add_texture("ripples", g, static=False).set_scene(scene)
    plan = rg.plan()
    assert any("BAKE" in ln and "rust" in ln for ln in plan)
    assert any("LIVE" in ln and "ripples" in ln for ln in plan)
    assert any("bind_scene" in ln for ln in plan)


def test_prepare_bakes_static_keeps_dynamic_live_and_binds():
    g = _graph()
    scene = InstancedScene()
    chair = Definition("chair", box(1, 1, 1), "metal")
    scene.place(chair, translation([-2, 0, 0])); scene.place(chair, translation([2, 0, 0]))
    scene.place(Definition("haze", object(), "fog", geometry_kind=VOLUME))
    rg = RenderGraph(res=32).add_texture("rust", g, static=True).add_texture("ripples", g, static=False).set_scene(scene)
    prep = rg.prepare()
    assert isinstance(prep, PreparedScene)
    assert isinstance(prep.texture("rust"), BakedTexture)
    assert prep.texture("ripples") is g
    assert prep.surface_mesh.n_vertices == 2 * box(1, 1, 1).n_vertices
    assert len(prep.volume_instances) == 1


def test_prepare_without_a_scene():
    rg = RenderGraph(res=16).add_texture("t", _graph(), static=True)
    prep = rg.prepare()
    assert prep.surface_mesh.n_vertices == 0 and prep.volume_instances == []


def test_stages_declare_needs_and_produces():
    rg = RenderGraph()
    stages = rg._stages()
    names = [s.name for s in stages]
    assert "bake_textures" in names and "bind_scene" in names
    bake = next(s for s in stages if s.name == "bake_textures")
    assert "resolved_textures" in bake.produces
