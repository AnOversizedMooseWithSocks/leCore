"""Modeling-app backlog capstone: the Sampler -- a placeable read-probe, the read-dual of FieldEffect."""
import numpy as np
from holographic_sdf import sphere
from holographic_scene_doc import Scene
from holographic_sampler import (Sampler, contribution_of, dominant_owner, total_contribution,
                                 owners_from_sdfs, place_sampler, sampler_triggers)

FIELD = lambda P: np.asarray(P, float)[:, 2]           # height field = z


class _Shift:
    def __init__(self, base, off): self.base = base; self.off = np.asarray(off, float)
    def eval(self, P): return self.base.eval(np.asarray(P, float) - self.off)


def test_point_mode_reads_at_spot():
    s = Sampler(sphere(0.1), FIELD, mode="point")
    assert abs(s.sample(at=(0, 0, 2.0)) - 2.0) < 1e-9


def test_volume_mode_mean_and_integral():
    s = Sampler(sphere(1.0), FIELD, mode="volume", radius=1.0, falloff="linear")
    mean = s.sample(at=(0, 0, 0), bounds=([-1, -1, -1], [1, 1, 1]), n=400, seed=0, aggregate="mean")
    assert abs(mean) < 0.15                              # symmetric region -> ~0 mean height
    integral = s.sample(at=(0, 0, 0), bounds=([-1, -1, -1], [1, 1, 1]), n=400, seed=0, aggregate="sum")
    assert np.isfinite(integral)


def test_volume_shifted_region_shifts_mean():
    # a field = z, sampled in the upper half [0,2] -> positive mean height
    s = Sampler(sphere(2.0), FIELD, mode="volume", radius=2.0, falloff="linear")
    mean = s.sample(at=(0, 0, 1), bounds=([-2, -2, 0.0], [2, 2, 2]), n=500, seed=0, aggregate="mean")
    assert mean > 0.3                                    # reads the higher region


def test_labeled_bundle_separates_and_collapses():
    dim = 512; rng = np.random.default_rng(0)
    hA = rng.standard_normal(dim); hA /= np.linalg.norm(hA)
    hB = rng.standard_normal(dim); hB /= np.linalg.norm(hB)
    owners = owners_from_sdfs([(hA, sphere(1.0)), (hB, _Shift(sphere(1.0), [3, 0, 0]))])
    s = Sampler(sphere(5.0), lambda P: np.ones(len(P)), mode="volume", radius=5.0)
    lab = s.sample_labeled(owners, at=(1.5, 0, 0), bounds=([-2, -2, -2], [5, 2, 2]), n=600, seed=1)
    cA = contribution_of(lab, hA); cB = contribution_of(lab, hB)
    assert cA > 0 and cB > 0                             # both present, separable by handle
    assert dominant_owner(lab, [hA, hB]) in (0, 1)
    assert abs(total_contribution(lab, [hA, hB]) - (cA + cB)) < 1e-6   # collapsible


def test_placeable_in_scene():
    scene = Scene(dim=128, seed=0)
    s = Sampler(sphere(0.1), FIELD, mode="point")
    h = place_sampler(scene, s, name="probe")
    assert h in scene.objects and scene.get(h).params["is_sampler"]
    assert scene.get(h).params["sampler"] is s


def test_trigger():
    assert sampler_triggers(2.0, 1.0, "above") and not sampler_triggers(0.5, 1.0, "above")
    assert sampler_triggers(0.5, 1.0, "below")


def test_deterministic():
    s = Sampler(sphere(1.0), FIELD, mode="volume")
    a = s.sample(at=(0, 0, 0), bounds=([-1, -1, -1], [1, 1, 1]), n=200, seed=7)
    b = s.sample(at=(0, 0, 0), bounds=([-1, -1, -1], [1, 1, 1]), n=200, seed=7)
    assert a == b
