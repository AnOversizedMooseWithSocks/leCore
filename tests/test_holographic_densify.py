"""CI wrapper for coarse-to-fine splat densification (C1). The module ships its asserts in `_c1_selftest`:
coarse-to-fine densify_fit reaches a markedly better optimum than the 210-step one-shot aniso_fit on a multi-scale
target because the staged placement is a better warm start. This collects that measured check."""
import numpy as np
import pytest

from holographic.rendering.holographic_splat import _c1_selftest, densify_fit


def test_holographic_densify_selftest():
    _c1_selftest()


def test_explicit_schedule_runs_once_and_reports_the_measured_choice():
    ys, xs = np.mgrid[0:16, 0:16]
    target = np.exp(-((xs - 8) ** 2 + (ys - 7) ** 2) / 18.0)
    stats = {}
    _, rendered = densify_fit(target, 3, stage_steps=(2, 3, 4), stats=stats)
    measured = float(((rendered - target) ** 2).mean())
    assert stats["candidates"] == 1
    assert stats["stage_steps"] == (2, 3, 4)
    assert stats["candidate_mse"] == [stats["mse"]]
    assert stats["mse"] == pytest.approx(measured, rel=0, abs=1e-15)


def test_empty_or_negative_schedule_refuses():
    target = np.zeros((8, 8))
    with pytest.raises(ValueError):
        densify_fit(target, 3, stage_steps=())
    with pytest.raises(ValueError):
        densify_fit(target, 3, stage_steps=(2, -1, 3))
