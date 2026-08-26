"""Forecasting sweep (sec.5.5): the scheduler cost model as a forecaster -- calibrated (measured) capacity."""
import numpy as np
from holographic.scene_and_pipeline.holographic_superschedule import calibrated_capacity, should_superpose, pack_capacity


def test_measured_capacity_moves_with_target():
    cap90, curve = calibrated_capacity(512, gated=True, target_recall=0.9, seed=0)
    cap99, _ = calibrated_capacity(512, gated=True, target_recall=0.99, seed=0)
    assert cap99 <= cap90            # stricter target -> smaller safe load
    assert cap90 >= 1
    assert dict(curve)[1] >= 0.99    # perfect recall at the smallest load


def test_recall_falls_beyond_capacity():
    cap, curve = calibrated_capacity(512, gated=True, target_recall=0.9, seed=0)
    d = dict(curve)
    # the last measured point is at/just past the crossover, so its recall is below target
    last_n = max(d)
    assert d[last_n] < 0.9 or last_n == max(4, int(0.30 * 512))


def test_should_superpose_gate():
    dim = 512
    cap, _ = calibrated_capacity(dim, gated=True, target_recall=0.9, seed=0)
    assert should_superpose(max(1, cap // 2), dim, target_recall=0.9) is True
    assert should_superpose(cap + max(5, cap), dim, target_recall=0.9) is False


def test_deterministic():
    assert calibrated_capacity(256, seed=0)[0] == calibrated_capacity(256, seed=0)[0]
