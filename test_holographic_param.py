"""Connectable parameters: a value can be a constant OR a map / field / wired output."""
import numpy as np
from holographic_param import Param, resolve_param


def test_constant_and_field_and_map_resolve():
    pts = np.array([[0.2, 0.0], [0.8, 0.0], [0.5, 0.0]])
    assert np.allclose(resolve_param(0.3, pts), 0.3)                       # bare scalar
    assert np.allclose(resolve_param(Param(value=0.3), pts), 0.3)          # constant socket
    assert np.allclose(resolve_param(Param(field=lambda P: P[:, 0]), pts), [0.2, 0.8, 0.5])  # field
    mp = np.array([0.0, 1.0])                                              # a 2-cell map over x in [0,1]
    got = resolve_param(Param(map=mp, domain=(np.array([0.0, 0.0]), np.array([1.0, 1.0]))), pts)
    assert got.shape == (3,)


def test_source_wire_and_dangling_default():
    pts = np.array([[0.2, 0.0], [0.9, 0.0]])
    ctx = {"curv": Param(field=lambda P: P[:, 0] * 2.0)}
    assert np.allclose(resolve_param(Param(source="curv"), pts, ctx=ctx), [0.4, 1.8])  # follows the wire
    assert np.allclose(resolve_param(Param(source="missing", default=0.7), pts), 0.7)  # dangling -> default


def test_param_is_backward_compatible_scalar():
    # a faculty that used to take a number still works: a scalar resolves to itself
    assert float(resolve_param(0.5)) == 0.5
    assert resolve_param(Param(value=0.5)) == 0.5
