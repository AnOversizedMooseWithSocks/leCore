"""Modeling-app backlog: measurement + units -- dimensioned mesh integrals measured from geometry."""
import numpy as np
from holographic.misc.holographic_metrology import surface_area, volume, bounding_box, centroid, distance, edge_length, angle_at, dihedral_angle, degrees, _unit_cube


def test_area_and_volume_of_cube():
    cube = _unit_cube()
    assert abs(surface_area(cube).to("m2") - 6.0) < 1e-9
    assert abs(volume(cube).to("m3") - 1.0) < 1e-9


def test_unit_conversion_is_one_multiply():
    cube = _unit_cube()
    assert abs(volume(cube).to("L") - 1000.0) < 1e-6         # 1 m^3 = 1000 L
    d = edge_length(cube, 0, 1)
    assert abs(d.to("ft") - 3.280839895) < 1e-6              # 1 m in feet


def test_dimensional_algebra_refuses_nonsense():
    cube = _unit_cube()
    A = surface_area(cube); d = edge_length(cube, 0, 1)
    assert abs((A + A).to("m2") - 12.0) < 1e-9               # area + area OK
    try:
        _ = A + d; assert False                             # area + length must raise
    except ValueError:
        pass
    try:
        _ = volume(cube).to("m2"); assert False             # volume expressed as area must raise
    except ValueError:
        pass


def test_bounding_box_and_centroid():
    cube = _unit_cube()
    bb = bounding_box(cube)
    assert np.allclose(bb.size, [1, 1, 1])
    assert abs(bb.diagonal.to("m") - np.sqrt(3)) < 1e-9
    assert np.allclose(bb.center, [0.5, 0.5, 0.5])
    assert np.allclose(centroid(cube), [0.5, 0.5, 0.5], atol=1e-9)


def test_distance_dimensioned():
    d = distance([0, 0, 0], [3, 4, 0])
    assert abs(d.to("m") - 5.0) < 1e-9 and abs(d.to("cm") - 500.0) < 1e-6


def test_angles():
    assert abs(degrees(angle_at((1, 0, 0), (0, 0, 0), (0, 1, 0))) - 90.0) < 1e-9
    assert abs(degrees(dihedral_angle(_unit_cube(), 0, 4)) - 90.0) < 1e-6


def test_deterministic():
    assert surface_area(_unit_cube()).to("m2") == surface_area(_unit_cube()).to("m2")
