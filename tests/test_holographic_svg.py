"""Holographic vector-graphics (SVG) scenes (holographic_svg.py): a scene of typed primitives encodes into one
hypervector and decodes back, two scenes morph by interpolating their vectors, the composed-manifold diffusion
generates novel scenes, and any scene renders as crisp, resolution-independent SVG."""
import numpy as np

from holographic.io_and_interop.holographic_svg import HolographicSVG, _selftest


def test_selftest_passes():
    _selftest()


def _scene():
    return [(1, 0.30, 0.35, 0.12, 1), (0, 0.70, 0.62, 0.10, 0), (2, 0.52, 0.40, 0.09, 2)]


def test_round_trip_recovers_attributes():
    svg = HolographicSVG(dim=4096, seed=0)
    scene = _scene()
    dec = svg.decode(svg.encode(scene), len(scene))
    assert all(a[0] == b[0] for a, b in zip(scene, dec))          # type exact (cleanup)
    assert all(a[4] == b[4] for a, b in zip(scene, dec))          # colour exact (cleanup)
    perr = np.mean([abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3]) for a, b in zip(scene, dec)]) / 3
    assert perr < 0.05                                            # position/size close (continuous decode)


def test_morph_midpoint_lies_between_endpoints():
    svg = HolographicSVG(dim=4096, seed=0)
    A = [(1, 0.20, 0.30, 0.13, 1), (0, 0.75, 0.70, 0.11, 3)]
    B = [(1, 0.75, 0.30, 0.09, 2), (0, 0.25, 0.70, 0.13, 0)]
    mid = svg.morph(A, B, steps=5)[2]
    # the first primitive's x moved from ~0.20 toward ~0.75 -> the midpoint x is genuinely between them
    assert A[0][1] + 0.1 < mid[0][1] < B[0][1] - 0.1


def test_generate_is_diverse_and_deterministic():
    svg = HolographicSVG(dim=4096, seed=0)
    scenes = [tuple(svg.generate(k=4, seed=s)) for s in range(6)]
    assert len(set(scenes)) >= 4                                  # distinct novel scenes
    assert tuple(svg.generate(k=4, seed=2)) == scenes[2]          # deterministic in the seed


def test_to_svg_is_well_formed_and_covers_every_primitive():
    svg = HolographicSVG(dim=1024, seed=1)
    scene = [(0, 0.3, 0.3, 0.1, 0), (1, 0.7, 0.7, 0.1, 1), (2, 0.5, 0.5, 0.1, 2)]
    out = svg.to_svg(scene)
    assert out.startswith("<svg") and out.rstrip().endswith("</svg>")
    assert 'viewBox=' in out                                      # resolution-independent (no baked pixel size)
    assert out.count("<rect") == 2 and out.count("<circle") == 1 and out.count("<polygon") == 1


def test_render_geometry_scales_with_size():
    # the SAME scene at two viewBox sizes draws proportionally -- it is vector, not a fixed raster
    svg = HolographicSVG(dim=512, seed=0)
    scene = [(1, 0.5, 0.5, 0.2, 0)]
    small = svg.to_svg(scene, size=100)
    big = svg.to_svg(scene, size=400)
    assert 'r="20.00"' in small and 'r="80.00"' in big           # radius 0.2 -> 20 at 100, 80 at 400


def test_too_many_primitives_rejected():
    svg = HolographicSVG(dim=512, seed=0, max_slots=2)
    try:
        svg.encode([(0, 0.5, 0.5, 0.1, 0)] * 3)
        assert False, "should reject more primitives than slots"
    except ValueError:
        pass
