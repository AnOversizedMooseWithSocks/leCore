"""CircularEncoder (I2) at the seams: through the mind, the hour-of-day use it was named for, bind
compatibility with the engine's own circular convolution (rotation-as-bind on the circle), and a bundle of
encoded angles reading as a circular density with the mode where it was planted."""
import numpy as np

import lecore
from holographic.agents_and_reasoning.holographic_ai import bind


def test_the_hour_of_day_case_through_the_faculty():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    e = mind.circular_encoder(512, period=24.0)
    a, b, c = e.encode(23.9), e.encode(0.1), e.encode(12.0)
    assert float(a @ b) > 0.5                                    # 12 minutes apart across midnight
    assert float(a @ c) < 0.3                                    # half a day apart
    assert abs(e.decode(a) - 23.9) < 0.2 or abs(e.decode(a) - 23.9 - 24.0) < 0.2


def test_wrap_exactness_and_gap_only_similarity():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    e = mind.circular_encoder(1024, period=2 * np.pi)
    for x in (0.3, 2.9, 5.5):
        assert float(np.max(np.abs(e.encode(x) - e.encode(x + 2 * np.pi)))) < 1e-12
    # gap-only: every pair at the same circular distance reads the same cosine (within noise of the draw)
    g = 0.4
    cs = [float(e.encode(x) @ e.encode(x + g)) for x in (0.0, 1.7, 4.2, 6.0)]
    assert max(cs) - min(cs) < 1e-9, cs                          # shift-invariant by construction, exactly


def test_rotation_is_a_bind_on_the_circle():
    """The FPE property inherited on the circle: binding with encode(delta) rotates every encoded angle by
    delta -- bind(encode(x), encode(d)) == encode(x+d) -- because circular convolution adds the phases and
    the harmonics are shared. The engine's own bind, no special case."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    e = mind.circular_encoder(1024, period=2 * np.pi)
    x, d = 1.3, 0.9
    rotated = bind(e.encode(x), e.encode(d))
    target = e.encode(x + d)
    cos = float(rotated @ target / (np.linalg.norm(rotated) * np.linalg.norm(target) + 1e-12))
    assert cos > 0.999, cos


def test_a_bundle_of_angles_is_a_circular_density():
    """Sum encoded headings drawn around a mode; the decoded argmax of the bundle sits at the mode, across
    the wrap if that is where the mode lives -- the circular analogue of the kernel-density read the RBF
    ScalarEncoder was designed for on the line."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    e = mind.circular_encoder(1024, period=2 * np.pi, concentration=0.8)
    mode = 0.05                                                  # deliberately ON the wrap
    angles = (mode + 0.25 * rng.standard_normal(80)) % (2 * np.pi)
    bundle = np.sum([e.encode(a) for a in angles], axis=0)
    peak = e.decode(bundle)
    gap = min(abs(peak - mode), 2 * np.pi - abs(peak - mode))
    assert gap < 0.15, (peak, mode)
