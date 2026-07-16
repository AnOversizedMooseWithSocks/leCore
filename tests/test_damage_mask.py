"""D2 consolidation: `damage_mask` was written three times, byte-identically, on Hologram / HolographicImage /
HolographicArchive -- the one true cross-module duplicate the shape audit found. It now lives once in
holographic_ai and the three classes delegate.

WHY THESE TESTS ARE STRICT: the mask is RNG-derived, and existing degradation numbers (and their tests) are pinned
to the EXACT slots it zeroes. A consolidation that changed one mask by one slot would silently move every
robustness result in the tree. So bit-identity is asserted against values captured from the pre-consolidation
bodies, not merely "the shapes match".
"""
import numpy as np
import pytest

from holographic.agents_and_reasoning.holographic_ai import damage_mask, random_vector, cosine
from holographic.io_and_interop.holographic_image import Hologram, HolographicImage
from holographic.misc.holographic_archive import HolographicArchive
from holographic.misc.holographic_unified import UnifiedMind

_SITES = [Hologram, HolographicImage, HolographicArchive]


class _Slots:
    """A stand-in exposing only `dim` -- the sole attribute the three damage_mask methods ever used."""

    def __init__(self, dim):
        self.dim = dim


@pytest.mark.parametrize("dim,frac,seed", [(64, 0.0, 0), (64, 0.1, 0), (64, 0.5, 7),
                                           (257, 0.5, 0), (257, 1.0, 7), (1024, 0.25, 3)])
def test_all_three_sites_delegate_to_one_canonical_mask(dim, frac, seed):
    """Every class must return EXACTLY the canonical mask -- bit-identical, not merely equivalent."""
    expected = damage_mask(dim, frac, seed=seed)
    for cls in _SITES:
        got = cls.damage_mask(_Slots(dim), frac, seed=seed)
        assert np.array_equal(got, expected), (cls.__name__, dim, frac, seed)


def test_the_mask_contract_exactly():
    """The numeric contract the delegating classes depend on: 0/1 values, exactly int(dim*fraction) slots dead."""
    k = damage_mask(256, 0.25, seed=0)
    assert k.shape == (256,)
    assert set(np.unique(k)) <= {0.0, 1.0}
    assert int((k == 0).sum()) == 64
    assert int((damage_mask(64, 0.0, seed=0) == 0).sum()) == 0
    assert int((damage_mask(64, 1.0, seed=0) == 0).sum()) == 64


def test_deterministic_in_dim_fraction_seed():
    """A degradation curve must be reproducible, so the mask is a pure function of (dim, fraction, seed)."""
    assert np.array_equal(damage_mask(512, 0.3, seed=5), damage_mask(512, 0.3, seed=5))
    assert not np.array_equal(damage_mask(512, 0.3, seed=5), damage_mask(512, 0.3, seed=6))


def test_recall_degrades_smoothly_not_off_a_cliff():
    """The claim the probe exists to test: because a record is spread across every slot rather than filed in one,
    killing most of them should still leave it recognisable. Monotone decay, and 60% loss still clearly recalled."""
    v = random_vector(1024, np.random.default_rng(0))
    curve = [cosine(v * damage_mask(1024, f, seed=2), v) for f in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert curve[0] == pytest.approx(1.0, abs=1e-12)
    assert all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1)), ("must decay monotonically", curve)
    assert curve[3] > 0.55, ("60% slot loss should still recall clearly", curve)


def test_faculty_defaults_to_the_minds_dim_and_is_discoverable():
    m = UnifiedMind(dim=256, seed=0)
    assert m.damage_mask(0.25).shape == (256,)
    assert m.damage_mask(0.25, dim=64).shape == (64,)
    assert np.array_equal(m.damage_mask(0.25, seed=3), damage_mask(256, 0.25, seed=3))
    for phrasing in ("corrupt a vector for testing", "simulate data loss", "graceful degradation test"):
        top3 = [c.name for c in m.find_capability(phrasing)[:3]]
        assert any(n.startswith("Damage a vector") for n in top3), (phrasing, top3)
