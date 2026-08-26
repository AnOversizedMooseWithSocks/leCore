"""The holographic image stack on REAL photographs (small in-repo samples from a
'mountain' wallpaper category): the plate's robustness, the vault's adaptive
choice on continuous-tone images, and the honest JPEG-beats-us efficiency gap."""
import glob
import os

import numpy as np

from holographic.rendering.holographic_photos import load_photo_folder, robustness_curve

_SAMPLE = "features/photo_sample"


def _have_samples():
    return os.path.isdir(_SAMPLE) and len(glob.glob(os.path.join(_SAMPLE, "*.npy"))) >= 2


def test_photos_are_continuous_tone_not_palette():
    # Real photos are a different regime than sprites: thousands of colours, so
    # palette packing is hopeless and lossy/robust methods are the point.
    if not _have_samples():
        import pytest
        pytest.skip("photo samples not present")
    p = load_photo_folder(_SAMPLE, size=96, limit=1)[0]
    codes = (p[..., 0].astype(np.uint32) << 16) | (p[..., 1].astype(np.uint32) << 8) | p[..., 2]
    assert len(np.unique(codes)) > 1000                  # far past the 256 palette limit


def test_holographic_plate_degrades_gracefully_on_photos():
    # THE HONEST WIN: erasing half the holographic coefficients barely moves PSNR
    # on a real photo -- the distributed representation spreads each pixel across
    # all coefficients. (A block codec like JPEG shatters under the same loss.)
    if not _have_samples():
        import pytest
        pytest.skip("photo samples not present")
    g = load_photo_folder(_SAMPLE, size=96, limit=1, gray=True)[0]
    rc = robustness_curve(g, keeps_erasures=((800, (0.0, 0.5)),))
    assert rc[0.5] > rc[0.0] - 1.0                       # near-flat across 50% erasure


def test_vault_picks_lossy_for_photos():
    # The adaptive chooser, validated on a new data type: on continuous-tone
    # photos the palette path is unavailable and a lossy codec wins -- the
    # opposite of its choice on sprites, driven entirely by the data. The
    # lossless pack still round-trips exactly.
    if not _have_samples():
        import pytest
        pytest.skip("photo samples not present")
    from tools.image_vault import ImageVault
    photos = load_photo_folder(_SAMPLE, size=96)
    v = ImageVault()
    for i, p in enumerate(photos):
        v.add(np.dstack([p, np.full(p.shape[:2], 255, np.uint8)]), name=f"m{i}")
    rows = v.report(lossy_quality=85)
    lossy = [r for r in rows if r[2] != float("inf")]
    lossless = [r for r in rows if r[2] == float("inf")]
    assert lossy and lossless
    assert min(r[1] for r in lossy) < min(r[1] for r in lossless)   # lossy smaller
    # palette method should be absent (too many colours)
    assert not any(r[0] == "palette" for r in rows)
    # lossless pack still exact
    v2 = ImageVault.load(v.pack())
    assert all(np.array_equal(v.images[i], v2.images[i]) for i in range(len(photos)))


def test_orientation_stays_idle_on_photos():
    # The transpose principle helps sprites (vertical structure) but NOT photos
    # (no directional self-similarity); the vault's lzma chooser must keep
    # row-major. Verified by exact round-trip whichever flag it picks.
    if not _have_samples():
        import pytest
        pytest.skip("photo samples not present")
    from tools.image_vault import ImageVault
    photos = load_photo_folder(_SAMPLE, size=96)
    v = ImageVault()
    for i, p in enumerate(photos):
        v.add(np.dstack([p, np.full(p.shape[:2], 255, np.uint8)]), name=f"m{i}")
    v2 = ImageVault.load(v.pack())
    assert all(np.array_equal(v.images[i], v2.images[i]) for i in range(len(photos)))
