"""Two doors opened at sweep 134's close-out, and the up/down/sideways check on both.

The above/below matrix left 18 catalog cards naming module-level functions -- importable
in-process, NOT callable over /invoke. These two are the pair that shows the difference between
the two correct remedies:

  aces_tonemap        WIRED. There was no door at all, and the demo-scene review named it the
                      highest value per line of the 18: a physically-lit render is FLOAT and
                      clips to white, so this is the last step before anything is viewable.
  element_flame_color RE-POINTED, not wired. mind.element('Na')['flame_color'] already returns
                      it, so a second door would have been a REDUNDANT door -- a discoverability
                      tax, and worse than the mis-pointed card it would have "fixed".
"""
import numpy as np
import lecore
import pytest


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


def test_aces_tonemap_is_reachable_as_a_door_and_actually_tonemaps(mind):
    """A value of 4.0 is out of display range; after the transform it must be inside it, or the
    door is wired to something that does not do the job."""
    out = np.asarray(mind.aces_tonemap(np.array([[[0.1, 0.5, 4.0]]])))
    assert out.shape == (1, 1, 3)
    assert out[0, 0, 2] < 1.0, "an HDR 4.0 must come back inside display range"
    assert (out >= 0.0).all()


def test_aces_tonemap_is_deterministic(mind):
    img = np.array([[[0.1, 0.5, 4.0], [2.0, 0.2, 0.05]]])
    assert np.array_equal(np.asarray(mind.aces_tonemap(img)),
                          np.asarray(mind.aces_tonemap(img)))


def test_element_flame_color_reaches_through_the_door_that_already_existed(mind):
    """THE RE-POINT, pinned. The capability was never missing -- only the card's pointer was. If
    this ever stops working, the card is lying again and the remedy is the card, not a new verb."""
    rgb = mind.element("Na")["flame_color"]
    assert len(rgb) == 3 and all(0.0 <= float(c) <= 1.0 for c in rgb)


def test_UP_DOWN_SIDEWAYS_on_aces_tonemap(mind):
    """All three directions, pinned. It is elementwise, so it is polymorphic by construction --
    which is worth ASSERTING rather than assuming, because a later 'optimisation' that reshapes
    to (H,W,3) would silently take two of these away."""
    img = np.array([[[0.1, 0.5, 4.0], [2.0, 0.2, 0.05]]])
    assert np.asarray(mind.aces_tonemap(img[:, :1])).shape == (1, 1, 3)          # DOWN: one pixel
    assert np.asarray(mind.aces_tonemap(np.stack([img, img * 2.0]))).shape == (2, 1, 2, 3)  # UP
    assert np.asarray(mind.aces_tonemap(np.array([0.1, 0.5, 4.0]))).shape == (3,)  # SIDEWAYS


def test_the_SIDEWAYS_direction_sweep_133_named_is_now_CLOSED(mind):
    """Sweep 133's up/down/sideways check found feedback_step wearing the FIELD costume and not
    the SEQUENCE one -- a 1-D hypervector raised on the shape unpack, and the check's own rule is
    that a missed direction is a missed faculty. Sweep 134 closed it. All three ranks now pass,
    and this test is the one that would go red if the sequence costume were ever dropped again."""
    assert np.asarray(mind.feedback_step(np.arange(16.0), zoom=1.0, decay=0.9)).shape == (16,)
    assert np.asarray(mind.feedback_step(np.zeros((8, 8)), zoom=1.0, decay=0.9)).shape == (8, 8)
    assert np.asarray(mind.feedback_step(np.zeros((8, 8, 3)), zoom=1.0, decay=0.9)).shape == (8, 8, 3)
