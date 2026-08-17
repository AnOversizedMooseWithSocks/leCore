"""Fur as an SDF shell: length is a DISTANCE, coverage is a FIELD.

Two symptoms recurred in every groom: the length was never the right scale, and coverage was
spotty. Both come from groom_hair's parameterisation -- an abstract `length` number and
`n_strands` scattered in an axis-aligned box. SOTA's field formulation removes both by
construction: Kajiya & Kay (SIGGRAPH 1989) render fur as a VOLUMETRIC TEXTURE, the production
lineage as "concentric layers from the skin outwards", and HISR (2024) states the hybrid in
SDF terms -- a hard SDF filled with "opaque materials" and outside it a translucent region
"with volume densities".
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_furshell as FS

SPHERE = lambda P: np.linalg.norm(np.atleast_2d(np.asarray(P, float)), axis=1) - 1.0


def test_length_is_a_distance_in_model_units():
    """THE SCALE FIX. Fur stops at exactly the requested offset, so 'length' can no longer
    mean something different on a mouse than on a head."""
    L = 0.15
    fur = FS.fur_shell(SPHERE, L, strand_scale=40.0)
    assert float(np.asarray(fur(np.array([[1.0 + L * 1.2, 0, 0]])))[0]) == 0.0
    ring = np.stack([np.full(400, 1.02), np.linspace(-0.4, 0.4, 400), np.zeros(400)], 1)
    assert float(np.asarray(fur(ring)).max()) > 0.0


def test_intent_survives_a_change_of_model_scale():
    """'Short fur' must stay short when the model is resized -- the control that was missing.
    A 10x model gets 10x the fur length for the same stated fraction."""
    small = FS.fur_length_for(((-1,) * 3, (1,) * 3), 0.04)
    big = FS.fur_length_for(((-10,) * 3, (10,) * 3), 0.04)
    assert abs(big / small - 10.0) < 1e-9


def test_coverage_does_not_depend_on_sampling():
    """THE SPOTTY FIX. Density is evaluated per point, so the covered FRACTION is invariant
    to how many samples you take -- unlike a strand count over a box, where coverage depends
    on how much of the box the surface fills."""
    L = 0.15
    fur = FS.fur_shell(SPHERE, L, strand_scale=40.0)

    def frac(n):
        rng = np.random.default_rng(0)
        th = rng.uniform(0, np.pi, n)
        ph = rng.uniform(0, 2 * np.pi, n)
        rr = 1.0 + L * 0.35
        P = np.stack([rr * np.sin(th) * np.cos(ph), rr * np.cos(th),
                      rr * np.sin(th) * np.sin(ph)], 1)
        return float((np.asarray(fur(P)) > 0).mean())
    assert abs(frac(3000) - frac(6000)) < 0.06


def test_density_and_length_fields_modulate_the_shell():
    """One shell, many regions: a density field of zero grows nothing, and a length field
    shortens the fur where it says to -- the beard/scalp case without two grooms."""
    L = 0.2
    none = FS.fur_shell(SPHERE, L, density_fn=lambda P: np.zeros(len(P)))
    P = np.stack([np.full(200, 1.05), np.linspace(-0.3, 0.3, 200), np.zeros(200)], 1)
    assert float(np.asarray(none(P)).max()) == 0.0
    short = FS.fur_shell(SPHERE, L, length_fn=lambda P: np.full(len(P), 0.2))
    full = FS.fur_shell(SPHERE, L)
    far = np.array([[1.0 + L * 0.5, 0, 0]])
    assert float(np.asarray(short(far))[0]) == 0.0        # past the SHORTENED tip
    assert np.asarray(full(far))[0] >= 0.0                # but within the full length


def test_reach_guard_refuses_an_overlong_shell():
    """A shell thicker than the reach self-intersects (L3's bound). The guard must refuse."""
    assert not FS.shell_is_valid(0.5, 0.3)[0]
    assert FS.shell_is_valid(0.1, 0.3)[0]
