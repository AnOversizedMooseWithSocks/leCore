"""holographic_iokinds.py -- the coarse DATATYPE vocabulary a capability can CONSUME or PRODUCE (S3.1).

WHY A CLOSED, COARSE ENUM (not free-form type strings): the point of io-shape tagging is to answer two questions --
"what can I run on this <thing> I'm holding?" (filter by `consumes`) and "what gets me from A to B?" (chain
`produces`->`consumes`). Those questions only work if the kinds are a SMALL, SHARED vocabulary: if one capability
says it produces 'point cloud' and the next says it consumes 'points', the pipeline never links. So the kinds are a
frozen set, coarse on purpose -- 'points' covers point clouds / particle sets / splat centers alike; the fine
distinctions live in the capability's own docstring, not in the routing vocabulary.

GROUNDED IN MEMBERS, not aspiration: every kind below is a datatype that many capabilities in the tree actually
take or return (measured: field 199, mesh 151, sdf 95, image 78, points 51, transform 50, scalar 35, selection 18,
curve 22, skeleton/bone a handful, plus sdf_scene as the composed-scene container). A kind with no members would be
a dead menu entry -- the io equivalent of a dark module -- so the set stays tied to what exists.

Deterministic, stdlib only. This module holds ONLY the vocabulary + validation; the consumes/produces tags live on
catalog entries (holographic_catalog) and the filtering/pipeline logic lives on the catalog + faculties.
"""

# The closed kind vocabulary. Coarse by design (see module docstring). ORDER is stable for deterministic listing.
IO_KINDS = (
    "mesh",        # a triangle/quad mesh: vertices + faces (the polygon half of the modeler)
    "points",      # an unordered point set: point clouds, particle positions, splat centers (coarse: all "points")
    "sdf",         # a signed-distance function / field callable or node (the implicit-surface half)
    "sdf_scene",   # a COMPOSED sdf scene: several sdf parts + materials, the whole implicit scene as one object
    "field",       # a sampled grid field over space: density, velocity, heat -- anything read at continuous coords
    "image",       # a 2-D raster: an RGB(A) or scalar image / texture
    "hypervector", # a VSA/HRR vector (or bundle of them): the engine's native representation
    "transform",   # an affine transform: a matrix, a (R,t) pair, or a delta -- something you APPLY to geometry
    "selection",   # a sub-object selection: a set of vertex/edge/face indices with a mode
    "scalar",      # a single number or small tuple of numbers: a measurement, a cost, a distance
    "curve",       # a 1-D curve/spline: control points + a parameterization (splines, knots, strokes)
    "skeleton",    # a bone hierarchy / armature: joints + parenting (drives skinning)
    "timeseries",  # a sampled-in-TIME sequence: (t, y) samples or a value trace -- light curves, sim traces, audio
                   #   param streams. Distinct from `field` (space) and `scalar` (one number); it is what a
                   #   periodogram/phase-fold consumes and what nbody energy-over-time / audio_param_bus produce.
    "spectrum",    # a per-WAVELENGTH / per-frequency reading: SED samples, Stokes-vs-lambda, receptor responses --
                   #   what observe_spectrum / falsecolor_spectral / rm_synthesis take as their spectral axis.
)

_IO_KINDS_SET = frozenset(IO_KINDS)


def is_kind(k):
    """True iff `k` is a known io kind. The one place the vocabulary is checked -- callers validate `consumes` /
    `produces` tuples against this so a typo ('point' vs 'points') is caught at registration, not silently at
    pipeline time (a mismatched kind that never links is the exact failure the closed vocabulary prevents)."""
    return k in _IO_KINDS_SET


def validate_kinds(kinds, where=""):
    """Validate a tuple of kind strings; raise ValueError naming the offender (and `where`) on the first unknown.
    Returns the kinds unchanged on success, so it can wrap a value inline. Empty tuple is valid ('unspecified')."""
    for k in kinds or ():
        if not is_kind(k):
            raise ValueError("unknown io kind %r%s -- must be one of %s" % (
                k, (" in " + where) if where else "", ", ".join(IO_KINDS)))
    return tuple(kinds or ())


def _selftest():
    """Contracts:
    1. the vocabulary is a closed, non-empty, de-duplicated tuple.
    2. is_kind accepts every listed kind and rejects a near-miss ('point', 'meshes').
    3. validate_kinds passes a good tuple and raises on a typo, naming the offender.
    """
    assert len(IO_KINDS) == len(set(IO_KINDS)) and len(IO_KINDS) >= 10
    assert all(is_kind(k) for k in IO_KINDS)
    assert not is_kind("point") and not is_kind("meshes") and not is_kind("")   # coarse != sloppy
    assert validate_kinds(("mesh", "points")) == ("mesh", "points")
    assert validate_kinds(()) == ()
    try:
        validate_kinds(("mesh", "point"), where="test")                         # 'point' is the classic typo
        raise AssertionError("expected ValueError on the 'point' typo")
    except ValueError as e:
        assert "point" in str(e) and "test" in str(e), str(e)
    print("holographic_iokinds selftest OK: %d closed kinds; is_kind accepts them and rejects near-misses; "
          "validate_kinds catches the 'point'/'points' typo by name" % len(IO_KINDS))


if __name__ == "__main__":
    _selftest()
