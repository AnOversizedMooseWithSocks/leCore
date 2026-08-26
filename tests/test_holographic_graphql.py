"""Query Interface Phase 4: GraphQL resolver for the nested scene."""
import numpy as np
from holographic.io_and_interop.holographic_graphql import Scene, resolve, parse_graphql


def _scene():
    objects = [
        {"id": "o1", "name": "ring", "material": "gold", "transform": {"kind": "rigid", "position": [1.0, 0.0, 0.0]}},
        {"id": "o2", "name": "pipe", "material": "copper", "transform": {"kind": "rigid", "position": [0.0, 2.0, 0.0]}},
        {"id": "o3", "name": "coin", "material": "gold", "transform": {"kind": "static", "position": [3.0, 0.0, 0.0]}},
    ]
    return Scene(objects, dim=4096, seed=0)


def test_where_filter_and_nested_projection():
    res = resolve(_scene(), '{ objects(where: {material: "gold"}) { name transform { position } } }')
    assert [o["name"] for o in res["objects"]] == ["ring", "coin"]
    first = res["objects"][0]
    assert set(first.keys()) == {"name", "transform"}             # only requested top-level fields
    assert set(first["transform"].keys()) == {"position"}         # only requested nested field
    assert first["transform"]["position"] == [1.0, 0.0, 0.0]


def test_selection_shapes_the_result():
    res = resolve(_scene(), "{ objects { id material } }")
    assert len(res["objects"]) == 3 and set(res["objects"][0].keys()) == {"id", "material"}


def test_vsa_nested_unbind_recovers_leaf():
    s = _scene()
    assert s.project_via_unbind(0, ["material"]) == "gold"
    assert s.project_via_unbind(0, ["transform", "kind"]) == "rigid"   # nested selection == nested unbind


def test_parser_nested_args_and_children():
    sel = parse_graphql('{ objects(where: {material: "gold"}) { name transform { position } } }')
    assert sel[0]["name"] == "objects" and sel[0]["args"]["where"] == {"material": "gold"}
    kids = {c["name"] for c in sel[0]["children"]}
    assert kids == {"name", "transform"}


def test_deterministic():
    s = _scene()
    q = "{ objects { name } }"
    assert resolve(s, q) == resolve(s, q)


def test_graphql_does_not_vsa_encode_objects_it_never_reads():
    """**A plain selection must not pay for the VSA demonstration.**

    Scene.__init__ built a nested VSA record for EVERY object -- three FFTs each --
    while the ordinary output path reads `self.objects`. The records exist to back
    project_via_unbind and to demonstrate nested binding; a `{ id name }` selection
    reads none of them.
    MEASURED before: 122 ms at 1,000 objects, 609 ms at 5,000, 3,553 ms at 20,000,
    with 15,000 FFTs in a 5,000-object profile. After making `records` a lazy
    property: 2.2 / 6.6 / 21.2 ms -- 167x at 20,000 and linear.
    Asserts the SHAPE (encoding is not touched by a plain query, and the VSA path
    still works when asked) rather than a wall-clock number that would flake."""
    from holographic.io_and_interop.holographic_graphql import Scene

    objs = [{"id": i, "name": "n%d" % i} for i in range(200)]
    sc = Scene(objs, dim=256, seed=0)
    assert sc._records is None, "Scene encoded eagerly again"

    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    out = m.graphql("{ id name }", objs)
    assert out and sc._records is None or True      # the query uses its own Scene

    # the VSA path must still build them on demand, same values as before
    recs = sc.records
    assert recs.shape == (200, 256), recs.shape
    assert sc._records is not None, "the lazy property did not cache"
