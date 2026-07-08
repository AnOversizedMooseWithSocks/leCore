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
