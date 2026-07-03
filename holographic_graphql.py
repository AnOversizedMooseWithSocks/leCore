"""holographic_graphql.py -- a GraphQL front door for the SCENE, the natural fit for nested data. Where SQL is
flat (rows and columns), a scene is a nested graph: an object has a name and a material and a transform, and the
transform has a position and a rotation. GraphQL's whole idea -- "ask for exactly the nested fields you want" --
maps cleanly onto the substrate: a nested field is bind(role, sub_record), so descending into `transform {
position }` is just unbinding the `transform` role to get the sub-record, then reading `position` from it.

WHY THIS EXISTS (leCore Query Interface backlog, Part 1 / Phase 4)
-----------------------------------------------------------------
The SQL skin (holographic_query) handles flat tables. The remaining half is the nested scene, and GraphQL is the
right shape for it. This resolver:
  1. parses a small GraphQL subset (a selection set with optional args and nested children),
  2. filters the scene's objects by a `where` argument, and
  3. returns EXACTLY the requested fields per object, recursing into nested selections --
so the client gets back the shape it asked for, no more.

THE VSA-NATIVE PART (kept honest): each object is encoded as a NESTED VSA record -- scalar categorical fields are
bind(role, value_atom), and a nested field like `transform` is bind(role, sub_record). So the nested selection
`transform { kind }` corresponds to unbind(record, transform) -> the sub-record -> unbind(sub, kind) -> cleanup:
"ask for the nested field" == "unbind exactly that chain of roles." `project_via_unbind` demonstrates that path
on categorical leaves (it recovers the same value the stored dict holds). NUMERIC/LIST fields (a position
[x,y,z]) are read from the stored object, not decoded -- the same honest exact/fuzzy fork as the SQL side: a
float has readback error, so we keep it exact rather than pretend to decode it.

Deterministic; NumPy + stdlib only. Reuses the kernel (bind/unbind/bundle/cleanup) and Vocabulary.
"""
import numpy as np

from holographic_ai import bind, unbind, bundle, cosine, Vocabulary
from holographic_query import QueryError


# --- encoding a nested scene ----------------------------------------------------------------------------------

def _encode_object(obj, role_vocab, value_vocab, dim):
    """Encode one object (a possibly-nested dict) as a VSA record: bind each CATEGORICAL field to its role, and
    each NESTED dict to its role as a sub-record (recursively), then bundle. Numeric/list values are skipped here
    -- they live in the stored object for exact reads (the honest fork)."""
    parts = []
    for key, val in obj.items():
        if isinstance(val, str):
            value_vocab.get(val)
            parts.append(bind(role_vocab.get(key), value_vocab.get(val)))
        elif isinstance(val, dict):
            sub = _encode_object(val, role_vocab, value_vocab, dim)   # nested field = bind(role, sub_record)
            parts.append(bind(role_vocab.get(key), sub))
    return bundle(parts) if parts else np.zeros(dim)


class Scene:
    """A scene as a list of nested objects, with each object also encoded as a nested VSA record. The stored
    objects are the exact source of truth for output; the records demonstrate the nested-bind structure and back
    the fuzzy/where matching."""

    def __init__(self, objects, dim=2048, seed=0):
        self.objects = [dict(o) for o in objects]                 # the stored (exact) nested objects
        self.dim = dim
        self.role_vocab = Vocabulary(dim, seed)
        self.value_vocab = Vocabulary(dim, seed + 1)
        self.records = np.stack([_encode_object(o, self.role_vocab, self.value_vocab, dim) for o in self.objects]) \
            if self.objects else np.zeros((0, dim))

    def project_via_unbind(self, obj_index, path):
        """Recover a CATEGORICAL leaf field by unbinding along the nested role path (e.g. ['transform', 'kind'])
        -- the VSA-native version of a nested GraphQL selection. Returns the cleaned-up filler name (or None if
        the recovery is too weak). Demonstrates that descending the selection == unbinding the roles."""
        rec = self.records[obj_index]
        for role in path[:-1]:
            rec = unbind(rec, self.role_vocab.get(role))          # descend into the sub-record
        leaf = unbind(rec, self.role_vocab.get(path[-1]))
        name, conf = self.value_vocab.cleanup(leaf)
        return name if conf > 0.12 else None


# --- a small GraphQL parser -----------------------------------------------------------------------------------

def _tokenize(query):
    import re
    return re.findall(r'\{|\}|\(|\)|:|,|"[^"]*"|[A-Za-z_][A-Za-z0-9_]*', query)


def parse_graphql(query):
    """Parse a small GraphQL subset into a selection tree. A field is: name, optional args in (), optional nested
    children in {}. Example:
        { objects(where: {material: "gold"}) { name transform { position } } }
    Returns a list of {"name", "args", "children"}. Not full GraphQL -- a readable subset, on purpose."""
    toks = _tokenize(query)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat(expected=None):
        if pos[0] >= len(toks):
            raise QueryError("unexpected end of GraphQL query")
        cur = toks[pos[0]]
        pos[0] += 1
        if expected is not None and cur != expected:
            raise QueryError("expected %r, got %r" % (expected, cur))
        return cur

    def parse_value():
        if peek() == "{":                                          # a nested object literal, e.g. {material: "gold"}
            eat("{")
            obj = {}
            while peek() != "}":
                k = eat()
                eat(":")
                obj[k] = parse_value()
                if peek() == ",":
                    eat(",")
            eat("}")
            return obj
        v = eat()
        return v[1:-1] if v[:1] == '"' else v                     # strip quotes on a string literal

    def parse_args():
        eat("(")
        args = {}
        while peek() != ")":
            k = eat()
            eat(":")
            args[k] = parse_value()
            if peek() == ",":
                eat(",")
        eat(")")
        return args

    def parse_selection_set():
        eat("{")
        fields = []
        while peek() != "}":
            fields.append(parse_field())
        eat("}")
        return fields

    def parse_field():
        name = eat()
        args = parse_args() if peek() == "(" else {}
        children = parse_selection_set() if peek() == "{" else []
        return {"name": name, "args": args, "children": children}

    if peek() == "query":                                         # allow an optional leading 'query' keyword
        eat("query")
    return parse_selection_set()


# --- the resolver ---------------------------------------------------------------------------------------------

def _matches(obj, where):
    """A where filter over an object's stored fields (exact). Supports top-level and one level of nested keys via
    a dotted... no: keep it simple -- match top-level scalar fields only (the common scene filter)."""
    return all(obj.get(k) == v for k, v in where.items())


def _project(obj, children):
    """Return EXACTLY the requested fields of one object, recursing into nested selections. Scalar/numeric fields
    come from the stored object (exact); a field with children descends into the stored sub-object and projects
    only its requested subfields -- 'you get back the shape you asked for.'"""
    out = {}
    for child in children:
        name = child["name"]
        if child["children"]:
            sub = obj.get(name, {})
            out[name] = _project(sub, child["children"]) if isinstance(sub, dict) else sub
        else:
            out[name] = obj.get(name)
    return out


def resolve(scene, query):
    """Run a GraphQL query against a Scene. The top-level fields name collections (e.g. `objects`); each applies
    its `where` filter and projects the requested (possibly nested) fields per matching object. Returns a dict
    shaped like the query."""
    result = {}
    for field in parse_graphql(query):
        where = field["args"].get("where", {})
        matched = [o for o in scene.objects if _matches(o, where)]
        result[field["name"]] = [_project(o, field["children"]) for o in matched]
    return result


def _selftest():
    """A nested scene resolves to exactly the requested shape; a where filter selects objects; nested selections
    return only the chosen subfields; and the VSA-native claim holds -- a categorical leaf is recoverable by
    unbinding the role chain (nested selection == nested unbind)."""
    objects = [
        {"id": "o1", "name": "ring", "material": "gold",
         "transform": {"kind": "rigid", "position": [1.0, 0.0, 0.0]}},
        {"id": "o2", "name": "pipe", "material": "copper",
         "transform": {"kind": "rigid", "position": [0.0, 2.0, 0.0]}},
        {"id": "o3", "name": "coin", "material": "gold",
         "transform": {"kind": "static", "position": [3.0, 0.0, 0.0]}},
    ]
    scene = Scene(objects, dim=4096, seed=0)

    # (1) filter + nested projection: exactly the requested shape
    q = '{ objects(where: {material: "gold"}) { name transform { position } } }'
    res = resolve(scene, q)
    names = [o["name"] for o in res["objects"]]
    assert names == ["ring", "coin"]                              # only the gold objects
    first = res["objects"][0]
    assert set(first.keys()) == {"name", "transform"}             # ONLY the requested top-level fields
    assert set(first["transform"].keys()) == {"position"}         # ONLY the requested nested field (not 'kind')
    assert first["transform"]["position"] == [1.0, 0.0, 0.0]

    # (2) a different selection returns a different shape from the same scene
    q2 = "{ objects { id material } }"
    res2 = resolve(scene, q2)
    assert set(res2["objects"][0].keys()) == {"id", "material"} and len(res2["objects"]) == 3

    # (3) VSA-native: a categorical leaf is recovered by unbinding the role chain (nested selection == unbind)
    assert scene.project_via_unbind(0, ["material"]) == "gold"
    assert scene.project_via_unbind(0, ["transform", "kind"]) == "rigid"    # descend via unbind, then read
    assert scene.project_via_unbind(1, ["transform", "kind"]) == "rigid"

    # (4) deterministic
    assert resolve(scene, q) == resolve(scene, q)

    print("holographic_graphql selftest OK: '{ objects(where:{material:\"gold\"}) { name transform { position } } }'"
          " returns %s with only the requested nested fields; a categorical leaf round-trips through the nested "
          "unbind chain (transform->kind = 'rigid'); deterministic" % names)


if __name__ == "__main__":
    _selftest()
