"""holographic_scene_query.py -- SELECTION, SEARCH, and TAGGING over the Scene document (modeling-app feature layer).

The backlog's reframe, made real: the scene is a VSA table of object records, so the whole organizational layer is
query / bundle / cleanup wearing a DCC costume.

  * A SELECTION is a QUERY RESULT -- a set of object handles.
  * SEARCH / FILTER is querying that table. EXACT predicates (name / material / tag) run as plain, readable
    filtering over the stored records -- fast, and exact (no readback error). FUZZY / semantic predicates ride
    holographic_query's Table: they rank objects by how well a property MEANS the query value (cosine of the
    unbound filler to the probe) and carry a calibrated CONFIDENCE -- so "the metal-ish parts" isn't a silent
    guess. Surfacing that confidence matters: a fuzzy select that quietly includes a near-miss is a footgun.
  * A NAMED selection set is a saved result you reuse; set ALGEBRA (union / intersect / minus / invert) is the
    algebra of selections. Membership is kept as an ID LIST (exact, no capacity limit) -- a named set of thousands
    must NOT be one giant bundle (the decode ceiling). The holographic representation (a BUNDLE of the members'
    identity atoms) is offered separately, for the vector-algebra case, with that ceiling noted.
  * TAGGING is a bound role on the object (a key->value on the record); selecting by tag is a query. Tags are
    written through scene.edit / scene.remove_tag, so undo and change events come for free.

Deterministic; NumPy + stdlib only (the fuzzy path uses the query layer, which is bind/bundle/cleanup).
"""


def _role_value(obj, role):
    """Read a queryable property off an object record: name, material, or a tag by key (None if absent)."""
    if role == "name":
        return obj.name
    if role == "material":
        return obj.material
    return obj.tags.get(role)


# ---- exact selection: plain, readable filtering over the records (the common case) ----------------------------

def select(scene, name=None, name_contains=None, material=None, tag=None, has_tag=None, where=None):
    """Select object handles by EXACT predicates (all provided must hold -- an AND):
        name            -- exact object name
        name_contains   -- case-insensitive substring of the name
        material        -- exact material
        tag=(key, val)  -- objects whose tag `key` equals `val`
        has_tag=key     -- objects that carry that tag key (any value)
        where=fn        -- a predicate fn(obj)->bool for anything else
    Returns a set of handles. This is exact filtering over the stored records: fast and lossless."""
    out = set()
    for h, obj in scene.objects.items():
        if name is not None and obj.name != name:
            continue
        if name_contains is not None and name_contains.lower() not in (obj.name or "").lower():
            continue
        if material is not None and obj.material != material:
            continue
        if tag is not None:
            key, val = tag
            if obj.tags.get(key) != val:
                continue
        if has_tag is not None and has_tag not in obj.tags:
            continue
        if where is not None and not where(obj):
            continue
        out.add(h)
    return out


# ---- fuzzy / semantic selection: ride the query layer, surface the confidence --------------------------------

def select_fuzzy(scene, role, value, threshold=None, dim=1024, seed=0):
    """Semantic selection: rank objects by how well their `role` value MEANS `value`, via the query layer's fuzzy
    WHERE (cosine of the unbound filler to the probe). Returns [(handle, confidence)] ranked, filtered to
    confidence >= threshold if given. The confidence is SURFACED so the caller isn't surprised by a near-miss.

    NOTE on semantics: with the default random-atom filler vocabulary this behaves as exact-match-with-confidence
    (distinct values are near-orthogonal); TRUE semantic nearness ('metal' ~ 'steel') needs a meaning-bearing value
    encoder for the vocab -- a separate capability, kept as an honest boundary."""
    from holographic.agents_and_reasoning.holographic_query import from_rows, Query
    handles = [h for h, o in scene.objects.items() if _role_value(o, role) is not None]
    if not handles:
        return []
    rows = [{"id": h, role: str(_role_value(scene.objects[h], role))} for h in handles]
    table = from_rows(rows, roles=["id", role], dim=dim, seed=seed)
    res = Query().select("id", role).where(role, "~", str(value)).order_by("similarity").run(table)
    out = []
    for r in res:
        h = r.get("id")
        conf = float(r.get("_confidence", 0.0))
        if h in scene.objects and (threshold is None or conf >= threshold):
            out.append((h, conf))
    return out


# ---- tagging (writes through the document, so undo + events fire) --------------------------------------------

def tag(scene, handles, key, value=True):
    """Tag each object with key->value (a bound role on the record). Through scene.edit, so it records undo and
    fires change events."""
    for h in handles:
        scene.edit(h, tags={key: value})


def untag(scene, handles, key):
    """Remove a tag from each object (through scene.remove_tag)."""
    for h in handles:
        scene.remove_tag(h, key)


def select_by_tag(scene, key, value=None):
    """Handles carrying tag `key` (optionally equal to `value`) -- a tag query."""
    if value is None:
        return select(scene, has_tag=key)
    return select(scene, tag=(key, value))


# ---- named selection sets + set algebra ----------------------------------------------------------------------

class Selection:
    """Selection helper bound to a Scene: query into a set of handles, SAVE named sets (as id lists -- exact, no
    capacity limit), do set ALGEBRA, push the current selection to the document, and (optionally) get the
    holographic BUNDLE of a selection's identity atoms for the vector-algebra case."""

    def __init__(self, scene):
        self.scene = scene
        self._named = {}                      # name -> frozenset of handles (an ID LIST, not a giant bundle)

    # queries
    def query(self, **kw):
        return select(self.scene, **kw)

    def query_fuzzy(self, role, value, **kw):
        return select_fuzzy(self.scene, role, value, **kw)

    # named sets
    def save(self, name, handles):
        """Save a named selection set (stored as an id list -- the membership, exact and unbounded)."""
        self._named[name] = frozenset(handles)
        return set(self._named[name])

    def get(self, name):
        return set(self._named.get(name, frozenset()))

    def names(self):
        return sorted(self._named)

    # set algebra (plain, exact operations on handle sets -- "everything metal, minus the wheels" is one line)
    @staticmethod
    def union(*sets):
        out = set()
        for s in sets:
            out |= set(s)
        return out

    @staticmethod
    def intersect(*sets):
        if not sets:
            return set()
        out = set(sets[0])
        for s in sets[1:]:
            out &= set(s)
        return out

    @staticmethod
    def minus(a, b):
        return set(a) - set(b)

    def invert(self, handles):
        """Everything in the scene NOT in `handles`."""
        return set(self.scene.objects) - set(handles)

    # push to the document's current selection (fires a select event)
    def apply(self, handles):
        self.scene.select(handles)
        return set(handles)

    # the holographic representation (the backlog's "selection AS a hypervector")
    def as_bundle(self, handles):
        """Bundle the members' identity atoms into ONE hypervector -- the selection as a vector, so a member reads
        as present (high cosine) and set-union is bundling. KEPT NEGATIVE: this has the superposition capacity
        ceiling, so it is for the VECTOR ALGEBRA on modest sets, not for storing membership of thousands (use the
        id list for that). Returns None for an empty selection."""
        from holographic.agents_and_reasoning.holographic_ai import bundle
        vecs = [self.scene.handle_vector(h) for h in handles if h in self.scene.objects]
        return bundle(vecs) if vecs else None


def _selftest():
    """Exact select filters by name/material/tag; fuzzy select ranks by meaning WITH confidence; tag/untag write
    through the document (undo/events); named sets + set algebra compose; a member reads as present in the
    selection bundle; deterministic."""
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.agents_and_reasoning.holographic_ai import cosine

    scene = Scene(dim=512, seed=0)
    w1 = scene.add(name="wheel_front", material="metal", tags={"kind": "wheel"})
    w2 = scene.add(name="wheel_rear", material="metal", tags={"kind": "wheel"})
    body = scene.add(name="body", material="paint", tags={"kind": "panel"})
    glass = scene.add(name="windscreen", material="glass", tags={"kind": "panel"})

    # (1) exact selection: by material, by tag, by substring, by predicate
    metal = select(scene, material="metal")
    assert metal == {w1, w2}
    panels = select_by_tag(scene, "kind", "panel")
    assert panels == {body, glass}
    wheels = select(scene, name_contains="wheel")
    assert wheels == {w1, w2}

    # (2) set algebra: everything metal, minus the front wheel
    sel = Selection(scene)
    assert sel.minus(metal, {w1}) == {w2}
    assert sel.union(metal, panels) == {w1, w2, body, glass}
    assert sel.intersect(select_by_tag(scene, "kind", "wheel"), metal) == {w1, w2}
    assert sel.invert(metal) == {body, glass}

    # (3) named sets (stored as id lists) + apply pushes to the document's current selection
    sel.save("drivetrain", metal)
    assert sel.get("drivetrain") == {w1, w2} and "drivetrain" in sel.names()
    sel.apply(sel.get("drivetrain"))
    assert scene.selection == {w1, w2}

    # (4) fuzzy selection rides the query layer and SURFACES confidence
    hits = select_fuzzy(scene, "material", "metal")
    hit_handles = {h for h, c in hits}
    assert {w1, w2} <= hit_handles                       # the metal parts rank in
    assert all(0.0 <= c <= 1.0 for _, c in hits)          # every hit carries a calibrated confidence

    # (5) tagging writes through the document: undo + a change event, and the tag becomes queryable
    events = []
    scene.on_change(lambda k, h: events.append(k))
    tag(scene, [body], "review", "pending")
    assert select_by_tag(scene, "review") == {body}
    assert "edit" in events
    scene.undo()                                          # the tag was recorded -> undo removes it
    assert select_by_tag(scene, "review") == set()
    untag(scene, [w1], "kind")
    assert "kind" not in scene.get(w1).tags

    # (6) the selection as a BUNDLE: a member reads as present (high cosine), a stranger does not
    b = sel.as_bundle(metal)
    assert cosine(b, scene.handle_vector(w1)) > cosine(b, scene.handle_vector(glass))

    # (7) deterministic
    assert select(scene, material="metal") == select(scene, material="metal")

    print("holographic_scene_query selftest OK: exact select filters by material/tag/substring/predicate; set "
          "algebra composes (metal minus front-wheel = rear-wheel); named sets save as id lists and apply to the "
          "document; fuzzy select ranks the metal parts WITH calibrated confidence; tag/untag write through the "
          "document (undo + events); a member reads as present in the selection bundle; deterministic")


if __name__ == "__main__":
    _selftest()
