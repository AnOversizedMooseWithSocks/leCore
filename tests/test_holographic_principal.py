"""Tests for holographic_principal.py -- one scoped identity for any actor."""
import numpy as np
from holographic.misc.holographic_principal import Principal
from holographic.agents_and_reasoning.holographic_query import Database
from holographic.misc.holographic_bus import MessageBus


def _trio():
    db, bus = Database(), MessageBus()
    a = Principal(None, db, "alice", workspace="lab", kind="user", dim=512).connect(bus)
    b = Principal(None, db, "bob", workspace="lab", kind="user", dim=512).connect(bus)
    c = Principal(None, db, "carol", workspace="lab", kind="agent", dim=512).connect(bus)
    return db, bus, a, b, c


def test_distinct_namespaces_own_writes_only():
    _, _, a, b, _ = _trio()
    assert a.namespace_name != b.namespace_name
    assert a.can_write(a.namespace_name)
    assert not a.can_write(b.namespace_name)


def test_directed_messages_reach_only_addressee():
    _, bus, a, b, c = _trio()
    b.send(bus, to="alice", payload={"hi": 1})
    got = a.poll(bus)
    assert len(got) == 1 and got[0].sender == "bob" and got[0].payload == {"hi": 1}
    assert b.poll(bus) == [] and c.poll(bus) == []          # no crossed inboxes


def test_namespace_name_encodes_workspace_and_kind():
    _, _, a, _, c = _trio()
    assert a.namespace_name == "ws:lab/user:alice"
    assert c.namespace_name == "ws:lab/agent:carol"


def test_provenance_tag_traces_to_the_principal():
    from holographic.caching_and_storage.holographic_provenance import of_source
    _, _, a, _, c = _trio()
    v = np.random.default_rng(2).standard_normal(512); v /= np.linalg.norm(v)
    tagged = a.tag(v)
    cg = float(np.dot(of_source(tagged, "alice", 512), v))
    cb = float(np.dot(of_source(tagged, "carol", 512), v))
    assert cg > 0.5 and cg > 2 * abs(cb)


def test_optional_overlay_only_when_base_given():
    db = Database()
    p = Principal(None, db, "x", dim=256)
    assert p.overlay is None                                 # no base -> no overlay


def test_kinds_are_carried():
    db = Database()
    for kind in ("agent", "user", "service", "peer"):
        p = Principal(None, db, "n_" + kind, kind=kind, dim=128)
        assert p.kind == kind
