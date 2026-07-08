"""Tests for holographic_access.py -- invite-gating + per-principal read grants."""
import pytest
from holographic.misc.holographic_access import AccessError, require_readable, can_read, grant, revoke, invite, apply_invite


class _P:
    def __init__(self, i, ns):
        self.id, self.namespace_name, self.grants = i, ns, {"read": set()}


def test_guest_sees_only_own_namespace_by_default():
    g = _P("guest", "ws:lab/user:guest")
    assert can_read(g, g.namespace_name)
    assert not can_read(g, "ws:lab/user:alice")
    with pytest.raises(AccessError):
        require_readable(g, "lab/scene")


def test_grant_then_readable():
    g = _P("guest", "ws:lab/user:guest")
    grant(g, read="lab/scene")
    require_readable(g, "lab/scene")                     # no raise
    assert can_read(g, "lab/scene") and not can_read(g, "lab/notes")


def test_grant_list_and_revoke():
    g = _P("guest", "ws:lab/user:guest")
    grant(g, read=["a", "b", "c"])
    assert all(can_read(g, ns) for ns in ("a", "b", "c"))
    revoke(g, read="b")
    assert can_read(g, "a") and not can_read(g, "b")


def test_invite_confers_grants_and_is_single_use():
    inv = invite(kind="user", grants={"read": ["lab/scene", "lab/props"]}, code="CODE1")
    assert inv.code == "CODE1" and inv.kind == "user" and inv.is_valid()
    p = _P("newcomer", "ws:lab/user:newcomer")
    apply_invite(p, inv)
    assert can_read(p, "lab/scene") and can_read(p, "lab/props")
    assert not inv.is_valid()                            # spent


def test_invite_codes_are_unguessable_when_unspecified():
    a, b = invite(), invite()
    assert a.code != b.code and len(a.code) >= 8
