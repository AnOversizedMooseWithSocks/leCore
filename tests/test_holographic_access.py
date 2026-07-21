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


def test_invite_button_and_join_button_round_trip():
    """The invite/join BUTTONS (create_invite_link / join_from_link): one call each, wrapping invite/admit so a UI
    button doesn't have to assemble a URL or know the grant vocabulary. A guest joins from a pasted LINK and another
    from a bare CODE; a bad code fails loudly."""
    import lecore
    from holographic.misc.holographic_access import AccessError

    host = lecore.UnifiedMind(dim=64, seed=0)

    # host clicks Invite -> a shareable link + bare code, default grant = read the workspace scene
    inv = host.create_invite_link(workspace="lab", base_url="http://127.0.0.1:5050/")
    assert "join=" in inv["link"] and inv["code"] and inv["code"] in inv["link"]
    assert inv["grants"] == {"read": ["lab/scene"]}

    # guest clicks Join pasting the full LINK
    alice = host.join_from_link(inv["link"], actor_id="alice")
    assert getattr(alice, "id", None) == "alice"

    # another guest joins by BARE CODE (typed into the join box)
    inv2 = host.create_invite_link(workspace="lab")
    bob = host.join_from_link(inv2["code"], actor_id="bob")
    assert getattr(bob, "id", None) == "bob"

    # base_url that already has a query string still gets a valid separator
    inv3 = host.create_invite_link(base_url="http://h/app?x=1")
    assert "?x=1&join=" in inv3["link"]

    # a bad/unknown code fails loudly, same contract as admit
    try:
        host.join_from_link("http://x/?join=deadbeefdeadbeef", "eve")
        assert False, "unknown code must raise"
    except AccessError:
        pass

    # a custom grant is honoured (share more than the default)
    inv4 = host.create_invite_link(workspace="lab", grants={"read": ["lab/scene", "lab/notes"]})
    assert inv4["grants"]["read"] == ["lab/scene", "lab/notes"]
