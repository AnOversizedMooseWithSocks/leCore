"""holographic_access.py -- who may READ what: invite-gating + per-principal read grants.

leCore's database already enforces "write only your OWN namespace" at one chokepoint (query._require_writable), with
cross-namespace READS always allowed -- fine for a TRUSTED host. The moment others connect (guests joining a workspace,
a public farm), that read-any default is too open. This adds the symmetric, stricter twin for guests:

  * per-principal READ GRANTS -- a guest is granted read access to SPECIFIC namespaces, and sees NOTHING else.
    Default is nothing granted, so "choose what to share" is an explicit act, not the current read-any default.
  * require_readable(principal, namespace) -- the read chokepoint mirroring _require_writable: raise unless the
    namespace is in the principal's read grants.
  * invites -- a host hands a guest an INVITE (a token) that admits it as a given kind with some initial grants;
    redeeming the invite makes a Principal with exactly those grants and nothing more.

Writes stay own-namespace-only (unchanged). This is opt-in: a trusted host that wants read-any simply doesn't call
require_readable. numpy-free; stdlib only (secrets for unguessable tokens).
"""
import secrets


class AccessError(Exception):
    """Raised when a principal tries to read a namespace it hasn't been granted."""


def _read_grants(principal):
    """The principal's set of granted-readable namespaces (creating the container if missing)."""
    grants = getattr(principal, "grants", None)
    if grants is None:
        grants = {}
        setattr(principal, "grants", grants)
    return grants.setdefault("read", set())


def _as_list(x):
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def grant(principal, read=None):
    """Grant this principal READ access to a namespace (or list of namespaces). Returns the principal (chainable)."""
    g = _read_grants(principal)
    for ns in _as_list(read):
        g.add(ns)
    return principal


def revoke(principal, read=None):
    """Stop sharing: remove READ access to a namespace (or list). Returns the principal."""
    g = _read_grants(principal)
    for ns in _as_list(read):
        g.discard(ns)
    return principal


def can_read(principal, namespace):
    """True iff this principal has been granted read access to `namespace`. Its own namespace is always readable."""
    if namespace == getattr(principal, "namespace_name", None):
        return True
    return namespace in _read_grants(principal)


def require_readable(principal, namespace):
    """The read chokepoint -- the symmetric twin of the DB's _require_writable. Raises AccessError unless `namespace`
    is readable by `principal` (granted, or the principal's own). A guest sees nothing until the host grants it."""
    if not can_read(principal, namespace):
        raise AccessError("%s is not shared with %s" % (namespace, getattr(principal, "id", principal)))


class Invite:
    """A one-token admission: it says a guest may connect as `kind` and, on redemption, starts with these read grants
    (and nothing else). The host creates it and hands the guest the .code; redeeming binds it to a specific actor id.
    Single-use by default -- redeeming marks it used so a leaked code can't admit a crowd."""

    def __init__(self, kind="user", grants=None, code=None, single_use=True):
        self.kind = kind
        self.read = set(_as_list((grants or {}).get("read")))
        self.code = code or secrets.token_hex(8)
        self.single_use = single_use
        self.redeemed_by = None

    def is_valid(self):
        return not (self.single_use and self.redeemed_by is not None)

    def __repr__(self):
        return "Invite(kind=%r, code=%r, read=%s)" % (self.kind, self.code, sorted(self.read))


def invite(kind="user", grants=None, code=None):
    """Make an invite token admitting a guest as `kind` with initial read `grants` (e.g. {'read': ['lab/scene']})."""
    return Invite(kind=kind, grants=grants, code=code)


def apply_invite(principal, inv):
    """Redeem an invite onto a freshly created principal: copy its kind's grants over and mark the invite used."""
    for ns in inv.read:
        grant(principal, read=ns)
    if inv.single_use:
        inv.redeemed_by = getattr(principal, "id", principal)
    return principal


def _selftest():
    # a stand-in principal with an own-namespace and a grants container
    class _P:
        def __init__(self, i, ns):
            self.id, self.namespace_name, self.grants = i, ns, {"read": set()}

    guest = _P("guest", "ws:lab/user:guest")

    # default: a guest sees NOTHING but its own namespace
    assert can_read(guest, guest.namespace_name)                 # own namespace always readable
    assert not can_read(guest, "ws:lab/user:alice")
    try:
        require_readable(guest, "lab/scene")
        assert False, "should have raised"
    except AccessError:
        pass

    # grant one namespace -> now readable; others still not
    grant(guest, read="lab/scene")
    require_readable(guest, "lab/scene")                         # no raise
    assert can_read(guest, "lab/scene") and not can_read(guest, "lab/notes")

    # grant a list, then revoke one
    grant(guest, read=["lab/notes", "lab/props"])
    assert can_read(guest, "lab/notes") and can_read(guest, "lab/props")
    revoke(guest, read="lab/notes")
    assert not can_read(guest, "lab/notes") and can_read(guest, "lab/props")

    # an invite carries initial grants and is single-use
    inv = invite(kind="user", grants={"read": ["lab/scene", "lab/props"]}, code="TESTCODE")
    assert inv.code == "TESTCODE" and inv.kind == "user" and inv.is_valid()
    newcomer = _P("newcomer", "ws:lab/user:newcomer")
    apply_invite(newcomer, inv)
    assert can_read(newcomer, "lab/scene") and can_read(newcomer, "lab/props")
    assert not inv.is_valid()                                    # single-use: now spent

    print("OK: holographic_access self-test passed (a guest reads only its own namespace by default; require_readable "
          "raises until granted; grant/revoke share and un-share specific namespaces; an invite confers initial grants "
          "and is single-use)")


if __name__ == "__main__":
    _selftest()
