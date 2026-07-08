"""holographic_principal.py -- ONE scoped identity for ANY actor: agent, user, service, or a whole peer leCore.

The audit's key finding: the isolation a single agent needs is the SAME isolation a user, a service, or a guest peer
node needs -- so there is one primitive, not three. A Principal bundles four isolations, and the point is that they
are the DEFAULT, not a discipline you have to remember:

  * namespace -- a private database namespace ("ws:<workspace>/<kind>:<id>"). A principal WRITES only here.
  * inbox     -- a private directed bus topic ("to:<id>"). A principal READS only its own messages.
  * role      -- a provenance role vector (source_role). Everything a principal contributes is tagged with its origin,
                 so two principals' contributions never get confused (this is the same provenance the bus sender,
                 opponent, and merge_forks read).
  * overlay   -- (optional) a private copy-on-write learning overlay over a shared partition base, for a principal
                 that learns without disturbing the shared base or other principals.

Because every actor is a Principal, multiplayer (many users in a workspace), swarm (many agents), and federation (a
guest peer node) are the SAME isolation problem solved once. A kind="peer" principal is a whole other leCore instance
connecting as a guest -- it uses the host's tools (remote_tools) under the same scoping and (Layer 7) access control.

Reuses: query.Database.add_namespace (namespace), bus.send/open_mailbox/poll (inbox), provenance.source_role (role),
partition.SharedMind.branch (overlay). numpy/stdlib only; deterministic.
"""
from holographic.caching_and_storage.holographic_provenance import source_role, from_external


class Principal:
    """A scoped identity. Construct with a partition base and a database (either may be None), plus the actor id and
    which workspace/kind it is. The isolation primitives are set up on construction; call connect(bus) once so the
    principal can pull its inbox.

        alice = Principal(mind.base, mind.db, "alice", workspace="lab", kind="user").connect(mind.bus())
        alice.send(mind.bus(), to="bob", payload={...})     # only bob's inbox sees it, stamped sender="alice"
        for msg in alice.poll(mind.bus()): ...              # only alice's own inbox
    """

    def __init__(self, base, db, actor_id, workspace="default", kind="agent", dim=1024, seed=0):
        self.id = str(actor_id)
        self.kind = kind                                        # agent | user | service | peer
        self.workspace = workspace
        self.dim = dim
        self._seed = seed

        # namespace: this principal's private, writable store. Its name encodes workspace + kind + id so two
        # principals can never share one. (If no db is given, the principal still has an inbox + role + overlay.)
        self.namespace_name = "ws:%s/%s:%s" % (workspace, kind, self.id)
        self.namespace = db.add_namespace(self.namespace_name, tier="workspace") if db is not None else None

        # inbox: the directed topic this principal listens on. connect(bus) opens the mailbox that collects it.
        self.inbox = "to:%s" % self.id

        # role: this principal's provenance tag -- everything it contributes is bound to this.
        self.role = source_role(self.id, dim, seed)

        # overlay: an OPTIONAL private learning overlay over a shared frozen base (partition). None unless a base is
        # given -- most principals (users, services, peers) need only namespace + inbox + role.
        self.overlay = base.branch(self.id) if base is not None else None

        # access: which OTHER namespaces this principal may READ. Default: nothing (its own is always readable). A host
        # grants specific namespaces to share; see holographic_access. Writes stay own-namespace-only regardless.
        self.grants = {"read": set()}

    def connect(self, bus):
        """Open this principal's inbox mailbox on the bus so it can PULL its own directed messages. Chainable."""
        bus.open_mailbox(self.id, [self.inbox])
        return self

    def send(self, bus, to, payload):
        """Send a directed message to another principal, stamped with THIS principal as the sender. Delivered only to
        `to`'s inbox -- signals don't broadcast across principals."""
        return bus.send(to, payload, sender=self.id)

    def poll(self, bus, limit=None):
        """This principal's waiting inbox messages -- only its own. Another principal's inbox is invisible here."""
        return bus.poll(self.id, limit=limit)

    def tag(self, vec):
        """Bring an external vector into the shared space tagged with THIS principal's origin (provenance binding), so
        its contribution can always be traced back and never confused with another's."""
        return from_external(vec, self.id, dim=self.dim, seed=self._seed)

    def can_write(self, namespace):
        """The isolation rule, made checkable: a principal may write ONLY its own namespace."""
        return namespace == self.namespace_name

    def can_read(self, namespace):
        """A principal may read its OWN namespace always, and any other only if it's been GRANTED (default: nothing).
        See holographic_access.grant / require_readable."""
        from holographic.misc.holographic_access import can_read
        return can_read(self, namespace)

    def __repr__(self):
        return "Principal(%r, kind=%r, workspace=%r)" % (self.id, self.kind, self.workspace)


def _selftest():
    import numpy as np
    from holographic.agents_and_reasoning.holographic_query import Database
    from holographic.misc.holographic_bus import MessageBus

    db = Database()
    bus = MessageBus()
    alice = Principal(None, db, "alice", workspace="lab", kind="user", dim=512).connect(bus)
    bob = Principal(None, db, "bob", workspace="lab", kind="user", dim=512).connect(bus)
    carol = Principal(None, db, "carol", workspace="lab", kind="agent", dim=512).connect(bus)

    # --- distinct namespaces; each writes only its own ---
    assert alice.namespace_name != bob.namespace_name
    assert alice.can_write(alice.namespace_name) and not alice.can_write(bob.namespace_name)

    # --- directed messaging: bob sends to alice; only alice sees it, stamped from bob ---
    bob.send(bus, to="alice", payload={"hi": 1})
    alice_msgs = alice.poll(bus)
    assert len(alice_msgs) == 1 and alice_msgs[0].sender == "bob"
    assert carol.poll(bus) == [] and bob.poll(bus) == []       # no crossed signals

    # --- provenance: alice's tag is recoverable by her role, not carol's ---
    v = np.random.default_rng(0).standard_normal(512); v /= np.linalg.norm(v)
    tagged = alice.tag(v)
    from holographic.caching_and_storage.holographic_provenance import of_source
    good = of_source(tagged, "alice", 512)
    bad = of_source(tagged, "carol", 512)
    cg = float(np.dot(good, v) / (np.linalg.norm(good) * np.linalg.norm(v)))
    cb = float(np.dot(bad, v) / (np.linalg.norm(bad) * np.linalg.norm(v)))
    assert cg > 0.6 and cg > 2 * abs(cb)

    print("OK: holographic_principal self-test passed (3 principals: distinct namespaces, own-namespace writes only, "
          "directed messages reach only the addressee stamped with the sender, no crossed inboxes, provenance tags "
          "trace to the right principal)")


if __name__ == "__main__":
    _selftest()
