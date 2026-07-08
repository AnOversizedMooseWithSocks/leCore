"""holographic_workspace.py -- WORKSPACES: durable DB coexists with transient 3D/sim sessions (query backlog WS3-WS6).

The requirement is coexistence without stepping on each other. Three tiers (defined on the Database in WS1):
  * system     -- read-only, the mind's live state; restored by a reset, never persisted as user data.
  * persistent -- the durable user database (+ installed programs); survives resets, switches, and sessions.
  * workspace  -- transient per-session scratch (ws:<name>): loose tables plus the 3D scene / sim / render context;
                  cleared independently, and clearing one never touches the persistent DB or another workspace.

This module adds the manager on top of those tiers: make / switch / clear workspaces, a reset that keeps your data,
export/import a single workspace, and a combine with an EXPLICIT collision policy (a merge needs a decision, not a
guess). Pure stdlib + the existing query layer; deterministic (replay-based save).

KEPT NEGATIVE (loud): reset_to_default keeps the persistent tier and drops workspaces, but the SYSTEM tier is the
mind's job to re-publish (this manager does not fabricate it). Combine's default policy is 'error' on a name clash --
you must choose 'suffix' / 'left' / 'right' deliberately.
"""
from holographic.agents_and_reasoning.holographic_query import Database, UserTable, QueryError


class Workspace:
    """A transient session context: its own workspace-tier namespace (ws:<name>) for scratch tables, plus optional
    handles for the 3D scene / simulation / render session that belong to this session. Cleared as a unit."""

    def __init__(self, name):
        self.name = name
        self.ns = "ws:" + name
        self.scene = None                                    # a 3D scene handle (bundle of objects), if any
        self.sim = None                                      # a simulation handle (fields), if any
        self.render = None                                   # a render session, if any
        self.meta = {}


class WorkspaceManager:
    """Manages the coexistence of one persistent DB with many transient workspaces over a single Database."""

    def __init__(self, db=None):
        self.db = db if db is not None else Database()
        self.workspaces = {}                                 # name -> Workspace
        self.active = None

    # ---- WS3: make / switch / clear ----
    def new_workspace(self, name):
        """Create a fresh workspace (a 'workspace'-tier namespace ws:<name>) and make it active."""
        if name in self.workspaces:
            raise QueryError("workspace %r already exists" % name)
        self.db.create_namespace("ws:" + name, tier="workspace")
        ws = Workspace(name)
        self.workspaces[name] = ws
        self.active = name
        return ws

    def switch_workspace(self, name):
        """Make an existing workspace active (the persistent DB is always visible regardless)."""
        if name not in self.workspaces:
            raise QueryError("no such workspace %r" % name)
        self.active = name
        return self.workspaces[name]

    def clear_workspace(self, name):
        """Drop one workspace and everything in its namespace -- WITHOUT touching the persistent DB or any other
        workspace. This is the isolation guarantee."""
        if name not in self.workspaces:
            raise QueryError("no such workspace %r" % name)
        self.db.drop_namespace("ws:" + name)
        del self.workspaces[name]
        if self.active == name:
            self.active = None
        return self

    def active_workspace(self):
        return self.workspaces.get(self.active) if self.active else None

    # ---- WS4: the 'new session' button that does NOT wipe your data ----
    def reset_to_default(self):
        """Drop every workspace (transient), KEEP the persistent tier (the durable DB + installed programs), and
        leave the system tier for the mind to re-publish. The whole point: a reset returns to a clean session
        without destroying user data."""
        for name in list(self.workspaces):
            self.clear_workspace(name)
        self.active = None
        return self

    # ---- WS5: export / import one workspace ----
    def export_workspace(self, name):
        """Serialise one workspace's scratch tables by replay (columns/dim/seed/rows) -- portable and deterministic."""
        if name not in self.workspaces:
            raise QueryError("no such workspace %r" % name)
        ns = "ws:" + name
        body = self.db.namespaces.get(ns, {"tables": {}})
        tables = {tn: {"columns": list(t.roles), "dim": t.dim, "seed": getattr(t, "seed", 0), "rows": [dict(r) for r in t.rows]}
                  for tn, t in body["tables"].items() if isinstance(t, UserTable)}
        return {"name": name, "tables": tables}

    def import_workspace(self, blob):
        """Rebuild a workspace from an export blob (re-encodes byte-identically -- the seed fixes every atom)."""
        name = blob["name"]
        if name in self.workspaces:
            raise QueryError("workspace %r already exists; clear it first" % name)
        ws = self.new_workspace(name)
        for tn, spec in blob["tables"].items():
            t = UserTable(tn, spec["columns"], dim=spec["dim"], seed=spec["seed"])
            for row in spec["rows"]:
                t.insert(row)
            self.db.namespaces[ws.ns]["tables"][tn] = t
        return ws

    # ---- WS6: combine two workspaces (union of DBs) with an EXPLICIT collision policy ----
    def combine_workspaces(self, a, b, into, on_collision="error"):
        """Union the tables of workspaces `a` and `b` into a new workspace `into`. A merge across sources needs a
        DECISION on name clashes, not a guess: on_collision in {'error','suffix','left','right'}. 'suffix' keeps both
        (tables tagged _a / _b), 'left'/'right' pick a winner, 'error' refuses. Combining scenes would bundle their
        objects and sims would overlay their fields -- the same union move at the 3D/sim layer (left to those handles)."""
        for w in (a, b):
            if w not in self.workspaces:
                raise QueryError("no such workspace %r" % w)
        if on_collision not in ("error", "suffix", "left", "right"):
            raise QueryError("on_collision must be error/suffix/left/right, got %r" % on_collision)
        dst = self.new_workspace(into)
        atab = self.db.namespaces["ws:" + a]["tables"]
        btab = self.db.namespaces["ws:" + b]["tables"]
        shared = set(atab) & set(btab)
        for src_ws, tabs, suf in ((a, atab, "_a"), (b, btab, "_b")):
            for tn, t in tabs.items():
                if tn in shared:
                    if on_collision == "error":
                        raise QueryError("table %r exists in both %r and %r (choose an on_collision policy)" % (tn, a, b))
                    if on_collision == "left" and src_ws != a:
                        continue
                    if on_collision == "right" and src_ws != b:
                        continue
                    name = tn + suf if on_collision == "suffix" else tn
                else:
                    name = tn
                self.db.namespaces[dst.ns]["tables"][name] = t
        return dst


def _selftest():
    """Persistent data survives a reset; a workspace is isolated (clearing it leaves the persistent DB and a sibling
    workspace intact); export/import round-trips a workspace; combine unions two workspaces with a collision policy."""
    from holographic.agents_and_reasoning.holographic_query import run_sql

    mgr = WorkspaceManager()
    db = mgr.db

    # persistent durable DB
    db.create_namespace("userdb", tier="persistent")
    db.create_table("userdb.notes", ["txt"], dim=256, seed=0)
    db.insert("userdb.notes", {"txt": "durable"})

    # two workspaces with their own scratch
    mgr.new_workspace("sessionA")
    db.create_table("ws:sessionA.tmp", ["x"], dim=256, seed=0)
    db.insert("ws:sessionA.tmp", {"x": "A-scratch"})
    mgr.new_workspace("sessionB")
    db.create_table("ws:sessionB.tmp", ["x"], dim=256, seed=0)
    db.insert("ws:sessionB.tmp", {"x": "B-scratch"})

    # clearing A leaves B and the persistent DB untouched
    mgr.clear_workspace("sessionA")
    assert mgr.db.tier_of("ws:sessionA") is None                     # gone
    assert run_sql("SELECT x FROM tmp", db.resolve("ws:sessionB.tmp"))[0]["x"] == "B-scratch"
    assert run_sql("SELECT txt FROM notes", db.resolve("userdb.notes"))[0]["txt"] == "durable"

    # reset_to_default drops workspaces, KEEPS the persistent DB
    mgr.reset_to_default()
    assert mgr.workspaces == {} and mgr.db.tier_of("ws:sessionB") is None
    assert mgr.db.tier_of("userdb") == "persistent"                  # durable data survived the reset
    assert run_sql("SELECT txt FROM notes", db.resolve("userdb.notes"))[0]["txt"] == "durable"

    # export / import a workspace round-trips
    mgr.new_workspace("work")
    db.create_table("ws:work.items", ["name"], dim=256, seed=1)
    db.insert("ws:work.items", {"name": "widget"})
    blob = mgr.export_workspace("work")
    mgr.clear_workspace("work")
    mgr.import_workspace(blob)
    assert run_sql("SELECT name FROM items", db.resolve("ws:work.items"))[0]["name"] == "widget"

    # combine two workspaces (union) with an explicit collision policy
    mgr.reset_to_default()
    mgr.new_workspace("left"); db.create_table("ws:left.t", ["v"], dim=128, seed=0); db.insert("ws:left.t", {"v": "L"})
    mgr.new_workspace("right"); db.create_table("ws:right.t", ["v"], dim=128, seed=0); db.insert("ws:right.t", {"v": "R"})
    try:
        mgr.combine_workspaces("left", "right", "both")             # collision on 't' with default 'error'
        raise AssertionError("collision was not refused")
    except QueryError:
        pass
    mgr.combine_workspaces("left", "right", "both2", on_collision="suffix")
    tables = set(db.namespaces["ws:both2"]["tables"])
    assert {"t_a", "t_b"} <= tables                                  # both kept, disambiguated

    print("holographic_workspace selftest OK: persistent data survives reset_to_default; clearing one workspace "
          "leaves the persistent DB and a sibling workspace intact; export/import round-trips a workspace; combine "
          "unions two workspaces with an explicit collision policy (suffix -> t_a, t_b); deterministic")


if __name__ == "__main__":
    _selftest()
