"""holographic_queryfolder.py -- WS7 folders: a shallow grouping tree over a query Database (database > folder > table).

WHY
---
A flat pile of tables gets as unwieldy as a flat pile of files. Folders add ONE grouping level between a database and
its tables, so related tables sit together and a search can scope to just one group. This is the compositional ladder
continued upward -- rows -> tables -> FOLDERS -> databases -- and it is NOT new storage: a folder REFERENCES existing
tables by their qualified name; it never copies them.

The one design decision that matters (so multi-membership does not break lifecycle):
  * HOME folder -- every table has exactly ONE. That is OWNERSHIP: it governs the table's lifecycle (dropping a home
    folder drops its tables) and its tier is the table's namespace tier. Like the directory a file actually lives in.
  * ASSOCIATION links -- a table may also be linked into zero or more OTHER folders. That is pure GROUPING (like tags
    / hard links): unlinking never deletes the table, it just ungroups it.

Scoped search doubles as a speedup: no scope searches everything; a folder scope searches only that subtree's tables
-- a narrower result AND a smaller candidate set, the way a spatial tree prunes a scene.

KEPT NEGATIVES (loud)
  * Keep the tree SHALLOW -- database > folder > table is usually enough; deep nesting is as annoying as a flat pile,
    and the value is grouping, not depth. Sub-folders are allowed but discouraged.
  * An association link is a GROUPING, not a copy -- one stored table referenced from several folders, so edits are
    seen everywhere and lifecycle follows the HOME folder only.
"""
from holographic_query import QueryError


class _FolderNode:
    """One node in the folder tree: it owns tables (home), links tables (association), and may hold sub-folders."""

    def __init__(self, name):
        self.name = name
        self.home = set()          # qualified table names whose HOME is here (ownership -> lifecycle)
        self.linked = set()        # qualified table names GROUPED here by association (no ownership)
        self.subfolders = {}       # name -> _FolderNode

    def child(self, seg, create=True):
        if seg not in self.subfolders:
            if not create:
                raise QueryError("no such folder segment %r" % seg)
            self.subfolders[seg] = _FolderNode(seg)
        return self.subfolders[seg]


class FolderTree:
    """A shallow grouping tree over a Database's tables. Folders reference tables by qualified name ('ns.table');
    they are an overlay, not storage. One home folder per table (ownership), plus any number of association links."""

    def __init__(self, db):
        self.db = db
        self.root = _FolderNode("")     # the database root (folders hang off it)
        self._home = {}                 # qualified table name -> home folder path (so a table has exactly one home)

    # ---- building the tree ---------------------------------------------------------------------------------
    def create_folder(self, path):
        """Create a folder (or nested path 'a/b'). Returns the path. Shallow is preferred -- see the kept negative."""
        node = self.root
        for seg in _segments(path):
            node = node.child(seg, create=True)
        return path

    def _node(self, path, create=False):
        node = self.root
        for seg in _segments(path):
            node = node.child(seg, create=create)
        return node

    # ---- ownership (home) vs grouping (association) --------------------------------------------------------
    def set_home(self, qualified_table, folder_path):
        """Put a table's HOME in `folder_path` (ownership -> lifecycle). A table has exactly one home, so this moves
        it if it already had one. The table must exist in the Database (checked, so a folder can't own a ghost)."""
        self._require_table(qualified_table)
        old = self._home.get(qualified_table)
        if old is not None:
            self._node(old).home.discard(qualified_table)     # leave its previous home
        self._node(folder_path, create=True).home.add(qualified_table)
        self._home[qualified_table] = folder_path
        return folder_path

    def link(self, qualified_table, folder_path):
        """Add an ASSOCIATION link (grouping, no ownership) -- the table shows up in this folder too, but its
        lifecycle still follows its home. A no-op if the folder IS the table's home (that is already ownership)."""
        self._require_table(qualified_table)
        if self._home.get(qualified_table) == folder_path:
            return folder_path
        self._node(folder_path, create=True).linked.add(qualified_table)
        return folder_path

    def unlink(self, qualified_table, folder_path):
        """Remove an association link -- ungroups the table from this folder. NEVER deletes the table (that is what
        home ownership is for). Refuses to unlink a table from its home (use set_home to move ownership)."""
        if self._home.get(qualified_table) == folder_path:
            raise QueryError("%r is this table's HOME folder; unlink is for associations, use set_home to move it"
                             % folder_path)
        self._node(folder_path).linked.discard(qualified_table)
        return folder_path

    # ---- reading the tree ----------------------------------------------------------------------------------
    def tables_in(self, path="", recursive=True):
        """The tables visible in a folder: its home tables + its associated (linked) tables, and -- if recursive --
        everything under its sub-folders too. Deduplicated. This is what a scoped search runs over."""
        node = self.root if path in ("", None) else self._node(path)
        return sorted(self._collect(node, recursive))

    def _collect(self, node, recursive):
        seen = set(node.home) | set(node.linked)
        if recursive:
            for sub in node.subfolders.values():
                seen |= self._collect(sub, True)
        return seen

    def resolve(self, path):
        """Resolve a 'folder/.../table' path to the actual Table object in the Database (resolve-or-null: returns
        None if the last segment is not a table home/linked here). The table itself lives in its namespace."""
        segs = _segments(path)
        if not segs:
            return None
        *folders, table = segs
        node = self.root
        for seg in folders:
            if seg not in node.subfolders:
                return None
            node = node.subfolders[seg]
        hit = next((q for q in (node.home | node.linked) if _bare(q) == table), None)
        return self._table_obj(hit) if hit else None

    def drop_folder(self, path):
        """Drop a folder: its HOME tables are deleted (ownership -> lifecycle, like deleting a directory), its
        associations are simply dropped (grouping, no deletion), and its sub-folders drop recursively. Returns the
        list of deleted qualified table names."""
        node = self._node(path)
        deleted = self._drop_node(node)
        # detach this folder from its parent
        parent_path = "/".join(_segments(path)[:-1])
        parent = self.root if not parent_path else self._node(parent_path)
        parent.subfolders.pop(_segments(path)[-1], None)
        return deleted

    def _drop_node(self, node):
        deleted = []
        for q in list(node.home):                 # home tables are OWNED -> delete them from the Database
            self._delete_table(q)
            self._home.pop(q, None)
            deleted.append(q)
        node.linked.clear()                       # associations are just groupings -> drop the links, keep the tables
        for sub in list(node.subfolders.values()):
            deleted += self._drop_node(sub)
        node.subfolders.clear()
        return deleted

    # ---- Database glue -------------------------------------------------------------------------------------
    def _require_table(self, qualified):
        if self._table_obj(qualified) is None:
            raise QueryError("no such table %r (a folder references existing tables, it does not create them)"
                             % qualified)

    def _table_obj(self, qualified):
        ns, _, name = qualified.partition(".")
        body = self.db.namespaces.get(ns)
        return body["tables"].get(name) if body else None

    def _delete_table(self, qualified):
        ns, _, name = qualified.partition(".")
        if self.db.tier_of(ns) == "system":                   # honour the read-only wall
            raise QueryError("%r is in the system tier and cannot be dropped via a folder" % qualified)
        body = self.db.namespaces.get(ns)
        if body:
            body["tables"].pop(name, None)


def _segments(path):
    return [s for s in str(path).replace(".", "/").split("/") if s]


def _bare(qualified):
    return qualified.split(".")[-1]


def _selftest():
    from holographic_query import Database
    db = Database()
    db.add_namespace("user", tier="persistent")
    for t in ("sales", "returns", "catalog", "scratch"):
        db.create_table("user." + t, ["id"], dim=256, seed=0)

    ft = FolderTree(db)
    ft.create_folder("reports")
    ft.set_home("user.sales", "reports")              # sales lives in reports (ownership)
    ft.set_home("user.returns", "reports")
    ft.set_home("user.catalog", "reference")          # catalog's home is reference
    ft.link("user.catalog", "reports")                # but it is ALSO grouped into reports (association)

    # scoped search: 'reports' sees its two home tables + the linked catalog
    assert set(_bare(q) for q in ft.tables_in("reports")) == {"sales", "returns", "catalog"}
    assert set(_bare(q) for q in ft.tables_in("reference")) == {"catalog"}

    # resolve a folder path to the real table object
    assert ft.resolve("reports/sales") is db.namespaces["user"]["tables"]["sales"]
    assert ft.resolve("reports/nope") is None

    # unlink an association: catalog leaves reports but is NOT deleted (still in reference, still in the DB)
    ft.unlink("user.catalog", "reports")
    assert set(_bare(q) for q in ft.tables_in("reports")) == {"sales", "returns"}
    assert db.namespaces["user"]["tables"].get("catalog") is not None
    # cannot unlink a table from its HOME (that would orphan it)
    try:
        ft.unlink("user.catalog", "reference"); assert False
    except QueryError:
        pass

    # drop a folder: its HOME tables are deleted; associations elsewhere are untouched
    ft.link("user.scratch", "reports")                # scratch's home is root (unset) -> associate it into reports
    deleted = ft.drop_folder("reports")
    assert set(_bare(q) for q in deleted) == {"sales", "returns"}   # only the home tables
    assert db.namespaces["user"]["tables"].get("sales") is None     # sales was owned by reports -> gone
    assert db.namespaces["user"]["tables"].get("scratch") is not None  # scratch was only linked -> survives

    print("OK: holographic_queryfolder self-test passed (home=ownership vs association=grouping, scoped tables_in, "
          "resolve, unlink-keeps-table, drop-folder-deletes-only-home -- WS7)")


if __name__ == "__main__":
    _selftest()
