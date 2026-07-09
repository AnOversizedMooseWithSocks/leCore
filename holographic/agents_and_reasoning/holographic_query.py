"""holographic_query.py -- a query front door over the VSA store: a role-bound record IS a database row, so a
query is a PROJECTION over records. This is the projection core plus a small SQL subset and the two things a
plain database can't do -- FUZZY (semantic) WHERE and a calibrated per-row CONFIDENCE.

WHY THIS EXISTS (leCore Query Interface backlog, Part 1 / Phases 1-3)
--------------------------------------------------------------------
A record = roles bound to fillers, bundled -- that is a ROW (roles are columns, fillers are values). A table is a
set of such records (a codebook, a definition library, a scene's objects already are one). Recovering a column
is unbind(record, role) + cleanup -- already the primitive holographic_relations uses. So a query engine is not a
new store; it is a projection over the store we have. The discipline (Part 2): LOWER, don't interpret -- a WHERE
is one cosine of a probe against the record matrix, a SELECT is one batched unbind. Two Python crossings (parse,
read out), no per-row Python loop.

WHAT MAKES IT A GOOD *VSA* DATABASE (falls out of the substrate):
* FUZZY WHERE: `colour ~ 'red'` ranks rows by cosine of the unbound filler to red -- exact match is the special
  case, semantic match is native (plain SQL needs a bolt-on vector extension).
* CALIBRATED CONFIDENCE: every fuzzy result carries how strong its match is (cosine, and against a codebook a
  RecallNull-style read) -- a database that can say "this is a real match" vs "noise, abstaining."
* Round-trip: exact stored props round-trip losslessly (they are kept beside the vectors); the vector part is
  the fuzzy, recall-robust layer.

HONEST SCOPE (kept loud, per the backlog):
* EXACT vs FUZZY is a real semantic fork: `density = 5000` on a DECODED vector has readback error, so exact
  predicates run on the STORED props, fuzzy predicates on the VECTORS -- two clearly-labelled modes, never
  silently mixed (why Table keeps both).
* The SQL here is a small, documented SUBSET (SELECT / FROM / WHERE / ORDER BY / LIMIT), hand-rolled -- not full
  SQL. Determinism: ORDER BY ties break on a stable key.
* Sparse schema = natural NULL: a missing column is an absent bind, so SELECT of it returns null-ish, not an error.
This module is Phases 1-3; GraphQL, aggregation/GROUP BY, the capability registry + EXECUTE, namespaces and user
databases (Phases 4-13) build ON this core. Deterministic; NumPy + stdlib.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, bundle, cosine, Vocabulary


class Table:
    """A set of role-bound records with a shared role vocabulary (the 'columns') and the exact stored values
    beside them -- so we answer BOTH fuzzy predicates (on the vectors) and exact predicates (on the stored rows)
    honestly. `value_vocab` maps categorical fillers to vectors for fuzzy matching and cleanup."""

    def __init__(self, records, roles, role_vocab, value_vocab, rows):
        self.records = np.asarray(records, float)   # (n, dim) the VSA rows
        self.roles = list(roles)                    # column names
        self.role_vocab = role_vocab                # Vocabulary: role name -> role vector (for unbind)
        self.value_vocab = value_vocab              # Vocabulary: filler name -> filler vector (fuzzy + cleanup)
        self.rows = list(rows)                      # exact stored values per record (exact WHERE / output)

    def __len__(self):
        return len(self.records)


def _encode_row(row, roles, role_vocab, value_vocab, dim):
    """Turn one {column: value} dict into a VSA record: bind each CATEGORICAL value to its column role and bundle.
    Numeric values are NOT encoded (they live in the stored row for exact predicates -- encoding a float would add
    readback error, the honest fork). Shared by from_rows and UserTable.insert so there is ONE encoding, not two."""
    parts = []
    for col, val in row.items():
        if col not in roles:
            continue
        if isinstance(val, str):                                    # categorical filler -> a codebook atom
            value_vocab.get(val)
            parts.append(bind(role_vocab.get(col), value_vocab.get(val)))
    return bundle(parts) if parts else np.zeros(dim)


def from_rows(rows, roles, dim=1024, seed=0):
    """Ingest tabular data (a list of {column: value} dicts) into a Table: bind each CATEGORICAL value to its
    column role and bundle -> one record per row. Numeric values are kept only in the stored rows (exact
    predicates run on those; encoding a float into a vector would add readback error -- the honest fork).
    Returns a Table. A missing column is simply an absent bind (natural NULL)."""
    role_vocab = Vocabulary(dim, seed)
    value_vocab = Vocabulary(dim, seed + 1)
    for r in roles:
        role_vocab.get(r)
    records = [_encode_row(row, roles, role_vocab, value_vocab, dim) for row in rows]
    return Table(np.array(records) if records else np.zeros((0, dim)), roles, role_vocab, value_vocab, list(rows))


class QueryError(Exception):
    """A query/database error with a clear message (bad name, a write to the read-only system wall, etc.)."""


class ConstraintError(QueryError):
    """B5 -- a write was refused because it would violate a declared constraint (NOT NULL / UNIQUE / PK / FK / CHECK)."""


class Rollback(Exception):
    """B6 -- raise inside a `with transaction(...)` block to abort cleanly: every table rolls back and the exception
    is swallowed (an explicit, tidy ROLLBACK)."""


class _Transaction:
    """B6 -- single-writer atomicity: snapshot the given tables on entry; on any exception roll them ALL back to that
    snapshot; on a clean exit keep every change (commit)."""

    def __init__(self, tables):
        self._tables = tables

    def __enter__(self):
        self._snaps = [t._snapshot() for t in self._tables]
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:                              # any error -> roll every table back to pre-transaction
            for t, snap in zip(self._tables, self._snaps):
                t._restore(snap)
            return exc_type is Rollback                       # swallow an explicit Rollback; re-raise anything else
        return False                                          # clean exit -> commit (keep the changes)


def transaction(*tables):
    """B6 -- run a batch of writes over one or more tables all-or-nothing. Everything inside the `with` block commits
    together, or (on any exception, including `raise Rollback`) rolls back together to the pre-transaction state.
    Reuses a full per-table snapshot. Kept negative: isolation between CONCURRENT writers is deferred (B8); this is
    single-writer atomicity + durability of the batch, which covers many workloads."""
    return _Transaction(tables)


class UserTable(Table):
    """A WRITABLE table: CREATE allocates it with its columns as role vectors; INSERT encodes one more row into a
    record and stores its exact values beside it. Insertion is order-INDEPENDENT (a record is a bundle = a sum),
    so keep an explicit order column if order matters. This is what makes a user database a live thing you own,
    not a fixed snapshot."""

    def __init__(self, name, columns, dim=1024, seed=0):
        role_vocab = Vocabulary(dim, seed)
        value_vocab = Vocabulary(dim, seed + 1)
        for c in columns:
            role_vocab.get(c)
        super().__init__(np.zeros((0, dim)), list(columns), role_vocab, value_vocab, [])
        self.name = name
        self.dim = dim
        self.seed = seed
        self._pk = None                                        # B4: the primary-key column, if one is set
        self._pk_index = {}                                    # key value -> [row indices] (kept in sync on insert)
        # B5 constraints, enforced on insert:
        self._not_null = set()                                 # columns that may not be None/absent
        self._unique = set()                                   # columns whose live values must be distinct
        self._fk = {}                                          # col -> (ref_table, ref_col)
        self._checks = []                                      # [(name, predicate(row) -> bool)]

    # ---- B5: declare constraints ----
    def not_null(self, *cols):
        """Require these columns to be present and non-None on every inserted row."""
        self._not_null.update(cols)
        return self

    def unique(self, *cols):
        """Require these columns' live values to be distinct across rows."""
        self._unique.update(cols)
        return self

    def foreign_key(self, col, ref_table, ref_col):
        """Require col's value to exist in ref_table.ref_col (a live row). resolve-or-null stays the soft opt-out --
        a None value is allowed (an unset reference); a non-None value must resolve."""
        self._fk[col] = (ref_table, ref_col)
        return self

    def check(self, predicate, name="CHECK"):
        """Require predicate(row) to hold on every insert (an arbitrary row-level rule, e.g. lambda r: r['age'] >= 0)."""
        self._checks.append((name, predicate))
        return self

    def _value_exists(self, col, value):
        """Is `value` present in `col` among LIVE (non-tombstoned) rows? Used by UNIQUE and FK checks."""
        return any(r.get(col) == value and not r.get("_deleted") for r in self.rows)

    def _enforce_constraints(self, row):
        """Raise ConstraintError if `row` would violate any declared constraint. Called before an insert persists."""
        for c in self._not_null:
            if row.get(c) is None:
                raise ConstraintError("NOT NULL violated: column %r is required" % c)
        for c in self._unique:
            v = row.get(c)
            if v is not None and self._value_exists(c, v):
                raise ConstraintError("UNIQUE violated: %r = %r already exists" % (c, v))
        for c, (ref, rc) in self._fk.items():
            v = row.get(c)
            if v is not None and not ref._value_exists(rc, v):
                raise ConstraintError("FOREIGN KEY violated: %s=%r not found in %s.%s" % (c, v, ref.name, rc))
        for cname, pred in self._checks:
            if not pred(row):
                raise ConstraintError("CHECK violated: %s" % cname)

    def set_primary_key(self, col):
        """B4 -- declare a primary-key column and build its hash index, so `WHERE pk = X` is an O(1) dict lookup
        instead of the measured O(n) scan. A PRIMARY KEY is also UNIQUE and NOT NULL (B5). The index is rebuilt
        deterministically here and maintained on insert; the replay model rebuilds it on load, so it never drifts."""
        if col not in self.roles:
            raise QueryError("cannot key on %r: not a column" % col)
        self._pk = col
        self._pk_index = {}
        for i, r in enumerate(self.rows):                     # (re)build from whatever is already stored
            self._pk_index.setdefault(r.get(col), []).append(i)
        self._not_null.add(col)                               # a PK is NOT NULL + UNIQUE
        self._unique.add(col)
        return self

    def pk_lookup(self, value):
        """Live row indices for a primary-key value -- O(1) via the index, skipping tombstoned versions."""
        return [i for i in self._pk_index.get(value, []) if not self.rows[i].get("_deleted")]

    def add_column(self, name):
        """P13 -- add a queryable column with NO migration. Append the role (allocate its codebook vector); existing
        records are UNTOUCHED (no re-encoding, no table rewrite), so old rows are simply sparse on the new column
        (they never bound it -> read as None), and new rows may set it. This is what 'a record only holds the roles
        you bind' buys: ALTER TABLE ADD COLUMN is free here, where SQL locks/rewrites a big table."""
        if name not in self.roles:
            self.roles.append(name)
            self.role_vocab.get(name)                          # allocate the new role vector; old records don't use it
        return self

    def _snapshot(self):
        """B6 -- a full copy of the mutable state (records + rows + pk index) for transactional rollback."""
        return (self.records.copy(), [dict(r) for r in self.rows],
                {k: list(v) for k, v in self._pk_index.items()})

    def _restore(self, snap):
        """B6 -- restore the state captured by _snapshot (roll back to before the transaction)."""
        self.records, self.rows, self._pk_index = snap[0], snap[1], snap[2]

    def insert(self, row):
        """Append one row: encode it to a record (same encoding as from_rows) and store the exact values. B5:
        constraints are enforced FIRST, so a violating row is refused and the table is left unchanged."""
        self._enforce_constraints(row)                        # B5: refuse the row before any state changes
        rec = _encode_row(row, self.roles, self.role_vocab, self.value_vocab, self.dim)
        self.records = rec[None, :] if len(self.records) == 0 else np.vstack([self.records, rec[None, :]])
        self.rows.append(dict(row))
        if self._pk is not None:                              # B4: keep the primary-key index in sync
            self._pk_index.setdefault(row.get(self._pk), []).append(len(self.rows) - 1)
        return self


class Database:
    """A collection of NAMESPACES, each holding tables (and views). The 'system' namespace is READ-ONLY -- it is
    where the mind publishes its own tables (the scene, the capability registry, recall memory); user namespaces
    are writable. The one rule that makes it safe to open the door this wide: you can SELECT from ANY namespace,
    but you can only write to your OWN. That rule is enforced at a single write chokepoint (_require_writable).

    Persistence is by REPLAY: a user table is just (columns, dim, seed, rows), so to_state saves those and
    from_state rebuilds the vectors deterministically by re-inserting -- no need to serialize the vocab vectors,
    and the reload is byte-identical because the seed fixes every atom."""

    def __init__(self):
        self.namespaces = {}                                       # name -> {"writable", "tables", "views"}
        self.add_namespace("system", writable=False)

    # -- namespaces ---------------------------------------------------------------------------------------------
    def add_namespace(self, name, writable=True, tier=None):
        """Create a namespace. WS1: each namespace has a TIER -- 'system' (read-only, the mind's live state),
        'persistent' (the durable user DB, survives reset/sessions), or 'workspace' (transient session scratch,
        cleared with the workspace). If `tier` is omitted it is inferred from `writable` for backward compatibility."""
        if tier is None:
            tier = "system" if not writable else "persistent"
        if name not in self.namespaces:
            self.namespaces[name] = {"writable": tier != "system", "tier": tier, "tables": {}, "views": {}}
        return self

    def create_namespace(self, name, tier="persistent"):
        """WS1 -- create a writable namespace in a given tier ('persistent' or 'workspace')."""
        if tier == "system":
            raise QueryError("cannot create a namespace in the read-only 'system' tier")
        self.add_namespace(name, writable=True, tier=tier)
        return self

    def drop_namespace(self, name):
        """WS -- remove a namespace and everything in it (used to clear a workspace). The system tier is protected."""
        if self.namespaces.get(name, {}).get("tier") == "system":
            raise QueryError("the 'system' tier cannot be dropped")
        self.namespaces.pop(name, None)
        return self

    def tier_of(self, name):
        """The tier of a namespace ('system' | 'persistent' | 'workspace'), or None if it does not exist."""
        return self.namespaces.get(name, {}).get("tier")

    def create_database(self, name):
        """Create a user-owned WRITABLE namespace beside the read-only system one."""
        if name == "system":
            raise QueryError("'system' is reserved and read-only")
        return self.add_namespace(name, writable=True)

    def register_system(self, table_name, table):
        """Publish a read-only table under system.* (e.g. system.actions = the capability registry)."""
        self.namespaces["system"]["tables"][table_name] = table
        return self

    # -- resolve + the write wall -------------------------------------------------------------------------------
    def _split(self, qualified):
        if "." not in qualified:
            raise QueryError("use a qualified name 'namespace.table', got %r" % (qualified,))
        return qualified.split(".", 1)

    def resolve(self, qualified):
        """Get a table by 'namespace.table'. Reads are allowed from ANY namespace, including system."""
        ns, name = self._split(qualified)
        if ns not in self.namespaces or name not in self.namespaces[ns]["tables"]:
            raise QueryError("no such table %r" % (qualified,))
        entry = self.namespaces[ns]["tables"][name]
        if getattr(self, "_cold", None) is not None:          # cold storage on: warm-on-access + track recency
            from holographic.caching_and_storage.holographic_coldstore import Cold
            with self._cold["lock"]:
                if isinstance(entry, Cold):
                    entry = entry.warm()                      # inflate a cooled table transparently
                    self.namespaces[ns]["tables"][name] = entry
                self._touch(ns, name)
        return entry

    # -- cold storage: fold up INACTIVE user tables to save memory, unfurl on access (OPT-IN) ------------------
    def enable_cold_storage(self, keep_warm=8, codec="zlib", spill_dir=None):
        """Turn on OPT-IN auto-cooling for this database's USER tables. cool_idle() compresses the tables you haven't
        touched lately (freeing their RAM); the next query that resolves one warms it back transparently. System
        tables (the mind's live state) are never cooled. OFF by default, so nothing changes unless you ask.

        Safe for DISTRIBUTED use by construction: if a cold-enabled database is pickled (e.g. shipped to a worker as a
        shared read-only cache), it is WARMED first and arrives with cooling DISABLED -- a plain, immutable, warm copy.
        So a worker's reads never mutate the shared cache, and no lock or spill-file path ever crosses a process
        boundary. Re-enable it in the child only if you actually want cooling there."""
        import threading
        from collections import OrderedDict
        self._cold = {"keep_warm": keep_warm, "codec": codec, "spill_dir": spill_dir,
                      "lru": OrderedDict(), "lock": threading.RLock()}
        return self

    def disable_cold_storage(self):
        """Warm everything back and turn cooling off."""
        self.warm_all()
        self._cold = None
        return self

    def _touch(self, ns, name):
        """Mark (ns, name) most-recently-used in the LRU order (called from resolve while holding the lock)."""
        lru = self._cold["lru"]
        lru.pop((ns, name), None)
        lru[(ns, name)] = True

    def _user_tables(self):
        """(namespace, table) for every WRITABLE (non-system) table -- the only ones we ever cool."""
        return [(ns, nm) for ns, body in self.namespaces.items()
                if body.get("writable") and ns != "system" for nm in list(body["tables"])]

    def cool_idle(self):
        """Compress every user table EXCEPT the `keep_warm` most-recently-resolved ones, freeing their RAM. Call this
        when the database is IDLE -- no query or transaction in flight -- because cooling replaces a table object with
        its compressed form, and warming later builds a fresh object; doing that mid-operation could strand a live
        reference. Warming on the next resolve() is automatic. Returns how many tables were cooled."""
        if getattr(self, "_cold", None) is None:
            return 0
        from holographic.caching_and_storage.holographic_coldstore import Cold
        with self._cold["lock"]:
            keep = self._cold["keep_warm"]
            recent = set(list(self._cold["lru"])[-keep:]) if keep else set()
            cooled = 0
            for ns, nm in self._user_tables():
                entry = self.namespaces[ns]["tables"][nm]
                if isinstance(entry, Cold) or (ns, nm) in recent:
                    continue                                   # already cold, or one of the hot ones we keep warm
                c = Cold(entry, codec=self._cold["codec"], spill_dir=self._cold["spill_dir"])
                c.cool()
                self.namespaces[ns]["tables"][nm] = c
                cooled += 1
            return cooled

    def warm_all(self):
        """Inflate every cooled table back to a live Table (used before serialising or sharing). Safe anytime."""
        if getattr(self, "_cold", None) is None:
            return self
        from holographic.caching_and_storage.holographic_coldstore import Cold
        with self._cold["lock"]:
            for body in self.namespaces.values():
                for nm, entry in list(body["tables"].items()):
                    if isinstance(entry, Cold):
                        body["tables"][nm] = entry.warm()
        return self

    def cold_stats(self):
        """Memory picture: how many user tables are warm vs cold, and the compressed footprint of the cold ones."""
        from holographic.caching_and_storage.holographic_coldstore import Cold
        warm = cold = cold_bytes = 0
        for body in self.namespaces.values():
            for entry in body["tables"].values():
                if isinstance(entry, Cold):
                    cold += 1
                    cold_bytes += entry.cold_bytes()
                else:
                    warm += 1
        return {"warm": warm, "cold": cold, "cold_bytes": cold_bytes,
                "enabled": getattr(self, "_cold", None) is not None}

    def __getstate__(self):
        """Pickle in a DISTRIBUTED-SAFE form: warm every cooled table and drop the cold-storage machinery (its lock and
        LRU state). A worker that receives this database gets a plain, immutable, warm copy with cooling OFF, so reads
        can't mutate the shared cache and the (unpicklable) lock / spill paths never cross the process boundary."""
        self.warm_all()
        state = dict(self.__dict__)
        state["_cold"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_cold" not in self.__dict__:
            self._cold = None

    def _require_writable(self, ns):
        """The single write chokepoint -- the system wall lives here."""
        if ns not in self.namespaces:
            raise QueryError("no such namespace %r" % (ns,))
        if not self.namespaces[ns]["writable"]:
            raise QueryError("namespace %r is read-only (the system wall): reads are fine, writes are refused"
                             % (ns,))

    # -- CREATE / INSERT ----------------------------------------------------------------------------------------
    def create_table(self, qualified, columns, dim=1024, seed=0):
        """CREATE TABLE namespace.name (cols): allocate a writable UserTable in a writable namespace."""
        ns, name = self._split(qualified)
        self._require_writable(ns)
        t = UserTable(name, columns, dim=dim, seed=seed)
        self.namespaces[ns]["tables"][name] = t
        return t

    def insert(self, qualified, row):
        """INSERT one row into a user table (refused for system.* -- the wall)."""
        ns, _name = self._split(qualified)
        self._require_writable(ns)
        self.resolve(qualified).insert(row)
        return self

    # -- bookmarks: cross-namespace INSERT ... SELECT (Phase 11) ------------------------------------------------
    def insert_select(self, dest, columns, source, where=None, mode="snapshot", id_col="id"):
        """Pull rows from `source` (often a system table) into the user table `dest` -- a bookmark. Two honest
        flavours: mode='snapshot' copies the VALUES (survives later source changes, never dangles); mode='reference'
        stores the source id and resolves LIVE at read (always current, but can dangle -> resolve-or-null). The
        write only ever lands in the user's table; cross-namespace READS are always allowed."""
        ns, _ = self._split(dest)
        self._require_writable(ns)
        q = Query().select(*columns)
        if where:
            q.where(*where)
        for r in q.run(self.resolve(source)):
            if mode == "reference":
                self.resolve(dest).insert({"_ref_source": source, "_ref_id": r.get(id_col)})
            else:
                self.resolve(dest).insert({c: r.get(c) for c in columns})
        return self

    def resolve_reference(self, dest_row):
        """Read a reference bookmark LIVE: resolve its stored (source, id) against the current source, returning
        the row or None if it has been deleted (resolve-or-null -- references can dangle)."""
        src = dest_row.get("_ref_source")
        rid = dest_row.get("_ref_id")
        if src is None:
            return None
        try:
            table = self.resolve(src)
        except QueryError:
            return None
        for row in table.rows:
            if row.get("id") == rid:
                return row
        return None                                                # the referenced row is gone -> null

    # -- views: a saved SELECT re-run on read (Phase 12) --------------------------------------------------------
    def create_view(self, qualified, select_cols, source, where=None, order=None):
        """A VIEW is a stored query re-run on read (LIVE): since a query lowers to a plan, a view is a saved plan.
        A live view reflects a source change on the next read; a materialised snapshot would not (that is just a
        UserTable filled by insert_select)."""
        ns, name = self._split(qualified)
        self._require_writable(ns)
        self.namespaces[ns]["views"][name] = {"select": list(select_cols), "source": source,
                                              "where": where, "order": order}
        return self

    def run_view(self, qualified):
        """Execute a view: re-run its saved SELECT against the CURRENT source."""
        ns, name = self._split(qualified)
        v = self.namespaces.get(ns, {}).get("views", {}).get(name)
        if v is None:
            raise QueryError("no such view %r" % (qualified,))
        q = Query().select(*v["select"])
        if v["where"]:
            q.where(*v["where"])
        if v["order"]:
            q.order_by(*v["order"])
        return q.run(self.resolve(v["source"]))

    # -- catalog + persistence (Phase 13) -----------------------------------------------------------------------
    def catalog(self):
        """SHOW DATABASES/TABLES: the catalog as plain rows -- every namespace, table/view, and its writability."""
        rows = []
        for ns, body in sorted(self.namespaces.items()):
            for tname in sorted(body["tables"]):
                rows.append({"namespace": ns, "name": tname, "kind": "table", "writable": body["writable"]})
            for vname in sorted(body.get("views", {})):
                rows.append({"namespace": ns, "name": vname, "kind": "view", "writable": body["writable"]})
        return rows

    def to_state(self, tiers=None):
        """Serialise the USER namespaces by REPLAY: save each table's (columns, dim, seed, rows) and each view's
        spec -- deterministic, small, and re-encodes byte-identically on load (the seed fixes every atom). The
        read-only system tables are the mind's live state, not the user's data, so they are NEVER saved here. WS2:
        pass `tiers` (e.g. ['persistent']) to save only certain tiers -- persistent for the durable DB, 'workspace'
        for a single session's scratch."""
        self.warm_all()                                       # cold storage: inflate any cooled tables so none are missed
        out = {"namespaces": {}}
        for ns, body in self.namespaces.items():
            if ns == "system" or not body["writable"]:
                continue
            if tiers is not None and body.get("tier") not in tiers:      # WS2: tier filter
                continue
            tables = {name: {"columns": t.roles, "dim": t.dim, "seed": getattr(t, "seed", 0), "rows": t.rows}
                      for name, t in body["tables"].items() if isinstance(t, UserTable)}
            out["namespaces"][ns] = {"tables": tables, "views": dict(body.get("views", {})), "tier": body.get("tier")}
        return out

    @classmethod
    def from_state(cls, state, system_tables=None):
        """Rebuild a Database from to_state by REPLAYING inserts (deterministic). Optionally re-attach live system
        tables (which were not persisted)."""
        db = cls()
        if system_tables:
            for name, table in system_tables.items():
                db.register_system(name, table)
        for ns, body in state.get("namespaces", {}).items():
            tier = body.get("tier", "persistent")             # WS2: restore into the saved tier (default persistent)
            if tier == "workspace":
                db.create_namespace(ns, tier="workspace")
            else:
                db.create_database(ns)
            for name, spec in body.get("tables", {}).items():
                db.create_table("%s.%s" % (ns, name), spec["columns"], dim=spec["dim"], seed=spec["seed"])
                for row in spec["rows"]:
                    db.resolve("%s.%s" % (ns, name)).insert(row)
            db.namespaces[ns]["views"] = dict(body.get("views", {}))
        return db


def _parse_values(text):
    """Parse a VALUES tuple's contents into Python values: quoted -> str, numeric -> int/float, else bareword str.
    A tiny readable parser -- not a full SQL literal grammar."""
    import re
    vals = []
    for tok in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", text):   # split on commas outside quotes
        tok = tok.strip()
        if tok[:1] in "'\"" and tok[-1:] in "'\"":
            vals.append(tok[1:-1])
        elif re.match(r"^-?\d+$", tok):
            vals.append(int(tok))
        elif re.match(r"^-?\d+\.\d+$", tok):
            vals.append(float(tok))
        else:
            vals.append(tok)
    return vals


def run_db_sql(sql, db):
    """A small SQL skin over a Database: CREATE DATABASE, CREATE TABLE ns.t (cols), INSERT INTO ns.t (cols)
    VALUES (...), and SELECT ... FROM ns.table (which resolves the qualified table and runs the Part-1 query on
    it). Writes to system.* are refused by the Database's wall. A documented subset, on purpose -- bookmarks and
    views stay on the clearer object API (insert_select / create_view)."""
    import re
    s = sql.strip().rstrip(";")
    low = s.lower()
    if low.startswith("create database"):
        name = s[len("create database"):].strip()
        db.create_database(name)
        return {"created_database": name}
    if low.startswith("create table"):
        m = re.match(r"(?is)^create\s+table\s+([\w.]+)\s*\(([^)]*)\)\s*$", s)
        if not m:
            raise QueryError("CREATE TABLE ns.name (col1, col2, ...)")
        db.create_table(m.group(1), [c.strip() for c in m.group(2).split(",")])
        return {"created_table": m.group(1)}
    if low.startswith("insert into"):
        m = re.match(r"(?is)^insert\s+into\s+([\w.]+)\s*\(([^)]*)\)\s*values\s*\((.*)\)\s*$", s)
        if not m:
            raise QueryError("INSERT INTO ns.name (cols) VALUES (vals)  -- for bookmarks use insert_select()")
        cols = [c.strip() for c in m.group(2).split(",")]
        vals = _parse_values(m.group(3))
        db.insert(m.group(1), dict(zip(cols, vals)))
        return {"inserted": 1}
    # SELECT ... FROM a JOIN b ON key -- a relational join over the API (before the plain-SELECT path, which a join
    # query would otherwise match on its leading SELECT). See _run_join for the supported shape.
    if _re.search(r"(?is)\bjoin\b", s) and low.startswith("select"):
        return _run_join(s, db)
    # UPDATE ns.t SET c=v, ... WHERE ...  -- a WHERE is REQUIRED here on purpose: a networked SQL endpoint should not
    # let a typo rewrite a whole table. (Use the object API for a deliberate no-WHERE bulk write.)
    if low.startswith("update "):
        return _run_update(s, db)
    # DELETE FROM ns.t WHERE ...  -- WHERE required, same safety guard as UPDATE.
    if low.startswith("delete "):
        return _run_delete(s, db)
    # DROP TABLE ns.t  -- remove a table from its namespace (the system tier is protected by the wall).
    if low.startswith("drop table"):
        m = re.match(r"(?is)^drop\s+table\s+([\w.]+)\s*$", s)
        if not m:
            raise QueryError("DROP TABLE ns.name")
        qn = m.group(1)
        ns, _, name = qn.partition(".")
        if db.tier_of(ns) == "system":
            raise QueryError("cannot drop a table in the read-only 'system' tier")
        if ns not in db.namespaces or name not in db.namespaces[ns]["tables"]:
            raise QueryError("no such table %r" % qn)
        del db.namespaces[ns]["tables"][name]
        return {"dropped_table": qn}
    # SELECT: pull the qualified table name, resolve it, run the Part-1 query on the bare name
    m = re.match(r"(?is)^select\s+.+?\s+from\s+([\w.]+)", s)
    if not m:
        raise QueryError("unsupported statement (CREATE DATABASE/TABLE, INSERT ... VALUES, or SELECT ... FROM t)")
    qn = m.group(1)
    table = db.resolve(qn)
    bare = qn.split(".")[-1] if "." in qn else qn
    return run_sql(s.replace(qn, bare, 1), table)


def _run_update(s, db):
    """UPDATE ns.t SET c1=v1, c2=v2 WHERE <predicate>. Resolves the table, parses the assignments and the (required)
    WHERE, and delegates to update() -- which tombstones + re-inserts, the append-only write model. Returns the count."""
    import re
    m = re.match(r"(?is)^update\s+([\w.]+)\s+set\s+(.+?)\s+where\s+(.+)$", s)
    if not m:
        raise QueryError("UPDATE ns.name SET col=val, ... WHERE predicate  (a WHERE is required)")
    table = db.resolve(m.group(1))
    # parse "a = 1, b = 'x'" -> {a: 1, b: 'x'}; the value pattern allows quoted strings (which may contain commas)
    pairs = re.findall(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,]+)", m.group(2))
    changes = {col: _coerce_value(val) for col, val in pairs}
    n = update(table, m.group(3).strip(), changes)               # update() takes a WHERE STRING
    return {"updated": n}


def _run_delete(s, db):
    """DELETE FROM ns.t WHERE <predicate>. Resolves the table and delegates to delete() (tombstone the matches).
    WHERE is required -- the same guard as UPDATE against a whole-table wipe from a typo."""
    import re
    m = re.match(r"(?is)^delete\s+from\s+([\w.]+)\s+where\s+(.+)$", s)
    if not m:
        raise QueryError("DELETE FROM ns.name WHERE predicate  (a WHERE is required)")
    table = db.resolve(m.group(1))
    n = delete(table, m.group(2).strip())
    return {"deleted": n}


def _run_join(s, db):
    """SELECT cols FROM a [INNER|LEFT] JOIN b ON key [WHERE col op val]. Resolves both tables, hash-joins on the key,
    then applies an optional single-predicate WHERE and projects the requested columns. Returns the joined rows.
    A readable subset: one join, one optional predicate -- for anything richer, use the object join()/Query API."""
    import re
    m = re.match(r"(?is)^select\s+(.+?)\s+from\s+([\w.]+)\s+(inner\s+|left\s+)?join\s+([\w.]+)\s+on\s+(\w+)"
                 r"(?:\s+where\s+(\w+)\s*(=|!=|>|<|>=|<=)\s*('[^']*'|\"[^\"]*\"|\S+))?\s*$", s)
    if not m:
        raise QueryError("SELECT cols FROM a JOIN b ON key [WHERE col op val]")
    cols_raw, lqn, how_raw, rqn, key, wcol, wop, wval = m.groups()
    left, right = db.resolve(lqn), db.resolve(rqn)
    how = "left" if (how_raw and how_raw.strip().lower() == "left") else "inner"
    rows = join(left, right, on=key, how=how)                    # the exact hash-join on the stored values
    # optional single-predicate WHERE, applied to the joined dicts
    if wcol is not None:
        want = _coerce_value(wval)
        rows = [r for r in rows if _cmp(r.get(wcol), wop, want)]
    # column projection: '*' keeps everything, otherwise keep the named columns
    cols = [c.strip() for c in cols_raw.split(",")]
    if cols == ["*"]:
        return rows
    return [{c: r.get(c) for c in cols} for r in rows]


def _cmp(a, op, b):
    """Compare two stored values by a WHERE operator (used by the join's optional predicate)."""
    if a is None:
        return False
    try:
        if op == "=":
            return a == b
        if op == "!=":
            return a != b
        if op == ">":
            return a > b
        if op == "<":
            return a < b
        if op == ">=":
            return a >= b
        if op == "<=":
            return a <= b
    except TypeError:
        return False
    return False


def _coerce_value(val):
    """A WHERE literal -> Python value: quoted -> str, numeric -> float, else bareword str."""
    import re
    val = val.strip()
    if val[:1] in "'\"" and val[-1:] in "'\"":
        return val[1:-1]
    return float(val) if re.match(r"^-?\d+(\.\d+)?$", val) else val


# B3: a WHERE predicate tree. A node is ('pred', col, op, value) or ('and'|'or', left, right). A small, readable
# recursive-descent parser with the usual precedence (OR lowest, AND higher, parentheses highest) -- NOT a full SQL
# grammar, on purpose. The single-predicate case is just a leaf, so the old behaviour is a special case of this.
import re as _re
_WHERE_TOK = _re.compile(r"'[^']*'|\"[^\"]*\"|[()]|>=|<=|!=|=|>|<|~|[^\s()=<>~!]+")


def _tokenize_where(s):
    """Split a WHERE string into tokens: quoted strings, parens, comparison operators, and barewords (which include
    column names, numbers, and the keywords AND/OR). Operators split even without surrounding spaces (legs>2)."""
    return _WHERE_TOK.findall(s)


def parse_where(text):
    """Parse a WHERE string into a predicate tree. Raises ValueError on a malformed clause."""
    toks = _tokenize_where(text)
    if not toks:
        raise ValueError("empty WHERE")
    pos = [0]
    tree = _parse_or(toks, pos)
    if pos[0] != len(toks):
        raise ValueError("unexpected trailing tokens in WHERE: %s" % " ".join(toks[pos[0]:]))
    return tree


def _parse_or(toks, pos):
    node = _parse_and(toks, pos)
    while pos[0] < len(toks) and toks[pos[0]].upper() == "OR":
        pos[0] += 1
        node = ("or", node, _parse_and(toks, pos))
    return node


def _parse_and(toks, pos):
    node = _parse_factor(toks, pos)
    while pos[0] < len(toks) and toks[pos[0]].upper() == "AND":
        pos[0] += 1
        node = ("and", node, _parse_factor(toks, pos))
    return node


def _parse_factor(toks, pos):
    if pos[0] < len(toks) and toks[pos[0]] == "(":                # a parenthesised sub-expression
        pos[0] += 1
        node = _parse_or(toks, pos)
        if pos[0] >= len(toks) or toks[pos[0]] != ")":
            raise ValueError("unbalanced parentheses in WHERE")
        pos[0] += 1
        return node
    if pos[0] + 2 >= len(toks) + 1 or pos[0] + 2 > len(toks):     # need col op value
        raise ValueError("incomplete predicate in WHERE (expected col op value)")
    col, op, val = toks[pos[0]], toks[pos[0] + 1], toks[pos[0] + 2]
    if op not in ("=", ">", "<", ">=", "<=", "!=", "~"):
        raise ValueError("bad operator %r in WHERE" % op)
    pos[0] += 3
    return ("pred", col, op, _coerce_value(val))


def _where_columns(tree):
    """Every column named anywhere in a predicate tree (for schema validation)."""
    if tree[0] == "pred":
        return {tree[1]}
    return _where_columns(tree[1]) | _where_columns(tree[2])


def _tree_has_fuzzy(tree):
    """True if any leaf of the predicate tree is a fuzzy (~) match -- such queries rank by similarity by default."""
    if tree[0] == "pred":
        return tree[2] == "~"
    return _tree_has_fuzzy(tree[1]) or _tree_has_fuzzy(tree[2])


def _compare(v, op, val):
    """Compare a stored value against a literal with a SQL operator (used by HAVING). None fails any comparison."""
    if v is None:
        return False
    if op == "=":
        return v == val
    if op == "!=":
        return v != val
    num = isinstance(v, (int, float))
    if op == ">":
        return num and v > val
    if op == "<":
        return num and v < val
    if op == ">=":
        return num and v >= val
    if op == "<=":
        return num and v <= val
    return False


def project(record, cols, table):
    """SELECT: unbind each requested role and clean it up to the nearest known filler -- relations.NAME, batched.
    Returns {col: filler_name_or_None}. A column that was never bound cleans up weakly -> reported as None."""
    out = {}
    for c in cols:
        if c not in table.role_vocab.vectors:
            out[c] = None
            continue
        noisy = unbind(record, table.role_vocab.get(c))
        name, conf = table.value_vocab.cleanup(noisy)             # snap to the nearest filler; cleanup returns (name, sim)
        # weak recovery (no strong match) -> null-ish, so a missing/absent column reads as None not a wrong guess
        out[c] = name if conf > 0.15 else None
    return out


class Query:
    """A projection over a Table: select columns, filter (exact on stored rows OR fuzzy on the vectors), rank (by
    a column or by fuzzy similarity), limit. `run` returns a list of dict rows, each with a `_confidence`."""

    def __init__(self):
        self._select = None            # columns to return (None = all roles)
        self._where = None             # (col, op, value) ; op in {'=','>','<','~'}
        self._order = None             # (col_or_'similarity', descending)
        self._limit = None
        self._group = None             # GROUP BY column (exact grouping on the stored value)
        self._aggs = []                # aggregate specs: list of (func, col, label)
        self._distinct = False         # B9: SELECT DISTINCT -> dedupe result rows
        self._offset = None            # B9: OFFSET n -> skip n rows before LIMIT
        self._having = None            # B9: HAVING (label, op, value) -> filter groups after aggregation

    def select(self, *cols):
        self._select = list(cols) if cols else None
        return self

    def where(self, col, op, value):
        self._where = ("pred", col, op, value)                   # a single predicate is just a leaf of the tree
        return self

    def where_tree(self, tree):
        """B3: filter by a WHERE predicate TREE ('and'/'or'/parens over leaf predicates)."""
        self._where = tree
        return self

    def _leaf_mask(self, table, col, op, val):
        """Evaluate ONE predicate over the table -> (keep_set, scores_or_None). Exact ops run on the STORED props
        (lossless); '~' is the fuzzy cosine of the unbound filler to the probe (scores kept for ranking)."""
        n = len(table)
        if op == "~":
            probe = table.value_vocab.get(val)                   # a novel probe still gets a (random) direction
            role = table.role_vocab.get(col)
            scores = [float(cosine(unbind(table.records[i], role), probe)) for i in range(n)]
            return set(i for i in range(n) if scores[i] > 0.25), scores

        def ok(i):
            v = table.rows[i].get(col)
            if v is None:
                return False
            if op == "=":
                return v == val
            if op == "!=":
                return v != val
            num = isinstance(v, (int, float))
            if op == ">":
                return num and v > val
            if op == "<":
                return num and v < val
            if op == ">=":
                return num and v >= val
            if op == "<=":
                return num and v <= val
            return False
        return set(i for i in range(n) if ok(i)), None

    def _eval_where(self, table, tree):
        """Evaluate a WHERE predicate tree -> (keep_set, sims). AND/OR intersect/union the per-leaf row masks;
        sims carry any fuzzy leaf's score (max where two combine) for ranking, else 1.0."""
        if tree[0] == "pred":
            keep, scores = self._leaf_mask(table, tree[1], tree[2], tree[3])
            return keep, (scores if scores is not None else [1.0] * len(table))
        lkeep, lsims = self._eval_where(table, tree[1])
        rkeep, rsims = self._eval_where(table, tree[2])
        keep = (lkeep & rkeep) if tree[0] == "and" else (lkeep | rkeep)
        sims = [max(lsims[i], rsims[i]) for i in range(len(table))]
        return keep, sims

    def group_by(self, col):
        self._group = col
        return self

    def aggregate(self, func, col, label=None):
        """Add an aggregate to compute per group: func in COUNT/SUM/AVG/MIN/MAX, over the stored column `col`
        ('*' for COUNT). `label` is the output key (defaults to e.g. 'AVG(density)')."""
        func = func.upper()
        self._aggs.append((func, col, label or ("%s(%s)" % (func, col))))
        return self

    def order_by(self, key, descending=True):
        self._order = (key, descending)
        return self

    def limit(self, k):
        self._limit = int(k)
        return self

    def distinct(self, on=True):
        self._distinct = bool(on)                                # B9: dedupe the result rows
        return self

    def offset(self, k):
        self._offset = int(k)                                    # B9: skip the first k rows (after ordering)
        return self

    def having(self, label, op, value):
        self._having = (label, op, value)                        # B9: filter GROUPs by an aggregate result
        return self

    def run(self, table):
        cols = self._select or list(table.roles)

        # F1: a query column must be a DECLARED role of the table. A column that exists but is absent from a given
        # row is fine (sparse rows -> None), but a column the table never declared is an error, not a confident null.
        known = set(table.roles)
        for c in cols:
            if c not in known:
                raise QueryError("column %r does not exist (columns: %s)" % (c, ", ".join(table.roles)))
        if self._where is not None:
            for _wc in _where_columns(self._where):
                if _wc not in known:
                    raise QueryError("column %r does not exist (columns: %s)" % (_wc, ", ".join(table.roles)))
        if self._group is not None and self._group not in known:
            raise QueryError("column %r does not exist (columns: %s)" % (self._group, ", ".join(table.roles)))
        for _func, _col, _label in self._aggs:
            if _col != "*" and _col not in known:
                raise QueryError("column %r does not exist (columns: %s)" % (_col, ", ".join(table.roles)))
        agg_labels = {lab for (_f, _c, lab) in self._aggs}       # ORDER BY may target an aggregate's output label
        if (self._order is not None and self._order[0] != "similarity"
                and self._order[0] not in known and self._order[0] not in agg_labels):
            raise QueryError("column %r does not exist (columns: %s)" % (self._order[0], ", ".join(table.roles)))

        n = len(table)
        sims = None                                                # per-row match strength; built lazily (fuzzy only)

        # B4 fast path: a single `pk = value` predicate reads the O(1) hash index directly -- no O(n) scan at all.
        _pk = getattr(table, "_pk", None)
        _pk_fast = (self._where is not None and _pk is not None and self._where[0] == "pred"
                    and self._where[1] == _pk and self._where[2] == "=")
        if _pk_fast:
            idx = table.pk_lookup(self._where[3])                 # already live (skips tombstones), O(1)
        else:
            idx = [i for i in range(n) if not table.rows[i].get("_deleted")]   # B2: scans skip tombstoned rows
            if self._where is not None:
                keep, sims = self._eval_where(table, self._where)  # B3: intersect/union the per-leaf masks
                idx = [i for i in idx if i in keep]

        # AGGREGATION (Phase 5): if the query GROUPs or asks for aggregates, collapse the filtered rows into one
        # row per group and compute the aggregates -- COUNT/SUM/AVG/MIN/MAX exact on the STORED props, plus a VSA
        # _centroid (the group's bundle: superposition = the group's prototype vector).
        if self._aggs or self._group is not None:
            return self._run_aggregate(table, idx)

        # rank
        if self._order is not None:
            key, desc = self._order
            if key == "similarity":
                _s = sims if sims is not None else [1.0] * n
                idx.sort(key=lambda i: (_s[i], -i), reverse=desc)   # tie-break by row index (stable, deterministic)
            else:
                idx.sort(key=lambda i: (table.rows[i].get(key, 0), -i), reverse=desc)
        elif sims is not None and self._where is not None and _tree_has_fuzzy(self._where):
            idx.sort(key=lambda i: (sims[i], -i), reverse=True)       # fuzzy queries rank by match by default

        # OFFSET/LIMIT: without DISTINCT they trim the index directly (cheap); with DISTINCT they must apply to the
        # DEDUPED rows, so we defer them until after projection.
        if not self._distinct:
            if self._offset is not None:                             # B9: OFFSET skips rows before LIMIT
                idx = idx[self._offset:]
            if self._limit is not None:
                idx = idx[:self._limit]

        results = []
        seen = set()
        for i in idx:
            row = {}
            for c in cols:
                if c in table.rows[i]:
                    row[c] = table.rows[i][c]                         # stored exact value (avoid the O(codebook) decode)
                else:
                    row[c] = project(table.records[i], [c], table)[c]  # only decode from the vector when truly absent
            if self._distinct:                                       # B9: skip a row whose projected values were seen
                sig = tuple((c, row[c]) for c in cols)
                if sig in seen:
                    continue
                seen.add(sig)
            row["_confidence"] = float(sims[i]) if sims is not None else 1.0
            results.append(row)

        if self._distinct:                                           # OFFSET/LIMIT now apply to the deduped rows
            if self._offset is not None:
                results = results[self._offset:]
            if self._limit is not None:
                results = results[:self._limit]
        return results

    def _compute_agg(self, func, col, members, table):
        """One aggregate over a group's rows. COUNT is exact; SUM/AVG/MIN/MAX run on the STORED numeric props, so
        they are EXACT too (no encoder readback error -- that is exactly why Table keeps the stored props beside
        the vectors). Non-numeric or absent values are skipped; an empty numeric set returns None."""
        if func == "COUNT":
            return len(members)
        vals = [table.rows[i].get(col) for i in members]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return None
        if func == "SUM":
            return sum(vals)
        if func == "AVG":
            return sum(vals) / len(vals)
        if func == "MIN":
            return min(vals)
        if func == "MAX":
            return max(vals)
        raise ValueError("unknown aggregate %r" % (func,))

    def _run_aggregate(self, table, idx):
        """Group the filtered rows by the GROUP BY column's stored value (exact grouping), then compute each
        aggregate per group. Also returns each group's _centroid -- the bundle (superposition) of its records,
        which IS the group's prototype vector (the VSA aggregate). Deterministic group order."""
        groups = {}
        for i in idx:
            key = table.rows[i].get(self._group) if self._group is not None else "__all__"
            groups.setdefault(key, []).append(i)
        # deterministic group order: Nones last, then by key
        keys = sorted(groups.keys(), key=lambda k: (k is None, str(k)))
        out = []
        for key in keys:
            members = groups[key]
            row = {}
            if self._group is not None:
                row[self._group] = key
            for func, col, label in self._aggs:
                row[label] = self._compute_agg(func, col, members, table)
            row["_count"] = len(members)
            row["_centroid"] = bundle([table.records[i] for i in members]) if members else None
            out.append(row)
        if self._having is not None:                             # B9: HAVING filters GROUPs by an aggregate result
            hlabel, hop, hval = self._having
            out = [r for r in out if _compare(r.get(hlabel), hop, hval)]
        # order/limit apply to the grouped rows (by an aggregate label or the group column)
        if self._order is not None:
            okey, desc = self._order
            out.sort(key=lambda r: (r.get(okey) is None, r.get(okey)), reverse=desc)
        if self._offset is not None:                             # B9: OFFSET on grouped rows too
            out = out[self._offset:]
        if self._limit is not None:
            out = out[:self._limit]
        return out


# --- a small, documented SQL subset: SELECT ... FROM ... [WHERE ...] [ORDER BY ...] [LIMIT n] -------------------

def parse_sql(sql):
    """Parse the documented subset into a plan dict. Supports: SELECT [DISTINCT] cols/aggregates FROM t
    [WHERE tree] [GROUP BY g] [HAVING agg op v] [ORDER BY k] [LIMIT n] [OFFSET m]. WHERE is a predicate TREE
    (AND/OR/parens); aggregates are FUNC(col) (COUNT/SUM/AVG/MIN/MAX). A small readable subset, on purpose."""
    import re
    s = sql.strip().rstrip(";")
    m = re.match(r"(?is)^\s*select\s+(.+?)\s+from\s+(\w+)(?:\s+where\s+(.+?))?"
                 r"(?:\s+group\s+by\s+(\w+))?(?:\s+having\s+(.+?))?(?:\s+order\s+by\s+(.+?))?"
                 r"(?:\s+limit\s+(\d+))?(?:\s+offset\s+(\d+))?\s*$", s)
    if not m:
        raise ValueError("unsupported query (subset is SELECT [DISTINCT] cols FROM t [WHERE ...] [GROUP BY g] "
                         "[HAVING ...] [ORDER BY k] [LIMIT n] [OFFSET m])")
    sel, frm, where, group, having, order, lim, off = m.groups()
    plan = {"select": None, "aggs": [], "from": frm, "where": None,
            "group": group.strip() if group else None, "order": None,
            "limit": int(lim) if lim else None, "offset": int(off) if off else None,
            "distinct": False, "having": None}
    sel = sel.strip()
    if sel.lower().startswith("distinct "):                        # B9: SELECT DISTINCT
        plan["distinct"] = True
        sel = sel[len("distinct "):].strip()
    if sel != "*":                                                 # classify each SELECT item: aggregate vs column
        cols = []
        for item in sel.split(","):
            item = item.strip()
            am = re.match(r"(?i)^(\w+)\s*\(\s*([\w*]+)\s*\)$", item)
            if am:
                func, arg = am.group(1).upper(), am.group(2)
                plan["aggs"].append((func, arg, "%s(%s)" % (func, arg)))
            else:
                cols.append(item)
        plan["select"] = cols or None
    if where:
        plan["where"] = parse_where(where)                        # B3: a predicate TREE (AND/OR/parens)
    if having:                                                     # B9: HAVING (one predicate on an aggregate/column)
        hm = re.match(r"(?is)^\s*(\w+\s*\(\s*[\w*]*\s*\)|\w+)\s*(>=|<=|!=|=|>|<)\s*(.+?)\s*$", having.strip())
        if not hm:
            raise ValueError("unsupported HAVING (one predicate: aggregate op value)")
        hlabel, hop, hval = hm.groups()
        hlabel = re.sub(r"\s+", "", hlabel)                        # 'COUNT ( * )' -> 'COUNT(*)'
        if "(" in hlabel:                                          # uppercase the function name to match the agg label
            fn, rest = hlabel.split("(", 1)
            hlabel = fn.upper() + "(" + rest
        plan["having"] = (hlabel, hop, _coerce_value(hval))
    if order:
        parts = order.strip().split()
        plan["order"] = (parts[0], not (len(parts) > 1 and parts[1].lower() == "asc"))
    return plan


def _run_single_select(sql, table):
    """Lower ONE SELECT to a Query and run it (no UNION handling)."""
    plan = parse_sql(sql)
    q = Query()
    if plan["select"]:
        q.select(*plan["select"])
    if plan["distinct"]:
        q.distinct()
    if plan["where"] is not None:
        q.where_tree(plan["where"])
    if plan.get("group"):
        q.group_by(plan["group"])
    for func, arg, label in plan.get("aggs", []):
        q.aggregate(func, arg, label)
    if plan.get("having") is not None:
        q.having(*plan["having"])
    if plan["order"]:
        q.order_by(plan["order"][0], plan["order"][1])
    if plan["limit"] is not None:                                  # F3: LIMIT 0 is a real limit (return no rows)
        q.limit(plan["limit"])
    if plan["offset"] is not None:
        q.offset(plan["offset"])
    return q.run(table)


def run_sql(sql, table):
    """Lower a SQL string to a Query and run it against a Table -- the SQL skin over the projection core. B9: handles
    UNION / UNION ALL by running each SELECT and combining (a plain UNION anywhere dedupes the final result)."""
    import re
    if re.search(r"(?is)\bunion\b", sql):
        parts = re.split(r"(?is)\bunion\s+all\b|\bunion\b", sql)
        ops = re.findall(r"(?is)\bunion\s+all\b|\bunion\b", sql)
        combined = []
        for p in parts:
            combined.extend(_run_single_select(p, table))
        dedupe = any("all" not in o.lower() for o in ops)          # any plain UNION -> distinct the whole result
        if not dedupe:
            return combined
        seen, out = set(), []
        for r in combined:
            sig = tuple(sorted((k, v) for k, v in r.items() if k != "_confidence"))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(r)
        return out
    return _run_single_select(sql, table)


def similar_to(table, target_row, k=10):
    """PROMOTE P1 -- whole-row "more like this". SQL's LIKE is substring-only and pgvector similarity is per-column;
    this ranks by the cosine of the WHOLE record vector (every column at once), with a per-row confidence. Reuses the
    kernel's cosine over the stored record vectors -- the same 'find the stored thing most like this' analog recall."""
    q = _encode_row(target_row, table.roles, table.role_vocab, table.value_vocab, table.dim)
    scored = sorted(((i, float(cosine(table.records[i], q))) for i in range(len(table))),
                    key=lambda t: (t[1], -t[0]), reverse=True)                # stable, deterministic tie-break
    out = []
    for i, sim in scored[:k]:
        row = dict(table.rows[i])
        row["_confidence"] = sim
        out.append(row)
    return out


def cluster(table, into=3, seed=0):
    """PROMOTE P2 -- semantic GROUP BY. SQL's GROUP BY is exact-value only; this groups rows by SIMILARITY of their
    whole record vectors (cosine k-means on the unit sphere), and reports each cluster's COHERENCE (mean cosine of
    members to their centroid) so a real group is distinguishable from a loose one. Reuses organizer._cosine_kmeans.
    Kept negative: unsupervised -- a suggestion, not a decree; `into` is a hint (empty clusters are dropped)."""
    from holographic.scene_and_pipeline.holographic_organizer import _cosine_kmeans
    n = len(table)
    if n == 0:
        return []
    X = np.asarray(table.records, float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)               # unit sphere -> dot is cosine
    into = max(1, min(int(into), n))
    cent, assign = _cosine_kmeans(X, into, np.random.default_rng(seed))
    clusters = []
    for j in range(into):
        members = [i for i in range(n) if assign[i] == j]
        if not members:
            continue                                                          # drop empty clusters (a hint, not a law)
        coherence = float((X[assign == j] @ cent[j]).mean())
        clusters.append({"cluster": j, "size": len(members), "coherence": coherence,
                         "rows": [dict(table.rows[i]) for i in members]})
    clusters.sort(key=lambda c: (-c["coherence"], c["cluster"]))              # tightest groups first, deterministic
    return clusters


def anomalies(table, seed=0):
    """PROMOTE P3 -- the weird rows, with a CALIBRATED "how weird". SQL forces z-score gymnastics or an export; this
    reports, per row, the false-alarm probability that its NEAREST OTHER row is only a noise-level match (RecallNull's
    calibrated novelty). A high score => the row has no real neighbour => anomalous; if every row has a real neighbour
    the scores are all tiny -> nothing is actually anomalous (it can ABSTAIN instead of always naming N outliers)."""
    from holographic.agents_and_reasoning.holographic_honesty import RecallNull
    n = len(table)
    if n < 2:
        return [dict(r) for r in table.rows]
    C = np.asarray(table.records, float)
    units = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    null = RecallNull().fit(C, seed=seed)                                     # noise floor for THIS table's geometry
    sims = units @ units.T
    np.fill_diagonal(sims, -np.inf)                                          # exclude self -> nearest OTHER row
    out = []
    for i in range(n):
        best = float(sims[i].max())
        row = dict(table.rows[i])
        row["_anomaly_score"] = float(null.pvalue(best))                     # calibrated: is that best match real?
        row["_nn_similarity"] = best
        out.append(row)
    out.sort(key=lambda r: (-r["_anomaly_score"], r["_nn_similarity"]))       # most anomalous first
    return out


def near_duplicates(table, threshold=0.9):
    """PROMOTE P4 -- fuzzy dedup / entity resolution. SQL needs Levenshtein UDFs + self-joins per column; here two
    rows are near-duplicates when their WHOLE record vectors are cosine-close. Returns connected groups (union-find
    over the >=threshold pairs), each with its mean intra-similarity, tightest first. Reuses the kernel cosine. Kept
    negative: it PROPOSES candidates with a confidence -- a human/threshold confirms; it never auto-merges."""
    n = len(table)
    if n < 2:
        return []
    units = np.asarray(table.records, float)
    units = units / (np.linalg.norm(units, axis=1, keepdims=True) + 1e-12)
    sims = units @ units.T
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):                                                       # union every above-threshold pair
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue                                                          # a singleton is not a duplicate group
        pair_sims = [sims[a, b] for a in members for b in members if a < b]
        out.append({"size": len(members), "similarity": float(np.mean(pair_sims)),
                    "rows": [dict(table.rows[i]) for i in members]})
    out.sort(key=lambda g: -g["similarity"])                                  # tightest duplicate groups first
    return out


def explain_match(table, row_a, row_b):
    """PROMOTE P5 -- WHY are two rows similar? SQL can't say. For each column: a CATEGORICAL (string) column is
    compared on the VECTOR side -- unbind its role from both records and cosine the recovered fillers (the resonator's
    decompose as interpretability); a NUMERIC column is compared on the EXACT stored side (equal or not), because
    numerics deliberately never enter the record vector (the honest exact/fuzzy fork). Columns that agree DRIVE the
    match, columns that disagree PULL against it. `overall` is the whole-record (categorical) cosine."""
    ra = _encode_row(row_a, table.roles, table.role_vocab, table.value_vocab, table.dim)
    rb = _encode_row(row_b, table.roles, table.role_vocab, table.value_vocab, table.dim)
    per_col = {}
    for role in table.roles:
        va, vb = row_a.get(role), row_b.get(role)
        if isinstance(va, str) and isinstance(vb, str):                      # fuzzy/vector side (categoricals)
            rv = table.role_vocab.get(role)
            per_col[role] = float(cosine(unbind(ra, rv), unbind(rb, rv)))
        elif va is not None and vb is not None:                              # exact/stored side (numerics)
            per_col[role] = 1.0 if va == vb else 0.0
        else:
            per_col[role] = 0.0                                             # one side missing -> no agreement
    ranked = sorted(per_col.items(), key=lambda kv: -kv[1])
    return {"overall": float(cosine(ra, rb)), "per_column": per_col,
            "drives": [c for c, s in ranked if s > 0.5],                     # strong agreement -> drives the match
            "against": [c for c, s in ranked if s < 0.1]}                    # disagreement -> pulls against it


def recommend(table, example_rows, k=10, exclude=True):
    """PROMOTE P6 -- "more like these, not these". Bundle the example record vectors into one taste vector, recall
    the nearest rows by cosine, and (by default) exclude the examples themselves. Reuses bundle + cosine."""
    ex_vecs = [_encode_row(r, table.roles, table.role_vocab, table.value_vocab, table.dim) for r in example_rows]
    taste = bundle(ex_vecs)                                                   # the superposed "what I like" vector
    ex_rows = [dict(r) for r in example_rows]
    scored = sorted(((i, float(cosine(table.records[i], taste))) for i in range(len(table))),
                    key=lambda t: (t[1], -t[0]), reverse=True)
    out = []
    for i, sim in scored:
        row = dict(table.rows[i])
        if exclude and row in ex_rows:                                       # skip the examples themselves
            continue
        r = dict(row)
        r["_confidence"] = sim
        out.append(r)
        if len(out) >= k:
            break
    return out


def _joined_row(lrow, rrow, lroles, rroles, lkey, rkey, suffix):
    """Merge a left+right row-dict into one. Shared column NAMES (other than the join key) are disambiguated with the
    suffixes; a right key that shares the left key's name is dropped (it duplicates the left value)."""
    shared = set(lroles) & set(rroles)
    out = {}
    for c in lroles:
        out[c + suffix[0] if (c in shared and c != lkey) else c] = lrow.get(c)
    for c in rroles:
        if c == rkey and rkey == lkey:                            # same-named shared key -> already present
            continue
        out[c + suffix[1] if (c in shared and c != rkey) else c] = rrow.get(c)
    return out


def join(left, right, on, how="inner", suffix=("_l", "_r")):
    """B1 -- an exact HASH-JOIN on a shared key. `on` is a key name present in both tables, or a (left_key, right_key)
    pair. Build a dict on the RIGHT table's key once (the hash side), then probe it per LEFT row -> O(n+m), not the
    O(n*m) nested loop. `how` in {'inner','left'} (a left join null-fills the right side on a miss). Exact on the
    STORED values -- lossless. Kept negative: unindexed it is still O(n*m); the hash side makes it O(n+m)."""
    lkey, rkey = (on, on) if isinstance(on, str) else on
    if lkey not in left.roles:
        raise QueryError("join key %r is not a column of the left table" % lkey)
    if rkey not in right.roles:
        raise QueryError("join key %r is not a column of the right table" % rkey)
    index = {}                                                    # the hash side: right key value -> its right rows
    for rrow in right.rows:
        if rrow.get("_deleted"):                                  # B2: skip tombstoned rows so a join sees LIVE data
            continue
        index.setdefault(rrow.get(rkey), []).append(rrow)
    out = []
    for lrow in left.rows:
        if lrow.get("_deleted"):                                  # (an UPDATE tombstones the old row + re-inserts a
            continue                                              #  new one, so without this a join would double-count)
        matches = index.get(lrow.get(lkey), [])
        if matches:
            for rrow in matches:
                out.append(_joined_row(lrow, rrow, left.roles, right.roles, lkey, rkey, suffix))
        elif how == "left":                                       # keep the unmatched left row, null-fill the right
            out.append(_joined_row(lrow, {}, left.roles, right.roles, lkey, rkey, suffix))
    return out


def scalar_key_encoder(lo, hi, dim=1024, n=64, seed=0):
    """A smooth encoder for NUMERIC join keys: nearby numbers map to nearby vectors (RBF bumps over [lo,hi]), so a
    fuzzy_join can match keys that are CLOSE in value -- a proximity join (timestamps within a window, prices near
    each other) SQL makes painful. Returns a function value -> unit vector."""
    rng = np.random.default_rng(seed)
    anchors = np.linspace(lo, hi, n)
    basis = rng.standard_normal((n, dim))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True) + 1e-12
    width = (hi - lo) / max(n - 1, 1)

    def enc(x):
        w = np.exp(-0.5 * ((float(x) - anchors) / width) ** 2)               # a soft bump centred on x
        v = w @ basis
        return v / (np.linalg.norm(v) + 1e-12)
    return enc


def fuzzy_join(left, right, on, threshold=0.5, key_encoder=None, suffix=("_l", "_r")):
    """B1 bonus -- a SEMANTIC join SQL can't do: match rows whose KEY vectors are cosine-close, not exactly equal.
    Both tables' keys are encoded in a SHARED space (each table has its own vocabulary, so identical keys would
    otherwise be orthogonal -- the shared encoder is what makes the match meaningful). Default encoder is categorical
    (identical string keys match, distinct ones don't -> reduces to an exact join, the honest baseline); pass a
    `scalar_key_encoder` to match NUMERIC keys that are merely CLOSE (the real semantic win). Each pair carries a
    `_confidence`. Kept negative: APPROXIMATE and O(n*m); use the exact hash-join when keys are exact."""
    lkey, rkey = (on, on) if isinstance(on, str) else on
    if key_encoder is None:
        shared = Vocabulary(left.dim, seed=0)                                # one shared categorical space for the key
        key_encoder = lambda v: shared.get(str(v))
    lvecs = [key_encoder(lrow.get(lkey)) for lrow in left.rows]
    rvecs = [key_encoder(rrow.get(rkey)) for rrow in right.rows]
    out = []
    for i, lrow in enumerate(left.rows):
        for j, rrow in enumerate(right.rows):
            sim = float(cosine(lvecs[i], rvecs[j]))
            if sim >= threshold:
                row = _joined_row(lrow, rrow, left.roles, right.roles, lkey, rkey, suffix)
                row["_confidence"] = sim
                out.append(row)
    out.sort(key=lambda r: -r["_confidence"])                                # best matches first
    return out


def _matching_indices(table, where):
    """Live row indices matching a WHERE (a SQL predicate string, or a pre-parsed predicate tree)."""
    tree = parse_where(where) if isinstance(where, str) else where
    keep, _ = Query()._eval_where(table, tree)
    return [i for i in keep if not table.rows[i].get("_deleted")]


def delete(table, where):
    """B2 DELETE -- tombstone the matching rows (scans skip them from now on). A record is a bundle (a sum), so you
    cannot cleanly subtract one; instead mark it dead, LSM/event-sourced style. Returns the number tombstoned. Kept
    negative: the row's vector still sits in the store (crosstalk) until compact() -- fine for exact reads, mildly
    degrades fuzzy recall until GC."""
    idx = _matching_indices(table, where)
    for i in idx:
        table.rows[i]["_deleted"] = True
    return len(idx)


def update(table, where, changes):
    """B2 UPDATE -- tombstone each matching row and APPEND a new version carrying the changes (again: you can't edit
    a bundle in place, so retire + re-insert). Returns the number updated."""
    idx = _matching_indices(table, where)
    for i in idx:
        new = dict(table.rows[i])
        new.update(changes)
        new.pop("_deleted", None)
        table.rows[i]["_deleted"] = True                          # retire the old version
        table.insert(new)                                         # append the new one (order-independent bundle)
    return len(idx)


def compact(table):
    """B2 GC -- replay the table WITHOUT the tombstoned rows, returning a fresh table. This is where the retired
    rows' vector crosstalk actually goes away (delete/update only mark; compact reclaims). Reuses the insert path."""
    live = []
    for r in table.rows:
        if not r.get("_deleted"):
            r2 = dict(r)
            r2.pop("_deleted", None)
            live.append(r2)
    t = UserTable(table.name, list(table.roles), dim=table.dim, seed=table.seed)
    t.role_vocab = table.role_vocab                              # keep the (deterministic) encoding stable
    t.value_vocab = table.value_vocab
    for r in live:
        t.insert(r)
    return t


def capability_registry(mind, dim=2048, seed=0):
    """Part 2 / Phase 6: build the mind's capability registry as a VSA TABLE -- one row per public faculty, so
    'what can this mind do?' becomes an ordinary Part-1 data query (introspection = a SELECT over the registry).
    Columns: `name`, `domain` (a heuristic tag from the faculty name), `doc` (its one-line docstring).

    HONEST SCOPE (kept): the domain is a keyword HEURISTIC over the method name, not a curated taxonomy -- good
    enough to answer 'which faculties are about rendering?' but not authoritative; the doc is the method's own
    first docstring line, truncated. A real deployment would register domains deliberately (name + handler)."""
    import inspect
    domain_keys = [
        ("forecast", "forecasting"), ("analog", "forecasting"), ("horizon", "forecasting"),
        ("conformal", "forecasting"), ("recurrent", "forecasting"), ("market", "forecasting"),
        ("svgf", "render"), ("render", "render"), ("splat", "render"), ("denoise", "render"),
        ("adaptive_sample", "render"), ("pipeline", "render"), ("path", "render"),
        ("query", "query"), ("make_table", "query"),
        ("scheduler", "compute"), ("distribute", "compute"), ("fuse", "compute"),
        ("recall", "memory"), ("remember", "memory"), ("recognize", "memory"), ("archive", "memory"),
        ("decide", "agent"), ("reinforce", "agent"), ("act", "agent"),
        ("field", "physics"), ("particle", "physics"), ("fluid", "physics"), ("collide", "physics"),
        ("bind", "kernel"), ("bundle", "kernel"), ("cleanup", "kernel"), ("encode", "kernel"),
        ("mesh", "geometry"), ("sdf", "geometry"), ("sculpt", "geometry"),
    ]
    rows = []
    for name in dir(mind):
        if name.startswith("_"):
            continue
        attr = getattr(type(mind), name, None)
        if not callable(attr):
            continue
        doc = (inspect.getdoc(attr) or "").strip().split("\n")[0][:120]
        domain = "general"
        for key, dom in domain_keys:
            if key in name:
                domain = dom
                break
        rows.append({"name": name, "domain": domain, "doc": doc})
    return from_rows(rows, ["name", "domain", "doc"], dim=dim, seed=seed)


def explain_program(machine, program_vec, init_acc=None):
    """Part 2 / Phase 7: EXPLAIN = a DRY RUN. Run the program with NO handlers, so every APPLY is a no-op and the
    heavy work is skipped -- but the machine still walks the whole program, so the trace tells which faculties it
    WOULD call and how many steps it takes, WITHOUT executing them. The pipeline's plan()->EXPLAIN, one level
    down. Returns {faculties_called, n_steps, trace}."""
    _acc, trace = machine.run(program_vec, init_acc=init_acc, handlers=None)
    faculties = [arg for (op, arg) in trace if op == "APPLY"]
    return {"faculties_called": faculties, "n_steps": len(trace), "trace": trace}


def _selftest():
    """Round-trip a small 'materials' table; exact numeric WHERE on stored props; fuzzy WHERE ranks by meaning
    with a confidence; ORDER BY + LIMIT; the SQL subset parses and runs; deterministic."""
    rows = [
        {"name": "gold", "colour": "yellow", "density": 19300},
        {"name": "copper", "colour": "orange", "density": 8960},
        {"name": "silver", "colour": "grey", "density": 10490},
        {"name": "iron", "colour": "grey", "density": 7870},
        {"name": "lead", "colour": "grey", "density": 11340},
    ]
    roles = ["name", "colour", "density"]
    t = from_rows(rows, roles, dim=2048, seed=0)

    # (1) round-trip: projecting a record recovers its categorical fillers
    p = project(t.records[0], ["name", "colour"], t)
    assert p["name"] == "gold" and p["colour"] == "yellow"

    # (2) exact numeric WHERE runs on the stored props (no readback error) + ORDER BY + LIMIT
    res = run_sql("SELECT name, density FROM materials WHERE density > 9000 ORDER BY density LIMIT 2", t)
    assert [r["name"] for r in res] == ["gold", "lead"]           # the two densest above 9000, descending

    # (3) fuzzy WHERE ranks by meaning and carries a confidence
    res2 = Query().select("name", "colour").where("colour", "~", "grey").order_by("similarity").run(t)
    greys = {r["name"] for r in res2}
    assert {"silver", "iron", "lead"} <= greys                    # the grey metals rank in
    assert all("_confidence" in r for r in res2)
    assert res2[0]["_confidence"] > 0.5                           # a real match scores strongly

    # (4) exact string WHERE via SQL
    res3 = run_sql("SELECT name FROM materials WHERE colour = 'yellow'", t)
    assert [r["name"] for r in res3] == ["gold"]

    # (5) Phase 5: GROUP BY + aggregates (COUNT/AVG exact on stored props) + a per-group centroid (bundle)
    grp = run_sql("SELECT colour, COUNT(*), AVG(density) FROM materials GROUP BY colour ORDER BY colour ASC", t)
    by = {r["colour"]: r for r in grp}
    assert by["grey"]["COUNT(*)"] == 3 and abs(by["grey"]["AVG(density)"] - 9900.0) < 1e-6
    assert by["yellow"]["COUNT(*)"] == 1 and by["grey"]["_centroid"] is not None
    glob = run_sql("SELECT MIN(density), MAX(density) FROM materials", t)
    assert glob[0]["MIN(density)"] == 7870 and glob[0]["MAX(density)"] == 19300

    # (6) deterministic
    assert run_sql("SELECT name FROM materials WHERE density > 9000 ORDER BY density", t) == \
           run_sql("SELECT name FROM materials WHERE density > 9000 ORDER BY density", t)

    print("holographic_query selftest OK: round-trip recovers fillers (gold/yellow); exact numeric WHERE+ORDER "
          "BY+LIMIT returns the densest [gold, lead]; fuzzy 'colour ~ grey' ranks the grey metals with confidence "
          "%.2f; SQL subset parses; deterministic" % res2[0]["_confidence"])


if __name__ == "__main__":
    _selftest()
