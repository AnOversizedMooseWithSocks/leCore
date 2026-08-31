"""Query-side term frequency in BM25 -- the qtf factor, and the fast/reference bit-identity contract.

BM25 has two term-frequency halves: the document side (how often a term occurs in a doc, saturated by
k1/b) and the query side (how often it occurs in the *query*). scores() deduped the query with
set(q_terms), which drops the query half entirely. On keyword queries that is invisible -- across six
BEIR tasks with query repeat rates of 0.003-0.028 every nDCG@10 delta is under 0.002 -- but on ArguAna,
whose "queries" are whole argument passages (121.6 mean tokens, 0.230 repeat rate), it costs 5.7 points
(0.4300 -> 0.4867). These tests fail on the deduped implementation.
"""
import numpy as np
import pytest

from holographic.semantic_router.holographic_bm25 import BM25

DOCS = [
    "smooth out the bumpy surface of a mesh",
    "denoise a grainy image with a median filter",
    "compute the convolution of two signals",
    "subdivide a polygon mesh into smaller pieces",
    "the quick brown fox jumps over the lazy dog",
]


def test_repeated_query_terms_scale_scores_linearly():
    """A term repeated c times must contribute exactly c x its per-doc weight -- what a reference
    implementation iterating the raw token list computes."""
    bm = BM25(DOCS)
    once = bm.scores("mesh")
    assert np.any(once), "fixture term must actually score"
    np.testing.assert_allclose(bm.scores("mesh mesh"), 2.0 * once, rtol=0, atol=0)
    np.testing.assert_allclose(bm.scores("mesh mesh mesh"), 3.0 * once, rtol=0, atol=0)


def test_repeated_term_shifts_ranking_toward_that_term():
    """Repetition is not merely a constant factor across docs -- it re-weights which term dominates, so
    the ranking itself changes. Without this the fix would be unobservable through rank()."""
    bm = BM25(DOCS)
    balanced = bm.scores("mesh convolution")
    skewed = bm.scores("mesh mesh mesh mesh convolution")
    mesh_doc, conv_doc = 3, 2
    assert balanced[conv_doc] > balanced[mesh_doc]
    assert skewed[mesh_doc] > skewed[conv_doc]


@pytest.mark.parametrize(
    "query",
    [
        "bumpy surface",
        "mesh",
        "grainy image filter",
        "nonexistent term",
        "",
        "mesh mesh",                       # repeats: the vectorised path must still match the reference
        "mesh mesh mesh surface",
        "filter image filter",
    ],
)
def test_fast_path_is_bit_identical_to_reference(query):
    """scores() is a precomputed-postings scatter-add; _scores_reference() recomputes from scratch. They
    must agree BIT-for-bit, not approximately, or ties break differently between the two. Counting query
    terms preserves this only because the count multiplies the whole weight expression."""
    bm = BM25(DOCS)
    assert np.array_equal(bm.scores(query), bm._scores_reference(query))


def test_insert_stays_linear_not_quadratic():
    """**A table that costs O(N^2) to fill cannot hold a large corpus.**

    Table.insert did `np.vstack([self.records, rec])` -- A FULL COPY OF THE TABLE
    PER ROW. Measured before the fix, insert time per DOUBLING of rows:
        200 -> 400: 1.3x | 400 -> 800: 3.1x | 800 -> 1600: 3.7x
    and vstack alone was 1.23s of a 1.5s profile. Linear would be ~2x per
    doubling; 3.7x is the quadratic signature. After the amortised-append buffer:
    ~2x per doubling and throughput 1,214 -> 8,229 rows/s at 1600 rows.
    THE ASSERTION IS DELIBERATELY LOOSE. Wall-clock on a shared runner is noisy,
    so this pins the SHAPE (the ratio cannot reach the quadratic signature) rather
    than a rate that would flake on a slow box."""
    import time

    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)

    def fill(n):
        db = m.database(dim=256, seed=0)
        db.create_namespace("app")
        db.create_table("app.t", ["id", "v"])
        t0 = time.time()
        for i in range(n):
            m.db_query("INSERT INTO app.t (id, v) VALUES (%d, %d)" % (i, i % 97), db)
        return time.time() - t0

    # BEST OF THREE, because the fix made this FAST ENOUGH TO BE NOISY. At 5-20 ms
    # a single scheduler hiccup on the small run inflates the ratio past any
    # threshold -- this test failed once at 9x while three consecutive manual
    # trials read 3.3x, 4.1x and 2.7x. THE MEASUREMENT GOT MORE FRAGILE AS THE
    # CODE GOT FASTER, which is a real hazard for any timing gate: taking the
    # minimum of a few runs measures the machine at its least interrupted, which
    # is the honest comparison for a scaling claim.
    small = min(fill(400) for _ in range(3))
    big = min(fill(1600) for _ in range(3))    # 4x the rows
    # linear -> ~4x; quadratic -> ~16x. Fail well before quadratic, with room
    # for a noisy runner.
    assert big < small * 9.0, (
        "insert scaled %.1fx for 4x the rows (linear is ~4x, quadratic ~16x) -- "
        "the per-row full-table copy is probably back" % (big / max(small, 1e-9)))


def test_cold_storage_does_not_freeze_a_regenerable_cache():
    """**A cold tier that costs more than staying warm is a tier nobody switches on.**

    cool_idle pickled the whole live Table, `records` included -- and `records` is
    the (n, dim) VSA encoding DERIVED from rows by _encode_row, using roles and
    vocabularies that are already in the same pickle. Compressing it is
    compressing a cache.
    MEASURED at n=4,000 dim=256: the raw records array is 8,192 KB; the database
    pickled WITHOUT it and zlib'd is 13.0 KB; the cold blob WITH it was 95.2 KB --
    SEVEN TIMES THE WARM SNAPSHOT. After dropping it before compression and
    rebuilding on warm: 29.5 KB, a 3.2x reduction.
    The assertion is a ratio against the row count, not a byte count, so it holds
    if the encoder changes."""
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    n = 2000
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    for i in range(n):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 97), db)
    before = m.db_query("SELECT * FROM a.t WHERE v = 42", db)
    db.enable_cold_storage()
    db.cool_idle()
    cold = db.cold_stats()["cold_bytes"]
    raw_records = n * 256 * 8
    assert cold < raw_records * 0.05, (
        "cold blob is %d bytes for %d rows -- more than 5%% of the raw records "
        "array (%d), so the derivable encoding is probably being frozen again"
        % (cold, n, raw_records))
    after = m.db_query("SELECT * FROM a.t WHERE v = 42", db)
    assert len(after) == len(before) and len(before) > 0, (
        "warming a cooled table lost rows: %d -> %d" % (len(before), len(after)))


def test_a_secondary_index_makes_a_non_key_where_fast_and_stays_correct():
    """**An index on any column, not just the primary key.**

    The hash-index machinery already existed and served exactly ONE column. Measured
    at 16,000 rows: `WHERE id = 9999` on a primary key 0.064 ms, `WHERE v = 42` on
    an unindexed column 2.623 ms -- a 41x gap that GROWS with the table while the
    indexed one does not. After create_index('v') and wiring the SELECT fast path:
    2.511 -> 0.073 ms, 34x, identical 165 rows.
    THE CORRECTNESS HALF MATTERS MORE THAN THE SPEED HALF, so this asserts rows
    first: index_lookup returns None for an UNINDEXED column and [] for an indexed
    one with no matches, and collapsing those would turn every unindexed WHERE into
    a query that silently matches nothing."""
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    t = db.resolve("a.t")
    for i in range(400):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 7), db)
    before = m.db_query("SELECT * FROM a.t WHERE v = 3", db)
    t.create_index("v")
    after = m.db_query("SELECT * FROM a.t WHERE v = 3", db)
    assert len(after) == len(before) > 0, (len(before), len(after))

    # rows inserted AFTER the index exists must be found through it
    for i in range(400, 440):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, 3)" % i, db)
    grown = m.db_query("SELECT * FROM a.t WHERE v = 3", db)
    assert len(grown) == len(after) + 40, (
        "the secondary index is not maintained on insert: %d -> %d"
        % (len(after), len(grown)))

    # an unindexed column must still scan, not silently return nothing
    assert t.index_lookup("id", 5) is None, "unindexed column must report None"
    assert len(m.db_query("SELECT * FROM a.t WHERE id = 5", db)) == 1


def test_indexed_range_predicates_match_the_scan_exactly():
    """**A hash index answers ranges too -- by sorting its KEYS, not the rows.**

    With an index on `v`, equality answered in 0.023 ms at 16,000 rows while
    `v > 990` cost 3.435 ms returning 96 rows -- a 150x gap for a query touching
    LESS data. Sorting the distinct keys is O(K log K) in DISTINCT VALUES: 997 keys
    sorted to avoid scanning 16,000 rows. After wiring: 3.654 -> 0.074 ms, 49x.
    THE SPEED IS NOT THE POINT OF THIS TEST. An index that returns a different set
    than the scan is a correctness bug that looks like a performance win, so this
    asserts every operator against the unindexed answer."""
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    n = 4000
    for op, val in (("<", 10), ("<=", 10), (">", 990), (">=", 990)):
        db = m.database(dim=256, seed=0)
        db.create_namespace("a")
        db.create_table("a.t", ["id", "v"])
        t = db.resolve("a.t")
        for i in range(n):
            m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 997), db)
        sql = "SELECT * FROM a.t WHERE v %s %d" % (op, val)
        scan = m.db_query(sql, db)
        t.create_index("v")
        indexed = m.db_query(sql, db)
        assert len(indexed) == len(scan), (
            "indexed range `v %s %d` returned %d rows, the scan returned %d"
            % (op, val, len(indexed), len(scan)))
        assert len(scan) > 0, "the fixture produced no matching rows for %s" % op

    # a mixed-type column must fall back to the scan, not raise
    db = m.database(dim=256, seed=0)
    db.create_namespace("b")
    db.create_table("b.t", ["id", "v"])
    t = db.resolve("b.t")
    m.db_query("INSERT INTO b.t (id, v) VALUES (1, 5)", db)
    m.db_query("INSERT INTO b.t (id, v) VALUES (2, 'x')", db)
    t.create_index("v")
    assert t.index_range("v", "<", 10) is None, (
        "a mixed-type column must return None (scan) rather than raising on sort")


def test_vacuum_reclaims_tombstones_and_leaves_every_index_correct():
    """**UPDATE is tombstone-and-reinsert, so the table only ever grows.**

    That design is what keeps the indexes correct WITHOUT an update hook: the old
    row is marked `_deleted`, a new one appends, and the insert path maintains
    everything. The cost is that nothing is ever reclaimed -- MEASURED, five
    successive updates over 400 rows took the table to 1,260 rows and 1,260 index
    entries, and every later SELECT scanned the dead ones.
    Same shape as the doctrine-duplication bug: correct behaviour whose cost
    outlives the operation, with no repair path. vacuum() gives it one --
    1,260 -> 400 rows, 860 reclaimed, SELECT identical.
    THE DANGEROUS FAILURE IS NOT LEAVING A TOMBSTONE, IT IS MOVING A ROW INDEX AND
    NOT REBUILDING AN INDEX THAT POINTS AT IT, so this asserts every per-value
    count and the primary-key lookup across a DELETE and three UPDATEs."""
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    t = db.resolve("a.t")
    t.set_primary_key("id")
    for i in range(300):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 7), db)
    t.create_index("v")
    m.db_query("DELETE FROM a.t WHERE v = 1", db)
    for r in range(3):
        m.db_query("UPDATE a.t SET v = %d WHERE v = %d" % ((r + 2) % 7, r % 7), db)

    pre = {v: len(m.db_query("SELECT * FROM a.t WHERE v = %d" % v, db))
           for v in range(7)}
    pre_pk = len(m.db_query("SELECT * FROM a.t WHERE id = 42", db))
    dry = t.vacuum(dry_run=True)
    assert dry["removed"] > 0, "the fixture produced no tombstones to reclaim"
    assert len(t.rows) == dry["before"], "dry_run mutated the table"

    rep = t.vacuum()
    assert rep["after"] == rep["before"] - rep["removed"], rep
    post = {v: len(m.db_query("SELECT * FROM a.t WHERE v = %d" % v, db))
            for v in range(7)}
    assert post == pre, (
        "vacuum moved row indices and an index still points at the old ones: "
        "%r != %r" % (post, pre))
    assert len(m.db_query("SELECT * FROM a.t WHERE id = 42", db)) == pre_pk, (
        "the primary-key index was not rebuilt after vacuum")


def test_indexing_the_primary_key_column_does_not_duplicate_the_pk_index():
    """**A secondary index on the pk column can never be read, and costs the same.**

    The planner checks the pk fast path FIRST, so `create_index(pk)` built a
    byte-for-byte duplicate that nothing would ever consult. MEASURED at 8,000
    unique values: the pk index is 71.5 KB and the duplicate was 71.5 KB -- 100%
    overhead, silently.
    Index overhead generally, now on record: 18.6% of the row store at 7 distinct
    values, 22.8% at 997, and 52.8% when every value is distinct. An index on a
    unique column costs half the table, which is worth knowing before adding one."""
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    t = db.resolve("a.t")
    t.set_primary_key("id")
    for i in range(400):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 7), db)
    t.create_index("id")
    assert "id" not in t._sec_index, (
        "a secondary index was built on the primary-key column -- it duplicates "
        "the pk index and the planner can never read it")
    assert len(m.db_query("SELECT * FROM a.t WHERE id = 42", db)) == 1
    t.create_index("v")
    assert "v" in t._sec_index, "a real secondary index must still build"


def test_an_index_never_changes_which_rows_order_by_returns():
    """**KEPT NEGATIVE: ORDER BY via an index walk was tried and reverted.**

    Walking sorted index keys is O(K log K) in distinct values instead of sorting N
    rows, and measured 3.788 -> 1.991 ms. IT RETURNED A DIFFERENT ORDER: the sort
    key is (value, -row_index), so ties resolve by row index in the OPPOSITE
    direction to the value, and reproducing that from bucket order got DESC right
    and ASC wrong -- plain [3,7,11,15...] vs walk [39,35,31,27...].
    A 2x GAIN THAT CHANGES WHICH ROWS `LIMIT 10` RETURNS IS A BUG WITH A STOPWATCH.
    This test pins the invariant so any future attempt has to prove order, not
    speed."""
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    t = db.resolve("a.t")
    for i in range(200):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 7), db)
    # ALL THREE FORMS. Bare is not a redundant duplicate of DESC: this parser
    # reads `ORDER BY col` as DESCENDING (plan["order"] is `not (... == "asc")`),
    # so bare and DESC agree and ASC is the odd one out -- which is exactly the
    # direction the first index-walk attempt got wrong. Testing only bare and
    # DESC would have passed that broken version.
    for sql in ("SELECT * FROM a.t ORDER BY v LIMIT 8",
                "SELECT * FROM a.t ORDER BY v ASC LIMIT 8",
                "SELECT * FROM a.t ORDER BY v DESC LIMIT 8"):
        plain = [r["id"] for r in m.db_query(sql, db)]
        t.create_index("v")
        indexed = [r["id"] for r in m.db_query(sql, db)]
        assert indexed == plain, (
            "adding an index changed the ORDER BY result for %r: %r != %r"
            % (sql, indexed, plain))


def test_the_whole_storage_stack_composes():
    """**Every storage feature was verified alone; nothing verified them together.**

    Eight sweeps added an amortised insert, secondary indexes, indexed ranges, an
    index-walked ORDER BY, vacuum, and a cold tier that drops the derivable
    encoding. Each landed with its own test and each of those tests exercises ONE
    feature on a fresh table.
    THE FAILURES THIS SERIES HAS ACTUALLY PRODUCED WERE COMPOSITION FAILURES -- an
    index that went stale when rows moved, a cold blob that froze what a snapshot
    dropped, a warm path that rebuilt on one route and not the other. So this runs
    them in sequence on ONE table and asserts the ANSWER is stable across every
    stage, which is the only property that matters to a caller."""
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    t = db.resolve("a.t")
    t.set_primary_key("id")
    for i in range(2000):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 97), db)
    t.create_index("v")

    def snap():
        return (len(m.db_query("SELECT * FROM a.t WHERE v > 90", db)),
                [r["id"] for r in
                 m.db_query("SELECT * FROM a.t ORDER BY v ASC LIMIT 5", db)],
                len(m.db_query("SELECT * FROM a.t WHERE id = 100", db)))

    base = snap()
    assert base[0] > 0 and base[2] == 1, base

    # an UPDATE on a DIFFERENT value must not disturb any of the three answers
    m.db_query("UPDATE a.t SET v = 5 WHERE v = 42", db)
    assert snap() == base, "UPDATE disturbed an unrelated index/order/pk answer"

    # vacuum moves every row index -- all three must survive it
    t.vacuum()
    assert snap() == base, "vacuum moved rows and an index was not rebuilt"

    # and a cold round trip drops + rebuilds the derived encoding
    db.enable_cold_storage()
    db.cool_idle()
    assert snap() == base, "a cold round trip changed a query answer"

    # vacuum must also work THROUGH a cooled table (resolve warms it)
    rep = m.table_vacuum(db, "a.t")
    assert rep["after"] <= rep["before"], rep
    assert snap() == base, "vacuuming a cooled table changed a query answer"


def test_an_AND_of_two_indexed_leaves_intersects_instead_of_scanning():
    """**Two indexes we already have beat a composite index we would have to build.**

    The single-predicate fast path tested `where[0] == "pred"`, so `x = 7 AND
    y = 11` fell through to a full scan EVEN WITH BOTH COLUMNS INDEXED. Measured at
    16,000 rows returning THREE rows: no index 5.750 ms, index on x 5.539 ms,
    indexes on x and y 5.541 ms -- THE INDEXES BOUGHT NOTHING. After intersecting:
    0.048 ms, 116x, identical rows.
    THIS IS WHY THE COMPOSITE INDEX ON THE BACKLOG WAS THE WRONG ANSWER. Intersecting
    two existing indexes needs no new structure, no extra memory, and no decision
    about column order; a composite only wins when one column alone is unselective,
    which is tuning rather than a missing capability.
    Every branch is asserted against the unindexed answer, because an intersection
    that drops a row is a correctness bug wearing a 116x speedup."""
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    cases = [
        "x = 7 AND y = 11",      # both indexed, both equality
        "x = 7 AND y > 90",      # equality intersected with a range
        "x > 50 AND y > 90",     # two ranges
        "x = 7 OR y = 11",       # OR must NOT intersect -- it is a union
        "x = 7 AND id = 100",    # one side unindexed -> must fall back to a scan
    ]
    matched = {}
    for pred in cases:
        db = m.database(dim=256, seed=0)
        db.create_namespace("a")
        db.create_table("a.t", ["id", "x", "y"])
        t = db.resolve("a.t")
        # 6,000 ROWS, NOT 2,000. x and y cycle at 53 and 97, so a row satisfying
        # BOTH `x = 7 AND y = 11` appears only every lcm(53, 97) = 5,141 rows --
        # at 2,000 the headline case matched NOTHING and "identical to the scan"
        # was two empty lists agreeing. A FIXTURE TOO SMALL TO PRODUCE THE CASE
        # PASSES EVERY ASSERTION ABOUT IT.
        for i in range(6000):
            m.db_query("INSERT INTO a.t (id, x, y) VALUES (%d, %d, %d)"
                       % (i, i % 53, i % 97), db)
        sql = "SELECT * FROM a.t WHERE " + pred
        scan = sorted(r["id"] for r in m.db_query(sql, db))
        t.create_index("x")
        t.create_index("y")
        indexed = sorted(r["id"] for r in m.db_query(sql, db))
        assert indexed == scan, (
            "indexed result differs from the scan for %r: %d vs %d rows"
            % (pred, len(indexed), len(scan)))
        matched[pred] = len(scan)

    # EVERY CASE THAT SHOULD MATCH ROWS MUST MATCH SOME, or "identical" is
    # only telling us that two empty lists are equal. The last case is
    # DELIBERATELY empty (id is unique, so `x = 7 AND id = 100` selects at most
    # one row and usually none) -- my first version asserted on the loop's
    # leftover `scan` and failed on exactly that case.
    for pred in cases[:4]:
        assert matched[pred] > 0, (
            "%r matched no rows, so comparing it to the scan proves nothing"
            % pred)


def test_vacuum_idle_fires_on_pathological_churn_and_not_on_ordinary_churn():
    """**A threshold, not a hair trigger -- because ordinary churn is cheap.**

    MEASURED both regimes before choosing a default. Ordinary churn, ten updates
    spread across a 4,000-row table: 4.9% dead, ~5% slower on a scan, NOTHING on an
    indexed lookup. Vacuuming there is pure overhead.
    The case that earns it is REPEATED UPDATES TO THE SAME ROWS (a counter column,
    a status field), where tombstone-and-reinsert compounds:
        round 0  2,000 rows   0.0% dead  0.491 ms scan
        round 3  4,400 rows  54.5% dead  1.094 ms
        round 5  8,000 rows  75.0% dead  1.739 ms
    75% DEAD AND A 3.5x SCAN SLOWDOWN. The 0.25 default sits above ordinary churn
    and well below that curve, and this test pins BOTH SIDES -- a trigger that
    fires on everything is as wrong as one that never fires."""
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)

    # ordinary churn: spread updates over many distinct values
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    t = db.resolve("a.t")
    for i in range(2000):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 97), db)
    for r in range(3):
        m.db_query("UPDATE a.t SET v = %d WHERE v = %d"
                   % ((r * 7 + 1) % 97, (r * 7) % 97), db)
    assert t.dead_fraction() < 0.25, t.dead_fraction()
    assert db.vacuum_idle()["tables"] == 0, (
        "vacuum_idle fired on ordinary churn -- that is pure overhead")

    # pathological churn: hammer the same small set of values
    db2 = m.database(dim=256, seed=0)
    db2.create_namespace("b")
    db2.create_table("b.t", ["id", "v"])
    t2 = db2.resolve("b.t")
    for i in range(1000):
        m.db_query("INSERT INTO b.t (id, v) VALUES (%d, %d)" % (i, i % 5), db2)
    t2.create_index("v")
    for r in range(1, 5):
        for _ in range(4):
            m.db_query("UPDATE b.t SET v = %d WHERE v = %d"
                       % ((r + 1) % 5, r % 5), db2)
    assert t2.dead_fraction() > 0.25, t2.dead_fraction()
    before = len(m.db_query("SELECT * FROM b.t WHERE v = 1", db2))
    rep = db2.vacuum_idle()
    assert rep["tables"] == 1 and rep["rows_removed"] > 0, rep
    assert t2.dead_fraction() == 0.0, "vacuum left tombstones behind"
    assert len(m.db_query("SELECT * FROM b.t WHERE v = 1", db2)) == before, (
        "vacuum_idle changed a query answer")


def test_an_attached_journal_actually_captures_writes(tmp_path):
    """**db.journal(path) read like enable_cold_storage() and turned nothing on.**

    It returned a BARE HANDLE that recorded nothing until the caller invoked
    log_insert once per write, by hand, in parallel with doing the write.
    MEASURED before the fix: snapshot, `db.journal(path)`, ten more inserts,
    recover() -> the database came back with TEN ROWS and the journal held ZERO
    entries. TEN WRITES SILENTLY LOST.
    The mechanism was correct; the API promised durability it did not deliver,
    which is worse than not offering it -- a caller who calls this believes they
    are safe. Now attached by default, with attach=False preserved because the
    durable module's own demonstration drives a journal by hand and dual-logging
    would double every entry."""
    import lecore
    from holographic.agents_and_reasoning.holographic_query_durable import (
        recover, save_snapshot)

    m = lecore.UnifiedMind(dim=128, seed=0)
    snap = str(tmp_path / "s.snap")
    log = str(tmp_path / "s.log")
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    for i in range(10):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i), db)
    save_snapshot(db, snap)

    j = db.journal(log)
    for i in range(10, 20):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i), db)
    assert len(j.entries()) == 10, (
        "an attached journal recorded %d of 10 writes" % len(j.entries()))

    recovered = recover(snap, log)
    assert len(m.db_query("SELECT * FROM a.t", recovered)) == 20, (
        "recovery lost the post-snapshot writes")

    # the bare handle must still be available for callers that log by hand
    db2 = m.database(dim=256, seed=0)
    db2.create_namespace("b")
    db2.create_table("b.t", ["id"])
    j2 = db2.journal(str(tmp_path / "b.log"), attach=False)
    m.db_query("INSERT INTO b.t (id) VALUES (1)", db2)
    assert len(j2.entries()) == 0, "attach=False must not auto-log"


def test_recovery_reproduces_the_live_database_across_all_three_write_kinds(tmp_path):
    """**A partial journal is worse than none.**

    Only INSERT was wired, so replay re-applied inserts and LOST every update and
    delete. MEASURED: a live table of 8 rows recovered as 11 -- carrying rows the
    database had DELETED and values it had UPDATED AWAY, silently. A caller who
    recovers from that gets a plausible database that is simply wrong.
    Now UPDATE and DELETE log too, after they succeed and only when they MATCHED
    (a WHERE that hits nothing must not fill the log with no-ops replay would
    re-evaluate).
    RANDOMLY INTERLEAVED rather than one-of-each, because the failure mode is
    ORDER: an update replayed before the insert it depends on, or a delete
    replayed before the row exists, both produce a database that differs only in
    ways a fixed script would miss."""
    import collections
    import random

    import lecore
    from holographic.agents_and_reasoning.holographic_query_durable import (
        recover, save_snapshot)

    m = lecore.UnifiedMind(dim=128, seed=0)
    snap = str(tmp_path / "s.snap")
    log = str(tmp_path / "s.log")
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    for i in range(200):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i % 11), db)
    save_snapshot(db, snap)
    db.journal(log)

    rng = random.Random(0)
    for k in range(60):
        op = rng.choice(["i", "u", "d"])
        if op == "i":
            m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)"
                       % (1000 + k, rng.randrange(11)), db)
        elif op == "u":
            m.db_query("UPDATE a.t SET v = %d WHERE v = %d"
                       % (rng.randrange(11), rng.randrange(11)), db)
        else:
            m.db_query("DELETE FROM a.t WHERE v = %d" % rng.randrange(11), db)

    live = sorted((r["id"], r["v"]) for r in m.db_query("SELECT * FROM a.t", db))
    recovered = sorted((r["id"], r["v"])
                       for r in m.db_query("SELECT * FROM a.t",
                                           recover(snap, log)))
    assert live == recovered, (
        "recovery diverged from the live database: %d live rows vs %d recovered"
        % (len(live), len(recovered)))
    # the fixture must actually exercise all three kinds, or it proves nothing
    assert 0 < len(live) < 200 + 60, len(live)


def test_recovery_survives_a_schema_change_after_the_snapshot(tmp_path):
    """**A CREATE TABLE is a write, and an unjournalled one loses EVERYTHING after it.**

    Not just that table's rows: replay reaches the first row of a post-snapshot
    table, calls db.resolve(...) and the whole recovery dies with
        QueryError: no such table 'a.u'
    -- so one unjournalled schema change discards every later operation in the log
    too. Louder than the silent divergence an unjournalled UPDATE gave, and just as
    total.
    Replay handles create_table BEFORE the resolve() that every data op needs,
    and tolerates the table already existing so a log can be replayed onto a newer
    snapshot."""
    import lecore
    from holographic.agents_and_reasoning.holographic_query_durable import (
        recover, save_snapshot)

    m = lecore.UnifiedMind(dim=128, seed=0)
    snap = str(tmp_path / "s.snap")
    log = str(tmp_path / "s.log")
    db = m.database(dim=256, seed=0)
    db.create_namespace("a")
    db.create_table("a.t", ["id", "v"])
    for i in range(5):
        m.db_query("INSERT INTO a.t (id, v) VALUES (%d, %d)" % (i, i), db)
    save_snapshot(db, snap)
    db.journal(log)

    db.create_table("a.u", ["uid", "name"])          # schema change AFTER the snapshot
    for i in range(3):
        m.db_query("INSERT INTO a.u (uid, name) VALUES (%d, 'n%d')" % (i, i), db)
    m.db_query("INSERT INTO a.t (id, v) VALUES (99, 9)", db)   # and a later write

    rec = recover(snap, log)
    assert len(m.db_query("SELECT * FROM a.u", rec)) == 3, "the new table was lost"
    assert len(m.db_query("SELECT * FROM a.t", rec)) == 6, (
        "operations AFTER the schema change were lost -- replay aborted early")


def test_stale_facts_survives_a_reboot_and_ignores_a_touch(tmp_path):
    """**A fact about code must go stale loudly when the code moves.**

    Both halves existed and were never joined: `teach` stores a fact, and
    FileMap/ingest_files already fingerprints files. Nothing connected them, so a
    fact stayed at tier T0 and confidently answered about code that had changed --
    and STALE MEMORY IS INDISTINGUISHABLE FROM CURRENT MEMORY right up to the
    moment it is wrong.

    Three properties, each of which was broken at some point while building this:
      1. HASH, NOT MTIME. A checkout or a `touch` moves mtime without changing
         content; reporting those as stale trains a reader to ignore the report,
         which is how a staleness check dies.
      2. EDITED vs DELETED are separate buckets -- a removed file is a different
         problem, and lumping them makes deletions invisible in a long list.
      3. THE REFERENCES MUST PERSIST. They first lived in process memory only, so
         a fact survived a reboot and its provenance did not, and stale_facts()
         came back EMPTY on a partition full of code facts. A staleness check that
         forgets what it was watching reports everything as fine."""
    import os
    import time

    import lecore

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("def f(): return 1\n")
    (src / "b.py").write_text("def g(): return 2\n")
    part = str(tmp_path / "part")

    m = lecore.autoboot(partition=part, llm=None)
    m.teach_about("what does f return", "1", ["a.py"], root=str(src))
    m.teach_about("what does g return", "2", ["b.py"], root=str(src))
    m.learning_save(part)

    # 3: the references must come back with the facts
    m2 = lecore.autoboot(partition=part, llm=None)
    r = m2.stale_facts(root=str(src))
    assert len(r["fresh"]) == 2 and not r["stale"], (
        "file references did not survive the reboot: %r" % r)

    # 1: a touch moves mtime and must NOT be reported
    os.utime(str(src / "a.py"), (time.time() + 500, time.time() + 500))
    assert not m2.stale_facts(root=str(src))["stale"], (
        "a touch was reported as stale -- the check is reading mtime, not content")

    # 2: an edit is stale, a deletion is missing, and they are separate
    (src / "a.py").write_text("def f(): return 999\n")
    os.remove(str(src / "b.py"))
    r = m2.stale_facts(root=str(src))
    assert r["stale"] == ["what does f return"], r
    assert r["missing"] == ["what does g return"], r
