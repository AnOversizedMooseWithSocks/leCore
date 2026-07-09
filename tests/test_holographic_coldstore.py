"""Tests for holographic_coldstore.py -- compress inactive structures, inflate on demand."""
import os
import numpy as np
import pytest
from holographic.caching_and_storage.holographic_coldstore import Cold, ColdStore


def test_cool_warm_array_is_exact_and_shrinks():
    a = np.tile(np.arange(1000, dtype=np.float64), 200)        # very redundant
    c = Cold(a)
    c.cool()
    assert c.is_cold() and c.cold_bytes() < c.warm_bytes()
    assert np.array_equal(c.get(), a) and not c.is_cold()      # exact round trip, warm again
    assert 0 < c.ratio() < 1


def test_codecs():
    a = np.tile(np.arange(500.), 50)
    z = Cold(a, codec="zlib"); z.cool()
    x = Cold(a, codec="lzma"); x.cool()
    n = Cold(a, codec="none"); n.cool()
    assert x.cold_bytes() <= z.cold_bytes() <= n.cold_bytes()  # lzma smallest, none largest
    assert np.array_equal(x.get(), a) and np.array_equal(n.get(), a)


def test_spill_to_disk_frees_ram(tmp_path):
    c = Cold({"rows": list(range(5000))}, spill_dir=str(tmp_path))
    c.cool()
    assert c._blob is None and c._path and os.path.exists(c._path)   # blob on disk, not RAM
    assert c.get() == {"rows": list(range(5000))}
    c.warm()
    assert c._path is None                                      # spill file cleaned up


def test_coldstore_keeps_k_warm():
    s = ColdStore(keep_warm=2)
    for i in range(6):
        s.put("t%d" % i, np.full(500, i, dtype=np.int64))
    st = s.stats()
    assert st["warm"] <= 2 and st["cold"] >= 4
    assert int(s.get("t0")[0]) == 0                             # cold -> transparently warmed
    assert s.stats()["approx_saved_bytes"] > 0


def test_coldstore_get_marks_recent():
    s = ColdStore(keep_warm=1)
    s.put("a", np.zeros(300)); s.put("b", np.ones(300))
    s.get("a")                                                 # touch a -> a warm, b cooled
    assert not s._items["a"].is_cold()


def test_cool_a_real_table():
    from holographic.agents_and_reasoning.holographic_query import Database
    db = Database(); db.add_namespace("s")
    db.create_table("s.widgets", ["id", "name"])
    t = db.namespaces["s"]["tables"]["widgets"]
    for i in range(20):
        t.insert({"id": i, "name": "row%d" % i})
    c = Cold(t, codec="lzma"); c.cool()
    assert c.is_cold()
    t2 = c.get()
    assert len(t2.rows) == 20 and t2.rows[5]["name"] == "row5"


def test_cool_whole_database():
    from holographic.agents_and_reasoning.holographic_query import Database
    db = Database(); db.add_namespace("s")
    db.create_table("s.t", ["id"])
    db.namespaces["s"]["tables"]["t"].insert({"id": 1})
    c = Cold(db); c.cool()
    db2 = c.get()
    assert "s" in db2.namespaces and "t" in db2.namespaces["s"]["tables"]


def test_remove_and_contains():
    s = ColdStore(keep_warm=4)
    s.put("k", [1, 2, 3])
    assert "k" in s and len(s) == 1
    s.remove("k")
    assert "k" not in s


# ---- Database auto-cooling (opt-in) + distributed safety ---------------------------------------------------
def _db_with_tables(n_tables=5, rows=30):
    from holographic.agents_and_reasoning.holographic_query import Database
    db = Database(); db.add_namespace("app")
    for t in range(n_tables):
        db.create_table("app.t%d" % t, ["id", "val"])
        for i in range(rows):
            db.resolve("app.t%d" % t).insert({"id": i, "val": "r%d" % i})
    return db


def test_db_cool_idle_keeps_recent_warm():
    db = _db_with_tables()
    db.enable_cold_storage(keep_warm=2)
    db.resolve("app.t3"); db.resolve("app.t4")     # touch these -> stay warm
    cooled = db.cool_idle()
    s = db.cold_stats()
    assert cooled == 3 and s["warm"] == 2 and s["cold"] == 3


def test_db_resolve_warms_transparently_same_data():
    db = _db_with_tables()
    db.enable_cold_storage(keep_warm=1)
    db.cool_idle()
    t = db.resolve("app.t0")                       # was cold -> warmed here
    assert t.rows[5]["val"] == "r5"
    assert not any(c > 0 for c in [db.cold_stats()["cold"]] if False)  # (sanity, no-op)


def test_db_off_by_default_is_backward_compatible():
    db = _db_with_tables()
    assert db.cold_stats()["enabled"] is False     # nothing changes unless you enable it
    assert db.cool_idle() == 0                      # a no-op when disabled


def test_db_to_state_warms_cold_tables():
    db = _db_with_tables()
    db.enable_cold_storage(keep_warm=1)
    db.cool_idle()                                 # some tables are cold now
    st = db.to_state()
    assert len(st["namespaces"]["app"]["tables"]) == 5   # ALL tables serialized, none dropped


def test_db_pickle_is_distributed_safe():
    """Shipping a cold-enabled DB to a worker must yield a WARM copy with cooling OFF, so reads can't mutate the
    shared read-only cache and no lock/spill crosses the process boundary."""
    import pickle
    db = _db_with_tables()
    db.enable_cold_storage(keep_warm=1)
    db.cool_idle()
    assert db.cold_stats()["cold"] >= 1

    shipped = pickle.loads(pickle.dumps(db))       # what a worker process receives
    assert shipped.cold_stats()["enabled"] is False        # cooling disabled in the copy
    assert shipped.cold_stats()["cold"] == 0               # everything arrived warm (immutable)

    before = dict(shipped.cold_stats())
    shipped.resolve("app.t0"); shipped.resolve("app.t2")   # worker reads
    assert dict(shipped.cold_stats()) == before            # reads did NOT mutate the cache

    assert db.cold_stats()["enabled"] is True              # the original still cools


def test_db_cold_through_distributed_coordinator():
    from holographic.scene_and_pipeline.holographic_coordinator import Coordinator, InProcessBackend
    from holographic.scene_and_pipeline.holographic_distribute import reduce_sum
    db = _db_with_tables(n_tables=1, rows=100)
    for i in range(100):
        db.resolve("app.t0").rows[i]["amt"] = i
    db.enable_cold_storage(keep_warm=1); db.cool_idle()

    def worker(bucket, cache):
        tbl = cache.resolve("app.t0")
        return sum(tbl.rows[i]["amt"] for i in bucket)

    with Coordinator(InProcessBackend()) as c:
        total = c.run([list(range(50)), list(range(50, 100))], worker, cache=db, reduce=reduce_sum)
    assert total == sum(range(100))
    assert db.cold_stats()["enabled"] is True      # cache survived the job intact
