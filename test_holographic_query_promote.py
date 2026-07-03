"""Query layer PROMOTE (P1-P3): shipped faculties exposed as query verbs -- similarity, clustering, calibrated anomaly."""
import numpy as np
from holographic_query import UserTable, similar_to, cluster, anomalies


def _grouped():
    t = UserTable("critters", ["legs", "sound"], dim=1024, seed=0)
    for r in ([{"legs": 4, "sound": "meow"}] * 3 + [{"legs": 2, "sound": "tweet"}] * 3
              + [{"legs": 0, "sound": "hiss"}]):                # the snake is the lone outlier
        t.insert(r)
    return t


def test_p1_similar_to_ranks_like_rows_first():
    t = _grouped()
    res = similar_to(t, {"legs": 4, "sound": "meow"}, k=3)
    assert all(r["sound"] == "meow" for r in res) and res[0]["_confidence"] > 0.99


def test_p1_confidence_is_present_and_ordered():
    t = _grouped()
    res = similar_to(t, {"legs": 2, "sound": "tweet"}, k=7)
    confs = [r["_confidence"] for r in res]
    assert confs == sorted(confs, reverse=True)               # ranked by similarity, descending


def test_p2_cluster_separates_groups_by_coherence():
    t = _grouped()
    cl = cluster(t, into=2, seed=0)
    tight = cl[0]                                             # tightest cluster first
    assert tight["coherence"] > 0.99 and tight["size"] == 3   # the three identical meows
    assert all(c["size"] > 0 for c in cl)                    # empty clusters dropped


def test_p2_cluster_empty_table():
    assert cluster(UserTable("e", ["a"], dim=256, seed=0), into=3) == []


def test_p3_anomaly_flags_the_outlier():
    t = _grouped()
    ranked = anomalies(t, seed=0)
    assert ranked[0]["sound"] == "hiss"                      # the lone snake is most anomalous
    assert ranked[0]["_anomaly_score"] > 0.5                 # its nearest neighbour is noise-level
    assert ranked[-1]["_anomaly_score"] < 0.05              # duplicated rows are clearly not anomalous


def test_p3_calibrated_can_abstain_when_all_have_neighbours():
    # every row has a duplicate -> no row's nearest neighbour is noise -> nothing is actually anomalous
    t = UserTable("pairs", ["v"], dim=1024, seed=0)
    for r in [{"v": "a"}, {"v": "a"}, {"v": "b"}, {"v": "b"}]:
        t.insert(r)
    assert max(r["_anomaly_score"] for r in anomalies(t, seed=0)) < 0.2


def test_p3_deterministic():
    t = _grouped()
    a = [r["_anomaly_score"] for r in anomalies(t, seed=0)]
    b = [r["_anomaly_score"] for r in anomalies(t, seed=0)]
    assert a == b


def test_promotes_work_through_the_database_front_door():
    """The promoted verbs operate on a Table resolved through the Database -- the query front door, not a silo."""
    from holographic_query import Database
    db = Database()
    db.create_database("app")
    db.create_table("app.critters", ["legs", "sound"])
    for r in ([{"legs": 4, "sound": "meow"}] * 3 + [{"legs": 0, "sound": "hiss"}]):
        db.insert("app.critters", r)
    t = db.resolve("app.critters")
    assert similar_to(t, {"legs": 4, "sound": "meow"}, k=1)[0]["sound"] == "meow"
    assert anomalies(t, seed=0)[0]["sound"] == "hiss"
    assert cluster(t, into=2, seed=0)[0]["coherence"] > 0.99
