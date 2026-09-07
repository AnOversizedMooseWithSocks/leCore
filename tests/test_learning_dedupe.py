"""Learning-memory storage contracts (sweep 166/167): what is stored is deduplicated and
instanced; what is REPLAYED is bit-identical to before.

Every test here pins a number that was measured wrong first:
  * 263,023 taught rows, 385 distinct, one row 42,496 times, learning_load 225 s.
  * keep-first dedupe broke "blue -> grey -> blue again" (reload disagreed with live).
  * 795 audit rows, 307 distinct keys, 307 distinct values: 62% of the floor's bytes were copies.
  * a container with no audit arrays raised KeyError 'aud_k_0' instead of loading.
"""
import io
import json
import os
import zipfile

import numpy as np
import pytest

import lecore


def _mind():
    return lecore.UnifiedMind(dim=64, seed=0, plugins=())


def _taught_rows(path):
    m = json.loads(zipfile.ZipFile(path).read("manifest.json"))
    return [s for s in m["sections"] if s["kind"] == "lecore.learning.taught"][0]["meta"]["texts"]


# ---- taught record: one copy per distinct row, last occurrence wins ----------------------

def test_identical_teaches_leave_one_row():
    m = _mind()
    for _ in range(300):
        m.teach("what colour is the sky", "blue")
    assert len(m.zoo["ladder"].taught_log) == 1


def test_reteaching_an_earlier_answer_survives_reload(tmp_path):
    """KEEP-FIRST WAS WRONG: replay is last-wins per question, so the surviving copy must sit
    where the latest teach put it. Both orders."""
    for seq, want in ((["blue", "grey", "blue"], "blue"), (["grey", "blue", "grey"], "grey")):
        m = _mind()
        for a in seq:
            m.teach("q", a)
        assert m.ask("q")["answer"] == want
        m.learning_save(str(tmp_path))
        n = _mind()
        n.learning_load(str(tmp_path))
        assert n.ask("q")["answer"] == want, seq
        assert len(n.zoo["ladder"].taught_log) == 2


def test_taught_log_survives_reassignment():
    """Curation and distillation REASSIGN taught_log; the seen-set must rebuild, not go stale."""
    m = _mind()
    m.teach("a", "1"); m.teach("b", "2")
    lad = m.zoo["ladder"]
    lad.taught_log = [["a", "1", "shared", "taught"]]          # a curator dropped 'b'
    assert lad._log_taught(["b", "2", "shared", "taught"]) is True, "stale seen-set refused a row the log no longer holds"
    assert lad._log_taught(["a", "1", "shared", "taught"]) is False
    assert len(lad.taught_log) == 2


def test_bloated_container_repairs_on_load(tmp_path):
    """THE 263,023-ROW CASE, in miniature: a partition written before the write-side dedupe
    carries the same row many times. Load dedupes (last wins), the next save writes the
    compact record, and the answers do not change."""
    m = _mind()
    m.teach("q1", "a1"); m.teach("q2", "a2")
    m.learning_save(str(tmp_path))
    p = str(tmp_path / "learning" / "state.lecore")
    # bloat it by hand: repeat every row 500 times
    src = zipfile.ZipFile(p)
    man = json.loads(src.read("manifest.json"))
    sec = [s for s in man["sections"] if s["kind"] == "lecore.learning.taught"][0]
    sec["meta"]["texts"] = sec["meta"]["texts"] * 500
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
        for i in src.infolist():
            dst.writestr(i.filename, json.dumps(man) if i.filename == "manifest.json" else src.read(i.filename))
    open(p, "wb").write(buf.getvalue())
    assert len(_taught_rows(p)) == 1000
    n = _mind()
    n.learning_load(str(tmp_path))
    assert getattr(n, "_learning_load_deduped", 0) == 998
    assert len(n.zoo["ladder"].taught_log) == 2
    assert n.ask("q1")["answer"] == "a1" and n.ask("q2")["answer"] == "a2"
    n.learning_save(str(tmp_path))
    assert len(_taught_rows(p)) == 2, "the repair did not reach the file"


# ---- audit floor: instanced storage, identical replay ---------------------------------------

def _audit_members(path):
    return {i.filename.split("/")[-1]: i.file_size for i in zipfile.ZipFile(path).infolist() if "aud_" in i.filename}


def test_instanced_audit_round_trips_bit_identically(tmp_path):
    """Unique key/value tables plus per-tile reference sequences; the replayed sequence --
    order, repeats, values -- is the one that was written."""
    m = _mind()
    rng = np.random.default_rng(0)
    for i in range(12):
        m.teach("question %d" % i, "answer %d" % (i % 4))     # answers repeat: shared values
    for i in range(12):
        m.teach("question %d" % i, "answer %d" % (i % 4))     # repeats: same pairs again
    m.learning_save(str(tmp_path))
    p = str(tmp_path / "learning" / "state.lecore")
    mem = _audit_members(p)
    assert any(k.startswith("aud_kref_") for k in mem), "audit not instanced"
    assert "aud_K_q.npy" in mem and "aud_V_q.npy" in mem
    inst = getattr(m, "_audit_instanced", None)
    assert inst and inst["unique_keys"] <= inst["rows"] and inst["unique_keys"] == 12
    # COMPARE TWO LOADED MINDS, not live vs loaded: the q8 per-row pack (cosine 0.99996,
    # pre-existing, same in the stacked form) means a loaded trace never bit-matched the
    # live float64 one. What instancing must preserve is that a load from the instanced
    # file equals a load from the same file re-saved -- i.e. the stored sequence is exact.
    n = _mind()
    n.learning_load(str(tmp_path))
    n.learning_save(str(tmp_path))
    n2 = _mind()
    n2.learning_load(str(tmp_path))
    for ta, tb in zip(n.experience.tiles, n2.experience.tiles):
        assert np.array_equal(ta._trace, tb._trace), "trace not bit-identical across instanced save/load"
        assert len(ta._audit) == len(tb._audit)
        for (k1, v1), (k2, v2) in zip(ta._audit, tb._audit):
            assert np.array_equal(np.asarray(k1, np.float32), np.asarray(k2, np.float32))
            assert np.array_equal(np.asarray(v1, np.float32), np.asarray(v2, np.float32))
    assert n.experience.stats == n2.experience.stats
    for i in range(12):
        assert n2.ask("question %d" % i)["answer"] == "answer %d" % (i % 4)
    # idempotent: two saves with nothing in between are the same size. (Asks sit BEFORE
    # this bracket on purpose: an ask ingests text into the semantic table -- measured as
    # the other growth channel -- so a save after asks legitimately differs.)
    n2.learning_save(str(tmp_path))
    size = os.path.getsize(p)
    n2.learning_save(str(tmp_path))
    assert os.path.getsize(p) == size


def test_instanced_is_smaller_than_stacked_when_rows_repeat(tmp_path):
    """The point of instancing: repeated rows cost a reference, not a copy."""
    m = _mind()
    for _ in range(3):
        for i in range(10):
            m.teach("q%d" % i, "same answer")
    m.learning_save(str(tmp_path))
    p = str(tmp_path / "learning" / "state.lecore")
    mem = _audit_members(p)
    rows = sum(len(t._audit) for t in m.experience.tiles)
    ref_bytes = sum(v for k, v in mem.items() if "ref_" in k)
    table_bytes = sum(v for k, v in mem.items() if "aud_K_" in k or "aud_V_" in k)
    inst = m._audit_instanced
    # value atoms are content-derived from the PAIR, so a shared answer text is still ten
    # atoms; the repeats (3x) are what instancing collapses
    assert inst["unique_keys"] == 10 and inst["unique_values"] == 10, inst
    # the delta rule already skips most predicted repeats (measured: 14 rows from 30
    # teaches); instancing collapses the repeats that got through
    assert rows > 10, rows
    # a stacked store would be rows * 2 * dim bytes of codes; the tables are unique * dim
    dim = m.experience.tiles[0]._audit[0][0].__len__() if m.experience.tiles[0]._audit else 2048
    stacked = rows * 2 * dim
    assert table_bytes + ref_bytes < stacked, (table_bytes, ref_bytes, stacked)


def test_old_format_audit_still_loads(tmp_path):
    """A container written by the pre-instancing writer (aud_kq_<tile> per tile) keeps loading
    and answers the same -- and re-saving it produces the instanced form."""
    from holographic.unified.holographic_unified_p20_zoo import _q8_pack
    m = _mind()
    for i in range(6):
        m.teach("old %d" % i, "ans %d" % i)
    m.learning_save(str(tmp_path))
    p = str(tmp_path / "learning" / "state.lecore")
    # rewrite the container in the OLD per-tile stacked form
    src = zipfile.ZipFile(p)
    man = json.loads(src.read("manifest.json"))
    sec = [s for s in man["sections"] if s["kind"] == "lecore.learning.experience"][0]
    K = np.load(io.BytesIO(src.read("sections/%d/aud_K_q.npy" % man["sections"].index(sec))))
    names = {i.filename.split("/")[-1]: i.filename for i in src.infolist()}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
        for i in src.infolist():
            if "aud_" in i.filename:
                continue
            dst.writestr(i.filename, src.read(i.filename) if i.filename != "manifest.json" else b"")
        secdir = os.path.dirname(names["aud_K_q.npy"])
        arrays = {}
        for ti, t in enumerate(m.experience.tiles):
            aud = t._audit
            if not aud:
                continue
            ks = np.stack([np.asarray(k, np.float32) for k, _ in aud]); vs = np.stack([np.asarray(v, np.float32) for _, v in aud])
            for nm, arr in zip(("kq", "klo", "khi"), _q8_pack(ks)): arrays["aud_%s_%d" % (nm, ti)] = arr
            for nm, arr in zip(("vq", "vlo", "vhi"), _q8_pack(vs)): arrays["aud_%s_%d" % (nm, ti)] = arr
        for nm, arr in arrays.items():
            b = io.BytesIO(); np.save(b, arr); dst.writestr("%s/%s.npy" % (secdir, nm), b.getvalue())
        keep = [a for a in sec["arrays"] if not str(a).startswith("aud_")] if isinstance(sec["arrays"], list) else {}
        sec["arrays"] = (keep + list(arrays)) if isinstance(sec["arrays"], list) else dict(sec["arrays"], **{k: "%s/%s.npy" % (secdir, k) for k in arrays})
        dst.writestr("manifest.json", json.dumps(man))
    open(p, "wb").write(buf.getvalue())
    n = _mind()
    n.learning_load(str(tmp_path))
    for i in range(6):
        assert n.ask("old %d" % i)["answer"] == "ans %d" % i
    n.learning_save(str(tmp_path))
    assert any(k.startswith("aud_kref_") for k in _audit_members(p)), "re-save did not instance"


def test_container_without_audit_arrays_loads(tmp_path):
    """A merely compact partition must open; replay rebuilds the text-backed writes."""
    m = _mind()
    m.teach("x", "y")
    m.learning_save(str(tmp_path))
    p = str(tmp_path / "learning" / "state.lecore")
    src = zipfile.ZipFile(p)
    man = json.loads(src.read("manifest.json"))
    for s in man["sections"]:
        if s["kind"] == "lecore.learning.experience":
            s["arrays"] = [a for a in s["arrays"] if not str(a).startswith("aud_")] if isinstance(s["arrays"], list) \
                else {k: v for k, v in s["arrays"].items() if not k.startswith("aud_")}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
        for i in src.infolist():
            if "aud_" in i.filename:
                continue
            dst.writestr(i.filename, json.dumps(man) if i.filename == "manifest.json" else src.read(i.filename))
    open(p, "wb").write(buf.getvalue())
    n = _mind()
    n.learning_load(str(tmp_path))                          # raised KeyError 'aud_k_0' before
    assert n.ask("x")["answer"] == "y"
