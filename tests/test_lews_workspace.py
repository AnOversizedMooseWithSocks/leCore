"""The .lews shared-workspace standard (holographic_lews): schema versions + migration, canonical kinds, the live
Workspace (lock, atomic write, journal, revisions, conflicts), foreign kinds carried verbatim, and two 'apps' seeing
each other's work through the journal."""
import json
import os
import tempfile

import numpy as np
import pytest

import lecore
from holographic.io_and_interop import holographic_lews as L
from holographic.io_and_interop.holographic_container import image_section, load_container


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


@pytest.fixture
def root():
    d = tempfile.mkdtemp()
    yield d
    import shutil; shutil.rmtree(d, ignore_errors=True)


def test_sections_are_schema_stamped_and_canonical_kinds_validate():
    s = L.mesh_section([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], sid="m")
    assert s["meta"]["schema"] == 1 and s["arrays"]["verts"].dtype == np.float32 and s["arrays"]["faces"].dtype == np.int32
    with pytest.raises(ValueError):
        L.mesh_section([[0, 0, 0]], [[0, 1, 2]])
    with pytest.raises(ValueError):
        L.material_section("x", library="nosuchmineral")
    assert L.material_section("gem", library="amethyst")["meta"]["library"] == "amethyst"
    sc = L.scene_section([{"id": "o", "mesh": "m"}])
    assert len(sc["meta"]["objects"][0]["transform"]) == 16
    with pytest.raises(ValueError):
        L.scene_section([{"id": "o", "transform": [1, 2, 3]}])


def test_migration_chain_and_read_only_newer():
    L.register_kind_schema("t.kind", 3, "test", migrate={1: lambda s: {**s, "meta": {**s["meta"], "v2": True}},
                                                          2: lambda s: {**s, "meta": {**s["meta"], "v3": True}}})
    up = L.upgrade_section({"kind": "t.kind", "id": "a", "meta": {"schema": 1}, "arrays": {}})
    assert up["meta"]["schema"] == 3 and up["meta"]["v2"] and up["meta"]["v3"]
    same = L.upgrade_section({"kind": "t.kind", "id": "a", "meta": {"schema": 3}, "arrays": {}})
    assert "_read_only" not in same["meta"]
    newer = L.upgrade_section({"kind": "t.kind", "id": "a", "meta": {"schema": 4}, "arrays": {}})
    assert newer["meta"]["_read_only"] is True
    L.register_kind_schema("t.gap", 2, "test")                                   # no migration registered
    with pytest.raises(ValueError):
        L.upgrade_section({"kind": "t.gap", "id": "a", "meta": {"schema": 1}, "arrays": {}})
    foreign = L.upgrade_section({"kind": "nobody.knows", "id": "z", "meta": {"schema": 7}, "arrays": {}})
    assert "_read_only" not in foreign["meta"]                                     # unknown kinds: opaque, not flagged


def test_two_apps_share_a_workspace_through_the_journal(root):
    painter = L.Workspace(root, app="lestudio"); modeller = L.Workspace(root, app="polystudio")
    img = np.zeros((4, 4, 4), np.float32); img[..., 3] = 1
    s = image_section(img, name="t"); s["id"] = "tex"
    r1 = painter.put(s)
    r2 = modeller.put(L.mesh_section([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], sid="m"))
    r3 = modeller.put(L.scene_section([{"id": "o", "mesh": "m", "texture": "tex"}]))
    assert (r1, r2, r3) == (1, 2, 3)
    assert [e["id"] for e in painter.changes_since(r1)] == ["m", "scene"]
    img[..., 0] = 1; s2 = image_section(img, name="t2"); s2["id"] = "tex"
    r4 = painter.put(s2)
    ch = modeller.changes_since(r3); assert len(ch) == 1 and ch[0]["rev"] == r4 and ch[0]["app"] == "lestudio"
    assert modeller.get("tex")["meta"]["name"] == "t2" and modeller.get("tex")["meta"]["rev"] == r4
    d = modeller.describe(); assert d["rev"] == 4 and d["lews"] == L.LEWS_SPEC and {x["id"] for x in d["sections"]} == {"tex", "m", "scene"}
    # the file on disk is a valid container at all times, with the LEWS meta
    cont = load_container(open(os.path.join(root, L.Workspace.FILE), "rb").read())
    assert cont["meta"]["lews"] == L.LEWS_SPEC and cont["meta"]["rev"] == 4
    # delete + journal
    r5 = modeller.delete("m"); assert r5 == 5 and modeller.get("m") is None
    assert painter.changes_since(r4)[0]["op"] == "delete"
    assert modeller.delete("nope") == 5                                            # absent id: no new revision


def test_conflict_detection_and_lock_cleanup(root):
    a = L.Workspace(root, app="a"); b = L.Workspace(root, app="b")
    s = L.sdf_section("sphere(1)", sid="s")
    r = a.put(s)
    b.put(L.sdf_section("box(1,1,1)", sid="s"))                                     # b changes it first
    with pytest.raises(L.ConflictError):
        a.put(L.sdf_section("sphere(2)", sid="s"), expected_rev=r)
    assert a.put(L.sdf_section("sphere(2)", sid="s"), expected_rev=r + 1) == r + 2
    assert not os.path.exists(os.path.join(root, L.Workspace.LOCK))                # lock released after every write
    # a stale lock left by a dead holder is broken after lock_timeout
    open(os.path.join(root, L.Workspace.LOCK), "w").write("{}")
    old = os.path.join(root, L.Workspace.LOCK); os.utime(old, (1, 1))
    c = L.Workspace(root, app="c", lock_timeout=0.5)
    assert c.put(L.camera_section((0, 0, 3), (0, 0, 0), sid="cam")) == r + 3


def test_foreign_kind_round_trips_byte_identical(root):
    w = L.Workspace(root, app="x")
    foreign = {"kind": "other.app.thing", "id": "f", "meta": {"k": [1, {"z": 2}], "schema": 5}, "arrays": {"a": np.arange(6, dtype=np.uint8).reshape(2, 3)}}
    w.put(foreign)
    w.put(L.sdf_section("sphere(1)", sid="s"))                                        # another write must carry it through
    got = L.Workspace(root, app="y").get("f", upgrade=False)
    assert got["meta"]["k"] == [1, {"z": 2}] and got["arrays"]["a"].dtype == np.uint8 and np.array_equal(got["arrays"]["a"], foreign["arrays"]["a"])
    assert L.section_hash(got) == L.section_hash(foreign)


def test_mind_verbs_are_wired(mind, root):
    w = mind.lews_open(root, app="agent")
    w.put(mind.lews_material_section("point", sid="mat", library="quartz"))
    w.put(mind.lews_camera_section((0, 1, 3), (0, 0, 0), sid="cam"))
    d = mind.lews_describe(root); assert {s["kind"] for s in d["sections"]} == {L.MATERIAL_KIND, L.CAMERA_KIND}
    ch = mind.lews_changes(root, 0); assert len(ch) == 2 and json.dumps(ch)          # JSON-safe for /invoke
    assert mind.lews_register_kind("myapp.brush", 2, "brush") == "myapp.brush" and L.kind_schema_version("myapp.brush") == 2


# ---------------------------------------------------------------- sweep 162: the live session on the directory
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "lews")


def test_notes_advance_the_rev_without_rewriting_the_container(root):
    ws = L.Workspace(root, app="lestudio")
    r1 = ws.put(L.camera_section([0, 0, 4], [0, 0, 0], sid="cam"))
    size, mtime = os.path.getsize(ws.path), os.path.getmtime(ws.path)
    r2 = ws.bump("moose", "selection", {"layer": 3})
    assert r2 == r1 + 1 and ws.rev() == r2
    assert (os.path.getsize(ws.path), os.path.getmtime(ws.path)) == (size, mtime)   # the ZIP was not touched
    # the next section write continues the SAME counter (journal and container agree on one truth)
    assert ws.put(L.camera_section([1, 0, 4], [0, 0, 0], sid="cam")) == r2 + 1
    feed = L.Workspace(root, app="polystudio", create=False).since(r1, exclude="moose")
    assert [e["op"] for e in feed] == ["put"]                # the note's author is excluded from its own echo


def test_presence_is_a_heartbeat_keyed_by_person_with_a_sticky_host(root):
    painter = L.Workspace(root, app="lestudio"); modeller = L.Workspace(root, app="polystudio")
    painter.touch("moose", name="Moose", activity={"tool": "brush", "section": "tex1"})
    modeller.touch("agent-7", activity={"tool": "extrude", "section": "m1"})
    ro = modeller.roster(ttl=30)
    assert [r["who"] for r in ro] == ["moose", "agent-7"]
    assert ro[0]["host"] and ro[0]["app"] == "lestudio" and ro[1]["app"] == "polystudio"
    assert ro[1]["activity"] == {"tool": "extrude", "section": "m1"}
    # five re-heartbeats from the same person are ONE row, and the host does not flap
    for _ in range(5):
        painter.touch("moose")
    assert [r["who"] for r in painter.roster()] == ["moose", "agent-7"] and painter.roster()[0]["host"]
    assert painter.roster()[0]["app"] == "lestudio"          # a bare heartbeat keeps the announced app
    # silence past the ttl removes a participant; a clean leave removes it now
    assert painter.participants(ttl=0.0) == []
    modeller.touch("agent-7"); assert modeller.drop("agent-7") == []


def test_two_processes_see_each_other_present(root):
    """A second OS process heart-beats and posts a note; the first sees both without any shared memory."""
    import subprocess, sys
    ws = L.Workspace(root, app="lestudio"); ws.touch("moose", app="lestudio")
    code = ("import sys; from holographic.io_and_interop.holographic_lews import Workspace; "
            "w=Workspace(sys.argv[1], app='polystudio', create=False); w.touch('agent-7', activity={'tool':'extrude'}, app='polystudio'); "
            "print(w.bump('agent-7','object_added',{'id':'cube'}))")
    out = subprocess.run([sys.executable, "-c", code, root], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "0"}).stdout.strip()
    assert int(out) == 1
    who = {r["who"]: r for r in ws.roster()}
    assert set(who) == {"moose", "agent-7"} and who["agent-7"]["activity"]["tool"] == "extrude" and who["moose"]["host"]
    assert ws.since(0, exclude="moose")[-1]["meta"] == {"id": "cube"}


@pytest.mark.skipif(not os.path.exists(os.path.join(FIXTURES, "golden_ops.lews")), reason="fixture absent")
@pytest.mark.parametrize("name", ["golden_ops.lews", "golden_r47.lews"])
def test_lestudio_golden_files_open_and_survive_another_apps_edit(root, name):
    """leStudio's own golden .lews fixtures (kind lestudio.document, assets, brushes) are valid LEWS instances: a
    modeller opens one, adds its own sections, and every leStudio section hashes identically afterwards."""
    fx = os.path.join(FIXTURES, name)
    before = {s["id"]: L.section_hash(s) for s in load_container(open(fx, "rb").read())["sections"]}
    ws = L.Workspace.from_file(fx, os.path.join(root, "ws"), app="polystudio")
    d = ws.describe()
    assert d["lews"] == L.LEWS_SPEC and d["sections"][0]["kind"] == "lestudio.document" and d["sections"][0]["known"]
    assert ws.changes_since(0)[0]["op"] == "import" and ws.changes_since(0)[0]["meta"]["source_app"] == "lestudio"
    ws.put(L.mesh_section([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], sid="m1"))
    after = {s["id"]: L.section_hash(s) for s in load_container(ws.export_bytes())["sections"]}
    assert all(after[k] == v for k, v in before.items()) and "m1" in after
    # and the exported file still opens with the plain container reader another leStudio would use
    assert load_container(ws.export_bytes())["meta"]["active"] == "D1"


def test_mind_verbs_presence_note_wait_import(mind, root):
    fx = os.path.join(FIXTURES, "golden_r47.lews")
    if os.path.exists(fx):
        d = mind.lews_import(fx, os.path.join(root, "imp"), app="polystudio")
        assert d["sections"][0]["kind"] == "lestudio.document"
    mind.lews_open(root, app="lestudio")
    mind.lews_touch(root, "moose", activity={"tool": "brush"}, name="Moose", app="lestudio")
    mind.lews_touch(root, "agent-7", activity={"tool": "extrude"}, app="polystudio")
    r = mind.lews_note(root, "moose", "selection", {"layer": 3})
    ro = mind.lews_presence(root)
    assert [(p["who"], p["app"], p["host"]) for p in ro] == [("moose", "lestudio", True), ("agent-7", "polystudio", False)]
    got = mind.lews_wait(root, 0, timeout=0.2, exclude="nobody")
    assert got[-1]["kind"] == "selection" and got[-1]["rev"] == r
    assert mind.lews_wait(root, r, timeout=0.15) == []          # nothing new: returns after the timeout, empty
    assert mind.lews_leave(root, "agent-7") == ["moose"]
    json.dumps(ro); json.dumps(got)                              # JSON-safe for /invoke


def test_assets_stored_once_journal_refuses_inline_arrays_gc_exact(mind, root):
    mind.lews_open(root, app="lestudio")
    tip = np.ones((3, 3), np.float32)
    k = mind.lews_put_asset(root, tip, "tip"); r = L.Workspace(root, create=False).rev()
    assert mind.lews_put_asset(root, tip.copy(), "again") == k and L.Workspace(root, create=False).rev() == r
    with pytest.raises(ValueError):
        mind.lews_journal_section("img", [{"op": "bad", "pixels": np.zeros(3)}])
    ws = L.Workspace(root, create=False, app="lestudio")
    ws.put(mind.lews_journal_section("img", [{"op": "stamp", "asset": k, "x": 1, "y": 1, "seed": 3}]))
    orphan = mind.lews_put_asset(root, np.zeros((2, 2), np.float32), "orphan")
    assert mind.lews_gc_assets(root, dry_run=True) == [orphan]
    assert mind.lews_gc_assets(root) == [orphan] and mind.lews_get_asset(root, orphan) is None
    assert np.array_equal(mind.lews_get_asset(root, k), tip)
    # the journal is plain JSON any app can read, and the asset key inside it resolves from another app
    J = L.Workspace(root, create=False, app="polystudio").get("journal:img")
    json.dumps(J["meta"]); assert L.journal_asset_refs(J["meta"]["ops"]) == {k}


def test_mint_is_persisted_per_prefix_and_safe_across_processes(mind, root):
    import subprocess, sys
    mind.lews_open(root, app="lestudio")
    assert [mind.lews_mint(root, "L"), mind.lews_mint(root, "L"), mind.lews_mint(root, "O")] == ["L1", "L2", "O1"]
    code = ("import sys; from holographic.io_and_interop.holographic_lews import Workspace; "
            "w=Workspace(sys.argv[1], app='polystudio', create=False); print(w.mint('L'))")
    out = subprocess.run([sys.executable, "-c", code, root], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "0"}).stdout.strip()
    assert out == "L3" and mind.lews_mint(root, "L") == "L4"
    ws = L.Workspace(root, create=False)
    assert ws.id_counters() == {"L": 4, "O": 1} and [e["op"] for e in ws.changes_since(0)] == ["mint"] * 5


def test_preset_kind_is_json_and_readable_by_the_other_app(mind, root):
    w = mind.lews_open(root, app="polystudio")
    w.put(mind.lews_preset_section("museum", "polystudio.render", {"exposure": 1.2, "hdri": "voortrekker"}, tags=["lighting"]))
    with pytest.raises((TypeError, ValueError)):
        mind.lews_preset_section("bad", "x", {"tip": np.ones(3)})       # an array is an asset, not a preset
    p = L.Workspace(root, app="lestudio", create=False).sections("lecore.preset")[0]
    assert p["meta"]["target"] == "polystudio.render" and p["meta"]["params"]["exposure"] == 1.2 and p["meta"]["tags"] == ["lighting"]
    json.dumps(p["meta"])
