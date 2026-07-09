"""Tests for holographic_assets.py -- external asset relocation, change detection, and distributed resolution."""
import os
import time
import shutil
import tempfile
import pytest
from holographic.misc.holographic_assets import AssetLibrary, relocation, apply_relocation, common_suffix_len, find_by_relative_tail, find_by_hash, fingerprint


# ---- pure relocation logic (no disk) ----------------------------------------------------------------------
def test_relocation_finds_the_moved_parent():
    old = "/Users/me/Documents/project/textures/water/wave.png"
    new = "/Users/me/Projects/project/textures/water/wave.png"
    old_prefix, new_prefix, preserved = relocation(old, new)
    assert old_prefix == "/Users/me/Documents"
    assert new_prefix == "/Users/me/Projects"
    assert preserved == ["project", "textures", "water", "wave.png"]


def test_apply_relocation_rewrites_siblings():
    old_p, new_p = "/a/Documents", "/a/Projects"
    got = apply_relocation("/a/Documents/project/models/boat.obj", old_p, new_p)
    assert got == "/a/Projects/project/models/boat.obj"
    # a path NOT under the old prefix is left alone (returns None)
    assert apply_relocation("/b/other/thing.png", old_p, new_p) is None
    # folder-by-folder matching: /a/Documents2 is not under /a/Documents
    assert apply_relocation("/a/Documents2/x.png", old_p, new_p) is None


def test_common_suffix_len():
    assert common_suffix_len(["a", "b", "c"], ["x", "b", "c"]) == 2
    assert common_suffix_len(["a"], ["b"]) == 0


# ---- disk-based scenarios ---------------------------------------------------------------------------------
@pytest.fixture
def project(tmp_path):
    """An OLD project tree with three assets in a folder hierarchy."""
    old = tmp_path / "Documents" / "project"
    rels = ["textures/water/splashes/wave.png", "textures/stone/wall.png", "models/boat.obj"]
    for rel in rels:
        p = old.joinpath(*rel.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(("data:" + rel).encode())
    return tmp_path, old, rels


def test_relink_all_from_one(project):
    tmp_path, old, rels = project
    lib = AssetLibrary()
    for rel in rels:
        lib.add(str(old.joinpath(*rel.split("/"))), role=rel)
    assert len(lib.missing()) == 0

    # move the whole project, then re-point ONE asset
    new = tmp_path / "Projects" / "project"
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))
    assert len(lib.missing()) == 3

    one = str(new.joinpath("textures", "water", "splashes", "wave.png"))
    rep = lib.relink(lib.assets[0].path, one)
    assert len(lib.missing()) == 0                            # the other two were found automatically
    assert any(r["how"] == "prefix-swap" for r in rep["relinked"])


def test_search_by_structure(project):
    tmp_path, old, rels = project
    lib = AssetLibrary()
    ref = lib.add(str(old.joinpath("models", "boat.obj")))
    # reorganise: move it somewhere a clean prefix-swap wouldn't predict
    dest = tmp_path / "elsewhere" / "boat.obj"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(ref.path, str(dest))
    assert len(lib.missing()) == 1
    lib.search_under(str(tmp_path))                          # find by trailing name/structure
    assert len(lib.missing()) == 0
    assert lib.assets[0].path == str(dest)


def test_change_detection(project):
    tmp_path, old, rels = project
    lib = AssetLibrary()
    ref = lib.add(str(old.joinpath("textures", "stone", "wall.png")))
    assert ref.status() == "ok"
    time.sleep(0.01)
    with open(ref.path, "ab") as f:
        f.write(b" edited")
    os.utime(ref.path, None)
    assert ref.status() == "modified"                        # detected the on-disk edit
    ref.refresh()
    assert ref.status() == "ok"                              # acknowledged


def test_change_detection_by_hash(project):
    tmp_path, old, rels = project
    lib = AssetLibrary()
    ref = lib.add(str(old.joinpath("models", "boat.obj")), with_hash=True)
    assert ref.fp["sha256"]
    # a "touch" that changes mtime but NOT content -> hash says unchanged
    time.sleep(0.01); os.utime(ref.path, None)
    assert ref.status(with_hash=True) == "ok"                # content identical -> ok despite mtime change


def test_distributed_content_hash_resolve(project):
    tmp_path, old, rels = project
    lib = AssetLibrary()
    ref = lib.add(str(old.joinpath("models", "boat.obj")), with_hash=True)
    # the same file appears elsewhere under a different name (another machine's layout)
    other = tmp_path / "machineB" / "assets" / "renamed_boat.obj"
    other.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ref.path, str(other))
    os.remove(ref.path)                                      # the original path is now gone
    assert not ref.exists()
    resolved = lib.resolve(ref.id, roots=[str(tmp_path / "machineB")])
    assert resolved == str(other)                            # found by CONTENT, not path


def test_manifest_round_trip(project):
    tmp_path, old, rels = project
    lib = AssetLibrary()
    for rel in rels:
        lib.add(str(old.joinpath(*rel.split("/"))), role=rel, with_hash=True)
    path = str(tmp_path / "assets.json")
    lib.save(path)
    lib2 = AssetLibrary.load(path)
    assert [a.path for a in lib2.assets] == [a.path for a in lib.assets]
    assert [a.fp.get("sha256") for a in lib2.assets] == [a.fp.get("sha256") for a in lib.assets]


def test_report(project):
    tmp_path, old, rels = project
    lib = AssetLibrary()
    for rel in rels:
        lib.add(str(old.joinpath(*rel.split("/"))))
    os.remove(lib.assets[0].path)
    rep = lib.report()
    assert rep["counts"].get("missing") == 1 and rep["counts"].get("ok") == 2
