"""Tests for holographic_filemap.py -- ingest a folder/zip/file into a queryable, asset-tracked file map."""
import os
import zipfile
import shutil
import pytest
from holographic_filemap import ingest, FileMap


@pytest.fixture
def tree(tmp_path):
    files = {
        "readme.md": "renders water with a caustic shader and normal maps",
        "src/shader.glsl": "vec3 normal = computeNormal(); float caustic = refractLight(normal);",
        "src/util.py": "def load_texture(path): return open(path,'rb').read()",
        "textures/water/wave.png": "PNGDATA",
        "models/boat.obj": "v 0 0 0",
        "notes.txt": "todo fix the lighting setup and the boat mesh",
    }
    for rel, c in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(c)
    return tmp_path


def test_ingest_folder_counts_and_kinds(tree):
    fm = ingest(str(tree))
    assert len(fm) == 6
    assert fm.kinds()["image"] == 1 and fm.kinds()["model"] == 1 and fm.kinds()["code"] == 2


def test_query_by_name_and_kind(tree):
    fm = ingest(str(tree))
    assert [e.relpath for e in fm.find("*.png")] == [os.path.join("textures", "water", "wave.png")]
    assert len(fm.by_kind("text")) == 2                        # readme.md + notes.txt
    assert len(fm.find("src/*")) == 2


def test_keyword_search(tree):
    fm = ingest(str(tree))
    hits = [e.relpath for e, _ in fm.search_text("normal caustic")]
    assert any("shader" in h for h in hits)                    # both words in the shader
    any_hits = [e.relpath for e, _ in fm.search_text("lighting", mode="any")]
    assert any("notes" in h for h in any_hits)


def test_meaning_search(tree):
    fm = ingest(str(tree))
    fm.build_meaning_index(dim=256)
    hits = fm.find_by_meaning("lighting and shading", k=3)
    assert hits and all(len(h) == 2 for h in hits)             # (entry, score) tuples, top-first


def test_tree_is_the_file_map(tree):
    fm = ingest(str(tree))
    t = fm.tree()
    assert "textures" in t and "water" in t["textures"] and "wave.png" in t["textures"]["water"]
    assert "src" in t and "shader.glsl" in t["src"]


def test_ingest_zip(tmp_path):
    src = tmp_path / "proj"
    (src / "a").mkdir(parents=True)
    (src / "a" / "notes.txt").write_text("water shader lighting")
    (src / "img.png").write_text("PNG")
    z = tmp_path / "proj.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(src / "a" / "notes.txt", "a/notes.txt")
        zf.write(src / "img.png", "img.png")
    fm = ingest(str(z))
    assert len(fm) == 2
    assert [e.relpath for e, _ in fm.search_text("lighting")] == [os.path.join("a", "notes.txt")]


def test_ingest_single_file(tmp_path):
    p = tmp_path / "solo.txt"
    p.write_text("hello world")
    fm = ingest(str(p))
    assert len(fm) == 1 and fm.files[0].kind == "text"


def test_relocation_on_ingested_tree(tree, tmp_path):
    fm = ingest(str(tree))
    moved = tmp_path.parent / (tmp_path.name + "_moved")
    shutil.move(str(tree), str(moved))
    assert len(fm.missing()) == 6
    rel0 = fm.files[0].relpath
    fm.relink(fm.assets.assets[0].path, str(moved / rel0))     # fix ONE -> the rest follow
    assert len(fm.missing()) == 0


def test_change_detection_on_ingested_tree(tree):
    import time
    fm = ingest(str(tree))
    target = [e for e in fm.files if e.relpath == "notes.txt"][0]
    ref = [a for a in fm.assets.assets if a.path == target.path][0]
    assert ref.status() == "ok"
    time.sleep(0.01)
    with open(target.path, "a") as f:
        f.write(" more")
    os.utime(target.path, None)
    assert ref.status() == "modified"


def test_metadata_queries(tree):
    fm = ingest(str(tree))
    assert isinstance(fm.larger_than(0), list)
    assert len(fm.by_ext(".py")) == 1
