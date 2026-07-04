"""Tests for holographic_queryfolder (WS7 folders) + SceneCoder.combine (WS6 combine-scenes)."""
from holographic_query import Database, QueryError
from holographic_queryfolder import FolderTree, _bare
from holographic_scene import SceneCoder, COLOURS, SHAPES, TEXTURES


def _db():
    db = Database(); db.add_namespace("user", tier="persistent")
    for t in ("sales", "returns", "catalog", "scratch"):
        db.create_table("user." + t, ["id"], dim=256, seed=0)
    return db


def test_home_ownership_and_association_grouping():
    ft = FolderTree(_db())
    ft.set_home("user.sales", "reports"); ft.set_home("user.returns", "reports")
    ft.set_home("user.catalog", "reference"); ft.link("user.catalog", "reports")
    assert {_bare(q) for q in ft.tables_in("reports")} == {"sales", "returns", "catalog"}
    assert {_bare(q) for q in ft.tables_in("reference")} == {"catalog"}


def test_resolve_folder_path():
    db = _db(); ft = FolderTree(db)
    ft.set_home("user.sales", "reports")
    assert ft.resolve("reports/sales") is db.namespaces["user"]["tables"]["sales"]
    assert ft.resolve("reports/nope") is None


def test_unlink_keeps_table_but_home_unlink_refused():
    db = _db(); ft = FolderTree(db)
    ft.set_home("user.catalog", "reference"); ft.link("user.catalog", "reports")
    ft.unlink("user.catalog", "reports")
    assert {_bare(q) for q in ft.tables_in("reports")} == set()
    assert db.namespaces["user"]["tables"].get("catalog") is not None       # not deleted
    try:
        ft.unlink("user.catalog", "reference"); assert False                # can't unlink from home
    except QueryError:
        pass


def test_drop_folder_deletes_only_home_tables():
    db = _db(); ft = FolderTree(db)
    ft.set_home("user.sales", "reports"); ft.set_home("user.returns", "reports")
    ft.link("user.scratch", "reports")                                      # scratch only LINKED
    deleted = ft.drop_folder("reports")
    assert {_bare(q) for q in deleted} == {"sales", "returns"}
    assert db.namespaces["user"]["tables"].get("sales") is None             # owned -> deleted
    assert db.namespaces["user"]["tables"].get("scratch") is not None       # linked -> survives


def test_folder_references_only_existing_tables():
    ft = FolderTree(_db())
    try:
        ft.set_home("user.ghost", "reports"); assert False
    except QueryError:
        pass


def test_scoped_search_recursive():
    db = _db(); ft = FolderTree(db)
    ft.set_home("user.sales", "reports")
    ft.set_home("user.returns", "reports/q1")                               # a sub-folder
    assert {_bare(q) for q in ft.tables_in("reports", recursive=True)} == {"sales", "returns"}
    assert {_bare(q) for q in ft.tables_in("reports", recursive=False)} == {"sales"}


def test_combine_scenes_is_a_bundle():
    sc = SceneCoder(dim=4096, seed=0)
    a = sc.encode_scene([{"colour": COLOURS[0], "shape": SHAPES[0], "texture": TEXTURES[0]},
                         {"colour": COLOURS[1], "shape": SHAPES[1], "texture": TEXTURES[0]}])
    b = sc.encode_scene([{"colour": COLOURS[2], "shape": SHAPES[2], "texture": TEXTURES[0]}])
    combined = sc.combine(a, b)
    assert sc.count_objects(combined) == sc.count_objects(a) + sc.count_objects(b)
