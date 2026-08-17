"""holographic_container.py -- an app-neutral, forward-compatible CONTAINER of typed sections (leStudio backlog 11).

WHY THIS EXISTS
---------------
Several apps are being built ON leCore -- an image editor/compositor, a 3-D modeller, a video editor -- and they all
need to share ONE workspace file so a document from one app can travel into another. The reusable kernel is small and
has nothing to do with images: a container is a ZIP of a `manifest.json` plus binary array payloads, where the
manifest is a list of TYPED SECTIONS -- {kind, id, meta, arrays} -- and **a section whose `kind` this reader does not
understand round-trips UNTOUCHED**. That last property is the whole point: the image editor can save a mesh section
it cannot open, hand the file to the 3-D app, and get its own document sections back byte-for-byte. The container
carries data it does not understand rather than dropping it.

leStudio's `.lews` v2 (save_workspace / load_workspace) is the reference implementation; this lifts the SECTION
machinery into the core so every app registers its own kinds instead of each re-owning the file format. SDF trees,
meshes, palettes, fields, agent traces, brushes, documents -- all become persistable in one file, and a new kind
added by any app does not break an older reader of the same file.

THE MODEL
---------
A SECTION is a plain dict:
    {"kind": str,            # a namespaced type tag, e.g. "lestudio.document", "lecore.mesh" (opaque to the reader)
     "id":   str,            # a within-file identifier (default "")
     "meta": <json-able>,    # small structured metadata (a dict/list of json scalars) -- schema is the kind's business
     "arrays": {name: ndarray}}   # the bulk numeric payload; NUMERIC only (no object/pickle -- see SAFETY)

`save_container(sections, meta=None) -> bytes` and `load_container(data) -> {"meta", "sections"}` are the whole API.
`meta` is an optional top-level dict describing the file as a whole (app name, version, active id, ...).

SAFETY (deliberate, not incidental)
-----------------------------------
Arrays are serialised with numpy's `.npy` format at `allow_pickle=False`, so a container can NEVER carry executable
pickle -- loading a foreign file runs no code and reconstructs only plain numeric arrays. An object-dtype array is
refused at save time with a clear error rather than silently pickled. This matters precisely because the point of the
format is to open files written by OTHER apps.

DETERMINISM (why the bytes are stable)
--------------------------------------
Byte-identical save->load->save is a stated requirement (a version-control diff of a workspace should be empty when
nothing changed). Three sources of non-determinism are removed: (1) the `.npy` format carries NO timestamp -- it is a
header + the raw buffer; (2) every ZIP entry is written with a FIXED date_time (the 1980 zip epoch) instead of the
wall clock; (3) entries are emitted in a FIXED order (manifest first, then sections in list order, arrays sorted by
name) and the manifest is `json.dumps(..., sort_keys=True)`. With `compress=False` (ZIP_STORED) the bytes are stable
regardless of zlib; the default `compress=True` (DEFLATE, fixed level) is deterministic within a given zlib and much
smaller -- which is what a save/load/save round-trip in one environment tests.

KEPT NEGATIVE: this is NOT the `workspace_manager` (holographic_workspace) -- that checkpoints a live DB's scratch
tables by REPLAY. This is a FILE FORMAT for typed sections. They compose (a workspace export can be carried as one
section kind), but they are different things: reach for the container to ship a project file between apps, for the
manager to snapshot engine state within a session.
"""

import io
import json
import zipfile

import numpy as np

FORMAT = "lecore.container"          # manifest tag, so a reader can recognise the format
VERSION = 1
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)   # the earliest date_time a ZIP entry can carry -- fixed, so no wall-clock leaks in


def _npy_bytes(arr):
    """Serialise one ndarray to raw .npy bytes. allow_pickle=False -> numeric arrays only, no executable payload, and
    the format carries no timestamp so the bytes depend only on dtype/shape/data (deterministic)."""
    a = np.asarray(arr)
    if a.dtype == object:
        raise ValueError("container arrays must be numeric; got an object-dtype array (pickle is refused for safety)")
    buf = io.BytesIO()
    np.lib.format.write_array(buf, a, allow_pickle=False)
    return buf.getvalue()


def _load_npy(data):
    """Inverse of _npy_bytes. allow_pickle=False, so a hostile file cannot execute code on load."""
    return np.lib.format.read_array(io.BytesIO(data), allow_pickle=False)


def _writestr(z, name, data, compress):
    """Add one entry with a FIXED timestamp (not the wall clock) so the archive is byte-stable. The compress flag is
    per-entry to keep it simple; we use one value for the whole file."""
    zi = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
    zi.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    zi.external_attr = 0o600 << 16                    # fixed perms -> one less thing to vary the bytes
    z.writestr(zi, data)


# ---- L-3: A SECTION-KIND REGISTRY, and L-4: the canonical image kind.
# save_container accepts ANY kind string, and nothing could tell you which ones
# a build understands -- so an app hardcoded the kinds it recognised and could
# not tell a user WHY a section was inert. It carried them silently instead.
_KINDS = {}


def register_kind(kind, describe=""):
    """Declare a section kind this build understands. Returns the kind."""
    _KINDS[str(kind)] = str(describe)
    return kind


def known_kinds():
    """{kind: description} for every kind this build understands.

    A UI can now say "this file also contains 3 polystudio.object sections (not
    editable here)" instead of carrying them silently -- which is the difference
    between a format that survives round-trips and one a user trusts."""
    return dict(_KINDS)


def describe_sections(container):
    """Per-section {kind, known, describe} for a loaded container. What a UI shows."""
    out = []
    for sec in (container or {}).get("sections", ()):
        k = sec.get("kind")
        out.append({"kind": k, "known": k in _KINDS,
                    "describe": _KINDS.get(k, "")})
    return out


#: L-4: THE CANONICAL IMAGE SECTION. Every app wants to publish "here is an
#: image to use as a texture", and without one name each app PAIR needs its own
#: adapter -- lestudio.document to polystudio.texture to whatever a video editor
#: writes, N^2 adapters for N apps. One documented kind makes it N.
#: Contract: `arrays["image"]` is (H, W, 4) float 0..1 STRAIGHT alpha (the same
#: convention composite_layers uses, so a composite result is publishable
#: unchanged), and meta carries colour_space (default "srgb") and dpi.
IMAGE_KIND = register_kind(
    "lecore.image",
    "an RGBA float 0..1 image any app can use as a texture; meta: "
    "{colour_space, dpi}")
register_kind("lestudio.document", "leStudio layer document")
register_kind("polystudio.object", "Poly Studio scene object")


def image_section(image, colour_space="srgb", dpi=72.0, name=None):
    """Build a `lecore.image` section from an RGB or RGBA float array."""
    a = np.asarray(image, np.float32)
    if a.ndim != 3 or a.shape[-1] not in (3, 4):
        raise ValueError("image must be (H, W, 3) or (H, W, 4), got %r"
                         % (a.shape,))
    if a.shape[-1] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], -1)
    return {"kind": IMAGE_KIND, "arrays": {"image": a},
            "meta": {"colour_space": str(colour_space), "dpi": float(dpi),
                     "name": name or "image"}}


def read_image_section(section):
    """(image, meta) from a `lecore.image` section, whoever wrote it."""
    if section.get("kind") != IMAGE_KIND:
        raise ValueError("not a %s section: %r" % (IMAGE_KIND,
                                                   section.get("kind")))
    return np.asarray(section["arrays"]["image"], np.float32), dict(
        section.get("meta") or {})


def save_container(sections, meta=None, compress=True):
    """Serialise a list of typed SECTIONS into one container file -> bytes.

    Each section is a dict {"kind": str, "id": str (default ""), "meta": <json-able> (default {}), "arrays":
    {name: ndarray} (default {})}. `meta` is optional top-level metadata for the file as a whole. Unknown kinds are
    never interpreted -- they are stored and returned verbatim, which is what makes the format forward-compatible
    across apps. Deterministic: saving the same sections twice yields identical bytes (see the module docstring).

    Raises on a non-numeric (object-dtype) array or non-JSON-serialisable meta -- both are caller errors that would
    otherwise corrupt a file other apps must read."""
    man_sections = []
    payloads = []                                     # (zip_path, npy_bytes), collected then written in fixed order
    for i, sec in enumerate(sections):
        kind = str(sec.get("kind", "unknown"))
        sid = str(sec.get("id", ""))
        smeta = sec.get("meta", {})
        arrays = sec.get("arrays") or {}
        names = sorted(arrays.keys())                 # sorted -> the manifest and the files share a stable order
        for name in names:
            payloads.append(("sections/%d/%s.npy" % (i, name), _npy_bytes(arrays[name])))
        man_sections.append({"kind": kind, "id": sid, "meta": smeta, "arrays": names})
    manifest = {"format": FORMAT, "version": VERSION, "meta": (meta if meta is not None else {}),
                "sections": man_sections}
    # json.dumps validates JSON-serialisability (raises TypeError on a bad meta) and sort_keys makes it deterministic.
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        _writestr(z, "manifest.json", manifest_bytes, compress)     # manifest first, always
        for path, data in payloads:                                 # then payloads in section-then-sorted-name order
            _writestr(z, path, data, compress)
    return buf.getvalue()


def load_container(data):
    """Inverse of save_container: bytes -> {"meta": <top-level meta>, "sections": [ {kind, id, meta, arrays}, ... ]}.

    Sections come back in their saved order; each section's `arrays` is a dict {name: ndarray}. A section whose kind
    this caller does not understand is returned exactly as stored -- the reader is expected to keep it and write it
    back out, so a file round-trips through an app that only understands SOME of its kinds. Raises on a file that is
    not a container (bad/missing manifest)."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        manifest = json.loads(z.read("manifest.json"))
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise ValueError("not a lecore container (bad or missing manifest.json): %s" % (e,))
    if manifest.get("format") != FORMAT:
        raise ValueError("unrecognised container format %r (expected %r)" % (manifest.get("format"), FORMAT))
    sections = []
    for i, sm in enumerate(manifest.get("sections", [])):
        arrays = {}
        for name in sm.get("arrays", []):
            arrays[name] = _load_npy(z.read("sections/%d/%s.npy" % (i, name)))
        sections.append({"kind": sm.get("kind", "unknown"), "id": sm.get("id", ""),
                         "meta": sm.get("meta", {}), "arrays": arrays})
    return {"meta": manifest.get("meta", {}), "sections": sections}


def _sections_equal(a, b):
    """True iff two section lists carry the same kinds/ids/meta and bit-identical arrays -- the round-trip contract."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x["kind"] != y["kind"] or x["id"] != y["id"] or x["meta"] != y["meta"]:
            return False
        if sorted(x["arrays"]) != sorted(y["arrays"]):
            return False
        for k in x["arrays"]:
            if not np.array_equal(x["arrays"][k], y["arrays"][k]):
                return False
            if x["arrays"][k].dtype != y["arrays"][k].dtype:
                return False
    return True


def _selftest():
    """The container's real contracts, as numeric/byte asserts (not a smoke test): a full round-trip preserves
    kinds/ids/meta/arrays and dtypes; an UNKNOWN kind round-trips untouched; save->load->save is BYTE-IDENTICAL;
    saving the same input twice is deterministic; object arrays and non-container bytes are refused."""
    rng = np.random.default_rng(0)
    sections = [
        {"kind": "lestudio.document", "id": "D1", "meta": {"w": 640, "h": 480, "name": "untitled"},
         "arrays": {"layer0": rng.random((8, 8, 3)).astype(np.float32),
                    "mask0": (rng.random((8, 8)) > 0.5)}},                       # bool array survives
        {"kind": "lecore.mesh", "id": "M1", "meta": {"n_faces": 2},
         "arrays": {"verts": rng.random((4, 3)), "faces": np.array([[0, 1, 2], [1, 2, 3]], np.int64)}},
        {"kind": "some.future.kind.this.reader.never.heard.of", "id": "X9",     # <-- the forward-compat case
         "meta": {"opaque": [1, 2, {"nested": True}]},
         "arrays": {"blob": rng.integers(0, 255, (5, 5), np.uint8)}},
        {"kind": "empty.section", "id": "", "meta": {}, "arrays": {}},           # a section with NO arrays is legal
    ]
    blob = save_container(sections, meta={"app": "test", "active": "D1"})

    # 1. ROUND-TRIP: everything comes back, arrays bit-identical, top-level meta preserved.
    got = load_container(blob)
    assert got["meta"] == {"app": "test", "active": "D1"}, got["meta"]
    assert _sections_equal(sections, got["sections"]), "round-trip changed the sections"
    # dtypes specifically (a float32 layer must not silently promote to float64, etc.)
    d = {s["id"]: s for s in got["sections"]}
    assert d["D1"]["arrays"]["layer0"].dtype == np.float32
    assert d["D1"]["arrays"]["mask0"].dtype == np.bool_
    assert d["M1"]["arrays"]["faces"].dtype == np.int64

    # 2. FORWARD COMPATIBILITY: the unknown kind is returned exactly as stored -- kind string, nested meta, and blob.
    fut = d["X9"]
    assert fut["kind"] == "some.future.kind.this.reader.never.heard.of"
    assert fut["meta"] == {"opaque": [1, 2, {"nested": True}]}
    assert np.array_equal(fut["arrays"]["blob"], sections[2]["arrays"]["blob"])

    # 3. BYTE-IDENTICAL save->load->save: load a container and write it straight back -> the same bytes. This is the
    #    property that makes a workspace diff empty when nothing changed, and that carries foreign kinds losslessly.
    re_saved = save_container(got["sections"], meta=got["meta"])
    assert re_saved == blob, "save->load->save was not byte-identical"

    # 4. DETERMINISM: saving the same input twice is identical (no wall-clock timestamp leaked into the zip).
    assert save_container(sections, meta={"app": "test", "active": "D1"}) == blob

    # 5. compress=False is also a valid, deterministic round-trip (and byte-stable regardless of zlib).
    stored = save_container(sections, compress=False)
    assert _sections_equal(sections, load_container(stored)["sections"])
    assert save_container(sections, compress=False) == stored

    # 6. SAFETY + guards (each must RAISE, not silently corrupt): an object-dtype array, and non-container bytes.
    for bad in (lambda: save_container([{"kind": "k", "arrays": {"o": np.array([{"x": 1}], object)}}]),
                lambda: load_container(b"not a zip at all"),
                lambda: load_container(save_container([], compress=False).replace(b"lecore.container", b"x.y"))):
        try:
            bad()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("a bad input must raise, not corrupt the file")

    print("OK: holographic_container self-test passed (round-trip preserves kinds/ids/meta/arrays + dtypes; an "
          "UNKNOWN kind round-trips untouched; save->load->save is BYTE-IDENTICAL and deterministic; object arrays "
          "and non-container bytes are refused)")


if __name__ == "__main__":
    _selftest()
