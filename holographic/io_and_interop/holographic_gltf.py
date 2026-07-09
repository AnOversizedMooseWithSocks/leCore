"""Binary glTF (.glb) emission and parsing (FWD-2): the boundary between the NumPy back end and three.js.

WHY THIS MODULE EXISTS
----------------------
The mesh kernel (holographic_mesh.py, FWD-1) can represent and operate on explicit geometry, but nothing the
back end produces is visible until it can be HANDED to the front end, and three.js speaks glTF. glTF 2.0
already models almost the entire scene graph a renderer needs -- meshes, materials, transforms, cameras,
lights, skins, animation -- so the whole back-end -> front-end boundary very nearly collapses to "emit
glTF". This module is the minimal, spec-conformant core of that: turn a `Mesh` into a single-file binary
glTF (`.glb`) that three.js's `GLTFLoader` accepts, and parse one back. It is the thing that lets you SEE
anything; the vertical slice (a cube through this boundary) de-risks the single most important new plumbing
before any modeling features are piled on.

WHAT IT PROVIDES
  * `mesh_to_glb(mesh, ...)` -> bytes -- a valid `.glb`: a 12-byte header + a JSON chunk (the scene graph)
    + a BIN chunk (the packed vertex/index buffers). POSITION / NORMAL / TEXCOORD_0 / COLOR_0 attributes
    and a triangle index buffer, with the glTF-required POSITION min/max bounds.
  * `glb_to_mesh(data)` -> Mesh -- parse a `.glb` back to a mesh (positions, normals, uvs, faces). The
    round-trip that proves the boundary offline (the ultimate proof is loading it in three.js).
  * `write_glb(mesh, path)` / `read_glb(path)` -- file convenience.

  Materials are kept to a single default PBR material (a base colour); the rich material / multi-node /
  animation surface is glTF too, and is the natural growth path -- but the slice stays a single mesh, on the
  principle the backlog states: prove the boundary, then add features.

THE DELTA CHANNEL (deferred, flagged loudly)
  The forward backlog pairs this with a JSON patch/delta channel so a single-node edit transmits as a
  bounded patch rather than re-shipping the whole scene -- and the architecture sweep (ARCH-2) identifies
  that same delta protocol as the single highest-value architectural addition, because it serves the WHOLE
  core (memory updates, index maintenance, recipe edits), not just the viewport. It is intentionally NOT in
  this module: it is its own faculty with its own measurement bar, and folding it in here would violate the
  "prove the boundary first" discipline. This module ships the full-scene emission; the delta channel is the
  next item.

DETERMINISM (per ISA.md)
  The bytes are a pure deterministic function of the mesh: buffers are packed in a fixed attribute order
  (POSITION, NORMAL, UV, COLOR, INDICES), each region 4-byte aligned, JSON keys emitted in a fixed order via
  a stable serialisation. The same mesh yields a byte-identical `.glb` run to run (asserted in the tests) --
  the EXACT class, appropriate for a serialised artifact.

NO NEW DEPENDENCIES
  Pure Python `struct` + `json` + NumPy. glTF is little-endian; all packing uses explicit little-endian
  dtypes so the output is correct regardless of host byte order.
"""

import json
import struct

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh

# --- glTF constant codes (from the glTF 2.0 spec) ---
_GLB_MAGIC = 0x46546C67          # "glTF" as a little-endian uint32
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A         # "JSON"
_CHUNK_BIN = 0x004E4942          # "BIN\0"

_COMP_FLOAT = 5126               # componentType: 32-bit float
_COMP_USHORT = 5123              # componentType: unsigned 16-bit int (index buffer, small meshes)
_COMP_UINT = 5125                # componentType: unsigned 32-bit int (index buffer, large meshes)

_TARGET_ARRAY = 34962            # ARRAY_BUFFER (vertex attributes)
_TARGET_ELEMENT = 34963          # ELEMENT_ARRAY_BUFFER (indices)

_MODE_TRIANGLES = 4              # primitive.mode


def _pad4(b, fill=b"\x00"):
    """Pad a bytes object up to the next 4-byte boundary (glTF requires 4-byte chunk/region alignment)."""
    rem = (-len(b)) % 4
    return b + fill * rem


def mesh_to_glb(mesh, base_colour=(0.8, 0.8, 0.8, 1.0), generator="holostuff", material=None):
    """Serialise a `Mesh` to a single-file binary glTF (`.glb`) and return the bytes.

    Lays out one BIN buffer in a fixed attribute order, gives each attribute its own (4-byte aligned)
    bufferView and accessor, writes the glTF-required POSITION min/max, and wraps it all in a one-mesh,
    one-node, one-scene graph with a single default PBR material. The result is what three.js's GLTFLoader
    consumes.
    """
    buf = mesh.to_buffers()
    pos = np.ascontiguousarray(buf["position"], dtype="<f4")        # little-endian float32 VEC3
    nrm = np.ascontiguousarray(buf["normal"], dtype="<f4")
    has_uv = "uv" in buf
    has_col = "colour" in buf
    uv = np.ascontiguousarray(buf["uv"], dtype="<f4") if has_uv else None
    col = np.ascontiguousarray(buf["colour"], dtype="<f4") if has_col else None

    n_vert = pos.shape[0]
    indices = np.asarray(buf["indices"], dtype=np.int64).reshape(-1)
    # pick the smallest index type that fits -- uint16 for small meshes (the common case), else uint32
    if int(indices.max(initial=0)) < 65536:
        idx_bytes = indices.astype("<u2").tobytes()
        idx_comp = _COMP_USHORT
    else:
        idx_bytes = indices.astype("<u4").tobytes()
        idx_comp = _COMP_UINT

    # --- build the BIN blob, recording (byteOffset, byteLength) for each region ---
    blob = bytearray()
    views = []          # each: dict(byteOffset, byteLength, target)

    def add_region(raw, target):
        raw = _pad4(bytes(raw))                     # 4-byte align this region's END so the next starts aligned
        off = len(blob)
        blob.extend(raw)
        views.append({"byteOffset": off, "byteLength": len(raw), "target": target})
        return len(views) - 1

    pos_view = add_region(pos.tobytes(), _TARGET_ARRAY)
    nrm_view = add_region(nrm.tobytes(), _TARGET_ARRAY)
    uv_view = add_region(uv.tobytes(), _TARGET_ARRAY) if has_uv else None
    col_view = add_region(col.tobytes(), _TARGET_ARRAY) if has_col else None
    idx_view = add_region(idx_bytes, _TARGET_ELEMENT)

    # --- accessors (POSITION carries the required min/max bounds) ---
    pmin = pos.min(axis=0).tolist()
    pmax = pos.max(axis=0).tolist()
    accessors = [
        {"bufferView": pos_view, "componentType": _COMP_FLOAT, "count": n_vert, "type": "VEC3",
         "min": pmin, "max": pmax},
        {"bufferView": nrm_view, "componentType": _COMP_FLOAT, "count": n_vert, "type": "VEC3"},
    ]
    attributes = {"POSITION": 0, "NORMAL": 1}
    if has_uv:
        attributes["TEXCOORD_0"] = len(accessors)
        accessors.append({"bufferView": uv_view, "componentType": _COMP_FLOAT, "count": n_vert, "type": "VEC2"})
    if has_col:
        attributes["COLOR_0"] = len(accessors)
        accessors.append({"bufferView": col_view, "componentType": _COMP_FLOAT, "count": n_vert, "type": "VEC4"})
    idx_accessor = len(accessors)
    accessors.append({"bufferView": idx_view, "componentType": idx_comp,
                      "count": int(indices.size), "type": "SCALAR"})

    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes,
                                    "indices": idx_accessor,
                                    "material": 0,
                                    "mode": _MODE_TRIANGLES}]}],
        "materials": [material.to_gltf_dict() if material is not None
                      else {"pbrMetallicRoughness": {"baseColorFactor": list(base_colour),
                                                     "metallicFactor": 0.0,
                                                     "roughnessFactor": 0.8}}],
        "accessors": accessors,
        "bufferViews": [{"buffer": 0, **v} for v in views],
        "buffers": [{"byteLength": len(blob)}],
    }

    # --- assemble the .glb container: header + JSON chunk + BIN chunk ---
    # sort_keys gives a stable byte-identical serialisation run to run (determinism).
    json_bytes = _pad4(json.dumps(gltf, sort_keys=True, separators=(",", ":")).encode("utf-8"), fill=b"\x20")
    bin_bytes = _pad4(bytes(blob), fill=b"\x00")

    total = 12 + (8 + len(json_bytes)) + (8 + len(bin_bytes))
    out = bytearray()
    out += struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total)          # 12-byte header
    out += struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes  # JSON chunk
    out += struct.pack("<II", len(bin_bytes), _CHUNK_BIN) + bin_bytes     # BIN chunk
    return bytes(out)


def _read_accessor(accessor, views, blob):
    """Read one accessor's data out of the BIN blob as a NumPy array (float32 for VEC*, the right uint for
    SCALAR indices). Honours the bufferView byteOffset/byteLength; assumes a tightly-packed accessor (no
    interleaving / no accessor byteOffset), which is exactly how `mesh_to_glb` writes them."""
    view = views[accessor["bufferView"]]
    start = view["byteOffset"]
    raw = blob[start:start + view["byteLength"]]
    comp = accessor["componentType"]
    count = accessor["count"]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    dtype = {_COMP_FLOAT: "<f4", _COMP_USHORT: "<u2", _COMP_UINT: "<u4"}[comp]
    arr = np.frombuffer(raw, dtype=dtype, count=count * ncomp)
    return arr.reshape(count, ncomp) if ncomp > 1 else arr


def glb_to_mesh(data):
    """Parse a binary glTF (`.glb`) back into a `Mesh`. Reads the FIRST mesh primitive: POSITION, the
    triangle indices, and NORMAL / TEXCOORD_0 if present. The faces come back as triangles (glTF stores a
    triangle index buffer); positions and the triangle set are recovered exactly. This is the offline proof
    that what we emitted is well-formed and lossless across the boundary."""
    if len(data) < 12:
        raise ValueError("not a .glb: too short")
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("not a .glb: bad magic")
    if version != _GLB_VERSION:
        raise ValueError(f"unsupported glTF version {version}")

    # walk the chunks
    off = 12
    json_obj = None
    blob = b""
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + clen]
        off += clen
        if ctype == _CHUNK_JSON:
            json_obj = json.loads(chunk.decode("utf-8"))
        elif ctype == _CHUNK_BIN:
            blob = chunk
    if json_obj is None:
        raise ValueError("no JSON chunk in .glb")

    views = json_obj["bufferViews"]
    accessors = json_obj["accessors"]
    prim = json_obj["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]

    pos = _read_accessor(accessors[attrs["POSITION"]], views, blob).astype(float)
    idx = _read_accessor(accessors[prim["indices"]], views, blob).astype(np.int64).reshape(-1, 3)
    nrm = (_read_accessor(accessors[attrs["NORMAL"]], views, blob).astype(float)
           if "NORMAL" in attrs else None)
    uv = (_read_accessor(accessors[attrs["TEXCOORD_0"]], views, blob).astype(float)
          if "TEXCOORD_0" in attrs else None)

    faces = [tuple(int(i) for i in tri) for tri in idx]
    return Mesh(pos, faces, normals=nrm, uvs=uv)


def write_glb(mesh, path, **kw):
    """Write a mesh to a `.glb` file."""
    with open(path, "wb") as fh:
        fh.write(mesh_to_glb(mesh, **kw))


def read_glb(path):
    """Read a mesh from a `.glb` file."""
    with open(path, "rb") as fh:
        return glb_to_mesh(fh.read())


def validate_glb(data):
    """A structural conformance check on a `.glb`: the container a real GLTFLoader requires. Returns a dict
    of findings (all True on a good file). This is the offline stand-in for "three.js accepted it" -- it
    checks the things a loader checks before it ever touches the geometry."""
    ok = {}
    magic, version, total = struct.unpack_from("<III", data, 0)
    ok["magic_glTF"] = (magic == _GLB_MAGIC)
    ok["version_2"] = (version == _GLB_VERSION)
    ok["length_matches"] = (total == len(data))
    ok["length_4_aligned"] = (len(data) % 4 == 0)
    # chunks
    off = 12
    chunk_types = []
    aligned = True
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk_types.append(ctype)
        if clen % 4 != 0:
            aligned = False
        off += 8 + clen
    ok["has_json_chunk"] = (_CHUNK_JSON in chunk_types)
    ok["has_bin_chunk"] = (_CHUNK_BIN in chunk_types)
    ok["json_chunk_first"] = (len(chunk_types) > 0 and chunk_types[0] == _CHUNK_JSON)
    ok["chunks_4_aligned"] = aligned
    # POSITION accessor must carry min/max (a hard glTF requirement) -- re-read the JSON chunk to check
    j_off = 12
    clen, ctype = struct.unpack_from("<II", data, j_off)
    gltf = json.loads(data[j_off + 8:j_off + 8 + clen].decode("utf-8"))
    pos_acc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    ok["position_has_bounds"] = ("min" in pos_acc and "max" in pos_acc)
    return ok


def _selftest():
    from holographic.mesh_and_geometry.holographic_mesh import box, tetrahedron, grid

    # --- a cube through the boundary: emit, structurally validate, parse back ---
    m = box(2.0, 2.0, 2.0)
    m.uvs = np.zeros((m.n_vertices, 2))          # give it trivial UVs so TEXCOORD_0 is exercised
    data = mesh_to_glb(m)

    checks = validate_glb(data)
    assert all(checks.values()), f"glb failed structural validation: {checks}"
    assert len(data) % 4 == 0, "glb total length must be 4-aligned"

    back = glb_to_mesh(data)
    # positions survive exactly (we wrote and read float32); the cube's 8 verts come back
    assert back.n_vertices == 8, back.n_vertices
    assert np.allclose(back.vertices.astype(np.float32), m.vertices.astype(np.float32)), \
        "positions must survive the glTF round-trip"
    # the triangle count matches (6 quads -> 12 triangles)
    assert back.n_faces == 12, back.n_faces
    assert back.normals is not None and back.uvs is not None, "normals + uvs should round-trip"

    # --- determinism: the same mesh yields byte-identical glb ---
    assert mesh_to_glb(box(2, 2, 2)) == mesh_to_glb(box(2, 2, 2)), "glb must be byte-reproducible"

    # --- a triangle mesh (tetra) and an open mesh (grid) also emit + parse cleanly ---
    for mk in (tetrahedron(), grid(3, 3)):
        d = mesh_to_glb(mk)
        assert all(validate_glb(d).values())
        r = glb_to_mesh(d)
        assert np.allclose(r.vertices.astype(np.float32), mk.vertices.astype(np.float32))

    # --- file round-trip ---
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".glb")
    os.close(fd)
    try:
        write_glb(m, path)
        r = read_glb(path)
        assert r.n_vertices == 8
    finally:
        os.remove(path)

    print(f"holographic_gltf selftest: ok (cube -> {len(data)}-byte .glb, structurally valid, "
          f"round-trips positions/normals/uvs; byte-reproducible; tetra+grid+file round-trips clean)")


if __name__ == "__main__":
    _selftest()
