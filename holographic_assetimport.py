"""holographic_assetimport.py -- import the file formats artists actually hand you.

WHAT IT LOADS
  * OBJ (+ its MTL)     -- Wavefront geometry with UVs/normals + the materials it references (map_* textures loaded).
  * glTF / GLB          -- the modern interchange: geometry AND its PBR materials (base colour / metallic / roughness /
                           normal / occlusion / emissive), per-vertex UVs and normals, embedded textures, AND -- for
                           rigged models -- ANIMATIONS (keyframed node transforms, sampled at any time) and SKINS
                           (joints + inverse-bind matrices + per-vertex weights).
  * a texture SET       -- a folder of PBR maps exported from Adobe Substance 3D Painter (or any tool): basecolor /
                           normal / roughness / metallic / height / ao / emissive, matched by their file names, built
                           into one PBRMaterial.
  * volumetric grids    -- a 3-D density grid (.npy, or raw floats + dims) wrapped as a field the volume renderer
                           marches (mind.render_volume).

HONEST BOUNDARIES (kept loud, stated up front so nothing surprises you):
  * We import the OPEN, EXPORTED forms. The PROPRIETARY project files need their vendor's engine and are NOT parsed
    here: Substance's .sbsar / .spp (export the texture maps from Painter instead), and OpenVDB's .vdb (a sparse
    hierarchical format -- export a dense .npy/.raw grid, or convert with the OpenVDB tools).
  * Image decoding uses PIL, imported LAZILY only when a texture is actually loaded, so importing leCore stays
    NumPy-only. A texture that can't be found/decoded becomes None (the factor-level material still works).
  * OBJ handling is the common case (v / vt / vn / f / usemtl / mtllib, polygons fan-triangulated); exotic OBJ
    features (free-form curves, smoothing groups) are ignored, not errored.

Readable + stdlib/NumPy only (os, json, struct, base64). Reuses the engine's own primitives: Mesh (geometry),
materials_from_mtl + PBRMaterial + TextureMap (materials), glb_to_mesh (glTF geometry), volume_render (the field).
"""
import os
import json
import struct

import numpy as np


# =========================================================================================================
# shared: a lazy image loader (PIL lives at the boundary, never in the core import path)
# =========================================================================================================
def _image_from_path(path):
    """Load an image FILE into a float (H, W, 3) array in [0,1], or None if missing/undecodable."""
    if not path or not os.path.exists(path):
        return None
    try:
        from PIL import Image                                  # lazy: only when a texture is actually loaded
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return None


def _image_from_bytes(data):
    """Load an image from raw bytes (a texture embedded in a .glb), or None."""
    try:
        from PIL import Image
        import io
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return None


def _texmap(arr):
    """Wrap an image array as a UV-sampleable TextureMap (or None)."""
    if arr is None:
        return None
    from holographic_materialio import TextureMap
    return TextureMap(arr, wrap="repeat")


# =========================================================================================================
# OBJ (+ MTL)
# =========================================================================================================
class LoadedMesh:
    """The result of importing a mesh asset: positions + triangle faces + (optional) UVs/normals, the material NAME
    each face uses, the materials themselves (name -> PBRMaterial), and -- for rigged/animated glTF -- the ANIMATIONS
    (keyframed node transforms), the SKINS (joints + inverse-bind matrices), and the per-vertex skin weights. `.mesh()`
    hands back a plain engine Mesh (positions + faces) for the existing geometry pipeline."""

    def __init__(self, positions, faces, uv=None, normals=None, face_material=None, materials=None,
                 animations=None, skins=None, joints=None, weights=None, nodes=None,
                 node_graph=None, morph_targets=None, morph_weights=None):
        self.positions = np.asarray(positions, float)          # (Nv, 3)
        self.faces = np.asarray(faces, int) if len(faces) else np.zeros((0, 3), int)   # (Nf, 3) position indices
        self.uv = uv                                           # OBJ: (Nf,3,2) per-corner; glTF: (Nv,2) per-vertex
        self.normals = normals                                 # OBJ: (Nf,3,3) per-corner; glTF: (Nv,3) per-vertex
        self.face_material = face_material or []               # length Nf: the material name per face
        self.materials = materials or {}                       # name -> PBRMaterial
        self.animations = animations or []                     # list[AnimationClip] -- keyframed node transforms
        self.skins = skins or []                               # list[dict]: {'joints': [...], 'inverse_bind': (J,4,4)}
        self.joints = joints                                   # (Nv, 4) per-vertex joint indices (skinning), or None
        self.weights = weights                                 # (Nv, 4) per-vertex joint weights (skinning), or None
        self.nodes = nodes or []                               # glTF node NAMES (animations target by index)
        self.node_graph = node_graph or []                     # per node {name, local(4x4 rest), children} -- the rig
        self.morph_targets = morph_targets                     # (T, Nv, 3) position deltas per morph target, or None
        self.morph_weights = morph_weights                     # (T,) default morph weights, or None

    def mesh(self):
        from holographic_mesh import Mesh
        return Mesh(self.positions, [tuple(f) for f in self.faces])

    def __repr__(self):
        extra = ""
        if self.animations:
            extra += ", %d animation(s)" % len(self.animations)
        if self.skins:
            extra += ", %d skin(s)" % len(self.skins)
        return "LoadedMesh(%d verts, %d faces, %d materials%s%s)" % (
            len(self.positions), len(self.faces), len(self.materials),
            ", uv" if self.uv is not None else "", extra)


def _face_token(tok, nv, nt, nn):
    """Parse one 'f' token 'v', 'v/vt', 'v//vn' or 'v/vt/vn' -> (vi, ti, ni) 0-indexed (or -1). OBJ is 1-indexed and
    allows NEGATIVE indices (relative to the end), which we resolve here."""
    a = (tok.split("/") + ["", ""])[:3]
    def one(s, n):
        if s == "":
            return -1
        i = int(s)
        return (n + i) if i < 0 else (i - 1)                   # negative = from the end; else 1-indexed -> 0-indexed
    return one(a[0], nv), one(a[1], nt), one(a[2], nn)


def load_obj(path):
    """Load a Wavefront .obj (and its .mtl, if referenced) into a LoadedMesh. Polygons are fan-triangulated; UVs and
    normals are kept per triangle corner when present; each face remembers its usemtl material; the mtllib is parsed
    into PBRMaterials with their map_* textures loaded."""
    V, VT, VN = [], [], []                                     # the raw v / vt / vn tables
    tris, tri_uv, tri_n, tri_mat = [], [], [], []
    cur_mtl, mtllib = None, None
    base = os.path.dirname(os.path.abspath(path))

    with open(path, "r", errors="ignore") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            k = p[0]
            if k == "v":
                V.append([float(x) for x in p[1:4]])
            elif k == "vt":
                VT.append([float(x) for x in p[1:3]])
            elif k == "vn":
                VN.append([float(x) for x in p[1:4]])
            elif k == "mtllib":
                mtllib = " ".join(p[1:])
            elif k == "usemtl":
                cur_mtl = " ".join(p[1:])
            elif k == "f":
                corners = [_face_token(t, len(V), len(VT), len(VN)) for t in p[1:]]
                for i in range(1, len(corners) - 1):           # fan-triangulate the polygon
                    fan = (corners[0], corners[i], corners[i + 1])
                    tris.append([c[0] for c in fan])
                    tri_uv.append([(VT[c[1]] if 0 <= c[1] < len(VT) else [0.0, 0.0]) for c in fan])
                    tri_n.append([(VN[c[2]] if 0 <= c[2] < len(VN) else [0.0, 0.0, 0.0]) for c in fan])
                    tri_mat.append(cur_mtl)

    materials = {}
    if mtllib:
        materials = _load_mtl(os.path.join(base, mtllib))

    uv = np.array(tri_uv, float) if VT and tris else None
    nrm = np.array(tri_n, float) if VN and tris else None
    return LoadedMesh(np.array(V, float), tris, uv=uv, normals=nrm, face_material=tri_mat, materials=materials)


def _load_mtl(mtl_path):
    """Parse an .mtl into {name: PBRMaterial}, loading the map_* texture FILES (relative to the mtl) onto the maps."""
    if not os.path.exists(mtl_path):
        return {}
    from holographic_materialio import materials_from_mtl
    text = open(mtl_path, "r", errors="ignore").read()
    mats = {m.name: m for m in materials_from_mtl(text)}       # Kd/Ks/Ns/Pr/Pm etc. (the engine's own parser)

    # materials_from_mtl reads the numeric fields; here we also load the referenced texture IMAGES (map_Kd, ...),
    # which live next to the .mtl, and attach them so the imported material carries its textures, not just factors.
    base = os.path.dirname(os.path.abspath(mtl_path))
    cur = None
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "newmtl":
            cur = mats.get(" ".join(p[1:]))
        elif cur is not None and p[0] in ("map_Kd", "map_Ke", "map_Pr", "map_Pm", "map_Bump", "bump", "norm"):
            tex = _texmap(_image_from_path(os.path.join(base, p[-1])))   # last token = the file (skip -options)
            if p[0] == "map_Kd":
                cur.base_color_map = tex
            elif p[0] == "map_Ke":
                cur.emissive_map = tex
            elif p[0] == "map_Pr":
                cur.roughness_map = tex
            elif p[0] == "map_Pm":
                cur.metallic_map = tex
            else:                                              # bump / normal -> carried as an attribute
                cur.normal_map = tex
    return mats


# =========================================================================================================
# glTF / GLB -- geometry (via the engine's glb_to_mesh) + PBR materials + embedded textures
# =========================================================================================================
_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942


def _glb_chunks(data):
    """Split a .glb blob into (gltf_json_dict, bin_bytes). The container is a 12-byte header then length-prefixed
    chunks; we pull the JSON chunk (the scene graph) and the BIN chunk (packed buffers / embedded images)."""
    magic, _ver, _total = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("not a .glb (bad magic)")
    off = 12
    gltf, binblob = None, b""
    while off + 8 <= len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        body = data[off + 8: off + 8 + clen]
        if ctype == _CHUNK_JSON:
            gltf = json.loads(body.decode("utf-8"))
        elif ctype == _CHUNK_BIN:
            binblob = body
        off += 8 + clen
    if gltf is None:
        raise ValueError("no JSON chunk in .glb")
    return gltf, binblob


def _gltf_image_array(gltf, binblob, image_index, base_dir):
    """Decode glTF image #image_index to an array: either embedded in the BIN buffer (via a bufferView), a data: URI,
    or an external file next to the .gltf. Returns (H,W,3) float or None."""
    images = gltf.get("images", [])
    if not (0 <= image_index < len(images)):
        return None
    img = images[image_index]
    if "bufferView" in img:                                    # embedded in the .glb BIN chunk
        bv = gltf["bufferViews"][img["bufferView"]]
        start = bv.get("byteOffset", 0)
        return _image_from_bytes(binblob[start: start + bv["byteLength"]])
    uri = img.get("uri", "")
    if uri.startswith("data:"):                                # inline base64 data URI
        import base64
        return _image_from_bytes(base64.b64decode(uri.split(",", 1)[1]))
    if uri:                                                    # external file beside the .gltf
        return _image_from_path(os.path.join(base_dir, uri))
    return None


def _gltf_materials(gltf, binblob, base_dir):
    """Build a list of PBRMaterial from the glTF material array: baseColor/metallic/roughness factors + their
    textures, plus normal and emissive. Texture -> image index chases material.texture.source through the tables."""
    from holographic_materialio import PBRMaterial

    def tex_array(tex_ref):
        if not tex_ref or "index" not in tex_ref:
            return None
        tex = gltf.get("textures", [])[tex_ref["index"]]
        return _gltf_image_array(gltf, binblob, tex.get("source", -1), base_dir)

    out = []
    for m in gltf.get("materials", [{}]):
        pbr = m.get("pbrMetallicRoughness", {})
        bc = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])
        mat = PBRMaterial(name=m.get("name", "material"),
                          base_color=tuple(bc), metallic=float(pbr.get("metallicFactor", 1.0)),
                          roughness=float(pbr.get("roughnessFactor", 1.0)),
                          emissive=tuple(m.get("emissiveFactor", [0.0, 0.0, 0.0])))
        mat.base_color_map = _texmap(tex_array(pbr.get("baseColorTexture")))
        # glTF packs metallic+roughness in ONE texture (B=metallic, G=roughness); we carry it on both channels
        mr = _texmap(tex_array(pbr.get("metallicRoughnessTexture")))
        mat.metallic_map = mat.roughness_map = mr
        mat.emissive_map = _texmap(tex_array(m.get("emissiveTexture")))
        mat.normal_map = _texmap(tex_array(m.get("normalTexture")))     # carried (attribute)
        mat.ao_map = _texmap(tex_array(m.get("occlusionTexture")))      # ambient occlusion (attribute)
        out.append(mat)
    return out


def _accessor(gltf, index, blob):
    """Read glTF accessor #index out of the BIN blob as a NumPy array -- a ROBUST reader that (unlike the emitter's
    tightly-packed one) honours the accessor's OWN byteOffset and handles every component type and shape real files
    use, including MAT4 (inverse-bind matrices) and the small int types skin JOINTS/WEIGHTS come in. Returns
    (count, ncomp) for vectors/matrices, (count,) for scalars."""
    acc = gltf["accessors"][index]
    bv = gltf["bufferViews"][acc["bufferView"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}[acc["type"]]
    dtype = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[acc["componentType"]]
    count = acc["count"]
    arr = np.frombuffer(blob, dtype=np.dtype(dtype), count=count * ncomp, offset=start)
    return arr.reshape(count, ncomp) if ncomp > 1 else arr.copy()


def _quat_to_mat(q):
    """A unit quaternion (x, y, z, w) -> a 3x3 rotation matrix (the standard glTF convention)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]], float)


def _compose_trs(T, R, S):
    """Compose translation (3,), rotation quaternion (4,), scale (3,) into a 4x4 local matrix -- M = T * R * S."""
    M = np.eye(4)
    M[:3, :3] = _quat_to_mat(R) * np.asarray(S)               # scale columns, then rotate
    M[:3, 3] = T
    return M


def _interp_linear(times, values, t):
    """Linearly interpolate `values` (K, C) keyed at `times` (K,) at scalar time t, clamping past the ends."""
    if len(times) == 1 or t <= times[0]:
        return np.asarray(values[0], float)
    if t >= times[-1]:
        return np.asarray(values[-1], float)
    i = int(np.searchsorted(times, t)) - 1
    span = times[i + 1] - times[i]
    a = 0.0 if span <= 0 else (t - times[i]) / span
    return (1 - a) * np.asarray(values[i], float) + a * np.asarray(values[i + 1], float)


def _interp_quat(times, values, t):
    """Slerp a rotation channel (quaternions) at scalar time t, clamping past the ends. Uses the engine's slerp."""
    from holographic_ai import slerp
    if len(times) == 1 or t <= times[0]:
        return np.asarray(values[0], float)
    if t >= times[-1]:
        return np.asarray(values[-1], float)
    i = int(np.searchsorted(times, t)) - 1
    span = times[i + 1] - times[i]
    a = 0.0 if span <= 0 else (t - times[i]) / span
    return np.asarray(slerp(np.asarray(values[i], float), np.asarray(values[i + 1], float), a), float)


class AnimationClip:
    """One glTF animation: keyframed node transforms over time. `channels` is {node_index: {path: (times, values)}}
    where path is 'translation' / 'rotation' / 'scale' / 'weights'. `sample(t)` returns {node_index: 4x4 LOCAL matrix}
    with translation/scale interpolated linearly and rotation SLERPed (the engine's slerp) -- ready to feed a scene
    graph. `duration` is the last keyframe time. This is what makes a rigged glTF actually move."""

    def __init__(self, name, channels):
        self.name = name
        self.channels = channels
        self.duration = max((tv[0][-1] for paths in channels.values() for tv in paths.values() if len(tv[0])),
                            default=0.0)

    def nodes(self):
        return sorted(self.channels)

    def sample(self, t):
        """The local transform of every animated node at time t, as {node_index: 4x4}."""
        out = {}
        for node, paths in self.channels.items():
            T = _interp_linear(*paths["translation"], t) if "translation" in paths else np.zeros(3)
            R = _interp_quat(*paths["rotation"], t) if "rotation" in paths else np.array([0.0, 0.0, 0.0, 1.0])
            S = _interp_linear(*paths["scale"], t) if "scale" in paths else np.ones(3)
            out[node] = _compose_trs(T, R, S)
        return out

    def sample_channel(self, node, path, t):
        """The raw interpolated value of one channel (e.g. morph 'weights') at time t, or None if absent."""
        paths = self.channels.get(node, {})
        if path not in paths:
            return None
        times, values = paths[path]
        return _interp_quat(times, values, t) if path == "rotation" else _interp_linear(times, values, t)

    def __repr__(self):
        return "AnimationClip(%r, %d nodes, %.3fs)" % (self.name, len(self.channels), self.duration)


def _gltf_animations(gltf, blob):
    """Parse the glTF `animations` array into a list of AnimationClip. Each animation is a set of CHANNELS (a target
    node + a path) driven by SAMPLERS (input = keyframe times accessor, output = keyframe values accessor). We read
    the accessors and group them by node so sampling gives a whole transform at once."""
    clips = []
    for ai, anim in enumerate(gltf.get("animations", [])):
        samplers = anim.get("samplers", [])
        channels = {}                                          # node -> {path -> (times, values)}
        for ch in anim.get("channels", []):
            target = ch.get("target", {})
            node = target.get("node")
            path = target.get("path")                          # translation / rotation / scale / weights
            if node is None or path is None:
                continue
            samp = samplers[ch["sampler"]]
            times = _accessor(gltf, samp["input"], blob)       # (K,)
            values = _accessor(gltf, samp["output"], blob)     # (K, C)
            channels.setdefault(node, {})[path] = (np.asarray(times, float), np.asarray(values, float))
        if channels:
            clips.append(AnimationClip(anim.get("name", "animation_%d" % ai), channels))
    return clips


def _gltf_skins(gltf, blob):
    """Parse the glTF `skins` array: each skin is a set of JOINT nodes and their INVERSE-BIND matrices (which move a
    vertex from mesh space into each joint's local space before the joint's animated transform is applied). Returns a
    list of {'joints': [node indices], 'inverse_bind': (J,4,4) or None}."""
    skins = []
    for sk in gltf.get("skins", []):
        joints = list(sk.get("joints", []))
        ibm = None
        if "inverseBindMatrices" in sk:
            m = _accessor(gltf, sk["inverseBindMatrices"], blob)          # (J, 16) column-major
            ibm = m.reshape(-1, 4, 4).transpose(0, 2, 1)                  # -> (J,4,4) row-major
        skins.append({"joints": joints, "inverse_bind": ibm})
    return skins


def _node_rest_local(n):
    """A glTF node's REST local transform as a 4x4: its explicit `matrix` (column-major in the file), or composed
    from its translation/rotation/scale."""
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T          # glTF matrices are column-major
    T = np.array(n.get("translation", [0.0, 0.0, 0.0]), float)
    R = np.array(n.get("rotation", [0.0, 0.0, 0.0, 1.0]), float)
    S = np.array(n.get("scale", [1.0, 1.0, 1.0]), float)
    return _compose_trs(T, R, S)


def _gltf_node_graph(gltf):
    """The node hierarchy the skinning deformer needs: per node its name, REST local matrix, and child indices. The
    deformer walks this to turn each joint's animated local transform into a GLOBAL one."""
    out = []
    for i, n in enumerate(gltf.get("nodes", [])):
        out.append({"name": n.get("name", "node_%d" % i),
                    "local": _node_rest_local(n),
                    "children": list(n.get("children", []))})
    return out


def _gltf_morph_targets(gltf, blob):
    """Morph (blend-shape) targets on the first mesh primitive: each target's POSITION deltas, plus the default
    weights. Returns ((T, Nv, 3) deltas or None, (T,) weights or None)."""
    meshes = gltf.get("meshes", [])
    if not meshes:
        return None, None
    prim = meshes[0].get("primitives", [{}])[0]
    targets = prim.get("targets", [])
    if not targets:
        return None, None
    deltas = [_accessor(gltf, tg["POSITION"], blob) for tg in targets if "POSITION" in tg]
    stack = np.stack(deltas) if deltas else None
    w = meshes[0].get("weights") or prim.get("weights")
    return stack, (np.array(w, float) if w else None)


def _gltf_mesh_attribute(gltf, blob, semantic):
    """Read a per-vertex attribute (e.g. 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0') from the FIRST primitive of the first
    mesh, or None if the mesh doesn't carry it."""
    meshes = gltf.get("meshes", [])
    if not meshes:
        return None
    prim = meshes[0].get("primitives", [{}])[0]
    idx = prim.get("attributes", {}).get(semantic)
    return None if idx is None else _accessor(gltf, idx, blob)


def load_glb(path):
    """Load a .glb (binary glTF) into a LoadedMesh: geometry via the engine's glb_to_mesh, its PBR materials + embedded
    textures, its per-vertex UVs/normals, and -- for rigged models -- its ANIMATIONS (keyframed node transforms),
    SKINS (joints + inverse-bind matrices), and per-vertex skin joints/weights. (A .gltf + external files also works if
    you point at the .gltf; geometry then comes from whatever glb_to_mesh can read.)"""
    from holographic_gltf import glb_to_mesh
    base_dir = os.path.dirname(os.path.abspath(path))

    if path.lower().endswith(".glb"):
        data = open(path, "rb").read()
        gltf, binblob = _glb_chunks(data)
        try:
            mesh = glb_to_mesh(data)                            # positions + triangles + uvs/normals (existing)
        except Exception:
            mesh = None                                         # animation-only / unreadable geometry -> still import the rest
    else:                                                       # a .gltf JSON (buffers/images external)
        gltf = json.load(open(path))
        binblob = b""
        mesh = None

    mats = _gltf_materials(gltf, binblob, base_dir)
    materials = {m.name: m for m in mats}
    positions = mesh.vertices if mesh is not None else np.zeros((0, 3))
    faces = mesh.faces if mesh is not None else []
    uv = getattr(mesh, "uvs", None) if mesh is not None else None          # per-vertex TEXCOORD_0
    normals = getattr(mesh, "normals", None) if mesh is not None else None

    # skinning + animation (present only on rigged models)
    joints = _gltf_mesh_attribute(gltf, binblob, "JOINTS_0") if binblob else None
    weights = _gltf_mesh_attribute(gltf, binblob, "WEIGHTS_0") if binblob else None
    animations = _gltf_animations(gltf, binblob) if binblob else []
    skins = _gltf_skins(gltf, binblob) if binblob else []
    nodes = [n.get("name", "node_%d" % i) for i, n in enumerate(gltf.get("nodes", []))]
    node_graph = _gltf_node_graph(gltf)
    morph_targets, morph_weights = (_gltf_morph_targets(gltf, binblob) if binblob else (None, None))

    lm = LoadedMesh(positions, faces, uv=uv, normals=normals, materials=materials,
                    animations=animations, skins=skins, joints=joints, weights=weights, nodes=nodes,
                    node_graph=node_graph, morph_targets=morph_targets, morph_weights=morph_weights)
    lm.face_material = [mats[0].name] * len(lm.faces) if mats else []       # single-material glTF -> all faces use it
    return lm


# =========================================================================================================
# Substance 3D Painter texture SET -- a folder of exported maps -> one PBRMaterial
# =========================================================================================================
# how a channel is recognised from a file name (Painter/glTF/UE/Unity export conventions, all lower-cased)
_CHANNEL_KEYS = {
    "base_color": ("basecolor", "base_color", "albedo", "diffuse", "color", "col"),
    "roughness":  ("roughness", "rough", "rgh"),
    "metallic":   ("metallic", "metalness", "metal", "mtl"),
    "normal":     ("normal", "nrm", "normalgl", "normaldx"),
    "height":     ("height", "displacement", "disp", "bump"),
    "ao":         ("ambientocclusion", "occlusion", "ao"),
    "emissive":   ("emissive", "emission", "emit", "glow"),
}


def _classify_map(filename):
    """Which PBR channel does this file name look like? Longest keyword match wins ('base_color' beats 'col')."""
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    best, best_len = None, 0
    for channel, keys in _CHANNEL_KEYS.items():
        for kw in keys:
            if kw in stem and len(kw) > best_len:
                best, best_len = channel, len(kw)
    return best


def load_texture_set(folder, name=None):
    """Build ONE PBRMaterial from a folder of exported texture maps (Adobe Substance 3D Painter, or any tool that
    writes basecolor/roughness/metallic/normal/height/ao/emissive PNGs). Maps are matched by their file NAMES. The
    numeric factors default sensibly (metallic 1, roughness 1) so the maps drive them. Returns the PBRMaterial;
    unmatched channels stay None. NOTE: this reads the EXPORTED maps -- Painter's .spp/.sbsar project files are
    proprietary and not parsed."""
    from holographic_materialio import PBRMaterial
    IMG_EXT = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".exr")
    found = {}
    for fn in sorted(os.listdir(folder)):
        if os.path.splitext(fn)[1].lower() not in IMG_EXT:
            continue
        ch = _classify_map(fn)
        if ch and ch not in found:                             # first match per channel (sorted -> deterministic)
            found[ch] = os.path.join(folder, fn)

    mat = PBRMaterial(name=name or os.path.basename(os.path.normpath(folder)),
                      metallic=1.0, roughness=1.0)
    mat.base_color_map = _texmap(_image_from_path(found.get("base_color")))
    mat.roughness_map = _texmap(_image_from_path(found.get("roughness")))
    mat.metallic_map = _texmap(_image_from_path(found.get("metallic")))
    mat.emissive_map = _texmap(_image_from_path(found.get("emissive")))
    mat.normal_map = _texmap(_image_from_path(found.get("normal")))     # carried attributes for downstream use
    mat.height_map = _texmap(_image_from_path(found.get("height")))
    mat.ao_map = _texmap(_image_from_path(found.get("ao")))
    mat.channels_found = sorted(found)                          # so you can see what was matched
    return mat


# =========================================================================================================
# Volumetric grids -- a 3-D density array wrapped as a field for the volume renderer
# =========================================================================================================
class GridField:
    """A 3-D density grid presented as a callable field(points(N,3)) -> density, so it drops straight into
    volume_render / mind.render_volume. Samples the grid by TRILINEAR interpolation inside `bounds`; outside is 0.
    `bounds` = (min_corner, max_corner) places the grid in world space."""

    def __init__(self, grid, bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))):
        self.grid = np.asarray(grid, np.float32)               # (nx, ny, nz)
        self.lo = np.asarray(bounds[0], float)
        self.hi = np.asarray(bounds[1], float)
        self.n = np.array(self.grid.shape, float)

    def __call__(self, P):
        P = np.asarray(P, float)
        # world -> grid coordinates in [0, n-1]
        g = (P - self.lo) / np.maximum(self.hi - self.lo, 1e-9) * (self.n - 1)
        inside = np.all((g >= 0) & (g <= self.n - 1), axis=-1)
        out = np.zeros(len(P), np.float32)
        if not np.any(inside):
            return out
        gi = g[inside]
        i0 = np.floor(gi).astype(int)
        i1 = np.minimum(i0 + 1, self.grid.shape - np.array([1, 1, 1]))
        f = gi - i0                                            # trilinear weights
        gx = self.grid
        def s(a, b, c):
            return gx[i0[:, 0] * 0 + [i0, i1][a][:, 0], [i0, i1][b][:, 1], [i0, i1][c][:, 2]]
        c00 = s(0, 0, 0) * (1 - f[:, 0]) + s(1, 0, 0) * f[:, 0]
        c01 = s(0, 0, 1) * (1 - f[:, 0]) + s(1, 0, 1) * f[:, 0]
        c10 = s(0, 1, 0) * (1 - f[:, 0]) + s(1, 1, 0) * f[:, 0]
        c11 = s(0, 1, 1) * (1 - f[:, 0]) + s(1, 1, 1) * f[:, 0]
        c0 = c00 * (1 - f[:, 1]) + c10 * f[:, 1]
        c1 = c01 * (1 - f[:, 1]) + c11 * f[:, 1]
        out[inside] = c0 * (1 - f[:, 2]) + c1 * f[:, 2]
        return out


def load_volume(path, dims=None, dtype="float32", bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))):
    """Load a volumetric density grid into a (GridField, bounds) you can hand to mind.render_volume.
      * .npy               -- a 3-D array, read directly (dims inferred).
      * .raw / .bin / .dat -- flat floats; pass dims=(nx,ny,nz) and dtype.
    Returns (GridField, bounds). NOTE: OpenVDB (.vdb) is a proprietary sparse format and is NOT parsed here -- export
    a dense .npy/.raw grid (or convert with the OpenVDB tools) and load that."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vdb":
        raise ValueError("OpenVDB .vdb is a sparse proprietary format -- export a dense .npy/.raw grid instead "
                         "(see the module docstring).")
    if ext == ".npy":
        grid = np.load(path)
    else:
        if dims is None:
            raise ValueError("raw volume needs dims=(nx,ny,nz)")
        grid = np.fromfile(path, dtype=np.dtype(dtype)).reshape(dims)
    if grid.ndim != 3:
        raise ValueError("expected a 3-D grid, got shape %s" % (grid.shape,))
    return GridField(grid, bounds=bounds), bounds


# =========================================================================================================
# one dispatcher by extension
# =========================================================================================================
def import_asset(path):
    """Import any supported artist file by its extension: .obj -> LoadedMesh (+mtl), .glb/.gltf -> LoadedMesh (+PBR),
    .npy/.raw volume -> (GridField, bounds). For a Substance texture SET, point load_texture_set at the folder."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return load_obj(path)
    if ext in (".glb", ".gltf"):
        return load_glb(path)
    if ext in (".npy", ".raw", ".bin", ".dat", ".vol"):
        return load_volume(path)
    raise ValueError("unsupported extension %r (obj/glb/gltf/npy/raw, or load_texture_set for a Painter folder)" % ext)


def _selftest():
    import tempfile
    import shutil

    root = tempfile.mkdtemp(prefix="lecore_assetimport_")
    try:
        # ---- OBJ + MTL ----
        obj = ("mtllib m.mtl\n"
               "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
               "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
               "vn 0 0 1\n"
               "usemtl red\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")
        mtl = "newmtl red\nKd 0.8 0.1 0.1\nPr 0.4\nPm 0.0\n"
        open(os.path.join(root, "m.obj"), "w").write(obj)
        open(os.path.join(root, "m.mtl"), "w").write(mtl)
        lm = load_obj(os.path.join(root, "m.obj"))
        assert lm.positions.shape == (4, 3) and lm.faces.shape == (2, 3)
        assert lm.uv is not None and lm.uv.shape == (2, 3, 2)
        assert "red" in lm.materials and abs(lm.materials["red"].base_color[0] - 0.8) < 1e-6
        assert lm.face_material == ["red", "red"]
        m = lm.mesh(); assert len(m.vertices) == 4

        # ---- glTF/GLB round trip: emit a box with a material, read it back WITH the material ----
        from holographic_mesh import box
        from holographic_gltf import mesh_to_glb
        from holographic_materialio import PBRMaterial
        blob = mesh_to_glb(box(), material=PBRMaterial(name="steel", base_color=(0.2, 0.3, 0.9, 1.0),
                                                       metallic=1.0, roughness=0.3))
        gp = os.path.join(root, "b.glb"); open(gp, "wb").write(blob)
        glm = load_glb(gp)
        assert len(glm.positions) > 0 and glm.materials
        only = list(glm.materials.values())[0]
        assert abs(only.metallic - 1.0) < 1e-6 and abs(only.base_color[2] - 0.9) < 1e-6

        # ---- texture SET: a folder of named maps -> one PBRMaterial ----
        try:
            from PIL import Image
            tdir = os.path.join(root, "painter"); os.makedirs(tdir)
            for nm, tint in [("brick_basecolor.png", (200, 80, 60)), ("brick_roughness.png", (180, 180, 180)),
                             ("brick_normal.png", (128, 128, 255)), ("brick_metallic.png", (10, 10, 10))]:
                a = np.zeros((8, 8, 3), np.uint8); a[:] = tint; Image.fromarray(a).save(os.path.join(tdir, nm))
            ts = load_texture_set(tdir)
            assert ts.base_color_map is not None and ts.roughness_map is not None
            assert ts.normal_map is not None and ts.metallic_map is not None
            assert set(ts.channels_found) >= {"base_color", "roughness", "normal", "metallic"}
            tset_note = "texture set -> PBRMaterial with 4 maps"
        except ImportError:
            tset_note = "texture set skipped (no PIL in this env)"

        # ---- volume: a density grid -> a field the volume renderer can sample ----
        grid = np.zeros((16, 16, 16), np.float32); grid[6:10, 6:10, 6:10] = 1.0   # a solid cube of density
        np.save(os.path.join(root, "v.npy"), grid)
        gf, bounds = load_volume(os.path.join(root, "v.npy"))
        centre = gf(np.array([[0.0, 0.0, 0.0]]))                # world origin is inside the dense cube
        edge = gf(np.array([[0.95, 0.95, 0.95]]))               # near the corner is empty
        assert centre[0] > 0.5 and edge[0] < 0.5

        print("OK: holographic_assetimport self-test passed (OBJ+MTL: 4v/2f + red material w/ UVs; glTF/GLB round "
              "trips geometry + a steel PBR material; %s; volume .npy -> a trilinear density field the renderer "
              "marches)" % tset_note)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
