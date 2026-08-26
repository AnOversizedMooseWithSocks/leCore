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
    from holographic.materials_and_texture.holographic_materialio import TextureMap
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
        from holographic.mesh_and_geometry.holographic_mesh import Mesh
        return Mesh(self.positions, [tuple(f) for f in self.faces])

    def split_by_material(self):
        """Split this loaded scene into one LoadedMesh PER MATERIAL, each with its faces reindexed to a compact
        vertex set and its UVs/normals subset to match. Returns an ordered dict {material_name: LoadedMesh}.

        WHY (client P2): load_glb merges an entire multi-material scene into ONE mesh, so a consumer that renders
        or LODs per material -- i.e. ANY consumer, because sampling a multi-material scan with a single texture
        paints most faces with the WRONG image (the fishing-spider file) -- had to re-implement face grouping,
        vertex reindexing and UV subsetting every time. face_material already records the per-face material name;
        this makes the correct path the one-call path.

        Reindexing is essential, not cosmetic: a per-material submesh that kept the scene's global vertex indices
        would carry the whole scene's vertex array (huge) and index into positions its own faces never touch. Each
        output remaps to only the vertices its faces use, in first-seen order (deterministic). UVs are handled in
        BOTH conventions this loader produces -- glTF per-vertex (Nv,2) is subset by the vertex remap; OBJ
        per-corner (Nf,3,2) is subset by the FACE selection -- so the split is correct whichever importer ran.
        A face with no material name groups under "" (empty string), never silently dropped."""
        from collections import OrderedDict

        fm = list(self.face_material or [])
        nf = len(self.faces)
        if not fm or len(fm) != nf:
            # No per-face material (or a length mismatch we must not paper over): the whole thing is one group.
            # Returning a single-entry dict keeps the caller's loop correct rather than special-casing None.
            name = fm[0] if fm else ""
            return OrderedDict([(name, self)])

        # group face indices by material name, first-seen order for determinism
        groups = OrderedDict()
        for fi, name in enumerate(fm):
            groups.setdefault(name, []).append(fi)

        uv = self.uv
        uv_per_vertex = uv is not None and np.asarray(uv).ndim == 2      # glTF (Nv,2); OBJ is (Nf,3,2)
        nrm = self.normals
        nrm_per_vertex = nrm is not None and np.asarray(nrm).ndim == 2

        out = OrderedDict()
        for name, fis in groups.items():
            fis = np.asarray(fis, int)
            sub_faces = self.faces[fis]                                  # (k,3) global vertex indices
            used = np.unique(sub_faces)                                 # the vertices this material actually touches
            remap = {int(g): i for i, g in enumerate(used)}            # global -> compact
            new_faces = np.array([[remap[int(v)] for v in f] for f in sub_faces], int)
            new_pos = self.positions[used]

            new_uv = None
            if uv is not None:
                new_uv = np.asarray(uv)[used] if uv_per_vertex else np.asarray(uv)[fis]
            new_nrm = None
            if nrm is not None:
                new_nrm = np.asarray(nrm)[used] if nrm_per_vertex else np.asarray(nrm)[fis]

            mats = {name: self.materials[name]} if name in (self.materials or {}) else {}
            out[name] = LoadedMesh(new_pos, new_faces, uv=new_uv, normals=new_nrm,
                                   face_material=[name] * len(new_faces), materials=mats)
        return out

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
    from holographic.materials_and_texture.holographic_materialio import materials_from_mtl
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
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial

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
    """A quaternion (x, y, z, w) -> a 3x3 rotation matrix (the standard glTF convention). RENORMALISED first.

    WHY THE NORMALISE IS NOT DEFENSIVE PADDING -- it is a measured bug fix: the formula below is exact only for
    a UNIT quaternion, and glTF stores rotations as float32, which are unit only to ~1.7e-8. Feeding the raw
    keyframe in produced a matrix with det 0.99999993 -- not a rotation -- and every posed vertex inherited the
    error: an analytic 90-degree bone swing landed 1.0e-07 off, and the residual traced exactly here. A
    near-unit quaternion is the NORMAL case for imported data, not a malformed one, so the reader must handle it
    rather than assume the file is perfect. Zero-norm (a malformed channel) falls back to identity."""
    q = np.asarray(q, float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
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
    from holographic.agents_and_reasoning.holographic_ai import slerp
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

    def sample(self, t, rest_trs=None):
        """The local transform of every animated node at time t, as {node_index: 4x4}.

        `rest_trs` is {node_index: (T, R, S)} giving each node's OWN transform, used for any path this clip does
        not animate. Pass it whenever you have it -- omitting it is only safe for a fully-animated node.

        WHY IT MATTERS (a silent, whole-skeleton bug): glTF says an animation overrides ONLY the paths it
        targets; every other path keeps the node's own value. This defaulted un-animated paths to
        translation=(0,0,0) / scale=(1,1,1) instead. A rotation-only channel -- the most common thing in a rig,
        since bones rotate about a fixed offset -- therefore collapsed the bone's rest translation to zero and
        pulled the skeleton to the origin. The defaults are kept for the no-rest call so old callers behave
        exactly as before."""
        out = {}
        rest_trs = rest_trs or {}
        for node, paths in self.channels.items():
            rT, rR, rS = rest_trs.get(node, (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), np.ones(3)))
            T = _interp_linear(*paths["translation"], t) if "translation" in paths else np.asarray(rT, float)
            R = _interp_quat(*paths["rotation"], t) if "rotation" in paths else np.asarray(rR, float)
            S = _interp_linear(*paths["scale"], t) if "scale" in paths else np.asarray(rS, float)
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
    """The node hierarchy the skinning deformer needs: per node its name, REST local matrix, its rest TRS, and
    child indices. The deformer walks this to turn each joint's animated local transform into a GLOBAL one.

    `trs` is carried ALONGSIDE `local` because you cannot override one path of a composed matrix: an animation
    that only rotates a bone must keep that bone's rest translation, which is only recoverable from the
    decomposed values. A node given as an explicit `matrix` has no TRS in the file -- glTF forbids animating
    such a node -- so its trs is None and the deformer keeps `local` verbatim."""
    out = []
    for i, n in enumerate(gltf.get("nodes", [])):
        trs = None
        if "matrix" not in n:
            trs = (np.array(n.get("translation", [0.0, 0.0, 0.0]), float),
                   np.array(n.get("rotation", [0.0, 0.0, 0.0, 1.0]), float),
                   np.array(n.get("scale", [1.0, 1.0, 1.0]), float))
        out.append({"name": n.get("name", "node_%d" % i),
                    "local": _node_rest_local(n),
                    "trs": trs,
                    "children": list(n.get("children", []))})
    return out


def _gltf_morph_targets(gltf, blob):
    """Morph (blend-shape) target POSITION deltas for the WHOLE scene, concatenated in gltf.scene_primitives
    order so each target's rows line up 1:1 with glb_to_mesh's vertex table. Returns ((T, Nv, 3) deltas or None,
    (T,) default weights or None).

    THE SAME BUG, A THIRD LAYER DOWN, fixed on the same canonical walk: this read the first primitive's targets
    and returned deltas as long as THAT primitive, so applying them to a multi-mesh scene's vertex table would
    deform the wrong vertices (or IndexError). Deltas are in the primitive's own object space, so they are
    transformed by the node matrix exactly as glb_to_mesh transforms positions -- but as a DELTA, i.e. the
    rotation/scale block only, never the translation (a translation would shift every vertex by the node's
    offset the moment any weight went non-zero).

    T is taken from the widest primitive: a scene can legally give different primitives different target counts,
    and a primitive with fewer (or none) ZERO-FILLS the missing ones -- a zero delta is exactly "this chunk does
    not move for that shape key", which is the truth. KEPT NEGATIVE: only POSITION deltas are read; NORMAL and
    TANGENT deltas are ignored (they exist in the spec and would improve shading on strong morphs), so normals
    on a morphed mesh are the REST normals until something recomputes them."""
    from holographic.io_and_interop.holographic_gltf import scene_primitives
    meshes = gltf.get("meshes", [])
    nodes = gltf.get("nodes", [])
    if not meshes:
        return None, None
    prims = scene_primitives(gltf)
    if not any(meshes[mi]["primitives"][pi].get("targets") for mi, pi, _M, _ni in prims
               if mi < len(meshes) and pi < len(meshes[mi].get("primitives", []))):
        return None, None
    T = max(len(meshes[mi]["primitives"][pi].get("targets", [])) for mi, pi, _M, _ni in prims)
    chunks = [[] for _ in range(T)]
    weights = None
    for mi, pi, M, ni in prims:
        prim = meshes[mi]["primitives"][pi]
        n = _accessor_count(gltf, prim.get("attributes", {}).get("POSITION"))
        targets = prim.get("targets", [])
        R = M[:3, :3]                                          # DELTA transform: no translation, by construction
        for t in range(T):
            tg = targets[t] if t < len(targets) else None
            if tg is None or "POSITION" not in tg:
                chunks[t].append(np.zeros((n, 3)))
                continue
            d = np.asarray(_accessor(gltf, tg["POSITION"], blob), float)
            chunks[t].append(d @ R.T if not np.allclose(R, np.eye(3)) else d)
        if weights is None:
            w = meshes[mi].get("weights") or prim.get("weights")
            if w:
                weights = np.array(w, float)
    stack = np.stack([np.vstack(c) for c in chunks])
    return stack, weights


def _gltf_mesh_attribute(gltf, blob, semantic):
    """Read a per-vertex attribute (e.g. 'JOINTS_0', 'WEIGHTS_0') for the WHOLE scene, concatenated in
    gltf.scene_primitives order so its rows line up 1:1 with glb_to_mesh's vertex table. Primitives that lack
    the attribute contribute ZERO rows of the right width (a common, legal mix: a rigged body chunk beside an
    unrigged prop). Returns None only when NO primitive in the scene carries it.

    WHY THIS IS NOT "the first primitive" ANY MORE -- the bug this fixes was real and silent: it read mesh[0]'s
    first primitive while glb_to_mesh returned the whole scene, so a rigged two-mesh file loaded 16 positions
    against 8 weight rows. Nothing raised; skinning simply indexed the wrong vertices. This is the SAME bug the
    geometry reader had, one layer down -- which is exactly why the traversal now lives in ONE place
    (gltf.scene_primitives) and every payload rides it."""
    from holographic.io_and_interop.holographic_gltf import scene_primitives
    meshes = gltf.get("meshes", [])
    if not meshes:
        return None
    chunks, found = [], False
    for mesh_i, prim_i, _M, _ni in scene_primitives(gltf):
        try:
            prim = meshes[mesh_i]["primitives"][prim_i]
        except (IndexError, KeyError):
            continue
        attrs = prim.get("attributes", {})
        n = _accessor_count(gltf, attrs.get("POSITION"))          # this primitive's vertex count, always known
        idx = attrs.get(semantic)
        if idx is None:
            chunks.append(("zeros", n))                           # placeholder: width isn't known until we see one
        else:
            chunks.append(("data", _accessor(gltf, idx, blob)))
            found = True
    if not found:
        return None
    width = next(c[1].shape[1] for c in chunks if c[0] == "data" and np.asarray(c[1]).ndim == 2)
    dtype = next(np.asarray(c[1]).dtype for c in chunks if c[0] == "data")
    rows = [np.zeros((c[1], width), dtype) if c[0] == "zeros" else np.asarray(c[1]) for c in chunks]
    return np.vstack(rows)


def _scene_chunks(gltf):
    """[(node_index, start, end)] -- the vertex range each scene primitive contributed to glb_to_mesh's table, in
    the canonical scene_primitives order. The record of the walk, kept so later passes can ask "whose vertex is
    this?" without re-deriving the traversal (which is how the traversal got duplicated and drifted before)."""
    from holographic.io_and_interop.holographic_gltf import scene_primitives
    meshes = gltf.get("meshes", [])
    out, cursor = [], 0
    for mesh_i, prim_i, _M, ni in scene_primitives(gltf):
        try:
            prim = meshes[mesh_i]["primitives"][prim_i]
        except (IndexError, KeyError):
            continue
        n = _accessor_count(gltf, prim.get("attributes", {}).get("POSITION"))
        out.append((ni, cursor, cursor + n))
        cursor += n
    return out


def _gltf_skin_binding(gltf, blob):
    """Read the scene's skin binding as ONE consistent table: (joints, weights, joint_nodes) where `joints` and
    `weights` have a row per vertex of glb_to_mesh's vertex table, and every joint index refers to a position in
    `joint_nodes` -- a GLOBAL, scene-wide list of joint node indices.

    WHY A REMAP IS REQUIRED (the trap under the trap): a JOINTS_0 value is NOT a node index. It indexes the
    `joints` array of the SKIN ON THAT PRIMITIVE'S NODE. Two chunks that use different skins therefore both say
    "joint 0" and mean different bones. Concatenating them raw -- which is all the old reader could ever do,
    since it never knew which node it was standing on -- silently welds two skeletons together. Here every
    chunk's local indices are remapped into one global joint list, so index 0 means one bone across the whole
    scene. Single-skin files (the overwhelming majority) remap to themselves, unchanged.

    Returns (None, None, []) when the scene has no skinned primitive."""
    from holographic.io_and_interop.holographic_gltf import scene_primitives
    meshes = gltf.get("meshes", [])
    skins = gltf.get("skins", [])
    nodes = gltf.get("nodes", [])
    if not meshes or not skins:
        return None, None, []

    joint_nodes = []                                          # the global joint list, in first-seen order
    index_of = {}
    for sk in skins:                                          # deterministic: skins array order, not walk order
        for nd in sk.get("joints", []):
            if nd not in index_of:
                index_of[nd] = len(joint_nodes)
                joint_nodes.append(nd)

    j_chunks, w_chunks, found = [], [], False
    for mesh_i, prim_i, _M, ni in scene_primitives(gltf):
        try:
            prim = meshes[mesh_i]["primitives"][prim_i]
        except (IndexError, KeyError):
            continue
        attrs = prim.get("attributes", {})
        n = _accessor_count(gltf, attrs.get("POSITION"))
        skin_i = nodes[ni].get("skin") if (0 <= ni < len(nodes)) else None
        if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs or skin_i is None:
            j_chunks.append(("zeros", n)); w_chunks.append(("zeros", n))
            continue
        local = np.asarray(_accessor(gltf, attrs["JOINTS_0"], blob))
        wts = np.asarray(_accessor(gltf, attrs["WEIGHTS_0"], blob), float)
        lut = np.array([index_of[nd] for nd in skins[skin_i].get("joints", [])], np.int32)
        remapped = lut[np.clip(local.astype(np.int64), 0, max(len(lut) - 1, 0))] if len(lut) else local * 0
        j_chunks.append(("data", remapped.astype(np.int32)))
        w_chunks.append(("data", wts))
        found = True
    if not found:
        return None, None, []
    width = next(np.asarray(c[1]).shape[1] for c in j_chunks if c[0] == "data")
    J = np.vstack([np.zeros((c[1], width), np.int32) if c[0] == "zeros" else c[1] for c in j_chunks])
    W = np.vstack([np.zeros((c[1], width), float) if c[0] == "zeros" else c[1] for c in w_chunks])
    return J, W, joint_nodes


def _accessor_count(gltf, index):
    """The element count of accessor `index` (0 when absent) -- needed to size the zero-fill for a primitive
    that lacks an attribute its siblings carry."""
    if index is None:
        return 0
    return int(gltf["accessors"][index].get("count", 0))


def load_glb(path):
    """Load a .glb (binary glTF) into a LoadedMesh: geometry via the engine's glb_to_mesh, its PBR materials + embedded
    textures, its per-vertex UVs/normals, and -- for rigged models -- its ANIMATIONS (keyframed node transforms),
    SKINS (joints + inverse-bind matrices), and per-vertex skin joints/weights. (A .gltf + external files also works if
    you point at the .gltf; geometry then comes from whatever glb_to_mesh can read.)"""
    from holographic.io_and_interop.holographic_gltf import glb_to_mesh
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

    # skinning + animation (present only on rigged models). The skin binding is read as ONE table so that a
    # joint index means the same bone across every chunk of a multi-mesh scene -- see _gltf_skin_binding.
    joints = weights = None
    joint_nodes = []
    if binblob:
        joints, weights, joint_nodes = _gltf_skin_binding(gltf, binblob)
        if joints is None:                                    # unskinned JOINTS_0 (rare/malformed): raw passthrough
            joints = _gltf_mesh_attribute(gltf, binblob, "JOINTS_0")
            weights = _gltf_mesh_attribute(gltf, binblob, "WEIGHTS_0")
    animations = _gltf_animations(gltf, binblob) if binblob else []
    skins = _gltf_skins(gltf, binblob) if binblob else []
    nodes = [n.get("name", "node_%d" % i) for i, n in enumerate(gltf.get("nodes", []))]
    node_graph = _gltf_node_graph(gltf)
    morph_targets, morph_weights = (_gltf_morph_targets(gltf, binblob) if binblob else (None, None))

    lm = LoadedMesh(positions, faces, uv=uv, normals=normals, materials=materials,
                    animations=animations, skins=skins, joints=joints, weights=weights, nodes=nodes,
                    node_graph=node_graph, morph_targets=morph_targets, morph_weights=morph_weights)
    lm.joint_nodes = joint_nodes          # what a joint INDEX in lm.joints refers to: a node index, scene-wide
    # WHICH NODE each vertex came from, as [(node_index, start, end)] in scene_primitives order. The walk knows
    # this and used to throw it away; the deformer needs it, because glb_to_mesh bakes each chunk's REST node
    # transform into its positions and a skinned chunk must be taken back to its own space before the joint
    # matrices are applied. Cheap to keep, impossible to reconstruct afterwards.
    lm.chunks = _scene_chunks(gltf) if path.lower().endswith(".glb") else []
    # per-face material NAMES from the reader's per-face material INDICES (the whole-scene glb_to_mesh attaches
    # face_material; -1 = no material declared). The old line here assumed a single-material file and stamped
    # mats[0] on every face -- wrong the moment a multi-material scan arrived (crab: pedestal mat 0, body mat 1).
    fm_idx = getattr(mesh, "face_material", None) if mesh is not None else None
    if fm_idx is not None and mats:
        lm.face_material = [mats[i].name if 0 <= i < len(mats) else "" for i in fm_idx]
    else:
        lm.face_material = [mats[0].name] * len(lm.faces) if mats else []
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
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial
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


def _bone_glb():
    """One cube fully bound to ONE bone at rest translation (0,2,0), animated rotation 0 -> 90 degrees about Z.
    Analytic BY CONSTRUCTION: the posed cube must equal Rz(angle) applied about the point (0,2,0), so the test
    compares against maths rather than against a previous run. Test fixture, not a public API."""
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.io_and_interop.holographic_gltf import (mesh_to_glb, _GLB_MAGIC, _GLB_VERSION,
                                                             _CHUNK_JSON, _CHUNK_BIN)
    blob = mesh_to_glb(box())
    jl = struct.unpack("<I", blob[12:16])[0]
    g = json.loads(blob[20:20 + jl])
    binary = bytearray(blob[20 + jl + 8:])
    nv = g["accessors"][g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["count"]

    def add(a, ct, at):
        off = len(binary)
        binary.extend(a.tobytes())
        while len(binary) % 4:
            binary.append(0)
        g["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": a.nbytes})
        g["accessors"].append({"bufferView": len(g["bufferViews"]) - 1, "componentType": ct,
                               "count": len(a), "type": at})
        return len(g["accessors"]) - 1

    J = np.zeros((nv, 4), np.uint8)
    W = np.zeros((nv, 4), np.float32); W[:, 0] = 1.0
    ja, wa = add(J, 5121, "VEC4"), add(W, 5126, "VEC4")
    g["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"] = ja
    g["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"] = wa
    ibm = np.eye(4)[None, :, :].copy(); ibm[0][:3, 3] = [0, -2, 0]     # inverse of the bone's rest translation
    ia = add(ibm.transpose(0, 2, 1).reshape(1, 16).astype(np.float32), 5126, "MAT4")
    h = np.float32(np.sin(np.pi / 4))
    ta = add(np.array([0.0, 1.0], np.float32), 5126, "SCALAR")
    qa = add(np.array([[0, 0, 0, 1], [0, 0, h, h]], np.float32), 5126, "VEC4")
    g["nodes"] = [{"mesh": 0, "skin": 0}, {"name": "bone", "translation": [0.0, 2.0, 0.0]}]
    g["skins"] = [{"joints": [1], "inverseBindMatrices": ia}]
    g["animations"] = [{"name": "swing",
                        "samplers": [{"input": ta, "output": qa, "interpolation": "LINEAR"}],
                        "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}]}]
    g["scenes"] = [{"nodes": [0, 1]}]; g["scene"] = 0
    g["buffers"][0]["byteLength"] = len(binary)
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, 12 + 8 + len(js) + 8 + len(binary))
    out += struct.pack("<II", len(js), _CHUNK_JSON) + js
    out += struct.pack("<II", len(binary), _CHUNK_BIN) + bytes(binary)
    return bytes(out)


def _selftest_pose():
    """Pin the deformer against ANALYTIC truth, not a golden run: a cube bound to a bone at (0,2,0) swung 90
    degrees about Z must land exactly on Rz(90) about that point. Also pins the two silent traps this exposed --
    an un-animated path must keep its REST value, and a float32 quaternion must be renormalised before it
    becomes a matrix."""
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".glb")
    os.write(fd, _bone_glb()); os.close(fd)
    try:
        lm = load_glb(path)
        rest = np.asarray(lm.positions, float)
        m0, r0 = pose_asset(lm, time=0.0)
        m1, r1 = pose_asset(lm, time=1.0)
        assert r1["mode"] == "animated" and r1["joints"] == 1 and r1["skinned_vertices"] == len(rest)
        assert np.abs(np.asarray(m0.vertices) - rest).max() < 1e-9, "t=0 must reproduce the bind pose"
        o = np.array([0.0, 2.0, 0.0])
        Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        expect = (rest - o) @ Rz.T + o
        err = np.abs(np.asarray(m1.vertices) - expect).max()
        assert err < 1e-9, "posed cube is %.2e off the analytic 90-degree swing" % err
        # the REST trap: the bone is rotation-only, so its rest translation (0,2,0) must survive sampling
        loc = lm.animations[0].sample(1.0, rest_trs={1: lm.node_graph[1]["trs"]})[1]
        assert np.allclose(loc[:3, 3], [0, 2, 0]), "an un-animated path must keep the node's REST value"
        assert np.allclose(lm.animations[0].sample(1.0)[1][:3, 3], [0, 0, 0])   # no rest given -> old default
        # halfway is a real slerp, not a snap to either end
        mh, _ = pose_asset(lm, time=0.5)
        Ph = np.asarray(mh.vertices)
        assert np.abs(Ph - rest).max() > 1e-3 and np.abs(Ph - expect).max() > 1e-3
    finally:
        os.unlink(path)
    print("pose selftest OK (bind pose exact; 90-degree swing matches analytic Rz to %.1e; rest-aware sampling; "
          "slerp midpoint distinct from both ends)" % err)


def _morph_glb():
    """A TWO-mesh .glb where only the FIRST mesh has a morph target, and the second node is translated. Pins
    both halves: deltas must span the whole vertex table (zero-filled for the chunk that has none), and a delta
    must NOT pick up the node's translation. Test fixture, not a public API."""
    import copy
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.io_and_interop.holographic_gltf import (mesh_to_glb, _GLB_MAGIC, _GLB_VERSION,
                                                             _CHUNK_JSON, _CHUNK_BIN)
    blob = mesh_to_glb(box())
    jlen = struct.unpack("<I", blob[12:16])[0]
    g = json.loads(blob[20:20 + jlen])
    binary = bytearray(blob[20 + jlen + 8:])
    nv = g["accessors"][g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["count"]
    d = np.zeros((nv, 3), np.float32); d[:, 1] = 0.5                      # every vertex rises by 0.5 in Y
    off = len(binary); binary.extend(d.tobytes())
    while len(binary) % 4:
        binary.append(0)
    g["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": d.nbytes})
    g["accessors"].append({"bufferView": len(g["bufferViews"]) - 1, "componentType": 5126,
                           "count": nv, "type": "VEC3"})
    da = len(g["accessors"]) - 1
    g["meshes"].append(copy.deepcopy(g["meshes"][0]))
    g["meshes"][0]["primitives"][0]["targets"] = [{"POSITION": da}]
    g["meshes"][0]["weights"] = [0.0]
    g["nodes"] = [{"mesh": 0}, {"mesh": 1, "translation": [3.0, 0.0, 0.0]}]
    g["scenes"] = [{"nodes": [0, 1]}]; g["scene"] = 0
    g["buffers"][0]["byteLength"] = len(binary)
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, 12 + 8 + len(js) + 8 + len(binary))
    out += struct.pack("<II", len(js), _CHUNK_JSON) + js
    out += struct.pack("<II", len(binary), _CHUNK_BIN) + bytes(binary)
    return bytes(out)


def _selftest_morph_multimesh():
    """Pin morph targets across a multi-mesh scene: deltas span the WHOLE vertex table, the chunk without a
    target zero-fills in place, and a delta carries the node's rotation/scale but NEVER its translation."""
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".glb")
    os.write(fd, _morph_glb()); os.close(fd)
    try:
        lm = load_glb(path)
        D = np.asarray(lm.morph_targets)
        n = len(lm.positions) // 2
        assert D.shape == (1, len(lm.positions), 3), "deltas must span the scene vertex table, got %s" % (D.shape,)
        assert np.allclose(D[0, :n, 1], 0.5), "the morphed chunk's deltas must survive"
        assert np.allclose(D[0, n:], 0.0), "the chunk without a target must ZERO-FILL, not shift rows"
        assert np.allclose(D[0, :, 0], 0.0), "a DELTA must never pick up the node's 3.0 translation"
        assert np.allclose(np.asarray(lm.morph_weights), [0.0])
    finally:
        os.unlink(path)
    print("morph multimesh selftest OK (deltas span the scene table; unmorphed chunk zero-fills; delta carries "
          "no node translation)")


def _rigged_glb(two_skins=False):
    """Build a rigged TWO-mesh .glb in memory: chunk A rigged, chunk B either unrigged (zero-fill case) or on a
    SECOND skin whose joint 0 is a different bone (the index-collision case). Test fixture, not a public API."""
    import copy
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.io_and_interop.holographic_gltf import (mesh_to_glb, _GLB_MAGIC, _GLB_VERSION,
                                                             _CHUNK_JSON, _CHUNK_BIN)
    blob = mesh_to_glb(box())
    jlen = struct.unpack("<I", blob[12:16])[0]
    g = json.loads(blob[20:20 + jlen])
    binary = bytearray(blob[20 + jlen + 8:])
    nv = g["accessors"][g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]["count"]

    def add(arr, ctype, atype):
        off = len(binary)
        binary.extend(arr.tobytes())
        while len(binary) % 4:
            binary.append(0)
        g["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
        g["accessors"].append({"bufferView": len(g["bufferViews"]) - 1, "componentType": ctype,
                               "count": len(arr), "type": atype})
        return len(g["accessors"]) - 1

    J = np.zeros((nv, 4), np.uint8)
    W = np.zeros((nv, 4), np.float32); W[:, 0] = 1.0
    ja, wa = add(J, 5121, "VEC4"), add(W, 5126, "VEC4")
    g["meshes"].append(copy.deepcopy(g["meshes"][0]))
    g["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"] = ja
    g["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"] = wa
    if two_skins:
        g["meshes"][1]["primitives"][0]["attributes"]["JOINTS_0"] = ja
        g["meshes"][1]["primitives"][0]["attributes"]["WEIGHTS_0"] = wa
        g["nodes"] = [{"mesh": 0, "skin": 0}, {"mesh": 1, "skin": 1, "translation": [3.0, 0, 0]},
                      {"name": "boneA"}, {"name": "boneB"}]
        g["skins"] = [{"joints": [2]}, {"joints": [3]}]
        g["scenes"] = [{"nodes": [0, 1, 2, 3]}]
    else:
        g["nodes"] = [{"mesh": 0, "skin": 0}, {"mesh": 1, "translation": [3.0, 0, 0]}, {"name": "boneA"}]
        g["skins"] = [{"joints": [2]}]
        g["scenes"] = [{"nodes": [0, 1, 2]}]
    g["scene"] = 0
    g["buffers"][0]["byteLength"] = len(binary)
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, 12 + 8 + len(js) + 8 + len(binary))
    out += struct.pack("<II", len(js), _CHUNK_JSON) + js
    out += struct.pack("<II", len(binary), _CHUNK_BIN) + bytes(binary)
    return bytes(out)


def _selftest_rigged_multimesh():
    """Pin the skin binding of a MULTI-MESH rigged scene -- two bugs of the same family, both silent:
    (1) JOINTS/WEIGHTS covered only the first primitive while positions covered the whole scene (16 vs 8 rows);
    (2) joint INDICES are per-skin, so two chunks both saying 'joint 0' meant different bones."""
    import tempfile, os
    for two_skins in (False, True):
        fd, path = tempfile.mkstemp(suffix=".glb")
        os.write(fd, _rigged_glb(two_skins=two_skins)); os.close(fd)
        try:
            lm = load_glb(path)
            J = np.asarray(lm.joints); W = np.asarray(lm.weights)
            n = len(lm.positions) // 2
            assert len(J) == len(lm.positions), "joints must have a row per scene vertex, got %d vs %d" % (
                len(J), len(lm.positions))
            assert len(W) == len(lm.positions)
            assert np.allclose(W[:n, 0], 1.0), "the rigged chunk's weights must survive the concat"
            if two_skins:
                assert J[0, 0] != J[n, 0], "two skins both said 'joint 0' -- the remap must separate them"
                assert lm.joint_nodes[J[0, 0]] == 2 and lm.joint_nodes[J[n, 0]] == 3
            else:
                assert (J[n:] == 0).all() and np.allclose(W[n:], 0.0), "unrigged chunk must ZERO-FILL in place"
                assert lm.joint_nodes == [2]
        finally:
            os.unlink(path)
    print("rigged multimesh selftest OK (rows align with the scene vertex table; unrigged chunk zero-fills; "
          "two skins' colliding 'joint 0' remapped to distinct global bones)")


def _world_transforms(node_graph, locals_by_node):
    """Compose every node's LOCAL transform down the hierarchy into a WORLD transform. Roots are the nodes that
    are nobody's child. Returns {node_index: 4x4}. Cycles (malformed files) are broken by a visited set rather
    than recursing forever."""
    child_of = set()
    for n in node_graph:
        child_of.update(n["children"])
    world = {}
    seen = set()

    def walk(i, acc):
        if i in seen:
            return                                            # malformed graph: refuse to loop
        seen.add(i)
        M = acc @ locals_by_node.get(i, node_graph[i]["local"])
        world[i] = M
        for c in node_graph[i]["children"]:
            if 0 <= c < len(node_graph):
                walk(c, M)

    for i in range(len(node_graph)):
        if i not in child_of:
            walk(i, np.eye(4))
    for i in range(len(node_graph)):                           # orphaned by a cycle: still give it something
        if i not in world:
            world[i] = locals_by_node.get(i, node_graph[i]["local"])
    return world


def pose_asset(lm, time=0.0, clip=0):
    """POSE a rigged asset at `time` -- the composition that turns an imported rig into moving geometry, and the
    last missing link of the glTF import chain. Returns (Mesh, report).

    Every piece of this already existed and none of them were connected: load_glb reads the animation channels,
    AnimationClip.sample interpolates them, _gltf_node_graph carries the hierarchy (its docstring literally says
    "the deformer walks this"), skins carry the inverse-bind matrices, and meshskin does linear blend skinning.
    Nothing composed them, so a rigged .glb imported and then sat in its bind pose forever.

    The chain, in the order the glTF spec requires:
      1. sample the clip at `time` -> LOCAL matrices for animated nodes, with un-animated paths keeping the
         node's REST value (the rotation-only-bone trap);
      2. compose the hierarchy -> WORLD matrix per node, at this time and at rest;
      3. joint matrix for global joint k = world_anim(joint_node) @ inverse_bind(joint_node);
      4. take each chunk's vertices back out of the REST node transform glb_to_mesh baked into them (the spec
         ignores a skinned mesh node's own transform; our positions already have it applied, so it must be
         undone or it is counted twice), then blend by the vertex's joints/weights.

    Falls back gracefully and says so in the report: no animation -> the bind pose; no skin -> rigid node
    animation still moves whole chunks. KEPT NEGATIVE: linear blend skinning, so the classic candy-wrapper
    collapse on a 180-degree twist is present and NOT fixed (dual-quaternion skinning is the known answer);
    morph target weights are read but not applied here -- pose_asset does bones, not shape keys."""
    from holographic.mesh_and_geometry.holographic_meshskin import linear_blend_skin_indexed
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.array(lm.positions, float)
    graph = list(getattr(lm, "node_graph", []) or [])
    clips = list(getattr(lm, "animations", []) or [])
    report = {"time": float(time), "vertices": len(V), "clip": None, "duration": 0.0,
              "animated_nodes": 0, "joints": 0, "skinned_vertices": 0, "mode": "bind_pose"}
    if not graph:
        return Mesh(V, [tuple(f) for f in lm.faces], uvs=getattr(lm, "uv", None)), report

    # 1. the animated local transforms at `time` (rest-aware)
    locals_anim = {}
    if clips and 0 <= clip < len(clips):
        c = clips[clip]
        rest_trs = {i: g["trs"] for i, g in enumerate(graph) if g.get("trs") is not None}
        locals_anim = c.sample(float(time), rest_trs=rest_trs)
        report.update({"clip": c.name, "duration": float(c.duration), "animated_nodes": len(locals_anim),
                       "mode": "animated"})

    # 2. world transforms, posed and at rest
    world = _world_transforms(graph, locals_anim)
    world_rest = _world_transforms(graph, {})

    # 3. joint matrices, in the GLOBAL joint index space lm.joints already speaks
    joint_nodes = list(getattr(lm, "joint_nodes", []) or [])
    ibm_of = {}
    for sk in getattr(lm, "skins", []) or []:
        ibm = sk.get("inverse_bind")
        for j, nd in enumerate(sk.get("joints", [])):
            if nd not in ibm_of:                              # first skin to define a joint wins (deterministic)
                ibm_of[nd] = ibm[j] if ibm is not None and j < len(ibm) else np.eye(4)
    JM = np.stack([world.get(nd, np.eye(4)) @ ibm_of.get(nd, np.eye(4)) for nd in joint_nodes])         if joint_nodes else np.zeros((0, 4, 4))
    report["joints"] = len(joint_nodes)

    joints = getattr(lm, "joints", None)
    weights = getattr(lm, "weights", None)
    out = V.copy()
    chunks = list(getattr(lm, "chunks", []) or []) or [(-1, 0, len(V))]
    for ni, s, e in chunks:
        if e <= s:
            continue
        # 4a. undo the REST node transform glb_to_mesh baked in -> this chunk's own object space
        Wr = world_rest.get(ni, np.eye(4))
        inv = np.linalg.inv(Wr) if not np.allclose(Wr, np.eye(4)) else np.eye(4)
        local_pts = V[s:e] @ inv[:3, :3].T + inv[:3, 3]
        claimed = (joints is not None and weights is not None and len(joint_nodes)
                   and np.asarray(weights)[s:e].sum() > 1e-12)
        if claimed:
            # 4b. skinned: blend the joint matrices (they already carry the chunk back to world space)
            out[s:e] = linear_blend_skin_indexed(local_pts, JM, np.asarray(joints)[s:e],
                                                 np.asarray(weights, float)[s:e])
            report["skinned_vertices"] += int(e - s)
        else:
            # 4c. unskinned chunk: it still rides its node's ANIMATED transform (a prop on a moving arm)
            Wa = world.get(ni, Wr)
            out[s:e] = local_pts @ Wa[:3, :3].T + Wa[:3, 3]
    if report["skinned_vertices"] and report["mode"] == "bind_pose":
        report["mode"] = "bind_pose_skinned"
    return Mesh(out, [tuple(f) for f in lm.faces], uvs=getattr(lm, "uv", None)), report


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
        from holographic.mesh_and_geometry.holographic_mesh import box
        from holographic.io_and_interop.holographic_gltf import mesh_to_glb
        from holographic.materials_and_texture.holographic_materialio import PBRMaterial
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


def asset_base_texture(loaded_mesh):
    """Return the render-ready (texture, uvs, base_color) for a LOADED mesh -- the pointer from an imported (or
    self-derived) mesh to a TEXTURED render_mesh call, without going back through a file path.

    WHY THIS EXISTS: preview_asset renders a textured image straight from a PATH, but a mesh you DECIMATED or
    RETOPOLOGISED yourself has no path -- and getting from a LoadedMesh's materials to the (texture, uvs) that
    render_mesh wants took manual buffer digging (a whole render arc hand-extracted the embedded JPEG because
    nothing surfaced this). This is that pointer, and it is the SAME material-pick preview_asset uses (chosen by
    face COVERAGE so a multi-material scan renders in the skin most of its surface wears, 8-bit normalised to
    [0,1]), factored out so both callers share one code path.

    Returns (texture (H,W,3) float in [0,1] or None, uvs (n,2) float or None, base_color (3,) fallback). Feed
    the pair straight to render_mesh(mesh, cam, texture=texture, uvs=uvs); if texture is None the base_color is
    the flat fallback. uvs come from loaded_mesh.uv (per-vertex TEXCOORD_0)."""
    import numpy as _np
    from collections import Counter as _Counter
    uv = getattr(loaded_mesh, "uv", None)
    uv = _np.asarray(uv, float) if uv is not None else None
    tex = None
    base = (0.8, 0.8, 0.8)
    fm = list(getattr(loaded_mesh, "face_material", []) or [])
    counts = _Counter(n for n in fm if n)
    ordered = ([loaded_mesh.materials[n] for n, _ in counts.most_common() if n in loaded_mesh.materials]
               or list(getattr(loaded_mesh, "materials", {}).values()))
    for mat in ordered:
        bcm = getattr(mat, "base_color_map", None)
        if bcm is not None and getattr(bcm, "image", None) is not None:
            tex = _np.asarray(bcm.image, float)
            if tex.max() > 1.5:
                tex = tex / 255.0                            # 8-bit -> [0,1]; render_mesh wants floats
            break
        base = tuple(mat.base_color[:3])
    return tex, uv, base


def preview_asset(path, camera=None, width=512, height=384, ambient=0.5, smooth=True, eye_dir=(0.55, 0.35, 0.7), fit=False):
    """ONE-CALL textured preview of an asset file: import (with materials + embedded textures), auto-frame, and
    rasterize with the base-colour texture applied. Returns (image (H,W,3) float, LoadedMesh).

    WHY THIS EXISTS: every piece already existed -- load_glb extracts the embedded texture, render_mesh renders
    textured -- but showing an imported model WITH its texture took five composition steps (LoadedMesh -> Mesh,
    attach uv, normalise the texture to [0,1], frame a camera, pass texture+uvs), two of them non-obvious. A
    whole debugging arc rendered with a synthetic checker because nothing pointed from the import to the
    textured render. This is that pointer. `camera=None` auto-frames from the mesh bounds along `eye_dir`; `fit=True` uses fit_camera for an exact aspect-aware fit (measured ~4x the frame coverage on a small-bbox asset) instead of the bbox-diagonal heuristic.

    Uses the FIRST material carrying a base_color_map (multi-material assets render with that one map -- the
    per-face material split is the importer's existing report, not re-solved here). No map -> flat base_color.
    """
    import lecore
    lm = import_asset(path)
    if isinstance(lm, tuple):
        raise ValueError("preview_asset previews meshes; %r imported as a volume" % (path,))
    mesh = lm.mesh()
    uv = getattr(mesh, "uvs", None)
    if uv is None and getattr(lm, "uv", None) is not None:
        uv = np.asarray(lm.uv, float)
        if uv.ndim == 2 and len(uv) == len(mesh.vertices):
            mesh.uvs = uv                                    # per-vertex uvs (the glb path)
        else:
            uv = None                                        # per-corner OBJ uvs need a split; flat render then
    # SHARED with asset_base_texture: pick the base-colour map by FACE COVERAGE (a multi-material crab scan
    # previews in the skin most of its surface wears), 8-bit normalised. Factored out so a self-derived mesh
    # (decimated/retopo'd, no path) can get the same (texture, uvs) for a textured render_mesh call.
    tex, _uv_unused, base = asset_base_texture(lm)
    from collections import Counter
    fm = list(getattr(lm, "face_material", []) or [])
    counts = Counter(n for n in fm if n)
    ordered = [lm.materials[n] for n, _ in counts.most_common() if n in lm.materials] or               list(getattr(lm, "materials", {}).values())
    chosen_name = None
    for mat in ordered:                                     # recover which material won (for the face subset below)
        bcm = getattr(mat, "base_color_map", None)
        if bcm is not None and getattr(bcm, "image", None) is not None:
            chosen_name = mat.name
            break
    if chosen_name is not None and fm and len(fm) == len(mesh.faces) and len(set(fm)) > 1:
        keep = [i for i, n in enumerate(fm) if n == chosen_name]
        if 0 < len(keep) < len(mesh.faces):
            from holographic.mesh_and_geometry.holographic_mesh import Mesh as _Mesh
            sub = _Mesh(mesh.vertices, [mesh.faces[i] for i in keep],
                        normals=getattr(mesh, "normals", None), uvs=getattr(mesh, "uvs", None))
            mesh = sub                                       # unreferenced verts are harmless for a raster pass
    m = lecore.UnifiedMind(dim=64, seed=0)                   # smallest mind: this is a raster call, not VSA work
    V = np.asarray(mesh.vertices, float)
    if camera is None:
        c = V.mean(0)
        if fit:
            # fit_camera centres on the PROJECTED bbox and fits it to the frame exactly, aspect-aware.
            # MEASURED on a small-bbox asset (crab, 0.086u): the heuristic below fills 36%x22% of frame;
            # fit_camera fills 75%x46% -- ~4x the pixel coverage. Opt-in (fit=False default) so the historical
            # framing of every existing caller is untouched; a rescale of the eye distance would flip renders.
            from holographic.rendering.holographic_render import fit_camera as _fit
            camera = _fit(mesh, direction=eye_dir, fov_deg=50.0, aspect=float(width) / float(height))
        else:
            diag = float(np.linalg.norm(V.max(0) - V.min(0))) or 1.0
            camera = {"eye": (c + np.asarray(eye_dir, float) * diag).tolist(), "target": c.tolist()}
    kw = dict(texture=tex, uvs=np.asarray(mesh.uvs, float)) if (tex is not None and uv is not None) \
        else dict(base_color=base)
    img = m.render_mesh(mesh, camera, width=width, height=height, ambient=ambient, smooth=smooth, **kw)
    return np.asarray(img), lm


def _selftest_asset_base_texture():
    """Pin: asset_base_texture returns a render-ready texture from a mesh carrying a base_color_map, and the
    same texture preview_asset would pick -- so a self-derived (decimated/retopo'd) mesh can be textured
    without a path. Built on a synthetic two-colour textured quad."""
    import tempfile, os
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.io_and_interop.holographic_gltf import mesh_to_glb
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial, TextureMap
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    F = [(0, 1, 2), (0, 2, 3)]
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    teximg = np.zeros((2, 2, 3), np.float32); teximg[:, 0] = (1, 0, 0); teximg[:, 1] = (0, 0, 1)
    mesh = Mesh(V, F, uvs=uv)
    mat = PBRMaterial(name="t", base_color_map=TextureMap(teximg))
    glb = mesh_to_glb(mesh, material=mat, texture=teximg)   # writer needs the explicit texture arg (as _selftest_preview does)
    pth = tempfile.mktemp(suffix=".glb"); open(pth, "wb").write(glb)
    try:
        lm = load_glb(pth)
        tex, u, base = asset_base_texture(lm)
        assert tex is not None, "a mesh with a base_color_map must yield a texture"
        assert tex.shape[2] == 3 and 0.0 <= tex.min() and tex.max() <= 1.0, "texture must be [0,1] RGB"
        assert u is not None and u.shape[1] == 2, "uvs must come back for a textured render"
        assert len(base) == 3, "base_color fallback is a 3-tuple"
        print("asset_base_texture selftest OK (render-ready texture %s + uvs %s from a loaded mesh)"
              % (tex.shape, u.shape))
    finally:
        os.remove(pth)


def _selftest_preview():
    """Pin: a .glb with an embedded texture previews TEXTURED (colour variation on the surface), and the uvs
    used are the asset's own. Built on a synthetic two-colour textured quad so the assertion is exact-ish."""
    import tempfile
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.io_and_interop.holographic_gltf import mesh_to_glb
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial, TextureMap
    # a unit quad with uvs spanning the texture; texture: left half red, right half blue
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    F = [(0, 1, 2), (0, 2, 3)]
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    teximg = np.zeros((8, 8, 3)); teximg[:, :4] = [1, 0, 0]; teximg[:, 4:] = [0, 0, 1]
    mat = PBRMaterial(name="two", base_color_map=TextureMap(teximg))
    quad = Mesh(V, F, uvs=uv)
    blob = mesh_to_glb(quad, material=mat, texture=teximg)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "q.glb")
        open(p, "wb").write(blob)
        img, lm = preview_asset(p, camera={"eye": [0.5, 0.5, 2.0], "target": [0.5, 0.5, 0.0]},
                                width=64, height=64, ambient=1.0, smooth=False)
    fg = img[img.sum(axis=2) > 0.2]
    assert len(fg) > 100
    red = (fg[:, 0] > 0.4) & (fg[:, 2] < 0.3)
    blue = (fg[:, 2] > 0.4) & (fg[:, 0] < 0.3)
    assert red.sum() > 20 and blue.sum() > 20, (red.sum(), blue.sum())   # BOTH halves visible = uvs really used
    print("preview_asset selftest OK (embedded texture round-trips: %d red + %d blue px on the surface)"
          % (red.sum(), blue.sum()))


if __name__ == "__main__":
    _selftest_preview(); _selftest_asset_base_texture(); _selftest_rigged_multimesh(); _selftest_morph_multimesh(); _selftest_pose()   # the first guard above already ran _selftest(); module-as-script runs both in order
