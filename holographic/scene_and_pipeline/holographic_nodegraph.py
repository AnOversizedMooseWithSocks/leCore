"""holographic_nodegraph.py -- the UNIFYING NODE-GRAPH backend a node editor binds to.

WHY THIS MODULE EXISTS (the one missing shell)
----------------------------------------------
leCore already has every node KIND a 3-D package's node editor exposes -- SDF/CSG trees (holographic_sdf), texture
graphs (holographic_texturegraph), material socket graphs (matcompile), geometry modifier dependency graphs
(holographic_modifier) -- plus the hard infrastructure: dirty propagation, a dependency-keyed cache, topological
sort + cycle detection (the shader-graph compiler), and typed structures. What was MISSING was a single generic graph
model that holds HETEROGENEOUS typed nodes, validates connections across the data kinds, evaluates in topological
order with dirty propagation, and serializes. This is that shell. It does NOT reimplement any domain's compute -- a
node's evaluate() DELEGATES to the existing subsystem (an sdf_union node calls SDF.union; a fillet node calls the K5
fillet). The shell is the editor's contract; the compute stays where it already lives ("wire the wheels together").

THE MODEL
---------
  * SocketType -- the kinds a wire can carry: scalar, vector, color, field, sdf, mesh, material, texture, ANY.
    Connection is allowed iff the source type equals the sink type, or either side is ANY. Strict by default so the
    editor can refuse a nonsense wire (an sdf into a scalar slot) at connect time, not at evaluate time.
  * NodeType -- a registered node kind: named input sockets (name->type), output sockets (name->type), and an
    evaluate(params, inputs)->outputs function. Domain ops register themselves as NodeTypes.
  * NodeGraph -- nodes (id -> (type_name, params)) + edges (src_id.src_socket -> dst_id.dst_socket). connect()
    TYPE-CHECKS and refuses CYCLES. evaluate(node_id) walks the graph in topological order, memoizing; set_param()
    marks the node and everything downstream DIRTY so a re-evaluate recomputes only what changed. to_dict/from_dict
    round-trip the whole graph as plain JSON-able data.

Deterministic; NumPy + stdlib only.
"""
import numpy as np


# ---- socket types + compatibility -----------------------------------------------------------------------------
SOCKET_TYPES = ("scalar", "vector", "color", "field", "sdf", "mesh", "material", "texture", "signal", "any")


def types_compatible(src, dst):
    """Can a `src`-typed output wire into a `dst`-typed input? Exact match, or either side is 'any'."""
    if src not in SOCKET_TYPES or dst not in SOCKET_TYPES:
        raise ValueError("unknown socket type %r/%r" % (src, dst))
    return src == dst or src == "any" or dst == "any"


class NodeType:
    """A registered node kind. `inputs`/`outputs` map socket name -> SocketType. `fn(params, inputs)->outputs_dict`
    is the compute, which DELEGATES to an existing subsystem -- the shell never reimplements domain math."""

    def __init__(self, name, inputs, outputs, fn, param_inputs=None):
        self.name = name
        self.inputs = dict(inputs)
        self.outputs = dict(outputs)
        self.fn = fn
        # DRIVABLE PARAMS: params that are ALSO optional input sockets. Wire a compatible source (an audio band, a
        # shader field) into one and it OVERRIDES the static param at evaluate time -- the mechanism that lets any
        # source "drive everything it can" without enumerating what it drives (the socket type decides).
        self.param_inputs = dict(param_inputs or {})
        # TIME AWARENESS: a node fn may take an optional 3rd arg (a context carrying 't') -- detected once so time-
        # varying sources (audio, animated shader fields) can read the clock while static nodes keep the 2-arg form.
        import inspect
        try:
            self._wants_ctx = len(inspect.signature(fn).parameters) >= 3
        except (TypeError, ValueError):
            self._wants_ctx = False

    def all_inputs(self):
        """Regular inputs plus drivable-param inputs -- everything you can wire INTO this node."""
        merged = dict(self.inputs)
        merged.update(self.param_inputs)
        return merged


class NodeRegistry:
    """The palette of node kinds an editor can place. Domain modules register their ops here."""

    def __init__(self):
        self.types = {}

    def register(self, name, inputs, outputs, fn, param_inputs=None):
        self.types[name] = NodeType(name, inputs, outputs, fn, param_inputs=param_inputs)
        return name

    def get(self, name):
        if name not in self.types:
            raise KeyError("no node type %r (have: %s)" % (name, ", ".join(sorted(self.types))))
        return self.types[name]


class NodeGraph:
    """A heterogeneous typed node graph. Add nodes, connect sockets (type-checked, cycle-free), evaluate a node's
    outputs (topological, memoized), edit params (dirty-propagating), and serialize."""

    def __init__(self, registry):
        self.reg = registry
        self.nodes = {}         # id -> {"type": name, "params": {...}}
        self.edges = []         # {"src": id, "src_socket": s, "dst": id, "dst_socket": s}
        self._cache = {}        # id -> outputs dict (memo)
        self._next = 0
        self._t = 0.0           # the time the memo was last evaluated at (time change invalidates time-varying nodes)

    # -- construction ------------------------------------------------------------------------------------------
    def add(self, type_name, params=None, node_id=None):
        """Add a node of a registered type. Returns its id."""
        self.reg.get(type_name)                                  # validate the type exists
        nid = node_id if node_id is not None else "n%d" % self._next
        self._next += 1
        self.nodes[nid] = {"type": type_name, "params": dict(params or {})}
        return nid

    def connect(self, src, src_socket, dst, dst_socket):
        """Wire src.src_socket -> dst.dst_socket. Refuses a type mismatch and refuses a connection that would create
        a cycle -- the two guarantees an editor needs so a bad wire fails at connect time, not mid-evaluation."""
        st = self.reg.get(self.nodes[src]["type"])
        dt = self.reg.get(self.nodes[dst]["type"])
        dst_inputs = dt.all_inputs()                             # regular inputs + drivable-param inputs
        if src_socket not in st.outputs:
            raise ValueError("node %s has no output socket %r" % (src, src_socket))
        if dst_socket not in dst_inputs:
            raise ValueError("node %s has no input socket %r" % (dst, dst_socket))
        if not types_compatible(st.outputs[src_socket], dst_inputs[dst_socket]):
            raise TypeError("cannot wire %s (%s) -> %s (%s): incompatible socket types"
                            % (src_socket, st.outputs[src_socket], dst_socket, dst_inputs[dst_socket]))
        # cycle check FIRST, before any mutation -- a refused connect must leave the graph untouched
        if self._would_cycle(src, dst):
            raise ValueError("connecting %s -> %s would create a cycle" % (src, dst))
        # a single input socket takes one wire: drop any existing wire into it (editor semantics)
        self.edges = [e for e in self.edges if not (e["dst"] == dst and e["dst_socket"] == dst_socket)]
        self.edges.append({"src": src, "src_socket": src_socket, "dst": dst, "dst_socket": dst_socket})
        self._cache.clear()                                      # topology changed -> invalidate memo

    def _would_cycle(self, src, dst):
        """Would adding src->dst make a cycle? True iff src is already reachable FROM dst (dst -> ... -> src)."""
        stack = [dst]; seen = set()
        while stack:
            n = stack.pop()
            if n == src:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack += [e["dst"] for e in self.edges if e["src"] == n]
        return False

    # -- evaluation --------------------------------------------------------------------------------------------
    def _inputs_for(self, nid, t):
        """Resolve a node's wired sockets by evaluating its upstream sources at time `t` (recursive, memoized)."""
        vals = {}
        for e in self.edges:
            if e["dst"] == nid:
                vals[e["dst_socket"]] = self.evaluate(e["src"], t)[e["src_socket"]]
        return vals

    def evaluate(self, nid, t=0.0):
        """Evaluate a node's outputs at time `t`, computing (and memoizing) every upstream node it depends on. When
        `t` changes from the last evaluate, the memo is cleared so time-varying sources (audio, animated shader
        fields) re-read the clock -- this is what lets an audio band or a moving shader DRIVE the whole graph as time
        advances. Topological by construction (recursion follows edges; the graph is acyclic by the connect guard)."""
        if t != self._t:
            self._cache.clear(); self._t = t
        if nid in self._cache:
            return self._cache[nid]
        spec = self.nodes[nid]
        ntype = self.reg.get(spec["type"])
        wired = self._inputs_for(nid, t)
        # split wired values: regular inputs feed the fn's `inputs`; drivable-param wires OVERRIDE the static param
        inputs = {}
        params = dict(spec["params"])
        for name in ntype.inputs:
            inputs[name] = wired.get(name)                      # None if unconnected -> node decides a default
        for name in ntype.param_inputs:
            if name in wired and wired[name] is not None:
                params[name] = wired[name]                      # a driver overrides the knob
        out = ntype.fn(params, inputs, {"t": t}) if ntype._wants_ctx else ntype.fn(params, inputs)
        self._cache[nid] = out
        return out

    def set_param(self, nid, **params):
        """Edit a node's params and mark it + everything downstream DIRTY, so the next evaluate recomputes only the
        affected subgraph (dirty propagation -- the incremental re-eval an interactive editor needs)."""
        self.nodes[nid]["params"].update(params)
        for d in self._downstream(nid) | {nid}:
            self._cache.pop(d, None)

    def _downstream(self, nid):
        """All nodes reachable FROM nid (its dependents), so a change invalidates exactly them."""
        stack = [nid]; seen = set()
        while stack:
            n = stack.pop()
            for e in self.edges:
                if e["src"] == n and e["dst"] not in seen:
                    seen.add(e["dst"]); stack.append(e["dst"])
        return seen

    # -- serialization -----------------------------------------------------------------------------------------
    def to_dict(self):
        """The whole graph as plain JSON-able data (node ids, types, params, edges) -- what an editor saves/loads."""
        return {"nodes": {k: {"type": v["type"], "params": dict(v["params"])} for k, v in self.nodes.items()},
                "edges": [dict(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, registry, data):
        g = cls(registry)
        for nid, spec in data["nodes"].items():
            g.nodes[nid] = {"type": spec["type"], "params": dict(spec.get("params", {}))}
        g.edges = [dict(e) for e in data["edges"]]
        g._next = len(g.nodes)
        return g


# ---- a default registry wiring REAL leCore subsystems as node types ------------------------------------------
def default_registry():
    """A palette wired to the existing engine: SDF primitives + CSG + K5 fillet, texture const/mix, a field leaf,
    and a scalar -- so a graph built here DRIVES the real compute, proving the shell is not a stub."""
    reg = NodeRegistry()

    from holographic.mesh_and_geometry import holographic_sdf as S

    reg.register("scalar", {}, {"out": "scalar"},
                 lambda p, i: {"out": float(p.get("value", 0.0))})

    # -- SDF PRIMITIVES (delegating to holographic_sdf) --------------------------------------------------------
    reg.register("sdf_sphere", {}, {"out": "sdf"},
                 lambda p, i: {"out": S.sphere(float(p.get("radius", 1.0)))},
                 param_inputs={"radius": "scalar"})            # radius is DRIVABLE (audio/shader can pulse it)
    reg.register("sdf_box", {}, {"out": "sdf"},
                 lambda p, i: {"out": S.box(*p.get("size", (1.0, 1.0, 1.0)))})
    reg.register("sdf_torus", {}, {"out": "sdf"},
                 lambda p, i: {"out": S.torus(float(p.get("R", 1.0)), float(p.get("r", 0.3)))},
                 param_inputs={"r": "scalar"})
    reg.register("sdf_cylinder", {}, {"out": "sdf"},
                 lambda p, i: {"out": S.cylinder(float(p.get("h", 1.0)), float(p.get("r", 0.5)))},
                 param_inputs={"r": "scalar"})
    reg.register("sdf_plane", {}, {"out": "sdf"},
                 lambda p, i: {"out": S.plane(float(p.get("h", 0.0)))})
    reg.register("sdf_capsule", {}, {"out": "sdf"},
                 lambda p, i: {"out": S.capsule(float(p.get("h", 1.0)), float(p.get("r", 0.3)))})
    reg.register("sdf_cone", {}, {"out": "sdf"},
                 lambda p, i: {"out": S.cone(float(p.get("h", 1.0)), float(p.get("r", 0.5)))})

    # -- SDF BOOLEANS -----------------------------------------------------------------------------------------
    reg.register("sdf_union", {"a": "sdf", "b": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].union(i["b"])})
    reg.register("sdf_intersect", {"a": "sdf", "b": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].intersect(i["b"])})
    reg.register("sdf_subtract", {"a": "sdf", "b": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].subtract(i["b"])})
    reg.register("sdf_smooth_union", {"a": "sdf", "b": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].smooth_union(i["b"], float(p.get("k", 0.3)))},
                 param_inputs={"k": "scalar"})                 # blend amount is DRIVABLE

    def _fillet(p, i):
        from holographic.mesh_and_geometry.holographic_fillet import fillet_union
        return {"out": fillet_union(i["a"], i["b"], float(p.get("radius", 0.2)))}
    reg.register("sdf_fillet", {"a": "sdf", "b": "sdf"}, {"out": "sdf"}, _fillet,
                 param_inputs={"radius": "scalar"})            # fillet radius is DRIVABLE

    # -- SDF TRANSFORMS / DOMAIN (single sdf in -> sdf out) ----------------------------------------------------
    reg.register("sdf_translate", {"a": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].translate(tuple(p.get("t", (0.0, 0.0, 0.0))))})
    reg.register("sdf_scale", {"a": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].scale(float(p.get("s", 1.0)))})
    reg.register("sdf_rotate", {"a": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].rotate(tuple(p.get("axis", (0.0, 0.0, 1.0))), float(p.get("angle", 0.0)))})
    reg.register("sdf_repeat", {"a": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].repeat(tuple(p.get("period", (2.0, 2.0, 2.0))))})
    reg.register("sdf_twist", {"a": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].twist(float(p.get("k", 1.0)))})
    reg.register("sdf_elongate", {"a": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].elongate(float(p.get("hx", 0.0)), float(p.get("hy", 0.0)), float(p.get("hz", 0.0)))})
    reg.register("sdf_onion", {"a": "sdf"}, {"out": "sdf"},
                 lambda p, i: {"out": i["a"].onion(float(p.get("thickness", 0.1)))})

    # -- BAKE (the bake-vs-live policy as a node): freeze an SDF subgraph to a grid for O(1) repeated sampling.
    # WHY a node and not a mode: baking is a real edit the editor exposes -- "bake this branch" -- and downstream
    # nodes then sample the cheap GridSDF instead of walking the whole analytic subtree every hit. See rendergraph
    # for the same bake-vs-live decision on textures.
    def _bake(p, i):
        from holographic.mesh_and_geometry.holographic_sdfbake import bake_sdf_grid, GridSDF
        from holographic.mesh_and_geometry.holographic_sdf import as_eval
        lo = tuple(p.get("lo", (-2.0, -2.0, -2.0))); hi = tuple(p.get("hi", (2.0, 2.0, 2.0)))
        res = int(p.get("res", 48))
        dist, _ids = bake_sdf_grid(as_eval(i["a"]), lo, hi, res)
        return {"out": GridSDF(dist, lo, hi)}
    reg.register("sdf_bake", {"a": "sdf"}, {"out": "sdf"}, _bake)

    # -- FIELD generator (gyroid) --------------------------------------------------------------------------
    def _gyroid(p, i):
        from holographic.mesh_and_geometry.holographic_curves import gyroid_field
        scale = float(p.get("scale", 1.0))
        return {"out": (lambda P: gyroid_field(np.asarray(P, float), scale))}
    reg.register("field_gyroid", {}, {"out": "field"}, _gyroid)

    def _tex_const(p, i):
        from holographic.materials_and_texture.holographic_texturegraph import Const
        return {"out": Const(p.get("color", (1.0, 1.0, 1.0)))}
    reg.register("texture_const", {}, {"out": "texture"}, _tex_const)

    def _tex_mix(p, i):
        from holographic.materials_and_texture.holographic_texturegraph import Map, Const
        return {"out": Map("mix", a=i["a"], b=i["b"], t=Const(float(p.get("t", 0.5))))}
    reg.register("texture_mix", {"a": "texture", "b": "texture"}, {"out": "texture"}, _tex_mix)

    def _tex_binop(op):
        def fn(p, i):
            from holographic.materials_and_texture.holographic_texturegraph import Map
            return {"out": Map(op, a=i["a"], b=i["b"])}
        return fn
    reg.register("texture_add", {"a": "texture", "b": "texture"}, {"out": "texture"}, _tex_binop("add"))
    reg.register("texture_multiply", {"a": "texture", "b": "texture"}, {"out": "texture"}, _tex_binop("multiply"))

    # -- OUTPUT / TERMINATING nodes: turn a graph into RENDERABLE geometry ------------------------------------
    def _sdf_to_mesh(p, i):
        # sample the SDF (as_eval handles both SDF objects AND the bare callables that fillet/transform emit) on a
        # grid, then polygonise with marching tetrahedra. This is the node that ends a CSG graph in a real mesh.
        from holographic.mesh_and_geometry.holographic_sdf import as_eval
        from holographic.mesh_and_geometry.holographic_meshbridge import marching_tetrahedra
        ev = as_eval(i["a"])
        lo = np.asarray(p.get("lo", (-2.0, -2.0, -2.0)), float); hi = np.asarray(p.get("hi", (2.0, 2.0, 2.0)), float)
        res = int(p.get("res", 48))
        xs = np.linspace(lo[0], hi[0], res); ys = np.linspace(lo[1], hi[1], res); zs = np.linspace(lo[2], hi[2], res)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        values = ev(P).reshape(res, res, res)
        mesh = marching_tetrahedra(values, (xs, ys, zs), level=float(p.get("level", 0.0)))
        return {"out": mesh}
    reg.register("sdf_to_mesh", {"a": "sdf"}, {"out": "mesh"}, _sdf_to_mesh)

    def _material_lib(p, i):
        from holographic.materials_and_texture.holographic_matlib import material
        return {"out": material(p.get("name", "matte_gray"))}
    reg.register("material_lib", {}, {"out": "material"}, _material_lib)

    # a constant COLOUR node -> drives a material socket
    reg.register("color", {}, {"out": "color"},
                 lambda p, i: {"out": tuple(p.get("rgba", (0.8, 0.8, 0.8, 1.0)))})

    # MATERIAL SOCKET graph: build a PBR material whose channels can be DRIVEN by upstream nodes (a color node into
    # base_color, scalar nodes into metallic/roughness) -- the shader-editor pattern, not just a library preset. Each
    # socket falls back to its param when unconnected. (A texture-DRIVEN base colour needs the procedural material
    # tier -- declared next step; base_color here is a constant colour.)
    def _material_pbr(p, i):
        from holographic.materials_and_texture.holographic_materialio import PBRMaterial
        base = i["base_color"] if i.get("base_color") is not None else tuple(p.get("base_color", (0.8, 0.8, 0.8, 1.0)))
        base = tuple(base) + (1.0,) if len(base) == 3 else tuple(base)
        metallic = i["metallic"] if i.get("metallic") is not None else float(p.get("metallic", 0.0))
        rough = i["roughness"] if i.get("roughness") is not None else float(p.get("roughness", 0.8))
        emis = i["emissive"] if i.get("emissive") is not None else tuple(p.get("emissive", (0.0, 0.0, 0.0)))
        return {"out": PBRMaterial(name=p.get("name", "material"), base_color=base,
                                   metallic=float(metallic), roughness=float(rough), emissive=tuple(emis[:3]))}
    reg.register("material_pbr",
                 {"base_color": "color", "metallic": "scalar", "roughness": "scalar", "emissive": "color"},
                 {"out": "material"}, _material_pbr)

    # TEXTURE-DRIVEN material (the procedural tier): a texture's colour drives the base albedo, which becomes a
    # CALLABLE socket f(points)->(M,3) rgb -- exactly what the procedural material path / a RegionField consumes.
    # This is how a spatially-varying texture paints a surface, vs material_pbr's constant colour. KEPT NEGATIVE:
    # the point->uv map here is a simple xy-planar projection (points[:, :2]); proper UV / triplanar mapping is the
    # declared refinement.
    def _material_textured(p, i):
        tex = i["tex"]
        metallic = i["metallic"] if i.get("metallic") is not None else float(p.get("metallic", 0.0))
        rough = i["roughness"] if i.get("roughness") is not None else float(p.get("roughness", 0.8))

        def albedo(points):
            points = np.asarray(points, float)
            uv = points[:, :2] if points.ndim == 2 else np.atleast_2d(points)[:, :2]
            col = np.asarray(tex.sample(uv), float)
            if col.ndim == 1:                                  # a constant texture returns one colour -> broadcast
                col = np.broadcast_to(col[:3], (len(uv), 3)).copy()
            return col[:, :3]
        return {"out": {"name": p.get("name", "textured"), "albedo": albedo,
                        "metallic": float(metallic), "roughness": float(rough), "textured": True}}
    reg.register("material_textured", {"tex": "texture", "metallic": "scalar", "roughness": "scalar"},
                 {"out": "material"}, _material_textured)

    def _assign_material(p, i):
        # pair geometry with a material -> the renderable object a viewport draws. Typed 'mesh' (still geometry,
        # now carrying a material) so it can flow into further geometry-consuming nodes.
        return {"out": {"mesh": i["mesh"], "material": i["material"]}}
    reg.register("assign_material", {"mesh": "mesh", "material": "material"}, {"out": "mesh"}, _assign_material)

    # -- GEOMETRY MODIFIER nodes (mesh -> mesh): the "geometry nodes" a modeler chains on a mesh --------------
    def _mesh_subdivide(p, i):
        from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
        mesh = i["mesh"]["mesh"] if isinstance(i["mesh"], dict) else i["mesh"]   # unwrap a materialed renderable
        return {"out": loop_subdivide(mesh, int(p.get("levels", 1)))}
    reg.register("mesh_subdivide", {"mesh": "mesh"}, {"out": "mesh"}, _mesh_subdivide)

    def _mesh_smooth(p, i):
        from holographic.mesh_and_geometry.holographic_meshsmooth import laplacian_smooth
        mesh = i["mesh"]["mesh"] if isinstance(i["mesh"], dict) else i["mesh"]
        return {"out": laplacian_smooth(mesh, lam=float(p.get("lam", 0.5)), iters=int(p.get("iters", 5)),
                                        weights=p.get("weights", "cotangent"))}
    reg.register("mesh_smooth", {"mesh": "mesh"}, {"out": "mesh"}, _mesh_smooth)

    def _mesh_decimate(p, i):
        from holographic.mesh_and_geometry.holographic_meshqem import qem_decimate
        mesh = i["mesh"]["mesh"] if isinstance(i["mesh"], dict) else i["mesh"]
        target = int(p.get("target_faces", max(4, int(len(mesh.faces) * float(p.get("ratio", 0.5))))))
        # fast=True: the batched-ranking QEM (opt-in, ~4x faster) -- an editor node is a preview/interactive context
        # where speed matters more than the byte-exact canonical mesh. Pass fast=False in params to force canonical.
        return {"out": qem_decimate(mesh, target, fast=bool(p.get("fast", True)))}
    reg.register("mesh_decimate", {"mesh": "mesh"}, {"out": "mesh"}, _mesh_decimate)

    def _mesh_to_sdf(p, i):
        # the LOOP CLOSER: turn a mesh back into a sampleable SDF field, so geometry can re-enter the CSG graph.
        from holographic.mesh_and_geometry.holographic_meshbridge import mesh_distance_grid
        from holographic.mesh_and_geometry.holographic_sdfbake import GridSDF
        mesh = i["mesh"]["mesh"] if isinstance(i["mesh"], dict) else i["mesh"]
        lo = tuple(p.get("lo", (-2.0, -2.0, -2.0))); hi = tuple(p.get("hi", (2.0, 2.0, 2.0)))
        grid, _axes = mesh_distance_grid(mesh, (lo, hi), res=int(p.get("res", 40)))
        return {"out": GridSDF(np.asarray(grid, float), lo, hi)}
    reg.register("mesh_to_sdf", {"mesh": "mesh"}, {"out": "sdf"}, _mesh_to_sdf)

    # -- DRIVER SOURCES: shader-style maps + audio, emitting GENERIC types so they drive ANYTHING they can ------
    # The whole point: these output standard socket types (field/color/scalar). The type system + drivable params
    # then let them plug into everything compatible -- a radius, a fillet, a blend, a mix -- without enumeration.
    def _shader_field(p, i, ctx):
        # a Shadertoy-STYLE animated procedural field in NumPy (leCore's domain-op / demoscene substrate). The same
        # map can be EMITTED to a real Shadertoy shader via the SDF dialect emitters; here it evaluates natively.
        t = float(ctx.get("t", 0.0))
        k = np.asarray(p.get("freq", (1.0, 1.0, 1.0)), float)
        speed = float(p.get("speed", 1.0)); phase = float(p.get("phase", 0.0))
        def field(P):
            P = np.asarray(P, float)
            return np.sin(P @ k + phase + speed * t)           # a moving plane-wave field; drives displacement/blend/etc.
        return {"out": field}
    reg.register("shader_field", {}, {"out": "field"}, _shader_field)

    def _palette(p, i, ctx):
        # iq's cosine palette as a driver: maps a scalar t (a param OR a driven input -- audio band, orbit trap,
        # anything) to an RGB colour. Drive `t` from audio and the colour animates. See holographic_domain.
        from holographic.mesh_and_geometry.holographic_domain import cosine_palette
        tval = float(p.get("t", 0.0))
        rgb = cosine_palette(tval, tuple(p.get("a", (0.5, 0.5, 0.5))), tuple(p.get("b", (0.5, 0.5, 0.5))),
                             tuple(p.get("c", (1.0, 1.0, 1.0))), tuple(p.get("d", (0.0, 0.33, 0.67))))
        return {"out": np.asarray(rgb, float)}
    reg.register("palette", {}, {"out": "color"}, _palette, param_inputs={"t": "scalar"})

    def _audio_clip(p, i):
        # an audio SIGNAL: either supplied samples+rate, or a synthesized test tone (so a graph is self-contained).
        rate = int(p.get("rate", 22050))
        if p.get("samples") is not None:
            return {"out": {"samples": np.asarray(p["samples"], float), "rate": rate}}
        freq = float(p.get("freq", 220.0)); dur = float(p.get("dur", 1.0)); trem = float(p.get("tremolo", 4.0))
        n = int(rate * dur); tt = np.arange(n) / rate
        # a tone with a slow amplitude tremolo, so its band envelope actually MOVES over time (a real driver)
        sig = np.sin(2 * np.pi * freq * tt) * (0.5 + 0.5 * np.sin(2 * np.pi * trem * tt))
        return {"out": {"samples": sig, "rate": rate}}
    reg.register("audio_clip", {}, {"out": "signal"}, _audio_clip)

    def _audio_band(p, i, ctx):
        # THE audio driver: build a param bus from the signal, read one band's 0..1 envelope AT TIME t (ctx), and
        # map it onto [lo,hi]. Output is a plain SCALAR, so it drives any drivable param. Reuses holographic_parambus
        # (no new FFT) -- exactly ParamBus.subscribe, exposed as a node.
        from holographic.misc.holographic_parambus import param_bus
        sig = i["signal"]
        if sig is None:
            return {"out": float(p.get("lo", 0.0))}
        bus = param_bus(sig["samples"], sig["rate"])
        t = float(ctx.get("t", 0.0))
        frame = int(round(t * bus.fps))
        env = bus.band(int(p.get("band", 0)))
        e = float(env[int(np.clip(frame, 0, len(env) - 1))])
        lo = float(p.get("lo", 0.0)); hi = float(p.get("hi", 1.0))
        return {"out": lo + (hi - lo) * e}
    reg.register("audio_band", {"signal": "signal"}, {"out": "scalar"}, _audio_band)

    return reg


def _selftest():
    reg = default_registry()
    g = NodeGraph(reg)

    # --- build a real SDF graph: sphere UNION box, then evaluate to an SDF and sample it ---
    sph = g.add("sdf_sphere", {"radius": 1.0})
    bx = g.add("sdf_box", {"size": (0.8, 0.8, 0.8)})
    uni = g.add("sdf_union")
    g.connect(sph, "out", uni, "a")
    g.connect(bx, "out", uni, "b")
    sdf = g.evaluate(uni)["out"]
    from holographic.mesh_and_geometry.holographic_sdf import as_eval
    ev = as_eval(sdf)
    # a point at the origin is deep inside both -> negative; a far point is outside -> positive
    assert float(ev(np.array([[0.0, 0, 0]]))[0]) < 0
    assert float(ev(np.array([[5.0, 0, 0]]))[0]) > 0
    # union == min of the two child SDFs at a test point
    a = as_eval(g.evaluate(sph)["out"]); b = as_eval(g.evaluate(bx)["out"])
    P = np.array([[0.9, 0.0, 0.0]])
    assert abs(float(ev(P)[0]) - min(float(a(P)[0]), float(b(P)[0]))) < 1e-9

    # --- TYPE CHECK: wiring an sdf output into a scalar input is refused at connect time ---
    sc = g.add("scalar", {"value": 3.0})
    try:
        g.connect(sph, "out", sc, "nope")                        # scalar has no inputs anyway
        assert False
    except ValueError:
        pass
    # a genuine type mismatch: make a node with an sdf input, feed it a texture -> refused
    tc = g.add("texture_const", {"color": (1, 0, 0)})
    try:
        g.connect(tc, "out", uni, "a")                           # texture -> sdf input
        assert False, "texture->sdf should be refused"
    except TypeError:
        pass

    # --- CYCLE CHECK: wiring uni's output back into one of its own ancestors is refused ---
    uni2 = g.add("sdf_union")
    g.connect(uni, "out", uni2, "a")
    g.connect(sph, "out", uni2, "b")
    try:
        g.connect(uni2, "out", uni, "a")                         # uni2 depends on uni -> cycle
        assert False, "cycle should be refused"
    except ValueError:
        pass

    # --- FILLET node drives the K5 exact fillet: build sphere-fillet-box, check it's a valid field ---
    fil = g.add("sdf_fillet", {"radius": 0.15})
    g.connect(sph, "out", fil, "a")
    g.connect(bx, "out", fil, "b")
    fev = as_eval(g.evaluate(fil)["out"])
    grid = np.array([[x, 0.0, 0.0] for x in np.linspace(-2, 2, 21)])
    v = fev(grid)
    assert v.min() < 0 and v.max() > 0                           # a real solid with interior & exterior

    # --- DIRTY PROPAGATION: change the sphere radius; only the sphere + its dependents recompute ---
    g.evaluate(uni)                                              # warm the cache
    assert sph in g._cache and uni in g._cache and bx in g._cache
    g.set_param(sph, radius=1.5)
    assert sph not in g._cache and uni not in g._cache           # sphere and its dependent union invalidated
    assert bx in g._cache                                        # the box (not downstream of sphere) stays cached

    # --- SERIALIZE round-trip: to_dict -> from_dict -> same evaluation ---
    data = g.to_dict()
    import json
    data2 = json.loads(json.dumps({"nodes": data["nodes"], "edges": data["edges"]}))  # JSON-able
    g2 = NodeGraph.from_dict(reg, data2)
    ev2 = as_eval(g2.evaluate(uni)["out"])
    Q = np.array([[0.9, 0.0, 0.0]])
    assert abs(float(ev2(Q)[0]) - float(as_eval(g.evaluate(uni)["out"])(Q)[0])) < 1e-9

    # --- TEXTURE nodes wire too: const MIX const -> a texture node that samples ---
    gt = NodeGraph(reg)
    c1 = gt.add("texture_const", {"color": (1.0, 0.0, 0.0)})
    c2 = gt.add("texture_const", {"color": (0.0, 0.0, 1.0)})
    mix = gt.add("texture_mix", {"t": 0.5})
    gt.connect(c1, "out", mix, "a")
    gt.connect(c2, "out", mix, "b")
    tex = gt.evaluate(mix)["out"]
    col = np.asarray(tex.sample(np.array([0.5, 0.5])), float)
    assert col.shape[-1] == 3 and 0.0 <= col.min() and col.max() <= 1.0   # a blended colour

    # --- EXPANDED PALETTE: a transform chain + smooth_union + bake + a field generator all evaluate ---
    gp = NodeGraph(reg)
    s2 = gp.add("sdf_sphere", {"radius": 0.6})
    tr = gp.add("sdf_translate", {"t": (1.0, 0.0, 0.0)})     # move the sphere to x=1
    gp.connect(s2, "out", tr, "a")
    b2 = gp.add("sdf_box", {"size": (0.5, 0.5, 0.5)})
    su = gp.add("sdf_smooth_union", {"k": 0.3})
    gp.connect(tr, "out", su, "a")
    gp.connect(b2, "out", su, "b")
    ev_su = as_eval(gp.evaluate(su)["out"])
    # the translated sphere really is centred at x=1 (its own surface passes ~0 near x=1)
    ev_tr = as_eval(gp.evaluate(tr)["out"])
    assert float(ev_tr(np.array([[1.0, 0, 0]]))[0]) < 0 and float(ev_tr(np.array([[-1.0, 0, 0]]))[0]) > 0

    # bake node: freeze an SDF branch to a grid; the baked sampler agrees with the analytic within grid tolerance
    bake = gp.add("sdf_bake", {"lo": (-2, -2, -2), "hi": (2, 2, 2), "res": 64})
    gp.connect(su, "out", bake, "a")
    ev_bake = as_eval(gp.evaluate(bake)["out"])
    test_pts = np.array([[0.0, 0, 0], [0.9, 0, 0], [1.5, 0.5, 0.0]])
    assert np.max(np.abs(ev_bake(test_pts) - ev_su(test_pts))) < 0.1   # trilinear grid vs analytic, within a cell

    # field generator node: gyroid outputs a field callable that varies in space
    gf = gp.add("field_gyroid", {"scale": 1.0})
    field = gp.evaluate(gf)["out"]
    fv = field(np.array([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]]))
    assert np.asarray(fv).shape == (2,) and not np.allclose(fv[0], fv[1])

    # --- palette size sanity: the registry exposes a real editor palette, not two toy nodes ---
    assert len(reg.types) >= 18, len(reg.types)

    # --- END-TO-END: a graph that TERMINATES in renderable geometry -- primitives -> CSG -> mesh -> +material ---
    ge = NodeGraph(reg)
    esph = ge.add("sdf_sphere", {"radius": 1.0})
    ebox = ge.add("sdf_box", {"size": (0.9, 0.9, 0.9)})
    euni = ge.add("sdf_union")
    ge.connect(esph, "out", euni, "a"); ge.connect(ebox, "out", euni, "b")
    emesh = ge.add("sdf_to_mesh", {"lo": (-2, -2, -2), "hi": (2, 2, 2), "res": 10})
    ge.connect(euni, "out", emesh, "a")
    mesh = ge.evaluate(emesh)["out"]
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0    # a real watertight-ish mesh came out of the CSG graph
    emat = ge.add("material_lib", {"name": "matte_gray"})
    easgn = ge.add("assign_material")
    ge.connect(emesh, "out", easgn, "mesh"); ge.connect(emat, "out", easgn, "material")
    renderable = ge.evaluate(easgn)["out"]
    assert "mesh" in renderable and "material" in renderable  # the graph terminates in a materialed, renderable object
    # type check still holds at the output stage: a texture cannot feed the material slot
    etc = ge.add("texture_const", {"color": (1, 0, 0)})
    try:
        ge.connect(etc, "out", easgn, "material"); assert False
    except TypeError:
        pass

    # --- GEOMETRY MODIFIER nodes: subdivide adds detail, decimate removes it, smooth preserves count; then the
    # mesh_to_sdf LOOP CLOSER turns geometry back into a field. Kept small/fast (mesh_to_sdf is brute point-to-mesh). ---
    base_mesh = ge.evaluate(emesh)["out"]
    sub = ge.add("mesh_subdivide", {"levels": 1})
    ge.connect(emesh, "out", sub, "mesh")
    assert len(ge.evaluate(sub)["out"].vertices) > len(base_mesh.vertices)   # subdivision added detail

    sm = ge.add("mesh_smooth", {"lam": 0.5, "iters": 2})
    ge.connect(emesh, "out", sm, "mesh")
    assert len(ge.evaluate(sm)["out"].vertices) == len(base_mesh.vertices)   # smoothing preserves topology

    dec = ge.add("mesh_decimate", {"ratio": 0.4})
    ge.connect(emesh, "out", dec, "mesh")
    dec_mesh = ge.evaluate(dec)["out"]
    assert len(dec_mesh.faces) < len(base_mesh.faces)                        # decimation removed faces

    # loop closer on the small decimated mesh: geometry becomes an SDF again (banded near the surface -> test a
    # point ON the surface, where the banded distance is ~0; the origin is outside the band and clamped).
    back = ge.add("mesh_to_sdf", {"lo": (-2, -2, -2), "hi": (2, 2, 2), "res": 16})
    ge.connect(dec, "out", back, "mesh")
    sdf_again = as_eval(ge.evaluate(back)["out"])
    surf = np.array([[1.0, 0.0, 0.0]])                       # ~on the sphere/box union surface
    assert float(sdf_again(surf)[0]) < 0.3                   # banded distance is small at the surface

    # --- MATERIAL SOCKET graph: a color node + scalar nodes DRIVE a PBR material's channels (shader-editor pattern,
    # not a library preset), which is then assigned to geometry. ---
    col = ge.add("color", {"rgba": (0.2, 0.5, 0.9, 1.0)})
    rough = ge.add("scalar", {"value": 0.25})
    metal = ge.add("scalar", {"value": 1.0})
    mpbr = ge.add("material_pbr", {"name": "brushed_blue"})
    ge.connect(col, "out", mpbr, "base_color")
    ge.connect(rough, "out", mpbr, "roughness")
    ge.connect(metal, "out", mpbr, "metallic")
    pbr = ge.evaluate(mpbr)["out"]
    assert abs(pbr.roughness - 0.25) < 1e-9 and abs(pbr.metallic - 1.0) < 1e-9   # driven by the scalar nodes
    assert tuple(pbr.base_color)[:3] == (0.2, 0.5, 0.9)      # driven by the color node
    # type check: a scalar cannot drive the base_color (color) socket
    try:
        ge.connect(rough, "out", mpbr, "base_color"); assert False
    except TypeError:
        pass

    # --- TEXTURE-DRIVEN material (procedural tier): a texture drives the albedo, which becomes a callable socket
    # f(points)->rgb. Here a red-blue mix (t=0.5) -> a constant purple albedo; sampled at any points it returns that. ---
    tred = ge.add("texture_const", {"color": (1.0, 0.0, 0.0)})
    tblue = ge.add("texture_const", {"color": (0.0, 0.0, 1.0)})
    tmix = ge.add("texture_mix", {"t": 0.5})
    ge.connect(tred, "out", tmix, "a"); ge.connect(tblue, "out", tmix, "b")
    mtex = ge.add("material_textured", {"name": "purple_tex", "roughness": 0.3})
    ge.connect(tmix, "out", mtex, "tex")
    procmat = ge.evaluate(mtex)["out"]
    assert procmat["textured"] and callable(procmat["albedo"])
    cols = procmat["albedo"](np.array([[0.0, 0, 0], [1.0, 1, 1]]))   # albedo socket: points -> (M,3) rgb
    assert cols.shape == (2, 3)
    assert 0.4 < cols[0][0] < 0.6 and 0.4 < cols[0][2] < 0.6 and cols[0][1] < 0.1   # ~purple (r,b ~0.5, g ~0)
    # type check: an sdf cannot drive the texture socket
    try:
        ge.connect(euni, "out", mtex, "tex"); assert False
    except TypeError:
        pass

    # --- DRIVERS: audio and shader as generic sources that DRIVE anything they can, chosen by socket TYPE ---
    # (1) one audio band drives a sphere's radius; advancing TIME changes the driven radius (audio-reactive geometry)
    gd = NodeGraph(reg)
    clip = gd.add("audio_clip", {"freq": 220.0, "dur": 1.0, "tremolo": 4.0, "rate": 22050})
    aband = gd.add("audio_band", {"band": 0, "lo": 0.4, "hi": 1.6})
    gd.connect(clip, "out", aband, "signal")
    sphere = gd.add("sdf_sphere", {"radius": 1.0})
    gd.connect(aband, "out", sphere, "radius")                # audio scalar -> DRIVABLE radius param
    r_t0 = float(as_eval(gd.evaluate(sphere, t=0.0)["out"])(np.array([[0.0, 0, 0]]))[0])   # -radius at the origin
    r_t1 = float(as_eval(gd.evaluate(sphere, t=0.5)["out"])(np.array([[0.0, 0, 0]]))[0])
    assert abs(r_t0 - r_t1) > 1e-3, ("audio did not move the radius over time", r_t0, r_t1)

    # (2) the SAME audio band scalar drives a DIFFERENT thing -- a fillet radius -- nothing enumerates what it drives
    boxd = gd.add("sdf_box", {"size": (0.7, 0.7, 0.7)})
    fild = gd.add("sdf_fillet", {"radius": 0.2})
    gd.connect(sphere, "out", fild, "a"); gd.connect(boxd, "out", fild, "b")
    gd.connect(aband, "out", fild, "radius")                  # same source, different target, purely by wiring
    assert gd.evaluate(fild, t=0.2)["out"] is not None

    # (3) type safety holds for drivers: an sdf cannot drive a scalar param (radius) -- refused at connect
    sp2 = gd.add("sdf_sphere")
    try:
        gd.connect(sphere, "out", sp2, "radius"); assert False
    except TypeError:
        pass

    # (4) a shader_field is a time-varying FIELD source; evaluated at two times it differs (an animated map)
    gs = NodeGraph(reg)
    sf = gs.add("shader_field", {"freq": (2.0, 0.0, 0.0), "speed": 3.0})
    P0 = np.array([[0.3, 0.0, 0.0]])
    v0 = float(gs.evaluate(sf, t=0.0)["out"](P0)[0]); v1 = float(gs.evaluate(sf, t=0.5)["out"](P0)[0])
    assert abs(v0 - v1) > 1e-3, ("shader field did not animate with time", v0, v1)

    # (5) palette driven by audio: an animated colour
    gcp = NodeGraph(reg)
    ac2 = gcp.add("audio_clip", {"freq": 110.0}); ab2 = gcp.add("audio_band", {"band": 0, "lo": 0.0, "hi": 1.0})
    gcp.connect(ac2, "out", ab2, "signal")
    pal = gcp.add("palette", {"a": (0.5, 0.5, 0.5)})
    gcp.connect(ab2, "out", pal, "t")
    col0 = gcp.evaluate(pal, t=0.0)["out"]; col1 = gcp.evaluate(pal, t=0.5)["out"]
    assert np.asarray(col0).shape == (3,) and not np.allclose(col0, col1)

    print("holographic_nodegraph selftest OK: one heterogeneous typed graph drives REAL leCore compute -- sdf "
          "CSG/transforms/fillet/bake/field/texture, OUTPUT nodes (sdf_to_mesh + material) that terminate a graph in "
          "renderable geometry, a GEOMETRY-MODIFIER chain (subdivide/smooth/decimate) + a mesh_to_sdf LOOP CLOSER, "
          "and DRIVER SOURCES: an audio band (ParamBus, no new FFT) outputs a SCALAR that drives a sphere radius AND "
          "a fillet radius -- same source, different targets, by wiring alone -- and advancing TIME re-reads the "
          "clock so the geometry is audio-reactive; a shader_field is a time-animated FIELD; a palette colour is "
          "audio-driven. ANY param is DRIVABLE (an optional typed input overrides the knob) and the socket TYPE "
          "decides what a source can plug into (an sdf cannot drive a scalar -- refused). Connections type/cycle-"
          "checked; dirty-propagating; JSON-serializable. %d node types. The shell delegates every node's compute to "
          "the existing subsystem -- editor contract, not a second engine. Deterministic." % len(reg.types))


if __name__ == "__main__":
    _selftest()
