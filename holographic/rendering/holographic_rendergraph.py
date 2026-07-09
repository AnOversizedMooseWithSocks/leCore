"""holographic_rendergraph.py -- CMP5: let the PIPELINE compose the CMP1-CMP4 graphs, and make 'adaptive' reach down.

The pipeline (holographic_pipeline) already orders render/sim STAGES by what they need/produce. CMP1-CMP4 gave us the
graphs BELOW a render: texture maps (CMP1), multi- and layered materials (CMP2/3), and a type-checked instanced scene
(CMP4). CMP5 is the join: an orchestrator that PREPARES those graphs into a render-ready form as pipeline stages, and
-- the point -- makes the one decision that "adaptive" is really about at this level:

    BAKE a static texture graph to a grid (look it up in O(1)), or SAMPLE it live (walk the tree each hit)?

Baking wins when a graph is STATIC and sampled many times (you pay the tree walk once, per grid cell, then every hit
is a cheap interpolation). Live wins when the graph is cheap or CHANGES per frame (baking it would be wasted work,
re-done every frame). So the prepare step looks at each texture and decides -- and plan() tells you what it will do and
WHY before it runs, exactly like the render pipeline's plan().

KEPT NEGATIVE (loud): baking trades MEMORY for speed and adds INTERPOLATION error -- a grid at resolution R reproduces
a smooth map well but blurs detail finer than a cell (raise R, or keep sharp maps live). This is the same trade
holographic_matbake records for the 3-D material bake; here it is the 2-D texture case.

Reuses: holographic_texturegraph (CMP1 -- the graph + sample_grid), holographic_instancing (CMP4 -- bind + flatten the
scene), holographic_pipeline.Stage (the needs/produces step abstraction, so this really is 'the pipeline composing the
graphs'), and holographic_matbake for the 3-D MATERIAL bake (bake_material) when you want it. Plain NumPy; readable.
"""
import numpy as np


class BakedTexture:
    """A 2-D texture graph evaluated onto a grid once, then read back by BILINEAR interpolation -- O(1) per sample,
    no matter how deep the graph was. Same .sample(uv) interface as a live graph, so it is a drop-in replacement. This
    is the 2-D twin of holographic_matbake.BakedField (which bakes 3-D material channels); same idea, same trade."""

    def __init__(self, grid, lo=0.0, hi=1.0):
        self.grid = np.asarray(grid, float)                     # (res,res) scalar or (res,res,C) colour; grid[v][u]
        self.res = self.grid.shape[0]
        self.lo = float(lo)
        self.hi = float(hi)

    def sample(self, uv):
        """Bilinearly sample at uv=(u,v). Points outside [lo,hi] clamp to the edge (like the 3-D BakedField)."""
        u, v = float(uv[0]), float(uv[1])
        span = max(self.hi - self.lo, 1e-12)
        fu = np.clip((u - self.lo) / span, 0.0, 1.0) * (self.res - 1)   # continuous grid coords
        fv = np.clip((v - self.lo) / span, 0.0, 1.0) * (self.res - 1)
        u0, v0 = int(np.floor(fu)), int(np.floor(fv))
        u1, v1 = min(u0 + 1, self.res - 1), min(v0 + 1, self.res - 1)
        du, dv = fu - u0, fv - v0                               # interpolation weights in [0,1]
        g = self.grid                                          # grid is indexed [v][u] (rows are v, cols u)
        top = (1 - du) * g[v0, u0] + du * g[v0, u1]            # interpolate along u at the two v rows...
        bot = (1 - du) * g[v1, u0] + du * g[v1, u1]
        return (1 - dv) * top + dv * bot                       # ...then along v


def bake_texture(graph, res=64, lo=0.0, hi=1.0):
    """Bake a CMP1 texture graph to a res x res grid (via the graph's own sample_grid) -> a BakedTexture you sample in
    O(1). This is the ONE precompute; do it when a map is static and sampled a lot."""
    from holographic.materials_and_texture.holographic_texturegraph import sample_grid
    grid = sample_grid(graph, res=res, lo=lo, hi=hi)
    return BakedTexture(grid, lo=lo, hi=hi)


def resolve_texture(graph, res=64, bake="auto", static=True, lo=0.0, hi=1.0):
    """The ADAPTIVE decision, in one place. Return a thing you can .sample(uv):
      * bake=True         -> always bake (a BakedTexture).
      * bake=False        -> always keep it live (the graph itself).
      * bake='auto'       -> bake when the map is STATIC (amortise the tree walk over many hits); keep it live when it
                             is NOT static (it would be re-baked every frame -- wasted work).
    A BakedTexture and a live graph share the same .sample(uv), so downstream code doesn't care which it got."""
    if bake is True:
        return bake_texture(graph, res=res, lo=lo, hi=hi)
    if bake is False:
        return graph
    return bake_texture(graph, res=res, lo=lo, hi=hi) if static else graph


class _Texture:
    """Bookkeeping for one named texture the RenderGraph manages: the graph, whether it is static, and (after prepare)
    what it resolved to and why."""

    def __init__(self, name, graph, static=True):
        self.name = name
        self.graph = graph
        self.static = static
        self.resolved = None
        self.decision = None            # 'bake' or 'live', filled in by prepare()


class PreparedScene:
    """The render-ready result of RenderGraph.prepare(): resolved textures (baked or live, all .sample(uv)-able) and
    the bound scene geometry (one merged surface mesh + the volume instances kept aside)."""

    def __init__(self, textures, surface_mesh, volume_instances):
        self.textures = textures                # name -> BakedTexture | live graph
        self.surface_mesh = surface_mesh        # one merged Mesh (may be empty)
        self.volume_instances = volume_instances

    def texture(self, name):
        return self.textures[name]


class RenderGraph:
    """Orchestrate the CMP1-CMP4 graphs into a render-ready scene, as PIPELINE STAGES. Register named texture graphs
    (each static or not) and an instanced scene; plan() shows what prepare() will do and WHY (bake vs live per texture,
    then bind the scene); prepare() runs the stages and returns a PreparedScene. The bake-vs-live choice is where
    'adaptive' reaches all the way down to the maps."""

    def __init__(self, res=64):
        self.textures = []              # list of _Texture
        self.scene = None               # a CMP4 InstancedScene (optional)
        self.res = res

    def add_texture(self, name, graph, static=True):
        """Register a named texture graph. static=False means it changes per frame -> prepare() keeps it live."""
        self.textures.append(_Texture(name, graph, static=static))
        return self

    def set_scene(self, instanced_scene):
        """Register a CMP4 InstancedScene to bind + flatten during prepare()."""
        self.scene = instanced_scene
        return self

    def _stages(self):
        """The pipeline stages this render graph runs -- each declares needs/produces so the ordering is explicit.
        Reuses holographic_pipeline.Stage as the step abstraction (this is 'the pipeline composing the graphs')."""
        from holographic.scene_and_pipeline.holographic_pipeline import Stage
        stages = [
            Stage(name="bake_textures", needs=("textures",), produces=("resolved_textures",), phase=0,
                  why="bake STATIC texture graphs to a grid (O(1) lookup); keep dynamic ones live"),
            Stage(name="bind_scene", needs=("scene",), produces=("surface_mesh", "volume_instances"), phase=0,
                  why="type-check the material<->geometry bindings and flatten surface instances to one mesh (CMP4)"),
        ]
        return stages

    def plan(self):
        """A readable, side-effect-free description of what prepare() will do -- one line per texture (bake or live,
        and why) plus the scene-binding step. Mirrors the render pipeline's plan(): see before you run."""
        lines = []
        for t in self.textures:
            if t.static:
                lines.append("bake_textures: '%s' -> BAKE to a %dx%d grid (static, amortise over many hits)"
                             % (t.name, self.res, self.res))
            else:
                lines.append("bake_textures: '%s' -> keep LIVE (not static -- baking would be re-done every frame)"
                             % t.name)
        if self.scene is not None:
            n_surf = len(self.scene.surface_instances())
            n_vol = len(self.scene.volume_instances())
            lines.append("bind_scene: bind %d definition(s), flatten %d surface instance(s) to one mesh, keep %d "
                         "volume instance(s) aside" % (len(self.scene.definitions()), n_surf, n_vol))
        return lines

    def prepare(self, lo=0.0, hi=1.0):
        """Run the stages and return a PreparedScene. bake_textures resolves each texture (bake vs live); bind_scene
        flattens the CMP4 surface instances into one mesh. Deterministic: baking just evaluates the graph on a grid."""
        # stage bake_textures
        resolved = {}
        for t in self.textures:
            t.resolved = resolve_texture(t.graph, res=self.res, bake="auto", static=t.static, lo=lo, hi=hi)
            t.decision = "bake" if t.static else "live"
            resolved[t.name] = t.resolved
        # stage bind_scene (CMP4): flattening also triggers the compose-time binding checks already done at build time
        if self.scene is not None:
            surface_mesh = self.scene.flatten_surface()
            volumes = self.scene.volume_instances()
        else:
            from holographic.mesh_and_geometry.holographic_mesh import Mesh
            surface_mesh, volumes = Mesh(np.zeros((0, 3)), []), []
        return PreparedScene(resolved, surface_mesh, volumes)


def _selftest():
    from holographic.materials_and_texture.holographic_texturegraph import Map, Const, field_leaf
    from holographic.misc.holographic_instancing import Definition, InstancedScene, VOLUME
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.scene_and_pipeline.holographic_scenegraph import translation

    # a static texture graph (a smooth fbm blend) -- bake it and check the baked lookup matches the live graph
    g = Map("mix", a=Const([1.0, 0, 0]), b=Const([0, 0, 1.0]), t=field_leaf("fbm", n_dims=2, seed=0))
    baked = bake_texture(g, res=128)
    errs = [float(np.max(np.abs(np.asarray(g.sample([u, v])) - np.asarray(baked.sample([u, v])))))
            for u, v in [(0.2, 0.3), (0.5, 0.5), (0.8, 0.1), (0.33, 0.77)]]
    assert max(errs) < 0.05, errs                              # bilinear bake tracks the smooth graph (interp error)

    # the adaptive decision
    assert isinstance(resolve_texture(g, static=True), BakedTexture)          # static -> baked
    assert resolve_texture(g, static=False) is g                              # dynamic -> live (the graph itself)
    assert isinstance(resolve_texture(g, bake=True), BakedTexture)            # forced bake
    assert resolve_texture(g, bake=False) is g                               # forced live

    # orchestrate: two textures (one static, one dynamic) + a CMP4 instanced scene
    scene = InstancedScene()
    chair = Definition("chair", box(1, 1, 1), "metal")
    scene.place(chair, translation([-2, 0, 0])); scene.place(chair, translation([2, 0, 0]))
    scene.place(Definition("haze", object(), "fog", geometry_kind=VOLUME))
    rg = RenderGraph(res=48)
    rg.add_texture("rust", g, static=True).add_texture("ripples", g, static=False).set_scene(scene)

    plan = rg.plan()
    assert any("BAKE" in ln and "rust" in ln for ln in plan)
    assert any("LIVE" in ln and "ripples" in ln for ln in plan)
    assert any("bind_scene" in ln for ln in plan)

    prep = rg.prepare()
    assert isinstance(prep.texture("rust"), BakedTexture)     # static -> baked
    assert prep.texture("ripples") is g                       # dynamic -> live
    assert prep.surface_mesh.n_vertices == 2 * box(1, 1, 1).n_vertices   # 2 surface instances merged
    assert len(prep.volume_instances) == 1

    print("OK: holographic_rendergraph self-test passed (baked texture matches the live graph within %.3f interp "
          "error; adaptive resolve bakes static / keeps dynamic live; plan() reports bake-vs-live + the scene bind; "
          "prepare() bakes 'rust', keeps 'ripples' live, and flattens 2 surface instances to one mesh)" % max(errs))


if __name__ == "__main__":
    _selftest()
