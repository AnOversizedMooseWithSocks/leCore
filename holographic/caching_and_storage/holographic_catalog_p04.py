"""holographic_catalog_p04 -- part 4/6 of the capability registry (split from holographic_catalog).

MECHANICAL SPLIT, no edits. holographic_catalog.py hit 81% of the 1 MB agent-read cap, so the file
that makes capabilities discoverable was becoming the one file an agent could not open. The parts are
called IN ORDER by default_catalog() and the emitted catalog is byte-identical -- verified by hashing
every capability field before and after. Order matters: find_capability ranks by score and ties break
by registration order, so a reordering would silently move search results.

Add new capabilities to the LAST part, or to whichever part is topically right -- never to a new file
without registering it in default_catalog(), or it will simply not exist.
"""


def register_p04(c):
    """Register this part's capabilities on `c`. Called by default_catalog() in order."""
    c.register_capability("Shrinkwrap (snap a mesh onto a surface)", "SHRINKWRAP: move each vertex onto its CLOSEST POINT on a target surface (Blender shrinkwrap / retopo-snap): m.shrinkwrap(mesh, target, factor=1.0) -> (new_mesh, residual). factor 1.0 lands on the surface, 0.5 halfway, 0.0 no-op; topology preserved; residual = distance each vertex closed. THE retopo finisher: a box model / remesh has clean TOPOLOGY but approximate POSITIONS -- one pass snaps positions onto the reference (fixed our box-model residual 0.0158 -> ~0). KEPT NEG: closest-POINT not normal-raycast; a thin target can pull to the wrong side (small factor, repeat).",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, grid, Mesh; tgt=grid(6,6,width=6.0,height=6.0); lift=Mesh(np.asarray(box().vertices,float)+np.array([0,0,2.0]),[tuple(f) for f in box().faces]); sw,res=m.shrinkwrap(lift, tgt, factor=1.0); (bool(np.allclose(np.asarray(sw.vertices)[:,2],0,atol=1e-6)), round(float(res.max()),2))",
                          native=True, aliases=("shrinkwrap a mesh", "snap a mesh onto a surface", "project a mesh onto another",
                                                "conform a mesh to a surface", "retopo snap", "wrap a mesh to a target",
                                                "pull vertices onto a surface"))
    c.register_capability("UV shell (texture-carrying envelope)", "UV SHELL (cage-bake as geometry): freeze a texture onto a slightly-inflated ENVELOPE so it survives ANY topology change. make_uv_shell pushes vertices OUTWARD along normals, keeping faces + UVs. project_uv_from_shell reads each new vertex UV from the closest shell point, so a LOD/retopo/remesh recovers the texture regardless of topology; returns (uvs, residual). Freeze once, project onto any geometry. MEASURED: mantis LOD and retopo both re-textured from ONE shell, residual 0.0017. KEPT NEG: uniform offset can pinch in deep concavity; closest-point can grab a thin feature's far side.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; b=box(); V=np.asarray(b.vertices,float); uv=(V[:,:2]-V[:,:2].min(0)); uv=uv/(uv.max(0)+1e-9); shell=m.make_uv_shell(b, uv, offset=0.1); puv,res=m.project_uv_from_shell(b, shell); (puv.shape==uv.shape, float(res.mean())<0.2)",
                          native=True, aliases=("uv shell", "texture shell", "cage bake uvs", "keep texture through a remesh",
                                                "project texture onto new topology", "reproject uvs after decimation",
                                                "envelope to carry a texture map"))
    c.register_capability("Depth from a hazy/foggy image (haze + defocus)", "RELATIVE DEPTH from a single HAZY/shallow-DoF photo -- the NO-WEIGHTS fix for scenes where shape-from-shading INVERTS depth (fog reads as near). Fuses HAZE (atmospheric scattering, Tarel-Hautiere veil; m.haze_depth) + DEPTH-OF-FIELD (local sharpness; m.sharpness_depth) via a guided filter; hand to photo_to_3d. Returns depth (H,W) in [0,1], 1=nearest. MEASURED: on the foggy forest photo it more than DOUBLED near/far separation vs SfS (+0.13 vs +0.06), fixing the inversion. KEPT NEG: relative not metric; needs real haze or DoF (else use shape_from_shading); sky-guard clamps bright sky to far.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); img=np.zeros((60,90,3)); yy,xx=np.mgrid[0:60,0:90]; img[:]=0.3+0.5*(yy/59.0)[...,None]; d=m.fuse_depth(img); (d.shape==(60,90), float(d[40:].mean())>=0.0)",
                          native=True, aliases=("depth from a foggy image", "haze depth", "depth from fog", "dehaze depth",
                                                "atmospheric depth from a photo", "defocus depth", "depth of field depth",
                                                "fix shape from shading on outdoor photos", "depth from a hazy photo"))
    c.register_capability("Auto-weighted depth from a photo (vanishing-point gated)", "AUTO-WEIGHTED single-image depth (m.auto_fuse_depth): fuse HAZE + SHARPNESS, each weighted by how well it AGREES with the scene's LINEAR PERSPECTIVE -- the cue tracking depth for THIS photo dominates, an INVERTED cue auto-down-weighted, no per-image hand-tuning. Vanishing point from oblique Hough lines (m.vanishing_point + confidence); native cue full weight, flipped one discounted; fixed fallback if no confident VP. Returns depth (H,W), 1=nearest. MEASURED: tracks->haze 0.73, bridge->0.57, forest->0.78. Feed depth_to_mesh. KEPT NEG: VP prior gives the depth AXIS not true depth; relative.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); img=np.zeros((50,70,3)); yy,xx=np.mgrid[0:50,0:70]; img[:]=(0.25+0.55*yy/49.0)[...,None]; d=m.auto_fuse_depth(img); (d.shape==(50,70), 0.0<=float(d.mean())<=1.0)",
                          native=True, aliases=("auto depth from a photo", "automatic depth cue weighting", "vanishing point depth",
                                                "detect the vanishing point", "perspective-weighted depth", "auto fuse depth cues",
                                                "best depth cue for this photo"))
    c.register_capability("Ground-plane depth (forward-looking perspective)", "GROUND-PLANE DEPTH from linear perspective (m.ground_plane_depth): for a forward-looking camera the ground recedes to the horizon, so depth rises with height up to the VP row. THE cue that captures a track/road recession when HAZE and DEFOCUS are weak (mostly-in-focus scene). auto_fuse_depth uses it as the BACKBONE (haze/sharpness add relief) at a confident VP -- fixed misty-tracks flat depth (std 0.13->0.26). Returns depth (H,W), 1=nearest. KEPT NEG: assumes a level forward-looking camera, ground at bottom -- meaningless for top-down/portrait (gated behind a confident VP); RAMP only.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); img=np.zeros((60,80,3)); yy,xx=np.mgrid[0:60,0:80]; img[:]=(0.2+0.6*yy/59.0)[...,None]; d=m.ground_plane_depth(img, vp=(40,5)); (d.shape==(60,80), float(d[50:].mean())>float(d[:10].mean()))",
                          native=True, aliases=("ground plane depth", "perspective depth ramp", "road recession depth",
                                                "depth from linear perspective", "forward-looking depth", "horizon depth ramp",
                                                "depth for a road or track scene"))
    c.register_capability("Depth map to a clean height-field mesh", "DEPTH MAP -> a CLEAN triangulated HEIGHT-FIELD MESH for single-view photo-to-3D (m.depth_to_mesh). 2 triangles per pixel block, dropped where depth jumps > `discontinuity` (so near foreground is not welded to far -- no melted mesh). Regular grid = ZERO non-manifold edges (unlike dual-contour points_to_mesh), smoothable/textured. Accepts ANY depth (1=near); pair with fuse_depth. Returns (mesh, vertex_colours). MEASURED: bridge photo -> 104k-vert textured relief, 0 non-manifold edges. KEPT NEG: single-view FRONT relief not a solid; relative depth; wrong discontinuity melts or shreds.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); img=np.zeros((40,60,3)); yy,xx=np.mgrid[0:40,0:60]; img[:]=(0.3+0.5*yy/39.0)[...,None]; d=m.fuse_depth(img); mesh,vcol=m.depth_to_mesh(d, colour=img, discontinuity=0.1); (mesh.n_vertices>0, vcol is not None)",
                          native=True, aliases=("depth map to mesh", "height field mesh from depth", "mesh a depth map",
                                                "photo to a clean mesh", "triangulate a depth image", "relief mesh from a photo",
                                                "turn a depth map into geometry"))
    c.register_capability("Skin a skeleton (B-Mesh base mesh)", "SKIN A SKELETON (B-Mesh, SDF route): wrap a stick figure -- verts (n,3), edges [(i,j)...], per-vertex radii (n,) -- in ONE watertight surface (faculty m.skin_skeleton). Each edge becomes a capsule; branches MERGE automatically (smooth_union stitches for free), then marching-cubes to a Mesh. THE base-mesh route: model a creature from ~20 joints not 200 extrudes. MEASURED: an 18-joint mantis skeleton skins to a watertight blob at 0.29 silhouette IoU vs the original. KEPT NEG: organic isotropic-triangle topology, NOT edge-loops -- a BLOCK-OUT to retopo/quad_remesh onto, not a final cage.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); sk=m.skin_skeleton(np.array([[0,0,0],[1.0,0,0],[0.5,0.8,0]]), [(0,1),(0,2)], np.array([0.2,0.15,0.12]), resolution=36); (sk.n_vertices>0, sk.is_closed())",
                          native=True, aliases=("skin a skeleton", "skin modifier", "base mesh from a stick figure",
                                                "tube mesh from edges with radii", "creature from joints", "b-mesh",
                                                "blockout mesh from a skeleton"))
    c.register_capability("Fit a base mesh to a target", "FIT A BASE MESH TO A TARGET (the closed block-out loop): skin a skeleton into a watertight base mesh, SHRINKWRAP it onto a target, report the silhouette-fit gain (faculty m.fit_base_mesh). The block-out-then-snap loop, an OPTIMISATION target since it returns iou_base and iou_fitted. Returns {base, fitted, residual, iou_base, iou_fitted}. MEASURED: a crude 1-edge capsule fitted to a stretched-box target jumped 0.64 -> 0.97 mean IoU. KEPT NEG: closest-point shrinkwrap -- the skeleton must roughly COVER the target parts; fits SHAPE not TOPOLOGY (retopo after).",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, Mesh; tgt=Mesh(np.asarray(box().vertices,float)*np.array([2.0,0.6,0.6]),[tuple(f) for f in box().faces]); r=m.fit_base_mesh(tgt, np.array([[-1.0,0,0],[1.0,0,0]]), [(0,1)], np.array([0.4,0.4]), resolution=28); r['iou_fitted']>r['iou_base']",
                          native=True, aliases=("fit a base mesh to a target", "block out and snap to a reference",
                                                "auto-fit a skeleton to a mesh", "skin then shrinkwrap",
                                                "fit a blockout to a sculpt", "conform a base mesh"))
    c.register_capability("Voxel remesh (uniform cleanup)", "VOXEL REMESH (Blender Voxel Remesh): rebuild a mesh as a UNIFORM watertight surface via a signed-distance grid + re-marching (faculty m.voxel_remesh). The standard cleanup for messy/self-intersecting/non-manifold/multi-shell input before retopo -- any tangle becomes one clean closed surface at `resolution` cells per axis. A compose of mesh_to_sdf_grid + marching tetrahedra. Pairs with skin_skeleton (clean the block-out) then quad_remesh (get quads). KEPT NEG: uniform density rounds off features below the cell size (raise resolution or crease after); wants a roughly-closed input.",
                          example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; vr=m.voxel_remesh(box(), resolution=36); (vr.n_faces>0, m.mesh_report(vr)['is_closed'])",
                          native=True, aliases=("voxel remesh", "remesh a mesh uniformly", "clean up a messy mesh",
                                                "rebuild a mesh watertight", "uniform remesh", "fix a non-manifold mesh by remeshing",
                                                "remesh with a voxel grid"))
    c.register_capability("Metaball mesh (soft-blob base mesh)", "METABALL MESH (Blender metaballs / soft-blob base mesh): sum-of-Gaussians field at `centers` (n,3), spread `radius`, marched at `level` -- overlapping blobs FUSE smoothly (faculty m.metaball_mesh). The organic-blob base-mesh route complementing skin_skeleton (blobs where branch-stitching gets ugly). Returns a watertight Mesh. MEASURED: two overlapping blobs fuse to one watertight shell. KEPT NEG: isotropic-triangle blob topology (retopo after); too high a `level` on far centers yields separate shells.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); mb=m.metaball_mesh(np.array([[0.0,0,0],[0.4,0,0]]), radius=0.4, resolution=32); (mb.n_faces>0, m.mesh_report(mb)['is_closed'])",
                          native=True, aliases=("metaball mesh", "soft blob surface", "sum of gaussians mesh",
                                                "merge blobs into a mesh", "metaballs", "blob base mesh"))
    c.register_capability("Bake a normal map (high to low)", "BAKE a normal map (optionally AO) from a HIGH-poly onto a LOW-poly's UVs -- the 'keep the sculpt detail on the retopo' step (faculty m.bake_normal_map). Per texel: find its 3-D point, project to the CLOSEST point on the high-poly, read that normal, store it. Default TANGENT-space (portable, flat = lavender 0.5,0.5,1.0); world_space=True for a raw static map; ao=True + ao_samples adds an occlusion pass. Returns an (size,size,3) image. MEASURED: a high-poly bump bakes as non-flat R/G against a lavender flat. KEPT NEG: closest-point with no cage limit (a floating detail bleeds); AO is coarse.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import grid, Mesh; low=grid(5,5,width=2.0,height=2.0); LV=np.asarray(low.vertices,float); uv=(LV[:,:2]-LV[:,:2].min(0)); uv=uv/uv.max(0); HV=LV.copy(); r=np.linalg.norm(HV[:,:2]-HV[:,:2].mean(0),axis=1); HV[:,2]=0.4*np.exp(-(r/0.5)**2); nm=m.bake_normal_map(low, uv, Mesh(HV,[tuple(f) for f in low.faces]), size=24); nm.shape",
                          native=True, aliases=("bake a normal map", "bake high poly to low poly", "normal map baking",
                                                "keep sculpt detail on the retopo", "bake ambient occlusion", "transfer detail to a texture"))
    c.register_capability("Auto-retopo (blockout to quad cage)", "AUTO-RETOPO: turn a messy BLOCK-OUT (skin_skeleton blob, metaball, boolean mess) into a clean quad-dominant cage in ONE call (m.auto_retopo): voxel_remesh COARSE (keep ~12-20) -> quad_remesh -> optional catmull_clark(subdivide). With target=, shrinkwraps onto it and scores IoU. Returns {mesh, quad_fraction, report, iou?}. ENDS the base-mesh pipeline: place joints -> skin -> auto_retopo -> clean model. MEASURED: a skinned blob -> 0.77-1.00 quad fraction, watertight. KEPT NEG: uniform topology not artist edge FLOW -- a base asset, not a hero face; quad_remesh cost rises fast with tris.",
                          example="import numpy as np, lecore, warnings; warnings.filterwarnings('ignore'); m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_meshtools import skin_skeleton; sk=skin_skeleton(np.array([[0.0,0,0],[1.0,0,0]]), [(0,1)], np.array([0.3,0.3]), resolution=16); r=m.auto_retopo(sk, voxel_resolution=10); (r['quad_fraction']>0.5, r['report']['is_closed'])",
                          native=True, aliases=("auto retopo", "automatic retopology", "blockout to quad cage",
                                                "turn a blob into quads", "clean up a blockout to a cage", "auto retopologize"))
    c.register_capability("Mesh report (topology scoreboard)", "MESH REPORT: one-call topology + shape scoreboard as a DICT (holographic_meshtools.mesh_report; faculty m.mesh_report): verts, faces, quad/tri/ngon fraction, boundary_edges (open holes/seams), nonmanifold_edges, is_manifold, is_closed, euler_characteristic, valence_histogram, regular_fraction (valence-4 for a quad mesh), bbox min/max/span, centroid. What lets an agent SEE a mesh's state cheaply and BRANCH on it -- e.g. boundary_edges>0 means fill before subdividing; quad_fraction<1 means triangulate/remesh first. Returns a dict (not a print) so it can drive logic. Deterministic.",
                          example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; r=m.mesh_report(box()); (r['quad_fraction'], r['is_closed'], r['euler_characteristic'])",
                          native=True, aliases=("mesh report", "topology scoreboard", "mesh statistics", "inspect a mesh",
                                                "quad percentage and valence", "is my mesh watertight", "mesh quality check"))
    c.register_capability("Turnaround + silhouette-IoU critic", "TURNAROUND: render a mesh from the standard views (top/front/side/3q) in ONE call and, given a ref_mesh, score how well the silhouettes MATCH per view (faculty m.turnaround). Returns {sheet, views, iou {view:IoU}, mean_iou}. IoU = intersection-over-union of the two foreground masks under the same camera; 1.0 = identical outline. THE critic loop that caught the mantis slurped legs -- now a NUMBER an agent can OPTIMISE (fix the lowest view). MEASURED: mesh vs itself 1.0 every view; half-size copy 0.22. KEPT NEG: silhouette only, blind to interior topology; pair with mesh_report.",
                          example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; r=m.turnaround(box(), ref_mesh=box(), width=64, height=64); round(r['mean_iou'],3)",
                          native=True, aliases=("turnaround render", "compare model to reference", "silhouette iou",
                                                "does my model look right", "multi-view render", "score a model against a reference",
                                                "orthographic views of a mesh"))
    c.register_capability("Proportional edit (soft grab with falloff)", "PROPORTIONAL EDIT (Blender O + G): move selected vertices and drag neighbours with a geodesic falloff, in one call (faculty m.proportional_edit(mesh, selection, translate, radius, falloff) -> new Mesh, topology unchanged). Grabbed verts move fully, neighbours ease to 0 at `radius` along the surface (falloff linear/smooth/sharp) -- reshape a whole region with ONE grab instead of moving every ring by hand. Delegates the falloff to soft_selection_weights (the geodesic engine). KEPT NEG: translate only (no rotate/scale falloff); radius is geodesic.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import grid; g=grid(8,8,width=8.0,height=8.0); V=np.asarray(g.vertices,float); c=int(np.argmin(np.linalg.norm(V[:,:2]-V[:,:2].mean(0),axis=1))); out=m.proportional_edit(g,[c],(0,0,1.5),2.5); round(float(np.asarray(out.vertices)[c,2]),3)",
                          native=True, aliases=("proportional editing", "soft selection move", "grab with falloff",
                                                "move vertices with falloff", "soft grab", "reshape a region smoothly",
                                                "pull a vertex and drag neighbors"))
    c.register_capability("Catmull-Clark subdivision (quad box modelling)", "CATMULL-CLARK subdivide (catmull_clark): m.mesh_catmull_clark(cage, levels, creases=) -- THE box-modelling subd surface (1978 masks): every face becomes quads, so a quad cage STAYS ALL-QUAD (Loop triangulates, wrong for a cage). SEMI-SHARP CREASES (DeRose 1998): creases={(vi,vj):sharpness} holds edges sharp for `sharpness` levels then smooths -- sharp edges with NO support loops (build via m.mesh_crease_edges). Chi preserved; closed stays closed. MEASURED: cube 6->24->96 all-quad, spread 0.23->0.009 smooth vs 0.15 all-creased (stays boxy). KEPT NEG: subdivision only, no closed-form limit.",
                          example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; cc=m.mesh_catmull_clark(box(),2); (len(cc.faces), all(len(f)==4 for f in cc.faces), cc.is_manifold())",
                          native=True, aliases=("catmull clark subdivision", "subdivision surface", "subdivide a quad cage",
                                                "smooth a box model", "subd modelling", "box modeling subdivision",
                                                "turn a cage into a smooth surface", "crease an edge", "semi-sharp crease",
                                                "hold an edge sharp", "sharp edge subdivision", "mark edge sharp",
                                                "auto crease sharp edges", "crease the sharp edges", "detect and crease creases"))
    c.register_capability("Dialect emitters (WGSL / C / JS / Zig from the Python kernel)", "leCore's kernels are written "
                          "once, in Python, and the browser needs them in WGSL. mind.emit_kernel(fn, dialect) walks "
                          "the same AST that code_structure decomposes and a dialect table supplies the type names, "
                          "the intrinsic names and the declaration syntax -- so the hand-written compute shader "
                          "becomes a PROJECTION of the authoritative Python kernel: one source of truth, two "
                          "runtimes, no drift. Dialects: wgsl, c_f64, c_f32, js, zig_f64, zig_f32. BOUNDED LOOPS "
                          "EMIT: `for i in range(<int literal>)` -- the shader fBm/octave shape -- with explicit "
                          "counter promotion ((double)i / f32(i) / @floatFromInt) and mutable accumulators; a "
                          "variable trip count still refuses. THE BAR IS EXECUTED, "
                          "not asserted: "
                          "mind.validate_kernel COMPILES the emitted C with cc and RUNS it on the same inputs. "
                          "MEASURED on the sphere SDF, smoothstep and cosine over 200 random inputs: c_f64 is "
                          "BIT-IDENTICAL to the Python original (same order of operations, same doubles); c_f32 "
                          "differs by 8.0e-08 to 3.4e-07. KEPT NEGATIVE 1: A WGSL KERNEL CANNOT BE BIT-IDENTICAL TO "
                          "ITS PYTHON ORIGINAL -- WGSL's f32 is single precision and NumPy is double, so the bar is "
                          "'to float tolerance' and THE TOLERANCE IS f32 EPSILON, not a number anybody chooses. "
                          "c_f32 exists so that tolerance is measured by running it. KEPT NEGATIVE 2: the emitted "
                          "WGSL is NOT executed by any test here -- there is no GPU and no browser. Its arithmetic "
                          "semantics are validated through c_f32, which shares the IR and differs only in a table; "
                          "what is NOT validated is WGSL's own precision guarantees, its fast-math latitude, or "
                          "whether the shader compiles. That is a real gap, stated. KEPT NEGATIVE 3: `bind` is NOT "
                          "emittable and that is not a missing feature -- it is a circular convolution by FFT, a "
                          "whole-array cooperative algorithm, and its WGSL is a workgroup FFT, a different artifact. "
                          "A scalar emitter that pretended otherwise would emit an O(D^2) loop nest and call it a "
                          "bind. K10's rule is obeyed throughout: the emitter REFUSES rather than guesses, because a "
                          "wrong int/double is a wrong answer at no tolerance. ZIG (opt-in, `pip install ziglang`, "
                          "numba's exact contract -- every test passes without it): validate_kernel with a zig_* "
                          "dialect compiles `-O ReleaseSafe` and RUNS. MEASURED: zig_f64 BIT-IDENTICAL on the round-"
                          "box SDF over 200 inputs; zig_f32 max 7.0e-07. KEPT NEGATIVE 4: Zig REFUSES unused locals/"
                          "params at compile time -- a dead assignment emits but will not build, and we do not "
                          "suppress that. KEPT NEGATIVE 5: ReleaseFast licenses float reassociation and is NOT the "
                          "deterministic mode. KEPT NEGATIVE 6: std.math.pow is not libm pow (measured 1-ulp gap), "
                          "so f64 bit-identity is a property of the builtin intrinsics only. The zig wheel also "
                          "backstops the C path: run_c falls back to `zig cc` when no system compiler exists.",
                          example="src = 'def sdf_sphere(px: float, py: float, pz: float, r: float) -> float:\\n"
                                  "    d = sqrt(px * px + py * py + pz * pz)\\n    return d - r\\n'; "
                                  "print(mind.emit_kernel(src, 'wgsl')); print(mind.emit_kernel(src, 'c_f32'))",
                          native=True, aliases=("emit wgsl", "wgsl emitter", "transpile a kernel",
                                                "one source of truth two runtimes", "compute shader from python",
                                                "emit c", "dialect emitter", "code generation", "kernel port",
                                                "webgpu shader", "emit zig", "zig code generation",
                                                "compile and run generated code", "native kernel",
                                                "compile a kernel to a fast binary", "zig cc fallback"), semantic="convert/emit", consumes=("sdf",), produces=("scalar",))
    c.register_capability("Native batch kernels (Zig shared library, on the fly)", "compile a scalar Python "
                          "kernel ONCE to a native .so (content-hash cached), batch-evaluate via ctypes -- Z2. "
                          "mind.zig_batch_eval runs it; mind.zig_regime_map races it against the strongest honest "
                          "baseline, the same kernel vectorized in NumPy. MEASURED verdict: a modest REAL 2-5x, "
                          "peaking near n=1e5, ~2x at n=1e6 (memory-bandwidth bound). opt='safe' f64 is "
                          "BIT-IDENTICAL to NumPy incl. the SIMD tail. Kept negatives: no 10-40x win exists "
                          "(early estimates wrong, on record); first call pays ~1-2 s compile; timings include a "
                          "per-call SoA copy. Opt-in wheel, numba's contract. See holographic_zigrun.",
                          example="src = 'def k(px: float, r: float) -> float:\\n"
                                  "    return sqrt(px * px) - r\\n'; "
                                  "print(mind.zig_batch_eval(src, [[1.5, 2.0, -0.7], [0.5, 0.5, 0.5]]))",
                          native=True, aliases=("native batch kernel", "compile kernel to shared library", "dispatcher",
                                                "when to use native code", "auto accelerate a kernel",
                                                "gpu accelerated subprocess", "fast native evaluation",
                                                "simd kernel", "on the fly native code", "zig batch",
                                                "regime map", "race against numpy", "optimized sub-process"), semantic="simulate/run", consumes=(), produces=("scalar",))
    # Z5 lives inside the same capability: mind.zig_dispatch_policy is the decision, zig_batch_eval the action --
    # one entry, because two entries for one workflow is a discoverability tax.
    c.register_capability("One kernel, two runtimes: Zig raymarcher, bit-identical", "the demoscene bar, "
                          "EXECUTED (Z4): a scene SDF written once in Python is sphere-traced by sphere_trace AND "
                          "a Zig loop compiled on the fly from the SAME text. mind.zig_march_compare marches the "
                          "same rays through both, shades both with the same code, reports. MEASURED: f64 t/hit "
                          "BIT-IDENTICAL over 110k rays x 96 steps, frames BYTE-IDENTICAL, zig 3.8x (safe==fast: "
                          "determinism is free here). Kept negative: the f32 march is a DIFFERENT PROGRAM -- a "
                          "1-ulp hit-branch flip changes the step count, so it gets a measurement, never a "
                          "tolerance. See holographic_zigmarch.",
                          example="print(mind.zig_march_compare(width=64, height=48))",
                          native=True, aliases=("zig raymarcher", "native sphere trace", "compare two renders",
                                                "one kernel two runtimes", "bit identical render",
                                                "cpu shader", "native sdf render", "march an sdf natively"), semantic="simulate/run")
    c.register_capability("Explain code in English (deterministic, layered)", "mind.explain_code(src) turns "
                          "Python source into plain English under a strict honesty contract (C1): four labeled "
                          "layers per function: signature; data flow; a control-flow census; and an idiom "
                          "layer, the only one that speaks PURPOSE, on a shape match (names/constants blanked, "
                          "so iq's box under any renaming matches) OR a min/max of registered primitives read as "
                          "a named union/intersection/subtraction (C6 composition). Unmatched: 'not recognized', "
                          "never a guess. mind.register_code_idiom + register_composition_primitive grow it. See "
                          "holographic_codeverbal.",
                          example="print(mind.explain_code('def lerp(a: float, b: float, t: float) -> float:"
                                  "\\n    return a + (b - a) * t\\n')['text'])",
                          native=True, aliases=("explain code", "explain what code does in english",
                                                "summarize a function", "describe the logic flow of a program",
                                                "find variables in source code", "what does this code do",
                                                "code to english", "verbalize code", "register code idiom",
                                                "recognize a union of shapes", "composition of primitives",
                                                "detect composed shapes", "what shapes make up this sdf"), semantic="analyze/describe", consumes=(), produces=("scalar",))
    c.register_capability("Translate kernels between languages (one IR, exact)", "mind.translate_kernel(src, from_dialect, "
                          "to_dialect) moves a kernel between python, C, WGSL, JS and Zig through the ONE "
                          "shared IR: parse back (C2's reverse parsers, inverted "
                          "from the emit tables so they cannot drift), then re-emit. THE BAR IS EXECUTED: round-trip "
                          "byte-identity over all 144 dialect pairs, asserted per pair, none sampled; hand-"
                          "written C with real precedence parses too. Refusals by name outside the kernel "
                          "grammar (K10); zigv_* is derived exhaust, not parsed. dialect= on mind.explain_code "
                          "gives English for all 7 languages through ONE verbalizer (C4). See holographic_codeparse.",
                          example="c = mind.emit_kernel('def lerp(a: float, b: float, t: float) -> float:\\n"
                                  "    return a + (b - a) * t\\n', 'c_f64'); "
                                  "print(mind.translate_kernel(c, 'c_f64', 'zig_f64'))",
                          native=True, aliases=("translate code between languages", "convert c to zig",
                                                "port a kernel", "transpile between dialects",
                                                "explain c code in english", "parse a shader back to python",
                                                "code to code translation", "round trip a kernel"), semantic="convert/emit", consumes=(), produces=("scalar",))
    c.register_capability("Kernel from a description (constrained English -> SDF)", "mind."
                          "kernel_from_description(text, name, dialect) turns a CONTROLLED-VOCABULARY description "
                          "into a geometry kernel: registered parametric forms (sphere, box, plane -- iq's exact "
                          "SDF formulae) composed with union/intersect/subtract, returned as Python or emitted "
                          "to any dialect. NOT free-form NL->code (out of scope): outside the vocabulary it "
                          "REFUSES BY NAME, and colour/material words are NOTED as ignored, not dropped -- an SDF "
                          "has no colour. mind.register_geometry_form grows it. See holographic_codecompose.",
                          example="print(mind.kernel_from_description('a sphere radius 0.4 at (1, 0, 0) union a "
                                  "floor at height -0.5'))",
                          native=True, aliases=("generate code from a description", "build an sdf from words",
                                                "english to code", "describe a shape and get a kernel",
                                                "natural language to kernel", "make an sdf from a sentence",
                                                "compose primitives from words", "text to sdf"), semantic="create/emit", consumes=(), produces=("sdf",))
    c.register_capability("Triage code in an unknown language (observations, not comprehension)",
                          "mind.triage_code(src) reports honest STRUCTURAL OBSERVATIONS about code in a language "
                          "leCore has no parser for (C5): ranked identifier word pieces (camelCase/snake_case "
                          "split), literal inventory, nesting depth, bracket balance, and a WEAK language hint "
                          "WITH its evidence. Every field is checkable against the source; NONE claims to know "
                          "what the code does -- grammar induction from one sample is a hallucination this "
                          "refuses. Triage, not comprehension: explain_code falls back here on an unknown "
                          "dialect. See holographic_codetriage.",
                          example="print(mind.triage_code('fn quicksort(xs: List) { let pivot = xs[0]; }', "
                                  "as_text=True))",
                          native=True, aliases=("analyze code in an unknown language", "triage unfamiliar code",
                                                "extract identifiers from source", "what language is this",
                                                "structural observations of code", "split camelcase names",
                                                "inspect foreign code", "code i cannot parse",
                                                "which code file should I edit", "where should I make this change",
                                                "triage a source file", "assess a code file before editing"), semantic="analyze/measure", consumes=(), produces=("scalar",))
    c.register_capability("Optional accelerators & extras (what's installed, what it buys)",
                          "mind.accelerator_report() lists every optional dependency with installed-or-not, "
                          "version, WHAT IT UNLOCKS with the measured numbers, and the exact pip command. NumPy "
                          "is the only required row. Highlights: ziglang [zig] -- native batch kernels, measured "
                          "2-5x over vectorised NumPy, 3.8x on the raymarch demo, BIT-IDENTICAL in safe mode, "
                          "one wheel, whole toolchain; pillow [images] -- jpg/webp via mind.save_render (PNG "
                          "stays stdlib on purpose); numba [jit], cupy [gpu], sympy [symbolic], flask [ui]. "
                          "All opt-in: the engine runs and passes every test with none of them.",
                          example="import json; print(json.dumps(mind.accelerator_report(), indent=1))",
                          native=True, aliases=("optional dependencies", "which accelerators are installed",
                                                "how do i speed this up", "install zig", "enable gpu",
                                                "what does pillow unlock", "pip extras", "accelerator status",
                                                "save a jpg", "make the engine faster"), semantic="analyze/describe", consumes=(), produces=("scalar",))
    c.register_capability("Canonical element + delta chain (instancing, generalised)", "a renderer's instancing says "
                          "'these two objects are the same mesh'; this says 'these two objects are the same ANYTHING, "
                          "modulo a recognised delta'. mind.canonical_form(V, family) splits an element into "
                          "(canonical, delta) with V = canonical @ A.T + b EXACTLY (1e-12); mind.recognize_elements "
                          "collapses a scene into classes; mind.canon_storage_report carries the baseline. THREE "
                          "FAMILIES, and choosing one is the whole decision: `rigid` (7-float delta) recognises "
                          "congruent copies, `similarity` (8) recognises similar copies at any size, `affine` "
                          "(3*rank + 3) collapses shape. MEASURED on 200 triangles from 5 base shapes under random "
                          "rotation + translation + scale: rigid finds 200 classes (UNDER-fits -- scale is not in the "
                          "family, so nothing matches, 0.56x), similarity finds 5 (exactly the generating family, "
                          "1.09x), affine finds 5 (0.98x). AFFINE GIVES 5 AND NOT 1 for a reason worth having: "
                          "whitening a triangle's hull makes it exactly EQUILATERAL (all three sides sqrt(6), "
                          "measured), so the shape really is collapsed -- what remains is the in-hull ROTATION, and "
                          "pinning that on a symmetric configuration needs a vertex ORDER an unordered point set does "
                          "not carry. 'Every non-degenerate triangle is affinely the same' is a statement about "
                          "ORDERED triangles. KEPT NEGATIVE: A TRIANGLE CAN NEVER PAY. Its hull is rank 2, so the "
                          "affine delta is 3*2+3 = 9 floats for a 9-float triangle -- break-even before storing a "
                          "single canonical. The dividend scales with the ELEMENT against an O(1) delta: 0.75x at 3 "
                          "vertices, 2.96x at 12, 22x at 100, 143x at 2000 -- and zlib manages only 1.04x on float64 "
                          "coordinates, so this IS a codec for large elements, unlike the same idea applied to source "
                          "code (which came out 1.12x LARGER than zlib). Per-triangle canonicalisation is a "
                          "RECOGNISER; its dividend is the dependency-keyed compute cache, not storage.",
                          example="import numpy as np; rng = np.random.default_rng(0); base = rng.normal(size=(50,3)); "
                                  "els = [base @ np.linalg.qr(rng.normal(size=(3,3)))[0].T + rng.normal(size=3) for _ in range(30)]; "
                                  "r = mind.canon_storage_report(els, 'rigid'); "
                                  "print(r['classes'], round(r['ratio'],1), r['beats_zlib'])",
                          native=True, aliases=("store a mesh as canonical plus deltas", "canonical element",
                                                "recognize that two triangles are the same up to a transform",
                                                "instancing generalized", "delta chain for geometry",
                                                "shape recognition", "congruent", "similar shapes",
                                                "canonicalize a point set", "deduplicate geometry"))
    c.register_capability("The projective ceiling (where the transform tower stops)", "compose any chain of "
                          "transform generators and you get ONE 4x4, exactly (3.3e-16 against applying the chain step "
                          "by step). So the whole transform IS the composed group element. **BUT A GROUP IS NOT A "
                          "LANGUAGE**: in a language a word is not a letter, while in a group the composition of "
                          "generators is another group element drawn from the SAME set. Words and letters live in one "
                          "alphabet -- that is what CLOSURE means, and it is why DL11's edit chain collapses to a "
                          "single (S,T) instead of needing a sequence: the recoverable object is the group element, "
                          "not the spelling. So the hierarchy is real and it is NOT letters -> words -> sentences. It "
                          "is a chain of subgroups ordered by NORMALITY: translations <| Aff(3) < PGL(4). 'Which "
                          "layer am I on' is not a question about length; it is the question 'can I push a delta "
                          "through?', and the answer is yes exactly when the layer below is normal. THE CEILING: a "
                          "4x4 is AFFINE when its bottom row is [0,0,0,1] -- when it fixes the plane at infinity. "
                          "mind.is_affine_matrix is that boolean. Conjugating a translation by a ROTATION gives "
                          "T(A t) to 1.1e-16, but conjugating it by a PERSPECTIVE gives a matrix that is not a "
                          "translation and NOT EVEN AFFINE (mind.affine_normality measures both). **Aff is a subgroup "
                          "of PGL but NOT a normal one**, and the tower's whole mechanism -- push the delta onto the "
                          "other operand, collapse the chain, read the equivariance table -- rests on normality and "
                          "stops here. TEXTURE PROJECTION IS THAT CEILING IN A RENDERER: interpolating (u,v) linearly "
                          "in screen space assumes the triangle-to-texture map is affine, and under perspective it is "
                          "not. mind.texture_projection_error, at vertex depths (1, 4, 1.5): affine max error 0.3310 "
                          "-- A THIRD OF THE TEXTURE -- against 2.2e-16 for the homogeneous (u/w, v/w, 1/w) divide. "
                          "**The extra parameter is not another letter in the same alphabet. It is an extra "
                          "COORDINATE**, carried through the transform and divided out at the end -- the `q` of a "
                          "homogeneous (u,v,q) texture coordinate. It enlarges the space the alphabet acts on, and by "
                          "doing so breaks the affine group's normality. That is why the fix is a divide and not a "
                          "matrix. KEPT NEGATIVE: a projective map is not 'affine plus a bit' -- it is linear on a "
                          "HIGHER-dimensional homogeneous space whose shadow on the affine chart is nonlinear, and "
                          "`nearest_affine` deliberately does not exist, because projecting a perspective onto the "
                          "affine subgroup throws away the only thing that made it perspective. With equal depths the "
                          "affine map is exact: the ceiling only bites under perspective.",
                          example="print(mind.affine_normality()); print(mind.texture_projection_error()); "
                                  "from holographic.mesh_and_geometry.holographic_projectivetower import projective; "
                                  "from holographic.mesh_and_geometry.holographic_grouptower import translation; "
                                  "print('affine?', mind.is_affine_matrix(mind.compose_word([translation([0.1,0.2,0.3]), projective([0.1,0,0])])))",
                          native=True, aliases=("projective transform", "homography", "perspective divide",
                                                "texture projection", "uvq", "plane at infinity",
                                                "4x4 transform", "is a word a letter", "affine ceiling",
                                                "perspective correct interpolation", "why uv needs a divide",
                                                "sketchup texture projection"))
    c.register_capability("The transform tower (which layer of the affine group)", "patterns, transformations, "
                          "rotations and scaling form a hierarchy the way letters -> words -> sentences -> document "
                          "does, and the hierarchy is the LEVI DECOMPOSITION of the affine group: Aff(n) = GL(n) "
                          "semidirect R^n, with GL(n) = center x SL(n). Bottom to top: hypervectors (the atoms); "
                          "TRANSLATION (the abelian ideal -- the content); ROTATION and SHEAR (the sl(n) part -- "
                          "non-commuting peers); SCALE (central -- commutes with the whole linear part). It is not a "
                          "picture, it makes predictions, and mind.commutator_table() checks every one: [T,T'] = 0 "
                          "(the ideal is abelian); [S,R] = [S,Sh] = 0 (scale is central in GL); [R,Sh] = 0.23 (the "
                          "peers do not commute); [S,T] = 0.49 -- SCALE IS CENTRAL IN THE LINEAR PART AND NOT IN THE "
                          "AFFINE GROUP, because s(x+t) = sx + st, so scale acts ON the ideal rather than commuting "
                          "past it. And in TWO dimensions the rotations commute with each other (SO(2) is abelian), "
                          "so 'non-commuting peers' is rotation-vs-SHEAR there and only becomes rotation-vs-rotation "
                          "in 3-D ([Rx,Ry] = 0.50). THE IDEAL IS NORMAL, and that is the whole mechanism: "
                          "mind.semidirect_law verifies A T(t) A^-1 = T(A t) to 1.1e-16 for rotation, shear and "
                          "scale. **That one line is three things this engine already found**: it is "
                          "shade_adjoint's 'push the delta onto the other operand' (conjugation); it is DL11's group "
                          "closure (why an affine edit chain collapses to one (S,T)); and it is why the equivariance "
                          "table has the shape it has -- an operator's law under a delta is a statement about which "
                          "layer the delta lives in. WHICH LAYER CAN A TRANSFORM BANK HOLD? mind.is_diagonalisable "
                          "answers by measurement: a single Fourier spectrum represents a TRANSLATION to 3.8e-16 and "
                          "a rotation to 5.4e-01 and a scale to 1.3e-01. **Exactly the ideal, and nothing above it** "
                          "-- a convolution algebra is COMMUTATIVE, so it can only represent an abelian group, and "
                          "the FPE law bind(encode(x), encode(t)) == encode(x+t) says translation IS the group "
                          "operation of the encoding. So the TransformBank is a REPRESENTATION OF THE ABELIAN IDEAL, "
                          "not a cache of transforms, and its refusal to hold a scale is the tower speaking. (Its own "
                          "'rotation' -- a cyclic index shift -- is a TRANSLATION in index space; it was never the "
                          "tower's rotation layer. The name was the bug, again.) HOW SCALE GETS IN: change the AXIS, "
                          "not the algebra. mind.mellin_promotes_scale shows a dilation is a translation on a LOG "
                          "axis -- relative error 1e-15 there against 2.81 on the linear one -- so it joins the ideal "
                          "and becomes a bind. **A layer you cannot diagonalise, you relocate.** THE ONE ENTRY "
                          "POINT is lecore.classify_transform(fn) (also mind.classify_transform): hand it ANY "
                          "callable on points and it MEASURES which floor it stands on -- {layer, name, "
                          "diagonalisable, bankable, delta_pushable, why}. It accepts a callable OR A MATRIX -- (n,n) "
                          "linear, (n,n+1) affine, or (n+1,n+1) homogeneous applied WITH the divide, so a perspective "
                          "POSTed as a 4x4 correctly classifies as beyond the affine ceiling. A matrix is data; a "
                          "callable is not, and a capability an agent cannot call does not exist. "
                          "It gets translation, rotation, shear, "
                          "scale, a perspective and a non-group nonlinearity all correct. `delta_pushable` is the "
                          "question the tower exists to answer: it is shade_adjoint's licence, DL11's closure and "
                          "the equivariance table's shape, in one boolean. And the fact is on the MAIN CLASS: "
                          "Hypervector.transform_layer() answers 'always the abelian ideal', because bind is a "
                          "circular convolution and a convolution algebra can only represent an ABELIAN group -- so "
                          "no hypervector operator can EVER be a rotation or a shear, and `permute` is not an "
                          "exception (it is a translation in INDEX space, and two permutes compose by adding their "
                          "shifts, exactly). Hypervector.commutes_with(other) measures it: 2.8e-17. "
                          "TransformBank.tower_layer() says the bank IS that ideal. lecore exports TOWER, "
                          "classify_transform, commutator_table, semidirect_law, hypervector_layer, "
                          "affine_normality, is_affine and texture_projection_error at the top level.",
                          example="import numpy as np, lecore; "
                                  "print(lecore.classify_transform(lambda x: x + np.array([0.1, 0, 0]))['name']); "
                                  "print(lecore.classify_transform(lambda x: 1.7 * x)['name']); "
                                  "print(lecore.classify_transform(lambda x: x / (1 + 0.3 * x[2]))['name']); "
                                  "print(mind.hypervector_layer()['name'])",
                          native=True, aliases=("which floor is this transform on", "classify a transform",
                                                "can I push a delta through this",
                                                "transform tower", "transform hierarchy", "levi decomposition",
                                                "affine group", "abelian ideal", "why scale is central",
                                                "commutator table", "semidirect product",
                                                "which transforms are binds", "group structure of transforms",
                                                "scale rotation translation hierarchy"))
    c.register_capability("Transform bank (a prebuilt map of hypervector transforms)", "keep the engine's transforms "
                          "-- patterns, shifts, rotations -- in a prebuilt map, held as their Fourier spectra. "
                          "mind.transform_bank(dim) gives add_random_unitary / add_rotation(k) / apply / apply_batch "
                          "/ apply_chain / power / inverse_spectrum / stats. MEASURED at D=4096: one bind costs "
                          "140.5 us of which the operand's own rfft is 39.3 us, so CACHING A SPECTRUM SAVES 28% -- "
                          "1.42x, and that is NOT the reason to build this. **COMPOSITION IS THE PAYOFF**: circular "
                          "convolution is diagonal in the Fourier basis, so a CHAIN of transforms is the PRODUCT of "
                          "their spectra and k binds collapse into ONE -- 8 sequential binds 1217.5 us against a "
                          "single composed spectrum at 90.2 us, **13.5x**, exact to 5.7e-17. That is iterate.step_k's "
                          "trick generalised from powers of ONE operator to a chain of DIFFERENT ones, and it is "
                          "DL11's group closure in the VSA algebra. A cyclic ROTATION really is a bind (verified to "
                          "1.1e-15 against np.roll), a UNITARY's inverse is its conjugate spectrum (and a Gaussian "
                          "atom's is REFUSED -- N11 measured cosine 0.744), and a POWER is a power, fractional or "
                          "huge, at constant cost. **SCALE IS NOT IN THE BANK.** A dilation is not shift-invariant, "
                          "so it is not diagonal in the Fourier basis and NO spectrum represents it: fit one on a "
                          "vector and apply it to a second and the relative error is 1.579 -- the wrong object, not a "
                          "lossy fit (mind.scale_is_not_a_bind measures it). DL11 said so; the Mellin lift makes "
                          "scale a SHIFT on a log axis, which is a different bank over a different axis. **The map is "
                          "a group representation, not a lookup table**, and refusing the transforms the algebra does "
                          "not diagonalise is the feature. KEPT NEGATIVES: composition is exact but NOT bit-identical "
                          "(5.7e-17: one inverse transform instead of k, different rounding), batching one transform "
                          "across M vectors pays only 1.6x-2.3x because the transforms dominate not the loop, and the "
                          "bank costs 1.002x the bytes of its atoms -- I guessed 2x, and an rfft of a real vector is "
                          "Hermitian, so half the coefficients are never stored.",
                          example="import numpy as np; b = mind.transform_bank(512); "
                                  "[b.add_random_unitary('t%d' % i) for i in range(4)]; b.add_rotation('rot7', 7); "
                                  "v = np.random.default_rng(0).normal(size=512); "
                                  "print(np.abs(b.apply('rot7', v) - np.roll(v, 7)).max()); "
                                  "print(b.stats(), round(mind.scale_is_not_a_bind(), 3))",
                          native=True, aliases=("transform bank", "prebuilt map of transforms",
                                                "cache a transform operator", "precomputed rotation vectors",
                                                "reuse a bind operator", "compose a chain of transforms",
                                                "spectrum cache", "group representation", "rotation as a bind",
                                                "why scale is not a bind"))
    c.register_capability("Dependency-keyed cache (key on what the operator reads)", "Part C's compute model: every "
                          "triangle is THE canonical triangle plus a recognised chain of deltas; a computation runs "
                          "on the canonical ONCE and its RESULT is transformed through the deltas, while deltas the "
                          "computation never reads -- a material, for a geometric quantity -- never enter the cache "
                          "key at all. mind.delta_cache(op, canonical, policy=...) and mind.delta_cache_report carry "
                          "the comparison. Which deltas an operator reads is MEASURED, not guessed: "
                          "mind.equivariance_table decides. MEASURED on 400 triangles (64 rotation deltas x 8 "
                          "materials; the first 400 contain 50 distinct shapes): brute 400 computes; `read_set` 50 "
                          "computes, 8.0x, BIT-IDENTICAL, because the material never enters the key; `equivariant` 1 "
                          "compute, 400x, because `area` is measured INVARIANT under rotation so the shape delta "
                          "drops out too. KEPT NEGATIVE: the equivariant path is NOT bit-identical -- max|diff| "
                          "8.3e-17. Rotating a triangle and re-integrating accumulates round-off the canonical "
                          "evaluation never incurs, so the CACHE is the one that is right and the BRUTE path carries "
                          "the error; `max_abs_diff` is reported rather than a boolean, and `equivariant` is opt-in "
                          "because this engine's constitution says a change at 1e-12 has still flipped a creature's "
                          "trajectory. C4: THE CACHE IS ONLY SOUND OVER DETERMINISTIC EVALUATORS. mind.is_deterministic "
                          "is the gate, and DeltaCache REFUSES an evaluator that draws from a global RNG stream "
                          "(measured: the same input returned 0.4019 then 0.3188) -- the cache would serve its first "
                          "draw forever while the uncached path kept drawing, and the cache would get blamed. Key the "
                          "sampler by its input's coordinates with hash_unit. PART C, END TO END: "
                          "mind.evaluate_elements(elements, op, op_name, family) takes RAW point sets with no shape "
                          "ids -- canonmesh.recognize derives the classes (C3), the equivariance table says what the "
                          "operator reads (C2), and the cache keys on that (C1). MEASURED on 200 raw triangles from 5 "
                          "base shapes: area under `similarity` is 5 classes, 5 computes, exact to 1.4e-14 -- 40x. A "
                          "COMPOSITE FAMILY'S VERDICT IS THE WEAKEST OF ITS PARTS (mind.family_verdict): area is "
                          "invariant under `rigid` but only equivariant under `similarity`, because a uniform scale "
                          "moves it. AND RECOGNITION ALONE IS NOT ENOUGH -- reusing the canonical's area directly "
                          "under `similarity` is wrong by 8.54. `max_x` finds 5 classes and still does 200 computes, "
                          "because `recompute` means there is no dividend and it says so.",
                          example="import numpy as np; from holographic.mesh_and_geometry.holographic_equivariance import area; "
                                  "rng = np.random.default_rng(0); "
                                  "bases = [rng.normal(size=(3,3)) for _ in range(5)]; "
                                  "els = [1.5*b @ np.linalg.qr(rng.normal(size=(3,3)))[0].T + rng.normal(size=3) for b in bases for _ in range(20)]; "
                                  "vals, st = mind.evaluate_elements(els, area, 'area', family='similarity'); "
                                  "print(len(els), '->', st)",
                          native=True, aliases=("evaluate elements", "cache over recognised classes",
                                                "dependency keyed memoization", "cache a computation by its dependencies",
                                                "read set of a computation", "skip work whose inputs did not change",
                                                "per triangle cache", "canonical plus delta caching",
                                                "instancing generalized", "deferred shading", "delta id",
                                                "is my evaluator deterministic", "coordinate keyed sampling"))
    c.register_capability("Canonical affine recovery (Fourier-Mellin + refine)", "recover the canonical (S, T) of "
                          "an arbitrary translate/scale edit history: after(x) = before((x - T) / S). Translate and "
                          "scale do NOT commute, and scale is not diagonal in the linear-frequency basis -- but the "
                          "family CLOSES: every order of a chain collapses to some single affine group element, and "
                          "that element is the recoverable object. mind.affine_compose(chain) is the exact group law; "
                          "mind.recover_affine(before, after) inverts it blind. THE LIFT: |FFT| discards the "
                          "translation, and resampling the magnitude spectrum onto a LOG-frequency axis turns the "
                          "dilation into a SHIFT -- so the estimator is the same cross-correlation-with-a-parabola "
                          "that est_dx uses on images (Reddy & Chatterji's Fourier-Mellin move). Scale becomes "
                          "translation; the engine already knew how to find a translation. Then a shrinking-grid "
                          "refine on the (s, t) manifold. MEASURED: 3.7e-04 scale error on a 4-edit chain, alignment "
                          "0.9995. KEPT NEGATIVE: **the SUPPORT BAND is the gate, not log-vs-plain magnitudes.** The "
                          "backlog says to use log magnitudes 'because dilation scales spectrum amplitude, which "
                          "tilts plain correlation' -- measured, that reason is wrong: multiplying one signal by a "
                          "constant scales the whole cross-correlation and leaves the argmax exactly where it was "
                          "(peak 7.0 either way). What decides it is the band: on a narrowband spectrum the log axis "
                          "is mostly noise floor, log amplifies it, and the peak pins at ZERO shift for every true "
                          "scale (1.05, 1.2, 1.5 all recover 1.00). Band to the support and both work to ~0.5%. "
                          "SECOND KEPT NEGATIVE: the group law is exact on the PARAMETERS; repeated RESAMPLING is "
                          "not. Four interpolated resamples do NOT reproduce one resample by (S, T) -- max|chain - "
                          "direct| is 0.157 at n=1024, 0.058 at 2048, 0.0045 at 8192 -- so recovery from a chained "
                          "signal fits the affine that best explains a slightly-blurred observation. AND STATE THE "
                          "UNIT: the scale lands at 3.7e-04, the SHIFT at 0.37 SAMPLES, not the 1e-4 the backlog "
                          "reports. HONEST SCOPE: 1-D. Two dimensions adds rotation and needs the log-POLAR resample "
                          "of the full Fourier-Mellin transform.",
                          example="import numpy as np; from holographic.sampling_and_signal.holographic_registration import resample_affine; "
                                  "x = np.linspace(0,1,2048); f = np.sin(2*np.pi*(20*x + 60*x**2)) * np.exp(-((x-0.5)**2)/0.06) + 0.5*np.sin(2*np.pi*180*x)*np.exp(-((x-0.3)**2)/0.005); "
                                  "S, T = mind.affine_compose([(1.03, 4.0), (0.98, -2.5), (1.05, 3.1)]); "
                                  "g = resample_affine(f, S, T); r = mind.recover_affine(f, g); "
                                  "print('true', (round(S,4), round(T,4)), '->', round(r['scale'],4), round(r['alignment'],5))",
                          native=True, aliases=("recover a scale and shift between two signals", "recover_affine",
                                                "fourier mellin registration", "estimate the dilation",
                                                "canonical affine edit", "register two signals", "image registration 1d",
                                                "log polar", "scale becomes translation", "affine group law",
                                                "edit history canonical form"))
    c.register_capability("Conformal UV unwrap (LSCM) + the metric that sees folds", "least-squares conformal maps "
                          "(Levy, Petitjean, Ray & Maillot, SIGGRAPH 2002): the angle-preserving unwrap, as ONE "
                          "linear least-squares solve on the mesh -- no iteration, no autodiff. mind.mesh_lscm(mesh) "
                          "or mind.mesh_uv_unwrap(mesh, method='lscm'). MEASURED (mean quasi-conformal ratio "
                          "sigma1/sigma2; 1.0 is conformal): a flat patch gives lscm 1.00000, isomap 1.10866, planar "
                          "1.00000 -- LSCM is EXACT on a developable surface; a hemisphere cap gives lscm 1.086, "
                          "isomap 1.878, planar 4.390. THREE KEPT NEGATIVES: (1) LSCM buys angles with AREA -- 0.4420 "
                          "area spread on the cap against isomap's 0.2957. Compare charts on the functional they "
                          "optimise, or you will conclude the wrong thing; mind.mesh_uv_report prints angle, area and "
                          "stretch for every method. (2) REPORT THE MEDIAN, not the mean: the mean quasi-conformal "
                          "ratio is unbounded -- one near-degenerate face sends sigma2 to 0, and a cap stretched 6x "
                          "in z gives LSCM a mean of 398.0 against a median of 4.8. (3) NEITHER the stretch metric "
                          "NOR the mean ratio can see a FOLD: on that stretched cap isomap has a BETTER mean (2.573 "
                          "vs 398.038) while folding 128 of 256 faces against LSCM's 72. Half its map is inverted and "
                          "every scalar summary says it is fine. mind.mesh_uv_angle_distortion reports `flipped`, and "
                          "a fold is a MINORITY orientation, not a negative determinant -- a globally mirrored chart "
                          "(classical MDS returns one routinely) has every det < 0 and no folds at all.",
                          example="from holographic.mesh_and_geometry.holographic_meshuv import flat_grid_mesh; "
                                  "m = flat_grid_mesh(6); uv = mind.mesh_lscm(m); "
                                  "print(mind.mesh_uv_angle_distortion(m, uv))",
                          native=True, aliases=("lscm", "least squares conformal maps", "conformal map",
                                                "unwrap a mesh into UV", "uv coordinates", "texture atlas",
                                                "angle distortion of a parameterization", "quasi-conformal",
                                                "does my uv map fold", "flipped triangles", "uv distortion",
                                                "parameterization"))
    c.register_capability("Progressive LOD stream (rank-ordered TT cores)", "the brain/muscle format contract: "
                          "leCore bakes, the front end consumes. mind.stream_encode(X) emits {descriptor, levels} "
                          "where every byte PREFIX is itself a valid, coarser field -- rank-ordered TT cores are a "
                          "progressive LOD. mind.stream_prefix(payload, max_bytes) picks the richest level that fits, "
                          "from the DESCRIPTOR alone (shape, dtype, full_ranks, per-level bytes, rel_rms, rel_max), "
                          "so the consumer knows what a prefix costs and what it is worth before fetching it. "
                          "mind.stream_decode reconstructs any level; mind.stream_report carries the ladder. "
                          "MEASURED on a 6-mode separable field (20^3): 6 levels, 314 B at 57% RMS error to 3,914 B "
                          "at 1.4e-15, and a 10% RMS budget costs 20.4x fewer bytes than dense. THE GUARANTEE IS IN "
                          "RMS, NOT MAX-ABS -- TT truncation is Frobenius-optimal, so adding a rank always lowers the "
                          "L2 error and can still make one voxel WORSE: on white noise the max-abs error rises at 4 of "
                          "15 levels while the RMS falls at every one. A progressive format must publish which norm "
                          "its monotonicity is in. TWO MORE KEPT NEGATIVES: the ladder is a property of the FIELD's "
                          "rank, not of the format -- white noise never reaches a 10% budget below FULL rank, where "
                          "the TT is only 1.8x smaller than dense; and a coarse level is the same shape SMOOTHED, not "
                          "a smaller field -- rank is not resolution, and a front end wanting fewer samples needs a "
                          "mip chain, which is a different object.",
                          example="import numpy as np; g = np.linspace(0,1,12); X,Y,Z = np.meshgrid(g,g,g,indexing='ij'); "
                                  "F = sum(w*np.sin((k+1)*np.pi*X)*np.cos((k+1)*np.pi*Y)*np.exp(-(k+1)*Z) for k,w in enumerate([1,.5,.25,.12])); "
                                  "p = mind.stream_encode(F); r = mind.stream_report(F, p); "
                                  "print(r['monotone_rms'], p['descriptor']['bytes'], mind.stream_prefix(p, 1000))",
                          native=True, aliases=("progressive level of detail stream", "progressive LOD", "LOD stream",
                                                "stream a field to the browser", "rank ordered payload",
                                                "truncate to a byte budget", "format contract for the front end",
                                                "brain muscle protocol", "tensor train stream", "streaming payload",
                                                "descriptor", "byte budget"))
    c.register_capability("Equivariance table (the cache policy, measured)", "for each (operator, transform) pair, "
                          "WHICH of the three mechanisms applies: INVARIANT (the delta drops out of the cache key), "
                          "EQUIVARIANT (the delta becomes a transform of the output), ADJOINT (the delta moves to the "
                          "other operand), or RECOMPUTE (no law exists). mind.equivariance_table() MEASURES it rather "
                          "than asserting it; mind.cache_policy(op, transform) turns a verdict into a key decision; "
                          "mind.classify_equivariance runs one cell. THE FINDING, and it cost two wrong cells: my "
                          "first pass reported area under shear and normal under reflection as RECOMPUTE. Both were a "
                          "MISSING LAW, not a missing law -- area(Ax) = |det A| * ||A^-T n|| * area(x) and "
                          "normal(Ax) = sign(det A) * normalize(A^-T n), each exact to 1e-12 for every affine family. "
                          "**RECOMPUTE must mean NO LAW EXISTS, not 'I did not write one down'** -- a table that says "
                          "recompute where a law exists is a cache that never fires, and it looks exactly like a "
                          "table that is merely honest. AND THE READ-SET IS THE POINT: area's law reads the NORMAL, "
                          "so the key must carry the normal's class too. Every non-rigid law here reads the normal. "
                          "THE ADJOINT, corrected: Part C's 'shade a rotated triangle by unrotating the light', "
                          "shade(Ax, L) == shade(x, A^-1 L), is exact for a ROTATION (3.9e-16) and WRONG for "
                          "everything else -- including a plain uniform scale, by 0.38, because the normal is "
                          "renormalised and the scale does not cancel. mind.shade_adjoint carries the correction, "
                          "which (again) reads the normal. `max_x` is registered as a genuine recompute case so the "
                          "negative branch is exercised by something real: which vertex attained the maximum is "
                          "information the scalar threw away.",
                          example="t = mind.equivariance_table(); print(t['area']); "
                                  "print(mind.cache_policy('area', 'shear')); print(mind.cache_policy('max_x', 'rotate'))",
                          native=True, aliases=("equivariance table", "equivariance", "invariance",
                                                "is this operator invariant under rotation", "cache policy",
                                                "does the delta drop out of the cache key",
                                                "transform the result not the input", "which cache policy applies",
                                                "adjoint transfer", "canonical plus delta", "jacobian law",
                                                "unrotate the light"))
    c.register_capability("Cloud stack (closed-form shadow rays)", "single-scattered volumetric clouds assembled "
                          "from shipped parts: volint's CLOSED-FORM line integral over an FPE density field, plus the "
                          "renderer's Henyey-Greenstein phase. mind.cloud_transmittance is Beer-Lambert on a tau that "
                          "costs ONE inner product per ray -- no marching. mind.cloud_single_scatter marches the VIEW "
                          "ray (it must: the integrand contains the transmittance being accumulated) and evaluates "
                          "every SHADOW ray in closed form. THE CLOSED FORM PAYS ON THE SHADOW RAY, and it is not a "
                          "speed-for-accuracy trade: MEASURED at 64 rays x 32 view steps against a 64-step marched "
                          "shadow, the closed form uses 32 density evaluations against 2,080 (65x fewer), runs 52x "
                          "faster, and is 13x MORE ACCURATE (3.03e-07 vs a 16-step march's 3.94e-06) -- because it is "
                          "the exact integral and the march is the one carrying error. mind.cloud_report carries the "
                          "comparison. HONEST SCOPE: the view integral still marches (volint's own note: absorption "
                          "does not want marching, scattering still does); multiple scattering is not here; and the "
                          "closed form's physical SCALE is a fitted constant whose accuracy is that of the short "
                          "march it was calibrated against (3.5e-05 at calibration_steps=24, 5.1e-07 at 256). "
                          "`optical_depth` takes a PER-RAY L -- passing a median instead is a 1000x accuracy loss.",
                          example="import numpy as np; from holographic.misc.holographic_volint import HolographicVolume; "
                                  "from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder; "
                                  "rng = np.random.default_rng(0); enc = VectorFunctionEncoder(3, dim=256, bounds=[(-1,1)]*3, bandwidth=2.5, seed=0); "
                                  "vol = HolographicVolume.from_blobs(enc, rng.uniform(-0.5,0.5,size=(16,3)), calibration_steps=96); "
                                  "O = np.stack([np.full(8,-0.95), np.zeros(8), np.linspace(-0.2,0.2,8)], axis=1); D = np.tile([1.,0,0],(8,1)); "
                                  "print(mind.cloud_report(vol, O, D, 1.9, (0,1,0), ceiling=0.95, view_steps=8, reference_shadow_steps=32))",
                          native=True, aliases=("render a cloud", "cloud stack", "volumetric clouds",
                                                "closed form transmittance", "shadow ray without marching",
                                                "single scattering", "henyey greenstein", "beer lambert",
                                                "fog", "atmosphere", "participating media", "optical depth",
                                                "volumetric scattering integration", "analytic segment integral",
                                                "energy conserving fog accumulation", "frostbite volumetric integration",
                                                "fewer steps for the same volume quality",
                                                "reduce volumetric banding at low step counts"))
    c.register_capability("Points to mesh (isosurface / surface reconstruction)", "the inverse of "
                          "sdf_surface_points, which the engine could do in one direction only. "
                          "mind.sdf_from_points(points, normals, lo, hi, res) builds a signed distance grid from an "
                          "ORIENTED point cloud -- distance to the nearest sample, signed by that sample's normal -- "
                          "and mind.surface_nets(field, grids) extracts a WATERTIGHT quad mesh by dual isosurface "
                          "extraction: one vertex per sign-changing cell at the mean of its edge crossings, one quad "
                          "per sign-changing grid edge. mind.points_to_mesh runs both; mind.mesh_report scores it "
                          "(watertight, Euler characteristic, max surface error). MEASURED on a unit sphere, 600 "
                          "samples, 32^3 grid (cell 0.1032): 1,804 vertices, 1,802 quads, watertight, Euler = 2, max "
                          "vertex error 0.0454 -- 0.44 CELLS, and the cell size is the honest baseline because a dual "
                          "extractor cannot place a vertex better than the cell it lives in. HONEST SCOPE: this is "
                          "naive surface nets, NOT Dual Contouring -- averaging the crossings rounds off SHARP "
                          "features, which DC's QEF solve (Ju et al., SIGGRAPH 2002) recovers. Smooth surfaces. "
                          "THREE KEPT NEGATIVES: (1) the point-cloud SDF is LEAST accurate near the surface, exactly "
                          "where the extractor reads it (max err 0.2225 within 0.1 of the surface, 0.0695 beyond 0.6) "
                          "-- distance-to-nearest-SAMPLE overestimates distance-to-SURFACE by up to the sample "
                          "spacing; (2) accuracy is set by the CLOUD, not the grid -- the near-surface error tracks "
                          "the spacing at 1.3-1.7x, so refining the grid under a sparse cloud buys nothing; (3) the "
                          "MESH is watertight AND ORIENTED -- every directed edge once, every normal along +grad -- "
                          "which `watertight` alone cannot see: mind.mesh_is_oriented(quads) is the stronger check, "
                          "and before it existed the sphere was watertight with 228 duplicated directed edges and 98 "
                          "of 200 normals pointing inward, so Mesh.half_edges() refused it. Orienting needs TWO sign "
                          "flips composed (the crossing's direction AND the frame's parity, since (1,0,2) is an odd "
                          "permutation); fixing only the first left 136 of 408 normals outward. The "
                          "MESH is 4.7x more accurate than the FIELD it came from, because averaging twelve edge "
                          "crossings cancels per-sample noise -- do not read one error as the other, in either "
                          "direction. The all-pairs distance matrix is chunked: unchunked, a 24^3 grid against 9,600 "
                          "points allocated 133M floats and the process was killed.",
                          example="import numpy as np; rng = np.random.default_rng(0); "
                                  "p = rng.normal(size=(400,3)); p /= np.linalg.norm(p, axis=1, keepdims=True); "
                                  "V, Q, F, g = mind.points_to_mesh(p, p, np.full(3,-1.6), np.full(3,1.6), 20); "
                                  "print(mind.mesh_report(V, Q, sdf=lambda X: np.linalg.norm(X,axis=1)-1.0))",
                          native=True, aliases=("marching cubes", "dual contouring", "surface nets", "isosurface",
                                                "convert splats to a mesh", "surface reconstruction from points",
                                                "sdf from a point cloud", "point cloud to mesh", "mesh from an sdf",
                                                "extract a surface", "watertight mesh", "poisson reconstruction"))
    c.register_capability("Fill the gaps in a field (inpaint / impute)", "fill the unknown cells of a field, "
                          "dispatched on TYPE. mind.inpaint(field, known) sends a float array to a harmonic (Laplace) "
                          "solve -- each hole relaxes to the mean of its four neighbours, known cells pinned -- and an "
                          "integer array to a majority neighbour vote, because a discrete field has no mean and "
                          "averaging it is a category error. mind.fill_report scores ON THE HOLES ONLY. MEASURED "
                          "(48x48, 59% erased, 8 seeds): harmonic MAE 0.0015 mean (range 0.0012-0.0018); majority "
                          "accuracy 0.9653 mean (0.9553-0.9749), and 0.9990 in region INTERIORS -- nearly all the "
                          "error is boundary error, so the overall number is a property of the FIELD while the "
                          "interior number is a property of the ALGORITHM. THE BOUNDARY CONDITION IS THE GATE: "
                          "periodic=False (edge-clamped) is the default, because wrapping a non-periodic field with "
                          "np.roll solves a different problem and costs 5.4x (MAE 0.00666 vs 0.00123). DECLARED "
                          "NEGATIVES, measured, do not rebuild them: a VSA record (one vector per cell, roles bound "
                          "per channel) LOSES to both of these on both channels -- temperature MAE 0.0248 vs harmonic "
                          "0.0077, material accuracy 94.2% vs majority 96.0%; per-step cleanup in a multi-role NCA "
                          "DOUBLES the continuous error (0.0248 -> 0.0485) for zero categorical benefit, because "
                          "cleanup is per-role but the bundle is shared; and merely encoding a scalar into a 2-role "
                          "record and reading it back costs MAE 0.0160, more than twice what a harmonic solve achieves "
                          "while actually reconstructing missing values.",
                          example="import numpy as np; N = 32; y, x = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N), indexing='ij'); "
                                  "f = 0.3*x + 0.4*np.exp(-((x-0.6)**2 + (y-0.3)**2)/0.05); "
                                  "known = np.random.default_rng(0).random((N,N)) > 0.5; "
                                  "print(mind.fill_report(f, mind.inpaint(f, known), known))",
                          native=True, aliases=("inpaint", "inpaint a hole", "impute missing values",
                                                "fill in missing data", "label propagation", "hole filling",
                                                "missing data", "impute", "fill gaps in a field",
                                                "extrapolate a sparse field", "harmonic inpainting",
                                                "laplace solve on holes", "majority vote fill", "gap filling"))
    c.register_capability("Frame-to-frame motion by one unbind (reprojection velocity)", "recover the translation "
                          "between two frames with ONE unbind: cross-correlation in the Fourier domain is "
                          "conj(F(a))*F(b), and its peak is the shift. This is TAA's analytic reprojection velocity, "
                          "and it is the engine's core operator applied to images. mind.est_dx(a, b) returns (dy, dx) "
                          "to sub-pixel precision; mind.reproject(a, b, tile=None) warps a forward to predict b; "
                          "mind.reproject_report(a, b) carries every baseline. MEASURED on a REAL rendered frame "
                          "warped by a known amount: 0.0705 px mean error, 0.1087 px worst; integer shifts exact. "
                          "FOUR KEPT NEGATIVES, all measured: (1) `normalize=True` -- textbook PHASE correlation -- is "
                          "2.3x WORSE at sub-pixel, because it sharpens the peak toward a delta and a parabola needs "
                          "curvature -- and WHITE NOISE is the worst case for the same reason, its autocorrelation "
                          "being a delta; (2) a Hann window, the textbook wrap-bias fix, is worse still (2.05 px, 1.17 px "
                          "even after mean removal); (3) the residual is THE SCENE, not estimator error -- warping "
                          "lifts a lateral pan from 23.23 dB to 36.84 dB but plateaus. With the camera FIXED and the "
                          "scene moving (the only non-vacuous control -- a far-away camera makes the two frames "
                          "IDENTICAL, and warping nothing perfectly proves nothing), two spheres at the SAME depth "
                          "gain 11.65 dB from a warp while the same slide at DIFFERENT depths gains 6.06 dB: parallax "
                          "halves what one translation can explain, and a depth slide (a scale change) gains only "
                          "5.48 dB; (4) TILING LOSES ON UNIFORM MOTION "
                          "(pan: 40.46 dB global vs 36.67-40.67 tiled) and wins only on a non-uniform field (dolly: "
                          "34.82 vs 37.43 at tile 48) -- and the per-tile shift SPREAD does NOT tell you which regime "
                          "you are in (a pure translation has the largest spread and global still wins by 22 dB). So "
                          "the backlog's 'one unbind per tile INSTEAD of motion vectors from geometry' does not hold: "
                          "the unbind is an excellent ESTIMATOR, not a substitute for knowing how the camera moved.",
                          example="import numpy as np; from holographic.rendering.holographic_reproject import warp; "
                                  "x = np.linspace(0, 6, 64); a = np.outer(np.sin(x), np.cos(1.7*x)) + 0.3*np.outer(x, x[::-1]); "
                                  "b = warp(a, 1.4, -2.6, wrap=True); "
                                  "print('truth (1.4, -2.6) ->', np.round(mind.est_dx(a, b), 3))",
                          native=True, aliases=("est_dx", "reprojection velocity", "motion vectors between frames",
                                                "estimate the shift between two images", "phase correlation",
                                                "temporal reprojection", "TAA", "optical flow", "image registration",
                                                "subpixel shift", "warp the previous frame", "frame prediction"))
    c.register_capability("Code as canonical shape + name delta (exact, not a codec)", "a statement is (canonical "
                          "SHAPE) + (name DELTA): erase the identity-carrying leaves -- names, attributes, constants, "
                          "argument names -- and what remains is pure structure; what you erased is the delta. Part "
                          "C's triangle, applied to code. mind.code_decompose(stmt) splits it, mind.code_recompose "
                          "inverts it EXACTLY (a delta of the wrong length RAISES rather than short-reading into "
                          "plausible wrong code), mind.code_structure(src) / mind.code_rebuild(cb, stream) do a whole "
                          "module, and mind.code_shape_census(src) measures the split. THE BAR, MET: 63,121 of 63,121 "
                          "statement subtrees reconstruct bit-exactly, and 421 of 421 modules rebuild to a "
                          "byte-identical normalized source -- 'normalized' being precise, because ast.unparse is a "
                          "FIXED POINT on every module here and the reparsed AST is identical. MEASURED census: "
                          "identifiers kept 1.19x reuse, identifiers erased 2.34x -- erasing them collapses ~49% of "
                          "distinct statements. STATE THE UNIT WITH THE NUMBER: the same census over FUNCTIONS reads "
                          "1.13x, and reading one as a refutation of the other is a unit error. KEPT NEGATIVE: this is "
                          "NOT a compressor. mind.code_byte_report(src) reports the structure at 1.12x LARGER than "
                          "zlib on the whole tree, because 83.2% of shapes occur exactly once -- code's tail is long. "
                          "The shape is a semantic KEY (structural search, duplicate detection, refactor targeting), "
                          "and never a cache key.",
                          example="import ast; tmpl, delta = mind.code_decompose('total = a + 7'); print(delta); "
                                  "print(ast.unparse(mind.code_recompose(tmpl, ['x', 'y', 9])))",
                          native=True, aliases=("code structure", "canonical shape and name delta",
                                                "decompose code into shape and names", "ast round trip",
                                                "reconstruct source from a structure", "statement shape",
                                                "structural search", "find duplicate code", "shape census",
                                                "code as canonical plus delta", "exact ast decomposition"))
    c.register_capability("Selftest coverage census (which modules have a real _selftest)",
                          "which engine modules carry a real _selftest and which advertise a __main__ but assert "
                          "nothing (a false green -- and the exact backfill worklist). mind.selftest_coverage() "
                          "returns {runnable, missing, missing_modules, coverage} by a pure AST scan (no import, no "
                          "subprocess), so an agent driving the engine can ask 'is the codebase covered by its own "
                          "selftests?' without shelling out. The actual RUN of every selftest is the CLI/CI tool "
                          "tools/run_selftests.py; this is the instant census behind it, and it exists because an "
                          "above/below sweep found the walker had no mind door.",
                          example="c = mind.selftest_coverage(); print(round(c['coverage'], 3), c['missing'])",
                          native=True, aliases=("selftest coverage", "which modules lack a selftest",
                                                "test coverage census", "modules missing tests",
                                                "is the engine covered by tests", "which modules have no selftest",
                                                "audit test coverage", "self test census", "untested modules"))
    c.register_capability("Memoize a pure function (the purity gate is the point)", "skip re-execution of PURE work "
                          "whose inputs repeat. mind.memoize_pure(fn) keys on (the function's EXACT canonical source, "
                          "its arguments) and REFUSES a function that is not pure -- is_pure rejects the clock, RNG, "
                          "IO, global writes, and transitive impurity through a call-graph fixpoint, while accepting a "
                          "locally-allocated container. A cache over an impure function returns a stale answer "
                          "silently, so the gate raises instead. MEASURED: 36x on a repeated 256x256 SVD, "
                          "bit-identical. THE BACKLOG CALLS THIS 'shape-keyed memoization', AND THAT NAME IS A BUG: a "
                          "canonical shape erases identifiers and constants, so `def f(x): return x + 1` and "
                          "`def g(x): return x + 2` have the SAME shape and would share a cache entry. "
                          "mind.canonical_shape(fn) exists, and is a COMPRESSION primitive, never a cache key. "
                          "KEPT NEGATIVE: the key costs O(input bytes) -- fingerprinting a 512x512 array costs 1.747 "
                          "ms while A.sum() costs 0.084 ms, so a cheap function of a large array loses 21x; ask "
                          "mind.machine_place with the function's own cost as the baseline. TWO BACKLOG NUMBERS DID "
                          "NOT REPRODUCE: shape reuse is 1.13x (node type + depth) or 1.87x (control flow), not 2.36x "
                          "-- it is a property of the equivalence relation, not the code; and tree purity is 35.4% "
                          "(781 of 2,188 module-level functions), not 76%. HONEST SCOPE: the gate resolves callees "
                          "within ONE module, so a function that calls an IMPORTED helper is refused as unresolved "
                          "(sound, and why tucker.rank_gate is rejected -- it reaches fix_eigvec_signs from another "
                          "module). Cross-module resolution wants types.",
                          example="import numpy as np; from holographic.simulation_and_physics.holographic_island import island_energy; "
                                  "f = mind.memoize_pure(island_energy); X = np.zeros((64,3)); V = np.ones((64,3)); "
                                  "f(X, V); f(X, V); print(f.cache_stats())",
                          native=True, aliases=("memoize", "memoize a pure function", "cache a function keyed on its inputs",
                                                "skip repeated work", "pure function cache", "content addressed memoization",
                                                "shape keyed memoization", "canonical shape of a function",
                                                "is it safe to cache this", "lru cache but safe", "purity gate"))
    c.register_capability("Scatter / gather (any rank, any kernel, exact on demand)", "deposit values onto a grid of "
                          "ANY rank at continuous coordinates, and read them back through the SAME kernel -- scatter "
                          "and gather are adjoint. mind.scatter(points, values, shape, kernel=) is rank-agnostic "
                          "(verified 1-D through 4-D), mass-preserving (a partition-of-unity kernel's weights sum to "
                          "1), handles vector values (N,C), and clamps or wraps at the edges. kernel='nearest' is the "
                          "GPU's scatter -- an atomic add at an index, ties rounding UP by stated convention -- and "
                          "scattering ones at integer coordinates IS np.bincount, so a nearest scatter is a HISTOGRAM. "
                          "kernel='bilinear' spreads over 2^D cells; 'bspline' is the smooth MPM kernel. "
                          "mind.scatter_exact(...) is PERMUTATION-INVARIANT: a scatter is a reduce PER CELL and "
                          "np.add.at accumulates in point order, so a float scatter of the same points reordered gives "
                          "a different grid -- MEASURED, 4,000 points onto 16x16 with weights spanning 16 orders of "
                          "magnitude, the float scatter differs by 1.12e-08 under a permutation (9.31e-09 for a "
                          "nearest histogram) and the exact one does not differ at all. scatter_to_field and "
                          "scatter_to_field_3d are the graphics doors onto this same function.",
                          example="import numpy as np; idx = np.random.default_rng(0).integers(0, 8, size=200); "
                                  "hist = mind.scatter(idx[:, None].astype(float), np.ones(200), (8,), kernel='nearest'); "
                                  "print(np.array_equal(hist, np.bincount(idx, minlength=8).astype(float)))",
                          native=True, aliases=("scatter", "gather", "scatter add", "atomic add",
                                                "scatter values into an output array", "deposit particles onto a grid",
                                                "accumulate into arbitrary indices", "order independent scatter",
                                                "splat values to a grid", "histogram", "histogram of values",
                                                "bincount", "particle to grid", "grid to particle", "P2G", "G2P",
                                                "adjoint of sampling", "deterministic histogram"))
    c.register_capability("The machine model (leCore's hardware units + memory tiers)", "THE SPEC SHEET, and the "
                          "first thing to read before building anything that smells like a cache, a kernel, a "
                          "scheduler or a lookup -- the odds are the unit already exists and has a measured cost "
                          "model. mind.machine_map() lists every COMPUTE unit (SIMD lanes, SIMT width, texture unit, "
                          "gather, kernel fusion, batched operator power, RT core, per-thread RNG, atomics-free wave "
                          "scheduler, occupancy gate) and every MEMORY tier (compiled operator, fat-margin cache, "
                          "baked grid, content-addressed cache, compressed RAM, cold store, durable delta chain), "
                          "each with the real module+symbol, its setup cost, its marginal cost, how that marginal "
                          "cost SCALES, and the conditions under which it must NOT be used. mind.machine_place(...) "
                          "answers the only question that matters -- does the work amortize the setup -- and returns "
                          "break_even_n = inf when a unit can NEVER pay. mind.machine_spec_sheet() re-MEASURES all 17 "
                          "units on your box (a spec sheet that cannot re-measure itself is a rumour), and "
                          "mind.machine_place_unit(name, baseline_ns, n_calls) runs the placement on those MEASURED "
                          "numbers rather than on ones you remembered. CAVEAT, and it is the program's oldest error: "
                          "`baseline_ns` must be the cost of what the unit REPLACES -- kernel_fusion replaces N passes, "
                          "gather replaces N fetches. Priced against a raw array read (130 ns) almost every unit "
                          "correctly reports NEVER; if everything says never, check the denominator. "
                          "KEPT NEGATIVE, measured: the textbook latency ladder (registers < L1 < L2 < RAM) is FALSE "
                          "here -- a dense array index (132 ns) beats the fat-margin cache (3,485 ns) and the texture "
                          "unit (376,032 ns) on a single scalar access, because NONE of these is a scalar unit. They "
                          "are BATCH units: BakedGrid costs 61,765 ns/point at N=1 and 274 ns/point at N=10,000, and "
                          "`gather`'s marginal cost is CONSTANT in N (182,010x at N=2,048 -- when the rule is reused).",
                          example="sheet = mind.machine_spec_sheet(); "
                                  "print(mind.machine_place_unit('t2_baked_grid', baseline_ns=50_000, n_calls=10**6, sheet=sheet)); "
                                  "print(mind.machine_unit('gather_unit')['do_not_use_when'])",
                          native=True, aliases=("machine model", "hardware units", "spec sheet", "cost model",
                                                "what hardware units does this engine have", "gpu equivalent",
                                                "what is the gpu equivalent here", "memory hierarchy", "cache hierarchy",
                                                "which tier should my data live in", "is it worth caching this",
                                                "break even for a cache", "should I bake this or compute it each time",
                                                "amortize", "setup vs marginal cost", "L1 L2 L3", "registers",
                                                "warp", "texture unit", "rt core", "tensor core", "occupancy",
                                                "which unit should I use", "pattern to use"))
    c.register_capability("Compressed-domain compute (never touch the decompressed field)", "blur, add, scale and "
                          "query a 2-D field by operating on its rank-r FACTORS, never forming the array. "
                          "mind.low_rank_field(X) returns a LowRankField with .blur(kernel_1d) / .add(other) / "
                          ".scale(a) / .query(i,j) / .to_dense(); mind.worth_factoring(X) is the honest gate; "
                          "mind.factored_field_report(X, k) re-runs the comparison for you. The bandwidth wall is "
                          "physics (this box reads ~12.3 GB/s, a GPU's HBM does 1-3 TB/s) -- you do not out-bandwidth "
                          "a GPU, you flank it by never touching decompressed data. MEASURED (1024x1024 smooth field, "
                          "rank 3, 171x fewer bytes): separable blur 66.60 ms / 8.4 MB dense vs 2.53 ms / 0.049 MB "
                          "factored, error 3.11e-15; add two fields 16.8 MB vs 0.066 MB, error 5.83e-14; a point query "
                          "takes 1.7 us and 72 bytes against materialising 8.4 MB. FOUR KEPT NEGATIVES: (1) the blur "
                          "must be SEPARABLE -- a 2-D kernel is outside the algebra and is REFUSED, not approximated; "
                          "(2) add inflates rank (six naive adds take rank 2 -> 14) so it recompresses, lossily, at a "
                          "tolerance; (3) NONLINEAR ops do not survive -- ReLU on factors differs from ReLU on the "
                          "field by 1.283, so clamp/threshold/min/max need to_dense(); (4) if the field is not low "
                          "rank, factoring COSTS more -- white noise gates to rank 197 of 256 and worth_factoring "
                          "returns False. WIRED (B2) as fieldhome.Field.low_rank, a fourth backend beside "
                          "callable/dense/sparse. AND THE GATE IS AN ERROR BUDGET, not rank_gate's 99% ENERGY: "
                          "measured on REAL fields (SDF slices, not synthetic outer products), a sphere SDF at 99% "
                          "energy is rank 2 and 7.45% WRONG, a box SDF 18.19% wrong, and fbm noise passes the energy "
                          "gate at rank 5 with 28.54% error -- an SDF that wrong does not sphere-trace. Use "
                          "mind.rank_for_error(X, max_abs_error) and mind.worth_factoring(X, max_error=...): at 1% of "
                          "amplitude a sphere SDF needs rank 4 (16x fewer bytes, pays), a box SDF rank 12 (5.3x), fbm "
                          "rank 50 (1.27x, marginal), white noise rank 124 (refused). DEFERRED for postfx: it STREAMS "
                          "frames, so an SVD costs 53.7x the FFT blur it would accelerate at 128^2 and 91.7x at 256^2 "
                          "-- LowRankField pays where a field is baked once and queried many times.",
                          example="import numpy as np; x = np.linspace(0,1,256); "
                                  "X = np.outer(np.sin(3*np.pi*x), np.cos(2*np.pi*x)); "
                                  "k = np.array([1.,4,6,4,1]); k /= k.sum(); "
                                  "print(mind.factored_field_report(X, k)); print(mind.worth_factoring(X))",
                          native=True, aliases=("compressed domain", "low rank field", "operate on factors",
                                                "blur a field without decompressing it", "factored ops",
                                                "operate on tensor train cores directly", "never decompress",
                                                "add two compressed fields", "query a compressed field at a point",
                                                "bandwidth wall", "low rank factorization of a field", "svd field",
                                                "separable blur", "compressed compute", "rank gate"))
    c.register_capability("Hierarchical superposition (cleanup between levels)", "hold far more items in one vector "
                          "than the flat capacity law allows, by cleaning up BETWEEN levels. mind.hierarchical_pack "
                          "superposes G group-keyed chunks; mind.hierarchical_recall unbinds the group key, SNAPS the "
                          "noisy chunk to its exact pattern in a chunk codebook (the crosstalk reset), then unbinds the "
                          "leaf; mind.flat_recall is the baseline it must beat, shipped beside it. MEASURED (D=2048, 8 "
                          "items/group, 16 shared patterns): flat recall 100% / 90% / 56.7% / 18.3% at G = 4 / 16 / 32 / "
                          "64 groups, while hierarchical recall stays at 100% throughout. Capacity is bounded by the "
                          "WORST SINGLE LEVEL, not by the product of levels. KEPT NEGATIVE (theorem-shaped): "
                          "superposition is LINEAR, so naive bundle-of-bundles with product roles IS one flat bundle -- "
                          "measured identical to 2.78e-16. Nesting alone buys nothing; the mid-level cleanup is the "
                          "entire mechanism. SECOND NEGATIVE, correcting the backlog: shared chunks do NOT buy recall "
                          "(64 distinct patterns for 64 groups still recalls 100%) -- they buy a SMALL CODEBOOK, 16 "
                          "patterns instead of 64, and that is where R1's promoted chunks pay. Say it plainly: the "
                          "single vector holds the STRUCTURE, the codebooks hold the content. "
                          "R3 -- THE ONE CODEBOOK FAMILY, third consumer: mind.chunk_codebook_vectors(codebook, items, "
                          "leaf_keys) turns R1's LEARNED chunk codebook (mind.learn_chunks) into these chunk vectors. "
                          "R1 learns WHICH chunks recur; R2 realizes each as a map_bind product; this realizes each as "
                          "a pack superposition -- same identities, different vectors. Reproduced on a learned "
                          "codebook: flat 100/95/70/30 at G=4/16/32/64, hierarchical 100/100/100/100. "
                          "THIRD NEGATIVE, and it is the dangerous one: if a group is NOT in the chunk codebook (R1 "
                          "was allowed too few merges), the mid-level cleanup snaps to the NEAREST entry -- the wrong "
                          "chunk -- and returns an item with every appearance of success. Measured: uncovered group "
                          "chunk_similarity 0.036, covered 0.502. Pass min_chunk_similarity=0.15 to ABSTAIN instead of "
                          "lying, and mind.chunk_coverage(...) tells you the fraction at risk (60 merges covered 8 of "
                          "16 groups; 150 covered all 16).",
                          example="import numpy as np; from holographic.agents_and_reasoning.holographic_ai import unitary_vector; "
                                  "from holographic.misc.holographic_superposed import pack; r = np.random.default_rng(0); "
                                  "at = lambda n: np.stack([unitary_vector(512, r) for _ in range(n)]); "
                                  "lk, gk, items = at(4), at(8), at(16); "
                                  "chunks = np.stack([pack(lk, items[p*4:(p+1)*4]) for p in range(4)]); "
                                  "S = mind.hierarchical_pack(gk, chunks[[0,1,2,3,0,1,2,3]]); "
                                  "r = mind.hierarchical_recall(S, gk[3], lk[2], chunks, items, min_chunk_similarity=0.15); "
                                  "print(r['item_index'], r['abstained'])",
                          native=True, aliases=("hierarchical superposition", "chunked memory", "mid-level cleanup",
                                                "cleanup between levels", "store many items in one vector and recall them",
                                                "how many items can i bundle before recall fails", "capacity",
                                                "chunked memory with a shared codebook", "bundle of bundles",
                                                "nested superposition", "crosstalk reset", "recall capacity",
                                                "two level memory", "group and leaf"))
    c.register_capability("Learned chunk codebook (iterated pair promotion)", "learn the RECURRING CHUNKS of a symbol "
                          "stream by iterated pair promotion (BPE -- Gage 1994; Sennrich et al. 2016), where the merged "
                          "chunks are factoring and storage codebooks, not tokenizer vocabulary. mind.learn_chunks(stream) "
                          "returns a plain-data codebook; mind.chunk_encode / mind.chunk_decode round-trip it LOSSLESSLY; "
                          "mind.structure_score(stream) is the one-number probe for whether a stream has reusable "
                          "structure at all. THE ONE CODEBOOK FAMILY (R3): the same codebook feeds recursive factoring "
                          "(R2), hierarchical superposition's mid-level cleanup (W5) and the edit codec (DL8) -- three "
                          "consumers, one structure. MEASURED: a workflow stream of 6,000 symbols tokenizes to 1,392 "
                          "(4.3x) with mean chunk depth 4.31 and max depth 16; a uniform control stalls at 1.3x, mean "
                          "depth 1.34, max depth 2. No structure, no recursion dividend -- and this measures it before "
                          "anything is built on top. KEPT NEGATIVE: it is NOT a byte compressor. On the same stream zlib "
                          "takes 1,820 bytes and the codebook+tokens take 3,578; mind.chunk_byte_report(...) reports both "
                          "so the token ratio cannot be mistaken for a compression claim. Deterministic: count ties break "
                          "on the pair, never on dict insertion order.",
                          example="from holographic.agents_and_reasoning.holographic_chunkcodebook import workflow_stream; "
                                  "s = workflow_stream(); cb = mind.learn_chunks(s); "
                                  "assert mind.chunk_decode(mind.chunk_encode(s, cb), cb) == s; print(mind.chunk_stats(s, cb))",
                          native=True, aliases=("chunk codebook", "bpe", "byte pair encoding", "pair promotion",
                                                "learn a codebook of repeated pairs from a stream", "chunk promotion",
                                                "tokenize a sequence into learned chunks", "find repeated motifs in a sequence",
                                                "repeated motifs", "does my data have repeating structure",
                                                "structure probe", "reusable chunks", "macro codebook",
                                                "promote frequent chunks", "learned vocabulary", "sequence chunking",
                                                "recursion dividend", "shared codebook"))
    c.register_capability("Physics event codec (a trace as base + interruptions)", "record a simulation as its BASE "
                          "state plus its EVENTS -- the impulses and contacts where the deterministic flow was "
                          "interrupted -- and regenerate everything else. Between events physics is a deterministic "
                          "function of the state, so the states were never data. mind.record_physics_trace(...) gives "
                          "(trace, EventTrace); mind.replay_physics_trace(ev) reconstructs it BIT-IDENTICALLY; "
                          "mind.physics_compression_report(trace, ev) reports the codec's size beside every baseline it "
                          "claims to beat, so the comparison travels with the capability. MEASURED (600 frames x 16 "
                          "bodies, 663 events): raw 460,800 bytes; zlib(raw) 308,090; zlib(frame deltas) 87,057; EVENT "
                          "CODEC 6,360 -- 13.7x over the bar, and lossless. KEPT NEGATIVE 1: the win is event SPARSITY "
                          "(663 events replace 9,600 state rows), NOT a codebook -- a quantized impulse codebook adds "
                          "only ~2x and it is LOSSY, and the loss amplifies because events decide which events happen "
                          "next (at q=0.1 the replay leaves a box of half-extent 2.0 by 4.47). KEPT NEGATIVE 2: "
                          "DeltaChain is the wrong tool here -- it skips unchanged rows, but a sim moves every body "
                          "every frame, so it takes 614,144 bytes, MORE than the raw 460,800. Dense mutation with "
                          "sparse causes is a different structure from sparse mutation.",
                          example="trace, ev = mind.record_physics_trace(n=8, frames=200); "
                                  "assert (mind.replay_physics_trace(ev) == trace).all(); "
                                  "print(mind.physics_compression_report(trace, ev))",
                          native=True, aliases=("event codec", "physics codec", "compress a physics simulation trace",
                                                "compress a simulation", "record a replay", "deterministic replay",
                                                "seed and events", "netcode", "state sync", "delta compress a stream of states",
                                                "impulse events", "contact events", "replay a trace",
                                                "compress a physics trace", "trace compression", "lockstep",
                                                "store a simulation compactly", "sync physics over the network",
                                                "shrink a recorded sim", "sparse events"))
    c.register_capability("Fat-margin cache (for a query that drifts)", "when a query DRIFTS -- a camera nudging "
                          "forward, a cursor, an agent, a recall neighbourhood -- do not key the cache on the exact "
                          "query: bake an ENLARGED region around it and serve everything that lands inside. Catto's "
                          "enlarged AABB (he grows a moving body's box so it need not re-insert into the broadphase "
                          "every frame), generalized past physics. mind.margin_cache(builder, margin).get(p) -> "
                          "(value, hit); mind.drift_scale(queries) is the variation probe pointed at the QUERY STREAM "
                          "instead of the data; mind.suggest_margin(queries, target) picks the smallest margin meeting "
                          "a hit-rate target by REPLAYING the stream (empirical on purpose: a random walk's exit time "
                          "scales like (R/sigma)^2 but the measured rebuilds sit ~1.8x off, so a fitted law is worse "
                          "than a replay). MEASURED on a unit-step 2-D walk of 400 queries: margin 0 -> 0% hits / 400 "
                          "rebuilds; 1.0 -> 35.5% / 258; 3.0 -> 85.0% / 60; 6.0 -> 95.0% / 20. KEPT NEGATIVE: this is "
                          "NOT the sleep tracker's two-threshold hysteresis -- a margin cache has exactly ONE radius, "
                          "because a cache entry has no state to hover at a bar and flicker between; an inner "
                          "threshold would never be read. Cousins, not the same mechanism. WIRED (C4) into "
                          "RenderSession.preview(reuse_margin=...), where the drifting query is the CAMERA POSE: 20 "
                          "drifting frames at margin 0.12 give 19 hits and 1 rebuild. THE GATE IS NOT A HIT-RATE "
                          "TARGET -- a hit serves a STALE value, and on a rendered frame the max error saturates at "
                          "the FIRST reuse (0.5864, a silhouette edge) while the mean creeps 0.0001 -> 0.0051. Use "
                          "mind.suggest_margin_for_error(queries, values, max_mean_error, max_abs_error=...) and "
                          "mind.replay_margin_error(...): a value that jumps 0->1 passes a mean-only budget at margin "
                          "0.1929 and serves a completely wrong answer (max error 1.00), while the max-error bound "
                          "stops at 0.094558 and 0.095158 is already catastrophic. The admissible margin is a CLIFF. "
                          "SECOND CORRECTION: lightcache and domecache are NOT clients -- they are stateless per-frame "
                          "screen-space stride caches with no query stream to drift.",
                          example="import numpy as np; q = np.cumsum(np.random.default_rng(0).normal(size=(400,2)), axis=0); "
                                  "mc = mind.margin_cache(lambda p: ('bake', tuple(p)), margin=mind.suggest_margin(q, 0.9)); "
                                  "vals = [mc.get(x) for x in q]; print(mc.stats())",
                          native=True, aliases=("fat margin", "margin cache", "drifting query", "cache reuse",
                                                "cache a result for a query that keeps moving slightly",
                                                "avoid rebuilding a cache every frame", "hysteresis cache",
                                                "reuse a render tile when the camera barely moved", "enlarged region",
                                                "how big should my cache region be", "cache invalidation",
                                                "temporal reuse", "camera drift", "query drift", "rebuild less often",
                                                "variation probe", "drift statistics"))
    c.register_capability("Graph-colour waves (lock-free deterministic parallelism)", "schedule conflicting work into "
                          "WAVES that touch disjoint resources, so a wave runs fully parallel with no locks and no "
                          "atomics. mind.conflict_graph(item_keys) builds the graph (two tasks conflict iff they share "
                          "a resource); mind.color_waves(n, edges) colours it greedily in ascending index, so the "
                          "schedule is DETERMINISTIC -- same input, same waves, same order, every machine and every run, "
                          "which is exactly how Box3D earns its cross-platform determinism. mind.plan_write_waves(keys) "
                          "applies it to database write batches: the single-writer lock serialises writers because two "
                          "MIGHT touch the same row; colouring proves when they cannot. MEASURED: 2,000 transactions "
                          "over 300 keys colour into 24 waves, mean size 83.3 -- 83x lock-free parallelism, every wave "
                          "verified conflict-free. A physics constraint graph, a mesh's edge adjacency, a farm's "
                          "conflict graph and a DB write set are the same object; greedy is not optimal (colouring is "
                          "NP-hard) and does not need to be -- one extra wave costs one extra pass.",
                          example="n, edges = mind.conflict_graph([{'a','b'}, {'b','c'}, {'d'}]); waves = mind.color_waves(n, edges)",
                          native=True, aliases=("graph colouring", "graph coloring", "colour a graph", "waves",
                                                "lock free", "run tasks in parallel without locks", "no atomics",
                                                "deterministic parallelism", "conflict graph", "conflict free batches",
                                                "batch database writes that do not conflict", "wave scheduling",
                                                "group work so nothing collides", "parallel scheduling",
                                                "schedule conflicting tasks", "write batches", "key overlap"))
    c.register_capability("Partition-invariant sums (same answer at any bucket count)", "sum contributions so the "
                          "result is BIT-IDENTICAL no matter how the work is split -- 4-way, 7-way, one bucket, or one "
                          "bucket per item. mind.reduce_sum_exact_partitioned(buckets) fixes one global fixed-point "
                          "scale (from the global peak and count, both partition-invariant since max and len are), then "
                          "each bucket reduces to an int64 accumulator that merges in any order: integer addition is "
                          "exact, associative and commutative, so the accumulators form a monoid. MEASURED (700 "
                          "contributions spanning 16 orders of magnitude): plain float 4-way vs 7-way differs by 3e-08; "
                          "this is bit-identical across 1-, 4-, 7-, 13- and 700-way splits and under row shuffles. "
                          "KEPT NEGATIVE: reduce_sum_exact is order-independent but NOT partition-independent -- if a "
                          "farm float-sums INSIDE each bucket first, the rounding has already diverged and no exact "
                          "merge can undo it. Exactness must reach the leaves. This is determinism that survives "
                          "re-partitioning a running farm, which is the invariance Box3D does not claim. THE SAME "
                          "MONOID GIVES A SCAN (G3): mind.scan_exact(x) is a prefix sum that is bit-identical however "
                          "the array is blocked, and mind.scan_exact_blocked(x, k) proves it for every k from 1 to N. "
                          "A blocked FLOAT scan -- what every parallel scan actually is -- disagrees with itself: "
                          "4-block vs 7-block differ by 1.14e-12 on uniform data, 3.87e-07 across 16 orders of "
                          "magnitude, and 92.0 (9.2e-15 relative) on [1e16, 1, -1e16] repeated. KEPT NEGATIVE: the "
                          "exact scan is NOT more accurate than np.cumsum -- it is more REPRODUCIBLE. A sequential "
                          "cumsum wins on precision (7.8e-16 vs 6.5e-15 relative); it just cannot run on eight blocks "
                          "and give the same bits. If you are not blocking the scan, do not use it. "
                          "mind.distribute_exact(buckets, worker) and Coordinator.run_exact(...) are the wired doors: "
                          "the worker returns the bucket's CONTRIBUTIONS, not their sum, and that contract change IS "
                          "the fix. Swapping reduce_sum_exact into distribute() does NOT repair it -- by then the "
                          "worker has already float-summed inside its own bucket.",
                          example="import numpy as np; d = np.random.default_rng(0).normal(size=(64,3)); "
                                  "total, info = mind.distribute_exact(np.array_split(d, 7), lambda b, c: np.asarray(b, float)); "
                                  "print(info['scale'], total)",
                          native=True, aliases=("partition invariant", "bit exact sum", "reproducible sum",
                                                # G3: the prefix sum, same monoid
                                                "prefix sum of an array", "prefix sum", "scan", "scan an array",
                                                "running total", "cumulative sum bit exact", "blocked scan",
                                                "parallel scan", "cumsum reproducible",
                                                "same answer no matter how many machines i use", "reduce_sum_exact",
                                                "my sim gives different results on different nodes", "float associativity",
                                                "deterministic reduction", "exact accumulation", "order independent sum",
                                                "bit identical across nodes", "farm determinism", "rns"))
    c.register_capability("Name a contact type (bounce/slide/rest/jam)", "NAME what KIND of contact happened (holographic_collide.classify_contact) from {overlap, velocity, restitution}: bins the scalars to categories, then match_record against the contact-type records (bounce/slide/rest_contact/penetration/jam) + decide_or_abstain. m.classify_contact(overlap, velocity, restitution) -> {type, confident, record}. A LABEL/DISPATCH layer over the numeric resolvers (advance_ccd computes the RESPONSE; this names the SITUATION for per-type dispatch + a logged reason). KEPT NEG: a label, not a replacement; bins collapse magnitude.", example="import lecore; m=lecore.UnifiedMind(); print(m.classify_contact(0.02, 2.0, 0.8)['type'])", native=True, module="collide", aliases=("classify a collision type", "what kind of contact is this", "name the contact bounce or rest", "categorize a physics collision", "contact type from overlap and velocity", "is this a bounce or a jam"), semantic="analyze/match", consumes=("scalar",), produces=("selection",))
    c.register_capability("Tunnelling & CCD (speculative margins, conservative advancement)", "stop fast bodies "
                          "passing through thin walls. mind.time_of_impact(X, V, dt, sdf) sweeps each point along its "
                          "path and returns (hit, toi, contact) -- continuous collision detection by conservative "
                          "advancement; mind.advance_ccd(...) advances one step without tunnelling and cancels the "
                          "into-surface velocity (restitution bounces); mind.sdf_offset(sdf, margin) is the speculative "
                          "contact margin, which for an SDF-native engine costs ONE SUBTRACTION (no inflated AABBs). "
                          "The core CCD query -- how far can I move without hitting anything -- IS the SDF value, so "
                          "this is sphere tracing and it reuses the renderer's raymarch.sphere_trace: same march that "
                          "renders a pixel, same distance query Walk-on-Spheres steps by, no dedicated CCD pass. "
                          "MEASURED: a 30 m/s body stepping 0.5 m per frame passes clean through a 0.1 m wall under "
                          "discrete resolution and is stopped exactly on it here. KEPT NEGATIVE: a margin DETECTS "
                          "proximity but does not PREVENT tunnelling -- it resolves an already-crossed body out the "
                          "WRONG side, because a point sample has no memory of the swept path. The sdf argument accepts "
                          "a callable, an sdf node, or a DSL STRING like '(sphere 1.0)' -- the string form is what lets "
                          "an agent call these over HTTP, since a callable cannot cross a JSON boundary. "
                          "mind.resolve_swept_collision(X_prev, X, sdf) is the POSITIONAL twin for a PBD solver, and "
                          "softbody.step(continuous=True) is the wired door: nodes the sweep does not catch come back "
                          "bit-identical, so it is a strict addition.",
                          example="hit, toi, contact = mind.time_of_impact([[-3,0,0]], [[120,0,0]], 1/60., '(sphere 1.0)')",
                          native=True, aliases=("ccd", "continuous collision detection", "tunnelling", "tunneling",
                                                "stop a fast bullet going through a thin wall",
                                                "my object passes through the floor", "swept collision",
                                                "time of impact", "toi", "when will my object hit the ground",
                                                "conservative advancement", "speculative margin", "contact margin",
                                                "grow a collider by a small amount", "offset an sdf",
                                                "sphere tracing", "fast moving object collision", "bullet through paper",
                                                "prevent objects passing through each other", "swept sphere"))
    c.register_capability("Modal jump solver (skip the substeps)", "advance a LINEAR physics island in closed form "
                          "instead of substepping it: within a contact mode a soft-constraint system is the affine "
                          "recurrence s <- A s + b, so N substeps are ONE eigendecomposition and t=10s costs the same "
                          "as t=1s. mind.affine_jump(state, A, b, k) is the stateless jump; mind.modal_solver(...) "
                          "keeps a per-mode factorization and re-diagonalizes only at contact-mode SWITCHES; "
                          "mind.should_jump(dim, k) is the measured gate (jump pays at k >= 20*dim); "
                          "mind.escalation_plan(dim, k, energy=...) is THE ESCALATION LADDER (X11) that picks "
                          "{sleep | jump | substep} per island per frame -- Catto's '4 substeps' dial and our closed "
                          "form are two ends of one axis, and the descriptor chooses the rung. mind.soft_chain_bank + "
                          "mind.advance_bank are the TUNING BANK (X8): M stiffness/damping variants advanced in ONE "
                          "batched eigendecomposition (M=32 x 1,920 substeps: 4.3x over substepping the batch, exact "
                          "to 1.9e-12). KEPT NEGATIVE: that is NOT a superposition -- a trajectory is linear in the "
                          "FORCING (blend exactly, mind.blend_forcings, 1.1e-16) and nonlinear in the OPERATOR "
                          "(blending stiffness gives 2.9e-01 of error), so variants batch as arrays and there is "
                          "no capacity budget to spend; the backlog's 'M <= D/256' came from the retracted sqrt(M/D) "
                          "law. MEASURED: a "
                          "12-body chain (hertz=15, zeta=0.7) matches 3,840 substeps to 2.5e-12 at 8x the speed. "
                          "HONEST SCOPE: the win is where contact topology is STABLE (machinery, ragdolls at rest, "
                          "suspensions); where contacts churn, substepping is still the right tool and the gate says "
                          "so -- it degrades to stepping, never worse. Kept negative: a free-body island is a Jordan "
                          "block with no eigenbasis; it is REFUSED and stepped, not silently jumped.",
                          example="A, b, h = mind.soft_chain_matrices(12, hertz=15.0, zeta=0.7); "
                                  "s = mind.affine_jump(np.zeros(24), A, b, 3840)",
                          native=True, aliases=("modal jump", "closed form physics", "skip substeps",
                                                "skip thousands of physics substeps", "substepping too slow",
                                                "my machinery sim is too slow", "fast forward a simulation",
                                                "fast forward a ragdoll to where it settles",
                                                "advance a spring network without stepping", "linear recurrence",
                                                "affine recurrence", "matrix power", "eigendecomposition",
                                                "is it worth diagonalizing this system", "contact mode",
                                                "mode switch", "jump ahead in time", "soft constraint chain",
                                                "escalation ladder", "choose how many substeps to use",
                                                "tuning bank", "variant bank", "evaluate many parameter variants in one pass",
                                                "sweep friction and stiffness settings at once", "parameter sweep",
                                                "blend forcings", "many variants at once",
                                                "pick the right solver for this island", "how many substeps",
                                                "damped oscillator system", "Catto soft step", "propagate ahead"))
    c.register_capability("Islands + sleep (solve only what is still moving)", "partition a system into ISLANDS -- "
                          "the connected components of its constraint graph -- and step only the AWAKE ones, so a "
                          "pile of settled bodies costs nothing. mind.islands(n, edges) is the flood fill (a physics "
                          "island, a mesh shell, a farm bucket and a DDM subdomain are the same object); "
                          "mind.island_energy(pos, vel) is the sleep sensor; mind.island_sleep_tracker() adds "
                          "HYSTERESIS (sleep after N quiet frames, wake instantly above an outer bar -- one threshold "
                          "flickers on float noise, measured); mind.step_islands(...) carries a sleeping island's rows "
                          "through BIT-IDENTICALLY. And SLEEP IS THE CLOSED FORM: mind.settle_island(state, U) jumps "
                          "straight to the fixed point via iterate.limit() instead of stepping until it settles. "
                          "Measured negative: that fixed point is NOT rest -- modes with |eigenvalue|~1 persist, so a "
                          "diffusive island settles to its MEAN; only a strictly contractive operator settles to zero.",
                          example="isl = mind.islands(6, [(0,1),(1,2),(4,5)]); tr = mind.island_sleep_tracker(); "
                                  "state, awake, asleep = mind.step_islands(np.zeros((6,3)), isl, lambda s: s+1.0, tracker=tr); "
                                  "print(awake, asleep)",
                          native=True, aliases=("island", "islands", "sleep", "sleeping bodies", "put bodies to sleep",
                                                "put resting bodies to sleep", "skip simulating objects that stopped moving",
                                                "solve only the parts that are still moving", "connected components",
                                                "constraint graph", "group bodies connected by constraints",
                                                "island decomposition", "wake and sleep", "at rest", "settled",
                                                "jump a settled system to its final state", "fixed point of a system",
                                                "sleep threshold", "hysteresis", "awake islands", "skip idle work",
                                                "steady state", "settle", "quiescent", "energy probe",
                                                # C1/C2: the two wired clients
                                                "softbody sleep", "skip sleeping cloth", "solve only moving nodes",
                                                "coordinator waves", "lock free coordinator", "wave schedule"))
    c.register_capability("Soft constraints (hertz + damping ratio)", "make any constraint SPRINGY instead of rigid, "
                          "in physical units: mind.project_onto_constraints(x, projs, stiffness=(hertz, zeta), dt=h) "
                          "specifies a constraint by its natural frequency (hertz) and damping ratio (zeta; 1.0 = "
                          "critically damped, no overshoot) instead of a hand-tuned per-sweep omega. Catto's Soft Step "
                          "parameterization: the same (hertz, zeta) means the same physics at ANY substep count, where "
                          "the same omega does not -- so the substep count becomes an accuracy dial, not a physics dial. "
                          "stiffness=(inf, zeta) is the hard projection exactly. Because PBD, FABRIK/IK, the resonator "
                          "and the PnP denoise loop are all ONE iterated projection, they all gain softness from this one "
                          "dial. mind.soft_relaxation(hertz, zeta, dt) exposes the factor itself. Kept negative: being "
                          "position-level it cannot RING -- zeta is a rate dial, not an overshoot dial; underdamped "
                          "bounce needs the velocity solver (dynamics). WIRED (C3): mind.solve_ik(..., stiffness=(hz, "
                          "zeta), dt=...) makes an IK chain springy, and SoftBody.step(solver='pbd', stiffness=...) "
                          "makes its constraints soft -- both gated on stiffness=(inf, zeta) being BIT-IDENTICAL to the "
                          "rigid default. Measured: an IK end-effector lags its target by 0.3673 / 0.0336 / 0.0000 at "
                          "2 / 8 / 40 Hz; a stretched PBD bone relaxes to 1.7498 at 2 Hz and 1.028 at 20 Hz against a "
                          "rest length of 1.0. The XPBD path ignores it -- its per-constraint compliance already IS "
                          "this idea.",
                          example="x, n, ok = mind.project_onto_constraints(x0, [proj], iters=64, stiffness=(15.0, 1.0), dt=1/240)",
                          native=True, aliases=("soft constraint", "soft constraints", "stiffness", "hertz",
                                                "damping ratio", "zeta", "springy constraint", "make it springy",
                                                "how stiff should my constraint be", "spring stiffness",
                                                "soft body stiffness in hertz", "compliance", "XPBD compliance",
                                                "under-relaxation", "omega", "soft step", "Catto soft constraint",
                                                "substep invariant", "why does my solver change with more substeps",
                                                "rigid vs springy", "joint softness", "cloth stiffness",
                                                "damping for a joint", "constraint stiffness", "soft_relaxation"))
    c.register_capability("Import artist file formats (OBJ/glTF/textures/volume)", "import the files artists hand you: "
                          "mind.load_obj('model.obj') reads Wavefront geometry + its .mtl (UVs, normals, per-face "
                          "material, map_* textures); mind.load_glb('model.glb') reads glTF/GLB geometry AND its full PBR "
                          "channels (base colour / metallic-roughness / normal / occlusion / emissive) with embedded "
                          "textures and per-vertex UVs/normals, AND for rigged models its ANIMATIONS (keyframed node "
                          "transforms -- clip.sample(t), rotations slerped) and SKINS (joints + inverse-bind + weights); "
                          "mind.load_texture_set(folder) turns a folder of Adobe Substance 3D Painter export maps "
                          "(basecolor/roughness/metallic/normal/height/ao/emissive, matched by name) into one "
                          "PBRMaterial; mind.load_volume('grid.npy') wraps a 3-D density grid as a field for "
                          "render_volume. mind.import_asset(path) dispatches by extension. Once a rigged glTF is loaded, "
                          "mind.deform_mesh(loaded, clip, t) actually MOVES it -- linear-blend skinning by the animated "
                          "skeleton plus morph-target blending, returning the deformed mesh at time t. Stdlib+NumPy; PIL "
                          "lazy for textures. HONEST: proprietary .sbsar/.spp and sparse OpenVDB .vdb need their vendor tools -- "
                          "import the exported open forms.",
                          example="lm = mind.load_obj('chair.obj'); glb = mind.load_glb('robot.glb'); mat = mind.load_texture_set('exports/brick'); vol, b = mind.load_volume('smoke.npy')",
                          native=True, aliases=("import", "load obj", "load gltf", "load glb", "mtl", "wavefront",
                                                "substance painter", "adobe painter", "texture set", "pbr material import",
                                                "load model", "import mesh", "volumetric", "load volume", "vdb", "voxel",
                                                "density grid", "import material", "3d file", "asset import",
                                                "animation", "skin", "rigged", "keyframe", "skeleton", "uv", "channels",
                                                "deform", "skinning", "linear blend skinning", "morph", "blend shape",
                                                "pose a rig", "animate a model"))
    c.register_capability("Cold storage (compress inactive data)", "shrink INACTIVE data to save memory and disk, and "
                          "inflate it back on demand: store = mind.cold_store(keep_warm=8) keeps only the K most-recently-"
                          "used values live and compresses the rest, warming any of them transparently on get(); "
                          "mind.cool(big_table) wraps ONE value so c.cool() frees its RAM and c.get() brings it back "
                          "bit-identical. Works on tables, whole databases, big arrays, any picklable structure; "
                          "codec='lzma' packs smaller, spill_dir=... writes cold blobs to disk. Honest: high-entropy VSA "
                          "vectors barely compress (the win there is freeing the live object / spilling to disk); "
                          "redundant/text/structured data compresses a lot. The query Database can auto-cool its own "
                          "idle tables: db.enable_cold_storage(keep_warm=K) then db.cool_idle() compresses tables you "
                          "haven't queried lately and a query warms them back -- and a DB shipped to a distributed "
                          "worker arrives warm + cooling-off, so a shared read-only cache is never mutated.",
                          example="store = mind.cold_store(keep_warm=4); store.put('t1', big_table); store.get('t1')  # transparently warmed",
                          native=True, aliases=("cold storage", "compress inactive", "evict", "spill to disk", "cool",
                                                "warm", "fold up", "shrink memory", "free ram", "compress table",
                                                "compress database", "lazy inflate", "lru cache eviction", "page out",
                                                "auto cool tables", "idle table compression"))
    c.register_capability("File map ingest (folder / zip -> queryable)", "point at a FOLDER, a .zip, or a file and "
                          "digest it into a queryable FILE MAP: fm = mind.ingest_files('project/') (or 'bundle.zip'). "
                          "Query it by NAME/glob (fm.find('*.png')), KIND (fm.by_kind('model'): image/text/model/data/"
                          "code/archive), METADATA (larger_than/newer_than/by_ext), text CONTENT (fm.search_text('shader "
                          "normal') -- an inverted index over the text files), and MEANING (fm.build_meaning_index() then "
                          "fm.find_by_meaning('lighting')). fm.tree() is the folder hierarchy. Every file is also tracked "
                          "for RELOCATION/CHANGE (fm.missing()/changed()/relink(one,new)/resolve_assets(roots)), so a "
                          "moved/edited tree self-heals. Stdlib only; text indexing is size-capped.",
                          example="fm = mind.ingest_files('my_project.zip'); fm.find('*.obj'); fm.search_text('normal map'); fm.tree()",
                          native=True, aliases=("ingest", "ingest files", "index a folder", "digest a folder", "read a zip",
                                                "scan folder", "file map", "make files queryable", "search my files",
                                                "index files", "folder to database", "query a directory", "catalog files",
                                                "import a folder", "unzip and index"))
    c.register_capability("Asset relocation / relink (external files)", "track the EXTERNAL files a scene depends on "
                          "(textures, models, ...) and repair their paths when they move -- the '3-D missing textures' "
                          "problem. lib = mind.asset_library(); lib.add(path); then when a folder moves, lib.relink("
                          "one_asset, its_new_path) re-finds every OTHER moved file automatically (it works out the "
                          "moved parent and rewrites the rest, then structurally SEARCHES for anything reorganised). "
                          "lib.changed() spots files edited on disk (size/mtime or content hash); lib.search_under("
                          "folder) finds missing files under a folder; lib.resolve(asset, roots=) locates a file by "
                          "CONTENT HASH across machines (the distributed fallback). Saves/loads a JSON manifest.",
                          example="lib = mind.asset_library(); lib.add('project/textures/water/wave.png'); lib.relink(lib.assets[0], 'newroot/project/textures/water/wave.png')",
                          native=True, aliases=("asset", "assets", "relink", "relocate", "missing textures", "broken path",
                                                "fix paths", "external files", "find moved files", "asset paths",
                                                "texture path", "reconnect assets", "repath", "file moved", "asset manifest"))
    c.register_capability("Message bus + agent (LLM) bridge", "connect a person AND an agent to the running tool at "
                          "once, and let the app PUSH to the agent instead of the agent polling: mind.bus() is a "
                          "message bus (publish/subscribe by topic, mailboxes to pull an inbox, history); "
                          "mind.run_task('render', fn, background=True) runs a job and publishes 'render.done' with a "
                          "small summary when it finishes; mind.agent_bridge(llm=my_fn).notify_on('render.done', 'does "
                          "it look right?') calls YOUR llm (any text->reply callable -- no LLM library is imported, so "
                          "it's fully optional) and posts the reply on the bus. Over HTTP a remote agent uses "
                          "/bus/publish + /bus/poll. The LLM is optional; leCore runs with no agent attached.",
                          example="bridge = mind.agent_bridge(llm=my_llm); bridge.notify_on('render.done', 'does it look right?'); mind.run_task('render', lambda: scene.render(), background=True)",
                          native=True, aliases=("message bus", "event bus", "pubsub", "publish subscribe", "agent bridge",
                                                "llm bridge", "notify the agent", "push notification", "on render done",
                                                "connect an agent", "send message to agent", "mailbox", "inbox",
                                                "trigger the llm", "watch for events", "task done event"))
    # --- agent-friendly discovery: describe / suggest / route / autocomplete over the whole engine ---
    c.register_capability("Agent skills (discover & route)", "the AGENT-FRIENDLY layer: mind.skills() lists every "
                          "capability + method with how to CALL it (skill descriptions, real signatures); "
                          "mind.suggest(task) ranks capabilities for a plain-English task WITH a confidence + the call; "
                          "mind.route(task) is a decision node ('act' with the call when confident, else 'choose' the "
                          "options); mind.complete_method(prefix) autocompletes method names; mind.describe_skill(name) is a "
                          "skill card. Also over HTTP: GET /skills, POST /skills/suggest|route|complete|card",
                          example="mind.route('render a scene'); mind.suggest('edit an image'); mind.complete_method('learn_')",
                          native=True, aliases=("agent", "agentic", "skills", "skill description", "autocomplete",
                                                "suggest", "decision tree", "route", "list abilities", "available skills",
                                                "which tool", "find a tool", "capabilities", "manifest", "discover", "help"))
    # --- domain families surfaced by the catalog-gap sweep (tools existed, homes did not) ---
    # io-tag correction, caught by test_io_shape_pipeline_hierarchy: this entry claimed consumes=('mesh',
    # 'sdf_scene'), which MANUFACTURED a fake mesh->image edge. The path tracer's signature is
    # path_trace(sdf, camera, ...) -- there is no mesh path anywhere in it, so suggest_pipeline('points',
    # 'image') routed through a step that would raise the moment an agent actually called it (and it WON the
    # route, because the BFS tie-break sorts by name and capital 'R' sorts before lowercase 'render_mesh').
    # The honest mesh->image producer is "Rasterize a mesh (z-buffer, textured)" / faculty m.render_mesh.
    # A wrong io tag is worse than a missing one: it is a confidently-suggested broken pipeline.
    c.register_capability("Rendering (path trace)", "render a scene to an image: path_trace (Monte-Carlo global "
                          "illumination), a camera controller, indirect-light gather + irradiance cache "
                          "(globalillum), precomputed radiance transfer (prt), volumetric integration, and lens/DOF + "
                          "post-FX. The analysis-by-synthesis render path", example="mind.path_trace(scene); mind.camera(); from holographic.rendering.holographic_raymarch import sphere_trace",
                          native=True, aliases=("render a scene", "path trace", "ray tracing", "global illumination",
                                                "camera", "depth of field", "lens", "volumetric render", "radiance transfer",
                                                "prt", "ambient occlusion", "post processing", "gbuffer", "raytrace", "render", 'render a mesh with vertex colours', 'draw a mesh with no texture just vertex colors'), module="render", consumes=('sdf_scene',), produces=('image',))
    c.register_capability("Rasterize a mesh (z-buffer, textured)", "RASTERISE a mesh to an (H,W,3) image, z-buffer + Lambert (rasterize_mesh; faculty m.render_mesh). TEXTURED (default-off): texture=(H,W,3) + per-vertex uvs -> each fragment BILINEARLY samples at its barycentric UV. VERTEX COLOURS (VCOL): vertex_colors=(V,3/4) or mesh.colours renders a mesh with NO texture, barycentric-interpolated -- what a recall bake / coloured DCC mesh needs. smooth=True = Gouraud normals (curved not faceted); two_sided=True = |n.l| for thin/unorientable meshes. All default-off, byte-identical absent. KEPT NEG: textured/vcol/smooth need vectorized=True.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.rendering.holographic_render import Camera; b=box(); uv=np.array([[0,0],[1,0],[1,1],[0,1]]*2,float); chk=np.stack([np.indices((8,8)).sum(0)%2]*3,-1).astype(float); img=m.render_mesh(b, Camera(eye=(2.2,1.6,2.4),target=(0,0,0),fov_deg=40), width=64, height=64, texture=chk, uvs=uv); img.shape",
                          native=True, aliases=("render a mesh with a texture", "textured mesh rendering", "rasterize a mesh",
                                                "show a textured model", "preview a mesh with its texture", "z-buffer render",
                                                "display uv mapped texture", "software rasterizer"))
    c.register_capability("Smooth a bumpy mesh surface (Taubin no-shrink)", "SMOOTH / denoise a bumpy mesh surface (holographic_meshsmooth): m.mesh_smooth(mesh) runs Taubin lambda|mu no-shrink smoothing -- a low-pass over vertex positions using cotangent weights that removes surface noise/bumps WITHOUT the shrinkage plain Laplacian smoothing causes. Exposes lam/mu/iters. The go-to for a jagged / noisy / faceted mesh from marching-cubes, scanning, or photogrammetry. KEPT NEG: it is a low-pass, so it also softens INTENDED sharp features; and it over-smooths an already-clean mesh (needs a noise estimate, no auto-tune).", example="import lecore; from holographic.mesh_and_geometry.holographic_mesh import box; m=lecore.UnifiedMind(); sm=m.mesh_smooth(box()); print(len(sm.vertices))", native=True, module="meshsmooth", aliases=(
                                                # the full phrasing, not just the stem: route_or_abstain scores a
                                                # query against an IN-VOCABULARY NOISE FLOOR that RISES as the
                                                # catalog grows (null_mean 2.88 -> 3.49 once the merge restored 30
                                                # entries), so this hit cleared the floor at z=1.08 before and
                                                # abstained at z=0.63 after, with its score unchanged at 4.50.
                                                # Raising the TARGET is the additive fix; lowering the floor would
                                                # weaken every gate that uses it.
                                                "smooth a bumpy mesh", "denoise a mesh", "remove mesh noise",
                                                "smooth out the bumpy surface", "smooth a mesh", "remove bumps from a mesh", "denoise a mesh surface", "make a jagged mesh smooth", "taubin smoothing", "relax mesh vertices", "smooth a noisy scan"), semantic="create/emit", consumes=("mesh",), produces=("mesh",))
    c.register_capability("Mesh editing (DCC)", "modeling/DCC edits on a Mesh: extrude/inset faces (meshpoly; extrude/inset quad_walls=True emit pure-quad side/ring walls for a Catmull-Clark cage; loop_cut takes cuts=N + factor for N spaced parallel loops), "
                          "subdivide + smooth (meshsubdiv, Catmull-Clark), deform/warp (deform), rig-skin-pose a "
                          "skeleton (blendpose), UV unwrap (chart), decimate/QEM, booleans, and mesh<->SDF. "
                          "Blender-parity polygon editing", example="mind.deform(mesh, ...); mind.mesh_to_sdf(mesh); from holographic.mesh_and_geometry.holographic_meshverbs import extrude_face",
                          native=True, aliases=("edit a mesh", "extrude", "bevel", "inset", "subdivide", "smooth a mesh",
                                                "decimate", "reduce polygons", "uv unwrap", "unwrap uv", "rig", "skin",
                                                "pose a skeleton", "skeleton", "deform", "boolean", "remesh", "dcc", "modeling"), consumes=('mesh',), produces=('mesh',))
    c.register_capability("SDF & procedural geometry", "implicit + procedural geometry: signed distance fields (sdf), "
                          "sphere-trace raymarching with ambient occlusion (raymarch), sculpting, procedural terrain "
                          "(procgen), spatial tiling + octree, and voxelization. Native-first shape building",
                          example="from holographic.rendering.holographic_raymarch import sphere_trace; mind.terrain(...); from holographic.mesh_and_geometry.holographic_sdf import ...",
                          native=True, aliases=("sdf", "signed distance field", "raymarch", "sphere trace", "sculpt",
                                                "procedural terrain", "procedural geometry", "voxelize", "voxel", "octree",
                                                "tile in space", "implicit surface", "marching"))
    c.register_capability("Domain operators & cosine palette (demoscene)", "infinite procedural worlds from a tiny "
                          "kernel (holographic_domain, Quilez/Shadertoy style): domain WARPS that pre-transform "
                          "the query point of any SDF or field -- domain_repeat (tile into an infinite or finite "
                          "lattice), domain_fold (kaleidoscopic mirror symmetry), domain_twist / domain_bend "
                          "(helix / arc). smooth_min / smooth_max are the crease-free metaball union / intersection "
                          "/ subtraction (iq's smin). cosine_palette turns one scalar into a smooth colour, "
                          "random_palette makes a seed-driven scheme. One shape becomes a crystal; no assets",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                          "lat=m.domain_repeat(sphere(0.3), 1.0); m.cosine_palette(0.5).tolist()",
                          native=True, aliases=("domain repetition", "infinite tiling of a shape", "tile a shape",
                                                "fold space for symmetry", "kaleidoscope", "mirror the domain",
                                                "twist a shape", "bend a shape", "smooth minimum", "smin", "metaball",
                                                "elongate a shape", "stretch a primitive along an axis",
                                                "opelongate", "make a capsule from a sphere",
                                                "blend two shapes smoothly", "cosine color palette", "cosine gradient",
                                                "procedural palette", "random color palette", "iq palette", "demoscene",
                                                "infinite lattice", "opRep", "smooth union of sdf"))
    c.register_capability("Palette colour stops (plottable swatches)", "turn a cosine palette into a small table of "
                          "plottable RGB colours -- the companion to random_palette, which returns cosine "
                          "COEFFICIENTS (a,b,c,d), NOT colours. mind.palette_stops(seed, n) evaluates the palette at "
                          "n even points -> an (n,3) float RGB array for a swatch strip, gradient ramp, or legend; "
                          "pass coeffs=(a,b,c,d) to sample a KNOWN palette. Pure composition of random_palette + "
                          "cosine_palette, so the stops ARE the palette's colours -- it exists so callers stop "
                          "interpolating the coefficients as colours (which ships garbage). Deterministic per seed",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); m.palette_stops(seed=7, n=8).tolist()",
                          native=True, aliases=("palette stops", "palette color stops", "list of rgb colors from a palette",
                                                "sample a palette into colors", "swatches from a seed", "color swatches",
                                                "generate n colors", "gradient stops", "rgb colors from random palette",
                                                "palette to colors", "colors for a legend", "theme colors from a seed",
                                                "sample cosine palette as rgb", "plottable palette colours"))
    c.register_capability("Navigation & planning", "find a way through a space or structure: A*/shortest-path route "
                          "planning (plan), slime-mould flow networks (flow), tree/graph navigation (navigator), and "
                          "maze solving. Pathfinding on the VSA substrate", example="from holographic.scene_and_pipeline.holographic_plan import ...; mind.solve_maze(world); from holographic.misc.holographic_flow import ...",
                          native=True, aliases=("navigation", "plan a route", "pathfinding", "shortest path", "maze",
                                                "slime mould", "flow network", "route", "navigate", "wayfinding", "traverse",
                                                "slime mold maze solver", "pheromone pathfinding", "solve a maze"))
    c.register_capability("Learning & agents", "gradient-free learning on the substrate: an RL agent with a value head "
                          "+ drives (agent), a holographic classifier, an echo-state reservoir (reservoir), "
                          "mixture-of-experts (moe), KAN, forward-forward, recurrent/predictive nets, and dreaming. NPC "
                          "brains and on-line learners with NO autodiff", example="mind.agent(...); mind.classify(x); mind.reservoir(...)",
                          native=True, aliases=("reinforcement learning", "rl agent", "train a classifier", "classify",
                                                "policy", "npc brain", "game ai", "reservoir", "echo state", "mixture of experts",
                                                "moe", "kan", "forward forward", "gradient free", "learn a policy", "predictor", "agent"))
    c.register_capability("Data analysis", "analyse data with VSA-native methods: optimal transport / Wasserstein "
                          "(transport), graph Laplacian + spectral filtering (graphsignal), Nystrom embedding / "
                          "dimensionality reduction, persistent-homology topology, kernel density estimate, "
                          "point-cloud structure (cosmic), and time-series / market analysis", example="from holographic.misc.holographic_transport import wasserstein; from holographic.misc.holographic_graphsignal import laplacian_filter",
                          native=True, aliases=("data analysis", "cluster", "optimal transport", "wasserstein", "graph laplacian",
                                                "spectral", "dimensionality reduction", "embedding", "topology", "persistent homology",
                                                "kernel density", "point cloud", "time series", "statistics", "analytics"))
    c.register_capability("Symbolic reasoning", "recover structure symbolically: symbolic regression to find a formula "
                          "(symbolic), resonator networks that FACTOR a bound vector into its parts (sbc/resonator), "
                          "is_a taxonomy climbing, and relational reasoning over records. Turning data and vectors back "
                          "into laws", example="from holographic.agents_and_reasoning.holographic_symbolic import ...; mind.climb('dog'); from holographic.misc.holographic_sbc import ...",
                          native=True, aliases=("symbolic regression", "find a formula", "factor a vector", "resonator",
                                                "factorization", "decompose a signal", "reason", "reasoning", "climb hierarchy",
                                                "relational", "law from data"))
    c.register_capability("Signal & spectral", "1-D signal processing: FFT / spectral analysis (spectral), "
                          "faint-signal detection in noise with a calibrated false-discovery rate (signal_structure), "
                          "drifting-narrowband / de-Doppler search (dedoppler), spectral flatness, and bandwidth. The "
                          "radio-SETI-style detection stack", example="from holographic.sampling_and_signal.holographic_spectral import ...; from holographic.sampling_and_signal.holographic_dedoppler import ...",
                          native=True, aliases=("signal processing", "fft", "spectral", "spectrum", "detect a signal",
                                                "faint signal", "narrowband", "doppler", "dedoppler", "drift", "flatness",
                                                "bandwidth", "frequency", "audio"))
    c.register_capability("analyze_axes", "which axis of a multi-dimensional dataset is the INDEX (carrier -- the "
                          "boring, regular axis like time or scanline order) and which is the PAYLOAD (content -- "
                          "the axis whose value defines what each item means). Per axis, measures marginal "
                          "information and content coupling, then recommends INDEX (a cheap, comparability-preserving "
                          "carrier) or BIND (fold the value into content, only when the axis is informative and its "
                          "conjunction with content is the unit). The auto-schema / auto-decomposition entry point",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "vid=np.random.default_rng(0).standard_normal((20,8,8)); m.analyze_axes(vid, categorical=[])",
                          native=True, aliases=("axis role", "index vs payload", "carrier vs content",
                                                "which axis is the carrier", "which axis is boring",
                                                "index or bind", "should time be a feature", "schema discovery",
                                                "discover data format", "decompose a tensor", "axis information",
                                                "marginal information per axis", "content coupling",
                                                "elevate the boring dimension", "time as index", "payload axis",
                                                "which dimension to fold in"))
    c.register_capability("comparability_cost", "MEASURE the price of binding a boring axis into content "
                          "(holographic_axisrole): adjacent-slice similarity when the axis is INDEXED (raw slices) "
                          "vs BOUND (each slice rotated by a distinct per-slice key). On a boring carrier the "
                          "indexed similarity is high and the bound similarity collapses toward 0 -- the "
                          "similarity destroyed by the wrong role choice, in one number, against the raw indexed "
                          "baseline",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "vid=np.random.default_rng(0).standard_normal((20,64)); m.comparability_cost(vid, 0)",
                          native=True, aliases=("cost of binding an axis", "binding destroys similarity",
                                                "private subspace rotation", "why not bind time",
                                                "comparability", "similarity collapse", "measure binding cost"))
    c.register_capability("analytic_signal", "represent a signed series as ROTATION (holographic_analytic): the "
                          "analytic signal z = value + i*Hilbert(value) = amplitude * exp(i*phase). Returns the "
                          "instantaneous amplitude (envelope / circle radius), unwrapped phase (how far it has "
                          "rotated), and instantaneous frequency (how fast the sign turns over). amplitude*cos(phase) "
                          "reconstructs the signal EXACTLY. The 'sign as rotation' framework: a negative value is a "
                          "rotation, magnitude is the radius. NumPy-only Hilbert transform, no scipy",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "x=np.cos(np.linspace(0,20,512)); a=m.analytic_signal(x); a['amplitude'][:3]",
                          native=True, aliases=("analytic signal", "hilbert transform", "sign as rotation",
                                                "value as rotation", "instantaneous phase", "instantaneous frequency",
                                                "instantaneous amplitude", "envelope of a signal", "phasor of a signal",
                                                "quadrature", "rotate to make negative", "phase of a signal",
                                                "represent negative as rotation", "circle encoding of a value"))
    c.register_capability("monotone_cost", "MEASURE the price of clockwise-only (one-way) rotation on a real signed "
                          "series (holographic_analytic): reconstruct with the full reversible phase vs a phase "
                          "clamped to advance one way, and report the excess error and reversal fraction. Sharp "
                          "finding: a real scalar signal is ALREADY a one-way rotation (symmetric spectrum -> "
                          "non-negative instantaneous frequency), so this reads ~0 -- a single real channel cannot "
                          "carry a reversal. The real group-vs-monoid price lives on the complex path "
                          "(phasor_monotone_cost)",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "x=np.cos(np.linspace(0,20,512)); m.monotone_cost(x)",
                          native=True, aliases=("clockwise only rotation", "one way rotation cost", "monotone phase",
                                                "irreversible rotation", "ratchet cost", "group versus monoid",
                                                "can only rotate one direction", "cost of one directional rotation",
                                                "reversal fraction", "monocomponent signal test"))
    c.register_capability("phasor_monotone_cost", "the group-vs-monoid price of clockwise-only rotation where it "
                          "actually lives: a TRUE complex / I-Q rotation (holographic_analytic). A complex series "
                          "carries a genuine rotation DIRECTION in its two channels and can truly reverse; clamping "
                          "it one-way loses the reversal at a large well-defined cost. The quadrature encoder with "
                          "both channels present -- drop to one direction and you pay",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "z=np.exp(1j*np.cumsum(np.r_[np.full(64,0.2),np.full(64,-0.2)])); m.phasor_monotone_cost(z)",
                          native=True, aliases=("complex rotation reversal cost", "iq signal one way", "phasor reversal",
                                                "quadrature encoder direction", "two channel rotation",
                                                "clockwise only complex", "reversal cost of a phasor"))
    c.register_capability("identify_dynamics", "identify MASS / MOMENTUM / dynamics from a measurement series "
                          "(holographic_sysid), via whichever honest door the data opens: a FORCE channel (fit "
                          "m*a+c*v+k*x=F -> mass, damping, stiffness); an INTERACTION (momentum conservation -> the "
                          "mass ratio); or a KNOWN FORCE LAW + constant (orbit + G -> central mass, Kepler). A "
                          "trajectory ALONE is REFUSED with the gauge theorem (F=ma exposes only F/m; mass is "
                          "unidentifiable without a force channel) -- kinematics is offered instead. General: lab "
                          "carts, collider events, orbits; a market 'mass' would be the force door with order flow",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "t=np.arange(0,4,0.001); m.identify_dynamics(x=np.cos(2*t), dt=0.001, force=8*np.cos(2*t)*0-2*4*np.cos(2*t))",
                          native=True, aliases=("estimate mass from data", "mass from trajectory and force",
                                                "system identification", "fit equation of motion", "momentum of an object",
                                                "identify dynamics", "mass ratio from collision", "weigh an object",
                                                "learn dynamics coefficients", "damping and stiffness from data",
                                                "gauge freedom mass force", "can i get mass from a trajectory"), module="dynamics", consumes=('timeseries',), produces=('transform',))
    c.register_capability("central_mass_from_orbit", "weigh a CENTRAL BODY from a bound orbit (holographic_sysid): "
                          "Kepler's third law M = 4*pi^2*a^3/(G*T^2); semi-major axis from radius extremes, period "
                          "from the unwrapped bearing (the monotone-rotation winding picture). 2-D or inclined 3-D "
                          "orbits (best-fit plane). REFUSES on under one full observed orbit rather than "
                          "extrapolating. How astronomy weighs stars and black holes with no force sensor -- the "
                          "known force law + its constant break the mass gauge",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "T=3.156e7; tt=np.linspace(0,1.2*T,2000); R=1.496e11; "
                          "pos=np.stack([R*np.cos(2*np.pi*tt/T),R*np.sin(2*np.pi*tt/T)],axis=1); "
                          "m.central_mass_from_orbit(pos, tt[1]-tt[0])",
                          native=True, aliases=("kepler third law", "mass of a star from an orbit", "weigh a star",
                                                "central mass", "orbital period mass", "mass of a black hole from orbits",
                                                "astronomy mass estimate", "semi major axis period", "weigh the sun"))
    c.register_capability("diagnose_scaling", "detect WHICH limit a workload is hitting "
                          "(holographic_scalinglaw): scale each declared knob (dim, tiles, bits, resolution, "
                          "samples -- anything) in isolation, measure the error response, rank the levers. A limit "
                          "is diagnosed by which knob's doubling reduces the error; a WALL is when no knob does "
                          "(scaling is the wrong tool -- change the approach). The house dim-doubling rule "
                          "generalised to every resource and made executable, with the probe table as evidence",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "m.diagnose_scaling(lambda dim,tiles: 1.0/dim**0.5, {'dim':64,'tiles':4})",
                          native=True, aliases=("which limit am i hitting", "should i scale dimensions or tile",
                                                "variance limited or margin limited", "double the dimension test",
                                                "pick a scaling lever", "diagnose a bottleneck", "scaling diagnosis",
                                                "is this a wall or a scaling problem", "detect what needs scaling",
                                                "rank scaling knobs", "capacity or resolution limited"))
    c.register_capability("auto_scale", "automatic scaling (holographic_scalinglaw): repeatedly diagnose from the "
                          "current operating point and double the most responsive knob until the target error is "
                          "met, a WALL is diagnosed (no knob helps -- stop and say so), or the round budget is "
                          "spent. Every step carries the probe that justified it. The capacity-adaptive pattern "
                          "(octree, load-gated record) generalised to any workload with declared knobs",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "m.auto_scale(lambda dim: 1.0/dim**0.5, {'dim':64}, target_error=0.05)",
                          native=True, aliases=("automatic scaling", "scale until target met", "auto scale a workload",
                                                "adaptive scaling loop", "keep doubling until it works",
                                                "scale up automatically", "generic capacity adaptation"))
    c.register_capability("diagnose_bake", "should you raise the DIMENSION or the BANDWIDTH for an n-D texture "
                          "bake of THIS field? (holographic_scalinglaw): wires diagnose_scaling to bake_nd on a "
                          "held-out query set, so the engine's most-repeated tuning rule ('double D -- if error "
                          "drops you are variance-limited, else raise the bandwidth') becomes one call instead of "
                          "a manual re-bake-and-eyeball. verdict is 'scale:dim' (more dimension pays) or "
                          "'scale:margin' (widen/narrow the kernel; more dimension is wasted), each carrying its "
                          "measured error drop -- run it before committing to an expensive high-dimension bake",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "ax=np.linspace(0,1,40); P=np.stack(np.meshgrid(ax,ax,indexing='ij'),-1); "
                          "m.diagnose_bake([ax,ax], np.sin(2*np.pi*P[...,0])*np.cos(2*np.pi*P[...,1]))['verdict']",
                          native=True, aliases=("tune the bake dimension", "raise dimension or bandwidth for a bake",
                                                "is my bake variance limited", "diagnose a texture bake",
                                                "should i raise dim or margin", "pick bake dimension",
                                                "auto-tune bake parameters", "bias or variance limited bake"))
    c.register_capability("rectify_carrier", "REPAIR a nearly-boring carrier axis into a clean uniform index "
                          "(holographic_axisrole): a non-monotone axis (delta sometimes negative) is lifted by "
                          "cumulative ARC LENGTH -- the monotone/covering-lift from sign-as-rotation, absorbing "
                          "small reversals into one-way progress -- then an irregular axis is RESAMPLED onto a "
                          "uniform grid by interpolation. Marginal info measured before/after (after = 0.0, ideal "
                          "carrier). monotone_fraction reports how much repair was needed; a largely-reversing "
                          "axis (below ~0.9) means content is a PATH not a function of the axis -- inspect by hand",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "t=np.cumsum(np.random.default_rng(0).exponential(1.0,200)); "
                          "m.rectify_carrier(t, np.sin(0.1*t))['marginal_info_after']",
                          native=True, aliases=("fix an irregular time axis", "resample to uniform spacing",
                                                "interpolate to constant delta", "normalize a carrier axis",
                                                "make an axis monotone", "arc length reparametrization",
                                                "axis sometimes goes negative", "repair the index axis",
                                                "non uniform sampling to uniform", "rectify the boring dimension"))
    c.register_capability("winding_map", "when a carrier axis LARGELY reverses and revisits coordinates: is "
                          "content a FUNCTION of the axis or a PATH over it? (holographic_winding). Splits into "
                          "monotone LAPS, measures lap agreement. Verdicts: 'function' -> merged noise-averaged "
                          "profile (multi-pass = free denoise); 'hysteresis' -> per-direction branches, merging "
                          "REFUSED (the average is a curve no pass traced); 'path' -> per-lap curves, no merge. "
                          "Disagreement numbers travel with every verdict",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "x=np.linspace(0,1,80); c=np.concatenate([x,x[::-1],x]); "
                          "m.winding_map(c, np.sin(6*c))['verdict']",
                          native=True, aliases=("hysteresis detection", "up sweep down sweep differ",
                                                "content revisits the same coordinate", "merge multiple scans",
                                                "back and forth sweep", "lap decomposition", "split into laps",
                                                "is it a function or a path", "is my data a function or a path",
                                                "multi pass averaging",
                                                "covering space by direction", "reversing carrier axis"))
    c.register_capability("explore_series", "AUTO-EXPLORE an unlabeled multi-axis series (holographic_scaffold): "
                          "try every axis as the candidate scaffold (score = continuity * (1 - marginal info), "
                          "table returned); rectify the winner's wobbling coordinates; decompose each channel "
                          "along the carrier into its generating law (MDL-gated); recompose and account variance "
                          "-- each channel returns its explained fraction AND its residual (the hand-off to the "
                          "next level). Verdict structured / weakly structured / no structure found, decided by "
                          "measurement; noise is never dressed as law. Raw cube in; schema, laws, leftovers out",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "u=np.linspace(0,1,200); s=np.stack([np.sin(4*np.pi*u), 0.8*u],axis=1); "
                          "m.explore_series(s)['verdict']",
                          native=True, aliases=("explore unlabeled data", "find the primary axis automatically",
                                                "auto decompose a data series", "discover structure without labels",
                                                "what is the schema of this data", "automatic data exploration",
                                                "find patterns and signals automatically", "unsupervised exploration",
                                                "scaffold discovery", "decompose until the boring axis is found",
                                                "explain a raw data cube"))
    c.register_capability("demux_series", "ONE stream, MANY sources (holographic_demux): detect round-robin "
                          "INTERLEAVING in a 1-D stream (the Contact move -- sample i belongs to channel i mod K; "
                          "the stride is FOUND by delta-continuity, recovery is bit-exact, smallest-K Occam over "
                          "the harmonic ladder, honest K=1 when nothing separates), then GROUP channels into "
                          "OBJECTS by |correlation| (a multi-mesh animated delta stream resolves into its meshes; "
                          "mirrored axes included). Each object is ready for explore_series: decode each channel "
                          "separately. Score table + correlation matrix travel as evidence",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "u=np.linspace(0,1,200); x=np.empty(400); x[0::2]=np.sin(6*u); x[1::2]=u; "
                          "m.demux_series(x)['stride']",
                          native=True, aliases=("separate interleaved channels", "demultiplex a stream",
                                                "how many channels are interleaved", "split a multiplexed signal",
                                                "detect multiple objects in one series", "group channels into objects",
                                                "channels that move together", "separate signal channels",
                                                "multiple meshes in one stream", "time division multiplexing",
                                                "decode each channel separately"))
    c.register_capability("cross_channel_links", "find DELAYED-COPY / shared-component links between channels "
                          "(holographic_demux): per ordered pair, scan lags of the normalized cross-correlation; "
                          "a peak at lag L with gain g means channel j ~ g * channel i delayed by L -- structure "
                          "INVISIBLE to per-channel decomposition (a delayed copy of noise decomposes to nothing "
                          "on both channels, yet the pair is lawful together). The residual pass explore_series's "
                          "leftovers exist for; direction falls out of which ordering peaks. Statistical sample "
                          "guard: too few samples for the threshold -> links refused, not fabricated",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "s=np.random.default_rng(0).standard_normal(300); d=np.zeros(300); d[5:]=0.9*s[:-5]; "
                          "m.cross_channel_links(np.stack([s,d],axis=1))['links'][0]",
                          native=True, aliases=("delayed copy of another channel", "cross correlation lag",
                                                "which channel leads which", "echo detection between channels",
                                                "shared components across channels", "lagged relationship",
                                                "residual link analysis", "channel lead lag"))


_PART = "holographic_catalog_p04"


def _selftest():
    """Delegates to holographic_catalog.check_catalog_part -- one home for the shared contract."""
    from holographic.caching_and_storage.holographic_catalog import check_catalog_part
    n = check_catalog_part(_PART, register_p04)
    print("%s selftest OK -- %d capabilities, no internal duplicates" % (_PART, n))


if __name__ == "__main__":
    _selftest()
