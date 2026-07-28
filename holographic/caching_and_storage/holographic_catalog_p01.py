"""holographic_catalog_p01 -- part 1/6 of the capability registry (split from holographic_catalog).

MECHANICAL SPLIT, no edits. holographic_catalog.py hit 81% of the 1 MB agent-read cap, so the file
that makes capabilities discoverable was becoming the one file an agent could not open. The parts are
called IN ORDER by default_catalog() and the emitted catalog is byte-identical -- verified by hashing
every capability field before and after. Order matters: find_capability ranks by score and ties break
by registration order, so a reordering would silently move search results.

Add new capabilities to the LAST part, or to whichever part is topically right -- never to a new file
without registering it in default_catalog(), or it will simply not exist.
"""


def register_p01(c):
    """Register this part's capabilities on `c`. Called by default_catalog() in order."""

    # --- search / recall: the INDICES (audit named ~7) ---
    c.register_capability(
        "Index (search)", "nearest-neighbour / recall over a pile of vectors with ONE interface (Index.nearest(q,k)): "
        "exact cosine scan for small sets, sub-linear RP-forest for large, plus a calibrated abstain",
        example="from holographic.caching_and_storage.holographic_index import Index; Index(vectors, labels=names).nearest(query, k=5)",
        native=True, aliases=("knn", "nearest", "lookup", "recall", "retrieve", "similarity", "search", "index"))
    c.register_capability("holographic_spatial.knn", "EUCLIDEAN k-nearest over a POINT cloud (a spatial grid) -- a "
                          "different metric than the cosine Index; use for geometry, not vectors",
                          example="SpatialGrid(points).knn(query, k)", native=True, aliases=("spatial", "euclidean", "points", "knn"), module="tree", consumes=('points',), produces=('selection',))
    c.register_capability("holographic_rayindex", "which pixels/objects a RAY touches (ray<->object index) -- not a "
                          "nearest(query,k); a distinct spatial ray structure", example="build_ray_index(ctx, camera, w, h)",
                          native=True, aliases=("ray", "pixels", "reshade", "spatial", "bvh"))
    c.register_capability("holographic_tree.HoloForest", "sub-linear approximate nearest-neighbour search over many "
                          "vectors (random-projection forest) with cross-tree agreement", example="HoloForest(V).recall(q,k)",
                          native=True, aliases=("forest", "ann", "knn"), module="tree", consumes=('hypervector',), produces=('selection',))
    c.register_capability("holographic_pivot", "recursive pivot-tree index for nearest-neighbour search",
                          example="from holographic.misc.holographic_pivot import ...", native=True, aliases=("pivot", "index"))
    c.register_capability("holographic_archive", "content-addressable image memory (WHT plates), damage-tolerant",
                          example="from holographic.misc.holographic_archive import ...", native=True, aliases=("image", "store", "recall"), module="archive", consumes=('image',), produces=('image',))

    # --- caching / baking: the CACHES (audit named ~9) = bake_and_query ---
    c.register_capability(
        "Cache (bake-and-query)", "bake a slow evaluator over what VARIES (position/view/time/constant) then look it "
        "up cheaply -- one shared grid-sample core over the scattered bakes (matbake, sdfbake, viewlut, anim)",
        example="from holographic.caching_and_storage.holographic_cachehome import Cache; Cache.bake(fn, vary='position', lo=lo, hi=hi, res=24)",
        native=True, aliases=("bake", "precompute", "lookup", "cache", "memoise", "irradiance", "lut", "grid"))
    c.register_capability("holographic_domecache", "cached DOME / sky-ambient light: bake PRT at coarse anchors, "
                          "smooth interpolate, recompute edges (three-tier)", example="render_scene_document(..., dome_cache=True)",
                          native=True, aliases=("dome", "ambient", "ao", "sky"))
    c.register_capability("holographic_lightcache", "cached SOFT AREA lights + one-bounce INDIRECT / global "
                          "illumination, baked noise-free at anchors (the shared cached_screen_shade engine)",
                          example="render_scene_document(..., soft_light_cache=True, indirect_cache=True)",
                          native=True, aliases=("gi", "indirect", "bounce", "area", "penumbra", "shadow", "speckle"))
    c.register_capability("holographic_modulate", "modulate/demodulate primitive (= bind/unbind): split radiance into "
                          "albedo x irradiance to denoise or upscale the smooth part cleanly",
                          example="from holographic.misc.holographic_modulate import demodulate, remodulate", native=True,
                          aliases=("albedo", "irradiance", "denoise", "upscale", "demodulate"))
    c.register_capability("holographic_matbake", "bake POSITION-dependent material channels to a grid, trilinear "
                          "lookup", example="from holographic.materials_and_texture.holographic_matbake import ...", native=True, aliases=("material", "bake"), consumes=(), produces=('field',))
    c.register_capability("holographic_prt", "precomputed radiance transfer: bake light transport, relight by a dot "
                          "product", example="from holographic.misc.holographic_prt import precompute_transfer, shade_prt", native=True,
                          aliases=("relight", "sh", "transfer", "light"))

    # --- 2D image editing & generation, text generation, language learning, utilities (curated families) ---
    c.register_capability("2D image editing & generation", "the engine's 2D IMAGE toolkit: edit (recolor_image / "
                          "colour transfer, sharpen_loop, svgf_denoise, downscale), generate & blend (blend_images "
                          "crossfade/morph, pattern_field procedural noise/fbm/checker/stripes, svg_canvas vector "
                          "drawing), store & compare (image_archive damage-tolerant recall, compare_images / "
                          "image_distance perceptual similarity). Raster and vector, all on the VSA substrate",
                          example="mind.recolor_image(img, ref); mind.blend_images(a, b); mind.sharpen_image(img); mind.splat_points(pts, cam, 128, 128)",
                          native=True, aliases=("2d", "image", "edit an image", "generate an image", "draw", "draw a picture",
                                                "make a drawing", "paint", "paint on a canvas", "canvas", "sharpen", "blur",
                                                "downscale", "resize", "recolor", "colour transfer", "color transfer",
                                                "crossfade", "morph", "sprite", "vector graphics", "svg", "procedural texture",
                                                "picture", "photo", "raster", "pixels", "deblur", "sharpen an image",
                                                "point cloud", "splat points", "render points to an image", "warp an image"))
    c.register_capability("Image analysis (classic CV)", "SEE with arithmetic (holographic_vision, now mind doors): "
                          "image_edges (self-calibrating Sobel edge map), image_corners (Harris interest points), "
                          "image_lines (Hough dominant lines, edge detection chained in), image_colours (k-means "
                          "palette + fractions), image_signature (one fixed-length descriptor per image -- colour + "
                          "edge-orientation + layout, for retrieval/dedup/perceptual distance), image_classes "
                          "(cluster unlabeled images into k visual classes). Pure NumPy, deterministic per seed",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "img=np.zeros((32,32,3)); img[:,12:20]=[0.9,0.2,0.1]; "
                          "(m.image_edges(img).sum() > 0, m.image_colours(img, k=2)[1].tolist())",
                          native=True, aliases=("find edges in an image", "detect corners in an image",
                                                "find lines in an image", "dominant colors of an image",
                                                "image palette", "cluster images by appearance",
                                                "image feature vector", "perceptual image descriptor",
                                                "analyze an image", "computer vision", "edge detection",
                                                "corner detection", "hough transform", "image similarity"))
    c.register_capability("Segment a photo into object regions (demux)", "DEMUX a photo into per-object REGIONS -- the segmentation front end of the photo->3D pipeline. mind.segment_image(rgb, k) k-means-clusters pixels in (r,g,b,x,y), splits each colour cluster into 4-connected components, merges tiny regions. Returns region dicts largest-first: id, mask, area, fraction, bbox, centroid, mean_color, shape (circle/rectangle/line/triangle), circularity/extent/aspect. Deterministic; numpy+stdlib. HONEST: splits on APPEARANCE not semantics (a shadow can split a floor) -- the per-region stats are a coarse guess the primitive-fit stage refines.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); img=np.zeros((40,40,3)); img[:,:,2]=1.0; img[10:30,10:30]=(1.0,0.0,0.0); [round(r['fraction'],2) for r in m.segment_image(img, k=2)]",
                          native=True, aliases=("segment an image", "segment a photo into objects", "demux a scene into regions",
                                                "separate objects in a photo", "colour segmentation", "color segmentation",
                                                "region segmentation", "split an image into regions", "connected components of an image",
                                                "find objects in an image", "extract objects from a photo", "foreground regions"))
    c.register_capability("Tighten a selection to opaque pixels (auto-shrink marquee)", "SHRINK a rectangular raster selection to its NON-TRANSPARENT content -- the auto-shrink-to-opaque-pixels Photoshop/GIMP do, so a rotate/scale pivots about the DRAWING's centre, not the loose marquee's empty centre. mind.tighten_selection(alpha, bbox, threshold): alpha is (H,W) 0..1 or 0..255, an (H,W,4) RGBA image, or a bool mask; bbox=(r0,c0,r1,c1) inclusive is the marquee (None=whole image). Returns {empty, bbox, centre, area}: bbox is the tight box, centre the (row,col) pivot. empty=True means KEEP the original selection. Deterministic, numpy-only.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); a=np.zeros((100,100)); a[20:30,60:70]=1.0; r=m.tighten_selection(a, bbox=(0,0,99,99)); (r['bbox'], r['centre'])",
                          native=True, aliases=("auto shrink selection to drawn pixels", "shrink selection to non-transparent pixels",
                                                "tighten selection to content", "exclude transparent pixels from selection",
                                                "crop selection to opaque pixels", "trim transparent border from a selection",
                                                "bounding box of the drawn area", "rotate about the drawing centre not the selection box",
                                                "fix rotate pivot for a transparent selection", "shrink marquee to content",
                                                "selection bounds from alpha", "auto crop selection to what I drew"))
    c.register_capability("Build a scene from a photo (image -> editable scene)", "BUILD A SCENE FROM A PHOTO (machine-initialised) -- the demux->fit->assemble front half of image->3D. mind.scene_from_image(image, k, max_objects) segments the photo, keeps the most object-like foreground regions, maps each region's silhouette+colour to a primitive, assembles a live SemanticScene you can adjust/render/refine_to_target/to_node_graph. Returns {scene, regions, roles, objects}. Deterministic. HONEST: shape from silhouette, colour from region mean; DEPTH not reconstructed (z=0) -- a STARTING POINT the critic + drill-down refine; quality bounded by the segmentation.",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); img=np.ones((60,90,3)); yy,xx=np.mgrid[0:60,0:90]; img[(yy-30)**2+(xx-25)**2<=12**2]=(0.85,0.15,0.15); img[20:45,58:82]=(0.15,0.25,0.85); [o['shape'] for o in m.scene_from_image(img, k=3, max_objects=2)['objects']]",
                          native=True, aliases=("build a scene from a photo", "photo to scene", "image to editable scene",
                                                "reconstruct a scene from an image", "model a photo automatically", "photo to 3d scene",
                                                "make a 3d scene from a picture", "auto build a scene from an image", "image to scene",
                                                "turn a photo into a 3d scene", "scene from a photo"))
    c.register_capability("Floor and wall backdrop for a scene", "give a scene a matching FLOOR and WALL so a render competes with a photo's whole frame instead of empty sky. Set scene.environment['ground_color']=(r,g,b) to recolour the floor and scene.environment['backdrop_color']=(r,g,b) to add a vertical wall behind the scene; render() applies both (default None -> neutral gray floor + sky, byte-identical old behaviour). scene_from_image(background=True) sets them AUTOMATICALLY from the photo's floor/wall regions. Measured: a matching backdrop is the single biggest fidelity lever when matching a photo (it is most of the frame).",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.build_scene('a red sphere'); s.environment['ground_color']=(0.2,0.14,0.09); s.environment['backdrop_color']=(0.72,0.72,0.7); s.render(width=64,height=48).shape",
                          native=True, aliases=("add a floor to a scene", "ground plane colour", "wall behind the scene",
                                                "backdrop colour", "set the floor colour", "add a background wall",
                                                "match the photo background", "floor and wall", "environment backdrop"))
    c.register_capability("ascii_view", "render any image to TEXT (holographic_ascii) -- the terminal / log / "
                          "SSH projection backend with a real resolution knob (`width` in characters). Modes by "
                          "detail-per-character: ramp (luminance glyphs, ~70 levels), edge (oriented | / - \\ "
                          "glyphs where the gradient is strong), braille (2x4 dots = 8 pixels per character, "
                          "Bayer-dithered -- the max-detail mode), half (2 full-color pixels per character via "
                          "ANSI fg/bg). ansi='256'|'truecolor' colors any mode; deterministic to the byte and "
                          "fully vectorised (240^2 to 100 columns of braille in ~5 ms)",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.ascii_view(np.tile(np.linspace(0,1,64),(64,1)), width=40, mode='braille'))",
                          native=True, aliases=("ascii art from an image", "render image to terminal",
                                                "print an image as characters", "text representation of an image",
                                                "braille image", "ansi color image", "terminal graphics",
                                                "view a render in the console", "image to text art",
                                                "ascii projection", "console output of an image"))
    c.register_capability("ascii_sdf", "preview a 3-D SDF scene as TEXT (holographic_ascii): raymarch + shade + "
                          "ASCII in one call -- the 'see my SDF over SSH' path, no manual render loop. Takes a "
                          "live SDF, a domain-warped scene, or its DSL text; default camera looks down -z, or "
                          "pass (origin, forward). Modes ramp/edge/braille/half, ansi color, named ramps. Small "
                          "by design (a preview) -- for a full frame, raymarch and pass the image to ascii_view",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_sdf import sphere; "
                          "print(m.ascii_sdf(sphere(1.0), width=40, mode='braille'))",
                          native=True, aliases=("preview an sdf in the terminal", "ascii render an sdf",
                                                "show a signed distance field as text", "raymarch to ascii",
                                                "text preview of a 3d scene", "sdf to ascii", "console sdf preview"))
    c.register_capability("ascii_field", "project a 2-D scalar FIELD straight to TEXT (holographic_ascii) -- "
                          "composability past finished images: hand it any callable f(points)->values (a bake_nd "
                          "slice, a noise function, a heightmap), it samples over a region, self-normalises, and "
                          "renders. The seam that lets the ASCII backend consume the engine's native fields, not "
                          "just image arrays. Modes ramp/edge/braille/half, ansi color, named ramps",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.ascii_field(lambda P: np.sin(6*P[:,0])*np.cos(6*P[:,1]), width=40, ramp='blocks'))",
                          native=True, aliases=("ascii a field", "print a field as text", "visualize a field in the terminal",
                                                "render a heightmap as ascii", "text plot of a 2d function",
                                                "field to ascii", "console field plot"))
    c.register_capability("depth_from_image", "SHAPE FROM SHADING: estimate a relative DEPTH MAP from a single "
                          "image (C1 of photo-to-3D) -- no learned weights, no torch. The missing "
                          "front end for photo_to_3d / unproject, which both need a depth map. Returns depth (H,W) "
                          "normalised [0,1]. HONEST: shape-from-shading is ill-posed (bas-relief ambiguity), so "
                          "this is a plausible RELATIVE surface, not metric depth",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "img=np.random.default_rng(0).uniform(0,1,(32,32)); print(m.depth_from_image(img).shape)",
                          native=True, aliases=("estimate depth from a photo", "monocular depth map",
                                                "depth from a single image", "shape from shading",
                                                "depth map from an image", "guess depth from a picture",
                                                "single image depth estimation", "relative depth from shading"))
    c.register_capability("image_to_3d", "END-TO-END PHOTO-TO-3D from a single image (C1->C2->C3): estimate depth "
                          "by shape-from-shading, unproject to camera-space points, and fit per-pixel 3-D GAUSSIANS "
                          "on the confident front-facing pixels (abstaining on edges, grazing angles, and the "
                          "unobserved back). Returns positions/colours/radii/confidences + abstain mask. Single "
                          "view reconstructs the VISIBLE FRONT, not a watertight object",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "img=np.random.default_rng(0).uniform(0,1,(32,32,3)); r=m.image_to_3d(img); print(r['positions'].shape)",
                          native=True, aliases=("3d gaussians from an image", "image to gaussian splats",
                                                "photo to 3d", "picture to 3d points", "gaussian splatting from a photo",
                                                "3d from a single photo", "image to point cloud", "photo to gaussians",
                                                "3d from one image", "turn a photo into 3d", "photo to 3d model"))
    c.register_capability("image_to_mesh", "END-TO-END image -> MESH (the visible FRONT, NOT a watertight solid): "
                          "shape-from-shading depth, unproject to points, oriented normals, then surface "
                          "reconstruction (dual contouring). Returns (verts, quads, field, grids). Single-view + "
                          "relative depth, so it meshes a height-field surface -- for splats use image_to_3d. "
                          "repair=True runs weld+split-nonmanifold+fill (default-off, byte-identical): MEASURED, it "
                          "turns the dual-contour output MANIFOLD (non-manifold edges -> 0) so the cross-field retopo "
                          "accepts it -- pass repair=True then mesh_repair(triangulate=True) for a retopo-ready mesh",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "img=np.random.default_rng(0).uniform(0,1,(24,24)); v,q,f,g=m.image_to_mesh(img,res=32); print(len(v)>0)",
                          native=True, aliases=("mesh from a photo", "image to 3d mesh", "reconstruct a mesh from a picture",
                                                "photo to mesh", "surface reconstruction from an image",
                                                "3d model from a photo", "picture to mesh", "photogrammetry"))
    c.register_capability("four_surface_demo", "ONE KERNEL, FOUR SURFACES (W19): given one SDF scene, return its "
                          "four backend representations -- GLSL (Shadertoy), WGSL (browser GPU), a braille ASCII "
                          "raymarch, and the canonical DSL text -- all provably the same field (the C emission "
                          "matches the CPU eval that the ascii/PNG paths march). Author once, render everywhere; "
                          "the demo that explains the whole engine in one screen",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_sdf import box; "
                          "d=m.four_surface_demo(box(0.4,0.4,0.4).rounded(0.1)); print(sorted(d.keys()))",
                          native=True, aliases=("one kernel four surfaces", "same scene four ways",
                                                "render a scene as glsl wgsl ascii", "author once render everywhere",
                                                "all backends of a scene", "scene to every format"))
    c.register_capability("2D SDF + extrude/revolve", "2-D signed distance shapes and the operators that lift them "
                          "into 3-D (holographic_sdf2d, W10): draw a cross-section (circle, box, rounded_box, "
                          "ngon, polygon) then EXTRUDE it into a prism along Z (a logo -> a badge, a gear profile "
                          "-> a gear) or REVOLVE it around Y into a solid of revolution (a vase, a bottle; an "
                          "offset circle -> a torus, exact). The result is a 3-D SDF that raymarches / meshes / "
                          "voxelizes like any other",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "prism=m.sdf_extrude(m.sdf2d('ngon', sides=6, r=0.8), height=0.3); print(prism(__import__('numpy').zeros((1,3))).round(3))",
                          native=True, aliases=("2d sdf", "2d sdf shape", "extrude a 2d profile", "revolve a profile",
                                                "lathe a shape", "solid of revolution", "extrude a shape",
                                                "spin a profile", "prism from a cross section", "polygon sdf",
                                                "2d shape to 3d", "make a vase", "extrude a logo"), consumes=(), produces=('sdf',))
    c.register_capability("sdf_curvature", "MEAN CURVATURE of an SDF surface (W13) -- the field Laplacian "
                          "(divergence of the unit gradient). POSITIVE on convex edges/ridges, NEGATIVE in "
                          "concave creases/cavities, ~0 on flat regions (a sphere of radius r reads 2/r). Drives "
                          "cavity darkening, edge highlighting, and curvature-aware LOD -- the shading cue behind "
                          "the cavity/edge look",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_sdf import sphere; print(m.sdf_curvature(sphere(1.0), np.array([[1.,0,0]])).round(2))",
                          native=True, aliases=("sdf curvature", "mean curvature of a surface", "surface curvature",
                                                "cavity shading", "edge detection on an sdf", "convexity of a shape",
                                                "curvature shading", "ridge and valley detection"))
    c.register_capability("warped_noise", "DOMAIN-WARPED fBm (W11, iq's warped noise / dFBM) -- fbm sampled at a "
                          "point displaced by a vector of other fbm fields, giving the swirling, flowing, marbled "
                          "look plain fbm cannot make: smoke, magma, wood grain, weather fronts. Returns "
                          "f(points)->[0,1]; warp=0 reduces to plain fbm. The most demoscene-recognisable noise",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "f=m.warped_noise(scale=2.0,seed=0,warp=0.5); print(f(np.zeros((1,3))).round(3))",
                          native=True, aliases=("domain warped fbm", "warped noise", "turbulence noise", "flow noise",
                                                "swirling noise", "marble texture", "smoke noise", "dfbm",
                                                "fbm domain warp", "flowing procedural texture"))
    c.register_capability("ladder_forecast_calibrated", "forecast a numeric series with the ladder predictor "
                          "wrapped in a CALIBRATED prediction interval (holographic_ladder) -- an uncalibrated "
                          "forecast is not a measurement. Rolls the predictor over the series to gather residuals "
                          "on held-out data, calibrates a conformal forecaster, and returns the next point forecast "
                          "plus an interval with MEASURED coverage (not assumed). Falls back to point-only when the "
                          "history is too short to calibrate honestly",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "r=m.ladder_forecast_calibrated([0,1,2,3,4]*30); print(r['interval'] is not None)",
                          native=True, aliases=("forecast with a confidence interval", "calibrated forecast",
                                                "prediction interval for a series", "forecast with error bars",
                                                "how sure is this forecast", "conformal forecast",
                                                "forecast with measured coverage", "next value with an interval"))
    c.register_capability("edit_history", "the UNDO/REDO log AND EDITABLE CONSTRUCTION HISTORY for an interactive "
                          "edit session (holographic_edithistory) -- an EditHistory you thread scene state through: "
                          "do(state, cmd) applies and records, undo/redo walk it bit-identically (tie-safe replay). "
                          "Also .rebuild(base) replays the whole recipe, and .replace_command(i, new_cmd, base) "
                          "edits a PAST operation's parameters and re-evaluates downstream (the Maya/C4D reach-back). "
                          "Build commands with vertex_move_command / capture_edit_command",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "h=m.edit_history(); P=[[0,0,0],[1,0,0]]; "
                          "s=h.do(P,m.vertex_move_command([1],[0,1,0])); print(np.allclose(h.undo(s),P))",
                          native=True, aliases=("undo redo", "undo a geometry edit", "edit history",
                                                "command log for editing", "reversible edit stack",
                                                "undo a mesh edit", "editable construction history",
                                                "edit a past operation parameter", "parametric history",
                                                "re-evaluate a recipe with changed parameters"))
    c.register_capability("vertex_move_command", "a reversible VERTEX MOVE command (holographic_edithistory) for "
                          "the undo log -- apply adds a delta to the given vertices, invert subtracts it "
                          "(closed-form inverse, O(edit) memory). Feed to edit_history.do",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.vertex_move_command([1],[0,1,0]).name)",
                          native=True, aliases=("reversible move command", "undoable vertex move",
                                                "move command for undo", "record a vertex move",
                                                "make a move undoable"))
    c.register_capability("capture_edit_command", "wrap an ARBITRARY geometry edit into a reversible command "
                          "(holographic_edithistory) by snapshotting before/after positions of just the touched "
                          "vertices -- O(edit) memory, for edits with no cheap algebraic inverse (a bevel, a "
                          "smooth). Feed to edit_history.do",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.capture_edit_command([0],[[9,9,9]],[[0,0,0]]).name)",
                          native=True, aliases=("make any edit undoable", "record an arbitrary edit",
                                                "snapshot inverse command", "wrap an edit for undo",
                                                "undoable geometry edit"))
    c.register_capability("residue_system", "exact integer arithmetic in vectors via a RESIDUE NUMBER SYSTEM "
                          "(holographic_extras) -- encode integers in [0,M) as CRT residues carried in "
                          "hypervectors, then add/subtract/scale with vector ops that are EXACT (no floating "
                          "error), decoding back to the integer. The number-theoretic view of VSA bundling",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "rs=m.residue_system([3,5,7]); "
                          "print(rs.decode(rs.add(rs.encode(20),rs.encode(30))))",
                          native=True, aliases=("residue number system", "exact modular arithmetic",
                                                "crt integer arithmetic", "modular arithmetic in vectors",
                                                "exact integer math with hypervectors"))
    c.register_capability("vsa_region", "a REGION of space as a signed-distance ball with boolean algebra "
                          "(holographic_extras) -- union/intersect/subtract/complement of spherical regions, plus "
                          "contains() and steer(). The set-algebra complement to sdf_scene: compose regions of "
                          "interest for selection or routing",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "r=m.vsa_region([0,0,1.0],1.0).union(m.vsa_region([0,0,-1.0],1.0)); "
                          "print(bool(r.contains([0,0,1.0])))",
                          native=True, aliases=("region of space", "spherical region algebra",
                                                "region of interest", "boolean region composition",
                                                "combine regions of space"))
    c.register_capability("predictive_filter", "a SURPRISE filter (holographic_extras) -- observe(vec) returns "
                          "(is_novel, surprise); slow drift is absorbed by a moving prediction while an abrupt "
                          "change fires once. Pass only surprising observations downstream, stay quiet on "
                          "predictable ones -- an event gate for a stream",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "pf=m.predictive_filter(); print(pf.observe(np.ones(64))[0] in (True,False))",
                          native=True, aliases=("surprise filter", "novelty detector", "event gate for a stream",
                                                "predictive novelty filter", "only report surprising observations"))
    c.register_capability("sdf_scene", "build an SDF SCENE from parts (holographic_sdfscene) -- 'a scene is a set "
                          "of SDF parts'. Pass (sdf_fn, material) pairs and optional (center,radius) bounds; get "
                          ".eval (nearest-surface distance = min over parts, what a ray-marcher calls), .part_ids / "
                          ".material_at (argmin, material lookup), .parts_near (spatial cull). The SDF-scene state "
                          "model, composing parts the way a splat scene bundles primitives",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "sc=m.sdf_scene([(lambda p: np.linalg.norm(np.asarray(p,float),axis=-1)-1.0,'red')]); "
                          "print(float(sc.eval(np.array([[0,0,0.0]]))[0]))",
                          native=True, aliases=("sdf scene", "compose sdf parts", "scene of sdf primitives",
                                                "build a scene from signed distance functions",
                                                "sdf scene with materials", "combine sdf shapes into a scene"),
                          semantic="create/scene",
                          consumes=("sdf",), produces=("sdf_scene",))
    c.register_capability("snap_to_grid", "GEOMETRIC grid snap (holographic_snap) -- snap a 3-D point to the "
                          "nearest grid node of spacing `increment` (scalar or per-axis; a zero axis is left "
                          "alone). The 'snap to grid' a modeler holds Ctrl for. Distinct from guide_snap (VSA "
                          "codebook cleanup)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.snap_to_grid([0.4,0.6,-0.3],1.0))",
                          native=True, aliases=("snap to grid", "round to grid increment", "grid snapping",
                                                "snap a point to the grid", "quantize to grid"),
                          semantic="transform/snap",
                          consumes=("points",), produces=("points",))
    c.register_capability("snap_to_vertices", "snap a point to the NEAREST vertex (holographic_snap) -- returns "
                          "{index, position, distance} or None if beyond max_dist. The vertex-snap that makes two "
                          "verts coincide exactly",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.snap_to_vertices([4.6,0.1,0.0],[[0,0,0],[5,0,0]])['index'])",
                          native=True, aliases=("snap to nearest vertex", "snap a vertex to another",
                                                "vertex snapping", "snap to a point", "find nearest vertex to snap"),
                          semantic="transform/snap",
                          consumes=("points",), produces=("points",))
    c.register_capability("snap_transform_delta", "snap a TRANSFORM DELTA so the dragged point lands on a target "
                          "(holographic_snap) -- target 'grid'/'vertex'/'edge'; returns {delta (corrected), "
                          "snapped_to}. The form the gizmo uses: it has a raw delta and the point being dragged, and "
                          "wants the delta adjusted so that point snaps. Keeps transform and snap layers separate",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.snap_transform_delta([0.4,0,0],'grid',1.0,moved_point=[0.4,0,0])['snapped_to'])",
                          native=True, aliases=("snap a move to the grid", "snap while dragging",
                                                "snap a transform", "constrain a move to a snap target",
                                                "snap the gizmo delta"),
                          semantic="transform/snap",
                          consumes=("transform",), produces=("transform",))
    c.register_capability("transform_selection", "the GIZMO BACKEND (holographic_transform_space) -- transform "
                          "selected vertices about a PIVOT (median/active/cursor/bbox), in a SPACE "
                          "(world/local/view), under an axis CONSTRAINT mask: the triple that turns a raw matrix "
                          "into the move/rotate/scale a modeler expects. translate/rotate/scale about the pivot; "
                          "pass weights for PROPORTIONAL editing. Non-destructive",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "P=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]; "
                          "print(m.transform_selection(P,[0,1,2,3],translate=[1,1,1],constraint=(1,0,0))[0])",
                          native=True, aliases=("translate rotate scale a selection", "move a selection",
                                                "gizmo transform", "axis constrained move", "transform in a space",
                                                "rotate about a pivot", "proportional edit transform"),
                          semantic="transform/gizmo",
                          consumes=("mesh", "selection", "transform"), produces=("mesh",))
    c.register_capability("pivot_point", "resolve the PIVOT for a transform (holographic_transform_space) -- "
                          "'median' (centroid), 'bbox' (box centre), 'cursor' (a given point), or 'active' (a "
                          "chosen vertex). The point a rotate/scale turns around",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.pivot_point([[0,0,0],[2,0,0]],[0,1],'bbox'))",
                          native=True, aliases=("pivot point", "transform pivot", "center of a selection",
                                                "rotation center", "where to rotate around"),
                          semantic="transform/pivot",
                          consumes=("mesh", "selection"), produces=("transform",))
    c.register_capability("pick_mesh", "VIEWPORT PICK on a REAL mesh (holographic_raypick) -- from a cursor (u,v in "
                          "-1..1) return the nearest 'face' or 'vertex' clicked, as {kind, index, position, "
                          "distance} or index:None on a miss. The generalization of pick_element (demo cage) onto a "
                          "user's arbitrary geometry -- one call from 'clicked here' to 'selected this'",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "mesh={'vertices':[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],'faces':[[0,1,2,3]]}; "
                          "print(m.pick_mesh(mesh,0.0,0.0)['index'])",
                          native=True, aliases=("pick a face on a mesh", "click to select a mesh element",
                                                "viewport pick real geometry", "select geometry under the cursor",
                                                "pick mesh by screen position"),
                          semantic="select/pick",
                          consumes=("mesh",), produces=("selection",))
    c.register_capability("ray_mesh_intersect", "RAY-VS-MESH picking (holographic_raypick) -- cast a ray at a mesh "
                          "and return the NEAREST hit {face, position, distance, barycentric} or None. "
                          "Moller-Trumbore per triangle with an AABB broad phase; quads report the original face. "
                          "How viewport picking hits a user's real geometry",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "mesh={'vertices':[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],'faces':[[0,1,2,3]]}; "
                          "print(m.ray_mesh_intersect(mesh,[0,0,5],[0,0,-1])['face'])",
                          native=True, aliases=("ray triangle intersection", "cast a ray at a mesh",
                                                "ray hits a mesh face", "pick a face with a ray",
                                                "moller trumbore", "ray mesh hit test"),
                          semantic="select/pick",
                          consumes=("mesh",), produces=("scalar",))
    c.register_capability("ray_sdf_intersect", "RAY-VS-SDF picking (holographic_raypick) -- sphere-trace a ray into "
                          "an SDF (any sdf_fn(pt)->distance) and return the hit {position, distance, normal, steps} "
                          "or None. The native pick for the field/procedural half of a scene -- exact to the field, "
                          "no triangulation",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "sph=lambda p: float(np.linalg.norm(np.asarray(p,float))-1.0); "
                          "print(round(m.ray_sdf_intersect(sph,[0,0,3],[0,0,-1])['distance'],1))",
                          native=True, aliases=("ray march an sdf", "cast a ray into an sdf",
                                                "sphere trace a ray", "sdf ray hit", "raymarch pick"),
                          semantic="select/pick",
                          consumes=("sdf",), produces=("scalar",))
    c.register_capability("screen_ray", "build a world-space RAY from a screen coordinate (holographic_raypick) -- "
                          "(u,v) in -1..1 under the cursor -> (origin, direction), so 'the user clicked here' "
                          "becomes a geometry query for ray_mesh_intersect / ray_sdf_intersect",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "o,d=m.screen_ray(0.0,0.0); print(o)",
                          native=True, aliases=("screen to world ray", "cursor to ray", "unproject a screen point",
                                                "make a pick ray", "ray from a screen coordinate"),
                          semantic="select/pick")
    c.register_capability("skin_bind_weights", "AUTO-SKIN BINDING (holographic_meshskin) -- compute per-vertex bone "
                          "weights from bone anchor points, the 'bind' step that produces the weights skin_mesh "
                          "consumes. Inverse-distance falloff to the nearest bones, keeping max_influences and "
                          "renormalizing to a PARTITION OF UNITY (rigid motion stays exact). The distance-based "
                          "auto-bind a rig starts from",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "w=m.skin_bind_weights([[0,0,0],[5,0,0.0]],[[0,0,0],[5,0,0.0]],max_influences=2); "
                          "print(np.round(w.sum(axis=1),3).tolist())",
                          native=True, aliases=("bind mesh to skeleton", "compute skin weights from bones",
                                                "automatic skin weights", "rig bind weights",
                                                "distance based skin binding", "skin binding"),
                          semantic="animate/skin",
                          consumes=("mesh", "skeleton"), produces=("scalar",))
    c.register_capability("transport", "An animation TRANSPORT / playhead (holographic_anim) -- start/pause/step/seek/"
                          "scrub/rewind/fast-forward over a frame function, which the keyframe timeline + frame cache "
                          "lacked. frame_fn(frame)->state computes any frame on demand; caches computed frames so "
                          "rewind/scrub-back/replay is O(1). play(speed): 1=fwd, -1=rewind, 2=fast-forward, 0.5=slow. "
                          "Deterministic scrub (same state however you arrived)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); import numpy as np; "
                          "t=m.transport(lambda f: np.array([[float(f),0,0]]), n_frames=10); "
                          "t.seek(5); print(t.frame)",
                          native=True, aliases=("play an animation", "pause the simulation", "rewind to a frame",
                                                "scrub the timeline", "fast forward animation", "seek to a frame",
                                                "animation playhead", "step through frames"),
                          semantic="modify/transform", consumes=(), produces=("scalar",))
    c.register_capability("field_displace", "Displace a mesh's vertices along their normals by a SCALAR FIELD or SDF "
                          "sampled at each vertex (holographic_autodisplace) -- the field-driven modifier. field is "
                          "any .eval SDF (mandelbulb/fold_fractal) or a callable, so a FRACTAL drives the relief. An "
                          "optional per-vertex weight MASK (from a texture map) gates it so detail grows only where "
                          "the map paints -- the per-face fractal modifier. Generalizes auto_displace beyond RGB",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import grid; "
                          "print(m.field_displace(grid(nx=16,ny=16), m.mandelbulb(iterations=5), amount=0.2).n_faces)",
                          native=True, aliases=("displace a mesh by a fractal", "per face modifier from a texture",
                                                "drive geometry from a field", "vertex displacement from an sdf",
                                                "mandelbulb modifier on a mesh", "texture masked displacement",
                                                "apply a fractal modifier to geometry"),
                          semantic="modify/deform", consumes=("mesh",), produces=("mesh",))
    c.register_capability("creature", "Build a Spore-style non-humanoid CREATURE from a body-plan spec "
                          "(holographic_creature) -- a spine with limbs attached at fractional positions, bilateral "
                          "symmetry, and generic organic joint constraints (a cone at each mount, no-hyperextension "
                          "hinges). spec: {spine:{length,segments,axis,curve}, limbs:[{at,dir,segments,length,radius,"
                          "mirror,cone_deg,hinge_deg}], head, body:<morph block>}. Returns the Creature + its morph-"
                          "aware skin SDF (meshes, emits Shadertoy). Generalises the humanoid to arbitrary body plans",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "cre,body=m.creature(m.quadruped_spec()); print(len(cre.chains),'mainImage' in m.to_shadertoy(body))",
                          native=True, aliases=("build a creature from parts", "procedural creature body",
                                                "spore creature editor", "make a quadruped", "non-humanoid rig",
                                                "spine with limbs", "custom animal body", "tentacled creature"),
                          semantic="create/emit", consumes=(), produces=("sdf",))
    c.register_capability("creature_pose", "Build a CREATURE from a spec and pose its limbs to targets via CONSTRAINED "
                          "IK in one deterministic call (holographic_creature). targets = {chain_name: (x,y,z)}; chain "
                          "names are 'L0','L0m','L1',... (m = mirrored twin). Joint limits (muscle/fat tightened) are "
                          "enforced so limbs never hyperextend. Returns (Creature, skin_sdf)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "cre,body=m.creature_pose(m.quadruped_spec(), {'L0':(0.3,-0.5,0.4)}); print('mainImage' in m.to_shadertoy(body))",
                          native=True, aliases=("pose a creature", "animate creature limbs", "reach a creature leg",
                                                "pose a non-humanoid", "put a creature in a pose"),
                          semantic="animate/pose", consumes=(), produces=("sdf",))
    c.register_capability("quadruped_spec", "A ready-made creature body plan -- a quadruped (spine + two mirrored leg "
                          "pairs + head) (holographic_creature). A concrete starting spec for creature(); copy + edit "
                          "the dict to change proportions, add limbs, or attach a head",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(len(m.quadruped_spec()['limbs']))",
                          native=True, aliases=("quadruped body plan", "four legged creature template",
                                                "animal spec", "starter creature spec"),
                          semantic="create/emit", consumes=(), produces=("scalar",))
    c.register_capability("solve_ik_limited", "CONSTRAINED inverse kinematics with anatomical JOINT LIMITS "
                          "(holographic_iklimit) -- reach a target while keeping each joint in range: no hyperextended "
                          "elbows/knees (one-direction hinge), ball joints within a cone. Constrained FABRIK "
                          "(Aristidou-Lasenby): alternates a FABRIK reach with a root->tip limit projection. `limits` "
                          "is per-bone None/hinge/cone in radians (hinge axis may be 'auto' so the bend plane follows "
                          "the limb). Returns (joints, reach_error); error>0 when limits correctly block an out-of-"
                          "range target. Kept negative: angle limits only, no self-collision",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "arm=np.array([[0,0,0.],[0.4,0,0],[0.8,0,0]]); "
                          "lim=[None,{'type':'hinge','axis':'auto','lo':0.0,'hi':2.6}]; "
                          "print(round(m.solve_ik_limited(arm,np.array([0.3,0,0.4]),lim)[1],2))",
                          native=True, aliases=("constrained inverse kinematics", "ik with joint limits",
                                                "prevent hyperextension", "natural pose ik", "clamp joint angles",
                                                "limited ik solver", "range of motion ik"),
                          semantic="analyze/measure", consumes=("points",), produces=("points",))
    c.register_capability("humanoid", "Build a parametric biped HUMANOID with automatic IK rigging + CHARACTER-EDITOR "
                          "morphs (holographic_humanoid) -- a named skeleton + a morphable primitive skin. Pose limbs "
                          "by IK targets (FABRIK, keeps bone lengths). `body` params (see body_params) drive game-"
                          "style sliders: global weight/muscle/fat distributed across the body by region, per-segment "
                          "muscle/fat/length, and optional breast geometry (size/sag/separation/nipple). Returns the "
                          "Humanoid + its morphed skin SDF (meshes, emits Shadertoy). Base build is unchanged at 0",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "b=m.body_params(); b['muscle']=0.6; h,body=m.humanoid(body=b); print('mainImage' in m.to_shadertoy(body))",
                          native=True, aliases=("make a humanoid", "biped character rig", "human figure model",
                                                "stick figure with ik", "poseable character", "rigged human body",
                                                "humanoid with inverse kinematics", "customizable character body"),
                          semantic="create/emit", consumes=(), produces=("sdf",))
    c.register_capability("body_params", "The neutral CHARACTER-EDITOR parameter block for humanoid() "
                          "(holographic_humanoid) -- every slider at 0. Copy + adjust: global weight/muscle/fat in "
                          "[-1,1] (distributed across the body by region); segments[name] = {muscle, fat, length} for "
                          "torso/neck/shoulder/upper_arm/forearm/hip/thigh/shin; breasts = None or {size, sag, "
                          "separation, nipple_diameter, nipple_depth}. Pass as humanoid(body=...)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "b=m.body_params(); b['fat']=0.5; print(sorted(b.keys()))",
                          native=True, aliases=("character editor sliders", "body morph controls",
                                                "muscle and fat sliders", "body customization parameters",
                                                "humanoid body sliders", "weight muscle fat controls"),
                          semantic="create/emit", consumes=(), produces=("scalar",))
    c.register_capability("fit_pose", "Fit a HUMANOID rig to KEYPOINTS -- the honest 'approximate a pose' "
                          "(holographic_humanoid). 3-D keypoints (joint -> xyz, e.g. mocap) -> a direct IK fit; 2-D "
                          "image keypoints (joint -> uv) + a camera -> a bone-length-constrained lift + IK. Returns "
                          "the posed Humanoid. KEPT NEGATIVE: fits KEYPOINTS, does NOT detect them in pixels (that "
                          "needs a learned model the engine forbids); a monocular 2-D lift is depth-ambiguous (A "
                          "plausible pose, not THE unique one)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "h=m.fit_pose({'l_wrist':(0.4,0.9,0.2),'r_wrist':(-0.5,0.3,0.1)}); print(round(float(h.joints['l_wrist'][0]),1))",
                          native=True, aliases=("fit a pose to keypoints", "pose a skeleton to joints",
                                                "estimate pose from keypoints", "match a rig to joint positions",
                                                "pose from mocap points", "fit a humanoid to points",
                                                "approximate pose from keypoints"),
                          semantic="analyze/measure", consumes=("points",), produces=("sdf",))
    c.register_capability("fit_primitives", "Approximate a (M,3) point cloud with a UNION of PRIMITIVES, best-fit per "
                          "cluster (holographic_primfit) -- the honest model for a HARD-SURFACE or NON-FRACTAL organic "
                          "shape (a 'creature', a part) that fold_fractal and the affine-IFS library can't represent. "
                          "Per cluster it fits a SPHERE (round), an ORIENTED BOX (blocky, via PCA), and a CAPSULE "
                          "(elongated limb) and keeps the best -- unioned into an EXACT SDF you can raymarch / "
                          "sdf_to_mesh / to_shadertoy. quality = improvement over one bounding sphere; auto_k grows K",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "rng=np.random.default_rng(0); d=rng.normal(size=(400,3)); d/=np.linalg.norm(d,axis=1,keepdims=True); "
                          "print(m.fit_primitives(d*0.7, k=4)['kinds'])",
                          native=True, aliases=("approximate a shape with primitives", "fit sdf primitives to a shape",
                                                "sphere box capsule fit", "decompose a shape into primitives",
                                                "cover a point cloud with primitives", "fit a creature with primitives",
                                                "union of spheres boxes capsules"),
                          semantic="analyze/measure", consumes=("points",), produces=("sdf",))
    c.register_capability("ifs_generate", "Generate a plant/fractal point cloud from an AFFINE IFS via the chaos game "
                          "(holographic_ifs) -- a Barnsley fern, fractal tree, sierpinski, dragon, ... from a handful "
                          "of 6-number affine maps. The botanical/branching model that fold_fractal (a Mandelbox fold) "
                          "is not. Pass a named system or an AffineIFS; get (n,2) points. Mesh via sdf_from_points -> "
                          "sdf_to_mesh for geometry",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.ifs_generate('barnsley_fern', n=5000).shape)",
                          native=True, aliases=("generate a fern", "barnsley fern", "make a fractal tree",
                                                "chaos game fractal", "sierpinski triangle points",
                                                "affine ifs attractor", "draw a fern"),
                          semantic="create/emit", consumes=(), produces=("points",))
    c.register_capability("ifs_fit", "Match a 2-D point cloud to the CLOSEST NAMED affine-IFS system (holographic_ifs) "
                          "-- the honest 'fit a fern/tree': snap to the closest of {barnsley_fern, culcita_fern, "
                          "sierpinski, fractal_tree, dragon_curve} by occupancy signature, with a measured baseline. "
                          "quality beats baseline when the target really resembles a known system. The botanical "
                          "companion to fold_fit (Mandelbox). Kept negative: snap-to-library, not arbitrary-IFS "
                          "recovery, not rotation-invariant",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.ifs_fit(m.ifs_generate('barnsley_fern', n=5000))['name'])",
                          native=True, aliases=("fit a fern", "which fractal is this point cloud", "identify a plant fractal",
                                                "match a point cloud to a named fractal", "fit an affine ifs",
                                                "recognize a fern or tree", "what plant fractal is this"),
                          semantic="analyze/measure", consumes=("points",), produces=("scalar",))
    c.register_capability("fit_shape", "CLOSEST-FIT a target to a procedural formula + its SHADERTOY / GLSL "
                          "(holographic_fitshape) -- the capstone. An (M,3) POINT CLOUD -> a fractal SDF recipe via "
                          "fold_fit, emitted as a Shadertoy raymarch shader; an (M,2) POINT CLOUD -> the closest NAMED "
                          "affine-IFS (fern/tree/sierpinski via ifs_fit); a 2-D IMAGE/HEIGHT/TEXTURE -> a procedural "
                          "fBm matched to its statistical signature + a GLSL snippet. Reports measured quality vs "
                          "baseline + a note. Kept negative: texture path is a family match, not parameter recovery",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_foldfit import surface_points; "
                          "print(m.fit_shape(surface_points((2.1,0.5,1.0),n=200))['kind'])",
                          native=True, aliases=("find the closest formula for a shape", "fit a shape and get shadertoy",
                                                "match a model to a fractal", "closest procedural fit",
                                                "shape to shadertoy code", "fit a texture to a formula",
                                                "represent a shape with an equation", "what formula makes this shape"),
                          semantic="analyze/measure", consumes=("points",), produces=("scalar",))
    c.register_capability("to_shadertoy", "Emit a complete runnable SHADERTOY fragment shader for an SDF "
                          "(holographic_sdf) -- map + raymarch + normals + lighting + mainImage, ready for "
                          "shadertoy.com. Works for the fractal SDFs (fold_fractal/mandelbulb/menger) too, with a "
                          "header note that a distance estimate needs conservative steps. The 'get the shadertoy code' "
                          "primitive",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print('mainImage' in m.to_shadertoy(m.mandelbulb(iterations=6)))",
                          native=True, aliases=("get the shadertoy code", "export an sdf to shadertoy",
                                                "emit a fragment shader", "sdf to runnable glsl",
                                                "make a shadertoy from a fractal", "raymarch shader for an sdf"),
                          semantic="convert/emit", consumes=("sdf",), produces=("scalar",))
    c.register_capability("sdf_to_mesh", "FRACTAL / SDF -> MESH, the one-liner (holographic bridge) -- march an SDF "
                          "object (fold_fractal/mandelbulb/menger/any .eval field) to a watertight Mesh ready for "
                          "mesh_to_softbody and the whole mesh+simulation pipeline. Fixes the two traps: an SDF isn't "
                          "a bare callable (wraps .eval), and an all-positive distance ESTIMATOR returns 0 faces at "
                          "level 0 (auto-offsets the iso). bounds auto-probed",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.sdf_to_mesh(m.mandelbulb(iterations=6), resolution=32).n_faces)",
                          native=True, aliases=("mesh a fractal", "convert an sdf to a mesh", "polygonize a mandelbulb",
                                                "marching cubes on a fractal", "turn a distance field into a mesh",
                                                "make a static mesh from an sdf", "fractal to geometry"),
                          semantic="convert/emit", consumes=("sdf",), produces=("mesh",))
    c.register_capability("fold_fit", "INFER a fold RECIPE from an observed point cloud (holographic_foldfit) -- the "
                          "INVERSE of fold_fractal. Recover the (scale,min_radius,fold_limit) whose Mandelbox fractal "
                          "best fits a (M,3) target: a coarse grid over recipe space then a local refine via optimize. "
                          "The pattern-recognition payoff -- self-similarity detection as parameter estimation. Returns "
                          "{recipe,loss,baseline,improved}; the baseline-improvement RATIO is the discriminative signal "
                          "(the loss is necessary not sufficient -- a DE can contain the points)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_foldfit import surface_points; "
                          "t=surface_points((2.1,0.5,1.0),n=200); print(m.fold_fit(t)['improved'])",
                          native=True, aliases=("fit a fractal recipe to a shape", "infer IFS from a point cloud",
                                                "recover fold parameters", "inverse fractal problem",
                                                "self-similarity fit", "estimate a mandelbox recipe",
                                                "what fractal made this"),
                          semantic="analyze/measure", consumes=("points",), produces=("scalar",))
    c.register_capability("milk_parse", "PARSE a Milkdrop `.milk` preset (holographic_milkdrop) into settings + "
                          "per_frame_init/per_frame/per_pixel equation families + captured warp/comp shaders. Then "
                          "run_frame(state, audio, time, frame) evaluates the per-frame equations deterministically, "
                          "driving the motion vars from audio envelopes (pair with audio_param_bus). The EQUATION "
                          "layer; warp mesh + pixel shaders are stored for the renderer, not run here",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "p=m.milk_parse('per_frame_1=q1 = q1 + 1\\nzoom=1.0'); "
                          "s=p.initial_state(); p.run_frame(s, {'bass':1.0}); print(s['q1'])",
                          native=True, aliases=("parse a milkdrop preset", "read a .milk file", "load a milk preset",
                                                "milkdrop preset reader", "import a milkdrop visualization",
                                                "run milkdrop equations"),
                          semantic="convert/parse", consumes=(), produces=("scalar",))
    c.register_capability("milk_eval", "Evaluate ONE ns-eel2 expression (Milkdrop's equation language) against a "
                          "variable dict (holographic_milkdrop) -- SAFE (a whitelisted recursive-descent grammar, "
                          "never Python eval), deterministic. Unknown vars read as 0, divide-by-zero is 0, an "
                          "unsupported function raises. The safe expression evaluator milk_parse compiles per equation",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.milk_eval('sqrt(sqr(3)+sqr(4)) + bass', {'bass': 1.0}))",
                          native=True, aliases=("evaluate a milkdrop expression", "ns-eel expression evaluator",
                                                "safe math expression evaluator", "eval a preset equation",
                                                "parse and evaluate a formula"),
                          semantic="measure/eval", consumes=(), produces=("scalar",))
    c.register_capability("mandelbulb", "The MANDELBULB distance-estimator SDF (holographic_sdf) -- the 3D Mandelbrot "
                          "analogue (White-Nylander polar power z^n+c in spherical coords, analytic DE). power=8 is "
                          "the classic bulb. The ESCAPE-TIME fractal family in 3D (vs fold_fractal's Mandelbox FOLD "
                          "engine). Raymarches + orbit-traps with the existing renderer",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(round(float(m.mandelbulb().eval([[0,0,0]])[0]),3))",
                          native=True, aliases=("mandelbulb", "3d mandelbrot fractal", "power 8 bulb fractal",
                                                "polar power fractal sdf", "white nylander fractal", "spherical z^n+c fractal"),
                          semantic="create/emit", consumes=(), produces=("sdf",))
    c.register_capability("escape_time", "The 2D ESCAPE-TIME fractal FIELD (holographic_sdf) -- Mandelbrot (default) "
                          "or Julia (julia_c=(re,im)): z -> z^power+c in the complex plane, returned as a (h,w) array "
                          "of SMOOTH continuous escape counts ready for a palette. The 2D sibling of mandelbulb; same "
                          "z^n+c recurrence read as a field. center/span frame the view",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.escape_time(width=64,height=64,max_iter=50).shape)",
                          native=True, aliases=("mandelbrot set", "julia set", "escape time fractal",
                                                "mandelbrot field", "2d fractal escape count", "complex z^2+c fractal",
                                                "draw the mandelbrot set"),
                          semantic="create/emit", consumes=(), produces=("image",))
    c.register_capability("fold_fractal", "The KALEIDOSCOPIC-IFS / MANDELBOX distance-estimator SDF (holographic_sdf) "
                          "-- the general FOLD ENGINE behind the fractal-forums 3D fractals and the Nishitsuji tweet-"
                          "shader look. Iterates box-fold + sphere-fold + scale; a four-float recipe that regenerates "
                          "megabytes of deterministic self-similar structure. Raymarches + orbit-traps with the "
                          "existing renderer. A distance ESTIMATE (inexact)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "ff=m.fold_fractal(iterations=10,scale=2.0); print(round(float(ff.eval([[0.5,0.5,0.5]])[0]),4))",
                          native=True, aliases=("mandelbox fractal", "kaleidoscopic ifs", "fold a fractal",
                                                "iterated fold rotate scale fractal", "KIFS distance estimator",
                                                "box fold sphere fold fractal", "sierpinski by folding",
                                                "nishitsuji fractal shader", "demoscene fractal sdf"),
                          semantic="create/emit", consumes=(), produces=("sdf",))
    c.register_capability("mesh_auto_seam", "AUTO-MARK SEAMS for UV unwrapping (holographic_meshseam) -- choose "
                          "which edges to cut WITHOUT naming a path. Returns the sorted (lo,hi) seam edges (the 'red "
                          "edges' a modeler marks). Where mesh_cut_seam / mesh_shortest_seam cut a GIVEN seam, this "
                          "SELECTS one: method='crease' seams along sharp edges (dihedral > threshold), where an "
                          "artist cuts so the seam is hidden. Empty on a smooth surface (no creases)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import box; "
                          "print(len(m.mesh_auto_seam(m.mesh_triangulate(box(2,2,2)))))",
                          native=True, aliases=("auto mark seams", "automatically place uv seams",
                                                "choose where to cut a mesh for uv", "mark seams by curvature",
                                                "seam along sharp edges", "find seams for unwrapping",
                                                "where to place uv seams"),
                          semantic="analyze/measure", consumes=("mesh",), produces=("selection",))
    c.register_capability("mesh_rip_vertex", "RIP a shared vertex apart (holographic_eulerops) -- give every face "
                          "incident to a vertex its OWN copy at the same position, so the faces are no longer joined "
                          "there. The INVERSE of a weld at one vertex; topology only, positions unchanged (the mesh "
                          "looks identical but is torn there). Ripping a manifold interior vertex opens the surface",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import box; "
                          "print(m.mesh_rip_vertex(box(2,2,2),0).n_vertices)",
                          native=True, aliases=("rip a vertex", "unweld a vertex", "tear a mesh at a vertex",
                                                "split a shared vertex", "separate faces at a vertex",
                                                "duplicate a vertex per face", "rip vertices apart"),
                          semantic="modify/deform", consumes=("mesh",), produces=("mesh",))
    c.register_capability("mesh_split_vertices", "SPLIT every vertex per-face (holographic_eulerops) -- give each "
                          "face its own private copies of its corners, so no two faces share a vertex. The full "
                          "INVERSE of a weld (weld_mesh): a 'polygon soup' with every face independent (flat/faceted "
                          "shading, no shared normals). Positions unchanged. weld_mesh undoes it",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import box; "
                          "print(m.mesh_split_vertices(box(2,2,2)).n_vertices)",
                          native=True, aliases=("split all vertices", "unweld a mesh", "make a polygon soup",
                                                "split vertices to make faces independent", "unindex a mesh",
                                                "flat shade by splitting vertices", "explode shared vertices"),
                          semantic="convert/emit", consumes=("mesh",), produces=("mesh",))
    c.register_capability("mesh_pack_uv", "PACK UV ISLANDS (holographic_meshuv) -- unwrap each connected component "
                          "(UV island) of a mesh SEPARATELY, then lay the islands out in non-overlapping cells of the "
                          "unit UV square. The 'pack islands' / smart-UV step that mesh_lscm and mesh_uv_unwrap skip "
                          "(they solve every piece in one frame, so disconnected islands overlap). Each island scaled "
                          "uniformly (no stretch) into its cell",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import grid; "
                          "print(m.mesh_pack_uv(m.mesh_triangulate(grid(3,3))).shape)",
                          native=True, aliases=("pack uv islands", "smart uv project", "lay out uv islands",
                                                "pack islands in the unit square", "non-overlapping uv layout",
                                                "arrange uv charts", "uv atlas packing"),
                          semantic="convert/emit", consumes=("mesh",), produces=("points",))
    c.register_capability("mesh_fill_holes", "FILL open holes (boundary loops) of a mesh with faces "
                          "(holographic_meshverbs2) -- close it up. mode='fan' caps each loop with a centroid + "
                          "triangle fan (always works); mode='grid' bridges a big even loop with a coarser quad strip "
                          "(Blender grid fill), falling back to fan otherwise. `max_sides` (Blender Sides) fills only "
                          "loops up to that many edges (0=all) -- close small holes, leave a big outer border open. "
                          "The 'fill holes' / 'grid fill' step after a boolean, scan, or deleting a face",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import box; "
                          "from holographic.mesh_and_geometry.holographic_mesh import Mesh; "
                          "b=box(2,2,2); holed=Mesh(b.vertices,[tuple(f) for f in b.faces][1:]); "
                          "print(m.mesh_fill_holes(holed).is_closed())",
                          native=True, aliases=("fill a hole in a mesh", "grid fill a hole", "patch a hole with quads",
                                                "cap an open loop", "close a hole in a mesh", "fill holes",
                                                "fill an open boundary with faces"),
                          semantic="create/emit", consumes=("mesh",), produces=("mesh",))
    c.register_capability("Mesh repair (weld + split non-manifold + fill + compact)", "REPAIR a raw mesh (holographic_meshtools): m.mesh_repair(mesh) WELDS near-dup vertices, SPLITS non-manifold vertices into umbrellas (makes it MANIFOLD so cross-field retopo accepts it), optionally FILLS holes, DROPS unreferenced; triangulate=True gives uniform triangles. Returns (repaired, report) with before/after counts, manifold/closed flags, split count -- makes a marching-cubes / import / boolean / photo-to-mesh result RETOPO-READY. m.mesh_weld / m.mesh_make_manifold are single-step ops. Deterministic; never raises. KEPT NEG: a pure X-junction over-splits into open sheets.",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import Mesh; book=Mesh(np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1.],[0,0,-1]]),[(0,1,2),(0,1,3),(0,1,4),(0,1,5)]); rm,rep=m.mesh_repair(book, fill_holes=False); (book.is_manifold(), rm.is_manifold(), rep['split_vertices'])",
                          native=True, aliases=("repair a broken mesh", "fix a mesh", "weld duplicate vertices", "merge vertices by distance",
                                                "make a mesh watertight", "remove degenerate triangles", "clean up a mesh", "mesh cleanup",
                                                "fix a non-manifold mesh", "make a mesh manifold", "weld a mesh", "heal a mesh", "retopo-ready mesh"),
                          semantic="create/emit", consumes=("mesh",), produces=("mesh",))
    c.register_capability("Split a loaded mesh into per-material submeshes", "mind.split_by_material(loaded_mesh) "
                          "-> ordered {material_name: LoadedMesh}, each reindexed to its own compact vertex set "
                          "with UVs/normals subset to match. A .glb import MERGES the whole scene into one mesh, so "
                          "sampling a multi-material scan with a single texture paints most faces with the WRONG "
                          "image (the fishing-spider file). Split first, then render/LOD each material with its own "
                          "texture. face_material already records the per-face name; this is the one-call path that "
                          "was otherwise re-implemented (group + reindex + subset UVs) by every consumer.",
                          example="import lecore, numpy as np; "
                                  "from holographic.io_and_interop.holographic_assetimport import LoadedMesh; "
                                  "lm=LoadedMesh(np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float), "
                                  "np.array([[0,1,2],[1,3,2]],int), face_material=['red','blue']); "
                                  "print(list(lecore.UnifiedMind(dim=64,seed=0).split_by_material(lm)))",
                          native=True, aliases=("split a mesh by material", "separate a glb into per-material meshes",
                                                "group faces by material", "per material submesh", "one mesh per "
                                                "material", "multi-material scan wrong texture", "split loaded mesh",
                                                "extract submesh for each material"),
                          semantic="convert/split", consumes=("mesh",), produces=("mesh",)),
    c.register_capability("Whole-scene .glb import (multi-mesh, node transforms, per-face materials)", "glb_to_mesh reads the WHOLE glTF scene via gltf.scene_primitives -- THE canonical vertex order (node transforms composed, every primitive concatenated, normals via inverse-transpose, per-face material on Mesh.face_material). Every per-vertex reader rides that ONE walk: load_glb aligns JOINTS/WEIGHTS to the same table and remaps per-skin joint indices into one global list (lm.joint_nodes). WHY: the first-primitive reader returned a 24-vert cube from a 312,578-vert scan, and gave rigged scenes 16 positions against 8 weights. Engine-emitted files round-trip byte-identically.", example="import lecore; from holographic.io_and_interop.holographic_gltf import glb_to_mesh, mesh_to_glb; from holographic.mesh_and_geometry.holographic_mesh import box; m2 = glb_to_mesh(mesh_to_glb(box())); (len(m2.vertices), m2.face_material[:2])", native=True, module="gltf", aliases=("glb imports only part of the model", "multi mesh gltf import", "imported model missing pieces", "gltf node transforms ignored", "glb shows a cube instead of my model", "read all meshes from a glb scene", "rigged glb loads wrong weights", "skin weights dont match vertex count"), semantic="io/import"),

    c.register_capability("Orientation-field preservation check (Extended Gaussian Image)", "m.mesh_egi_compare(ref, mesh) measures ORIENTATION-FIELD preservation: the Extended Gaussian Image (Horn 1984) -- each face's area binned by its normal on the direction sphere -- compared as 1-normalised-L1 in [0,1]. The COMPLEMENT of the silhouette sweep, found while hunting a one-image silhouette check: a decimated sphere keeps silhouette 0.99 while EGI collapses to 0.06 -- outline and surface character are ORTHOGONAL, so guard both. O(F), ~0.14s on 322k faces, translation-invariant. NOT on the guard's 0.95 IoU scale; read it as how much surface character changed.", example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; b=triangulate_ngons(box()); r=m.mesh_egi_compare(b, b); r['similarity']", native=True, module="render", aliases=("did decimation destroy surface detail", "compare normal distributions of two meshes", "check shading character survived optimization", "normal field similarity", "extended gaussian image compare", "surface orientation preserved"), semantic="analyze/measure")

    c.register_capability("Fit a camera to frame a mesh (exact, aspect-aware, projected-bbox centred)", "m.fit_camera(mesh, direction, width, height) FRAMES a subject: the camera dict {eye,target,up,fov_deg} that fits every vertex inside the frame, centred, ready for m.render_mesh. Distance solved exactly (dist >= max over verts of |x|/tx+z), no iteration. Centres on the PROJECTED bbox, NOT the centroid -- a scan's verts bunch where the scanner saw detail, so centroid framing clips one edge while the other has slack (measured on a ladybird scan). Bounding-sphere framing ignores aspect and wastes the frame on flat wide subjects. Measured need: preview_asset left a crab at 4% of frame.", example='import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; cam = m.fit_camera(box(), width=640, height=360); sorted(cam.keys())', native=True, module="render", aliases=("fit the camera to the model", "frame the subject in a render", "my model is tiny in the frame", "model is cut off at the edges", "auto framing for a preview", "camera distance to fit the bounding box"), semantic="analyze/measure")

    c.register_capability("Iterative linear solve (shared conjugate gradient, complex-aware)", "m.solve_linear_cg(A, b, x0=None) solves A x = b for Hermitian positive-definite A by conjugate gradient -- the PROMOTED shared solver (holographic_numerics.cg, ledger P1) that replaced two independent CG copies (image's real-only, crossfield's complex-Hermitian). Complex systems use conjugated inner products; real input is BIT-IDENTICAL to the historical solver (measured 0.000e+00); x0 warm-starts, which is most of an inverse-iteration outer loop's speed. Matvec-closure form: import holographic_numerics.cg (closures do not cross JSON). Returns x, deterministic.", example='import numpy as np, lecore; m=lecore.UnifiedMind(); A=np.array([[4.,1.],[1.,3.]]); b=np.array([1.,2.]); x=m.solve_linear_cg(A,b); bool(np.abs(A@x-b).max()<1e-9)', native=True, module="numerics", aliases=("solve a linear system iteratively", "conjugate gradient solver", "solve without inverting the matrix", "hermitian positive definite solve", "iterative solver for a big system", "cg solve"), semantic="analyze/measure")

    c.register_capability("Surface-route retopology (field-aligned quads, silhouette-safe by construction)", 'm.surface_retopo(mesh, density) gives a SCAN or dense mesh field-aligned QUAD topology whose vertices never leave the source surface, so the silhouette survives BY CONSTRUCTION (measured: 323 faces at IoU 0.989, 77% quads). Chain: cross_field -> position_field (IFAM 4-PoSy) -> extract_quads (IFAM 4.4) -> shrinkwrap. Use INSTEAD of auto_retopo for scans: voxelising fails the 0.95 gate at every affordable resolution on thin features (0.785/0.825/0.884/0.935 at res 12/20/32/48) -- an SDF cannot represent what it cannot sample. guide_dirs puts loops where deformation lives. Guarded, linear knob.', example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; q,r = m.surface_retopo(loop_subdivide(triangulate_ngons(box()), levels=3), density=1.5); (r['faces'] > 0, round(r['quad_fraction'], 2))", native=True, module="crossfield", aliases=("retopologize a scan", "make animation friendly topology", "clean quad topology for a model", "retopo without wrecking the silhouette", "edge loops that follow the form", "quad remesh a photogrammetry scan"), semantic="create/emit")

    c.register_capability('Consistent face winding (orientation repair: the precondition every field solver needs)', 'm.mesh_orient(mesh) makes face winding CONSISTENT -- flood-fill 2-colouring over the dual graph, flipping any face that traverses a shared edge the same way as the neighbour that reached it. THE PRECONDITION for field work: cross_field/guided_cross_field/surface_retopo all require consistent winding and photogrammetry scans do not have it. Already-oriented meshes return BIT-IDENTICAL. Non-manifold edges (3+ faces) are SKIPPED and counted -- a different defect (use m.mesh_repair); measured, a ladybird LOD had 490. Non-orientable components are left alone and reported, never guessed.', example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, Mesh; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; b=triangulate_ngons(box()); bad=[tuple(reversed(f)) if i%2 else tuple(f) for i,f in enumerate(b.faces)]; o,r = m.mesh_orient(Mesh(np.asarray(b.vertices,float), bad)); (r['oriented'], r['flipped']>0)", native=True, module='meshtools', aliases=('fix flipped faces', 'make the winding consistent', 'orient a mesh consistently', 'my normals point inward', 'mesh is not consistently oriented', 'repair face orientation'), semantic='convert/emit')

    c.register_capability('Transform a mesh by a matrix (reflection-aware: det<0 flips winding)', "m.transform_mesh(mesh, matrix) applies a 3x3/4x4 matrix AND FLIPS FACE WINDING WHEN THE MATRIX REFLECTS (det<0). m.convert_up_axis(mesh,'z','y') re-orients between up-axis conventions via a PROPER rotation. WHY: a mirror/axis-swap/negative-scale leaves a mesh perfectly self-consistent and entirely INSIDE-OUT -- measured, the naive swap V[:,[0,2,1]] gives a box reporting oriented=True with 0% outward normals, and mesh_orient CANNOT fix it (it repairs neighbours DISAGREEING; global inversion has no disagreement to find). Different defects. Singular matrices raise rather than collapse.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; b=triangulate_ngons(box()); r=m.transform_mesh(b, np.diag([1.,1.,-1.])); u=m.convert_up_axis(b,'z','y'); (len(r.faces)==len(b.faces), len(u.faces)==len(b.faces))", native=True, module='meshtools', aliases=('apply a matrix to a mesh', 'mirror a mesh without turning it inside out', 'change the up axis of a model', 'convert z-up to y-up', 'my normals inverted after a transform', 'transform mesh vertices by a matrix'), semantic='convert/emit')

    c.register_capability('Topology preservation gate (islands / holes punched / holes filled)', "m.mesh_topology_delta(src, out) checks the invariants THE SILHOUETTE GATE CANNOT SEE: islands_created (a reducing op must never detach geometry), holes_created (never punch holes in a closed mesh), holes_filled (never close holes that EXISTED -- a scan's holes are DATA; filling them invents surface never measured), euler_changed, nonmanifold_added, plus a `preserved` verdict. WHY SEPARATE: an outline is blind to anything inside it -- measured, surface_retopo scored 0.973 IoU (a PASS) while punching 6 boundary edges into a CLOSED box. Integers, no tolerance. Pairs with silhouette + EGI.", example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, Mesh; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; import numpy as np; b=triangulate_ngons(box()); holed=Mesh(np.asarray(b.vertices,float), [tuple(f) for f in b.faces][:-2]); d=m.mesh_topology_delta(b, holed); (d['holes_created'], d['preserved'])", native=True, module='meshtools', aliases=('did the decimation create disconnected pieces', 'check for face islands after a mesh operation', 'did we punch holes in the mesh', 'are holes being filled that should not be', 'topology invariants before and after', 'verify no detached geometry'), semantic='analyze/measure')

    c.register_capability('Bisect a monotone knob to a target budget (shared decimate/rate-distortion engine)', 'Bisect a MONOTONE probe(knob) to hit a target budget -- grow/shrink a knob until probe(knob) crosses a target, tracking the closest hit. The shared engine behind decimate_to (bisect a grid to a face count) and ratedistortion (bisect a scale to a target cosine): one move, parameterised. midpoint arith=(lo+hi)//2 for integer grids, geom=sqrt(lo*hi) for continuous scale; tol best-tracks within a tolerance or None sweeps fixed iters; key reads a budget number off a probed object; the caller owns its own iteration count via on_probe (so promoting it never moved a recorded iters value).', example="import lecore; m=lecore.UnifiedMind(); r=m.bisect_to_budget(lambda k:k, 20, 0, 4, midpoint='arith', max_iters=12, tol=0.10, bracket=True); r", native=True, module='numerics', aliases=('bisect to a budget', 'binary search a monotone parameter', 'find the knob value for a target', 'grow a parameter until it hits a target', 'solve for the setting that meets a budget', 'bracket and bisect to a face count or cosine'), semantic='analyze/measure')

    c.register_capability('Smallest eigenpair of a sparse operator (matvec-only, no scipy)', "Smallest eigenpair of a Hermitian PSD operator from ONLY its matvec -- no matrix materialised, no scipy. The two-phase solver behind cross_field's sparse path, promoted (M7): phase 1 a safe fixed shift that favours the bottom of the spectrum from any start, phase 2 a Rayleigh shift for a superlinear gap-independent endgame; CG inner solves on the shifted matvec; exits on the eigen-residual (successive-iterate agreement false-converges, measured). Caller supplies the Gershgorin bound c and may keep its own matvec count via on_matvec. Returns (u, lambda_min, matvecs).", example='import numpy as np, lecore; m=lecore.UnifiedMind(); rng=np.random.default_rng(3); Q=rng.standard_normal((30,30)); A=Q@Q.T; c=float(np.abs(A).sum(1).max()); u,lam,mv=m.smallest_eigenpair(lambda x: A@x, 30, c, dtype=float); (round(lam,6), round(float(np.linalg.eigh(A)[0][0]),6))', native=True, module='numerics', aliases=('smallest eigenvector of an operator', 'dominant eigenpair without scipy', 'matvec only eigensolver', 'spectral solve without building the matrix', 'smallest eigenvalue of a laplacian', 'inverse iteration eigensolver'), semantic='analyze/measure')

    c.register_capability('Closest point on a mesh (shared correspondence machine for transfer + bakes)', 'Closest point on a mesh to each query point -- the shared correspondence machine behind uv/attribute transfer AND the high-to-low bakes (M14: one projection, many channels). Builds a uniform spatial hash over triangles ONCE and ring-searches it per point; returns (face_index, barycentric, distance) so the caller reads whatever it needs (position, normal, uv, weight) off the single projection instead of re-casting. m.mesh_closest_point(mesh, points). The dedup of four inline copies of the same grid+ring-search; bit-identical to each (same cell rule, ring order, first-seen tie-break).', example='import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; b=triangulate_ngons(box()); r=m.mesh_closest_point(b, [[0.4,0.4,0.4]]); (r[0][0], round(r[0][2],3))', native=True, module='meshtools', aliases=('closest point on a mesh', 'surface correspondence between two meshes', 'project points onto a surface', 'closest face and barycentric coords', 'one projection for uv and normal transfer', 'spatial hash closest point query'), semantic='analyze/measure')

    c.register_capability('Graded power-of-two size levels (2:1-balanced, for adaptive retopo)', "Per-vertex power-of-two size LEVELS from a target edge length, 2:1-BALANCED so the level jump across any mesh edge is at most 1 -- the graded size field behind adaptive retopo (M1): refine where the surface bends, coarsen where it is flat, WITHOUT breaking the quad extractor's lattice. rho(v) = rho0*2^k(v); 2^k lattices have nested cell walls so cells at different levels still align (the only artefact is a hanging node, and |dk|<=1 caps it to one per coarse edge). Feed target_edge = clamp(rho0/(1+curvature)). m.graded_levels(mesh, target_edge, rho0). Returns (levels, rho).", example='import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; s=loop_subdivide(triangulate_ngons(box()),levels=2); V=np.asarray(s.vertices); te=np.where(V[:,0]>0,0.1,0.8); k,rho=m.graded_levels(s,te,0.4); (int(k.min()),int(k.max()))', native=True, module='crossfield', aliases=('graded sizing for retopo', 'balanced refinement levels from curvature', 'power of two size field', '2 to 1 balance a level field', 'adaptive lattice sizing without breaking the extractor', 'refine where the mesh bends'), semantic='analyze/measure')

    c.register_capability('Single-branch skeleton curve (medial ridge collapsed to a polyline)', "Collapse a mesh's medial-axis ridge into a single-branch CENTERLINE CURVE (ordered polyline) -- the 1-D skeleton of a LIMB-LIKE shape, for rigging bones and centerline measurement. m.skeleton_curve(mesh) returns {curve (ordered points), depth=medial radius along it, n_ridge}. Orders ridge points along their principal axis and averages cross-sections (a cylinder collapses to a straight line on its axis, radial 0.00). KEPT NEGATIVE: SINGLE-BRANCH -- one PCA axis cuts corners on a bent/branched shape (residual 0.48 on an L-tube); those need branch segmentation first, then this per branch.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_skeleton import _cylinder; cv=m.skeleton_curve(_cylinder(), res=20); (len(cv['curve'])>=3, round(float(np.sqrt(cv['curve'][:,0]**2+cv['curve'][:,1]**2).mean()),2))", native=True, module='skeleton', aliases=('collapse skeleton to a curve', 'centerline polyline of a limb', 'skeleton as a polyline', '1d curve from medial voxels', 'bone centerline for rigging', 'reduce a limb to a line', 'trace the middle of a shape', 'spine polyline of a limb', 'ridge to polyline'), semantic='analyze/measure')


_PART = "holographic_catalog_p01"


def _selftest():
    """A part must REGISTER something and must not duplicate a name inside itself.

    WHY A REAL SELFTEST RATHER THAN A BUDGET LINE. The split created six modules with no `__main__`, and the
    selftest-budget test caught it immediately -- correctly, because a module that asserts nothing is a false
    green. Budgeting them would have been the lazy fix: it silences the alarm without testing anything. These
    parts have a real, cheap contract -- register onto a fresh Catalog, register a NON-EMPTY set, and do not
    collide with themselves -- so it costs nothing to assert it, and it catches the failure that actually
    threatens a mechanical split: a chunk boundary that swallowed or repeated a registration."""
    from holographic.caching_and_storage.holographic_catalog import Catalog
    c = Catalog()
    register_p01(c)
    caps = c.all()
    assert caps, "%s registered NOTHING -- a part that registers nothing is a silently missing chunk" % _PART
    names = [x.name for x in caps]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, "%s registers the same name twice: %s" % (_PART, dupes)
    print("%s selftest OK -- %d capabilities, no internal duplicates" % (_PART, len(caps)))


if __name__ == "__main__":
    _selftest()
