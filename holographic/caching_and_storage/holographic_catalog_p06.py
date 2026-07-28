"""holographic_catalog_p06 -- part 6/6 of the capability registry (split from holographic_catalog).

MECHANICAL SPLIT, no edits. holographic_catalog.py hit 81% of the 1 MB agent-read cap, so the file
that makes capabilities discoverable was becoming the one file an agent could not open. The parts are
called IN ORDER by default_catalog() and the emitted catalog is byte-identical -- verified by hashing
every capability field before and after. Order matters: find_capability ranks by score and ties break
by registration order, so a reordering would silently move search results.

Add new capabilities to the LAST part, or to whichever part is topically right -- never to a new file
without registering it in default_catalog(), or it will simply not exist.
"""


def register_p06(c):
    """Register this part's capabilities on `c`. Called by default_catalog() in order."""
    c.register_capability("holographic_catalog", "THIS catalog: search the engine's own capabilities before building "
                          "a duplicate (register_capability / find_capability)", example="find_capability('search vectors')",
                          native=False, aliases=("catalog", "capability", "registry", "find", "discover", "duplicate"))

    # --- the pipeline (consolidation R1): the one entry point that composes a render/sim run ---
    c.register_capability(
        "Pipeline (render/sim)", "compose a render or sim run as ordered stages that declare what they need/produce; "
        "dispatch among render strategies (pathtrace/raymarch/prt/radiance) and catch a missing input before running",
        example="from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig, RenderSpec", native=False,
        aliases=("pipeline", "stage", "compose", "run", "render", "strategy", "dispatch", "route"))

    # --- top-level DOMAIN pipelines: one findable pointer per subsystem, so no whole domain is buried ---
    c.register_capability("Make a 3-D primitive by name (placed, one door)", "every SDF primitive shipped "
                          "reachable only by import: asked for a sphere this mind returned a Lipschitz "
                          "worst-view bound, asked for a cube the sky-observation capability. Ten phrasings, "
                          "ten unrelated fallbacks. kind is a word you'd type -- cube/ball/floor/donut/cone/"
                          "capsule/ellipsoid/torus/cylinder/octahedron plus the fractals -- and position/"
                          "rotate/scale are applied in the ONE order that cannot go wrong (scale, rotate, "
                          "THEN translate: rotating after translating orbits the world origin instead of "
                          "spinning in place). Feed the result to scene.add(geometry=...) or render_sdf",
                          example="import lecore; m=lecore.UnifiedMind(); "
                                  "print(m.shape('cube', bx=0.4, by=0.4, bz=0.4, position=(1,0.5,0)).to_dsl())",
                          native=True, module="sdf",
                          aliases=("make a sphere", "add a cube", "create a box shape", "give me a ground plane",
                                   "build a cylinder", "a torus shape", "basic 3d shapes to start with",
                                   "primitive shapes", "put a ball in the scene", "make a floor",
                                   "what shapes can I make", "add geometry to my scene"))
    c.register_capability("The SDF DSL, described well enough to write one", "sdf_parse has always taken a "
                          "compact s-expression for a whole shape tree -- (kind params... children...) -- and "
                          "the node names and parameter counts lived in a module-level dict nothing surfaced. "
                          "A grammar you can only use if you already know it is not a usable grammar. Returns "
                          "every node kind with what its numbers MEAN, sorted primitives -> modifiers -> "
                          "combinators (the order you build in), plus an example that parses",
                          example="import lecore; m=lecore.UnifiedMind(); print(m.sdf_grammar()['example'])",
                          native=True, module="sdf",
                          aliases=("how do I write an sdf string", "what nodes does the sdf dsl have",
                                   "sdf syntax", "shape language reference", "what can I put in sdf_parse",
                                   "csg operators available", "union two shapes together",
                                   "subtract one shape from another", "smooth blend two blobs"))
    c.register_capability("Read a render back (PNG -> array, the see-then-fix loop)", "the engine could WRITE a "
                          "PNG and could not READ one -- a grep for IHDR found only the encoder. That single "
                          "missing direction blocked every render->look->adjust->render cycle, because 'look' "
                          "had nowhere to start, and it is why compare_image_files reached for Pillow (an "
                          "unguarded third-party import in a stdlib-only core). Pure zlib+struct. rgb01 gives "
                          "(H,W,3) float ready to feed straight back in. KEPT NEG: round trip is to ~1/255, not "
                          "exact -- save_png is 8-bit, so assert a tolerance. Interlaced PNGs RAISE rather than "
                          "decode wrongly",
                          example="import lecore; m=lecore.UnifiedMind(); "
                                  "m.save_render('/tmp/x.png', __import__('numpy').zeros((8,8,3))); "
                                  "print(m.load_image('/tmp/x.png').shape)",
                          native=True, module="render",
                          aliases=("read a png file into an array", "load an image from disk",
                                   "open a render I saved earlier", "decode a png",
                                   "get pixels out of an image file", "look at my own render",
                                   "did my render change", "check the image I just saved",
                                   "read an image back in", "png to numpy array"))
    c.register_capability("Lighting (domain)", "one home for lighting: the light types (point/directional/spot/area/"
                          "dome/IES) and the shade INTEGRAL in each mode -- direct NEE, PRT relight, environment SH; "
                          "render methods call it", example="from holographic.rendering.holographic_lightinghome import Lighting, RectLight",
                          native=True, aliases=("lighting", "light", "lamp", "shadow", "dome", "area", "ies", "spot",
                                                "nee", "direct", "prt", "irradiance"))
    c.register_capability("Build a path-tracer light by name (aimed, one door)", "ten light classes shipped and "
                          "NINE were reachable by nothing -- and mind.light() returns the RASTERISER's Light, "
                          "which raises inside the path tracer. This is the one door for render_scene_document: "
                          "kind is a word you'd type ('softbox', 'sun', 'hdri', 'spot'), and `target` AIMS the "
                          "panel/disk/spot for you instead of making you hand-build u_vec/v_vec half-edges -- "
                          "measured as where 3-D authoring stalls. Reach for 'dome' first: an environment light "
                          "is shadowed, so contact AO is free. KEPT NEG: dome + a bright sky double-counts the "
                          "environment for diffuse -- use one or the other",
                          example="import lecore; m=lecore.UnifiedMind(); "
                                  "print(type(m.scene_light('softbox', position=(2,3,2), target=(0,0,0), intensity=60.0)).__name__)",
                          native=True, module="lights",
                          aliases=("add a softbox light to my scene", "area light with soft shadows",
                                   "environment lighting from a sky dome", "hdri lighting", "make a spotlight",
                                   "key light and fill light", "sun lamp", "point light in my render",
                                   "how do I light a scene", "aim a light at something", "studio light",
                                   "light for the path tracer", "why does my light crash the renderer",
                                   # the catalog SELFTEST's own probe -- the symptom phrasing a user
                                   # brings ("speckle"/"fireflies"), re-ranked out when two merges
                                   # added ~57 capabilities.
                                   "my placed light has speckle noise", "noisy speckled light",
                                   "fireflies in my render"))
    c.register_capability("Shadow / visibility (domain)", "test whether light or the environment reaches a point: "
                          "SDF soft shadow (Quilez penumbra), ambient occlusion, hard shadow-ray (NEE), and PRT baked "
                          "visibility -- one home of strategies render paths call",
                          example="from holographic.rendering.holographic_shadowhome import Shadow; Shadow.soft(sdf, P, Ldir)",
                          native=True, aliases=("shadow", "visibility", "occlusion", "ambient occlusion", "penumbra",
                                                "shadow ray", "soft shadow", "unoccluded"))
    c.register_capability("Geometry (domain)", "build and edit shapes three ways: explicit MESH (half-edge + verbs), "
                          "implicit SDF (CSG + raymarch), and SPLATS (Gaussian clouds) -- convertible via meshbridge",
                          example="from holographic.mesh_and_geometry.holographic_mesh import Mesh; from holographic.mesh_and_geometry.holographic_sdf import box, sphere",
                          native=True, aliases=("geometry", "mesh", "sdf", "splat", "shape", "model", "csg", "subdivide"))
    c.register_capability("Texture (domain)", "procedural + example-based surface detail as FIELDS you plug into a "
                          "Material channel: fbm noise, Voronoi/cellular cracks, divergence-free curl, patch synthesis; "
                          "plus the weathering set (burn/oxidation/inclusions)",
                          example="from holographic.materials_and_texture.holographic_texturehome import Texture; Param(field=Texture.voronoi(kind='edge'))",
                          native=True, aliases=("texture", "noise", "fbm", "voronoi", "curl", "procedural", "weathering",
                                                "pattern", "detail", "cellular"))
    c.register_capability("Texture graph (composable maps)", "build a texture as a TREE of maps: an op "
                          "(mix/multiply/over/scale/remap/...) over TYPED inputs -- map | color | field | number -- each of "
                          "which may be another map, so graphs nest to any depth. Sampling walks the tree; the input types "
                          "are checked at COMPOSE time so a bad graph (a colour used as a weight, a missing input) is refused "
                          "up front, not rendered wrong. Encode a graph to a hypervector to cache/search it. CMP1",
                          example="mind.texture_op('mix', a=mind.texture_leaf(value=[1,0,0]), b=mind.texture_leaf(value=[0,0,1]), t=mind.texture_leaf('fbm', n_dims=2)); mind.sample_texture(g, [0.3,0.7])",
                          native=True, aliases=("texture graph", "map graph", "shader graph", "compose texture",
                                                "layered texture", "node graph", "blend maps", "mix textures", "procedural graph",
                                                "compose a texture from noise and colors", "combine noise and colours",
                                                "mix noise with colors", "build a texture from nodes"))
    c.register_capability("Simulation (domain)", "a shared STEP LOOP over any solver (fluids/smoke, fire/combustion, "
                          "softbody/cloth, hair, MPM, collision, reaction-diffusion) -- each keeps its own math; the "
                          "scaffold gives them one step(dt) and exposes their field for the Pipeline to render. "
                          "mind.simulation(solver, step_fn, field_fn) wraps ANY solver in process; "
                          "mind.run_simulation(kind, steps) is the stateless twin for /invoke -- build a known "
                          "solver ('fluid' or 'automaton'), run it, and return its field grid as plain JSON (the "
                          "live wrapper holds a solver+adapter that does not survive serialization).",
                          example="grid = mind.run_simulation('fluid', 30)   # step a fresh fluid and return its density",
                          native=True, aliases=("simulation", "solver", "fluid", "smoke", "fire", "cloth", "softbody",
                                                "step", "advance", "sim loop", "mpm", "reaction diffusion",
                                                "particle system", "particles", "emitter", "mass spring", "spring",
                                                "rigid body", "collision"), module="fluid", consumes=('field',), produces=('field',))
    c.register_capability("Encoders (number to vector)", "turn raw values into hypervectors: scalar & fractional-power "
                          "encoding (encoders/fpe -- nearby numbers map to nearby vectors), N-D coordinate fields "
                          "(fpefield), complex-phasor FHRR (fhrr), sparse block codes (sbc), geometric-algebra Clifford "
                          "(clifford), and exact integer arithmetic over phasors (rns). How data ENTERS the substrate",
                          example="from holographic.io_and_interop.holographic_encoders import ScalarEncoder; from holographic.sampling_and_signal.holographic_fpe import ...",
                          native=True, aliases=("encode", "encoder", "number to vector", "scalar encoding",
                                                "fractional power encoding", "fpe", "encode coordinates", "phasor", "fhrr",
                                                "sparse block codes", "sbc", "clifford", "geometric algebra",
                                                "exact integer arithmetic", "rns", "embed a value"))
    c.register_capability("Physics & chemistry (domain)", "physical/chemical PROPERTIES and their evolution: the matter "
                          "model (Mixture/matter_step: smoke->oil separation), diffusion, equilibrium propagation, "
                          "thin-film iridescence, oxidation/weathering", example="from holographic.misc.holographic_mixture import Mixture, matter_step",
                          native=True, aliases=("physics", "chemistry", "matter", "mixture", "diffusion", "material properties",
                                                "iridescence", "oxidation", "phase"))
    c.register_capability("Adaptive rendering", "the render call that picks its own methods/quality: the converging "
                          "sampler that stops per-pixel when the confidence interval is tight, and the render-method "
                          "auto-picker", example="from holographic.rendering.holographic_gbuffer import render_auto, converge_samples",
                          native=True, aliases=("adaptive", "auto", "quality", "converge", "raytracing mode", "render mode"))
    c.register_capability("Render graph (bake vs live)", "the PIPELINE composing the texture/material/scene graphs: "
                          "mind.render_graph() registers texture graphs (static or dynamic) + a CMP4 instanced scene, "
                          "then plan() shows what it will do and WHY and prepare() runs it. The adaptive decision it "
                          "adds is BAKE a static texture graph to a grid (O(1) bilinear lookup, mind.bake_texture) vs "
                          "SAMPLE it live -- baking amortises a deep graph over many hits, live avoids re-baking a "
                          "changing map every frame. Trade: memory + interpolation error. CMP5",
                          example="rg = mind.render_graph(); rg.add_texture('rust', graph, static=True).set_scene(scene); rg.plan(); prep = rg.prepare()",
                          native=True, aliases=("render graph", "bake texture", "bake vs live", "prepare scene",
                                                "resolve textures", "orchestrate render", "material lod", "precompute texture",
                                                "static texture", "render pipeline graphs"))
    c.register_capability("Preview (swatch & material ball)", "SEE what you composed: mind.preview_texture(graph) "
                          "renders a CMP1 texture graph as a flat RGB swatch, and mind.preview_material(material) "
                          "renders a material on the classic MATERIAL BALL sphere (Cook-Torrance shaded, using the "
                          "material's roughness/metallic channels) -- works on a plain Material or a CMP2/CMP3 "
                          "layered/multi material. Returns a float image in [0,1] to save/view. The missing step "
                          "between composing a texture/material and looking at it.",
                          example="img = mind.preview_texture(graph); ball = mind.preview_material(layered_material)",
                          native=True, aliases=("preview", "swatch", "material ball", "material preview", "texture preview",
                                                "see the texture", "render swatch", "thumbnail", "material sphere",
                                                "visualize texture", "visualize material", "look at the material"))
    c.register_capability("Make water (one-call Gerstner ocean preset)", "ONE CALL -> a WATER surface: mind.make_water(res, extent, t, seed, preset) sums deterministic Gerstner/trochoidal waves (Fournier & Reeves 1986; Tessendorf 2001 dispersion + steepness bound) into {height, positions, normals, bank}; shaded=True adds a sun-shaded preview. Presets 'ocean'/'calm'/'storm'; overrides: wind_heading, n_waves, choppiness, wavelength_range. Animate with t (same seed = coherent frames; dispersion kills looping). EXACT analytic normals. Height feeds spectral_ocean to EVOLVE; positions feed the meshers. KINEMATIC (no breaking) -- overturn via free_surface.",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); w=m.make_water(res=64, preset='ocean', shaded=True); (w['height'].shape, w['image'].shape)",
                          native=True, aliases=("make water for my scene", "generate ocean water surface", "water preset",
                                                "gerstner waves", "animated water surface", "ocean heightfield generator",
                                                "choppy waves", "water waves heightfield", "sea surface", "waves for a lake",
                                                "one call water", "procedural ocean"))
    c.register_capability("Quick material ball (plain numbers, no channels)", "The material-editor SHORTCUT: mind.quick_material(color, roughness, metallic, res) -> the classic MATERIAL BALL image from plain numbers -- no encoder, no channel fields. Shades with the SAME Cook-Torrance BRDF the real renderer uses, so the ball predicts a render. quick_material((1,0.2,0.1), roughness=0.15, metallic=1.0) = polished red metal. Deliberately carries NO textures -- for textured/layered materials build a real Material and use preview_material; this is the one-slider entry.",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); ball=m.quick_material(color=(1,0.3,0.1), roughness=0.2, metallic=1.0, res=64); ball.shape",
                          native=True, aliases=("quick material preview", "material ball from numbers", "preview roughness and metallic",
                                                "simple material ball", "show me a shiny red metal", "material editor preview",
                                                "try a material without textures", "one call material ball", "pbr sliders preview"))
    c.register_capability("Water body (container-first water tool)", "EVERYTHING between 'I want water' and pixels: mind.water_body(container, level, preset, ...) -> a WaterBody. container=None -> OPEN water over `extent` m; 'glass'/'pool'/'bowl' -> a vessel filled to `level` with real Gerstner RIPPLES on top (vessel-scaled, animated by t); any SDF -> the cavity. Liquid from the material library (colour from matlib, IOR from the library -- oil refracts at 1.47). Waves tunable at every scale (choppiness, wind_heading, wavelength_range). .render('fast'|'final') has PRE-BALANCED lighting (raster ~2s / refractive trace); .at_time(t) animates coherently.",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); wb=m.water_body(extent=50.0, seed=1, res=96); img=wb.render('fast', width=160, height=120); img.shape",
                          native=True, aliases=("fill a container with water", "water in a glass", "put water in an object",
                                                "easy water tool", "water scene helper", "pool of water", "bowl of water",
                                                "assemble a water effect", "water with ripples in a cup", "simple ocean scene",
                                                "one call water scene with lighting", "ready to render water",
                                                "render water in one call", "water render over http", "water image for an agent"))
    c.register_capability("Cloud scene (presets x quality tiers)", "GOOD CLOUDS IN ONE WORD EACH: mind.cloud_scene(preset, quality) wraps make_cloud's tuning into named choices. Presets: 'cumulus', 'wispy', 'storm', 'sunset'. Quality tiers MEASURED: 'fast' ~6s (192px), 'balanced' ~20s (288px), 'final' ~2min (384px) -- the full lighting (self-shadow, HG silver lining, multi-scatter) is in EVERY tier; tiers trade resolution/steps only. texture='musgrave'/'voronoi'/'fbm' (opt-in) shapes the density from the procedural texture MENU instead of the built-in cumulus -- streaky/cellular/billow clouds, no grid bake. Any make_cloud keyword overrides.",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); img=m.cloud_scene(preset='wispy', quality='fast', seed=1); img.shape",
                          native=True, aliases=("easy clouds", "cloud preset", "make good clouds fast", "cloud scene helper",
                                                "storm clouds", "sunset clouds", "wispy clouds", "fluffy cumulus",
                                                "clouds quality settings", "quick cloud render", "one word cloud tool",
                                                "clouds from a texture", "musgrave clouds", "texture driven cloud density"))
    c.register_capability("Procedural texture menu (2D + 3D standard set)", "The texture menu every 3D app ships, by NAME: mind.proc_texture(name, **params) -> a field f(P (M,3)); mind.texture_image(name, size) -> a 2D image; mind.texture_volume(name, res) -> a 3D grid (cloud densities). Menu: noise, fbm, white, voronoi (f1/f2/f2f1/cell/smooth), musgrave (ridged/hybrid), wave (bands/rings), marble, wood, brick, magic, checker, stripes, gradient, dots. ONE field serves all three samplers -- 2D texturing is the 3D solid on a plane (slide z through the marble). Deterministic in seed; the direct-eval costume of texturehome's VSA fields.",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); img=m.texture_image('voronoi', size=64, kind='f2f1', scale=5, seed=1); vol=m.texture_volume('fbm', res=16, seed=0); (img.shape, vol.shape)",
                          native=True, aliases=("procedural texture", "voronoi texture", "musgrave texture", "marble texture",
                                                "wood grain texture", "brick texture", "3d noise texture", "cellular noise",
                                                "solid texture", "texture like blender", "standard texture set",
                                                "noise texture for clouds", "texture menu"))
    c.register_capability("Mask refraction (2D lens/droplet distortion)", "Refract an image through a 2D SHAPE: mind.mask_refraction(image, mask, strength, ior, ...) reads the mask as a LENS -- jump-flood distance-to-edge -> a meniscus height -> small-angle Snell displaces pixels by -(ior-1)*strength*grad(height): distortion is STRONGEST NEAR THE MASK EDGE, zero on the plateau and outside (a droplet or glass blob over the image). profile 'lens'/'dome'; chromatic adds dispersion fringes; ripple=(amp,scale) adds fbm shimmer. Screen-space single-interface (no TIR/caustics -- true refraction is path_trace's dielectric).",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); yy,xx=np.mgrid[0:64,0:64]; bg=np.stack([np.mod(xx//8+yy//8,2).astype(float)]*3,-1); mask=(xx-32)**2+(yy-32)**2<20**2; r=m.mask_refraction(bg, mask, strength=8.0); r.shape",
                          native=True, aliases=("refraction effect", "refract through a mask", "water droplet distortion",
                                                "glass blob effect", "2d refraction", "lens distortion from a shape",
                                                "distort image near mask edge", "screen space refraction",
                                                "water shimmer on an image", "droplet lens effect"))
    c.register_capability("Sculpt-mode preparation (guarded mesh -> SDF cache)", "The SAFE switch into sculpting: mind.sculpt_prepare(mesh, resolution, silhouette=0.95) builds the SDF cache (grid+axes) AND the sculptable remesh in one call, held to a worst-view silhouette-IoU floor so conversion cannot silently change shape. Two levers in cost order: retry the SIGN (flood fill leaks through touching shells, WORSENING with resolution: 0.734@48 -> 0.250@96; winding robust 0.954+), then escalate resolution x1.5 for thin features; unreachable floor -> loud ValueError with the ladder. Sharp low-poly corners round intrinsically -- lower the floor or silhouette=None knowingly.",
                          example="import numpy as np; import lecore; from holographic.mesh_and_geometry.holographic_meshbridge import sculpt_prepare; from holographic.mesh_and_geometry.holographic_sdf import sphere; from holographic.mesh_and_geometry.holographic_meshbridge import marching_tetrahedra_vec, mesh_to_sdf_grid",
                          native=True, aliases=("prepare a mesh for sculpting", "sculpt mode conversion", "sdf cache from a mesh",
                                                "convert mesh to sculptable", "switch to sculpt mode safely", "guarded voxel conversion",
                                                "mesh changes shape when sculpting", "keep the shape when converting",
                                                "silhouette guard for conversion", "sculpt cache"))
    c.register_capability("Texture sampler + ramps (textures as numbers, numbers as textures)", "The two directions of one identity. READ: mind.sample_image(image, uv) samples a raster bilinearly/nearest with clamp/repeat (GPU half-texel convention) -- drive any parameter from a painted map; mind.image_field(image) wraps it as f(P (M,3)) so a painted map plugs in anywhere a field goes (Material channels, cloud densities). WRITE: mind.values_to_texture(v) makes numbers sampleable (roundtrip EXACT at texel centres); mind.ramp(positions, values, interp='linear'/'constant'/'smooth') is the ColorRamp -- stops exact in every mode, ends clamp; mind.ramp_texture bakes the strip.",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); tex=m.values_to_texture(np.array([0.2,0.8,0.5])); v=m.sample_image(tex,[[0.5/3,0.5]]); r=m.ramp([0,1],[0.0,1.0]); (float(v[0]), float(r([0.25])[0]))",
                          native=True, aliases=("texture sampler", "sample an image at uv coordinates", "use a texture as a number",
                                                "color ramp with stops", "gradient ramp", "assign values to a texture",
                                                "bake values into a texture", "bilinear image sample", "ramp texture",
                                                "map a value through a gradient", "lookup table texture", "drive a parameter from a map"))
    c.register_capability("Mixture matter model (oil & water, dye, smoke -- one advected-field core)", "Smoke, dye mixing, salt fingering, and oil-and-water SEPARATION are ONE advected-field matter model, not four simulators: mind.make_mixture(shape, buoyancy, tension) builds component channels riding one shared incompressible flow; mind.matter_step(mix, vx, vy, dt, drift_strength) advances it, DELEGATING to the fluid faculties -- no second solver. Channels diffuse at their own rates (salt fingering); drift + double-well hooks (off by default) give demixing/immiscible behaviour. KEPT NEGATIVE: sharp immiscible interfaces are the diffuse-interface trade; fractions clamp to a partition.",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); mix=m.make_mixture((16,16)); type(mix).__name__",
                          native=True, module="mixture",
                          aliases=("oil and water separating mixture model", "mixture model", "phase separation",
                                   "demixing simulation", "immiscible fluids", "dye mixing in water",
                                   "salt fingering", "multi component fluid", "matter model", "oil water demix"))
    c.register_capability("Style transfer (grade toward a reference image)", "Make one image FEEL like another: mind.color_transfer(img, reference, mode, strength) matches the reference's colour statistics -- 'meanstd' (Reinhard 2001) or 'covariance' (Monge-Kantorovich whiten-then-colour: handles correlated teal-orange grades). Sizes need not match; strength blends 0..1. COMPOSES: the 'style_transfer' step in postfx_chain grades a frame inside any chain -- ('style_transfer', {'reference': ref}) then bloom/grain/aces. Family: ST2 texture_synthesis, ST3 guided super-res. GLOBAL statistics: moves colour, not content; extreme palette gaps can wash out.",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); img=np.random.default_rng(0).uniform(0,1,(32,32,3)); ref=np.random.default_rng(1).uniform(0,1,(24,24,3)); out=m.color_transfer(img, ref, strength=0.8); out.shape",
                          native=True, aliases=("style transfer", "apply the style of one image to another",
                                                "make my render look like a painting", "match the colors of a reference image",
                                                "stylize an image", "transfer the look of a photo", "neural style transfer",
                                                "post process with a style", "color grade toward a reference",
                                                "match a movie look", "consistent grade across frames"))
    c.register_capability("Textured object render (paint composed maps)", "paint a COMPOSED texture or material "
                          "(CMP1 graph / CMP2-3 material) onto an object and render it: "
                          "mind.render_textured(scene, {object_name: texture_graph}) marches the scene, UV-wraps each "
                          "texture onto its object (spherical map on a sphere, planar on a box), and shades with the "
                          "real Cook-Torrance BRDF + a light + a hard shadow. This is the composability stack driving "
                          "a full 3-D render, not just a swatch. Honest: textbook UV (seams), single hard light.",
                          example="tex = mind.texture_op('mix', a=mind.texture_leaf(value='orange'), b=mind.texture_leaf(value='purple'), t=mind.texture_leaf('fbm', n_dims=2)); mind.render_textured(scene, {scene.names()[0]: tex})",
                          native=True, aliases=("textured render", "paint texture on object", "wrap texture", "uv render",
                                                "texture the sphere", "composed texture render", "map onto object"))
    c.register_capability("Denoise (domain)", "clean a render or signal with one home: image SVGF (variance-guided "
                          "a-trous) or demodulated (divide albedo out), sharpen, and the signal manifold denoisers "
                          "(adaptive/manifold/codebook/trajectory)",
                          example="from holographic.rendering.holographic_denoisehome import Denoise; Denoise.image(img, N, A, D, method='svgf')",
                          native=True, aliases=("denoise", "svgf", "clean", "smooth", "nlm", "demodulate", "sharpen",
                                                "noise reduction", "restore"))
    c.register_capability("Compute (VSA-native)", "stay in the vector/frequency domain with no Python hops: FUSE a "
                          "bind/bundle/permute chain into ~2 FFTs (measure the FFT drop), the fuse-runs SCHEDULER, "
                          "width, and running logic as a VSA PROGRAM. Rule: push decisions/cleanups to the boundaries",
                          example="from holographic.misc.holographic_computehome import Compute; Compute.fuse_record(keys, values)",
                          native=True, aliases=("compute", "fuse", "fused", "schedule", "execute", "program", "machine",
                                                "fft", "chain", "collapse", "vsa native"))
    c.register_capability("Memory (cache hierarchy)", "keep the hot working set where the CPU can reach it fast: FFT "
                          "spectrum residency (skip recomputing a reused transform), batched contiguous bind (one FFT "
                          "for a whole record), tiling to fit a cache level, and the opt-in GPU / numba backends",
                          example="from holographic.simulation_and_physics.holographic_memoryhome import Memory; Memory.bind_cached(a, b, cache)",
                          native=True, aliases=("memory", "cache", "residency", "resident", "spectrum cache", "batch",
                                                "bind_batch", "backend", "gpu", "jit", "working set", "hot"))
    c.register_capability("Cache key cost (identity vs content addressing)", "the price of a cache KEY, measured "
                          "rather than assumed. SpectrumCache shipped keying on a sha256 of the whole atom -- and "
                          "hashing D floats costs MORE than transforming them (D=1024: 21.5us hash vs 13.0us rfft), "
                          "so the cache measured 0.40x-0.82x scalar and 0.50x-0.70x inside fusion: SLOWER than no "
                          "cache, while its docstring claimed 1.4x. key='identity' keys on the array object (O(1), "
                          "pinned so the id cannot be recycled): 2.4x-2.6x scalar, 3.7x-4.3x in fuse_record, "
                          "bit-identical. Content keying stays the default and is required when byte-identical "
                          "arrays arrive as distinct objects",
                          example="c = mind.spectrum_cache(key='identity'); mind.fuse_record(keys, values, spectrum_cache=c)",
                          native=True, aliases=("cache key cost", "identity keyed cache", "content addressing cost",
                                                "is my cache slower than no cache", "cheap cache key for a big array",
                                                "avoid rehashing an immutable array", "hashing costs more than the work",
                                                "make the spectrum cache actually fast", "cache without hashing the contents",
                                                "why is my cache slow"))
    c.register_capability("Function-granularity reachability (the engine audits itself)", "the other audits "
                          "reason about MODULES and all report zero gaps -- a module passes if it has a "
                          "docstring, public exports and a reference from UnifiedMind. None looks INSIDE the "
                          "file, so functions can be reachable by nothing while their module passes. This one "
                          "partitions every public engine function into faculty / catalogued / called / "
                          "TEST-ONLY / orphan. TEST-ONLY is the valuable bucket: works, tested, exposed "
                          "nowhere -- so by this repo's own rule it does not exist. Conservative, never deletes",
                          example="mind.audit_orphans()['counts']",
                          native=True, aliases=("find dead code", "which functions are never called",
                                                "unused methods", "audit the codebase", "map the codebase",
                                                "what is built but not wired", "orphan functions",
                                                "code that exists but cannot be reached", "self audit",
                                                "is anything unreachable", "audit my own code"))
    c.register_capability("Agent reachability (referenced somewhere vs callable from /invoke)", "the orphan audit "
                          "asks 'is this name referenced anywhere?' and answers YES for a symbol whose only "
                          "caller is itself import-only by design -- a consolidation home, a declared negative. "
                          "Alive in the import graph, dead to /invoke. This asks whether the route GOES "
                          "anywhere. shadowed = referenced only from cul-de-sacs; dark = a public CLASS with no "
                          "faculty and no catalog entry (the orphan audit collects functions only, so classes "
                          "were invisible to it). MEASURED: 9 of 10 path-tracer light classes are dark while "
                          "every module audit read 0 gaps. ADVISORY, under-reports, never a delete list",
                          example="mind.audit_agent_reach()['counts']",
                          native=True, module="orphanaudit",
                          aliases=("can an agent actually call this", "which classes can I not construct",
                                   "what can I not reach through the mind", "half wired module",
                                   "built but I cannot call it", "why can't I use this class",
                                   "is this class exposed anywhere", "dark classes", "shadowed functions",
                                   "does the import chain go anywhere", "agent reachable surface",
                                   "alive in the graph but dead to a caller"))
    c.register_capability("Search the engine's own source by meaning", "find_capability searches the CATALOG "
                          "-- 674 of 7,572 functions; for the other 6,898 there was nothing. This indexes every "
                          "public engine function by name tokens, first docstring line and CALLEE NAMES (who "
                          "you call is what you do) and answers 'what else looks like this?', which is Rule 0's "
                          "actual question. KEPT NEGATIVE IN THE DEFAULT: the hypervector encoding LOST to "
                          "token-set Jaccard on the same features (recall@1 0.175 vs 0.542) and uses 2.8x more "
                          "memory, winning only query latency 8.3x -- so Jaccard is the default and the vector "
                          "path is opt-in",
                          example="import lecore; m=lecore.UnifiedMind(); [l for l,_s in m.code_search('subdivide a mesh', k=3)]",
                          native=True, aliases=("find similar code", "search the codebase semantically",
                                                "what other function looks like this one", "code similarity",
                                                "semantic search over my own source", "find near duplicate functions",
                                                "what else does what this does", "search my source",
                                                "which function does this already", "analogy over code"))
    c.register_capability("Code health: complexity x exposure x exercise (risk, not size)", "raw cyclomatic "
                          "complexity ranks the WRONG thing, and measuring it proved it: the top-scoring "
                          "functions here (parse_description 65, mesh_parts 57, rebake_texture 54) are all "
                          "exercised -- they score high BECAUSE they are load-bearing, and load-bearing code "
                          "got tests. Risk is the cross product: 1858 functions no test mentions, 22 at "
                          "CC>=20, and the worst cell is an ADVERTISED catalog capability at CC 46 that "
                          "nothing tests. Stdlib ast; 0.92 top-100 rank agreement with radon. Mention scan, "
                          "not coverage",
                          example="import lecore; m=lecore.UnifiedMind(); m.audit_complexity(limit=3)['totals']",
                          native=True, aliases=("cyclomatic complexity", "code complexity", "code health",
                                                "how complex is this function", "which code is risky",
                                                "complex and untested", "where should i add tests",
                                                "code metrics", "maintainability", "technical debt map"))
    c.register_capability("Antiperiodic (Mobius) fraction -- is a circle the wrong carrier?", "a circular "
                          "encoding CANNOT hold a sign-flipping pattern: it wraps theta and theta+pi onto the "
                          "same point, destroying the antiperiodic half on encode. Split two periods by halves "
                          "-- (a+b)/2 periodic, (a-b)/2 antiperiodic -- an exact orthogonal split with no FFT "
                          "bin-parity bookkeeping, parts summing back bit-for-bit. Reads ~1.0 for f(t+T)=-f(t), "
                          "~0.0 for f(t+T)=+f(t), 0.5 for a 50/50 sum. The diagnostic that turns 'circle or "
                          "Mobius strip?' from a guess into a measurement",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); t=np.arange(256); (round(m.antiperiodic_fraction(np.cos(np.pi*t/128)),3), round(m.antiperiodic_fraction(np.cos(2*np.pi*t/128)),3))",
                          native=True, aliases=("antiperiodic fraction", "mobius strip or circle",
                                                "sign flipping component", "antiperiodic split",
                                                "does this repeat or invert", "half period sign flip",
                                                "is a circular encoding wrong here", "axial vs circular"))
    c.register_capability("IES photometric file (a real luminaire's measured falloff)", "parse an IESNA LM-63 "
                          "file -- the format lighting manufacturers actually publish -- into a "
                          "(candela_profile, max_vertical_angle) pair usable as a light's angular falloff. "
                          "Takes the file TEXT not a path, so it works on an upload, a string inside a scene "
                          "description, or a file you read yourself. This is how a render stops using an "
                          "invented cosine falloff and starts using the measured distribution of an actual "
                          "fixture",
                          example="import lecore; m=lecore.UnifiedMind(); m.load_ies('IESNA:LM-63-2002\\nTILT=NONE\\n1 1000 1 3 1 1 -1 0 0 0\\n1.0 1.0 0.0\\n0 45 90\\n0\\n1000 500 0\\n')[1]",
                          native=True, aliases=("ies file", "photometric file", "lm-63", "luminaire profile",
                                                "real world light falloff", "load a light profile",
                                                "manufacturer light data", "measured light distribution"))
    c.register_capability("Transform (warp)", "move / rotate / warp across representations: VSA bind (rigid) + "
                          "permute (order), 4x4 matrices (translate/scale/rotate/compose/decompose/look_at + "
                          "quaternions), clifford rotors, anisotropic steering -- one facade",
                          example="from holographic.misc.holographic_transformhome import Transform; Transform.translation(t)",
                          native=True, aliases=("transform", "warp", "rotate", "translate", "scale", "rigid", "affine",
                                                "matrix", "quaternion", "rotor", "bind", "permute", "gizmo",
                                                # rev. 9 discoverability audit: multi-word phrasings for the KIT
                                                # ("quaternion from axis and angle", "translation matrix") lost to
                                                # the transform-TOWER theory entries. Minimal honest additions --
                                                # a first, wider set of nine shifted an unrelated pinned ranking
                                                # (catalog entries superpose; every alias perturbs every query).
                                                "translation matrix", "rotation matrix",
                                                "axis angle", "quaternion from axis and angle"))
    c.register_capability("Blend (combine)", "combine things into one: bundle (superposition, weighted = soft "
                          "mixture), lerp / slerp interpolation, Frechet mean on the sphere, front-to-back alpha "
                          "composite, and dict/scene merge with a conflict policy",
                          example="from holographic.misc.holographic_blendhome import Blend; Blend.bundle(vectors, weights)",
                          native=True, aliases=("blend", "combine", "merge", "interpolate", "lerp", "slerp", "mix",
                                                "composite", "superpose", "average", "crossfade", "morph"))
    c.register_capability("Scale (distribute)", "make something bigger than one box / one pass can hold: partition a "
                          "job, run the pieces independently, reassemble with a commutative monoid -- map_reduce, "
                          "load-balanced partition, image tiles / volume bricks; strategies tiling/octree/multires/"
                          "superposed/sparsefield", example="from holographic.misc.holographic_scalehome import Scale; Scale.map_reduce(buckets, worker, reduce='sum')",
                          native=True, aliases=("scale", "distribute", "partition", "map reduce", "tile", "brick",
                                                "parallel", "shard", "chunk", "monoid", "scale out"))
    c.register_capability("Query / database (domain)", "treat VSA stores as a database: SQL over tables, similarity/"
                          "time-travel/diff, durable + concurrent + graph + history query layers",
                          example="from holographic.agents_and_reasoning.holographic_query import run_sql, UserTable", native=False,
                          aliases=("query", "sql", "database", "table", "history", "diff", "time travel"))
    c.register_capability("Token sampling (temperature + nucleus)",
                          "stochastic next-symbol draw over any {symbol: weight} distribution -- the GENERATION dual "
                          "of argmax prediction. Promoted from the char generator into one primitive; wired as "
                          "PredictiveMemory.sample / generate_sampled and the mind's sample_instruction / "
                          "sample_recipe over the recipe grammar. Measured reason: a greedy generator limit-cycles "
                          "(MMD2 0.599 vs 0.011 sampled; 15x verbatim-copy) or flatlines on heavy-tailed streams. "
                          "Kept negatives in the docstring: nucleus/low-T delete rare events on heavy-tailed "
                          "alphabets; well-formedness (e.g. alternation) is the caller's decode-loop job",
                          example="from holographic.agents_and_reasoning.holographic_tokensample import sample_from_distribution; sample_from_distribution({'a': 0.7, 'b': 0.3}, temperature=1.0, top_p=1.0)",
                          native=True,
                          aliases=("sample", "sampler", "temperature sampling", "nucleus sampling", "top p",
                                   "stochastic generation", "sample the next token", "sample an instruction",
                                   "generate without limit cycling", "draw from a distribution",
                                   "instead of always picking the best", "stuck repeating in a loop",
                                   "pick a symbol randomly by weight", "weighted random choice",
                                   "roll a weighted die", "sample from a dict of scores"))

    # --- QUANTUM: the complex-wavefunction stack (Schrodinger split-operator, current, dot, Aharonov-Bohm) ---
    c.register_capability("Quantum field (complex wavefunction)", "a COMPLEX wavefunction psi on a grid -- the central quantum object; gaussian_packet launches a wave packet, set_potential/set_vector_potential install a well and magnetic flux, probability_density is |psi|^2. The quantum complement to the real-valued wave_field",
                          example="import lecore; m=lecore.UnifiedMind(); qf=m.quantum_field((128,128),dx=0.2); qf.gaussian_packet((30,64),6.0,(0.8,0.0)); qf.norm()",
                          native=True, aliases=("quantum", "wavefunction", "complex field", "psi", "quantum state", "electron wave", "quantum simulation", "wave function on a grid", "schrodinger field"), semantic="create/emit", consumes=(), produces=("field",))
    c.register_capability("Schrodinger solver (split-operator TDSE)", "evolve a quantum wavefunction in time by the time-dependent Schrodinger equation, UNITARILY (norm conserved to machine precision) via a split-step Fourier method -- the kinetic step is the analytic continuation of the heat propagator. Explicit Euler is unstable and NOT used (recorded negative)",
                          example="import lecore; m=lecore.UnifiedMind(); qf=m.quantum_field((128,128),dx=0.2); qf.gaussian_packet((30,64),6.0,(0.8,0.0)); m.quantum_solver(qf).run(50,0.02); qf.norm()",
                          native=True, aliases=("schrodinger", "schrodinger equation", "solve schrodinger", "time dependent schrodinger", "evolve a wavefunction", "quantum time evolution", "split operator", "split step fourier", "propagate a wave packet", "TDSE"), semantic="simulate/step", consumes=("field",), produces=("field",))
    c.register_capability("Probability current (quantum flow)", "the probability current j = (hbar/m) Im(psi* grad psi) - (q/m) A |psi|^2 of a wavefunction -- where |psi|^2 is flowing; streamlines of j are the glowing threads in an interferometer and a loop with circulation is a probability vortex. j/|psi|^2 feeds advect_field",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(); qf=m.quantum_field((96,96),dx=0.2); qf.gaussian_packet((30,48),6.0,(0.8,0.0)); jx,jy=m.probability_current(qf.psi,dx=0.2)",
                          native=True, aliases=("probability current", "quantum current", "probability flow", "where the probability is moving", "quantum flux", "probability velocity", "streamlines of psi", "probability vortex", "glowing threads"), semantic="analyze/measure", consumes=("field",), produces=("field",))
    c.register_capability("Quantum dot / transmission (resonant scatterer)", "a quantum dot as a potential well or barrier, and the MEASURED transmission of a packet past it (swept over energy) -- the resonance/tunnelling emerges from the solver, it is not painted on. Compare with and without the dot for the honest baseline",
                          example="import lecore; m=lecore.UnifiedMind(); d=m.quantum_dot_well((160,80),(80,40),depth=-8.0,width=2.5); m.quantum_transmission(0.7,dot_V=d,shape=(160,80),steps=300)",
                          native=True, aliases=("quantum dot", "resonant scatterer", "transmission", "tunnelling", "tunneling", "resonance", "fano", "breit wigner", "potential barrier", "how much gets through", "scattering off a well", "particle in a box", "particle in a box energy levels", "bound state energy levels", "energy levels of a well"), semantic="simulate/step", consumes=("field",), produces=("scalar",))
    c.register_capability("Aharonov-Bohm ring (magnetic flux phase)", "thread magnetic flux through a ring interferometer and MEASURE the Aharonov-Bohm phase the two arms accumulate -- equal to q*Phi/hbar even though the field is zero on the arms (only the enclosed flux is physical). quantum_solenoid_A builds the vector potential",
                          example="import lecore; m=lecore.UnifiedMind(); m.aharonov_bohm_phase(1.0,ring_radius=30)",
                          native=True, aliases=("aharonov bohm", "aharonov-bohm", "magnetic flux phase", "enclosed flux", "vector potential phase", "interferometer", "ring interferometer", "flux threaded ring", "AB phase", "gauge phase"), semantic="simulate/step", consumes=("field",), produces=("scalar",))
    c.register_capability("Two-slit interferometer (quantum)", "build a two-slit wall (high potential with two openings) for a wave packet -- the two slits become coherent sources and interference fringes appear downstream; the canonical warm-up before the Aharonov-Bohm ring",
                          example="import lecore; m=lecore.UnifiedMind(); qf,V=m.quantum_two_slit(shape=(128,128))",
                          native=True, aliases=("two slit", "double slit", "two-slit experiment", "slit interference", "young double slit", "quantum interference fringes", "coherent sources"), semantic="create/emit", consumes=(), produces=("field",))
    c.register_capability("Polarized light (Stokes state)", "the STATE of polarized light as a Stokes vector [S0,S1,S2,S3] (holographic_stokes): total intensity plus linear (Q,U) and CIRCULAR (V / handedness) polarization. Field-native (a whole image is (...,4)); reports degree-of-polarization, e-vector angle and handedness; scalar radiance lifts/round-trips byte-identically. The circular channel is the one the mantis shrimp uniquely sees",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); print(m.stokes_report(m.stokes_circular(1.0, handedness=1))['docp'])",
                          native=True, aliases=("polarization", "polarisation", "stokes vector", "degree of polarization", "e-vector angle", "circular polarization", "linearly polarized light", "unpolarized light", "handedness of light", "polarized reflection", "polarized light state"), semantic="create/emit", consumes=("spectrum",), produces=("spectrum",))
    c.register_capability("Identify an element by its properties", "IDENTIFY the element(s) whose categorical fingerprint {category, state} matches given properties (holographic_elements.identify_element) -- the REVERSE of element() (which looks up BY name). m.identify_element({'category':'noble_gas','state':'gas'}) -> the noble gases, ranked by match_record over all 43 element records, gated by decide_or_abstain. confident is False when several elements share the fingerprint (honest under-determined answer; narrow with more fields). KEPT NEG: categorical only -- atomic number/mass excluded.", example="import lecore; m=lecore.UnifiedMind(); r=m.identify_element({'category':'noble_gas','state':'gas'}); print([s for s,sc in r['ranked'][:3]], r['confident'])", native=True, module="elements", aliases=("which element is this", "identify an element from its properties", "find the element that is an inert gas", "reverse periodic table lookup", "classify an element by category", "what element has these properties"), semantic="analyze/match", consumes=("scalar",), produces=("selection",))
    c.register_capability("Optical elements (Mueller matrices)", "how optical elements TRANSFORM polarized light, as real 4x4 Mueller matrices (holographic_mueller): polarizer, wave plate / retarder (a quarter-wave plate converts linear<->circular -- the mantis R8 mechanism), optical ROTATOR (= Faraday rotation), depolarizer, and polarizing dielectric (Fresnel) reflection. Elements COMPOSE (a light path folds to one matrix) and apply to a Stokes vector or a whole field",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); print(m.stokes_report(m.apply_mueller(m.mueller_matrix('quarter_wave', angle=np.pi/4), m.stokes_linear(1.0, 0.0)))['docp'])",
                          native=True, aliases=("mueller matrix", "polarizer", "wave plate", "quarter wave plate", "half wave plate", "retarder", "optical rotator", "faraday rotation", "fresnel polarization", "birefringence", "transform polarized light", "polarizing filter"), semantic="transform/warp", consumes=("spectrum",), produces=("spectrum",))
    c.register_capability("Rotation-measure synthesis (Faraday depth)", "recover the FARADAY DEPTH of polarized light -- the line-of-sight magnetic field a radio telescope reads from a galaxy's polarized glow (holographic_rmsynth; Brentjens & de Bruyn 2005). Transforms complex polarization P=Q+iU over wavelength^2 into a spectrum over Faraday depth phi, peaked to {rm, polarized_intensity, angle0}. Field-native over an image cube; handles unevenly-sampled bands with gaps. The SEQUENCE costume of the Stokes state (U1). rm_synthesis / rmtf / rm_peak / rm_phi_grid / rm_resolution / stokes_faraday_depth", example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); L=np.linspace(0.03,0.24,200); P=2.0*np.exp(2j*(0.5+42.0*L)); g=m.rm_phi_grid(L); print(round(m.rm_peak(m.rm_synthesis(L,g,P=P),g)['rm'],1))", native=True, aliases=("rotation measure synthesis", "faraday depth", "faraday rotation measure", "RM synthesis", "line of sight magnetic field", "polarization angle vs wavelength", "magnetic field from polarization", "faraday dispersion function", "radio polarization analysis", "recover rotation measure", "stokes q u fft", "polarization over wavelength"), semantic="analyze/measure", consumes=("spectrum",), produces=("spectrum",))
    c.register_capability("Faraday sky map (telescope as observer)", "the TELESCOPE AS OBSERVER: Faraday rotation on a whole sky (holographic_rmsynth). faraday_rotate is the forward model -- rotate an intrinsic polarized signal by rm*lambda^2 across a band, the sky a radio dish receives (intensity + circular untouched). faraday_rm_map is the inverse -- recover a per-pixel Faraday-depth (line-of-sight magnetism) MAP from a sky Stokes cube (...,nchan,4) in one call, by rm synthesis over the whole field. The SAME polarization core reads a mantis eye and a radio telescope (the sensor unifier). faraday_rotate / faraday_rm_map", example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); L=np.linspace(0.03,0.24,140); s0=np.zeros((2,2,4)); s0[...,0]=1; s0[...,1]=1; cube=m.faraday_rotate(s0,L,np.array([[15.,-40.],[70.,-5.]])); print(np.round(m.faraday_rm_map(L,cube)['rm']).tolist())", native=True, aliases=("faraday rotation", "faraday rotate a sky", "rotation measure map", "RM map", "line of sight magnetism map", "recover magnetic field per pixel", "polarization sky cube to RM", "simulate faraday rotation", "telescope polarization observer", "galaxy magnetic field map", "radio polarization sky"), semantic="analyze/measure", consumes=("image",), produces=("image",))
    c.register_capability("Sky observation (cube + world axes)", "a SKY OBSERVATION as first-class data (holographic_skydata): a data cube + WORLD AXES (WCS-lite -- linear RA/Dec/freq/wavelength via crval/crpix/cdelt), plus meta. Convert pixel<->world, get an axis' real coordinates, turn a frequency axis into the lambda^2 the Faraday tools want, and reshape to (...,nchan,4) ready for faraday_rm_map. Deterministic save/load (json header + npy, no pickle). No astropy/FITS parser in core; header-dict + npy is the ingest contract. make_skydata / sky_world_coords / sky_lambda2 / sky_stokes_cube / save_skydata / load_skydata", example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); ax=[m.make_sky_axis('freq',5,'Hz',crval=1e9,cdelt=2e8)]; sky=m.make_skydata(np.zeros((5,)),ax); print(round(float(m.sky_lambda2(sky)[0]),4))", native=True, aliases=("sky data cube", "telescope observation container", "world coordinate axes", "WCS lite", "pixel to sky coordinate", "radio image cube", "frequency axis to lambda squared", "load a telescope cube", "gridded sky observation", "observation with RA Dec freq", "ingest a sky map"), semantic="create/emit", consumes=(), produces=("image",))
    c.register_capability("Star system from parameters", "PLUG DATA IN, GET A STAR SYSTEM (holographic_starsystem): assemble parameters -- a star's temperature/radius/mass and each planet's orbit (a,e), radius, temperature -- into a deterministic, JSON-serializable scene RECIPE. Star gets a blackbody colour; each planet a biome by temperature, a closed-form Kepler orbit (star at a focus), a position, and a seed to regenerate its surface via fractal_planet on demand. Same params+seed = byte-identical. Delegates to blackbody + fractal_planet + Kepler geometry. star_system / kepler_ellipse / kepler_position / temperature_to_biome / planet_field", example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); r=m.star_system({'star':{'temp_K':5772},'planets':[{'a':1.0,'e':0.02,'radius':0.09,'temp_K':288}]}); print(r['planets'][0]['biome'])", native=True, aliases=("build a star system", "star system from parameters", "procedural solar system", "assemble a planetary system", "plug data in to see a system", "planets on kepler orbits", "make a solar system", "star with planets", "orbit geometry", "kepler orbit", "planet temperature to biome", "simulate a star system"), semantic="create/emit", consumes=(), produces=("scalar",))
    c.register_capability("N-body gravity simulation", "N-BODY GRAVITY (holographic_nbody): integrate bodies pulling on each other under softened Newtonian gravity, O(N^2) direct sum, with a VELOCITY-VERLET symplectic integrator so total energy stays bounded (orbits close instead of spiralling). nbody_simulate runs it and reports the honest energy drift + an optional trajectory; circular_orbit_velocity seeds a stable orbit. The dynamics counterpart to star_system's closed-form orbits (they agree). Barnes-Hut / Poisson-field are declared accelerator paths. nbody_simulate / nbody_accel / nbody_energy / nbody_step", example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); p=np.array([[0.,0.],[1.,0.]]); v=np.array([[0.,0.],[0.,m.circular_orbit_velocity(1000,1,1.0)]]); print(m.nbody_simulate(p,v,np.array([1000.,1.]),0.001,50,G=1.0,softening=1e-3)['energy_drift']<0.01)", native=True, aliases=("n-body simulation", "nbody gravity", "gravitational simulation", "simulate orbits", "planets orbiting", "verlet integrator", "symplectic integrator", "gravity between bodies", "orbital dynamics", "evolve a star system", "galaxy dynamics", "run a gravity simulation"), semantic="simulate/step", consumes=("points",), produces=("points",))
    c.register_capability("Star cluster (many systems)", "a STAR CLUSTER -- many star systems in a field (holographic_starsystem; the UP direction of star_system). Masses come from a Salpeter IMF (mostly red dwarfs, a few blue giants) and colour each star by its main-sequence temperature, so it looks like a real population. Even low-discrepancy placement by default, or pass a density_field (e.g. a cosmic-web map from the maze/Physarum solver) to cluster systems along large-scale structure (Burchett 2020 MCPM). Deterministic recipe. star_cluster / sample_imf / mass_to_temperature", example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); c=m.star_cluster(30,seed=0,extent=2.0); print(c['n']==30)", native=True, aliases=("star cluster", "galaxy cluster", "many star systems", "population of stars", "cluster of stars", "initial mass function", "salpeter imf", "distribute stars in a field", "cosmic web of stars", "simulate a star cluster", "field of stars"), semantic="create/emit", consumes=(), produces=("points",))
    c.register_capability("Nebula (volumetric gas & dust)", "a NEBULA -- turbulent volumetric gas/dust you can render (holographic_nebula). nebula_volume builds a 3-D density field (res^3, [0,1]) with wispy filaments and dark voids from the engine's own FractalNoise; pass star positions to carve CAVITIES where stars blow bubbles (ties to star_cluster). nebula_field_fn wraps it as the callable render_volume marches (trilinear), so it drops into the ray-marcher; nebula_column is the cheap column-density look. An artist's nebula, not a hydro sim (fluid advection declared). nebula_volume / nebula_field_fn / nebula_column", example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); v=m.nebula_volume(res=24,seed=0); print(v.shape==(24,24,24))", native=True, aliases=("nebula", "gas cloud volume", "interstellar dust cloud", "emission nebula", "volumetric gas", "turbulent gas field", "star forming cloud", "3d density volume nebula", "molecular cloud", "make a nebula", "gas and dust cloud"), semantic="create/emit", consumes=(), produces=("field",))
    c.register_capability("Period of a signal (Lomb-Scargle)", "find the PERIOD of an unevenly-sampled signal (holographic_lombscargle; Lomb 1976, Scargle 1982) -- what a plain FFT can't do on gappy real observations. best_period searches a data-derived frequency grid and returns {period, power, fap}; false_alarm_probability runs a permutation null (times fixed) so a peak's significance is measured, not assumed; phase_fold shows a period is real by folding coherently. Closes the loop: a light curve -> a period -> Kepler -> star_system. best_period / lomb_scargle / lomb_scargle_auto / phase_fold", example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); rng=np.random.default_rng(0); t=np.sort(rng.uniform(0,20,120)); y=np.sin(2*np.pi*t/2.5); print(round(m.best_period(t,y,min_period=0.5,max_period=8)['period'],1))", native=True, aliases=("lomb scargle", "lomb scargle periodogram", "period of a light curve", "find period unevenly sampled", "periodogram irregular sampling", "detect periodicity with gaps", "orbital period from radial velocity", "phase fold a time series", "false alarm probability", "period finding", "how long is the period"), semantic="analyze/measure", consumes=("timeseries",), produces=("scalar",))
    c.register_capability("Observer (spectrum to sensor readings)", "an OBSERVER: turn a spectrum into sensor readings by integrating it against sensitivity curves (holographic_observer). A human eye (3 CIE curves), a mantis eye (~12 receptors), or a telescope bandpass are all the same object with different channels -- one core, many sensors. Field-native: a hyperspectral image (...,nlam) gives per-pixel readings (...,nchan) in one call. The human observer reproduces blackbody_rgb byte-identically (blackbody is this observer on a Planck spectrum). human_observer / make_observer / observe_spectrum / spectrum_to_rgb / observer_receptor_bank / xyz_to_srgb", example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); from holographic.misc import holographic_blackbody as bb; L=np.linspace(380,780,90); print(np.array_equal(m.spectrum_to_rgb(bb.planck_radiance(L*1e-9,5000.0)), bb.blackbody_rgb(5000.0)))", native=True, aliases=("observer", "custom sensor", "sensor response", "spectrum to color", "spectrum to rgb", "color matching functions", "CIE observer", "what the eye sees", "multi-band receptor", "camera spectral response", "integrate spectrum through filters", "hyperspectral to color", "mantis shrimp eye"), semantic="transform/warp", consumes=("spectrum",), produces=("image",))
    c.register_capability("Mantis-shrimp vision (12-band + polarization)", "see as a MANTIS SHRIMP does: 12 spectral receptors from deep UV to far red PLUS linear and CIRCULAR polarization (holographic_observer.mantis_view). The circular channels use a quarter-wave retarder (the R8 rhabdomere, Chiou 2008) before linear detectors -- the sense mantis shrimp uniquely have. Composes the observer (O1) and Mueller elements (P2). Field-native. KEPT NEGATIVE (Thoen 2014): a DIRECT per-receptor readout, NOT colour-opponent -- mantis colour discrimination is measured coarse. mantis_receptors / polarization_readout / mantis_view", example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); L=np.linspace(300,720,140); b=np.exp(-0.5*((L-500)/60)**2); S=np.zeros(L.shape+(4,)); S[...,0]=b; S[...,3]=b; print(m.mantis_view(S,L)['handedness_sign'])", native=True, aliases=("mantis shrimp vision", "mantis shrimp eye", "see ultraviolet and polarization", "circular polarization vision", "twelve band eye", "twelve photoreceptors", "see what a mantis shrimp sees", "UV plus polarization sensor", "handedness of light detector", "stomatopod vision", "many band eye readings"), semantic="transform/warp", consumes=("spectrum",), produces=("image",))
    c.register_capability("See what the mantis sees (false colour)", "FALSE COLOUR: show a human what a non-human sensor sees (holographic_falsecolor). Map invisible channels onto R/G/B -- ULTRAVIOLET becomes a chosen hue, e-vector ANGLE becomes hue (strength = saturation), circular HANDEDNESS becomes a red/blue diverging map. mantis_falsecolor turns a mantis_view into three images (colour, polarization, handedness). Field-native. EVERY map is a CHOICE (Eno), not true colour. wavelength_to_rgb / hsv_to_rgb / falsecolor_spectral / falsecolor_polarization / falsecolor_handedness / mantis_falsecolor", example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); L=np.linspace(300,720,140); S=np.zeros(L.shape+(4,)); S[...,0]=np.exp(-0.5*((L-330)/20)**2); S[...,3]=S[...,0]; fc=m.mantis_falsecolor(m.mantis_view(S,L)); print(float(fc['color'].max())>0)", native=True, aliases=("false color", "false colour", "see what the mantis sees", "visualize polarization as color", "map invisible channels to rgb", "see ultraviolet as visible color", "polarization angle to hue", "handedness color map", "wavelength to rgb", "make UV visible", "visualize a non-human sensor", "hsv to rgb"), semantic="convert/emit", consumes=("image",), produces=("image",))
    c.register_capability("Doppler velocity & drift acceleration", "read VELOCITY and ACCELERATION out of a spectral shift or drift (holographic_dedoppler). doppler_velocity turns an observed vs rest wavelength into a line-of-sight velocity (classical v=c*z, or relativistic, which stays below c); redshift gives z; doppler_shift is the forward model (velocity -> observed wavelength). drift_acceleration turns a narrowband frequency drift rate (Hz/s -- what detect_drifting finds) into the emitter's acceleration a=-c*(df/dt)/f: the SETI reading of a drifting tone. Field-native. doppler_velocity / redshift / doppler_shift / drift_acceleration", example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); lr=656.28e-9; print(round(float(m.doppler_velocity(m.doppler_shift(lr,3e5),lr))/1e3,1))", native=True, aliases=("doppler velocity", "redshift to velocity", "radial velocity from wavelength", "relativistic doppler", "doppler shift", "wavelength shift to speed", "drift rate to acceleration", "how fast is it moving", "recession velocity", "line of sight velocity", "SETI drift acceleration", "how fast is a star moving", "speed of a source from its spectrum", "velocity from a spectral line"), semantic="analyze/measure", consumes=("timeseries",), produces=("scalar",))
    c.register_capability("Authoritative game world shard (fixed-tick, deterministic)", "build a GAME on the engine (holographic_gameshard.GameShard): authoritative fixed-dt world tick fed by an ordered player-command queue, deterministic by construction (same command log -> identical sha256 digest: free lockstep verification). Collision culls via spatial_hash_pairs; richer dynamics delegate to rigid_body. AOI snapshot() + stateless delta_since() for clients; region departures for handoff over the distributed bus (massive-world sharding). save/load digest-identical; negatives in the module docstring.", example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); s=m.game_shard(seed=0); s.submit({'tick':0,'player':'a','seq':0,'op':'spawn','id':1,'pos':(0,0,0)}); print(s.step()['n'])", native=True, aliases=("build a video game", "game server", "multiplayer game world", "authoritative server tick", "game loop", "fixed timestep game", "deterministic lockstep", "player input command queue", "area of interest snapshot", "interest management", "state delta sync", "shard a massive world", "massively multiplayer world", "world region handoff", "entity simulation for a game", "mmo world shard"), semantic="simulate/step")
    c.register_capability("run_game_shard", "one-shot JSON game-world run (holographic_gameshard.run_shard): the agent-invokable face of the game shard -- pass a command list, tick count, and optionally a saved state blob; returns final state, per-tick lockstep digests, region departures, and an optional area-of-interest snapshot. Stateless on the wire: the state travels with the caller, so any distributed farm worker can serve the next call.", example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); r=m.run_game_shard([{'tick':0,'player':'a','seq':0,'op':'spawn','id':1,'pos':(0,0,0)}], 3); print(len(r['digests']))", native=True, aliases=("run a game tick over http", "invoke game world remotely", "stateless game step", "agent playable world", "step a game world with json", "resume a saved game world"), semantic="simulate/step")
    c.register_capability("Massive sharded game world (deterministic migration)", "scale a game to a MASSIVE world (holographic_gameshard.ShardWorld): a lazy grid of authoritative shards -- cost tracks occupied cells, not world size. Entities crossing a cell boundary migrate deterministically with exact velocity/mass carried over; snapshots span shard seams; a world-level sha256 digest gives lockstep verification across the whole grid. collect_only handoffs + receive() are the bus-transport seam: identical payloads in-process or across the distributed farm. run_game_world is the JSON /invoke face.", example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); r=m.run_game_world([{'op':'spawn','id':1,'pos':(3.5,1,1),'vel':(2,0,0)}], 5, cell=4.0, dt=0.1); print(r['migrated'])", native=True, aliases=("massive game world", "massively multiplayer world", "shard entities across regions", "entity migration between shards", "cross shard snapshot", "world scale simulation", "distribute a game world across machines", "open world game backend", "seamless world regions"), semantic="simulate/step")
    c.register_capability("game_bus_host", "run a game world ON the existing distributed system (holographic_gameshard.BusShardHost): each farm node owns a set of world cells and exchanges entity handoffs over the message/distributed bus -- one topic per cell, so ownership can move without topology re-learning. The interaction layer's handshake with the data layer (bus/coordinator/presence); duplicates none of it, per the coordinator's own monoid rule (a game tick is non-monoid feedback: it runs whole on one worker). Rounds are barriered (publish R, join R+1); pinned equal to the single-process world to 1e-12.", example="from holographic.scene_and_pipeline.holographic_distbus import MessageBus; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); bus=MessageBus(); w=m.game_world(cell=4.0,dt=0.1); h=m.game_bus_host(bus,w,[(0,0,0)]); w.spawn(1,(1,1,1)); print(h.tick()['n'])", native=True, aliases=("run game shards on the farm", "game world over the message bus", "connect game to distributed system", "multiplayer across machines", "node owns world regions", "handoff entities over the bus"), semantic="simulate/step")
    c.register_capability("Game world SSE streaming (per-client deltas)", "watch or drive a game world from a BROWSER (holographic_gameshard.WorldStreamer + service /game + /game/stream): POST /game creates a room, routes player commands, and advances the authoritative clock; GET /game/stream is an SSE push of per-client DELTAS -- first event is the full area-of-interest as 'added', later events only what changed, the wire format a three.js client feeds straight into its scene graph. advance=1 makes a stream the designated clock; a lock keeps mid-tick command POSTs replayable. Needs serve(threads=True) so an open stream never blocks input.", example="import lecore; from holographic.simulation_and_physics.holographic_gameshard import ShardWorld, WorldStreamer; w=ShardWorld(cell=8.0,dt=0.1); w.spawn(1,(1,1,1)); st=WorldStreamer(w); print(len(st.next_event('c1', center=(1,1,1), radius=5)['added']))", native=True, aliases=("stream game to browser", "watch the world live", "server sent events game", "three.js game client feed", "push world deltas to client", "live multiplayer view over http", "game room http api"), semantic="simulate/step")

    # --- GEOMETRY KERNEL (modeling-app backend: tolerance authority + exact predicates + intersection + trim + 2D) ---
    c.register_capability("Model tolerance + exact geometric predicates", "the geometry kernel foundation: ONE ModelTolerance authority (abs/rel/angular) every boolean/snap/intersection consults so they agree on equal, plus orient2d/orient3d EXACT-sign predicates (float fast path, Fraction exact fallback) that decide collinear/coplanar ties deterministically instead of by a fuzzy epsilon. See holographic_geomkernel.", example="import lecore; m=lecore.UnifiedMind(); m.orient2d((0,0),(1,0),(0,1))", native=True, aliases=("model tolerance", "geometric tolerance", "orient2d", "orient3d", "robust predicate", "exact sign of a determinant", "is a point left of a line", "collinear test", "are three points collinear", "which side of a line is a point", "coplanar test", "tolerance authority"))
    c.register_capability("Curve-curve intersection", "where two curves cross (K1): all intersections of two polylines as records {point, segment indices, parameters}, crossings decided by the exact orient2d so a near-tangency is not swallowed; plus self-intersections (what an offset curve must clean up) and split-at-crossing. See holographic_curveint.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); m.curve_intersect(np.array([[-1.,0],[1,0]]), np.array([[0,-1.],[0,1]]))", native=True, aliases=("curve curve intersection", "intersect two curves", "where do two curves cross", "spline intersection", "where do two splines cross", "do these curves cross", "find where curves meet", "self intersection of a curve", "polyline intersection", "segment intersection", "curve crossing points"), consumes=('curve',), produces=('selection',))
    c.register_capability("Surface-surface intersection (SSI)", "trace the intersection curve of two implicit surfaces f=0, g=0 (K2, the kernel keystone) by a predict-correct FIELD MARCH: tangent = grad f x grad g, corrector = Newton projection onto both surfaces (one more iterate-a-projection). Returns polylines; fit a NURBS for a trim loop. Tangencies reported degenerate, not marched into noise. See holographic_surfint.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); sA=lambda P: np.linalg.norm(np.asarray(P,float),axis=1)-1.0; sB=lambda P: np.linalg.norm(np.asarray(P,float)-np.array([1.,0,0]),axis=1)-1.0; len(m.surface_intersect(sA,sB,(-1.5,-1.5,-1.5),(2,1.5,1.5)))", native=True, aliases=("surface surface intersection", "SSI", "intersect two surfaces", "intersection curve of two surfaces", "trim curve from two surfaces", "where two surfaces meet", "solid intersection curve", "implicit surface intersection"))
    c.register_capability("Trimmed surface", "a surface restricted to trim loops in parameter space (K3): inside an outer loop and outside holes -- how Rhino represents a trimmed face. Robust point-in-trim (exact orient2d), trim-respecting tessellation, and a bridge that projects a 3-D SSI curve to a (u,v) trim loop. See holographic_trimsurf.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); flat=lambda u,v: np.array([u,v,0.0]); ts=m.trimmed_surface(flat, [[0,0],[1,0],[1,1],[0,1]]); ts.is_inside(0.5,0.5)", native=True, aliases=("trimmed surface", "trim a surface", "surface with a hole", "trimmed nurbs face", "cut a region from a surface", "trim loop", "bounded surface patch"))
    c.register_capability("2D region boolean + curve offset", "union/difference/intersection of two closed polygonal regions by exact even-odd membership (K4, the SketchUp-face/drafting layer), plus parallel-curve OFFSET with the folded loops a concave offset makes cleaned up via self-intersection removal. See holographic_region2d.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); A=np.array([[0,0],[1,0],[1,1],[0,1]]); B=np.array([[0.5,0],[1.5,0],[1.5,1],[0.5,1]]); round(m.region_boolean_area(A,B,'intersection'),2)", native=True, aliases=("2d boolean", "region boolean", "union of two polygons", "polygon difference", "clip polygons", "offset a curve", "parallel curve", "inset a polygon", "curve offset", "2d region union"))
    c.register_capability("2D constraint sketch solver", "a parametric 2-D sketch solved by ITERATED PROJECTION (K8, the SketchUp-inference / dimensioned-drawing engine): add points, declare constraints (fix/coincident/horizontal/vertical/distance/parallel/perpendicular/point-on-line), solve to a fixed point (Gauss-Seidel relaxation, the same iterate-a-projection pattern as IK/PBD/resonator), and read under/well/over-constrained. See holographic_sketch2d.", example="import lecore; m=lecore.UnifiedMind(); s=m.sketch2d(); a=s.add_point(0,0); b=s.add_point(3,0.3); s.fix(a); s.horizontal(a,b); s.distance(a,b,4.0); s.solve()['satisfied']", native=True, aliases=("2d constraint solver", "sketch constraint solver", "parametric sketch", "solve a dimensioned drawing", "make lines parallel or perpendicular", "constrain distance between points", "coincident constraint", "geometric constraint solver", "under or over constrained sketch"))
    c.register_capability("CAD export: STL + DXF", "write geometry OUT in the two open exchange formats a modeler needs (K7): mesh_to_stl (ASCII STL for 3-D meshes, tris/quads/ngons, per-facet normals) and polylines_to_dxf (minimal DXF R12 for 2-D drawings, POLYLINE/VERTEX, closed loops flagged -- the format Rhino/AutoCAD read). Pure strings; the caller writes the file. See holographic_cadexport.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); m.mesh_to_stl(np.array([[0.,0,0],[1,0,0],[0,1,0]]), [(0,1,2)])[:5]", native=True, aliases=("export STL", "write STL file", "export DXF", "write a 2d drawing", "save mesh for 3d printing", "dxf export", "stl export", "export a drawing to autocad", "write geometry to a file"))
    c.register_capability("Parametric surface analysis (curvature + draft)", "curvature ON a parametric surface (K9), not sampled off a mesh: Gaussian/mean/principal curvature at (u,v) via the first and second fundamental forms (sphere K=1/R^2, cylinder K=0, saddle K<0), plus the moldability DRAFT ANGLE for a pull direction (positive drafts, ~0 vertical wall, negative undercut) and a developable test. See holographic_surfanalysis.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); sph=lambda u,v: np.array([2*np.cos(u)*np.sin(v),2*np.sin(u)*np.sin(v),2*np.cos(v)]); round(m.surface_curvature(sph,0.7,1.0)['gaussian'],3)", native=True, aliases=("surface curvature", "gaussian curvature of a surface", "mean curvature", "principal curvatures", "draft angle", "moldability analysis", "is a surface developable", "curvature of a nurbs surface", "fundamental forms", "undercut detection"))
    c.register_capability("Object snap: midpoint + intersection", "modeling object-snaps (K10) on top of the existing vertex/edge/grid snap: snap a dragged point to the nearest EDGE MIDPOINT, or to the nearest INTERSECTION of 2-D polylines (crossings found by the robust curve intersector). Returns hit records for the picking layer. See holographic_snap.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); V=np.array([[0.,0,0],[5,0,0]]); m.snap_to_midpoints([2.4,0.2,0], V, [[0,1]])['position'][0]", native=True, aliases=("midpoint snap", "snap to midpoint", "intersection snap", "snap to intersection", "object snap", "osnap", "snap to where lines cross", "snap to edge midpoint"))
    c.register_capability("Edge fillet + chamfer (exact radius)", "round or bevel the crease where two implicit surfaces meet (K5), the field-native fillet: fillet_union/intersection/difference give an EXACT constant-radius circular arc at the edge (iq rounded booleans) -- a true dimensioned radius, unlike smooth_union whose k is a soft blend, not a radius; chamfer_union gives the flat 45-degree bevel. Result is an SDF that raymarches/meshes/emits. KEPT NEGATIVE: a 3-way vertex is only approximately r. See holographic_fillet.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); px=lambda P: np.asarray(P,float)[:,0]; py=lambda P: np.asarray(P,float)[:,1]; f=m.fillet_union(px,py,0.3); float(f(np.array([[0.,0.3,0]]))[0])", native=True, aliases=("fillet an edge", "round an edge", "constant radius fillet", "chamfer an edge", "bevel an edge", "round the corner between two surfaces", "edge blend", "rolling ball fillet", "fillet between two solids"))
    c.register_capability("B-rep solid topology (Euler-Poincare validity)", "the boundary-representation foundation (K6): the vertex/edge/loop/face/shell topology of an exact solid, with Euler-Poincare validity (V-E+F-R=2(S-H)), genus, closed-2-manifold checking, and a bridge that lets each FACE carry a trimmed analytic surface (K3). A B-rep face is a trimmed surface, not a polygon -- that is what distinguishes this from the mesh Euler ops. HONEST SCOPE: topology+validity+face-geometry; B-rep booleans (SSI re-stitch) are the declared next step. See holographic_brep.", example="import lecore; m=lecore.UnifiedMind(); m.brep_validate(m.brep_box())['genus']", native=True, aliases=("b-rep", "boundary representation", "solid topology", "euler poincare validity", "is this a valid solid", "faces edges loops shells", "genus of a solid", "closed manifold check", "exact solid representation"))
    c.register_capability("B-rep boolean (finished solid modeling)", "the FINISHED B-rep boolean -- union/difference/intersection of two solids into one watertight B-rep (the SSI-driven re-stitch turning K2/K3/K6 into full solid modeling). Routes both solids through the SDF (intersection seam + field combine + marching, reusing route_csg), wraps the watertight result as a B-rep, VALIDATED with K6 (closed 2-manifold, Euler, volume vs inclusion-exclusion). analytic=True recovers POLYGONAL faces (~100x fewer, same volume). See holographic_brepbool.", example="import lecore; m=lecore.UnifiedMind(); a=m.brep_box(lo=(-1,-1,-1),hi=(1,1,1)); b=m.brep_box(lo=(0,0,0),hi=(2,2,2)); r=m.brep_boolean(a,b,'union',bounds=((-1.5,-1.5,-1.5),(2.5,2.5,2.5))); r._boolean_report['closed_manifold']", native=True, aliases=("b-rep boolean", "boolean of two solids", "union two solids", "subtract one solid from another", "intersect two solids", "solid modeling boolean", "csg on solids", "merge two solids", "re-stitch solids"))
    c.register_capability("Node-graph editor backend", "the unifying NODE-GRAPH a 3-D node editor binds to: one heterogeneous graph of TYPED nodes (scalar/color/field/sdf/mesh/material/texture), 40-node palette (SDF CSG/transforms, bake, fields, textures, geometry modifiers, sdf_to_mesh, PBR/material sockets, audio drivers). ANY param is DRIVABLE; type/cycle-checked; memoized eval; dirty-propagating; JSON-serializable. DRILL-DOWN: list_nodes() overviews, describe(id) shows a node's exact knobs+values+socket types+wiring, describe_type(name) a kind's schema, set_param(id, knob=value) sets an EXACT value.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); g=m.node_graph(); s=g.add('sdf_sphere',{'radius':1.0}); g.describe(s)['params']; g.set_param(s, radius=2.5); g.describe(s)['params']['radius']", native=True, aliases=("node editor", "node based editor", "node graph editor", "wire nodes together", "shader node graph", "geometry nodes", "material node graph", "connect nodes with typed sockets", "visual node graph", "node graph backend", "dataflow node editor", "audio reactive", "audio drives parameters", "shader as a map", "drive a parameter with a signal", "time varying node graph", "make geometry react to music", "music reactive visuals", "drill down to a setting", "list a node's parameters", "what can I adjust on this node", "set an exact value on a node", "inspect a node", "node parameters", "adjust exact settings", "tweak a node's value"))
    c.register_capability("Semantic scene to node graph (drill down to exact settings)", "bridge the high-level SEMANTIC scene to an exact, editable NODE GRAPH ('as above, so below'). scene.to_node_graph() emits each object as an sdf primitive + sdf_translate at its EXACT size/position, unioned. Returns {graph, output, objects: name->node id; materials: name->pbr node (materials=True, colour/metallic/roughness); renderables: name->assign_material node (renderable=True, meshed geometry + material -> drawable)}. describe(id) drills in; set_param(id, radius=/roughness=/res=..) sets an EXACT value. English is the fast way in; the node graph is the precise finish.",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.build_scene('a big red sphere and a box'); ng=s.to_node_graph(); g=ng['graph']; sid=[i for n,i in ng['objects'].items() if 'sphere' in n][0]; g.set_param(sid, radius=3.0); g.describe(sid)['params']['radius']",
                          native=True, module="scene_semantic", aliases=("drill down from a command to exact settings", "semantic scene to node graph",
                                                "convert a described scene to nodes", "adjust exact settings of a described object",
                                                "fine tune a semantic scene", "as above so below", "high level command to exact node",
                                                "edit exact parameters of a scene object", "scene to node graph", "drill down to exact settings",
                                                "adjust exact colour or roughness of an object", "renderable node graph from a scene"))
    c.register_capability("Rotate or tilt a scene object", "ROTATE / TILT a scene object about an axis (closes the axis-aligned limitation, so leaves can splay into a rosette): scene.adjust('tilt the cone 30 degrees'), scene.adjust('rotate the box 45 about y'), scene.adjust('lean it left'). Sets rotation (axis, angle_deg); the realizer wraps it in a rotation-EXACT SDF (query points rotated about the centre, distance-preserving). tilt/lean default to x, rotate/turn/spin to y (turntable); 'about x/y/z' picks the axis; a left/down/back word negates; repeats on the same axis ACCUMULATE.",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.build_scene('a green cone'); s.adjust('tilt the cone 40 degrees'); s.objects[0]['rotation']",
                          native=True, aliases=("rotate an object", "tilt a shape", "tilt the cone", "rotate the box",
                                                "lean an object", "turn an object", "spin it", "orient at an angle",
                                                "rotate a scene object", "tilt a leaf outward", "face a different direction"))
    c.register_capability("Per-object render passes (which object made each pixel)", "BIDIRECTIONAL LOOKUP: scene.render_passes(want=['mask','depth','normal','position']) returns, per pixel, WHICH object produced it -- one Cryptomatte-style matte per object keyed by NAME ('object:<name>'), plus the requested G-buffer passes and a 'beauty'. The trace-back the renderer already computes (union SDF's nearest-object id at each hit), now surfaced: an EXACT per-object mask for OUR renders (no colour segmentation), which the FOCUSED critic (propose_edits(focus=...)) and per-object material/texture work build on. Deterministic.",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.build_scene('a red sphere and a blue box'); p=s.render_passes(want=['mask'], width=64, height=48); sorted(k for k in p if k.startswith('object:'))",
                          native=True, aliases=("which object did each pixel hit", "per object mask from a render", "trace a pixel to its object",
                                                "object id pass", "cryptomatte", "g-buffer", "render passes", "per object coverage matte",
                                                "aov render channels", "depth and normal pass", "bidirectional pixel lookup"))
    c.register_capability("Critique & refine a scene toward a target image (image->3D loop)", "THE CRITIC of the image->3D loop: scene.propose_edits(target_image[, geometry=True]) renders candidate edits (lighting/brightness/material/colour, and with geometry=True coarse move/scale), scores each by how much it cuts the perceptual distance, returns them RANKED by improvement -- nothing applied. scene.refine_to_target(target_image, max_steps) greedily applies the best edit until converged/out of budget. Deterministic; feed the top into adjust(). KEPT NEGATIVE: ranks colour/lighting/material and COARSE geometry well, blind to FINE geometry (node-graph drill-down's job).",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(); g=m.build_scene('a red sphere'); g.adjust('make it night'); t=np.asarray(g.render(width=48,height=36),float); s=m.build_scene('a red sphere'); s.propose_edits(t, candidates=['make it night','make it brighter'], width=48, height=36)['proposals'][0]['command']",
                          native=True, aliases=("propose edits to match a target image", "critique a render against a target",
                                                "automatically improve a scene to match a photo", "suggest changes to reduce image difference",
                                                "refine a scene toward an image", "image to 3d refinement", "hill climb scene edits",
                                                "self improve a scene", "match a scene to a reference image", "make the scene look like this photo",
                                                "reposition objects to match a photo"))
    c.register_capability("B-rep membership + boolean face classification", 'the classification half of a solid boolean (toward K6 booleans): point_in_brep tests whether points are inside a B-rep solid (delegates to the generalized winding number), and brep_boolean_faces decides which whole faces of A survive a union/difference/intersection with B, flagging faces that straddle the B boundary (which need a K2-SSI split). HONEST SCOPE: whole-face granularity; the SSI face-split + re-stitch is the declared next step. See holographic_brepbool.', example="import lecore; m=lecore.UnifiedMind(); c=m.brep_box(); bool(m.point_in_brep(c, [[0,0,0]])[0])", native=True, aliases=("is a point inside a solid", "point in solid", "point inside a brep", "solid membership", "boolean of two solids", "classify faces for a boolean", "inside outside test for a solid", "solid boolean classification"))
    c.register_capability("Move / rotate / scale an object (and actually render the rotation)", "scene_to_render placed objects by translation + uniform scale and DROPPED any rotation -- documented, invisible to the caller, with NO downstream error, so the picture silently disagreed with the document. mind.place(scene, handle, position=, rotation=, scale=) writes the transform (Euler degrees, axis+angle, or a 3x3; each argument replaces only its own component); render with affine=True to have the rotation actually RENDERED. Exact to 1e-12 against the matrix. OFF BY DEFAULT: turning it on moves every scene with a rotated object. KEPT NEG: uniform scale only",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.new_scene(); h=m.scene_add(s, name='c', geometry=m.shape('cube')); m.place(s, h, position=(1,0,0), rotation=(0,45,0)); print(m.scene_info(s)['objects'][0]['rotated'])",
                          native=True, module="scene_render",
                          aliases=("move an object in the scene", "rotate an object I already placed",
                                   "set position rotation scale of an object", "turn a cube 45 degrees",
                                   "place an object at a location", "tilt an object",
                                   "my rotation is not showing up in the render",
                                   "orient an object", "why did my object not rotate",
                                   "position an object in the scene document", "scale an object"))
    c.register_capability("Load an HDRI environment map (.hdr RGBE -> unbounded radiance)", "image-based lighting needed one missing piece and this is it: a Radiance .hdr/.pic (RGBE) reader giving UNBOUNDED linear radiance. DomeLight's color already took a callable and sky_dome already sampled an equirectangular env -- but load_image reads 8 bits, and an 8-bit env is the wrong input because an HDRI's sun is thousands of times brighter than its sky. MEASURED: a flat dome vs a procedural sky FIELD differ 0.0054 (invisible); the same env mirrored differs 0.0336. Gradients don't pay, DIRECTIONAL structure does. KEPT NEG: no .exr; XYZE raises; never clip the result",
                          example="import lecore; m=lecore.UnifiedMind(); # env=m.load_hdr('sky.hdr'); L=m.scene_light('dome', color=lambda d: m.sky_dome(d, env=env))\nprint(m.sky_dome([[0,1,0]]).shape)",
                          native=True, module="render",
                          aliases=("load an hdri environment map", "image based lighting",
                                   "light my scene with a real sky photo", "read a radiance hdr file",
                                   "load a high dynamic range image", "use a panorama to light the scene",
                                   "equirectangular environment map", "rgbe encoded image",
                                   "ibl environment", "hdri lighting", "open an hdr file"))
    c.register_capability("Texture a scene object (named procedural or image, JSON-safe)", "texture a Scene object BY NAME ('wood','marble','checker',... or an (H,W,3) image, None removes) -- JSON-safe end to end, which is the point: scene_to_render already honoured an albedo_socket callable and proc_texture already built one, but a CALLABLE cannot cross POST /invoke, so over HTTP texturing was impossible while every part worked in-process. This builds the callable server-side from JSON. SOLID texture (evaluated at world points -- grain carves through, no UVs needed). KEPT NEG: albedo only; image mapping is world-XZ planar (triplanar needs normals the socket contract lacks)",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.new_scene(); h=s.add(name='b', geometry=m.shape('sphere')); m.scene_set_texture(s, h, 'wood', scale=3.0, colors=((0.35,0.2,0.08),(0.75,0.55,0.3)))",
                          native=True, module="scene_render",
                          aliases=("put an image on the cube", "wood grain texture on my object",
                                   "apply an image texture to an object", "texture an object in my scene",
                                   "procedural texture on a scene object", "make the ball checkered",
                                   "marble texture", "paint a texture onto a shape",
                                   "remove the texture from an object", "tile an image over the floor",
                                   "make the cube look like wood", "slap a texture on it",
                                   "give the sphere a pattern"))
    c.register_capability("Animate the scene document (keyframes -> frames -> GIF)", "keyframes in, frames out, optionally an animated GIF -- the see->fix loop for MOTION. Composes Timeline + place + render_preview (a Timeline cannot cross /invoke). keys = {handle: {position/rotation/scale: [[t,value],...]}}, seconds. save_gif is stdlib GIF89a, deterministic (fixed 252-colour lattice, no median-cut). sky_keys={'hour':[[t,h],...],...} animates the sky per frame (timelapse; with no lights given the sky drives the dome so the ground follows). KEPT NEG: preview quality; Euler lerp, no quaternions; the last frame's transforms persist (undoable)",
                          example="import lecore; m=lecore.UnifiedMind(); s=m.new_scene(); h=s.add(name='b', geometry=m.shape('sphere')); f=m.render_animation(s, m.camera(eye=(0,1,3), target=(0,0,0)), {h: {'position': [[0,[-1,0,0]],[1,[1,0,0]]]}}, n_frames=4, width=32, height=24)",
                          native=True, module="scene_render",
                          aliases=("animate an object in my scene document", "keyframe the cube position",
                                   "render an animation of my scene", "render frames over time",
                                   "turn my scene into a video", "animate the scene and save frames",
                                   "make a gif of my scene", "bouncing ball animation",
                                   "move an object between two keyframes", "save an animated gif",
                                   "day to night timelapse animation", "animate the time of day",
                                   "sunset timelapse render"))
    c.register_capability("Describe to document (words -> handled, renderable scene objects)", "words -> the CANONICAL Scene document: named, handled objects you can texture, place, keyframe and path-trace. leCore had TWO scene systems that could not talk -- build_scene's SemanticScene and the Scene document (handles/undo, where every parity faculty landed) -- so an agent starting from words was cut off from all of it (8/8 audit phrasings missed). REUSES interpret_description + realize_scene; parsed colours become PBRMaterials; unknown words are REPORTED, never dropped. KEPT NEG: realizer has no rotation; SDFs arrive pre-placed so document transforms start identity",
                          example="import lecore; m=lecore.UnifiedMind(); r=m.describe_to_scene('a red cube and a green sphere'); print(sorted(r['handles']), r['unknown'])",
                          native=True, module="unified",
                          aliases=("turn a text description into scene document objects",
                                   "convert build_scene output to the scene document",
                                   "semantic scene into editable document", "describe a scene then keyframe it",
                                   "from words to objects I can texture and animate",
                                   "promote a described scene to the real document",
                                   "make a described scene renderable with the path tracer",
                                   "words to primitives with handles", "create a scene by describing it"))
    c.register_capability("Refine a scene toward a target image (the self-improving loop)", "hand a described scene a TARGET IMAGE and the engine improves itself toward it -- past screenshot-and-hope: Blender's integration shows an agent its render but cannot score candidate edits against a goal and apply the best. apply=True runs the bounded greedy loop (applied/start/final/history); apply=False only SCORES, ranked, touching nothing. Verified live: 'a red sphere' toward a night target, 0.2625 -> 0.0000 -- it rediscovered 'make it night' itself. Deterministic. KEPT NEG: edits are sentences, so it works on SemanticScene; promote via describe_to_scene after",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(); g=m.build_scene('a red sphere'); g.adjust('make it night'); t=np.asarray(g.render(width=96,height=72),float); s=m.build_scene('a red sphere'); print(m.refine_scene(s, t)['applied'])",
                          native=True, module="scene_semantic",
                          aliases=("critique my render and improve it", "match my scene to this image",
                                   "automatically refine a scene toward a target image",
                                   "score candidate edits against a goal", "self improving render loop",
                                   "make my scene look like this picture", "close the loop on a render",
                                   "propose edits ranked by improvement"))
    c.register_capability("Fetch an external asset (pinned, content-addressed, replayable)", "fetch an external asset (HDRI/model/texture) into a CONTENT-ADDRESSED cache. The network meets the determinism rule the way randomness does: BY PINNING. Unpinned fetch returns the sha256 to record; a PINNED fetch that is cached is served from disk with NO network I/O -- a recipe of (url, sha256) pairs replays bit-identically offline forever, which download-on-demand can never do. Mismatch = deleted + raises naming BOTH hashes. Opt-in (nothing in core imports it), http(s) only, 512 MB ceiling. Feed results to load_hdr / import_asset / asset_library",
                          example="import lecore; m=lecore.UnifiedMind(); # r=m.fetch_asset('https://example.com/sky.hdr'); print(r['sha256'])  # then pin it:\n# env=m.load_hdr(m.fetch_asset(url, sha256=r['sha256'])['path'])\nprint('see holographic_assetfetch')",
                          native=True, module="assetfetch",
                          aliases=("download a file from a url", "fetch an asset from the internet",
                                   "get a model file from polyhaven", "http download with checksum",
                                   "cache a downloaded file", "verify a download against a hash",
                                   "download an hdri", "pin an external asset",
                                   "reproducible asset download", "pull a file from the web reproducibly"))
    c.register_capability("Parametric sky (time of day, sun, moon, stars, high cloud layers)", "a PARAMETRIC sky: hour drives a keyed gradient palette AND the sun's arc; stars are a hash of direction (same seed = same sky forever), fading by daylight and by cloud; moon=True auto-places opposite the sun; SEVEN cloud kinds (cirrus/cirrostratus/cirrocumulus/altocumulus/altostratus/stratocumulus/nimbostratus): Beer-Lambert shells, per-kind extinction/threshold/warp/erosion; cellular kinds keep GAPS. time_s/wind/evolve ANIMATE clouds (wind drifts; evolve slides through the solid noise so shapes MORPH; sky_keys feeds frame time). KEPT NEG: low clouds refused toward cloud_scene",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(); sky=m.sky_model(hour=19.0, clouds=[('cirrus',0.5)]); print(np.round(sky([[0,1,0]]),3))",
                          native=True, module="skymodel",
                          aliases=("time of day sky gradient", "night sky with stars",
                                   "render the moon in the sky", "starfield generator", "sunset sky colors",
                                   "cloudy sky with sun shining through", "cirrus or stratus cloud layer",
                                   "procedural sky model", "sunny daytime sky", "partially cloudy sky",
                                   "environmental sky primitive", "sky sphere environment",
                                   "mackerel sky", "broken cloud deck", "thin cloud veil",
                                   "animated moving clouds", "clouds changing shape over time"))
    c.register_capability("Sky-synced sun light (auto position/colour, optional cloud shadows)", "scene_light('sun', sky=<sky_model closure>) -- direction, colour, and day-scaling read from the SKY'S OWN sun state (one source of truth: the disk overhead and the light on the ground cannot disagree; below the horizon it contributes nothing). cloud_shadows=True gates intensity per shading point by the sky's cloud transmittance toward the sun -- the SAME shell and layer densities the sky paints, riding the existing intensity-field mechanism (no tracer changes). shadow_scale (default 60) is declared artistic licence: scene metres vs shell km. Custom directional lighting: omit sky=",
                          example="import lecore; m=lecore.UnifiedMind(); sky=m.sky_model(hour=9.5, clouds=[('stratocumulus',0.6)]); sun=m.scene_light('sun', sky=sky, cloud_shadows=True)",
                          native=True, module="lights",
                          aliases=("sun light for my scene", "light that follows the sun in the sky",
                                   "directional light synced to the sky", "cloud shadows on the ground",
                                   "sunlight through the clouds", "automatic sun position lighting",
                                   "patches of sun and shade", "sun light driven by time of day"))


    # ------------------------------------------------------------------------------------------------
    # RESTORED IN THE J-3D MERGE. The catalog split was authored against a PRE-FORK catalog, so every
    # capability registered after that fork -- the whole GPU/agent/compute layer, 30 entries -- was
    # silently absent from the parts. The FACULTIES were fine (place_work, wgsl_matmul, declare... all
    # still on the mind), which is exactly why no audit caught it: catalog_gaps and skill_lint check
    # that REGISTERED capabilities have homes and runnable examples, and neither can see an ABSENCE.
    # A test asserting a specific phrasing still routes was the only thing that noticed.
    # LESSON: when a registry is REORGANISED on a branch, diff the resulting NAME SET against the
    # base, never just the file. Re-registered here at the end of the last part, so ties still break
    # in the original registration order relative to everything the parts already hold.
    # ------------------------------------------------------------------------------------------------

    c.register_capability(
        "Agent tool-use loop (with a gate below the model)", "hands a model the relevant manifest, parses "
        "its tool call, dispatches through invoke(), feeds the result back, iterates. Over HTTP this worked; "
        "in process every embedder wrote their own loop, routing around the choke point. THE DIFFERENTIATOR "
        "IS THE GATE BELOW IT: route_or_abstain scores the task against a null BEFORE any step, and below the "
        "floor the loop refuses and the MODEL IS NEVER CONSULTED. Measured with a stub that always claims "
        "done: has-tool 20/20, no-tool 0/20 -- FALSE-ACTION RATE 0%. Refuses non-finite args and off-manifest "
        "tools; never guesses an unparsed reply",
        example="mind.attach_llm(my_fn); mind.agent_loop('smooth a bumpy mesh')",
        native=True, aliases=("let a model use my tools", "in process tool use loop",
                              "run an agent against the catalog", "agent loop",
                              "refuse a step when no tool fits", "model picks tools and i run them",
                              "tool calling loop without http"))

    c.register_capability(
        "Agent-socket benchmark (false-action rate)", "PRE-REGISTERED primary metric: false-action rate on a "
        "NO-TOOL set -- the number reference systems do not publish. The no-tool set is built by REMOVAL: each "
        "task is a real capability's own author-written alias asked against an index rebuilt WITHOUT that "
        "capability, so it is a coherent idiomatic request with nothing behind it and every near neighbour "
        "still present. Strictly harder than word salad. MEASURED 60/20 seeded: resolution 100.0%, FALSE-ACTION "
        "RATE 0.0%, variance ZERO, model calls 0. KEPT NEGATIVE: rungs 1-5 fired 0/60",
        example="mind.agent_benchmark(n_has=60, n_no=20); mind.catalog_without(['some capability'])",
        native=True, aliases=("measure the false action rate", "benchmark the agent socket",
                             "how often does it act when no tool exists", "agent benchmark",
                             "remove a capability and see if it still answers",
                             "does it refuse when nothing fits"))

    c.register_capability(
        "Batched bind on ANY GPU (circular convolution)", "bind IS a plain circular convolution (verified to "
        "7e-15), so it can be rfft->multiply->irfft in O(D log D) or DIRECT in O(D^2). Direct is ~100x more "
        "arithmetic and is the right trade: it reuses the SAME workgroup-reduction shape as the matvec and "
        "matmul kernels -- no bit-reversal, no twiddle tables, no multi-stage barriers -- and ARITHMETIC IS "
        "WHAT A GPU HAS. Batched on purpose: a single bind is ~0.03ms on CPU, below any dispatch floor. "
        "Correctness verified against bind_batch; the crossover needs a real device",
        example="out = mind.wgsl_bind_batch(a_stack, b_stack)   # (K, D) each",
        native=True, aliases=("bind many vectors at once on the gpu", "batched bind on any gpu",
                              "circular convolution on the gpu", "gpu bind", "convolve a batch on the gpu"))

    c.register_capability(
        "Bring your own query embedder (dense routing seam)", "install ANY callable text->vector so "
        "route_semantic can reach the dense index from FREE TEXT -- today the shipped artifact is the "
        "document side only (509 modules x 128d) and free text returns an honest None. Same contract as "
        "attach_llm: leCore imports no model SDK. VERIFIED BY DEFAULT with a round-trip space probe: the "
        "index lives in ONE space, a cosine against a different model's vectors is MEANINGLESS yet still "
        "returns confident ranks. Dimension is checkable, space is not -- so sampled modules must self-recall "
        "on their own docstrings (chance 5/509)",
        example="mind.set_embedder(my_encode); mind.route_semantic('smooth a bumpy mesh'); mind.set_embedder(None)",
        native=True, aliases=("supply my own embedding model", "bring your own vector encoder",
                              "plug in an external embedder", "use a sentence transformer for routing",
                              "dense retrieval with my own model", "set embedder",
                              "make free text routing work", "external encoder for capability search",
                              "route by meaning with my own embeddings"))

    # --- exact / matrix-free TRANSFORMS ---
    c.register_capability(
        "Bundle capacity as a measured load ratio", "how many things fit in a bundle -- answered with its "
        "THREE VARIABLES attached (readout, dimension, quality floor), measured at call time, never a "
        "constant. The folklore '20-32 instructions' was a LINEAR-readout artifact: naive cosine holds safe "
        "M/D = 0.02 while cosamp/amp hold 0.17 (44 items at D=256, 174 at D=1024 -- 8.7x more, and the "
        "ratio COLLAPSES across dims, which is why capacity is m/D not a count). Reference numbers are for "
        "an INCOHERENT dictionary; coherence inverts the ranking, so pass codebook= for your atoms. Gate is "
        "mean minus sd: a lucky-seed capacity is not a capacity",
        example="mind.bundle_capacity(512, 'cosamp'); mind.measure_recovery_curve(512, 'amp')",
        native=True, aliases=("how many things fit in a bundle", "safe number of items to superpose",
                              "capacity of a bundle at this dimension", "load ratio before recovery fails",
                              "will recovery still work with this many items", "bundle capacity",
                              "how many items can i pack into one vector", "superposition limit"))

    c.register_capability(
        "Bundle recovery (unmix a superposition)", "recover the components of cue = sum_i w_i * codebook[i] -- FIVE "
        "members: LINEAR one-shot correlate + top-m (washes out at load); occlusion_recall GREEDY matching pursuit "
        "(cheap, never revisits); iht_recall projected gradient (revises its support); cosamp_recall batch-select + "
        "least-squares (exact coefficients, best on COHERENT dictionaries); amp_recall Onsager-corrected AMP (K "
        "OPTIONAL, flat cost, best at HEAVY load). NEITHER DOMINATES -- measured D=512/N=2048: all tie at 1.000 to "
        "M/D=0.17; AMP 0.558 vs CoSaMP 0.167 at M/D=0.33; but on a coherent dictionary AMP 0.052 vs CoSaMP 1.000",
        example="mind.cosamp_recall(cue, codebook, K); mind.iht_recall(cue, codebook, K); mind.occlusion_recall(cue, codebook, K)",
        native=True, aliases=("recover many items from one bundle", "find which codebook entries are in this sum",
                              "unmix a superposition into its parts", "what went into this bundle",
                              "sparse recovery against a dictionary", "greedy solver for a mixture of atoms",
                              "decode a superposition one piece at a time", "unbundle", "unbundling",
                              "compressed sensing", "matching pursuit", "sparse recovery", "demix",
                              "how many things fit in a bundle", "pull the parts out of a sum of vectors",
                              "which atoms are in this mixture", "recovery family"))

    # --- caching / baking: the CACHES (audit named ~9) = bake_and_query ---
    c.register_capability(
        "Clean up many cues at once (batched cleanup)", "the missing UP direction of cleanup, and it pays on "
        "the CPU ALONE: one (K,D)x(D,M) matmul instead of K separate matvecs is 2.58x at K=32, 5.36x at K=64, "
        "5.92x at K=128 -- BLAS getting one big matmul rather than K small ones, with no device involved. "
        "backend='wgsl' routes the same computation to ANY GPU, DEFAULT OFF because the host<->device "
        "crossover has never been measured on real hardware and the one thing worse than not using a device "
        "is using it on a guess. Indices resolve by lowest index on both paths, so ties cannot move",
        example="idx, scores = mind.cleanup_batch(codebook, queries)   # backend='wgsl' to try a device",
        native=True, aliases=("clean up many cues at once", "batch cleanup", "recall many vectors at once",
                              "nearest atom for a stack of queries", "batched nearest neighbour"))

    c.register_capability(
        "Decision-safe quantization (does the ARGMAX survive?)", "measure the top-1 FLIP RATE when an index "
        "is quantized -- not reconstruction error, the DECISION. A code can hold cosine 0.9999 and still "
        "change which entry wins, and a flipped argmax is a different answer. Returns flip_rate plus the "
        "margin distribution, because a rate without margins says what happened, not why. MEASURED on the "
        "509x128 routing index: normal queries flip 0.00% down to 2 BITS; queries midway between two "
        "documents collapse to margin ~0.058 and flip at 8. FLIP RATE IS GOVERNED BY MARGIN, not by corpus "
        "size or bit width",
        example="mind.decision_flip_rate(index, queries, bits=8); mind.crowded_subset(index, 200)",
        native=True, aliases=("does quantization change the answer", "top 1 flip rate",
                              "is this index decision safe", "argmax flips under compression",
                              "how few bits can i use for retrieval", "quantization decision safety",
                              "will compressing my vectors change which one wins",
                              "margin distribution of a codebook", "re-prove quantization on a new index"))

    c.register_capability(
        "Declare a body, let the ladder fill it", "describe what you want; the engine walks rungs "
        "cheapest-and-most-provable FIRST and stops at the first clearing its gate: 0 route_or_abstain -> "
        "invoke, 1 typed plan, 2 synthesize_procedure (EXACT, execution-verified), 3 fill_capability_gap "
        "(TOL). Every result carries rung/mechanism/exactness/reversibility/confidence/why PLUS a descent "
        "log saying why each rung above declined -- that log IS the explanation. REFUSAL IS A RESULT: an "
        "unresolvable request returns ok=False, never a guess. max_rung=5 keeps it deterministic; every "
        "gate is NaN-guarded because a NaN score WINS an unguarded argmax",
        example="mind.declare('smooth a bumpy mesh'); mind.declare_explain('...'); f = mind.declares(fn)",
        native=True, aliases=("declare a method and let the engine fill it in",
                              "resolve an empty function body at runtime",
                              "try cheap deterministic ways before calling a model",
                              "which rung answered my request", "escalating ladder of mechanisms",
                              "fill in a stub", "agent socket", "let the engine work out how",
                              "explain how this would be answered", "refuse instead of guessing"))

    c.register_capability(
        "Hadamard codebook (cleanup as one transform)", "cleanup WITHOUT scanning every atom: atoms are the "
        "sign-permuted rows of a Hadamard matrix, so correlating against ALL of them is one Walsh-Hadamard "
        "transform -- O(D log D) not O(K*D), atoms generated not stored, rows mutually orthogonal so crosstalk "
        "is exactly zero, and argmax is the exact ML nearest-codeword decode (Reed-Muller's Green machine). "
        "MEASURED at equal K and D: 6.9x at D=1024, 219x at D=8192. KEPT NEGATIVES: LOSES at D=256 (0.49x, "
        "crossover ~D=512), and K is CAPPED at 2*D by construction",
        example="cb = mind.hadamard_codebook(1024); cb.cleanup(cue); mind.hadamard_codebook_measure()",
        native=True, aliases=("cleanup without comparing against every codebook entry",
                              "find the nearest codebook entry without scanning every one",
                              "structured codebook so cleanup is a transform", "nearest codeword in log time",
                              "speed up cleanup when the codebook is huge", "sublinear cleanup",
                              "reed muller decoding", "maximum likelihood nearest codeword",
                              "green machine decoder", "fast nearest atom", "orthogonal codebook",
                              "cleanup faster than a matmul", "decode a codeword with a fast transform"))

    c.register_capability(
        "How many cores can I actually use (+ should I pool?)", "cpu_budget() is NOT os.cpu_count(), which "
        "LIES IN A CONTAINER -- it reports the HOST's cores and ignores cgroup quota and affinity, so "
        "--cpus=2 on a 64-core box answers 64 and a pool sized from it spawns 64 interpreters to share 2 "
        "cores: slower than sequential and 64x the memory. Takes the MINIMUM of affinity, cgroup v2/v1 quota "
        "and cpu_count. should_pool() then decides if a pool pays, refusing on <2 cores, <2 buckets, or work "
        "per bucket below ~4x the 0.2ms dispatch cost",
        example="mind.cpu_budget(); mind.should_pool(n_buckets=8, est_ms_per_bucket=50.0)",
        native=True, aliases=("how many cores do i have", "detect available cpus",
                              "pick a worker count automatically", "should i use a process pool",
                              "is parallelism worth it here", "how many workers should i start",
                              "cpu count in a container"))

    c.register_capability(
        "How many slots can I drop under memory pressure", "device memory is a hard ceiling with no swap, so "
        "pressure means failure rather than slowdown -- a distributed representation can DEGRADE instead. "
        "Dropping slots reduces the EFFECTIVE DIMENSION, so the budget is the load-ratio law: recall holds "
        "while n_items/(keep*dim) stays under the safe ratio. NO NEW THEORY -- verified across 5 configs. "
        "CORRECTION KEPT LOUD: the 100%-at-40%-destroyed figure is about DAMAGE (zeroed slots, no memory "
        "saved); TRUNCATING to 40% at the same load gives 85%, not 100%. Different quantities",
        example="mind.drop_budget(dim=1024, n_items=16)   # -> keep 78%, 1792 bytes saved",
        native=True, aliases=("how many slots can i drop", "degrade instead of running out of memory",
                              "shrink a vector under memory pressure", "memory budget for a bundle",
                              "how much can i truncate"))

    c.register_capability(
        "Make the attached LLM a planner-visible tool", "attach_llm sets the mind's _llm and a bus bridge "
        "but does NOT register the model as a tool -- so Planner.plan, optimize_toolchain, CircuitBreaker and "
        "SkeletonLibrary were all BLIND to it: the one tool that can do fuzzy language work was the one the "
        "planner could not reach. llm_tool() registers it like any other tool (keyword vector, success rate, "
        "breaker). THE POINT: a registered model can be FAILED OVER AWAY FROM -- measured, a flaky model's "
        "breaker opens after 3 failures and the planner is then only offered the deterministic tool. A system "
        "whose only mechanism IS the model cannot do that",
        example="mind.attach_llm(my_fn); tool = mind.llm_tool(description='rewrite text')",
        native=True, aliases=("let the planner use the language model", "register an llm as a tool",
                              "make the model visible to the planner", "fail over away from a flaky model",
                              "llm as a tool", "use my model in a plan",
                              "what happens when the model keeps failing"))

    c.register_capability(
        "Measure where the GPU starts winning (crossover)", "the ONE number blocking the compute backlog: "
        "should_offload's thresholds are ARITHMETIC FROM PCIe BANDWIDTH, not measurements, and everything "
        "downstream is wired and default-off waiting on them. Sweeps CPU vs device across dim/count/batch and "
        "reports the crossover in bytes. HANDLES THE TIMING TRAP -- GPU calls are async, so it reads every "
        "result back to force completion; timing a launch instead of an execution is the classic spectacular "
        "wrong number. REFUSES TO FLATTER A SOFTWARE ADAPTER: llvmpipe/WARP get a MEANINGLESS banner",
        example="print(mind.gpu_crossover(kind='cleanup', text=True))",
        native=True, aliases=("measure the gpu crossover", "benchmark cpu vs gpu",
                              "find where the device starts winning", "is my gpu actually faster",
                              "when should i use the gpu", "gpu benchmark"))

    c.register_capability(
        "NTT exact integer binding", "bind/convolve with ZERO rounding error: the same circular convolution "
        "bind() does, computed as a Number-Theoretic Transform over Z_q, so it is EXACT and BIT-IDENTICAL ON "
        "EVERY MACHINE -- numpy.fft is not (SIMD width reorders the summation; NumPy #11926), and here a ULP "
        "flip is an argmax flip. Integer input only; the modulus bound is checked and RAISES rather than "
        "wrapping. KEPT NEGATIVES: 19-50x SLOWER than the float bind (exactness, never speed), and unbind is "
        "still HRR's QUASI-inverse -- cleanup is not deleted",
        example="mind.ntt_bind(a, b); mind.ntt_unbind(c, a); mind.ntt_convolve(a, b); mind.ntt_measure_vs_fft()",
        native=True, aliases=("exact circular convolution with integers", "bind two vectors with no rounding error",
                              "modular arithmetic convolution", "number theoretic transform",
                              "convolution that is identical on every machine", "integer only binding",
                              "bind without floating point", "exact bind", "reproducible convolution",
                              "deterministic binding across cpus", "ntt", "exact convolution",
                              "binding with no rounding", "bit exact binding"))

    c.register_capability(
        "Null-reference a synthesis threshold", "is the 0.85 coherence bar MEANINGFUL on your library? "
        "synthesize_for_goal accepts a chain when coherence clears a bare constant -- and that constant "
        "encodes an assumption about how coherent a RANDOM goal can get, which is a property of the LIBRARY, "
        "not the algorithm. Re-runs the identical synthesis on random unit goals (no chain behind them by "
        "construction) and reports where the real score sits. MEASURED: real goals 1.000, random 0.14-0.24, "
        "so 0.85 separates -- the number the constant hides. Wired into declare(null_check=True)",
        example="mind.gap_gate_null(library, goal_sig); mind.declare(req, args=..., null_check=True)",
        native=True, aliases=("is my threshold meaningful", "null reference a coherence gate",
                              "check a synthesis threshold against chance",
                              "score versus its own null for capability synthesis",
                              "is 0.85 a real bar", "validate a gate constant"))

    c.register_capability(
        "Query expansion gated on faithfulness", "let a model rewrite a request into catalog vocabulary "
        "before retrieval, then REFUSE the rewrite unless it keeps the original's meaning. MEASURED: random "
        "padding cannot smuggle a no-tool query past the router (0/8 -- the null is built at MATCHED TOKEN "
        "COUNT so dilution scores worse), but a TARGETED rewrite sails through (1/3: 'purple monkey "
        "dishwasher' -> 'smooth a bumpy mesh' routes confidently). A NULL DETECTS IRRELEVANCE, NOT "
        "INFIDELITY. So the primary gate is overlap with the ORIGINAL; both gates apply, not either",
        example="mind.attach_llm(my_fn); mind.expand_query('how do i fix a lumpy model')",
        native=True, aliases=("rewrite my query into catalog words", "query expansion",
                              "let the model rephrase before searching",
                              "stop a rewrite from changing what i asked", "expand a search query",
                              "is this rewrite faithful"))

    c.register_capability(
        "Reduce and argmax on ANY GPU (WGSL)", "sum/max/min and argmax over a 1-D array on Vulkan/Metal/DX12/"
        "WebGPU. The primitive that unlocks the VSA kernels: elementwise maps serve rendering and NONE of "
        "bundle/cleanup/resonator/amp/htcodebook, which are all cross-invocation reductions. TWO-STAGE -- "
        "workgroup partials in shared memory, host finishes -- because a grid-wide barrier does not exist in "
        "WGSL and atomics are float-nondeterministic. ARGMAX splits deliberately: value on device, INDEX on "
        "host by lowest index, so ties break canonically. Measured 200/200 on adversarial exact ties",
        example="mind.wgsl_reduce('sum', data); idx, val = mind.wgsl_argmax(similarities)",
        native=True, aliases=("sum an array on the gpu", "gpu reduction", "argmax on the gpu",
                              "reduce a vector on any gpu", "find the max on the graphics card",
                              "cleanup on the gpu"))

    c.register_capability(
        "Resonator restart budget advisor", "how many restarts does YOUR factoring problem need -- measured "
        "on your own codebooks. The F>=4 'capacity cliff' is a SEARCH BUDGET, not a capacity limit: same "
        "network, same dimension, 25% at restarts=4 and 100% at 256. The default was NOT raised, and the "
        "reason is the cost profile: a bigger cap is nearly free when an answer exists (early exit) and 13x "
        "slower when there is NONE, because a refusal must exhaust the budget. The sequence is PREFIX-STABLE, "
        "so raising it could not flip an existing answer -- the objection is cost alone",
        example="mind.advise_restarts([bookA, bookB], targets=(0.95,))",
        native=True, aliases=("how many restarts does my resonator need", "pick a search budget",
                              "how long should i search before giving up", "advise a restart count",
                              "is my factoring failing from budget or capacity"))

    c.register_capability(
        "Resource policy (what this process may use)", "the OPERATOR says what is allowed -- cpu_cores cap, "
        "pool allow/deny, gpu auto/on/off, device_memory_mb -- because cpu_budget() answers what is "
        "PHYSICALLY AVAILABLE, which is not what this process MAY TAKE on a shared box or beside the user's "
        "real work. A POLICY CAPS, IT DOES NOT COMMAND: cpu_cores=4 means never more than 4 and the measured "
        "gates still decide inside it. Precedence explicit > policy > env > auto. Reports the SOURCE of every "
        "value and flags which settings change NUMERICS (gpu) versus only speed (cores, pool)",
        example="mind.resource_policy(cpu_cores=4, gpu='off'); mind.resource_policy()",
        native=True, aliases=("limit how many cores it uses", "turn off the gpu",
                              "configure resource limits", "set a cpu limit",
                              "stop it using all my cores", "system configuration settings",
                              "what is it allowed to use", "restrict hardware usage"))

    c.register_capability(
        "Return the tie, then verify which candidate works", "decide_or_abstain detects a knife-edge then THROWS THE "
        "ALTERNATIVES AWAY. tied_candidates returns the set within margin (a clear winner gives a ONE-element "
        "set, never empty -- 'no ambiguity' and 'no answer' must not look alike); verify_and_keep tries them "
        "in rank order and keeps the first that VERIFIES, reporting all-failed instead of guessing. Not a "
        "learned tie-breaker: at a real tie candidates are EQUALLY GOOD, so verification beats learning. "
        "MEASURED: 0% ties on a random codebook, 84% on a coherent one under noise -- a degraded-regime tool",
        example="t = mind.tied_candidates(ranked, margin=0.01); mind.verify_and_keep(t['candidates'], check)",
        native=True, aliases=("what were the runner up matches", "return several candidates instead of one",
                              "how close was the second best answer", "try both and see which works",
                              "handle an ambiguous match", "dont guess when its a tie",
                              "adapt instead of breaking on a tie"))

    c.register_capability(
        "Run a kernel on ANY GPU via WGSL (vendor-neutral)", "emit_kernel already projects an annotated "
        "Python kernel into WGSL; this DISPATCHES it -- @compute entry point, storage bindings, bounds guard "
        "-- on Vulkan / Metal / DX12 / WebGPU, where use_gpu's CuPy backend is CUDA/NVIDIA ONLY. The shader "
        "is a PROJECTION of the authoritative Python, so verify_wgsl_kernel can DIFFERENTIALLY TEST the two "
        "on real data (CuPy cannot: no shared source). Works on software adapters, so correctness is "
        "CI-testable with no GPU. SCOPE: elementwise f32 maps; a cross-invocation reduction is not solved",
        example="info = mind.wgsl_device(); mind.verify_wgsl_kernel(my_fn, data, extra_args=(2.0,))",
        native=True, aliases=("run this on any gpu", "use my amd or intel gpu", "gpu without cuda",
                              "run a kernel on metal or vulkan", "webgpu compute",
                              "check my shader matches the python", "vendor neutral gpu"))

    c.register_capability(
        "Spin up local worker processes (parallel execution)", "a PERSISTENT process pool -- each worker its "
        "own interpreter with its own GIL, so GIL-bound work actually runs in parallel on ONE machine, and a "
        "big read-only cache is published ONCE into shared_memory (zero-copy) instead of pickled per bucket. "
        "This is the one that CREATES workers; `farm` is the cross-machine sibling and only CONSUMES hosts "
        "you already started. Pass it as distribute_compute(backend=...). VERIFIED bit-identical to "
        "in-process. Workers must be TOP-LEVEL picklable functions. Default stays single-process -- measure "
        "on your own hardware first",
        example="pool = mind.local_pool(n=4); mind.distribute_compute(buckets, my_fn, backend=pool); pool.close()",
        native=True, aliases=("spin up another instance", "start a second worker", "use more cores",
                              "launch a local worker pool", "run work in parallel across processes",
                              "parallel execution on one machine", "balance load across instances",
                              "make it use all my cpus", "local process pool"))

    c.register_capability(
        "Use the GPU (optional CuPy backend, NVIDIA only)", "turn the optional CuPy backend on for the heavy "
        "array-parallel kernels (fluid, shader, deptrace, proc_texture, memoryhome -- 5 modules). Returns "
        "whether the GPU is now ACTIVE: requested AND a CUDA device present. Falls back to NumPy silently "
        "otherwise. SELECTIVE BY DESIGN -- a big FFT or matmul wins because the transfer amortises, a small "
        "per-vector op LOSES to the transfer. HONEST: this is CUDA/NVIDIA ONLY, and GPU matches NumPy only "
        "to a TOLERANCE, so the bit-exact and tie-sensitive paths stay on CPU. Throughput, not determinism",
        example="mind.use_gpu(True)   # -> False when no CUDA device is present",
        native=True, aliases=("use my gpu", "offload work to cuda", "run this on the graphics card",
                              "do i have a gpu", "enable cuda acceleration",
                              "make it faster with my graphics card", "use hardware acceleration",
                              "turn on the gpu", "gpu acceleration"))

    c.register_capability(
        "VSA cleanup on ANY GPU (matvec + argmax, fused)", "the codebook similarity is 98-100% of a cleanup's "
        "cost at any real M (the argmax is single-digit microseconds), so the SIMILARITY is what to offload. "
        "One workgroup per row, rows never communicate. Similarity and argmax FUSED in one dispatch -- "
        "splitting pays submission twice and ships the intermediate back. Index resolves host-side by lowest "
        "index (canonical tie rule). MEASURED RISK: a similarity gap <=1e-7 can flip (3/150); that is 4 orders "
        "below any sensible tie margin, so pair with tied_candidates",
        example="idx, sc = mind.wgsl_cleanup_batch(codebook, queries); mind.wgsl_matmul(codebook, queries)",
        native=True, aliases=("cleanup on the gpu", "matrix times vector on the gpu",
                              "codebook similarity on any gpu", "nearest atom on the graphics card",
                              "matvec on the gpu", "vsa recall on the gpu",
                              "clean up many cues at once", "batched cleanup on the gpu"))

    c.register_capability(
        "Walk on Decomposed Subdomains (short walks + exact solve)", "SHORT random walks estimate local "
        "coupling between interface points; the sparse system is then solved DETERMINISTICALLY by the shared "
        "conjugate gradient. Sampling does local coupling, exact linear algebra does the rest. MEASURED vs "
        "pure WoS at 32 walks: 0.043 vs 0.075 error (about HALF). KEPT NEGATIVES: BIASED by interface "
        "resolution, so unbiased WoS OVERTAKES at high budgets; the paper's low-variance headline does NOT "
        "reproduce -- this earns sample efficiency. 2-D rectangle + Dirichlet; use wost for general SDFs",
        example="pts = mind.wods_interface_grid(6, 6); mind.wods_solve(pts, g); mind.wods_measure_vs_pure_wos()",
        native=True, aliases=("split a domain into pieces and solve each one",
                              "estimate a local solution operator by random walks",
                              "combine local solvers into one global sparse system",
                              "monte carlo pde with fewer samples", "domain decomposition",
                              "subdomain solver", "cheaper grid free solve", "walk on decomposed subdomains",
                              "solve a pde with a tight sample budget"))

    c.register_capability(
        "Walsh-Hadamard transform (exact, matrix-free)", "the O(D log D) WHT, D a power of two: every butterfly is "
        "one add and one subtract -- no twiddles, no stored matrix, nothing to round. On INTEGER input it is "
        "BIT-EXACT and machine-independent, which numpy.fft is not (pocketfft's SIMD summation order is "
        "microarchitecture-dependent, NumPy #11926) -- and in this engine a ULP flip is an argmax flip. "
        "wht_exact refuses float so the guarantee is enforced. KEPT NEGATIVE, measured: 4-9x SLOWER than "
        "numpy.rfft at D=256..16384 -- it is an EXACTNESS tool, not an FFT speedup",
        example="mind.wht(x); mind.wht_exact(x); mind.wht_inverse(y); mind.wht_measure_vs_fft()",
        native=True, aliases=("fast walsh hadamard transform", "walsh hadamard", "hadamard transform",
                              "transform that uses only additions and subtractions",
                              "exact integer orthogonal transform", "matrix free transform",
                              "deterministic transform across cpus", "transform without rounding error",
                              "fwht", "wht", "sequency transform", "exact transform for integers",
                              "bit exact transform", "structured operator without a stored matrix"))

    # --- bundle RECOVERY: unmix a superposition (the four-member family) ---
    # WHY A CURATED HOME: all four members were wired mind faculties and auto-registered from their
    # docstrings, so they answered to their PAPERS' names (cosamp, iterative hard thresholding) and to
    # nothing else. Measured before this entry: 0/6 stranger phrasings surfaced any of them, 2/2
    # implementer names did. A research sweep duly read that hole as "ships but is not wired into
    # unbundling" and filed re-wiring them as an actionable item -- work that was already done. The
    # defect was the vocabulary, exactly as with mesh_box/camera above.
    c.register_capability(
        "What GPU do I have, and would offloading pay?", "use_gpu() returns a bare bool that conflates FOUR "
        "states -- no CuPy, CuPy but no device, a device the resource policy forbids, and enabled -- three "
        "of which the user can fix. gpu_report() separates them and covers BOTH paths (CuPy = NVIDIA-only "
        "and transparent; WGSL = vendor-neutral and explicit), because a CuPy-only report tells an Apple or "
        "AMD user they have no GPU. should_offload() is the pre-gate: refuses on no device, too little data, "
        "too little work per byte, or REPEATED ROUND TRIPS (fuse first). Thresholds PROVISIONAL, unmeasured",
        example="mind.gpu_report(); mind.should_offload(n_bytes=10**8, flops_per_byte=50.0)",
        native=True, aliases=("what gpu do i have", "is the gpu worth using here",
                              "should i offload this to the gpu", "why is my gpu not being used",
                              "check gpu availability", "is my graphics card being used"))

    c.register_capability(
        "Where should this work run (one placement oracle)", "three oracles answered three placement "
        "questions and none knew about the others -- machine_place_unit, should_pool, should_offload -- so a "
        "caller reconciled them by hand and NOTHING reconciled them with resource_policy: an oracle could "
        "recommend a device the operator had forbidden. This composes them. POLICY VETO FIRST (no arithmetic "
        "makes a banned device faster), then CHEAPEST-CORRECT: unit, pool, device -- the device last because "
        "it is the only one that changes the NUMBERS, not just the speed. Device answers are marked provisional",
        example="mind.place_work(n_buckets=64, est_ms_per_bucket=50.0, n_bytes=10**8, flops_per_byte=40.0)",
        native=True, aliases=("where should this work run", "should this go on the gpu or cpu",
                              "pick the best place to run this", "cpu pool or gpu",
                              "one answer for where to run", "placement decision"))

    c.register_capability(
        "qFHRR quantized phase (3-8 bits per dimension)", "store FHRR phasors as INTEGER phase indices "
        "instead of complex128: 4 bits/dim at 16 levels, a 96.9% cut, and bind/unbind become EXACT modular "
        "integer arithmetic -- unbind is a TRUE inverse returning the indices bit for bit, unlike the "
        "real-valued path's ~0.70 quasi-inverse. KEPT NEGATIVES: BUNDLING IS NOT CLOSED (it leaves the "
        "representation via atan2 + round, and that round is itself a tie), so this does NOT delete "
        "tie-arbitration; and bundle fidelity saturates at ~0.892 vs a complex bundle however fine the "
        "phase grid, because magnitude is discarded",
        example="q = mind.qfhrr_quantize(v); mind.qfhrr_bind(q, k); mind.qfhrr_unbind(c, k); mind.qfhrr_measure_fidelity()",
        native=True, aliases=("store a hypervector at three or four bits per dimension",
                              "quantize phase angles to integers", "bind by adding phase indices modulo k",
                              "shrink a codebook by quantizing", "low bit width vector representation",
                              "integer phase binding", "compress hypervectors", "quantized vsa",
                              "exact unbind", "fewer bits per dimension", "qfhrr", "quantized fhrr",
                              "shrink hypervector memory footprint"))


_PART = "holographic_catalog_p06"




def _selftest():
    """Delegates to holographic_catalog.check_catalog_part -- one home for the shared contract."""
    from holographic.caching_and_storage.holographic_catalog import check_catalog_part
    n = check_catalog_part(_PART, register_p06)
    print("%s selftest OK -- %d capabilities, no internal duplicates" % (_PART, n))


if __name__ == "__main__":
    _selftest()
