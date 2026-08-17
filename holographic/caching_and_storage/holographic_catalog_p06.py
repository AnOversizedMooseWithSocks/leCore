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
                          "encoding (taper='kaiser:beta' shapes similarity SIDELOBES by aperture-taper design -- "
                          "measured -13 -> -37.5 dB, weak-item margin 1.5x -> 18.2x beyond the mainlobe, price "
                          "2.7x mainlobe width -- redistribution not creation), N-D coordinate fields "
                          "(fpefield), complex-phasor FHRR (fhrr), sparse block codes (sbc), geometric-algebra Clifford "
                          "(clifford), and exact integer arithmetic over phasors (rns). How data ENTERS the substrate",
                          example="from holographic.io_and_interop.holographic_encoders import ScalarEncoder; from holographic.sampling_and_signal.holographic_fpe import ...",
                          native=True, aliases=("encode", "encoder", "number to vector", "scalar encoding",
                                                "fractional power encoding", "fpe", "encode coordinates", "phasor", "fhrr",
                                                "sparse block codes", "sbc", "clifford", "geometric algebra",
                                                "exact integer arithmetic", "rns", "embed a value",
                                                "suppress similarity sidelobes", "kernel taper",
                                                "weak item buried under strong", "phased array kernel"))
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


    # --- ORGANICS (crystals / grass / plants / growth scrubbing / creature idle). The aliases here were
    # written by testing stranger phrasings against the LIVE index first: "cubic lattice", "salt crystal
    # structure", "put grass on my terrain mesh", "cover a surface in plants", "make a bush", "vegetation
    # generator", "watch it grow step by step" and "make the creature move a little" all returned unrelated
    # fallbacks before these entries existed. Each alias below is a phrasing that MEASURABLY failed, not one
    # that sounded plausible to the implementer.
    c.register_capability(
        "Crystal lattices (the 14 Bravais lattices)", "atom SITE positions for any of the 14 Bravais lattices "
        "(7 crystal systems x P/I/F/C centring), their nearest-neighbour bonds, and a faceted habit SDF from "
        "Miller-index planes; feed sites to metaball_mesh for ball-and-stick or scatter_mesh for unit cells",
        example="pts = mind.lattice_sites('cubic', centring='F', extent=2); bonds, d = mind.lattice_bonds(pts)",
        native=True, aliases=("cubic lattice", "salt crystal structure", "how do crystals stack",
                              "bravais lattice", "crystal unit cell", "make a crystal", "atom positions",
                              "face centred cubic", "body centred cubic", "diamond lattice",
                              "seven crystal systems", "crystal facets", "mineral shape", "gemstone shape"))

    c.register_capability(
        "Scatter meshes over a surface (grass, rocks, plants)", "place many copies of a mesh across the AREA of "
        "another mesh -- area-weighted sampling, normal-aligned frames with yaw/scale jitter, optional blue-noise "
        "thinning and a density mask; merge into one mesh or share one definition across n instances",
        example="lawn = mind.scatter_mesh(ground, mind.grass_blade(), 500, seed=0, mode='merge')",
        native=True, aliases=("put grass on my terrain mesh", "cover a surface in plants", "lawn", "grass field",
                              "scatter grass", "instance a model many times over a mesh", "fur on a mesh",
                              "rocks on a hill", "barnacles on a hull", "sprinkle objects on a surface",
                              "distribute meshes over an area", "populate a surface", "foliage scatter"))

    c.register_capability(
        "Procedural plants & trees (L-system grammar)", "grow branching plants and trees from rewrite rules: "
        "expand an L-system, walk it with a 3-D turtle into branch segments, then mesh it as tapered limbs; also "
        "greebles, seeded procedural objects and terrain vegetation",
        example="ls = mind.lsystem('F', {'F': 'F[+F]F[-F]F'}); mesh, segs, scene = mind.grow_plant(ls, 3)",
        native=True, aliases=("branching plant generator", "make a bush", "vegetation generator", "procedural tree",
                              "grow a tree from rules", "l-system", "turtle graphics", "foliage generation",
                              "tree generator", "shrub", "fern", "plant model", "branch structure"))

    c.register_capability(
        "Scrub through growth (staged growers + verification)", "step a plant, crystal or dendrite from nothing to "
        "finished form: discrete stages or a continuous t in [0,1]. grow_at is PURE, so scrubbing backwards is safe; "
        "growth_report checks purity and that nothing retracts between stages. t is growth PROGRESS, not time",
        example="stages = mind.grow_stages('crystal', {'system': 'cubic', 'centring': 'F'}, 8); "
                "rep = mind.growth_report('crystal', {'system': 'cubic', 'centring': 'F'})",
        native=True, aliases=("watch it grow step by step", "scrub through growth", "growth stages",
                             "step through crystal formation", "growth preview", "intermediate growth stages",
                             "time lapse of growth", "grow over time", "check the growth is right",
                             "verify growth order", "replay growth"))

    c.register_capability(
        "Creature idle animation (show where the joints bend)", "a simple looping idle that flexes each joint within "
        "its OWN stored limit, so knees/elbows/hips visibly show where they are and which way they bend -- the limit "
        "is the driver, so an impossible bend cannot be shown. A limits demo, NOT locomotion (no gait or balance)",
        example="c = mind.creature(mind.quadruped_spec()); frames = mind.creature_idle_frames(c, 24)",
        native=True, aliases=("make the creature move a little", "show me where the joints bend",
                              "idle animation", "creature idle", "which way does the elbow bend",
                              "animate a creature", "simple animation for a monster", "joint limit preview",
                              "test my rig", "does the knee bend the right way", "breathing idle"))


    # --- ORGANICS, second pass (trees by space colonization, Spore-style creature skin + spine edits).
    # Same discipline: these aliases are phrasings that MEASURABLY returned unrelated fallbacks before
    # the entry existed -- "leaves on a tree" returned a code-shape capability, "oak tree" returned
    # file_tree, "make the belly fatter" returned NOTHING, "fat torso thin neck" returned a cache.
    c.register_capability(
        "Trees by space colonization (branches, taper, leaves)", "grow a tree by racing branches into an "
        "attractor crown (Runions 2007) -- natural limb spread and a shaped canopy; thickness by the da Vinci "
        "rule (parent^2 = sum of children^2); leaves placed at the golden angle. Segments come out in growth "
        "order, so scrubbing is free",
        example="A = mind.crown_attractors(n=250); t = mind.grow_tree(A); mesh = mind.tree_mesh(t)",
        native=True, aliases=("leaves on a tree", "oak tree", "how thick should a branch be",
                              "tree branches thickness", "grow a realistic tree", "space colonization",
                              "branch taper", "canopy", "conifer", "foliage on branches",
                              "realistic tree generator", "limb growth", "crown shape"))

    c.register_capability(
        "Spore-style creature skin & spine sculpting", "skin a creature as a chain of METABALLS spaced from "
        "their own radius, so stretching a spine or limb adds balls instead of stretching a shape, and limbs "
        "blend into the torso; per-node spine radii give a fat belly and a thin neck, and the spine can be "
        "extended, subdivided, re-thickened and reshaped -- every edit returning a new spec",
        example="s = mind.spine_profile(mind.quadruped_spec(), [.06,.16,.2,.16,.06]); "
                "mesh = mind.creature_skin_mesh(mind.creature(s, skin=False), s)",
        native=True, aliases=("make the belly fatter", "fat torso thin neck", "sculpt the torso",
                              "spore style body", "blobby creature skin", "stretch the spine",
                              "adjust thickness around the spine", "add more spine segments",
                              "thicken the body", "smooth skin over bones", "metaball body",
                              "edit the backbone", "make a creature rounder"))


    # --- ORGANICS, third pass: the HOLOGRAPHIC half (parts as a bound record, symmetry groups, skin
    # weights as a soft mixture, scatter layers as content-addressable bundles, generated variants).
    # Aliases again from measured misses: "snap parts onto a creature" returned shrinkwrap, "library of
    # body parts" returned Water body, "which bone controls this vertex" returned a cross field,
    # "is anything scattered here" returned Cloud stack, "five fins around the body" returned a cache.
    c.register_capability(
        "Creature parts, sockets & symmetry (holographic assembly)", "a rigblock LIBRARY whose parts are codebook "
        "atoms with authored deformation handles; attaching to sockets builds ONE bound record, so the layout is "
        "queryable (what is on this socket), comparable between creatures, and mirrored or radially repeated by a "
        "symmetry GROUP. assembly_report measures the bundle's recall margin rather than assuming capacity",
        example="lib = mind.part_library(); lib.define('horn', handles={'length': (0.5, 2.0)}); "
                "a, v = mind.attach_part({}, 'shoulder', 'horn', lib, symmetry='bilateral')",
        native=True, aliases=("snap parts onto a creature", "library of body parts", "rigblock",
                              "what part is on the left shoulder", "mirror parts to both sides",
                              "five fins around the body", "radial symmetry", "attach horns",
                              "body part library", "symmetry group", "creature accessories",
                              "put an eye on the head"))

    c.register_capability(
        "Skin weights from metaball provenance (soft mixture over bones)", "per-vertex bone weights read off which "
        "bones produced the metaballs near each vertex -- skinning as the soft mixture of experts it already is. "
        "Returns the compact indexed form linear_blend_skin_indexed wants. Distance-based, not geodesic: touching "
        "limbs bleed weight",
        example="C, R, bones = mind.creature_metaballs(cr); idx, w, names, book = "
                "mind.skin_weights_from_balls(mesh.vertices, C, R, bones)",
        native=True, aliases=("which bone controls this vertex", "automatic skin weights",
                              "bind vertices to bones", "weight painting", "rig a generated creature",
                              "bone influence per vertex", "skinning weights"))

    c.register_capability(
        "Content-addressable scatter layers & generated variants", "bundle every placement of a scattered field "
        "into ONE vector bound by region code, so you can ask whether anything is scattered near a point without a "
        "spatial index; plus deterministic spec VARIANTS, so a pool of twenty different plants is a loop over seeds "
        "rather than twenty authored assets, and never needs storing",
        example="g = mind.scatter_mesh(ground, mind.grass_blade(), 500, holographic=True); "
                "mind.region_occupancy(g['layer'], g['instance'], g['transforms'][0, :3, 3])",
        native=True, aliases=("is anything scattered here", "query the grass field", "twenty different ferns",
                              "random variations of a plant", "what is scattered near this point",
                              "plant permutations", "vary a plant", "scattered field query",
                              "grass blades from hair strands", "ribbon strands"))


    # --- ORGANICS, final pass (paint mode + scatter bake/LOD). Measured misses again: "skin markings"
    # returned pose_asset, "colour the body" returned Water body, "level of detail for grass" returned
    # Gabor volumes, "thin distant grass" returned iridescent thin-film, "cheaper geometry far away"
    # returned reprojection velocity.
    c.register_capability(
        "Paint a creature (markings bound to the rig)", "procedural per-vertex colour mixing a BONE tint with a "
        "pattern (stripes/dots/checker/noise): because the tint reads the skin weights, markings follow the "
        "anatomy and travel with a pose instead of swimming through world-space noise, and a limb is its own "
        "colour region for free. Per-vertex -- finer detail needs a texture bake",
        example="idx, w, names, book = mind.skin_weights_from_balls(V, C, R, bones); "
                "cols = mind.paint_creature(V, idx, w, names, pattern='stripes')",
        native=True, aliases=("skin markings", "colour the body", "stripes on an animal",
                              "paint a creature", "creature colours", "fur pattern", "spots and stripes",
                              "colour a monster", "texture a creature", "animal markings"))

    c.register_capability(
        "Scatter bake & level of detail (measured)", "bake a scattered population once, then serve any distance "
        "from the cache: thin the population and drop to a coarser blade as it recedes. Thinning is deterministic "
        "and NESTED, so the far set is a subset of the near set and blades never flicker as the camera moves. "
        "Reports exact triangle counts against the full-resolution baseline",
        example="b = mind.scatter_bake(g['transforms'], mind.grass_blade()); print(b.report())",
        native=True, aliases=("level of detail for grass", "thin distant grass", "cheaper geometry far away",
                              "cache a scatter", "lod for scattered objects", "reduce grass in the distance",
                              "bake the grass", "scatter performance", "too many blades"))


    # --- ORGANIC CREATURE MATERIALS. Measured misses before registration: 'chitin' and 'exoskeleton'
    # returned NOTHING at all; 'scales for a lizard' returned Fourier-Mellin; 'frog skin' and 'eel skin'
    # returned pose_asset; 'beetle shell' returned the soap-bubble thin-film.
    c.register_capability(
        "Creature skin materials (scales, chitin, mucus, fur)", "integument by TAXON -- reptile/fish scales, "
        "amphibian glands, insect chitin plates, worm annuli, mammal pores -- as channel fields evaluated in the "
        "creature's own BODY frame, so scale rows elongate down the body and travel with it. Feeds render_surface "
        "as a solid 3-D texture: no UV unwrap, no seams",
        example="mat = mind.creature_surface_material('reptile', axis=(0,0,1)); "
                "img = mind.render_surface(field, cam, 256, 256, {0: mat})",
        native=True, aliases=("scales for a lizard", "frog skin", "eel skin", "beetle shell", "chitin",
                              "fish scales", "snake skin", "reptile skin", "amphibian skin",
                              "insect shell", "worm skin", "mammal skin", "wet slimy skin",
                              "scaly texture", "creature skin"))

    c.register_capability(
        "Layered creature anatomy (bone, organ, dermis, epidermis, coat)", "an integument as the stack of tissues it "
        "really is, composed through the layered-material order schema so base<diffuse<specular<coat is enforced at "
        "compose time. INSECTS REFUSE A BONE LAYER -- an arthropod's rigid structure is its exoskeleton. Interior "
        "layers only show through translucency; this is not an x-ray",
        example="st = mind.anatomy_stack('reptile'); print(st['layers'], st['interior_visible'])",
        native=True, aliases=("bone and organ layers", "layered skin", "exoskeleton", "endoskeleton",
                              "what is my creature made of", "skin layers", "tissue stack",
                              "anatomy material", "dermis and epidermis", "organ material",
                              "bone material", "invertebrate"))


    # --- THE CREATURE EDITOR LOOP. Measured misses: "undo my last change" returned file_undo, "save a
    # creature to a file" returned a tensor-train container, "where did i click on the model" returned a
    # cache, "stick a horn on the back" returned a decomposition contract, "dna points" returned
    # vanishing points, "is my creature valid" / "how complex is my creature" returned the idle animation.
    c.register_capability(
        "Creature editor session (undo, save, validate, build)", "the object an app drives: holds the creature "
        "document, records every change so it can be undone and redone, saves and loads as JSON that round-trips "
        "exactly, validates that the thing is buildable, meters a Spore-style complexity budget, and builds skin "
        "plus placed parts. Edits return self, so a UI can chain and still undo each step",
        example="ed = mind.creature_editor(); ed.extend_spine(2).set_thickness(0.5, 0.22, falloff=0.3); "
                "ed.undo(); print(ed.validate(), ed.complexity())",
        native=True, aliases=("spore creature creator", "creature editor", "undo my last change",
                              "save a creature to a file", "load a creature", "is my creature valid",
                              "how complex is my creature", "dna points", "creature document",
                              "edit a creature", "creature save format", "redo", "creature budget"))

    c.register_capability(
        "Creature sockets: where a part lands on the body", "attach points in ANATOMY space -- (t along the spine, "
        "theta around it) -- resolved by marching the creature's own skin field outward. Because a socket stores "
        "anatomy coordinates rather than a world position, a part rides the skin through every later spine and "
        "thickness edit. Includes the inverse (click point -> socket) and a viewport ray pick, which round-trip",
        example="cr, f = ed.creature(), ed.field(); s = mind.pick_socket(cr, f, eye, ray); "
                "ed.add_part('horn', s['t'], s['theta'], symmetry='bilateral')",
        native=True, aliases=("click to place a part", "where did i click on the model",
                              "stick a horn on the back", "drag a part onto the body",
                              "attach a part to a creature", "part placement", "snap a part to the surface",
                              "socket position", "pick a point on a creature", "put horns on it"))


    # --- THE PARAMETRIC PART LIBRARY. Measured misses: "give it eyes" returned scene-from-description,
    # "add a mouth" returned encyclopedia_add, "three toed foot" returned Arrow of time, "how many
    # fingers" returned batched cleanup, "tapered tube" returned sweep_tube (which cannot taper).
    c.register_capability(
        "Creature body parts (eyes, mouths, feet, claws, fins)", "eleven PARAMETRIC body parts -- eye (optionally "
        "on a stalk), mouth, foot and hand with a variable number of digits, claw, horn, spike, fin, antenna, ear "
        "-- each with authored handle ranges that clamp. Procedural, so digits=3 versus digits=5 is a genuinely "
        "different foot rather than one mesh scaled, which is what a rigblock handle is supposed to mean",
        example="lib = mind.creature_parts(); ed = mind.creature_editor(part_library=lib); "
                "ed.add_part('eye', 0.95, 0.7, symmetry='bilateral')",
        native=True, aliases=("give it eyes", "add a mouth", "three toed foot", "how many fingers",
                              "make a claw", "wings or fins", "antennae", "body part library",
                              "what parts can i use", "horns", "eyeball", "hand with fingers",
                              "foot with toes", "part palette"))

    c.register_capability(
        "Tapered sweep (varying-radius tube along a path)", "sweep a circular cross-section whose RADIUS varies "
        "along the path, in a rotation-minimizing frame so the ring does not spin on a curve. The shipped "
        "sweep_tube takes one profile for the whole tube and so cannot taper -- which is the one thing every "
        "organic appendage does, and why every part in the creature library is built on this",
        example="import numpy as np; P = np.stack([np.zeros(6), np.zeros(6), np.linspace(0, 1, 6)], 1); "
                "m = mind.sweep_profile(P, np.linspace(0.1, 0.01, 6))",
        native=True, aliases=("tapered tube", "varying radius sweep", "cone along a curve",
                              "taper a sweep", "swept profile", "tapered extrusion"))


    # --- GAIT. Measured misses: "trot gallop" and "moonwalk" returned NOTHING; "make it walk" and
    # "does my creature walk properly" collided with the Walk-on-Stars PDE solver; "locomotion"
    # returned the IDLE animation, which is explicitly not locomotion; "stride length" returned
    # packet_demux.
    c.register_capability(
        "Creature gait (make it walk, any body plan)", "locomotion for a generated creature: legs are found by "
        "MEASUREMENT (which limbs reach the ground), stride comes from their reach, and phase offsets come from "
        "the classic gait diagrams -- walk, trot, pace, bound, gallop for tetrapods, a metachronal wave for any "
        "other leg count. Body speed is DERIVED from stride/(duty*period), never a free input, which is what "
        "keeps planted feet from sliding. Legs are posed through the rig's own limit-constrained IK",
        example="rep = mind.gait_report(cr, gait='trot'); frames = mind.gait_frames(cr, n_frames=24)",
        native=True, aliases=("make it walk", "trot gallop", "locomotion", "walk cycle", "gait",
                              "animate walking", "step cycle", "stride length", "which limbs are legs",
                              "quadruped walk", "biped walk", "hexapod walk", "run cycle"))

    c.register_capability(
        "Foot slip (is this walk actually correct?)", "the objective measure of a walk: how far a PLANTED foot "
        "slides in world space, absolutely and as a fraction of stride. Sliding feet are the moonwalk artifact, "
        "and this is a number rather than a matter of taste. Also reports distance travelled, measured duty per "
        "foot, and any leg the IK could not place. KEPT NEGATIVE: a foot that never moves cannot slip, so a low "
        "score means nothing unless `unreachable` is empty -- check it first",
        example="r = mind.gait_report(cr); print(r['slip_ratio'], r['unreachable'], r['duty_measured'])",
        native=True, aliases=("foot slip", "moonwalk", "does my creature walk properly",
                              "feet sliding", "walk quality", "sliding feet", "verify a walk cycle",
                              "gait measurement", "duty factor"))


    # --- BODY SHAPING / MESH QUALITY / PART FUSION. Measured misses: "barrel chested" returned
    # NOTHING; "flat belly" returned flat_recall; "why is my creature lumpy" returned a navigator;
    # "beaded mesh" returned closest-point; "make the horn look attached" returned a message bus.
    c.register_capability(
        "Creature body shape (non-circular cross-section)", "metaballs are spheres, so every cross section of a "
        "creature is a CIRCLE and no profile edit changes that -- a fat belly is still a round belly. This warps "
        "SPACE in the body frame instead of using ellipsoid primitives, so every ball becomes the same ellipse for "
        "free: broaden across, flatten front-to-back, raise a spinal crest, flatten the underside. Measured: no "
        "evaluation cost over spheres",
        example="w = mind.section_warp(cr, width=1.5, depth=0.75, ridge=0.3, belly=0.35); "
                "f = mind.creature_skin_field(cr, spec, warp=w)",
        native=True, aliases=("make the body wider than deep", "flat belly", "barrel chested",
                              "body shape not round", "squash the body", "body silhouette",
                              "cross section shape", "slab sided", "deep chested", "spinal ridge"))

    c.register_capability(
        "Mesh quality guard (why is my creature lumpy?)", "how many marching cells span the THINNEST feature, and "
        "the resolution that would fix it. A feature needs at least 4 cells to mesh smoothly; below 2 it BEADS "
        "into visible lumps -- which is what a thin limb does inside the bounding box of a much larger body, "
        "because the cell size is set by the whole body and the limb gets no say",
        example="q = mind.skin_quality(cr, spec, resolution=104); print(q['verdict'], q['recommended_resolution'])",
        native=True, aliases=("why is my creature lumpy", "beaded mesh", "what resolution do i need",
                              "lumpy limbs", "bumpy surface", "marching resolution", "sausage limbs",
                              "mesh looks bad"))

    c.register_capability(
        "Fuse parts into the skin (attached, not glued)", "parts become metaballs in the SAME implicit surface as "
        "the body, so a horn is one continuous mesh with a smooth fillet where it meets the flank instead of a "
        "cone resting against it. Only tapered-tube parts survive being reduced to a ball chain; fins, hands and "
        "mouths would become blobs and are returned for placing as geometry instead",
        example="fld, fused, unfused = mind.fused_field(cr, spec, spec['sockets'], lib)",
        native=True, aliases=("make the horn look attached", "parts blended into the body",
                              "fuse a part", "seamless parts", "parts look glued on",
                              "blend a horn into the skin", "continuous skin"))


    # --- LIMB SOCKETS / AUTO FEET / CEL SHADING. Measured misses: "flat cartoon look" returned
    # conditional statistics, "outline the creature" returned the parts list, "put feet on the legs"
    # returned the gait.
    c.register_capability(
        "Feet on the legs (limb sockets, not spine sockets)", "attach points on a LIMB -- at a fraction along it, "
        "angle around it, and optionally down its own AXIS, which is what a foot needs because a foot goes on the "
        "END of a leg rather than its side. auto_feet identifies legs the way the gait does (which limbs reach the "
        "ground) and sockets a foot on each, so an arm correctly gets none",
        example="feet = mind.auto_feet(cr, ed.field(), part='foot', scale=1.2); "
                "ed.spec['sockets'].extend(feet)",
        native=True, aliases=("put feet on the legs", "attach a foot", "socket on a limb",
                              "limb tip socket", "hands on the arms", "part on a leg",
                              "knee spur", "foot placement on a creature"))

    c.register_capability(
        "Cel shading (flat cartoon bands + silhouette)", "quantise the light into flat BANDS and darken the "
        "silhouette, per vertex, so it composes with vertex paint and the existing rasteriser instead of needing a "
        "new render path. The silhouette comes from GEOMETRY (the surface turning away from the eye) rather than "
        "from filtering the image, which would trace colour edges on a flat belly just as happily. Darkens the "
        "silhouette only -- interior creases need an image pass with depth and ids",
        example="cols = mind.toon_shade(mesh, cols, cam['eye'], bands=3, rim=0.42); "
                "img = mind.render_mesh(mesh, cam, vertex_colors=cols, lights=[], ambient=1.0)",
        native=True, aliases=("cel shading", "toon shading", "flat cartoon look", "outline the creature",
                              "posterize shading", "non photorealistic render", "comic book look",
                              "quantize shading bands", "cartoon render", "rim darkening"))


    c.register_capability("Tiered memory (adaptive short-term / long-term with promotion & demotion)",
        "mind.tiered_memory(hot_capacity=K) is the ST/LT conductor over existing levers: a bounded EXACT hot "
        "dict (O(1), zero loss -- low overhead for what matters), and demoted items in a CONSTANT-size "
        "superposed trace plus zlib-compressed exact spill (low disk/RAM for what doesn't). Demotion picks the "
        "lowest importance = recency-decay x (1+hits), with a recency-window veto (kept negative: pure "
        "frequency ordering starved every new item, twice). LT access verifies trace vs spill, then PROMOTES "
        "back to hot. get() returns (value, tier).",
        example="tm=mind.tiered_memory(hot_capacity=4); [tm.put(k,(k*7)%256) for k in range(9)]; "
                "print(tm.get(0), tm.stats())",
        native=True, aliases=("short term and long term memory", "adaptive memory tiers",
                              "consolidate short term into long term", "promote important memories",
                              "demote stale memories", "move memories between tiers",
                              "low overhead for what matters", "spend less disk on unimportant data",
                              "working memory with archive", "importance based eviction",
                              "hot and cold memory", "memory that forgets gracefully",
                              # the value-head move applied to the POLICY itself (policy='holo'):
                              "cache policy as a hypervector", "importance as a bundle readout",
                              "eviction decided inside the vsa", "holographic cache policy",
                              # Quilez-seat persistence: the trace is a derived view, save the rule
                              "save memory as the rule not the bytes",
                              "persist a cache and regenerate its trace",
                              "constant size save for tiered memory"))

    c.register_capability("Celled memory (domain repetition over the capacity law -- unbounded pairs, bounded cells)",
        "mind.celled_memory() escapes the capacity wall the HONEST way: cells of EXACTLY n* pairs (the "
        "measured limit IS the tile size -- Quilez opRep applied to memory), one shared seed-derived "
        "codebook, warm/cold cell tiers with the crossing cost measured, exact key->cell directory. "
        "MEASURED on real corpus pairs at dim 4096: ONE memory 70x past the law recalls at 0.007 "
        "(interference collapse, as the law predicts); celled recalls 1.000 across 71 cells. Kept "
        "negative: a holographic directory would re-buy the interference the cells escape.",
        example="cm=mind.celled_memory(dim=2048, vocab=4096); import numpy as np; "
                "ks=np.arange(500); cm.store(ks,(ks*7)%4096); print((cm.recall(ks)==(ks*7)%4096).mean(), cm.stats())",
        native=True, aliases=("store more pairs than the capacity law allows", "escape the capacity limit",
                              "unbounded associative memory", "tile memory into cells",
                              "domain repetition for memory", "memory beyond the interference wall",
                              "millions of key value pairs holographically", "scale superposed memory"))

    c.register_capability("Learn this codebase (the map, the menu, and the method)",
        "Reading order for new eyes, human or AI: (1) docs/ARCHITECTURE.md -- the whole system then the "
        "parts; (2) CAPABILITIES.md -- the auto-generated menu of every capability with runnable examples "
        "(this very catalog, exported); (3) tools/showcase.py -- the flagship claims as live assertions. "
        "THE METHOD: it is often easier to use leCore to learn leCore -- find_capability/suggest/route ARE "
        "semantic search over this catalog and beat grep for 'where does X live'. llms.txt/AGENTS.md carry "
        "the same guidance for AI assistants landing on the repo.",
        example="print(open('docs/ARCHITECTURE.md').read()[:400])",
        native=True, aliases=("how do I learn this codebase", "where do I start", "reading order",
                              "explain the architecture", "how is this organized", "onboarding",
                              "documentation entry point", "map of the project"))

    c.register_capability("Routed roles (the semantic system staffs the swarm)",
        "mind.dispatch_roles(tasks, spec): task phrases ('leave a map of the target', 'move along the "
        "shared map', 'adjust the texture gains') route to registry roles (scout/mover/texturer) via "
        "the engine's OWN BM25 -- leCore staffing leCore; nobody hand-builds member stacks. Builders "
        "close over spec (targets, steps, channels), so dispatch COMPOSES. AMBIGUITY IS AN ERROR: no "
        "match or two tasks claiming one role raises WITH NAMES -- silent misstaffing is a ghost. "
        "Pinned end-to-end: routed members converge in the workspace loop.",
        example="import lecore, numpy as np; m=lecore.UnifiedMind(); [r for r,_ in m.dispatch_roles(['leave a map of the target direction','adjust the texture gains'], {'target_params': np.ones(3)})]",
        native=True, aliases=("route tasks to swarm roles", "staff the swarm", "assign agent roles",
                              "texture the scene routes to texturer", "role dispatch"))

    c.register_capability("Shared workspace for swarm roles (coordinate through slots, not chatter)",
        "mind.shared_workspace() + render_critique_loop(workspace=): named slots the roles read and "
        "write while deliberating -- the designer leaves the layout, the texturer reads it and leaves "
        "gains. Writes BUFFER within a round and commit together (even on no-improvement rounds: a "
        "scout that only leaves a map IS the round's progress -- the first pin run proved bootstrap "
        "dies otherwise); collisions resolve to the LOWEST member index and are LOGGED, never silent. "
        "Pinned: coordination is LOAD-BEARING (the mover fails without the scout's slot).",
        example="import lecore; m=lecore.UnifiedMind(); ws=m.shared_workspace(); ws.write(0,'layout',[1,2]); ws.commit(1); ws.read('layout')",
        native=True, aliases=("shared workspace between agents", "swarm scratchpad", "roles coordinate",
                              "blackboard for the swarm", "agents share scene state"))

    c.register_capability("The inner eye's 2D toolset (image ops as installed chain steps)",
        "mind.image_op_library(h, w): the classic editing bench as FAC-ready callables, verdicts "
        "MEASURED AT IMAGE SCALE (probe scale= names the certification DOMAIN -- at unit scale a "
        "threshold certified linear on the zero function): blur/unsharp/sobel certify, flip/rot90/warp "
        "are PERMUTATIONS (D ints), brightness/contrast install; threshold/gamma REFUSE and ride "
        "HOST:APPLY. Chains track state dim across rectangular steps. Compose with "
        "render_critique_loop: the eye can look at ANY pipeline's output.",
        example="import lecore; m=lecore.UnifiedMind(); lib=m.image_op_library(4,4); import numpy as np; sorted(lib.keys())[:5]",
        native=True, aliases=("blur inside the weights", "image pipeline installed", "2d editing in the model",
                              "installed image filters", "which image ops install", "flip is a permutation"))

    c.register_capability("The inner eye (render, look, iterate, THEN speak the picture)",
        "mind.render_critique_loop: swarm-role members propose scene params, an INSTALLED chain "
        "renders, the frame goes through the model's OWN vision (eye is injectable: the assimilated "
        "Qwen3.5-VL tower on the host; ReferenceEye in CI -- the seam IS the honesty), a critic scores "
        "in EYE SPACE (kept negative: pixel-space critics reward changes the eye cannot see -- pinned "
        "with a checkerboard the eye pools away), loop until satisfied, emit PGM through the mouth. "
        "Deterministic: same intent, same picture, every run. Stalls stop honestly.",
        example="import numpy as np; from holographic.agents_and_reasoning.holographic_machine import HoloMachine; from holographic.agents_and_reasoning.holographic_innereye import ReferenceEye; import lecore; m=lecore.UnifiedMind(); Wf=np.abs(np.random.default_rng(0).standard_normal((16,2)))*50; eye=ReferenceEye(4,4,embed_dim=8,patch=2); mm=HoloMachine(dim=2,seed=9,data=['a']); mm.functions_symbolic={}; pgm,rep=m.render_critique_loop(mm,[('FAC',('f',lambda p: Wf@p)),('HALT',None)],np.zeros(2),[('d',lambda p,s,r: p+0.1)],eye,eye(Wf@np.array([0.6,0.6])),4,4,satisfy=0.99,max_rounds=20); rep['satisfied']",
        native=True, aliases=("look at a render before outputting", "inner eye loop",
                              "render critique iterate", "model looks at its own render",
                              "design render look loop", "swarm renders and inspects"))

    c.register_capability("The installed generative model (HDRIFT head: model == one certified matrix)",
        "mind.drift_head(model): a drift generative model's readout is its (d+1) x D moment matrix "
        "[mu; nu_j] -- certified DENSE at 0.0, so the model ships as ONE weight matrix. MODEL "
        "ARITHMETIC IN WEIGHT SPACE, exact: head(A)+head(B) == head(compose(A,B)) at 0.0; subtract == "
        "ablate; transport == a certified linear action on rows (3.6e-16). drift_head_load inverts "
        "(field bit-identical). HONEST BOUNDARY: the sampling recurrence is nonlinear -- the projector "
        "refuses it (residual 8e-2); enc = host-feature lane, generation stays host-shape.",
        example="import numpy as np, lecore; from holographic.sampling_and_signal.holographic_hdrift import DriftModel, drift_moments, drift_compose; from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder; m=lecore.UnifiedMind(); r=np.random.default_rng(0); e=VectorFunctionEncoder(2, dim=512, bounds=[(-3,3),(-3,3)], bandwidth=6.0, seed=1); A=DriftModel(e, *drift_moments(r.standard_normal((80,2))*0.3, e), 80); B=DriftModel(e, *drift_moments(r.standard_normal((80,2))*0.3+1.0, e), 80); float(np.max(np.abs(m.drift_head(drift_compose(A,B)) - (m.drift_head(A)+m.drift_head(B)))))",
        native=True, aliases=("add two generative models", "model arithmetic in weight space",
                              "install the drift head", "merge distributions by adding weights",
                              "generative model as a matrix", "ship the model as weights"))

    c.register_capability("VSA load-bearing audit (the ablation table)",
        "mind.ablation_table(seeds=...): for each subsystem, run the DUMBEST honest non-"
        "holographic baseline on the SAME task, data, and metric; measure both across seeds with "
        "the variance harness; confidence intervals decide the verdict -- load-bearing (holo lower "
        "CI above baseline upper), decorative (baseline wins), or tie. The honest answer to 'where "
        "is VSA actually the reason it works', system-wide. FDR-corrected verdicts included.",
        example="import holographic.misc.holographic_ablate as ab; ab.verdict({'mean': 0.9, 'ci': (0.88, 0.92)}, {'mean': 0.5, 'ci': (0.48, 0.52)})['verdict']",
        native=True, aliases=("is vsa load bearing here", "ablation table", "honest baseline comparison",
                              "which subsystems need vsa", "vsa vs simple baseline"))

    c.register_capability("Roles as powers of one shift (the affordable role machine)",
        "mind.roles_by_shift(pairs, dim=): encode role-filler pairs where role k IS the k-th power "
        "of ONE cyclic shift -- the oldest VSA trick, and the fix that made the in-weights role "
        "machine affordable (one permutation instead of one circulant PER role: the circulant "
        "design wanted 228 percent of a 3584-wide MLP for eight roles). Roles are INTEGERS (shift "
        "counts); decode via holographic_vsaroles.decode_structure; capacity() measures the load "
        "law. The origin design behind the weight installs.",
        example="import numpy as np, lecore; m = lecore.UnifiedMind(dim=64, seed=0); m.roles_by_shift([(0, np.ones(32)), (1, 0.5 * np.ones(32))], dim=32).shape == (32,)",
        native=True, aliases=("roles as shifts", "role filler machine", "cheap role binding",
                              "powers of one operator", "affordable roles in weights"))

    c.register_capability("The thesis (one data type, many costumes -- why none of this is junk)",
        "docs/THE_THESIS.md: for visitors who see 600 modules and conclude bloat. Everything -- data "
        "AND functionality -- is a hypervector or an operator on them, one algebra; modules MULTIPLY. "
        "The junk test w/ receipts: cleanup IS a denoiser (24/24 at half-brain); IK/PBD/PnP/resonator "
        "= one solver (rig CCD 8e-17 rad); mesh subdivision ran on symbol sequences; a mince is "
        "block_shuffle; sphere tracing became a certified retrieval bound. Plus the discipline that "
        "keeps sprawl honest, and a ten-minute skeptic tour.",
        example="import pathlib; t = pathlib.Path('docs/THE_THESIS.md').read_text(); 'one algebra wearing 600 costumes' in t",
        native=True, aliases=("is this junk", "why is this codebase so big", "unrelated modules",
                              "what is the unifying idea", "why hypervectors for everything",
                              "the thesis", "one data type many costumes", "why should I care about vsa"))

    c.register_capability("Precision ladder (certified int8 rung: exact answers at quantized speed)",
        "Index(method='int8') and the auto ladder: row-scaled int8 scan (numba OPT-IN kernel; "
        "absent numba the route does not exist) with a SPECTRUM-IMMUNE certified dot-error bound "
        "(s_r/2)|q|1 + (qs/2)|x|1 + (s_r qs/4)D -- conservative candidates PROVABLY contain every "
        "true top-k row incl ties; f64 rescore; near-tie storms fall to exact. THE BENCHMARK: "
        "100k x768 hard: recall 1.000 @ 9.7 ms (FAISS Flat exact: 27.1); 1M x128: 1.000 @ 34.8 ms "
        "(only exactness in the table). Whitened data killed dimension-domain bounds twice; "
        "PRECISION-domain lifting is the lever the spectrum cannot touch.",
        example="import numpy as np; from holographic.caching_and_storage.holographic_index import Index; X=np.random.default_rng(0).standard_normal((2000,64)); i8=Index(X, method='int8'); ex=Index(X, method='exact'); q=X[3]+0.05*np.random.default_rng(1).standard_normal(64); [i for i,_ in i8.nearest(q,k=8)] == [i for i,_ in ex.nearest(q,k=8)]",
        native=True, aliases=("int8 index", "quantized exact search", "precision ladder",
                              "certified quantized scan", "exact recall at quantized speed"))

    c.register_capability("Bake persistence (screens to_state / restore, hash-guarded)",
        "Index.screens_state() / screens_restore(state): persist the Lloyd bake (centroids, "
        "blocks, contiguous rows, radii) so the ~40s 1M bake is paid ONCE EVER; restore is "
        "seconds. A sha256 of the corpus travels with the state -- restoring onto different "
        "items REFUSES loudly (a bake is a derived fact about one exact corpus). Round-trip "
        "answers bit-equal, pinned. Includes the BULK-FINISH worst-case guard: when 32 blocks "
        "prune nothing, sphere delegates to the exact fast path -- 1M dust measured 8527 -> 55 "
        "ms/q, recall 1.000. HoloForest's to_state convention, applied to screens.",
        example="import numpy as np; from holographic.caching_and_storage.holographic_index import Index; X=np.random.default_rng(0).standard_normal((512,16)); a=Index(X, method='sphere'); st=a.screens_state(); Index(X, method='sphere').screens_restore(st).nearest(X[0], k=2) == a.nearest(X[0], k=2)",
        native=True, aliases=("save the index bake", "persist the screens", "restore a baked index",
                              "bake once query forever", "hash guarded index state"))

    c.register_capability("Composable index (merge and ablate corpora without rebuild)",
        "Index.merge(other) / Index.ablate(source): HDRIFT's compose/ablate applied to retrieval "
        "-- THE INDEX AS A COMMUTATIVE MONOID. Baked block families concatenate with provenance; "
        "every sphere bound is a fact about its own members so CERTIFIED EXACTNESS survives union "
        "untouched (zero re-Lloyd). MEASURED LAWS (pinned): exact-over-union; merge(A,B).ablate(B) "
        "answers == A alone; commutative up to tie order; merge 2.8 ms vs rebuild. Pruning after "
        "merge = the bakes side by side, never re-optimized (priced). Sphere/ladder family.",
        example="import numpy as np; from holographic.caching_and_storage.holographic_index import Index; a=Index(np.eye(8)[:4], method='sphere'); b=Index(np.eye(8)[4:], method='sphere'); a.nearest(np.eye(8)[0]); b.nearest(np.eye(8)[7]); len(a.merge(b).items) == 8",
        native=True, aliases=("merge two indexes", "combine corpora without rebuild", "ablate a corpus source",
                              "composable index", "index algebra", "add and remove corpora"))

    c.register_capability("Retrieval dispute harness (FAISS + HoloForest + leCore, hard data only)",
        "tools/benchmarks_faiss.py: the NEUTRAL INSTRUMENT for benchmark disputes -- same hard "
        "data (real anchors + on-manifold offspring cliques at EVERY scale; a friendliness gate "
        "REFUSES near-orthogonal separable data), exact float64 ground truth computed by the "
        "harness, leCore pays its full ingest, FAISS configs stated in the output. MEASURED 100k "
        "x768: leCore fast recall 1.000 @ 23.4ms BEATS FAISS Flat exact (27.1ms); IVF 0.875 / "
        "HNSW 0.853 -- approximate engines drop 12-15%% recall on clique data where friendly "
        "benchmarks show ~0.99. Three gate bugs kept as negatives in the module docstring.",
        example="import subprocess; r=subprocess.run(['python3','tools/benchmarks_faiss.py','--scales','1000','--queries','8'],capture_output=True,text=True,timeout=600); 'recall' in r.stdout",
        native=False, aliases=("faiss benchmark", "retrieval dispute harness", "benchmark against faiss",
                               "independent benchmark harness", "recall benchmark hard data",
                               "compare index engines"))

    c.register_capability("Semantic rig (bones, hinges, and IK handles for the memory itself)",
        "mind.semantic_rig(): rig the framework like a bound mesh. Bones from each substrate's "
        "SYMMETRY GROUP: Givens hinges (GDN, full orthogonal) / rfft band-phase bones (HRR, cyclic; "
        "Nyquist excluded). IK = closed-form CCD under limits (planted pose 1e-16 rad). POSE = a new "
        "edit primitive: isometry, zero capacity cost (write pays crosstalk). SKINNING: key-space "
        "regions -- ortho topology exact; random keys leak at sqrt(nA/D); CANDY-WRAPPER quantitative "
        "(0.707 at full coverage), pinned not patched. Family: solve_ik / skin_mesh.",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); r=m.semantic_rig(dim=96, hrr_dim=1024, n_items=12); r['gdn']['restore_err'] < 1e-12",
        native=True, aliases=("semantic rig", "rig the memory like a mesh", "pose the memory",
                              "ik handles for the framework", "bones and joints for hypervectors",
                              "memory with a skeleton", "adaptive shape with trigger response",
                              "skin weights for memory", "candy wrapper", "regional memory handles", "bone chains for memory", "auto rig from the data", "re-address the memory", "twist bones", "kinematic redundancy"))

    c.register_capability("Shufflebrain (Pietsch's surgeries on holographic memory, measured)",
        "mind.shufflebrain_battery(): Pietsch's surgeries, measured. Rotation = COHERENT TRANSFORM; "
        "focal lesion: holographic keeps all items, localized loses half; cleanup identifies 24/24 at "
        "half-brain; GDN orthogonal-covariant vs HRR cyclic-only. GRAFT (S2): a CHANNEL, not a "
        "destination -- identify through it, consolidate FRESH = full transfer, host untouched; "
        "in-place pays the capacity law (kept negative, with mincing). docs/PANEL_pietsch_hologramic.md",
        example="import lecore; m=lecore.UnifiedMind(dim=64, seed=0); r=m.shufflebrain_battery(dim=512, n_items=12); abs(r['rotation']['vs_rotated']-r['rotation']['baseline'])<0.01",
        native=True, aliases=("shufflebrain", "pietsch battery", "rotate the memory trace",
                              "does memory survive brain surgery", "hologramic memory test",
                              "lesion the memory and measure recall", "memory graft experiment",
                              "graft amplification", "transfer memories between minds", "two speed transfer", "mince law", "spectral lesion", "literal resolution loss", "aligned mass fraction", "graft between trained models", "model graft rejection", "behavior transfer between models"))

    c.register_capability("Ouroboros (the closed memory loop: leCore eats the installed model's memory)",
        "THE NAMED PROCESS: a model with leCore installed in its weights OUTPUTS memory -- GDN head "
        "state (an outer-product accumulator, leCore's own HRR trace) and durable notes -- and "
        "server-side leCore CONSUMES it as an ordinary data structure, then feeds it back. MEASURED "
        "on exact GDN algebra: read 0.935; external write reads 0.951 by the model's own readout "
        "(zero forward passes); delete -> -0.24; capacity 0.932 pred / 0.905 meas; transcript "
        "consolidation 0.767 -> 0.918 (self-rehearsal = pollution, kept negative). Durable side: "
        "memory_write/memory_search per-tenant partition. docs/ZOO.md 7-8.",
        example="from holographic_mcp import MCPServer; import tempfile; s=MCPServer(memory_root=tempfile.mkdtemp()); s.handle({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'memory_write','arguments':{'text':'ouroboros lives'}}})['result']['isError']",
        native=True, aliases=("ouroboros", "closed memory loop", "feed the model's memory back",
                              "manage the installed model's memory", "the snake eats its tail",
                              "external memory of the installed model",
                              "the leap", "leap outside the training data"))

    c.register_capability("MCP server (mount leCore in any Model Context Protocol host)",
        "holographic_mcp.py: JSON-RPC 2.0 over stdio, stdlib-only, delegating to /tools + /invoke. "
        "Tools: lecore_map/find/describe/invoke; corpus_bind/ask; void_explore(handle_b=...) = the "
        "FEDERATED LEAP (A's licensed gaps instantiated in B, warrant attached); memory_write/"
        "search per-tenant partition; receipt_verify + lecore.receipt sha256 pair on EVERY call -- "
        "determinism is the proof system (charge once, serve the hash). Cost in _meta.",
        example="from holographic_mcp import MCPServer; s=MCPServer(); r=s.handle({'jsonrpc':'2.0','id':1,'method':'tools/list'}); [t['name'] for t in r['result']['tools']]",
        native=True, aliases=("mcp", "model context protocol", "serve tools over mcp", "openzoo",
                              "mount lecore in claude desktop", "mcp stdio server", "proof of inference receipt", "charge once serve the hash", "federated leap", "what does the zoo know that my corpus lacks"))

    c.register_capability("The memory mountain (measure your own cache tiers; the tiers predict the benchmarks)",
        "mind.memory_mountain(): streaming GB/s vs working set, tier detection (peak / knee / floor), "
        "predict_streaming_ms from the measured floor. THIS box: peak ~90 GB/s @ 0.5-1 MB (L2), floor "
        "~26 GB/s from 4 MB -- and bytes/floor REPRODUCED the fast-arbiter table to ~15% (exact f64 "
        "9.1 pred / 10.4 meas; f32 4.5/5.1; screens 1.6/1.9): the fast-path wins ARE the mountain "
        "wearing different working sets. KEPT NEGATIVES: the left flank is DISPATCH overhead (a Python "
        "probe cannot see L1, and says so); L3/RAM merge to ONE floor on a virtualized host.",
        example="import lecore; m=lecore.UnifiedMind(); curve,tiers=m.memory_mountain(sizes=[256e3,1e6,8e6,32e6]); tiers['peak_gbs'] > tiers['floor_gbs']",
        native=True, aliases=("measure cache bandwidth", "detect cache size", "memory mountain",
                              "L1 L2 L3 boundaries", "how fast is my ram", "why is the matvec this slow"))

    c.register_capability("The time machine (unitary dynamics: reversible, random-access, superposable time)",
        "mind.time_machine(): for UNITARY steps (|spectrum|=1) time is an ADDRESSABLE AXIS: time_jump "
        "reaches step 977 in one spectral power (5e-13) and t<0 REVERSES exactly (1.4e-15 back; decaying "
        "steps refuse WITH eig_min^t -- the probe measured 1.4e+121 first). bundle_sims: K sims in ONE "
        "vector (circulant steps commute with binding, 1.6e-15); members read at the 1/sqrt(K) LAW; "
        "evolve_functional: a PRECOMMITTED ensemble readout, EXACT. KEPT NEGATIVE: keyed functionals are "
        "NOT exact (cosine 0.34 -- crosstalk survives weighting).",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(); tm=m.time_machine(); spec=tm.make_unitary_step(64, seed=3); x=np.random.default_rng(0).standard_normal(64); y=tm.time_jump(x, spec, 500); back=tm.time_jump(y, spec, -500); float(np.max(np.abs(back-x)))",
        native=True, aliases=("run the simulation backwards", "jump to timestep t", "time travel state",
                              "reverse the dynamics", "many simulations one vector", "ensemble in superposition",
                              "undo n steps"))

    c.register_capability("The HRNN collapse (n timesteps as ONE installed operator)",
        "mind.collapse_recurrence(machine, step_program, n): a linear recurrence x_t = M x_(t-1) + b IS "
        "leCore's HRNN (decay inside M) -- and n applications of one operator ARE one operator, so 100 "
        "sim steps collapse to a single certified affine matvec. MEASURED: 156x on endpoint queries at "
        "2e-15 vs the stepped trajectory; affine drift+decay collapses exactly (geometric-series "
        "offset); the certificate prices the SPECTRUM (eig_max^n -- explosive recurrences announce "
        "themselves at compile); HOST links (clamps, branches) REFUSE with names -- sim_program_run "
        "stays the referee and the drift instrument.",
        example="import numpy as np; from holographic.agents_and_reasoning.holographic_machine import HoloMachine; import lecore; m=lecore.UnifiedMind(); mm=HoloMachine(dim=6,seed=7,data=['a']); mm.functions_symbolic={}; run,cert=m.collapse_recurrence(mm,[('FAC',('d',lambda f: 0.9*f)),('HALT',None)],40); (round(float(run(np.ones(6))[0]),6), round(cert['eign_max'],6))",
        native=True, aliases=("collapse a recurrence", "n steps in one matvec", "hrnn in the weights",
                              "fast forward the simulation", "skip to the end state", "decay gate installed"))

    c.register_capability("Simulation in the weights (installed physics step, drift-audited)",
        "mind.sim_program_run(machine, step_program, init, n_steps): compile ONE physics step (linear "
        "projections install certified; clamps ride as marked HOST:APPLY links), iterate it installed "
        "with the state fed back -- the chain IS the integrator. Returns (trajectory, manifest, DRIFT "
        "curve vs the live step): measured 100-step PBD chain at drift identically 0.0. Any nonzero "
        "drift is the certificate residual compounding -- visible, never hidden.",
        example="import numpy as np; from holographic.agents_and_reasoning.holographic_machine import HoloMachine; import lecore; m=lecore.UnifiedMind(); mm=HoloMachine(dim=6,seed=7,data=['a']); mm.functions_symbolic={}; tr,man,dr=m.sim_program_run(mm,[('FAC',('s',lambda f: f*0.9)),('HALT',None)],np.ones(6),10); (tr.shape, float(dr.max()))",
        native=True, aliases=("run a physics sim in the weights", "installed simulation",
                              "physics step as a model", "drift curve", "simulate inside the model"))

    c.register_capability("Render to text from the weights (installed image formation -> PGM)",
        "mind.raster_program_pgm(machine, program, params, w, h): run an installed image-formation "
        "chain (RECTANGULAR linear maps certify -- 3 lights -> 64 pixels) and emit the frame as PGM P2 "
        "ASCII -- the picture leaves through the mouth, no file I/O; byte-exact vs the live path "
        "(pinned). Quantization to 0..255 ints is the SERIALIZER's job, stated in the docstring.",
        example="import numpy as np; from holographic.agents_and_reasoning.holographic_machine import HoloMachine; import lecore; m=lecore.UnifiedMind(); W=np.full((4,2),40.0); mm=HoloMachine(dim=2,seed=7,data=['a']); mm.functions_symbolic={}; pgm,_=m.raster_program_pgm(mm,[('FAC',('f',lambda q: W@q)),('HALT',None)],np.ones(2),2,2); print(pgm)",
        native=True, aliases=("render from the weights", "picture out of the model", "installed render",
                              "emit an image as text", "pgm from the model"))

    c.register_capability("Cleanup as one attention head (certified agreement, priced ties)",
        "mind.cleanup_as_attention(codebook, beta) expresses exact cleanup as y = A^T softmax(beta*Ax) "
        "-- ONE attention head, codebook as keys AND values: the host's own mechanism. "
        "mind.attention_read_certificate(codebook, queries, beta) MEASURES agreement vs exact cleanup "
        "on YOUR queries (real wiki: 0.575 @beta=4, 1.000 @beta>=16). PRE-REGISTERED NEGATIVE, held by "
        "theorem: softmax averages exactly-tied rows -- the lowest-index tie rule is inexpressible; "
        "ties are the agreement floor.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(); rng=np.random.default_rng(0); A=rng.standard_normal((50,16)); A/=np.linalg.norm(A,axis=1,keepdims=True); q=A[:8]+0.05*rng.standard_normal((8,16)); m.attention_read_certificate(A,q,beta=64.0)",
        native=True, aliases=("install cleanup as attention", "attention read certificate",
                              "measure attention agreement", "cleanup as a head", "softmax vs argmax gap"))

    c.register_capability("Mesh through the weights, OBJ out the mouth (installed 3D program)",
        "mind.mesh_program_obj(machine, program, verts, faces): compile FAC steps (rigid transforms "
        "certify BLOCKDIAG -- 9+3 params/step), run the chain INSTALLED with the mesh's flattened "
        "vertices as the state, and get the transformed mesh back as an OBJ TEXT DUMP -- the token "
        "stream is the output device, no file I/O anywhere. BYTE-EXACT vs the live-faculty path "
        "(pinned). host_fallback=True lets refused steps ride as marked HOST:APPLY links.",
        example="import numpy as np; from holographic.agents_and_reasoning.holographic_machine import HoloMachine; from holographic.agents_and_reasoning.holographic_compileinstall import mesh_program_obj; mm=HoloMachine(dim=12,seed=3,data=['a']); mm.functions_symbolic={}; obj,_=mesh_program_obj(mm,[('FAC',('s',lambda f: f*2.0)),('HALT',None)],np.eye(4,3),[(0,1,2)]); print(obj[:60])",
        native=True, aliases=("run a mesh through installed weights", "obj from the model",
                              "3d program in the weights", "emit a mesh as text", "installed mesh transform"))

    c.register_capability("Byte-plane float packing (compress the 'incompressible', byte-exact)",
        "mind.float_pack_bytes / float_unpack_bytes: general codecs get ~1.08x on float embeddings "
        "(interleaved sign/exponent/mantissa reads as noise). Byte-plane TRANSPOSE groups like bytes "
        "before lzma: 1.19x on the same real bytes, byte-exact round trip (f32/f64, any shape, F-order "
        "handled). KEPT NEGATIVE, measured: row-delta before planing adds NOTHING -- embedding rows are "
        "not sequentially correlated; the filter ships without it.",
        example="import numpy as np, lecore; m=lecore.UnifiedMind(); A=(np.random.default_rng(0).standard_normal((50,16))*0.1).astype(np.float32); b=m.float_pack_bytes(A); (np.array_equal(m.float_unpack_bytes(b), A), len(b) < A.nbytes)",
        native=True, aliases=("compress embeddings lossless", "float compression byte exact",
                              "byte plane shuffle", "pack float arrays smaller", "embeddings wont compress"))

    c.register_capability("Flagship benchmarks (real data, SOTA context, negatives loud)",
        "tools/benchmarks_flagship.py + docs/BENCHMARKS.md: calibrated abstention realized-vs-promised "
        "FA on SHUFFLED-REAL noise (0.013 @ alpha=0.01, power 1.000 -- within binomial CI; no SOTA ships "
        "the promise); screens recall 0.97 [0.94,0.99] self-measured with honest demotion (context: DARTH "
        "SLOs; 2026 HNSW silent-degradation post-mortems); 175-byte model re-bakes 2048 params "
        "bit-identically (Tracr stores weights, we store the rule); codec baselines set as the honest bar. "
        "KEPT NEGATIVE: screens wall-clock LOSES to exact BLAS at 36k -- scan savings != latency there.",
        example="import subprocess; print(subprocess.run(['python3','tools/benchmarks_flagship.py'],capture_output=True,text=True).stdout[:300])",
        native=True, aliases=("benchmarks", "sota comparison", "how does it compare to hnsw",
                              "beat state of the art", "benchmark scores", "prove it with numbers"))

    c.register_capability("The showcase (runnable proof of what makes this engine different)",
        "tools/showcase.py executes the six flagship claims as LIVE assertions in ~2 s on a laptop CPU: "
        "calibrated abstention (0/100 noise passed at alpha=0.01), the recall budget demoting honestly with "
        "the measurement attached, exact tiled search with cross-tile tie identity, the ~250-byte model file "
        "re-baking bit-identical weights, VM==installed==symbolic three-referee conformance, and the one "
        "determinism tie rule. docs/SHOWCASE.md is the companion for new eyes: run the proof before "
        "summarizing the project.",
        example="import subprocess; print(subprocess.run(['python3','tools/showcase.py'],capture_output=True,text=True).stdout[-200:])",
        native=True, aliases=("what makes this project special", "summarize this project", "demo",
                              "show me what it can do", "why is this different", "elevator pitch",
                              "prove the claims", "quickstart"))

    c.register_capability("Native holographic micro-model (baked; the model IS the program)",
        "mind.native_model(dim, seed, program): a from-scratch model, no pretrained host -- layers are the "
        "certified parameterizations (circulant/permutation/dense), registers are recurrent state, forward() "
        "= the compiled program (pinned == VM). BAKED: save() writes a ~250-byte RULE file; load() re-bakes "
        "bit-identical weights. unitary=True bakes norm-preserving atoms for DEEP programs (depth-256 error "
        "7.8e82 -> 6e-15 measured; the conditioning warning names this switch). to_dense(op) exports the "
        "literal host matrix.",
        example="mdl=mind.native_model(512, 7, [('LOAD','a'),('BIND','k'),('HALT',None)], data=['a','k']); "
                "y=mdl.forward(); print(mdl.layers())",
        native=True, aliases=("model without pretrained weights", "bake a model from a program",
                              "deep bind chain explodes", "norm preserving atoms", "unitary bake",
                              "neurosymbolic", "interpretable by construction", "white box model",
                              "auditable AI model", "model library in one file", "many programs one model",
                              "function library as weights",
                              "tiny model file regenerates weights", "vsa native model",
                              "export layer as matrix", "the model is the program"))

    c.register_capability("Compile a VM program into installed form (conformance + manifest)",
        "mind.compile_program_installed(machine, program): a symbolic HoloMachine program becomes a chain of "
        "projector-CERTIFIED matvecs + register slots; REPEAT of a linear body collapses to ONE operator "
        "power (spectral, exact). CONFORMANCE PINNED: VM and installed chain agree NUMERICALLY (allclose, "
        "not cosine) on a REPEAT+STORE/RECALL program. Nonlinear bodies refuse. Every compile yields the "
        "manifest (kind, payload SHAPE, residual per op); save_manifest writes the sidecar.",
        example="from holographic.agents_and_reasoning.holographic_machine import HoloMachine; "
                "mach=HoloMachine(dim=512, seed=7, data=['a','k']); mach.functions_symbolic={}; "
                "run,man=mind.compile_program_installed(mach, [('LOAD','a'),('BIND','k'),('HALT',None)]); "
                "print(man['chain'])",
        native=True, aliases=("run a program in the weights", "compile to installed opcodes",
                              "manifest schema", "model card fields", "what installs into weights",
                              "which units cannot install",
                              "vm conformance installed", "repeat as operator power",
                              "manifest of installed capabilities", "what is installed in the model"))

    c.register_capability("Out-of-core exact search (top-k over on-disk arrays of any size)",
        "mind.out_of_core_search(path, queries, k) runs EXACT tie-safe top-k over an .npy file WITHOUT "
        "loading it: np.memmap + the tiled fold stream tiles from disk, so memory is bounded by the tile "
        "whatever the file size. MEASURED: 600 MB file, 40.5 ms/q k=5, peak RSS 0.75 GB. The 2026 ANN "
        "consensus calls exact 'not applicable' at scale and ships approximate+rerank; this is the honest "
        "inversion -- exact all the way down, recall 1.0 by construction, deterministic ties.",
        example="import numpy as np; np.save('/tmp/d.npy', np.random.default_rng(0).standard_normal((5000,64))); "
                "v,i = mind.out_of_core_search('/tmp/d.npy', np.random.default_rng(1).standard_normal(64), k=3); print(i[:,0])",
        native=True, aliases=("search a file bigger than memory", "exact search on disk",
                              "top k over a huge npy", "streaming nearest neighbours",
                              "dataset does not fit in ram"))

    c.register_capability("The projector (measure a faculty into installed form, or refuse)",
        "mind.project_faculty(f, dim): probe a callable, CERTIFY on held-out inputs: permutation / "
        "circulant / blockdiag / dense / rectangular; refusals retry HOST vocabulary (rmsnorm, "
        "gated/SwiGLU) then ENGINE kinds (powerlaw: gamma/tone certify at 1e-16 -- render chains lost "
        "their last host links). scale= names the DOMAIN. Census: 8.8% facade / 8.6% module verdict "
        "rate -- frame hypothesis REFUTED; the ore is the 11.4% module refusals (vocabulary targets); "
        "FAC closures make this a LOWER bound.",
        example="import numpy as np; p=mind.project_faculty(lambda v: np.roll(v,3), 64); "
                "print(p['kind'], p['residual'])",
        native=True, aliases=("turn a function into a matrix", "can this install into weights",
                              "block diagonal detection", "certify against host layers", "rmsnorm target",
                              "installability census", "what fraction installs",
                              "compile a faculty into the model", "project code into vsa form",
                              "is this operation linear", "measure an operator into installed form"))

    c.register_capability("Trace energy partition (the saturation ledger: signal / crosstalk / damage)",
        "mind.trace_partition(trace, atoms[, stored_idx]) splits a bundle's FIXED energy into signal "
        "(least-squares onto stored atoms), the law's ~n/dim crosstalk floor, and damage above it. "
        "Fractions SUM TO 1 by construction -- the ledger attributes power, never creates it. Membership "
        "MAD-gated when stored_idx unknown (estimated=True). Selftest: clean~all-signal; injected damage "
        "moves only the damage account.",
        example="import numpy as np; A=np.random.default_rng(0).standard_normal((128,512)); "
                "A/=np.linalg.norm(A,axis=1,keepdims=True); t=A[:9].sum(0); print(mind.trace_partition(t, A))",
        native=True, aliases=("how much of this bundle is signal", "signal versus crosstalk fraction",
                              "is my trace damaged or just loaded", "memory health report",
                              "energy budget of a superposition", "saturation ledger"))

    c.register_capability("Tiled matmul-reduce (exact per-query max/argmax/sum, memory bounded by the tile)",
        "holographic_tiledreduce.tiled_matreduce(items, Q) reduces an (N x D)x(D x Q) product per query "
        "WITHOUT the (N,Q) matrix: a pure FOLD (step(state, tile) -> state over a commutative monoid), so "
        "peak memory is tile x Q whatever N is, and the step is REPEAT-expressible for the installed side. "
        "MEASURED: bit-identical argmax to dense on 12k REAL text vectors (strict-> preserves the first-index "
        "tie rule -- planted cross-tile ties pinned), FASTER than dense at these shapes (0.13 vs 0.22s), 3 MB "
        "vs 19 MB. This is what turned calibrated abstention's 7.45 GiB death at N=500k into a 0.9 GB loop.",
        example="import numpy as np; from holographic.sampling_and_signal.holographic_tiledreduce import "
                "tiled_matreduce; X=np.random.default_rng(0).standard_normal((5000,64)); "
                "b,a=tiled_matreduce(X, X[:3].T); print(a)",
        aliases=("argmax over a huge matrix without memory", "chunked similarity max", "tiled reduction",
                 "exact search bounded memory", "abstention at large scale", "blockwise matmul reduce"))

    c.register_capability("Deterministic top-k (the tie-safe shortlist rule, stated once)",
        "holographic_determinism.topk_det(scores, k): indices of the k best, descending, ties to the LOWEST "
        "index -- argmax_tiebreak extended to a list, and the ISA-1 pattern applied at k>1 (the same "
        "shortlist rule had been hand-copied into THREE sites, each with its own kept-negative comment about "
        "the k+1 boundary bug). Index.nearest, Index.nearest_batch and BM25.rank now DELEGATE here; planted "
        "discrete-tie traps pin bit-identity. Conformance home for ANY substrate's top-k.",
        example="import numpy as np; from holographic.misc.holographic_determinism import topk_det; "
                "print(topk_det(np.array([3.,1.,3.,2.]), 2))",
        aliases=("stable top k", "tie safe shortlist", "deterministic ranking rule", "top k contract",
                 "ties resolve lowest index",
                 # outsider vocabulary: the determinism SPINE should win these, not a leaf module
                 "reproducible AI", "deterministic machine learning", "bit identical results",
                 "same answer every run", "reproducible builds for models"))

    c.register_capability("Superposed key-value memory (capacity law + allocator + gated resonator decode)",
        "mind.superposed_memory(vocab=V) stores pairs as ONE vector (sum of bind(k,v)). "
        "codebook='hadamard' GENERATES atoms (O(dim), zero crosstalk; vocab<=2*dim refused); 'lazy' seeds "
        "rows per-index for unbounded vocab (1M measured: O(1) build, recall 1.0, 0.6 vs 32 GB dense). "
        "memory_capacity_law PREDICTS how many pairs fit; allocate_memory_dim inverts it BEFORE storing. "
        "recall(decoder='pic') cancels interference to ~1.5x the one-shot wall, LOAD-GATED past its phase "
        "transition (kept negative: undamped PIC there is worse). int8 decision-free; sign keeps ~70%.",
        example="import numpy as np; mem=mind.superposed_memory(vocab=256); n=mind.memory_capacity_law(vocab=256); "
                "ks=np.arange(n); vs=(ks*7)%256; r=mem.store(ks,vs).recall(ks, decoder='pic'); "
                "print(n, (r['values']==vs).mean(), r['decoder'])",
        native=True, aliases=("how many pairs fit in a vector", "associative memory size",
                              "store facts in one hypervector", "key value bundle capacity",
                              "what dimension do I need for N items", "allocate dimension before storing",
                              "iterative cleanup decoder", "interference cancellation recall",
                              "CDMA memory", "superposition recall limit"))

    c.register_capability("State demand meter (how much memory does this stream need)",
        "Measure BEFORE allocating: mind.state_demand(x) returns the TT-SVD bond dimensions (causal-state "
        "counts) of the stream's block distribution, thresholded against the SAME stream shuffled -- iid "
        "reads rank 1, period-p reads p, sampling noise cannot inflate it. mind.entropy_rate(x) returns "
        "{h, E}: h~0 marks the deterministic regime (a generator exists), h>0 prices irreducible novelty; "
        "dense-regime GUARDED (refuses block lengths the sample count cannot support -- the measured "
        "silent-low-bias failure of naive plug-in). Feed demand into allocate_memory_dim.",
        example="import numpy as np; x=np.tile(np.arange(4),5000); print(mind.state_demand(x)['ranks'], "
                "mind.entropy_rate(x)['h'])",
        native=True, aliases=("how much state does this stream need", "count causal states",
                              "bond dimension of a process", "entropy rate of a signal",
                              "how many bits to remember this", "is this stream predictable",
                              "excess entropy", "memory demand before allocating"))

    c.register_capability("Compressibility gate with horizon (does a generator exist, certified at this window)",
        "mind.compressibility_check(x): stage 1 rejects on measured entropy rate (catches walk/AR/white -- "
        "the nulls a phase-randomised surrogate provably misses); stage 2 calibrates fit_deterministic's "
        "correlation against phase-randomised surrogates of the SAME signal (catches spectrum-matched "
        "stochastic imposters). Every verdict carries a HORIZON: the same process honestly earns different "
        "verdicts at different windows (measured -- short windows of a long randomisation ARE locally pure), "
        "so a pass certifies THIS window only; extrapolating past it is the caller's declared risk.",
        example="import numpy as np; t=np.arange(2000.); print(mind.compressibility_check(np.sin(2*np.pi*t/210))['passed'])",
        native=True, aliases=("is this signal compressible", "does a deterministic generator exist",
                              "is this noise or structure", "compressibility test with null",
                              "should I fit or refuse", "certified at what horizon",
                              "entropy gate before fitting", "generator existence check"))

    c.register_capability("The Holographic RNN (measure -> identify | price | refuse, with provenance)",
        "mind.holographic_rnn().process_stream(x) walks the abstention ladder: calibrated gate first (a "
        "pass = a GENERATOR exists AT THIS HORIZON, identified via fit_deterministic, returned as bytes + "
        "predict); a refusal prices state demand (TT ranks) and routes to associative() (capacity-law "
        "allocation, load-gated resonator decode) or classifier() (trajectory readout carrying BOTH "
        "invariances: arrival-time traps AND Levy areas -- neither subsumes the other). Incompressible "
        "streams are refused WITH an allocator quote. Every verdict carries {regime, h, horizon, why}.",
        example="import numpy as np; eng=mind.holographic_rnn(); t=np.arange(1000.); "
                "r=eng.process_stream(np.sin(2*np.pi*t/150)); print(r['regime'], r['horizon'])",
        native=True, aliases=("holographic rnn", "better rnn", "sequence model that abstains",
                              "recurrent model with provenance", "route a stream to the right model",
                              "should I fit a generator or a memory", "sequence engine",
                              "rnn that measures first", "adaptive sequence architecture",
                              "trajectory classifier with signatures"))

    c.register_capability("Stream sentinel (regime watch + change events + priced recorder)",
        "mind.stream_sentinel().watch(x) slides the HRNN ladder along a stream, segments it by regime, and "
        "raises change events (regime flip or entropy-rate jump) carrying BOTH windows' provenance -- alarms "
        "arrive with evidence. record(x) stores each window at its cheapest FAITHFUL form: generator params "
        "(~30 floats, prefix-fit/suffix-CERTIFIED so a lone tone's surrogate degeneracy cannot block it), "
        "quantile symbols at the measured rate, or raw floats -- noise is never fake-compressed. replay() "
        "reconstructs in-window only (no extrapolation past any horizon), certificates riding every entry.",
        example="import numpy as np; s=mind.stream_sentinel(); t=np.arange(4000.); "
                "x=np.concatenate([np.sin(2*np.pi*t[:2000]/170), np.random.default_rng(0).standard_normal(2000)]); "
                "w=s.watch(x); print(len(w['events']), [seg['regime'] for seg in w['segments']])",
        native=True, aliases=("watch a stream for changes", "regime change detector", "drift monitor",
                              "segment a signal by behavior", "compress telemetry honestly",
                              "priced recorder", "anomaly boundary detector", "when did the process change",
                              "adaptive stream compression", "record a stream with certificates"))

    c.register_capability("Horizon profile (one stream, verdicts across scales -- drift localization)",
        "mind.holographic_rnn().route_profile(x) runs the routing ladder on tail-anchored windows at "
        "geometric scales and returns the verdict PER HORIZON. Compressibility is scale-relative (measured), "
        "so one verdict is a point sample; the profile is the function. Scale DISAGREEMENT is the signal: a "
        "regime change appears as the small-window verdict diverging from the large one, and the divergence "
        "scale brackets when it happened (measured on a sine->noise splice: h climbs 0.87 -> 1.34 -> 1.98 as "
        "the window narrows onto the noise). Memoised meters keep the repeated sub-window work cheap.",
        example="import numpy as np; e=mind.holographic_rnn(); x=np.concatenate([np.sin(np.arange(1500.)/24), "
                "np.random.default_rng(0).standard_normal(500)]); print([(p['window'],p['regime']) for p in e.route_profile(x)])",
        native=True, aliases=("did the regime change", "when did this stream change", "drift detection",
                              "verdict at multiple scales", "is it still the same process",
                              "localize a change point", "multi-horizon check", "stream drift profile"))

    c.register_capability("Triage cascade (a little model that amortises an expensive check, fast-REJECT only)",
        "mind.triage_cascade() fronts an expensive predicate (default: the full compressibility gate, "
        "n_null+1 generator fits) with a tiny ridge over cheap features (memoised entropy rate, top-bin "
        "power, derivative skew, lag-1 autocorr). THE CONTRACT: the fast path may only reject -- every "
        "accept runs the full machinery, so accept decisions are the oracle's by construction. Calibrated "
        "so training positives are never fast-rejected (threshold below the lowest positive, minus a "
        "safety spread); held-out false-reject rate is measured, not assumed. Trained heads save()/load().",
        example="import numpy as np; t=np.arange(600.); casc=mind.triage_cascade(); "
                "casc.fit([np.sin(2*np.pi*t/105), np.random.default_rng(0).standard_normal(600)]); "
                "print(casc(np.random.default_rng(1).standard_normal(600))['path'])",
        native=True, aliases=("speed up an expensive check", "fast pre-filter before fitting",
                              "little model shortcut", "amortize a slow test", "cheap gate first",
                              "branch predictor for faculties", "triage before computing",
                              "skip the expensive path when obvious"))

    c.register_capability("Train a model, one front door (honest about when it's actually trained)",
        "mind.train_model(examples, labels=...) routes to the right learner and tells the truth: "
        "sequences + labels -> trajectory classifier, REFUSING to call an underdetermined ridge "
        "'trained' (the measured learning-curve knee -- flat until n_train exceeds the feature count, "
        "0.62 -> 0.91 across it -- enforced at the API, with the row count that flips the verdict); "
        "(keys, values) -> pair memory with dimension allocated from the capacity law BEFORE storing; "
        "a bare stream -> the HRNN ladder (a generator with predict(), or an honest verdict). Every "
        "result: {kind, trained, why}; every model ...",
        example="import numpy as np; r=mind.train_model((np.arange(50), (np.arange(50)*7)%%97)); "
                "print(r['kind'], r['trained'], r['model'].recall(np.arange(5))['values'])",
        native=True, aliases=("train a model on my data", "make a classifier from examples",
                              "learn from labeled sequences", "train and export a model",
                              "is my model actually trained", "how many examples do I need",
                              "fit a model to a stream", "one call training"))

    c.register_capability("Structure fingerprint + drift (did this stream change between releases)",
        "mind.structure_fingerprint(x) -> {h, E, ranks, demand_bits, horizon}: a tiny structural signature "
        "of any stream (asset bytes, CI timings, solver residuals), memoised so logging it per artifact per "
        "release is near-free. mind.structure_drift(a, b) compares two and answers in MEASURED units -- "
        "'entropy rate moved 0.00 -> 1.99', 'state demand moved: max rank 4 -> 1' -- with tolerances set "
        "from this tree's own observed spreads. The regression detector for pipelines: structure changes "
        "move the fingerprint before they break a unit test.",
        example="import numpy as np; a=mind.structure_fingerprint(np.tile(np.arange(4),2000)); "
                "b=mind.structure_fingerprint(np.random.default_rng(0).integers(0,4,8000)); "
                "print(mind.structure_drift(a,b)['why'])",
        native=True, aliases=("did this data change", "detect drift between versions",
                              "structural regression check", "fingerprint a stream",
                              "compare two datasets structurally", "pipeline output changed",
                              "release regression detector", "signature of a signal"))

    c.register_capability("Nested memory library (many knowledge bases in ONE vector, one-unbind queries)",
        "mind.nested_memory(n_bases=M, facts_per_base=n) allocates ONE vector (flat capacity law at "
        "the product load M*n) holding a whole shelf of pair-memory bases: add(name, keys, values) "
        "superposes a base under its name atom; shelve(name, memory) ingests an existing trained "
        "SuperposedMemory; query(name, keys) answers (base, key) -> value in a SINGLE unbind, because "
        "bind's associativity flattens sum_i bind(name_i, sum_j bind(k,v)) into composite-key pairs "
        "-- no base is ever reconstructed to be read. Load-gated PIC decode; int8 decision-free; "
        "exports at 1 bit/dim.",
        example="lib=mind.nested_memory(n_bases=2, facts_per_base=3); import numpy as np; "
                "lib.add('a', np.arange(3), np.arange(3)*7); lib.add('b', np.arange(3), np.arange(3)*11); "
                "print(lib.query('b', np.arange(3))['values'])",
        native=True, aliases=("many databases in one vector", "library of memories",
                              "nested knowledge bases", "memory of memories",
                              "query across model shelf", "holographic library",
                              "two level lookup one operation", "shelve a trained memory"))

    c.register_capability("Easy model: train it, ask it, save it (three verbs, any kind)",
        "m = mind.easy_model(data, labels=...) trains the right model (pair memory / sequence "
        "classifier / generator) behind ONE handle; m.ask(query) answers regardless of kind; NOTE its "
        "scope honestly: exact or fuzzy (edit-distance, correction reported) lookup over YOUR stored "
        "keys and labels, NOT natural-language prompting -- semantic queries belong to "
        "find_capability, comprehension to DECLARE (key ids -> values, sequences -> labels, a count "
        "-> that many forecast steps); m.save(path) and mind.load_easy_model(path) round-trip it as a "
        "small npz. All train_model honesty guards apply: ...generator ...",
        example="import numpy as np; m2=mind.easy_model((np.arange(30),(np.arange(30)*7)%%64)); "
                "print(m2.ask(np.arange(3))['answer']); m2.save('/tmp/em.npz'); "
                "print(mind.load_easy_model('/tmp/em.npz').ask(np.arange(3))['answer'])",
        native=True, aliases=("easiest way to train a model", "train then query a model",
                              "simple model api", "ask a trained model", "three verb model",
                              "load a model and use it", "train ask save", "beginner model training"))

    c.register_capability("HRNN domain recipes (forecasting, markets, science, data, text, audio)",
        "mind.hrnn_recipes(topic) is the use-case front door: a working call sequence per domain -- "
        "'forecasting' (certify-then-extend, horizon-scoped), 'market analysis' (route + fingerprint "
        "+ drift; returns honestly refused), 'scientific study' (generator existence, causal-state "
        "demand, trajectory classification), 'data processing' (per-release fingerprints, triage "
        "cascades), 'text generation' (price the corpus; generation routes to n-gram faculties; "
        "comprehension to DECLARE), 'audio' (streams route like any signal). Every recipe carries an "
        "HONEST field stating what the mechanism will not do.",
        example="print(mind.hrnn_recipes()); print(mind.hrnn_recipes('weather forecasting')['how'])",
        native=True, aliases=("forecast the weather", "weather forecasting", "analyze market data",
                              "predict a time series", "process my data with hrnn", "scientific data analysis",
                              "generate text with hrnn", "audio analysis", "what can the hrnn do for me",
                              "hrnn use cases", "domain recipes"))

    c.register_capability("Behavior meter for creature and agent minds (wrong-habit alarm)",
        "mind.behavior_meter(actions, rewards, prev=last_epoch) is the two-meter learning instrument: "
        "entropy rate of the action stream measures policy FORMATION (h_norm 1.0 = acting at chance, "
        "<0.6 = crystallised), rewards measure CORRECTNESS, and formation advancing while reward stays "
        "flat fires the WRONG-HABIT ALARM -- measured live on a real CreatureMind that crystallised "
        "(h 1.96 -> 0.97) at policy-correct 0.25. Neither meter alone can see a confidently wrong habit. "
        "Cheap (memoised fingerprints), online, per creature per epoch; actions may be any hashables.",
        example="import numpy as np; r=np.random.default_rng(0); "
                "e0=mind.behavior_meter(r.integers(0,4,240), rewards=r.random(240)*0.1); "
                "e1=mind.behavior_meter(np.tile(np.arange(4),60), rewards=r.random(240)*0.1, prev=e0); "
                "print(e1['formation'], e1['alarm'])",
        native=True, aliases=("is my creature actually learning", "wrong habit alarm",
                              "agent learning progress meter", "did the policy crystallise wrong",
                              "creature behavior formation", "reinforcement learning sanity check",
                              "behavior entropy meter", "policy formation vs correctness"))

    c.register_capability("Dynamic model synthesis (emit the pipeline as an inspectable recipe)",
        "mind.synthesize_model(data, labels=...) measures the data and EMITS the pipeline as a JSON recipe "
        "-- each stage (adapter, features, readout/decoder, guards) recorded WITH the measurement that "
        "justified it -- then compiles and trains it. Recipes are artifacts like stored VM programs: "
        "diffable, versionable, replayable. v1 scope stated honestly: choices are measurement-driven rules "
        "over the shipped stages, not open-ended codegen. Also: mind.find_capability is now BAKED (per-cap "
        "haystacks precomputed + cross-session memo keyed by catalog hash: 50ms -> 1ms warm, repeats free).",
        example="r=mind.synthesize_model((__import__('numpy').arange(40),(__import__('numpy').arange(40)*3)%%50)); "
                "print([s['stage']+':'+s['choice'] for s in r['recipe']['stages']])",
        native=True, aliases=("build a model pipeline automatically", "synthesize a model on the fly",
                              "emit a training recipe", "why did it choose this model",
                              "dynamic model construction", "model as a stored program")) 

    c.register_capability("Certified surrogate layer (serve computation from a model, never fabricate)",
        "mind.make_surrogate(fn, sample_inputs) runs fn ONCE over the samples and returns a callable with a "
        "three-way contract: CERTIFIED EXTENSION where the ladder certifies a generator (measured 9078x on "
        "a fine-step simulation, NRMSE 0.041), EXACT hash-replay on seen inputs, and the real computation "
        "(memoised) otherwise -- never fabrication. .provenance states which contract is in force and why. "
        "For big-vocab context stores, mind.big_pair_memory streams seed-derived codebooks in chunks (the "
        "MQAR pattern) so the state is ONE vector and materialised codebooks cost nothing.",
        example="s=mind.make_surrogate(lambda i: float(__import__('numpy').sin(2*3.14159*i/40)), range(400)); "
                "print(s(7)['path'], s(450)['path'])",
        native=True, aliases=("replace heavy computation with a model", "surrogate for a simulation",
                              "cache a computation as a model", "certified surrogate",
                              "speed up repeated computation", "big vocabulary pair memory",
                              "context memory at dictionary scale", "streamed codebook memory"))

    c.register_capability("Enriched capability search (dictionary-augmented routing) + recipe replay",
        "mind.find_capability_enriched(q): words the catalog does not know are looked up in the "
        "in-tree 144k dictionary and their definition tokens (suffix-stemmed) join the search -- "
        "'prognosticate the morrow' reaches forecasting, 'an augury of my ledgers' reaches "
        "drift/fingerprints. Additive by construction (tokens only added: raw hits can never be "
        "lost); expansions reported, never silent. Also mind.replay_model_recipe(recipe, data): "
        "retrain from a stored synthesis recipe and ASSERT the stage choices reproduce -- a recipe is "
        "a contract, drift raises with the diff.",
        example="r=mind.find_capability_enriched('prognosticate the morrow'); "
                "print(r['expansions'], [str(c)[:30] for c in r['results'][:2]])",
        native=True, aliases=("search with synonyms", "dictionary augmented search", "enriched routing",
                              "find capability with rare words", "replay a training recipe",
                              "reproduce a model exactly", "rag routing"))

    c.register_capability("Scale advisor (consult every capacity law BEFORE hitting the wall)",
        "mind.advise_scale(n_pairs=..., vocab=..., dim=..., bundle_k=..., factors=..., depth=...) "
        "applies every measured law in one checkpoint -- pair capacity via allocate (alpha-exact "
        "dim), the PIC decoder transition, the bundle linear-readout ceiling k*~0.13*D (sparse "
        "decoders hold ~8.7x more), the factorization hard wall F=4 (split factor groups beyond it) "
        "-- and returns margins, the BINDING constraint, and a concrete prescription; fix=True "
        "returns the corrected spec. Empirical knobs (depth) route to mind.auto_scale by name, which "
        "doubles the responsive knob and diagnoses genuine walls.",
        example="print(mind.advise_scale(n_pairs=200, vocab=1000, dim=512, fix=True)['prescription'])",
        native=True, aliases=("hit a capacity wall", "how big should dim be", "auto scale capacity",
                              "overcome depth limit", "memory is full", "bundle overloaded",
                              "too many factors", "which constraint is binding", "grow the dimension",
                              "capacity check before building", "nesting depth wall", "tree too deep", "probe depth limit"))

    c.register_capability("Compute plan (amortisation tiers before the backend race)",
        "mind.compute_plan(n, calls_expected, repeat_fraction=..., stream=...) routes a computation "
        "through the FULL menu, not just raw backends: exact-replay memo for repeats (recomputing a "
        "known answer is the only true waste), certified surrogate when a sample output stream passes "
        "the ladder (measured 9078x; exact-cycle for symbolic), then the real zig policy (measured "
        "2-5x regime) and the real gpu_crossover row when hardware has measured one -- an unmeasured "
        "device is NAMED blocked, never guessed. Where the GPU wins raw throughput, the winning move "
        "is often shrinking the work: superposition, ...",
        example="print(mind.compute_plan(10**6, repeat_fraction=0.9)['tier']); "
                "print(mind.compute_plan(10**6)['why'])",
        native=True, aliases=("should this run on the gpu", "cpu or gpu decision", "route a computation",
                              "avoid recomputing", "beat the gpu", "compute dispatch policy",
                              "which backend should I use", "shrink the work"))

    c.register_capability("Convergence guard for adaptive sampling (is the CLT stop trustworthy here?)",
        "mind.convergence_guard(increments) checks the assumption the variance-based stop silently "
        "makes: adaptive_sample_budget's interval is exactly right for i.i.d. increments and exactly "
        "a lie for a pixel whose stream still carries structure -- drift (a caustic path being "
        "discovered) or correlated/periodic sampling. Measured trap on record: two streams with "
        "near-identical CLT half-widths (~0.008, both claiming converged), one with TRUE error 0.083 "
        "-- 10x its interval.",
        example="import numpy as np; r=np.random.default_rng(0); "
                "print(mind.convergence_guard(r.standard_normal(400)*0.1)['iid_ok']); "
                "print(mind.convergence_guard(r.standard_normal(400)*0.1+np.linspace(0,.2,400))['why'][:60])",
        native=True, aliases=("is this pixel really converged", "can I stop sampling", "clt assumption check",
                              "adaptive sampling guard", "simulation reached steady state check",
                              "settle detector", "render convergence check", "iid increments test"))

    c.register_capability("Settle-gated simulation runner (pay for dynamics, not equilibrium)",
        "mind.run_until_settled(step, state, steps) runs any simulation step function until the "
        "residual stream passes convergence_guard (i.i.d.: no drift, no order), then serves remaining "
        "frames from the settled state. MEASURED on the real fluid solver: a 600-frame decaying 64x64 "
        "shot settled at step 96 -- 504 frames served from equilibrium, 4.7x wall-clock, final-frame "
        "max error vs the fully-simulated ground truth 0.00e+00."
        "cloth settling, particle systems; pair with make_surrogate for settled-but-oscillatory regimes.",
        example="import numpy as np; r=mind.run_until_settled(lambda v: v*0.7, np.ones(32), steps=200); "
                "print(r['why'])",
        native=True, aliases=("stop simulating when settled", "simulation early exit", "fluid settle speedup",
                              "skip equilibrium frames", "softbody relaxation stop", "cloth settle",
                              "speed up my simulation", "graphics optimization pass")) 

    c.register_capability("Behavior pool (LOD for minds: tick 50k NPCs on one box)",
        "mind.behavior_pool() manages a population of ticking agents with behavior level-of-detail: "
        "an agent whose recent output stream certifies as an EXACT CYCLE (the symbolic surrogate "
        "contract) is demoted to a served cycle at near-zero cost; any input to that agent promotes "
        "it back to live ticking instantly; agents that never certify -- driven, chaotic, LEARNING -- "
        "are never demoted, and report() says which and why. pool.add(name, tick_fn, state); "
        "pool.step_all(inputs); pool.report()."
        "behavior costs what its information content costs.",
        example="p=mind.behavior_pool(window=48); p.add('npc', lambda st,inp: ((st or 0)%%4,(st or 0)+1), 0); "
                "[p.step_all() for _ in range(120)]; print(p.report()['why'])",
        native=True, aliases=("tick many agents cheaply", "npc crowd on one server", "behavior level of detail",
                              "agent pool", "mmorpg npc optimization", "demote idle npcs",
                              "game server agent scaling", "lots of npcs cheap", "npc archetypes share memory", "cohort compaction", "per region ai health", "zone behavior monitor"))

    c.register_capability("Streaming meters (live convergence guard and entropy, O(1) per sample)",
        "mind.stream_meter(window=256) returns an online meter: push(x) per sample or block (O(1) in "
        "stream length), verdict() runs the convergence guard on the live window -- BIT-IDENTICAL to the "
        "batch guard on the same bytes, the property the selftest pins -- and entropy() gives the live "
        "rate report. For audio blocks, simulation residuals, and agent action streams: the instruments "
        "run where the data is born instead of re-scanning history. Warmup answers iid_ok=None honestly. "
        "Over HTTP the meter travels as an object handle (call push/verdict via the handle registry).",
        example="sm=mind.stream_meter(window=64); import numpy as np; "
                "[sm.push(v) for v in np.random.default_rng(0).standard_normal(64)*0.1]; "
                "print(sm.verdict()['iid_ok'])",
        native=True, aliases=("online convergence check", "live entropy meter", "streaming guard",
                              "real time stream analysis", "audio block meter", "push samples get verdict",
                              "monitor a live stream", "incremental meters"))

    c.register_capability("Carrier-elevated deep trees (depth survives: readable leaves at depth 32)",
        "mind.encode_tree_carrier(tree) encodes each tree LEVEL on its own carrier, making depth "
        "contribution linear instead of geometric. Measured against the flat encoder's "
        "dim-independent wall (depth_probe): holistic separability moves d5 -> d7 and decays "
        "polynomially instead of snapping to 1.0; the real payoff is DEPTH-ADDRESSABILITY -- unbind "
        "level-d's carrier, strip the position tag, clean up: deep-leaf recovery 0.94-1.00 at depths "
        "7-32 where the flat encoding carries ZERO bits about the leaf."
        "depth must survive. The advisor (advise_scale) prescribes exactly this lever past depth 4.",
        example="v=mind.encode_tree_carrier(('add',('mul','x','y'),'z')); print(len(v), float((v*v).sum()))",
        native=True, aliases=("encode a deep tree", "tree too deep to encode", "carrier levels",
                              "depth addressable structure", "read a leaf from a deep tree",
                              "nested structure beyond depth limit", "deep hierarchy encoding", "why is my deep tree unreadable", "capacity warning at encode time", "unmix a bundle with sparse decoding", "recover all bundle members", "omp bundle readout"))

    c.register_capability("Fluid boundaries & performance (leStudio backlog: dtype, RGB, walls, ROI)",
        "The fluid stack is float32-clean end to end (P1: advect 2.62 -> 1.25 ms at 144x192; "
        "projection and diffuse preserve input dtype -- every float32 pipeline stays float32). advect "
        "accepts (H,W,C) fields sharing one backtrace (P2: RGB dye 7.23 -> 3.01 ms/step combined with "
        "P1), plus out= buffer reuse and roi=(y0,y1,x0,x1) windows (P4: sound for advection -- the "
        "backtrace is local; projection stays global; coarse-global + fine-local is the standard "
        "hybrid)."
        "projection only.",
        example="import numpy as np; z=np.zeros((32,32),np.float32); "
                "print(mind.fluid_step(z,z.copy(),z.copy(),boundary='wall')[0].dtype)",
        native=True, aliases=("fluid walls not torus", "canvas edges are walls", "ink wraps around the edge",
                              "mass conserving boundary", "advect rgb dye", "float32 fluid",
                              "solve only a region of the grid", "fluid roi", "document solid mask"))

    c.register_capability("Adaptive path tracing (CI-driven sampling: stop when the pixel is proven)",
        "mind.path_trace_adaptive(sdf, camera, tol=0.02) samples in blocks and stops each pixel when "
        "its CLT 95% half-width falls under tol*scale -- the statedemand stopping rule per pixel, "
        "valid because Monte-Carlo samples are iid by construction (scope stated, not assumed). Sky "
        "and flat regions stop at min_spp; edges and high-variance paths run to max_spp. MEASURED "
        "(lit sphere, 48x48): 84% of a flat 128-spp render's samples avoided, error 7x under the "
        "contracted tolerance, spp 16-112 spatially adaptive. Uses path_trace's own active mask -- "
        "the shipped tracer, not a fork.",
        example="import numpy as np; from holographic.rendering.holographic_render import Camera; "
                "img,rep=mind.path_trace_adaptive(lambda P: np.linalg.norm(P,axis=-1)-1.0, Camera(), "
                "width=24, height=24, max_spp=32); print(rep['why'])",
        native=True, aliases=("adaptive sampling render", "stop sampling converged pixels",
                              "render faster same quality", "variance based sampling",
                              "spend samples where noisy", "progressive render with a stopping rule"))

    c.register_capability(
        "Creature readability reports (webbing, silhouette gaps, per-part ids)",
        "the three instruments a creature rebuild is judged against. webbing_report counts "
        "NON-ADJACENT bone pairs with material in the corridor between them (one global implicit "
        "surface webs independent limbs together) -- 50/99 on the shipped quadruped, and it RESPONDS "
        "to blend (k=0.01 -> 10, k=0.30 -> 52). silhouette_report counts ENCLOSED negative-space "
        "holes (a blob scores 0; the quadruped scores 0). part_ids says which rig segment owns a "
        "point, for the flat per-part colour seam test. KEPT NEGATIVE: a corridor blocked by a third "
        "bone is `shielded` and excluded -- watch the TREND, not the absolute.",
        example="cr, _sdf = mind.creature(mind.quadruped_spec()); "
                "print(mind.creature_webbing_report(cr)['webbing_pairs'], "
                "mind.creature_silhouette_report(cr, res=64)['holes'])",
        native=True, aliases=("does my creature look like a blob", "webbing between limbs",
                              "limbs melting into the body", "measure creature readability",
                              "negative space between legs", "silhouette gaps", "webbing pairs",
                              "why does my creature look wrong", "is the skin blending too much",
                              "which part owns this point", "flat colour per part test"))


    c.register_capability(
        "One rig type (shared skeleton view + capability roles)",
        "there is not a creature rig and a humanoid rig. mind.rig(x) is the shared joints + "
        "per-segment bones + chains view over ANY skeleton, with canonical '<chain>#<index>' tags "
        "IDENTICAL to the skin's bone_of, so weights, reports and roles join with no translation "
        "table. rig_invariant pins one bone = one rigid segment (no bending mid-shaft) on creature "
        "AND humanoid. rig_roles labels foot/tip/torso from geometry. KEPT NEGATIVE: "
        "find_by_role is the EXACT dict and is authoritative; the VSA unbind path is exact at 16 "
        "segments but recall falls to 0.04 by 128 (module docstring has the table).",
        example="cr, _s = mind.creature(mind.quadruped_spec()); print(mind.rig_invariant(cr)); "
                "print(mind.rig_roles(cr).find_by_role('foot'))",
        native=True, aliases=("one rig for creatures and humanoids", "shared skeleton type",
                              "which segments are feet", "tag parts by role", "capability tags",
                              "does my rig have one bone per segment", "bone bends in the middle",
                              "find the legs on any body plan", "unify creature and humanoid rig",
                              "segment tags", "how big is my creature", "reference length of a rig"))


    c.register_capability(
        "Creature skin as a composition tree (metaball groups)",
        "the fix for limbs melting into the torso. The old skin summed everything into ONE global "
        "field with ONE blend radius, so anything near anything else blended. creature_tree compiles "
        "the rig into the existing SDF DSL: parent-child segments blend at their shared joint, all "
        "else HARD-unions, so webbing between unrelated limbs is UNEXPRESSIBLE. MEASURED: webbing "
        "76 -> 0, negative space 0.130 -> 0.443. Returns an SDF; default-off. Joint blend is "
        "RELATIVE to the limb (D-7). KEPT NEGATIVE: an ABSOLUTE 0.30 blend still webs (58 smooth / "
        "60 exact-fillet) -- the bounded operator did not rescue it.",
        example="cr, _s = mind.creature(mind.quadruped_spec()); sdf = mind.creature_tree(cr); "
                "print(mind.creature_webbing_report(cr, field=sdf)['webbing_pairs'])",
        native=True, aliases=("limbs melt into the body", "stop the skin blending everything",
                              "metaball groups", "webbing between limbs fix", "creature sdf tree",
                              "composition tree skin", "blend only at joints",
                              "why do my legs merge together", "separate limbs from the torso",
                              "creature skin without melting"))


    c.register_capability(
        "Volumetric tissue: bone, muscle, fat, skin as nested fields",
        "real anatomy, not a shading trick. tissue_fields returns a nested SDF per tissue, grown "
        "OUTWARD from bone -- set muscle and fat PER BONE and the skin falls out, which is why one "
        "skeleton can be a whippet or a bulldog. tissue_at(P) names the tissue at a point; "
        "anatomy_report checks bone-in-muscle-in-fat-in-skin (0/396 violations). "
        "tissue_visible_field hides layers and/or cuts with a plane -- hide the skin and the WHOLE "
        "skeleton shows in place, no separate geometry. ORGANS are metaballs in anatomy space (the "
        "one place metaballs are right), fitted inside muscle with bone subtracted.",
        example="cr, _s = mind.creature(mind.quadruped_spec()); f = mind.tissue_fields(cr); "
                "print(mind.anatomy_report(cr, fields=f)['fractions'])",
        native=True, aliases=("see inside my creature", "show the skeleton", "muscle and fat sliders",
                              "what tissue is at this point", "cross section of a creature",
                              "hide the skin", "cutaway view", "layers of anatomy",
                              "is the bone inside the skin", "skin weights from anatomy",
                              "fat belly thick thighs", "x-ray my creature", "put organs in my creature",
                              "where is the heart", "viscera", "internal organs"))


    c.register_capability(
        "Hybrid body plans (a centaur is a spec, not a code path)",
        "a limb can mount ON another limb ({'on': 'torso', 'u': 0.85}) instead of on the spine, which "
        "is the one structural thing a hybrid needs. mind.centaur_spec() is a horse with a humanoid "
        "torso and arms on it -- and NOTHING in the engine knows what a centaur is. VERIFIED end to "
        "end on the same code as a quadruped: 26 segments, webbing 0, negative space 0.585, anatomy "
        "nesting 0 violations, and it WALKS on its four horse legs (slip 7.0% of stride) with the "
        "arms correctly not treated as legs. Chains can be named for readability; unnamed keeps the "
        "old L0/L0m tags byte-for-byte.",
        example="cr, _s = mind.creature(mind.centaur_spec()); print(mind.rig_invariant(cr)['segments'], "
                "mind.gait_report(cr)['slip_ratio'])",
        native=True, aliases=("centaur", "hybrid creature", "half horse half human",
                              "mount a limb on another limb", "torso on a quadruped",
                              "minotaur", "mermaid", "chimera body plan", "arms on a horse"))


    c.register_capability(
        "Body-relative sizing (texture that survives a resize)",
        "a spatial frequency must declare its reference length, or the same animal wears finer skin "
        "just for being bigger: tripling a creature took an insect from 17 sclerite plates to 38. "
        "Pass creature= to creature_material and the pattern is sized BY THE BODY. scale_invariance_probe "
        "is the generalised check: this bug hit DISTANCES four times. rotation_invariance_probe is its "
        "directional twin -- it hit DIRECTIONS three more (mirror plane, limb dir, spine arch). "
        "Neither knows which quantities SHOULD vary: shape belongs to the BODY frame, gravity to the "
        "WORLD.",
        example="cr, _s = mind.creature(mind.quadruped_spec()); "
                "mat = mind.creature_material('insect', creature=cr)",
        native=True, aliases=("my creature breaks when I rotate it", "rotation invariance check",
                              "which frame should this be in", "body relative or world relative",
                              "my scales get smaller when I resize", "texture changes with body size",
                              "how many scales across the body", "physical scale size",
                              "keep the texture the same when scaling", "body relative texture",
                              "does this survive a resize", "scale invariance check",
                              "absolute vs relative units"))


    c.register_capability(
        "One pipeline, three bodies (the unification regression)",
        "the honest test of D-1 is not that a quadruped still works -- it is that a HYBRID and a rig "
        "recovered from a POINT CLOUD walk the same calls. creature_regression_report runs one "
        "skin/tissue/readability pipeline over a quadruped, a centaur and a fitted rig: 0 webbed "
        "pairs and 0 nesting violations on each, 16/26/4 segments. rig_from_primitives closes the "
        "loop -- a fitted capsule IS a bone segment, so observe and generate produce the same type. "
        "KEPT NEGATIVE: a fitted rig has no spine so it gets no organs, reported not hidden.",
        example="print(mind.creature_regression_report(res=48))",
        native=True, aliases=("does the centaur work", "hybrid body plan test",
                              "rig from a point cloud", "fitted rig", "close the loop",
                              "photo to creature rig", "do all body plans use one code path",
                              "regression specs", "prove the unification"))


    c.register_capability(
        "Recover a body from a mesh (spine, thickness, inferred tissue)",
        "the observe half of the pipeline. rig_from_mesh takes the medial-axis centerline as a real "
        "SPINE chain carrying the medial radius -- the shape's own thickness measurement -- so a "
        "scanned body gets parented segments, joint blending, anatomy space AND organs, none of "
        "which rig_from_primitives could give it. infer_tissue_fractions derives muscle/fat from the "
        "gap between fitted bone and observed skin, body_params-shaped so an inferred body drives "
        "tissue_fields like an authored one. KEPT NEGATIVES: single-branch (torso, not limbs); the "
        "muscle/fat split is not observable from a silhouette.",
        example="cr, _s = mind.creature(mind.quadruped_spec()); "
                "lo, hi = mind.rig(cr).extent(); "
                "mesh = mind.mesh_from_sdf(mind.creature_tree(cr), (tuple(lo-0.1), tuple(hi+0.1)), res=32, vectorized=True); "
                "rig, thick = mind.rig_from_mesh(mesh, res=24); print(len(rig.tags))",
        native=True, aliases=("find the spine of a mesh", "recover a rig from a scan",
                              "how thick is this body", "infer muscle and fat",
                              "backbone from geometry", "reverse engineer a creature",
                              "mesh to rig", "medial axis to spine"))


    c.register_capability(
        "Scaffold meshing (a cage on the skeleton, projected onto the field)",
        "the fix for lumpy limbs. Marching cubes sizes ONE grid for the whole body, so a thin limb "
        "gets a couple of cells across it and beads; a scaffold's density follows the SKELETON. "
        "MEASURED on the quadruped's thinnest segment: radial ripple 25.4% (marching, res 40) -> 1.6% "
        "(scaffold), with FEWER verts (7,570 vs 10,754), landing on the isosurface to 1e-16. A "
        "composition, not a new mesher: skin_skeleton + shrinkwrap_field + creature_tree. KEPT "
        "NEGATIVE: closest-POINT, so a cage vertex nearer a neighbouring limb is pulled onto it.",
        example="cr, _s = mind.creature(mind.quadruped_spec()); "
                "out = mind.creature_scaffold_mesh(cr); print(len(out['mesh'].vertices))",
        native=True, aliases=("my limbs look lumpy", "beading on thin limbs", "better mesh than marching cubes",
                              "quad cage on a skeleton", "project a mesh onto an sdf",
                              "scaffold polygonisation", "retopo onto a field", "shrinkwrap onto a field"))


    c.register_capability(
        "Creature readability: proportion as a SEARCH, not a rule table",
        "per Togelius et al.'s search-based PCG, quality comes from an evaluation function you SEARCH, "
        "so this scores specs with the metric already trusted for the field rebuild rather than "
        "hand-coding proportions. TWO TERMS, because one is degenerate: negative space alone is "
        "MONOTONE in limb thickness (0.470 -> 0.332), so maximising it yields a spider-legged wisp; "
        "mass dominance runs the other way (0.817 -> 0.516), giving an interior optimum. Webbing is a "
        "hard GATE, not a term. Also grounds a creature (A-4) so it reads as an animal.",
        example="print(mind.creature_proportion_search(mind.quadruped_spec(), res=48)['chosen'])",
        native=True, aliases=("my creature looks wrong but I don't know why", "proportion rules",
                              "does this read as a creature", "make the limbs subordinate",
                              "stand my creature on the ground", "line of action",
                              "my creature has no neck", "the head does not read", "head to body ratio",
                              "score a body plan", "big shape small shape", "why does it look spindly"))


    c.register_capability(
        "Make a creature (spec -> body, skin, mesh, parts, grounded)",
        "THE entry point. build_creature(spec) runs the whole pipeline: rig, metaball-group skin, "
        "scaffold mesh, role-driven part sockets (feet on whatever the feet are, eyes and mouth on the "
        "head, ANY body plan, no per-plan table), grounding and a readability score. It exists because "
        "dogfooding found find_capability('make a creature') returned the parts library, the body-shape "
        "module and the editor -- everything except how to make one. KEPT NEGATIVE: a placed foot does "
        "not yet make a leg READ as footed; the limb's capsule already caps that space (0.58% of "
        "pixels change).",
        example="out = mind.build_creature(mind.quadruped_spec(), quads=True); "
                "print(len(out['mesh'].faces), out['quads']['quad_fraction'])",
        native=True, aliases=("make a creature", "create a creature", "design a monster from scratch",
                              "build an animal", "generate a creature from a spec", "creature pipeline",
                              "how do I make a creature", "put feet and eyes on my creature",
                              "attach parts to a body", "creature with quads and lods",
                              "retopologise my creature", "game ready creature mesh",
                              "make my creature fat", "obese creature", "pot belly", "fat tummy",
                              "how do I control body fat", "muscular creature",
                              "my creature loses legs when I rotate it", "spine along a different axis",
                              "body relative limb directions", "dir_space body"))


    c.register_capability(
        "Ratios that carry their denominator (and role-driven part coverage)",
        "measured_ratio(n, d, of=...) will not state a percentage without naming what it is a percentage "
        "OF -- the measurement twin of D-7's reference length. It exists because 'parts change 0.58% of "
        "pixels, so parts do not read' was ONE mistake: that was 0.58% of the whole IMAGE (~95% "
        "background); against the BODY the same parts add 11% of silhouette. Alongside it, "
        "creature_auto_sockets places parts by ROLE -- ground tip -> foot, LATERAL tip -> hand, head -> "
        "eyes/mouth. One rule set: quadruped 4+0, centaur 4+2, humanoid 2+2.",
        example="print(mind.measured_ratio(694, 6162, of='body silhouette')['value']); "
                "cr, _s = mind.creature(mind.quadruped_spec()); print(len(mind.creature_auto_sockets(cr)))",
        native=True, aliases=("percent of what", "state the denominator", "is this ratio meaningful",
                              "put hands on the arms", "where do parts go on any body plan",
                              "horns and ears and spikes", "part coverage by role"))


    c.register_capability(
        "Convolution surfaces (hands, feet, digits without bulges)",
        "the right tool for extremities. Bloomenthal & Shoemake 1991: summing convolutions of "
        "CONTIGUOUS skeletal primitives gives NO bulge at joints (superposition), where a "
        "smooth_union's blend IS the bulge -- measured, right-angle corner 0.1065 vs 0.0788, 26% "
        "less. Fuentes Suarez/Hubert/Zanni 2019 adds ELLIPSOIDAL sections so a sole is flat (4:1) "
        "without extra primitives. KEPT NEGATIVE, Bloomenthal's own: convolution SUMS, so separate "
        "digits still web (3-toe fan: 1 blob summed, 3 grouped) -- grouping is required.",
        example="g, _a = mind.foot_skeleton(digits=3); f = mind.convolution_groups(g); "
                "print(float(f([[0.0, 0.0, 0.05]])[0]))",
        native=True, aliases=("how do I model a hand", "fingers and toes", "no bulge at joints",
                              "convolution surface", "skeleton to smooth surface",
                              "why do my joints bulge", "flat sole section", "model a foot properly"))


    c.register_capability(
        "Crystal FORMS ({hkl} means every equivalent face)",
        "in crystallography braces denote the FORM -- all symmetry-equivalent faces -- so cubic {100} "
        "is SIX faces (a cube) and {111} is EIGHT (an octahedron). crystal_habit takes EXPLICIT faces "
        "and adds only the centrosymmetric pair, so asking for {100} and expecting a cube silently "
        "gives a SLAB: measured volume 5.76 where a cube is 1.00, and not invariant under a 90-degree "
        "turn (pointwise field difference 0.97). Pass form=True and you get a real cube (1.0022) and "
        "octahedron (0.8495 vs 0.866 analytic), symmetric to 1e-16.",
        example="s = mind.crystal_habit('cubic', ((1,0,0),), 0.5, form=True)",
        native=True, aliases=("make a cube crystal", "crystal form vs face", "miller indices braces",
                              "octahedron crystal", "why is my crystal a slab", "crystal symmetry faces"))


    c.register_capability(
        "Grow crystals (on a surface, as a cluster, in a geode, gated by a field)",
        "crystal_habit gives the SHAPE a lattice permits; this places it. One rule covers every case: "
        "A CRYSTAL GROWS PERPENDICULAR TO ITS SUBSTRATE, because a crystal pointing away from the wall "
        "keeps reaching fresh solution while one lying flat gets buried. So a druse radiates, and a "
        "geode is the SAME call with inward=True. `where` takes a weight FIELD, so crystals grow only "
        "where a material says -- measured, gating lifted the mean field value under the crystals from "
        "0.313 to 0.577. Geode measured hollow: 0.00 filled at centre, 1.00 rind.",
        example="g = mind.crystal_geode(radius=0.7, count=40); "
                "print(float(mind.crystal_cut(g).cut_face_normal[0]))",
        native=True, aliases=("grow crystals on a surface", "crystal cluster", "druse",
                              "make a geode", "crystals inside a rock", "crystals in a cavern",
                              "crystals only where the material is", "crystal growth",
                              "geode cross section", "crystal lining a cavity"))


    c.register_capability(
        "Beer-Lambert absorption (why a thick gem is darker than a thin one)",
        "light is attenuated by the DISTANCE it travels INSIDE a transmissive solid, so depth reads: a "
        "gem's thick parts come out darker and more saturated than its edges. Albedo alone tints once "
        "per interaction and cannot tell a thick crystal from a sliver, which is why gems looked like "
        "coloured glass. The path tracer takes sigma per RGB as an 8th material channel and uses the "
        "interior path length it already computes for refraction. MEASURED on glass pixels: sigma=0 "
        "gives (0.93,0.93,0.96), absorbing gives (0.11,0.10,0.16) -- darker AND hue-shifted, since "
        "each channel is absorbed at its own rate.",
        example="cb = mind.material_trace_channels('amethyst'); "
                "print(mind.material_absorption('amethyst'))",
        native=True, aliases=("beer lambert", "absorption through glass", "why is my gem flat",
                              "thick crystal darker", "extinction coefficient", "gem depth",
                              "transmissive material depth", "attenuate light through a solid"))


    c.register_capability(
        "Crystal imperfections (cloudiness, inclusions, phantoms, fractures, chips)",
        "a PERFECT crystal reads as glass; specimens read as mineral because they are flawed in "
        "structured ways. All but chipping are MATERIAL modifiers -- they change absorption, albedo "
        "and roughness by position and leave the shape alone, so they compose with any habit and any "
        "growth mode. Cloudiness is SCATTERING not pigment, so it raises absorption NEUTRALLY across "
        "RGB (measured [1.92,1.92,1.92]) and desaturates 0.571 -> 0.296. Phantoms are the same habit "
        "scaled down, so they hug it to |d| 0.011. Chips only ever subtract.",
        example="cl = mind.crystal_cloudiness(seed=1); "
                "cb = mind.crystal_flawed_material('quartz', cloud=cl)",
        native=True, aliases=("milky quartz", "cloudy crystal", "inclusions in a crystal",
                              "rutilated quartz", "phantom quartz", "crystal flaws",
                              "imperfections in a gem", "chipped crystal", "fractures in quartz"))


    c.register_capability(
        "Render a specimen (adaptive trace + denoise + graded, one call)",
        "trace -> G-buffer -> SVGF denoise -> firefly clamp -> SEARCHED exposure, composed. `tol` "
        "replaces a sample count: state the quality and converged pixels stop being sampled (measured "
        "83% of a flat 48-spp render's samples avoided at equal mean radiance; grain 0.080 -> 0.032 "
        "after denoise). Every step already shipped separately and was hand-wired per render, which is "
        "how the adaptive tracer went unused by an entire crystal arc. CAUTION: at min_spp=8 the block "
        "CI is optimistic and pins spp at the floor; 16 escalates properly.",
        example="img, rep = mind.render_specimen(sdf, (1.3,0.7,1.5), (0,0,0), mat, "
                "mind.sky_model(hour=10.0), width=48, height=40); print(rep['sample_saving'])",
        native=True, aliases=("render a crystal", "render until converged", "one call render",
                              "denoised render", "adaptive sampling render", "how do I render a gem",
                              "trace denoise and tone map"))


    c.register_capability(
        "Pose a creature mid-stride, and budget a render before running it",
        "build_creature(pose=0.25) builds the body MID-WALK: the legs are IK-solved so the feet land "
        "where the gait says, and support then comes from the gait's CONTACT set instead of geometry "
        "-- a walking body legitimately lifts a foot, and the static test called that unstable "
        "(supported False on a normal walk cycle). render_plan MEASURES cost on a tiny tile instead of "
        "extrapolating: four overruns here came from linear estimates, which understate glass because "
        "more samples means more rays marching through interiors.",
        example="o = mind.build_creature(mind.quadruped_spec(), pose=0.25, cage_res=16); "
                "print(o['ground']['planted'], o['ground']['supported'])",
        native=True, aliases=("pose a creature", "walking creature", "creature mid stride",
                              "how long will this render take", "render budget",
                              "will this render finish", "animate and build"))


    # --- CYCLE CERTIFICATE. Measured misses: "does this repeat" returned the Mobius fraction, "is it
    # cycling" returned token sampling, "limit cycle detection" returned the creature gait.
    c.register_capability(
        "Cycle certificate (does this sequence repeat, and at what period)", "the SMALLEST period at which every "
        "recent frame matches the one p back, certified at a numeric tolerance -- or certified=False, never a best "
        "guess. Works on any sequence, not just a simulation: a REGIME STREAM is a square wave that a harmonic fit "
        "rings on (NRMSE 0.584) while an exact cycle certificate replays it at 0.037",
        example="c = mind.certify_cycle(series.reshape(-1, 1), tol=0.15); print(c['period'], c['certified'])",
        native=True, aliases=("does this repeat", "is it cycling", "find the period of a sequence",
                              "limit cycle detection", "period of a repeating state",
                              "has the simulation started looping", "detect a loop", "cycle period"))


    # --- MULTI-TONE GENERATOR. Measured misses: "two tones at once" returned the Zig raymarcher,
    # "beating oscillator" returned the modal jump solver, "multi tone fit" returned the sentinel.
    c.register_capability(
        "Multi-tone generator (independent, incommensurate frequencies)", "fit a signal as a sum of INDEPENDENT "
        "sinusoids -- the generator class the harmonic fit cannot express, since that one fits harmonics of ONE "
        "fundamental and refuses on incommensurate tones (beating oscillators, two-rotor vibration, tidal "
        "constituents). Greedy matching pursuit with off-grid refinement, deliberately NOT a sparse solve over a "
        "frequency dictionary: a dense dictionary is coherent and blows up across its density parameter",
        example="m = mind.fit_multitone(x, n_tones=3); print(m['ok'], [1/f for f in m['frequencies']])",
        native=True, aliases=("two tones at once", "incommensurate frequencies", "beating oscillator",
                              "sum of sinusoids", "multi tone fit", "several frequencies at once",
                              "two rotors", "fit multiple sine waves", "not a harmonic stack"))


    # --- THE HRNN FRONT DOOR. Measured misses, all of them the phrasing a NON-SPECIALIST would type:
    # "analyse my time series" -> Arrow of time; "is my data predictable" -> Cold storage;
    # "what should i do with this stream" -> Compute plan; "explain this signal" -> Event study.
    # The ladder was reachable only by someone who already knew it existed.
    c.register_capability(
        "Explain a stream (one call: what it is, what to do, what it will NOT do)", "hand it any 1-D series and get "
        "plain English back: whether it has a GENERATOR (a few floats that reproduce and extend it), real "
        "STRUCTURE (predictable but not reducible), is INCOMPRESSIBLE (independent facts -- do not fit a model), "
        "or UNMEASURED (a refusal, not a finding). Carries the recommended next call as runnable code, the honest "
        "refusal, the measured evidence, a predict() callable when there is one, and the verdict as a hypervector",
        example="r = mind.explain_stream(series); print(r['headline'], r['what_to_do'], r['wont_do'])",
        native=True, aliases=("analyse my time series", "what should i do with this stream",
                              "is my data predictable", "explain this signal", "what is this data",
                              "should i fit a model to this", "does my data have a pattern",
                              "understand a signal", "one call for a stream", "analyse a series",
                              "is this signal random", "what kind of data is this"))


    # --- FLEET ANOMALY BY VERDICT ALGEBRA. Measured misses: "is this sensor behaving like the others"
    # returned the spectrum Observer, "which machine is misbehaving" returned the hardware model,
    # "compare sensors in different units" returned conditional statistics.
    c.register_capability(
        "Fleet anomaly (compare sensors by STRUCTURE, across units)", "summarise a whole cohort of streams as ONE "
        "hypervector, then ask whether a stream behaves unlike its cohort. Compares STRUCTURE, not values, so it "
        "is EXACTLY invariant to scale, offset and sign -- a pressure sensor and a temperature sensor are "
        "directly comparable with no normalisation and no per-sensor calibration, and the signature does not grow "
        "with cohort size. Catches DRIFT, which amplitude and spectral baselines miss. Kept negative: a FLATLINE "
        "is not caught (a constant IS a generator) -- pair it with an amplitude check",
        example="sig = mind.fleet_signature(streams); r = mind.fleet_anomaly(new_stream, sig); "
                "print(r['score'], r['floor'], r['anomalous'])",
        native=True, aliases=("is this sensor behaving like the others", "which machine is misbehaving",
                              "compare sensors in different units", "fleet anomaly", "cohort outlier",
                              "which sensor is faulty", "sensor drift across a fleet",
                              "compare many streams", "is this one different from the rest"))


    # --- CONVERGENCE ACCELERATION. Measured misses: "make my solver converge faster" returned the
    # compute plan, "skip iterations" returned the settle-gated runner (which stops at equilibrium --
    # this PREDICTS the limit before reaching it).
    c.register_capability(
        "Convergence acceleration (jump to a solver's limit, or decline)", "a convergence sequence is a STREAM, "
        "so the ladder's question applies to it: does it have a generator? When it does, three iterates give the "
        "limit in closed form -- measured 7 iterations where plain iteration needed 70, to machine precision. A "
        "jump is taken ONLY if it VALIDATES against one more step, because naive extrapolation on a multi-mode "
        "solve measured 250x WORSE than simply iterating. Works on any fixed-point iteration: relaxation sweeps, "
        "physics settling, IK passes",
        example="r = mind.accelerate_convergence(step_fn, x0); print(r['iters'], r['jumps'], r['why'])",
        native=True, aliases=("make my solver converge faster", "skip iterations", "jump to the answer",
                              "accelerate an iterative solve", "extrapolate a fixed point",
                              "fewer iterations", "speed up relaxation", "converge in fewer steps"))


    # ---------------------------------------------------------------- HDRIFT: generative media
    c.register_capability(
        "Holographic drift generative model (HDRIFT: train on points, generate by drift)",
        "the generative model AS d+1 moment hypervectors: mind.drift_train(points) encodes ONCE (bandwidth "
        "probed from the data; a collapsing dataset is REFUSED, not served as a mean-generator) and "
        "mind.drift_generate samples by particle drift read off the vectors by dot products -- attraction to "
        "the data field minus repulsion from the batch's own field (the corrective for the measured "
        "attraction-only memorisation, max-cos 1.000). No adversary, no backprop, no learned weights; field "
        "cost is independent of N. labels= packs every class into ONE vector set; condition= unbinds one",
        example="mdl = mind.drift_train(pts); X = mind.drift_generate(mdl, n=32); print(mind.generation_audit(X, pts))",
        native=True, aliases=("train a generative model", "generate new samples like my data", "gan",
                              "holographic gan", "hgan", "drift model", "generative model without a discriminator",
                              "sample from a learned distribution", "make more data like this",
                              "conditional generation by label"))

    c.register_capability(
        "Drift model algebra (compose + ablate + transport trained models)",
        "the verbs no per-dataset-trained generator has, each a vector operation because the model IS "
        "vectors: mind.drift_compose(a, b) MERGES two models trained separately (moments add, evidence-"
        "weighted); mind.drift_ablate(a, b) REMOVES b's contribution (unlearning / a negative prompt with "
        "no retraining -- exact when b's data is a subset of a's, an approximation otherwise, stated); "
        "mind.drift_transport(m, delta) MOVES the whole distribution by shift-is-a-bind with the first-"
        "moment cross-term the naive shift drops. Models must share one encoder space (enforced)",
        example="ab = mind.drift_compose(a, b); mind.drift_generate(ab, n=16)",
        native=True, aliases=("combine two trained models", "merge generative models", "subtract a model",
                              "make the model forget", "unlearn a class", "negative prompt",
                              "shift a distribution", "move a trained model", "model arithmetic"))

    c.register_capability(
        "Generation audit (memorisation + coverage gate)",
        "novelty and mode coverage of generated samples against their training set in ONE report, because "
        "memorisation manifests as SUCCESS (perfect samples) and fixing it usually costs coverage -- so "
        "both are measured together. novelty ~0 = memorised (nearest-training distance in units of the "
        "training set's own NN scale); coverage = fraction of k data modes some sample lands nearest to. "
        "mind.generate_media attaches this automatically; nothing generated should ship without it",
        example="a = mind.generation_audit(samples, train); print(a['novelty_mean'], a['coverage'])",
        native=True, aliases=("is my model memorising", "novelty of generated samples", "mode collapse check",
                              "coverage of modes", "did it just copy the training data",
                              "overfitting check for a generator"))

    c.register_capability(
        "Train on images and generate more (media in, media out)",
        "mind.train_media_model(images) fits each image to k anisotropic splats (hand-derived-gradient "
        "Adam) and drifts in SPLAT-PARAMETER space -- dozens of dimensions, not thousands, which is the "
        "curse-of-dimensionality answer the 2026 drifting papers solve with a frozen network encoder. "
        "mind.generate_media(model, meta, n) drifts new splat sets and renders them, ALWAYS attaching the "
        "generation audit when audit_train is given. HONEST v1 SCOPE: generated images render isotropic "
        "(soft-edged) splats; the aniso structure is not yet carried through the drift space",
        example="mdl, meta = mind.train_media_model(images, k=8); out = mind.generate_media(mdl, meta, n=4)",
        native=True, aliases=("train a model on my images", "generate images like these",
                              "make more images like this folder", "image generation from examples",
                              "learn the style of these pictures", "media generation"))

    c.register_capability(
        "Write a WAV audio file",
        "mind.write_wav(path, samples, rate) writes float samples in [-1,1] to 16-bit PCM -- the OUT half "
        "of read_wav, shipped in holographic_audio all along but never wired to the mind (a generation "
        "pipeline that cannot emit audio is not a pipeline). Round-trips read_wav to 1/32768",
        example="mind.write_wav('/tmp/tone.wav', np.sin(np.linspace(0, 2*np.pi*440, 8000)), 8000)",
        native=True, aliases=("write a wav audio file", "save audio to a file", "export sound",
                              "emit a wav", "audio output file"))

    c.register_capability(
        "Auto-scale a drift model's knobs (dim x bandwidth through auto_scale)",
        "mind.drift_autoscale(points) routes HDRIFT's two knobs through the mind's EXISTING auto_scale "
        "loop -- eval is the bandwidth prober's spread-fidelity at the current operating point; the most "
        "responsive knob is doubled until the target is met or a WALL is named (no knob helps: stop and "
        "say so). No private tuner grown; every step in the trajectory carries the probe that justified it",
        example="traj = mind.drift_autoscale(pts, target_spread=0.9); print(traj)",
        native=True, aliases=("tune the drift model automatically", "autoscale generative knobs",
                              "pick dim and bandwidth for me", "scale the generator"))


    # ---------------------------------------------------------------- VOID-1: exploring the unknown
    c.register_capability(
        "Void explorer (what the corpus implies but does not contain)",
        "'undiscovered' as a MEASURED set, three warrants: mind.void_map finds bootstrap-null-gated "
        "low-density regions inside the support (sparsity the data's own noise explains is never called "
        "void; the instrument probes its own sharpest honest bandwidth -- the sampler's smooth kernel "
        "smears absence); mind.structured_voids is the Mendeleev move -- combinations every observed "
        "pairwise slot co-occurrence licenses but the full set lacks, REFUSED when the structure "
        "cannot beat a shuffle; mind.transfer_voids: present in B, absent in A -- instantiated "
        "elsewhere, the cross-disciplinary warrant",
        example="vm = mind.void_map(mdl, pts); sv = mind.structured_voids(rows); tv = mind.transfer_voids(a, b)",
        native=True, aliases=("explore the unknown", "find gaps in my knowledge", "what is missing from my data",
                              "undiscovered combinations", "mendeleev gaps", "predict missing entries", "my data is missing something", "what is my data missing",
                              "predict entries that should exist", "what should exist but does not",
                              "what does one dataset have that the other lacks", "map the voids",
                              "find holes in the dataset", "unknown unknowns in my corpus"))


    # ---------------------------------------------------------------- RESID-1: noise as unexplained data
    c.register_capability(
        "Residual explorer (noise is data without an explanation yet)",
        "'noise' is unexplained structure until matched nulls say otherwise: mind.residual_verdict "
        "explains a series, subtracts, and judges the remainder against AAFT AND a block shuffle -- "
        "'structured' only past both, else 'irreducible' with p-values (an efficient market's residual "
        "SHOULD read irreducible); mind.support_gauge: a CAUSAL inside/sparse/void monitor per step, "
        "the void closing as the trailing window absorbs it; mind.hidden_drivers: a common factor in a "
        "panel's RESIDUALS beyond surrogated nulls -- the puppet string no single series discloses",
        example="rv = mind.residual_verdict(y); g = mind.support_gauge(y); hd = mind.hidden_drivers(panel)",
        native=True, aliases=("noise is not noise", "structure hidden in the noise", "puppet strings in market data",
                              "the noise has patterns", "is the leftover signal meaningful",
                              "structure in my residuals", "common cause across my sensors",
                              "hidden influences across many series", "is this residual real or noise",
                              "am I outside anything the model has seen", "market state never seen before",
                              "common driver behind correlated moves", "unexplained co-movement",
                              "external factor influencing my data"))


    c.register_capability(
        "Dependence voids, the residual ladder, and one merged watch timeline",
        "mind.panel_gauge catches the void a single-series gauge cannot see: the state is the Fisher-z "
        "trailing CORRELATION structure, gauged causally -- a correlation crisis leaves all history "
        "while every marginal sleeps (planted and proven; outside the history's own bounding box is "
        "void BY GEOMETRY, never clipped). mind.residual_ladder climbs a structured residual through "
        "the next grammar (closed-form AR rung) until a rung prices it as noise or admits "
        "'rungs-exhausted'. mind.stream_watch merges sentinel regime events and gauge void/recovered "
        "events into ONE time-ordered timeline",
        example="pg = mind.panel_gauge(panel); rl = mind.residual_ladder(y); sw = mind.stream_watch(y)",
        native=True, aliases=("correlation regime change", "correlations all jumped together",
                              "relationships between my streams changed", "assets crashing at the same time",
                              "dependence structure never seen before", "climb the residual",
                              "which model finally explains the noise", "one timeline for all stream events",
                              "watch a stream for regime and void events", "2008 style correlation crisis",
                              "who leads whom in a panel", "lead lag relationship changed",
                              "tail dependence crash together", "volatility memory garch"))


    c.register_capability(
        "Market residual report (the stylized facts, measured on checked-in data)",
        "mind.market_residual_report runs the residual ladder over the vendored real datasets and "
        "names which grammar terminates each stream. First run reproduced finance's stylized facts "
        "with no market knowledge in the code: 1h returns level-clean but scale-structured "
        "(volatility clustering; the vol rung terminates), tick moves fire the AR rung with a "
        "NEGATIVE lag-1 coefficient (the bid-ask bounce, ~-0.21), tiny-n returns read irreducible "
        "(the EMH at acknowledged low power), and price levels are an AR fit's favourite meal. "
        "Slow-ish (surrogate ensembles per stream); the selftest pins a reduced pass",
        example="rep = mind.market_residual_report(); print({k: v['terminal'] for k, v in rep.items()})",
        native=True, aliases=("stylized facts of markets", "run the ladder on real market data",
                              "volatility clustering in real returns", "bid ask bounce",
                              "which model explains real returns", "efficient market check on real data"))


    c.register_capability(
        "Transit hunter (box-matched period search with a matched null)",
        "mind.transit_search: phase-coherent period search with Box Least Squares -- the BOX-matched "
        "filter, measured 6.3x more peak contrast than the sinusoid template near the detection floor, "
        "where planets are lost. Verdicts vs the block-shuffle null (red noise survives, phase "
        "coherence dies; the iid null flags red noise as planets -- reported, not used); harmonic "
        "families reported; an impassable p-floor refuses. mind.transit_detection_floor: the "
        "detection-limit curve with per-transit SNR. The ladder gained a fold rung: comb detects, "
        "BLS names, the folded median consumes",
        example="r = mind.transit_search(t, flux, 60, 400); print(r['verdict'], r['period'], r['family'])",
        native=True, aliases=("find a transit in a light curve", "exoplanet transit search",
                              "fold on the holographic substrate", "kernel fold uneven sampling",
                              "how faint a signal can you detect", "how faint a signal can you still detect", "subtract the periodic part",
                              "remove a known period from a series",
                              "box least squares", "periodic dip detection", "detection limit curve",
                              "find the period of repeating dips", "phase coherent period search",
                              "fold a residual at its period"))


    c.register_capability(
        "Pulsar panel (Hellings-Downs pattern test with a sky-scramble null)",
        "mind.hd_search asks the NANOGrav question of a panel of timing residuals: whiten each "
        "series (raw red-vs-red correlations are spurious, pinned), correlate every pair, judge the "
        "pattern with TWO matched nulls -- AAFT per series (does ANY cross-correlation exist) and the "
        "SKY SCRAMBLE (positions permuted against residuals: correlations survive, geometry dies). "
        "Verdicts: hd-consistent / correlated-not-sky-patterned (the monopole clock-error diagnosis) "
        "/ independent; amplitude a stated lower bound, the certified quantity is the curve SHAPE. "
        "mind.hd_panel_demo plants ground truth (hd | mono | none)",
        example="p, pos = mind.hd_panel_demo(); r = mind.hd_search(p, pos); print(r['verdict'], r['shape'])",
        native=True, aliases=("gravitational wave background", "hellings downs curve",
                              "correlated pulsar timing residuals", "pulsar timing array analysis",
                              "is the correlation explained by sky geometry", "sky scramble test",
                              "common signal across pulsars", "quadrupole correlation pattern",
                              "clock error versus gravitational waves"))


    c.register_capability(
        "Spectroscopist's bench (lines, identity with abstention, redshift verdict, decay)",
        "mind.spectral_lines: median continuum off, candidates gated against a max-hunting "
        "noise-only bootstrap (a permutation null contains its own lines -- pinned), sub-bin "
        "centers; with a catalog, cleanup-with-margin identification that ABSTAINS between lines. "
        "mind.redshift_verdict: ONE shared shift must explain every line vs scrambled catalogs -- "
        "a single match is numerology; z = median per-line. mind.fit_decay: A exp(-lambda t)+C, "
        "d^2 delta-method weights (d-weights read 17% low, pinned), bootstrap CI, bias-aware "
        "truncation flag. Doppler math delegates to dedoppler",
        example="fl = mind.spectral_lines(x, y, catalog=BALMER); rz = mind.redshift_verdict([l['center'] for l in fl['lines']], BALMER)",
        native=True, aliases=("find spectral lines", "identify emission lines", "what element is this line",
                              "measure the redshift", "radial velocity from spectrum", "fit an exponential decay",
                              "half life from counts", "randomized benchmarking decay", "ringdown rate",
                              "absorption line detection", "line list identification"))


    c.register_capability(
        "Quantum statistics (spacing-ratio regime classifier + the Bell verdict)",
        "mind.level_statistics reads integrable-vs-chaotic off a spectrum with the unfolding-free "
        "spacing RATIO (Atas 2013; a wrong unfolding manufactures or erases repulsion), classifying "
        "Poisson / GOE / GUE by bootstrap CI, REFUSING with the n that would decide when classes "
        "overlap. mind.chsh_verdict: pairing-scramble null (correlated at all?), bootstrap CI vs the "
        "classical bound 2 (beyond every local hidden-variable model?), and the TSIRELSON ALARM -- "
        "data past 2*sqrt(2) accuses the apparatus, not the theory. mind.chsh_demo plants quantum / classical / "
        "independent / broken trials",
        example="r = mind.level_statistics(eigvals); q = mind.chsh_verdict(*mind.chsh_demo(4000, 'quantum'))",
        native=True, aliases=("is my spectrum chaotic or integrable", "level spacing statistics",
                              "poisson vs wigner dyson", "random matrix statistics",
                              "bell test analysis", "chsh violation check", "quantum correlations test",
                              "does my data violate the classical bound", "level repulsion",
                              "eigenvalue statistics classifier"))


    c.register_capability(
        "Science report (one front door: transit / pulsar / spectrum / decay / levels / CHSH / series)",
        "mind.science_report(data, kind) routes named data to the matching science instrument and "
        "returns one uniform report {kind, verdict, why, result-with-audit-trail}. Kinds: "
        "light_curve (box transit hunt), pulsar_panel (Hellings-Downs + sky scramble), spectrum "
        "(lines + margin identification + one-shift-or-refuse redshift), decay (A exp(-lam t)+C), "
        "levels (Poisson/GOE/GUE spacing ratios), chsh (Bell verdict with the Tsirelson alarm), "
        "series (the residual interrogation tower). Unknown kind raises WITH the list -- the door "
        "never guesses. Citations map: docs/SCIENCE_INSTRUMENTS.md",
        example="rep = mind.science_report({'t': t, 'y': counts}, kind='decay'); print(rep['verdict'], rep['why'])",
        native=True, aliases=("analyze my scientific data", "run the science instruments",
                              "one report for my measurement", "which instrument fits my data",
                              "analyze my experiment", "science front door",
                              "statistics verdict for my data"))


    c.register_capability(
        "Audio drift (train on clips, generate more -- the abstention ladder as the adapter)",
        "mind.train_audio_drift maps each clip by what it honestly is: (freq, amp) tone parameters "
        "when the multitone r2 gate passes (frequency-sorted, phase is gauge), a log-band envelope "
        "when it is a STATIONARY texture, refused when neither (a chirp). A corpus must be ONE "
        "space; mixed corpora refuse with the counts. mind.generate_audio drifts in that space and "
        "resynthesizes deterministically (exact additive sine / seeded envelope-shaped noise), "
        "always attaching the audit + nearest-training spectral distance. Save with mind.write_wav",
        example="m2, meta = mind.train_audio_drift(clips, 8000); out = mind.generate_audio(m2, meta, n=4)",
        native=True, aliases=("generate audio like this folder", "train on my sound clips",
                              "make more sounds like these", "audio texture generation",
                              "synthesize similar tones", "sound model from examples"))


    c.register_capability(
        "Video drift (train on short clips, generate coherent motion)",
        "mind.train_video_drift turns each short clip into a keyframe-PAIR point [start splats, "
        "end-minus-start delta]: motion is the JOINT structure between keyframes -- the quantity "
        "the H1.4 verdict proved drift preserves and independent marginals scramble -- with end "
        "splats re-matched by nearest centre so the delta is motion, not relabelling. "
        "mind.generate_video drifts a pair, interpolates splat params across n_frames, renders "
        "every frame, and reports per-clip max frame-to-frame RMS in the audit: the smoothness "
        "claim carries its own number. Single-frame clips refuse",
        example="vm, vmeta = mind.train_video_drift(clips); out = mind.generate_video(vm, vmeta, n=2, n_frames=8)",
        native=True, aliases=("generate video like these clips", "train on my short clips",
                              "make more motion like this", "video texture generation",
                              "animate like my examples", "motion model from clips"))

    c.register_capability(
        "Codec atlas + honest router (which compressor, measured on YOUR data)",
        "machine_map applied to compression: mind.codec_atlas() is the SPEC SHEET -- every codec "
        "unit (zlib/lzma, low-rank/tucker/tt, rate-distortion, pack_images, event codec, "
        "sequence-predictive, generator rung, cold storage) with its real module+symbol, "
        "pays-condition, and kept negatives in one table. mind.codec_place(x, max_error=...) "
        "MEASURES every applicable unit on x and ranks by bytes, priced against the zlib "
        "baseline, with 'store raw' a first-class row. Lossy units run ONLY under a stated "
        "error budget (never 99% energy; loss is never volunteered). Refusal on incompressible "
        "data is the finding.",
        example="r = mind.codec_place(__import__('numpy').add.outer(__import__('numpy').sin(__import__('numpy').arange(64)/7.), __import__('numpy').cos(__import__('numpy').arange(64)/9.)), max_error=1e-6); print(r['best'], r['rows'][0])",
        native=True, aliases=("which codec should I use", "compare compressors on my data",
                              "benchmark all compressors", "pick a compression method automatically",
                              "codec atlas", "route data to the best compressor",
                              "will my data compress and how", "compression spec sheet"))

    c.register_capability(
        "Predictive residual codec (model + coded error, exact or budgeted)",
        "mind.residual_encode(y) compresses a 1-D signal as its piecewise LAWS plus the coded "
        "error: decompose_piecewise fits per-segment formulas (stored as exact recipes), the "
        "residual is byte-plane shuffled and entropy-coded. Exact by default -- decode is "
        "bit-identical (float fixup + verbatim patch list). With max_error, near-lossless "
        "within the budget (measured 8.5x vs zlib on a noisy 3-regime signal; exact mode caps "
        "at ~1.1x -- irreducible mantissa planes). Self-refuses into mode='raw' when the model "
        "head does not pay. mind.residual_decode(blob) inverts; codec_place routes 1-D here.",
        example="import numpy as np; y=np.sin(2*np.pi*np.arange(600.)/23); r=mind.residual_encode(y, max_error=1e-4); out=mind.residual_decode(r['blob']); print(r['report']['mode'], r['report']['ratio_vs_zlib'], float(np.abs(out-y).max()))",
        native=True, aliases=("pack this array smaller than zlib", "beat zlib on a float array", "quantize my weights", "quantize model weights with an error bound", "entropy code residuals after a model predicts",
                              "predictive residual codec", "compress a signal exactly with a model plus error",
                              "lossless model based compression", "store the law and the leftovers",
                              "model plus residual compression", "fit then code the error"))

    c.register_capability(
        "Surprise-weighted rate allocation (code the news finely, the expected coarsely)",
        "mind.surprise_code(batch, reference, fine_step) spends bits where the information is: "
        "the reference corpus's drift model reads density in one dot product (z=<enc(x),mu>), "
        "points in its VOID (the news) are quantized at fine_step, predicted points at "
        "fine_step*coarsen -- same news fidelity as uniform-fine coding, MEASURED 1.71x fewer "
        "bytes (coarsen sweep 16/64/128/256 -> 1.17/1.36/1.57/1.71x; the varint floor caps it). "
        "A chance gate refuses the split when the news share sits at the quantile's own "
        "expected level. Lossy by design on the predicted mass. mind.surprise_decode inverts.",
        example="import numpy as np; rng=np.random.default_rng(0); ref=rng.standard_normal((100,2))*0.05+0.5; batch=np.vstack([ref[:60], rng.uniform(0,1,(25,2))]); r=mind.surprise_code(batch, ref, fine_step=1e-4); print(r['report']['mode'], round(r['report']['ratio_vs_uniform_fine'],2))",
        native=True, aliases=("allocate bits where the information is",
                              "spend more bits on surprising samples",
                              "code the news finely and the expected coarsely",
                              "surprise weighted compression", "importance weighted quantization",
                              "variable rate coding by predictability",
                              "bit allocation by surprise"))

    c.register_capability(
        "Distributional codec (store the distribution, not the samples)",
        "mind.distribution_encode(points, bits=6) compresses a sample bank to its drift "
        "model's d+1 moment hypervectors, quantized at 4/6/8 bits with per-array scales -- "
        "MEASURED 10.5x (6-bit) / 21.5x (4-bit) vs zlib at coverage 1.0 on a 1500-point "
        "two-cluster bank. Decode returns a DriftModel to SAMPLE from: points LIKE the "
        "originals, never the originals (exactness wants codec_place/residual_encode). The "
        "report prices break_even_n (below it, pays=False) and carries the post-quantization "
        "generation audit, so a broken distribution is visible at encode time. "
        "mind.distribution_decode inverts.",
        example="import numpy as np; rng=np.random.default_rng(0); pts=np.vstack([c+0.05*rng.standard_normal((800,2)) for c in ([0.3,0.3],[0.7,0.7])]); r=mind.distribution_encode(pts); mdl=mind.distribution_decode(r['blob']); print(round(r['report']['ratio_vs_zlib'],1), r['report']['audit'])",
        native=True, aliases=("compress a point cloud to distribution moments",
                              "shrink this point cloud for storage",
                              "store distribution not samples", "distributional codec",
                              "summarize samples as a density model",
                              "replace a sample bank with a model",
                              "ship the moments not the points", "moment based compression"))

    c.register_capability(
        "Procedural storage (store the program, verify pointwise, or refuse)",
        "mind.store_procedural(y, tol=0.02) stores a 1-D signal as its PROGRAM, two tiers "
        "cheapest first: the generator bank + Gauss-Newton polish (blob CONSTANT in n -- "
        "MEASURED 76x at n=4k and 310x at n=16k from the SAME bytes; regenerable at any "
        "length, valid=False past 2x the verified window) or decompose_piecewise recipes "
        "(11.4x, original length only -- extension on per-segment axes is refused). Every "
        "tier is VERIFIED pointwise at tol*amplitude BEFORE commit; when both miss it "
        "refuses with measured errors and routes to residual_encode/codec_place. "
        "mind.regen_procedural(blob[, n]) plays it back.",
        example="import numpy as np; y=2.5*np.sin(2*np.pi*np.arange(3000.)/333)+7; r=mind.store_procedural(y); g=mind.regen_procedural(r['blob'], n=5000); print(r['report']['mode'], round(r['report']['ratio_vs_zlib']), g['valid'])",
        native=True, aliases=("compress by storing the program not the data",
                              "store the generator instead of the output",
                              "save a signal as a formula and regenerate it",
                              "fit a generator and store only the recipe",
                              "procedural storage round trip", "program as compression",
                              "constant size compression for lawful signals"))

    c.register_capability(
        "Mesh codec at a budget (and the measured refs-cost-what-deltas-save negative)",
        "mind.mesh_encode(mesh, max_error) compresses a triangle mesh: vertices quantized at "
        "a per-coordinate |err|<=max_error contract (verified on the decoded artifact), "
        "connectivity BIT-EXACT as varint index-deltas, all zlib'd -- MEASURED 2.5-2.7x vs "
        "zlib(raw). It prices the classic base+displacement hypothesis (decimate + closest-"
        "point refs + deltas) against this fair uniform coder and ships the smaller. KEPT "
        "NEGATIVE, the headline: explicit refs carry the information the anchors subtract, "
        "so uniform wins on every mesh measured; implicit refs are the deferred rung. "
        "mind.mesh_decode inverts.",
        example="import numpy as np; mesh=mind.mesh_from_sdf(lambda p: np.linalg.norm(np.atleast_2d(p),axis=1)-0.8, bounds=((-1,-1,-1),(1,1,1)), res=20); r=mind.mesh_encode(mesh, max_error=2e-3, try_base=False); V,F=mind.mesh_decode(r['blob']); print(r['report']['mode'], round(r['report']['ratio_vs_zlib'],2))",
        native=True, aliases=("compress a mesh", "mesh codec", "store a mesh smaller",
                              "coarse mesh plus displacement",
                              "compress geometry with a base and details",
                              "quantize mesh vertices at a budget", "shrink a mesh file"))

    c.register_capability(
        "Formal logic & Lean 4 export (prove, check, hand to an external authority)",
        "logic_prove: Horn forward chaining, proof tree, honest None (strategy="
        "'seminaive': same atoms, >=22x on large bases); logic_check_proof re-verifies "
        "INDEPENDENTLY (forged premises raise); lean_export emits Lean 4 "
        "(check='external' = both checkers agree); lean_verify runs installed lean; "
        "logic_consequences: least fixpoint + absurdity smoke (Lean never checks rule "
        "CONSISTENCY); logic_proof_measure sizes a checked proof; encode/decode_atom "
        "round-trip atoms (decode abstains); fact_capacity's NEGATIVE: bundled recall "
        "cliffs by load 8 independent of D -- INDEX fact bases. Deduction, not regression.",
        example="p=mind.logic_prove(['mortal',['socrates']], [{'head':['human',['socrates']],'name':'h'},{'head':['mortal',['?x']],'body':[['human',['?x']]],'name':'m'}]); print(mind.logic_check_proof(p, [{'head':['human',['socrates']],'name':'h'},{'head':['mortal',['?x']],'body':[['human',['?x']]],'name':'m'}]))",
        native=True, aliases=("lean4", "lean 4", "prove a theorem", "theorem prover",
                              "formal verification", "check a proof", "proof assistant",
                              "export to lean", "horn clauses", "forward chaining",
                              "unification", "verify a logical claim", "deduce a fact from rules",
                              "logic inference", "first-order logic",
                              "all consequences of rules", "everything derivable",
                              "fixpoint of rules", "detect inconsistent rules",
                              "contradiction in rules", "how complex is a proof",
                              "proof size", "decode a fact vector",
                              "how many facts fit", "fact capacity"))


    c.register_capability(
        "Conjecture & refute (learn Horn rules from examples, prove them in Lean)",
        "mind.logic_induce learns Horn clauses from positive/negative examples -- "
        "learning-from-failures (Cropper & Morel 2021, generate/test/constrain; LFF-"
        "style on the finite fragment, not Popper parity). Test is the engine's own T_P "
        "fixpoint, so RECURSIVE rules learn free (ancestor from parent, measured). Then "
        "deduces the theory's consequences, refutes vs negatives (count reported), and "
        "emits Lean 4 proving a positive FROM THE LEARNED RULES. rules=None when the "
        "space exhausts -- never a guess. See Formal logic for deduction.",
        example="out=mind.logic_induce([{'head':['parent',['tom','bob']],'name':'p0'},{'head':['parent',['bob','liz']],'name':'p1'}], [['ancestor',['tom','bob']],['ancestor',['tom','liz']]], [['ancestor',['bob','tom']]], 'ancestor', {'parent':2,'ancestor':2}); print(len(out['rules']), out['refuted_count'])",
        native=True, aliases=("learn rules from examples", "rule induction",
                              "inductive logic programming", "ILP", "conjecture and refute",
                              "induce a law from data", "learn horn clauses",
                              "find a rule that explains observations",
                              "learning from failures", "learn a recursive rule",
                              "generalize from examples", "hypothesis search"))


    c.register_capability(
        "Verified-knowledge memory (proofs as hypervectors, provenance kept)",
        "mind.proof_store proves a goal, runs the INDEPENDENT checker (unproven claims "
        "never enter), stores indexed rows in the substrate: goal atom, proof TREE "
        "(encode_tree_carrier), rule TRACE (seq_encode, complex kept complex). "
        "verify='external' records an installed Lean's verdict -- provenance "
        "('checked'/'lean_verified') travels with each record; the binary stays "
        "optional, its verdict is kept. mind.proof_recall: exact or k-nearest by "
        "goal/tree/trace cosine (self excluded), provenance-filtered, honest empties. "
        "Rows not bundles, per the fact_capacity negative.",
        example="mind.proof_store(['mortal',['socrates']], [{'head':['human',['socrates']],'name':'h'},{'head':['mortal',['?x']],'body':[['human',['?x']]],'name':'m'}]); print(mind.proof_recall(['mortal',['socrates']])['exact']['provenance'])",
        native=True, aliases=("remember a proof", "store verified knowledge",
                              "recall a proof", "similar proofs", "proof memory",
                              "verified knowledge base", "knowledge with provenance",
                              "find proofs like this", "proof cache",
                              "store theorems", "recall by structure"))


    c.register_capability(
        "Tabled goal-directed query (bindings for a goal with variables)",
        "mind.logic_query answers a goal containing variables (['ancestor',['tom','?w']]) "
        "backward from the goal, returning every ground binding with a checkable proof. "
        "TABLING (Chen & Warren 1996; XSB/SWI) makes it terminate on LEFT RECURSION and "
        "CYCLES where plain SLD diverges. MEASURED LAW: speedup tracks the goal's DEMAND "
        "CLOSURE not graph size -- 304x at demand 1, 0.3x (SLOWER) at demand 690 -- so "
        "budget caps the tabled answers and fallback=True reruns as a seminaive fixpoint, "
        "reporting which route ran. Never the silent default; see Formal logic to derive "
        "everything instead.",
        example="print(mind.logic_query(['ancestor',['tom','?w']], [{'head':['parent',['tom','bob']],'name':'p0'},{'head':['parent',['bob','liz']],'name':'p1'},{'head':['ancestor',['?x','?y']],'body':[['parent',['?x','?y']]],'name':'ab'},{'head':['ancestor',['?x','?z']],'body':[['parent',['?x','?y']],['ancestor',['?y','?z']]],'name':'as'}])['answers'])",
        native=True, aliases=("query with variables", "tabling", "tabled resolution",
                              "backward chaining", "goal directed search", "SLD resolution",
                              "answer a logic query", "what does X reach",
                              "bindings for a goal", "memoize subgoals", "occurs check"))


    c.register_capability(
        "Cell-aggregate morphogenesis (grow a body from soft cells, analytic gradients)",
        "morphogenesis_grow proliferates soft cells into a compact genus-0 aggregate "
        "(NO autodiff: closed-form gradients vs fd_gradient to 2e-9; soft-then-inflate "
        "anneal). morphogenesis_differentiate breaks symmetry by DIFFERENTIAL ADHESION "
        "(Mode 2: Gray-Scott RD modulated by a Wolpert gradient; control 0.824 vs "
        "0.257 sphericity). genome_encode/decode/locality/interpolate make a body plan "
        "ONE searchable vector (locality measured monotone; noise abstains). "
        "shape_memory_* hold morphologies as attractors: 1.00 recall vs 0.00 for a "
        "depth-matched scrambled control.",
        example="r=mind.morphogenesis_grow(n_cells=48, seed=3, steps=150); print(len(r['positions']), round(r['sphericity'],3))",
        native=True, aliases=("morphogenesis", "grow a creature body", "cell aggregate",
                              "reaction diffusion on cells", "turing pattern on a body",
                              "morphogen gradient", "limb bud", "symmetry breaking",
                              "genome encoding", "creature genome", "interpolate designs",
                              "encoding locality", "shape memory", "regeneration",
                              "recover from perturbation", "target morphology",
                              "soft cell simulation", "differential adhesion",
                              "particle relaxation packing", "pack soft spheres",
                              "body plan generation", "grow cells", "cell division growth",
                              "energy minimization on positions"))


    c.register_capability(
        "Tetrahedralize a point set with PROVED topology (limb-connection certificates)",
        "mind.tetrahedralize turns points into a volumetric tet mesh (Bowyer-Watson + "
        "alpha-complex, NumPy only) reporting adjacency, boundary, NON-MANIFOLD faces, "
        "components, Euler. mind.tet_connectivity_certificate PROVES every limb reaches "
        "the torso as a derivation (not a flood fill) and names orphans; "
        "mind.tet_certificate_lean exports a claim for external Lean. mind.tet_lod_chain "
        "makes each LOD level a RULE (nested prefix, 9.1x smaller than stored meshes) and "
        "REFUSES levels that orphan a limb. SCOPE: clean point sets, not TetGen. LAW: an "
        "attachment 1-2 cells across is NOT connected; 3 is minimum.",
        example="a=mind.morphogenesis_grow(n_cells=40,seed=0,steps=80); mesh=mind.tetrahedralize(a['positions'],a['radii']); print(mesh['T'], mesh['components'], mind.tet_connectivity_certificate(mesh,0,list(range(mesh['T'])))['ok'])",
        native=True, aliases=("tetrahedral mesh", "delaunay triangulation", "tetrahedralize",
                              "certified LOD", "volumetric LOD", "LOD without storing meshes",
                              "decimate without breaking topology",
                              "volumetric mesh from points", "alpha shape",
                              "limb attachment", "is my limb connected",
                              "mesh topology proof", "certify a mesh", "circumsphere",
                              "points to volume mesh"))


    c.register_capability(
        "Stable neo-Hookean tet elasticity + muscle fibers (hand-derived gradients)",
        "mind.fem_simulate solves a tet mesh quasistatically under STABLE neo-Hookean "
        "elasticity (Smith/De Goes/Kim 2018) plus activation-dependent muscle springs. "
        "Chosen over the classical log-J neo-Hookean because log J is UNDEFINED for "
        "inverted elements and generated meshes DO invert -- this energy stays finite and "
        "differentiable through inversion (pinned). NO autodiff: Piola-Kirchhoff stress "
        "hand-derived, checked vs fd_gradient to 2e-11, rest stress-free to 7e-17. "
        "fem_select_fibers picks axis-aligned edges; fem_rest_quality reports "
        "degenerate/INVERTED elements before you trust a solve.",
        example="a=mind.morphogenesis_grow(n_cells=30,seed=0,steps=60); mesh=mind.tetrahedralize(a['positions'],a['radii']); fib,rl=mind.fem_select_fibers(a['positions'],mesh['tets']); r=mind.fem_simulate(a['positions'],mesh['tets'],steps=60,fibers=fib,rest_lengths=rl,activation=0.7,pinned=[0]); print(round(r['history'][0],2), round(r['history'][-1],2))",
        native=True, aliases=("neo hookean", "hyperelastic material", "FEM tetrahedron",
                              "soft body FEM", "muscle actuation", "deformation gradient",
                              "piola kirchhoff stress", "element inversion",
                              "simulate a creature body", "strain energy density",
                              "lame parameters"))


    c.register_capability(
        "Tier contracts (certify a memory plan BEFORE it runs, fidelity clause included)",
        "NINE CERTIFY-OR-REFUSE contracts, one shape: certify, or refuse with the "
        "failing clause NAMED. tier_certify_plan (capacity, Horn-derived tier ban, "
        "FIDELITY from the measured D/M law), bake_certify (hypergeometric spot-check "
        "bound), differential_agreement, schedule_certify, demux_gated (measured 5% "
        "noise envelope), pose_certify, conservation_ledger (exact vs BOUNDED tested "
        "differently), lyapunov_certify (settle CERTIFIED for a true gradient flow), "
        "plan_certify (a GOAP plan's preconditions and goal).",
        example="print(mind.tier_certify_plan({'hot':{'capacity':8,'cost':1},'trace':{'capacity':10**6,'cost':10,'holographic':True,'dim':4096}}, [{'item':'b','tier':'trace','count':256}], min_recall=0.98)['violations'])",
        native=True, aliases=("tier contract", "certify a plan", "memory budget check",
                              "will this fit in cache", "roofline", "precondition check",
                              "refuse a plan", "memory hierarchy contract",
                              "eviction SLA", "fidelity guarantee",
                              "certify a bake", "spot check", "detection probability",
                              "how many samples to verify", "verify a lookup table",
                              "differential testing", "do two implementations agree",
                              "cross check backends", "compare implementations",
                              "schedule conflict", "is my schedule safe",
                              "race free schedule", "parallel wave check",
                              "estimate noise level", "is this answer trustworthy",
                              "gate a demux", "refuse outside the envelope",
                              "pose validity", "joint limit check", "certify a pose",
                              "conservation audit", "energy drift", "is my sim leaking",
                              "lyapunov", "has it really converged", "certify a settle",
                              "GOAP", "validate an action plan", "precondition missing"))


    c.register_capability(
        "Fixed-topology template wrap (vertex i means the same thing on every body)",
        "mind.template_wrap deforms ONE template mesh onto any target field KEEPING ITS "
        "FACE ARRAY -- the precondition for blendshapes, shared textures and cross-species "
        "morphing, none of which work while each creature meshes from scratch. Annealed "
        "projection (non-rigid ICP schedule, Amberg 2007) + Taubin no-shrink relaxation; an "
        "analytic field gives exact correspondence, not a nearest-point search. MEASURED: "
        "improves triangle quality 66.6 -> 38.3. template_wrap_quality reports landing "
        "error, ROBUST p95/p5 bunching, degenerate edges, flipped faces. NEGATIVE: needs "
        "matching topology.",
        example="import numpy as np; sph=lambda P: np.linalg.norm(P,axis=1)-1.0; t=mind.mesh_from_sdf(sph,((-1.4,)*3,(1.4,)*3),res=24,vectorized=True); ax=np.array([1.3,0.8,1.0]); ell=lambda P:(np.linalg.norm(P/ax,axis=1)-1.0)*ax.min(); V=mind.template_wrap(t.vertices,t.faces,ell,rounds=4); print(round(mind.template_wrap_quality(V,t.faces,ell)['surface_error'],4))",
        native=True, aliases=("template wrap", "shrink wrap a mesh", "fixed topology",
                              "vertex correspondence", "retopology", "same mesh new body",
                              "morph between creatures"))


    c.register_capability(
        "Blendshape basis with DECLARED local support (STAR's fix, without the scans)",
        "mind.blend_corrective authors one blendshape target that displaces only vertices "
        "within a GEODESIC radius of an anchor -- geodesic because a hand on a hip is "
        "millimetres away in space and a metre across the surface. SMPL's dense correctives "
        "capture spurious long-range coupling; STAR spends scan data LEARNING each joint's "
        "activation region, but an authored basis DECLARES it -- free and exact (measured "
        "overreach 0.000e+00; 8-15% of the mesh moves). blend_locality_report checks it. "
        "NEGATIVE: locality guaranteed, anatomical realism not.",
        example="import numpy as np; sph=lambda P: np.linalg.norm(P,axis=1)-1.0; msh=mind.mesh_from_sdf(sph,((-1.3,)*3,(1.3,)*3),res=18,vectorized=True); V=np.asarray(msh.vertices); s=int(np.argmax(V[:,1])); t=mind.blend_corrective(msh,s,0.8,'normal',0.2); print(mind.blend_locality_report(V,[t],msh,[s],[0.8])['max_overreach'])",
        native=True, aliases=("blendshape", "morph target", "pose corrective",
                              "local support", "shape basis", "sparse deformation",
                              "make a blendshape"))


    c.register_capability(
        "Face as a landmark graph + parts (procedural, no scans, non-human friendly)",
        "mind.face_landmarks places skull-canon landmarks (crown/brow/eye/nose/mouth/chin/"
        "jaw/cheek/ear/temple), bilateral pairs mirrored STRUCTURALLY. face_part_graph "
        "says which rigblock goes where as DATA, so a four-eyed noseless face is a list "
        "edit not a code path; face_expression gives per-landmark displacements driving "
        "blend_corrective. WHY NOT FLAME: 3DMMs fix topology and expression basis at scan "
        "time and assume adult human anatomy, fitting stylized/non-human assets unstably. "
        "NOT a likeness and NOT photo reconstruction -- no scan basis to fit.",
        example="lm = mind.face_landmarks((0.0,1.6,0.0), 0.24, 0.10); print(len(lm), sorted(lm)[:3], len(mind.face_part_graph(lm)))",
        native=True, aliases=("face", "facial landmarks", "head features", "expression",
                              "eyes nose mouth", "character face", "make a face"))


    c.register_capability(
        "LBS volume-loss bound (predict the candy wrapper, then refuse the pose)",
        "mind.skin_twist_shrink gives the CLOSED FORM |sum_b w_b exp(i theta_b)| for how "
        "much volume linear blend skinning loses under twist -- the two-bone case reduces "
        "to |cos(theta/2)|, so 90 deg keeps 0.707 and 180 deg collapses to ZERO (the candy "
        "wrapper). VERIFIED against the shipped skinning path to 1.1e-16, so it is a "
        "theorem about the code. mind.skin_pose_is_safe refuses a pinching pose BEFORE "
        "deforming; mind.skin_max_safe_twist inverts it (even 50/50 weights allow only 63.6 "
        "deg at a 0.85 floor). Exact for pure twist, conservative for bending.",
        example="import numpy as np; print(round(float(mind.skin_twist_shrink([0.5,0.5],[0.0,np.pi/2])),4), mind.skin_pose_is_safe([[0.5,0.5]],[0.0,np.pi])['ok'])",
        native=True, aliases=("candy wrapper", "volume loss", "skinning artifact",
                              "collapsed elbow", "twist limit", "is this pose safe"))


    c.register_capability(
        "Safe offset / wrap injectivity (the reach, both conditions)",
        "mind.wrap_is_injective says whether an offset or shrink-wrap will FOLD the mesh "
        "through itself -- a folded wrap still reads clean on surface error. Checks BOTH "
        "causes: LOCAL (offset under the smallest concave radius) and GLOBAL (collinear "
        "normals closer than twice the offset). The global term bites: armpits and finger "
        "gaps are LOW-curvature surfaces FACING each other, so a curvature-only check "
        "passes exactly the cases that fail. NEGATIVE: samples the reach, no medial axis.",
        example="import numpy as np; sph=lambda P: np.linalg.norm(P,axis=1)-1.0; msh=mind.mesh_from_sdf(sph,((-1.3,)*3,(1.3,)*3),res=14,vectorized=True); print(mind.wrap_is_injective(msh.vertices,msh.faces,0.05,sph,samples=200)['ok'])",
        native=True, aliases=("safe offset", "self intersection", "reach", "will this fold",
                              "offset distance", "shrink wrap safety", "medial axis limit"))


    c.register_capability(
        "SCALIS scale-invariant surfaces (thin features survive beside thick ones)",
        "mind.convolution_field_scalis integrates over the HOMOTHETIC measure ds/tau instead "
        "of absolute arc length, so a long thick segment no longer deposits more field than "
        "a short thin one. Plain convolution 'failed to reconstruct prescribed radii and "
        "was unable to model large shapes with fine details' (Zanni et al. 2013). MEASURED: "
        "exactly invariant (0.13241) across a 16x scale range where plain scales by lam; and "
        "on a spike 5.7x thinner than its trunk, plain renders it at 9% of the asked radius "
        "-- swallowed -- while SCALIS gives 123%. Default-off; opt in per field.",
        example="f = mind.convolution_field_scalis([((0,0,-0.5),(0,0,0.5),0.15,(1.,1.,1.))]); import numpy as np; print(round(float(f(np.array([[0.1,0.0,0.0]]))[0]),4))",
        native=True, aliases=("SCALIS", "scale invariant surface", "thin feature lost",
                              "convolution radius control", "tail tip vanishes",
                              "blend thin into thick"))


    c.register_capability(
        "Physically-based TISSUE materials (organs, bone, fat, skin -- not flat)",
        "mind.tissue_pbr gives base colour, roughness, metallic, SSS weight and a "
        "PER-CHANNEL subsurface radius for bone/skin/fat/muscle/organ/liver/lung/gut/"
        "spleen/chitin/keratin. Per-channel matters: red scatters deeper than blue in every "
        "soft tissue, and a scalar radius cannot give the warm silhouette that separates "
        "meat from red plastic. Christensen-Burley parameterisation; the ORDERING is "
        "grounded in measured SDOCT coefficients (bone/skin 1.95-2.13 /mm, liver 1.30-1.46, "
        "spleen 0.52-0.63) so viscera scatter furthest. NEGATIVE: single medium per tissue.",
        example="v = mind.tissue_pbr('skin'); print([round(x,2) for x in v['sss_radius']], v['sss_weight'])",
        native=True, aliases=("tissue material", "subsurface scattering", "organ material",
                              "skin shader", "bone material", "realistic flesh", "SSS"))



_PART = "holographic_catalog_p06"




def _selftest():
    """Delegates to holographic_catalog.check_catalog_part -- one home for the shared contract."""
    from holographic.caching_and_storage.holographic_catalog import check_catalog_part
    n = check_catalog_part(_PART, register_p06)
    print("%s selftest OK -- %d capabilities, no internal duplicates" % (_PART, n))


if __name__ == "__main__":
    _selftest()
