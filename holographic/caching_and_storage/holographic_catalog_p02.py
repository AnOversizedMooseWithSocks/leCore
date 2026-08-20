"""holographic_catalog_p02 -- part 2/6 of the capability registry (split from holographic_catalog).

MECHANICAL SPLIT, no edits. holographic_catalog.py hit 81% of the 1 MB agent-read cap, so the file
that makes capabilities discoverable was becoming the one file an agent could not open. The parts are
called IN ORDER by default_catalog() and the emitted catalog is byte-identical -- verified by hashing
every capability field before and after. Order matters: find_capability ranks by score and ties break
by registration order, so a reordering would silently move search results.

Add new capabilities to the LAST part, or to whichever part is topically right -- never to a new file
without registering it in default_catalog(), or it will simply not exist.
"""


def register_p02(c):
    """Register this part's capabilities on `c`. Called by default_catalog() in order."""
    c.register_capability('Interior distance / thickness field of a mesh', "The interior DEPTH of a mesh on a grid: distance from each inside point to the nearest surface (0 outside) -- a THICKNESS / wall-thickness field for finding thin walls, thick cores, and local part size. m.interior_distance_field(mesh, res) returns (depth grid, (lo,hi) bounds, cell size); depth is positive inside, larger = deeper. Built from the shared correspondence (closest_face_point) for distance and the winding number for inside/out. The skeleton is this field's ridge, but the field itself answers 'how thick is this part at each point'.", example='import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_skeleton import _cylinder; d,b,c=m.interior_distance_field(_cylinder(), res=16); (d.shape==(16,16,16), float(d.max())>0)', native=True, module='skeleton', aliases=('how thick is this part at each point', 'wall thickness of a model', 'thickness field of a mesh', 'distance from inside to the surface', 'local part size', 'solid depth grid'), semantic='analyze/measure')

    c.register_capability('Render-ready texture + uvs from a loaded mesh', 'Get the render-ready (texture, uvs, base_color) from a LOADED mesh -- the pointer from an imported (or self-decimated / retopologised) model to a TEXTURED render_mesh call WITHOUT a file path. m.asset_base_texture(loaded_mesh) returns (texture image in [0,1] or None, per-vertex uvs, base_color fallback); feed the pair straight to render_mesh(mesh, cam, texture=, uvs=). Picks the base-colour map by face COVERAGE (a multi-material scan renders in the skin most of its surface wears), 8-bit normalised. Same logic preview_asset uses, factored out so a mesh you built yourself can be textured too.', example="import lecore; m=lecore.UnifiedMind(); from holographic.io_and_interop.holographic_assetimport import load_glb, _rigged_glb; import tempfile,os; p=tempfile.mktemp(suffix='.glb'); open(p,'wb').write(_rigged_glb()); lm=load_glb(p); tex,uv,base=m.asset_base_texture(lm); os.remove(p); (len(base)==3, isinstance(base,tuple))", native=True, module='assetimport', aliases=('get texture and uvs to render a loaded mesh', 'render ready base color from an imported model', 'extract texture array from a mesh for render_mesh', 'texture and uv for a decimated mesh', 'pull the albedo image off a loaded glb', 'how do I texture a mesh I decimated myself'), semantic='convert/emit')

    c.register_capability('CVT remesh (Lloyd-relaxed isotropic decimation)', "CVT remeshing (CWF, Xu et al. SIGGRAPH 2024): m.cvt_remesh(mesh, n_sites) replaces cluster_decimate's axis-aligned grid with LLOYD-RELAXED surface sites -- k-means is the engine's codebook move, and the representatives reuse the bundled-quadric minimizer (the QEM term). MEASURED at equal vertex budget on a scanned mantis: min-angle median 22.8 -> 43.1 deg, slivers 14% -> 1%, components 41 -> 9, non-manifold edges 211 -> 82. Deterministic (farthest-point seeding, no rng). NOT provably manifold: gate with m.topology_gate. The R4 isotropic-fallback slot of the retopo backlog.", example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; q,rep=m.cvt_remesh(loop_subdivide(box(),3), n_sites=200, iterations=4); (len(q.faces)>0, rep['sites']==200)", native=True, module='meshqem', aliases=('remesh with well shaped triangles', 'isotropic remeshing', 'centroidal voronoi remesh', 'better triangle quality than grid decimation', 'lloyd relaxation on a mesh', 'reduce slivers when decimating'), semantic='modify/filter')

    c.register_capability('Gabor cloud render (single-scatter a Gabor field as volume)', "Render a Gabor field as a volumetric CLOUD (GAB-CLOUD): m.gabor_cloud_render(field, O, D, L, sun_dir, ceiling) single-scatters a fitted GaborField through the engine's cloud renderer. The field satisfies the density protocol (.density + finite-segment .optical_depth, verified 1e-6 vs quadrature via a pure-NumPy complex erf), so cloud_single_scatter's CLOSED-FORM shadow rays work unchanged -- measured 49x fewer density evals at 8e-5 error, same as on FPE volumes. Call field.lod(cutoff) first for a cheaper coarse cloud, no refit. Returns (radiance, density_evals).", example="import numpy as np, lecore; m=lecore.UnifiedMind(); ax=np.linspace(0,1,20); X=np.stack(np.meshgrid(ax,ax,ax,indexing='ij'),-1); r2=((X-0.5)**2).sum(-1); rho=np.clip(np.exp(-r2/0.08)*(1+0.4*np.cos(20*X[...,0])),0,None); f,rep=m.gabor_volume(rho,K=16); rad,ev=m.gabor_cloud_render(f, np.array([[0.5,0.5,-0.5]]), np.array([[0.,0.,1.]]), 2.0, np.array([0.3,1.,0.2]), 1.2, view_steps=8); np.isfinite(rad).all()", native=True, module='gaborfield', aliases=('render a gabor field as a cloud', 'light and shadow a gabor volume', 'single scatter a fitted gabor field', 'volumetric render of gabor kernels', 'cloud from gabor primitives with lod', 'closed form shadow rays on a gabor field'), semantic='render/frame')

    c.register_capability('Gabor field volumes (oriented primitives, closed-form rays, free LOD)', 'Gabor Fields (Condor SIGGRAPH 2026): m.gabor_volume(rho, K) fits a density grid with Gaussian-envelope x cosine-wave primitives; Gaussians and oriented Gabors compete per slot. MEASURED +13-14 dB over equal Gaussians on oriented content; ray integrals CLOSED FORM (2e-16 vs quadrature), transmittance one call/ray; LOD FREE via field.lod(cutoff). anisotropic=True fits oriented ellipsoid envelopes (+2-6 dB on filaments, opt-in, worse on blobs). KEPT NEG: fit cost once per asset; GAB-CV control variates declared negative (deterministic renderer, no variance).', example="import numpy as np, lecore; m=lecore.UnifiedMind(); ax=np.linspace(0,1,20); X=np.stack(np.meshgrid(ax,ax,ax,indexing='ij'),-1); r2=((X-0.5)**2).sum(-1); rho=np.clip(np.exp(-r2/0.08)*(1+0.4*np.cos(30*X[...,0])),0,None); f,rep=m.gabor_volume(rho, K=12); (rep['psnr_db']>0, len(f.lod(1e-9).A)==rep['gaussians'])", native=True, module='gaborfield', aliases=('render clouds with gabor kernels', 'volumetric level of detail without mipmaps', 'fit a volume with oriented primitives', 'closed form ray integral through a cloud', 'prune volume detail by frequency', 'gaussian mixture with wave modulation'), semantic='analyze/measure')

    c.register_capability('Retopo destruction fixes (singular-cell snap + feature-sized lattice)', 'Retopo mesh-destruction fixes (R2+R5): surface_retopo(snap_singular=True) rescues degenerate lattice cells by QEx-style per-vertex re-keying (additive: never changes kept faces); feature_sized=True computes local thickness via feature_size_field (a SpatialMemory recall of the nearest opposing wall) and grades the lattice finer where the surface is thin. MEASURED on a scanned mantis at coarse density: baseline shatters into 12 components; snap alone 5; sizing alone 5; BOTH -> 1 component, intact. Both default OFF; process_scan takes retopo_snap= / retopo_sized=. Gate with m.topology_gate (R1).', example='import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; q,r=m.surface_retopo(loop_subdivide(box(),2), density=1.0, silhouette=None, snap_singular=True, feature_sized=True); len(q.faces)>0', native=True, module='crossfield', aliases=('stop retopo from shattering the mesh', 'rescue dropped cells in quad extraction', 'keep thin legs during retopo', 'feature size aware remeshing', 'fix holes introduced by retopology', 'local thickness field'), semantic='modify/filter')

    c.register_capability('Manifold cleanup (make a retopo strictly manifold for QEM/half-edge)', "Strict-manifold cleanup for retopo (R3): m.manifold_cleanup(mesh) splits non-manifold 'fin' edges so QEM decimate / half-edge consumers ACCEPT the result -- MEASURED on a scan retopo: 142 non-manifold edges -> 0, 1 component preserved, ~93% faces kept, QEM then accepts (LOD-on-retopo unblocked). process_scan(manifold=True) opts in. The cost is honest and REPORTED: a few small holes for strict manifoldness (24 on the mantis). KEPT NEGATIVES: four local surgeries all traded the defect for holes or fragments; a lossless fix needs a manifold-guaranteeing extraction (R3-proper, filed).", example="import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, Mesh; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; b=triangulate_ngons(box()); F=[tuple(int(i) for i in f) for f in b.faces]; a,c,d=F[0]; fin=Mesh(np.asarray(b.vertices,float), F+[(a,d,c)]); out,rep=m.manifold_cleanup(fin); (rep['manifold'], rep['non_manifold_after']==0)", native=True, module='meshtools', aliases=('make retopo mesh manifold for decimation', 'fix fins so qem decimate accepts the mesh', 'strict manifold cleanup with reported cost', 'unblock lod on a retopo mesh', 'remove non-manifold fin edges', 'resolve cone points in a scan retopo'), semantic='modify/filter')

    c.register_capability('Topology gate (reject remeshes that punch holes or shatter components)', 'Topology invariant gate (R1): m.topology_report(mesh) gives PER-COMPONENT V/E/F, euler chi, boundary-loop count + fingerprints, and genus; m.topology_gate(before, after) ACCEPTS a remesh only if components, genus, and boundary loops are preserved -- an INTENDED hole is a loop present in the input, a NEW loop / new component / genus change is destruction, rejected with the violation NAMED. Replaces silent keep_largest amputation (measured: 11% of a scanned mantis dropped) with a loud, retryable verdict; process_scan reports it per shard_cleanup stage as topology_ok / dropped_fraction.', example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; b=triangulate_ngons(box()); ok,rep=m.topology_gate(b,b); (ok, m.topology_report(b)['per_component'][0]['genus']==0)", native=True, module='meshtools', aliases=('did the remesh break the mesh', 'check for new holes after retopo', 'genus and boundary loop check', 'detect mesh fragmentation', 'protect intended holes from being flagged', 'euler characteristic per component'), semantic='analyze/measure')

    c.register_capability('Spatial memory (position hypervectors: closest-point as associative recall)', 'EVERY CLOSEST-POINT IS A RECALL (H5): positions become hypervectors via fractional power encoding (nearby points -> similar vectors, spearman 0.967); nearest-point queries are argmax cosine over an item store -- one matmul, no spatial hash. m.spatial_recall(points, queries, payloads=, k=) returns (indices, resonant payload readout, report). Measured 4.1x vs brute at scan scale; recalled points within 1% of true nearest (p95); colour readout 0.034 RGB. KEPT NEGATIVE: no bundle mode -- FPE keys are correlated and cross-talk in superposition (33% at K=128).', example="import numpy as np, lecore; m=lecore.UnifiedMind(); rng=np.random.default_rng(0); P=rng.random((200,3)); Q=P[:10]+0.01; idx,out,rep=m.spatial_recall(P, Q, payloads=P, k=1); (rep['n_points']==200, idx.shape==(10,1), out.shape==(10,3))", native=True, module='spatialmem', aliases=('find the nearest stored point by similarity', 'position keyed memory', 'encode 3d points as hypervectors', 'closest point without a spatial hash', 'look up what is near a location', 'holographic nearest neighbour'), semantic='analyze/measure')

    c.register_capability('Holographic texture bake (scatter/gather fast path)', "Fast HOLOGRAPHIC texture re-bake via scatter/gather (H1): m.mesh_rebake_texture(src, src_uv, texture, target, method='scatter') SCATTERS source colour into a volumetric grid keyed by 3-D position, then GATHERS colour at every texel in one vectorised pass -- the closest-point projection loop is a hand-rolled scatter/gather. Measured ~1500x faster (62s->0.03s scatter) at colour error 0.066-0.088 RGB. method='project' (default) stays exact. KEPT NEGATIVE: scatter quality is bounded by SOURCE VERTEX DENSITY and two walls in one cell bleed -- opt-in for DENSE scans; raise grid if a feature smears.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import grid; s=grid(8,8,width=1.0,height=1.0); V=np.asarray(s.vertices,float); s.uvs=V[:,:2].copy(); tex=np.zeros((32,32,3)); tex[:,:,0]=np.linspace(0,1,32)[None,:]; mm,uu,img,rep=m.mesh_rebake_texture(s, np.asarray(s.uvs), tex, s, size=128, method='scatter'); (rep['method'], rep['grid']>0)", native=True, module='meshtools', aliases=('fast texture bake', 'holographic rebake', 'scatter gather texture bake', 'bake texture without the closest point loop', 'speed up texture reprojection', 'volumetric colour bake'), semantic='convert/uv')

    c.register_capability('Scan-to-asset pipeline (repair, retopo, LOD, fresh UVs, rebake)', "ONE WORKFLOW to repair a scan and reduce polys, keeping its texture -- in the correct order: repair the ORIGINAL -> retopo the repaired mesh -> LOD (a COARSER RETOPO when retopo=True, because decimating a quad retopo re-shatters it -- measured; QEM decimation when retopo=False) -> shard cleanup -> FRESH per-face atlas + reproject the original texture (rebake; never a transfer of the scan's fragmented uvs). m.process_scan(mesh, uv=, texture=, retopo=, lod=) covers four workflows: retopo+lod, retopo only, lod only, repair only. Returns (mesh, uv, image, report with every stage's numbers).", example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; out,u,img,rep=m.process_scan(triangulate_ngons(box()), retopo=False); ([s['stage'] for s in rep['stages']], rep['faces']>0)", native=True, module='meshtools', aliases=('repair a scan and reduce polys with texture', 'scan to clean textured low poly', 'full mesh processing pipeline', 'repair retopo and rebake in one call', 'clean up a photogrammetry scan for games', 'one call scan to asset'), semantic='analyze/pipeline')

    c.register_capability('Drop small disconnected mesh components (retopo shard cleanup)', 'Remove small disconnected COMPONENTS from a mesh -- the cleanup a field-guided retopo needs, because extracting quads from a scan leaves isolated cells (a mantis retopo shattered into 88 components: one body + ~75 shards that render as speckle and break UV packing). m.mesh_drop_small_components(mesh, keep_largest=True) keeps only the biggest surface; min_faces=N or min_fraction=f keep components above a size threshold. Re-indexes verts, carries uvs/normals. Returns (mesh, report). Built on the shared graph flood. Removes only -- cannot reconnect a split body.', example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import Mesh; V=np.array([[0,0,0],[1,0,0],[0,1,0],[5,5,5],[6,5,5],[5,6,5]],float); mesh=Mesh(V,[(0,1,2),(3,4,5)]); body,rep=m.mesh_drop_small_components(mesh, keep_largest=True); (rep['components_before'],rep['components_after'],rep['faces_after'])", native=True, module='meshtools', aliases=('keep only the largest connected component', 'remove small disconnected pieces', 'drop mesh shards and islands', 'clean up a fragmented retopo', 'keep the biggest surface piece', 'strip loose disconnected geometry'), semantic='modify/filter')

    c.register_capability('Graph connected components (the generic flood fill)', "Partition nodes into CONNECTED COMPONENTS under an undirected edge list -- the generic GRAPH FLOOD FILL under every 'island' in the engine: physics constraint graphs, mesh edge adjacency, conflict graphs, DDM subdomain splits. m.graph_connected_components(n_nodes, edges) returns a list of sorted index lists, ordered by each component's smallest member (deterministic, independent of edge order); isolated nodes are singletons. The reusable primitive that mesh_connected_components and route's component count both delegate to -- one flood fill for every island in the engine.", example='import lecore; m=lecore.UnifiedMind(); comps=m.graph_connected_components(5, [(0,1),(1,2),(3,4)]); (len(comps)==2, comps[0]==[0,1,2], comps[1]==[3,4])', native=True, module='island', aliases=('flood fill a graph', 'partition nodes into connected components', 'split a graph into islands', 'group connected nodes', 'connected components of an edge list', 'label connected graph nodes'), semantic='analyze/measure')

    c.register_capability('Rig from parts (joint tree + skin weights from a segmentation)', 'M2 -- assemble a RIG (joint tree + bound skin weights) from a mesh_parts segmentation (m.rig_from_parts). COMPOSITION of M9 + skin_bind_weights + part adjacency: the core part roots a BFS tree, each elongated limb gets a proximal+distal joint so it can bend, and a LABEL-AWARE bind restricts each vertex to its own + parent part (MEASURED: 57->87% own-part binding, one-limb pose isolated 11000x in-vs-out on the mantis). Feed weights + per-joint transforms to linear_blend_skin to pose. Run mesh_parts on a welded mesh first. Returns a rig dict (joints, bones, parent, joint_part, weights, core).', example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, Mesh; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; S=triangulate_ngons(loop_subdivide(box(),4)); V=np.asarray(S.vertices,float); V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9); d=np.array([0.,-1,0]); V=V+d*(3*np.clip((V@d-0.7)/0.3,0,1)**1.2)[:,None]; mesh=Mesh(V,[tuple(int(i) for i in f) for f in S.faces]); lab,rep=m.mesh_parts(mesh); rig=m.rig_from_parts(mesh,lab,rep); np.allclose(rig['weights'].sum(1),1,atol=1e-6)", native=True, module='meshskin', aliases=('build a rig from segmented parts', 'auto rig a creature from its limbs', 'turn mesh parts into a skeleton', 'make a bone hierarchy and bind weights', 'rig template from part labels', 'assemble joints and skinning from parts'), semantic='create/emit')

    c.register_capability('Holistic lattice cleanup (FHRR resonator factoring of FPE coordinates)', 'R6 (gated) -- factor a BOUND PRODUCT of fractional-power-encoded integer coordinates back to its integers via a Fourier-HRR RESONATOR (Frady/Kent 2020; m.fpe_lattice_resonator). For the HOLISTIC-ONLY regime: coordinates never observed, only the single bound product prod z_a^k (a lattice point stored inside a structure, or under correlated phase noise). Iterated cleanup over power-codebooks converges to the integer tuple -- VERIFIED 200/200 at 0.6 rad noise, 51x51 codebooks in dim 1024. KEPT NEGATIVE: for DIRECT noisy coords np.round dominates (83% at sigma 0.3); do NOT use the resonator there.', example="import numpy as np, hashlib, lecore; m=lecore.UnifiedMind(); b=lambda s:np.exp(1j*np.random.default_rng(int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big')).uniform(-np.pi,np.pi,1024)); zu,zv=b('u'),b('v'); coords,rep=m.fpe_lattice_resonator((zu**7)*(zv**13),[zu,zv],[21,21]); coords==[7,13]", native=True, module='fpe', aliases=('factor a bound product of lattice coordinates', 'recover integer coordinates from a hypervector', 'resonator cleanup to nearest lattice point', 'decode a fractional-power-encoded position', 'snap a holographic coordinate to a lattice', 'factor an fpe product back to integers'), semantic='analyze/measure')

    c.register_capability('Low eigenvectors of an operator (matvec-only, no scipy)', 'The k LOWEST eigenvectors of a Hermitian PSD operator from its MATVEC alone (m.low_eigenvectors) -- the low band (mesh eigenmaps, Fiedler order, modal shapes) where dense eigh is unaffordable. Block shifted inverse iteration on the shared cg. VERIFIED vs eigh on a sphere: residual 2.5e-11. Also reachable as laplacian_eigenbasis(L, n_basis, method=\'iterative\') -- the H3 fold; dense stays the default (KEPT NEG, measured: eigh wins ~30x on a DENSE matvec; this pays only for sparse/implicit operators). Deterministic. Returns (eigenvalues, eigenvectors).', example='import numpy as np, lecore; m=lecore.UnifiedMind(); A=np.random.default_rng(0).standard_normal((30,30)); A=A@A.T; w,U=m.low_eigenvectors(lambda x:A@x,30,float(np.abs(A).sum(1).max()),k=4,dtype=float,shift=float(np.linalg.eigvalsh(A)[0]-0.5),iters=80); np.allclose(np.sort(w),np.linalg.eigvalsh(A)[:4],atol=1e-2)', native=True, module='numerics', aliases=('smallest eigenvectors of a large matrix', 'sparse eigensolver without scipy', 'a few low eigenvectors near a shift', 'inverse iteration eigenpairs', 'fiedler vector via matvec', 'modal shapes of an operator'), semantic='analyze/measure')

    c.register_capability('Mesh as a sequence (SATO-SEQ: stable serialization + hypervector encode)', 'SATO-SEQ -- serialise a mesh to a STABLE token sequence (m.mesh_to_tokens) and bind a sequence into one FHRR hypervector (m.seq_encode / m.seq_decode). Three deterministic vertex orders: morton (Z-order curve, byte-stable under input permutation), zyx (PolyGen lexicographic), fiedler (spectral seriation). Coords quantised to `bits` bits (3 tokens/vertex). Sequence -> hypervector by permutation-power binding; past the ~dim/8 capacity cliff it stores block vectors (round-trips exactly). Clean-room from Morton/PolyGen, NOT the GPL-3.0 SATO code. Returns (tokens, order, grid).', example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; toks,idx,grid=m.mesh_to_tokens(box(),order='morton',bits=8); H=m.seq_encode(toks[:48],dim=1024,seed=0,vocab_size=256); m.seq_decode(H,48,dim=1024,seed=0,vocab_size=256)==toks[:48]", native=True, module='meshseq', aliases=('turn a mesh into a sequence', 'serialize a mesh to tokens', "morton order a mesh's vertices", 'encode a mesh as a hypervector', 'spectral vertex ordering of a mesh', 'tokenize a mesh for a sequence model'), semantic='analyze/measure')

    c.register_capability('Global worst view over the sphere (Lipschitz / DIRECT, no dense sweep)', "M16 -- find the GLOBAL worst view of a mesh over S^2 without a dense turntable sweep (m.worst_view). A per-direction quality metric (silhouette IoU, render error) is optimised on the sphere by branch-and-bound over an icosahedral subdivision. mode='direct' (default) is Lipschitz-CONSTANT-FREE (DIRECT, Jones 1993) -- safe when the metric jumps at occlusion; MEASURED 1704 evals, 0.34 deg from truth, BEATS a 2562 dense sweep. mode='certified' is Piyavskii B&B returning an optimality certificate (needs a Lipschitz bound; costs more). Deterministic. Returns (best_dir, best_value, report).", example="import numpy as np, lecore; m=lecore.UnifiedMind(); g=np.array([0.4,-0.6,0.7]); g=g/np.linalg.norm(g); d,v,rep=m.worst_view(lambda x:float(np.exp(-8*np.arccos(np.clip(np.asarray(x)@g,-1,1))**2)),mode='direct',max_evals=1200); np.degrees(np.arccos(np.clip(d@g,-1,1)))<2.0", native=True, module='worstview', aliases=('find the worst view of a mesh', 'global optimization on the sphere', 'hardest camera angle for a mesh', 'branch and bound worst viewpoint', 'lipschitz search over view directions', 'worst silhouette view without a sweep'), semantic='analyze/measure')

    c.register_capability('Stripe patterns (field-following even stripes on a surface)', 'Knoppel-Crane STRIPE PATTERNS (SIGGRAPH 2015): m.stripe_pattern(mesh, direction_field, frequency) places evenly-spaced stripes that FOLLOW a per-vertex tangent direction field -- the co-oriented iso-lines a quad layout, texture alignment, or hatching wants. ONE smallest-eigenvector problem: Hermitian energy (cotan weights, edge phase increment freq*<edge,field>), smallest eigenvector via the shipped matvec-only eigensolver. MEASURED: phase follows the field to 0.006 rad median edge residual on a sphere. Stripes = level sets of angle(psi); mask cos(angle(psi))>0. Returns (psi, report).', example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, Mesh; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; S=triangulate_ngons(loop_subdivide(box(),3)); V=np.asarray(S.vertices,float); V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9); N=V.copy(); ax=np.array([0.,0,1]); X=ax-N*(N@ax)[:,None]; X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); psi,rep=m.stripe_pattern(Mesh(V,[tuple(int(i) for i in f) for f in S.faces]), X, frequency=18.0); rep['phase_residual_median']<0.05", native=True, module='crossfield', aliases=('stripe pattern on a surface', 'evenly spaced lines aligned to a direction field', 'knoppel crane stripe patterns', 'phase texture following a vector field', 'co-oriented iso-stripes on a mesh', 'hatching aligned to a field'), semantic='create/emit')

    c.register_capability('Mesh Laplacian eigenmaps (cotan spectrum for spectral analysis)', "R6 foundation -- the low SPECTRUM of a mesh's cotan Laplace-Beltrami operator (m.mesh_laplacian_eigenmaps): the eigenfunctions a spectral analysis builds on (spectral segmentation, quadrangulation layout, shape descriptors). Cotan weights (Pinkall-Polthier) + lumped mass, solved as the symmetrised generalised eigenproblem via eigh (exact, fine to a few thousand verts). VALIDATED on a sphere: eigenvalues cluster at l(l+1)=0,2,6,12 and the first eigenspace recovers x,y,z at R2=1.000. SCALAR vertex operator, distinct from the crossfield CONNECTION Laplacian. Returns (eigenvalues, eigenfunctions).", example='import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, Mesh; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; S=triangulate_ngons(loop_subdivide(box(),3)); V=np.asarray(S.vertices,float); V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9); w,phi=m.mesh_laplacian_eigenmaps(Mesh(V,[tuple(int(i) for i in f) for f in S.faces]),k=6); abs(w[0])<1e-5', native=True, module='crossfield', aliases=('laplacian eigenvectors of a mesh', 'eigenfunctions of the mesh laplacian', 'spectral embedding of a surface', 'cotan laplace beltrami spectrum', 'harmonic basis for a mesh', 'shape descriptor from the laplacian'), semantic='analyze/measure')

    c.register_capability('Morse critical points (minima maxima saddles of a scalar field)', 'Count and classify the CRITICAL POINTS (minima, maxima, saddles) of a scalar field on a mesh (m.morse_critical_points) -- the singularity structure a Morse-Smale complex is built from, for spectral quad layout and feature analysis. Discrete lower-star test on each 1-ring; obeys Euler-Poincare (minima - saddles + maxima = chi), verified chi=2 on a sphere. Deterministic (field ties broken by vertex id). Returns {minima, maxima, saddles, indices}.', example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, Mesh; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; S=triangulate_ngons(loop_subdivide(box(),3)); V=np.asarray(S.vertices,float); V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9); c=m.morse_critical_points(Mesh(V,[tuple(int(i) for i in f) for f in S.faces]), V[:,2]); c['minima']-c['saddles']+c['maxima']==2", native=True, module='crossfield', aliases=('critical points of a function on a surface', 'minima maxima and saddles of a field', 'morse smale singularities', 'count saddles on a mesh', 'topological features of a scalar field', 'euler characteristic from critical points'), semantic='analyze/measure')

    c.register_capability('Mesh part segmentation (limbs and body via surface Reeb graph)', "M9 -- segment a mesh into LIMBS AND BODY (m.mesh_parts) via the Reeb graph of geodesic distance on the SURFACE, so thin limbs survive (the voxel skeleton found only 45 points on a mantis's legs; this found 12 parts in 0.2s, each one connected blob, aspect splitting limbs 7.5-13.4 from core 1.2). Dijkstra from an extremity -> distance bands -> connected components per band = Reeb nodes -> branch decomposition -> per-vertex labels; twigs absorbed. Weld scans first. m.match_symmetric_parts pairs left/right limbs. Returns (labels, report).", example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box, Mesh; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; S=triangulate_ngons(loop_subdivide(box(),4)); V=np.asarray(S.vertices,float); V=V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9); d=np.array([0.,-1,0]); V=V+d*(3*np.clip((V@d-0.7)/0.3,0,1)**1.2)[:,None]; lab,rep=m.mesh_parts(Mesh(V,[tuple(int(i) for i in f) for f in S.faces])); rep['n_parts']>=1", native=True, module='skeleton', aliases=('segment a mesh into limbs and body', 'split a creature into parts', 'label the limbs of a model', 'reeb graph part decomposition', 'which vertices belong to which limb', "find a character's arms and legs"), semantic='analyze/measure')

    c.register_capability('Curve skeleton / medial axis of a mesh (interior distance ridge)', "Curve SKELETON / medial axis of a mesh: the ridge (local maxima) of the interior distance field -- the deepest, surface-equidistant points tracing the shape's backbone, for rigging, thickness, and part detection. m.mesh_skeleton(mesh) returns {points, depth=medial radius (local half-thickness), bounds}. GENERALISES existing machines: distance from the shared correspondence (closest_face_point), inside/out from the winding number -- not a new algorithm. Validated: a cylinder's ridge lands on its axis (radial 0.02). KEPT NEGATIVE: a voxel ridge, res-limited, not yet a connected 1-D curve.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_skeleton import _cylinder; sk=m.mesh_skeleton(_cylinder(), res=20); (len(sk['points'])>0, round(float(np.sqrt(sk['points'][:,0]**2+sk['points'][:,1]**2).mean()),2))", native=True, module='skeleton', aliases=('skeleton of a mesh', 'medial axis', 'medial surface', 'centerline of a shape', 'curve skeleton for rigging', 'backbone of a 3d model', 'spine of a model', 'find the bones inside a character', 'auto-rig skeleton extraction', 'thickness / medial radius of a mesh'), semantic='analyze/measure')

    c.register_capability('Bake a displacement (height) map (high to low, same projection as the normal bake)', "m.bake_normal_map(low, low_uv, high, displacement=True, max_distance=D) also bakes a DISPLACEMENT (height) map alongside the normal map, from the SAME closest-point projection -- one cast, two channels read out (the holographic 'add a dimension to one pass, project out what you need' move). Signed: positive=bump, negative=dent, along the low-poly normal. CLAMPED to max_distance -- the cage a displacement map REQUIRES because a stray far hit moves GEOMETRY, not just shading (unlike a normal map). Makes a low-poly render as true high-poly detail (silhouette-changing), not just shaded detail.", example='import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide; hi=loop_subdivide(triangulate_ngons(box()),levels=3); lo,_=m.mesh_decimate_to(hi,target_faces=120,min_silhouette_iou=None); uv=np.asarray(lo.vertices)[:,:2]; uv=(uv-uv.min(0))/(uv.max(0)-uv.min(0)+1e-9); n,d=m.bake_normal_map(lo,uv,hi,size=32,displacement=True,max_distance=0.3); (n.shape, d.shape)', native=True, module='meshtools', aliases=('bake a displacement map', 'height map from high poly to low poly', 'make the low poly have real depth not just shading', 'displacement bake with a cage', 'add high poly detail to a low poly silhouette', 'one pass normal and displacement'), semantic='create/emit')

    c.register_capability("Turntable silhouette sweep (fast orthographic 3-D preservation check)", "m.silhouette_sweep(ref_mesh, mesh, n_azimuth=6) is the fast 3-D preservation check behind the default-on modification guards -- the shape analogue of validating a denoise against its signal: rotate the pair under a fixed ORTHOGRAPHIC camera (azimuths across [0,pi); theta and theta+pi give the same outline, a symmetry perspective breaks) plus the top, mask each silhouette (edge-sample + flood fill, no shading), and score IoU per direction under the REFERENCE's frame. ~2s warm on 322k faces; ranks degradation like the perspective critic. Returns {iou, worst, worst_view, mean, seconds}.", example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons; b=triangulate_ngons(box()); r=m.silhouette_sweep(b, b, n_azimuth=4, size=64); (r['worst'], r['mean'])", native=True, module="render", aliases=("check the silhouette survived decimation", "compare model outline before and after", "did optimization change the shape", "rotating silhouette comparison", "fast shape preservation check", "silhouette iou sweep"), semantic="analyze/measure")

    c.register_capability("Decimate to a target face count / fraction with an optional silhouette guard", 'm.mesh_decimate_to(mesh, target_faces=N | target_fraction=p, min_silhouette_iou=x) is decimation UNDER CONTROL: an explicit face budget hit by deterministic bisection (grid is monotone in faces), and an OPTIONAL silhouette guard -- the outline is scored vs the SOURCE from 4 views and the search walks BACK if the WORST view drops below the floor, shipping more faces than asked LOUDLY (report.budget_missed_for_silhouette) instead of silently slurped limbs (crab: asked 3000, shipped 15215 at >=0.97). No target -> mesh UNTOUCHED: never-modify is a policy. Returns (mesh, report).', example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_meshtools import _uv_sphere_fixture; s=_uv_sphere_fixture(24); out, rep = m.mesh_decimate_to(s, target_fraction=0.3, keep_uv=False); (rep['modified'], rep['budget_error'] < 0.35)", native=True, module="meshqem", aliases=("decimate to a target face count", "reduce mesh to a percentage", "limit decimation so the shape survives", "dont let optimization destroy the model", "keep the silhouette while simplifying", "control how much a mesh is reduced"), semantic="modify/weld")

    c.register_capability("Reproject a uv map onto changed topology (seam-aware)", "m.mesh_reproject_uv(source, source_uv, target) puts a uv map back on a mesh whose FACE COUNT CHANGED (decimate, remesh, retopo) so the texture lines up. Per-CORNER and cut-aware: a retopo WELDS both sides of a seam into ONE vertex, which cannot carry a seam's two uvs, so per-vertex transfer smears the faces there. Side is a per-corner CONSTRAINT (majority-vote home, ambiguous samples abstain). Measured: cylinder 3.36% pixels smeared -> 0.00%; sphere incl. poles -> 0 defects. Returns (mesh, uv, report). keep_uv='auto' calls it. Fragmented scan atlas -> raises, names mesh_rebake_texture.", example="import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_meshtools import _uv_cylinder_fixture; from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate; src=_uv_cylinder_fixture(); lod=cluster_decimate(src, grid=7, keep_uv=False); mesh, uv, rep = m.mesh_reproject_uv(src, np.asarray(src.uvs), lod); (rep['seam_splits'], rep['finite'])", native=True, module="meshtools", aliases=("reproject uv after decimation", "keep uvs through retopo", "texture doesnt line up after optimizing", "uvs lost after remesh", "transfer texture coordinates to new topology", "my texture is smeared at the seam"), semantic="convert/uv")

    c.register_capability("Pose a rigged asset at a time (animation + skin -> moving geometry)", "m.pose_asset(loaded_mesh, time=t) turns an imported rig into geometry that MOVES: samples the clip (un-animated paths keep the node's REST value -- a rotation-only bone must not lose its offset), composes the hierarchy to world, builds joint matrices from the inverse-bind, and linear-blend-skins. Returns (Mesh, report); report['mode'] = animated / bind_pose / bind_pose_skinned. Every piece existed, nothing composed them, so rigged .glb files sat in bind pose forever. Pinned on ANALYTIC truth: a 90-degree swing lands within 4e-16. KEPT NEGATIVE: linear blend skinning.", example="import lecore, tempfile, os; m=lecore.UnifiedMind(); from holographic.io_and_interop.holographic_assetimport import load_glb, _bone_glb; fd,p=tempfile.mkstemp(suffix='.glb'); os.write(fd,_bone_glb()); os.close(fd); lm=load_glb(p); posed,rep=m.pose_asset(lm, time=1.0); os.unlink(p); (rep['mode'], rep['joints'])", native=True, module="assetimport", aliases=("rigged glb doesnt move", "play an animation on an imported model", "pose a character at a time", "apply bone animation to a mesh", "skeleton deform imported model", "my glb animation does nothing"), semantic="animate/pose")

    c.register_capability("One-call textured preview of an asset file", "PREVIEW an .obj/.glb/.gltf WITH ITS OWN TEXTURE in one call (holographic_assetimport.preview_asset): m.preview_asset(path) imports the file with materials + embedded textures (load_glb / load_obj), attaches the uvs, normalises the base-colour map, auto-frames a camera from the bounds, and rasterizes textured+smooth. Returns (image, LoadedMesh). Every piece existed; the COMPOSITION did not -- a debugging arc rendered with a synthetic checker because nothing pointed from import to textured render. Validated by uv-readback: rendered surface == texture(mesh uvs), mean err 0.009.", example="import lecore; m=lecore.UnifiedMind(); # img, lm = m.preview_asset('model.glb'); img.shape", native=True, module="assetimport", aliases=("render a glb with its texture", "textured preview of an imported model", "show my model with its materials", "preview a gltf file", "render an asset file with textures", "see the real texture on my mesh"), semantic="render/raster")

    c.register_capability("Textured LOD that routes by measurement (atlas report + re-bake)", "A decimated mesh that STILL WEARS ITS TEXTURE: m.mesh_textured_lod(mesh, texture) measures the atlas (m.uv_atlas_report) and picks the route -- coherent atlas -> cheap uv transfer; fragmented scan atlas -> re-bake into a new per-face atlas (m.mesh_rebake_texture). WHY: a scan's atlas had 4079 islands at a MEDIAN OF 1 FACE, so per-vertex transfer put 90% of LOD faces across island boundaries and rendered as speckle, no error raised. Measured: speckle energy 0.136 -> 0.054 (source 0.055); render error 0.120 -> 0.057. keep_uv='auto' now REFUSES fragmented transfers and names the right route.", example="import numpy as np, lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import grid; s=grid(6,6,width=1.0,height=1.0); V=np.asarray(s.vertices,float); s.uvs=V[:,:2].copy(); r=m.uv_atlas_report(s); (r['islands'], r['transferable'])", native=True, module="meshtools", aliases=("texture looks speckled after decimation", "lod loses its texture", "uv transfer scrambles my scan texture", "rebake texture onto a decimated mesh", "will my uvs survive retopo", "textured level of detail"), semantic="convert/uv")

    c.register_capability("Texture-preserving mesh repair & decimation (attribute-aware weld)", "FIX for 'losing texture information' in mesh optimization: merge_by_distance/mesh_repair are ATTRIBUTE-AWARE (attrs='auto') -- welds only vertices agreeing in position AND uv AND normal (the glTF render-duplicate weld), so UV-SEAM splits stay split and arrays are CARRIED corner-exact (pinned). Measured on a .glb scan: ALL 4956 duplicate groups were seams -- the old position-only weld scrambled the atlas and dropped uvs. cluster_decimate/voxel_remesh now PROJECT uvs via transfer_uv; qem already carried. Attr-free meshes: bit-identical old path.", example="import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import Mesh; V=np.array([[0,0,0],[0,0,0],[1,0,0],[1,0,0],[0,1,0],[2,1,0]],float); UV=np.array([[.2,.2],[.2,.2],[.1,.5],[.9,.5],[.3,.3],[.7,.7]]); r,rep=m.mesh_repair(Mesh(V,[(0,2,4),(1,3,5)],uvs=UV), fill_holes=False); (rep['uvs_carried'], len(r.vertices))", native=True, module="meshtools", aliases=("texture lost after mesh cleanup", "repair strips my uvs", "keep uvs when decimating a mesh", "weld destroys uv seams", "mesh optimization loses texture coordinates", "preserve texture through remesh"), semantic="modify/weld")

    c.register_capability("Robust mesh-to-SDF sign for scan soups (winding number)", "FIX for open/scan meshes shredding in mesh->SDF conversion: m.mesh_to_sdf_grid(mesh, bounds, sign='auto') and m.voxel_remesh(mesh, sign='auto') route edge-closed meshes to the original flood path BIT-IDENTICALLY, and meshes with boundary edges (a Sketchfab .glb scan measured 71% boundary -- flood leaked, marched garbage blobs) to the GENERALISED WINDING NUMBER sign (Jacobson 2013) via fast cluster-dipoles (Barill 2018; 113x measured over the exact sum). Pinned: slit-sphere soup interior signed 4% by flood vs 100% by winding at equal res.", example="import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; g,axes=m.mesh_to_sdf_grid(box(), ((-1.2,-1.2,-1.2),(1.2,1.2,1.2)), res=16, sign='auto'); (g.shape, float(g.min())<0)", native=True, module="meshbridge", aliases=("glb import renders as garbage blobs", "voxel remesh shreds my scanned mesh", "open mesh to sdf conversion broken", "fix inside outside for triangle soup", "winding number sign for mesh to field", "imported scan becomes disconnected chunks"), semantic="convert/isosurface")

    c.register_capability("CAD mass properties (volume / COM / inertia tensor)", "MASS PROPERTIES of a closed triangle mesh (holographic_meshtools.mass_properties): m.mass_properties(mesh, density=1.0) returns exact VOLUME, surface AREA, CENTRE OF MASS, MASS, the full INERTIA TENSOR about the COM, and PRINCIPAL moments + axes -- signed-tetrahedron integration with the Tonon (2004) covariance formula, shipped correctly once (the naive re-derivation yields impossible NEGATIVE moments; a selftest pins that). Negative volume flags inward winding. Deterministic, exact on analytic solids (cube to 1e-12).", example="import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; mp=m.mass_properties(box()); (round(mp['volume'],6), mp['principal_moments'])", native=True, module="meshtools", aliases=("volume and center of mass of a mesh", "inertia tensor of a solid", "moment of inertia of a 3d model", "how heavy is this mesh", "principal axes of a part", "cad mass properties"), semantic="measure/area", consumes=("mesh",))

    c.register_capability("Exact planar cross-section (area / perimeter / contours)", "CROSS-SECTION a triangle mesh with a plane (holographic_meshtools.section): m.mesh_section(mesh, plane_point, plane_normal) returns the exact enclosed AREA (winding-signed shoelace over the triangle/plane segments -- holes subtract automatically), PERIMETER, CONTOUR count, and the world-space POLYLINES. No rasterising or field sampling -- the numeric contour, from the geometry itself. Unit cube at mid-height: area 1, perimeter 4, 1 contour, to 1e-12.", example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; s=m.mesh_section(box(), (0,0,0.0), (0,0,1)); (round(s['area'],6), s['contours'])", native=True, module="meshtools", aliases=("cut a mesh with a plane and measure", "cross section area of a solid", "slice a model and get the outline", "section plane through a part", "measure a cut plane", "contour where a plane cuts a mesh"), semantic="measure/area", consumes=("mesh",))

    c.register_capability("Draft-angle moldability report (mesh)", "MOLDABILITY report for a triangle mesh vs a pull direction (holographic_meshtools.draft_report): m.draft_report(mesh, pull_dir, min_draft_deg=2) returns area-weighted MOLDABLE / PARTING (near-vertical, risky) / UNDERCUT fractions plus the full per-face draft-angle distribution -- READ-ONLY numbers, not painted faces. Complements draft_angle (per-point, parametric surfaces). Cube vs +Z: 1/6 moldable, 4/6 parting, 1/6 undercut, exact.", example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; r=m.draft_report(box(), (0,0,1)); (round(r['undercut_fraction'],4), round(r['parting_fraction'],4))", native=True, module="meshtools", aliases=("can this part be molded", "draft angle report", "undercut check on a mesh", "moldability analysis", "which faces are undercuts", "injection molding draft check"), semantic="measure/curvature", consumes=("mesh",))

    c.register_capability("Oriented bounding box (minimal-volume OBB)", "ORIENTED bounding box of a point set (holographic_fitshape.oriented_bbox): m.oriented_bbox(points) -> {center, axes, half_extents, volume} via PCA seed + coarse-to-fine rotation refinement, with a hard AABB FALLBACK so the OBB is NEVER worse than the axis-aligned box (a PCA-only OBB on an aligned cube can come out LARGER -- a real observed bug the fallback kills, pinned by selftest). A 45-degree-rotated box recovers ~its true volume where the AABB inflates 40%+. Deterministic, NumPy-only.", example="import lecore, numpy as np; m=lecore.UnifiedMind(); pts=np.random.default_rng(0).uniform(0,1,(200,3))*[1,2,3]; r=m.oriented_bbox(pts); (r['half_extents'].round(2), round(r['volume'],3))", native=True, module="fitshape", aliases=("tightest box around points", "oriented bounding box", "minimal bounding box of a model", "obb of a point cloud", "fit a rotated box", "bounding box that follows the shape"), semantic="measure/bounds", consumes=("points",))

    c.register_capability("Hydraulic terrain erosion (droplet simulation)", "ERODE a height grid hydraulically (holographic_terrain.erode): m.terrain_erode(height, droplets, steps, seed) runs the classic droplet simulation -- momentum downhill walk, capacity-limited sediment pickup, deposition on overload/uphill, evaporation, radius-brushed carving so channels have WIDTH. Carves drainage, softens peaks (max never grows). Additive: returns an eroded COPY. Deterministic under seed. NOTE: material leaving the tile edge is lost, like real drainage.", example="import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_terrain import Terrain; h=Terrain(seed=3).heightmap(48); e=m.terrain_erode(h, droplets=300, steps=20); round(float(abs(e-h).sum()),3)", native=True, module="terrain", aliases=("erode a terrain heightmap", "carve rivers into terrain", "hydraulic erosion", "make procedural terrain look weathered", "water erosion simulation", "drainage channels on a landscape"), semantic="simulate/run", consumes=("field",), produces=("field",))

    c.register_capability("Camera from vanishing points (focal + orientation)", "CALIBRATE a camera from two vanishing points of ORTHOGONAL line families (holographic_hazedepth.camera_from_vanishing_points): m.camera_from_vanishing_points(vp1, vp2, principal_point) -> {focal, R, principal_point} via the Caprile-Torre orthogonality relation f=sqrt(-(v1-pp).(v2-pp)) and Gram-Schmidt orientation. Consumes VP coords from vanishing_point() detection or user clicks. REFUSES a geometrically impossible pair (imaginary focal) instead of returning garbage. Round-trip selftest: f to 1e-6, axes to 1e-9.", example="import lecore; m=lecore.UnifiedMind(); cam=m.camera_from_vanishing_points((1120,240), (-480,290), (320,240)); round(cam['focal'],1)", native=True, module="hazedepth", aliases=("focal length from vanishing points", "calibrate camera from a single photo", "camera orientation from parallel lines", "estimate camera intrinsics from perspective", "vanishing point calibration", "recover camera from two vps"), semantic="analyze/measure", consumes=("points",))

    c.register_capability("Native batch kernels via the system C compiler", "C-COMPILER twin of the Zig runner (holographic_ccrun): m.c_batch_eval(kernel_source, [x, y, z]) emits the SAME c_f64/c_f32 IR the emitter already validates, compiles with cc/gcc/clang -O3 -shared, ctypes-calls the SoA batch loop, content-addressed cache (hashlib). Works wherever a C compiler exists -- i.e. almost everywhere Zig does not. f64 is BIT-IDENTICAL to the Python kernel; f32 ~3e-7 measured. REFUSES loudly with no compiler. KEPT NEG: no SIMD dialect -- -O3 autovectorizes; a hand-vector C path was maintenance without a measured win.", example="import lecore, numpy as np; m=lecore.UnifiedMind(); src='def k(x: float) -> float:\\n    return sqrt(x*x + 1.0)\\n'; m.c_batch_eval(src, [np.arange(4.0)])", native=True, module="ccrun", aliases=("compile a kernel with gcc", "native speedup without zig", "run sdf kernel as compiled c", "jit to c and run", "batch evaluate with a c compiler", "fast native kernel fallback"), semantic="simulate/run", )

    c.register_capability("True import footprint of an entry point (bundler's answer)", "WHAT DOES THIS ACTUALLY NEED to import (holographic_deptrace.footprint_report): m.import_footprint('lecore') returns the REQUIRED module closure vs what a naive follow-every-import tracer reports, plus required_external (the pip packages that must exist) and optional_external. Classifies each import by WHERE it sits: hard (top level, fatal if missing), guarded (inside try -- opt-in accelerator), deferred (inside a function -- never runs at import). MEASURED: import lecore needs 30 modules and numpy alone; a naive tracer says 499 (16.6x). ast-only, never imports the code.", example="import lecore; m=lecore.UnifiedMind(); r=m.import_footprint('lecore'); (r['required'], r['naive'], r['required_external'])", native=True, module="deptrace", aliases=("what modules does this actually need at runtime", "minimal dependency set for bundling a subset", "which third party packages does this code really need", "true dependency footprint", "what must i ship to embed this", "is anything importing torch at module level", "bundle a subset of the engine", "vendor part of lecore into another project"), semantic="analyze/describe")

    c.register_capability("Classify every import as hard / guarded / deferred", "IMPORT GRAPH with positions (holographic_deptrace.trace / import_edges): m.trace_imports(entry, follow=('hard',)) walks the closure and labels every edge HARD (module top level -- runs on import, ImportError fatal), GUARDED (lexically inside try -- optional accelerator, failure survivable), or DEFERRED (inside a function -- does not run at import at all). Returns modules, external/stdlib split, edge counts, unresolved. MEASURED on this engine: 1024 hard, 3 guarded, 2662 deferred -- the balloon is deferred self-imports, NOT try/except accelerators.", example="import lecore; m=lecore.UnifiedMind(); t=m.trace_imports('holographic.io_and_interop.holographic_ccrun'); (t['modules'], t['edges_by_kind'])", native=True, module="deptrace", aliases=("trace imports of a module", "static import graph of the engine", "find optional accelerator imports", "which imports run at import time", "are these imports lazy or eager", "import dependency analysis"), semantic="analyze/describe")

    c.register_capability("Collapse nodes into one reusable subgraph node", "GROUP a selection into ONE node (NodeGraph.collapse): g.collapse([n1, n2]) contracts the selection into a single subgraph node, re-pointing every external wire so the graph computes EXACTLY what it did before -- a refactor, not an edit. External sources become typed group INPUT sockets (deduped); inner outputs that feed outside, PLUS any with no consumer, become OUTPUTS (so collapsing a TERMINAL selection stays readable). Nests recursively; JSON-serializable into a FRESH registry. REFUSES a cycle-creating collapse, leaving the graph untouched. g.expand(id) is the inverse.", example="import lecore; m=lecore.UnifiedMind(); g=m.node_graph(); a=g.add('sdf_sphere', {'radius':1.0}); b=g.add('sdf_box'); u=g.add('sdf_union'); g.connect(a,'out',u,'a'); g.connect(b,'out',u,'b'); gid=g.collapse([a,b]); (gid, sorted(g.nodes))", native=True, module="nodegraph", aliases=("collapse nodes into a group", "make a reusable node group", "group selected nodes", "nested subgraph inside a node graph", "macro node from a selection", "node group like blender"), semantic="modify/graph")

    c.register_capability("Expand a subgraph node back into its nodes", "UNGROUP a subgraph node (NodeGraph.expand): g.expand(node_id) pastes the inner nodes back into the outer graph (ids re-prefixed so they cannot collide), re-attaches the original external sources and sinks, and returns the new ids -- the exact inverse of collapse, so grouping is never a one-way door. Result is unchanged by the round-trip (pinned by selftest). Raises ValueError on a node that is not a subgraph, KeyError on an unknown id.", example="import lecore; m=lecore.UnifiedMind(); g=m.node_graph(); a=g.add('sdf_sphere', {'radius':1.0}); b=g.add('sdf_box'); u=g.add('sdf_union'); g.connect(a,'out',u,'a'); g.connect(b,'out',u,'b'); gid=g.collapse([a,b]); (g.expand(gid), sorted(g.nodes))", native=True, module="nodegraph", aliases=("ungroup a node group", "expand a subgraph node", "flatten a nested node graph", "break apart a group node", "inline a subgraph", "undo a node collapse"), semantic="modify/graph")

    c.register_capability("Delete a node from a node graph", "REMOVE a node in place (NodeGraph.remove): g.remove(node_id) deletes the node and prunes every incident edge in O(edges), invalidating downstream memo entries -- the editor verb whose absence forced a serialize-drop-rebuild O(graph) workaround in every node-editor UI. Unknown id raises KeyError (a typo'd delete fails loudly). NOTE: a downstream node whose REQUIRED input lost its wire fails at evaluate time -- remove prunes topology, it does not invent defaults.", example="import lecore; m=lecore.UnifiedMind(); g=m.node_graph(); a=g.add('sdf_sphere', {'radius':1.0}); g.remove(a); a in g.nodes", native=True, module="nodegraph", aliases=("delete a node from the graph", "remove node and its connections", "node editor delete", "drop a node from a nodegraph", "prune a node", "erase a graph node"), semantic="modify/graph", )

    c.register_capability("Route a mesh to its minimal repair (defect-classified)", "ROUTE a mesh to the MINIMAL repair its defect needs (holographic_meshtools.route_repair), not the full pipeline: m.route_repair(mesh) diagnoses a categorical defect record {manifold, closed, duplicates}, MATCHES it against repair-strategy records (match_record), runs only the winning strategy ops -- a duplicate-only mesh welds with no hole-fill. Ambiguous defect -> decide_or_abstain falls back to full mesh_repair, so it never repairs LESS than needed. Returns (mesh, report) with {strategy, confident, defect}. Cheaper, self-explaining. KEPT NEG: categorical presence-of-defect, not hole SIZE.", example="import lecore; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import box; rm,rep=m.route_repair(box()); print(rep['strategy'], rep['confident'])", native=True, module="meshtools", aliases=("route a mesh defect to the right repair", "minimal mesh repair", "pick the repair a mesh needs", "diagnose and fix a mesh", "targeted mesh cleanup", "which mesh repair to run"), semantic="create/emit", consumes=("mesh",), produces=("mesh",))
    c.register_capability("Make a mesh manifold (split non-manifold vertices)", "MAKE A MESH MANIFOLD by splitting non-manifold vertices into connected UMBRELLAS (split_nonmanifold_vertices): incident faces are grouped across MANIFOLD edges only; a vertex whose faces form >1 umbrella (a bowtie, or an edge shared by >2 faces) is duplicated per umbrella. Resolves non-manifold EDGES too, so a cross-field retopo (which REFUSES a non-manifold mesh) accepts it. Unlike mesh_rip_vertex or mesh_split_vertices, this is the MINIMAL cut, a NO-OP on a clean mesh. Returns (mesh, report). KEPT NEG: a pure X-junction over-splits into disconnected sheets.",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(); from holographic.mesh_and_geometry.holographic_mesh import Mesh; book=Mesh(np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1.],[0,0,-1]]),[(0,1,2),(0,1,3),(0,1,4),(0,1,5)]); mm,rep=m.mesh_make_manifold(book); (mm.is_manifold(), rep['split_vertices'])",
                          native=True, aliases=("make a mesh manifold", "split non-manifold vertices", "fix non-manifold edges",
                                                "resolve a bowtie vertex", "cut non-manifold edges", "manifold repair", "unfan a vertex"),
                          semantic="create/emit", consumes=("mesh",), produces=("mesh",))
    c.register_capability("mesh_bevel_vertex", "BEVEL / CHAMFER a corner (holographic_meshverbs2) -- pull each edge "
                          "incident to a vertex back by `ratio` and cap the hole. segments=1 caps with one FLAT "
                          "facet; segments>=2 ROUNDS the corner into a smooth spherical dome (the 'bevel with N "
                          "segments' fillet). Preserves closed + manifold. The VERTEX bevel (edge bevel deferred)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import box; "
                          "print(m.mesh_bevel_vertex(box(2,2,2),0,ratio=0.3,segments=3).n_faces)",
                          native=True, aliases=("bevel a vertex", "chamfer a corner", "bevel with segments",
                                                "rounded bevel", "multi-segment bevel", "round a corner into a fillet",
                                                "smooth a sharp corner", "bevel a corner"),
                          semantic="modify/bevel", consumes=("mesh",), produces=("mesh",))
    c.register_capability("solidify_mesh", "SOLIDIFY / SHELL a mesh (holographic_meshtools) -- give a surface "
                          "thickness by offsetting a copy along the vertex normals, adding it as a reversed-winding "
                          "back wall, and bridging the open rim so the result is a CLOSED watertight solid. An open "
                          "sheet becomes a thick slab; a closed mesh becomes a hollow double wall. The 'solidify' "
                          "modifier",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import grid; "
                          "print(m.solidify_mesh(grid(4,4),0.2).is_closed())",
                          native=True, aliases=("solidify a mesh", "thicken a surface", "give a surface thickness",
                                                "add thickness to a mesh", "shell a surface", "make a hollow shell",
                                                "shell modifier", "turn a sheet into a solid slab"),
                          semantic="modify/extrude", consumes=("mesh",), produces=("mesh",))
    c.register_capability("mesh_symmetrize", "SYMMETRIZE a mesh across a plane (holographic_meshtools) -- keep the "
                          "half on one side, mirror it back, weld the seam, giving a bilaterally-symmetric mesh. "
                          "Unlike mirror (which doubles the whole mesh), this DISCARDS the far side first, so it "
                          "FIXES an off-axis sculpt instead of preserving the asymmetry. Composes mirror + weld",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import grid; "
                          "print(m.mesh_symmetrize(grid(6,6),axis=0).n_faces)",
                          native=True, aliases=("symmetrize a mesh", "make a mesh symmetric", "enforce symmetry",
                                                "mirror and weld one half", "fix an asymmetric mesh",
                                                "bilateral symmetry on a mesh", "make a sculpt symmetric"),
                          semantic="modify/deform", consumes=("mesh",), produces=("mesh",))
    c.register_capability("mesh_triangulate", "EAR-CLIP every face of a mesh into triangles "
                          "(holographic_meshverbs2), returning an all-triangle Mesh. The CONCAVE-CORRECT triangulate "
                          "(unlike the kernel's fan triangulate, which is convex-only): ear clipping (Meisters 1975) "
                          "tiles a concave n-gon exactly instead of the overlapping triangles a fan gives. No new "
                          "vertices, only the face list changes",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import box; "
                          "print(all(len(f)==3 for f in m.mesh_triangulate(box(2,2,2)).faces))",
                          native=True, aliases=("triangulate a mesh", "triangulate ngon faces",
                                                "ear clip a polygon", "convert quads to triangles",
                                                "triangulate concave faces", "quad to triangle conversion",
                                                "split polygons into triangles"),
                          semantic="convert/emit", consumes=("mesh",), produces=("mesh",))
    c.register_capability("mesh_poke", "POKE a polygon face (holographic_eulerops, FWD-7) -- add a vertex at the "
                          "face centroid (pushed out along the normal by height) and FAN the face into triangles, "
                          "one per edge. An n-gon becomes n triangles. V+1/E+n/F+(n-1), chi unchanged. Fan a quad to "
                          "triangles or raise a spike; the inverse of dissolving the center vertex",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.mesh_and_geometry.holographic_mesh import box; "
                          "print(m.mesh_poke(box(2,2,2),0,height=0.3).n_faces)",
                          native=True, aliases=("poke a face", "fan a face into triangles", "raise a spike on a face",
                                                "triangulate a face from its center", "add a center vertex to a polygon",
                                                "poke faces", "center-split a polygon"),
                          semantic="modify/subdivide", consumes=("mesh",), produces=("mesh",))
    c.register_capability("io_kinds", "the closed vocabulary of io DATATYPE kinds a capability can consume/produce "
                          "(holographic_iokinds) -- mesh, points, sdf, sdf_scene, field, image, hypervector, "
                          "transform, selection, scalar, curve, skeleton. The kinds the accepts=/produces= filter "
                          "and suggest_pipeline route over",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); print(m.io_kinds())",
                          native=True, aliases=("what datatypes exist", "list io kinds", "capability datatypes",
                                                "valid input output types", "what kinds can capabilities take"),
                          semantic="analyze/pipeline")
    c.register_capability("suggest_pipeline", "propose a PIPELINE from one datatype to another (holographic_catalog "
                          "+ holographic_iokinds) by chaining capabilities whose produces feeds the next's consumes. "
                          "Returns the shortest chain of {name, consumes, produces} steps, or None. The render-graph "
                          "idea over the whole catalog: the engine proposes a ROUTE from what you have to what you "
                          "want, not just one capability",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.suggest_pipeline('transform','selection'))",
                          native=True, aliases=("how do I get from points to a mesh", "chain capabilities",
                                                "build a pipeline", "route between datatypes",
                                                "what steps turn X into Y"),
                          semantic="analyze/pipeline")
    c.register_capability("find_capability_uris", "like find_capability but each result carries its disambiguating "
                          "capability URI(s) (holographic_catalog + holographic_capuri) so a caller NEVER gets a "
                          "bare ambiguous name. Returns [{name, does, example, uris}] -- one path for a unique name, "
                          "several for a colliding one. The collision fix at the discovery layer",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.find_capability_uris('snap to grid')[0]['uris'])",
                          native=True, aliases=("search capabilities with paths", "find a capability and its uri",
                                                "disambiguated capability search", "capability search with uris",
                                                "find functionality with full paths"),
                          semantic="analyze/pipeline")
    c.register_capability("pipeline_map", "the WHOLE workflow graph as data (pipelinemap + holographic_catalog): "
                          "every typed edge consume_kind->produce_kind->capability derived from the live "
                          "consumes/produces tags, plus per-kind producers/consumers, tag coverage, and a GAP "
                          "report (dead-end kinds produced-but-unconsumed, source-only kinds, untouched kinds). "
                          "Where suggest_pipeline answers ONE route, this is the whole map to plan over; also "
                          "writes docs/PIPELINE_MAP.md (mermaid) + pipelines.json",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.pipeline_map()['coverage'])",
                          native=True, aliases=("map the workflow", "workflow map", "pipeline diagram",
                                                "how do tools connect", "graph of tool inputs and outputs",
                                                "which tools feed which", "auto document the pipelines",
                                                "show the whole pipeline graph", "capability dependency graph"),
                          semantic="analyze/pipeline")
    c.register_capability("route_semantic", "route a request to the right MODULE by COSINE in nomic's embedding "
                          "space instead of token overlap -- catches meaning when words don't match ('squish a big "
                          "array down for storage' -> holographic_coldstore). Uses the shipped 96 KB 64d q8 index. "
                          "Takes a query vector, a build-time-cached phrase, OR free text when the N31 offline embedder ships "
                          "(SIF token-pool + ridge W, no model); returns None (caller falls back to token find_capability) "
                          "rather than fabricate an embedding. Measured 7/12 top-1 vs token 2/12",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.route_semantic('make my picture less grainy'))",
                          native=True, aliases=("route by meaning not keywords", "semantic search for a module",
                                                "find the module that means this", "cosine route a request",
                                                "which module handles this by meaning", "embedding router"),
                          semantic="analyze/route")
    c.register_capability("workflow_neighbors", "WHICH MODULES WORK TOGETHER (holographic_workflowgraph): the sparse workflow bones, from cross-references authors already wrote in docstrings (module A naming holographic_B). Edges are RARITY-weighted (a reference to a module few others mention counts more), hubs dropped, so bones stay SPECIFIC -- median out-degree 2 vs the io-kind graph 13-24. m.workflow_neighbors(module) -> [(module, weight)]; direction out/in/both. E.g. meshsmooth->graphsignal, resonator->chunkcodebook. KEPT NEG: author-stated, coverage uneven; relatedness, not runnable dataflow (the io graph does that).", example="import lecore; m=lecore.UnifiedMind(); print([n for n,_ in m.workflow_neighbors('meshsmooth', top=3)])", native=True, module="workflowgraph", aliases=("which modules work together", "related modules", "what modules go with this one", "module cross references", "workflow adjacency", "what should I use alongside this"), semantic="analyze/route")
    c.register_capability("workflow_propagate", "SPREAD scores one hop along the WORKFLOW BONES (holographic_workflowgraph.propagate): a module whose COLLABORATORS are strongly scored gets lifted even if its own text was never matched -- the structural complement to dense cosine and BM25, which both need shared words. m.workflow_propagate({module: score}) -> [(module, score)] best-first; alpha weights propagation vs the seed, alpha=0 returns the seed unchanged (sanity check). The mechanism for surfacing a module the query has NO vocabulary overlap with. KEPT NEG: ONE hop only -- multi-hop re-diffuses toward the smeared io-kind regime.", example="import lecore; m=lecore.UnifiedMind(); print(m.workflow_propagate({'mesh': 1.0}, alpha=0.8)[:2])", native=True, module="workflowgraph", aliases=("spread scores across related modules", "propagate activation along a graph", "lift related modules", "structural routing signal", "boost neighbors of a match", "graph propagation of relevance"), semantic="analyze/route")
    c.register_capability("bm25_rank", "INTERNAL ARM; front door is retrieval_dispatch. LEXICAL ranking by Okapi BM25 (holographic_bm25): rank a list of text docs by exact-term match to a query, with tf-saturation (k1) and length normalization (b). Pure NumPy/stdlib, no model. The complement to route_semantic's dense cosine -- catches asks whose query WORDS appear in the target text but whose embedding-geometry buries them (measured: 'bumpy surface'->meshsmooth, dense r22, BM25 top-5). Returns [(doc_index, score)]. KEPT NEG: cannot match a word absent from the docs (bag-of-words, no meaning).", example="import lecore; m=lecore.UnifiedMind(); print(m.bm25_rank('smooth bumpy surface', ['smooth a bumpy surface mesh','fluid solver'])[:1])", native=True, module="bm25", aliases=("keyword search over text", "bm25 lexical ranking", "rank documents by term overlap", "exact word match retrieval", "tf-idf style document ranking", "which text matches these keywords"), semantic="analyze/route")
    c.register_capability("guard_candidates", "RECALL GUARD (holographic_recallguard): wrap any ranked list with exact-containment tiers from a perfect_recall_index until a budget fills, returning candidates + a CERTIFICATE of the coordination level down to which completeness is guaranteed (every doc sharing >= c query terms is present -- a theorem, verified exhaustively in the selftest). Ranked head preserved. MEASURED (real NFCorpus vs BM25 top-200): reachable misses 1165 -> 732, Recall@1000 0.278 -> 0.311. KEPT NEG: lexically-reachable docs only; oversized low tiers make the certificate admit less, never lie.", example="import lecore; m=lecore.UnifiedMind(); ix=m.perfect_recall_index(tile=8); ix.add({'token':['cat','sat']}); ix.add({'token':['cat']}); print(m.guard_candidates([1],['cat','sat'],ix,budget=10))", native=True, module="recallguard", aliases=("guarantee nothing relevant is missed", "make search results provably complete", "certificate of retrieval completeness", "recover documents the ranker dropped", "safety net under ranked search", "exhaustive candidate generation with proof"), semantic="analyze/route")
    c.register_capability("perfect_recall_index", "GUARANTEED perfect recall (holographic_perfectrecall): exact AND-containment queries over any corpus size, zero false negatives (sparse binary superposition filters -- Bloom-as-VSA, Kleyko 2020) and zero false positives (sha256 verify, the depth test), under OR-baked tile probes with independent resolution (irradiance-map cull; probe saturation is a measured negative). Multi-channel (token/trigram/fields), instanced term codes, no BM25. Returns the EXACT ground-truth doc set. KEPT NEG: containment not relevance; ubiquitous terms degenerate to the timed scan.", example="import lecore; m=lecore.UnifiedMind(); ix=m.perfect_recall_index(tile=64); ix.add({'token':['cat','sat']}); ix.add({'token':['dog']}); print(ix.query(['cat']))", native=True, module="perfectrecall", aliases=("perfect recall search", "find every document containing these words", "exact match set no misses", "guaranteed no false negatives index", "bloom filter membership over documents", "unlimited corpus exact retrieval", "containment query which docs have all terms"), semantic="analyze/route")
    c.register_capability("retrieval_dispatch", "ADAPTIVE retrieval cascade (holographic_retrievaldispatch): exact-phrase short-circuit -> dense arm gated on top-1/top-2 margin (stop when proven, like adaptive path tracing) -> BM25 as a LAST-PASS denoise fit over the dense shortlist ONLY (O(shortlist), never the corpus), fused dense-dominant by RRF -> honest abstain. The lexical pass runs only on a narrow margin, only over the ambiguous window. Returns {ranked, stage, margin}. KEPT NEG: refine cannot rescue gold outside the shortlist; a confidently-wrong dense top-1 passes the gate un-refined.", example="import lecore; m=lecore.UnifiedMind(); print(m.retrieval_dispatch('fluid solver', ['smooth a mesh','fluid solver','render'])['stage'])", native=True, module="retrievaldispatch", aliases=("search my documents for the best match", "rank documents for a query", "adaptive search cascade", "pick the right retrieval method", "only run bm25 when needed", "search that stops when the answer is proven", "hybrid retrieval without scoring everything", "route between dense and lexical search", "last pass lexical refinement", "denoise a search shortlist"), semantic="analyze/route")
    c.register_capability("fuse_rankings", "RECIPROCAL RANK FUSION (holographic_bm25.reciprocal_rank_fusion): fuse several ranked id-lists into one by summing 1/(k+rank). Uses only RANKS, so no score calibration -- the right way to combine dense cosine (in [-1,1]) with BM25 (unbounded), whose raw scores are not comparable. An item ranked well by MORE retrievers rises. m.fuse_rankings([dense_order, bm25_order]) -> fused [(id, score)]. The hybrid-retrieval fuser the IR literature uses for vocabulary-mismatch.", example="import lecore; m=lecore.UnifiedMind(); print(m.fuse_rankings([[0,1,2],[0,2,1]])[:1])", native=True, module="bm25", aliases=("combine ranked lists", "reciprocal rank fusion", "merge two rankings", "fuse dense and sparse retrieval", "hybrid search fusion", "blend search results by rank"), semantic="analyze/route")
    c.register_capability("route_structured", "route a request to a MODULE by holographic role-STRUCTURE "
                          "instead of a bag-of-words mean (holographic_holoroute): parse request and module "
                          "into a {action, object, quality} record, bind+bundle via encode_record, match the "
                          "BOUND records. Separates the case a flat mean buries -- 'make my picture less grainy' "
                          "ranks denoise 1.000 vs fsr 0.409 where cosine put denoise at rank 237. Structure, not "
                          "the average. Returns [(name, score)] or [] if the request does not parse",
                          example="import lecore; m=lecore.UnifiedMind(dim=1024,seed=0); "
                          "print(m.route_structured('make my picture less grainy', "
                          "{'denoise':'reduce noise in an image','fsr':'upscale image resolution'})[:1])",
                          native=True, module="holoroute",
                          aliases=("route by structure not keywords", "match a request by roles and fillers",
                                   "holographic role router", "route by action object quality",
                                   "structured routing by binding", "which module by request structure"),
                          semantic="analyze/route")
    c.register_capability("match_record", "DOMAIN-GENERAL structured matching (holographic_relations): rank "
                          "candidates by how well their {role: filler} RECORD matches a query record, via "
                          "bound-record similarity (bind+bundle+cosine). The general form of route_structured "
                          "-- the SAME primitive classifies a physics regime {conserved,topology,motion}, a "
                          "market event {instrument,direction,magnitude}, an astronomy source {band,feature,"
                          "object}, or a mesh repair {defect,location,severity}. Exact match 1.0, partials "
                          "separate, empty query abstains. Returns [(name,score)]",
                          example="import lecore; m=lecore.UnifiedMind(dim=1024,seed=0); "
                          "print(m.match_record({'band':'radio','feature':'periodic'}, "
                          "{'pulsar':{'band':'radio','feature':'periodic'},'quasar':{'band':'radio','feature':'broadband'}})[:1])",
                          native=True, module="relations",
                          aliases=("match by structured record", "classify by role filler record",
                                   "nearest record by binding", "structure-aware nearest match",
                                   "rank candidates by their attributes", "which class does this record fit",
                                   "match physics regime market event astronomy source by structure"),
                          semantic="analyze/match")
    c.register_capability("match_prototype", "UNSTRUCTURED classification (holographic_relations, twin of "
                          "match_record): when an item has NO role schema -- a bag/blend, not a record -- match "
                          "it to the nearest class PROTOTYPE by cosine. The general form of the VSA intent "
                          "router: classify a question, gesture, regime, or style by the blend of its features. "
                          "build_prototypes({class:[examples]}) makes the prototypes; returns ranked [(class,"
                          "score)]. Pick vs match_record: has named roles -> match_record; role-free bag -> this",
                          example="import lecore; m=lecore.UnifiedMind(dim=1024,seed=0); "
                          "P=m.build_prototypes({'greet':['hello there','hi how are you'],'bye':['goodbye','see you']}); "
                          "print(m.match_prototype('hey hello',P)[:1])",
                          native=True, module="relations",
                          aliases=("classify without a schema", "nearest prototype match", "match a blend to a class",
                                   "intent style regime by example", "classify a bag of features",
                                   "which class does this blend fit"),
                          semantic="analyze/match")
    c.register_capability("decide_or_abstain", "the shared DECISION step for any classify/match "
                          "(holographic_relations): given ranked [(name,score)] from match_record / "
                          "match_prototype / any scorer, return (winner, score, confident) where confident "
                          "requires top-1 to beat top-2 by >= margin. One honest abstention rule instead of "
                          "each caller inventing its own -- abstains on a tie (flu~covid) rather than forcing a "
                          "pick. Cheap gap gate; for calibrated significance use a shuffle null",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.decide_or_abstain([('a',0.9),('b',0.4)], margin=0.1))",
                          native=True, module="relations",
                          aliases=("pick the winner or abstain", "confidence gate on a ranking",
                                   "abstain when the top isn't clearly ahead", "margin between top two",
                                   "trust the best only if separated", "decide or say unsure"),
                          semantic="analyze/decide")
    c.register_capability("resolve_capability_uri", "resolve a bare capability NAME or partial path to the FULL "
                          "capability URI(s) (holographic_capuri) -- 'rotation' -> both meshskin and scenegraph "
                          "paths; 'sdf/sphere' narrows to one. The disambiguation step when a name collides: supply "
                          "more of the path. Pairs with browse_capabilities (the menu) and capability_collisions",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "print(m.resolve_capability_uri('rotation'))",
                          native=True, aliases=("resolve a capability name", "disambiguate a function name",
                                                "full path of a capability", "which module has this function",
                                                "capability uri for a name"),
                          semantic="analyze/pipeline")
    c.register_capability("timeline", "a keyframe TIMELINE (holographic_anim) -- key(channel, t, value, interp) "
                          "then sample(channel, t) for the interpolated value at time t (vectorised over t). EASING "
                          "per key: 'linear' (default), 'step' (hold), 'smooth' (ease in-out), 'ease_in', 'ease_out'. "
                          "Key blendshape weights, deform params, or transforms and drive an animation from it",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "tl=m.timeline(); tl.key('x',0,0.0); tl.key('x',1,1.0,interp='ease_in'); "
                          "print(round(float(tl.sample('x',0.5)),2))",
                          native=True, aliases=("keyframe animation", "animation timeline", "ease in ease out",
                                                "animation curve easing", "keyframe a value over time",
                                                "interpolate keyframes", "keyframe with easing"),
                          semantic="animate/keyframe",
                          consumes=("scalar",), produces=("scalar",))
    c.register_capability("select_symmetric", "SYMMETRY SELECTION (holographic_meshselect) -- add a selection's "
                          "mirror-image elements across a world axis plane (axis 0/1/2 = x/y/z=0), so a symmetric "
                          "edit hits both sides. The selection-level complement to mirror_mesh (which mirrors "
                          "GEOMETRY): here nothing is created, we find the counterpart elements that already exist, "
                          "paired by reflected position",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "g={'vertices':[[-1,0,0],[1,0,0]],'faces':[]}; "
                          "print(len(m.select_symmetric(g,m.mesh_selection(g,'vertex').add([0]),axis=0)))",
                          native=True, aliases=("symmetric selection", "mirror a selection across an axis",
                                                "select the other side too", "select symmetric vertices",
                                                "symmetry select"),
                          semantic="select/symmetry",
                          consumes=("mesh", "selection"), produces=("selection",))
    c.register_capability("select_in_box", "REGION SELECT (holographic_meshselect) -- select every element inside "
                          "an axis-aligned box [lo,hi], the box/rubber-band select of a viewport. Edge/face modes "
                          "select if ANY vertex is in (inclusive). Pass a projection matrix or pt->(u,v) callable to "
                          "test in SCREEN coords instead -- that is frustum/rectangle select from the camera. "
                          "Returns a MeshSelection",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "g={'vertices':[[0,0,0],[5,5,0],[0.5,0.5,0]],'faces':[]}; "
                          "print(len(m.select_in_box(g,[-1,-1,-1],[1,1,1])))",
                          native=True, aliases=("box select", "region select vertices", "rubber band select",
                                                "frustum selection", "rectangle select", "select points in a box"),
                          semantic="select/region",
                          consumes=("mesh",), produces=("selection",))
    c.register_capability("soft_selection_weights", "SOFT SELECTION as a reusable per-vertex WEIGHT FIELD "
                          "(holographic_meshselect) -- 1 on the selection, falling off to 0 at a radius along the "
                          "surface (multi-source geodesic). Proportional editing: a transform moves each vertex by "
                          "weight*delta, dragging neighbours smoothly. Takes a MeshSelection or a vertex-index "
                          "list; falloff linear/smooth/sharp",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "g={'vertices':[[i,j,0] for j in range(3) for i in range(3)],"
                          "'faces':[[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]]}; "
                          "print(round(float(m.soft_selection_weights(g,[4],2.0)[4]),2))",
                          native=True, aliases=("soft selection falloff", "proportional editing weights",
                                                "falloff weights for a transform", "soft select weights",
                                                "smooth falloff selection"),
                          semantic="select/soft",
                          consumes=("mesh", "selection"), produces=("scalar",))
    c.register_capability("select_edge_loop", "select the EDGE LOOP through a seed edge (holographic_meshselect) -- "
                          "the ring of edges continuing straight across quads, the Alt-click loop-select users "
                          "expect from Blender/Maya. Walks both ways, stops at a pole or boundary (loops are only "
                          "well-defined on quads). Returns an edge-mode selection",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "g={'vertices':[[i,j,0] for j in range(3) for i in range(3)],"
                          "'faces':[[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]]}; "
                          "print(len(m.select_edge_loop(g,0)))",
                          native=True, aliases=("edge loop select", "loop select edges", "alt click edge loop",
                                                "select a ring of edges", "select an edge loop"),
                          semantic="select/loop",
                          consumes=("mesh", "selection"), produces=("selection",))
    c.register_capability("select_face_ring", "select the FACE RING from a seed face (holographic_meshselect) -- "
                          "the band of quads a loop cut runs through, walking quad to quad across shared edges. "
                          "Terminates at a non-quad or boundary. Returns a face-mode selection",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "g={'vertices':[[i,j,0] for j in range(3) for i in range(3)],"
                          "'faces':[[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]]}; "
                          "print(len(m.select_face_ring(g,0)))",
                          native=True, aliases=("face ring select", "select a ring of faces", "quad band select",
                                                "ring select faces", "select a face loop"),
                          semantic="select/loop",
                          consumes=("mesh", "selection"), produces=("selection",))
    c.register_capability("select_boundary_loops", "select the OPEN BOUNDARY edges of a mesh "
                          "(holographic_meshselect) -- the edges used by exactly one face (a hole rim or "
                          "open-surface border), the 'select the hole' step before filling or bridging. Returns an "
                          "edge-mode selection",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "g={'vertices':[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],'faces':[[0,1,2,3]]}; "
                          "print(len(m.select_boundary_loops(g)))",
                          native=True, aliases=("select boundary loop", "select the hole rim", "select open edges",
                                                "find mesh boundary", "select the border of a mesh"),
                          semantic="select/loop",
                          consumes=("mesh",), produces=("selection",))
    c.register_capability("mesh_selection", "a sub-object MESH SELECTION (holographic_meshselect) -- a persistent "
                          "set of VERTS/EDGES/FACES with a mode and set algebra (add/remove/toggle/union/intersect/"
                          "invert/select_all) plus mode CONVERSION (face->the verts it touches, verts->the faces "
                          "around them). The edit-mode selection a modeling app operates every edit on, "
                          "complementary to the object-level selection. Bind to a mesh so indices are validated",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "mesh={'vertices':[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],'faces':[[0,1,2,3]]}; "
                          "print(m.mesh_selection(mesh,'face').add([0]).to_mode('vertex').to_list())",
                          native=True, aliases=("select mesh vertices", "vertex edge face selection",
                                                "sub-object selection", "select geometry elements",
                                                "edit mode selection", "convert selection between modes",
                                                "selection set algebra"),
                          semantic="select/element",
                          consumes=("mesh",), produces=("selection",))
    c.register_capability("pick_element", "VIEWPORT PICKING for a 3D-modeling app (holographic_framebudget) -- "
                          "given a wireframe cage and a screen coordinate (-1..1 under the cursor), return which "
                          "element the user is pointing at: the nearest 'vertex', 'edge', or 'face' with its index "
                          "and position. Projects the cage's own verts to the screen and finds the closest -- "
                          "exact, deterministic, no GPU pick buffer. The select step before editing a vert/edge/face "
                          "in a viewport",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.scene_and_pipeline.holographic_framebudget import demo_frame_payload; "
                          "wf=demo_frame_payload({'width':64,'height':64},kinds=('wireframe',))['wireframe']; "
                          "print(m.pick_element(wf,0.0,0.0,want='vertex')['index'] is not None)",
                          native=True, aliases=("pick a vertex under the cursor", "select a vert edge or face",
                                                "ray pick a face", "click to select geometry",
                                                "which element is under the cursor", "viewport pick",
                                                "select geometry by screen position"))
    c.register_capability("workspace_manager", "a WORKSPACE MANAGER (holographic_workspace) -- durable user data "
                          "coexisting with transient 3D/sim SCENES, each in its own namespace. SAVE/LOAD a scene: "
                          "new_workspace, switch_workspace, export_workspace(name) -> a blob, import_workspace(blob) "
                          "rebuilds it BYTE-IDENTICALLY, combine_workspaces, reset_to_default. Also named "
                          "CHECKPOINTS: checkpoint(name,label) drops a save-point, restore_checkpoint rolls back to "
                          "it byte-identically, list_checkpoints. The persistence + save-point layer for a scene",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); wm=m.workspace_manager(); "
                          "wm.new_workspace('scene1'); print(wm.export_workspace('scene1')['name'])",
                          native=True, aliases=("save a workspace", "load a scene", "save my work",
                                                "persist a scene", "restore a workspace", "export a scene",
                                                "workspace save and load", "manage scenes", "checkpoint a scene",
                                                "named save point", "restore a checkpoint", "branch a workspace"))
    c.register_capability("Typed-section container (app-neutral workspace file)", "an app-neutral CONTAINER file "
                          "(holographic_container): a zip of a manifest + numeric array payloads, its body a list of "
                          "TYPED SECTIONS {kind, id, meta, arrays}. A section whose kind a reader does not understand "
                          "ROUND-TRIPS UNTOUCHED, so an image editor, a 3D app, and a video editor share ONE forward-"
                          "compatible file, each registering its own kinds. save_container(sections, meta) -> bytes; "
                          "load_container(bytes) -> {meta, sections}. Numeric-only (no pickle); byte-identical save/"
                          "load/save. Not workspace_manager (a live-DB checkpoint) -- the file FORMAT for typed data",
                          example="import numpy as np, lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "b=m.save_container([{'kind':'demo','id':'A','meta':{'n':1},'arrays':{'x':np.arange(4)}}]); "
                          "print(m.load_container(b)['sections'][0]['kind'])",
                          native=True, aliases=("save a project file", "workspace file for my app", "app project file",
                                                "bundle typed data into one file", "share a document between apps",
                                                "forward-compatible file format", "save meshes and images in one file",
                                                "container of arrays", "persist unknown kinds and round-trip them",
                                                "typed sections file", "one file for multiple apps", "cross-app workspace file"))
    c.register_capability("Frame-source protocol (temporal media seam)", "the CONTRACT for temporal media "
                          "(holographic_framesource): a FrameSource is any object with get() -> (frame, seq) plus "
                          "seekable/pausable flags; seq changes IFF the frame changes (cheap invalidation). The "
                          "engine owns the contract, NOT decoding (cv2/ffmpeg stay host-side). mind.map_frames("
                          "source, fn, cache) pulls a host source's current frame and memoises fn(frame) by seq; "
                          "mind.frame_key signs it; mind.synthetic_frame_source is a decoder-free synthetic clip. The "
                          "seam for video colour transfer / temporal NCA / optical flow",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); s=m.synthetic_frame_source(frames=4); "
                          "print(m.map_frames(s, lambda f: float(f.mean()))[1])",
                          native=True, aliases=("frame source protocol", "process video frames with caching",
                                                "per frame processing memoized by sequence", "seekable pausable frame provider",
                                                "apply an effect to each video frame", "video frame contract",
                                                "pull frames from a host source", "temporal media seam", "sequence numbered frames",
                                                "map a function over video frames", "frame invalidation by sequence"))
    c.register_capability("frame_server", "server-side REAL-TIME FRAME SERVING (holographic_framebudget) for "
                          "front-end clients that PULL frames -- the request/response form of a frame stream (the "
                          "HTTP service's POST /frame delegates to this). Keeps one frame-budget controller PER "
                          "SESSION; next_frame(session, target_fps, last_frame_ms) returns the quality preset to "
                          "render/simulate with, holding each client's target fps closed-loop. Two clients can run "
                          "at different rates (a phone at 30, a desktop at 60)",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); fs=m.frame_server(); "
                          "print(fs.next_frame('web', target_fps=60)['preset']['name'])",
                          native=True, aliases=("serve frames to a client", "stream frames to a front end",
                                                "per-session frame serving", "pull frames at a target rate",
                                                "adaptive frame server", "real-time frame endpoint",
                                                "serve real-time simulation frames"))
    c.register_capability("Stream to OBS (browser-source capture profile)", "The settings a streamer pastes into OBS to capture the leOS canvas as a BROWSER SOURCE -- the in-constitution way to stream leOS (OBS renders the page and does the ENCODING; the engine serves the page + frames via /frame, /frame/stream). mind.obs_capture_profile(base_url, preset, fps, transparent): preset '720p'/'1080p'/'1440p'/'4k' (match your OBS canvas -> no scaling); transparent=True gives transparent-bg CSS + a URL hint. Returns url, width, height, fps, frame_budget_ms, custom_css and step-by-step obs_steps. NOT an RTMP/NDI/virtual-camera encoder (needs ffmpeg/OS video I/O, outside core).",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); p=m.obs_capture_profile(preset='1080p', fps=30); (p['width'], p['height'], p['fps'])",
                          native=True, aliases=("stream to obs", "add leos to obs", "obs browser source settings",
                                                "capture the canvas in obs", "how do I stream this", "put this on a stream",
                                                "obs capture profile", "streaming setup for obs", "browser source width and fps",
                                                "transparent background for streaming", "record or stream the canvas",
                                                "use this in my stream"))
    c.register_capability("Invite button (shareable join link for a session)", "The INVITE BUTTON in one call: mint an invite and return a ready-to-share LINK + bare code a friend uses to join this multi-user session. mind.create_invite_link(workspace, base_url, grants, kind) wraps invite/admit -- default grants let the guest READ the workspace scene. Returns {code, link, workspace, kind, grants}: link is base_url?join=<code> for a Copy button, code for a 'type the code' box. The join side is mind.join_from_link. Wraps the low-level invite/principal/grant primitives so a UI button is one call, not an access-control lesson. Delegates; deterministic token via secrets.",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); inv=m.create_invite_link(workspace='lab'); ('join=' in inv['link'], bool(inv['code']))",
                          native=True, aliases=("invite someone to my session", "generate an invite link", "share a join link",
                                                "invite a friend to collaborate", "create a room invite", "get a link to invite people",
                                                "invite button", "let someone join my canvas", "share my session", "collaborate with a friend"))
    c.register_capability("Join button (enter a session from a link or code)", "The JOIN BUTTON in one call: admit a guest from EITHER a pasted invite LINK (...?join=<code>) OR a bare code. mind.join_from_link(link_or_code, actor_id) extracts the code from a URL if needed, redeems it via admit, and returns the scoped guest Principal (read-only to exactly what the invite granted; the guest's writes stay in their own namespace). Raises AccessError on an unknown/used code. The counterpart to create_invite_link so a join box accepts whatever the user pastes. Delegates to admit.",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); inv=m.create_invite_link(workspace='lab'); g=m.join_from_link(inv['link'], 'alice'); g.id",
                          native=True, aliases=("join a session with a code", "join from an invite link", "enter a shared session",
                                                "join a room by code", "accept an invite", "join button", "join my friend's canvas",
                                                "redeem an invite code", "connect to a shared world", "join a coop session"))
    c.register_capability("frame_budget_controller", "the FRAME-BUDGET CONTROLLER (holographic_framebudget) -- one "
                          "knob from a target FPS to concrete render + simulation quality, held closed-loop against "
                          "MEASURED frame time. Each frame: current() gives the quality preset, report(frame_ms) "
                          "feeds back the time; it DROPS a level on a budget miss and CLIMBS only after a streak of "
                          "comfortable frames (hysteresis). The conductor tying render_adaptive / "
                          "draft_vs_refine_simulation / LOD to a real-time target. Render and sim quality are "
                          "SEPARATE knobs -- a coarse render is a draft, a coarse chaotic sim a DIFFERENT run",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "ctrl=m.frame_budget_controller(target_fps=60, start_level=4); "
                          "ctrl.report(40.0); print(ctrl.current()['name'])",
                          native=True, aliases=("hit a target fps", "pick quality to hit a frame rate",
                                                "adapt quality to frame time", "real-time quality control",
                                                "map fps to quality level", "degrade gracefully to keep frame rate",
                                                "60 fps quality controller", "control quality for real-time display"))
    c.register_capability("regime_gate", "build a REGIME GATE (holographic_regimegate) -- route to a "
                          "superior-but-NICHE method only when a cheap detector says you are in its regime, and to "
                          "a safe fallback everywhere else. The honest way to RE-ENABLE a shelved 'only good in a "
                          "niche' method (a kept negative): the fallback stays the safe default, so a gate misfire "
                          "costs at most the default, never worse than the shelved method. Returns a gate; .apply(x) "
                          "gives (result, info) recording the score/threshold/path. The adaptive-dispatch pattern "
                          "as a reusable object",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "g=m.regime_gate('sharp', lambda x: abs(x), 5.0, lambda x: ('hi',x*2), lambda x: ('lo',x)); "
                          "print(g.apply(9.0)[1]['used'])",
                          native=True, aliases=("re-enable a niche method", "route by regime with a fallback",
                                                "gate a method behind a detector", "use a method only in its regime",
                                                "conditional dispatch with safe default", "regime gate",
                                                "shelved method behind a detector"))
    c.register_capability("Variance harness (honest measurement)", "the VARIANCE HARNESS (holographic_measure) -- "
                          "every headline number gets a mean, a spread, and a 95% bootstrap CI, not a lucky-seed "
                          "point estimate. measure(run_once, seeds) runs a scored experiment across seeds; "
                          "assert_robust passes only if the LOWER CI bound clears the floor (not just the mean); "
                          "is_fragile flags a claim whose spread could sink it on a couple of unlucky seeds; "
                          "measure_report formats it. The constitution's no-win-without-a-baseline discipline, "
                          "made invocable",
                          example="import lecore, numpy as np; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "s=m.measure(lambda seed: float(np.random.default_rng(seed).normal(0.7,0.1)), seeds=range(20)); "
                          "print(m.measure_report('score', s, floor=0.5))",
                          native=True, aliases=("measure across seeds", "mean spread and confidence interval",
                                                "is this result robust", "is this claim fragile",
                                                "bootstrap confidence interval", "variance harness",
                                                "honest measurement", "does the lower ci bound clear the floor"))
    c.register_capability("sweep_directions", "the UP/DOWN/SIDEWAYS completeness sweep (holographic_ladder) -- does "
                          "a corpus's structure hold in all three directions, or only one? DOWN: survives "
                          "DECOMPOSITION (are the parts analyzable)? UP: survives EMBEDDING in a larger corpus? "
                          "SIDEWAYS: which lens COSTUMES (sequence/structure) does it wear? Returns per-direction "
                          "ok + gaps + complete. Null-aware: irreducible data flags all three, never fabricating "
                          "structure. A capability that works in only one direction is an INCOMPLETE faculty",
                          example="import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "from holographic.agents_and_reasoning.holographic_ladder import _make_planted_corpus; "
                          "print(m.sweep_directions(_make_planted_corpus())['complete'])",
                          native=True, aliases=("up down sideways sweep", "check a capability in all directions",
                                                "does this work on components and wholes", "does structure survive embedding",
                                                "which lenses does this data wear", "completeness check",
                                                "is this faculty complete", "sweep the abstraction directions"))
    c.register_capability("iaaft_surrogate", "IAAFT surrogate -- the gold-standard null matching BOTH the exact "
                          "amplitude distribution AND (to convergence) the exact power spectrum (Schreiber & "
                          "Schmitz 1996). AAFT only approximates the spectrum; IAAFT iterates two projections "
                          "(impose target magnitudes / impose the amplitude distribution) until they agree -- the "
                          "iterate-a-projection move. Prefer over AAFT for strongly-coloured non-Gaussian signals "
                          "(fat-tailed autocorrelated data like price returns), at the cost of iterations",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "x=np.cumsum(np.random.default_rng(0).standard_normal(512)**3); "
                          "print(bool(np.allclose(np.sort(m.iaaft_surrogate(x)), np.sort(x))))",
                          native=True, aliases=("iaaft surrogate", "iterated surrogate",
                                                "exact spectrum and distribution null", "gold standard surrogate",
                                                "converged amplitude adjusted surrogate", "best surrogate for colored fat tails"))
    c.register_capability("amplitude_adjusted_surrogate", "AAFT surrogate -- the stricter null for NON-GAUSSIAN "
                          "signals (holographic_surrogate). Basic phase-randomization preserves the spectrum but "
                          "GAUSSIANIZES the marginal, destroying the fat tails of e.g. price returns; AAFT preserves "
                          "BOTH the exact amplitude distribution and (approximately) the spectrum. Use it when the "
                          "amplitude distribution matters (fat-tailed data); use phase_randomize when the signal is "
                          "~Gaussian and the spectrum must match exactly",
                          example="import numpy as np; import lecore; m=lecore.UnifiedMind(dim=256,seed=0); "
                          "x=np.random.default_rng(0).standard_normal(512)**3; "
                          "print(bool(np.allclose(np.sort(m.amplitude_adjusted_surrogate(x)), np.sort(x))))",
                          native=True, aliases=("aaft surrogate", "surrogate for fat tailed data",
                                                "null preserving the amplitude distribution", "amplitude adjusted null",
                                                "surrogate keeping the histogram", "non-gaussian surrogate",
                                                "fat tail preserving null"))


_PART = "holographic_catalog_p02"


def _selftest():
    """Delegates to holographic_catalog.check_catalog_part -- one home for the shared contract."""
    from holographic.caching_and_storage.holographic_catalog import check_catalog_part
    n = check_catalog_part(_PART, register_p02)
    print("%s selftest OK -- %d capabilities, no internal duplicates" % (_PART, n))


if __name__ == "__main__":
    _selftest()
