"""Part 06 of UnifiedMind's faculty surface -- 119 methods, mesh_collapse_edge .. route_representation.

NOT A STANDALONE MODULE. This is one slice of the single `UnifiedMind` class, which grew to 17.4k lines
in one file and went past the 1 MB cap an agent can read in a single pass -- so the engine could no
longer read its own central nervous system. The class is assembled from these parts by
holographic/misc/holographic_unified.py, which is still the only import path anyone uses.

Every method here is a real attribute of UnifiedMind at runtime (mixin, not delegation), so `mind.x()`,
`dir(mind)`, the doc generators and the service's tool introspection all behave exactly as before. The
bodies were moved by line range, not regenerated, so they are byte-identical to the originals.

KEPT NEGATIVE, so nobody "tidies" it: these part classes are NOT a public API and must never be
imported or subclassed directly. They carry no `__init__` and assume the state UnifiedMind.__init__
builds; instantiated alone they would fail on the first attribute access. The leading underscore on
the class name says so, and the reachability audit reads them as referenced-by-unified, not as
standalone capabilities.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder, _Index
from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
from holographic.misc.holographic_creature import HolographicMind
from holographic.unified import check_part


class _UnifiedPart06:

    def mesh_collapse_edge(self, mesh, keep, remove):
        """Merge edge endpoint `remove` into `keep` (holographic_eulerops, FWD-7): the decimation/LOD
        primitive and the inverse of split_edge. V-1, chi unchanged. Returns a new Mesh, or None if the
        collapse would break the manifold (the LINK CONDITION) -- a true precondition the caller must handle."""
        from holographic.misc.holographic_eulerops import collapse_edge
        return collapse_edge(mesh, keep, remove)

    def mesh_qem_decimate(self, mesh, target_faces, fast=False, uvs=None, silhouette=0.95, topology=True):
        """QEM decimation (Garland-Heckbert) to an explicit `target_faces` -- the QUALITY decimator (cluster is
        the fast one). `silhouette=0.95` (default): the result is swept against the input and the face budget
        is raised x1.5 until the outline survives (verdict in `.silhouette_report`); silhouette=None opts out.
        See holographic_meshqem.qem_decimate."""
        from holographic.mesh_and_geometry.holographic_meshqem import qem_decimate, silhouette_guarded
        src = self._as_mesh(mesh)
        floor = None if silhouette in (None, False) else float(silhouette)
        out, rep = silhouette_guarded(src, lambda n: qem_decimate(src, target_faces=n, fast=fast, uvs=uvs),
                                      int(target_faces), min_iou=floor)
        out.silhouette_report = rep
        out.topology_report = self._topology_check("mesh_qem_decimate", src, out, topology)
        return out

    def mesh_surface_deviation(self, mesh_a, mesh_b):
        """A decimation QUALITY metric (holographic_meshqem): (mean, max) point-to-surface distance from mesh_a's
        vertices to mesh_b's triangles -- how far one mesh's surface sits from the other's points. Use to measure
        how much a decimation (e.g. mesh_qem_decimate) moved the surface. Returns (mean, max)."""
        from holographic.mesh_and_geometry.holographic_meshqem import surface_deviation
        return surface_deviation(mesh_a, mesh_b)

    def mesh_lod_chain(self, mesh, targets=(0.5, 0.25, 0.125), silhouette=0.95):
        """Build a level-of-detail CHAIN (holographic_lod): QEM-decimate `mesh` to successively coarser levels at
        the given face-count fractions, measuring each level's surface deviation from the original. Returns a
        fine->coarse list of LODLevel(mesh, n_faces, mean_error, max_error); the first is the original (zero error).
        Pair with mesh_select_lod to choose a level by viewing distance."""
        from holographic.misc.holographic_lod import build_lod_chain
        from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guard_chain
        src = self._as_mesh(mesh)
        chain = build_lod_chain(src, targets=targets)
        floor = None if silhouette in (None, False) else float(silhouette)
        kept, rep = silhouette_guard_chain(src, chain, get_mesh=lambda lv: lv.mesh, min_iou=floor)
        # LODLevel has __slots__, so the verdict cannot ride on the levels; it rides on the mind instead
        # (self._last_lod_chain_silhouette), same pattern per chain faculty.
        self._last_lod_chain_silhouette = rep
        return kept

    def mesh_select_lod(self, chain, distance, pixel_threshold, screen_height_px=1080, fov_deg=60.0):
        """Choose a level of detail by SCREEN-SPACE ERROR (holographic_lod): the index of the coarsest level in
        `chain` whose error, projected to the screen at `distance`, stays under `pixel_threshold` -- the cheapest
        mesh that looks right. The engine's error-budget resolution selection (coarse_to_fine) carried to meshes:
        full detail up close, coarser far away. Returns an int index into the chain. Kept negative: the error is
        geometric surface deviation, not a perceptual/silhouette metric."""
        import math
        from holographic.misc.holographic_lod import select_lod
        return select_lod(chain, distance, pixel_threshold, screen_height_px=screen_height_px,
                          fov_rad=math.radians(fov_deg))

    def mesh_egi_compare(self, ref_mesh, mesh, nth=24, nph=48):
        """Orientation-field preservation (Extended Gaussian Image, Horn 1984): area-weighted normal
        distributions compared on the direction sphere, in [0,1]. The COMPLEMENT of silhouette_sweep -- sweep
        guards the OUTLINE and is blind to surface character; this reads how much the normal field coarsened
        and is blind to nothing about orientation but says nothing about the profile (measured: a decimated
        sphere keeps silhouette 0.99 while EGI reads 0.06). Not on the guard's 0.95 scale. See
        holographic_render.egi_similarity."""
        from holographic.rendering.holographic_render import egi_similarity
        return egi_similarity(self._as_mesh(ref_mesh), self._as_mesh(mesh), nth=nth, nph=nph)

    def fit_camera(self, mesh, direction=(1.0, 0.75, 1.1), up=(0.0, 1.0, 0.0), fov_deg=50.0,
                   width=512, height=512, margin=1.06):
        """FRAME a mesh: returns the camera dict {eye, target, up, fov_deg} that fits every vertex inside a
        `width` x `height` frame along `direction`, centred. Pass it straight to m.render_mesh. Solves the
        distance exactly (no iteration) and centres on the PROJECTED bbox -- a scan's centroid is not the
        centre of its outline, which is what leaves subjects clipped on one edge with slack on the other.
        Measured need: preview_asset's framing left a crab at 4% of frame; a hand-picked distance clipped a
        ladybird against all four edges. See holographic_render.fit_camera."""
        from holographic.rendering.holographic_render import fit_camera
        return fit_camera(self._as_mesh(mesh), direction=direction, up=up, fov_deg=fov_deg,
                          aspect=float(width) / float(height), margin=margin)

    def silhouette_sweep(self, ref_mesh, mesh, n_azimuth=6, size=128, include_top=True, ref_cache=None):
        """Orthographic TURNTABLE silhouette comparison: rotate the pair through `n_azimuth` directions across
        [0, pi) (theta and theta+pi are the same outline under orthographic projection) plus the top, and score
        IoU per direction under the REFERENCE's frame. The fast instrument behind the default-on modification
        guards (~2 s warm on a 322k source). Returns {iou, worst, worst_view, mean, seconds}. See
        holographic_render.silhouette_sweep."""
        from holographic.rendering.holographic_render import silhouette_sweep
        return silhouette_sweep(self._as_mesh(ref_mesh), self._as_mesh(mesh), n_azimuth=n_azimuth, size=size,
                                include_top=include_top, ref_cache=ref_cache)

    def mesh_decimate_to(self, mesh, target_faces=None, target_fraction=None, keep_uv="auto",
                         min_silhouette_iou=0.95, views_size=128, topology=True):
        """Decimate to an EXPLICIT face budget (`target_faces` or `target_fraction`), optionally guarded by a
        silhouette guard ON BY DEFAULT (min_silhouette_iou=0.95, orthographic turntable sweep): the result's
        outline is scored against the SOURCE across n azimuths + top and the decimation walks BACK if the WORST
        direction falls below the floor -- shipping more faces than asked, loudly
        (report['budget_missed_for_silhouette']), never a silently broken outline. min_silhouette_iou=None
        opts out: destructive modification is a choice, not the default. No target at all returns the mesh
        untouched: "never modify" is a valid policy. Returns (mesh, report). See
        holographic_meshqem.decimate_to."""
        from holographic.mesh_and_geometry.holographic_meshqem import decimate_to
        src = self._as_mesh(mesh)
        out, rep = decimate_to(src, target_faces=target_faces, target_fraction=target_fraction,
                               keep_uv=keep_uv, min_silhouette_iou=min_silhouette_iou, views_size=views_size)
        d = self._topology_check("mesh_decimate_to", src, out, topology)
        if d is not None:
            rep["topology"] = d
        return out, rep

    def mesh_cluster_decimate(self, mesh, grid=16, keep_uv="auto", silhouette=0.95, topology=True):
        """PARALLEL decimation by vertex clustering (holographic_meshqem, Rossignac-Borrel / Lindstrom) -- the O(n)
        counterpart of the greedy mesh_qem_decimate, for an IMPORTED mesh with no field behind it. Bins vertices into
        a grid^3 spatial lattice (the engine's floor-divide tiling), collapses each cell to ONE representative, remaps
        faces, drops degenerate ones. Every step is a vectorized array op -- no greedy edge-collapse search -- so it
        runs hundreds-to-thousands x faster than QEM (a 22k-face mesh in tens of ms vs minutes). The representative is
        VSA-native: a cell's quadric is the SUM (a bundle, superposition) of its faces' plane tensors, and the
        representative is that bundle's minimizer, clamped to the cell. Returns a new Mesh. KEPT NEGATIVE: clustering
        trades quality and manifoldness for parallel speed (a coarse grid can go non-manifold) -- mesh_qem_decimate
        stays the quality option, this is the fast one. Higher `grid` = finer = more faces kept.

        `keep_uv="auto"` REPROJECTS the source's uvs onto the result when the atlas can survive it (per-corner,
        re-splitting the seams -- the decimator welds them, and a welded seam vertex cannot carry the two uvs a
        seam needs), and otherwise leaves them off with `.uv_transfer_report` naming mesh_rebake_texture. True
        forces the old per-vertex transfer, False skips uvs entirely.

        `silhouette=0.95` (default, per owner directive): the result is swept against the input (orthographic
        turntable, worst direction) and the grid is REFINED x1.5 until the outline survives -- preservation is
        the default, destruction the opt-out (silhouette=None). The verdict rides on the result as
        `.silhouette_report`. The module-level cluster_decimate function is untouched and unguarded."""
        from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate, silhouette_guarded
        src = self._as_mesh(mesh)
        floor = None if silhouette in (None, False) else float(silhouette)
        out, rep = silhouette_guarded(src, lambda g: cluster_decimate(src, g, keep_uv=keep_uv), int(grid),
                                      min_iou=floor)
        out.silhouette_report = rep
        # M13: topology delta rides the result like silhouette_report does -- REPORTED by default (this
        # decimator measures clean on the fixtures, so for it the report is a regression trap, not a bug
        # hunt), refused only on request. An instrument must never flip yesterday's decisions.
        out.topology_report = self._topology_check("mesh_cluster_decimate", src, out, topology)
        return out

    def mesh_cluster_lod_chain(self, mesh, grids=(48, 24, 12), silhouette=0.95):
        """A fast PARALLEL level-of-detail chain (holographic_lod) for an IMPORTED mesh: vertex-cluster
        (mesh_cluster_decimate) at decreasing grid resolutions, measuring each level's deviation from the original.
        The O(n)-per-level counterpart of mesh_lod_chain (greedy QEM), for large meshes where the QEM chain is too
        slow. Returns a fine->coarse list of LODLevel(mesh, n_faces, mean_error, max_error); pair with
        mesh_select_lod. For a field-backed surface, prefer surface_mesh's FIELD-NATIVE LOD (re-march the source
        coarser) -- this is the path for a mesh that arrives with no field. Kept negative: inherits cluster_decimate's
        quality/manifoldness trade."""
        from holographic.misc.holographic_lod import build_cluster_lod_chain
        from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guard_chain
        src = self._as_mesh(mesh)
        chain = build_cluster_lod_chain(src, grids=grids)
        floor = None if silhouette in (None, False) else float(silhouette)
        kept, rep = silhouette_guard_chain(src, chain, get_mesh=lambda lv: lv.mesh, min_iou=floor)
        self._last_lod_chain_silhouette = rep
        return kept

    def mesh_to_field(self, mesh, bounds, res=48, band=None, method="shell"):
        """The mesh -> FIELD direction (holographic_meshbridge.mesh_distance_grid): decompose a mesh into a SIGNED
        banded distance field (a banded SDF) by TILING -- each triangle updates only the local block of grid voxels
        within `band` of it (a vectorized sub-array scatter-min by magnitude, the apply_local pattern), so the cost is
        O(F * block) not O(F * res^3). Signed (negative inside, by nearest face normal) so that |sample| near the
        surface is accurate to WELL UNDER a voxel -- an unsigned field's kink cannot resolve sub-voxel distances.
        Returns (grid res^3, (xs,ys,zs)). This is the gateway that lets an imported mesh be queried like a field:
        build once, then any number of points are O(V) samples (mesh_sample_field) -- e.g. a fast point-to-surface
        distance for decimation/LOD error.

        KEPT HONEST: the build is currently an F-triangle Python loop (~1ms/triangle) -- a batched closest-point would
        vectorize it (backlog); the nearest-normal sign can mis-sign deep concavities / non-watertight meshes
        (magnitude is always right); far interior voxels default to +band (no flood-fill sign yet), so this is a
        sample-near-the-surface field, not yet a re-marchable full SDF."""
        from holographic.mesh_and_geometry.holographic_meshbridge import mesh_distance_grid
        # method="scatter" is the ORIGINAL per-triangle path and the only one whose
        # cost is O(triangle count); "shell" is O(surface area) and approximate.
        # Dropping the flag pinned every caller to the fast approximation.
        return mesh_distance_grid(mesh, bounds, res=res, band=band, method=method)

    def mesh_sample_field(self, grid, axes, points):
        """Trilinearly sample a banded SDF (from mesh_to_field) at query points (N,3) -> (N,) SIGNED distances; take
        abs for unsigned surface distance (holographic_meshbridge.sample_distance_grid). The O(V) read that turns a
        once-built field into a point-to-surface distance for any points -- the cheap query the brute O(Va*Fb) scan
        could not give once the field exists."""
        from holographic.mesh_and_geometry.holographic_meshbridge import sample_distance_grid
        return sample_distance_grid(grid, axes, points)

    def mesh_reproject_uv(self, source_mesh, source_uv, target_mesh, uv_tol=1e-6, tie=0.5, disc_factor=5.0):
        """REPROJECT a uv map onto a mesh whose face count changed -- after decimation, remeshing, retopo, any
        topology edit. Per-CORNER and seam-aware: a retopo welds the two sides of a seam into one vertex, and one
        vertex cannot carry the two uvs a seam needs, so plain per-vertex transfer makes the faces around it span
        the whole atlas. Returns (mesh, uv, report); the mesh may gain vertices -- the seams have to go somewhere.
        Raises on a fragmented (per-triangle scan) atlas, where the answer is mesh_rebake_texture instead.
        See holographic_meshtools.reproject_uv."""
        from holographic.mesh_and_geometry.holographic_meshtools import reproject_uv
        return reproject_uv(self._as_mesh(source_mesh), source_uv, self._as_mesh(target_mesh),
                            uv_tol=uv_tol, tie=tie, disc_factor=disc_factor)

    def uv_atlas_report(self, mesh, uvs=None):
        """DIAGNOSE whether a mesh's UVs can survive retopo/LOD/remesh BEFORE you spend the time: island count,
        faces per island, and `transferable` -- False for a fragmented (per-triangle photogrammetry) atlas that
        no per-vertex uv transfer can preserve. The check that turns silent texture confetti into a readable
        field. See holographic_meshtools.uv_atlas_report."""
        from holographic.mesh_and_geometry.holographic_meshtools import uv_atlas_report
        return uv_atlas_report(self._as_mesh(mesh), uvs=uvs)

    def mesh_textured_lod(self, mesh, texture, uvs=None, grid=48, size=1024, margin=2, silhouette=0.95, method='auto'):
        """ONE CALL for a decimated mesh that STILL WEARS ITS TEXTURE, routed BY MEASUREMENT: a coherent atlas
        gets a cheap uv transfer (image reused); a fragmented scan atlas gets a full RE-BAKE into a new per-face
        atlas (the only correct route -- transfer would render as speckle). Returns (lod_mesh, uv, image,
        report); `report['route']` says which way it went and why. See holographic_meshtools.textured_lod."""
        from holographic.mesh_and_geometry.holographic_meshtools import textured_lod
        from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guarded
        src = self._as_mesh(mesh)
        floor = None if silhouette in (None, False) else float(silhouette)
        if floor is not None:
            # guard the GEOMETRY first (walk the grid up until the outline survives), THEN route the texture
            # onto the surviving grid -- texturing a destroyed mesh well is still a destroyed mesh.
            from holographic.mesh_and_geometry.holographic_meshqem import cluster_decimate
            _probe, rep = silhouette_guarded(src, lambda g: cluster_decimate(src, g, keep_uv=False),
                                             int(grid), min_iou=floor)
            grid = rep["knob"]
        out = textured_lod(src, texture, uvs=uvs, grid=grid, size=size, margin=margin, method=method)
        if floor is not None:
            out[3]["silhouette"] = rep
        return out

    def mesh_rebake_texture(self, source_mesh, source_uv, texture, target_mesh, size=1024, margin=2,
                            method="project", grid=None):
        """RE-BAKE a texture onto a NEW topology: build a per-face atlas for `target_mesh` and paint the source's
        colour into it. Topology-independent by construction -- the route when uv_atlas_report says
        transferable=False. method="project" (default) does texel -> closest point on source -> its uv -> sample
        (exact, slow). method="scatter" is the HOLOGRAPHIC fast path (~1500x on dense scans): scatter source
        colour into a volumetric grid, gather at each texel -- bounded by source vertex density, so opt-in for
        dense scans. method="recall" fills the atlas by SPATIAL MEMORY (position hypervectors, resonant top-k
        colour readout -- best quality per read, 0.034 RGB; sweet spot is vertex-scale queries, scatter owns
        texel-scale). Returns (mesh_with_split_verts, uv, image, report). See holographic_meshtools.rebake_texture."""
        from holographic.mesh_and_geometry.holographic_meshtools import rebake_texture
        return rebake_texture(self._as_mesh(source_mesh), source_uv, texture, self._as_mesh(target_mesh),
                              size=size, margin=margin, method=method, grid=grid)

    def pose_asset(self, lm, time=0.0, clip=0):
        """POSE a rigged asset at `time`: samples the animation clip, composes the node hierarchy, builds each
        joint's matrix from its inverse-bind, and linear-blend-skins the vertices. Returns (Mesh, report) --
        report['mode'] tells you whether it animated, fell back to the bind pose, or moved chunks rigidly. The
        composition that makes an imported .glb actually move; pass a LoadedMesh from load_glb / import_asset.
        See holographic_assetimport.pose_asset."""
        from holographic.io_and_interop.holographic_assetimport import pose_asset
        return pose_asset(lm, time=time, clip=clip)

    def asset_base_texture(self, loaded_mesh):
        """Render-ready (texture, uvs, base_color) for a LOADED or self-derived mesh -- the pointer from an
        imported/decimated/retopo'd mesh to a TEXTURED render_mesh call without a file path. Same
        coverage-based material pick preview_asset uses. Feed the pair to render_mesh(..., texture=, uvs=).
        See holographic_assetimport.asset_base_texture."""
        from holographic.io_and_interop.holographic_assetimport import asset_base_texture as _abt
        return _abt(loaded_mesh)

    def preview_asset(self, path, camera=None, width=512, height=384, ambient=0.5, smooth=True, fit=False, eye_dir=(0.55, 0.35, 0.7)):
        """ONE-CALL TEXTURED PREVIEW of an asset file (.obj/.glb/.gltf): import with materials + embedded
        textures, auto-frame, rasterize with the base-colour map applied. Returns (image (H,W,3), LoadedMesh).
        The pointer from import to textured render that was missing -- previously five manual composition steps.
        See holographic_assetimport.preview_asset."""
        from holographic.io_and_interop.holographic_assetimport import preview_asset
        return preview_asset(path, camera=camera, width=width, height=height, ambient=ambient, smooth=smooth, fit=fit, eye_dir=eye_dir)

    def mesh_to_sdf_grid(self, mesh, bounds, res=48, band=None, sign="auto"):
        """Convert an imported mesh into a FULL, re-marchable signed distance field
        (holographic_meshbridge.mesh_to_sdf_grid). Returns (grid res^3, (xs,ys,zs)) -- the gateway that lets an
        imported mesh JOIN the field-native world. `sign="auto"` routes edge-closed meshes to the original
        flood-fill path BIT-IDENTICALLY, and OPEN meshes / scan soups (any boundary edges) to the generalised
        winding number (Jacobson 2013, Barill 2018 fast clusters) -- the fix for the .glb-import regression where
        a 71%-boundary-edge Sketchfab scan flood-leaked into shredded garbage blobs. sign="flood"/"winding" force
        a path. KEPT HONEST: winding costs O(voxels x clusters); flood needs a watertight >= ~2-voxel band."""
        from holographic.mesh_and_geometry.holographic_meshbridge import mesh_to_sdf_grid
        # _as_mesh: accept {'vertices','faces'} JSON like voxel_remesh/render_mesh already do (C2) -- found by
        # the /invoke round-trip of THIS faculty failing on a plain-JSON mesh while its sibling accepted one
        return mesh_to_sdf_grid(self._as_mesh(mesh), bounds, res=res, band=band, sign=sign)

    def render_water(self, container=None, level=0.72, preset="ocean", size=1.0, extent=40.0, res=192,
                     t=0.0, seed=0, ripple=0.35, material="water", quality="fast", width=None, height=None,
                     **wave_overrides):
        """ONE-SHOT water render for JSON/agent clients: water_body(...) built AND rendered in a single call,
        returning the (H,W,3) image -- because a WaterBody OBJECT cannot cross the HTTP boundary (/invoke
        returns a repr stub for objects; found by the wiring sweep). Same parameters as water_body plus
        render's quality/width/height. In-process callers who want the object (animation via .at_time,
        custom cameras, the sdf scene) still use water_body. See holographic_ocean.water_body."""
        wb = self.water_body(container=container, level=level, preset=preset, size=size, extent=extent,
                             res=res, t=t, seed=seed, ripple=ripple, material=material, **wave_overrides)
        return wb.render(quality, width=width, height=height)

    def sculpt_prepare(self, mesh, resolution=48, silhouette=0.95, max_resolution=160, pad_frac=0.08,
                       n_azimuth=6, band=None):
        """Prepare a mesh for SCULPT MODE with the shape GUARDED: builds the SDF cache (grid + axes) and the
        sculptable remesh in one call, held to a worst-view silhouette-IoU floor so the conversion cannot
        silently change the shape. The ladder pulls TWO levers in cost order -- retry the sign method
        (flood-fill leaks through TOUCHING/COINCIDENT shells and gets WORSE with resolution, measured
        0.734@48 -> 0.250@96; the winding number is robust at 0.954+), then escalate resolution x1.5 for
        sub-cell thin features -- and REFUSES with the full report if the floor is unreachable
        (silhouette=None = explicit unguarded opt-out). Returns {mesh, grid, axes, bounds, report}; the grid
        and mesh are the same level of the same field, so brushes and raycasts agree with what the user sees.
        See holographic_meshbridge.sculpt_prepare; voxel_remesh is the mesh-only cousin."""
        from holographic.mesh_and_geometry.holographic_meshbridge import sculpt_prepare as _sp
        return _sp(self._as_mesh(mesh), resolution=resolution, silhouette=silhouette,
                   max_resolution=max_resolution, pad_frac=pad_frac, n_azimuth=n_azimuth, band=band)

    def voxel_remesh(self, mesh, resolution=64, pad=0.2, sign="auto", keep_uv="auto", silhouette=0.95, topology=True):
        """VOXEL REMESH (Blender Voxel Remesh): rebuild a mesh as a UNIFORM, watertight surface by sampling it into
        a signed-distance grid and re-marching -- the standard cleanup for messy, self-intersecting, non-manifold,
        or multi-shell input before retopo. Any tangle in becomes one clean closed surface out at `resolution` cells
        per axis. A compose of mesh_to_sdf_grid + the mesher. `keep_uv="auto"` carries the source's uvs across ONLY
        when uv_atlas_report says the atlas can survive a per-vertex transfer, and otherwise leaves them off and
        says why in `.uv_transfer_report` (use mesh_rebake_texture instead); keep_uv=True forces the old transfer,
        False skips it.

        `silhouette=0.95` (default): the remesh is swept against the input and `resolution` is refined x1.5 until
        the outline survives; verdict in `.silhouette_report`; silhouette=None opts out (destructive is a choice,
        not the default). See holographic_meshbridge.voxel_remesh."""
        from holographic.mesh_and_geometry.holographic_meshbridge import voxel_remesh as _vr
        from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guarded
        src = self._as_mesh(mesh)
        floor = None if silhouette in (None, False) else float(silhouette)
        out, rep = silhouette_guarded(src, lambda r: _vr(src, resolution=r, pad=pad, sign=sign, keep_uv=keep_uv),
                                      int(resolution), min_iou=floor)
        out.silhouette_report = rep
        # voxel_remesh REBUILDS the surface from an SDF, so topology CAN legitimately change (that is its
        # documented scope: block-outs, hole-closing). Reported, never refused by default -- refusing here
        # would gate the operator's own purpose; a caller who wants the strictness asks for it.
        out.topology_report = self._topology_check("voxel_remesh", src, out, topology)
        return out

    def metaball_mesh(self, centers, radius=0.5, level=0.5, resolution=48, pad=0.6):
        """METABALL MESH (soft-blob base mesh): sum-of-Gaussians field at `centers` (n,3), spread `radius`, marched
        at `level` -- overlapping blobs FUSE smoothly. The organic-blob base-mesh route complementing skin_skeleton
        (blobs where branch-stitching gets ugly). Returns a watertight Mesh. See holographic_meshbridge.metaball_mesh."""
        from holographic.mesh_and_geometry.holographic_meshbridge import metaball_mesh as _mb
        return _mb(centers, radius=radius, level=level, resolution=resolution, pad=pad)



    def mesh_to_field_vector(self, mesh, bounds, dim=2048, bandwidth=18.0, grid=12, seed=0):
        """FS-5: carry a surface as a SINGLE hypervector (edit = bind). Samples the mesh's signed distance on a
        coarse `grid`^3 lattice over `bounds` and bundles it into one FPE vector (HolographicField). On the result:
        `.value(points)` reads the (smoothed) signed field, `.translate(delta)` moves the WHOLE surface with one
        binding (exact -- value_shifted(x)=value_orig(x-delta)), `.union(other)` merges two surfaces by bundling, and
        `.surface(bounds, res)` re-extracts a mesh from the 0-level. This is the array-domain mesh_to_field's
        hypervector cousin -- moving and merging become VSA algebra. KEPT HONEST: a demonstration representation, not
        the fast path (FFT-bound build/extract); the marched extract is a smoothed, ~15%-biased, not-guaranteed-
        watertight estimate (bandwidth is the bias knob, dim the noise floor); the encoder bounds must exceed
        |sample|+|shift| or the FPE wraps; valid only within the sampled cloud. See holographic_fpefield."""
        from holographic.sampling_and_signal.holographic_fpefield import HolographicField
        return HolographicField.from_mesh(mesh, bounds, dim=dim, bandwidth=bandwidth, grid=grid, seed=seed)

    def mesh_point_distance(self, mesh, points, radius=2, signed=False):
        """Distance from query points (N,3) to a mesh, ACCELERATED by a vectorized spatial grid that culls the work
        (holographic_meshbridge.point_set_to_mesh_grid) -- ~20-110x faster than the brute O(N*F) scan and exact for
        near-surface queries (the regime that matters: decimation/LOD error, contact, snapping). Returns (N,) unsigned
        distance, or signed (negative inside) if `signed`. KEPT HONEST: APPROXIMATE by construction -- a query whose
        nearest triangle lies beyond `radius` cells returns +inf (raise `radius`, or use the exact brute path for
        far-field queries); see point_set_to_mesh_grid."""
        from holographic.mesh_and_geometry.holographic_meshbridge import point_set_to_mesh_grid
        return point_set_to_mesh_grid(points, mesh.vertices, mesh.faces, radius=radius, signed=signed)

    def mesh_field_lod(self, mesh, bounds, res=64, strides=(1, 2, 4), silhouette=0.95):
        """FIELD-NATIVE level-of-detail for an IMPORTED mesh (the decomposition closure): convert it to a full SDF
        once (mesh_to_sdf_grid), then RE-MARCH that field at coarser strides -- so the imported mesh coarsens exactly
        like a field-backed surface, no mesh decimation in the loop. Returns a fine->coarse list of meshes. This is
        the same field-native LOD that surface_mesh gives a native field, now reached by a mesh that arrived with no
        field. KEPT HONEST: quality is bounded by the SDF grid resolution and the nearest-normal sign; for a mesh that
        is already field-backed, use surface_mesh directly (no conversion needed)."""
        from holographic.mesh_and_geometry.holographic_meshbridge import mesh_to_sdf_grid, marching_tetrahedra_vec
        grid, axes = mesh_to_sdf_grid(mesh, bounds, res=res)
        out = []
        for s in strides:
            s = int(s)
            sub = grid[::s, ::s, ::s]
            subax = (axes[0][::s], axes[1][::s], axes[2][::s])
            out.append(marching_tetrahedra_vec(sub, subax, 0.0))
        # the chain guard: a coarse re-march whose outline is visibly destroyed is not a lower-quality option,
        # it is a wrong answer -- truncate it out of the menu (silhouette=None keeps everything).
        from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guard_chain
        floor = None if silhouette in (None, False) else float(silhouette)
        kept, rep = silhouette_guard_chain(self._as_mesh(mesh), out, min_iou=floor)
        self._last_lod_chain_silhouette = rep
        return kept

    def oct_encode_normals(self, normals, bits=8):
        """OCTAHEDRAL-encode unit normals to compact integer codes (holographic_octnormal, Cigolle et al. 2014):
        map each S^2 unit vector to 2 numbers (project onto the octahedron, unfold the lower hemisphere) and
        quantize to `bits` per component -- spending the budget on the sphere's 2 intrinsic DOF, not 3 ambient
        x/y/z. Returns integer codes (N,2) in [0, 2^bits). At equal storage this beats naive xyz quantization ~3x
        (the manifold-quantization win, reverse item R3's S^2 instance). Decode with oct_decode_normals. Kept
        negative: this is the S^2 map specifically -- the PRINCIPLE generalizes to FHRR phasors (phase angle) and
        normalized codes, the literal map does not."""
        from holographic.mesh_and_geometry.holographic_octnormal import oct_quantize
        return oct_quantize(normals, bits=bits)

    def oct_decode_normals(self, codes, bits=8):
        """Decode octahedral integer codes (N,2) back to unit normals (N,3) (holographic_octnormal). Inverse of
        oct_encode_normals."""
        from holographic.mesh_and_geometry.holographic_octnormal import oct_dequantize
        return oct_dequantize(codes, bits=bits)

    def mesh_split_face(self, mesh, f_index, i, j):
        """Cut polygon face `f_index` with a diagonal between its i-th and j-th corners (holographic_eulerops,
        FWD-7): MEF, the one operator that works on n-gons, not just triangles. E+1, F+1, chi unchanged.
        Returns a new Mesh."""
        from holographic.misc.holographic_eulerops import split_face
        return split_face(mesh, f_index, i, j)

    def mesh_poke(self, mesh, f_index, height=0.0):
        """POKE polygon face `f_index` (holographic_eulerops, FWD-7): add a vertex at the face centroid (pushed out
        along the face normal by `height`) and fan the face into triangles, one per original edge. An n-gon becomes
        n triangles. V+1, E+n, F+(n-1), chi unchanged -- a legal Euler edit. The 'poke faces' every modeler uses to
        fan a quad/ngon to triangles or raise a spike; the inverse of dissolving the center vertex. Returns a new
        Mesh. height=0 (default) keeps the center in the face plane (pure retopology, no shape change)."""
        from holographic.misc.holographic_eulerops import poke_face
        return poke_face(mesh, f_index, height=height)

    def mesh_rip_vertex(self, mesh, vertex):
        """RIP a shared vertex apart (holographic_eulerops): give every face incident to `vertex` its OWN copy at the
        same position, so the faces are no longer joined there -- the inverse of a weld at one vertex. Topology only,
        positions unchanged (the mesh looks identical but is torn at that vertex); ripping a manifold interior vertex
        opens the surface there. V rises by (incident_faces - 1). A no-op for a vertex used by <=1 face. Returns a
        new Mesh. The 'rip' a modeler does before pulling the pieces apart."""
        from holographic.misc.holographic_eulerops import rip_vertex
        return rip_vertex(mesh, vertex)

    def mesh_split_vertices(self, mesh):
        """SPLIT every vertex per-face (holographic_eulerops): give each face its own private copies of its corners,
        so no two faces share a vertex -- the full inverse of a weld (weld_mesh). The result is a 'polygon soup':
        same geometry, every face topologically independent (flat/faceted shading, no shared normals). Positions
        unchanged. weld_mesh(mesh_split_vertices(m)) recovers the welded mesh. Returns a new Mesh."""
        from holographic.misc.holographic_eulerops import split_vertices
        return split_vertices(mesh)

    def mesh_triangulate(self, mesh):
        """EAR-CLIP every face of `mesh` into triangles (holographic_meshverbs2), returning a new all-triangle Mesh.
        The concave-correct counterpart of the kernel's Mesh.triangulate() (which fans -- correct for CONVEX faces
        only). Ear clipping (Meisters 1975) repeatedly removes a convex corner whose triangle holds no other vertex,
        so a concave n-gon tiles exactly instead of the flipped/overlapping triangles a fan gives. No new vertices
        (unlike mesh_poke); only the face list changes. Deterministic. Returns a new Mesh."""
        from holographic.mesh_and_geometry.holographic_meshverbs2 import triangulate_ngons
        return triangulate_ngons(mesh)

    def mesh_smooth(self, mesh, lam=0.55, mu=-0.58, iters=8, weights="cotangent"):
        """Taubin lambda|mu no-shrink mesh smoothing / denoising (holographic_meshsmooth, FWD-4): the shipped
        `graphsignal.taubin_filter` WIRED onto explicit mesh geometry -- 3-D vertex positions as the signal,
        the mesh's 1-ring as the graph, cotangent (discrete Laplace-Beltrami) weights by default. Removes
        vertex noise while preserving the overall extent (no volume shrink, unlike a naive Laplacian). Faces
        are untouched, so connectivity and every Euler invariant are preserved -- only vertices move. Returns a
        new Mesh. `weights='uniform'` for the umbrella baseline. Kept negative: fixed strength over-smooths an
        already-clean mesh (it is a low-pass) -- needs a noise estimate; this exposes lam/mu/iters, no auto-tune."""
        from holographic.mesh_and_geometry.holographic_meshsmooth import taubin_smooth
        return taubin_smooth(mesh, lam=lam, mu=mu, iters=iters, weights=weights)

    def mesh_curvature(self, mesh, kind="mean"):
        """Per-vertex curvature of a mesh (holographic_meshcurvature, FWD-6). kind='mean' -> |H| via the
        cotangent Laplacian of positions (reusing FWD-4's weights; H=1/R on a sphere); kind='gaussian' -> the
        area-normalised angle defect (K=1/R^2 on a sphere), whose TOTAL equals 2*pi*chi exactly by discrete
        Gauss-Bonnet. Returns an array of length V. Kept negative: per-vertex values are noisy on coarse/
        irregular meshes (the mean is accurate); `mesh_curvature_confidence` scores per-vertex reliability."""
        from holographic.mesh_and_geometry.holographic_meshcurvature import mean_curvature, gaussian_curvature
        if kind == "mean":
            return mean_curvature(mesh)
        if kind == "gaussian":
            return gaussian_curvature(mesh)
        raise ValueError(f"kind must be 'mean' or 'gaussian', got {kind!r}")

    def mesh_curvature_confidence(self, mesh):
        """A per-vertex confidence in [0,1] for the curvature estimate (holographic_meshcurvature, FWD-6), from
        1-ring regularity -- low where the neighbourhood is sliver-heavy or sparse, so a caller can down-weight
        unreliable curvature rather than trust it (the noise kept-negative made actionable)."""
        from holographic.mesh_and_geometry.holographic_meshcurvature import curvature_confidence
        return curvature_confidence(mesh)

    def mesh_creases(self, mesh, threshold_deg=30.0):
        """The sharp edges of a mesh (holographic_meshcurvature, FWD-6): interior edges whose DIHEDRAL angle
        (between the two adjacent faces) exceeds `threshold_deg`. A cube returns its 12 edges; a smooth sphere
        returns none. Feeds crease-aware smoothing, adaptive subdivision, and shading-normal splitting. Returns
        a sorted list of (lo,hi) vertex-index edges."""
        from holographic.mesh_and_geometry.holographic_meshcurvature import detect_creases
        return detect_creases(mesh, threshold_deg=threshold_deg)

    def mesh_geodesic(self, mesh, source):
        """Single-source surface geodesic distance (holographic_meshgeodesic, FWD-5): shortest path ALONG mesh
        edges from vertex `source` to every vertex (Dijkstra with Euclidean edge weights) -- distance over the
        surface, not the straight line through the void. Returns an array of length V. The along-surface metric
        UV seams (FWD-3), soft selections, and remesh spacing all want. Kept negative: the edge-graph distance
        is approximate (a few percent over the smooth geodesic, with a tiny chord-effect undercut near source)."""
        from holographic.mesh_and_geometry.holographic_meshgeodesic import geodesic_distances
        return geodesic_distances(mesh, source)

    def mesh_soft_selection(self, mesh, source, radius, falloff="smooth"):
        """A soft-selection falloff in [0,1] per vertex by GEODESIC distance from `source` within `radius`
        (holographic_meshgeodesic, FWD-5): 1 at the source, smoothly to 0 at the radius. Because it measures
        along the surface it does NOT bleed to vertices near in 3-D space but far across the surface (a Euclidean
        brush's failure on a thin neck or a folded region). `falloff`: 'smooth' or 'linear'. Returns length-V."""
        from holographic.mesh_and_geometry.holographic_meshgeodesic import geodesic_soft_selection
        return geodesic_soft_selection(mesh, source, radius, falloff=falloff)

    def mesh_uv_unwrap(self, mesh, method="isomap"):
        """UV-unwrap a (disk-topology) mesh to 2-D coordinates (holographic_meshuv, FWD-3): classical MDS of the
        mesh's GEODESIC distance matrix -- Isomap on explicit edges -- so surface distances are preserved as well
        as a plane allows. The shipped manifold-chart machinery pointed at real mesh connectivity. Returns (V,2)
        UV in ~[0,1]^2. `method`: 'isomap' (geodesic; wins on curved surfaces), 'planar' (linear; exact on
        developable surfaces), 'spectral' (Laplacian eigenmaps). Kept negative: a CLOSED surface needs a seam
        (cut) first -- `holographic_meshuv.puncture` opens it crudely; a real seam is the ARCH-4 atlas piece."""
        from holographic.mesh_and_geometry.holographic_meshuv import uv_unwrap
        return uv_unwrap(mesh, method=method)

    def mesh_pack_uv(self, mesh, method="lscm", margin=0.02):
        """PACK UV ISLANDS (holographic_meshuv): unwrap each connected component (UV island) of `mesh` SEPARATELY,
        then lay the islands out in NON-OVERLAPPING cells of the unit UV square -- the 'pack islands' / smart-UV step
        that mesh_lscm and mesh_uv_unwrap skip (they solve every component in one frame, so disconnected pieces land
        on top of each other). `method`: 'lscm' (conformal per island) or 'isomap' (geodesic MDS). Each island is
        scaled uniformly (no UV stretch) into its cell with a `margin` gutter. Returns a (V,2) UV array in [0,1]^2.
        Composes the existing per-chart unwrap + a connected-components split; does not re-solve the unwrap."""
        from holographic.mesh_and_geometry.holographic_meshuv import pack_uv_islands
        return pack_uv_islands(mesh, method=method, margin=margin)

    def mesh_stable_uv(self, mesh, bounds=None, mode="triplanar", axis=2):
        """UVs that are a deterministic function of WORLD POSITION, so they DON'T move under local edits -- the
        stable counterpart to mesh_uv_unwrap (whose global MDS/eigenmap re-solves and can flip on any edit).
        Normalised by the FIXED `bounds` (field domain) so the scale is edit-invariant. mode='triplanar' picks
        each vertex's projection plane by its normal (curves don't fold); 'planar' drops `axis`. Stable texturing,
        not a seam-cut chart. See holographic_meshuv.stable_uv."""
        from holographic.mesh_and_geometry.holographic_meshuv import stable_uv
        return stable_uv(mesh, bounds=bounds, mode=mode, axis=axis)

    def mesh_face_type(self, mesh, face_type="quad", planarity=0.90, normal_tol=0.999):
        """Convert ANY triangle mesh's face standard (holographic_meshpoly) WITHOUT moving vertices (so stable
        keys survive): 'quad' merges coplanar triangle pairs into quad-dominant output; 'ngon' merges connected
        coplanar regions into single n-gons where the boundary is a clean loop (a flat wall -> one face);
        'triangle' returns it unchanged. Use after surface_mesh / on an imported mesh. See holographic_meshpoly."""
        if face_type == "quad":
            from holographic.mesh_and_geometry.holographic_meshpoly import triangles_to_quads
            return triangles_to_quads(mesh, planarity=planarity)
        if face_type == "ngon":
            from holographic.mesh_and_geometry.holographic_meshpoly import merge_coplanar
            return merge_coplanar(mesh, normal_tol=normal_tol)
        return mesh

    def mesh_face_counts(self, mesh):
        """{3: triangles, 4: quads, 5: n-gons} -- the face-standard summary. See holographic_meshpoly."""
        from holographic.mesh_and_geometry.holographic_meshpoly import face_type_counts
        return face_type_counts(mesh)

    def mesh_uv_distortion(self, mesh, uv):
        """The per-edge STRETCH distortion of a UV map (holographic_meshuv, FWD-3): spread of (UV edge length /
        3-D edge length), 0 = isometric, growing with Gaussian curvature (Gauss: a curved surface cannot flatten
        without stretch). The scale-invariant flatness score -- lower is a better parameterisation."""
        from holographic.mesh_and_geometry.holographic_meshuv import uv_distortion
        return uv_distortion(mesh, uv)

    def mesh_cut_seam(self, mesh, seam):
        """Cut a mesh open along a SEAM (holographic_meshseam, ARCH-4): given `seam` (an ordered list of vertex
        indices forming an edge path), duplicate the seam's interior vertices on a consistent side, opening a
        closed surface into a disk -- the REAL seam that FWD-3's crude `puncture` stood in for. Non-destructive
        (every face preserved, unlike puncture which deletes faces). A meridian-cut sphere becomes a disk (chi=1).
        Returns a new Mesh. Kept negative: seam CHOICE matters (a full pole-to-pole meridian unwraps worse than a
        pole-to-equator cut); a good atlas needs several cuts (deferred)."""
        from holographic.mesh_and_geometry.holographic_meshseam import cut_seam
        return cut_seam(mesh, seam)

    def mesh_shortest_seam(self, mesh, a, b):
        """A seam path between two vertices (holographic_meshseam, ARCH-4): the shortest edge path from `a` to `b`
        (Dijkstra on the mesh edge graph), e.g. a meridian from a pole to its antipode. Returns an ordered list of
        vertex indices to hand to mesh_cut_seam."""
        from holographic.mesh_and_geometry.holographic_meshseam import shortest_seam
        return shortest_seam(mesh, a, b)

    def mesh_auto_seam(self, mesh, threshold_deg=40.0, method="crease"):
        """AUTO-MARK SEAMS for UV unwrapping (holographic_meshseam): choose which edges to cut WITHOUT naming a path.
        Returns a sorted list of (lo,hi) seam edges -- the 'marked seams' (the red edges a modeler sees). Where
        mesh_cut_seam / mesh_shortest_seam cut a GIVEN seam, this SELECTS one. method='crease' (default) seams along
        the sharp edges (dihedral angle > threshold_deg), since an artist cuts where the surface already folds so the
        cut is hidden. Composes mesh_creases. Kept negative: a smooth closed surface has no creases -> empty set
        (honest, not an invented cut); use mesh_shortest_seam for a meridian there. Returns the edge list."""
        from holographic.mesh_and_geometry.holographic_meshseam import auto_seam
        return auto_seam(mesh, threshold_deg=threshold_deg, method=method)

    def mesh_extrude(self, mesh, face_index, distance, quad_walls=False):
        """EXTRUDE a face (holographic_meshverbs, FWD-7): lift face `face_index` along its outward normal by
        `distance` and wall it in -- the iconic modelling verb. Preserves the Euler characteristic and keeps a
        closed mesh a closed manifold; the cap moves exactly `distance` along the normal. Returns a new Mesh."""
        from holographic.mesh_and_geometry.holographic_meshverbs import extrude_face
        return extrude_face(mesh, face_index, distance, quad_walls=quad_walls)

    def curve_bezier(self, control, u):
        """Evaluate a BEZIER curve of any degree at parameter(s) u in [0,1] by de Casteljau (numerically stable).
        `control` is (k, dim). Returns the point(s). Passes through the first/last control point. Use for a smooth
        segment, a camera ease, or a tube centreline. See holographic_curves.bezier."""
        from holographic.mesh_and_geometry.holographic_curves import bezier
        return bezier(control, u)

    def curve_catmull_rom(self, control, n, alpha=0.5, closed=False):
        """Sample a CATMULL-ROM spline that INTERPOLATES `control` (passes through every point) -- the right
        curve for a camera path or a scatter path that must hit its keyframes. `alpha=0.5` (centripetal) avoids
        cusps on sharp turns. Returns (n, dim). See holographic_curves.catmull_rom."""
        from holographic.mesh_and_geometry.holographic_curves import catmull_rom
        return catmull_rom(control, n, alpha=alpha, closed=closed)

    def curve_bspline(self, control, n, degree=3, closed=False):
        """Sample a uniform B-SPLINE (default cubic) over `control` -- the smoothest (C^(degree-1)) of the
        splines, approximating not interpolating. The flowing-camera-move curve. Returns (n, dim). See
        holographic_curves.bspline."""
        from holographic.mesh_and_geometry.holographic_curves import bspline
        return bspline(control, n, degree=degree, closed=closed)

    def curve_frame(self, points, closed=False, minimizing=True):
        """Orthonormal FRAME (tangent, normal, binormal) at each point of a 3-D curve. `minimizing=True` gives
        the ROTATION-MINIMIZING frame (stable on straight and inflecting curves -- what a tube sweep / spline
        camera wants); False gives the true FRENET frame (flips at inflections, undefined on straight runs).
        Returns (T, N, B). See holographic_curves.rotation_minimizing_frame / frenet_frame."""
        from holographic.mesh_and_geometry.holographic_curves import rotation_minimizing_frame, frenet_frame
        return rotation_minimizing_frame(points, closed=closed) if minimizing else frenet_frame(points, closed=closed)

    def curve_resample_arc_length(self, points, n):
        """Resample a curve to `n` points spaced by EQUAL ARC LENGTH (evenly along the curve, not the parameter)
        -- what a tube wants so its rings are uniform, or a scatter wants for even spacing. See
        holographic_curves.resample_by_arc_length."""
        from holographic.mesh_and_geometry.holographic_curves import resample_by_arc_length
        return resample_by_arc_length(points, n)

    def sweep_tube(self, points, profile=None, radius=0.1, closed=False):
        """Sweep a 2-D `profile` (default a circle of `radius`) along a 3-D curve into a watertight TUBE mesh,
        oriented by the rotation-minimizing frame so it does not twist/tear. Returns (vertices, faces) -- pass to
        Mesh(). The curve->geometry bridge: a bezier/knot/helix becomes hair, cable, vine, neon, a knotted
        sculpture. `closed=True` welds a looped curve (a torus knot). See holographic_curves.sweep_tube."""
        from holographic.mesh_and_geometry.holographic_curves import sweep_tube
        return sweep_tube(points, profile=profile, radius=radius, closed=closed)

    def torus_knot(self, n=400, p=2, q=3, R=1.0, r=0.4):
        """A (p, q) TORUS KNOT curve: winds p times around the axis, q through the hole (p=2,q=3 = trefoil).
        Returns (n, 3) points -- sweep_tube it for the demoscene classic. See holographic_curves.torus_knot."""
        from holographic.mesh_and_geometry.holographic_curves import torus_knot
        return torus_knot(n=n, p=p, q=q, R=R, r=r)

    def trefoil_knot(self, n=400, scale=1.0):
        """The TREFOIL knot (simplest non-trivial knot) as (n, 3) points. See holographic_curves.trefoil."""
        from holographic.mesh_and_geometry.holographic_curves import trefoil
        return trefoil(n=n, scale=scale)

    def helix(self, n=200, radius=1.0, pitch=0.3, turns=3.0):
        """A HELIX curve: n points spiralling `turns` times at `radius`, rising `pitch` per turn. (n, 3). Sweep
        it for a spring, a screw, a DNA strand. See holographic_curves.helix."""
        from holographic.mesh_and_geometry.holographic_curves import helix
        return helix(n=n, radius=radius, pitch=pitch, turns=turns)

    def superellipsoid(self, nu=48, nv=48, e1=0.5, e2=0.5, a=1.0, b=1.0, c=1.0):
        """A SUPERELLIPSOID surface as a point grid: `e1,e2` squareness (1,1)=ellipsoid, ->0=box, >1=star; `a,b,c`
        semi-axes. Barr's rounded-solid family from two exponents. See holographic_curves.superellipsoid."""
        from holographic.mesh_and_geometry.holographic_curves import superellipsoid
        return superellipsoid(nu=nu, nv=nv, e1=e1, e2=e2, a=a, b=b, c=c)

    def gyroid_field(self, points, scale=1.0):
        """The GYROID triply-periodic minimal surface as an implicit FIELD (surface at f=0): sin x cos y + ... .
        Returns f at each point -- mesh the zero set, or use as an SDF-like field for infinite seamless lattice
        art. See holographic_curves.gyroid_field."""
        from holographic.mesh_and_geometry.holographic_curves import gyroid_field
        return gyroid_field(points, scale=scale)

    def klein_bottle(self, nu=48, nv=48, scale=1.0):
        """A KLEIN BOTTLE (figure-8 immersion) as a point grid -- the classic non-orientable surface showpiece.
        See holographic_curves.klein_bottle."""
        from holographic.mesh_and_geometry.holographic_curves import klein_bottle
        return klein_bottle(nu=nu, nv=nv, scale=scale)

    def voxelize_mesh(self, vertices, faces, res=32, pad=0.1, threshold=0.5):
        """VOXELISE a triangle mesh into an occupancy grid via the generalised WINDING NUMBER (Jacobson 2013) --
        robust to NON-watertight and self-intersecting meshes, unlike ray-parity. `res` voxels along the longest
        axis; a voxel is solid where |winding| >= threshold. Returns (occupancy (nx,ny,nz) bool, origin, spacing).
        O(voxels x triangles) -- for a heavy mesh use voxelize_sdf or decimate first. See
        holographic_voxelize.voxelize_mesh."""
        from holographic.mesh_and_geometry.holographic_voxelize import voxelize_mesh
        return voxelize_mesh(vertices, faces, res=res, pad=pad, threshold=threshold)

    def voxelize_sdf(self, sdf, lo, hi, res=32, iso=0.0):
        """VOXELISE an SDF / field into an occupancy grid: solid where field <= iso. O(voxels), no winding number
        -- the fast path for an implicit. `lo`,`hi` are box corners. Returns (occupancy, origin, spacing), the
        same layout as voxelize_mesh so they are interchangeable. See holographic_voxelize.voxelize_sdf."""
        from holographic.mesh_and_geometry.holographic_voxelize import voxelize_sdf
        return voxelize_sdf(sdf, lo, hi, res=res, iso=iso)

    def voxel_centres(self, occ, origin, spacing):
        """World-space centres (m, 3) of the SOLID voxels of an occupancy grid -- a point cloud of the volume,
        for instancing, point rendering, or feeding a mesher. See holographic_voxelize.voxel_centres."""
        from holographic.mesh_and_geometry.holographic_voxelize import voxel_centres
        return voxel_centres(occ, origin, spacing)

    def occupancy_to_mesh(self, occ, origin, spacing):
        """Extract a surface MESH (vertices, quads) from an occupancy grid via surface_nets -- closes the round
        trip mesh -> voxels -> mesh (a voxel-resolution resample of the input). See
        holographic_voxelize.occupancy_to_mesh."""
        from holographic.mesh_and_geometry.holographic_voxelize import occupancy_to_mesh
        return occupancy_to_mesh(occ, origin, spacing)

    def mesh_winding_number(self, points, vertices, faces):
        """Generalised WINDING NUMBER of each query point w.r.t. a triangle mesh: ~1 inside a closed surface, ~0
        outside, fractional near boundaries (robust inside/outside test for non-watertight meshes). Returns (n,)
        in turns. See holographic_voxelize.winding_number."""
        from holographic.mesh_and_geometry.holographic_voxelize import winding_number
        return winding_number(points, vertices, faces)

    def nurbs_curve(self, control, weights=None, n=100, degree=3, knots=None):
        """Evaluate a NURBS (rational B-spline) CURVE: `control` (k,dim) with per-point `weights` (default 1 -> a
        plain B-spline). Weights are what let a NURBS hold a CONIC exactly (a circle/arc), which a polynomial
        B-spline only approximates. Returns (n, dim). See holographic_nurbs.nurbs_curve."""
        from holographic.mesh_and_geometry.holographic_nurbs import nurbs_curve
        return nurbs_curve(control, weights=weights, n=n, degree=degree, knots=knots)

    def nurbs_surface(self, control_grid, weights=None, nu=40, nv=40, degree_u=3, degree_v=3,
                      knots_u=None, knots_v=None):
        """Evaluate a NURBS SURFACE (tensor-product rational B-spline): `control_grid` (ku,kv,3) control net with
        per-point `weights` (ku,kv). Samples an (nu,nv) grid -> (nu*nv, 3) points. The CAD surface primitive -- a
        rational patch that can be a sphere cap, cylinder, or a swoopy panel. See holographic_nurbs.nurbs_surface."""
        from holographic.mesh_and_geometry.holographic_nurbs import nurbs_surface
        return nurbs_surface(control_grid, weights=weights, nu=nu, nv=nv,
                             degree_u=degree_u, degree_v=degree_v, knots_u=knots_u, knots_v=knots_v)

    def nurbs_surface_mesh(self, control_grid, weights=None, nu=40, nv=40, degree_u=3, degree_v=3):
        """Tessellate a NURBS surface into a triangle MESH (vertices, faces) -- the bridge from a CAD patch to the
        engine's mesh pipeline (render / voxelise / DCC). See holographic_nurbs.nurbs_surface_mesh."""
        from holographic.mesh_and_geometry.holographic_nurbs import nurbs_surface_mesh
        return nurbs_surface_mesh(control_grid, weights=weights, nu=nu, nv=nv,
                                  degree_u=degree_u, degree_v=degree_v)

    def nurbs_circle(self, radius=1.0, n=100):
        """A NURBS CIRCLE -- the canonical proof that NURBS represent conics EXACTLY (every point at `radius` to
        ~1e-12, which a polynomial B-spline cannot do). Returns (n, 3). See holographic_nurbs.nurbs_circle."""
        from holographic.mesh_and_geometry.holographic_nurbs import nurbs_circle
        return nurbs_circle(radius=radius, n=n)

    def mesh_inset(self, mesh, face_index, ratio, quad_walls=False):
        """INSET a face (holographic_meshverbs, FWD-7): shrink face `face_index` toward its centroid by `ratio`,
        ringing it with new faces around a smaller central face (in-plane, so the central face stays coplanar; its
        area is exactly (1-ratio)^2 of the original). Preserves chi + closed + manifold. Returns a new Mesh."""
        from holographic.mesh_and_geometry.holographic_meshverbs import inset_face
        return inset_face(mesh, face_index, ratio, quad_walls=quad_walls)

    def mesh_dissolve_vertex(self, mesh, vertex):
        """DISSOLVE a vertex (holographic_meshverbs, FWD-7; the Euler KEV verb): remove `vertex` and its incident
        faces, then fan-triangulate the hole, leaving the surrounding ring fixed. Preserves chi + closed +
        manifold, one fewer vertex. Distinct from `mesh_collapse_edge` (the decimation cousin). Returns a Mesh."""
        from holographic.mesh_and_geometry.holographic_meshverbs import dissolve_vertex
        return dissolve_vertex(mesh, vertex)

    def mesh_bevel_vertex(self, mesh, vertex, ratio=0.25, segments=1):
        """BEVEL a corner (holographic_meshverbs2, FWD-7 remainder): pull each edge incident to `vertex` back toward
        its neighbour by `ratio`, chamfer every incident face, and cap the hole. `segments=1` (default) caps with one
        FLAT facet (byte-identical to the original bevel); `segments>=2` ROUNDS the corner into a smooth spherical
        dome of that many rings -- the 'bevel with N segments' fillet. Preserves closed + manifold. Returns a Mesh.
        Kept negative: this is the VERTEX bevel; the EDGE bevel (two-sided fan split) is deferred; ratio in (0,1)."""
        from holographic.mesh_and_geometry.holographic_meshverbs2 import bevel_vertex_segments
        return bevel_vertex_segments(mesh, vertex, ratio=ratio, segments=segments)

    def mesh_fill_holes(self, mesh, mode="fan", max_sides=0):
        """FILL open holes (boundary loops) of a mesh with faces (holographic_meshverbs2), returning a closed-up Mesh.
        mode='fan' (default, always works) caps each loop with a centroid vertex + a triangle fan; mode='grid' bridges
        a big-enough even loop with a coarser quad strip (Blender 'grid fill'), falling back to fan otherwise.
        `max_sides` (Blender 'Sides') fills only loops with at most that many edges (0 = all) -- set it to close small
        holes while leaving a large outer border open. Traces loops with face-consistent winding so the fill is
        manifold. Returns a new Mesh."""
        from holographic.mesh_and_geometry.holographic_meshverbs2 import fill_holes
        return fill_holes(mesh, mode=mode, max_sides=max_sides)

    def mesh_weld(self, mesh, tol=1e-5):
        """WELD vertices closer than `tol` into one (holographic_meshtools.merge_by_distance): snap to a tol grid,
        average each group, remap faces, drop any face that collapsed to a degenerate. The standard cleanup after a
        mirror / import / boolean / marching-cubes extraction. Returns a new Mesh. See mesh_repair for the full
        weld+fill+compact pass."""
        from holographic.mesh_and_geometry.holographic_meshtools import merge_by_distance
        return merge_by_distance(mesh, tol=tol)

    def mesh_repair(self, mesh, weld_tol=1e-5, fill_holes=True, max_fill_sides=0, drop_unreferenced=True,
                    split_nonmanifold=True, triangulate=False):
        """REPAIR a raw mesh by composing the standard cleanup ops (holographic_meshtools.mesh_repair): WELD
        near-duplicate vertices (also drops degenerate faces), SPLIT non-manifold vertices into connected umbrellas
        (makes the mesh MANIFOLD so the cross-field retopo will accept it), optionally FILL open holes, and DROP
        unreferenced vertices; triangulate=True ear-clips to uniform triangles (cross_field needs a single face arity).
        Returns (repaired_mesh, report) with before/after vertex+face counts, manifold/closed flags and the split
        count -- so a marching-cubes / import / boolean / photo-to-mesh result is made RETOPO-READY. Deterministic;
        never raises. For even topology follow with retopology (cross_field). See mesh_make_manifold for split-only."""
        from holographic.mesh_and_geometry.holographic_meshtools import mesh_repair
        return mesh_repair(mesh, weld_tol=weld_tol, fill_holes=fill_holes, max_fill_sides=max_fill_sides,
                           drop_unreferenced=drop_unreferenced, split_nonmanifold=split_nonmanifold,
                           triangulate=triangulate)

    def route_repair(self, mesh, margin=0.15, weld_tol=1e-5, max_fill_sides=0):
        """Route a mesh to the MINIMAL repair its defect needs instead of always running the full pipeline:
        diagnose the mesh into a categorical defect record {manifold, closed, duplicates}, match it against the
        repair-strategy records (match_record), and run only the ops the winning strategy names -- welds a
        duplicate-only mesh without a hole-fill pass, etc. If the defect is ambiguous (decide_or_abstain), it
        falls back to the full safe mesh_repair, so it never repairs LESS than needed. Returns (mesh, report)
        with {strategy, confident, defect} added. Cheaper and self-explaining than mesh_repair. See
        holographic_meshtools.route_repair."""
        from holographic.mesh_and_geometry.holographic_meshtools import route_repair
        return route_repair(mesh, mind=self, margin=margin, weld_tol=weld_tol, max_fill_sides=max_fill_sides)

    def mesh_make_manifold(self, mesh):
        """Make a mesh MANIFOLD by splitting non-manifold vertices into their connected umbrellas
        (holographic_meshtools.split_nonmanifold_vertices): at each vertex, incident faces are grouped across MANIFOLD
        edges only, and a vertex whose faces form more than one umbrella (a bowtie, or the endpoint of an edge shared
        by >2 faces) is duplicated per umbrella. Resolves non-manifold EDGES too, so a half-edge build / cross-field
        retopo accepts the result. Unlike mesh_rip_vertex (per-face) / mesh_split_vertices (explode-all), this is the
        minimal cut and a NO-OP on a clean mesh. Returns (mesh, report). See mesh_repair for the full weld+fill pass."""
        from holographic.mesh_and_geometry.holographic_meshtools import split_nonmanifold_vertices
        return split_nonmanifold_vertices(mesh)

    def transfer_uv(self, source_mesh, source_uv, target_vertices, cell_scale=1.0):
        """TRANSFER per-vertex UVs (or ANY per-vertex attribute) from source_mesh onto new target_vertices by
        closest-point projection + barycentric interpolation -- the step that makes RETOPO TEXTURE-PRESERVING (the
        remeshed surface lies on the original, so each new vertex takes the interpolated UV of its closest source
        triangle). Spatial-hash accelerated. Returns (attr (n,k), residual_distance (n,)) -- the residual is the
        honest error signal (large = the target strayed off the source surface). KEPT NEG: wrong across UV seams
        (closest triangle can be another island). See holographic_meshtools.transfer_uv."""
        from holographic.mesh_and_geometry.holographic_meshtools import transfer_uv as _tuv
        return _tuv(self._as_mesh(source_mesh), source_uv, target_vertices, cell_scale=cell_scale)

    def shrinkwrap(self, mesh, target_mesh, factor=1.0, cell_scale=1.0):
        """SHRINKWRAP: move each vertex of `mesh` onto its closest point on `target_mesh` (Blender shrinkwrap /
        retopo-snap). factor 1.0 lands on the surface, 0.5 pulls halfway, 0.0 is a no-op; topology unchanged.
        Returns (new_mesh, residual) where residual is the distance each vertex closed. THE retopo finisher: a box
        model / remesh has clean topology but approximate positions -- one pass snaps positions onto the reference
        (fixed our box-model surface residual 0.0158 -> ~0). See holographic_meshtools.shrinkwrap."""
        from holographic.mesh_and_geometry.holographic_meshtools import shrinkwrap as _sw
        return _sw(self._as_mesh(mesh), self._as_mesh(target_mesh), factor=factor, cell_scale=cell_scale)

    def make_uv_shell(self, mesh, uvs, offset=0.02, relative=True):
        """Build a UV SHELL (texture-carrying ENVELOPE): push every vertex OUTWARD along its normal by `offset`,
        keeping the faces and per-vertex `uvs`. A slightly-inflated cage that holds the texture INDEPENDENT of the
        surface topology -- freeze the map here, then project_uv_from_shell onto any modified mesh (LOD/retopo).
        relative=True scales offset by the bbox diagonal. Returns a Mesh with .uvs. See
        holographic_meshtools.make_uv_shell."""
        from holographic.mesh_and_geometry.holographic_meshtools import make_uv_shell as _mus
        return _mus(self._as_mesh(mesh), uvs, offset=offset, relative=relative)

    def project_uv_from_shell(self, new_mesh, shell, shell_uvs=None, cell_scale=1.0):
        """PROJECT a UV map from a texture-carrying SHELL onto a new mesh of ANY topology: for each new vertex, find
        the closest point on the shell and read its interpolated UV. The other half of the shell workflow -- after
        make_uv_shell freezes the map, this recovers texture coords on a LOD/retopo/remesh. Returns (uvs, residual).
        See holographic_meshtools.project_uv_from_shell."""
        from holographic.mesh_and_geometry.holographic_meshtools import project_uv_from_shell as _pufs
        return _pufs(self._as_mesh(new_mesh), self._as_mesh(shell), shell_uvs=shell_uvs, cell_scale=cell_scale)


    def skin_skeleton(self, verts, edges, radii, resolution=64, smooth_k=None, taper=True):
        """SKIN A SKELETON (B-Mesh): wrap a stick figure -- verts (n,3), edges [(i,j)...], per-vertex radii (n,) --
        in one watertight surface, so you model a creature from ~20 joints instead of extruding 200 faces. Each edge
        becomes a capsule; branches at a shared joint MERGE automatically (smooth_union), then marching-cubes to a
        Mesh. The base-mesh block-out to retopo onto (quad_remesh / shrinkwrap a cage). See
        holographic_meshtools.skin_skeleton."""
        from holographic.mesh_and_geometry.holographic_meshtools import skin_skeleton as _sk
        return _sk(verts, edges, radii, resolution=resolution, smooth_k=smooth_k, taper=taper)

    def fit_base_mesh(self, target_mesh, verts, edges, radii, resolution=64, smooth_k=None, shrink_factor=1.0):
        """FIT A BASE MESH TO A TARGET: skin the skeleton into a base mesh, SHRINKWRAP it onto target_mesh, and
        report the silhouette-fit gain. The block-out-then-snap loop (Blender skin modifier -> snap to sculpt), and
        an OPTIMISATION target since it returns iou_base/iou_fitted. Returns {base, fitted, residual, iou_base,
        iou_fitted}. See holographic_meshtools.fit_base_mesh."""
        from holographic.mesh_and_geometry.holographic_meshtools import fit_base_mesh as _fbm
        return _fbm(self._as_mesh(target_mesh), verts, edges, radii, resolution=resolution, smooth_k=smooth_k,
                    shrink_factor=shrink_factor)

    def bake_normal_map(self, low_mesh, low_uv, high_mesh, size=256, world_space=False, ao=False, ao_samples=0, displacement=None, max_distance=None):
        """BAKE a normal map (optionally AO) from a HIGH-poly onto a LOW-poly UVs -- keep the sculpt detail on the
        retopo. For each texel: low-poly 3-D point -> closest point on the high-poly -> its normal, stored tangent-
        space (default; portable lavender map) or world-space. ao=True + ao_samples returns (normal_img, ao_img). displacement=True ALSO bakes a signed height map (positive=bump, negative=dent) from the SAME closest-point projection (M14: one cast, many channels), clamped to max_distance -- the cage a displacement map needs because a stray hit moves GEOMETRY, not just shading. Outputs append in order normal[, ao][, displacement].
        Returns an (size,size,3) image. See holographic_meshtools.bake_normal_map."""
        from holographic.mesh_and_geometry.holographic_meshtools import bake_normal_map as _bnm
        return _bnm(self._as_mesh(low_mesh), low_uv, self._as_mesh(high_mesh), size=size,
                    world_space=world_space, ao=ao, ao_samples=ao_samples, displacement=displacement, max_distance=max_distance)

    def auto_retopo(self, mesh, voxel_resolution=16, subdivide=0, target=None, silhouette=0.95,
                    max_voxel_resolution=48):
        """AUTO-RETOPO: turn a messy BLOCK-OUT (a skin_skeleton blob, metaball, boolean mess) into a clean quad-
        dominant cage in one call -- voxel_remesh (coarse; keep voxel_resolution ~12-20) -> quad_remesh -> optional
        catmull_clark(subdivide). If target is given, shrinkwrap onto it and score silhouette IoU. Returns {mesh,
        quad_fraction, report, iou?}. The hand-off that ends the base-mesh pipeline: place joints -> clean model.
        See holographic_meshtools.auto_retopo."""
        from holographic.mesh_and_geometry.holographic_meshtools import auto_retopo as _ar
        from holographic.mesh_and_geometry.holographic_meshqem import silhouette_guarded
        src = self._as_mesh(mesh)
        floor = None if silhouette in (None, False) else float(silhouette)
        if floor is not None:
            # voxel_resolution is a CUBIC knob: res^3 grid -> res^3-ish faces -> cross_field. Declaring the
            # cost curve is what stops the guard walking it into an OOM (it did, twice, before R2). max_knob
            # caps the walk at a resolution whose marched mesh cross_field can still take; past that the guard
            # REFUSES loudly (report.refused_knob_cap) rather than dying, and the caller can raise the cap,
            # opt out with silhouette=None, or -- the real answer for scans -- use the surface route (R3).
            out, rep = silhouette_guarded(
                src, lambda r: _ar(src, voxel_resolution=r, subdivide=subdivide, target=target),
                int(voxel_resolution), min_iou=floor, knob_cost="cubic", max_knob=int(max_voxel_resolution),
                get_mesh=lambda d: d["mesh"] if isinstance(d, dict) else d)
            if isinstance(out, dict):
                out["silhouette_report"] = rep
            return out
        return _ar(self._as_mesh(mesh), voxel_resolution=voxel_resolution, subdivide=subdivide,
                   target=(self._as_mesh(target) if target is not None else None))












    def mesh_bridge(self, verts, loop_a, loop_b, closed=True):
        """BRIDGE two edge loops (holographic_meshverbs2, FWD-7 remainder): join two equal-length ordered vertex
        loops with a band of quads -- the verb that builds a tube between two openings. `verts` holds all points;
        `loop_a`/`loop_b` are equal-length index lists. Returns a Mesh of the band. Kept negative: requires
        equal-length, already-aligned loops (the caller supplies the correspondence); matching unequal loops is
        deferred."""
        from holographic.mesh_and_geometry.holographic_meshverbs2 import bridge_loops
        return bridge_loops(verts, loop_a, loop_b, closed=closed)

    def mesh_loop_cut(self, mesh, start_face, start_edge, cuts=1, factor=0.5):
        """LOOP-CUT: insert an edge loop (holographic_meshverbs2, FWD-7 remainder): trace the perpendicular loop of
        quads (enter through one edge, leave through the OPPOSITE edge) carrying `start_edge` of quad `start_face`,
        and split every crossed quad in two with a new mid-loop -- the verb that adds a ring of resolution.
        Preserves chi. Returns a Mesh. Kept negative: QUADS only (the opposite-edge trace is undefined on
        triangles); the trace stops at a boundary or when it returns to the start."""
        from holographic.mesh_and_geometry.holographic_meshverbs2 import loop_cut
        return loop_cut(mesh, start_face, start_edge, cuts=cuts, factor=factor)

    def pack_images(self, images):
        """Pack a FAMILY of 8-bit images as ONE reference plus per-image deltas, entropy-coded with zlib. Lossless
        and bit-exact (the residual is taken modulo 256); `mind.unpack_images(blob)` returns the originals byte for
        byte. The reference is chosen automatically (first / per-pixel mean / median -- whichever packs smallest).

        WHEN IT WINS, MEASURED: images sharing large BIT-IDENTICAL regions and differing in localized spots -- a
        logo suite, sprite variants, UI frames, scanned pages. On a 6-logo suite: 1,744 B against 3,553 B of
        per-file PNG and 3,162 B of gzip-the-whole-set.

        KEPT NEGATIVE, loud: it LOSES badly when each image is already compressible on its own. On six smooth
        gradients it packs to 32,274 B against per-file PNG's 1,987 B -- sixteen times WORSE. Run
        mind.pack_benchmark(images) and read the table; do not guess. See holographic_pack."""
        from holographic.misc.holographic_pack import pack
        return pack(images)

    def unpack_images(self, blob):
        """Reconstruct the image family packed by mind.pack_images -- bit-exact, byte for byte.
        See holographic_pack.unpack."""
        from holographic.misc.holographic_pack import unpack
        return unpack(blob)

    def pack_benchmark(self, images):
        """Should you set-pack this family at all? Returns rows (name, bytes, psnr) for raw, per-file PNG,
        gzip-the-PNGs, the set packer, and (only if Pillow is installed) per-file JPEG as a lossy reference.

        This exists because the answer is CONTENT-DEPENDENT and the packer loses by 16x on the wrong content. The
        PNG baselines use the engine's own stdlib encoder. See holographic_pack.benchmark."""
        from holographic.misc.holographic_pack import benchmark
        return benchmark(images)

    def refine_where_uncertain(self, coarse, uncertainty, refine_fn, frac=0.25, threshold=None):
        """COARSE-FIRST: run the cheap method everywhere, then pay for the expensive one ONLY where a per-cell
        uncertainty signal is high. `refine_fn(mask)` receives the boolean escalate mask. Returns
        (refined, mask, n_refined); the coarse result survives wherever the mask is False.

        Measured on adaptive anti-aliasing of a hard-edged disk (uncertainty = the coarse render's gradient): 6.2x
        fewer samples than supersampling everywhere, for a 21% RMSE cost. THE CONTROL THAT CLAIM OWES: the same
        budget spent at RANDOM cells leaves RMSE 3x worse -- so it is the SIGNAL that pays, not the budget.

        TWO NECESSARY CONDITIONS, and check both before believing a win. (1) The uncertainty must be CONCENTRATED --
        mind.uncertainty_concentration says so before any work is done; near 0 rules coarse-first out entirely.
        (2) The expensive method must be priced PER CELL: a greedy placement method (matching pursuit) is already
        adaptive, its cost is per primitive, and a mask tells it nothing -- measured 21.0 dB with and without, at
        0.9x the speed. And the trap that follows: a GREEDY coarse pass destroys the concentration its own
        refinement needs (0.416 for a uniform base, 0.106 for a greedy one, same size). Coarse-first wants a cheap,
        uniform, dumb base. See holographic_coarsefirst."""
        from holographic.misc.holographic_coarsefirst import refine_where_uncertain
        return refine_where_uncertain(coarse, uncertainty, refine_fn, frac=frac, threshold=threshold)

    def escalate_mask(self, uncertainty, frac=0.25, threshold=None):
        """WHERE to escalate: the top `frac` of cells by uncertainty (a fixed budget), or everything at or above an
        absolute `threshold`. Conservative -- ties at the cutoff are INCLUDED, because escalating a cell that did
        not need it wastes a little work while missing a hard one reintroduces the error.
        See holographic_coarsefirst.escalate_mask."""
        from holographic.misc.holographic_coarsefirst import escalate_mask
        return escalate_mask(uncertainty, frac=frac, threshold=threshold)

    def uncertainty_concentration(self, uncertainty):
        """The coarse-first GATE, and it costs nothing: how concentrated the uncertainty is (0..1), i.e. the share
        of it carried by the hardest 10% of cells, above what a uniform field would give.

        NECESSARY, NOT SUFFICIENT. Near 0 means coarse-first CANNOT help -- the hard work is everywhere and uniform
        refinement is simpler. High means it MIGHT: that region still owes a measured win against a uniform baseline
        AND a random-mask control at the same budget. See holographic_coarsefirst.concentration."""
        from holographic.misc.holographic_coarsefirst import concentration
        return concentration(uncertainty)

    def gradient_uncertainty(self, field):
        """A cheap, deterministic uncertainty signal for a 2-D field: local gradient magnitude. Where a coarse
        estimate changes fast it is probably under-resolved. Domain code should supply a better signal when it has
        one (a fit residual, a path-trace variance). See holographic_coarsefirst.gradient_uncertainty."""
        from holographic.misc.holographic_coarsefirst import gradient_uncertainty
        return gradient_uncertainty(field)

    def mesh_limit_surface(self, mesh):
        """The Loop LIMIT surface in closed form: where infinite subdivision would put every vertex, plus the EXACT
        limit normal there. Returns (positions, normals), both (nV, 3). O(V), and no subdivision is performed.

        This is `iterate`'s k -> infinity case, applied to the LOCAL Loop operator. The piece of that operator that
        is not shift-invariant is only the centre vertex: the ring-to-ring block is exactly the CIRCULANT of the
        kernel [3/8, 1/8, ..., 1/8], i.e. a bind operator, so `iterate.transfer` (an rfft) diagonalises it for free.
        Mode 0 (eigenvalue 5/8 at every valence) gives the limit position; modes +-1 span the tangent plane, so the
        normal is EXACT rather than an area-weighted approximation -- measured 0.0000 degrees against a 6-times
        subdivided icosphere, where the positions converge 6.0e-4 -> 3.7e-5 -> 2.3e-6 at k = 4/6/8. Warren's beta is
        read off the ring's spectrum, not hard-coded: beta = (1/n)(lambda_0 - lambda_1^2).

        HONEST SCOPE: this is the infinite-k case. A FINITE number of levels on an irregular mesh still needs the
        full Stam evaluation, which is not built -- use mind.mesh_subdivide(mesh, k) for that, at O(4^k).
        See holographic_meshsubdiv.loop_limit."""
        from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_limit
        return loop_limit(mesh)

    def mesh_subdivide(self, mesh, levels=1):
        """Loop-SUBDIVIDE a triangle mesh (holographic_meshsubdiv, FWD-8): refine each triangle into four and
        low-pass smooth with the Loop masks (a C2 limit surface). Two operations braided -- a topological refine
        (an Euler-operator sequence, the new part) and a graph-signal low-pass smooth (the spectral family FWD-4
        also uses). Each level multiplies faces by 4, adds one vertex per edge, preserves chi, and keeps a closed
        mesh a closed manifold; a flat mesh stays flat exactly, a curved one is smoothed toward the Loop limit.
        A non-triangle input is triangulated first (Loop is a triangle scheme). Returns a new Mesh."""
        from holographic.mesh_and_geometry.holographic_meshsubdiv import loop_subdivide
        return loop_subdivide(mesh, levels=levels)

    def mesh_catmull_clark(self, mesh, levels=1, creases=None):
        """CATMULL-CLARK subdivide (holographic_meshsubdiv.catmull_clark) -- THE quad box-modelling subdivision
        surface: every face becomes quads, so a quad cage STAYS all-quad (Loop, mesh_subdivide, triangulates -- the
        wrong smoother for a cage). Classic 1978 masks; boundary edges use the B-spline curve rules; chi preserved;
        a closed manifold stays one. `creases` maps an edge (vi,vj) -> sharpness s>=0 (DeRose 1998 semi-sharp: an
        edge held sharp for s levels then smoothing) -- default None is the pure smooth surface. Build the dict with
        mesh_crease_edges. The verb that turns a blocky cage into the smooth model, holding chosen edges sharp with
        NO extra support loops."""
        from holographic.mesh_and_geometry.holographic_meshsubdiv import catmull_clark as _cc
        return _cc(self._as_mesh(mesh), levels=levels, creases=creases)

    def mesh_crease_edges(self, edges, sharpness=5.0):
        """Build a CREASE map for mesh_catmull_clark from a list of edges: {(vi,vj): sharpness}. `edges` is a list
        of (vi,vj) vertex-index pairs (order-free); `sharpness` is the shared value (float, or a list matching
        edges). A high value (~5) reads as a hard crease; ~1-2 as a soft one. Convenience so you never hand-build
        the dict."""
        if isinstance(sharpness, (int, float)):
            return {(int(a), int(b)): float(sharpness) for (a, b) in edges}
        return {(int(a), int(b)): float(s) for (a, b), s in zip(edges, sharpness)}

    def mesh_auto_crease(self, mesh, threshold_deg=30.0, sharpness=5.0):
        """AUTO-CREASE: build a crease map for mesh_catmull_clark by tagging every edge whose DIHEDRAL angle exceeds
        threshold_deg -- the hard box-cage edges an artist would crease by hand, left off smooth regions. Returns
        {(vi,vj): sharpness} to pass as mesh_catmull_clark(creases=...). So a subdivided box keeps its corners with
        zero hand-tagging. See holographic_meshsubdiv.auto_crease_map."""
        from holographic.mesh_and_geometry.holographic_meshsubdiv import auto_crease_map as _acm
        return _acm(self._as_mesh(mesh), threshold_deg=threshold_deg, sharpness=sharpness)



    def solve_ik(self, joints, target, iters=20, tol=None, stiffness=None, dt=None):
        """Inverse kinematics by FABRIK (holographic_meshik, FWD-10): move a chain of `joints` (n+1, 3) so the
        end-effector reaches `target`, keeping every bone's rest length and the root fixed. Implemented LITERALLY
        through this mind's own `project_onto_constraints` engine -- FABRIK's forward/backward reaching IS a
        Gauss-Seidel sweep of bone-length + endpoint-pin projections, so IK is the iterate-a-projection faculty in
        the kinematic-chain costume. Returns (new_joints (n+1,3), n_sweeps). For an UNREACHABLE target the chain
        fully extends toward it (the honest degenerate outcome). Kept negative: plain FABRIK has no joint-angle
        limits -- a per-joint cone projection would slot into the same sweep but is not shipped here.

        SOFT IK (C3): `stiffness=(hertz, zeta)` + the substep `dt` makes the chain SPRINGY in physical units rather
        than rigid. `stiffness=(inf, zeta)` is BIT-IDENTICAL to the rigid default, so nothing changes unless you ask.
        Measured: the end-effector lags its target by 0.3673 / 0.0336 / 0.0000 at 2 / 8 / 40 Hz, and at a fixed
        physical horizon the dial holds its meaning where `omega` does not (omega=0.30 reaches 0.4253 at 5 sweeps
        and 0.0000 at 80; stiffness=(8, 1.0) reaches 0.0033 and 0.0001)."""
        from holographic.mesh_and_geometry.holographic_meshik import solve_ik as _solve_ik
        return _solve_ik(joints, target, iters=iters, tol=tol, stiffness=stiffness, dt=dt)

    def skin_mesh(self, mesh, transforms, weights):
        """Linear-blend-SKIN a mesh (holographic_meshskin, FWD-9): deform each vertex as the weighted combination
        of what each bone transform would do to it -- v' = sum_b w_b (M_b v), weights a partition of unity. This
        is a SOFT mixture of expert bone-transforms (the soft/dense cousin of this engine's hard/sparse top-1
        `moe.GatedMixture` -- same experts+gating skeleton, different gating regime). `transforms` is (B,4,4),
        `weights` is (V,B). Returns a new Mesh (deformed vertices, faces untouched). A shared rigid transform is
        reproduced exactly. Kept negative: LBS averages matrices not rotations, so a 50/50 twist collapses the
        radius to cos(theta/2) (the candy-wrapper artifact) -- dual-quaternion skinning is the fix, not shipped."""
        from holographic.mesh_and_geometry.holographic_meshskin import skin_mesh as _skin_mesh
        return _skin_mesh(mesh, transforms, weights)

    def rig_from_parts(self, mesh, labels, report, falloff=3.0):
        """M2: assemble a RIG (joint tree + label-aware bound skin weights) from a mesh_parts segmentation --
        COMPOSITION of M9 + skin_bind_weights + part adjacency. Core part roots the tree; each elongated limb
        gets a proximal+distal joint so it can bend; the bind restricts each vertex to its own + parent part
        (MEASURED: 57%%->87%% own-part binding, one-limb pose isolated 11000x on the mantis). Feed weights +
        per-joint transforms to linear_blend_skin to pose. Run mesh_parts on a welded mesh first. Returns a
        rig dict (joints, bones, parent, joint_part, weights, core). See holographic_meshskin.rig_from_parts."""
        from holographic.mesh_and_geometry.holographic_meshskin import rig_from_parts as _rfp
        import numpy as _np
        return _rfp(self._as_mesh(mesh), _np.asarray(labels, int), report, falloff=falloff)

    def skin_bind_weights(self, vertices, bones, falloff=2.0, max_influences=4):
        """AUTO-SKIN BINDING (holographic_meshskin) -- compute per-vertex bone weights from bone anchor points, the
        'bind' step that produces the weights skin_mesh consumes. Inverse-distance falloff to the nearest bones,
        keeping max_influences and renormalizing to a PARTITION OF UNITY (so rigid motion is exact). The
        distance-based auto-bind a rig starts from. Kept negative: ignores the surface (can bind across a thin gap);
        geodesic refinement is future. See holographic_meshskin.skin_bind_weights."""
        from holographic.mesh_and_geometry.holographic_meshskin import skin_bind_weights
        return skin_bind_weights(vertices, bones, falloff=falloff, max_influences=max_influences)

    def blend_pose(self, targets, weights):
        """The forward blendshape/skinning map for STRUCTURES (holographic_blendpose, ARCH-6): a soft weighted blend
        of pose-target structures, normalize(sum_i w_i targets_i) -- FWD-9's soft mixture, one rung up (mixing whole
        structures, not transforms). `targets` is (m,dim), `weights` (m,). Returns the (dim,) pose. Paired with
        solve_pose (the inverse)."""
        from holographic.misc.holographic_blendpose import blend_pose
        return blend_pose(targets, weights)

    def solve_pose(self, targets, goal, iters=400):
        """Inverse kinematics for a blendshape rig of STRUCTURES (holographic_blendpose, ARCH-6): solve the blend
        weights so blend_pose(targets, w) reaches `goal`, by handing two constraints to the SAME
        project_onto_constraints sweeper FWD-10 used for FABRIK -- FIT the goal (least-squares step, FABRIK's reach)
        and stay a VALID CONVEX BLEND (simplex projection, FABRIK's length constraint). The blend weights are the
        'joint angles'. Returns the weights (a valid convex blend). Exact when the goal is a blend of the targets;
        the CLOSEST valid blend otherwise (residual <= any single target). Kept negative: a goal outside the
        targets' convex blend is unreachable -- that needs a richer rig (more targets), not a better solver."""
        from holographic.misc.holographic_blendpose import solve_pose
        return solve_pose(targets, goal, iters=iters)

    def mesh_from_sdf(self, sdf, bounds, res=24, level=0.0, vectorized=False):
        """Extract a MESH from an implicit field (holographic_meshbridge, FWD-11; SDF -> mesh): sample the scalar
        field `sdf` (a callable points(N,3)->values, e.g. `sphere_sdf` or a `metaball_field` of Gaussian splats) on
        a res^3 grid over `bounds`=((x0,y0,z0),(x1,y1,z1)) and extract its `level` isosurface by MARCHING
        TETRAHEDRA -- the isosurface extractor the mesh kernel deliberately lacked, and the bridge that lets the
        engine's implicit/splat representations enter the mesh world. The result is a watertight, outward-oriented
        triangle Mesh (a closed genus-0 field gives chi=2). vectorized=True uses the parallel array-op marcher
        (marching_tetrahedra_vec, the case-table-RAM path) -- geometrically identical, ~6-14x faster at working grid
        sizes (default False keeps the per-cell marcher's exact vertex ordering for backward compatibility). Kept
        negative: resolution is the grid's -- features below the cell size are rounded. Returns a Mesh."""
        from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra, marching_tetrahedra_vec
        values, axes = sample_field(sdf, bounds, res)
        march = marching_tetrahedra_vec if vectorized else marching_tetrahedra
        return march(values, axes, level=level)

    def surface_mesh_stable(self, field, bounds, resolution=40, level=0.0, validate=True, face_type="triangle"):
        """Project a field to a mesh with STABLE vertex identity and a topology guarantee -- the entry point for
        a 3-D modeling app that needs faces/edges/verts that DON'T move on their own. Returns a dict:
          'mesh'     -- the watertight, 2-manifold Mesh.
          'keys'     -- a STABLE per-vertex identity array: keys[v] is the grid edge the vertex sits on, the SAME
                        integer in any extraction at this (resolution, bounds). Track vertices by KEY across
                        edits, NOT by array index -- a local edit renumbers indices (a crossing added/removed
                        shifts everything after it in sorted-key order) but never changes keys, so unchanged
                        geometry keeps its identity and the user sees no phantom movement of distant faces.
          'topology' -- (when validate) the validate_topology() report (manifold edges+verts, watertight, genus).
        `face_type` chooses the face standard projected out (marching tetrahedra emits triangles; quads/ngons are
        a deterministic merge ON TOP, leaving vertices -- and their keys -- untouched):
          'triangle' -- the raw marched triangles (default).
          'quad'     -- quad-DOMINANT: adjacent coplanar triangle pairs merged into convex quads, leftovers stay
                        triangles (triangles_to_quads).
          'ngon'     -- connected coplanar faces merged into single n-gons where the region's boundary is a clean
                        loop (merge_coplanar) -- a flat wall becomes one face, curved areas keep triangles.
        GUARANTEE: exact-corner field samples (the one case that makes marching tetrahedra non-manifold, where a
        vertex would land on a shared grid corner) are nudged a deterministic epsilon off the level so the
        crossing lands on the edge interior instead -- yielding clean 2-manifold output. Keys are tied to the
        grid, so they are stable across EDITS at a fixed resolution, not across resolution changes (a different
        resolution is a different mesh by definition). KEPT HONEST: the face GROUPING (which tris became a quad)
        is itself NOT edit-stable -- faces are a derived view; the vertices are the stable identity."""
        from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
        import numpy as _np
        values, axes = sample_field(field, bounds, resolution)
        scale = float(_np.abs(values).mean()) or 1.0
        exact = (values == level)
        if exact.any():
            values = values.copy()
            values[exact] = level - 1e-9 * scale               # push exact-corner hits just inside -> no bowtie
        mesh, keys = marching_tetrahedra_vec(values, axes, level=level, return_keys=True)
        if face_type == "quad":
            from holographic.mesh_and_geometry.holographic_meshpoly import triangles_to_quads
            mesh = triangles_to_quads(mesh)                    # vertices unchanged -> keys still valid
        elif face_type == "ngon":
            from holographic.mesh_and_geometry.holographic_meshpoly import merge_coplanar
            mesh = merge_coplanar(mesh)
        out = {"mesh": mesh, "keys": keys}
        if validate:
            out["topology"] = mesh.validate_topology()
        return out

    def validate_topology(self, mesh):
        """The full well-formedness report for a mesh (manifold edges AND vertices/bowties, watertight,
        degenerate faces, euler/genus) -- the guarantee a 3-D modeling app gates on before handing a mesh to a
        user. See Mesh.validate_topology."""
        return mesh.validate_topology()

    def mesh_to_sdf(self, mesh, points):
        """Signed distance from a MESH at query points (holographic_meshbridge, FWD-11; mesh -> implicit): the
        unsigned distance to the nearest triangle, signed by the nearest face normal (negative inside). The reverse
        of mesh_from_sdf, completing the mesh<->SDF bridge. `points` is (N,3); returns (N,). Kept negative: the
        nearest-normal sign is exact for convex-ish closed meshes but can mis-sign deep concavities (the magnitude
        is always right; a generalized winding number is the fix, not shipped)."""
        from holographic.mesh_and_geometry.holographic_meshbridge import mesh_to_sdf as _mesh_to_sdf
        return _mesh_to_sdf(mesh, points)

    def sculpt(self, field_fn, kind, p, radius, strength=0.3, **kw):
        """SCULPT a field with a falloff-weighted brush (holographic_sculpt, FS-1) -- a local edit of a field
        FUNCTION (vectorized: P of shape (N,3) -> (N,)) in a ball around point `p`, returning the EDITED field
        function. `kind` is one of inflate / carve / smooth / grab / flatten / pinch (grab needs drag=..., flatten
        needs level=...). Because a surface is carried as a field whose level-set IS the surface, sculpting the field
        and RE-EXTRACTING (marching_tetrahedra) gives resolution-independent topology editing -- the move a fixed mesh
        cannot do. The brush leaves the field BIT-IDENTICAL outside the ball (the falloff is exactly 0 past the
        radius), so the surface changes only where you brushed and the re-extract stays watertight/manifold. Works on
        ANY field, not just the surface SDF: the same operator reshapes the creature's value landscape (reward
        shaping) or a density/memory field -- the radius+falloff the bare `reinforce` lacks. KEPT HONEST: on a DENSE
        field the re-extract is still O(res^3) per stroke -- the narrow-band sparse field (the next FS item) is what
        makes a stroke cost O(brush). Delegates to holographic_sculpt.apply_brush."""
        from holographic.mesh_and_geometry.holographic_sculpt import apply_brush
        return apply_brush(field_fn, kind, p, radius, s=strength, **kw)

    def sparse_field(self, field_fn, bounds, voxel, band, tile=8):
        """Build a NARROW-BAND SPARSE field from a dense field function (holographic_sparsefield, FS-2) -- store,
        edit, and re-extract only the thin shell of voxels around the surface (|f| < band), so a brush stroke touches
        O(brush) voxels instead of O(res^3) and only the dirtied bricks re-mesh. This is what turns FS-1 sculpting
        from batchy into interactive. `field_fn(points (N,3)) -> values (N,)` is the SDF (negative inside), `bounds`
        = (min_corner, max_corner), `voxel` the cell size, `band` the half-width to store, `tile` voxels per brick.
        Returns a SparseField; call .apply_local(delta_fn, p, r) to edit (returns dirty bricks + touched count),
        .reinitialize() to restore the signed-distance property after edits, and .extract_local(dirty_bricks) to
        re-mesh only the dirty region (welded watertight). KEPT HONEST: the per-brick marching is pure Python and
        belongs on the GPU past a few hundred active bricks (holostuff is the authoring brain -- the band bookkeeping
        and SDF numerics -- the per-frame voxel grind is the GPU's muscle); the band must be reinitialized or
        distances drift; topology growth into unseeded interior space needs a re-seed. Delegates to
        holographic_sparsefield.SparseField.from_field."""
        from holographic.misc.holographic_sparsefield import SparseField
        return SparseField.from_field(field_fn, bounds[0], bounds[1], voxel, band, tile=tile)

    def surface_mesh(self, field, bounds=None, resolution=24, level=0.0, pixel_budget=None, distance=1.0,
                     lod_targets=(0.5, 0.25, 0.125), screen_height_px=1080, fov_deg=60.0, cache=False):
        """THE SCULPT LOOP'S RE-EXTRACT STEP (FS-4): turn ANY field representation into the drawable mesh, at the right
        detail for the view -- one entry point composing the parts FS-1..FS-3 shipped. `field` is either a field
        FUNCTION (points(N,3)->values; needs `bounds`=(min,max)) OR a SparseField from sparse_field().

        Without a `pixel_budget`, the full-resolution surface is projected and returned.

        With a `pixel_budget`, LOD is FIELD-NATIVE: rather than projecting the fine mesh and then QEM-DECIMATING it
        (greedy edge collapse -- O(collapses*edges), seconds), the SOURCE FIELD is coarsened (re-marched at a coarser
        grid stride / resolution) and re-projected, and the COARSEST level whose screen-space error at `distance`
        stays under the budget is returned. This is the whole thesis made load-bearing: the mesh is a PROJECTION of
        the field, so a coarser mesh is a coarser field projected -- measured ~thousands x faster than decimating the
        projection, because re-marching is a vectorized pass and the per-level error is read straight from the field
        (the full-res field value at a coarse vertex IS its distance to the true surface -- O(V) field samples, no
        O(V*F) mesh-to-mesh distance, no greedy collapse). The legacy QEM LOD (mesh_lod_chain) remains a separate
        faculty for an IMPORTED mesh that has no field behind it.

        cache=True (SparseField, no budget) uses the brick-mesh WORKING-SET CACHE (the ReflexCache idea: re-mesh only
        the bricks a stroke dirtied). The field's edits must go through apply_local; else call its cache_clear().

        This NAMES THE LOOP as iterate-a-projection -- each sculpt step EDITS the field then RE-PROJECTS to the
        surface, the same shape as the resonator/denoiser/dynamics. sculpt -> surface_mesh -> export_splats is the
        authoring cycle; the field is the source of truth, the mesh and splats are two projections. KEPT HONEST: the
        per-level error is the field-distance at the marched vertices (a true geometric deviation), not a
        silhouette/perceptual metric; the function-field path is dense O(res^3) per level (use the SparseField path
        for the interactive, O(brush) loop). `lod_targets` is retained for signature compatibility; the field-native
        LOD coarsens by RESOLUTION STRIDE, not face fraction."""
        from holographic.misc.holographic_sparsefield import SparseField
        is_sparse = isinstance(field, SparseField)
        if not is_sparse and bounds is None:
            raise ValueError("a field FUNCTION needs bounds=(min_corner, max_corner)")
        # No budget: project the field once, at full resolution.
        if pixel_budget is None:
            if is_sparse:
                return field.extract_cached() if cache else field.extract_local()
            return self.mesh_from_sdf(field, bounds, res=resolution, level=level, vectorized=True)
        # Budget: FIELD-NATIVE LOD -- coarsen the SOURCE and re-project, never decimate the projection.
        if is_sparse:
            chain = field.lod_chain()
        else:
            chain = self._function_lod_chain(field, bounds, resolution, level)
        idx = self.mesh_select_lod(chain, distance, pixel_budget, screen_height_px=screen_height_px, fov_deg=fov_deg)
        return chain[idx].mesh

    def sdf_to_mesh(self, sdf, bounds=None, resolution=48, level=None):
        """FRACTAL / SDF -> MESH, the one-liner (holographic bridge). Marches an SDF OBJECT (from fold_fractal /
        mandelbulb / menger / sphere / any .eval-having field) to a watertight Mesh ready for mesh_to_softbody and the
        whole mesh + simulation pipeline. This exists because two sharp edges trip everyone: (1) an SDF object is NOT
        a bare callable, so surface_mesh's `field=` rejects it -- we wrap `.eval` here; (2) a distance-ESTIMATOR
        fractal (fold_fractal) is ALL-POSITIVE (its DE never dips below 0), so marching at level 0 finds no crossing
        and returns 0 faces SILENTLY -- we detect an all-positive field and offset the iso `level` a hair into the
        solid so it actually meshes. Pass an explicit `level` to override. `bounds` defaults to a symmetric box sized
        to the field (probed), so `m.sdf_to_mesh(m.mandelbulb())` just works. Returns a Mesh.

        KEPT NEGATIVE: an all-positive DE has no true zero surface -- the offset picks a near-surface isocontour, not
        a canonical boundary; the mesh is 'the fractal at iso=eps', which is the honest thing a raymarcher shows too."""
        import numpy as _np
        if not hasattr(sdf, "eval"):
            raise TypeError("sdf_to_mesh expects an SDF object with .eval(P); wrap a bare function yourself")
        if bounds is None:
            # probe a coarse box for the field's support: grow until the corners read clearly outside.
            b = 1.0
            for _ in range(6):
                corners = _np.array([[s * b for s in c] for c in
                                     [(1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1)]], float)
                if _np.min(sdf.eval(corners)) > 0.25 * b:      # corners comfortably outside -> box contains it
                    break
                b *= 1.6
            bounds = ([-b, -b, -b], [b, b, b])
        # auto-detect an all-positive distance estimator and offset the iso level into the solid (the 0-faces fix).
        if level is None:
            rng = _np.random.default_rng(0)
            lo, hi = _np.asarray(bounds[0], float), _np.asarray(bounds[1], float)
            probe = lo + (hi - lo) * rng.uniform(size=(4000, 3))
            dvals = sdf.eval(probe)
            if _np.min(dvals) >= 0.0:                          # never negative -> no zero crossing to march
                level = float(_np.percentile(dvals, 20))       # a near-surface isocontour that DOES cross
            else:
                level = 0.0
        return self.surface_mesh(lambda P: sdf.eval(_np.asarray(P)), bounds=bounds, resolution=resolution, level=level)

    def field_displace(self, mesh, field, amount=0.1, weight=None, invert=False, bias=0.0):
        """Displace a mesh's vertices along their normals by a SCALAR FIELD or SDF sampled at each vertex
        (holographic_autodisplace) -- the general, field-driven modifier. `field` is any .eval-having SDF (mandelbulb,
        fold_fractal) or a bare callable P->values, so a FRACTAL can drive the relief. `weight` is the per-vertex MASK
        (an array or callable in [0,1], e.g. sampled from a texture map) so the detail grows only WHERE THE MAP PAINTS
        it -- the 'per-face mandelbulb modifier from a texture' case. `invert` flips the sign, `bias` recenters.
        Returns a new Mesh. Generalizes auto_displace (which only reads an RGB image) to any field. Kept negative:
        displaces along existing normals (adds relief, no re-topology) -- mesh finely first for deep fractal detail."""
        from holographic.mesh_and_geometry.holographic_autodisplace import field_displace
        return field_displace(mesh, field, amount=amount, weight=weight, invert=invert, bias=bias)

    def _function_lod_chain(self, field, bounds, resolution, level):
        """Field-native LOD for a field FUNCTION: re-sample + march at decreasing resolutions (coarsen the source),
        with each level's error read from the function at the coarse vertices (|f(vertex) - level| = the vertex's
        distance to the true level set, for an SDF). Returns a fine->coarse list of LODLevel for select_lod -- the
        same 'coarsen the source, project, read the error from the field' move as SparseField.lod_chain."""
        from holographic.misc.holographic_lod import LODLevel
        from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
        import numpy as _np
        levels = []
        res = int(resolution)
        for r in (res, res // 2, res // 4):
            if r < 2:
                break
            vals, axes = sample_field(field, bounds, r)
            m = marching_tetrahedra_vec(vals, axes, level=level)
            if m.n_faces == 0:
                continue
            if levels and m.n_faces >= levels[-1].n_faces:
                continue
            if not levels:
                mean_e = max_e = 0.0
            else:
                d = _np.abs(_np.asarray(field(m.vertices), float) - level)    # field deviation at marched vertices
                mean_e, max_e = float(d.mean()), float(d.max())
            levels.append(LODLevel(m, m.n_faces, mean_e, max_e))
        return levels

    def route_representation(self, operation):
        """The routing POLICY (holographic_route, ARCH-7): the representation whose capability set supports
        `operation` -- e.g. booleans ("union"/"intersection"/"difference") route to "sdf" (field min/max), explicit
        "boundary"/"render" to "mesh", soft "blend" to "splat". The decision layer on top of the FWD-11 bridge."""
        from holographic.scene_and_pipeline.holographic_route import representation_for
        return representation_for(operation)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p06_mesh_collapse_edge", "_UnifiedPart06")
    print("holographic_unified_p06_mesh_collapse_edge selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
