"""Screen-space-error level-of-detail policy (holographic_lod).

WHY THIS MODULE EXISTS
----------------------
From the geometry->stack backlog: the QEM decimator (holographic_meshqem) produces a simplified mesh and
surface_deviation measures how far it strayed, but nothing yet DECIDES which simplification to show. This is that
decision -- the standard real-time-rendering level-of-detail policy: pick the coarsest mesh whose error, once
projected to the screen, is below a pixel budget. Far away a coarse mesh is indistinguishable from the full one, so
showing the full one is wasted; up close the error projects large, so detail is kept.

THE REVERSE-THESIS CONNECTION (why this belongs here, not just in a renderer): this is the engine's own
error-budget RESOLUTION SELECTION carried to meshes. coarse_to_fine (holographic_resolution) refines a query only
until an error budget is met; multires_pyramid keeps a signal at several scales; the equidistribution rule places
resolution where it is needed. select_lod is exactly that rule for geometry -- the coarsest level of a decimation
chain that meets a (screen-space) error budget. The budget is in pixels; the chain is QEM's; the principle is the
one the engine already uses for signals and queries.

WHAT IT PROVIDES
  * build_lod_chain(mesh, targets) -- decimate the mesh (via QEM) to a chain of successively coarser levels,
    measuring each level's surface deviation (mean, max) from the ORIGINAL. Returns fine->coarse LODLevel records.
  * screen_space_error(world_error, distance, screen_height_px, fov_rad) -- project a world-space error to screen
    pixels: sse = world_error * screen_height / (2 * distance * tan(fov/2)).
  * select_lod(chain, distance, pixel_threshold, ...) -- the index of the COARSEST level whose max screen-space
    error is still under the pixel threshold (the cheapest mesh that looks right at that distance).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * the chain decimates monotonically: fewer faces at each level, and surface deviation that only grows.
  * screen_space_error falls with distance (same error, farther -> fewer pixels).
  * select_lod picks finer levels up close and coarser ones far away (monotone in distance), and the choice is
    TIGHT -- the next-coarser level would breach the budget.
  * a tighter pixel threshold (or higher screen resolution) forces a finer level.

DETERMINISM (per ISA.md)
  QEM decimation and surface_deviation are deterministic, and the projection is closed-form -- same mesh and same
  viewing parameters give the same chain and the same selection (asserted).

KEPT NEGATIVES (loud)
  * The error driving the policy is GEOMETRIC surface deviation (a Hausdorff-style distance), not a perceptual or
    silhouette metric -- a coarse mesh can be within the pixel budget on average yet show a visible silhouette
    break. The policy is as good as surface_deviation is, which is a distance, not a render.
  * The projection assumes the error subtends the screen the way a frontoparallel segment does; it ignores
    foreshortening and screen position. It is the standard conservative LOD estimate, not a per-pixel bound.
  * The chain inherits QEM's limits (closed meshes, boundary handling) -- this selects among levels, it does not
    improve them.
"""

from collections import namedtuple
import math

import numpy as np

from holographic_meshqem import qem_decimate, surface_deviation, cluster_decimate

# one level of the chain: the mesh, its face count, and how far it strayed from the original (mean, max)
LODLevel = namedtuple("LODLevel", ["mesh", "n_faces", "mean_error", "max_error"])


def build_lod_chain(mesh, targets=(0.5, 0.25, 0.125)):
    """Decimate `mesh` (via QEM) to a chain of coarser levels at the given face-count FRACTIONS of the original,
    measuring each level's surface deviation from the ORIGINAL mesh. Returns a fine->coarse list of LODLevel; the
    first level is the original (zero error)."""
    base_faces = mesh.n_faces
    chain = [LODLevel(mesh, base_faces, 0.0, 0.0)]
    for frac in targets:
        tf = max(1, int(round(base_faces * frac)))
        if tf >= chain[-1].n_faces:
            continue                                       # no coarsening achieved; skip degenerate level
        coarse = qem_decimate(mesh, tf)
        mean_e, max_e = surface_deviation(coarse, mesh)
        chain.append(LODLevel(coarse, coarse.n_faces, float(mean_e), float(max_e)))
    return chain


def build_cluster_lod_chain(mesh, grids=(48, 24, 12)):
    """The PARALLEL counterpart of build_lod_chain, for an IMPORTED mesh with no field behind it: vertex-cluster the
    mesh (cluster_decimate) at decreasing grid resolutions, measuring each level's surface deviation from the
    ORIGINAL. Each level is O(n) vectorized array ops (no greedy edge-collapse search), so the whole chain builds in
    a fraction of the time the QEM chain takes -- the trade is quality and possibly manifoldness (see
    cluster_decimate). Returns a fine->coarse list of LODLevel; the first level is the original (zero error).

    Use build_lod_chain (QEM) when quality matters and the mesh is small; build_cluster_lod_chain when the mesh is
    large and speed matters; or convert the mesh to a field (mesh_to_sdf) and use the FIELD-NATIVE LOD (re-march
    coarser) when you want to leave mesh-land entirely. KEPT NEGATIVE: a level's error is NOT monotonic in grid
    resolution -- cell alignment with the surface matters, so a coarser grid can land representatives closer; the
    chain is monotone in FACE COUNT, not in error. SPEED: the per-level error uses surface_deviation's fast path (the
    vectorized spatial index) -- ~110x faster than the brute scan and exact here, because a decimated mesh's vertices
    are near the original surface; surface_deviation falls back to the exact scan for any out-of-reach vertex."""
    base_faces = mesh.n_faces
    chain = [LODLevel(mesh, base_faces, 0.0, 0.0)]
    for g in grids:
        coarse = cluster_decimate(mesh, g)
        if coarse.n_faces == 0 or coarse.n_faces >= chain[-1].n_faces:
            continue                                       # this grid bought no coarsening; skip
        mean_e, max_e = surface_deviation(coarse, mesh)    # fast=True: grid-culled, near-surface-exact, brute fallback
        chain.append(LODLevel(coarse, coarse.n_faces, float(mean_e), float(max_e)))
    return chain


def screen_space_error(world_error, distance, screen_height_px=1080, fov_rad=math.radians(60.0)):
    """Project a world-space error to screen pixels at a viewing distance. A segment of length `world_error` at
    `distance` subtends world_error/(distance*tan(fov/2)) of the half-view, i.e. that fraction of half the screen
    in pixels. Returns pixels (0 at infinite distance)."""
    if distance <= 0:
        return float("inf")
    return float(world_error * screen_height_px / (2.0 * distance * math.tan(fov_rad / 2.0)))


def select_lod(chain, distance, pixel_threshold, screen_height_px=1080, fov_rad=math.radians(60.0)):
    """Index of the COARSEST level in `chain` whose MAX screen-space error is still under `pixel_threshold` at this
    distance -- the cheapest mesh that looks right. The original (level 0, zero error) always qualifies, so the
    result is well-defined; far away, coarser levels qualify too and the coarsest wins."""
    pick = 0
    for i, lvl in enumerate(chain):
        if screen_space_error(lvl.max_error, distance, screen_height_px, fov_rad) < pixel_threshold:
            pick = i                                       # chain is fine->coarse, so the last qualifier is coarsest
    return pick


# =====================================================================================================
# Self-test -- monotone chain; error falls with distance; selection is distance-monotone and tight.
# =====================================================================================================
def _selftest():
    from holographic_meshsmooth import _icosphere

    full = _icosphere(2)                                   # V66 F128 unit sphere
    chain = build_lod_chain(full, targets=(0.5, 0.25, 0.125))
    assert len(chain) >= 3, "expected several LOD levels"

    # --- monotone: fewer faces and growing deviation down the chain ---
    faces = [lvl.n_faces for lvl in chain]
    maxerr = [lvl.max_error for lvl in chain]
    assert all(faces[i] > faces[i + 1] for i in range(len(faces) - 1)), "face count must strictly decrease"
    assert all(maxerr[i] <= maxerr[i + 1] for i in range(len(maxerr) - 1)), "deviation must only grow"

    # --- screen-space error falls with distance ---
    e = chain[-1].max_error
    assert screen_space_error(e, 5.0) > screen_space_error(e, 50.0), "farther -> fewer pixels"

    # --- distance monotone: never finer as you move away ---
    picks = [select_lod(chain, d, 2.0) for d in (2.0, 5.0, 15.0, 50.0, 200.0)]
    assert all(picks[i] <= picks[i + 1] for i in range(len(picks) - 1)), f"LOD must coarsen with distance: {picks}"
    assert picks[0] == 0 and picks[-1] > 0, "full detail up close, coarser far away"

    # --- the choice is TIGHT: at a distance picking a coarse level, the next-coarser breaches the budget ---
    d, thr = 50.0, 2.0
    p = select_lod(chain, d, thr)
    if p + 1 < len(chain):
        assert screen_space_error(chain[p + 1].max_error, d) >= thr, "the next-coarser level must exceed the budget"

    # --- tighter threshold forces a finer level ---
    coarse_pick = select_lod(chain, 50.0, 10.0)
    fine_pick = select_lod(chain, 50.0, 0.5)
    assert fine_pick <= coarse_pick, "a tighter pixel budget cannot select a coarser level"

    # --- higher screen resolution forces a finer level (more pixels -> more visible error) ---
    assert select_lod(chain, 50.0, 2.0, screen_height_px=4320) <= select_lod(chain, 50.0, 2.0, screen_height_px=540)

    # --- determinism ---
    chain2 = build_lod_chain(full, targets=(0.5, 0.25, 0.125))
    assert [l.n_faces for l in chain2] == faces and [l.max_error for l in chain2] == maxerr
    assert select_lod(chain, 50.0, 2.0) == select_lod(chain2, 50.0, 2.0)

    print(f"holographic_lod selftest: ok (chain F{faces} with max deviation {[round(x, 3) for x in maxerr]}; "
          f"screen error falls with distance; LOD picks by distance {picks} (full up close, F{chain[picks[-1]].n_faces} "
          f"at 200 units); selection tight (next-coarser breaches budget); tighter threshold / higher resolution -> "
          f"finer; deterministic)")


if __name__ == "__main__":
    _selftest()
