# Poly Studio 1.2.0 → leCore: what was hand-rolled, what the engine now has, what to upstream

Audited `polystudio_standalone.zip` (backend.py 9,230 lines, app.js 5,737) with `find_capability` for each helper.

| Poly Studio | engine faculty | verdict |
|---|---|---|
| `ccrun.py` (C twin of zigrun, measured 4.0x at n=1e5) | `holographic_ccrun` / `c_batch_eval` | already upstreamed; the app should import the engine's and delete its copy |
| `.lews` import/export via `save_container`/`load_container` | `holographic_lews` (this sweep) | works as-is (schema 1); move to `Workspace.put` + `changes_since` for live sharing; use `lecore.mesh`/`lecore.scene` instead of `polystudio.object` |
| `_planar_uv`, `_auto_uv`, `_fix_uv_seams` | `mesh_lscm`, `mesh_uv_unwrap`, `transfer_uv`, `mesh_repair` | engine has conformal unwrap + seam-aware transfer; the app's planar fallback is honest but weaker — call the engine's |
| `_midpoint_refine` | `holographic_meshsubdiv`, `mesh_catmull_clark` | duplicate; delegate |
| `_banded_grid_chunked` | `mesh_to_sdf_grid` (banded, winding_flood, chunked since sweep 148) | duplicate; delegate |
| `_hist_*` undo/redo with branches | `edit_history`, `creature_editor` session undo | overlap; the branch-and-replay behaviour is worth upstreaming into `edit_history` as a flag |
| `_gauss_blur_np`, `_nn_voxel` | image ops exist under `recolor_image`/postfx; nearest-voxel has no direct verb | small; `_nn_voxel` is a candidate for `sample_distance_grid`'s family |
| `_photo_cache_put` (render cache by params) | `margin_cache`, dependency-keyed cache | overlap |
| `/api/agent/tools` from the live url_map | `holographic_service` `/tools` + `/invoke` | the app re-derived the engine's pattern for its own Flask blueprint — could mount the engine's service instead |
| `quality_gate.py` (absolute-threshold render regression: terracing, edge tones, fringe, exactness) | none | **upstream**: a real measurement tool the engine's own render sweeps needed (sweep 151's aliasing would have been caught) |
| `viewport_guard.py` | none (front-end) | app-side; stays |
| `LECORE_CORE_BACKLOG.md` LC-1 `path_trace(region=, base=)` | `render_surface` has `pixel_mask=`/`base=`; `path_trace` does not | open gap, well specified — next |
| LC-2 `/api/photo` on `JobManager` | `holographic_jobs` | open question; fold error measurement is the gate |

The pattern across the table: the app hand-rolled where it did not *find* the engine faculty (UV, subdivision,
grid bake, cache) — a discoverability failure the catalog aliases should absorb — and built two things the engine
did not have (the C runner, since upstreamed; the quality gate, still to upstream).
