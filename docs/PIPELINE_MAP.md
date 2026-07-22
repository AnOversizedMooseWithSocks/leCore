# leCore Pipeline Map

*The workflow graph, auto-derived by `pipelinemap.py` from the catalog's `consumes`/`produces` tags. Nodes are io-kinds; an edge means some capability turns the source kind into the target kind. This is a VIEW of the live tags -- to change it, tag capabilities, not this file.*

> **Coverage: 110 of 2289 capabilities carry io-kind tags (4%).** The graph below is that tagged subset. Untagged capabilities are real but do not yet declare a typed edge -- backfilling tags grows the map.

```mermaid
graph LR
    curve["curve"] -->|sweep_tube| mesh["mesh"]
    curve["curve"] -->|Curve-curve intersection| selection["selection"]
    field["field"] -->|Denoise multi-way data (low-rank tensor prior) +9| field["field"]
    field["field"] -->|grid_to_hypervector| hypervector["hypervector"]
    field["field"] -->|occupancy_to_mesh| mesh["mesh"]
    field["field"] -->|Aharonov-Bohm ring (magnetic flux phase) +2| scalar["scalar"]
    field["field"] -->|near_surface_to_sdf| sdf["sdf"]
    hypervector["hypervector"] -->|hypervector_to_grid| field["field"]
    hypervector["hypervector"] -->|Recursive factoring (past the resonator's cliff)| hypervector["hypervector"]
    hypervector["hypervector"] -->|Rate-distortion report (bits per vector at a fidelity)| scalar["scalar"]
    hypervector["hypervector"] -->|tree.HoloForest| selection["selection"]
    image["image"] -->|Denoise multi-way data (low-rank tensor prior) +3| image["image"]
    image["image"] -->|depth_to_mesh +3| mesh["mesh"]
    mesh["mesh"] -->|Voxelization +2| field["field"]
    mesh["mesh"] -->|JSON-drivable objects (mesh/camera coercion) +1| image["image"]
    mesh["mesh"] -->|Make a mesh manifold (split non-manifold vertices) +19| mesh["mesh"]
    mesh["mesh"] -->|mesh_pack_uv| points["points"]
    mesh["mesh"] -->|mesh_sample_field +3| scalar["scalar"]
    mesh["mesh"] -->|mesh_auto_seam +7| selection["selection"]
    mesh["mesh"] -->|pivot_point| transform["transform"]
    points["points"] -->|points_to_mesh| mesh["mesh"]
    points["points"] -->|N-body gravity simulation +4| points["points"]
    points["points"] -->|ambient_occlusion +5| scalar["scalar"]
    points["points"] -->|fit_pose +1| sdf["sdf"]
    points["points"] -->|spatial.knn| selection["selection"]
    scalar["scalar"] -->|Honesty & measurement +1| scalar["scalar"]
    scalar["scalar"] -->|Identify an element by its properties +1| selection["selection"]
    sdf["sdf"] -->|Voxelization +1| field["field"]
    sdf["sdf"] -->|mesh_from_sdf +1| mesh["mesh"]
    sdf["sdf"] -->|collide_sdf +1| points["points"]
    sdf["sdf"] -->|Dialect emitters (WGSL / C / JS / Zig from the Python kernel) +3| scalar["scalar"]
    sdf["sdf"] -->|domain_bend +3| sdf["sdf"]
    sdf["sdf"] -->|sdf_scene| sdf_scene["sdf_scene"]
    sdf_scene["sdf_scene"] -->|Rendering (path trace) +1| image["image"]
    selection["selection"] -->|transform_selection| mesh["mesh"]
    selection["selection"] -->|soft_selection_weights| scalar["scalar"]
    selection["selection"] -->|select_edge_loop +2| selection["selection"]
    selection["selection"] -->|pivot_point| transform["transform"]
    skeleton["skeleton"] -->|skin_mesh +1| mesh["mesh"]
    skeleton["skeleton"] -->|skin_bind_weights| scalar["scalar"]
    spectrum["spectrum"] -->|Mantis-shrimp vision (12-band + polarization) +1| image["image"]
    spectrum["spectrum"] -->|Optical elements (Mueller matrices) +2| spectrum["spectrum"]
    timeseries["timeseries"] -->|Doppler velocity & drift acceleration +1| scalar["scalar"]
    timeseries["timeseries"] -->|phase_fold| timeseries["timeseries"]
    timeseries["timeseries"] -->|identify_dynamics| transform["transform"]
    transform["transform"] -->|transform_selection| mesh["mesh"]
    transform["transform"] -->|snap_transform_delta| transform["transform"]
```

## By io-kind

### `mesh`
- **produced by:** Make a mesh manifold (split non-manifold vertices), Mesh editing (DCC), Mesh repair (weld + split non-manifold + fill + compact), Route a mesh to its minimal repair (defect-classified), Smooth a bumpy mesh surface (Taubin no-shrink), Split a loaded mesh into per-material submeshes, depth_to_mesh, field_displace, image_to_3d, image_to_mesh, mesh_bevel_vertex, mesh_fill_holes, mesh_from_sdf, mesh_poke, mesh_rip_vertex, mesh_smooth, mesh_split_vertices, mesh_subdivide, mesh_symmetrize, mesh_triangulate, mesh_uv_unwrap, occupancy_to_mesh, photo_to_3d, points_to_mesh, sdf_to_mesh, skin_mesh, skin_skeleton, solidify_mesh, sweep_tube, transform_selection
- **consumed by:** JSON-drivable objects (mesh/camera coercion), Make a mesh manifold (split non-manifold vertices), Mesh editing (DCC), Mesh repair (weld + split non-manifold + fill + compact), Route a mesh to its minimal repair (defect-classified), Smooth a bumpy mesh surface (Taubin no-shrink), Split a loaded mesh into per-material submeshes, Voxelization, field_displace, mesh_auto_seam, mesh_bevel_vertex, mesh_fill_holes, mesh_pack_uv, mesh_poke, mesh_rip_vertex, mesh_sample_field, mesh_selection, mesh_smooth, mesh_split_vertices, mesh_subdivide, mesh_symmetrize, mesh_to_field, mesh_triangulate, mesh_uv_unwrap, pick_mesh, pivot_point, ray_mesh_intersect, render_mesh, select_boundary_loops, select_edge_loop, select_face_ring, select_in_box, select_symmetric, skin_bind_weights, skin_mesh, soft_selection_weights, solidify_mesh, transform_selection, voxelize_mesh

### `points`
- **produced by:** N-body gravity simulation, collide_sdf, emit_from_surface, mesh_pack_uv, snap_to_grid, snap_to_vertices, solve_ik_limited
- **consumed by:** N-body gravity simulation, ambient_occlusion, collide_sdf, fit_pose, fit_primitives, fit_shape, fold_fit, spatial.knn, ifs_fit, mesh_sample_field, points_to_mesh, sample_field, snap_to_grid, snap_to_vertices, solve_ik_limited

### `sdf`
- **produced by:** domain_bend, domain_fold, domain_repeat, domain_twist, fit_pose, fit_primitives, near_surface_to_sdf
- **consumed by:** Dialect emitters (WGSL / C / JS / Zig from the Python kernel), Voxelization, ambient_occlusion, bake_sdf, collide_sdf, domain_bend, domain_fold, domain_repeat, domain_twist, emit_from_surface, mesh_from_sdf, ray_sdf_intersect, sdf_scene, sdf_to_mesh, to_shadertoy

### `sdf_scene`
- **produced by:** sdf_scene
- **consumed by:** Rendering (path trace), render_scene

### `field`
- **produced by:** Denoise multi-way data (low-rank tensor prior), Hydraulic terrain erosion (droplet simulation), Probability current (quantum flow), Schrodinger solver (split-operator TDSE), Simulation (domain), Voxelization, advect_field, bake_sdf, diffuse_field, harmonic_fill, hypervector_to_grid, inpaint, majority_fill, mesh_to_field, voxelize_mesh
- **consumed by:** Aharonov-Bohm ring (magnetic flux phase), Denoise multi-way data (low-rank tensor prior), Hydraulic terrain erosion (droplet simulation), Probability current (quantum flow), Quantum dot / transmission (resonant scatterer), Schrodinger solver (split-operator TDSE), Simulation (domain), advect_field, diffuse_field, grid_to_hypervector, harmonic_fill, inpaint, majority_fill, near_surface_to_sdf, occupancy_to_mesh, sample_field

### `image`
- **produced by:** Denoise multi-way data (low-rank tensor prior), Faraday sky map (telescope as observer), JSON-drivable objects (mesh/camera coercion), Mantis-shrimp vision (12-band + polarization), Observer (spectrum to sensor readings), Rendering (path trace), See what the mantis sees (false colour), archive, render_mesh, render_scene
- **consumed by:** Denoise multi-way data (low-rank tensor prior), Faraday sky map (telescope as observer), See what the mantis sees (false colour), depth_to_mesh, archive, image_to_3d, image_to_mesh, photo_to_3d

### `hypervector`
- **produced by:** Recursive factoring (past the resonator's cliff), grid_to_hypervector
- **consumed by:** Rate-distortion report (bits per vector at a fidelity), Recursive factoring (past the resonator's cliff), tree.HoloForest, hypervector_to_grid

### `transform`
- **produced by:** identify_dynamics, pivot_point, snap_transform_delta
- **consumed by:** snap_transform_delta, transform_selection

### `selection`
- **produced by:** Curve-curve intersection, Identify an element by its properties, Name a contact type (bounce/slide/rest/jam), spatial.knn, tree.HoloForest, mesh_auto_seam, mesh_selection, pick_mesh, select_boundary_loops, select_edge_loop, select_face_ring, select_in_box, select_symmetric
- **consumed by:** pivot_point, select_edge_loop, select_face_ring, select_symmetric, soft_selection_weights, transform_selection

### `scalar`
- **produced by:** Aharonov-Bohm ring (magnetic flux phase), Dialect emitters (WGSL / C / JS / Zig from the Python kernel), Doppler velocity & drift acceleration, Honesty & measurement, Period of a signal (Lomb-Scargle), Quantum dot / transmission (resonant scatterer), Rate-distortion report (bits per vector at a fidelity), ambient_occlusion, fit_shape, fold_fit, ifs_fit, mesh_sample_field, ray_mesh_intersect, ray_sdf_intersect, sample_field, skin_bind_weights, soft_selection_weights, timeline, to_shadertoy
- **consumed by:** Honesty & measurement, Identify an element by its properties, Name a contact type (bounce/slide/rest/jam), timeline

### `curve`
- **produced by:** _(nothing tagged)_
- **consumed by:** Curve-curve intersection, sweep_tube

### `skeleton`
- **produced by:** _(nothing tagged)_
- **consumed by:** skin_bind_weights, skin_mesh, skin_skeleton

### `timeseries`
- **produced by:** phase_fold
- **consumed by:** Doppler velocity & drift acceleration, Period of a signal (Lomb-Scargle), identify_dynamics, phase_fold

### `spectrum`
- **produced by:** Optical elements (Mueller matrices), Polarized light (Stokes state), Rotation-measure synthesis (Faraday depth)
- **consumed by:** Mantis-shrimp vision (12-band + polarization), Observer (spectrum to sensor readings), Optical elements (Mueller matrices), Polarized light (Stokes state), Rotation-measure synthesis (Faraday depth)

## Gaps (the find-a-gap report)

- **dead-end kinds** (produced, nothing tagged consumes them): _none_
- **source-only kinds** (consumed, nothing tagged produces them -- user-supplied or untagged producer): `curve`, `skeleton`
- **untouched kinds** (in the vocabulary, in no tagged edge yet): _none_

