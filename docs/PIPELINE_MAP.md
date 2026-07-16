# leCore Pipeline Map

*The workflow graph, auto-derived by `pipelinemap.py` from the catalog's `consumes`/`produces` tags. Nodes are io-kinds; an edge means some capability turns the source kind into the target kind. This is a VIEW of the live tags -- to change it, tag capabilities, not this file.*

> **Coverage: 71 of 393 capabilities carry io-kind tags (18%).** The graph below is that tagged subset. Untagged capabilities are real but do not yet declare a typed edge -- backfilling tags grows the map.

```mermaid
graph LR
    curve["curve"] -->|Curve-curve intersection| selection["selection"]
    field["field"] -->|Denoise multi-way data (low-rank tensor prior) +3| field["field"]
    field["field"] -->|Denoise multi-way data (low-rank tensor prior)| image["image"]
    field["field"] -->|Aharonov-Bohm ring (magnetic flux phase) +1| scalar["scalar"]
    hypervector["hypervector"] -->|Recursive factoring (past the resonator's cliff)| hypervector["hypervector"]
    hypervector["hypervector"] -->|Rate-distortion report (bits per vector at a fidelity)| scalar["scalar"]
    hypervector["hypervector"] -->|tree.HoloForest| selection["selection"]
    image["image"] -->|Denoise multi-way data (low-rank tensor prior)| field["field"]
    image["image"] -->|Denoise multi-way data (low-rank tensor prior) +3| image["image"]
    mesh["mesh"] -->|Voxelization| field["field"]
    mesh["mesh"] -->|Rendering (path trace)| image["image"]
    mesh["mesh"] -->|Make a mesh manifold (split non-manifold vertices) +14| mesh["mesh"]
    mesh["mesh"] -->|mesh_pack_uv| points["points"]
    mesh["mesh"] -->|ray_mesh_intersect +2| scalar["scalar"]
    mesh["mesh"] -->|mesh_auto_seam +7| selection["selection"]
    mesh["mesh"] -->|pivot_point| transform["transform"]
    points["points"] -->|N-body gravity simulation +3| points["points"]
    points["points"] -->|fit_shape +2| scalar["scalar"]
    points["points"] -->|fit_pose +1| sdf["sdf"]
    points["points"] -->|spatial.knn| selection["selection"]
    scalar["scalar"] -->|Honesty & measurement +1| scalar["scalar"]
    scalar["scalar"] -->|Identify an element by its properties +1| selection["selection"]
    sdf["sdf"] -->|Voxelization| field["field"]
    sdf["sdf"] -->|sdf_to_mesh| mesh["mesh"]
    sdf["sdf"] -->|Dialect emitters (WGSL / C / JS / Zig from the Python kernel) +2| scalar["scalar"]
    sdf["sdf"] -->|sdf_scene| sdf_scene["sdf_scene"]
    sdf_scene["sdf_scene"] -->|Rendering (path trace)| image["image"]
    selection["selection"] -->|transform_selection| mesh["mesh"]
    selection["selection"] -->|soft_selection_weights| scalar["scalar"]
    selection["selection"] -->|select_edge_loop +2| selection["selection"]
    selection["selection"] -->|pivot_point| transform["transform"]
    skeleton["skeleton"] -->|skin_bind_weights| scalar["scalar"]
    spectrum["spectrum"] -->|Mantis-shrimp vision (12-band + polarization) +1| image["image"]
    spectrum["spectrum"] -->|Optical elements (Mueller matrices) +2| spectrum["spectrum"]
    timeseries["timeseries"] -->|Doppler velocity & drift acceleration +1| scalar["scalar"]
    timeseries["timeseries"] -->|identify_dynamics| transform["transform"]
    transform["transform"] -->|transform_selection| mesh["mesh"]
    transform["transform"] -->|snap_transform_delta| transform["transform"]
```

## By io-kind

### `mesh`
- **produced by:** Make a mesh manifold (split non-manifold vertices), Mesh editing (DCC), Mesh repair (weld + split non-manifold + fill + compact), Route a mesh to its minimal repair (defect-classified), Smooth a bumpy mesh surface (Taubin no-shrink), field_displace, mesh_bevel_vertex, mesh_fill_holes, mesh_poke, mesh_rip_vertex, mesh_split_vertices, mesh_symmetrize, mesh_triangulate, sdf_to_mesh, solidify_mesh, transform_selection
- **consumed by:** Make a mesh manifold (split non-manifold vertices), Mesh editing (DCC), Mesh repair (weld + split non-manifold + fill + compact), Rendering (path trace), Route a mesh to its minimal repair (defect-classified), Smooth a bumpy mesh surface (Taubin no-shrink), Voxelization, field_displace, mesh_auto_seam, mesh_bevel_vertex, mesh_fill_holes, mesh_pack_uv, mesh_poke, mesh_rip_vertex, mesh_selection, mesh_split_vertices, mesh_symmetrize, mesh_triangulate, pick_mesh, pivot_point, ray_mesh_intersect, select_boundary_loops, select_edge_loop, select_face_ring, select_in_box, select_symmetric, skin_bind_weights, soft_selection_weights, solidify_mesh, transform_selection

### `points`
- **produced by:** N-body gravity simulation, mesh_pack_uv, snap_to_grid, snap_to_vertices, solve_ik_limited
- **consumed by:** N-body gravity simulation, fit_pose, fit_primitives, fit_shape, fold_fit, spatial.knn, ifs_fit, snap_to_grid, snap_to_vertices, solve_ik_limited

### `sdf`
- **produced by:** fit_pose, fit_primitives
- **consumed by:** Dialect emitters (WGSL / C / JS / Zig from the Python kernel), Voxelization, ray_sdf_intersect, sdf_scene, sdf_to_mesh, to_shadertoy

### `sdf_scene`
- **produced by:** sdf_scene
- **consumed by:** Rendering (path trace)

### `field`
- **produced by:** Denoise multi-way data (low-rank tensor prior), Probability current (quantum flow), Schrodinger solver (split-operator TDSE), Simulation (domain), Voxelization
- **consumed by:** Aharonov-Bohm ring (magnetic flux phase), Denoise multi-way data (low-rank tensor prior), Probability current (quantum flow), Quantum dot / transmission (resonant scatterer), Schrodinger solver (split-operator TDSE), Simulation (domain)

### `image`
- **produced by:** Denoise multi-way data (low-rank tensor prior), Faraday sky map (telescope as observer), Mantis-shrimp vision (12-band + polarization), Observer (spectrum to sensor readings), Rendering (path trace), See what the mantis sees (false colour), archive
- **consumed by:** Denoise multi-way data (low-rank tensor prior), Faraday sky map (telescope as observer), See what the mantis sees (false colour), archive

### `hypervector`
- **produced by:** Recursive factoring (past the resonator's cliff)
- **consumed by:** Rate-distortion report (bits per vector at a fidelity), Recursive factoring (past the resonator's cliff), tree.HoloForest

### `transform`
- **produced by:** identify_dynamics, pivot_point, snap_transform_delta
- **consumed by:** snap_transform_delta, transform_selection

### `selection`
- **produced by:** Curve-curve intersection, Identify an element by its properties, Name a contact type (bounce/slide/rest/jam), spatial.knn, tree.HoloForest, mesh_auto_seam, mesh_selection, pick_mesh, select_boundary_loops, select_edge_loop, select_face_ring, select_in_box, select_symmetric
- **consumed by:** pivot_point, select_edge_loop, select_face_ring, select_symmetric, soft_selection_weights, transform_selection

### `scalar`
- **produced by:** Aharonov-Bohm ring (magnetic flux phase), Dialect emitters (WGSL / C / JS / Zig from the Python kernel), Doppler velocity & drift acceleration, Honesty & measurement, Period of a signal (Lomb-Scargle), Quantum dot / transmission (resonant scatterer), Rate-distortion report (bits per vector at a fidelity), fit_shape, fold_fit, ifs_fit, ray_mesh_intersect, ray_sdf_intersect, skin_bind_weights, soft_selection_weights, timeline, to_shadertoy
- **consumed by:** Honesty & measurement, Identify an element by its properties, Name a contact type (bounce/slide/rest/jam), timeline

### `curve`
- **produced by:** _(nothing tagged)_
- **consumed by:** Curve-curve intersection

### `skeleton`
- **produced by:** _(nothing tagged)_
- **consumed by:** skin_bind_weights

### `timeseries`
- **produced by:** _(nothing tagged)_
- **consumed by:** Doppler velocity & drift acceleration, Period of a signal (Lomb-Scargle), identify_dynamics

### `spectrum`
- **produced by:** Optical elements (Mueller matrices), Polarized light (Stokes state), Rotation-measure synthesis (Faraday depth)
- **consumed by:** Mantis-shrimp vision (12-band + polarization), Observer (spectrum to sensor readings), Optical elements (Mueller matrices), Polarized light (Stokes state), Rotation-measure synthesis (Faraday depth)

## Gaps (the find-a-gap report)

- **dead-end kinds** (produced, nothing tagged consumes them): _none_
- **source-only kinds** (consumed, nothing tagged produces them -- user-supplied or untagged producer): `curve`, `skeleton`, `timeseries`
- **untouched kinds** (in the vocabulary, in no tagged edge yet): _none_

