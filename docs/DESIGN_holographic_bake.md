# Holographic bake: one correspondence, many channels projected out

## The measurement (not a hunch)
- The low->high **projection** is the cost: ~1.7-1.9s of a ~1.9s bake at size 48 on a 768-face high-poly.
  Everything else (writing a pixel) is free by comparison.
- A second quantity today = a **second full projection sweep of identical hits**. Measured: two normal calls,
  1.91s then 1.74s, bit-identical output -- the engine recomputed every closest-point from scratch.
- The projection `_high_normal_at(p)` already computes the closest point `q`, the barycentric weights `bc`,
  the squared distance `d2`, and the hit face `fi` -- then **discards all of it except the interpolated
  normal.** Normal, displacement, AO seed, and any transferred vertex attribute are all just DIFFERENT READS
  at that one hit. The information is already in hand and thrown away.

## The holographic move (the owner's exact framing: add channels to one pass, project out per operation)
Split the bake into two pieces along the seam the cost already draws:

1. `mesh_correspondence(low, low_uv, high, size, max_distance=None)` -> a **correspondence buffer**: for every
   covered texel, record (hit_point q, barycentric bc, hit_face fi, signed_distance along low normal, valid
   mask). ONE projection sweep. This is the expensive pass, paid ONCE. `max_distance` is the cage the normal
   bake never had (M12's real work) -- and it belongs HERE, because every channel needs it, not just one.

2. Cheap **projectors** that read a channel out of the buffer with NO new projection:
   - `bake_normal(corr, high)` -> interpolate high normal at bc (tangent or world)
   - `bake_displacement(corr)` -> the signed_distance already stored (M12 -- now literally free, a field read)
   - `bake_ao(corr, high, samples)` -> hemisphere rays from the stored hit points
   - `bake_transfer(corr, high, attr)` -> any per-vertex attribute at bc (colour, weights, ids)

`bake_maps(low, low_uv, high, channels=("normal","displacement","ao"))` runs the correspondence ONCE and
projects each requested channel. Asking for 3 maps costs ~1 projection + 3 cheap reads, not 3 projections.

## Why this is reuse, not a new subsystem
- The projection is `transfer_uv`'s closest-point lookup, which bake_normal_map's own docstring says it
  "reuses" -- except it doesn't reuse, it RE-IMPLEMENTS. mesh_correspondence is where that lookup finally
  lives once, and transfer_uv/bake_normal_map/displacement all become projectors over it.
- It's the same shape as the engine's existing `gather_field` instinct (compile the work so ONE operation
  serves many reads) -- transposed: there, many sample points -> one query; here, one point -> many channels.
- DISPLACEMENT STOPS BEING AN ITEM. M12 said "displacement is a parameter, the cage is the real work." Under
  this design the cage is in the correspondence (shared) and displacement is a one-line field read.

## Kept-negative discipline for when it's built
- Must be EXACT vs the current per-channel bakes (the projection is deterministic; a shared buffer changes
  nothing but when the work happens). Pin bit-equality of the normal map: fused == standalone.
- The buffer is memory: size*size * (3+3+1+1+1) floats. For 2048^2 that's ~50MB -- fine, but the projector
  API must stream/ tile if a caller wants one channel and not the buffer (levers 1 and 5: bake-once + tile).
