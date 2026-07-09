"""holographic_renderchannels.py -- RENDER CHANNELS / AOVs (inverse-rendering IR14).

Let the renderer output selectable, SEPARATE channels -- per data pass, per object, per material property -- each
with its own alpha, for compositing, science, and debugging. With no selection you get the BEAUTY pass (the fully
composed render) exactly as today.

The holographic reading, which is why most of this is exposure not new code: a RENDER CHANNEL IS AN UNBIND, and the
scene is a bundle at every level. The scene is a bundle of object records; an object is a bundle of (geometry,
appearance); a material is a bundle of channel roles. "Separate this out" is the engine's own DECOMPOSE -- unbind the
role you want. The material system already does it (material.channel(name) IS unbind(record, role(name))), and the
G-buffer (normal / depth / position / mask) is already computed by the sphere-trace the renderer runs. So the data /
object / material passes here are a selectable multi-buffer VIEW over operations the engine already performs.

The levels and the pass each gives you:
  * DATA (G-buffer): depth / normal / position / mask -- from one sphere-trace (already computed for shading).
  * OBJECT (Cryptomatte): per-object coverage matte -- the nearest-object id at each hit (unbind by object id).
  * MATERIAL: albedo / roughness / normal / ... -- material.channel(name), a shipped unbind (carries crosstalk).

KEPT NEGATIVES (loud):
  * LIGHTING passes (direct / indirect / diffuse / specular / GI / shadow) are the ONE genuinely-new bit and are NOT
    in this v1: the path tracer integrates over ALL paths and averages, so to split them it must accumulate a
    labelled buffer per contribution DURING the trace (tag each sample by bounce type) -- real renderer work, and
    making them sum exactly to beauty needs care at the MIS / Russian-roulette boundaries. Scoped out, named here.
  * material.channel() recovers a channel PLUS capacity crosstalk -- fine for a matte / debug / visualisation pass,
    but a channel meant for exact re-lighting wants the clean stored field (material.sample), not the recovery.
  * MEMORY: N channels = N buffers -- opt-in per channel via the selection, never all-on. DEEP compositing (many
    depth samples per pixel) is a heavier format, out of scope for v1.
NumPy + stdlib only; deterministic. Default (no selection) is BIT-IDENTICAL to render_sdf.
"""
import numpy as np

from holographic.rendering.holographic_raymarch import render_sdf, sphere_trace, sdf_normal

_GBUFFER = ("depth", "normal", "position", "mask")


def render_channels(sdf, camera, want=None, width=32, height=32, objects=None, **render_kw):
    """Render selectable AOV channels. `want` is a list of channel names (any of 'depth','normal','position','mask');
    `objects`, if given (a list of SDFs whose union is `sdf`), adds per-object coverage mattes 'object:<i>'. The
    'beauty' pass is always included and, when nothing else is requested, is the ONLY output and is bit-identical to
    render_sdf. Returns {name: buffer}; each buffer carries its own alpha (the hit mask / per-object coverage)."""
    beauty = render_sdf(sdf, camera, width=width, height=height, **render_kw)
    out = {"beauty": beauty}
    if not want and objects is None:
        return out                                           # default -> beauty only, unchanged

    # one primary sphere-trace serves every data / object pass (the same trace the renderer shades from)
    eye, dirs = camera.ray_dirs(width, height)
    D = dirs.reshape(-1, 3)
    O = np.broadcast_to(eye, D.shape)
    hit, t, P = sphere_trace(sdf, O, D)

    want = list(want or [])
    if "mask" in want:
        out["mask"] = hit.reshape(height, width).astype(float)         # coverage / alpha
    if "depth" in want:
        out["depth"] = np.where(hit, t, 0.0).reshape(height, width)    # ray distance at the hit (0 = miss)
    if "position" in want:
        out["position"] = P.reshape(height, width, 3)                  # world hit position
    if "normal" in want:
        N = np.zeros((len(D), 3))
        if hit.any():
            N[hit] = sdf_normal(sdf, P[hit])                           # surface normal at the hits
        out["normal"] = N.reshape(height, width, 3)

    if objects is not None:
        # per-object matte (Cryptomatte-style, hard-edged v1): the NEAREST object's id at each hit point.
        dists = np.stack([np.asarray(o.eval(P), float) for o in objects], axis=0)   # (n_obj, M)
        obj_id = np.argmin(dists, axis=0)                              # which object's surface is here
        for i in range(len(objects)):
            out["object:%d" % i] = (hit & (obj_id == i)).reshape(height, width).astype(float)

    return out


def material_channels(material, names=None):
    """The MATERIAL-level passes: recover each requested channel by UNBINDING it from the material record -- the
    literal 'a channel is an unbind'. Returns {name: recovered_field}. These carry capacity crosstalk (fine for a
    debug/matte pass; for exact values use material.sample, the clean stored field)."""
    names = list(names) if names is not None else list(material.channels.keys())
    return {name: material.channel(name) for name in names}


def composites_to_beauty(channels, width=None, height=None):
    """The compositor's invariant for OBJECT mattes: the per-object coverage mattes must tile the frame exactly --
    cover every hit pixel once, with no gaps and no double-counting. Returns the max abs difference between the sum
    of the object mattes and the beauty's own coverage (the 'mask' pass). 0 == they composite back perfectly."""
    mattes = [v for k, v in channels.items() if k.startswith("object:")]
    if not mattes or "mask" not in channels:
        return None
    return float(np.max(np.abs(sum(mattes) - channels["mask"])))


def _selftest():
    """Default is beauty-only and bit-identical to render_sdf; the G-buffer passes are valid (unit normals at hits,
    positive depth at hits, mask matches); per-object mattes composite back to the coverage exactly (no gaps/double
    count); material channels come back by unbind. Deterministic."""
    from holographic.rendering.holographic_render import Camera
    from holographic.mesh_and_geometry.holographic_sdf import box, sphere

    sdf = box(1.0, 0.7, 0.5)
    cam = Camera(eye=(2.5, 1.8, 2.5), target=(0, 0, 0), fov_deg=50.0)
    rkw = dict(width=40, height=40, ao=False, shadows=False, reflect=0.0)

    # (1) DEFAULT = beauty only, bit-identical to render_sdf
    only = render_channels(sdf, cam, **rkw)
    assert set(only) == {"beauty"}
    assert np.array_equal(only["beauty"], render_sdf(sdf, cam, **rkw))

    # (2) G-buffer passes are valid
    ch = render_channels(sdf, cam, want=["depth", "normal", "position", "mask"], **rkw)
    hit = ch["mask"] > 0.5
    assert ch["depth"].shape == (40, 40) and np.all(ch["depth"][hit] > 0)          # depth positive at hits
    assert ch["normal"].shape == (40, 40, 3)
    lens = np.linalg.norm(ch["normal"][hit], axis=-1)
    assert np.allclose(lens, 1.0, atol=1e-4)                                       # unit normals at hits
    assert ch["position"].shape == (40, 40, 3)

    # (3) OBJECT mattes composite back to the coverage exactly (two objects side by side)
    objs = [box(0.6, 0.6, 0.6).translate((-0.9, 0, 0)), sphere(0.7).translate((0.9, 0, 0))]
    union = objs[0].union(objs[1])
    chm = render_channels(union, cam, want=["mask"], objects=objs, **rkw)
    assert "object:0" in chm and "object:1" in chm
    err = composites_to_beauty(chm)
    assert err == 0.0                                                              # mattes tile the frame exactly
    # and no pixel is double-counted (the two mattes are disjoint)
    assert np.all(chm["object:0"] * chm["object:1"] == 0.0)

    # (4) MATERIAL channels come back by unbind, with high cosine to the stored directions
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    from holographic.materials_and_texture.holographic_material import Material, texture_field
    from holographic.agents_and_reasoning.holographic_ai import cosine
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=1)
    grid = [(u, v) for u in np.linspace(0.05, 0.95, 9) for v in np.linspace(0.05, 0.95, 9)]
    mat = Material(enc, {"albedo": texture_field(enc, grid, [0.6] * len(grid)),
                         "roughness": texture_field(enc, grid, [0.3] * len(grid))})
    mch = material_channels(mat)
    assert set(mch) == {"albedo", "roughness"}
    assert cosine(mch["albedo"], Material._unit(mat.channels["albedo"])) > 0.4     # recovered by unbind (+ crosstalk)

    # (5) deterministic
    assert np.array_equal(render_channels(sdf, cam, want=["depth"], **rkw)["depth"],
                          render_channels(sdf, cam, want=["depth"], **rkw)["depth"])

    print("holographic_renderchannels selftest OK: default is beauty-only and bit-identical to render_sdf; the "
          "G-buffer passes are valid (unit normals + positive depth at hits); per-object mattes composite back to "
          "the coverage EXACTLY (err %.1f, disjoint -- no gaps/double-count); material channels come back by unbind. "
          "A channel is an unbind; the scene is a bundle at every level. Deterministic" % err)


if __name__ == "__main__":
    _selftest()
