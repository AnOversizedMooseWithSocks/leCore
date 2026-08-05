"""Three creatures built entirely with the organics faculties added this arc.

Each one exercises a DIFFERENT part of the new functionality, so the three together are a working
demonstration rather than the same creature three times:

  1. STRIDER   spine editing (extend + belly profile) + bilateral parts + rig-bound paint
  2. URCHIN    radial-5 symmetry, the generalisation of the rig's single mirror plane
  3. SERPENT   a long extended spine with a tapered profile, showing metaball auto-density

Everything here goes through UnifiedMind faculties -- no reaching into module internals -- because a
demo that needs private access is a demo the faculties cannot actually support.
"""

import numpy as np

import lecore
from holographic.rendering.holographic_render import Light

m = lecore.UnifiedMind(dim=512, seed=0)


def build(spec, parts, symmetry, pattern, seed, resolution=104, bone_mix=0.65):
    """Spec -> rig -> metaball skin -> weights -> rig-bound paint. The whole R-1/R-3/R-7/R-9 chain."""
    cr = m.creature(spec, skin=False)
    C, R, bones = m.creature_metaballs(cr, spec, spacing=0.9)
    mesh = m.creature_skin_mesh(cr, spec, spacing=0.9, resolution=resolution)
    V = np.asarray(mesh.vertices, float)

    idx, w, names, _ = m.skin_weights_from_balls(V, C, R, bones, dim=256, seed=seed)
    cols = m.paint_creature(V, idx, w, names, pattern=pattern, pattern_scale=7.0,
                            seed=seed, bone_mix=bone_mix)

    # The holographic part layout: one bound record, queryable by socket.
    lib = m.part_library(dim=1024, seed=seed)
    for name, handles in parts:
        lib.define(name, handles=handles)
    assembly, vec = {}, None
    for name, _h in parts:
        assembly, vec = m.attach_part(assembly, "socket_%s" % name, name, lib,
                                      symmetry=symmetry[0], n=symmetry[1])
    return {"creature": cr, "mesh": mesh, "colours": cols, "balls": len(C),
            "assembly": assembly, "vector": vec, "library": lib,
            "report": m.assembly_report(assembly, lib)}


def render(mesh, cols, path, direction, size=720):
    """Rasterise with vertex colours, FRAMED BY THE SHIPPED fit_camera.

    Hand-picked eye positions gave 5.9% frame coverage on the first attempt -- the creatures are tall
    and thin, so a fixed distance either clips them or leaves them as a speck. fit_camera solves the
    distance exactly against the projected bbox, which is the whole reason it exists.

    LIGHTING, corrected by measurement. The first attempt used ambient=1.0 with no lights, on the
    reasoning that vertex colours must not be double-darkened. That rule applies when the colours are
    baked RADIANCE -- these are ALBEDO (paint), so with no lights there is no shading and every
    creature renders as a flat blob. Measured on the urchin, subject-pixel contrast (std is the form):
        flat  ambient=1.0, no lights ....... mean 0.370  std 0.155
        engine default lights ............. mean 0.092  std 0.090   (too dark)
        key + fill, ambient 0.45 .......... mean 0.515  std 0.236   <- chosen
    A key/fill pair nearly doubles the contrast against flat, which is the 3-D form appearing.
    """
    lights = [Light("directional", direction=(-0.55, 0.6, -0.55), intensity=1.25),
              Light("directional", direction=(0.7, 0.35, -0.2), intensity=0.45)]
    cam = m.fit_camera(mesh, direction=direction, up=(0.0, 0.0, 1.0), fov_deg=40.0,
                       width=size, height=size, margin=1.14)
    img = m.render_mesh(mesh, cam, width=size, height=size, vertex_colors=cols,
                        lights=lights, ambient=0.45, background=(0.07, 0.08, 0.10), smooth=True)
    img = _crop_to_subject(img, (0.07, 0.08, 0.10))
    m.save_render(path, np.clip(img, 0, 1))
    bg = np.array([0.07, 0.08, 0.10])
    cover = float((np.abs(img - bg).sum(-1) > 0.02).mean())
    return img, cover


def _crop_to_subject(img, bg, pad=0.05):
    """Crop to the rendered silhouette plus a margin.

    WHY THIS IS NEEDED ON TOP OF fit_camera: fit_camera correctly fits the whole mesh inside a SQUARE
    frame, but these creatures are tall and thin -- the serpent is 3.7 long and 0.4 wide. Fitting its
    length leaves it occupying ~2% of a square image, which is geometrically correct framing and a
    useless picture. Cropping to the silhouette makes the subject fill its own aspect ratio instead of
    the frame's. Measured, not assumed: coverage is reported per creature after the crop.
    """
    a = np.abs(img - np.asarray(bg)).sum(-1) > 0.02
    if not a.any():
        return img
    ys, xs = np.where(a)
    ph = int(pad * max(int(ys.max() - ys.min()), 1)); pw = int(pad * max(int(xs.max() - xs.min()), 1))
    y0, y1 = max(ys.min() - ph, 0), min(ys.max() + ph + 1, img.shape[0])
    x0, x1 = max(xs.min() - pw, 0), min(xs.max() + pw + 1, img.shape[1])
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------- 1. STRIDER --
# A long-legged quadruped. Spine EXTENDED by two segments, then given a belly with set_spine_radius.
strider = m.quadruped_spec()
strider = m.extend_spine(strider, 2)
strider = m.spine_profile(strider, [0.07, 0.11, 0.15, 0.16, 0.13, 0.09, 0.06])
strider = m.set_spine_radius(strider, 0.45, 0.19, falloff=0.35)
strider["limbs"] = [
    {"at": 0.20, "dir": [0.6, -1.4, -0.2], "segments": 3, "length": 0.85, "radius": 0.045, "mirror": True},
    {"at": 0.78, "dir": [0.6, -1.4, -0.2], "segments": 3, "length": 0.80, "radius": 0.045, "mirror": True},
]
strider["head"] = {"at": 1.0, "radius": 0.15}

# ----------------------------------------------------------------------------- 2. URCHIN --
# A compact radial body: short spine, many stubby limbs, RADIAL-5 part symmetry.
urchin = m.quadruped_spec()
urchin = m.spine_profile(urchin, [0.09, 0.17, 0.21, 0.17, 0.10])
urchin = m.reshape_spine(urchin, length=0.75, curve=0.0)
urchin["limbs"] = [
    # Limb length is set against the TORSO RADIUS, not chosen freely: metaballs merge anything that
    # passes close, so a limb whose tip clears the spine by less than ~3x the torso radius is simply
    # absorbed into the blob (measured: the first draft cleared by 1.6x and the limbs vanished).
    {"at": 0.28, "dir": [1.0, -0.40, 0.60], "segments": 3, "length": 0.78, "radius": 0.042, "mirror": True},
    {"at": 0.50, "dir": [1.0, 0.05, -0.85], "segments": 3, "length": 0.72, "radius": 0.042, "mirror": True},
    {"at": 0.72, "dir": [1.0, 0.50, 0.55], "segments": 3, "length": 0.70, "radius": 0.042, "mirror": True},
]
urchin["head"] = {"at": 1.0, "radius": 0.17}

# ---------------------------------------------------------------------------- 3. SERPENT --
# A long body: spine extended to 12 segments with a tapering profile, tiny limbs near the head.
serpent = m.quadruped_spec()
serpent = m.extend_spine(serpent, 8)
serpent = m.reshape_spine(serpent, curve=0.42)
serpent = m.spine_profile(serpent, [0.05, 0.09, 0.13, 0.15, 0.15, 0.14, 0.12,
                                    0.10, 0.085, 0.07, 0.055, 0.04, 0.028])
serpent["limbs"] = [
    {"at": 0.12, "dir": [0.9, -1.0, 0.1], "segments": 2, "length": 0.34, "radius": 0.032, "mirror": True},
]
serpent["head"] = {"at": 1.0, "radius": 0.13}


CREATURES = [
    ("strider", strider, [("horn", {"length": (0.4, 1.8)}), ("hoof", {"width": (0.3, 1.2)})],
     ("bilateral", 2), "stripes", 1, (1.0, -1.15, 0.45)),
    ("urchin", urchin, [("spike", {"length": (0.3, 2.0)}), ("eye", {"size": (0.5, 1.5)})],
     ("radial", 5), "dots", 2, (1.0, -1.05, 0.55)),
    ("serpent", serpent, [("fang", {"length": (0.5, 1.5)}), ("frill", {"width": (0.4, 1.6)})],
     ("bilateral", 2), "noise", 3, (1.0, -1.25, 0.38)),
]

if __name__ == "__main__":
    out = []
    for name, spec, parts, sym, pattern, seed, direction in CREATURES:
        b = build(spec, parts, sym, pattern, seed)
        path = "/home/claude/out/creature_%s.png" % name
        _img, cover = render(b["mesh"], b["colours"], path, direction)
        nseg = spec["spine"]["segments"]
        rep = b["report"]
        print("%-8s spine=%2d segs  balls=%4d  verts=%5d  sockets=%2d  recall=%.0f%% (margin %.3f)"
              " frame %.0f%%"
              % (name, nseg, b["balls"], len(np.asarray(b["mesh"].vertices)),
                 rep["n_parts"], 100 * rep["accuracy"], rep["min_margin"], 100 * cover))
        out.append((name, b))
