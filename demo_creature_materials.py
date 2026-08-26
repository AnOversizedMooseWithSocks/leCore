"""Beauty renders: the three creatures, each with an organic taxon material and real scale relief.

This is the QUALITY path, not the preview path:
  * the creature is rendered as its metaball DISTANCE field (not a mesh), so there is no tessellation
    to see -- the silhouette is exact at any zoom
  * the skin structure is displaced INTO the surface (`with_relief`), so scales and plates catch the
    light as geometry instead of being painted on
  * every material channel resolves PER HIT through `render_surface`, which makes the pattern a solid
    3-D texture that wraps the body with no UV unwrap and no seams
  * a studio sky gives the wet/chitinous coats something to reflect -- without one a glossy beetle is
    correctly, uselessly black

Taxon assignment matches each body plan to the integument it would plausibly wear, so the three
together show three of the six families rather than one material three times.
"""

import time

import numpy as np

import lecore
from holographic.rendering.holographic_render import Camera

m = lecore.UnifiedMind(dim=512, seed=0)

import demo_creatures as D  # noqa: E402  (the three creature specs, built with the organics faculties)


def studio_sky(dirs):
    """A soft overhead studio gradient. Glossy coats (chitin, mucus) get their character from what
    they REFLECT, so with no sky the render is physically fine and visually dead."""
    dirs = np.atleast_2d(np.asarray(dirs, float))
    t = (0.5 * (np.clip(dirs[:, 2], -1.0, 1.0) + 1.0))[:, None]
    return np.array([0.88, 0.92, 1.00])[None, :] * t + np.array([0.12, 0.13, 0.16])[None, :] * (1 - t)


def beauty(name, spec, taxon, direction, tint=None, amp=0.006, size=420, ambient=0.26,
           structure_strength=1.4, seed=2):
    """Build the creature, dress it in a taxon material, displace the structure, and trace it."""
    cr = m.creature(spec, skin=False)
    field = m.creature_skin_field(cr, spec, spacing=0.9)

    # The body axis is the spine, so the scale rows follow the animal rather than the world.
    nodes = np.array([np.asarray(cr.joints[n], float) for n in cr.spine_nodes])
    axis = nodes[-1] - nodes[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    origin = nodes.mean(0)

    skin = m.creature_material(taxon, axis=tuple(axis), origin=tuple(origin), seed=seed,
                               tint=tint, structure_strength=structure_strength)
    mat = m.creature_surface_material(taxon, axis=tuple(axis), origin=tuple(origin), seed=seed,
                                      tint=tint, structure_strength=structure_strength)
    relief = field.with_relief(skin["structure"], amplitude=amp)

    # Frame from a coarse mesh -- fit_camera needs vertices, and a low-res proxy is enough to solve
    # the distance exactly. The RENDER never touches this mesh.
    proxy = m.creature_skin_mesh(cr, spec, spacing=0.9, resolution=36)
    c = m.fit_camera(proxy, direction=direction, up=(0.0, 0.0, 1.0), fov_deg=38.0,
                     width=size, height=size, margin=1.10)
    cam = Camera(eye=tuple(c["eye"]), target=tuple(c["target"]), up=tuple(c["up"]),
                 fov_deg=c["fov_deg"])

    t0 = time.perf_counter()
    img = np.asarray(m.render_surface(relief, cam, size, size, {0: mat},
                                      light_dir=(0.55, -0.6, 0.55), ambient=ambient,
                                      sky=studio_sky, background=(0.07, 0.08, 0.10)))
    dt = time.perf_counter() - t0
    # THE SUBJECT MASK MUST BE EXACT, not a background-colour threshold. With a `sky`, non-hit pixels
    # are sky*0.5 + background*0.5 -- a GRADIENT, not the flat background -- so "differs from bg"
    # marks every pixel as subject. That silently made the first crop a no-op and folded the sky into
    # the reported mean/std. The sky here is a pure function of ray direction, so the expected
    # miss-colour is computable per pixel and the mask becomes exact rather than approximate.
    eye, dirs = cam.ray_dirs(size, size)
    miss = (np.clip(studio_sky(dirs.reshape(-1, 3)), 0, 1) * 0.5
            + np.asarray((0.07, 0.08, 0.10)) * 0.5).reshape(size, size, 3)
    msk = np.abs(img - miss).sum(-1) > 0.02

    img, msk = _crop(img, msk)
    path = "/home/claude/out/mat_%s.png" % name
    m.save_render(path, np.clip(img, 0, 1))
    g = img.mean(-1)
    mm = msk[1:, :] & msk[:-1, :]
    hf = float(np.abs(np.diff(g, axis=0))[mm].mean())      # structure energy reaching the pixels
    stack = m.anatomy_stack(taxon)
    print("%-8s %-9s %5.0fs  cover %4.1f%%  subject mean %.3f std %.3f  hf %.5f  layers: %s"
          % (name, taxon, dt, 100 * msk.mean(), img[msk].mean(), img[msk].std(), hf,
             " -> ".join(stack["layers"])))
    return path


def _crop(img, mask, pad=0.04):
    """Crop to the silhouette: these bodies are tall and thin, so a square frame is mostly empty.
    Takes the EXACT hit mask rather than re-deriving one from pixel colour."""
    if not mask.any():
        return img, mask
    ys, xs = np.where(mask)
    ph = int(pad * max(int(ys.max() - ys.min()), 1)); pw = int(pad * max(int(xs.max() - xs.min()), 1))
    y0, y1 = max(ys.min() - ph, 0), min(ys.max() + ph + 1, img.shape[0])
    x0, x1 = max(xs.min() - pw, 0), min(xs.max() + pw + 1, img.shape[1])
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


if __name__ == "__main__":
    beauty("strider", D.strider, "reptile", (1.0, -1.15, 0.45), tint=(0.40, 0.48, 0.26), amp=0.005)
    beauty("urchin", D.urchin, "insect", (1.0, -1.05, 0.55), tint=(0.46, 0.34, 0.16), amp=0.007)
    beauty("serpent", D.serpent, "worm", (1.0, -1.25, 0.38), tint=(0.66, 0.44, 0.44), amp=0.006)
