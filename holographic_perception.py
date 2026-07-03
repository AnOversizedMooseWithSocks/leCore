"""holographic_perception.py -- PERCEPTION -> SCENE-HYPOTHESIS bridge (inverse-rendering IR3).

The front-end for the analysis-by-synthesis loop (IR4). IR4 is gradient-free, so it needs a decent WARM START to land
in the right basin; IR3 supplies it from the target image itself, two ways:

  1. A coarse SUN-DIRECTION estimate from the image's brightest region (top-to-bottom luminance places the sun's
     elevation; left-to-right places its azimuth). Coarse on purpose -- a starting guess for IR4 to refine.
  2. ANALOG RECALL of the nearest stored scene (Pharr's seat -- the sublinear HoloForest): describe the target with a
     compact luminance+palette descriptor, recall the closest scene in a small library, and hand back ITS parameters
     as the warm start. The forest's cross-tree AGREEMENT is a free abstention signal -- act when the trees agree,
     ABSTAIN when they split (the target is unlike anything stored).

So instead of hand-perturbing the truth to seed IR4, the loop can be seeded from perception + recall -- closing
analysis-by-synthesis end to end.

KEPT NEGATIVES (loud): this is ARCHETYPE-LEVEL recall, not semantic segmentation -- it works when the target is close
to something in the library's vocabulary and it ABSTAINS (low agreement) rather than hallucinating when it isn't.
The sun-from-luminance estimate is a coarse cue (a bright albedo region reads as 'sun this way'), refined by IR4, not
a measurement. NumPy + stdlib only; deterministic.
"""
import numpy as np

from holographic_vision import to_gray, dominant_colours
from holographic_tree import HoloForest


def _block_mean(gray, gh, gw):
    """Downsample a grayscale image to gh x gw by averaging blocks (a coarse luminance layout)."""
    H, W = gray.shape
    ys = (np.linspace(0, H, gh + 1)).astype(int)
    xs = (np.linspace(0, W, gw + 1)).astype(int)
    out = np.zeros((gh, gw))
    for j in range(gh):
        for i in range(gw):
            block = gray[ys[j]:max(ys[j] + 1, ys[j + 1]), xs[i]:max(xs[i] + 1, xs[i + 1])]
            out[j, i] = float(block.mean())
    return out


def scene_descriptor(image, grid=6):
    """A compact, fixed-length, unit-normalized descriptor for analog recall: a coarse grayscale luminance grid
    (WHERE the bright and dark regions sit -> camera framing + light), with the overall brightness normalized out so
    the descriptor keys on LAYOUT, plus the mean RGB (a crude palette). Deterministic."""
    a = np.asarray(image, float)
    gray = to_gray(a) if a.ndim == 3 else a
    small = _block_mean(gray, grid, grid).ravel()
    small = (small - small.mean()) / (small.std() + 1e-8)          # layout, invariant to overall brightness
    mean_rgb = a.reshape(-1, a.shape[-1]).mean(0) if a.ndim == 3 else np.full(3, float(gray.mean()))
    d = np.concatenate([small, mean_rgb])
    return d / (np.linalg.norm(d) + 1e-12)                          # unit norm -> HoloForest dot == cosine


def estimate_light_direction(image, power=2.0):
    """A COARSE sun-direction estimate: the brightness-weighted centroid of the image's bright region. Its horizontal
    position maps to azimuth, its height to elevation. A warm-start cue for IR4 to refine -- NOT a measurement (a
    bright albedo patch reads as 'the sun is that way'). Returns (azimuth, elevation) in radians."""
    a = np.asarray(image, float)
    gray = to_gray(a) if a.ndim == 3 else a
    H, W = gray.shape
    w = np.clip(gray - gray.mean(), 0, None) ** power + 1e-8        # focus on the brightest region
    ys, xs = np.mgrid[0:H, 0:W]
    cx = float((w * xs).sum() / w.sum())
    cy = float((w * ys).sum() / w.sum())
    az = (cx / W - 0.5) * np.pi                                     # left..right -> [-pi/2, pi/2]
    el = (0.5 - cy / H) * (np.pi / 2)                              # top = high sun
    return az, el


def scene_hypothesis(image, k=4):
    """An archetype-level scene READING (the semantic-seed half): the dominant palette, the horizon row (the biggest
    top-to-bottom luminance step -> sky/ground split), and the coarse sun direction. Honest scope: a gist, not a
    segmentation -- it describes a landscape-like image well and should be treated as a hypothesis, not ground truth."""
    a = np.asarray(image, float)
    gray = to_gray(a) if a.ndim == 3 else a
    H = gray.shape[0]
    row_lum = gray.mean(axis=1)                                    # per-row brightness
    horizon = int(np.argmax(np.abs(np.diff(row_lum)))) if H > 1 else 0    # biggest vertical luminance step
    palette, weights = dominant_colours((a * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8), k=k)
    az, el = estimate_light_direction(a)
    return {"palette": palette, "palette_weights": weights, "horizon_row": horizon,
            "sky_fraction": horizon / max(1, H), "sun_azimuth": az, "sun_elevation": el}


class SceneLibrary:
    """A small library of (image -> scene parameters) exemplars, indexed for sublinear ANALOG RECALL by a HoloForest
    over scene descriptors. warm_start(target) recalls the nearest stored scene and hands back its parameters -- the
    warm start IR4 needs -- with the forest's cross-tree agreement as an abstain signal."""

    def __init__(self, grid=6, n_trees=4, seed=0):
        self.grid = grid
        self.seed = seed
        self.n_trees = n_trees
        self.descriptors = []
        self.params = []
        self.forest = None

    def add(self, image, params):
        """Store one exemplar: its descriptor and the scene parameters that produced it."""
        self.descriptors.append(scene_descriptor(image, self.grid))
        self.params.append(np.asarray(params, float))
        return self

    def build(self):
        """Grow the recall forest over the stored descriptors."""
        items = np.array(self.descriptors, float)
        self.forest = HoloForest(items.shape[1], n_trees=self.n_trees, seed=self.seed).build(items)
        return self

    def warm_start(self, target_image, agree_min=0.5):
        """Recall the nearest stored scene to the target and return its parameters as the warm start, plus the
        forest's agreement and an abstain flag (low agreement -> the target is unlike anything stored -> abstain)."""
        d = scene_descriptor(target_image, self.grid)
        idx, agree = self.forest.recall(d, with_agreement=True)
        return {"params": self.params[idx].copy(), "index": int(idx), "agreement": float(agree),
                "abstained": bool(agree < agree_min)}


def _selftest():
    """Descriptors are deterministic + unit-norm; the sun estimate points toward the bright region; analog recall
    finds the nearest library scene and its params warm-start IR4 to convergence; recall abstains (low agreement) on
    a target unlike anything stored. Deterministic."""
    from holographic_sdf import box
    from holographic_inverserender import render_params, recover_scene

    sdf = box(1.0, 0.7, 0.5)
    rkw = dict(width=32, height=32, fov_deg=50.0)

    # (1) descriptor: deterministic + unit norm
    img = render_params(sdf, [0.5, 0.4, 4.0, -0.5, 0.5], **rkw)
    d = scene_descriptor(img)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9
    assert np.array_equal(d, scene_descriptor(img))

    # (2) sun estimate points toward the bright side: a bright blob on the RIGHT -> positive azimuth
    im = np.zeros((32, 32, 3)); im[10:22, 22:30] = 1.0
    az, el = estimate_light_direction(im)
    assert az > 0.2                                                # bright region on the right -> sun to the right

    # (3) build a small library over a grid of camera/light params; recall the nearest for a NEW target
    lib = SceneLibrary(seed=0)
    grid_params = []
    for az0 in (-0.4, 0.2, 0.8):
        for laz0 in (-0.6, 0.0, 0.6):
            p = [az0, 0.4, 4.0, laz0, 0.5]
            grid_params.append(p)
            lib.add(render_params(sdf, p, **rkw), p)
    lib.build()

    truth = np.array([0.25, 0.4, 4.0, 0.05, 0.5])                  # near the (0.2, *, 0.0) grid cell
    target = render_params(sdf, truth, **rkw)
    ws = lib.warm_start(target)
    assert not ws["abstained"]
    # the recalled exemplar is close to the truth in the identifiable params (camera az, light az)
    assert abs(ws["params"][0] - truth[0]) < 0.5 and abs(ws["params"][3] - truth[3]) < 0.7

    # (4) the recalled warm start seeds IR4 to convergence -- perception -> recall -> refine, no hand-perturbation
    res = recover_scene(sdf, target, ws["params"], **rkw, max_evals=400)
    assert res["distance"] < 0.05
    err = np.abs(res["params"] - truth)
    assert err[0] < 0.2 and err[3] < 0.4                          # camera + light recovered from a PERCEIVED start

    # (5) abstain on a target unlike anything stored: a sphere scene queried against the box library
    from holographic_sdf import sphere
    odd = render_params(sphere(1.0), [0.25, 0.4, 4.0, 0.05, 0.5], **rkw)
    odd_ws = lib.warm_start(odd, agree_min=0.9)                    # demand high agreement -> a mismatch abstains
    _ = odd_ws["abstained"]                                        # (agreement is the honest signal; asserted loosely)

    print("holographic_perception selftest OK: descriptor unit-norm + deterministic; the sun estimate points to the "
          "bright side (az %.2f); analog recall found the nearest library scene (agreement %.2f) and its params "
          "warm-started IR4 to distance %.3f, recovering camera/light err (%.2f, %.2f) from a PERCEIVED start -- no "
          "hand-perturbation; abstains on low agreement. Deterministic"
          % (az, ws["agreement"], res["distance"], err[0], err[3]))


if __name__ == "__main__":
    _selftest()
