"""
holographic_scene.py -- compositional images: tag the parts, bind them into a
structure, and factor that structure back out with a resonator.

Two ideas drove this file, both from the same conversation:

  1. "We compute DCT coefficients but never use them as features for tagging."
     True -- the DCT lived only inside the compressor.  Here the DCT energy
     distribution becomes a *texture* tag (smooth / horizontal / vertical /
     busy), joining a *colour* tag (from HSV) and a *shape* tag (from geometry)
     to give every image -- or every object inside it -- an automatic label set.

  2. "Take a compositional approach, not a holistic one.  Use resonators."
     The holistic descriptor in holographic_vision averages the whole image into
     one vector, so a red circle beside a blue square blurs into 'purplish, mixed'
     -- the parts are lost.  Here each object is encoded as a *product* of its
     attribute atoms,  obj = colour (x) shape (x) texture,  and a scene is the
     superposition of its objects.  A ResonatorNetwork then *factors* a composite
     vector back into (colour, shape, texture) -- and with explain-away, pulls
     multiple objects out of one scene vector.  That is the thing a holistic
     vector simply cannot do.

Pipeline:  segment -> tag each part (colour/shape/texture) -> bind into roles ->
bundle into a scene -> resonate to recover the parts.  All numpy, building on
holographic_ai (bind/bundle), holographic_reasoning (ResonatorNetwork),
holographic_image (the DCT), and holographic_vision (shape geometry).
"""

import functools
import numpy as np

from holographic_ai import bind, bundle, random_vector, cosine
from holographic_reasoning import ResonatorNetwork
from holographic_image import _dct_matrix
import holographic_vision as hv


# ======================================================================
# vocabularies  --  the discrete tags each factor can take.
# ======================================================================

COLOURS = ["red", "yellow", "green", "cyan", "blue", "magenta", "grey"]
SHAPES = ["circle", "rectangle", "triangle", "line"]
TEXTURES = ["smooth", "horizontal", "vertical", "busy"]


# ======================================================================
# colour tag  --  dominant hue bucket from HSV (or 'grey' if washed out).
# ======================================================================

def colour_tag(rgb, mask=None):
    """Name the dominant hue of an image (or of the pixels under `mask`).
    Pixels that are too dark or too desaturated carry no reliable hue, so if
    too few colourful pixels survive we call it 'grey'."""
    hsv = hv.rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sel = (s >= 0.25) & (v >= 0.20)
    if mask is not None:
        sel &= mask
    if sel.sum() < 4:
        return "grey"
    # 6 hue wedges of 60 deg, red centred on 0; pick the most populated.
    wedge = (((h[sel] + 30) % 360) // 60).astype(int)   # 0=red..5=magenta
    counts = np.bincount(wedge, minlength=6)
    return COLOURS[int(counts.argmax())]


# ======================================================================
# texture tag  --  from the DCT coefficient layout.
# ======================================================================

def _resize_nn(gray, N):
    """Nearest-neighbour resize to N x N (small, dependency-free)."""
    H, W = gray.shape
    yi = (np.arange(N) * H / N).astype(int).clip(0, H - 1)
    xi = (np.arange(N) * W / N).astype(int).clip(0, W - 1)
    return gray[yi][:, xi]


def dct_features(gray, N=32):
    """Return the fractional AC energy of a region's 2-D DCT, split into bands:
        col   energy on the first column   (vertical frequencies -> HORIZONTAL bands)
        row   energy on the first row      (horizontal frequencies -> VERTICAL bands)
        off   everything else              (2-D / diagonal structure)
        low   the three lowest AC coeffs   (how gentle the variation is)
        ac    total AC energy / DC energy  (how much is happening at all)
    This is exactly the coefficient grid the compressor already builds; we just
    read structure out of *where* the energy sits instead of throwing it away."""
    g = _resize_nn(np.asarray(gray, float), N)
    M = _dct_matrix(N)
    C = M @ g @ M.T
    dc = abs(C[0, 0]) + 1e-9
    A = C.copy(); A[0, 0] = 0.0
    Et = float((A ** 2).sum()) + 1e-12
    col = float((C[1:, 0] ** 2).sum()) / Et             # first column
    row = float((C[0, 1:] ** 2).sum()) / Et             # first row
    off = max(0.0, 1.0 - col - row)
    low = float(C[0, 1] ** 2 + C[1, 0] ** 2 + C[1, 1] ** 2) / Et
    return dict(col=col, row=row, off=off, low=low, ac=Et ** 0.5 / dc)


def texture_tag(gray):
    """Classify region texture from its DCT energy layout:
        almost no AC energy, or all of it in the gentlest coefficients -> smooth
        energy mostly off the axes                                      -> busy
        first-column energy dominant                                    -> horizontal
        otherwise (first-row energy dominant)                           -> vertical
    Calibrated on canonical textures (see tests); approximate on arbitrary art."""
    f = dct_features(gray)
    if f["ac"] < 0.05 or f["low"] > 0.6:
        return "smooth"
    if f["off"] > max(f["col"], f["row"]):
        return "busy"
    return "horizontal" if f["col"] > f["row"] else "vertical"


# ======================================================================
# shape tag + segmentation  --  find the objects, label each one.
# ======================================================================

def _background(rgb):
    """Guess the background colour as the median of the four corners."""
    a = hv._as_float(rgb)
    corners = np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]])
    return np.median(corners, axis=0)


def foreground_mask(rgb, thresh=0.18):
    """Boolean mask of pixels that differ from the background colour."""
    a = hv._as_float(rgb)
    bg = _background(rgb)
    return np.sqrt(((a - bg) ** 2).sum(-1)) > thresh


def segment(rgb, min_area=20):
    """Split the foreground into connected components (4-connectivity), via a
    simple flood fill.  Returns a list of boolean masks, largest first --
    turning one picture into a list of objects to tag separately."""
    fg = foreground_mask(rgb)
    H, W = fg.shape
    seen = np.zeros((H, W), bool)
    masks = []
    for sy in range(H):
        for sx in range(W):
            if not fg[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]; comp = np.zeros((H, W), bool)
            while stack:                                # iterative flood fill
                y, x = stack.pop()
                if y < 0 or x < 0 or y >= H or x >= W or seen[y, x] or not fg[y, x]:
                    continue
                seen[y, x] = True; comp[y, x] = True
                stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
            if comp.sum() >= min_area:
                masks.append(comp)
    masks.sort(key=lambda m: -m.sum())
    return masks


def shape_tag(rgb, mask=None):
    """Geometric shape label.  Uses the foreground mask of the whole image, or a
    supplied object mask, and the geometry classifier from holographic_vision."""
    m = mask if mask is not None else foreground_mask(rgb)
    return hv.classify_shape(m)


# ======================================================================
# automatic tagging  --  the whole point of message 1.
# ======================================================================

def auto_tags(rgb, mask=None):
    """Extract a (colour, shape, texture) tag triple from raw pixels -- no
    labels, no training.  Texture comes from the DCT, colour from HSV, shape
    from geometry."""
    g = hv.to_gray(rgb)
    if mask is not None:
        ys, xs = np.nonzero(mask)
        if len(xs):
            g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return {"colour": colour_tag(rgb, mask), "shape": shape_tag(rgb, mask),
            "texture": texture_tag(g)}


def tag_objects(rgb):
    """Segment the image and auto-tag every object.  Returns a list of tag dicts
    (one per connected component) -- the compositional inventory of the scene."""
    return [auto_tags(rgb, mask=m) for m in segment(rgb)]


# ======================================================================
# compositional code  --  bind tags into a structure, resonate them back out.
# ======================================================================

class SceneCoder:
    """Encode images compositionally and recover their parts with a resonator.

    Each factor (colour, shape, texture) gets its own codebook of random atom
    vectors -- one per possible tag.  An object is the *product* (circular
    convolution) of its three atoms; a scene is the *superposition* of its
    objects.  factor() runs the resonator to read the tags back out of a single
    composite vector; factor_scene() adds explain-away to pull several objects
    out of one scene vector.
    """

    def __init__(self, dim=1024, seed=0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.vocab = [COLOURS, SHAPES, TEXTURES]
        # one codebook (n_tags x dim) of unit atoms per factor
        self.codebooks = [np.stack([random_vector(dim, rng) for _ in vocab])
                          for vocab in self.vocab]
        self.resonator = ResonatorNetwork(self.codebooks)

    # ---- encoding ----
    def _atoms(self, tags):
        return [self.codebooks[f][self.vocab[f].index(tags[k])]
                for f, k in enumerate(("colour", "shape", "texture"))]

    def encode(self, tags):
        """One object -> one composite vector (product of its three atoms)."""
        return functools.reduce(bind, self._atoms(tags))

    def encode_scene(self, tag_list):
        """Several objects -> one scene vector.  We keep it as an UNNORMALISED
        superposition (a plain sum), not a unit 'bundle'.  That detail is the
        whole game for multi-object recovery: explain-away subtracts unit-scale
        atoms, so the scene must carry each object at unit scale too.  Normalising
        (the textbook bundle) is exactly what used to cap recovery at a couple of
        objects."""
        return np.sum([self.encode(t) for t in tag_list], axis=0)

    # ---- decoding ----
    def _tags_from_idx(self, idx):
        return {"colour": COLOURS[idx[0]], "shape": SHAPES[idx[1]], "texture": TEXTURES[idx[2]]}

    def _product(self, idx):
        return functools.reduce(bind, [self.codebooks[f][idx[f]] for f in range(3)])

    def _best_factor(self, resid, iters, restarts, rng):
        """Factor once from the all-codebook start, then a few random restarts;
        keep whichever recovered atom-product best explains the residual."""
        best = self.resonator.factor(resid, iters=iters)
        score = cosine(self._product(best), resid)
        for _ in range(restarts):
            idx = self.resonator.factor(resid, iters=iters, init="random", rng=rng)
            s = cosine(self._product(idx), resid)
            if s > score:
                best, score = idx, s
        return best

    def factor(self, composite, iters=60):
        """Recover one object's tags from a composite vector via the resonator."""
        return self._tags_from_idx(self.resonator.factor(composite, iters=iters))

    def factor_scene(self, scene, n_objects, iters=60, restarts=0, sweeps=2, seed=0):
        """Recover n object tag-triples from a scene vector.  We peel objects off
        with explain-away, then run a few coordinate-descent *sweeps*: each object
        is re-factored from the scene minus the OTHER objects, so an early greedy
        mistake can still be corrected.  Sweeps alone reach 100% through ~5
        objects; `restarts` add robustness past that."""
        rng = np.random.default_rng(seed)
        scene = np.asarray(scene, float)
        idxs, resid = [], scene.copy()
        for _ in range(n_objects):                       # greedy peel
            idx = self._best_factor(resid, iters, restarts, rng)
            idxs.append(idx); resid = resid - self._product(idx)
        for _ in range(sweeps):                          # coordinate descent
            for k in range(n_objects):
                others = np.sum([self._product(idxs[j]) for j in range(n_objects) if j != k]
                                or [np.zeros_like(scene)], axis=0)
                idxs[k] = self._best_factor(scene - others, iters, restarts, rng)
        return [self._tags_from_idx(i) for i in idxs]


# ======================================================================
# scene drawing  --  multi-object test images (used by demo, tests, UI).
# ======================================================================

def make_scene(specs, S=96, seed=0, bg=(14, 22, 38)):
    """Draw several coloured shapes into one image.  `specs` is a list of
    (shape, colour) pairs; shapes are placed in a row.  Returns the RGB image."""
    palette = {"red": (235, 70, 70), "yellow": (235, 205, 60), "green": (70, 200, 110),
               "cyan": (60, 200, 210), "blue": (80, 120, 235), "magenta": (210, 80, 200)}
    img = np.empty((S, S, 3), np.uint8); img[:] = np.array(bg, np.uint8)
    n = len(specs); cell = S // n
    ry = (S - cell) // 2                                # centre shapes vertically
    for i, (shape, colour) in enumerate(specs):
        sub, mask = hv.make_shape(shape, cell, seed=seed + i, bg=bg, fg=palette[colour])
        x0 = i * cell
        region = img[ry:ry + cell, x0:x0 + cell]
        region[mask] = sub[mask]                        # paste shape pixels only
    return img


# ======================================================================
# demo
# ======================================================================

def _demo():
    print("holographic_scene -- compositional tagging + resonator factoring")
    print("-" * 64)

    # 1. automatic tags from raw pixels (DCT texture, HSV colour, geometry shape)
    img, _ = hv.make_shape("circle", 64, seed=1, fg=(235, 70, 70))
    print("auto-tags of a red circle :", auto_tags(img))

    # 2. texture tagging straight from the DCT, on canonical textures
    N = 48; yy, xx = np.mgrid[0:N, 0:N] / N
    cases = {"smooth": np.dstack([xx] * 3), "horizontal": np.dstack([0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 5 * yy))] * 3),
             "vertical": np.dstack([0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 5 * xx))] * 3),
             "busy": np.dstack([0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 6 * xx) * np.sin(2 * np.pi * 6 * yy))] * 3)}
    ok = sum(texture_tag(hv.to_gray(im)) == name for name, im in cases.items())
    print(f"DCT texture tagger        : {ok}/4 canonical textures correct")

    # 3. compositional round-trip: tag -> bind -> resonate -> tag
    coder = SceneCoder(dim=1024, seed=0)
    rng = np.random.default_rng(0); good = 0
    for _ in range(200):
        t = {"colour": rng.choice(COLOURS), "shape": rng.choice(SHAPES), "texture": rng.choice(TEXTURES)}
        good += coder.factor(coder.encode(t)) == t
    print(f"resonator round-trip      : {good}/200 single objects recovered exactly")

    # 4. the headline: a 2-object scene that holistic tagging cannot separate
    scene = make_scene([("circle", "red"), ("rectangle", "blue")], S=96, seed=2)
    holistic = colour_tag(scene)
    objs = tag_objects(scene)
    true_tags = [{"colour": "red", "shape": "circle"}, {"colour": "blue", "shape": "rectangle"}]
    sv = coder.encode_scene([{**o} for o in objs]) if objs else None
    recovered = coder.factor_scene(sv, 2) if sv is not None else []
    print(f"\n2-object scene  [red circle | blue rectangle]")
    print(f"  holistic colour tag     : '{holistic}'  (one label for the whole image -- blurred)")
    print(f"  compositional segments  : {[(o['colour'], o['shape']) for o in objs]}")
    print(f"  resonator factors scene : {[(o['colour'], o['shape']) for o in recovered]}")

    # multi-object recovery rate -- the old ceiling was ~50% at 3 objects
    rng = np.random.default_rng(0)
    for n in (3, 4, 5):
        ok = 0
        for _ in range(40):
            ts = [{"colour": str(rng.choice(COLOURS)), "shape": str(rng.choice(SHAPES)),
                   "texture": str(rng.choice(TEXTURES))} for _ in range(n)]
            got = coder.factor_scene(coder.encode_scene(ts), n, sweeps=2)
            key = lambda d: (d["colour"], d["shape"], d["texture"])
            ok += {key(t) for t in ts} == {key(g) for g in got}
        print(f"  {n}-object scene recovery : {ok}/40 = {ok / 40:.0%}")


if __name__ == "__main__":
    _demo()
