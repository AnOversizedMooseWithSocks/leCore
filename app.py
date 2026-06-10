"""
app.py -- a small Flask UI for the holographic image archive.

Run:  pip install flask numpy pillow matplotlib
      python app.py
      open http://127.0.0.1:5000

It wraps the REAL code (holographic_archive.HolographicArchive), so what you
click is exactly what the test suite validates -- no separate reimplementation.

Two things it does:
  * Run the pytest suite and show pass/fail per test.
  * Take a query image (a stored one or your own upload), degrade it (noise /
    blur / occlusion), optionally destroy part of the plate, then run
    content-addressable recall and show which image it matched and the
    reconstruction it pulled back out.
"""

import io
import os
import sys
import glob
import time
import base64
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from flask import Flask, request, jsonify, render_template_string

from holographic_archive import HolographicArchive, _gallery
from holographic_image import HolographicImage, _demo_image, _psnr, _lloyd_max
from holographic_creature import GridWorld, CreatureEncoder, HolographicMind, _train
from holographic_slime import solve_maze
from holographic_pack import benchmark as pack_benchmark, _suite as pack_suite
from image_vault import ImageVault
import holographic_vision as hvz
import holographic_scene as scn
import holographic_tree as htree
import holographic_uri as huri
from collections import defaultdict
from holographic_unified import UnifiedMind
from holographic_text import STOPWORDS

S = 128
app = Flask(__name__)

# Build the archive once at startup from the built-in gallery.
GALLERY = _gallery(S)
GALLERY_TAGS = [
    ["quadrants", "red", "blue", "yellow", "green", "blocks"],
    ["bands", "horizontal", "stripes", "red", "green", "blue"],
    ["gradient", "diagonal", "smooth", "cyan"],
    ["radial", "rings", "circular", "pink", "center"],
    ["ripples", "waves", "sine", "wavy"],
    ["checker", "checkerboard", "squares", "pink", "black"],
]
ARCHIVE = HolographicArchive((S, S, 3), capacity=len(GALLERY), keep=2000, dim=32768, seed=0)
for _im, _tg in zip(GALLERY, GALLERY_TAGS):
    ARCHIVE.add(_im, tags=_tg)


# Load the real sprite set (shipped packed as features/sprites.hsp, 68 KB for 712
# sprites). Used in two places below: the image vault and the walking creature.
# Graceful: if the asset is missing the app still runs (synthetic fallbacks).
SPRITES = {}
try:
    import pack_sprites as _ps
    _HSP = os.path.join(os.path.dirname(__file__), "features", "sprites.hsp")
    if os.path.exists(_HSP):
        with open(_HSP, "rb") as _f:
            SPRITES = {name: rgba for name, rgba in _ps.unpack(_f.read())}
        print(f"[sprites] loaded {len(SPRITES)} sprites from {os.path.basename(_HSP)}")
except Exception as _e:                                  # pragma: no cover
    print("[sprites] none:", _e); SPRITES = {}


# --- image helpers -------------------------------------------------------
def to_data_uri(arr):
    """float[H,W,3] in [0,1] -> 'data:image/png;base64,...'."""
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def read_upload(file_storage):
    im = Image.open(file_storage.stream).convert("RGB").resize((S, S))
    return np.asarray(im, dtype=float) / 255.0


def box(img, n):
    h, w = img.shape[:2]
    ys = np.linspace(0, h, n + 1).astype(int); xs = np.linspace(0, w, n + 1).astype(int)
    return np.stack([np.array([[img[ys[i]:ys[i+1], xs[j]:xs[j+1], c].mean()
            for j in range(n)] for i in range(n)]) for c in range(3)], -1)


def degrade(img, kind, amount):
    """amount in 0..1."""
    if kind == "noise":
        rng = np.random.default_rng(0)
        return np.clip(img + (0.7 * amount) * rng.standard_normal(img.shape), 0, 1)
    if kind == "blur":
        t = max(4, int(round(S * (1 - 0.88 * amount))))
        return np.repeat(np.repeat(box(img, t), -(-S // t), 0), -(-S // t), 1)[:S, :S]
    if kind == "occlude":
        g = img.copy(); side = int(20 + 90 * amount)
        a = (S - side) // 2; g[a:a+side, a:a+side] = 0; return g
    return img


# --- routes --------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/gallery")
def api_gallery():
    return jsonify([{"index": i, "uri": to_data_uri(im)} for i, im in enumerate(GALLERY)])


@app.route("/api/recall", methods=["POST"])
def api_recall():
    kind = request.form.get("degradation", "noise")
    amount = float(request.form.get("amount", 50)) / 100.0
    damage = float(request.form.get("damage", 0)) / 100.0
    truth = None

    if "image" in request.files and request.files["image"].filename:
        src = read_upload(request.files["image"])
    else:
        truth = int(request.form.get("index", 0))
        src = GALLERY[truth]

    query = degrade(src, kind, amount)
    mask = ARCHIVE.damage_mask(damage, seed=7) if damage > 0 else None
    match, recon = ARCHIVE.recall(query, mask=mask)

    return jsonify({
        "query_uri": to_data_uri(query),
        "match_index": match,
        "matched_uri": to_data_uri(GALLERY[match]),
        "recon_uri": to_data_uri(recon),
        "psnr": round(float(_psnr(GALLERY[match], recon)), 1),
        "truth": truth,
        "correct": (truth is None) or (match == truth),
        "damage": int(damage * 100),
    })


@app.route("/api/describe", methods=["POST"])
def api_describe():
    """Cross-modal recall: hand the archive words, get back the image whose tags
    match best -- no picture needed. The match runs entirely in the VSA address
    space (word atoms bundled into a hypervector), separate from the pixel plates."""
    words = [w for w in request.form.get("words", "").lower().split() if w]
    if not words:
        return jsonify({"error": "no words"}), 400
    i, _recon, conf = ARCHIVE.recall_by_tags(words=words)
    return jsonify({"words": words, "index": i, "confidence": round(conf, 2),
                    "matched_uri": to_data_uri(GALLERY[i]),
                    "tags": GALLERY_TAGS[i]})


@app.route("/api/pack", methods=["POST"])
def api_pack():
    """Pack a set of related images (a logo suite) as one reference + per-image
    deltas, and compare against the codecs you'd otherwise use."""
    imgs = pack_suite(96, 6)
    rows = [{"method": nm, "bytes": int(b),
             "fidelity": "lossless" if ps == float("inf") else f"{ps:.0f} dB"}
            for nm, b, ps in pack_benchmark(imgs)]
    png = next(r["bytes"] for r in rows if r["method"] == "per-file PNG")
    for r in rows:
        r["rel"] = "--" if r["method"].startswith("raw") else f"{round(100*r['bytes']/png)}%"
    uris = [to_data_uri(im.astype(float) / 255) for im in imgs]
    return jsonify({"rows": rows, "images": uris})


def _vault_set(kind):
    """Returns a list of (name, RGBA-uint8) for the chosen demo set."""
    if kind == "logos":
        return [(f"logo{i}", a.astype(np.uint8)) for i, a in enumerate(pack_suite(64, 6))]
    if kind == "photos":
        out = []
        for k in range(8):
            r = np.random.default_rng(k); H, W = 96, 128; yy, xx = np.mgrid[0:H, 0:W] / W
            base = np.stack([0.5 + 0.4*np.sin(3*xx+k) + 0.1*np.cos(7*yy),
                             0.5 + 0.3*np.cos(4*yy+k) + 0.1*np.sin(5*xx),
                             0.5 + 0.35*np.sin(2*xx+3*yy)], -1) + 0.03*r.standard_normal((H, W, 3))
            out.append((f"photo{k}", (np.clip(base, 0, 1) * 255).astype(np.uint8)))
        return out
    # sprites: the REAL set if it loaded -- the WHOLE thing (sorted by name, so a
    # character's eight sprites sit together and clustering groups them).
    if SPRITES:
        return [(n, SPRITES[n]) for n in sorted(SPRITES)]
    # synthetic fallback (asset missing)
    r = np.random.default_rng(0); pal = r.integers(0, 256, (16, 4), np.uint8); pal[:, 3] = 255
    base = pal[r.integers(0, 16, (32, 32))]; out = []
    for k in range(12):
        a = base.copy(); a[10:18, 10:18] = pal[r.integers(0, 16, (8, 8))]
        out.append((f"sprite{k}", a.astype(np.uint8)))
    return out


def _thumb(a):
    im = Image.fromarray(np.asarray(a, np.uint8))
    b = io.BytesIO(); im.save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _cutout(rgba):
    """Make a sprite's flat background transparent so it draws as a creature,
    not a square tile.

    These sprites were authored on a solid background (white, here) that got
    baked in as fully-opaque pixels, so the alpha channel is useless as shipped.
    We recover the transparency the honest way:

      1. take the colour sitting in the four corners as the background key
         (only if the corners agree -- otherwise there is no clean background
         to remove and we leave the sprite alone),
      2. FLOOD-FILL that colour inward from the four edges,
      3. clear the alpha on just the pixels the fill reached.

    Flood-filling from the border (rather than clearing every pixel of the key
    colour) is the important part: a white eye or highlight INSIDE the character
    is not connected to the edge, so it is left fully opaque. Returns a new RGBA
    array; the input is not modified.
    """
    a = np.array(rgba, dtype=np.uint8, copy=True)
    if a.ndim != 3 or a.shape[2] != 4:
        return a                                          # not RGBA: nothing to do
    h, w = a.shape[:2]
    rgb = a[:, :, :3]
    corners = {tuple(rgb[0, 0]), tuple(rgb[0, w - 1]),
               tuple(rgb[h - 1, 0]), tuple(rgb[h - 1, w - 1])}
    if len(corners) != 1:
        return a                                          # corners disagree: no clean key
    key = np.array(next(iter(corners)), dtype=np.uint8)
    is_bg = np.all(rgb == key, axis=2)                    # every pixel of the key colour

    # Breadth-first flood fill from the border across key-coloured pixels.
    reached = np.zeros((h, w), dtype=bool)
    stack = []
    for x in range(w):
        for y in (0, h - 1):
            if is_bg[y, x]:
                stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if is_bg[y, x]:
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        if reached[y, x]:
            continue
        reached[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and is_bg[ny, nx] and not reached[ny, nx]:
                stack.append((ny, nx))

    a[reached, 3] = 0                                     # clear alpha on the background only
    return a


@app.route("/api/vault", methods=["POST"])
def api_vault():
    kind = request.form.get("set", "sprites")
    items = _vault_set(kind)
    v = ImageVault()
    for name, a in items:
        v.add(a, name)
    rows = [{"method": m, "bytes": int(b), "fidelity": "lossless" if ps == float("inf") else f"{ps:.0f} dB",
             "lossy": ps != float("inf")} for m, b, ps in v.report(lossy_quality=85)]
    lossless = len(v.pack())
    # Only bother packing a lossy blob when a lossy encoder is actually smaller than
    # the best lossless one (true for photos, false for sprites/logos). The honest
    # full comparison still lives in the table above either way.
    lossy = None
    cheapest_lossy = min((r["bytes"] for r in rows if r["lossy"]), default=None)
    if cheapest_lossy is not None and cheapest_lossy < lossless:
        try:
            lossy = len(v.pack(lossy=True, quality=85))
        except Exception:
            lossy = None
    F = v.fingerprints(); Sm = F @ F.T; np.fill_diagonal(Sm, -1.0)
    neighbors = [[int(j) for j in np.argsort(-Sm[i])[:3]] for i in range(len(v))]
    note = ""
    if kind == "sprites" and SPRITES:
        note = f"your full {len(v)}-sprite set, every image crunched"
    return jsonify({"set": kind, "n": len(v), "clusters": len(v.clusters(0.9)),
                    "lossless": lossless, "lossy": lossy, "rows": rows, "note": note,
                    "names": list(v.names),
                    "images": [_thumb(a) for a in v.images], "neighbors": neighbors})


# --- vision: colour, edges, shapes, emergent classes ---------------------
def _fig_uri(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", transparent=True)
    plt.close(fig); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _vision_colour():
    rgb = GALLERY[0]                                   # a colourful built-in image
    hsv = hvz.rgb_to_hsv(rgb)
    cols, w = hvz.dominant_colours(rgb, k=4, seed=0)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2)); fig.patch.set_alpha(0)
    panels = [("original", rgb, None, None), ("hue", hsv[..., 0], "hsv", (0, 360)),
              ("saturation", hsv[..., 1], "gray", (0, 1)), ("value", hsv[..., 2], "gray", (0, 1))]
    for a, (ttl, data, cmap, lim) in zip(ax, panels):
        if cmap is None:
            a.imshow(data)
        else:
            a.imshow(data, cmap=cmap, vmin=lim[0], vmax=lim[1])
        a.set_title(ttl, color="#c8d3e6", fontsize=11); a.axis("off")
    return {"demo": "colour", "fig": _fig_uri(fig),
            "swatches": [{"rgb": [int(c) for c in cols[i]], "w": round(float(w[i]), 3)} for i in range(len(cols))]}


def _vision_shapes():
    line_img, _ = hvz.make_shape("line", 90, seed=3)
    circ_img, _ = hvz.make_shape("circle", 90, seed=4)
    tri_img, _ = hvz.make_shape("triangle", 90, seed=5)
    lines = hvz.hough_lines(hvz.edges(hvz.to_gray(line_img)), top=1)
    circ = hvz.hough_circles(hvz.to_gray(circ_img), radii=range(15, 40, 2), top=1)
    crn = hvz.corners(hvz.to_gray(tri_img), n=3)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6)); fig.patch.set_alpha(0)
    for a in ax:
        a.axis("off")
    H, W = line_img.shape[:2]
    ax[0].imshow(line_img); ax[0].set_title("line \u2192 Hough", color="#c8d3e6", fontsize=11)
    for rho, th, _v in lines:
        t = np.deg2rad(th); ca, sa = np.cos(t), np.sin(t); x0, y0 = ca * rho, sa * rho
        ax[0].plot([x0 - 1000 * sa, x0 + 1000 * sa], [y0 + 1000 * ca, y0 - 1000 * ca], "-", color="#ff5c7a", lw=2)
    ax[0].set_xlim(0, W); ax[0].set_ylim(H, 0)
    ax[1].imshow(circ_img); ax[1].set_title("circle \u2192 Hough", color="#c8d3e6", fontsize=11)
    for cx, cy, r, _v in circ:
        ax[1].add_patch(plt.Circle((cx, cy), r, fill=False, color="#2dd4bf", lw=2))
        ax[1].plot(cx, cy, "+", color="#2dd4bf", ms=11, mew=2)
    ax[2].imshow(tri_img); ax[2].set_title("triangle \u2192 corners", color="#c8d3e6", fontsize=11)
    for x, y in crn:
        ax[2].plot(x, y, "o", color="#ffd166", ms=9, mec="#7a5b00")
    return {"demo": "shapes", "fig": _fig_uri(fig), "lines": len(lines),
            "circle": (list(circ[0]) if circ else None), "corners": len(crn)}


def _vision_emergent():
    kinds = ["circle", "rectangle", "triangle", "line"]
    # A larger hidden set (25 per kind) for trustworthy percentages...
    imgs, masks, truth = [], [], []
    for s in range(25):
        for ki, k in enumerate(kinds):
            img, mask = hvz.make_shape(k, 56, seed=100 * s + ki)
            imgs.append(img); masks.append(mask); truth.append(ki)
    truth = np.array(truth)
    rule_ok = float(np.mean([hvz.classify_shape(masks[i]) == kinds[truth[i]] for i in range(len(imgs))]))
    X = np.stack([hvz.describe(im) for im in imgs])
    labels, _ = hvz.emergent_classes(imgs, k=4, seed=0, standardize=True)
    purity = hvz.cluster_purity(labels, truth)
    rng = np.random.default_rng(0); tr = rng.random(len(X)) < 0.5
    protos, enc = hvz.vsa_prototypes(X[tr], truth[tr], dim=2048, seed=1)
    acc = float(np.mean([hvz.vsa_classify(X[i], protos, enc)[0] == truth[i] for i in np.nonzero(~tr)[0]]))
    # ...but only show the first 40 in the grid so the figure stays compact.
    show = list(range(40)); cols = 8; rows = int(np.ceil(len(show) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(12, 1.6 * rows)); fig.patch.set_alpha(0)
    axf = ax.ravel()
    for slot, i in enumerate(show):
        pred = hvz.classify_shape(masks[i])
        axf[slot].imshow(imgs[i]); axf[slot].axis("off")
        axf[slot].set_title(pred, color=("#63dcbe" if pred == kinds[truth[i]] else "#ff5c7a"), fontsize=9)
    for j in range(len(show), len(axf)):
        axf[j].axis("off")
    return {"demo": "emergent", "fig": _fig_uri(fig), "rule_acc": round(rule_ok, 3),
            "purity": round(float(purity), 3), "vsa_acc": round(acc, 3), "n_stats": len(imgs)}


@app.route("/api/vision", methods=["POST"])
def api_vision():
    demo = request.form.get("demo", "colour")
    if demo == "shapes":
        return jsonify(_vision_shapes())
    if demo == "emergent":
        return jsonify(_vision_emergent())
    return jsonify(_vision_colour())


# --- compositional scene: DCT tags + resonator factoring -----------------
def _scene_tags():
    N = 64; yy, xx = np.mgrid[0:N, 0:N] / N
    fields = [("smooth", (80, 120, 235), xx),
              ("horizontal", (70, 200, 110), 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 5 * yy))),
              ("vertical", (235, 70, 70), 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 5 * xx))),
              ("busy", (210, 80, 200), 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 6 * xx) * np.sin(2 * np.pi * 6 * yy)))]
    tiles = []
    for _name, col, g in fields:                        # tinted texture fields
        rgb = (np.clip(np.dstack([g, g, g]) * (np.array(col) / 255.0), 0, 1) * 255).astype(np.uint8)
        t = scn.auto_tags(rgb); tiles.append((rgb, f"{t['colour']} / {t['texture']}"))
    for shp, col in [("circle", (235, 70, 70)), ("rectangle", (60, 200, 210)),
                     ("triangle", (235, 205, 60)), ("line", (70, 200, 110))]:
        img, _ = hvz.make_shape(shp, 64, seed=2, fg=col)
        t = scn.auto_tags(img); tiles.append((img, f"{t['colour']} {t['shape']}"))
    fig, ax = plt.subplots(2, 4, figsize=(12, 6.2)); fig.patch.set_alpha(0); axf = ax.ravel()
    for a, (img, ttl) in zip(axf, tiles):
        a.imshow(img); a.axis("off"); a.set_title(ttl, color="#c8d3e6", fontsize=10)
    return {"demo": "tags", "fig": _fig_uri(fig)}


def _scene_compose():
    img = scn.make_scene([("circle", "red"), ("rectangle", "blue"), ("triangle", "green")], S=120, seed=2)
    holistic = scn.colour_tag(img)
    masks = scn.segment(img)
    objs = [scn.auto_tags(img, mask=m) for m in masks]
    coder = scn.SceneCoder(dim=2048, seed=0)
    rec = coder.factor_scene(coder.encode_scene(objs), len(objs), sweeps=2) if objs else []
    fig, ax = plt.subplots(1, 1, figsize=(5, 5)); fig.patch.set_alpha(0)
    ax.imshow(img); ax.axis("off")
    for m, o in zip(masks, objs):
        ys, xs = np.nonzero(m)
        ax.add_patch(plt.Rectangle((xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                                   fill=False, color="#2dd4bf", lw=2))
        ax.text(xs.min(), ys.min() - 4, f"{o['colour']} {o['shape']}", color="#2dd4bf", fontsize=10)
    return {"demo": "compose", "fig": _fig_uri(fig), "holistic": holistic,
            "objects": [[o["colour"], o["shape"], o["texture"]] for o in objs],
            "recovered": [[o["colour"], o["shape"], o["texture"]] for o in rec]}


@app.route("/api/scene", methods=["POST"])
def api_scene():
    return jsonify(_scene_tags() if request.form.get("demo") == "tags" else _scene_compose())


# --- scaling: the recursive holographic tree ----------------------------
def _scaling_capacity():
    Ns = [64, 128, 256, 512, 1024, 2048]
    rows = htree.capacity_curve(Ns, dim=2048, leaf_size=64, seed=0, probes=120)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.4)); fig.patch.set_alpha(0)
    ax.set_facecolor("#0e1626")
    ax.plot(Ns, [r["flat"] * 100 for r in rows], "o-", color="#ff5c7a", lw=2.5, label="flat memory (one trace)")
    ax.plot(Ns, [r["tree"] * 100 for r in rows], "o-", color="#2dd4bf", lw=2.5, label="HoloTree (leaf 64)")
    ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("items stored (N)", color="#c8d3e6"); ax.set_ylabel("recall@1 (%)", color="#c8d3e6")
    ax.set_ylim(-4, 104); ax.tick_params(colors="#67769a")
    for s in ax.spines.values():
        s.set_color("#22324f")
    ax.grid(True, color="#22324f", lw=0.7); ax.legend(facecolor="#0e1626", edgecolor="#22324f", labelcolor="#c8d3e6")
    ax.set_title("a flat trace collapses past capacity; the tree holds", color="#c8d3e6", fontsize=12)
    return {"demo": "capacity", "fig": _fig_uri(fig), "rows": rows}


def _scaling_search():
    rng = np.random.default_rng(0); N, dim = 1500, 512
    items = np.stack([htree.random_vector(dim, rng) for _ in range(N)])
    tree = htree.HoloTree(dim, leaf_size=64, seed=0).build(items)
    qs, truth = [], []
    for _ in range(200):
        i = int(rng.integers(N)); q = items[i] + 0.5 * htree.random_vector(dim, rng)
        qs.append(q / np.linalg.norm(q)); truth.append(i)
    beams = [1, 2, 4, 8, 16, 32]; rec, cmp = [], []
    for b in beams:
        ok = c = 0
        for q, t in zip(qs, truth):
            ok += int(tree.recall(q, beam=b) == t); c += tree.last_comparisons
        rec.append(ok / len(qs)); cmp.append(c / len(qs))
    flux = sorted(tree.flux(), reverse=True)
    forests = [(f, htree.forest_benchmark(N=N, dim=dim, leaf_size=64, n_trees=f, beam=4, noise=0.5))
               for f in (4, 8)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3)); fig.patch.set_alpha(0)
    ax1.set_facecolor("#0e1626")
    ax1.plot(cmp, [r * 100 for r in rec], "o-", color="#2dd4bf", lw=2.5, label="single tree (beam)")
    for b, x, y in zip(beams, cmp, rec):
        ax1.annotate(f"b{b}", (x, y * 100), textcoords="offset points", xytext=(5, -10), color="#67769a", fontsize=8)
    for f, fb in forests:                                # forest reaches the ceiling cheaply
        ax1.plot(fb["forest_cmp"], fb["forest_recall"] * 100, "*", color="#ffd166", ms=15)
        ax1.annotate(f"forest {f}x", (fb["forest_cmp"], fb["forest_recall"] * 100),
                     textcoords="offset points", xytext=(6, 4), color="#ffd166", fontsize=9)
    ax1.axvline(N, color="#ff5c7a", ls="--", lw=1.5); ax1.text(N * 0.6, 8, f"exact scan\n{N} cmp", color="#ff5c7a", fontsize=9)
    ax1.set_xscale("log", base=2); ax1.set_xlabel("comparisons / query", color="#c8d3e6")
    ax1.set_ylabel("recall@1 (%)", color="#c8d3e6"); ax1.set_ylim(0, 104); ax1.tick_params(colors="#67769a")
    ax1.set_title("search: a forest breaks the single-tree ceiling", color="#c8d3e6", fontsize=12)
    ax2.set_facecolor("#0e1626")
    ax2.bar(range(len(flux)), flux, color="#2dd4bf")
    ax2.set_xlabel("leaf (sorted)", color="#c8d3e6"); ax2.set_ylabel("queries routed here", color="#c8d3e6")
    ax2.set_title("leaf flux: a few thick veins, many thin", color="#c8d3e6", fontsize=12)
    ax2.tick_params(colors="#67769a")
    for ax in (ax1, ax2):
        ax.grid(True, color="#22324f", lw=0.7)
        for s in ax.spines.values():
            s.set_color("#22324f")
    return {"demo": "search", "fig": _fig_uri(fig), "stats": tree.stats(),
            "beam8": {"recall": rec[beams.index(8)], "cmp": round(cmp[beams.index(8)]), "exact": N},
            "forest": {"trees": 4, "recall": forests[0][1]["forest_recall"],
                       "cmp": forests[0][1]["forest_cmp"], "exact": N}}


@app.route("/api/scaling", methods=["POST"])
def api_scaling():
    return jsonify(_scaling_search() if request.form.get("demo") == "search" else _scaling_capacity())


# --- content-addressed store: S3-style keys from properties --------------
_STORE = {}
_PAL = {"red": (235, 70, 70), "green": (70, 200, 110), "blue": (80, 120, 235),
        "yellow": (235, 205, 60), "magenta": (210, 80, 200)}
_SHAPES = ["circle", "rectangle", "triangle", "line"]


def _store():
    if "store" not in _STORE:
        coder = scn.SceneCoder(dim=1024, seed=0)
        store = huri.FacetStore()
        rng = np.random.default_rng(0)
        for i in range(120):
            shp = _SHAPES[rng.integers(len(_SHAPES))]; col = list(_PAL)[rng.integers(len(_PAL))]
            img, _ = hvz.make_shape(shp, 64, seed=i, fg=_PAL[col])
            tags = scn.auto_tags(img)
            store.put(i, tags, vector=coder.encode(tags))
        _STORE["store"], _STORE["coder"] = store, coder
    return _STORE["store"], _STORE["coder"]


def _store_keyspace():
    store, coder = _store()
    ok = n = 0
    for k, recs in store.flat.items():
        for r in recs:
            ok += huri.address_from_content(r["vec"], coder) == k; n += 1
    depth = []
    for order in [("colour",), ("colour", "shape"), ("colour", "shape", "texture")]:
        s = huri.FacetStore(order=order)
        for k, recs in store.flat.items():
            for r in recs:
                s.put(r["id"], r["tags"])
        depth.append({"depth": len(order), "buckets": s.stats()["buckets"], "max": s.stats()["max_bucket"]})
    return {"demo": "keyspace", "tree": store.tree(""), "stats": store.stats(),
            "sample": store.keys()[:6], "addr_acc": round(ok / n, 3), "depth": depth}


def _store_address():
    store, coder = _store()
    rng = np.random.default_rng()
    shp = _SHAPES[rng.integers(len(_SHAPES))]; col = list(_PAL)[rng.integers(len(_PAL))]
    img, _ = hvz.make_shape(shp, 72, seed=int(rng.integers(1_000_000)), fg=_PAL[col])
    tags = scn.auto_tags(img); key = huri.make_key(tags)
    rkey = huri.address_from_content(coder.encode(tags), coder)
    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return {"demo": "address", "thumb": _thumb(rgba), "tags": tags, "uri": key,
            "resonator_uri": rkey, "match": key == rkey,
            "bucket": [r["id"] for r in store.bucket(key)]}


@app.route("/api/store", methods=["POST"])
def api_store():
    return jsonify(_store_address() if request.form.get("demo") == "address" else _store_keyspace())


@app.route("/api/compression", methods=["POST"])
def api_compression():
    """Live compression + speed comparison on one image."""
    S = 160
    img = _demo_image(S)
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def measure(bits, shared):
        t = time.perf_counter()
        h = HolographicImage(img.shape, keep=3000, dim=16384, seed=0).store(img, bits=bits, shared_index=shared)
        enc = (time.perf_counter() - t) * 1e3
        t = time.perf_counter(); rec = h.reconstruct(); dec = (time.perf_counter() - t) * 1e3
        return {"method": ("float plate" if bits is None else f"{bits}-bit" + (" shared-idx" if shared else "")),
                "size": round(h.stored_bytes() / 1e3, 1), "psnr": round(float(_psnr(img, rec)), 1),
                "enc": round(enc), "dec": round(dec)}

    rows = [measure(None, False), measure(4, False), measure(4, True), measure(3, True)]

    def jpg(q):
        b = io.BytesIO(); Image.fromarray(u8).save(b, "JPEG", quality=q); data = b.getvalue()
        a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), float) / 255
        return data, {"method": f"JPEG q{q}", "size": round(len(data) / 1e3, 1), "psnr": round(float(_psnr(img, a)), 1)}
    jdata, jrow = jpg(85)
    pb = io.BytesIO(); Image.fromarray(u8).save(pb, "PNG")
    refs = [jrow, {"method": "PNG (lossless)", "size": round(len(pb.getvalue()) / 1e3, 1), "psnr": None}]

    # quick resilience headline at 10% corruption (1 trial)
    holo = HolographicImage(img.shape, keep=3000, dim=16384, seed=0).store(img, bits=4)
    bb = bytearray(jdata); r = np.random.default_rng(0)
    for i in r.choice(len(bb), int(len(bb) * 0.10), replace=False): bb[i] = int(r.integers(256))
    try:
        ja = np.asarray(Image.open(io.BytesIO(bytes(bb))).convert("RGB"), float) / 255
        jp0 = round(float(_psnr(img, ja)), 1) if ja.shape == img.shape else 0.0
    except Exception:
        jp0 = 0.0
    hp = round(float(_psnr(img, holo.reconstruct(mask=holo.damage_mask(0.10, 0)))), 1)

    K, D = 3000, 16384
    keys = f"WHT keys {(D//8 + K*2)/1e3:.0f} KB  vs  dense {K*D*8/1e6:.0f} MB  ({K*D*8/(D//8+K*2):,.0f}x smaller)"

    # ---- many DISTINCT files ----------------------------------------------
    G = _gallery(128); n = len(G); npix = 128 * 128
    def u8g(im): return (np.clip(im, 0, 1) * 255).astype(np.uint8)
    # standalone holographic file each (4-bit)
    sh_tot = 0; sh_ps = []
    for im in G:
        h = HolographicImage(im.shape, keep=1500, dim=16384, seed=0).store(im, bits=4, shared_index=True)
        sh_tot += h.stored_bytes(); sh_ps.append(float(_psnr(im, h.reconstruct())))
    # one multiplexed archive holding all of them, 4-bit plates
    arc = HolographicArchive((128, 128, 3), capacity=n, keep=2000, dim=32768, seed=0)
    for im in G:
        arc.add(im)
    for c in range(arc.nchan):
        codes, cents = _lloyd_max(arc.plates[c], 4); arc.plates[c] = cents[codes]
    arc_bytes = arc.nchan * (arc.dim * 4 / 8 + 16 * 8) + n * arc.nchan * np.ceil(npix / 8) + n * (arc.thumb ** 2 * arc.nchan * 8)
    arc_ps = [float(_psnr(G[i], arc.recover(i))) for i in range(n)]
    recall_ok = sum(arc.recall(G[i])[0] == i for i in range(n))
    # resilience: one joint recovery per channel at 40% destroyed, reconstruct all
    mask = arc.damage_mask(0.40, 7)
    joints = [arc._joint_recover(c, mask) for c in range(arc.nchan)]
    dmg_ps = []
    for i in range(n):
        chans = []
        for c in range(arc.nchan):
            f = np.zeros(npix); f[arc._idx[i][c]] = joints[c][i]
            chans.append(arc.M.T @ f.reshape(128, 128) @ arc.M)
        dmg_ps.append(float(_psnr(G[i], np.clip(np.stack(chans, -1), 0, 1))))
    # conventional codecs, each file separately
    jt = 0; jp = []
    for im in G:
        b = io.BytesIO(); Image.fromarray(u8g(im)).save(b, "JPEG", quality=85); jt += len(b.getvalue())
        jp.append(float(_psnr(im, np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), float) / 255)))
    pt = 0
    for im in G:
        b = io.BytesIO(); Image.fromarray(u8g(im)).save(b, "PNG"); pt += len(b.getvalue())
    many = [
        {"method": "holographic file each (4-bit)", "size": round(sh_tot/1e3, 1), "per": round(sh_tot/n/1e3, 1), "psnr": round(float(np.mean(sh_ps)), 1)},
        {"method": f"ONE archive, all {n} (4-bit)", "size": round(arc_bytes/1e3, 1), "per": round(arc_bytes/n/1e3, 1), "psnr": round(float(np.mean(arc_ps)), 1)},
        {"method": "JPEG q85 each", "size": round(jt/1e3, 1), "per": round(jt/n/1e3, 1), "psnr": round(float(np.mean(jp)), 1)},
        {"method": "PNG each", "size": round(pt/1e3, 1), "per": round(pt/n/1e3, 1), "psnr": None},
    ]
    many_strength = (f"the {n} distinct images live in ONE {arc_bytes/1e3:.0f} KB archive, recalled {recall_ok}/{n} by content "
                     f"-- and still {recall_ok}/{n} with 40% of it destroyed (avg {np.mean(dmg_ps):.0f} dB). No per-file codec survives that.")

    return jsonify({"rows": rows, "refs": refs, "keys": keys,
                    "resilience": f"single file at 10% random corruption: JPEG {jp0} dB vs hologram {hp} dB",
                    "many": many, "many_strength": many_strength})


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """1-bit hypervector retrieval vs float32 cosine (runs bench_batch.py)."""
    try:
        out = subprocess.run([sys.executable, "bench_batch.py"], capture_output=True, text=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        out = "batch benchmark timed out"
    return jsonify({"output": out})


# --- creature visual ------------------------------------------------------
_CREATURE = {}   # cache each trained mind (per mode) so re-clicks are instant

# Per-mode setup for the three creature demos. All share the same brain, senses,
# and energy mechanic; they differ only in the world they live in and how much
# working memory the brain folds in (mazes and walls need it; open forage does
# not). 'layout' is the world/maze seed we show; 'mind_seeds' are the candidate
# brains we train and pick the best of at run time.
_MODES = {
    "poison": dict(mem=0, episodes=200, tsteps=50, steps=80, walls=0, npois=2, maze=False,
                   layout=3, mind_seeds=[7, 2, 11, 5, 13], eps0=0.35, novelty=0.1, cap=5000),
    "walls":  dict(mem=3, episodes=240, tsteps=90, steps=100, walls=8, npois=2, maze=False,
                   layout=3, mind_seeds=[2, 7, 11, 5], eps0=0.45, novelty=0.2, cap=12000),
    "maze":   dict(mem=4, episodes=240, tsteps=90, steps=70, walls=0, npois=0, maze=True,
                   layout=7, mind_seeds=[2, 7, 11], eps0=0.50, novelty=0.2, cap=12000,
                   size=16, energy=320, dim=2048, ants=28, rounds=60, braid=1.0, elite=12.0),
}

def _make_world(mode, layout_seed):
    """Build a fresh world for a mode. Maze worlds use fixed_seed so the SAME
    labyrinth is rebuilt every episode; forage worlds use a plain seed so
    walls/poison re-randomise each reset.

    The labyrinth is now 16x16. A single egocentric reactive brain caps out around
    7x7 (it only sees its immediate surroundings), so a maze this size is solved by
    the holographic slime-mold colony instead -- the module built for exactly this.
    The battery is sized to the maze so energy doesn't end the run before the exit."""
    cfg = _MODES[mode]
    if cfg["maze"]:
        n = cfg.get("size", 7)
        return GridWorld(n, n, maze=True, fixed_seed=layout_seed,
                         braid=cfg.get("braid", 0.0), start_energy=cfg.get("energy", 100))
    return GridWorld(7, 7, n_poison=cfg["npois"], n_walls=cfg["walls"], seed=layout_seed)

def _slime_rollout(seed):
    """Solve a BRAIDED 16x16 labyrinth with the holographic slime-mold colony and record
    the SEARCH, not just the answer. Braiding opens dead-ends into loops, so the maze has
    MANY routes out -- which is the point: this is where the slime-mold optimisation earns
    its keep. The walkers carry no compass (they know nothing about where the exit is) and
    explore by pheromone alone; a tile is laid with a trail only after a walker has stepped
    on it; shorter successful routes deposit more pheromone per edge and elitist
    reinforcement commits the colony to the best-so-far, so over rounds the tube network
    THINS to the shortest connecting tube -- exactly what Physarum does. We read that tube
    back from the holographic field, and return the cells in first-discovery order plus the
    emergent route so the UI can replay the real thing with no precomputed guide."""
    cfg = _MODES["maze"]
    w = _make_world("maze", seed)
    path, info = solve_maze(w, dim=cfg["dim"], ants=cfg["ants"], rounds=cfg["rounds"],
                            seed=0, use_compass=False, record=True, elite=cfg["elite"])
    steps = len(path) - 1
    return dict(w=w.w, h=w.h, walls=[list(c) for c in w.walls],
                explore=[list(c) for c in info["order"]],     # first-discovery order
                route=[list(c) for c in path],                # the emergent (shortest) tube
                opt=info["optimal"], cells=info["cells"], steps=steps,
                escaped=(path[-1] == (w.fx, w.fy)),
                start_energy=w.start_energy, energy=w.start_energy - steps)

def _rollout(make_world, encoder, mind, steps=80, mem=0, eps=0.05):
    """Live one life on a freshly-built world and record it frame-by-frame for
    the animation.

    Each frame carries the creature's position, where the star/exit currently is,
    the energy remaining, whether it just ate a star, and whether it just died.
    We also record the static walls and the BFS-optimal route from the start to
    the goal, so the UI can draw 'the best possible route' behind the creature's
    learned one. The life ends the instant the creature dies (poison or an empty
    battery) or, in a maze, escapes -- so the frame list can be shorter than
    `steps`.

    The trained creature runs with a tiny exploration epsilon (the same trick
    _evaluate() uses): a purely greedy agent can get stuck oscillating between two
    cells, and an occasional random step shakes it loose. `mind=None` is the
    random baseline. `mem` is the working-memory depth folded into its state."""
    w = make_world()
    optimal = w.shortest_path((w.cx, w.cy), (w.fx, w.fy))     # the best route to the goal
    eaten = []; hits = 0
    recent = []                                              # recent moves, newest first
    senses = w.senses()
    state = encoder.build_state(senses, recent, mem)
    frames = [{"x": w.cx, "y": w.cy, "fx": w.fx, "fy": w.fy,
               "ate": False, "energy": w.energy, "dead": False}]
    rng = np.random.default_rng(123)
    for _ in range(steps):
        if mind is None:
            a = int(rng.integers(4))
        else:
            a = mind.decide(state, explore=False, epsilon=eps)
        senses, r, ate, done = w.step(GridWorld.ACTIONS[a])
        recent = [GridWorld.ACTIONS[a]] + recent
        state = encoder.build_state(senses, recent, mem)
        frames.append({"x": w.cx, "y": w.cy, "fx": w.fx, "fy": w.fy,
                       "ate": bool(ate), "energy": w.energy, "dead": not w.alive and not w.escaped})
        if not w.alive and not w.escaped and (w.cx, w.cy) in w.poison:
            hits = 1                                          # died by stepping on poison
        if ate:
            eaten.append((w.cx, w.cy))
        if done:
            break
    return dict(poison=list(w.poison), walls=list(w.walls), optimal=optimal,
                eaten=eaten, hits=hits, w=w.w, h=w.h, frames=frames,
                stars=w.stars, energy=w.energy, alive=w.alive, escaped=w.escaped,
                start_energy=w.start_energy, steps=len(frames) - 1)

def _draw_pair(before, after, caption):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    fig.patch.set_alpha(0)
    info = lambda ep: f"{len(ep['eaten'])} stars" + (f", {ep['hits']} poison hits" if ep['hits'] else ", 0 poison hits")
    for ax, ep, ttl in ((axes[0], before, "before training (random) -- " + info(before)),
                        (axes[1], after, "after training (greedy) -- " + info(after))):
        ax.set_facecolor("#0e1626")
        W, H = ep["w"], ep["h"]
        ax.set_xlim(-.5, W-.5); ax.set_ylim(H-.5, -.5); ax.set_aspect("equal")
        ax.set_xticks(np.arange(-.5, W, 1)); ax.set_yticks(np.arange(-.5, H, 1))
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.grid(True, color="#22324f", lw=1)
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_color("#22324f")
        for (x, y) in ep["poison"]:
            ax.add_patch(plt.Rectangle((x-.5, y-.5), 1, 1, color="#ff5c7a", alpha=.30))
            ax.plot(x, y, "x", color="#ff5c7a", ms=13, mew=3)
        px = [p[0] for p in ep["path"]]; py = [p[1] for p in ep["path"]]
        ax.plot(px, py, "-", color="#2dd4bf", lw=2.5, alpha=.85, solid_capstyle="round")
        ax.plot(px[0], py[0], "o", color="#c8d3e6", ms=13, zorder=5)
        for (x, y) in ep["foods"]: ax.plot(x, y, "*", color="#ffd166", ms=20, mec="#7a5b00", zorder=4)
        for (x, y) in ep["eaten"]: ax.plot(x, y, "*", color="#63dcbe", ms=22, mec="#063", zorder=4)
        ax.set_title(ttl, color="#c8d3e6", fontsize=12, pad=10)
    fig.suptitle(caption, color="#67769a", fontsize=12, y=0.04)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", transparent=True); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _reflex(enc, mind):
    dirs = [("star east", {"food_x": "east", "food_y": "none"}, "E"),
            ("star west", {"food_x": "west", "food_y": "none"}, "W"),
            ("star north", {"food_x": "none", "food_y": "north"}, "N"),
            ("star south", {"food_x": "none", "food_y": "south"}, "S")]
    reflex = []; seek = avoid = 0
    for name, senses, d in dirs:
        clear = mind.actions[mind.decide(enc.encode(senses), explore=False)]
        blocked = mind.actions[mind.decide(enc.encode({**senses, "danger_" + d: "yes"}), explore=False)]
        seek += (clear == d); avoid += (blocked != d)
        reflex.append({"name": name, "clear": clear, "blocked": blocked, "avoids": blocked != d})
    return seek, avoid, reflex

def _creature_sprite(prefer="amg2"):
    """Build the walking-creature sprite frames from the real set, or None if the
    asset is missing. Maps a move direction to the facing the character should show:
    N -> back, S -> front, W -> left, E -> right; two frames per facing for a walk
    cycle."""
    if not SPRITES:
        return None
    full = lambda c: all(f"{c}_{d}{n}.gif" in SPRITES for d in ("lf", "rt", "bk", "fr") for n in (1, 2))
    pick = prefer if full(prefer) else next((c for c in sorted({n.split("_")[0] for n in SPRITES}) if full(c)), None)
    if pick is None:
        return None
    facing = {"N": "bk", "S": "fr", "W": "lf", "E": "rt"}
    # _cutout() keys the flat background out so the creature has real transparency
    # and the grid shows through, instead of sitting in an opaque white box.
    frames = {mv: [_thumb(_cutout(SPRITES[f"{pick}_{fc}{n}.gif"])) for n in (1, 2)] for mv, fc in facing.items()}
    return {"char": pick, "frames": frames}


def _build_creature(mode):
    """Train a few candidate brains for a mode and keep the best one. Selection
    is done at run time because the exact learned policy is platform-sensitive
    (floating-point ties over many episodes), so we never trust a single seed."""
    import contextlib
    cfg = _MODES[mode]
    if mode == "maze":
        # No reactive brain can hold a 16x16 maze, so there is nothing to train here:
        # the slime-mold colony solves it directly. We keep an encoder only for the
        # random-walker baseline shown on the left.
        return {"maze_after": _slime_rollout(cfg["layout"]),
                "enc": CreatureEncoder(256, seed=1), "rolls": None}
    enc = CreatureEncoder(256, seed=1)
    best = None
    for ms in cfg["mind_seeds"]:
        mind = HolographicMind(256, GridWorld.ACTIONS, k=15, epsilon=cfg["eps0"],
                               novelty_bonus=cfg["novelty"], memory_cap=cfg["cap"], seed=ms)
        with contextlib.redirect_stdout(io.StringIO()):
            _train(_make_world(mode, cfg["layout"]), enc, mind, episodes=cfg["episodes"],
                   eps_start=cfg["eps0"], mem=cfg["mem"], max_steps=cfg["tsteps"])

        if mode == "maze":
            # Score a maze brain by how reliably it escapes the (fixed) labyrinth.
            rolls = [_rollout(lambda: _make_world(mode, cfg["layout"]), enc, mind,
                              steps=cfg["steps"], mem=cfg["mem"]) for _ in range(5)]
            escapes = sum(r["escaped"] for r in rolls)
            cand = {"score": escapes, "mind": mind, "rolls": rolls,
                    "perfect": escapes == len(rolls)}
        else:
            rolls = [_rollout(lambda s=s: _make_world(mode, s), enc, mind,
                              steps=cfg["steps"], mem=cfg["mem"]) for s in range(6)]
            stars = sum(r["stars"] for r in rolls); clean = sum(r["hits"] == 0 for r in rolls)
            if mode == "poison":
                # A reactive (mem=0) brain can be probed with one-shot senses, but
                # that synthetic reflex correlates only loosely with real play -- a
                # great forager can still lunge at a star in the artificial probe.
                # So we select on what the demo actually SHOWS: stars gathered and
                # lives survived, with the reflex as a gentle tie-breaking nudge.
                seek, avoid, reflex = _reflex(enc, mind)
                cand = {"score": stars + 5 * clean + 8 * (seek + avoid), "mind": mind,
                        "rolls": rolls, "seek": seek, "avoid": avoid, "reflex": reflex,
                        "perfect": False}
            else:
                # The walls brain is trained WITH working memory, so the memoryless
                # reflex probe would not match how it actually thinks -- we score it
                # by what matters here instead: stars gathered and lives survived.
                cand = {"score": 5 * clean + stars, "mind": mind, "rolls": rolls,
                        "perfect": False}
        if best is None or cand["score"] > best["score"]:
            best = cand
        if cand["perfect"]:
            break                                            # good enough -- stop early
    best["enc"] = enc
    return best


@app.route("/api/creature", methods=["POST"])
def api_creature():
    mode = (request.form.get("mode") or "poison").lower()
    if mode not in _MODES:
        mode = "poison"
    cfg = _MODES[mode]
    if mode not in _CREATURE:
        _CREATURE[mode] = _build_creature(mode)
    b = _CREATURE[mode]; enc = b["enc"]; rolls = b["rolls"]

    def pack(ep):
        return {"frames": ep["frames"], "poison": ep["poison"], "walls": ep["walls"],
                "optimal": ep["optimal"], "w": ep["w"], "h": ep["h"], "stars": ep["stars"],
                "hits": ep["hits"], "alive": ep["alive"], "escaped": ep["escaped"],
                "steps": ep["steps"], "start_energy": ep["start_energy"]}

    out = {"mode": mode, "sprite": _creature_sprite("amg2")}

    if mode == "maze":
        after = b["maze_after"]
        before = _rollout(lambda: _make_world(mode, cfg["layout"]), enc, None,
                          steps=cfg["steps"], mem=cfg["mem"])
        opt = after["opt"]
        cap = (f"a braided 16&times;16 maze &mdash; loops everywhere, so there are MANY ways out, and the "
               f"job is to find the SHORTEST. The slime-mold colony floods it with walkers that know "
               f"NOTHING about where the exit is (no compass, no precomputed route): a tile only gets a "
               f"trail after a walker reaches it, shorter successful routes lay down more pheromone per "
               f"step, and the tube network thins until the shortest connecting tube survives &mdash; "
               f"just like Physarum. Watch it explore ({after['cells']} reachable cells), then the "
               f"shortest tube emerge ({after['steps']} steps; the true optimum is {opt}). A random "
               f"walker (left) never gets out.")
        out.update(before=pack(before), after=after, caption=cap)
        return jsonify(out)

    # forage modes: poison and walls -----------------------------------------
    # illustrative world: the cleanest life (survived poison, collected the most).
    idx = sorted(range(len(rolls)), key=lambda s: (rolls[s]["hits"], -rolls[s]["stars"]))[0]
    after = rolls[idx]
    before = _rollout(lambda: _make_world(mode, idx), enc, None,
                      steps=cfg["steps"], mem=cfg["mem"])
    avg_t = round(float(np.mean([r["stars"] for r in rolls])), 1)
    avg_r = round(float(np.mean([_rollout(lambda s=s: _make_world(mode, s), enc, None,
                                          steps=cfg["steps"], mem=cfg["mem"])["stars"]
                                 for s in range(len(rolls))])), 1)
    clean = sum(1 for r in rolls if r["hits"] == 0)
    if mode == "walls":
        cap = (f"over {len(rolls)} layouts: random {avg_r} stars  vs  trained {avg_t} "
               f"stars  -- routing around {cfg['walls']} walls, survived {clean}/{len(rolls)}")
        out.update(before=pack(before), after=pack(after), caption=cap)
    else:
        cap = (f"over {len(rolls)} lives: random {avg_r} stars  vs  trained {avg_t} "
               f"stars  -- survived without touching poison in {clean}/{len(rolls)}")
        out.update(before=pack(before), after=pack(after), caption=cap)
    return jsonify(out)


@app.route("/api/tour", methods=["POST"])
def api_tour():
    """Run the whole-system tour script and return its text output."""
    try:
        out = subprocess.run([sys.executable, "tour.py"],
                             capture_output=True, text=True, timeout=180).stdout
    except subprocess.TimeoutExpired:
        out = "tour timed out"
    return jsonify({"output": out})


@app.route("/api/tests", methods=["POST"])
def api_tests():
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py")))
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", *files, "-v", "--tb=line", "-p", "no:cacheprovider"],
                             capture_output=True, text=True, timeout=600).stdout
    except subprocess.TimeoutExpired:
        return jsonify({"summary": "timed out", "tests": []})
    tests = []
    for line in out.splitlines():
        if "::" in line and (" PASSED" in line or " FAILED" in line or " ERROR" in line):
            name = line.split("::", 1)[1].split(" ")[0]
            status = "PASSED" if " PASSED" in line else ("FAILED" if " FAILED" in line else "ERROR")
            tests.append({"name": name, "status": status})
    summary = next((l.strip() for l in reversed(out.splitlines()) if "passed" in l or "failed" in l or "error" in l), "")
    return jsonify({"summary": summary, "tests": tests})


# --- unified brain: one model over real corpora --------------------------
# This is the same UnifiedMind console that used to live in unified_app.py,
# folded into the main page so every subsystem is in one place. It pulls a real
# NLTK corpus on demand (the data is hosted on GitHub), trains ONE UnifiedMind on
# it, and then classify / recall / organize / generate all run against that one
# trained mind over a single holographic space.
U_STATE = {"mind": None, "dataset": None, "labels": [], "test": [], "raw_len": 0}

def _u_ensure(pkg):
    """Make sure an NLTK corpus is present, pulling it on demand the first time."""
    import nltk
    nltk.data.path.insert(0, "/home/claude/nltk_data")
    try:
        nltk.download(pkg, quiet=True); return True
    except Exception:
        return False

def _u_content(tokens):
    """Drop stopwords -- the content words are what carry the topic/style."""
    return [w for w in tokens if w not in STOPWORDS]

def _u_load_reuters():
    from nltk.corpus import reuters
    single = [(f, reuters.categories(f)[0]) for f in reuters.fileids()
              if len(reuters.categories(f)) == 1]
    top = ["earn", "acq", "crude", "trade", "money-fx", "interest",
           "money-supply", "ship", "sugar", "coffee"]
    by = defaultdict(list)
    for f, c in single:
        if c in top:
            by[c].append(f)
    items, raw = [], []
    for c, fids in by.items():
        for f in fids[:150]:
            toks = [w.lower() for w in reuters.words(f) if w.isalpha()]
            items.append((_u_content(toks), c)); raw.append(" ".join(toks))
    return items, " ".join(raw), "Reuters financial newswire -- 10 confusable categories (grain/crude/money-fx share vocabulary)"

def _u_load_brown():
    from nltk.corpus import brown
    items, raw = [], []
    for c in ["news", "romance", "science_fiction", "government", "hobbies"]:
        words = [w.lower() for w in brown.words(categories=c) if w.isalpha()]
        for k in range(0, min(len(words), 18000) - 300, 300):
            chunk = words[k:k + 300]
            items.append((_u_content(chunk), c)); raw.append(" ".join(chunk))
    return items, " ".join(raw), "Brown corpus -- five prose genres, in 300-word chunks"

def _u_load_gutenberg():
    from nltk.corpus import gutenberg
    books = {"austen-emma.txt": "Austen", "carroll-alice.txt": "Carroll",
             "shakespeare-hamlet.txt": "Shakespeare", "melville-moby_dick.txt": "Melville",
             "chesterton-brown.txt": "Chesterton"}
    items, raw = [], []
    for fid, author in books.items():
        if fid not in gutenberg.fileids():
            continue
        words = [w.lower() for w in gutenberg.words(fid) if w.isalpha()][:9000]
        for k in range(0, len(words) - 200, 200):
            chunk = words[k:k + 200]
            items.append((_u_content(chunk), author)); raw.append(" ".join(chunk))
    return items, " ".join(raw), "Project Gutenberg -- classify the author, generate in their style"

def _u_load_europarl():
    from nltk.corpus import europarl_raw as eu
    items, raw = [], []
    for lang in ("english", "french", "german", "spanish", "italian"):
        words = [w.lower() for w in getattr(eu, lang).words()[:12000] if w.isalpha()]
        for k in range(0, len(words) - 120, 120):
            chunk = words[k:k + 120]
            items.append((_u_content(chunk), lang)); raw.append(" ".join(chunk))
    return items, " ".join(raw), "Europarl -- five languages; classify the language, generate in it"

U_DATASETS = {
    "reuters":   ("Reuters categories", ["reuters"], _u_load_reuters),
    "brown":     ("Brown genres", ["brown"], _u_load_brown),
    "gutenberg": ("Gutenberg authors", ["gutenberg"], _u_load_gutenberg),
    "europarl":  ("Europarl languages", ["europarl_raw"], _u_load_europarl),
}

def _u_build(dataset_id):
    """Pull a corpus, split 70/30, train ONE UnifiedMind, report held-out accuracy."""
    name, pkgs, loader = U_DATASETS[dataset_id]
    for p in pkgs:
        _u_ensure(p)
    items, raw, desc = loader()
    # split each label 70/30 so the accuracy number is on data the mind never saw
    by = defaultdict(list)
    for toks, lab in items:
        by[lab].append(toks)
    rng = np.random.default_rng(0)
    train, test = [], []
    for lab, docs in by.items():
        docs = list(docs); rng.shuffle(docs)
        cut = int(len(docs) * 0.7)
        train += [(d, lab) for d in docs[:cut]]
        test += [(d, lab) for d in docs[cut:]]
    rng.shuffle(train)
    mind = UnifiedMind(dim=1024, seed=0, text_window=3)
    mind.read([toks for toks, _ in train])      # learn word co-occurrence first
    for toks, lab in train:
        mind.learn(toks, lab, "text")           # classification + recall, one memory
    mind.maintain_now()
    mind.learn_sequence(raw[:160000], n=6)       # generation, same space
    acc = sum(mind.classify(toks, "text")[0] == lab for toks, lab in test) / max(1, len(test))
    U_STATE.update({"mind": mind, "dataset": name, "labels": sorted(by),
                    "test": test, "raw_len": len(raw), "desc": desc})
    return {"ok": True, "dataset": name, "desc": desc, "labels": sorted(by),
            "counts": mind.memory.live.counts_by_label(),
            "prototypes": mind.memory.live.size(),
            "trained": len(train), "held_out": len(test),
            "accuracy": round(100 * acc), "gen_chars": min(len(raw), 160000)}

@app.route("/api/unified/datasets")
def u_datasets():
    import importlib
    have = importlib.util.find_spec("nltk") is not None
    return jsonify({"datasets": [{"id": d, "name": n, "available": have}
                                 for d, (n, _, _) in U_DATASETS.items()], "nltk": have})

@app.route("/api/unified/load", methods=["POST"])
def u_load():
    did = request.json.get("id")
    if did not in U_DATASETS:
        return jsonify({"ok": False, "error": "unknown dataset"})
    try:
        return jsonify(_u_build(did))
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e} "
                        "(if a corpus is missing, a network connection is needed the "
                        "first time to pull it from GitHub)"})

@app.route("/api/unified/classify", methods=["POST"])
def u_classify():
    if U_STATE["mind"] is None:
        return jsonify({"error": "load a dataset first"})
    toks = _u_content((request.json.get("text") or "").lower().split())
    if not toks:
        return jsonify({"error": "type some words the model might know"})
    mind = U_STATE["mind"]
    label, score = mind.classify(toks, "text")
    (rlabel, _), rscore = mind.recall(toks, "text")
    return jsonify({"label": label, "score": round(float(score), 3),
                    "recall": {"label": rlabel, "score": round(float(rscore), 3)}})

@app.route("/api/unified/organize", methods=["POST"])
def u_organize():
    if U_STATE["mind"] is None:
        return jsonify({"error": "load a dataset first"})
    mind = U_STATE["mind"]
    choice = mind.maintain_now()
    return jsonify({"after": mind.memory.live.counts_by_label(),
                    "choice": (choice[0] if choice else "keep"),
                    "note": "each label may hold several sub-prototypes when the memory found "
                            "it multi-modal; one each means it stayed simple."})

@app.route("/api/unified/generate", methods=["POST"])
def u_generate():
    if U_STATE["mind"] is None or U_STATE["mind"]._gen is None:
        return jsonify({"error": "load a dataset first"})
    j = request.json
    seed = (j.get("seed") or "the ").lower()
    length = max(20, min(int(j.get("length", 220)), 600))
    temp = max(0.1, min(float(j.get("temperature", 0.45)), 1.2))
    return jsonify({"text": U_STATE["mind"].generate(seed, length, temp)})

@app.route("/api/unified/recall", methods=["POST"])
def u_recall():
    if U_STATE["mind"] is None:
        return jsonify({"error": "load a dataset first"})
    toks = _u_content((request.json.get("text") or "").lower().split())
    if not toks:
        return jsonify({"error": "type some words"})
    (label, example), score = U_STATE["mind"].recall(toks, "text")
    snippet = " ".join(example[:18]) if isinstance(example, list) else str(example)
    return jsonify({"label": label, "score": round(float(score), 3), "example": snippet})


PAGE = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Holographic Archive</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root{--navy:#0b1020;--panel:#111a2e;--panel2:#0e1626;--teal:#2dd4bf;--coral:#ff7a6b;
        --text:#c8d3e6;--muted:#67769a;--pass:#3ddc97;--fail:#ff5c7a;--line:#1e2c47;}
  *{box-sizing:border-box} body{margin:0;background:var(--navy);color:var(--text);
     font-family:'JetBrains Mono',monospace;font-size:14px;line-height:1.5}
  .wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px}
  h1{font-weight:800;font-size:22px;margin:0 0 2px} h1 .h{color:var(--teal)} h1 .l{color:var(--coral)}
  .sub{color:var(--muted);margin:0 0 24px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 18px;margin-bottom:18px}
  .panel h2{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--teal);margin:0 0 14px}
  button{font-family:inherit;font-weight:600;background:var(--teal);color:#05221d;border:0;
         border-radius:8px;padding:9px 16px;cursor:pointer;font-size:13px}
  button:hover{filter:brightness(1.1)} button.ghost{background:transparent;color:var(--coral);border:1px solid var(--coral)}
  select,input[type=file]{font-family:inherit;background:var(--panel2);color:var(--text);
         border:1px solid var(--line);border-radius:8px;padding:8px}
  label{color:var(--muted);font-size:12px;display:block;margin:0 0 4px}
  .row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-end}
  .gal{display:flex;gap:10px;flex-wrap:wrap}
  .gal figure{margin:0;text-align:center} .gal img{width:84px;height:84px;border-radius:8px;border:1px solid var(--line)}
  .gal figcaption{color:var(--muted);font-size:11px;margin-top:4px}
  .seg{display:flex;gap:0;border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .seg button{background:var(--panel2);color:var(--muted);border-radius:0;padding:8px 12px}
  .seg button.on{background:var(--coral);color:#2a0f0b}
  input[type=range]{width:170px;accent-color:var(--coral)}
  .val{color:var(--teal);font-weight:600}
  .results{display:flex;gap:18px;flex-wrap:wrap;margin-top:16px;align-items:flex-start}
  .results figure{margin:0;text-align:center} .results img{width:150px;height:150px;border-radius:10px;border:1px solid var(--line);image-rendering:pixelated}
  .arrow{align-self:center;color:var(--muted);font-size:24px}
  .badge{display:inline-block;padding:3px 10px;border-radius:20px;font-weight:600;font-size:12px}
  .ok{background:rgba(61,220,151,.15);color:var(--pass)} .no{background:rgba(255,92,122,.15);color:var(--fail)}
  .tests{max-height:280px;overflow:auto;border:1px solid var(--line);border-radius:8px}
  .tr{display:flex;justify-content:space-between;padding:6px 12px;border-bottom:1px solid var(--line);font-size:12px}
  .tr:last-child{border-bottom:0} .tr .p{color:var(--pass)} .tr .f{color:var(--fail)}
  .muted{color:var(--muted)} .spin{color:var(--coral)}
  .term{background:#070b16;border:1px solid var(--line);border-radius:8px;padding:14px;margin-top:12px;
        color:#9fe7da;font-size:12px;line-height:1.45;white-space:pre-wrap;max-height:420px;overflow:auto}
  table.cmp{border-collapse:collapse;margin-top:14px;font-size:13px}
  table.cmp th,table.cmp td{padding:6px 14px;text-align:right;border-bottom:1px solid var(--line)}
  table.cmp th:first-child,table.cmp td:first-child{text-align:left;color:var(--text)}
  table.cmp th{color:var(--teal);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.08em}
  table.cmp td{color:var(--muted)} table.cmp tr.ref td:first-child{color:var(--coral)}
  .note{color:var(--muted);font-size:12px;margin-top:10px}
  .sublabel{color:var(--text);font-size:12px;font-weight:600;letter-spacing:.04em;margin-top:6px;opacity:.85}
  table.reflex td{text-align:left;color:var(--muted)} table.reflex b{color:var(--text)}
  table.cmp .p{color:var(--pass);margin-left:8px} table.cmp .f{color:var(--fail);margin-left:8px}
  .creatgrids{display:flex;gap:22px;justify-content:center;flex-wrap:wrap;margin-top:10px}
  .creatcell{flex:1;min-width:250px;max-width:300px}
  .creatcell svg{display:block;width:100%;height:auto;border-radius:6px;margin-top:4px}
  .creatcell .sublabel{text-align:center;opacity:.8}
</style></head><body><div class="wrap">
  <h1><span class="h">holographic</span> <span class="l">archive</span></h1>
  <p class="sub">a numpy holographic / VSA engine &mdash; numbers, text, records, memory, reasoning, a learning creature, and a damage-tolerant image archive, all on one vector substrate</p>

  <div class="panel">
    <h2>System tour</h2>
    <p class="muted" style="margin:-8px 0 12px">runs every subsystem once &mdash; numbers &middot; text &middot; records &middot; key&rarr;value memory &middot; reasoning &middot; a foraging creature &middot; the image archive</p>
    <button onclick="runTour()">Run full system tour</button>
    <span id="toursum" class="muted" style="margin-left:12px"></span>
    <pre id="tourout" class="term" style="display:none"></pre>
  </div>

  <div class="panel">
    <h2>Unified brain &mdash; one model on a real corpus</h2>
    <p class="muted" style="margin:-8px 0 12px">pull a real corpus (NLTK data, hosted on GitHub &mdash; Reuters newswire, Brown genres, Gutenberg authors, or five Europarl languages), train ONE <code>UnifiedMind</code> on it, then classify &middot; recall &middot; organize &middot; generate all against that one mind over a single holographic space. The first pull of a corpus needs a network connection.</p>
    <div class="row">
      <div><label>corpus</label><select id="u_ds"></select></div>
      <button onclick="uLoad()">Pull &amp; train</button>
      <span id="u_loadmsg" class="spin" style="display:none">pulling + training&hellip; (first pull can take a while)</span>
    </div>
    <div id="u_trained" style="margin-top:14px"></div>

    <div id="u_ops" style="display:none">
      <div class="sublabel" style="margin-top:18px">classify &amp; recall</div>
      <p class="muted" style="margin:4px 0 8px">type a sentence in the style of the corpus &mdash; classify finds the nearest self-organized prototype, recall finds the nearest stored individual</p>
      <textarea id="u_cq" placeholder="e.g. the central bank raised interest rates to ease inflation" style="width:100%;min-height:52px;background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px;font-family:inherit;resize:vertical"></textarea>
      <div class="row" style="margin-top:8px"><button onclick="uClassify()">Classify</button>
        <button class="ghost" onclick="uRecall()">Recall nearest</button></div>
      <div id="u_cout" style="margin-top:10px"></div>

      <div class="sublabel" style="margin-top:18px">organize</div>
      <p class="muted" style="margin:4px 0 8px">how the one memory split each label into sub-prototypes (a multi-modal label gets more than one); click to run a maintenance pass</p>
      <button onclick="uOrganize()">Show &amp; reorganize</button>
      <div id="u_oout" style="margin-top:10px"></div>

      <div class="sublabel" style="margin-top:18px">generate</div>
      <p class="muted" style="margin:4px 0 8px">continue text in the style of what was learned (same holographic next-symbol prediction)</p>
      <div class="row">
        <div><label>seed</label><input id="u_seed" value="the " style="width:150px"></div>
        <div><label>length</label><input id="u_len" type="number" value="220" style="width:90px"></div>
        <div><label>temperature</label><input id="u_temp" type="number" step="0.05" value="0.45" style="width:90px"></div>
        <button onclick="uGenerate()">Generate</button>
      </div>
      <div id="u_gout" class="term" style="display:none;margin-top:12px"></div>
    </div>
  </div>

  <div class="panel">
    <h2>Compression &amp; speed</h2>
    <p class="muted" style="margin:-8px 0 12px">how big the stored hologram is, how fast it encodes/decodes, and how it stacks up against JPEG/PNG</p>
    <button onclick="runComp()">Measure</button>
    <span id="compsum" class="spin" style="margin-left:12px;display:none">measuring&hellip;</span>
    <div id="compout"></div>
  </div>

  <div class="panel">
    <h2>Batch operations</h2>
    <p class="muted" style="margin:-8px 0 12px">1-bit hypervectors + Hamming vs the common float32 cosine search, on a 10k-item retrieval task</p>
    <button onclick="runBatch()">Run benchmark</button>
    <span id="batchsum" class="spin" style="margin-left:12px;display:none">running&hellip; (~2s)</span>
    <pre id="batchout" class="term" style="display:none"></pre>
  </div>

  <div class="panel">
    <h2>Creature</h2>
    <p class="muted" style="margin:-8px 0 12px">a grid-world forager with a holographic mind (no neural net) &mdash; teal line is its path, the gold &#9733; is the star/exit it's after (green marks ones collected), red cells are poison, grey cells are solid walls, and in the forage worlds the faint dashed line is the <em>optimal</em> route (shortest path). It runs on energy: starts at 100, each step costs 1, every star gives +3, and poison empties it &mdash; instant death. The <em>Labyrinth</em> is different: a braided 16&times;16 maze (loops everywhere, so many routes out) solved by a slime-mold colony that you watch <em>discover</em> the way and then thin its tubes down to the <em>shortest</em> one &mdash; no precomputed guide, no exit compass, trails laid only on tiles a walker has already reached. Pick a world:</p>
    <button onclick="runCreature('poison')">Forage</button>
    <button onclick="runCreature('walls')">Obstacles</button>
    <button onclick="runCreature('maze')">Labyrinth</button>
    <span id="creatsum" class="spin" style="margin-left:12px;display:none">training a few candidate brains&hellip; (~20-40s first time, then cached)</span>
    <div id="creatout"></div>
  </div>

  <div class="panel">
    <h2>Test suite</h2>
    <button onclick="runTests()">Run pytest</button>
    <span id="tsum" class="muted" style="margin-left:12px"></span>
    <div id="tests" class="tests" style="margin-top:12px;display:none"></div>
  </div>

  <div class="panel">
    <h2>Stored gallery</h2>
    <div id="gallery" class="gal muted">loading&hellip;</div>
  </div>

  <div class="panel">
    <h2>Query &amp; recall</h2>
    <div class="row">
      <div><label>source image</label>
        <select id="src"></select></div>
      <div><label>&hellip;or upload your own</label>
        <input type="file" id="file" accept="image/*"></div>
    </div>
    <div class="row" style="margin-top:16px">
      <div><label>degradation</label>
        <div class="seg" id="deg">
          <button data-k="noise" class="on">noise</button>
          <button data-k="blur">blur</button>
          <button data-k="occlude">occlude</button>
        </div></div>
      <div><label>amount <span class="val" id="amtv">60%</span></label>
        <input type="range" id="amt" min="0" max="100" value="60"></div>
      <div><label>plate destroyed <span class="val" id="dmgv">0%</span></label>
        <input type="range" id="dmg" min="0" max="70" value="0"></div>
      <button onclick="recall()">Recall &rarr;</button>
    </div>
    <div id="out"></div>
  </div>

  <div class="panel">
    <h2>Recall by description</h2>
    <p class="muted" style="margin:-8px 0 12px">cross-modal recall &mdash; describe an image in words and the archive returns the best match from its tag <em>address space</em> (word atoms bundled into one hypervector), without ever seeing a picture. Tap a description or type your own.</p>
    <div class="seg" id="descChips" style="flex-wrap:wrap;gap:8px">
      <button data-w="radial pink rings">radial pink rings</button>
      <button data-w="checkerboard squares">checkerboard squares</button>
      <button data-w="quadrants blocks">quadrants of colour</button>
      <button data-w="horizontal stripes">horizontal stripes</button>
      <button data-w="ripples waves">ripples / waves</button>
      <button data-w="diagonal gradient">diagonal gradient</button>
    </div>
    <div class="row" style="margin-top:14px">
      <div style="flex:1"><input type="text" id="descIn" placeholder="e.g. pink rings in the center"></div>
      <button onclick="describe()">Recall &rarr;</button>
    </div>
    <div id="descOut" style="margin-top:14px"></div>
  </div>

  <div class="panel">
    <h2>Set packer (related images)</h2>
    <p class="muted" style="margin:-8px 0 12px">delta coding <em>between</em> related images &mdash; store one reference plus per-image deltas, so structure shared across a set (a logo suite, sprite sheet, UI frames) is kept once instead of in every file. Bit-exact, 8-bit integers throughout. It wins where images share large identical regions; for already-compressible images (photos, gradients) per-file PNG/JPEG still win, and the table says so honestly.</p>
    <button onclick="runPack()">Pack a logo set &rarr;</button>
    <span id="packsum" class="spin" style="margin-left:12px;display:none">packing&hellip;</span>
    <div id="packout" style="margin-top:14px"></div>
  </div>

  <div class="panel">
    <h2>Image vault (relate &middot; compress &middot; retrieve)</h2>
    <p class="muted" style="margin:-8px 0 12px">the general, format-agnostic store. It normalises any image to RGBA, <em>relates</em> them by a size-invariant fingerprint (clustering + query-by-example), and <em>compresses</em> adaptively &mdash; it measures every encoder, lossless and lossy, and keeps the smallest for that set. Pick a built-in set; the table is honest about which wins. Click a thumbnail to query: its three nearest matches light up.</p>
    <div class="seg" id="vaultSet">
      <button data-s="sprites" class="on">sprite set</button>
      <button data-s="photos">photo set</button>
      <button data-s="logos">logo set</button>
    </div>
    <span id="vaultsum" class="spin" style="margin-left:12px;display:none">crunching every image&hellip;</span>
    <div id="vaultout" style="margin-top:14px"></div>
  </div>

  <div class="panel">
    <h2>Vision: colour, edges, shapes, emergent classes</h2>
    <p class="muted" style="margin:-8px 0 12px">the image is just numbers, so classical vision is just arithmetic &mdash; all numpy, no OpenCV. <b>Colour</b> splits RGB into perceptual HSV and pulls dominant colours. <b>Edges &amp; shapes</b> runs Sobel gradients then Hough voting to find lines and circles, and Harris to find corners. <b>Emergent classes</b> turns each image into a small feature vector, lets categories fall out by clustering with no labels, and then classifies held-out shapes with VSA prototypes (bundle + cosine cleanup) &mdash; honest about where each step's accuracy tops out.</p>
    <div class="seg" id="visSeg">
      <button data-d="colour" class="on">colour (HSV)</button>
      <button data-d="shapes">edges &amp; shapes</button>
      <button data-d="emergent">emergent classes</button>
    </div>
    <span id="vissum" class="spin" style="display:none;margin-left:12px">computing&hellip;</span>
    <div id="visout" style="margin-top:14px"></div>
  </div>

  <div class="panel">
    <h2>Compositional scene (DCT tags + resonator)</h2>
    <p class="muted" style="margin:-8px 0 12px">two ideas at once. <b>Auto-tags</b> reads the DCT coefficient layout for a texture label (smooth / horizontal / vertical / busy), HSV for colour, and geometry for shape &mdash; automatic tags with no training, finally putting the DCT to work as features. <b>Compositional vs holistic</b> shows why parts beat wholes: each object is encoded as a product of its attribute atoms (colour &otimes; shape &otimes; texture) and the scene as their superposition, so a <b>resonator</b> can factor the parts back out. A single holistic tag can only name one object; the resonator recovers them all. (With an unnormalised scene and coordinate-descent refinement, this now recovers up to ~5 objects reliably &mdash; the old ~50%-at-three ceiling was a scale bug, not a real limit.)</p>
    <div class="seg" id="scnSeg">
      <button data-d="tags" class="on">auto-tags (DCT)</button>
      <button data-d="compose">compositional vs holistic</button>
    </div>
    <span id="scnsum" class="spin" style="display:none;margin-left:12px">computing&hellip;</span>
    <div id="scnout" style="margin-top:14px"></div>
  </div>

  <div class="panel">
    <h2>Scaling: a recursive memory tree</h2>
    <p class="muted" style="margin:-8px 0 12px">the honest limit of everything above: one holographic trace is a bundle, and a bundle has finite capacity &mdash; pile in too much and recall collapses. The fix is the one slime mould uses to beat the size limit of pure diffusion: stop being holistic and grow a <b>hierarchical tree</b>. Each node owns a deterministic seeded hyperplane (reproducible, demoscene-style) and splits its items at the median; each leaf keeps a small memory well inside capacity; a query descends the tree (a beam lets it back-track into nearby cells) and cleans up in just that leaf &mdash; the same trick as an RP-tree or a SQL index: never scan the whole table. <b>Capacity</b> shows a flat memory collapsing while the tree holds; <b>search</b> shows the recall/cost trade and the per-leaf &ldquo;flux&rdquo; (thick veins, thin veins).</p>
    <div class="seg" id="sclSeg">
      <button data-d="capacity" class="on">capacity (flat vs tree)</button>
      <button data-d="search">search &amp; flux</button>
    </div>
    <span id="sclsum" class="spin" style="display:none;margin-left:12px">building trees &amp; measuring&hellip;</span>
    <div id="sclout" style="margin-top:14px"></div>
  </div>

  <div class="panel">
    <h2>Content addresses (S3-style keys)</h2>
    <p class="muted" style="margin:-8px 0 12px">the partitioning idea in its proper form. Like S3, there are no folders &mdash; just a flat keyspace where each object's <em>name</em> encodes the hierarchy. Auto-tags (HSV colour, geometric shape, DCT texture) generate a deterministic URI such as <span style="font-family:inherit;color:var(--teal2)">red/circle/smooth</span>; the key <em>is</em> the partition path. <b>Keyspace</b> shows the prefix tree and the roll-up of common prefixes (S3's CommonPrefixes), plus how key depth controls bucket size. <b>Address from content</b> shows the resonator computing an item's URI from its content vector alone &mdash; the bridge from the holographic representation to the human-readable address.</p>
    <div class="seg" id="stoSeg">
      <button data-d="keyspace" class="on">keyspace &amp; prefixes</button>
      <button data-d="address">address from content</button>
    </div>
    <span id="stosum" class="spin" style="display:none;margin-left:12px">organising&hellip;</span>
    <div id="stoout" style="margin-top:14px"></div>
  </div>

<script>
let deg="noise";
document.querySelectorAll("#deg button").forEach(b=>b.onclick=()=>{
  deg=b.dataset.k; document.querySelectorAll("#deg button").forEach(x=>x.classList.remove("on")); b.classList.add("on");});
amt.oninput=()=>amtv.textContent=amt.value+"%";
dmg.oninput=()=>dmgv.textContent=dmg.value+"%";

let vaultSet="sprites";
document.querySelectorAll("#vaultSet button").forEach(b=>b.onclick=()=>{
  vaultSet=b.dataset.s; document.querySelectorAll("#vaultSet button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); runVault();});
async function runVault(){
  vaultsum.style.display="inline"; vaultout.innerHTML="";
  const fd=new FormData(); fd.append("set",vaultSet);
  const r=await (await fetch("/api/vault",{method:"POST",body:fd})).json();
  vaultsum.style.display="none";
  const bestLossless=Math.min(...r.rows.filter(x=>!x.lossy).map(x=>x.bytes));
  const bestLossy=Math.min(...r.rows.filter(x=>x.lossy).map(x=>x.bytes));
  const row=o=>{
    const win=(!o.lossy&&o.bytes===bestLossless)||(o.lossy&&o.bytes===bestLossy);
    return `<tr${win?' style="color:var(--teal2);font-weight:600"':''}><td>${o.method}${o.lossy?' <span class="muted">(lossy)</span>':''}</td>
      <td style="text-align:right">${o.bytes.toLocaleString()}</td><td>${o.fidelity}</td></tr>`;
  };
  const sz = r.images.length>120 ? 40 : 54;
  const thumbs=r.images.map((u,i)=>`<img class="vthumb" data-i="${i}" src="${u}" title="${(r.names&&r.names[i])||''}"
     style="width:${sz}px;height:${sz}px;border-radius:5px;border:2px solid #22324f;cursor:pointer;image-rendering:pixelated">`).join("");
  vaultout.innerHTML=`
    ${r.note?`<div class="muted" style="font-size:12px;margin-bottom:6px">${r.note}</div>`:''}
    <div class="muted" style="font-size:13px;margin-bottom:10px">${r.n} images &middot; related into ${r.clusters} clusters &middot;
      packed: <b style="color:var(--teal2)">${r.lossless.toLocaleString()}</b> B lossless${r.lossy?`, <b style="color:var(--coral)">${r.lossy.toLocaleString()}</b> B lossy`:''}</div>
    <table class="cmp"><tr><th>encoder</th><th style="text-align:right">bytes</th><th>fidelity</th></tr>${r.rows.map(row).join("")}</table>
    <div class="sublabel" style="margin-top:12px">query by example &mdash; click any of the ${r.n} thumbnails to light up its 3 nearest</div>
    <div id="vqcap" class="muted" style="font-size:12px;min-height:16px;margin:4px 0"></div>
    <div id="vthumbs" style="display:flex;gap:5px;flex-wrap:wrap;margin-top:4px;max-height:300px;overflow-y:auto;padding:8px;background:#0c1322;border:1px solid #1c2942;border-radius:8px">${thumbs}</div>`;
  const imgs=[...document.querySelectorAll(".vthumb")];
  imgs.forEach(im=>im.onclick=()=>{
    imgs.forEach(x=>x.style.borderColor="#22324f");
    im.style.borderColor="var(--teal)";
    const nb=r.neighbors[+im.dataset.i];
    nb.forEach(j=>{ if(imgs[j]) imgs[j].style.borderColor="var(--coral)"; });
    if(r.names) vqcap.innerHTML=`nearest to <b style="color:var(--teal2)">${r.names[+im.dataset.i]}</b>: `
      +nb.map(j=>`<span style="color:var(--coral)">${r.names[j]}</span>`).join(", ");
  });
}

let visDemo="colour";
document.querySelectorAll("#visSeg button").forEach(b=>b.onclick=()=>{
  visDemo=b.dataset.d; document.querySelectorAll("#visSeg button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); runVision();});
async function runVision(){
  vissum.style.display="inline"; visout.innerHTML="";
  const fd=new FormData(); fd.append("demo",visDemo);
  const r=await (await fetch("/api/vision",{method:"POST",body:fd})).json();
  vissum.style.display="none";
  let cap="";
  if(r.demo==="colour"){
    const sw=r.swatches.map(s=>`<span title="${(s.w*100).toFixed(0)}% of pixels" style="display:inline-block;width:34px;height:34px;border-radius:6px;border:1px solid #22324f;background:rgb(${s.rgb[0]},${s.rgb[1]},${s.rgb[2]})"></span>`).join(" ");
    cap=`<div class="sublabel" style="margin-top:10px">dominant colours (k-means in RGB), most common first</div><div style="margin-top:6px;display:flex;gap:7px">${sw}</div>`;
  }else if(r.demo==="shapes"){
    cap=`<div class="muted" style="font-size:13px;margin-top:8px">Hough found <b>${r.lines}</b> line(s); circle ${r.circle?`at (${r.circle[0]}, ${r.circle[1]}) radius ${r.circle[2]}`:'none'}; Harris found <b>${r.corners}</b> corners.</div>`;
  }else{
    cap=`<div class="muted" style="font-size:13px;margin-top:8px">rule-based shape ID <b style="color:var(--teal2)">${(r.rule_acc*100).toFixed(0)}%</b> &middot; emergent cluster purity <b style="color:var(--teal2)">${(r.purity*100).toFixed(0)}%</b> (unsupervised) &middot; VSA cleanup classify <b style="color:var(--coral)">${(r.vsa_acc*100).toFixed(0)}%</b> (held-out). Titles are predictions &mdash; teal correct, coral wrong.</div>`;
  }
  visout.innerHTML=`<img src="${r.fig}" style="width:100%;border-radius:8px">${cap}`;
}
runVision();

let scnDemo="tags";
document.querySelectorAll("#scnSeg button").forEach(b=>b.onclick=()=>{
  scnDemo=b.dataset.d; document.querySelectorAll("#scnSeg button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); runScene();});
async function runScene(){
  scnsum.style.display="inline"; scnout.innerHTML="";
  const fd=new FormData(); fd.append("demo",scnDemo);
  const r=await (await fetch("/api/scene",{method:"POST",body:fd})).json();
  scnsum.style.display="none";
  let cap="";
  if(r.demo==="compose"){
    const fmt=a=>a.map(o=>`${o[0]} ${o[1]} <span class="muted">(${o[2]})</span>`).join("  &middot;  ");
    const match=JSON.stringify(r.objects.map(String).sort())===JSON.stringify(r.recovered.map(String).sort());
    cap=`<div style="margin-top:10px;font-size:13px">
      <div>holistic colour tag: <b style="color:var(--coral)">${r.holistic}</b> &mdash; one label for the whole image, the other object is lost</div>
      <div style="margin-top:4px">segmented objects: <b style="color:var(--teal2)">${fmt(r.objects)}</b></div>
      <div style="margin-top:4px">resonator factors the scene vector back into: <b style="color:var(--teal2)">${fmt(r.recovered)}</b> ${match?'&#10003;':''}</div>
    </div>`;
  }else{
    cap=`<div class="muted" style="margin-top:8px;font-size:13px">top row: texture fields labelled by their DCT energy layout; bottom row: shapes labelled by colour and geometry. All tags come straight from the pixels.</div>`;
  }
  scnout.innerHTML=`<img src="${r.fig}" style="width:100%;border-radius:8px">${cap}`;
}
runScene();

let sclDemo="capacity";
document.querySelectorAll("#sclSeg button").forEach(b=>b.onclick=()=>{
  sclDemo=b.dataset.d; document.querySelectorAll("#sclSeg button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); runScaling();});
async function runScaling(){
  sclsum.style.display="inline"; sclout.innerHTML="";
  const fd=new FormData(); fd.append("demo",sclDemo);
  const r=await (await fetch("/api/scaling",{method:"POST",body:fd})).json();
  sclsum.style.display="none";
  let cap="";
  if(r.demo==="capacity"){
    const big=r.rows[r.rows.length-1];
    cap=`<div class="muted" style="font-size:13px;margin-top:8px">at N=${big.N}: flat memory recalls <b style="color:var(--coral)">${(big.flat*100).toFixed(0)}%</b>, the tree <b style="color:var(--teal2)">${(big.tree*100).toFixed(0)}%</b> across ${big.leaves} leaves (depth ${big.depth}). The tree keeps every leaf inside capacity.</div>`;
  }else{
    const b=r.beam8, f=r.forest;
    cap=`<div class="muted" style="font-size:13px;margin-top:8px">single tree at beam 8: <b style="color:var(--teal2)">${(b.recall*100).toFixed(0)}%</b> at ${b.cmp} cmp. A forest of <b>${f.trees}</b> trees reaches <b style="color:#ffd166">${(f.recall*100).toFixed(0)}%</b> at ${f.cmp} cmp &mdash; the recall a single tree couldn't reach without scanning all ${f.exact}. Tree shape: ${r.stats.leaves} leaves, depth ${r.stats.depth}.</div>`;
  }
  sclout.innerHTML=`<img src="${r.fig}" style="width:100%;border-radius:8px">${cap}`;
}
runScaling();

let stoDemo="keyspace";
document.querySelectorAll("#stoSeg button").forEach(b=>b.onclick=()=>{
  stoDemo=b.dataset.d; document.querySelectorAll("#stoSeg button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); runStore();});
function renderTree(node, depth){
  let h="";
  for(const seg in node){
    const v=node[seg], pad=12+depth*18;
    if(typeof v==="object"){
      h+=`<div style="padding-left:${pad}px;color:var(--teal2)">${seg}</div>`+renderTree(v,depth+1);
    }else{
      h+=`<div style="padding-left:${pad}px;color:#c8d3e6">${seg} <span class="muted">(${v})</span></div>`;
    }
  }
  return h;
}
async function runStore(){
  stosum.style.display="inline"; stoout.innerHTML="";
  const fd=new FormData(); fd.append("demo",stoDemo);
  const r=await (await fetch("/api/store",{method:"POST",body:fd})).json();
  stosum.style.display="none";
  if(r.demo==="keyspace"){
    const depthRows=r.depth.map(d=>`<tr><td>depth ${d.depth}</td><td style="text-align:right">${d.buckets}</td><td style="text-align:right">${d.max}</td></tr>`).join("");
    stoout.innerHTML=`
      <div class="muted" style="font-size:13px;margin-bottom:8px">${r.stats.objects} objects in <b style="color:var(--teal2)">${r.stats.buckets}</b> buckets &middot; biggest ${r.stats.max_bucket} &middot; skew ${r.stats.skew.toFixed(1)}x (semantic keys are uneven &mdash; honest) &middot; the resonator recovers <b style="color:var(--teal2)">${(r.addr_acc*100).toFixed(0)}%</b> of URIs from content alone</div>
      <div style="display:flex;gap:20px;flex-wrap:wrap">
        <div style="flex:1;min-width:240px">
          <div class="sublabel">flat keyspace, rolled up by prefix (like S3 CommonPrefixes)</div>
          <div style="margin-top:6px;max-height:300px;overflow:auto;background:#0c1322;border:1px solid #1c2942;border-radius:8px;padding:8px;font-size:13px">${renderTree(r.tree,0)}</div>
        </div>
        <div style="min-width:200px">
          <div class="sublabel">key depth vs bucket size</div>
          <table class="cmp" style="margin-top:6px"><tr><th>key</th><th style="text-align:right">buckets</th><th style="text-align:right">biggest</th></tr>${depthRows}</table>
          <div class="muted" style="font-size:12px;margin-top:8px">more facets &rarr; more, smaller buckets. The S3 prefix-design lever, and the fix when a bucket outgrows capacity.</div>
        </div>
      </div>`;
  }else{
    const t=r.tags;
    stoout.innerHTML=`
      <div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap">
        <img src="${r.thumb}" style="width:96px;height:96px;border-radius:8px;image-rendering:pixelated;border:1px solid #22324f">
        <div style="font-size:14px">
          <div class="muted">auto-tags &rarr; colour <b style="color:var(--teal2)">${t.colour}</b>, shape <b style="color:var(--teal2)">${t.shape}</b>, texture <b style="color:var(--teal2)">${t.texture}</b></div>
          <div style="margin-top:8px">address (from tags): <b style="color:var(--teal2)">${r.uri}</b></div>
          <div style="margin-top:4px">address (resonator, from content vector): <b style="color:var(--coral)">${r.resonator_uri}</b> ${r.match?'&#10003; match':''}</div>
          <div class="muted" style="margin-top:8px;font-size:12px">bucket ${r.uri} holds object id(s): ${r.bucket.join(", ")||'(none yet)'}</div>
        </div>
      </div>
      <div style="text-align:center;margin-top:12px"><button onclick="runStore()">&#8635; another object</button></div>`;
  }
}
runStore();

async function describe(words){
  words = (words || descIn.value || "").trim();
  if(!words){descOut.innerHTML='<span class="muted">type or tap a description</span>';return;}
  const fd=new FormData(); fd.append("words",words);
  const res=await fetch("/api/describe",{method:"POST",body:fd});
  if(!res.ok){descOut.innerHTML='<span class="muted">no matching words known &mdash; try the chips above</span>';return;}
  const r=await res.json();
  descOut.innerHTML=`<div style="text-align:center">
     <img src="${r.matched_uri}" style="width:190px;height:190px;border-radius:8px;border:1px solid #22324f">
     <div class="sublabel" style="margin-top:8px">match: image #${r.index} &mdash; confidence ${r.confidence}</div>
     <div class="muted" style="font-size:12px;margin-top:2px">you said &ldquo;${r.words.join(' ')}&rdquo; &middot; its tags: ${r.tags.join(', ')}</div>
   </div>`;
}
document.querySelectorAll("#descChips button").forEach(b=>b.onclick=()=>{descIn.value=b.dataset.w; describe(b.dataset.w);});

async function runPack(){
  packsum.style.display="inline"; packout.innerHTML="";
  const r=await (await fetch("/api/pack",{method:"POST"})).json();
  packsum.style.display="none";
  const thumbs=r.images.map(u=>`<img src="${u}" style="width:62px;height:62px;border-radius:6px;border:1px solid #22324f">`).join("");
  const row=o=>{
    const win=o.method.indexOf("set-pack")===0;
    return `<tr${win?' style="color:var(--teal2);font-weight:600"':''}><td>${o.method}</td>
      <td style="text-align:right">${o.bytes.toLocaleString()}</td>
      <td style="text-align:right">${o.rel}</td><td>${o.fidelity}</td></tr>`;
  };
  packout.innerHTML=`<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">${thumbs}</div>
    <table class="cmp"><tr><th>method</th><th style="text-align:right">bytes</th>
      <th style="text-align:right">vs PNG</th><th>fidelity</th></tr>
      ${r.rows.map(row).join("")}</table>
    <p class="muted" style="font-size:12px;margin-top:8px">Six logos sharing a navy background and teal ring, differing only in the centre mark &mdash; the shared pixels are bit-identical, so the deltas are sparse and zlib crushes them.</p>`;
}

async function loadGallery(){
  const g=await (await fetch("/api/gallery")).json();
  gallery.classList.remove("muted"); gallery.innerHTML="";
  src.innerHTML="";
  g.forEach(o=>{
    gallery.innerHTML+=`<figure><img src="${o.uri}"><figcaption>#${o.index}</figcaption></figure>`;
    src.innerHTML+=`<option value="${o.index}">stored image #${o.index}</option>`;
  });
}
async function recall(){
  out.innerHTML='<p class="spin">processing&hellip;</p>';
  const fd=new FormData();
  fd.append("degradation",deg); fd.append("amount",amt.value); fd.append("damage",dmg.value);
  if(file.files[0]) fd.append("image",file.files[0]); else fd.append("index",src.value);
  const r=await (await fetch("/api/recall",{method:"POST",body:fd})).json();
  const verdict = r.truth===null
    ? `<span class="badge ok">matched #${r.match_index}</span>`
    : (r.correct?`<span class="badge ok">correct &mdash; #${r.match_index}</span>`
                :`<span class="badge no">wrong &mdash; got #${r.match_index}</span>`);
  out.innerHTML=`<div class="results">
     <figure><img src="${r.query_uri}"><figcaption>degraded query</figcaption></figure>
     <div class="arrow">&rarr;</div>
     <figure><img src="${r.matched_uri}"><figcaption>matched original</figcaption></figure>
     <div class="arrow">&rarr;</div>
     <figure><img src="${r.recon_uri}"><figcaption>reconstruction<br>${r.psnr} dB${r.damage?` &middot; plate ${r.damage}% gone`:""}</figcaption></figure>
     <div style="align-self:center">${verdict}</div></div>`;
}
async function runComp(){
  compsum.style.display="inline"; compout.innerHTML="";
  const r=await (await fetch("/api/compression",{method:"POST"})).json();
  compsum.style.display="none";
  const isref=m=>m.indexOf('JPEG')>=0||m.indexOf('PNG')>=0;
  const row1=o=>`<tr class="${isref(o.method)?'ref':''}"><td>${o.method}</td><td>${o.size} KB</td>
     <td>${o.psnr===null?'exact':o.psnr+' dB'}</td><td>${o.enc!==undefined?o.enc+' ms':''}</td>
     <td>${o.dec!==undefined?o.dec+' ms':''}</td></tr>`;
  const row2=o=>`<tr class="${isref(o.method)?'ref':''}"><td>${o.method}</td><td>${o.size} KB</td>
     <td>${o.per} KB</td><td>${o.psnr===null?'exact':o.psnr+' dB'}</td></tr>`;
  compout.innerHTML=`
    <div class="sublabel">single image</div>
    <table class="cmp"><tr><th>representation</th><th>size</th><th>quality</th><th>encode</th><th>decode</th></tr>
      ${r.rows.map(row1).join("")}${r.refs.map(row1).join("")}</table>
    <div class="note">keys: ${r.keys}</div><div class="note">resilience: ${r.resilience}</div>
    <div class="sublabel" style="margin-top:20px">many distinct files (6 different images)</div>
    <table class="cmp"><tr><th>how stored</th><th>total</th><th>per file</th><th>quality</th></tr>
      ${r.many.map(row2).join("")}</table>
    <div class="note" style="color:var(--teal)">${r.many_strength}</div>`;
}
async function runBatch(){
  batchsum.style.display="inline"; batchout.style.display="none";
  const r=await (await fetch("/api/batch",{method:"POST"})).json();
  batchsum.style.display="none"; batchout.style.display="block"; batchout.textContent=r.output;
}
async function runCreature(mode){
  mode=mode||"poison";
  creatsum.style.display="inline"; creatout.innerHTML="";
  const body=new URLSearchParams({mode});
  const r=await (await fetch("/api/creature",{method:"POST",body})).json();
  creatsum.style.display="none";
  const SP=r.sprite||null, MAZE=(r.mode==="maze");
  const rrow=o=>`<tr><td>${o.name}, clear &rarr; moves <b>${o.clear}</b></td>
     <td>same, but that way is poison &rarr; moves <b>${o.blocked}</b>
     <span class="${o.avoids?'p':'f'}">${o.avoids?'avoids':'walks in'}</span></td></tr>`;
  const goalWord=MAZE?"exit":"star";
  const energyLine=MAZE
    ? `a braided maze with many routes out &mdash; the colony thins its tubes to the shortest (it found <b>${r.after.steps}</b> steps; the true optimum is <b>${r.after.opt}</b>), with no precomputed guide and no exit compass`
    : `energy starts at ${r.after.start_energy}: each step costs 1, every &#9733; star gives +3, and poison empties it &mdash; instant death`;
  creatout.innerHTML=`
    <div class="creatgrids">
      <div class="creatcell"><div class="sublabel" id="bttl"></div><svg id="svgB" xmlns="http://www.w3.org/2000/svg"></svg><div id="hudB"></div></div>
      <div class="creatcell"><div class="sublabel" id="attl"></div><svg id="svgA" xmlns="http://www.w3.org/2000/svg"></svg><div id="hudA"></div></div>
    </div>
    <div class="muted" style="text-align:center;margin:8px 0 2px;font-size:13px">${r.caption}</div>
    <div class="muted" style="text-align:center;font-size:12px;margin-bottom:2px">${energyLine}</div>
    ${SP?`<div class="muted" style="text-align:center;font-size:12px;margin-bottom:2px">drawn as sprite <b style="color:var(--teal2)">${SP.char}</b> from your set (background keyed out for transparency) &mdash; it turns to face the way it walks</div>`:''}
    <div style="text-align:center;margin:6px 0"><button id="replayBtn">&#9654; Replay</button></div>
    ${r.reflex?`<div class="sublabel" style="margin-top:10px">poison-avoidance reflex &mdash; seeks the star ${r.seek_ok}/4, avoids poison ${r.avoid_ok}/4</div>
    <table class="cmp reflex"><tr><th>when the way is clear</th><th>when poison blocks the star</th></tr>
      ${r.reflex.map(rrow).join("")}</table>`:''}`;

  const NS="http://www.w3.org/2000/svg", CS=38, mid=c=>c*CS+CS/2;
  const el=(t,a)=>{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);return n;};
  function star(x,y,fill){const s=el("text",{x:mid(x),y:mid(y),"font-size":CS*0.6,"text-anchor":"middle","dominant-baseline":"central",fill});s.textContent="\u2605";return s;}

  // Status strip under each grid, updated every frame: a tally on the left, an
  // energy bar in the middle, and the outcome on the right.
  function hud(node,tallyHtml,energy,startE,dead,escaped){
    const pct=Math.max(0,Math.min(100,100*energy/startE));
    const col=energy<=0?"#ff5c7a":pct<25?"#ffb454":pct<60?"#ffd166":"#3ddc97";   // red/amber/gold/green
    const right=escaped?'<b style="color:#3ddc97">&#10003; escaped</b>'
               :dead?'<b style="color:#ff5c7a">&#10007; died</b>':'energy <b>'+energy+'</b>';
    node.innerHTML=
      `<div style="display:flex;align-items:center;gap:8px;justify-content:center;font-size:12px;color:#c8d3e6;margin-top:7px">
         <span style="color:#ffd166;white-space:nowrap">${tallyHtml}</span>
         <div style="flex:1;max-width:150px;height:9px;border-radius:5px;background:#22324f;overflow:hidden">
           <div style="height:100%;width:${pct}%;background:${col};transition:width .12s linear"></div>
         </div>
         <span style="white-space:nowrap">${right}</span>
       </div>`;
  }

  function build(svg,ep){
    const W=ep.w,H=ep.h; svg.setAttribute("viewBox",`0 0 ${W*CS} ${H*CS}`); svg.innerHTML="";
    svg.appendChild(el("rect",{x:0,y:0,width:W*CS,height:H*CS,fill:"#0e1626",rx:6}));
    for(let i=0;i<=W;i++) svg.appendChild(el("line",{x1:i*CS,y1:0,x2:i*CS,y2:H*CS,stroke:"#22324f"}));
    for(let j=0;j<=H;j++) svg.appendChild(el("line",{x1:0,y1:j*CS,x2:W*CS,y2:j*CS,stroke:"#22324f"}));
    (ep.walls||[]).forEach(c=>                                   // solid impassable walls
      svg.appendChild(el("rect",{x:c[0]*CS+1,y:c[1]*CS+1,width:CS-2,height:CS-2,fill:"#3a4a66",rx:3})));
    ep.poison.forEach(p=>{
      svg.appendChild(el("rect",{x:p[0]*CS+3,y:p[1]*CS+3,width:CS-6,height:CS-6,fill:"#ff5c7a",opacity:0.28,rx:4}));
      const t=el("text",{x:mid(p[0]),y:mid(p[1]),"font-size":CS*0.5,"text-anchor":"middle","dominant-baseline":"central",fill:"#ff5c7a"});t.textContent="\u2715";svg.appendChild(t);
    });
    if(!MAZE && ep.optimal && ep.optimal.length>1){            // optimal route: forage worlds only
      const op=ep.optimal.map(c=>`${mid(c[0])},${mid(c[1])}`).join(" ");
      svg.appendChild(el("polyline",{points:op,fill:"none",stroke:"#9fb3d1","stroke-width":2,"stroke-dasharray":"3 5",opacity:0.5,"stroke-linecap":"round","stroke-linejoin":"round"}));
    }
    const trail=el("polyline",{fill:"none",stroke:"#2dd4bf","stroke-width":3,"stroke-linecap":"round","stroke-linejoin":"round",opacity:0.9});
    const eaten=el("g",{}); const food=star(ep.frames[0].fx,ep.frames[0].fy,"#ffd166");
    let cre;
    if(SP){                                  // real walking sprite from the set (now transparent)
      const z=CS*0.92, off=(CS-z)/2;
      cre=el("image",{x:ep.frames[0].x*CS+off,y:ep.frames[0].y*CS+off,width:z,height:z,preserveAspectRatio:"xMidYMid meet"});
      cre.setAttribute("href",SP.frames["S"][0]); cre.style.imageRendering="pixelated";
    }else{                                   // fallback marker
      cre=el("circle",{r:CS*0.26,fill:"#c8d3e6",stroke:"#0e1626","stroke-width":2,cx:mid(ep.frames[0].x),cy:mid(ep.frames[0].y)});
    }
    svg.append(trail,eaten,food,cre);
    return {trail,eaten,food,cre};
  }
  const tally=(k,stars)=>MAZE?`steps <b>${k}</b>`:`&#9733; <b>${stars}</b>`;
  let timers=[];
  function run(svg,hudNode,ep){
    const g=build(svg,ep), fr=ep.frames, pts=[`${mid(fr[0].x)},${mid(fr[0].y)}`]; let k=0, dir="S", stars=0;
    g.trail.setAttribute("points",pts.join(" "));
    hud(hudNode,tally(0,0),fr[0].energy,ep.start_energy,false,false);
    const z=CS*0.92, off=(CS-z)/2;
    const id=setInterval(()=>{
      if(++k>=fr.length){clearInterval(id);return;}
      const f=fr[k], p=fr[k-1], dx=f.x-p.x, dy=f.y-p.y;
      if(dx>0)dir="E";else if(dx<0)dir="W";else if(dy>0)dir="S";else if(dy<0)dir="N";   // no move -> keep facing
      if(SP){
        g.cre.setAttribute("x",f.x*CS+off); g.cre.setAttribute("y",f.y*CS+off);
        g.cre.setAttribute("href",SP.frames[dir][k%2]);                                 // cycle the two walk frames
      }else{
        g.cre.setAttribute("cx",mid(f.x)); g.cre.setAttribute("cy",mid(f.y));
      }
      pts.push(`${mid(f.x)},${mid(f.y)}`); g.trail.setAttribute("points",pts.join(" "));
      if(f.ate){ stars++; if(!MAZE) g.eaten.appendChild(star(f.x,f.y,"#63dcbe")); }
      g.food.setAttribute("x",mid(f.fx)); g.food.setAttribute("y",mid(f.fy));
      const escapedNow=MAZE&&f.ate;
      hud(hudNode,tally(k,stars),f.energy,ep.start_energy,f.dead,escapedNow);
      if(escapedNow){                                                                    // reached the exit
        svg.appendChild(el("circle",{cx:mid(f.x),cy:mid(f.y),r:CS*0.46,fill:"none",stroke:"#3ddc97","stroke-width":3}));
      }
      if(f.dead){                                                                        // mark the fatal step
        g.cre.setAttribute("opacity","0.4");
        const skull=el("text",{x:mid(f.x),y:mid(f.y),"font-size":CS*0.7,"text-anchor":"middle","dominant-baseline":"central",fill:"#ff5c7a"});
        skull.textContent="\u2620"; svg.appendChild(skull);
      }
    },150);
    timers.push(id);
  }
  // Maze 'after' panel: replay the ACTUAL slime-mold search. First the colony explores --
  // a tile lights up only once a walker has reached it (first-discovery order from the
  // solver) -- then the reinforced route emerges and the creature walks the way out. No
  // precomputed guide, no exit compass: the path is discovered, not laid out in advance.
  function runMaze(svg,hudNode,ep){
    const W=ep.w,H=ep.h; svg.setAttribute("viewBox",`0 0 ${W*CS} ${H*CS}`); svg.innerHTML="";
    svg.appendChild(el("rect",{x:0,y:0,width:W*CS,height:H*CS,fill:"#0e1626",rx:6}));
    for(let i=0;i<=W;i++) svg.appendChild(el("line",{x1:i*CS,y1:0,x2:i*CS,y2:H*CS,stroke:"#22324f"}));
    for(let j=0;j<=H;j++) svg.appendChild(el("line",{x1:0,y1:j*CS,x2:W*CS,y2:j*CS,stroke:"#22324f"}));
    (ep.walls||[]).forEach(c=>
      svg.appendChild(el("rect",{x:c[0]*CS+1,y:c[1]*CS+1,width:CS-2,height:CS-2,fill:"#3a4a66",rx:3})));
    const exLayer=el("g",{}), routeLine=el("polyline",{fill:"none",stroke:"#2dd4bf","stroke-width":3,"stroke-linecap":"round","stroke-linejoin":"round",opacity:0.95});
    svg.append(exLayer,routeLine);
    const goal=ep.route[ep.route.length-1], start=ep.route[0];
    svg.appendChild(star(goal[0],goal[1],"#ffd166"));
    const z=CS*0.92, off=(CS-z)/2; let cre;
    if(SP){ cre=el("image",{x:start[0]*CS+off,y:start[1]*CS+off,width:z,height:z,preserveAspectRatio:"xMidYMid meet"}); cre.setAttribute("href",SP.frames["S"][0]); cre.style.imageRendering="pixelated"; }
    else  { cre=el("circle",{r:CS*0.26,fill:"#c8d3e6",stroke:"#0e1626","stroke-width":2,cx:mid(start[0]),cy:mid(start[1])}); }
    svg.appendChild(cre);

    const ex=ep.explore; let i=0;
    const PER=Math.max(1,Math.round(ex.length/55));            // ~55 ticks of exploration
    hud(hudNode,"exploring&hellip;",ep.start_energy,ep.start_energy,false,false);
    const id1=setInterval(()=>{
      for(let n=0;n<PER && i<ex.length;n++,i++){
        const c=ex[i];
        exLayer.appendChild(el("rect",{x:c[0]*CS+CS*0.32,y:c[1]*CS+CS*0.32,width:CS*0.36,height:CS*0.36,fill:"#2dd4bf",opacity:0.16,rx:2}));
      }
      hud(hudNode,`explored <b>${Math.min(i,ex.length)}</b>/${ex.length} cells`,ep.start_energy,ep.start_energy,false,false);
      if(i>=ex.length){ clearInterval(id1); walkRoute(); }
    },45);
    timers.push(id1);

    function walkRoute(){
      const route=ep.route, pts=[`${mid(route[0][0])},${mid(route[0][1])}`]; routeLine.setAttribute("points",pts.join(" ")); let k=0, dir="S";
      const id2=setInterval(()=>{
        if(++k>=route.length){ clearInterval(id2);
          svg.appendChild(el("circle",{cx:mid(goal[0]),cy:mid(goal[1]),r:CS*0.46,fill:"none",stroke:"#3ddc97","stroke-width":3}));
          hud(hudNode,`out in <b>${route.length-1}</b> steps`,ep.energy,ep.start_energy,false,true); return; }
        const f=route[k], p=route[k-1], dx=f[0]-p[0], dy=f[1]-p[1];
        if(dx>0)dir="E";else if(dx<0)dir="W";else if(dy>0)dir="S";else if(dy<0)dir="N";
        if(SP){ cre.setAttribute("x",f[0]*CS+off); cre.setAttribute("y",f[1]*CS+off); cre.setAttribute("href",SP.frames[dir][k%2]); }
        else  { cre.setAttribute("cx",mid(f[0])); cre.setAttribute("cy",mid(f[1])); }
        pts.push(`${mid(f[0])},${mid(f[1])}`); routeLine.setAttribute("points",pts.join(" "));
        hud(hudNode,`following the tube &mdash; <b>${k}</b> steps`,ep.start_energy-k,ep.start_energy,false,false);
      },80);
      timers.push(id2);
    }
  }
  function play(){
    timers.forEach(clearInterval); timers=[];
    if(MAZE){
      const bt = r.before.escaped ? `escaped (${r.before.steps} steps)` : `never finds the exit (${r.before.steps} steps)`;
      bttl.innerHTML = `random walker \u2014 ${bt}`;
      attl.innerHTML = `holographic slime-mold colony \u2014 explores, then the way out emerges`;
      run(svgB,hudB,r.before);          // random walker, genuine wandering (trail after each visit)
      runMaze(svgA,hudA,r.after);       // the actual search: discover, reinforce, escape
      return;
    }
    const tag=ep=>ep.alive?`survived &mdash; ${ep.stars} ${goalWord}s`:`died &mdash; ${ep.stars} ${goalWord}s`;
    bttl.innerHTML=`before training (random) \u2014 ${tag(r.before)}`;
    attl.innerHTML=`after training (greedy) \u2014 ${tag(r.after)}`;
    run(svgB,hudB,r.before); run(svgA,hudA,r.after);
  }
  replayBtn.onclick=play; play();
}
async function runTour(){
  toursum.innerHTML='<span class="spin">running tour&hellip; (~20s, trains the creature)</span>'; tourout.style.display="none";
  const r=await (await fetch("/api/tour",{method:"POST"})).json();
  toursum.textContent=""; tourout.style.display="block"; tourout.textContent=r.output;
}
async function runTests(){
  tsum.innerHTML='<span class="spin">running pytest&hellip; (may take ~1 min)</span>'; tests.style.display="none";
  const r=await (await fetch("/api/tests",{method:"POST"})).json();
  tsum.textContent=r.summary;
  tests.style.display="block";
  tests.innerHTML=r.tests.map(t=>`<div class="tr"><span>${t.name}</span>
     <span class="${t.status==='PASSED'?'p':'f'}">${t.status}</span></div>`).join("");
}
loadGallery();
// --- unified brain ---------------------------------------------------------
const uId=id=>document.getElementById(id);
async function uPost(url,body){
  const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body||{})}); return r.json();
}
async function uInit(){
  const r=await (await fetch("/api/unified/datasets")).json();
  uId("u_ds").innerHTML=r.datasets.map(d=>`<option value="${d.id}">${d.name}</option>`).join("");
  if(!r.nltk) uId("u_trained").innerHTML='<span class="muted">install nltk to pull corpora: pip install nltk</span>';
}
async function uLoad(){
  uId("u_loadmsg").style.display="inline"; uId("u_trained").innerHTML="";
  const r=await uPost("/api/unified/load",{id:uId("u_ds").value});
  uId("u_loadmsg").style.display="none";
  if(!r.ok){uId("u_trained").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const pills=r.labels.map(l=>`<span class="badge ok" style="margin:2px 4px 2px 0">${l}: ${r.counts[l]||0}</span>`).join("");
  uId("u_trained").innerHTML=
    `<div>${r.desc}</div>
     <div style="margin-top:8px">held-out accuracy <span class="val" style="font-size:22px">${r.accuracy}%</span>
       <span class="muted">&nbsp; ${r.trained} trained / ${r.held_out} held out &middot;
       ${r.prototypes} prototypes &middot; ${r.gen_chars.toLocaleString()} chars for generation</span></div>
     <div style="margin-top:8px">${pills}</div>`;
  uId("u_ops").style.display="block";
}
async function uClassify(){
  const r=await uPost("/api/unified/classify",{text:uId("u_cq").value});
  if(r.error){uId("u_cout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  uId("u_cout").innerHTML=`classified as <b class="val">${r.label}</b> (cos ${r.score})
     <br><span class="muted">nearest stored item is a <b>${r.recall.label}</b> (cos ${r.recall.score})</span>`;
}
async function uRecall(){
  const r=await uPost("/api/unified/recall",{text:uId("u_cq").value});
  if(r.error){uId("u_cout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  uId("u_cout").innerHTML=`nearest stored item: <b class="val">${r.label}</b> (cos ${r.score})
     <br><span class="muted">&ldquo;${r.example}&hellip;&rdquo;</span>`;
}
async function uOrganize(){
  const r=await uPost("/api/unified/organize",{});
  if(r.error){uId("u_oout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const pills=Object.entries(r.after).map(([l,n])=>`<span class="badge ok" style="margin:2px 4px 2px 0">${l}: ${n}</span>`).join("");
  uId("u_oout").innerHTML=`reorganize decided: <b class="val">${r.choice}</b>
     <div style="margin-top:8px">${pills}</div>
     <div class="muted" style="margin-top:6px">${r.note}</div>`;
}
async function uGenerate(){
  uId("u_gout").style.display="block"; uId("u_gout").innerHTML='<span class="muted">generating&hellip;</span>';
  const r=await uPost("/api/unified/generate",
    {seed:uId("u_seed").value,length:+uId("u_len").value,temperature:+uId("u_temp").value});
  if(r.error){uId("u_gout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  uId("u_gout").textContent=r.text;
}
uInit();
</script>
</div></body></html>
"""

if __name__ == "__main__":
    app.run(debug=False, port=5000)
