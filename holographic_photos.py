"""Loading and testing the holographic image stack on REAL photographs.

Until now the image work was measured on GIF sprites -- palette images, <=88
colours, where lossless shared-palette packing wins. Photographs are the
opposite regime: thousands of colours, continuous tone, no palette. This module
loads a folder of real images (downsampled to a uniform working size) and is the
honest test bed for the lossy DCT coder, the vault's adaptive chooser, and the
holographic plate's robustness on data it was NOT designed around.

WHAT THE PHOTOS TAUGHT (measured on a 'mountain' wallpaper category, 256x256
luminance):
  * JPEG BEATS OUR DCT CODER ON EFFICIENCY, and we say so. HolographicImage keeps
    the top-K GLOBAL DCT coefficients in a fixed-size plate; JPEG uses 8x8 block
    DCT with entropy-coded coefficients. JPEG reaches ~31 dB at ~2 KB where the
    holographic coder needs ~5 KB. The plate is not a competitive photo codec and
    is not claimed to be.
  * BUT THE PLATE IS ROBUST WHERE JPEG IS BRITTLE -- the real, honest win. Erase a
    random 50% of the holographic coefficients and PSNR does not move (28.7 dB at
    0% and at 50%): every coefficient carries a little of the whole image, so
    partial loss degrades nothing. JPEG loses ~15 dB after a 10% byte loss and
    often fails to decode at all. Graceful degradation is the property the
    distributed representation buys, and photographs show it as clearly as
    sprites did.
  * THE VAULT'S ADAPTIVE CHOOSER GENERALIZES. On sprites it chose shared-palette
    + LZMA (lossless, low-colour); on 12k-colour photos that path is correctly
    unavailable and the chooser picks the lossy WebP/JPEG encoder (WebP ~90 KB /
    38 dB beating JPEG here). Same code, opposite verdict, driven entirely by the
    data -- the validation that 'measure every encoder, keep the smallest, report
    honestly' was the right design.
  * THE ORIENTATION PRINCIPLE STAYS IDLE, correctly. Transposing photo planes
    does not help (it costs ~4%), because photographs have no column-vs-row self-
    similarity the way character sprites do -- so the vault keeps row-major. The
    principle applies only where the data has directional structure, and the
    measurement is what says so.

Needs: numpy, PIL.
"""
import glob
import os

import numpy as np
from PIL import Image


def load_photo_folder(folder, size=256, limit=None, gray=False):
    """Load images under `folder`, center-fit to size x size. Returns a list of
    uint8 arrays (HxWx3, or HxW if gray). Real photos are large and varied; the
    uniform downsample makes them a clean, manageable test set."""
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg"))
                   + glob.glob(os.path.join(folder, "*.jpeg"))
                   + glob.glob(os.path.join(folder, "*.png"))
                   + glob.glob(os.path.join(folder, "*.npy")))
    if limit:
        paths = paths[:limit]
    out = []
    for p in paths:
        if p.endswith(".npy"):
            a = np.load(p)
            if a.shape[:2] != (size, size):
                a = np.asarray(Image.fromarray(a).resize((size, size), Image.LANCZOS), np.uint8)
        else:
            im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
            a = np.asarray(im, np.uint8)
        if gray:
            a = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.uint8)
        out.append(a)
    return out


def robustness_curve(gray_image, keeps_erasures=((1500, (0.0, 0.1, 0.3, 0.5)),), dim=4096):
    """Measure the holographic plate's PSNR vs random-erasure fraction on one
    grayscale image (values in 0..255 or 0..1). Returns {erase_fraction: psnr}.
    The signature property: PSNR is near-flat across erasure, unlike a block codec."""
    from holographic_image import HolographicImage, _psnr
    g = gray_image.astype(np.float64)
    if g.max() > 1.0:
        g = g / 255.0
    out = {}
    keep, erasures = keeps_erasures[0]
    h = HolographicImage(g.shape, keep=keep, dim=dim, seed=0).store(g, bits=8)
    for e in erasures:
        mask = h.damage_mask(e) if e > 0 else None
        out[e] = float(_psnr(g, h.reconstruct(mask)))
    return out
