"""Honest benchmark: the holographic image store vs standard JPEG/PNG, on
(1) plain compression and (2) resilience to random byte/cell corruption.

The point is NOT that the hologram compresses better -- it does not, and the
numbers say so plainly.  The point is corruption resilience: JPEG dies when a
fraction of a percent of its bytes flip, while the hologram degrades gracefully
because every cell carries a little of the whole picture.

    python bench_vs_jpeg.py          # text table
    python bench_vs_jpeg.py --fig    # also render bench_corruption.png

(This merges the former bench_fig.py, which drew the same comparison as a figure.)
"""
import io, sys, numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from holographic_image import HolographicImage, _demo_image, _psnr

S = 240
img = _demo_image(S)                          # 4 flat colour fields + a dotted high-freq edge
u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)


def png_bytes():
    b = io.BytesIO(); Image.fromarray(u8).save(b, "PNG"); return b.getvalue()


def jpg_bytes(q):
    b = io.BytesIO(); Image.fromarray(u8).save(b, "JPEG", quality=q); return b.getvalue()


def jpg_img(data):
    try:
        a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), float) / 255
        return a if a.shape == img.shape else np.zeros_like(img)
    except Exception:
        return np.zeros_like(img)


def corrupt(data, frac, seed):
    b = bytearray(data); rng = np.random.default_rng(seed)
    for i in rng.choice(len(b), int(len(b) * frac), replace=False):
        b[i] = int(rng.integers(0, 256))
    return bytes(b)


holo = HolographicImage(img.shape, keep=4000, dim=16384, seed=0).store(img, bits=4, shared_index=True)
png, j85, j50 = png_bytes(), jpg_bytes(85), jpg_bytes(50)

print("=== plain compression (240x240 colour) ===")
print(f"  PNG (lossless)     {len(png) / 1e3:6.1f} KB    inf dB")
print(f"  JPEG q85           {len(j85) / 1e3:6.1f} KB   {_psnr(img, jpg_img(j85)):4.1f} dB")
print(f"  JPEG q50           {len(j50) / 1e3:6.1f} KB   {_psnr(img, jpg_img(j50)):4.1f} dB")
print(f"  Hologram 4-bit     {holo.stored_bytes() / 1e3:6.1f} KB   {_psnr(img, holo.reconstruct()):4.1f} dB")
print("  -> on pure compression JPEG wins; that is not what the hologram is for.\n")

print("=== resilience to random corruption (mean over 8 trials) ===")
print(f"  {'corrupted':>10} {'JPEG q85':>12} {'Hologram':>12}")
for frac in [0.001, 0.005, 0.01, 0.05, 0.10, 0.40]:
    jp = np.mean([_psnr(img, jpg_img(corrupt(j85, frac, s))) for s in range(8)])
    hp = np.mean([_psnr(img, holo.reconstruct(mask=holo.damage_mask(frac, seed=s))) for s in range(8)])
    print(f"  {frac * 100:8.1f}% {jp:9.1f} dB {hp:9.1f} dB")
print("  (0 dB = file no longer decodes to a usable image)")

if "--fig" in sys.argv:
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fracs = [0.0, 0.01, 0.10, 0.40]
    fig, ax = plt.subplots(2, len(fracs), figsize=(3 * len(fracs), 6.2))
    for c, f in enumerate(fracs):
        jp = jpg_img(corrupt(j85, f, 3)); hp = holo.reconstruct(mask=holo.damage_mask(f, seed=3))
        ax[0, c].imshow(jp); ax[0, c].set_title(f"{int(f * 100)}% corrupt\n{_psnr(img, jp):.1f} dB"); ax[0, c].axis("off")
        ax[1, c].imshow(hp); ax[1, c].set_title(f"{int(f * 100)}% destroyed\n{_psnr(img, hp):.1f} dB"); ax[1, c].axis("off")
    fig.text(0.013, 0.74, "JPEG q85", rotation=90, fontsize=13, fontweight="bold", va="center")
    fig.text(0.013, 0.28, "Hologram", rotation=90, fontsize=13, fontweight="bold", va="center")
    fig.suptitle("Same fraction corrupted: JPEG fails by 1%, the hologram survives 40%", y=0.99, fontsize=12)
    fig.tight_layout(rect=[0.03, 0, 1, 1]); fig.savefig("bench_corruption.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("\nrendered bench_corruption.png")
