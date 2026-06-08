"""Honest benchmark: holographic store vs standard JPEG/PNG, on (1) plain
compression and (2) resilience to random byte/cell corruption."""
import io, numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from holographic_image import HolographicImage, _demo_image, _psnr

S = 240
img = _demo_image(S)                      # 4 flat colour fields + a dotted high-freq edge
u8  = (np.clip(img,0,1)*255).astype(np.uint8)

def png_bytes(): b=io.BytesIO(); Image.fromarray(u8).save(b,"PNG"); return b.getvalue()
def jpg_bytes(q): b=io.BytesIO(); Image.fromarray(u8).save(b,"JPEG",quality=q); return b.getvalue()
def jpg_psnr(data):
    try:
        a=np.asarray(Image.open(io.BytesIO(data)).convert("RGB"),float)/255
        if a.shape!=img.shape: return 0.0
        return _psnr(img,a)
    except Exception:
        return 0.0

# --- baseline (no corruption) ---
holo = HolographicImage(img.shape, keep=4000, dim=16384, seed=0).store(img, bits=4, shared_index=True)
png=png_bytes(); j85=jpg_bytes(85); j50=jpg_bytes(50)
print("=== plain compression (240x240 colour) ===")
print(f"  PNG (lossless)     {len(png)/1e3:6.1f} KB   (inf dB)")
print(f"  JPEG q85           {len(j85)/1e3:6.1f} KB   {jpg_psnr(j85):4.1f} dB")
print(f"  JPEG q50           {len(j50)/1e3:6.1f} KB   {jpg_psnr(j50):4.1f} dB")
print(f"  Hologram 4-bit     {holo.stored_bytes()/1e3:6.1f} KB   {_psnr(img,holo.reconstruct()):4.1f} dB")
print("  -> on pure compression JPEG wins; that is not what the hologram is for.\n")

# --- resilience: corrupt the SAME fraction of each representation ---
def corrupt(data, frac, seed):
    b=bytearray(data); rng=np.random.default_rng(seed)
    idx=rng.choice(len(b), int(len(b)*frac), replace=False)
    for i in idx: b[i]=int(rng.integers(0,256))
    return bytes(b)

jpeg_ref=jpg_bytes(85)
print("=== resilience to random corruption (mean over 8 trials) ===")
print(f"  {'corrupted':>10} {'JPEG q85':>12} {'Hologram':>12}")
for frac in [0.001,0.005,0.01,0.05,0.10,0.40]:
    jp=np.mean([jpg_psnr(corrupt(jpeg_ref,frac,s)) for s in range(8)])
    hp=np.mean([_psnr(img,holo.reconstruct(mask=holo.damage_mask(frac,seed=s))) for s in range(8)])
    print(f"  {frac*100:8.1f}% {jp:9.1f} dB {hp:9.1f} dB")
print("  (0 dB = file no longer decodes to a usable image)")
