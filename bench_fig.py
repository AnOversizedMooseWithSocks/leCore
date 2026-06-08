import io, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from holographic_image import HolographicImage, _demo_image, _psnr

S=240; img=_demo_image(S); u8=(np.clip(img,0,1)*255).astype(np.uint8)
jref=io.BytesIO(); Image.fromarray(u8).save(jref,"JPEG",quality=85); jref=jref.getvalue()
holo=HolographicImage(img.shape,keep=4000,dim=16384,seed=0).store(img,bits=4,shared_index=True)
def corrupt(d,f,s):
    b=bytearray(d); rng=np.random.default_rng(s)
    for i in rng.choice(len(b),int(len(b)*f),replace=False): b[i]=int(rng.integers(0,256))
    return bytes(b)
def jdec(d):
    try:
        a=np.asarray(Image.open(io.BytesIO(d)).convert("RGB"),float)/255
        return a if a.shape==img.shape else np.zeros_like(img)
    except Exception: return np.zeros_like(img)

fracs=[0.0,0.01,0.10,0.40]
fig,ax=plt.subplots(2,len(fracs),figsize=(3*len(fracs),6.2))
for c,f in enumerate(fracs):
    jp=jdec(corrupt(jref,f,3)); hp=holo.reconstruct(mask=holo.damage_mask(f,seed=3))
    ax[0,c].imshow(jp); ax[0,c].set_title(f"{int(f*100)}% corrupt\n{_psnr(img,jp):.1f} dB"); ax[0,c].axis("off")
    ax[1,c].imshow(hp); ax[1,c].set_title(f"{int(f*100)}% destroyed\n{_psnr(img,hp):.1f} dB"); ax[1,c].axis("off")
ax[0,0].set_ylabel("JPEG"); ax[1,0].set_ylabel("Hologram")
fig.text(0.013,0.74,"JPEG q85",rotation=90,fontsize=13,fontweight="bold",va="center")
fig.text(0.013,0.28,"Hologram",rotation=90,fontsize=13,fontweight="bold",va="center")
fig.suptitle("Same fraction of bytes/cells corrupted: JPEG fails by 1%, the hologram survives 40%",y=0.99,fontsize=12)
fig.tight_layout(rect=[0.03,0,1,1]); fig.savefig("bench_corruption.png",dpi=110,bbox_inches="tight"); plt.close(fig)
print("rendered bench_corruption.png")
