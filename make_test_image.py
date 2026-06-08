"""Asymmetric four-colour test image with a black dotted outline.
  upper-left = red, upper-right = blue, bottom-left = yellow, bottom-right = green
  overlaid: a black dotted line tracing an asymmetric closed shape."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

S = 400
img = np.zeros((S, S, 3))
h = S // 2
img[0:h, 0:h] = [1, 0, 0]      # upper-left  : red
img[0:h, h:S] = [0, 0, 1]      # upper-right : blue
img[h:S, 0:h] = [1, 1, 0]      # bottom-left : yellow
img[h:S, h:S] = [0, 1, 0]      # bottom-right: green

# --- a clearly asymmetric, irregular closed shape (no reflection symmetry) ---
rng = np.random.default_rng(7)
t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
R = 0.27 * S
r = np.ones_like(t)
for k in range(1, 7):                         # mismatched harmonics -> lopsided blob
    r += (0.34 / k) * np.sin(k * t + rng.uniform(0, 2 * np.pi))
r *= R
cx, cy = 0.47 * S, 0.52 * S     # off-centre, so it straddles the quadrants unevenly
x, y = cx + r * np.cos(t), cy + r * np.sin(t)

# --- resample the curve by arc length so the dots are evenly spaced ---
dx, dy = np.diff(x, append=x[0]), np.diff(y, append=y[0])
ds = np.hypot(dx, dy)
s = np.concatenate([[0], np.cumsum(ds)[:-1]])
total = s[-1] + ds[-1]
spacing, dot_r = 17.0, 4         # dot spacing and radius in pixels
dot_s = np.arange(0, total, spacing)
xd, yd = np.interp(dot_s, s, x), np.interp(dot_s, s, y)

yy, xx = np.mgrid[0:S, 0:S]
for cxi, cyi in zip(xd, yd):
    img[(xx - cxi) ** 2 + (yy - cyi) ** 2 <= dot_r ** 2] = [0, 0, 0]

plt.imsave("test_image.png", np.clip(img, 0, 1))
np.save("test_image.npy", np.clip(img, 0, 1))
print(f"saved test_image.png ({S}x{S}); {len(xd)} dots on the outline")
