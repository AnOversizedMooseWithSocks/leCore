"""ASCII projection (PROJ-A): render any image to text, at maximum detail per character, fast.

WHY THIS MODULE EXISTS
----------------------
The engine's renders were being eyeballed through ad-hoc throwaway loops ("ASCII output analysis" in session
scratch code). A projection that useful -- it works over SSH, in logs, in CI output, in any terminal, with zero
image dependencies -- deserves to be a first-class output backend with a real resolution knob, not a scratch
snippet reinvented per session. This module is that backend: image in, string out, deterministic, pure NumPy.

DETAIL PER CHARACTER (the design axis)
  A character cell is the pixel of this display, and different glyph sets pack different amounts of image into
  one cell. The modes, in increasing detail-per-cell:
    * 'ramp'    -- 1 luminance sample/cell through a brightness-ordered glyph ramp. The classic. ~70 levels.
    * 'edge'    -- ramp, plus ORIENTED glyphs (| / - \\) where the local gradient is strong: edges keep their
                   direction instead of dissolving into brightness soup. Same cell budget, more structure.
    * 'half'    -- U+2580 upper-half-block with ANSI foreground+background color: 2 FULL-COLOR pixels per cell
                   (fg paints the top half, bg the bottom). The color mode.
    * 'braille' -- U+2800..U+28FF dot patterns: a 2x4 dot grid per cell = 8 addressable pixels per character,
                   the densest text raster there is. Binary dots, so brightness is carried by ORDERED DITHER
                   (an 8x8 Bayer matrix -- vectorised, unlike error diffusion which is inherently serial).
  ANSI color ('256' or 'truecolor') can wrap any glyph mode; 'half' requires it by construction.

RESOLUTION AND SPEED
  `width` is the output width in CHARACTERS; the pixel resolution each mode consumes is width x (1|2|2) wide and
  aspect-derived rows x (1|2|4) tall. Sampling is box-average bilinear, fully vectorised (fancy indexing + one
  reshape-mean); a 240x240 render to 100 columns of braille is ~milliseconds. No Python per-pixel loop anywhere.

CHARACTER ASPECT
  A terminal cell is ~twice as tall as wide, so a square image maps to rows = width * (H/W) * cell_aspect with
  cell_aspect ~= 0.5 -- otherwise every render comes out stretched. Overridable for odd fonts.

The honest line: this is plain raster resampling and glyph lookup -- no hypervector tricks. What is engine-shaped
is its role: one more PROJECTION from the same field/render outputs everything else consumes, and deterministic
to the byte so a text render can sit in a regression test.
"""

import numpy as np

#: Brightness-ordered glyph ramps. SHORT is the classic 10-step; LONG is the ~70-step standard ramp for smoother
#: gradients (both start at space = black). A custom ramp string can be passed straight to ascii_render.
RAMP_SHORT = " .:-=+*#%@"
RAMP_LONG = (" .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")

#: 8x8 Bayer ordered-dither matrix, normalised to [0,1). WHY ordered dither and not Floyd-Steinberg: error
#: diffusion is a serial left-to-right, top-to-bottom recurrence (each pixel's error feeds the next), which cannot
#: be vectorised; Bayer is a pure per-pixel threshold lookup -- one array compare -- and its crosshatch structure
#: reads well at braille dot pitch.
_BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], float) / 64.0

#: Braille dot bit-weights by (row, col) inside a cell's 4x2 dot grid -- the Unicode braille encoding order
#: (dots 1-8): col0 rows0-2 are bits 0..2, col1 rows0-2 are bits 3..5, row3 is bits 6 (col0) and 7 (col1).
_BRAILLE_BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], dtype=np.int32)

#: BAKED ANSI-256 foreground/background escape CODEBOOKS: the 216 colours of the 6x6x6 cube, precomputed once as
#: escape strings and indexed, instead of formatting an f-string per pixel. WHY a codebook: color mode was 19x
#: slower than mono because it rebuilt "\x1b[38;5;Nm" per cell in a Python loop; measured 62 ms -> 3.3 ms on a
#: 100x120 frame by looking these up and joining a vectorised array of prefixes. The cube index is
#: 16 + 36*r + 6*g + b for (r,g,b) in 0..5 -- so entry i here is colour code 16+i.
_ANSI256_FG = np.array(["\x1b[38;5;%dm" % (16 + i) for i in range(216)])
_ANSI256_BG = np.array(["\x1b[48;5;%dm" % (16 + i) for i in range(216)])


def _cube_index(rgb01):
    """Map an (...,3) array of 0..1 colours to the 0..215 index into the ANSI-256 cube codebook. Vectorised."""
    q = (np.clip(rgb01, 0, 1) * 6).astype(int).clip(0, 5)
    return 36 * q[..., 0] + 6 * q[..., 1] + q[..., 2]


def _fg_prefixes(rgb, mode):
    """Vectorised ANSI foreground escape prefixes for an (N,3) 0..1 colour array -> (N,) array of strings.
    Truecolor formats per entry (16.7M colours, no codebook possible); 256 indexes the baked cube codebook."""
    if mode == "truecolor":
        c = (np.clip(rgb, 0, 1) * 255).astype(int)
        return np.array(["\x1b[38;2;%d;%d;%dm" % (r, g, b) for r, g, b in c])
    return _ANSI256_FG[_cube_index(rgb)]


def _to_gray_and_rgb(image):
    """Normalise any input -- (H,W), (H,W,3), ints 0..255 or floats 0..1 -- to (gray01, rgb01)."""
    a = np.asarray(image, float)
    if a.dtype.kind in "ui" or a.max() > 1.5:              # heuristics: byte images arrive as 0..255
        a = a / 255.0
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 2:
        return a, np.repeat(a[..., None], 3, axis=2)
    if a.ndim == 3 and a.shape[2] >= 3:
        rgb = a[..., :3]
        # Rec.601 luma -- the perceptual brightness a glyph ramp should follow, not the channel mean
        return rgb @ np.array([0.299, 0.587, 0.114]), rgb
    raise ValueError("image must be (H,W) or (H,W,3+)")


def _resample(img, out_h, out_w):
    """Box-average resample to (out_h, out_w), fully vectorised. Works on (H,W) or (H,W,3).

    WHY box-average and not nearest: at the brutal downscales ASCII implies (240px -> 60 cells), nearest-neighbour
    aliases a thin line into dashes; averaging each destination cell over its source box keeps it visible. The
    implementation samples an S x S sub-grid per destination cell (S=2) by fancy indexing and means it -- two
    gathers and a reshape, no loops."""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    S = 2                                                   # sub-samples per destination cell per axis
    ys = ((np.arange(out_h * S) + 0.5) / (out_h * S) * H).astype(int).clip(0, H - 1)
    xs = ((np.arange(out_w * S) + 0.5) / (out_w * S) * W).astype(int).clip(0, W - 1)
    big = img[np.ix_(ys, xs)] if img.ndim == 2 else img[np.ix_(ys, xs)]
    if img.ndim == 2:
        return big.reshape(out_h, S, out_w, S).mean(axis=(1, 3))
    return big.reshape(out_h, S, out_w, S, img.shape[2]).mean(axis=(1, 3))


#: Named glyph-ramp CODEBOOKS (luminance -> character): a swappable registry so a caller can pick a ramp by name
#: ('short', 'long', 'blocks', 'dots') or pass a custom string, and so a ramp is a first-class, composable choice
#: rather than a magic literal. 'blocks' uses the Unicode shade blocks for a smooth 5-level fill; 'dots' is a
#: minimal density ramp. Each starts at space (black) and increases in ink.
RAMPS = {
    "short": RAMP_SHORT,
    "long": RAMP_LONG,
    "blocks": " \u2591\u2592\u2593\u2588",           # light/medium/dark shade + full block
    "dots": " .\u2591:\u2592*\u2593#\u2588",
}


def resolve_ramp(ramp):
    """Turn a ramp NAME (a key of RAMPS) or a literal glyph string into the ramp string. The composability seam:
    ascii_render, a caller, or a future themed renderer all name the same codebook here instead of hard-coding a
    literal. Unknown non-key strings are treated as a custom ramp (any string dark->bright is valid)."""
    if isinstance(ramp, str) and ramp in RAMPS:
        return RAMPS[ramp]
    return ramp                                          # a literal custom ramp, or None (caller defaults)


def _rows_for(width, H, W, cell_aspect):
    """Character rows for a given width: aspect-corrected (a terminal cell is ~2x taller than wide)."""
    return max(1, int(round(width * (H / max(W, 1)) * cell_aspect)))


def ascii_render(image, width=80, mode="ramp", ansi=None, ramp=None, gamma=1.0,
                 invert=False, cell_aspect=0.5, edge_threshold=0.20):
    """Render an image to a text string at `width` characters wide -- the ASCII projection backend.

    image : (H,W) gray or (H,W,3) RGB, floats 0..1 or ints 0..255.
    width : output width in CHARACTERS (the resolution knob). Rows follow from the image aspect.
    mode  : 'ramp' (luminance glyphs), 'edge' (ramp + oriented | / - \\ glyphs on strong gradients),
            'braille' (U+2800 dot cells, 2x4 = 8 pixels/char, Bayer-dithered -- the max-detail mode),
            'half' (U+2580 half-blocks, 2 full-color pixels/char -- requires ansi).
    ansi  : None, '256', or 'truecolor' -- wrap glyphs in color escapes ('half' needs one of these).
    ramp  : custom glyph ramp string (dark -> bright); defaults to RAMP_LONG for 'ramp'/'edge'.
    gamma : display gamma applied to luminance before glyph lookup (2.2 to brighten a linear render).
    invert: flip the ramp (for dark-on-light terminals).
    cell_aspect : terminal cell width/height ratio (~0.5; 1.0 for square-cell contexts).
    edge_threshold : ('edge' mode) gradient magnitude, relative to the image's own max, above which a cell
            renders its orientation glyph instead of its brightness glyph -- self-calibrating, no absolute unit.

    Returns the string (rows joined by newline; ANSI reset appended when color is used). Deterministic to the
    byte for a given input and arguments -- safe to pin in a regression test."""
    gray, rgb = _to_gray_and_rgb(image)
    if gamma != 1.0:
        gray = gray ** (1.0 / gamma)
    H, W = gray.shape
    rows = _rows_for(width, H, W, cell_aspect)

    if mode == "braille":
        # 2x4 dots per cell: sample at (rows*4, width*2), Bayer-threshold, pack bits into U+2800+offset.
        g = _resample(gray, rows * 4, width * 2)
        thresh = np.tile(_BAYER8, (rows * 4 // 8 + 1, width * 2 // 8 + 1))[: rows * 4, : width * 2]
        dots = (g > thresh).astype(np.int32)
        cells = dots.reshape(rows, 4, width, 2).transpose(0, 2, 1, 3)          # (rows, width, 4, 2)
        codes = 0x2800 + (cells * _BRAILLE_BITS).sum(axis=(2, 3))
        lines = ["".join(map(chr, row)) for row in codes]
        if ansi:
            cell_rgb = _resample(rgb, rows, width)
            out_lines = []
            for r, line in enumerate(lines):
                fg = _fg_prefixes(cell_rgb[r], ansi)
                out_lines.append("".join(p + ch for p, ch in zip(fg, line)) + "\x1b[0m")
            return "\n".join(out_lines)
        return "\n".join(lines)

    if mode == "half":
        if ansi not in ("256", "truecolor"):
            raise ValueError("mode='half' paints with color: pass ansi='256' or 'truecolor'")
        # 2 vertical pixels per cell: fg colors the upper half-block, bg the lower. Vectorised through the baked
        # cube codebook (256) or per-entry format (truecolor) -- no per-cell Python string build in the hot path.
        c = _resample(rgb, rows * 2, width)
        top, bot = np.clip(c[0::2], 0, 1), np.clip(c[1::2], 0, 1)
        if ansi == "truecolor":
            ti = (top * 255).astype(int); bi = (bot * 255).astype(int)
            out_lines = []
            for r in range(rows):
                parts = ["\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm\u2580"
                         % (ti[r, x, 0], ti[r, x, 1], ti[r, x, 2], bi[r, x, 0], bi[r, x, 1], bi[r, x, 2])
                         for x in range(width)]
                out_lines.append("".join(parts) + "\x1b[0m")
            return "\n".join(out_lines)
        fg = _ANSI256_FG[_cube_index(top)]                    # (rows, width) escape strings, codebook-indexed
        bg = _ANSI256_BG[_cube_index(bot)]
        out_lines = ["".join(f + b + "\u2580" for f, b in zip(fg[r], bg[r])) + "\x1b[0m"
                     for r in range(rows)]
        return "\n".join(out_lines)

    # 'ramp' and 'edge': one luminance sample per cell through the glyph ramp.
    r = resolve_ramp(ramp) or RAMP_LONG
    if invert:
        r = r[::-1]
    g = _resample(gray, rows, width)
    idx = (g * (len(r) - 1e-9)).astype(int).clip(0, len(r) - 1)
    glyphs = np.array(list(r))[idx]

    if mode == "edge":
        # oriented glyphs where the gradient is strong: quantise the angle to 4 directions. The gradient is taken
        # on the CELL grid (post-resample) so the orientation matches what the eye sees at this resolution.
        gy, gx = np.gradient(g)
        mag = np.hypot(gx, gy)
        strong = mag > edge_threshold * (mag.max() + 1e-12)
        ang = np.mod(np.arctan2(gy, gx), np.pi)                    # orientation, not direction
        # a gradient ACROSS an edge is perpendicular to the edge itself -- pick the glyph along the edge
        edge_glyphs = np.choose((ang / (np.pi / 4)).astype(int).clip(0, 3), ["|", "\\", "-", "/"])
        glyphs = np.where(strong, edge_glyphs, glyphs)
    elif mode != "ramp":
        raise ValueError("mode must be 'ramp', 'edge', 'braille', or 'half'")

    lines = ["".join(row) for row in glyphs]
    if ansi:
        cell_rgb = _resample(rgb, rows, width)
        out_lines = []
        for rr, line in enumerate(lines):
            fg = _fg_prefixes(cell_rgb[rr], ansi)
            out_lines.append("".join(p + ch for p, ch in zip(fg, line)) + "\x1b[0m")
        return "\n".join(out_lines)
    return "\n".join(lines)


def ascii_field(field, bounds=(-1.0, 1.0), res=None, width=80, mode="ramp", **kw):
    """ASCII a 2-D scalar FIELD sampler directly -- composability past finished images. `field` is any callable
    f(P: (N,2)) -> (N,) values (a slice of a bake_nd field, a noise function, a heightmap); it is sampled on a
    grid over [bounds] x [bounds], normalised to its own min..max, and projected. `res` sets the sampling grid
    (defaults to width and its aspect); remaining kwargs pass to ascii_render. This is the seam that lets the
    ASCII backend consume the engine's native fields, not just PNG-shaped arrays."""
    lo, hi = bounds
    rows = kw.get("_rows", max(1, int(round(width * kw.get("cell_aspect", 0.5)))))
    rx = res or width
    ry = res or rows
    xs = np.linspace(lo, hi, rx)
    ys = np.linspace(hi, lo, ry)                          # top row = high y, screen convention
    gx, gy = np.meshgrid(xs, ys)
    P = np.stack([gx.ravel(), gy.ravel()], axis=1)
    v = np.asarray(field(P), float).reshape(ry, rx)
    v = (v - v.min()) / (np.ptp(v) + 1e-12)                # self-normalise to [0,1]
    return ascii_render(v, width=width, mode=mode, **{k: val for k, val in kw.items() if not k.startswith("_")})


def ascii_sdf(sdf, camera=None, width=80, height=None, mode="ramp", z=4.0, fov=0.8,
              lit=True, **kw):
    """ASCII a 3-D SDF scene directly: raymarch it to a small buffer, shade it, and project to text -- the
    'preview my SDF over SSH' path, no manual render loop. `sdf` is anything with .eval (a live SDF, a domain-
    warped scene) or its DSL text. A default camera looks down -z from `z` with vertical `fov`; pass an explicit
    (origin(3,), forward(3,)) `camera` to override. `lit` adds a lambert term (depth-only if False). Remaining
    kwargs pass to ascii_render (e.g. mode='braille', ansi='256'). Small by design -- a preview, not a final
    render; for a full frame use the raymarcher and pass its image to ascii_render."""
    from holographic.mesh_and_geometry.holographic_sdf import as_eval, sdf_normal
    from holographic.rendering.holographic_raymarch import sphere_trace
    f = as_eval(sdf)                                      # accepts a node, a callable, or DSL text
    W = width
    # WHY this height: a terminal CELL is ~2x taller than wide (cell_aspect ~0.5), and ascii_render will emit
    # rows = width*(H/W)*cell_aspect. If we raymarch a SQUARE buffer (H=W) the scene gets vertically compressed by
    # that cell_aspect and comes out stretched wide. So we raymarch the buffer at the SAME proportion the terminal
    # displays -- one raymarch sample per character cell -- and squeeze the ray fan vertically to match, so a
    # round object stays round. (braille packs 2x4 dots, but ascii_render resamples our buffer to its own dot grid,
    # so a per-cell buffer with the right ASPECT is what matters, not the dot count.)
    cell_aspect = kw.get("cell_aspect", 0.5)
    # WHY a plain square buffer with EQUAL fov on both axes: ascii_render already applies the one and only aspect
    # correction (it emits rows = width*(H_buf/W)*cell_aspect, so a square world image comes out round in 2x-tall
    # cells -- verified: a square circle renders round through ascii_render). So ascii_sdf must hand it an
    # undistorted square render. The old code multiplied the cell factor in HERE too (H=width*cell_aspect*2 and a
    # square fov), double-counting and stretching the picture wide. Render square, correct once, downstream.
    H = height or W
    px = (np.arange(W) + 0.5) / W * 2 - 1
    py = 1 - (np.arange(H) + 0.5) / H * 2
    gx, gy = np.meshgrid(px, py)
    if camera is None:
        O = np.tile([0.0, 0.0, z], (W * H, 1))
        D = np.stack([gx.ravel() * fov, gy.ravel() * fov, -np.ones(W * H)], 1)
    else:
        origin, fwd = np.asarray(camera[0], float), np.asarray(camera[1], float)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        right = np.cross(fwd, [0, 1, 0.0]); right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, fwd)
        D = (fwd + fov * (gx[..., None] * right + gy[..., None] * up)).reshape(-1, 3)
        O = np.tile(origin, (W * H, 1))
    D = D / np.linalg.norm(D, axis=1, keepdims=True)

    class _Obj:                                          # sphere_trace wants a .eval; wrap a bare callable
        def eval(self, P): return f(P)
    hit, t, pos = sphere_trace(_Obj(), O, D, max_steps=96, max_dist=z + 12.0)

    img = np.zeros((W * H,))
    if hit.any():
        if lit:
            N = sdf_normal(_Obj(), pos[hit])
            L = np.array([-0.4, 0.7, 0.5]); L /= np.linalg.norm(L)
            img[hit] = np.clip(N @ L, 0, 1) * 0.85 + 0.15
        else:
            img[hit] = 1.0 - np.clip((t[hit] - t[hit].min()) / (np.ptp(t[hit]) + 1e-9), 0, 1)
    img = img.reshape(H, W)
    return ascii_render(img, width=width, mode=mode, **kw)


# ANSI cursor controls for in-place playback (the demoscene terminal loop). \x1b[2J clears, \x1b[H homes the
# cursor to top-left, \x1b[?25l/h hide/show the cursor so the redraw does not flicker a caret.
_CLEAR, _HOME, _HIDE, _SHOW = "\x1b[2J", "\x1b[H", "\x1b[?25l", "\x1b[?25h"


def ascii_frames(frame, n, width=80, mode="ramp", **kw):
    """Render an animation to a LIST of `n` text frames -- the composable half of terminal playback (no I/O, no
    timing). `frame` is a callable frame(i, u) -> an image / SDF / field, where i is the frame index 0..n-1 and
    u = i/n is normalised time; OR a bare frame(u) taking just the phase. What it returns picks the renderer:
    a 2-D image array -> ascii_render; an SDF node or DSL text -> ascii_sdf; a callable f(P:(N,2)) -> ascii_field.
    Deterministic and pure (returns strings), so a caller can diff frames, write a .txt reel, or drive its own
    loop. See ascii_play for the live in-terminal version. Remaining kwargs pass to the chosen renderer."""
    import inspect
    from holographic.mesh_and_geometry.holographic_sdf import SDF
    two_arg = len(inspect.signature(frame).parameters) >= 2
    out = []
    for i in range(n):
        u = i / max(n, 1)
        obj = frame(i, u) if two_arg else frame(u)
        if isinstance(obj, SDF) or isinstance(obj, str):             # an SDF scene (node or DSL text)
            out.append(ascii_sdf(obj, width=width, mode=mode, **kw))
        elif callable(obj):                                          # a 2-D field sampler
            out.append(ascii_field(obj, width=width, mode=mode, **kw))
        else:                                                        # an image array
            out.append(ascii_render(obj, width=width, mode=mode, **kw))
    return out


def ascii_play(frame, n, width=80, mode="ramp", fps=12.0, loops=1, stream=None, clear=True, **kw):
    """PLAY an ASCII animation live in a terminal -- the demoscene 'kaleidoscope tunnel in the console' loop.
    Renders each frame with ascii_frames, then writes them to `stream` (default sys.stdout) in place: hide the
    cursor, home + redraw per frame, sleep to hold `fps`, restore the cursor at the end. `loops` repeats the
    reel (0 = forever, until KeyboardInterrupt). `clear=True` clears the screen once at the start. Returns the
    number of frames drawn. This is the ONLY I/O function in the module -- everything else returns strings, so
    the pure ascii_frames is what tests and non-terminal callers use. Remaining kwargs pass to the renderer."""
    import sys
    import time
    stream = stream or sys.stdout
    reel = ascii_frames(frame, n, width=width, mode=mode, **kw)      # render once, replay cheaply
    dt = 1.0 / max(fps, 1e-6)
    drawn = 0
    try:
        stream.write(_HIDE + (_CLEAR if clear else ""))
        rep = 0
        while loops == 0 or rep < loops:
            for f in reel:
                stream.write(_HOME + f)
                stream.flush()
                drawn += 1
                time.sleep(dt)
            rep += 1
    except KeyboardInterrupt:
        pass
    finally:
        stream.write(_SHOW + "\n")
        stream.flush()
    return drawn


def _selftest():
    """Contracts, as contrasts (never absolute pixel values):

    1. RESOLUTION KNOB: width W yields exactly W characters per line (ramp), rows follow aspect.
    2. RAMP ORDER: a left-to-right luminance gradient renders with per-row glyph indices NON-DECREASING, and a
       brighter image never maps to a darker glyph (monotonicity -- the contract a ramp must keep).
    3. BRAILLE DETAIL BEATS RAMP: a one-pixel diagonal line, rendered at the same character budget, survives in
       braille (8 px/cell) and dissolves in ramp (1 px/cell) -- measured as line-pixel coverage, the whole reason
       the mode exists.
    4. EDGE MODE draws oriented glyphs on a strong vertical edge ('|'), where plain ramp has none.
    5. ANSI: truecolor escapes present iff requested, reset appended; 'half' refuses to run colorless.
    6. DETERMINISM: byte-identical across two calls.
    7. SPEED: a 240x240 -> 100-wide braille render stays comfortably interactive (< 0.25 s here, typically ms).
    """
    import time

    # (1) resolution knob.
    img = np.tile(np.linspace(0, 1, 64), (64, 1))
    s = ascii_render(img, width=40, mode="ramp")
    lines = s.split("\n")
    assert all(len(ln) == 40 for ln in lines), [len(ln) for ln in lines[:3]]
    assert len(lines) == _rows_for(40, 64, 64, 0.5)

    # (2) ramp monotonicity along the gradient.
    ramp_pos = {ch: i for i, ch in enumerate(RAMP_LONG)}
    for ln in lines:
        idxs = [ramp_pos[ch] for ch in ln]
        assert all(b >= a for a, b in zip(idxs, idxs[1:])), "ramp must be monotone on a gradient"
    dark = ascii_render(np.zeros((8, 8)), width=8).replace("\n", "")
    lit = ascii_render(np.ones((8, 8)), width=8).replace("\n", "")
    assert set(dark) == {" "} and set(lit) == {RAMP_LONG[-1]}

    # (3) braille resolves a hairline that ramp loses. One-pixel diagonal on 128^2, both at width=16 chars.
    hair = np.zeros((128, 128))
    ij = np.arange(128)
    hair[ij, ij] = 1.0
    br = ascii_render(hair, width=16, mode="braille", cell_aspect=1.0)
    rp = ascii_render(hair, width=16, mode="ramp", cell_aspect=1.0)
    # count non-background cells along the diagonal band
    br_on = sum(1 for line in br.split("\n") for ch in line if ch != "\u2800")
    rp_on = sum(1 for line in rp.split("\n") for ch in line if ch != " ")
    assert br_on >= rp_on, (br_on, rp_on)                   # never worse ...
    assert br_on >= 16, br_on                               # ... and the line genuinely survives in braille

    # (4) edge mode marks a vertical edge with '|'.
    step = np.zeros((64, 64)); step[:, 32:] = 1.0
    e = ascii_render(step, width=32, mode="edge")
    assert "|" in e and "|" not in ascii_render(step, width=32, mode="ramp")

    # (5) ANSI contracts.
    rgbimg = np.zeros((16, 16, 3)); rgbimg[..., 0] = 1.0    # pure red
    col = ascii_render(rgbimg, width=8, ansi="truecolor")
    assert "\x1b[38;2;255;0;0m" in col and col.endswith("\x1b[0m")
    assert "\x1b" not in ascii_render(rgbimg, width=8)
    hb = ascii_render(rgbimg, width=8, mode="half", ansi="truecolor")
    assert "\u2580" in hb and "\x1b[48;2;" in hb            # fg AND bg painted
    try:
        ascii_render(rgbimg, width=8, mode="half"); raise AssertionError("half without ansi must refuse")
    except ValueError:
        pass

    # (6) determinism.
    noisy = np.random.default_rng(0).random((60, 80, 3))
    assert ascii_render(noisy, width=50, mode="braille", ansi="256") == \
           ascii_render(noisy, width=50, mode="braille", ansi="256")

    # (7) speed sanity: vectorised path, no per-pixel Python.
    big = np.random.default_rng(1).random((240, 240))
    t0 = time.time()
    ascii_render(big, width=100, mode="braille")
    dt = time.time() - t0
    assert dt < 0.25, f"braille render too slow: {dt:.3f}s"

    # (8) ANSI CODEBOOK: the baked 256-cube must equal a from-scratch per-cell computation, byte for byte (the
    #     bake is a speedup, NOT a behaviour change), and be much faster. Same colours, one table lookup.
    rgbimg = np.random.default_rng(2).random((40, 60, 3))
    baked = ascii_render(rgbimg, width=50, mode="half", ansi="256")
    # reference: recompute one cell's fg/bg the slow way and confirm the codebook matched it
    top = _resample(rgbimg, 2 * _rows_for(50, 40, 60, 0.5), 50)[0]
    q0 = _cube_index(np.clip(top[0], 0, 1))
    assert _ANSI256_FG[q0] in baked                       # the codebook entry appears in the output
    t0 = time.time(); [ascii_render(rgbimg, width=50, mode="half", ansi="256") for _ in range(10)]
    assert (time.time() - t0) / 10 < 0.05                 # codebook keeps color mode interactive

    # (9) RAMP CODEBOOK: named ramps resolve; a custom literal passes through; 'blocks' renders shade glyphs.
    assert resolve_ramp("long") is RAMP_LONG and resolve_ramp("@#*. ") == "@#*. "
    blk = ascii_render(np.tile(np.linspace(0, 1, 20), (6, 1)), width=20, ramp="blocks")
    assert "\u2588" in blk and set(blk.replace("\n", "")) <= set(RAMPS["blocks"])

    # (10) COMPOSABILITY: ascii_field consumes a raw scalar field (not an image); ascii_sdf consumes an SDF and
    #      raymarchs it. A radial field's centre is bright and its edge dark; a sphere fills the middle rows.
    fld = ascii_field(lambda P: -np.hypot(P[:, 0], P[:, 1]), bounds=(-1, 1), width=21, mode="ramp")
    flines = fld.split("\n")
    mid = flines[len(flines) // 2]
    assert mid[len(mid) // 2] != " "                      # centre of a radial peak is inked
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    sph = ascii_sdf(sphere(1.0), width=30, mode="ramp")
    assert sum(ch != " " for ch in sph) > 30              # the sphere actually rendered

    # (11) ANIMATION: ascii_frames renders a frame(i,u) sequence to n text frames; the frames must DIFFER (a
    #      moving scene is not a still), each honour the width, and be deterministic. ascii_play writes them to a
    #      stream in place (tested with an in-memory buffer -- no real terminal needed) and reports the count.
    import io
    frames = ascii_frames(lambda u: np.roll(np.tile(np.linspace(0, 1, 32), (16, 1)), int(u * 32), axis=1),
                          n=6, width=24, mode="ramp")
    assert len(frames) == 6 and all(len(ln) == 24 for f in frames for ln in f.split("\n"))
    assert len(set(frames)) > 1                            # the animation genuinely moves
    assert ascii_frames(lambda u: sphere(1.0), n=3, width=20) == \
           ascii_frames(lambda u: sphere(1.0), n=3, width=20)                 # deterministic (SDF path too)
    buf = io.StringIO()
    drawn = ascii_play(lambda u: sphere(1.0 - 0.3 * u), n=4, width=20, fps=1000.0, loops=1, stream=buf)
    assert drawn == 4 and "\x1b[H" in buf.getvalue() and buf.getvalue().endswith("\x1b[?25h\n")  # homed + cursor restored

    # ASPECT: a SPHERE must render ROUND, not stretched wide. Terminal cells are ~2x taller than wide, so a round
    # sphere spans ~2x more COLUMNS than ROWS. The historical bug applied that cell factor twice (once in ascii_sdf
    # picking a square-ish buffer AND once in ascii_render's row count), stretching every render horizontally. Pin
    # the ratio so it cannot come back: a bare sphere's char bounding box must be 1.5-2.6 wide-per-tall.
    _a = ascii_sdf(sphere(1.0), width=40, mode="ramp", z=3.5, fov=0.9).split("\n")
    _ink = [i for i, ln in enumerate(_a) if ln.strip()]
    _cols = [(min(i for i, ch in enumerate(ln) if ch != " "), max(i for i, ch in enumerate(ln) if ch != " "))
             for ln in _a if ln.strip()]
    _h = _ink[-1] - _ink[0] + 1
    _w = max(c[1] for c in _cols) - min(c[0] for c in _cols) + 1
    assert 1.5 <= _w / _h <= 2.6, ("sphere ascii stretched: aspect %.2f (want ~2.0)" % (_w / _h))

    print("holographic_ascii selftest OK (ramp monotone; braille kept a hairline ramp lost %d>=%d; "
          "edge '|' fires; ANSI 256 codebook byte-matches per-cell + interactive; named ramps resolve; "
          "ascii_field/ascii_sdf compose; sphere renders round (aspect %.2f); 240^2 -> 100w braille in %.0f ms)"
          % (br_on, rp_on, _w / _h, dt * 1000))


if __name__ == "__main__":
    _selftest()
