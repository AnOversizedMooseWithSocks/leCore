#!/usr/bin/env python3
"""Animate the finished opaque-fur leCore dog as a seamless character loop."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holographic.rendering.holographic_postfx import resample


def studio_background(width: int, height: int, lift: float = 0.0) -> np.ndarray:
    """Rebuild the exact studio plate used by render_better_dog.py."""
    scale = 1.25
    rw, rh = int(round(width * scale)), int(round(height * scale))
    yy, xx = np.mgrid[0:rh, 0:rw]
    t = yy[..., None] / max(rh - 1, 1)
    backdrop = (
        np.array([0.975, 0.955, 0.910])[None, None, :] * (1.0 - t)
        + np.array([0.885, 0.850, 0.780])[None, None, :] * t
    )
    floor_glow = np.exp(-((yy / rh - 0.72) / 0.24) ** 2)[..., None]
    backdrop = np.clip(backdrop + 0.018 * floor_glow, 0.0, 1.0)
    lift = float(np.clip(lift, 0.0, 1.0))
    dx = (xx / rw - 0.515) / (0.285 * (1.0 - 0.035 * lift))
    dy = (yy / rh - 0.775) / (0.052 * (1.0 - 0.050 * lift))
    shadow_alpha = (0.32 * (1.0 - 0.16 * lift)) * np.exp(-2.15 * (dx * dx + dy * dy))[..., None]
    backdrop = backdrop * (1.0 - shadow_alpha) + np.array([0.34, 0.30, 0.24]) * shadow_alpha
    return np.clip(resample(backdrop, scale=1.0 / scale)[:height, :width], 0.0, 1.0)


def _rgba_layer(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    rgba = np.dstack([np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)])
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def _translate(layer: Image.Image, dx: float, dy: float) -> Image.Image:
    return layer.transform(
        layer.size,
        Image.Transform.AFFINE,
        (1.0, 0.0, -float(dx), 0.0, 1.0, -float(dy)),
        resample=Image.Resampling.BICUBIC,
    )


def split_character(source: np.ndarray, background: np.ndarray):
    """Recover soft body, tail, and ear layers from the final render."""
    height, width = source.shape[:2]
    difference = np.max(np.abs(source - background), axis=2)
    dog_alpha = np.clip((difference - 0.012) / 0.070, 0.0, 1.0)

    sx, sy = width / 768.0, height / 768.0
    polygon = [
        (570 * sx, 255 * sy),
        (614 * sx, 245 * sy),
        (686 * sx, 164 * sy),
        (726 * sx, 188 * sy),
        (714 * sx, 250 * sy),
        (668 * sx, 307 * sy),
        (620 * sx, 350 * sy),
        (584 * sx, 333 * sy),
    ]
    part = Image.new("L", (width, height), 0)
    ImageDraw.Draw(part).polygon(polygon, fill=255)
    part = part.filter(ImageFilter.GaussianBlur(radius=max(1.0, 1.8 * sx)))
    partition = np.asarray(part, float) / 255.0
    xx = np.arange(width, dtype=float)[None, :]
    # Keep only a very short blend at the rump. A wide blend leaves a ghost of
    # the original tail behind when the plume reaches the end of its wag.
    base_feather = np.clip((xx - 568.0 * sx) / max(30.0 * sx, 1.0), 0.0, 1.0)
    partition *= base_feather

    tail_alpha = dog_alpha * partition
    body_alpha = dog_alpha * (1.0 - partition)

    # The ear is a darker, warmer connected region inside this polygon. Keep
    # its original pixels as a foreground layer, and paint plausible nearby
    # fur underneath so a bouncing ear reveals coat rather than a cutout hole.
    ear_polygon = [
        (160 * sx, 225 * sy),
        (225 * sx, 226 * sy),
        (260 * sx, 320 * sy),
        (242 * sx, 422 * sy),
        (176 * sx, 408 * sy),
        (150 * sx, 278 * sy),
    ]
    ear_region = Image.new("L", (width, height), 0)
    ImageDraw.Draw(ear_region).polygon(ear_polygon, fill=255)
    region = np.asarray(ear_region, float) / 255.0
    red, green, blue = source[..., 0], source[..., 1], source[..., 2]
    ear_colour = (
        (red > 0.18)
        & (red < 0.76)
        & (green < 0.39)
        & (blue < 0.28)
        & (red > 1.28 * green)
    )
    ear_raw = (region * ear_colour.astype(float) * dog_alpha * 255).astype(np.uint8)
    ear_mask = Image.fromarray(ear_raw, mode="L")
    ear_mask = ear_mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    ear_mask = ear_mask.filter(ImageFilter.GaussianBlur(radius=max(0.8, 1.0 * sx)))
    ear_partition = np.asarray(ear_mask, float) / 255.0
    ear_alpha = dog_alpha * ear_partition

    fx0, fx1 = int(260 * sx), int(320 * sx)
    fy0, fy1 = int(250 * sy), int(330 * sy)
    fur_fill = np.median(source[fy0:fy1, fx0:fx1].reshape(-1, 3), axis=0)
    body_rgb = source * (1.0 - ear_partition[..., None]) + fur_fill[None, None, :] * ear_partition[..., None]
    return (
        _rgba_layer(body_rgb, body_alpha),
        _rgba_layer(source, tail_alpha),
        _rgba_layer(source, ear_alpha),
        dog_alpha,
    )


def _bilinear_sample(image: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    """Bilinear RGBA sampling for the tail's progressive bend warp."""
    height, width = image.shape[:2]
    x0 = np.floor(sample_x).astype(int)
    y0 = np.floor(sample_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    fx = (sample_x - x0)[..., None]
    fy = (sample_y - y0)[..., None]
    a = image[y0c, x0c]
    b = image[y0c, x1c]
    c = image[y1c, x0c]
    d = image[y1c, x1c]
    out = a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy
    out[~valid] = 0.0
    return out


def flex_tail(layer: Image.Image, pivot: tuple[float, float], angle_deg: float) -> Image.Image:
    """Bend increasingly from a fixed rump to a freely wagging plume tip."""
    source = np.asarray(layer, float) / 255.0
    height, width = source.shape[:2]
    px, py = pivot
    radius = 225.0 * (width / 768.0)
    x0, x1 = max(0, int(px - 35)), min(width, int(px + radius + 25))
    y0, y1 = max(0, int(py - radius)), min(height, int(py + 85))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    dx, dy = xx - px, yy - py
    r = np.sqrt(dx * dx + dy * dy)
    start, end = 32.0 * width / 768.0, 178.0 * width / 768.0
    weight = np.clip((r - start) / max(end - start, 1.0), 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)
    angle = -np.radians(float(angle_deg)) * weight
    ca, sa = np.cos(angle), np.sin(angle)
    sample_x = px + ca * dx - sa * dy
    sample_y = py + sa * dx + ca * dy
    patch = _bilinear_sample(source, sample_x, sample_y)
    output = np.zeros_like(source)
    output[y0:y1, x0:x1] = patch
    return Image.fromarray((np.clip(output, 0.0, 1.0) * 255).astype(np.uint8), mode="RGBA")


def eyelid_colour(source: np.ndarray) -> tuple[int, int, int]:
    """Sample nearby fur so the blink inherits this exact character's coat."""
    height, width = source.shape[:2]
    sx, sy = width / 768.0, height / 768.0
    x0, x1 = int(112 * sx), int(142 * sx)
    y0, y1 = int(235 * sy), int(262 * sy)
    patch = source[y0:y1, x0:x1]
    rgb = np.median(patch.reshape(-1, 3), axis=0)
    return tuple(int(v * 255) for v in np.clip(rgb, 0.0, 1.0))


def make_frames(source_path: Path, frames: int = 40, fps: int = 20) -> list[Image.Image]:
    source_pil = Image.open(source_path).convert("RGB")
    source = np.asarray(source_pil, float) / 255.0
    height, width = source.shape[:2]
    background = studio_background(width, height)
    body, tail, ear, alpha = split_character(source, background)
    fur_colour = eyelid_colour(source)

    sx, sy = width / 768.0, height / 768.0
    tail_pivot = (588.0 * sx, 307.0 * sy)
    ear_pivot = (190.0 * sx, 263.0 * sy)
    eye = (153.0 * sx, 275.0 * sy)
    output: list[Image.Image] = []
    for index in range(int(frames)):
        phase = index / float(frames)
        # Two wags per idle loop; the slow vertical motion returns exactly to
        # its start, so frame N joins frame 0 without a pop.
        wag = 18.0 * np.sin(4.0 * np.pi * phase)
        ear_bounce = 3.4 * np.sin(4.0 * np.pi * phase - 0.55)
        lift = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
        bob = -2.8 * sy * lift
        drift = 0.8 * sx * np.sin(2.0 * np.pi * phase)

        frame_background = studio_background(width, height, lift=lift)
        frame = Image.fromarray((frame_background * 255).astype(np.uint8), mode="RGB").convert("RGBA")
        wagged_tail = flex_tail(tail, tail_pivot, wag)
        bounced_ear = ear.rotate(
            float(ear_bounce),
            resample=Image.Resampling.BICUBIC,
            center=ear_pivot,
        )
        frame.alpha_composite(_translate(wagged_tail, drift, bob))
        frame.alpha_composite(_translate(body, drift, bob))
        frame.alpha_composite(_translate(bounced_ear, drift, bob))

        # One soft blink, deliberately away from the loop seam.
        blink = float(np.exp(-0.5 * ((phase - 0.61) / 0.038) ** 2))
        if blink > 0.015:
            draw = ImageDraw.Draw(frame, "RGBA")
            cx, cy = eye[0] + drift, eye[1] + bob
            rx, ry = 12.0 * sx, 12.0 * sy
            fill_alpha = int(255 * min(1.0, blink * 1.35))
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fur_colour + (fill_alpha,))
            line_alpha = int(255 * blink)
            draw.arc(
                (cx - rx, cy - 2.5 * sy, cx + rx, cy + 5.5 * sy),
                start=8,
                end=172,
                fill=(55, 30, 20, line_alpha),
                width=max(2, int(round(3 * sx))),
            )
        output.append(frame.convert("RGB"))
    return output


def save_gif(frames: list[Image.Image], output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    palette = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
    quantized = [palette]
    quantized.extend(
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames[1:]
    )
    quantized[0].save(
        output,
        save_all=True,
        append_images=quantized[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=False,
        disposal=2,
    )


def save_mp4(frames: list[Image.Image], output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    full_ffmpeg = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
    encoder = str(full_ffmpeg) if full_ffmpeg.exists() else "ffmpeg"
    with tempfile.TemporaryDirectory(prefix="floofy-dog-frames-") as temp:
        frame_dir = Path(temp)
        for index, frame in enumerate(frames):
            frame.save(frame_dir / f"frame_{index:04d}.png")
        subprocess.run(
            [
                encoder,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(int(fps)),
                "-i",
                str(frame_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("figures/better_dog_lecore.png"))
    parser.add_argument("--gif", type=Path, default=Path("figures/floofy_dog_animated.gif"))
    parser.add_argument("--mp4", type=Path, default=Path("figures/floofy_dog_animated.mp4"))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    frames = make_frames(args.source, frames=args.frames, fps=args.fps)
    save_gif(frames, args.gif, args.fps)
    save_mp4(frames, args.mp4, args.fps)
    print(
        {
            "source": str(args.source),
            "gif": str(args.gif),
            "mp4": str(args.mp4),
            "frames": len(frames),
            "fps": int(args.fps),
            "duration_seconds": len(frames) / float(args.fps),
            "size": list(frames[0].size),
        }
    )


if __name__ == "__main__":
    main()
