#!/usr/bin/env python3
"""Place the opaque floofy dog inside a seamless animated leCore meadow world."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holographic.materials_and_texture.holographic_matlib import material
from holographic.rendering.holographic_skymodel import sky_model
from scripts.animate_floofy_dog import (
    _translate,
    eyelid_colour,
    flex_tail,
    save_gif,
    save_mp4,
    split_character,
    studio_background,
)


def _rgb(name: str, brightness: float = 1.0) -> tuple[int, int, int]:
    """Read a surface colour from the leCore material library."""
    values = np.asarray(material(name).base_color[:3]) * float(brightness)
    return tuple(int(round(v * 255)) for v in np.clip(values, 0.0, 1.0))


@lru_cache(maxsize=4)
def _sky_plate(width: int, height: int, horizon: int) -> Image.Image:
    """Sample the leCore parametric sky into the camera's upper hemisphere."""
    yy, xx = np.mgrid[0:horizon, 0:width].astype(float)
    screen_x = (xx / max(width - 1, 1) - 0.5) * 1.48
    screen_up = 0.04 + (1.0 - yy / max(horizon - 1, 1)) * 0.88
    directions = np.stack([screen_x, screen_up, -np.ones_like(screen_x)], axis=-1)
    radiance = sky_model(hour=16.35, clouds=(), sun_intensity=7.0)(directions.reshape(-1, 3))
    radiance = radiance.reshape(horizon, width, 3)
    # A gentle display transform retains the model's warm sun without clipping it flat.
    radiance = radiance / (1.0 + 0.22 * radiance)
    radiance = np.clip(radiance ** (1.0 / 1.08), 0.0, 1.0)
    plate = np.zeros((height, width, 3), dtype=np.uint8)
    plate[:horizon] = (radiance * 255).astype(np.uint8)
    return Image.fromarray(plate, mode="RGB").convert("RGBA")


def _hill_points(
    width: int,
    baseline: float,
    amplitude: float,
    phase: float,
    scroll: float = 0.0,
    cycles: float = 1.0,
) -> list[tuple[int, int]]:
    xs = np.linspace(-20.0, width + 20.0, 33)
    angle = 2.0 * np.pi * cycles * (xs + scroll) / max(width, 1)
    ys = baseline + amplitude * np.sin(angle + phase) + 0.42 * amplitude * np.sin(2.0 * angle - phase)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def _draw_cloud(layer: Image.Image, x: float, y: float, scale: float, opacity: int) -> None:
    draw = ImageDraw.Draw(layer, "RGBA")
    puffs = [
        (-56, 3, 72, 30),
        (-31, -14, 68, 42),
        (0, -25, 84, 54),
        (35, -10, 70, 40),
        (63, 4, 58, 28),
    ]
    for px, py, pw, ph in puffs:
        box = (
            x + (px - pw / 2) * scale,
            y + (py - ph / 2) * scale,
            x + (px + pw / 2) * scale,
            y + (py + ph / 2) * scale,
        )
        draw.ellipse(box, fill=(255, 249, 228, opacity))


def _draw_tree(draw: ImageDraw.ImageDraw, x: float, ground: float, scale: float, sway: float) -> None:
    trunk = _rgb("wood_oak", 0.82)
    dark = tuple(int(v * 0.73) for v in _rgb("moss", 0.92))
    mid = _rgb("grass", 0.82)
    light = tuple(min(255, int(v * 1.22)) for v in _rgb("grass", 0.96))
    draw.polygon(
        [
            (x - 13 * scale, ground),
            (x + 13 * scale, ground),
            (x + (7 + sway) * scale, ground - 126 * scale),
            (x + (-5 + sway) * scale, ground - 128 * scale),
        ],
        fill=trunk + (255,),
    )
    clusters = [
        (-46, -128, 58, dark),
        (-12, -159, 73, mid),
        (35, -145, 64, dark),
        (50, -107, 48, mid),
        (-54, -92, 49, mid),
        (4, -104, 76, light),
    ]
    for cx, cy, radius, colour in clusters:
        ox = (cx + sway * (0.8 + 0.003 * abs(cy))) * scale
        r = radius * scale
        draw.ellipse((x + ox - r, ground + cy * scale - r, x + ox + r, ground + cy * scale + r), fill=colour + (255,))


def _meadow_base(width: int, height: int, horizon: int, phase: float, travel: float = 0.0) -> Image.Image:
    world = _sky_plate(width, height, horizon).copy()

    # Puffy foreground clouds add a second, hand-shaped altitude layer over the parametric sky.
    cloud_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    _draw_cloud(cloud_layer, width * 0.27 + 8.0 * np.sin(2 * np.pi * phase), 155, 0.92, 184)
    _draw_cloud(cloud_layer, width * 0.71 - 6.0 * np.sin(2 * np.pi * phase), 104, 0.60, 154)
    cloud_layer = cloud_layer.filter(ImageFilter.GaussianBlur(radius=2.2))
    world.alpha_composite(cloud_layer)

    draw = ImageDraw.Draw(world, "RGBA")
    # Atmospheric hill layers, with colours tied to the engine's grass/moss presets.
    far = _rgb("grass", 1.08)
    far_points = _hill_points(width, 390, 36, 0.45, scroll=0.5 * travel, cycles=2.0) + [(width + 20, horizon + 95), (-20, horizon + 95)]
    draw.polygon(far_points, fill=far + (245,))
    near = tuple(int(v * 0.86) for v in _rgb("grass", 0.98))
    near_points = _hill_points(width, 438, 47, 2.1, scroll=0.75 * travel, cycles=4.0) + [(width + 20, height), (-20, height)]
    draw.polygon(near_points, fill=near + (255,))

    # The foreground is a vertical mix of the named grass and moss materials.
    grass_top = np.asarray(_rgb("grass", 1.03), float)
    grass_bottom = np.asarray(_rgb("moss", 0.72), float)
    ground = np.zeros((height - horizon, width, 4), dtype=np.uint8)
    t = np.linspace(0.0, 1.0, max(height - horizon, 1))[:, None, None]
    ground[..., :3] = ((1.0 - t) * grass_top + t * grass_bottom).astype(np.uint8)
    ground[..., 3] = 255
    world.alpha_composite(Image.fromarray(ground, mode="RGBA"), dest=(0, horizon))

    draw = ImageDraw.Draw(world, "RGBA")
    tree_sway = 2.3 * np.sin(2.0 * np.pi * phase)
    for repeat in (-1, 0, 1, 2):
        offset = repeat * width - travel
        _draw_tree(draw, 38 + offset, 535, 0.78, tree_sway)
        _draw_tree(draw, width - 42 + offset, 523, 0.88, -0.72 * tree_sway)

    # A soft path anchors the paws and pulls the eye into the small world.
    dirt = _rgb("dirt", 1.36)
    path = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pd = ImageDraw.Draw(path, "RGBA")
    for repeat in (-1, 0, 1, 2):
        offset = repeat * width - travel
        pd.ellipse((112 + offset, 525, 662 + offset, 710), fill=dirt + (205,))
        pd.ellipse((250 + offset, 615, 590 + offset, 795), fill=_rgb("grass_dry", 1.12) + (95,))
    path = path.filter(ImageFilter.GaussianBlur(radius=14.0))
    world.alpha_composite(path)

    # Deterministic grass and flowers. Their wind motion is sinusoidal, so the loop closes exactly.
    rng = np.random.default_rng(431)
    blade_x = rng.integers(0, width, 250)
    blade_y = rng.integers(475, height + 8, 250)
    blade_h = rng.integers(8, 29, 250)
    blade_tint = rng.random(250)
    wind = np.sin(2.0 * np.pi * phase)
    draw = ImageDraw.Draw(world, "RGBA")
    for x, y, h, tint in zip(blade_x, blade_y, blade_h, blade_tint):
        x = (float(x) - travel) % width
        colour = (36 + int(24 * tint), 91 + int(42 * tint), 25 + int(12 * tint), 150)
        tip_x = int(x + wind * h * (0.10 + 0.09 * tint))
        draw.line((int(x), int(y), tip_x, int(y - h)), fill=colour, width=1 + int(y > 650))

    flowers = [(83, 590, (255, 214, 62)), (691, 607, (245, 116, 126)), (117, 683, (190, 148, 240)),
               (639, 702, (255, 235, 115)), (52, 724, (245, 150, 186)), (726, 669, (235, 225, 255))]
    for i, (x, y, colour) in enumerate(flowers):
        x = (float(x) - travel) % width
        bob = 2.0 * np.sin(2.0 * np.pi * phase + i * 0.74)
        draw.line((x, y, x + wind * 1.8, y - 19 + bob), fill=(54, 116, 38, 220), width=2)
        cx, cy = x + wind * 1.8, y - 21 + bob
        for angle in np.linspace(0.0, 2.0 * np.pi, 6)[:-1]:
            px, py = cx + 5.2 * np.cos(angle), cy + 5.2 * np.sin(angle)
            draw.ellipse((px - 3.2, py - 3.2, px + 3.2, py + 3.2), fill=colour + (235,))
        draw.ellipse((cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5), fill=(240, 174, 43, 255))
    return world


def _dog_shadow(width: int, height: int, lift: float) -> Image.Image:
    sx, sy = width / 768.0, height / 768.0
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    squeeze = 1.0 - 0.06 * lift
    cx, cy = 392 * sx, 594 * sy
    rx, ry = 190 * sx * squeeze, 35 * sy * squeeze
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(25, 37, 15, int(94 - 18 * lift)))
    return layer.filter(ImageFilter.GaussianBlur(radius=17 * sx))


def _foreground_grass(width: int, height: int, phase: float, travel: float = 0.0) -> Image.Image:
    """A few close blades overlap the paws, integrating the character into the set."""
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    rng = np.random.default_rng(997)
    wind = np.sin(2.0 * np.pi * phase)
    for x, y, h in zip(rng.integers(0, width, 76), rng.integers(616, height + 5, 76), rng.integers(16, 43, 76)):
        x = (float(x) - travel) % width
        tip = x + wind * h * 0.18
        draw.line((int(x), int(y), int(tip), int(y - h)), fill=(39, 92, 25, 205), width=2)
    return layer


def make_world_frames(source_path: Path, frames: int = 40) -> list[Image.Image]:
    source_pil = Image.open(source_path).convert("RGB")
    source = np.asarray(source_pil, float) / 255.0
    height, width = source.shape[:2]
    body, tail, ear, _ = split_character(source, studio_background(width, height))
    fur_colour = eyelid_colour(source)

    sx, sy = width / 768.0, height / 768.0
    tail_pivot = (588.0 * sx, 307.0 * sy)
    ear_pivot = (190.0 * sx, 263.0 * sy)
    eye = (153.0 * sx, 275.0 * sy)
    horizon = int(round(452 * sy))
    output: list[Image.Image] = []
    for index in range(int(frames)):
        phase = index / float(frames)
        wag = 18.0 * np.sin(4.0 * np.pi * phase)
        ear_bounce = 3.4 * np.sin(4.0 * np.pi * phase - 0.55)
        lift = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
        bob = -2.8 * sy * lift
        drift = 0.8 * sx * np.sin(2.0 * np.pi * phase)

        frame = _meadow_base(width, height, horizon, phase)
        frame.alpha_composite(_dog_shadow(width, height, lift))
        frame.alpha_composite(_translate(flex_tail(tail, tail_pivot, wag), drift, bob))
        frame.alpha_composite(_translate(body, drift, bob))
        frame.alpha_composite(
            _translate(ear.rotate(float(ear_bounce), resample=Image.Resampling.BICUBIC, center=ear_pivot), drift, bob)
        )

        blink = float(np.exp(-0.5 * ((phase - 0.61) / 0.038) ** 2))
        if blink > 0.015:
            draw = ImageDraw.Draw(frame, "RGBA")
            cx, cy = eye[0] + drift, eye[1] + bob
            rx, ry = 12.0 * sx, 12.0 * sy
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fur_colour + (int(255 * min(1.0, blink * 1.35)),))
            draw.arc(
                (cx - rx, cy - 2.5 * sy, cx + rx, cy + 5.5 * sy),
                start=8,
                end=172,
                fill=(55, 30, 20, int(255 * blink)),
                width=max(2, int(round(3 * sx))),
            )
        frame.alpha_composite(_foreground_grass(width, height, phase))
        output.append(frame.convert("RGB"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("figures/better_dog_lecore.png"))
    parser.add_argument("--gif", type=Path, default=Path("figures/floofy_dog_meadow.gif"))
    parser.add_argument("--mp4", type=Path, default=Path("figures/floofy_dog_meadow.mp4"))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    frames = make_world_frames(args.source, frames=args.frames)
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
