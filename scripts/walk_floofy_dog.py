#!/usr/bin/env python3
"""Animate the opaque floofy dog walking in his leCore meadow world."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.animate_floofy_dog import (
    _rgba_layer,
    _translate,
    eyelid_colour,
    flex_tail,
    save_gif,
    save_mp4,
    split_character,
    studio_background,
)
from scripts.place_floofy_dog_in_world import _foreground_grass, _meadow_base
from holographic.misc.holographic_vision import _connected_components


def _polygon_mask(size: tuple[int, int], points: list[tuple[float, float]], blur: float) -> np.ndarray:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    return np.asarray(mask, float) / 255.0


def _components_near(mask: np.ndarray, seeds: list[tuple[float, float]]) -> np.ndarray:
    """Keep the leCore-labelled image parts closest to the hip, knee, and paw."""
    binary = np.asarray(mask > 0.055, bool)
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return mask
    y0, y1 = max(0, int(ys.min()) - 1), min(mask.shape[0], int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(mask.shape[1], int(xs.max()) + 2)
    cropped = binary[y0:y1, x0:x1]
    labels, _ = _connected_components(cropped)
    cy, cx = np.nonzero(cropped)
    keep: set[int] = set()
    for seed_x, seed_y in seeds:
        distance = (cx + x0 - seed_x) ** 2 + (cy + y0 - seed_y) ** 2
        nearest = int(np.argmin(distance))
        label = int(labels[cy[nearest], cx[nearest]])
        if label:
            keep.add(label)
    selected = np.zeros_like(binary)
    selected[y0:y1, x0:x1] = np.isin(labels, list(keep))
    return mask * selected


def split_walking_layers(body: Image.Image):
    """Separate four leg sprites and retain a soft torso cover for clean hip joints."""
    rgba = np.asarray(body, float) / 255.0
    height, width = rgba.shape[:2]
    sx, sy = width / 768.0, height / 768.0
    specs = [
        # name, polygon, hip, knee, paw, depth, phase sign
        ("front_far", [(180, 390), (270, 390), (290, 500), (285, 570), (220, 590), (175, 560)], (229, 406), (235, 480), (230, 553), "far", -1.0),
        ("front_near", [(250, 390), (365, 395), (370, 540), (350, 620), (250, 620), (245, 500)], (310, 410), (315, 492), (305, 583), "near", 1.0),
        ("rear_far", [(365, 365), (490, 365), (500, 450), (480, 500), (500, 550), (450, 580), (355, 545), (380, 440)], (431, 391), (444, 460), (421, 524), "far", 1.0),
        ("rear_near", [(440, 360), (565, 370), (575, 490), (555, 565), (485, 600), (420, 560), (460, 490)], (505, 391), (520, 467), (500, 548), "near", -1.0),
    ]
    masks: list[np.ndarray] = []
    regions: list[np.ndarray] = []
    for _, raw_points, _, _, _, _, _ in specs:
        points = [(x * sx, y * sy) for x, y in raw_points]
        region = _polygon_mask((width, height), points, max(1.0, 1.5 * sx))
        regions.append(region)
        masks.append(region * rgba[..., 3])

    # Polygons overlap around the hips and crossed paws in the source render. Give every source pixel
    # to exactly one limb so no fragment is duplicated when the legs separate during the stride.
    mask_stack = np.stack(masks, axis=0)
    owner = np.argmax(mask_stack, axis=0)
    masks = [mask * (owner == index) for index, mask in enumerate(masks)]
    removal_union = np.clip(np.maximum.reduce(regions), 0.0, 1.0)
    masks = [
        _components_near(
            mask,
            [
                (spec[2][0] * sx, spec[2][1] * sy),
                (spec[3][0] * sx, spec[3][1] * sy),
                (spec[4][0] * sx, spec[4][1] * sy),
            ],
        )
        for mask, spec in zip(masks, specs)
    ]

    legs = []
    yy = np.arange(height, dtype=float)[:, None]
    for (name, _, raw_hip, raw_knee, raw_paw, depth, sign), mask in zip(specs, masks):
        hip = (raw_hip[0] * sx, raw_hip[1] * sy)
        knee = (raw_knee[0] * sx, raw_knee[1] * sy)
        paw = (raw_paw[0] * sx, raw_paw[1] * sy)
        feather = max(7.0 * sy, 1.0)
        upper_weight = np.clip((knee[1] + feather - yy) / (2.0 * feather), 0.0, 1.0)
        lower_weight = 1.0 - upper_weight
        legs.append(
            {
                "name": name,
                "upper": _rgba_layer(rgba[..., :3], mask * upper_weight),
                "lower": _rgba_layer(rgba[..., :3], mask * lower_weight),
                "hip": hip,
                "knee": knee,
                "paw": paw,
                "depth": depth,
                "sign": sign,
            }
        )

    # Remove every pixel inside the authored leg regions from the static body,
    # including tiny source fragments that were rejected from the moving limbs.
    union = removal_union
    body_alpha = rgba[..., 3] * (1.0 - union)
    body_core = _rgba_layer(rgba[..., :3], body_alpha)

    # Re-cover only the upper overlap of the removed regions. This hides rotation seams at each hip
    # without restoring the original static lower legs.
    yy = np.arange(height, dtype=float)[:, None]
    upper = 1.0 - np.clip((yy - 418.0 * sy) / max(58.0 * sy, 1.0), 0.0, 1.0)
    cover_alpha = rgba[..., 3] * union * upper
    torso_cover = _rgba_layer(rgba[..., :3], cover_alpha)
    return body_core, torso_cover, legs


def _angle_delta(target: float, source: float) -> float:
    return float(np.arctan2(np.sin(target - source), np.cos(target - source)))


def _articulate_leg(leg: dict, target: tuple[float, float]) -> Image.Image:
    """Use two-link inverse kinematics so the knee bends and the paw reaches its target."""
    hip = np.asarray(leg["hip"], float)
    knee = np.asarray(leg["knee"], float)
    paw = np.asarray(leg["paw"], float)
    target_point = np.asarray(target, float)
    upper_length = float(np.linalg.norm(knee - hip))
    lower_length = float(np.linalg.norm(paw - knee))

    reach = target_point - hip
    distance = float(np.linalg.norm(reach))
    distance = float(np.clip(distance, abs(upper_length - lower_length) + 1e-4, upper_length + lower_length - 1e-4))
    direction = reach / max(float(np.linalg.norm(reach)), 1e-8)
    along = (upper_length * upper_length + distance * distance - lower_length * lower_length) / (2.0 * distance)
    height = float(np.sqrt(max(upper_length * upper_length - along * along, 0.0)))
    perpendicular = np.array([-direction[1], direction[0]])

    rest_direction = (paw - hip) / max(float(np.linalg.norm(paw - hip)), 1e-8)
    rest_perpendicular = np.array([-rest_direction[1], rest_direction[0]])
    bend_side = 1.0 if float(np.dot(knee - hip, rest_perpendicular)) >= 0.0 else -1.0
    new_knee = hip + direction * along + perpendicular * (bend_side * height)

    old_upper_angle = float(np.arctan2(knee[1] - hip[1], knee[0] - hip[0]))
    new_upper_angle = float(np.arctan2(new_knee[1] - hip[1], new_knee[0] - hip[0]))
    old_lower_angle = float(np.arctan2(paw[1] - knee[1], paw[0] - knee[0]))
    new_lower_angle = float(np.arctan2(target_point[1] - new_knee[1], target_point[0] - new_knee[0]))
    upper_rotation = -np.degrees(_angle_delta(new_upper_angle, old_upper_angle))
    lower_rotation = -np.degrees(_angle_delta(new_lower_angle, old_lower_angle))

    upper = leg["upper"].rotate(
        float(upper_rotation),
        resample=Image.Resampling.BICUBIC,
        center=tuple(hip),
    )
    lower = leg["lower"].rotate(
        float(lower_rotation),
        resample=Image.Resampling.BICUBIC,
        center=tuple(knee),
    )
    lower = _translate(lower, new_knee[0] - knee[0], new_knee[1] - knee[1])
    result = Image.new("RGBA", upper.size, (0, 0, 0, 0))
    result.alpha_composite(upper)
    result.alpha_composite(lower)
    rgba = np.asarray(result, float) / 255.0
    clean_alpha = _components_near(
        rgba[..., 3],
        [tuple(hip), tuple(new_knee), tuple(target_point)],
    )
    return _rgba_layer(rgba[..., :3], clean_alpha)


def _walking_shadow(width: int, height: int, compression: float) -> Image.Image:
    sx, sy = width / 768.0, height / 768.0
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    cx, cy = 411 * sx, 595 * sy
    rx = (177.0 - 8.0 * compression) * sx
    ry = (33.0 - 3.0 * compression) * sy
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(23, 34, 14, int(86 - 10 * compression)))
    return layer.filter(ImageFilter.GaussianBlur(radius=16.0 * sx))


def _footstep_dust(width: int, height: int, phase: float) -> Image.Image:
    """Small soft puffs mark alternating paw contacts without hiding the feet."""
    sx, sy = width / 768.0, height / 768.0
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    step = (phase * 4.0) % 1.0
    events = [
        (0.00, (294.0 * sx, 590.0 * sy)),
        (0.50, (505.0 * sx, 558.0 * sy)),
    ]
    for event, (px, py) in events:
        age = (step - event) % 1.0
        if age >= 0.28:
            continue
        life = age / 0.28
        alpha = int(62 * (1.0 - life) ** 1.7)
        drift = 22.0 * sx * life
        rise = 8.0 * sy * life
        radius = (4.0 + 9.0 * life) * sx
        for offset_x, offset_y, size in [(-0.8, 0.0, 0.72), (0.0, -0.35, 1.0), (0.9, 0.05, 0.62)]:
            cx = px - drift + offset_x * radius
            cy = py - rise + offset_y * radius
            r = radius * size
            draw.ellipse((cx - r, cy - r * 0.55, cx + r, cy + r * 0.55), fill=(196, 151, 89, alpha))
    return layer.filter(ImageFilter.GaussianBlur(radius=max(1.0, 2.6 * sx)))


def make_walk_frames(source_path: Path, frames: int = 64) -> list[Image.Image]:
    source_pil = Image.open(source_path).convert("RGB")
    source = np.asarray(source_pil, float) / 255.0
    height, width = source.shape[:2]
    body, tail, ear, _ = split_character(source, studio_background(width, height))
    body_core, torso_cover, legs = split_walking_layers(body)
    fur_colour = eyelid_colour(source)

    sx, sy = width / 768.0, height / 768.0
    tail_pivot = (588.0 * sx, 307.0 * sy)
    ear_pivot = (190.0 * sx, 263.0 * sy)
    eye = (153.0 * sx, 275.0 * sy)
    horizon = int(round(452 * sy))
    base_x = 16.0 * sx
    output: list[Image.Image] = []

    for index in range(int(frames)):
        phase = index / float(frames)
        # Four full strides carry one repeating meadow tile past the dog.
        stride_wave = np.sin(8.0 * np.pi * phase)
        step_wave = np.cos(8.0 * np.pi * phase)
        bob_amount = 0.5 * (1.0 - np.cos(16.0 * np.pi * phase))
        body_bob = -6.0 * sy * bob_amount
        body_sway = 1.5 * sx * np.sin(8.0 * np.pi * phase + 0.35)
        shift_x = base_x + body_sway
        travel = width * phase

        frame = _meadow_base(width, height, horizon, phase, travel=travel)
        frame.alpha_composite(_walking_shadow(width, height, bob_amount))
        frame.alpha_composite(_footstep_dust(width, height, phase))
        dog_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # The far pair is behind the coat. Diagonal pairs share a phase, producing a friendly trot.
        for leg in legs:
            if leg["depth"] != "far":
                continue
            gait = leg["sign"] * stride_wave
            lift = 32.0 * sy * max(0.0, leg["sign"] * step_wave)
            target = (leg["paw"][0] + 30.0 * sx * gait, leg["paw"][1] - lift)
            animated = _articulate_leg(leg, target)
            dog_layer.alpha_composite(_translate(animated, shift_x, body_bob))

        wag = 2.5 * np.sin(8.0 * np.pi * phase + 0.45)
        dog_layer.alpha_composite(_translate(flex_tail(tail, tail_pivot, wag), shift_x, body_bob))
        dog_layer.alpha_composite(_translate(body_core, shift_x, body_bob))

        for leg in legs:
            if leg["depth"] != "near":
                continue
            gait = leg["sign"] * stride_wave
            lift = 40.0 * sy * max(0.0, leg["sign"] * step_wave)
            target = (leg["paw"][0] + 38.0 * sx * gait, leg["paw"][1] - lift)
            animated = _articulate_leg(leg, target)
            dog_layer.alpha_composite(_translate(animated, shift_x, body_bob))

        dog_layer.alpha_composite(_translate(torso_cover, shift_x, body_bob))
        ear_angle = 5.2 * np.sin(16.0 * np.pi * phase - 0.7)
        animated_ear = ear.rotate(float(ear_angle), resample=Image.Resampling.BICUBIC, center=ear_pivot)
        dog_layer.alpha_composite(_translate(animated_ear, shift_x, body_bob))

        # A quick blink once per loop keeps the face alive without obscuring the walk read.
        blink = float(
            np.exp(-0.5 * ((phase - 0.38) / 0.024) ** 2)
            + np.exp(-0.5 * ((phase - 0.82) / 0.024) ** 2)
        )
        if blink > 0.015:
            eyelid = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(eyelid, "RGBA")
            cx, cy = eye[0] + shift_x, eye[1] + body_bob
            rx, ry = 12.0 * sx, 12.0 * sy
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fur_colour + (int(255 * min(1.0, blink * 1.35)),))
            draw.arc(
                (cx - rx, cy - 2.5 * sy, cx + rx, cy + 5.5 * sy),
                start=8,
                end=172,
                fill=(55, 30, 20, int(255 * blink)),
                width=max(2, int(round(3 * sx))),
            )
            dog_layer.alpha_composite(eyelid)

        # A tiny whole-body roll joins the separate limb motion into one soft, weighty step.
        body_roll = 0.65 * np.sin(8.0 * np.pi * phase + 0.18)
        dog_layer = dog_layer.rotate(
            float(body_roll),
            resample=Image.Resampling.BICUBIC,
            center=(405.0 * sx, 430.0 * sy),
        )
        frame.alpha_composite(dog_layer)
        frame.alpha_composite(_foreground_grass(width, height, phase, travel=travel))
        output.append(frame.convert("RGB"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("figures/better_dog_lecore.png"))
    parser.add_argument("--gif", type=Path, default=Path("figures/floofy_dog_walking.gif"))
    parser.add_argument("--mp4", type=Path, default=Path("figures/floofy_dog_walking.mp4"))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    frames = make_walk_frames(args.source, frames=args.frames)
    save_gif(frames, args.gif, args.fps)
    save_mp4(frames, args.mp4, args.fps)
    print(
        {
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
