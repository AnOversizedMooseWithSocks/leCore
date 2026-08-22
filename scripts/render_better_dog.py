#!/usr/bin/env python3
"""Build and render a friendly dog with leCore's SDF and rendering stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holographic.io_and_interop.holographic_coerce import as_camera
from holographic.mesh_and_geometry.holographic_creaturetree import bone_capsule
from holographic.mesh_and_geometry.holographic_groom import Strand, groom
from holographic.mesh_and_geometry.holographic_hairshade import render_hair
from holographic.mesh_and_geometry.holographic_mesh import Mesh
from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
from holographic.mesh_and_geometry.holographic_sdf import SDF, ellipsoid, plane, sphere, torus
from holographic.materials_and_texture.holographic_materialio import PBRMaterial
from holographic.rendering.holographic_render import Light, fit_camera, rasterize_mesh, save_image
from holographic.rendering.holographic_lights import make_light
from holographic.rendering.holographic_postfx import _fft_blur, resample
from holographic.rendering.holographic_scene_render import render_scene_document
from holographic.scene_and_pipeline.holographic_scene_doc import Scene


BODY_BOUNDS = ((-0.62, -0.98, -0.72), (0.62, 0.72, 2.24))


def _at(node: SDF, xyz) -> SDF:
    return node.translate(tuple(float(v) for v in xyz))


def _blend(nodes: list[SDF], radius: float) -> SDF:
    """Local smooth union: one coherent skin, without gluing distant legs."""
    result = nodes[0]
    for node in nodes[1:]:
        result = result.smooth_union(node, k=float(radius))
    return result


def dog_body_sdf() -> SDF:
    """A canine silhouette assembled from leCore's analytic primitives."""
    # Croup, waist, chest, raised neck, head, cheeks, and tapered muzzle.
    # A larger local blend makes the torso flow without visible primitive
    # rings while preserving the authored chest/waist silhouette.
    core = _blend(
        [
            _at(ellipsoid(0.31, 0.29, 0.35), (0.0, 0.015, 0.19)),
            _at(ellipsoid(0.29, 0.27, 0.60), (0.0, 0.015, 0.65)),
            _at(ellipsoid(0.34, 0.365, 0.38), (0.0, 0.045, 1.12)),
            bone_capsule((0.0, 0.09, 1.28), (0.0, 0.25, 1.48), 0.17),
            _at(ellipsoid(0.26, 0.25, 0.29), (0.0, 0.27, 1.59)),
            _at(ellipsoid(0.225, 0.18, 0.23), (0.0, 0.17, 1.73)),
            _at(ellipsoid(0.16, 0.13, 0.21), (0.0, 0.13, 1.88)),
        ],
        0.105,
    )

    appendages: list[SDF] = []
    for side in (-1.0, 1.0):
        xh = side * 0.225
        xf = side * 0.22
        # Offset the far-side feet along the body so all four read in the
        # three-quarter view instead of collapsing into two silhouettes.
        hz = 0.20 if side > 0.0 else 0.34
        fz = 1.10 if side > 0.0 else 1.24

        # Bent hind leg: hip -> knee -> hock -> paw.
        hind = _blend(
            [
                bone_capsule((xh, -0.08, hz), (xh, -0.37, hz + 0.25), 0.092),
                bone_capsule((xh, -0.37, hz + 0.25), (xh, -0.66, hz + 0.03), 0.068),
                bone_capsule((xh, -0.66, hz + 0.03), (xh, -0.81, hz + 0.11), 0.050),
                _at(ellipsoid(0.10, 0.064, 0.145), (xh, -0.855, hz + 0.21)),
                _at(ellipsoid(0.043, 0.042, 0.076), (xh - 0.047, -0.875, hz + 0.31)),
                _at(ellipsoid(0.043, 0.042, 0.076), (xh + 0.047, -0.875, hz + 0.31)),
            ],
            0.030,
        )
        appendages.append(hind)

        # Straighter front leg, with a compact forward-facing paw.
        front = _blend(
            [
                bone_capsule((xf, -0.10, fz), (xf, -0.46, fz + 0.025), 0.080),
                bone_capsule((xf, -0.46, fz + 0.025), (xf, -0.80, fz + 0.045), 0.059),
                _at(ellipsoid(0.095, 0.062, 0.14), (xf, -0.855, fz + 0.14)),
                _at(ellipsoid(0.041, 0.040, 0.073), (xf - 0.045, -0.875, fz + 0.235)),
                _at(ellipsoid(0.041, 0.040, 0.073), (xf + 0.045, -0.875, fz + 0.235)),
            ],
            0.028,
        )
        appendages.append(front)

    # A three-segment upward curl keeps the original lanky joke but reads as
    # a dog tail rather than a floating rod.
    tail = _blend(
        [
            bone_capsule((0.0, 0.08, -0.03), (0.025, 0.17, -0.27), 0.108),
            bone_capsule((0.025, 0.17, -0.27), (0.075, 0.34, -0.46), 0.086),
            bone_capsule((0.075, 0.34, -0.46), (0.115, 0.53, -0.49), 0.059),
        ],
        0.025,
    )
    appendages.append(tail)

    # Appendages are only lightly blended into the core. This preserves the
    # negative space between four legs—the visual fact that says "animal".
    body = core
    for item in appendages:
        body = body.smooth_union(item, k=0.035)
    return body


def ears_sdf(inner: bool = False) -> SDF:
    """Bilateral floppy ear shells; the inset layer supplies warm inner ears."""
    ears = []
    for side in (-1.0, 1.0):
        if inner:
            ears.append(
                _at(ellipsoid(0.040, 0.098, 0.036), (side * 0.358, 0.218, 1.515))
            )
        else:
            root = (side * 0.205, 0.365, 1.585)
            mid = (side * 0.280, 0.245, 1.540)
            tip = (side * 0.310, 0.105, 1.475)
            ears.append(
                _blend(
                    [
                        _at(ellipsoid(0.105, 0.125, 0.075), mid),
                        bone_capsule(root, mid, 0.075),
                        bone_capsule(mid, tip, 0.050),
                        _at(sphere(0.044), tip),
                    ],
                    0.020,
                )
            )
    return ears[0].union(ears[1])


def dark_features_sdf() -> SDF:
    """Eyes and nose, kept separate so leCore can shade them dark."""
    return _blend(
        [
            _at(sphere(0.048), (-0.175, 0.335, 1.72)),
            _at(sphere(0.058), (0.175, 0.335, 1.72)),
            _at(ellipsoid(0.088, 0.068, 0.064), (0.0, 0.145, 2.075)),
        ],
        0.008,
    )


def eye_highlights_sdf() -> SDF:
    """Tiny catchlights make the procedural face feel alert instead of vacant."""
    return _at(sphere(0.016), (-0.141, 0.357, 1.748)).union(
        _at(sphere(0.019), (0.236, 0.351, 1.744))
    )


def mouth_sdf() -> SDF:
    """A small upturned lip line on the camera-facing side of the muzzle."""
    return bone_capsule((0.130, 0.020, 1.965), (0.132, 0.055, 1.895), 0.011)


def collar_sdf() -> SDF:
    """A tilted torus following the dog's raised neck."""
    return _at(torus(0.205, 0.038).rotate((1.0, 0.0, 0.0), 0.84), (0.0, 0.150, 1.425))


def tag_sdf() -> SDF:
    """A small round identity tag hanging on the visible side of the collar."""
    return _at(ellipsoid(0.030, 0.055, 0.046), (0.178, 0.020, 1.505))


def ground_shadow_sdf() -> SDF:
    """A soft-looking contact patch built as a very flat leCore ellipsoid."""
    return _at(ellipsoid(0.46, 0.010, 1.02), (0.0, -0.925, 0.69))


def mesh_sdf(node: SDF, bounds, resolution: int) -> Mesh:
    values, axes = sample_field(node, bounds, int(resolution))
    return marching_tetrahedra_vec(values, axes, level=0.0)


def body_colours(vertices) -> np.ndarray:
    """Warm coat shading carried as per-vertex color through leCore's rasterizer."""
    v = np.asarray(vertices, float)
    height = np.clip((v[:, 1] + 0.90) / 1.55, 0.0, 1.0)[:, None]
    belly = np.array([0.84, 0.61, 0.34])
    back = np.array([0.66, 0.38, 0.19])
    colour = belly * (1.0 - height) + back * height

    # A cream muzzle/chest/sock pattern gives the silhouette readable regions
    # instead of one uninterrupted plastic-looking brown mass.
    cream = np.array([0.98, 0.82, 0.57])
    muzzle = (
        np.clip((v[:, 2] - 1.70) / 0.30, 0.0, 1.0)
        * np.clip((0.28 - v[:, 1]) / 0.24, 0.0, 1.0)
    )[:, None]
    socks = np.clip((-v[:, 1] - 0.57) / 0.26, 0.0, 1.0)[:, None]
    chest = (
        np.exp(-((v[:, 2] - 1.22) / 0.23) ** 2)
        * np.clip((0.18 - v[:, 1]) / 0.35, 0.0, 1.0)
        * np.clip((v[:, 0] + 0.04) / 0.28, 0.0, 1.0)
    )[:, None]
    marking = np.clip(0.60 * muzzle + 0.50 * socks + 0.42 * chest, 0.0, 0.72)
    colour = colour * (1.0 - marking) + cream * marking

    # Fine deterministic variation follows the procedural geometry through
    # per-vertex color, breaking the airbrushed look without a bitmap texture.
    grain = (
        np.sin(v[:, 0] * 67.0 + v[:, 2] * 31.0)
        * np.sin(v[:, 1] * 59.0 - v[:, 2] * 23.0)
    )[:, None]
    return np.clip(colour * (1.0 + 0.010 * grain), 0.0, 1.0)


def merge_coloured(items: list[tuple[Mesh, object]]) -> tuple[Mesh, np.ndarray]:
    vertices, faces, colours = [], [], []
    offset = 0
    for mesh, colour in items:
        v = np.asarray(mesh.vertices, float)
        f = np.asarray(mesh.triangulate(), int)
        vertices.append(v)
        faces.append(f + offset)
        c = np.asarray(colour, float)
        colours.append(c if c.shape == (len(v), 3) else np.tile(c, (len(v), 1)))
        offset += len(v)
    return Mesh(np.vstack(vertices), np.vstack(faces)), np.vstack(colours)


def fluffy_groom(body_node: SDF, n_strands: int = 100_000):
    """Dense leCore fur expanded from surface-authored guide strands."""
    target_count = max(1, int(n_strands))
    guide_count = min(14_000, target_count)
    guides = groom(
        body_node.eval,
        n_strands=guide_count,
        bounds=BODY_BOUNDS,
        length=0.022,
        n_pts=4,
        curl=0.0,
        lean=0.02,
        width=0.008,
        seed=19,
        length_jitter=0.32,
    )
    kept: list[Strand] = []
    for strand in guides:
        root = strand.root
        # Leave the muzzle and cream socks clean. Fur elsewhere follows the
        # analytic surface, with localized art-directed length variation.
        if root[2] > 1.78 or root[1] < -0.57:
            continue
        chest = np.exp(-((root[2] - 1.24) / 0.25) ** 2) * np.clip((0.18 - root[1]) / 0.55, 0.0, 1.0)
        tail = np.clip((-root[2] - 0.02) / 0.38, 0.0, 1.0)
        shoulder = np.exp(-((root[2] - 1.48) / 0.25) ** 2) * np.clip((root[1] + 0.10) / 0.55, 0.0, 1.0)
        length_scale = 1.0 + 1.55 * chest + 2.30 * tail + 0.42 * shoulder
        strand.points = root[None, :] + (strand.points - root[None, :]) * length_scale
        kept.append(strand)
    if not kept:
        return []
    if target_count <= len(kept):
        return kept[:target_count]

    # A compact guide groom carries the authored direction and regional
    # lengths. Deterministic tangent-plane jitter fans those guides into the
    # final dense coat without repeating expensive global surface projection.
    rng = np.random.default_rng(71)
    dense: list[Strand] = list(kept)
    while len(dense) < target_count:
        guide = kept[(len(dense) - len(kept)) % len(kept)]
        normal = np.asarray(guide.root_normal, float)
        axis = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
        tangent_a = axis - normal * float(np.dot(axis, normal))
        tangent_a /= np.linalg.norm(tangent_a) + 1e-12
        tangent_b = np.cross(normal, tangent_a)
        angle = 2.0 * np.pi * rng.random()
        radius = 0.014 * np.sqrt(rng.random())
        offset = radius * (np.cos(angle) * tangent_a + np.sin(angle) * tangent_b)
        length_variation = 0.84 + 0.32 * rng.random()
        root = guide.root + offset
        shape = (guide.points - guide.root[None, :]) * length_variation
        dense.append(
            Strand(
                root[None, :] + shape,
                root_normal=normal,
                width=guide.width,
                attrs={**guide.attrs, "dense_clone": True},
            )
        )
    return dense


def render_dog_mesh(
    output: Path,
    width: int = 768,
    height: int = 768,
    resolution: int = 88,
    fur_strands: int = 100_000,
) -> dict:
    body_node = dog_body_sdf()
    body = mesh_sdf(body_node, BODY_BOUNDS, int(resolution))
    ears = mesh_sdf(
        ears_sdf(),
        ((-0.40, -0.01, 1.37), (0.40, 0.48, 1.64)),
        max(34, int(resolution * 0.48)),
    )
    inner_ears = mesh_sdf(
        ears_sdf(inner=True),
        ((-0.40, 0.03, 1.41), (0.40, 0.42, 1.59)),
        max(30, int(resolution * 0.40)),
    )
    dark = mesh_sdf(
        dark_features_sdf(),
        ((-0.28, 0.04, 1.62), (0.28, 0.43, 2.16)),
        max(30, int(resolution * 0.52)),
    )
    highlights = mesh_sdf(
        eye_highlights_sdf(),
        ((-0.23, 0.32, 1.68), (0.23, 0.39, 1.78)),
        max(26, int(resolution * 0.34)),
    )
    mouth = mesh_sdf(
        mouth_sdf(),
        ((0.08, -0.01, 1.86), (0.17, 0.12, 2.05)),
        max(26, int(resolution * 0.42)),
    )
    collar = mesh_sdf(
        collar_sdf(),
        ((-0.28, -0.10, 1.16), (0.28, 0.42, 1.70)),
        max(42, int(resolution * 0.55)),
    )
    tag = mesh_sdf(
        tag_sdf(),
        ((0.13, -0.05, 1.44), (0.23, 0.09, 1.57)),
        max(28, int(resolution * 0.38)),
    )
    parts = [
        (body, body_colours(body.vertices)),
        (ears, (0.57, 0.30, 0.15)),
        (inner_ears, (0.66, 0.32, 0.24)),
        (collar, (0.035, 0.36, 0.34)),
        (tag, (0.92, 0.62, 0.16)),
        (dark, (0.045, 0.030, 0.022)),
        (highlights, (0.96, 0.93, 0.82)),
        (mouth, (0.20, 0.065, 0.050)),
    ]
    mesh, colours = merge_coloured(parts)
    protection_colours = np.vstack(
        [np.zeros((len(body.vertices), 3))]
        + [np.ones((len(part.vertices), 3)) for part, _ in parts[1:]]
    )

    camera = as_camera(
        fit_camera(
            mesh,
            direction=(3.45, 0.90, 1.45),
            up=(0.0, 1.0, 0.0),
            fov_deg=38.0,
            aspect=float(width) / float(height),
            margin=1.08,
        )
    )
    # Render above delivery resolution, then use leCore's resampler as SSAA.
    render_scale = 1.25
    render_width = max(int(width), int(round(width * render_scale)))
    render_height = max(int(height), int(round(height * render_scale)))
    flat_background = np.array((0.945, 0.925, 0.88), float)
    image = rasterize_mesh(
        mesh,
        camera,
        width=render_width,
        height=render_height,
        lights=[
            Light("directional", direction=(-0.42, -0.70, -0.48), intensity=0.88),
            Light("directional", direction=(0.72, -0.10, 0.28), intensity=0.28),
        ],
        ambient=0.38,
        background=tuple(flat_background),
        smooth=True,
        two_sided=True,
        vertex_colors=colours,
    )
    # A z-buffered feature ID pass prevents dense body fur from painting over
    # the nearer ears, eyes, mouth, collar, and tag. Black body geometry still
    # occludes their far sides, so this is a visibility mask rather than a
    # screen-space guess based on color.
    feature_ids = rasterize_mesh(
        mesh,
        camera,
        width=render_width,
        height=render_height,
        lights=[],
        ambient=1.0,
        background=(0.0, 0.0, 0.0),
        smooth=False,
        two_sided=True,
        vertex_colors=protection_colours,
    )

    # Warm studio cyclorama plus a feathered contact shadow. The rasterizer's
    # exact flat background acts as a clean procedural object matte.
    yy, xx = np.mgrid[0:render_height, 0:render_width]
    t = yy[..., None] / max(render_height - 1, 1)
    backdrop = (
        np.array([0.975, 0.955, 0.910])[None, None, :] * (1.0 - t)
        + np.array([0.885, 0.850, 0.780])[None, None, :] * t
    )
    floor_glow = np.exp(-((yy / render_height - 0.72) / 0.24) ** 2)[..., None]
    backdrop = np.clip(backdrop + 0.018 * floor_glow, 0.0, 1.0)

    dx = (xx / render_width - 0.515) / 0.285
    dy = (yy / render_height - 0.775) / 0.052
    shadow_alpha = 0.32 * np.exp(-2.15 * (dx * dx + dy * dy))[..., None]
    shadow_tint = np.array([0.34, 0.30, 0.24])[None, None, :]
    backdrop = backdrop * (1.0 - shadow_alpha) + shadow_tint * shadow_alpha

    matte = np.any(np.abs(image - flat_background[None, None, :]) > 1e-8, axis=2)[..., None]
    image = np.where(matte, image, backdrop)

    # leCore's strand renderer adds a true fiber-shaded fur layer. Keep the
    # coat lively while protecting the high-contrast face and accessories.
    strands = fluffy_groom(body_node, n_strands=fur_strands)
    hair_image, hair_alpha = render_hair(
        strands,
        camera,
        light_dir=(0.38, 0.72, 0.55),
        width=render_width,
        height=render_height,
        shader="kajiya",
        hair_color=(0.76, 0.49, 0.27),
        background=(0.0, 0.0, 0.0),
        smooth_levels=0,
        roughness=0.72,
        return_alpha=True,
    )
    object_pixels = matte[..., 0]
    protected = feature_ids.mean(axis=2) > 0.50

    # The groom is a coat, not a translucent effect. Convolve premultiplied
    # fiber color into an undercoat, drive dense body regions to opaque, then
    # lay the sharp individual hairs on top at full coverage.
    fur_palette = np.array([0.72, 0.43, 0.21])[None, None, :]
    fiber_light = np.clip(hair_image.mean(axis=2) / 0.62, 0.0, 1.0)
    fiber_shade = 0.78 + 0.34 * fiber_light
    toned_hair = np.clip(fur_palette * fiber_shade[..., None], 0.0, 1.0)
    blurred_alpha = _fft_blur(hair_alpha[..., None], sigma=1.50)
    blurred_premul = _fft_blur(toned_hair * hair_alpha[..., None], sigma=1.50)
    blurred_hair = blurred_premul / np.maximum(blurred_alpha, 1e-6)
    surface_light = np.clip(image.mean(axis=2) / 0.56, 0.72, 1.18)[..., None]
    lit_undercoat = np.clip(fur_palette * surface_light, 0.0, 1.0)
    blurred_hair = np.where(
        object_pixels[..., None],
        0.58 * blurred_hair + 0.42 * lit_undercoat,
        blurred_hair,
    )
    local_density = blurred_alpha[..., 0]
    coat_coverage = np.where(
        object_pixels,
        np.clip(local_density * 5.0, 0.0, 1.0),
        np.clip(local_density * 2.4, 0.0, 1.0),
    )
    dense_undercoat = object_pixels & (local_density > 0.035)
    coat_coverage = np.where(dense_undercoat, np.maximum(coat_coverage, 0.96), coat_coverage)
    coat_coverage = np.where(protected, 0.0, coat_coverage)[..., None]
    image = image * (1.0 - coat_coverage) + blurred_hair * coat_coverage

    sharp_coverage = (hair_alpha > 0.0) & (~protected)
    image = np.where(sharp_coverage[..., None], toned_hair, image)

    if render_width != int(width) or render_height != int(height):
        image = resample(image, scale=float(width) / float(render_width))
        image = image[: int(height), : int(width)]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(str(output), np.clip(image, 0.0, 1.0))
    return {
        "output": str(output),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "fur_strands": len(strands),
        "body_resolution": int(resolution),
        "body_sdf_nodes": body_node.cost()["nodes"],
        "engine": "leCore analytic SDF -> opaque groomed Kajiya-Kay fur coat -> SSAA studio composite",
    }


def coat_albedo(points) -> np.ndarray:
    """Continuous coat color sampled at path-traced surface hits."""
    p = np.atleast_2d(np.asarray(points, float))
    height = np.clip((p[:, 1] + 0.90) / 1.58, 0.0, 1.0)[:, None]
    belly = np.array([0.82, 0.55, 0.27])
    back = np.array([0.48, 0.205, 0.075])
    colour = belly * (1.0 - height) + back * height
    muzzle = np.clip((p[:, 2] - 1.66) / 0.45, 0.0, 1.0)[:, None]
    return np.clip(colour * (1.0 - 0.18 * muzzle) + np.array([0.88, 0.63, 0.31]) * (0.18 * muzzle), 0.0, 1.0)


def render_dog_pbr(output: Path, width: int = 768, height: int = 768) -> dict:
    """Render the analytic dog directly with leCore's physically based scene path."""
    body_node = dog_body_sdf()
    scene = Scene(seed=7)
    coat = PBRMaterial(
        name="warm_short_coat",
        base_color=(0.68, 0.38, 0.16, 1.0),
        metallic=0.0,
        roughness=0.72,
    )
    ear_mat = PBRMaterial(
        name="velvet_ear",
        base_color=(0.37, 0.13, 0.055, 1.0),
        metallic=0.0,
        roughness=0.82,
    )
    inner_mat = PBRMaterial(
        name="inner_ear",
        base_color=(0.55, 0.20, 0.15, 1.0),
        metallic=0.0,
        roughness=0.74,
    )
    dark_mat = PBRMaterial(
        name="eyes_and_nose",
        base_color=(0.018, 0.010, 0.006, 1.0),
        metallic=0.0,
        roughness=0.18,
    )
    mouth_mat = PBRMaterial(
        name="mouth",
        base_color=(0.22, 0.025, 0.018, 1.0),
        metallic=0.0,
        roughness=0.58,
    )

    scene.add(
        name="studio_floor",
        geometry=plane(-0.925),
        material="matte_white",
        overrides={"albedo_socket": lambda p: np.tile((0.72, 0.68, 0.60), (len(p), 1))},
    )
    scene.add(name="dog_body", geometry=body_node, material=coat,
              overrides={"albedo_socket": coat_albedo})
    scene.add(name="floppy_ears", geometry=ears_sdf(), material=ear_mat)
    scene.add(name="inner_ears", geometry=ears_sdf(inner=True), material=inner_mat)
    scene.add(name="eyes_and_nose", geometry=dark_features_sdf(), material=dark_mat)
    scene.add(name="eye_catchlights", geometry=eye_highlights_sdf(), material="ceramic_white")
    scene.add(name="mouth", geometry=mouth_sdf(), material=mouth_mat)

    framing_mesh = mesh_sdf(body_node, BODY_BOUNDS, 38)
    camera = as_camera(
        fit_camera(
            framing_mesh,
            direction=(3.45, 0.90, 1.45),
            up=(0.0, 1.0, 0.0),
            fov_deg=38.0,
            aspect=float(width) / float(height),
            margin=1.10,
        )
    )
    target = (0.0, -0.05, 0.80)
    lights = [
        make_light(
            "softbox",
            position=(2.8, 3.7, 3.4),
            target=target,
            width=2.5,
            height=2.5,
            color=(1.0, 0.82, 0.66),
            intensity=58.0,
        ),
        make_light(
            "softbox",
            position=(-2.4, 1.9, 2.1),
            target=target,
            width=2.0,
            height=2.0,
            color=(0.58, 0.70, 1.0),
            intensity=18.0,
        ),
        make_light(
            "dome",
            color=(0.40, 0.50, 0.68),
            ground_color=(0.20, 0.16, 0.12),
            intensity=0.22,
        ),
    ]
    dark_sky = lambda dirs: np.tile((0.012, 0.016, 0.024), (len(dirs), 1))
    image = render_scene_document(
        scene,
        camera,
        width=int(width),
        height=int(height),
        quality="fast",
        max_bounce=2,
        seed=11,
        sky=dark_sky,
        lights=lights,
        dome_cache=True,
        soft_light_cache=True,
        indirect_cache=True,
        demodulate=True,
        view="display",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(str(output), np.clip(image, 0.0, 1.0))
    return {
        "output": str(output),
        "size": [int(width), int(height)],
        "body_sdf_nodes": body_node.cost()["nodes"],
        "engine": "leCore analytic SDF scene -> cached soft-light PBR path tracer",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures/better_dog_lecore.png"))
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--resolution", type=int, default=88)
    parser.add_argument("--fur-strands", type=int, default=100_000)
    parser.add_argument("--renderer", choices=("pbr", "mesh"), default="mesh")
    args = parser.parse_args()
    if args.renderer == "mesh":
        result = render_dog_mesh(
            args.output,
            args.width,
            args.height,
            args.resolution,
            fur_strands=args.fur_strands,
        )
    else:
        result = render_dog_pbr(args.output, args.width, args.height)
    print(result)


if __name__ == "__main__":
    main()
