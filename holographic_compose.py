"""Forward compositional generation: run the resonator FORWARD to compose NEW scenes,
render them through the decomposition renderer, and vary them over time for procedural
video. This is the step up from the morph bundle (interpolate what's stored) to native
generation (compose what was never stored).

The decomposition path already proven elsewhere runs BACKWARD: image -> auto-tag ->
resonator factors the scene vector into its (colour, shape, texture) atoms. This module
runs the same machinery FORWARD:

    pick attribute tags  ->  encode_scene (bind + superpose)  ->  a scene vector
                         ->  make_scene (render)              ->  an actual image

and -- this is the honest part -- it does NOT just assert "we made something". A generated
scene is only meaningful if it can be ANALYSED BACK to the spec it was built from. So every
generator here is measured by ROUND-TRIP fidelity: the composed vector must factor back to
the tags it was composed from, and the rendered pixels must auto-tag back to the same
shape/colour. The combinations are drawn to be NOVEL (not a stored set), so a correct
round-trip proves genuine composition, not retrieval. Animation is just composing a frame
per step while one attribute varies; the trajectory is "real" iff each frame factors to its
intended value.

Built on the frozen core's primitives and the existing SceneCoder / make_scene; nothing new
is learned and no gradients are involved -- structure is driven forward, then verified.
"""
import itertools

import numpy as np

import holographic_scene as hs


def all_object_specs():
    """Every single-object tag-triple the attribute vocabularies can express
    (|COLOURS| x |SHAPES| x |TEXTURES|). The composable space."""
    return [{"colour": c, "shape": s, "texture": t}
            for c, s, t in itertools.product(hs.COLOURS, hs.SHAPES, hs.TEXTURES)]


def compose_object(coder, tags):
    """Run the resonator FORWARD on one object: bind its colour/shape/texture atoms into a
    single composite vector. The inverse of coder.factor()."""
    return coder.encode(tags)


def compose_scene(coder, tag_list):
    """Compose several objects into one scene vector (superpose the per-object products).
    The inverse of coder.factor_scene()."""
    return coder.encode_scene(tag_list)


def render_scene(tag_list, S=96, seed=0):
    """Render composed tags to an actual RGB image via the existing scene renderer.
    make_scene draws (shape, colour) per object; texture is carried in the vector but not
    painted, so render fidelity is judged on shape + colour."""
    specs = [(t["shape"], t["colour"]) for t in tag_list]
    return hs.make_scene(specs, S=S, seed=seed)


def roundtrip_object(coder, tags):
    """Compose one object forward, factor it back. True iff the recovered tags match --
    the proof that the composition is clean."""
    return coder.factor(compose_object(coder, tags)) == tags


def roundtrip_scene(coder, tag_list, sweeps=2):
    """Compose a multi-object scene forward, factor it back. True iff the recovered SET of
    tag-triples matches the composed set (order-free, since a scene is a superposition)."""
    got = coder.factor_scene(compose_scene(coder, tag_list), len(tag_list), sweeps=sweeps)
    key = lambda d: (d["colour"], d["shape"], d["texture"])
    return {key(t) for t in tag_list} == {key(g) for g in got}


def render_fidelity(tags, S=96, seed=0):
    """Render a single composed object and auto-tag the PIXELS back. Returns
    (shape_ok, colour_ok): does the generated image read back as the shape/colour it was
    composed from? This is what makes the output a real picture, not noise."""
    img = render_scene([tags], S=S, seed=seed)
    img = img / 255.0 if img.max() > 1 else img
    read = hs.auto_tags(img)
    return read.get("shape") == tags["shape"], read.get("colour") == tags["colour"]


def novel_specs(coder_seed=0, n=40, hold_from=None, rng=None):
    """Draw n single-object specs to GENERATE. By default these are random triples from the
    full space; pass hold_from a set of (colour,shape,texture) keys that were 'seen' and
    they are excluded, so a correct round-trip is provably composition, not recall."""
    rng = rng or np.random.default_rng(coder_seed)
    pool = all_object_specs()
    rng.shuffle(pool)
    out = []
    for t in pool:
        if hold_from and (t["colour"], t["shape"], t["texture"]) in hold_from:
            continue
        out.append(t)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Animation: compose a frame per step while ONE attribute sweeps a sequence.
# ---------------------------------------------------------------------------

def animate_attribute(coder, base_tags, attribute, values, S=96, seed=0):
    """Procedural animation: hold base_tags fixed and step `attribute` through `values`,
    composing AND rendering a frame for each. Returns a list of
    (value, scene_vector, rendered_image). The trajectory is intentional iff each frame's
    vector factors back to its intended `attribute` value (see animation_is_faithful)."""
    frames = []
    for v in values:
        tags = dict(base_tags)
        tags[attribute] = v
        vec = compose_object(coder, tags)
        img = render_scene([tags], S=S, seed=seed)
        frames.append((v, vec, img))
    return frames


def animation_is_faithful(coder, frames, attribute):
    """Fraction of animation frames whose composed vector factors back to the intended
    attribute value -- the honest measure that the sequence is a deliberate trajectory and
    not drift into noise."""
    if not frames:
        return 0.0
    ok = sum(coder.factor(vec)[attribute] == v for v, vec, _img in frames)
    return ok / len(frames)


def _demo():
    print("FORWARD COMPOSITIONAL GENERATION (compose new scenes, then verify by analysis)\n")
    coder = hs.SceneCoder(dim=1024, seed=0)
    rng = np.random.default_rng(0)

    # novel single-object compositions: compose forward, factor back
    novel = novel_specs(n=40, rng=rng)
    ok = sum(roundtrip_object(coder, t) for t in novel)
    print(f"novel single-object compose->factor : {ok}/{len(novel)} recovered exactly")

    # novel multi-object scenes
    for n in (2, 3, 4):
        good = 0
        for _ in range(30):
            tags = [{"colour": str(rng.choice(hs.COLOURS)), "shape": str(rng.choice(hs.SHAPES)),
                     "texture": str(rng.choice(hs.TEXTURES))} for _ in range(n)]
            good += roundtrip_scene(coder, tags)
        print(f"novel {n}-object scene compose->factor  : {good}/30")

    # render fidelity: the generated pixels read back as the composed shape/colour
    sh_ok = col_ok = 0
    N = 40
    for _ in range(N):
        t = {"colour": str(rng.choice([c for c in hs.COLOURS if c != "grey"])),
             "shape": str(rng.choice(hs.SHAPES)), "texture": "smooth"}
        s_ok, c_ok = render_fidelity(t, seed=int(rng.integers(1000)))
        sh_ok += s_ok; col_ok += c_ok
    print(f"render->auto-tag fidelity           : shape {sh_ok}/{N}, colour {col_ok}/{N}")

    # animation: sweep colour, every frame must factor to its intended colour
    frames = animate_attribute(coder, {"colour": "red", "shape": "circle", "texture": "smooth"},
                               "colour", ["red", "yellow", "green", "cyan", "blue"])
    print(f"colour-sweep animation faithfulness  : "
          f"{animation_is_faithful(coder, frames, 'colour'):.0%} of frames on-target")
    print("\nThese are NOVEL compositions verified by round-trip: the generated scene is real")
    print("because it can be analysed straight back to the spec it was built from.")


if __name__ == "__main__":
    _demo()
