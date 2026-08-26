"""holographic_scene_semantic.py -- describe a scene, build it, adjust its named objects in words, render or simulate.

WHAT THIS IS
------------
holographic_semantic already PARSES a description into objects and REALIZES + renders them. What was missing was the
thing a human or an agent actually wants to hold onto: a live scene you can talk to. This wraps the existing pieces in
one small, mutable SemanticScene:

    scene = scene_from_description("a big red metal sphere and a small blue glass box on a sunny day")
    scene.adjust("make the sphere bigger")        # reference a named object and change it, in plain words
    scene.adjust("change the box to metal")
    img = scene.render()                          # best-effort 3-D render of whatever it currently holds
    frames = scene.simulate(steps=40)             # a basic gravity drop of the objects

    # or start from an existing scene of named objects and adjust it semantically:
    scene = SemanticScene([{ "shape":"sphere", "color":"red", "material":"metal", "size":"big" }, ...])
    scene.set("the red sphere", material="glass")

Every object gets a human NAME (from its colour/size/material/shape, e.g. "big red metal sphere"), so you can refer to
it the way you'd describe it. Adjustment reuses the same controlled vocabulary the parser uses (SHAPES/COLORS/
MATERIALS/SIZES) -- deterministic and honest, not a general language model (see holographic_semantic's SCOPE note).

The rendering and geometry are the existing engine; simulate() is a deliberately SIMPLE rigid gravity-drop (point mass
per object, ground at y=0) so the "or simulate it" path is real end to end -- richer media (fluid/smoke/soft body) live
behind the Simulation home and take a field, not these surface objects.
"""
import os

import numpy as np

from holographic.simulation_and_physics.holographic_semantic import SHAPES, COLORS, MATERIALS, SIZES, RELATIONS, ARTICLES, _STOP, parse_description, realize_scene, find_objects, batch_set, _obj_name

# relative size words an adjust command may use, mapped to a concrete size word the parser/realizer understands
_RELATIVE_SIZE = {
    "bigger": "large", "larger": "large", "grow": "large", "enlarge": "large",
    "smaller": "small", "shrink": "small", "smallest": "tiny",
    "huge": "huge", "giant": "huge", "massive": "huge",
    "tiny": "tiny", "little": "small",
}


def parse_adjust(command):
    """Parse a plain-English edit into (reference, changes, explicit_all).
      * reference -- attribute constraints selecting which object(s) to change (e.g. {"shape":"box","color":"red"}).
      * changes   -- {field: value} to apply.
      * explicit_all -- True if the command targets ALL objects ('everything', 'all', 'it', ...).

    The one ambiguity is that a colour/material/size word can either SELECT ('the red box') or be the CHANGE ('make it
    red'). The rule: such a word is a SELECTOR when a shape noun FOLLOWS it ('red box'), otherwise it's the CHANGE
    ('make the box red'). Relative sizes ('bigger') are always a change. Deterministic and vocabulary-bounded --
    unknown words are ignored, so an unresolved target yields no selector and no change (a safe no-op upstream).

      "make the sphere bigger"        -> ({"shape":"sphere"}, {"size":"large"}, False)
      "change the red box to metal"   -> ({"shape":"box","color":"red"}, {"material":"metal"}, False)
      "make everything glass"         -> ({}, {"material":"glass"}, True)
      "make the dodecahedron golden"  -> ({}, {"material":"gold"}, False)   # no shape -> no selector -> no-op upstream
    """
    toks = command.lower().replace(",", " ").replace(".", " ").split()
    explicit_all = any(t in ("everything", "all", "every", "each", "both", "them", "they", "it") for t in toks)
    shape_positions = [i for i, t in enumerate(toks) if t in SHAPES]
    reference, changes = {}, {}
    for i, t in enumerate(toks):
        followed_by_shape = any(sp > i for sp in shape_positions)    # 'red box' -> red modifies a shape -> selector
        bucket = reference if followed_by_shape else changes
        if t in SHAPES:
            reference["shape"] = SHAPES[t]
        elif t in COLORS:
            bucket["color"] = t
        elif t in MATERIALS:
            bucket["material"] = MATERIALS[t]
        elif t in _RELATIVE_SIZE:
            changes["size"] = _RELATIVE_SIZE[t]                      # relative size is always the change
        elif t in SIZES:
            bucket["size"] = t
    return reference, changes, explicit_all


# verbs/glue an adjust command uses that aren't attributes -- so we don't mistake them for "unknown" content words
_COMMAND_WORDS = {"make", "change", "turn", "paint", "set", "into", "to", "recolor", "recolour", "and", "of", "with",
                  "on", "it", "them", "they", "everything", "all", "every", "each", "both", "please", "now",
                  "the", "a", "an", "is", "are", "be", "become", "more", "less", "very",
                  # common filler/verbs so we don't mistake them for unknown ATTRIBUTE words
                  "do", "something", "anything", "this", "that", "want", "can", "could", "would", "you", "i", "my",
                  "look", "looks", "give", "give", "let", "us", "we", "should", "will", "get", "keep", "up", "down"}
# the full palette of words a command CAN use (for 'did you mean' near-matching)
_KNOWN_VOCAB = sorted(set(SHAPES) | set(COLORS) | set(MATERIALS) | set(SIZES) | set(_RELATIVE_SIZE))


def interpret_command(scene, command, resolver=None):
    """Read an adjust command and, instead of silently failing on anything it doesn't understand, return a HELPFUL
    report: what it understood, what it will change, and -- when something is unclear -- SUGGESTIONS and clarifying
    QUESTIONS (the "did you mean ... ?" behaviour), so a person or an agent can refine the request. Applies nothing.

    Returns a dict:
      command      -- the original command
      understood   -- {"reference": {...}, "changes": {...}, "target_all": bool}   (after synonym resolution)
      read_as      -- {word: "field=value"} for out-of-vocabulary words it resolved (e.g. crimson -> color=red)
      matched      -- names of the objects the reference selects
      applied      -- whether adjust() would actually change anything
      questions    -- clarifying questions to ask back
      suggestions  -- concrete hints (known words, what's in the scene, 'did you mean X')
      unknown      -- content words it could not place at all
    """
    import difflib
    from holographic.simulation_and_physics.holographic_semantic import SynonymResolver
    if resolver is None:
        resolver = SynonymResolver()                            # table-only: deterministic, no corpus needed

    toks = command.lower().replace(",", " ").replace(".", " ").split()
    known = set(SHAPES) | set(COLORS) | set(MATERIALS) | set(SIZES) | set(_RELATIVE_SIZE) | _COMMAND_WORDS
    known |= {o["label"].lower() for o in scene.objects if o.get("label")}   # user nicknames are valid references too

    # do our best: resolve out-of-vocabulary words to known ones via the synonym table (crimson->red, chrome->metal),
    # substituting them in so parse_adjust picks them up; remember what we read for the report.
    read_as, unknown, new_toks = {}, [], []
    for t in toks:
        if t in known or not t.isalpha():
            new_toks.append(t)
            continue
        hit = None
        for field in ("shape", "color", "material", "size"):
            val = resolver.resolve(t, field)
            if val:
                hit = (field, val)
                break
        if hit:
            read_as[t] = "%s=%s" % hit
            new_toks.append(str(hit[1]))                        # the resolved value is itself a vocabulary word
        else:
            unknown.append(t)
            new_toks.append(t)

    reference, changes, explicit_all = parse_adjust(" ".join(new_toks))
    label_idx = scene._index_by_label(command)                  # a user nickname wins over attribute matching
    if label_idx:
        matched_idx = label_idx
    elif explicit_all:
        matched_idx = list(range(len(scene.objects)))
    elif reference:
        matched_idx = find_objects(scene.objects, **reference)
    else:
        matched_idx = []
    matched = [scene.objects[i]["name"] for i in matched_idx]
    applied = bool(changes) and bool(matched_idx)

    questions, suggestions = [], []
    scene_objs = ", ".join(scene.names()) if scene.objects else "nothing yet"

    # 1. words we couldn't place at all -> 'did you mean', by fuzzy match against the palette
    for w in unknown:
        close = difflib.get_close_matches(w, _KNOWN_VOCAB, n=2, cutoff=0.7)
        if close:
            suggestions.append("I don't recognise '%s' -- did you mean %s?" % (w, " or ".join(close)))
        else:
            suggestions.append("I don't recognise '%s'." % w)

    # 2. a change was asked for, but nothing was selected to change
    if changes and not applied:
        if reference:                                          # named a real attribute, but no object matched it
            questions.append("Nothing in the scene matches that. The scene has: %s. Which did you mean?" % scene_objs)
        else:                                                  # couldn't tell WHICH object
            shapes = ", ".join(sorted(set(SHAPES.values())))
            suggestions.append("I couldn't tell which object to change. The scene has: %s. Name it by shape (%s) or "
                               "colour, or say 'everything'." % (scene_objs, shapes))

    # 3. we know the target but not the change
    if (reference or explicit_all) and not changes:
        suggestions.append("I understood which object(s), but not what to change. You can set colour (e.g. 'red'), "
                           "material (e.g. 'metal', 'glass'), or size ('bigger'/'smaller').")

    # 4. nothing actionable at all
    if not changes and not reference and not explicit_all:
        suggestions.append("I didn't catch an edit. Try e.g. 'make the sphere bigger' or 'change the box to metal'. "
                           "The scene has: %s." % scene_objs)

    return {"command": command,
            "understood": {"reference": reference, "changes": changes, "target_all": explicit_all},
            "read_as": read_as, "matched": matched, "matched_idx": matched_idx, "applied": applied,
            "questions": questions, "suggestions": suggestions, "unknown": unknown}


# ---- a small, readable library of NAMED procedural textures -------------------------------------------------
# Each entry is ONE composed CMP1 texture graph. Add a texture by adding a line here; the words in the tuple are the
# names a user can say ('a rusty texture', 'make it marbled'). Kept fbm-based so they are deterministic and cheap.
def named_texture(name, seed=0):
    """Build a named procedural texture -> a CMP1 texture graph you can paint on an object (scene.paint / adjust
    'give the sphere a rusty texture'). Returns None for an unknown name so the caller can suggest the known ones;
    the known names are listed by texture_names()."""
    from holographic.materials_and_texture.holographic_texturegraph import Map, Const, field_leaf
    import numpy as _np
    n = name.lower()

    def fbm(s=seed, octaves=4):
        return field_leaf("fbm", n_dims=2, seed=s, octaves=octaves)

    if n in ("rust", "rusty", "rusted", "corroded"):
        return Map("mix", a=Const([0.50, 0.26, 0.12]), b=Const([0.88, 0.50, 0.22]), t=fbm())          # brown<->orange mottle
    if n in ("marble", "marbled", "veined"):
        veins = Map("saturate", x=Map("scale", x=fbm(octaves=6), k=Const(1.4)))
        return Map("mix", a=Const([0.95, 0.95, 0.93]), b=Const([0.50, 0.50, 0.58]), t=veins)           # white with dark veins
    if n in ("noise", "noisy", "grainy", "speckled"):
        return Map("saturate", x=Map("scale", x=fbm(octaves=5), k=Const(1.3)))                         # greyscale noise
    if n in ("cloud", "clouds", "cloudy"):
        return Map("mix", a=Const([0.52, 0.62, 0.85]), b=Const([0.98, 0.99, 1.0]), t=fbm(octaves=5))   # sky<->white puffs
    if n in ("moss", "mossy", "weathered"):
        return Map("mix", a=Const([0.28, 0.42, 0.16]), b=Const([0.58, 0.72, 0.36]), t=fbm(octaves=5))  # green weathering
    if n in ("lava", "molten", "fiery"):
        return Map("mix", a=Const([0.30, 0.05, 0.0]), b=Const([1.0, 0.55, 0.10]), t=Map("saturate", x=Map("scale", x=fbm(octaves=5), k=Const(1.5))))
    if n in ("stripe", "stripes", "striped"):
        # a stripe pattern from a raw sine FIELD (a callable is coerced to a texture leaf) blended between two colours
        stripes = (lambda pts: 0.5 + 0.5 * _np.sin(_np.asarray(pts, float)[:, 0] * 18.0))
        return Map("mix", a=Const([0.95, 0.95, 0.95]), b=Const([0.20, 0.20, 0.28]), t=stripes)
    return None


def texture_names():
    """The named textures you can ask for (one representative word each)."""
    return ["rusty", "marbled", "noisy", "cloudy", "mossy", "lava", "striped"]


# words that, in an adjust command, signal a TEXTURE request rather than an attribute change
_TEXTURE_TRIGGER = {"texture", "textured", "pattern", "patterned", "finish", "paint", "painted", "coat"}


def _texture_word_in(command):
    """If an adjust command asks for a NAMED texture, return that texture's canonical name; else None. A texture word
    on its own ('make the box rusty') or with a trigger ('give the sphere a marble texture') both count."""
    low = command.lower().replace(",", " ").replace(".", " ")
    toks = set(low.split())
    for word in toks:
        if named_texture(word) is not None:                     # the word names a texture we know how to build
            return word
    return None


# words that, in an adjust command, signal a LIGHTING request. Mapping to a LIGHTING preset (holographic_semantic.
# LIGHTING_WORDS): "make it sunset", "studio lighting", "moody look", "overcast". WHY a gate: most preset words
# ("sunset","studio","dramatic","overcast","night",...) are unambiguous -- they are NOT object attributes -- so they
# trigger on their own. The ONE collision is "golden", which is also a MATERIAL word (gold). To keep the existing
# "make it golden" -> material=gold behaviour byte-identical, an ambiguous preset word only counts as lighting when a
# lighting CUE is present ("light"/"lighting"/"lit"/"mood"/"look"/"hour"), or as the joined phrase "golden hour".
_LIGHTING_CUES = {"light", "lighting", "lit", "lights", "mood", "look", "illuminate", "illumination", "sunlight",
                  "daylight", "ambience", "ambiance", "hour", "atmosphere"}


def _lighting_word_in(command):
    """If an adjust command asks for a LIGHTING preset, return that preset's canonical name (per
    holographic_semantic.LIGHTING_WORDS); else None. Unambiguous preset words ("sunset","studio","dramatic",
    "overcast","night",...) trigger on their own; an AMBIGUOUS preset word that is also an object-attribute word
    (only "golden", which is also material=gold) triggers ONLY when a lighting cue is present -- so "make it golden"
    stays a material change while "golden hour"/"golden lighting" become the preset. Deterministic, vocabulary-bounded.
    See holographic_semantic.LIGHTING_WORDS / LIGHTING."""
    from holographic.simulation_and_physics.holographic_semantic import LIGHTING_WORDS
    low = command.lower().replace(",", " ").replace(".", " ")
    toks = low.split()
    tokset = set(toks)
    cue = bool(tokset & _LIGHTING_CUES)
    # the joined phrase "golden hour" -> "goldenhour" is an explicit, unambiguous lighting request
    for a, bb in zip(toks, toks[1:]):
        if (a + bb) in LIGHTING_WORDS:
            return LIGHTING_WORDS[a + bb]
    # a word is ambiguous if the parser would otherwise read it as an object attribute (shape/colour/material/size)
    ambiguous = set(SHAPES) | set(COLORS) | set(MATERIALS) | set(SIZES) | set(_RELATIVE_SIZE)
    for t in toks:                                             # first triggering word wins (stable, left-to-right)
        if t in LIGHTING_WORDS:
            if t in ambiguous and not cue:                     # e.g. bare "golden" -> leave it for the material path
                continue
            return LIGHTING_WORDS[t]
    return None

# relative BRIGHTNESS words (B2b): scale the whole-scene sun intensity up or down. Multiplicative so repeated
# commands compound ("brighter" twice = brighter still) and it is symmetric (up step and down step are reciprocals).
_BRIGHTEN = {"brighter", "brighten", "lighter", "lighten", "brightening"}
_DIM = {"dimmer", "dim", "darker", "darken", "dimming", "dimmed"}
_BRIGHT_INTENSIFY = {"much", "way", "far", "lot", "loads", "tons", "significantly", "considerably"}
_BRIGHT_STEP = 1.35            # one "brighter" multiplies sun intensity by this; "dimmer" by its reciprocal
_SUN_SCALE_MIN, _SUN_SCALE_MAX = 0.15, 4.0   # clamp so the scene never goes fully black or blows out


def _brightness_step(command):
    """If an adjust command asks to change the whole-scene BRIGHTNESS, return the multiplier to apply to
    environment["sun_scale"] (e.g. 1.35 for "brighter", ~0.74 for "dimmer"); else None. An intensifier ("much",
    "way", "a lot") squares the step. Returns None when a SHAPE selector is present ("make the sphere darker") --
    that is a per-object request this scene-level knob does not serve (a KEPT NEGATIVE: object-level brightness is
    deferred), so it falls through rather than silently dimming the whole scene. Deterministic, vocabulary-bounded."""
    toks = command.lower().replace(",", " ").replace(".", " ").split()
    tokset = set(toks)
    if tokset & set(SHAPES):                                   # object-selector present -> not a whole-scene request
        return None
    up = bool(tokset & _BRIGHTEN)
    down = bool(tokset & _DIM)
    if up == down:                                            # neither, or contradictory ("brighter and dimmer")
        return None
    step = _BRIGHT_STEP ** 2 if (tokset & _BRIGHT_INTENSIFY) else _BRIGHT_STEP
    return step if up else (1.0 / step)

# relative PLACEMENT (B2, "macro_relative_layout" in Blender-MCP terms): put/move one object relative to another.
# The realizer (holographic_semantic.realize_scene) ALREADY lays objects out from their "relation" (on/under/beside/
# inside/front/behind), so placement is pure REUSE -- we just set object X's relation to (relword, index-of-Y) and it
# re-lays-out on the next render. Deterministic and exact, which is the spatial precision reviewers say raw LLM->bpy
# struggles with. Prepositions are matched longest-first so "in front of" wins over "in", "on top of" over "on".
_PLACEMENT_PREPS = [                                          # (phrase words, canonical relword realize_scene knows)
    (("on", "top", "of"), "on"), (("in", "front", "of"), "front"), (("next", "to"), "beside"),
    (("on", "top"), "on"), (("upon",), "on"), (("on",), "on"), (("above",), "on"), (("over",), "on"),
    (("underneath",), "under"), (("beneath",), "under"), (("under",), "under"), (("below",), "under"),
    (("inside",), "inside"), (("within",), "inside"), (("in",), "inside"),
    (("beside",), "beside"), (("alongside",), "beside"), (("by",), "beside"),
    (("behind",), "behind"), (("front",), "front"),
]
_PLACEMENT_VERBS = {"put", "move", "place", "set", "drop", "stick", "position", "reposition", "attach", "sit", "seat"}


def _bounding_radius(shape, sdf, base_radius):
    """A conservative bounding radius for an object's SDF, from its exact size attributes -- used to size the
    sdf_to_mesh sampling box in to_node_graph(renderable=True) so the marching grid actually contains the surface.
    Readable per-shape; falls back to base_radius if an attribute is missing."""
    try:
        if shape == "sphere":
            return float(sdf.r)
        if shape == "box":
            import numpy as _np
            return float(_np.max(sdf.h))
        if shape in ("cylinder", "cone"):
            return float(max(sdf.r, sdf.hy))
        if shape == "torus":
            return float(sdf.R + sdf.rt)
    except AttributeError:
        pass
    return float(getattr(sdf, "r", base_radius))


def _placement_command(scene, command):
    """Parse "put the red sphere on top of the blue box" -> (x_index, relword, y_index), or None. Both objects are
    resolved with scene.select (nickname or attribute phrase). Longest-preposition-first so multi-word prepositions
    win. Self-placement (X on X) and unresolved references return None (a safe no-op upstream). See realize_scene
    for how the relword becomes a position."""
    toks = command.lower().replace(",", " ").replace(".", " ").split()
    # find the EARLIEST preposition occurrence; among ties at the same index, the longest phrase wins (already
    # ordered longest-first, so the first hit at the smallest index is correct).
    best = None                                               # (start_index, phrase_len, relword)
    for phrase, relword in _PLACEMENT_PREPS:
        L = len(phrase)
        for i in range(len(toks) - L + 1):
            if tuple(toks[i:i + L]) == phrase:
                if best is None or i < best[0] or (i == best[0] and L > best[1]):
                    best = (i, L, relword)
                break                                         # earliest occurrence of THIS phrase is enough
    if best is None:
        return None
    start, L, relword = best
    # GUARD against false positives: a normal command can contain "on"/"in" ("make the box on the left bigger").
    # Only treat this as placement if a placement VERB (put/move/place/...) appears before the preposition.
    if not any(t in _PLACEMENT_VERBS for t in toks[:start]):
        return None
    left = [t for t in toks[:start] if t not in _PLACEMENT_VERBS]   # drop the leading verb ("put"/"move"/...)
    right = toks[start + L:]
    if not left or not right:
        return None
    xi = scene.select(" ".join(left))
    yi = scene.select(" ".join(right))
    if not xi or not yi:
        return None
    x, y = xi[0], yi[0]                                       # first match on each side
    if x == y:                                                # "put the sphere on the sphere" -> no-op
        return None
    return (x, relword, y)

# TRANSLATE / SCALE by an explicit AMOUNT (B2, the direct-manipulation surface Blender-MCP exposes as move/scale).
# These write the OFFSET / SCALE_MUL fields realize_scene now reads. Deterministic and exact. Directions use the
# render camera convention: +x right, +y up, +z toward the viewer (so "forward" comes toward camera, "back" recedes).
_TRANSLATE_VERBS = {"move", "nudge", "shift", "translate", "slide", "push", "scoot", "reposition"}
_DIRECTIONS = {
    "left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0),
    "up": (0.0, 1.0, 0.0), "down": (0.0, -1.0, 0.0),
    "forward": (0.0, 0.0, 1.0), "forwards": (0.0, 0.0, 1.0), "toward": (0.0, 0.0, 1.0), "closer": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0), "backward": (0.0, 0.0, -1.0), "backwards": (0.0, 0.0, -1.0), "away": (0.0, 0.0, -1.0),
}


def _first_number(toks, default=1.0):
    """First numeric token as a float (e.g. "by 2" -> 2.0, "1.5x" -> 1.5); `default` if none. Readable and
    deterministic -- strips a trailing x/units so "2x"/"3units" parse."""
    for t in toks:
        s = t.rstrip("x").rstrip("units").rstrip("unit")
        try:
            return float(s)
        except ValueError:
            continue
    return float(default)


def _translate_command(scene, command):
    """Parse "move the red sphere left 2" -> (object_indices, (dx,dy,dz)); None if not a translate. Needs a move
    VERB and a DIRECTION word; the amount defaults to 1.0 unit. The target is resolved with scene.select (the shape/
    colour/nickname in the command). Multiple direction words sum (e.g. "up and right"). See realize_scene/offset."""
    toks = command.lower().replace(",", " ").replace(".", " ").split()
    if not any(v in toks for v in _TRANSLATE_VERBS):
        return None
    dirs = [np.asarray(_DIRECTIONS[t], float) for t in toks if t in _DIRECTIONS]
    if not dirs:
        return None
    amt = _first_number(toks, default=1.0)
    delta = np.sum(dirs, axis=0) * amt
    idx = scene.select(command)                               # reads the shape/colour/nickname selector out of it
    if not idx:
        return None
    return (idx, tuple(float(x) for x in delta))


# continuous SCALE words -> a multiplier (complementary to the discrete bigger/smaller size buckets).
_SCALE_UP = {"double": 2.0, "triple": 3.0, "quadruple": 4.0}
_SCALE_DOWN = {"halve": 0.5, "half": 0.5, "quarter": 0.25}
_SCALE_VERBS = {"scale", "resize", "size"}


def _scale_command(scene, command):
    """Parse a CONTINUOUS scale -> (object_indices, factor); None if not a scale request. Triggers on "scale ... "
    (up/down/by N), or the words double/triple/half/quarter/twice. "twice as big" -> 2.0, "half the size" -> 0.5,
    "scale up" -> 1.5, "scale down" -> 1/1.5, "scale by 2" -> 2.0. This is separate from the discrete bigger/smaller
    size buckets (parse_adjust) -- it is a smooth multiplier on scale_mul. See realize_scene."""
    toks = command.lower().replace(",", " ").replace(".", " ").split()
    tokset = set(toks)
    factor = None
    if "twice" in tokset or "2x" in tokset:
        factor = 2.0
    for w, f in _SCALE_UP.items():
        if w in tokset:
            factor = f
    for w, f in _SCALE_DOWN.items():
        if w in tokset:
            factor = f
    if factor is None and (tokset & _SCALE_VERBS):
        n = _first_number(toks, default=None) if any(c.isdigit() for c in command) else None
        if n is not None:
            factor = n                                        # "scale by 2" / "scale to 1.5x"
        elif "down" in tokset or "smaller" in tokset:
            factor = 1.0 / 1.5                                # "scale down"
        else:
            factor = 1.5                                      # bare "scale up" / "scale it"
    if factor is None or factor <= 0:
        return None
    idx = scene.select(command)
    if not idx:
        return None
    return (idx, float(factor))


# ROTATE / TILT: an axis + angle. Separate from translate/scale -- this is the verb that closes the 'rotation not
# modelled' negative for the semantic layer (realize_scene wraps the object in a _RotatedSDF).
_ROTATE_VERBS = {"rotate", "tilt", "turn", "lean", "spin"}
_AXIS_WORDS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _rotate_command(scene, command):
    """Parse "rotate the box 45 degrees" / "tilt the cone 30" -> (object_indices, (axis, angle_deg)); None if not a
    rotate. Needs a rotate VERB; angle defaults to 45 deg. Axis: an explicit "about x/y/z", else inferred -- tilt/lean
    about x (lean forward), rotate/turn/spin about y (turntable). A "left"/"down"/"back" direction negates the angle.
    The angle ACCUMULATES on repeat commands about the same axis. Sets an object's rotation field, which realize_scene
    turns into a _RotatedSDF. See _RotatedSDF."""
    toks = command.lower().replace(",", " ").replace(".", " ").split()
    if not any(v in toks for v in _ROTATE_VERBS):
        return None
    axis = None
    for j, t in enumerate(toks):
        if t in ("about", "around") and j + 1 < len(toks) and toks[j + 1] in _AXIS_WORDS:
            axis = _AXIS_WORDS[toks[j + 1]]
    if axis is None:                                          # infer from the verb: lean tilts, turn spins
        axis = _AXIS_WORDS["x"] if ({"tilt", "lean"} & set(toks)) else _AXIS_WORDS["y"]
    angle = _first_number(toks, default=45.0)
    if any(t in ("left", "down", "back", "counterclockwise", "ccw") for t in toks):
        angle = -angle
    idx = scene.select(command)
    if not idx:
        return None
    return (idx, (tuple(axis), float(angle)))


def _parse_name_command(command):
    """Recognise a naming command. Returns one of:
      ('rename', old, new)      for 'rename hero to champion'
      ('name', reference, label) for 'call the red sphere hero' / 'name it hero'
    or None if the command isn't about naming. Kept deliberately simple and readable."""
    import re
    low = command.strip().lower()
    m = re.match(r"^\s*rename\s+(.+?)\s+to\s+(.+?)\s*$", low)
    if m:
        return ("rename", m.group(1).strip(), m.group(2).strip())
    m = re.match(r"^\s*(?:call|name)\s+(.+)\s+(\w+)\s*$", low)   # 'call <reference...> <label>' -- label is the last word
    if m:
        return ("name", m.group(1).strip(), m.group(2).strip())
    return None


def _load_image_array(path):
    """Load an image FILE into a float (H, W, 3) array in [0,1], or None if it isn't there or can't be decoded. PIL is
    imported LAZILY here -- the core never needs it (importing leCore stays NumPy-only), and reading an external .png is
    a boundary feature. A missing/undecodable file returns None so rendering falls back to the object's colour instead
    of crashing."""
    if not os.path.exists(path):
        return None
    try:
        from PIL import Image                                 # lazy: only when someone actually renders an image file
        import numpy as _np
        return _np.asarray(Image.open(path).convert("RGB"), dtype=_np.float32) / 255.0
    except Exception:
        return None


class ExternalTexture:
    """A reference to an EXTERNAL image file used as a texture on a scene object. The scene's AssetLibrary tracks and
    repairs its path when it moves; the pixels are loaded LAZILY (and cached) the first time the scene renders it. If
    the file is missing or can't be decoded, load() returns None and the object simply renders with its flat colour --
    the scene never crashes on a bad path."""

    def __init__(self, path, role="diffuse", asset_id=None):
        self.path = path
        self.role = role
        self.asset_id = asset_id
        self._tex = None                                      # cached TextureMap once loaded
        self._loaded_from = None                              # the path we loaded, so a relink re-loads

    def load(self):
        """Return a UV-sampleable TextureMap for the current path, or None if the file can't be loaded. Re-loads if the
        path changed since last time (e.g. after a relink)."""
        if self._tex is not None and self._loaded_from == self.path:
            return self._tex
        arr = _load_image_array(self.path)
        if arr is None:
            self._tex = None
        else:
            from holographic.materials_and_texture.holographic_materialio import TextureMap
            self._tex = TextureMap(arr, wrap="repeat")
        self._loaded_from = self.path
        return self._tex


class SemanticScene:
    """A live, adjustable scene of NAMED objects plus an environment. Build it from a description or hand it a list of
    object dicts; then reference objects by how you'd describe them and change them in words, and render or simulate."""

    def __init__(self, objects, environment=None, mind=None):
        # each object is a dict {shape,color,material,size,relation}; ensure every one carries a stable descriptive
        # name, an optional user LABEL (a nickname you can rename and reference), and an optional composed TEXTURE.
        self.objects = [dict(o) for o in objects]
        for o in self.objects:
            o.setdefault("relation", None)
            o.setdefault("label", None)                         # a user-given nickname, e.g. 'hero' (None until named)
            o.setdefault("texture", None)                       # a CMP1 texture graph / material painted on it (or None)
            o["name"] = _obj_name(o)
        self.environment = dict(environment or {"sun": None, "sky": None})
        self.mind = mind
        self.feedback = None                                    # the clarifying report from the last adjust()
        self._assets = None                                     # lazy AssetLibrary for EXTERNAL texture/model files
        self.asset_roots = []                                   # folders to auto-search when a file has moved

    # ---- inspect ------------------------------------------------------------------------------------------
    def names(self):
        """How you refer to each object: its user LABEL if it has one (e.g. 'hero'), otherwise its descriptive name
        (e.g. 'big red metal sphere'). Either one works as a reference."""
        return [(o["label"] or o["name"]) for o in self.objects]

    def labels(self):
        """The user-given nicknames currently in the scene, as {label: descriptive name}. Empty until you name things."""
        return {o["label"]: o["name"] for o in self.objects if o["label"]}

    def describe(self):
        """A one-line human summary of what the scene currently holds (showing a label as 'label = description')."""
        if not self.objects:
            return "an empty scene"
        env = self.environment
        weather = ", ".join(v for v in (env.get("sun") and env["sun"] + " sun",
                                        env.get("sky") and env["sky"] + " sky") if v)
        body = "; ".join((o["label"] + " = " + o["name"]) if o["label"] else o["name"] for o in self.objects)
        return body + (" (%s)" % weather if weather else "")

    # ---- reference ----------------------------------------------------------------------------------------
    def _index_by_label(self, text):
        """If `text` mentions a user LABEL (as a whole word), return the matching object indices; else []. Checked
        BEFORE attribute matching so a nickname like 'hero' beats trying to read it as a shape/colour."""
        if not isinstance(text, str):
            return []
        import re
        low = text.lower()
        hits = []
        for i, o in enumerate(self.objects):
            lab = o.get("label")
            if lab and re.search(r"\b" + re.escape(lab.lower()) + r"\b", low):
                hits.append(i)
        return hits

    def select(self, reference):
        """Resolve a reference to object INDICES. `reference` may be a user LABEL ('hero'), a dict of attribute
        constraints, a plain phrase ('the red sphere', 'everything'), or None/'' for all objects. A label wins over
        attribute matching, so once you name something you can always reach it by that name."""
        if reference in (None, "", "all", "everything"):
            return list(range(len(self.objects)))
        by_label = self._index_by_label(reference)              # a nickname takes precedence
        if by_label:
            return by_label
        if isinstance(reference, str):
            ref_attrs, _, all_flag = parse_adjust(reference)     # read a phrase as a selector (ignore any change part)
            if all_flag:
                return list(range(len(self.objects)))
            reference = ref_attrs
        if not reference:
            return list(range(len(self.objects)))
        return find_objects(self.objects, **reference)

    def name(self, reference, label):
        """Give an object a nickname you can reference and rename: scene.name('the red sphere', 'hero'). After this,
        'hero' selects it in adjust/set/get. A label must be unique -- naming a second object 'hero' moves the name.
        Returns self (chainable)."""
        label = str(label).strip()
        if not label:
            raise ValueError("a label can't be empty")
        where = self.select(reference)
        if not where:
            self.feedback = {"applied": False, "suggestions": ["Nothing matches %r to name. The scene has: %s."
                             % (reference, ", ".join(self.names()) or "nothing")]}
            return self
        for o in self.objects:                                  # a label is unique: clear it wherever it was
            if o.get("label") == label:
                o["label"] = None
        self.objects[where[0]]["label"] = label                 # name the first match (naming is singular)
        return self

    def rename(self, old, new):
        """Rename an existing nickname: scene.rename('hero', 'champion'). Returns self."""
        new = str(new).strip()
        for o in self.objects:
            if o.get("label") == old:
                o["label"] = new
                return self
        self.feedback = {"applied": False, "suggestions": ["No object is named %r. Current names: %s."
                         % (old, ", ".join(self.labels()) or "none")]}
        return self

    def get(self, reference):
        """The object dict(s) matching a reference (a list; may be empty)."""
        return [self.objects[i] for i in self.select(reference)]

    # ---- adjust -------------------------------------------------------------------------------------------
    def set(self, reference, **fields):
        """Directly set attributes on the referenced object(s): scene.set('the red sphere', material='glass',
        size='big'). Values use the same words the parser uses. Returns self (chainable)."""
        where = self.select(reference)
        for i in where:
            for k, v in fields.items():
                self.objects[i][k] = MATERIALS.get(v, v) if k == "material" else v
            self.objects[i]["name"] = _obj_name(self.objects[i])
        return self

    def interpret(self, command):
        """PREVIEW an adjust command WITHOUT changing anything: returns the clarifying report (what it understood, what
        it would change, and -- when unclear -- suggestions + questions). This is the 'help me say it right' call; use
        it (or read .feedback after adjust) to offer the user options instead of a silent failure. See
        interpret_command."""
        return interpret_command(self, command)

    # some people will reach for suggest(); it's the same clarifying preview
    suggest = interpret

    def paint(self, reference, texture):
        """Attach a composed TEXTURE to the referenced object(s), so scene.render() paints it on. `texture` may be a
        texture NAME string ('rusty', 'marbled', ...) to use the built-in library, or a CMP1 texture graph / Material
        you built yourself. Returns self (chainable)."""
        if isinstance(texture, str):
            g = named_texture(texture)
            if g is None:
                self.feedback = {"applied": False, "suggestions": ["I don't know a %r texture. Try: %s."
                                 % (texture, ", ".join(texture_names()))]}
                return self
            texture = g
        for i in self.select(reference):
            self.objects[i]["texture"] = texture
        return self

    # ---- external asset files: track them, and repair paths when they move --------------------------------
    @property
    def assets(self):
        """The scene's AssetLibrary -- tracks the EXTERNAL image/model files objects reference, and repairs their
        paths when they move. Created on first use (a purely procedural scene never makes one)."""
        if self._assets is None:
            from holographic.misc.holographic_assets import AssetLibrary
            self._assets = AssetLibrary()
        return self._assets

    def set_asset_roots(self, roots):
        """Folders the scene may SEARCH to auto-find external files that have moved (used by render() and resolve).
        e.g. scene.set_asset_roots(['/Users/me/Projects']). Returns self."""
        self.asset_roots = list(roots)
        return self

    def attach_texture_file(self, reference, path, role="diffuse", with_hash=False):
        """Give the referenced object(s) an EXTERNAL image file as their texture. The file is registered in the scene's
        AssetLibrary (so it can be relinked/checked later) and the object renders with those pixels. Returns self."""
        ref = self.assets.add(path, role=role, with_hash=with_hash)
        et = ExternalTexture(path, role=role, asset_id=ref.id)
        for i in self.select(reference):
            self.objects[i]["texture"] = et
        return self

    def missing_assets(self):
        """The external files this scene references that are not where we last saw them (as AssetRefs)."""
        return self.assets.missing() if self._assets else []

    def check_assets(self, with_hash=False):
        """A report of every external file's status (ok / missing / modified) -- what a UI/agent would show."""
        return self.assets.report(with_hash=with_hash) if self._assets else {"counts": {}, "assets": []}

    def relink(self, asset_or_path, new_path, **kw):
        """Re-point ONE external asset to its new location and auto-find the rest (delegates to the AssetLibrary, then
        updates any object ExternalTexture whose file moved so the render picks up the new path). Returns the report."""
        rep = self.assets.relink(asset_or_path, new_path, **kw)
        self._sync_external_paths()
        return rep

    def resolve_assets(self, roots=None, with_hash=False):
        """Find missing external files by searching `roots` (defaults to the scene's asset_roots) -- by matching
        trailing folder structure, or by CONTENT HASH across machines. Updates the objects. Returns the report."""
        roots = roots if roots is not None else self.asset_roots
        report = {"relinked": [], "still_missing": []}
        for root in roots:
            r = self.assets.search_under(root, with_hash=with_hash)
            report["relinked"].extend(r["relinked"])
        report["still_missing"] = [a.path for a in self.assets.missing()]
        self._sync_external_paths()
        return report

    def _sync_external_paths(self):
        """After the AssetLibrary repairs paths, copy the current path back onto each object's ExternalTexture (matched
        by asset id) so the next render loads from the right place."""
        if not self._assets:
            return
        by_id = {a.id: a.path for a in self._assets.assets}
        for o in self.objects:
            t = o.get("texture")
            if isinstance(t, ExternalTexture) and t.asset_id in by_id:
                t.path = by_id[t.asset_id]

    def adjust(self, command):
        """Adjust the scene from a plain-English command. Handles three kinds of request:
          * NAMING:  'call the red sphere hero', 'rename hero to champion'  -> give/rename a nickname.
          * TEXTURE: 'give the sphere a rusty texture', 'make the box marbled' -> paint a named procedural texture.
          * CHANGE:  'make the sphere bigger', 'change hero to glass', 'make everything metal' -> set attributes.
        Objects can be referenced by a nickname (once named) or by description ('the red sphere'). Does its best --
        resolving synonyms (crimson->red, chrome->metal) -- and, when a command can't be placed, changes nothing and
        leaves a clarifying report in `self.feedback` rather than guessing. Returns self (chainable)."""
        # 1. a NAMING command?
        named = _parse_name_command(command)
        if named:
            if named[0] == "rename":
                self.rename(named[1], named[2])
                if self.feedback is None or self.feedback.get("applied") is not False:
                    self.feedback = {"applied": True, "did": "renamed %r to %r" % (named[1], named[2])}
            else:
                self.name(named[1], named[2])
                if self.feedback is None or self.feedback.get("applied") is not False:
                    self.feedback = {"applied": True, "did": "named that object %r" % named[2]}
            return self

        # 2. a TEXTURE request? ('rusty', 'marbled', ... optionally with 'texture'/'paint')
        texword = _texture_word_in(command)
        if texword:
            report = interpret_command(self, command)            # reuse the clarifier to resolve WHICH object (label-aware)
            where = report["matched_idx"]
            if not where and len(self.objects) == 1:
                where = [0]                                       # 'give it a rusty texture' with one object is unambiguous
            if not where:
                self.feedback = {"applied": False, "suggestions": ["Which object should get the %s texture? "
                                 "The scene has: %s." % (texword, ", ".join(self.names()))]}
                return self
            tex = named_texture(texword)
            for i in where:
                self.objects[i]["texture"] = tex
            self.feedback = {"applied": True, "did": "painted a %s texture on %s"
                             % (texword, ", ".join(self.objects[i]["name"] for i in where))}
            return self

        # 2b. a LIGHTING request? ("make it sunset", "studio lighting", "moody", "overcast"). This is a SCENE-LEVEL
        #     change (the environment), not a per-object one, so it is checked before the object-attribute path so a
        #     pure lighting command does not fall through and read as "unknown". Delegates to the environment["lighting"]
        #     slot the renderer already honours (holographic_semantic.LIGHTING) -- no new render machinery.
        preset = _lighting_word_in(command)
        if preset is not None:
            self.environment["lighting"] = preset
            self.feedback = {"applied": True, "did": "set the lighting to %r" % preset}
            return self

        # 2c. a relative BRIGHTNESS request? ("brighter", "make it dimmer", "much darker"). Scene-level: multiplies
        #     environment["sun_scale"] (clamped), which render() feeds to the sun intensity. Compounds across commands.
        step = _brightness_step(command)
        if step is not None:
            cur = float(self.environment.get("sun_scale", 1.0))
            new = max(_SUN_SCALE_MIN, min(_SUN_SCALE_MAX, cur * step))
            self.environment["sun_scale"] = new
            self.feedback = {"applied": True, "did": "%s the scene (sun_scale %.2f -> %.2f)"
                             % ("brightened" if step > 1.0 else "dimmed", cur, new)}
            return self

        # 2d. a relative PLACEMENT request? ("put the sphere on top of the box", "move the cone next to the sphere").
        #     Sets object X's relation to (relword, Y) -- the realizer re-lays-it-out. Reuses the existing relation
        #     system, so no new position field. Guarded to need a placement verb + two resolvable objects.
        place = _placement_command(self, command)
        if place is not None:
            xi, relword, yi = place
            self.objects[xi]["relation"] = (relword, yi)
            self.objects[xi]["name"] = _obj_name(self.objects[xi])
            self.feedback = {"applied": True, "did": "placed %s %s %s"
                             % (self.objects[xi]["name"], relword, self.objects[yi]["name"])}
            return self

        # 2e. TRANSLATE by an amount? ("move the sphere left 2", "nudge it up"). Accumulates into the offset field.
        tr = _translate_command(self, command)
        if tr is not None:
            idxs, delta = tr
            d = np.asarray(delta, float)
            for i in idxs:
                cur = np.asarray(self.objects[i].get("offset", (0.0, 0.0, 0.0)), float)
                self.objects[i]["offset"] = tuple(float(x) for x in (cur + d))
            self.feedback = {"applied": True, "did": "moved %s by (%.2f, %.2f, %.2f)"
                             % (", ".join(self.objects[i]["name"] for i in idxs), delta[0], delta[1], delta[2])}
            return self

        # 2f. SCALE by a continuous factor? ("scale the box up", "make the sphere twice as big", "halve it").
        sc_cmd = _scale_command(self, command)
        if sc_cmd is not None:
            idxs, factor = sc_cmd
            for i in idxs:
                cur = float(self.objects[i].get("scale_mul", 1.0))
                self.objects[i]["scale_mul"] = max(0.1, min(10.0, cur * factor))   # clamp: never vanish or explode
            self.feedback = {"applied": True, "did": "scaled %s by %.2fx"
                             % (", ".join(self.objects[i]["name"] for i in idxs), factor)}
            return self

        # 2g. ROTATE / TILT by an angle? ("tilt the cone 30 degrees", "rotate the box 45 about y").
        rot_cmd = _rotate_command(self, command)
        if rot_cmd is not None:
            idxs, (axis, angle) = rot_cmd
            for i in idxs:
                cur = self.objects[i].get("rotation")
                if cur and tuple(cur[0]) == tuple(axis):
                    self.objects[i]["rotation"] = (tuple(axis), float(cur[1]) + angle)   # accumulate same-axis
                else:
                    self.objects[i]["rotation"] = (tuple(axis), float(angle))
            self.feedback = {"applied": True, "did": "rotated %s by %.0f deg about %s"
                             % (", ".join(self.objects[i]["name"] for i in idxs), angle, tuple(axis))}
            return self

        # 3. the standard attribute CHANGE (target resolved label-aware inside interpret_command)
        report = interpret_command(self, command)
        self.feedback = report                                   # always available: what it read + any suggestions
        changes = report["understood"]["changes"]
        if changes and report["applied"]:
            for i in report["matched_idx"]:                      # label- or attribute-matched indices
                for k, v in changes.items():
                    self.objects[i][k] = v
                self.objects[i]["name"] = _obj_name(self.objects[i])
        return self

    def options(self):
        """The PALETTE of what you can say to this scene -- the object names you can reference and the attribute words
        you can set. Handy for a UI to show, or for an agent to ground its commands in. Returns a dict."""
        from holographic.simulation_and_physics.holographic_semantic import LIGHTING_WORDS
        return {"objects": self.names(),
                "shapes": sorted(set(SHAPES.values())),
                "colors": sorted(COLORS),
                "materials": sorted(set(MATERIALS.values())),
                "sizes": sorted(SIZES) + sorted(_RELATIVE_SIZE),
                "lighting": sorted(set(LIGHTING_WORDS.values()))}   # scene-level lighting presets (say 'make it sunset')

    # ---- realize / render / simulate ----------------------------------------------------------------------
    def realize(self, spacing=2.7, base_radius=0.7):
        """Turn the current objects into renderable geometry (a list of realized SDF objects). See
        holographic_semantic.realize_scene."""
        return realize_scene(self.objects, spacing=spacing, base_radius=base_radius)

    def to_node_graph(self, spacing=2.7, base_radius=0.7, materials=False, renderable=False):
        """DRILL DOWN from this high-level semantic scene to an EXACT, editable NODE GRAPH ("as above, so below").
        Each object becomes an sdf primitive node carrying its EXACT size params, wrapped in an sdf_translate at its
        EXACT laid-out position, and all are sdf_union'd into one output. Returns a dict:
          graph       -- a holographic_nodegraph.NodeGraph (drill in with describe(id), tune with set_param)
          output      -- the id of the final union node (evaluate it for the whole scene's SDF)
          objects     -- {object name -> primitive node id}, so an agent can reach ONE object's exact knobs by name
          materials   -- {object name -> material_pbr node id} when materials=True (else {}); each carries the
                         object's EXACT base_color / metallic / roughness as drill-into-able knobs.
          renderables -- {object name -> assign_material node id} when renderable=True (else {}): the object's
                         geometry MESHED (sdf_to_mesh) and PAIRED with its material -- a fully renderable node an
                         editor's viewport can draw. renderable=True implies materials=True.

        WHY: the semantic layer ('make it a big red sphere', 'move it left 2') is the fast, fuzzy way in; the node
        graph is the precise way to then set an EXACT radius/position/roughness an agent or artist wants. HONEST SCOPE:
        geometry is re-emitted faithfully (same sizes/positions) via the node palette's SDF system. Defaults
        materials=False/renderable=False -> byte-identical geometry-only bundle. renderable meshing uses a modest grid
        (res=24) per object's own bounds; a finer mesh is a set_param('res', ...) away. Deterministic."""
        from holographic.scene_and_pipeline.holographic_nodegraph import NodeGraph, default_registry
        if renderable:
            materials = True                                     # a renderable object needs a material to assign
        realized = self.realize(spacing=spacing, base_radius=base_radius)
        g = NodeGraph(default_registry())
        obj_nodes, obj_materials, obj_renderables = {}, {}, {}
        positioned = []                                          # translate-node ids to union together
        for i, ro in enumerate(realized):
            shape = self.objects[i]["shape"]
            sdf = ro["sdf"]
            pos = tuple(float(x) for x in getattr(sdf, "c", (0.0, 0.0, 0.0)))
            if shape == "sphere":
                prim = g.add("sdf_sphere", {"radius": float(sdf.r)})
            elif shape == "box":
                prim = g.add("sdf_box", {"size": tuple(float(x) for x in sdf.h)})
            elif shape == "cylinder":
                prim = g.add("sdf_cylinder", {"h": float(sdf.hy), "r": float(sdf.r)})
            elif shape == "cone":
                prim = g.add("sdf_cone", {"h": float(sdf.hy), "r": float(sdf.r)})
            elif shape == "torus":
                prim = g.add("sdf_torus", {"R": float(sdf.R), "r": float(sdf.rt)})
            else:                                                # any unmapped shape falls back to a sphere of its radius
                prim = g.add("sdf_sphere", {"radius": float(getattr(sdf, "r", base_radius))})
            obj_nodes[ro["name"]] = prim                         # drill in by object name -> its primitive node
            tr = g.add("sdf_translate", {"t": pos})
            g.connect(prim, "out", tr, "a")
            positioned.append(tr)
            if materials:
                # emit the object's material as an EXACT-param PBR node (colour/metallic/roughness are drill-into-able).
                # metallic/roughness reuse the renderer's own mapping so the node matches how the scene actually shades.
                from holographic.rendering.holographic_texturerender import _metal_for, _rough_for
                o = self.objects[i]
                col = COLORS.get(o.get("color"), (0.8, 0.8, 0.8))
                matname = o.get("material")
                emis = tuple(col) if matname == "emissive" else (0.0, 0.0, 0.0)
                mnode = g.add("material_pbr", {"name": ro["name"], "base_color": tuple(col) + (1.0,),
                                               "metallic": float(_metal_for(matname)),
                                               "roughness": float(_rough_for(matname)), "emissive": emis})
                obj_materials[ro["name"]] = mnode
            if renderable:
                # MESH the object (sdf_to_mesh) over its OWN bounds, then pair with its material -> a renderable node.
                brad = _bounding_radius(shape, sdf, base_radius) * 1.4 + 0.3   # generous box so the surface isn't clipped
                lo = tuple(float(pos[k] - brad) for k in range(3))
                hi = tuple(float(pos[k] + brad) for k in range(3))
                mesh_node = g.add("sdf_to_mesh", {"lo": lo, "hi": hi, "res": 24})
                g.connect(tr, "out", mesh_node, "a")
                asgn = g.add("assign_material")
                g.connect(mesh_node, "out", asgn, "mesh")
                g.connect(mnode, "out", asgn, "material")
                obj_renderables[ro["name"]] = asgn
        # left-fold the positioned primitives into one union (a deterministic binary chain)
        if not positioned:
            output = None
        else:
            output = positioned[0]
            for nxt in positioned[1:]:
                u = g.add("sdf_union")
                g.connect(output, "out", u, "a")
                g.connect(nxt, "out", u, "b")
                output = u
        return {"graph": g, "output": output, "objects": obj_nodes,
                "materials": obj_materials, "renderables": obj_renderables}

    # ---- the CRITIC (B7): propose ranked edits to make this scene look more like a TARGET image ----------------
    # This is the genuinely-missing stage of the image->3D self-improvement loop: given a target image and the
    # current scene, SCORE candidate semantic edits by how much each REDUCES the perceptual distance to the target,
    # and return them ranked. It reuses everything already built: the adjust() vocabulary generates candidates, render()
    # produces each trial image, and the perceptual image distance is the objective. Pure and DETERMINISTIC (seeded
    # render + deterministic metric + stable sort), so the same scene+target always yields the same proposal.
    def render_passes(self, want=("mask",), camera=None, width=200, height=150):
        """BIDIRECTIONAL LOOKUP -- render the scene and get back, per pixel, WHICH object produced it (plus the G-buffer
        passes). Returns a dict: 'beauty' (an sdf render), the requested data passes from `want`
        ('mask'/'depth'/'normal'/'position'), and one COVERAGE MATTE per object keyed by NAME ('object:<name>') -- a
        Cryptomatte-style (H,W) alpha telling you exactly which pixels are that object. This is the trace-back the
        renderer already computes internally (the union SDF's nearest-object id at each hit), now surfaced: it gives an
        EXACT per-object mask for OUR renders (no colour segmentation needed), which the focused critic and per-object
        texture/material work build on. Deterministic. See holographic_renderchannels.render_channels."""
        from holographic.simulation_and_physics.holographic_semantic import realize_scene, _UnionSDF
        from holographic.rendering.holographic_renderchannels import render_channels
        from holographic.rendering.holographic_render import Camera
        realized = self.realize()
        sdfs = [r["sdf"] for r in realized]
        if not sdfs:
            return {}
        union = _UnionSDF(sdfs)
        if camera is None:
            span = max(3.0, 1.6 * len(sdfs))
            camera = Camera(eye=(span * 0.4, span * 0.28, span), target=(0, 0, 0), fov_deg=42.0)
        passes = render_channels(union, camera, want=list(want), width=width, height=height, objects=sdfs)
        # rekey the per-object mattes from index -> the object's descriptive NAME (the bidirectional handle)
        out = {}
        for key, buf in passes.items():
            if key.startswith("object:"):
                i = int(key.split(":", 1)[1])
                out["object:%s" % realized[i]["name"]] = buf
            else:
                out[key] = buf
        return out

    def _clone(self):
        """A shallow-per-object copy of this scene, so a trial edit does not mutate the original. Object dicts are
        fresh (edits replace values, not shared sub-objects like a texture graph). See propose_edits."""
        return SemanticScene([dict(o) for o in self.objects], dict(self.environment), mind=self.mind)

    def _default_edit_candidates(self, materials=("metal", "glass", "matte", "gold"),
                                 colors=("red", "blue", "green", "white"), geometry=False, move_step=1.5):
        """The default edit vocabulary the critic tries: whole-scene LIGHTING/brightness moves, plus per-object
        MATERIAL and COLOUR swaps (skipping an object's current value). With geometry=True it ALSO proposes coarse
        TRANSFORM edits per object -- move left/right/up/down by move_step and scale up/down -- so the critic can fix
        POSITION and SIZE, not just appearance (measured: the perceptual metric ranks coarse moves correctly; it is
        only FINE geometry it cannot see). Bounded and deterministic. Override by passing your own `candidates`."""
        cmds = ["make it sunset", "make it night", "studio lighting", "make it dramatic", "make it overcast",
                "make it brighter", "make it dimmer"]
        for o in self.objects:
            name = o.get("name") or _obj_name(o)
            for mat in materials:
                if o.get("material") != mat:
                    cmds.append("make the %s %s" % (name, mat))
            for col in colors:
                if o.get("color") != col:
                    cmds.append("make the %s %s" % (name, col))
            if geometry:
                # coarse transform edits -- position (4 planar directions) + size. Depth (forward/back) is omitted to
                # bound the render count; exact positioning is the node-graph drill-down's job, not the critic's.
                for direction in ("left", "right", "up", "down"):
                    cmds.append("move the %s %s %s" % (name, direction, move_step))
                cmds.append("scale the %s up" % name)
                cmds.append("scale the %s down" % name)
        return cmds

    def propose_edits(self, target_image, candidates=None, camera=None, width=96, height=72, quality="fast",
                      top=8, geometry=False, focus=None):
        """THE CRITIC. Score each candidate semantic edit by how much it moves the RENDER toward `target_image`, and
        return the best `top` as a ranked list of dicts: {command, distance, improvement} (improvement = baseline
        distance MINUS the edit's distance; positive means it helps), best first. Nothing is applied to this scene --
        each candidate is tried on a clone. `candidates` defaults to _default_edit_candidates(geometry=geometry).
        With geometry=True the vocabulary ALSO includes coarse move/scale edits (fixes position/size, not just
        appearance). Deterministic.

        `focus` UN-BLINDS the critic to object-level edits by scoring only the SUBJECT region instead of the whole
        frame (where the background dominates the metric): None = whole frame; a (r0,c0,r1,c1) bbox = crop to it;
        'auto' = derive the subject bbox from the target by segmentation (segment_image foreground). Measured: a
        matching backdrop is most of a whole-frame score, so fine geometry/texture only registers under focus.

        This is the 'plan a change to better match the target' step of the image->3D refinement loop; feed the top
        command back into adjust() (or use refine_to_target to do that greedily)."""
        from holographic.io_and_interop.holographic_imagecompare import perceptual_distance
        import numpy as _np
        tgt = _np.asarray(target_image, float)
        bbox = _resolve_focus(tgt, focus)                       # None (whole frame) or a crop box
        _d = lambda a, b: float(perceptual_distance(_crop(a, bbox), _crop(b, bbox)))
        base_img = _np.asarray(self.render(camera=camera, width=width, height=height, quality=quality), float)
        base_d = _d(base_img, tgt)
        if candidates is None:
            candidates = self._default_edit_candidates(geometry=geometry)
        scored = []
        for cmd in candidates:
            trial = self._clone()
            trial.adjust(cmd)
            fb = trial.feedback
            if not (fb and fb.get("applied")):                  # the edit did nothing (unresolved ref) -> skip it
                continue
            img = _np.asarray(trial.render(camera=camera, width=width, height=height, quality=quality), float)
            d = _d(img, tgt)
            scored.append({"command": cmd, "distance": d, "improvement": base_d - d})
        # best improvement first; deterministic tie-break by command so the ranking is stable run-to-run
        scored.sort(key=lambda s: (-s["improvement"], s["command"]))
        return {"baseline_distance": base_d, "proposals": scored[:int(top)]}

    def refine_to_target(self, target_image, max_steps=4, min_improvement=1e-3, candidates=None,
                         camera=None, width=96, height=72, quality="fast", geometry=False, focus=None):
        """A bounded GREEDY driver over the critic: repeatedly propose_edits, APPLY the single best edit if it improves
        the match by at least `min_improvement`, and stop when nothing helps or `max_steps` is reached. Mutates this
        scene (it is the refinement). With geometry=True it may also move/scale objects; `focus` scores only the subject
        region (see propose_edits) so object edits register. Returns {applied, start_distance, final_distance, steps,
        history}. Deterministic."""
        from holographic.io_and_interop.holographic_imagecompare import perceptual_distance
        import numpy as _np
        tgt = _np.asarray(target_image, float)
        bbox = _resolve_focus(tgt, focus)
        start_d = float(perceptual_distance(_crop(_np.asarray(self.render(camera=camera, width=width, height=height,
                                                                          quality=quality), float), bbox), _crop(tgt, bbox)))
        applied, history = [], []
        cur_d = start_d
        for step in range(int(max_steps)):
            rep = self.propose_edits(target_image, candidates=candidates, camera=camera, width=width,
                                     height=height, quality=quality, top=1, geometry=geometry, focus=focus)
            if not rep["proposals"]:
                break
            best = rep["proposals"][0]
            if best["improvement"] < float(min_improvement):    # nothing helps enough -> converged
                break
            self.adjust(best["command"])                        # commit the best edit
            applied.append(best["command"])
            cur_d = best["distance"]
            history.append({"step": step, "command": best["command"], "distance": cur_d})
        return {"applied": applied, "start_distance": start_d, "final_distance": cur_d,
                "steps": len(applied), "history": history}

    def render(self, camera=None, width=200, height=150, quality="fast", **kw):
        """Best-effort 3-D RENDER of whatever the scene currently holds. Builds a default pulled-back camera if none is
        given, applies the environment (sun/sky), and calls the engine's single-pass renderer (quality='fast') or the
        Monte-Carlo path tracer (quality='hyperreal'). Returns an (H,W,3) image array. See holographic_semantic."""
        from holographic.simulation_and_physics.holographic_semantic import render_scene, render_scene_pbr
        from holographic.rendering.holographic_render import Camera
        objs = self.objects                                     # render_scene realizes these internally
        if camera is None:                                      # a sensible default view of the whole scene
            span = max(3.0, 1.6 * len(objs))
            camera = Camera(eye=(span * 0.4, span * 0.28, span), target=(0, 0, 0), fov_deg=42.0)
        # if any object carries a composed texture, use the TEXTURED renderer so those paints show (keyed by the
        # object's descriptive name, which is exactly what render_textured matches on).
        # if any object references EXTERNAL files, make sure their paths are current (auto-search the known roots for
        # anything that moved) before we try to load pixels.
        if self._assets is not None and self.asset_roots and self.assets.missing():
            self.resolve_assets()
        # build the {object name: texture} dict the textured renderer wants. An ExternalTexture is loaded to its pixels
        # here; if its file is missing/undecodable, load() returns None and we DROP it -- that object renders with its
        # flat colour rather than crashing on a bad path.
        textured = {}
        for o in objs:
            t = o.get("texture")
            if t is None:
                continue
            if isinstance(t, ExternalTexture):
                loaded = t.load()
                if loaded is not None:
                    textured[o["name"]] = loaded
            else:
                textured[o["name"]] = t
        # B2c: the textured and PBR renderers now honour lighting too. Pull the preset + brightness from the
        # environment and pass them (default None/1.0 -> byte-identical). Read them up front so all three paths agree.
        env_lighting = self.environment.get("lighting")
        env_sun_scale = float(self.environment.get("sun_scale", 1.0))
        if textured:
            from holographic.rendering.holographic_texturerender import render_textured
            tk = dict(kw)
            tk.setdefault("lighting", env_lighting)
            tk.setdefault("sun_scale", env_sun_scale)
            return render_textured(self, textured, camera=camera, width=width, height=height, **tk)
        sun = self.environment.get("sun") or "bright"
        sky = self.environment.get("sky") or "clear"
        lighting = env_lighting
        if quality == "hyperreal":
            if lighting is not None and "lighting" not in kw:   # B2c: PBR sun direction follows the preset
                kw["lighting"] = lighting
            return render_scene_pbr(objs, camera, width=width, height=height, sun=sun, sky=sky, **kw)
        if lighting is not None and "lighting" not in kw:
            kw["lighting"] = lighting
        # B2b: relative brightness. environment["sun_scale"] (default 1.0) multiplies the sun intensity so a spoken
        # "brighter"/"dimmer" changes the render. Only the FAST renderer threads it today (render_scene_pbr and the
        # TEXTURED renderer do not -- kept negatives in NOTES). An explicit sun_scale= in kw wins over the environment.
        if "sun_scale" not in kw:
            kw["sun_scale"] = float(self.environment.get("sun_scale", 1.0))
        # environment floor/wall colours (default None -> the renderer's neutral gray floor + sky, byte-identical).
        # scene_from_image sets these from the photo's floor/wall regions so the render has a matching backdrop.
        if "ground_color" not in kw and self.environment.get("ground_color") is not None:
            kw["ground_color"] = self.environment["ground_color"]
        if "backdrop" not in kw and self.environment.get("backdrop_color") is not None:
            kw["backdrop"] = self.environment["backdrop_color"]
        return render_scene(objs, camera, width=width, height=height, sun=sun, sky=sky, **kw)

    def simulate(self, steps=40, dt=0.05, gravity=9.8, start_height=3.0, spacing=2.7, base_radius=0.7):
        """A deliberately SIMPLE rigid-body drop of the scene's objects: each object is a point mass that starts above
        its layout position and falls under gravity until it rests on the ground (y = its radius). Returns a list of
        `steps` frames, each {object_name: (x, y, z)} -- a trajectory you can animate or feed to the renderer per frame.

        This is the honest, minimal 'or simulate it' path so the whole flow works end to end. For real participating
        media (smoke/fire/fluid) or deformable bodies, use the Simulation home (holographic_fluid / _softbody / ...),
        which operate on a field rather than these surface objects."""
        realized = self.realize(spacing=spacing, base_radius=base_radius)
        # lay out x,z from the realized geometry; give each object a radius and a staggered start height so they settle
        xs, zs, radii, names = [], [], [], []
        for k, ro in enumerate(realized):
            # realized objects carry an SDF whose first arg encodes size; fall back to base_radius
            r = base_radius * (1.0 if not self.objects[k].get("size") else _size_scale(self.objects[k]["size"]))
            xs.append((k - (len(realized) - 1) / 2.0) * spacing)
            zs.append(0.0)
            radii.append(max(0.2, r))
            names.append(realized[k].get("name", self.objects[k].get("name", "obj%d" % k)))
        y = [start_height + 0.4 * k for k in range(len(realized))]   # staggered so they don't all land at once
        v = [0.0] * len(realized)
        frames = []
        for _ in range(int(steps)):
            for i in range(len(realized)):
                v[i] -= gravity * dt
                y[i] += v[i] * dt
                floor = radii[i]                                 # rest when the object's bottom touches the ground
                if y[i] <= floor:
                    y[i] = floor
                    v[i] = 0.0                                    # inelastic stop (a simple, stable settle)
            frames.append({names[i]: (float(xs[i]), float(y[i]), float(zs[i])) for i in range(len(realized))})
        return frames

    # ---- encode (VSA) -------------------------------------------------------------------------------------
    def encode(self, mind=None):
        """Encode the scene into ONE composable, content-addressable hypervector (superpose bind(OBJ_i, record_i)).
        Needs a mind (this scene's, or one passed in). Returns (scene_vector, records, roles). See
        holographic_semantic.encode_scene."""
        from holographic.simulation_and_physics.holographic_semantic import encode_scene
        m = mind or self.mind
        if m is None:
            raise ValueError("encode() needs a mind -- build the scene with scene_from_description(text, mind) "
                             "or pass encode(mind=...)")
        return encode_scene(self.objects, m)


def _size_scale(size_word):
    """The uniform scale a size word implies (for the simple simulator's radius). Stretch sizes use their mean."""
    spec = SIZES.get(size_word)
    if not spec:
        return 1.0
    kind, val = spec
    return float(val) if kind == "uniform" else float(np.mean(val))


def interpret_description(text):
    """Report what a scene description will (and won't) produce, WITHOUT building it -- so when the parser gets little
    or nothing, we can suggest how to phrase it rather than just handing back an empty scene. Returns
    {objects, environment, unknown, suggestions}. Uses the synonym table to note words it can map (crimson->red)."""
    import difflib
    from holographic.simulation_and_physics.holographic_semantic import SynonymResolver
    resolver = SynonymResolver()
    parsed = parse_description(text)
    toks = text.lower().replace(",", " ").replace(".", " ").split()
    known = (set(SHAPES) | set(COLORS) | set(MATERIALS) | set(SIZES) | set(RELATIONS) | ARTICLES | _STOP |
             {"sun", "sunny", "sky", "cloud", "clouds", "cloudy", "overcast", "clear", "partly", "day", "bright", "soft"})
    unknown, suggestions = [], []
    for t in toks:
        if t in known or not t.isalpha():
            continue
        mapped = next((resolver.resolve(t, f) for f in ("shape", "color", "material", "size") if resolver.resolve(t, f)), None)
        if mapped:
            suggestions.append("I can read '%s' as '%s'." % (t, mapped))
        else:
            close = difflib.get_close_matches(t, _KNOWN_VOCAB, n=2, cutoff=0.7)
            if close:
                suggestions.append("I don't know '%s' -- did you mean %s?" % (t, " or ".join(close)))
            else:
                unknown.append(t)
    if not parsed["objects"]:
        suggestions.append("I didn't find any objects. I understand shapes (%s), colours, materials and sizes -- e.g. "
                           "'a big red metal sphere and a small blue glass box'." % ", ".join(sorted(set(SHAPES.values()))))
    return {"objects": parsed["objects"], "environment": parsed["environment"],
            "unknown": unknown, "suggestions": suggestions}


def _crop(img, bbox):
    """Crop an (H,W,..) image to a (r0,c0,r1,c1) bbox, or return it unchanged if bbox is None. Used by the focused
    critic to score only the subject region (removing the background's domination of a whole-frame metric)."""
    if bbox is None:
        return img
    r0, c0, r1, c1 = bbox
    return np.asarray(img)[r0:r1 + 1, c0:c1 + 1]


def _subject_bbox(image, k=5, pad=0.06, top=2):
    """The bounding box of the SUBJECT of a photo -- the union bbox of the most OBJECT-LIKE foreground regions (ranked
    by _object_score: compact + central), NOT of every region tagged foreground (over-segmentation would sprawl that to
    the whole frame). Padded a little. Returns (r0,c0,r1,c1), or the whole frame if segmentation finds no clear subject.
    This is what the focused critic scores within, so the background stops dominating the metric."""
    image = np.asarray(image, float)
    if image.max() > 1.5:
        image = image / 255.0
    H, W = image.shape[:2]
    from holographic.misc.holographic_vision import segment_image
    regions = segment_image(image, k=k)
    roles = [_region_role(r, H, W) for r in regions]
    fg = [r for r, ro in zip(regions, roles) if ro == "object"]
    if not fg:
        return (0, 0, H - 1, W - 1)
    fg.sort(key=lambda r: -_object_score(r, H, W))
    subj = fg[:max(1, int(top))]                             # the most object-like few, not the sprawling union
    r0 = min(r["bbox"][0] for r in subj); c0 = min(r["bbox"][1] for r in subj)
    r1 = max(r["bbox"][2] for r in subj); c1 = max(r["bbox"][3] for r in subj)
    dr, dc = int(pad * H), int(pad * W)
    return (max(0, r0 - dr), max(0, c0 - dc), min(H - 1, r1 + dr), min(W - 1, c1 + dc))


def _resolve_focus(target_image, focus):
    """Turn the critic's `focus` argument into a crop bbox: None -> None (whole frame); 'auto' -> the subject bbox from
    segmentation; a 4-tuple -> itself."""
    if focus is None:
        return None
    if isinstance(focus, str) and focus == "auto":
        return _subject_bbox(target_image)
    return tuple(int(x) for x in focus)


def _nearest_color_name(rgb):
    """The named COLOR closest to an (r,g,b) in [0,1] by L2 -- so a photo region's mean colour becomes a word the
    semantic scene understands. Approximate (COLORS are linear rgb, a photo is sRGB), but good enough to pick a name."""
    rgb = np.asarray(rgb, float)[:3]
    return min(COLORS, key=lambda name: float(np.sum((np.asarray(COLORS[name], float) - rgb) ** 2)))


def _region_to_object_spec(region, H, W, scene_w=4.4, scene_h=3.0):
    """STAGE B (silhouette path): map ONE segmented region to a SemanticScene object spec (shape/color/position/size)
    from its 2-D stats -- NOT from unprojected depth. WHY silhouette not point-cloud: measured, shape-from-shading
    depth is a brightness RELIEF, so unprojecting a region and fit_primitives'ing it returns nonsense (a wall came back
    a 'capsule'); the clean signal is the region's own silhouette (classify_shape + aspect/extent) + mean colour.
      shape:   circle->sphere, triangle->cone, line->cylinder(thin); rectangle-> cylinder if TALLER than wide
               (pot/column) else box.
      color:   nearest named colour to the region mean.
      position: the region centroid mapped into scene space (image x-> +x, image y-> +y up), so objects sit where the
               photo puts them (uses the absolute-position override in realize_scene).
      size:    scale_mul from the region's bounding-box diagonal vs the image diagonal.
    Deterministic. Returns an object dict ready for SemanticScene."""
    r0, c0, r1, c1 = region["bbox"]
    bh, bw = (r1 - r0 + 1), (c1 - c0 + 1)
    shp = region["shape"]
    if shp == "circle":
        shape = "sphere"
    elif shp == "triangle":
        shape = "cone"
    elif shp == "line":
        shape = "cylinder"
    else:                                                    # rectangle: tall -> cylinder (pot/column), else box
        shape = "cylinder" if bh > 1.15 * bw else "box"
    cy, cx = region["centroid"]
    x = (cx / max(W - 1, 1) - 0.5) * scene_w                 # image column -> scene x (centered)
    y = (0.5 - cy / max(H - 1, 1)) * scene_h                 # image row -> scene y (flip: image y is down)
    diag = float(np.hypot(bh, bw)) / float(np.hypot(H, W))   # region size fraction of the image
    scale_mul = float(np.clip(diag * 3.2, 0.25, 2.6))
    color = _nearest_color_name(region["mean_color"])
    material = "ceramic" if (sum(region["mean_color"]) / 3.0 > 0.55 and color in ("white", "gray", "grey")) else "matte"
    return {"shape": shape, "color": color, "material": material, "size": None,
            "position": (float(x), float(y), 0.0), "scale_mul": scale_mul,
            "name": "%s %s" % (color, shape), "relation": None}


def _region_role(region, H, W):
    """Coarse role of a region for scene assembly: 'background' (a big region that touches an image EDGE -- a wall
    hugging the top, a floor spanning the bottom/sides) vs 'object' (a compact, interior region -- the things we
    actually model). Heuristic, honest: photos put backdrop against the frame and subjects toward the middle."""
    r0, c0, r1, c1 = region["bbox"]
    touches_edge = (r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1)
    wide = (c1 - c0 + 1) > 0.6 * W or (r1 - r0 + 1) > 0.7 * H
    if region["fraction"] > 0.10 and touches_edge and wide:
        return "background"
    return "object"


def _object_score(region, H, W):
    """How OBJECT-like a region is (for ranking foreground candidates): compact (fills its bbox), reasonably central,
    and sizable -- so a centered pot outranks a sprawling floor fragment even when the fragment has more pixels."""
    cy, cx = region["centroid"]
    centrality = 1.0 - float(np.hypot((cx / max(W - 1, 1)) - 0.5, (cy / max(H - 1, 1)) - 0.5)) / 0.7071
    return float(region["extent"]) * max(centrality, 0.05) * float(np.sqrt(region["area"]))


def scene_from_image(image, k=5, seed=0, max_objects=3, background=False, mind=None):
    """STAGE B+C -- build a SemanticScene FROM A PHOTO (machine-initialised, not hand-authored). Segments the image
    (segment_image), assigns each region a coarse role (background wall/floor vs foreground object), maps the top
    `max_objects` foreground regions to primitive specs (_region_to_object_spec), and assembles a live SemanticScene
    you can then adjust(), render(), refine_to_target(), or drill into via to_node_graph(). Returns
    {scene, regions, roles, objects} (regions = the raw segmentation, objects = the specs used). Deterministic.

    HONEST SCOPE: shape comes from the region SILHOUETTE (colour segmentation), colour from the region mean; DEPTH is
    not reconstructed (shape-from-shading is a brightness relief -- measured too degenerate to fit primitives to), so
    z=0 for every object and front/back ordering is not recovered. background=True also adds the wall/floor regions as
    large flat boxes (coarse). This is a machine STARTING POINT for the critic/drill-down to refine, not a metric
    reconstruction."""
    image = np.asarray(image, float)
    if image.max() > 1.5:
        image = image / 255.0
    H, W = image.shape[:2]
    from holographic.misc.holographic_vision import segment_image
    regions = segment_image(image, k=k, seed=seed)
    roles = [_region_role(r, H, W) for r in regions]
    objs, used = [], []
    fg = [r for r, ro in zip(regions, roles) if ro == "object"]
    fg.sort(key=lambda r: -_object_score(r, H, W))          # most object-like first (compact + central), not raw area
    for r in fg[:int(max_objects)]:
        spec = _region_to_object_spec(r, H, W)
        objs.append(spec); used.append(spec)
    env = {"lighting": "overcast"}
    if background:
        # use the wall/floor REGION colours as the render's backdrop + ground (cleaner than placeholder boxes): the
        # topmost background region -> wall colour, the bottom-most -> floor colour.
        bg = [(r, ro) for r, ro in zip(regions, roles) if ro == "background"]
        if bg:
            top = min(bg, key=lambda t: t[0]["centroid"][0])[0]      # highest region -> wall
            bot = max(bg, key=lambda t: t[0]["centroid"][0])[0]      # lowest region -> floor
            env["backdrop_color"] = tuple(float(c) for c in top["mean_color"])
            env["ground_color"] = tuple(float(c) for c in bot["mean_color"])
    scene = SemanticScene(objs, env, mind=mind)
    return {"scene": scene, "regions": regions, "roles": roles, "objects": used}


def scene_from_description(text, mind=None):
    """Describe a scene in (controlled) natural language and get a live, adjustable SemanticScene back: the system
    parses the description, builds named objects + the environment, and hands you something you can adjust(),
    render(), simulate(), or encode(). The one-call 'describe it and let the system build it' entry point.

    If the description parses to NO objects (or leaves words unrecognised), the returned scene carries a clarifying
    report in `scene.feedback` -- suggestions for how to phrase it -- rather than being a silent empty scene."""
    from holographic.simulation_and_physics.holographic_semantic import SynonymResolver
    parsed = parse_description(text, resolver=SynonymResolver())     # do our best: map known synonyms (crimson->red)
    scene = SemanticScene(parsed["objects"], parsed["environment"], mind=mind)
    report = interpret_description(text)
    if not parsed["objects"] or report["unknown"] or report["suggestions"]:
        scene.feedback = report                                  # offer help instead of a silent (possibly empty) scene
    return scene


def _selftest():
    # describe -> build
    sc = scene_from_description("a big red metal sphere and a small blue glass box on a sunny day")
    assert len(sc.objects) == 2, sc.names()
    assert sc.environment["sun"] == "bright"
    names = sc.names()
    assert any("sphere" in n for n in names) and any("box" in n for n in names)

    # reference + adjust semantically
    sc.adjust("make the sphere bigger")
    sph = sc.get({"shape": "sphere"})[0]
    assert sph["size"] == "large", sph
    sc.adjust("change the box to metal")
    box = sc.get({"shape": "box"})[0]
    assert box["material"] == "metal", box
    sc.adjust("make everything glass")
    assert all(o["material"] == "glass" for o in sc.objects)

    # direct set + reference by description
    sc.set("the glass sphere", color="green")
    assert sc.get({"shape": "sphere"})[0]["color"] == "green"

    # honest no-op on an unmatched / unknown reference -- AND now it explains itself instead of failing silently
    sc.adjust("make the dodecahedron golden")                    # no such object, and 'dodecahedron' is unknown vocab
    assert all(o["material"] == "glass" for o in sc.objects), "unknown ref changed nothing"
    fb = sc.feedback
    assert not fb["applied"] and fb["suggestions"], fb          # it left suggestions
    assert any("dodecahedron" in s for s in fb["suggestions"])  # named the word it didn't know

    # does its best: resolve a known synonym (scarlet -> red) and say so
    sc.set("the box", material="matte")
    rep = sc.interpret("make the box scarlet")
    assert rep["read_as"].get("scarlet") == "color=red", rep
    sc.adjust("make the box scarlet")
    assert sc.get({"shape": "box"})[0]["color"] == "red"

    # preview: understood the target but not the change -> a helpful suggestion, nothing applied
    rep = sc.interpret("do something to the sphere")
    assert not rep["applied"] and rep["suggestions"]

    # options(): the palette a UI/agent can show
    opt = sc.options()
    assert "sphere" in opt["shapes"] and "metal" in opt["materials"] and opt["objects"]

    # NAMED OBJECTS: give a nickname, reference by it, rename it -----------------------------------------
    sc.name("the sphere", "hero")
    assert sc.labels().get("hero"), sc.labels()
    assert "hero" in sc.names()                                 # names() shows the nickname
    sc.adjust("make hero metal")                                # reference the object by its nickname
    assert sc.get("hero")[0]["material"] == "metal"
    sc.adjust("rename hero to champion")                        # rename via a plain command
    assert "champion" in sc.labels() and "hero" not in sc.labels()
    sc.adjust("call the box crate")                             # name a second object via a command
    assert "crate" in sc.labels()

    # TEXTURES: paint a named procedural texture by talking to the scene, and via the API -----------------
    sc.adjust("give champion a rusty texture")
    assert sc.get("champion")[0]["texture"] is not None, "the rusty texture should be attached"
    sc.paint("crate", "marbled")                               # the direct API, with a name from the library
    assert sc.get("crate")[0]["texture"] is not None
    # an unknown texture name is refused with a helpful suggestion, nothing attached
    sc.paint("crate", "zebra")
    assert not sc.feedback["applied"] and sc.feedback["suggestions"]

    # RENDER routes through the textured renderer when a texture is present (real image, in range)
    img = sc.render(width=48, height=40)
    assert np.asarray(img).shape == (40, 48, 3) and float(np.asarray(img).std()) > 0.02

    # simulate: a real gravity drop that settles on the ground
    frames = sc.simulate(steps=30)
    assert len(frames) == 30
    first_y = list(frames[0].values())[0][1]
    last_y = list(frames[-1].values())[0][1]
    assert last_y < first_y and last_y > 0.0, (first_y, last_y)   # fell, and rests above the ground

    # LIGHTING: talk to a live scene's lighting; delegates to environment["lighting"] (renderer honours it). ------
    # Regression trap for the "golden" collision (also a material word): bare "make it golden" must stay a MATERIAL
    # change, while an explicit lighting cue routes to the preset. Assert the exact contract, not just "no exception".
    from holographic.simulation_and_physics.holographic_scene_semantic import _lighting_word_in
    assert _lighting_word_in("make it sunset") == "sunset"
    assert _lighting_word_in("studio lighting") == "studio"
    assert _lighting_word_in("give it a moody look") == "dramatic"
    assert _lighting_word_in("golden hour") == "golden"          # joined phrase -> unambiguous preset
    assert _lighting_word_in("make it golden") is None           # bare ambiguous word -> stays material=gold
    sc.adjust("make it night")
    assert sc.environment["lighting"] == "night", sc.environment
    mats_before = [o["material"] for o in sc.objects]
    sc.adjust("make it golden")                                  # must be a MATERIAL change, lighting untouched
    assert sc.environment["lighting"] == "night", "the golden-collision must not steal the lighting preset"
    assert [o["material"] for o in sc.objects] != mats_before or all(o["material"] == "gold" for o in sc.objects)
    # lighting reaches the renderer: night is dimmer than studio (sun_i 0.5 vs 1.1), and re-render is deterministic.
    # Use a FRESH, UNTEXTURED scene: the standard renderer honours environment["lighting"]; the TEXTURED renderer
    # (render_textured) does NOT yet -- that is a KEPT NEGATIVE recorded in NOTES (textured-render lighting is a
    # separate follow-up touching the texture path), so we must not assert it on a textured scene.
    lit = scene_from_description("a red sphere and a metal box")
    lit.adjust("make it night");   dim = float(np.asarray(lit.render(width=40, height=32)).mean())
    lit.adjust("studio lighting"); brt = float(np.asarray(lit.render(width=40, height=32)).mean())
    assert brt > dim, ("studio should be brighter than night", brt, dim)
    assert "night" in lit.options()["lighting"], "options() must advertise lighting presets for discovery"

    # RELATIVE BRIGHTNESS (B2b): "brighter"/"dimmer" scale environment["sun_scale"], which the FAST renderer feeds
    # to the sun intensity. Assert the exact contract, the render effect, and that the default (sun_scale=1.0) is
    # byte-identical (multiply-by-1.0 identity -> a fresh scene renders the same before and after touching the knob).
    from holographic.simulation_and_physics.holographic_scene_semantic import _brightness_step
    assert abs(_brightness_step("make it brighter") - 1.35) < 1e-9
    assert abs(_brightness_step("make it dimmer") - 1.0 / 1.35) < 1e-9
    assert _brightness_step("make the sphere darker") is None          # object selector -> not a scene-level request
    br = scene_from_description("a red sphere and a metal box")
    base = float(np.asarray(br.render(width=40, height=32)).mean())    # sun_scale defaults to 1.0
    assert br.environment.get("sun_scale", 1.0) == 1.0
    same = float(np.asarray(br.render(width=40, height=32)).mean())
    assert base == same, "default render must be deterministic / unaffected by the sun_scale plumbing"
    br.adjust("make it brighter"); up = float(np.asarray(br.render(width=40, height=32)).mean())
    assert up > base, ("brighter must raise render brightness", up, base)
    br.adjust("make it much dimmer")                                    # 1.35 * (1/1.35**2) < 1 -> below base
    assert br.environment["sun_scale"] < 1.0, br.environment
    for _ in range(30):
        br.adjust("brighter")                                          # must clamp, never blow up
    assert br.environment["sun_scale"] <= 4.0 + 1e-9, "sun_scale must clamp to the ceiling"

    # B2c: the TEXTURED renderer now honours lighting too (was a kept-negative in B2a). A textured scene must render
    # DIMMER under night than studio -- proving lighting reaches the texture path, not just the flat one.
    tl = scene_from_description("a red sphere and a blue box")
    tl.adjust("give the sphere a rusty texture")                        # forces the textured render path
    tl.adjust("make it night");   tn = float(np.asarray(tl.render(width=40, height=32)).mean())
    tl.adjust("studio lighting"); ts = float(np.asarray(tl.render(width=40, height=32)).mean())
    assert ts > tn, ("textured render must honour lighting (studio brighter than night)", ts, tn)

    # RELATIVE PLACEMENT (B2, macro_relative_layout): "put X on top of Y" sets X's relation so the realizer moves it.
    # Deterministic and EXACT -- assert the sphere ends up physically ABOVE the box, and the verb-guard rejects a
    # normal command that merely contains "on" ("make the box on the left bigger" is NOT a placement).
    from holographic.simulation_and_physics.holographic_scene_semantic import _placement_command
    pl = scene_from_description("a red sphere and a blue box")
    assert _placement_command(pl, "make the box on the left bigger") is None    # verb-guard: not a placement
    assert _placement_command(pl, "put the sphere inside the box")[1] == "inside"
    pl.adjust("put the red sphere on top of the blue box")
    assert pl.objects[0].get("relation") == ("on", 1), pl.objects[0].get("relation")
    ys = {("sphere" in ro["name"]): getattr(ro["sdf"], "c", np.zeros(3))[1] for ro in pl.realize()}
    assert ys[True] > ys[False], ("placed sphere must sit above the box", ys)   # exact spatial result

    # TRANSLATE / SCALE by amount (B2): exact, deterministic. "move ... left 2" shifts x by -2; "twice as big" doubles
    # the radius; discrete "bigger" stays a size bucket (no collision); scale clamps.
    from holographic.simulation_and_physics.holographic_scene_semantic import _translate_command, _scale_command
    tsc = scene_from_description("a red sphere and a blue box")
    assert _translate_command(tsc, "move the red sphere left 2")[1] == (-2.0, 0.0, 0.0)
    assert _scale_command(tsc, "make the sphere twice as big")[1] == 2.0
    assert _scale_command(tsc, "make the box bigger") is None      # discrete size bucket, not continuous scale
    x0 = [getattr(ro["sdf"], "c", np.zeros(3))[0] for ro in tsc.realize() if "sphere" in ro["name"]][0]
    tsc.adjust("move the red sphere left 2")
    x1 = [getattr(ro["sdf"], "c", np.zeros(3))[0] for ro in tsc.realize() if "sphere" in ro["name"]][0]
    assert abs((x1 - x0) + 2.0) < 1e-9, ("translate must be exact", x0, x1)
    r0 = [getattr(ro["sdf"], "r", None) for ro in tsc.realize() if "sphere" in ro["name"]][0]
    tsc.adjust("make the red sphere twice as big")
    r1 = [getattr(ro["sdf"], "r", None) for ro in tsc.realize() if "sphere" in ro["name"]][0]
    assert abs(r1 / r0 - 2.0) < 1e-9, ("scale must be exact", r0, r1)

    # ROTATE / TILT (closes the axis-aligned negative): the verb sets an object's rotation, realize wraps it in a
    # _RotatedSDF, and the field actually changes (a point inside the upright cone falls outside once tilted).
    from holographic.simulation_and_physics.holographic_semantic import _RotatedSDF
    rsc = scene_from_description("a green cone")
    up_d = float(rsc.realize()[0]["sdf"].eval(np.array([[0.0, 0.7, 0.0]]))[0])
    rsc.adjust("tilt the cone 40 degrees")
    assert rsc.objects[0].get("rotation") == ((1.0, 0.0, 0.0), 40.0), rsc.objects[0].get("rotation")
    rsdf = rsc.realize()[0]["sdf"]
    assert isinstance(rsdf, _RotatedSDF)
    assert float(rsdf.eval(np.array([[0.0, 0.7, 0.0]]))[0]) != up_d, "tilt must change the field"
    rsc.adjust("tilt the cone 20 degrees")                            # same axis accumulates
    assert rsc.objects[0]["rotation"] == ((1.0, 0.0, 0.0), 60.0)

    # SCENE FROM IMAGE (stages B+C): a clean synthetic scene must round-trip to the right primitives. A red disc (left)
    # and a blue square (right) -> a red SPHERE left of a blue BOX. Proves the segment->map->assemble pipeline is sound
    # (photo quality is a segmentation limit, not a pipeline one).
    _H, _W = 80, 120
    _img = np.ones((_H, _W, 3))
    _yy, _xx = np.mgrid[0:_H, 0:_W]
    _img[(_yy - 40) ** 2 + (_xx - 30) ** 2 <= 15 ** 2] = (0.85, 0.15, 0.15)
    _img[30:60, 75:105] = (0.15, 0.25, 0.85)
    _res = scene_from_image(_img, k=3, max_objects=2)
    _reds = [o for o in _res["objects"] if o["color"] in ("red", "crimson", "maroon")]
    _blues = [o for o in _res["objects"] if o["color"] in ("blue", "navy")]
    assert _reds and _blues, [o["color"] for o in _res["objects"]]
    assert any(o["shape"] == "sphere" for o in _reds) and any(o["shape"] == "box" for o in _blues)
    assert _reds[0]["position"][0] < _blues[0]["position"][0]        # red disc is left of the blue square
    assert len(_res["scene"].objects) == 2

    # FLOOR + WALL backdrop: setting environment ground_color/backdrop_color changes the render (a matching backdrop);
    # absent, the render is byte-identical to the neutral default. Measured: a matching backdrop is the biggest single
    # fidelity lever against a photo (it is most of the frame).
    _bg = scene_from_description("a red sphere")
    _base = np.asarray(_bg.render(width=48, height=36), float)
    _bg.environment["ground_color"] = (0.20, 0.14, 0.09)             # dark wood floor
    _bg.environment["backdrop_color"] = (0.72, 0.72, 0.70)           # gray wall
    _withbg = np.asarray(_bg.render(width=48, height=36), float)
    assert not np.array_equal(_base, _withbg), "floor/wall colours must change the render"
    assert np.array_equal(_base, np.asarray(scene_from_description("a red sphere").render(width=48, height=36), float))

    # SEMANTIC -> NODE GRAPH bridge (drill down to exact settings): to_node_graph() emits an editable node graph whose
    # per-object primitive carries the EXACT radius/position, reachable by object NAME, and set_param tunes it exactly.
    ng = scene_from_description("a big red sphere and a blue box")
    ng.adjust("move the red sphere left 2")
    bundle = ng.to_node_graph()
    g2, objmap = bundle["graph"], bundle["objects"]
    sid = objmap[[n for n in objmap if "sphere" in n][0]]
    assert g2.describe(sid)["type"] == "sdf_sphere"
    tr = [e["dst"] for e in g2.edges if e["src"] == sid][0]        # its translate node holds the exact position
    assert g2.describe(tr)["params"]["t"][0] < -2.0, g2.describe(tr)["params"]["t"]
    g2.set_param(sid, radius=3.0)
    assert g2.describe(sid)["params"]["radius"] == 3.0            # exact node-level adjust
    assert hasattr(g2.evaluate(bundle["output"])["out"], "eval")  # the whole scene evaluates to one SDF
    assert bundle["materials"] == {}                             # materials default OFF -> geometry-only
    # materials=True carries each object's EXACT colour/metallic/roughness as a drill-into-able PBR node
    mb = scene_from_description("a red metal sphere and a blue box").to_node_graph(materials=True)
    gm = mb["graph"]; mname = [n for n in mb["materials"] if "sphere" in n][0]; mid = mb["materials"][mname]
    md = gm.describe(mid)
    assert md["type"] == "material_pbr" and md["params"]["metallic"] == 0.9  # metal -> metallic 0.9
    gm.set_param(mid, roughness=0.05)
    assert gm.describe(mid)["params"]["roughness"] == 0.05        # exact material knob adjust
    # renderable=True wires each object's MESHED geometry to its material (assign_material) -> a drawable node
    rb = scene_from_description("a red sphere").to_node_graph(renderable=True)
    gr = rb["graph"]; rid = list(rb["renderables"].values())[0]
    assert gr.describe(rid)["type"] == "assign_material"
    assert set(gr.describe(rid)["wired_in"]) == {"mesh", "material"}   # geometry + material both wired in
    meshnode = gr.describe(rid)["wired_in"]["mesh"][0]
    gr.set_param(meshnode, res=12)                                # drill down to the mesh resolution knob; keep it cheap
    out = gr.evaluate(rid)["out"]
    assert "mesh" in out and "material" in out and len(getattr(out["mesh"], "vertices", [])) > 0

    # THE CRITIC (B7): propose_edits ranks candidate edits by how much each moves the render toward a TARGET, and
    # refine_to_target greedily applies the best. Assert it RECOVERS a known target: render a 'night' goal, start from
    # default, and the top proposal must be the lighting edit that reaches it (distance -> ~0). Tiny + few candidates.
    goal = scene_from_description("a red sphere and a blue box"); goal.adjust("make it night")
    target = np.asarray(goal.render(width=40, height=32), float)
    crit = scene_from_description("a red sphere and a blue box")
    rep = crit.propose_edits(target, candidates=["make it night", "make it brighter", "make the red sphere green"],
                             width=40, height=32)
    assert rep["proposals"][0]["command"] == "make it night", rep["proposals"]     # ranks the right edit first
    assert rep["proposals"][0]["improvement"] > 0.0                                 # and it measurably helps
    # determinism: identical ranking on a repeat call
    rep2 = crit.propose_edits(target, candidates=["make it night", "make it brighter"], width=40, height=32)
    rep3 = crit.propose_edits(target, candidates=["make it night", "make it brighter"], width=40, height=32)
    assert [p["command"] for p in rep2["proposals"]] == [p["command"] for p in rep3["proposals"]]
    # the greedy driver reaches the target and stops
    driver = scene_from_description("a red sphere and a blue box")
    res = driver.refine_to_target(target, max_steps=2, candidates=["make it night", "make it brighter"],
                                  width=40, height=32)
    assert res["final_distance"] <= res["start_distance"] and "make it night" in res["applied"]
    # geometry=True lets the critic fix POSITION: target = a moved sphere; the top proposal must be the matching move.
    gtar = scene_from_description("a red sphere"); gtar.adjust("move the red sphere left 1.5")
    gimg = np.asarray(gtar.render(width=40, height=32), float)
    grep = scene_from_description("a red sphere").propose_edits(gimg, width=40, height=32, geometry=True, top=3)
    assert grep["proposals"][0]["command"] == "move the red sphere left 1.5", grep["proposals"]
    # and geometry=False must NOT propose a move (default appearance-only behaviour preserved)
    arep = scene_from_description("a red sphere").propose_edits(gimg, width=40, height=32, geometry=False, top=5)
    assert not any("move" in p["command"] for p in arep["proposals"])

    # BIDIRECTIONAL LOOKUP: render_passes gives a per-object coverage matte keyed by NAME, and the mattes tile the
    # foreground (each object's pixels are exactly attributed).
    ps = scene_from_description("a red sphere and a blue box").render_passes(want=("mask",), width=48, height=36)
    objkeys = [key for key in ps if key.startswith("object:")]
    assert len(objkeys) == 2 and all(np.asarray(ps[key]).sum() > 0 for key in objkeys)
    assert abs(float(sum(np.asarray(ps[key]) for key in objkeys).max()) - 1.0) < 1e-6  # no double-coverage

    # FOCUS un-blinds the critic: on a small red sphere over a BIG matching backdrop, recolouring the sphere barely
    # moves a whole-frame score but clearly moves a subject-focused one. Assert focus registers the object edit MORE.
    ftar = scene_from_description("a red sphere")
    ftar.environment["ground_color"] = (0.2, 0.14, 0.09); ftar.environment["backdrop_color"] = (0.7, 0.7, 0.68)
    ftar.adjust("make the red sphere green")
    ftimg = np.asarray(ftar.render(width=64, height=48), float)
    fstart = scene_from_description("a red sphere")
    fstart.environment["ground_color"] = (0.2, 0.14, 0.09); fstart.environment["backdrop_color"] = (0.7, 0.7, 0.68)
    cand = ["make the red sphere green"]
    imp_whole = fstart.propose_edits(ftimg, candidates=cand, width=64, height=48, focus=None)["proposals"][0]["improvement"]
    imp_focus = fstart.propose_edits(ftimg, candidates=cand, width=64, height=48, focus="auto")["proposals"][0]["improvement"]
    assert imp_focus > imp_whole, ("focus must register the object edit more strongly", imp_whole, imp_focus)

    print("OK: holographic_scene_semantic self-test passed (describe->build %d objects; adjust bigger/to-metal/"
          "everything-glass; set by description; NAME/rename + reference by nickname; paint a rusty/marbled TEXTURE "
          "and route render through it; unknown ref = no-op; simulate settles %.2f->%.2f)"
          % (len(sc.objects), float(first_y), float(last_y)))


if __name__ == "__main__":
    _selftest()
