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
import numpy as np

from holographic_semantic import (SHAPES, COLORS, MATERIALS, SIZES, RELATIONS, ARTICLES, _STOP, parse_description,
                                   realize_scene, find_objects, batch_set, _obj_name)

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
      "make the pyramid golden"       -> ({}, {"material":"gold"}, False)   # no shape -> no selector -> no-op upstream
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
    from holographic_semantic import SynonymResolver
    if resolver is None:
        resolver = SynonymResolver()                            # table-only: deterministic, no corpus needed

    toks = command.lower().replace(",", " ").replace(".", " ").split()
    known = set(SHAPES) | set(COLORS) | set(MATERIALS) | set(SIZES) | set(_RELATIVE_SIZE) | _COMMAND_WORDS

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
    if explicit_all:
        matched_idx = list(range(len(scene.objects)))
    elif reference:
        matched_idx = find_objects(scene.objects, **reference)
    else:
        matched_idx = []
    matched = [scene.objects[i]["name"] for i in matched_idx]
    applied = bool(changes) and (explicit_all or bool(matched_idx))

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
            "read_as": read_as, "matched": matched, "applied": applied,
            "questions": questions, "suggestions": suggestions, "unknown": unknown}


class SemanticScene:
    """A live, adjustable scene of NAMED objects plus an environment. Build it from a description or hand it a list of
    object dicts; then reference objects by how you'd describe them and change them in words, and render or simulate."""

    def __init__(self, objects, environment=None, mind=None):
        # each object is a dict {shape,color,material,size,relation}; ensure every one carries a stable name
        self.objects = [dict(o) for o in objects]
        for o in self.objects:
            o.setdefault("relation", None)
            o["name"] = _obj_name(o)
        self.environment = dict(environment or {"sun": None, "sky": None})
        self.mind = mind
        self.feedback = None                                    # the clarifying report from the last adjust()

    # ---- inspect ------------------------------------------------------------------------------------------
    def names(self):
        """The human name of each object (e.g. 'big red metal sphere') -- how you refer to them."""
        return [o["name"] for o in self.objects]

    def describe(self):
        """A one-line human summary of what the scene currently holds."""
        if not self.objects:
            return "an empty scene"
        env = self.environment
        weather = ", ".join(v for v in (env.get("sun") and env["sun"] + " sun",
                                        env.get("sky") and env["sky"] + " sky") if v)
        body = "; ".join(o["name"] for o in self.objects)
        return body + (" (%s)" % weather if weather else "")

    # ---- reference ----------------------------------------------------------------------------------------
    def select(self, reference):
        """Resolve a reference to object INDICES. `reference` may be a dict of attribute constraints, a plain phrase
        ('the red sphere', 'everything'), or None/'' for all objects."""
        if reference in (None, "", "all", "everything"):
            return list(range(len(self.objects)))
        if isinstance(reference, str):
            ref_attrs, _, all_flag = parse_adjust(reference)     # read a phrase as a selector (ignore any change part)
            if all_flag:
                return list(range(len(self.objects)))
            reference = ref_attrs
        if not reference:
            return list(range(len(self.objects)))
        return find_objects(self.objects, **reference)

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

    def adjust(self, command):
        """Adjust the scene from a plain-English command: 'make the sphere bigger', 'change the red box to metal',
        'make everything glass'. Does its best -- resolving known synonyms (crimson->red, chrome->metal) -- applies the
        change, and returns self (chainable).

        Honest AND helpful: a change hits ALL objects only when the command says so ('everything'/'all'/'it'); a
        command it can't fully place ('make the pyramid golden') changes nothing rather than guessing -- and instead of
        failing silently it leaves a clarifying report in `self.feedback` (understood/matched/questions/suggestions), so
        the caller can ask the user 'did you mean ...?'. Use interpret(command) to preview that report without applying."""
        report = interpret_command(self, command)
        self.feedback = report                                   # always available: what it read + any suggestions
        u = report["understood"]
        changes, reference, explicit_all = u["changes"], u["reference"], u["target_all"]
        if changes and report["applied"]:
            where = list(range(len(self.objects))) if explicit_all else find_objects(self.objects, **reference)
            for i in where:
                for k, v in changes.items():
                    self.objects[i][k] = v
                self.objects[i]["name"] = _obj_name(self.objects[i])
        return self

    def options(self):
        """The PALETTE of what you can say to this scene -- the object names you can reference and the attribute words
        you can set. Handy for a UI to show, or for an agent to ground its commands in. Returns a dict."""
        return {"objects": self.names(),
                "shapes": sorted(set(SHAPES.values())),
                "colors": sorted(COLORS),
                "materials": sorted(set(MATERIALS.values())),
                "sizes": sorted(SIZES) + sorted(_RELATIVE_SIZE)}

    # ---- realize / render / simulate ----------------------------------------------------------------------
    def realize(self, spacing=2.7, base_radius=0.7):
        """Turn the current objects into renderable geometry (a list of realized SDF objects). See
        holographic_semantic.realize_scene."""
        return realize_scene(self.objects, spacing=spacing, base_radius=base_radius)

    def render(self, camera=None, width=200, height=150, quality="fast", **kw):
        """Best-effort 3-D RENDER of whatever the scene currently holds. Builds a default pulled-back camera if none is
        given, applies the environment (sun/sky), and calls the engine's single-pass renderer (quality='fast') or the
        Monte-Carlo path tracer (quality='hyperreal'). Returns an (H,W,3) image array. See holographic_semantic."""
        from holographic_semantic import render_scene, render_scene_pbr
        from holographic_render import Camera
        objs = self.objects                                     # render_scene realizes these internally
        if camera is None:                                      # a sensible default view of the whole scene
            span = max(3.0, 1.6 * len(objs))
            camera = Camera(eye=(span * 0.4, span * 0.28, span), target=(0, 0, 0), fov_deg=42.0)
        sun = self.environment.get("sun") or "bright"
        sky = self.environment.get("sky") or "clear"
        if quality == "hyperreal":
            return render_scene_pbr(objs, camera, width=width, height=height, sun=sun, sky=sky, **kw)
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
        from holographic_semantic import encode_scene
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
    from holographic_semantic import SynonymResolver
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


def scene_from_description(text, mind=None):
    """Describe a scene in (controlled) natural language and get a live, adjustable SemanticScene back: the system
    parses the description, builds named objects + the environment, and hands you something you can adjust(),
    render(), simulate(), or encode(). The one-call 'describe it and let the system build it' entry point.

    If the description parses to NO objects (or leaves words unrecognised), the returned scene carries a clarifying
    report in `scene.feedback` -- suggestions for how to phrase it -- rather than being a silent empty scene."""
    from holographic_semantic import SynonymResolver
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
    sc.adjust("make the pyramid golden")                         # no pyramid, no known 'pyramid' shape
    assert all(o["material"] == "glass" for o in sc.objects), "unknown ref changed nothing"
    fb = sc.feedback
    assert not fb["applied"] and fb["suggestions"], fb          # it left suggestions
    assert any("pyramid" in s for s in fb["suggestions"])       # named the word it didn't know

    # does its best: resolve a known synonym (crimson -> red) and say so
    sc.set("the box", material="matte")
    rep = sc.interpret("make the box crimson")
    assert rep["read_as"].get("crimson") == "color=red", rep
    sc.adjust("make the box crimson")
    assert sc.get({"shape": "box"})[0]["color"] == "red"

    # preview: understood the target but not the change -> a helpful suggestion, nothing applied
    rep = sc.interpret("do something to the sphere")
    assert not rep["applied"] and rep["suggestions"]

    # options(): the palette a UI/agent can show
    opt = sc.options()
    assert "sphere" in opt["shapes"] and "metal" in opt["materials"] and opt["objects"]

    # simulate: a real gravity drop that settles on the ground
    frames = sc.simulate(steps=30)
    assert len(frames) == 30
    first_y = list(frames[0].values())[0][1]
    last_y = list(frames[-1].values())[0][1]
    assert last_y < first_y and last_y > 0.0, (first_y, last_y)   # fell, and rests above the ground

    print("OK: holographic_scene_semantic self-test passed (describe->build %d objects; adjust bigger/to-metal/"
          "everything-glass; set by description; unknown ref = no-op; simulate settles %.2f->%.2f)"
          % (len(sc.objects), float(first_y), float(last_y)))


if __name__ == "__main__":
    _selftest()
