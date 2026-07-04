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
    from holographic_texturegraph import Map, Const, field_leaf
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
            from holographic_materialio import TextureMap
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
            from holographic_assets import AssetLibrary
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
        if textured:
            from holographic_texturerender import render_textured
            return render_textured(self, textured, camera=camera, width=width, height=height, **kw)
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

    print("OK: holographic_scene_semantic self-test passed (describe->build %d objects; adjust bigger/to-metal/"
          "everything-glass; set by description; NAME/rename + reference by nickname; paint a rusty/marbled TEXTURE "
          "and route render through it; unknown ref = no-op; simulate settles %.2f->%.2f)"
          % (len(sc.objects), float(first_y), float(last_y)))


if __name__ == "__main__":
    _selftest()
