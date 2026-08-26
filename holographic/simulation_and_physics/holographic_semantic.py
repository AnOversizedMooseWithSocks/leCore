"""holographic_semantic.py -- a controlled SEMANTIC layer over the 3-D stack.

Ground words as VSA roles/fillers, parse a scene DESCRIPTION into VSA object RECORDS, bundle them into one composable
scene hypervector, query any attribute back, batch-edit, realize to a 3-D SDF scene, and emit UI control specs. The
through-line: a description becomes a VSA structure whose semantic values are ASSIGNED (bind), RECOVERABLE (unbind +
cleanup, bidirectional), QUERYABLE, and COMPOSABLE -- so the same scene vector can be rendered, edited, or handed to
the agent brain.

SCOPE (the kept boundary, in the spirit of holographic_lang): this is a CONTROLLED vocabulary + keyword grammar over
ONE domain -- 3-D scene description -- NOT general natural language. It knows a fixed set of shapes, colours,
materials, sizes and spatial relations and parses a sentence over THAT vocabulary deterministically (articles a/an =
introduce, the = refer-back; pre-modifiers "red ball" and post-modifiers "box with a glass material"; relations link
objects). There is no learned language model -- the engine has no torch / learned weights, so "red" maps to an rgb
because the table says so, and synonyms (ball<->sphere, metallic<->metal) are folded by the table, not inferred. What
VSA genuinely buys is the part that matters and is hard to fake: once parsed, each object is a bind/bundle RECORD that
is bidirectionally queryable (decode an attribute back at cosine ~1 even from the bundled scene), batch-editable, and
composable into one hypervector. The semantic value is assigned and recoverable; the language surface is honestly
narrow.
"""
import numpy as np


# --------------------------------------------------------------------------------------------------------------
# The controlled vocabulary -- the grounding tables (a word -> its meaning). Hand-built, deterministic, honest.
# --------------------------------------------------------------------------------------------------------------
SHAPES = {                                                   # word -> canonical shape
    "ball": "sphere", "sphere": "sphere", "orb": "sphere", "globe": "sphere", "marble": "sphere",
    "box": "box", "cube": "box", "block": "box", "crate": "box", "brick": "box",
    "cylinder": "cylinder", "tube": "cylinder", "pipe": "cylinder", "pillar": "cylinder",
    "column": "cylinder", "can": "cylinder", "barrel": "cylinder", "rod": "cylinder",
    "cone": "cone", "pyramid": "cone", "spike": "cone", "funnel": "cone", "teepee": "cone",
    "torus": "torus", "donut": "torus", "doughnut": "torus", "ring": "torus", "tyre": "torus", "tire": "torus",
}
COLORS = {                                                   # word -> linear rgb
    "red": (0.85, 0.18, 0.18), "green": (0.20, 0.70, 0.25), "blue": (0.20, 0.40, 0.85),
    "yellow": (0.90, 0.80, 0.20), "orange": (0.95, 0.55, 0.15), "purple": (0.60, 0.25, 0.70),
    "pink": (0.95, 0.55, 0.70), "cyan": (0.25, 0.80, 0.85), "white": (0.90, 0.90, 0.90),
    "black": (0.08, 0.08, 0.08), "gray": (0.50, 0.50, 0.50), "grey": (0.50, 0.50, 0.50),
    "teal": (0.10, 0.55, 0.55), "turquoise": (0.20, 0.75, 0.70), "navy": (0.10, 0.15, 0.45),
    "maroon": (0.45, 0.12, 0.15), "crimson": (0.80, 0.10, 0.22), "brown": (0.42, 0.26, 0.14),
    "tan": (0.75, 0.62, 0.44), "beige": (0.83, 0.76, 0.62), "cream": (0.94, 0.90, 0.78),
    "olive": (0.40, 0.42, 0.15), "lime": (0.62, 0.86, 0.20), "mint": (0.62, 0.90, 0.72),
    "lavender": (0.72, 0.66, 0.90), "violet": (0.55, 0.30, 0.82), "magenta": (0.85, 0.15, 0.70),
    "indigo": (0.28, 0.20, 0.62), "gold": (0.86, 0.68, 0.22), "silver": (0.75, 0.76, 0.80),
}
MATERIALS = {                                                # word -> canonical material
    "glass": "glass", "glassy": "glass", "transparent": "glass", "clear": "glass",
    "metal": "metal", "metallic": "metal", "chrome": "metal", "steel": "metal", "iron": "metal",
    "silver": "metal", "aluminium": "metal", "aluminum": "metal", "tin": "metal", "titanium": "metal",
    "wooden": "matte", "wood": "matte", "timber": "matte", "oak": "matte", "concrete": "matte",
    "stone": "matte", "cement": "matte", "cardboard": "matte", "rusty": "copper", "rust": "copper",
    "gold": "gold", "golden": "gold", "copper": "copper", "bronze": "copper", "brass": "gold",
    "matte": "matte", "rough": "matte", "clay": "matte", "chalk": "matte",
    "plastic": "plastic", "rubber": "plastic", "vinyl": "plastic",
    "ceramic": "ceramic", "porcelain": "ceramic",
    "mirror": "mirror", "reflective": "mirror", "shiny": "mirror", "polished": "mirror",
    "brushed": "brushed", "satin": "brushed", "glossy": "glossy", "burnished": "brushed",
    "emissive": "emissive", "glowing": "emissive", "glow": "emissive", "neon": "emissive",
    "luminous": "emissive", "lamp": "emissive", "lit": "emissive",
    "fog": "fog", "foggy": "fog", "mist": "fog", "misty": "fog", "vapor": "fog", "haze": "fog",
    "smoke": "smoke", "smoky": "smoke", "smog": "smoke",
    "fire": "fire", "flame": "fire", "burning": "fire", "ember": "fire",
    "cloud": "cloud", "cloudy": "cloud", "fluffy": "cloud", "cumulus": "cloud", "puffy": "cloud",
    "wax": "wax", "waxy": "wax", "jade": "wax", "marble": "wax", "skin": "wax", "candle": "wax",   # subsurface (SSS)
    "translucent": "translucent", "frosted": "translucent", "milky": "translucent",  # see-through diffuse
}
# materials that are VOLUMETRIC (participating media), rendered with volume_render, not as surfaces
_VOLUMETRIC = {"fog", "smoke", "fire", "cloud"}
SIZES = {                                                    # word -> ('uniform', s) or ('stretch', (sx,sy,sz))
    "big": ("uniform", 1.5), "large": ("uniform", 1.5), "huge": ("uniform", 2.0),
    "enormous": ("uniform", 2.3), "giant": ("uniform", 2.3), "gigantic": ("uniform", 2.5),
    "massive": ("uniform", 2.2), "medium": ("uniform", 1.0), "normal": ("uniform", 1.0),
    "small": ("uniform", 0.62), "tiny": ("uniform", 0.42), "little": ("uniform", 0.62),
    "miniature": ("uniform", 0.4), "elongated": ("stretch", (1.0, 1.0, 2.2)),
    "long": ("stretch", (1.0, 1.0, 2.2)), "short": ("stretch", (1.0, 0.55, 1.0)),
    "flat": ("stretch", (1.6, 0.4, 1.6)), "tall": ("stretch", (1.0, 2.0, 1.0)),
    "wide": ("stretch", (2.0, 1.0, 1.0)), "thin": ("stretch", (0.5, 1.4, 0.5)),
    "fat": ("stretch", (1.4, 1.0, 1.4)), "squat": ("stretch", (1.4, 0.6, 1.4)),
}
RELATIONS = {                                                # word -> canonical relation
    "inside": "inside", "within": "inside", "in": "inside",
    "on": "on", "atop": "on", "above": "above", "over": "above", "under": "under", "below": "under",
    "beneath": "under", "underneath": "under",
    "leaning": "leaning", "beside": "beside", "next": "beside", "near": "beside", "by": "beside",
    "behind": "behind", "front": "front",
    "diagonal": "diagonal", "diagonally": "diagonal",
}
ENVIRO_SKY = {"clear": "clear", "cloudy": "cloudy", "overcast": "cloudy", "partly": "partly"}

# Named LIGHTING presets (time-of-day / mood): each sets the sun DIRECTION, COLOUR and intensity + an ambient sky
# tint, so "at sunset" or "on an overcast morning" gives a visibly different, evocative light -- not just on/off.
# dir is a direction TOWARD the sun (will be normalised); warmer/lower sun = longer, more golden light.
LIGHTING = {                                                 # word -> dict(dir, sun_col, sun_i, amb, amb_col)
    "noon":     dict(dir=(-0.2, 0.95, -0.15), sun_col=(1.0, 0.99, 0.96), sun_i=1.15, amb=0.26, amb_col=(0.55, 0.62, 0.78)),
    "midday":   dict(dir=(-0.2, 0.95, -0.15), sun_col=(1.0, 0.99, 0.96), sun_i=1.15, amb=0.26, amb_col=(0.55, 0.62, 0.78)),
    "morning":  dict(dir=(-0.7, 0.42, -0.3),  sun_col=(1.0, 0.93, 0.82), sun_i=1.0,  amb=0.30, amb_col=(0.6, 0.66, 0.8)),
    "afternoon":dict(dir=(0.6, 0.55, -0.35),  sun_col=(1.0, 0.95, 0.85), sun_i=1.0,  amb=0.28, amb_col=(0.58, 0.63, 0.76)),
    "sunset":   dict(dir=(-0.85, 0.18, -0.2), sun_col=(1.0, 0.62, 0.35), sun_i=1.1,  amb=0.30, amb_col=(0.5, 0.45, 0.6)),
    "sunrise":  dict(dir=(0.85, 0.18, -0.2),  sun_col=(1.0, 0.68, 0.45), sun_i=1.05, amb=0.30, amb_col=(0.55, 0.5, 0.62)),
    "dusk":     dict(dir=(-0.7, 0.12, -0.2),  sun_col=(0.85, 0.55, 0.55), sun_i=0.8, amb=0.34, amb_col=(0.42, 0.44, 0.62)),
    "golden":   dict(dir=(-0.8, 0.25, -0.25), sun_col=(1.0, 0.78, 0.45), sun_i=1.15, amb=0.30, amb_col=(0.55, 0.5, 0.58)),
    "overcast": dict(dir=(-0.3, 0.85, -0.2),  sun_col=(0.9, 0.92, 0.95), sun_i=0.55, amb=0.5,  amb_col=(0.72, 0.75, 0.8)),
    "night":    dict(dir=(-0.4, 0.7, -0.3),   sun_col=(0.55, 0.62, 0.85), sun_i=0.5, amb=0.28, amb_col=(0.3, 0.36, 0.55)),
    "moonlit":  dict(dir=(-0.4, 0.7, -0.3),   sun_col=(0.6, 0.68, 0.9),  sun_i=0.55, amb=0.26, amb_col=(0.3, 0.36, 0.55)),
    "studio":   dict(dir=(-0.45, 0.8, -0.35), sun_col=(1.0, 1.0, 1.0),   sun_i=1.1,  amb=0.4,  amb_col=(0.7, 0.7, 0.72)),
    "dramatic": dict(dir=(-0.9, 0.3, 0.1),    sun_col=(1.0, 0.95, 0.88), sun_i=1.3,  amb=0.12, amb_col=(0.3, 0.34, 0.45)),
}
# words that map to a preset
LIGHTING_WORDS = {
    "noon": "noon", "midday": "midday", "morning": "morning", "afternoon": "afternoon",
    "sunset": "sunset", "dusk": "dusk", "evening": "sunset", "sunrise": "sunrise", "dawn": "sunrise",
    "golden": "golden", "goldenhour": "golden", "overcast": "overcast", "cloudy": "overcast",
    "night": "night", "nighttime": "night", "moonlit": "moonlit", "moonlight": "moonlit",
    "studio": "studio", "dramatic": "dramatic", "moody": "dramatic",
}

def lighting_params(sun="bright", lighting=None, sun_scale=1.0):
    """Resolve a lighting request into concrete shading numbers: dict(dir, sun_col, sun_i, amb, amb_col). This is the
    ONE place that turns a LIGHTING preset name (or the plain sun="bright"/"soft" word) plus a relative sun_scale into
    a sun DIRECTION, COLOUR, INTENSITY and AMBIENT tint -- so the fast renderer (_scene_setup) and the textured
    renderer agree instead of each hard-coding their own light. sun_scale multiplies the final intensity (B2b).
    Defaults (bright/None/1.0) give sun_i=1.0, white light, dir=(-0.45,0.8,-0.35), amb=0.24 -- the historical look.
    NumPy-free tuples out (readable, deterministic). See LIGHTING for the preset table."""
    import numpy as _np
    d = _np.array([-0.45, 0.8, -0.35]); d = d / _np.linalg.norm(d)
    dir_, sun_col, sun_i, amb, amb_col = tuple(d), (1.0, 1.0, 1.0), (1.0 if sun == "bright" else 0.7), 0.24, (1.0, 1.0, 1.0)
    preset = LIGHTING.get(lighting) if lighting else None
    if preset is not None:
        pd = _np.asarray(preset["dir"], float); pd = pd / (_np.linalg.norm(pd) + 1e-9)
        dir_, sun_col, sun_i, amb, amb_col = tuple(pd), tuple(preset["sun_col"]), preset["sun_i"], preset["amb"], tuple(preset["amb_col"])
    return {"dir": dir_, "sun_col": sun_col, "sun_i": float(sun_i) * float(sun_scale), "amb": amb, "amb_col": amb_col}

ARTICLES = {"a", "an", "the"}
_STOP = {"of", "with", "is", "to", "and", "that", "which", "sitting", "leaning", "resting"}   # glue words

# canonical material -> render_sdf kwargs (how a material LOOKS in the FAST renderer)
MATERIAL_RENDER = {
    "glass": dict(refract=0.85, ior=1.5, reflect=0.12),
    "metal": dict(pbr=(1.0, 0.28), reflect=0.45),
    "gold": dict(pbr=(1.0, 0.30), reflect=0.45),
    "copper": dict(pbr=(1.0, 0.35), reflect=0.40),
    "matte": dict(reflect=0.0),
    "plastic": dict(reflect=0.12),
    "ceramic": dict(reflect=0.18),
    "mirror": dict(reflect=0.85),
    "brushed": dict(reflect=0.55),                              # reflective but ROUGH -> a glossy (blurred) reflection
    "glossy": dict(reflect=0.35),
    "emissive": dict(reflect=0.0),
    "wax": dict(reflect=0.06, sss=1.0),                         # subsurface scattering (thin parts glow)
    "translucent": dict(reflect=0.08, translucent=1.0),        # diffuse see-through (frosted)
    None: dict(reflect=0.2),
}

# canonical material -> physically-based (metallic, roughness, emission_strength) for the PATH-TRACED renderer.
# These feed holographic_brdf.cook_torrance via holographic_pathtrace -- real GGX, not the fast renderer's crude term.
PBR_PARAMS = {
    "matte": (0.0, 0.88, 0.0), "plastic": (0.0, 0.34, 0.0), "ceramic": (0.0, 0.52, 0.0),
    "metal": (1.0, 0.25, 0.0), "gold": (1.0, 0.30, 0.0), "copper": (1.0, 0.36, 0.0),
    "mirror": (1.0, 0.045, 0.0), "glass": (0.0, 0.06, 0.0),     # glass = polished dielectric in the path tracer
    "emissive": (0.0, 0.6, 3.2), None: (0.0, 0.72, 0.0),
}
# albedo (reflection tint) for metals when no explicit colour is given
METAL_TINT = {"gold": (1.00, 0.78, 0.34), "copper": (0.95, 0.64, 0.54),
              "metal": (0.95, 0.96, 0.97), "mirror": (0.97, 0.97, 0.97)}


# --------------------------------------------------------------------------------------------------------------
# Parser: a scene DESCRIPTION -> objects + relations + environment (controlled grammar)
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# Synonym grounding by DISTRIBUTIONAL similarity (the dormant text module, put to work)
# --------------------------------------------------------------------------------------------------------------
# A small grounding corpus built so every synonym sits ADJACENT to its canonical word, so the random-indexing
# encoder (learn_word_vectors) drifts the synonym's vector toward the canonical it co-occurs with. This is how
# "crimson" learns it is near "red" with no table entry -- distributional meaning, no labels, no training loop.
SYNONYM_SEEDS = {
    "red": ["crimson", "scarlet", "ruby", "cherry", "maroon"],
    "blue": ["azure", "navy", "cobalt", "sapphire", "teal"],
    "green": ["emerald", "lime", "jade", "olive", "mint"],
    "yellow": ["golden", "amber", "lemon", "gold"],
    "orange": ["tangerine", "apricot", "rust"],
    "purple": ["violet", "magenta", "lavender", "indigo"],
    "black": ["charcoal", "ebony", "onyx", "jet"],
    "white": ["ivory", "snow", "pearl", "cream"],
    "gray": ["silver", "ash", "slate", "grey"],
    "metal": ["steel", "iron", "chrome", "titanium", "metallic", "brass"],
    "glass": ["transparent", "crystal", "clear", "glassy"],
    "matte": ["rubber", "plastic", "clay", "dull"],
    "mirror": ["polished", "gleaming", "reflective", "shiny", "chromed"],
    "sphere": ["spherical", "round", "orb", "globe", "ball"],
    "box": ["rectangular", "boxy", "cubic", "cube", "block", "crate"],
    "big": ["huge", "giant", "enormous", "massive", "large"],
    "small": ["tiny", "miniature", "petite", "little"],
    "elongated": ["stretched", "lengthy", "long"],
}


def _build_grounding_corpus(seeds):
    """One short line per (canonical, synonym) pair, canonical repeated so it dominates each synonym's context.
    Adjacency guarantees the encoder ties the two together within a window of 2."""
    lines = []
    for canon, syns in seeds.items():
        for s in syns:
            lines.append("%s %s %s" % (canon, s, canon))       # synonym flanked by its canonical word
    return lines


DEFAULT_GROUNDING_CORPUS = _build_grounding_corpus(SYNONYM_SEEDS)

# the known target words per field that an unknown word can resolve TO
_KNOWN = {
    "color": sorted(set(COLORS)),
    "material": sorted(set(MATERIALS.values())),
    "shape": sorted(set(SHAPES.values())),
    "size": sorted(set(SIZES)),
}


def _field_of(canon):
    """Which attribute field a canonical word belongs to."""
    if canon in COLORS:
        return "color"
    if canon in set(MATERIALS.values()):
        return "material"
    if canon in set(SHAPES.values()):
        return "shape"
    if canon in SIZES:
        return "size"
    return None


# the reliable reverse map: synonym -> (field, canonical). This is the DETERMINISTIC grounding (a synonym table);
# it is what actually resolves 'crimson'->red, 'chrome'->metal, 'spherical'->sphere in practice.
SYNONYM_TABLE = {}
for _canon, _syns in SYNONYM_SEEDS.items():
    _f = _field_of(_canon)
    for _s in _syns:
        if _s not in SHAPES and _s not in COLORS and _s not in MATERIALS and _s not in SIZES:
            SYNONYM_TABLE[_s] = (_f, _canon)


class SynonymResolver:
    """Resolve an out-of-vocabulary word to a known vocabulary word. Two grounding paths, honestly separated:

      * TABLE (default, reliable): the curated synonym table (SYNONYM_TABLE) -- deterministic, exact. This is what
        resolves 'crimson'->red, 'chrome'->metal, 'spherical'->sphere in practice.
      * LEARNED (opt-in, for REAL corpora): pass `corpus=` (or learned=True for the small default) to also fall back
        to nearest-known-by-distributional-similarity using the random-indexing word vectors
        (holographic_text.learn_word_vectors). MEASURED KEPT NEGATIVE: on a TINY synthetic corpus this is weak/noisy
        (~2/14 ranking accuracy, cosines ~0.05) -- random indexing needs a substantial real corpus where words
        co-occur many times in varied contexts (as holographic_text's demo_text shows on Gutenberg/Brown) before it
        beats the table. So the table leads; the learned path extends coverage when you have real text."""

    def __init__(self, corpus=None, learned=False, dim=512, window=2, seed=0, threshold=0.1):
        self.table = dict(SYNONYM_TABLE)
        self.enc = None
        self.threshold = threshold
        if corpus is not None or learned:
            from holographic.misc.holographic_text import learn_word_vectors
            self.enc = learn_word_vectors(corpus or DEFAULT_GROUNDING_CORPUS, dim=dim, window=window, seed=seed)

    def resolve(self, word, field):
        """Map `word` to the nearest known `field` value (table first, then learned), or None."""
        if word in self.table and self.table[word][0] == field:
            return self.table[word][1]
        if self.enc is None:
            return None
        from holographic.agents_and_reasoning.holographic_ai import cosine
        try:
            wv = self.enc.wordvec(word)
        except Exception:
            return None
        if wv is None or not np.any(wv):
            return None
        best, best_c = None, -1.0
        for cand in _KNOWN[field]:
            try:
                cv = self.enc.wordvec(cand)
            except Exception:
                continue
            if cv is None or not np.any(cv):
                continue
            c = float(cosine(wv, cv))
            if c > best_c:
                best, best_c = cand, c
        return best if (best is not None and best_c >= self.threshold) else None

    def classify_unknown(self, word):
        """Return (field, canonical_value) for the best resolution, or None. Table first, then learned."""
        if word in self.table:
            return self.table[word]
        if self.enc is None:
            return None
        from holographic.agents_and_reasoning.holographic_ai import cosine
        try:
            wv = self.enc.wordvec(word)
        except Exception:
            return None
        if wv is None or not np.any(wv):
            return None
        best = None
        for field in ("color", "material", "shape", "size"):
            r = self.resolve(word, field)
            if r is None:
                continue
            c = float(cosine(wv, self.enc.wordvec(r)))
            if best is None or c > best[2]:
                best = (field, r, c)
        return (best[0], best[1]) if best else None


def _classify(tok):
    """Which vocabulary table a token belongs to (or None). The lexer of the controlled grammar."""
    if tok in SHAPES:
        return "shape"
    if tok in COLORS:
        return "color"
    if tok in MATERIALS:
        return "material"
    if tok in SIZES:
        return "size"
    if tok in RELATIONS:
        return "relation"
    if tok in ARTICLES:
        return "article"
    return None


def parse_description(text, resolver=None):
    """Parse a controlled scene description into {'objects': [...], 'environment': {...}}. Each object is a dict
    {shape, color, material, size, relation}. `relation` is (relword, ref_index) or None. With a SynonymResolver,
    out-of-vocabulary words are mapped to the nearest known word by distributional similarity ('crimson' -> 'red').
    Honest, deterministic, vocabulary-bounded -- see the module SCOPE note."""
    text = text.lower().replace(",", " , ").replace(".", " . ")
    # split off environment clauses (those mentioning sun / sky / cloud) so they don't pollute object parsing
    env = {"sun": None, "sky": None, "lighting": None}
    obj_tokens = []
    for clause in _split_clauses(text):
        toks = clause.split()
        has_obj = any(t in SHAPES for t in toks) or any(MATERIALS.get(t) in _VOLUMETRIC for t in toks)
        env_subject = any(t in ("sun", "sky", "sunny") for t in toks)
        cloud_words = any(t in ("cloud", "clouds", "cloudy", "overcast") for t in toks)
        # LIGHTING / time-of-day: harvest a preset word wherever it appears (it's an environment cue, not an object)
        for t in toks:
            if t in LIGHTING_WORDS:
                env["lighting"] = LIGHTING_WORDS[t]
        # A clause can carry BOTH objects and weather ("...a glass box on a sunny day"). So we always HARVEST the
        # environment keywords, but only DROP the clause's tokens when it is PURELY environmental (no object in it) --
        # otherwise the objects in a mixed clause were silently lost (the "sunny day" bug).
        if env_subject or cloud_words:
            if "sun" in toks or "sunny" in toks:
                env["sun"] = "bright" if ("bright" in toks or "sunny" in toks) else "soft"
            if "partly" in toks and ("cloudy" in toks or "cloud" in toks or "clouds" in toks):
                env["sky"] = "partly"
            elif "cloudy" in toks or "overcast" in toks:
                env["sky"] = "cloudy"
            # only read "clear/blue" as SKY when the clause has no object -- otherwise "blue" is an object's colour
            elif "clear" in toks or ("blue" in toks and not has_obj):
                env["sky"] = "clear"
        has_lighting = any(t in LIGHTING_WORDS for t in toks)
        is_pure_env = (env_subject or cloud_words or has_lighting) and not has_obj
        if not is_pure_env:
            # strip standalone lighting/time-of-day words so "a sphere at sunset" doesn't treat 'sunset' as an
            # object token -- but keep any word that is ALSO a real shape/color/size so those still parse. A word
            # that is only a lighting cue (sunset, noon, golden, dramatic, overcast) is dropped from object tokens.
            for t in toks:
                if t in LIGHTING_WORDS and t not in SHAPES and t not in COLORS and t not in SIZES:
                    continue
                obj_tokens.append(t)

    objects = []
    mods = {"color": None, "material": None, "size": None}    # pre-modifiers waiting for their shape noun
    article = None
    pending_rel = None                                        # (relword, source_object_index) waiting for its ref
    _NOUN_TAG = {"material", "materials", "colour", "color", "surface", "finish", "texture"}

    def _flush_mods_to(obj):
        for k in ("color", "material", "size"):
            if mods[k] is not None:
                obj[k] = mods[k]

    j = 0
    while j < len(obj_tokens):
        tok = obj_tokens[j]
        nxt = obj_tokens[j + 1] if j + 1 < len(obj_tokens) else ""
        kind = _classify(tok)
        resolved_val = None
        if kind is None and resolver is not None and tok not in _STOP and tok not in _NOUN_TAG and len(tok) > 2:
            hit = resolver.classify_unknown(tok)                # distributional fallback: 'crimson' -> ('color','red')
            if hit is not None:
                kind, resolved_val = hit
        if kind == "article":
            article = tok
            j += 1; continue
        if kind in ("color", "material", "size"):
            val = MATERIALS[tok] if kind == "material" else tok   # canonicalize material synonyms (metallic->metal)
            if resolved_val is not None:
                val = resolved_val
            if kind == "material" and val in _VOLUMETRIC:      # fog/smoke/fire are blobs with no shape noun
                obj = {"shape": "sphere", "color": mods["color"], "material": val, "size": mods["size"],
                       "relation": None}
                objects.append(obj)
                if pending_rel is not None:
                    objects[pending_rel[1]]["relation"] = (pending_rel[0], len(objects) - 1); pending_rel = None
                mods = {"color": None, "material": None, "size": None}; article = None
                j += 1; continue
            if nxt in _NOUN_TAG and objects:                   # 'with a glass MATERIAL' -> attribute of last object
                objects[-1][kind] = val
            else:
                mods[kind] = val                               # pre-modifier waiting for its shape noun
            j += 1; continue
        if kind == "relation":
            if objects and pending_rel is None:                # keep the FIRST relation word ('leaning' over 'on')
                pending_rel = (RELATIONS[tok], len(objects) - 1)
            j += 1; continue
        if kind == "shape":
            # a RESOLVED shape word ('spherical') that precedes a real shape noun ('ball') is an adjective, not its
            # own object -- skip it so 'spherical ball' is ONE sphere, not two.
            if resolved_val is not None and (nxt in SHAPES or (resolver is not None and
                                             (resolver.classify_unknown(nxt) or (None,))[0] == "shape")):
                j += 1; continue
            canon = resolved_val if resolved_val is not None else SHAPES[tok]
            if article == "the":                               # 'the glass box' = REFER back to an existing object
                ref = _resolve_reference(objects, canon, mods)
                if ref is not None:
                    if pending_rel is not None:
                        objects[pending_rel[1]]["relation"] = (pending_rel[0], ref)
                        pending_rel = None
                    mods = {"color": None, "material": None, "size": None}
                    article = None
                    j += 1; continue
            obj = {"shape": canon, "color": None, "material": None, "size": None, "relation": None}
            _flush_mods_to(obj)
            objects.append(obj)
            if pending_rel is not None:                        # the just-created object is the ref of the pending rel
                objects[pending_rel[1]]["relation"] = (pending_rel[0], len(objects) - 1)
                pending_rel = None
            mods = {"color": None, "material": None, "size": None}
            article = None
            j += 1; continue
        j += 1                                                 # unknown token -> glue / ignored
    return {"objects": objects, "environment": env}


def _split_clauses(text):
    out, cur = [], []
    for t in text.split():
        if t in (",", "."):
            if cur:
                out.append(" ".join(cur)); cur = []
        else:
            cur.append(t)
    if cur:
        out.append(" ".join(cur))
    return out


def _resolve_reference(objects, shape, mods):
    """Find an existing object matching `shape` and any given modifiers (e.g. 'the glass box'). Returns its index."""
    best = None
    for i, o in enumerate(objects):
        if o["shape"] != shape:
            continue
        if mods["material"] and o["material"] != mods["material"]:
            continue
        if mods["color"] and o["color"] != mods["color"]:
            continue
        best = i
    return best


# --------------------------------------------------------------------------------------------------------------
# VSA encoding: object -> record vector; scene -> one composable hypervector; bidirectional decode
# --------------------------------------------------------------------------------------------------------------
_FIELDS = ("shape", "color", "material", "size")


def _record_fields(obj):
    """The {field: value_name} record for an object, skipping unset fields. 'none' fills an unset slot so the
    decoder always has a value to clean against."""
    return {f: (obj[f] if obj[f] is not None else "none") for f in _FIELDS}


def schema_codebooks():
    """The per-field codebooks (possible value names) the decoder cleans against -- the 'shape' of an object record."""
    return {
        "shape": sorted(set(SHAPES.values())) + ["none"],
        "color": sorted(COLORS) + ["none"],
        "material": sorted(set(MATERIALS.values())) + ["none"],
        "size": sorted(SIZES) + ["none"],
    }


def encode_objects(objects, mind):
    """Encode each object as a VSA record vector (mind.encode_record). Returns a list of (D,) vectors -- each one a
    bundle of role-bound attribute atoms, bidirectionally decodable."""
    return [mind.encode_record(_record_fields(o)) for o in objects]


def encode_scene(objects, mind):
    """Bundle the objects into ONE composable scene hypervector: superpose bind(OBJ_i, record_i). Returns
    (scene_vector, [record_vectors], [obj_role_atoms]). Unbind OBJ_i then decode to read object i back -- the scene
    is content-addressable by slot."""
    from holographic.agents_and_reasoning.holographic_ai import bind, bundle, derived_atom
    recs = encode_objects(objects, mind)
    roles = [derived_atom(mind.seed, "OBJ_%d" % i, mind.dim) for i in range(len(recs))]
    scene = bundle([bind(roles[i], recs[i]) for i in range(len(recs))]) if recs else np.zeros(mind.dim)
    return scene, recs, roles


def decode_object(vec, mind):
    """Decode an object record vector back to {field: value_name} by cleaning each unbound role against its field
    codebook -- the bidirectional read. Works on a clean record OR a noisy one unbound from the scene bundle (the
    codebook cleanup denoises the bundle crosstalk)."""
    return mind.decode_record(vec, schema_codebooks())


def query_scene(scene, roles, mind, slot):
    """Read object `slot` back OUT of the bundled scene hypervector: unbind its role, then decode. Demonstrates the
    scene is queryable by slot even after superposition."""
    from holographic.agents_and_reasoning.holographic_ai import unbind
    return decode_object(unbind(scene, roles[slot]), mind)


def find_objects(objects, **attr):
    """Find the indices of objects matching attribute=value constraints (e.g. find_objects(material='glass'))."""
    out = []
    for i, o in enumerate(objects):
        if all(o.get(k) == v for k, v in attr.items()):
            out.append(i)
    return out


def batch_set(objects, field, value, where=None):
    """Batch-edit: set `field`=`value` on all objects (or only those whose index is in `where`). Returns a NEW list
    (the parsed scene is immutable input). This is 'make all materials reflective' as one call."""
    out = []
    for i, o in enumerate(objects):
        o2 = dict(o)
        if where is None or i in where:
            o2[field] = value
        out.append(o2)
    return out


# --------------------------------------------------------------------------------------------------------------
# Realize: objects -> positioned SDF primitives with materials, ready to render
# --------------------------------------------------------------------------------------------------------------
class _SphereSDF:
    def __init__(self, c, r):
        self.c = np.asarray(c, float); self.r = float(r)

    def eval(self, P):
        return np.linalg.norm(np.asarray(P, float) - self.c, axis=1) - self.r


class _EllipsoidSDF:
    """iq's bounded ellipsoid distance k1*(k1-1)/k2 -- no exact closed form exists, but this bound never oversteps
    a raymarch. What a stretched 'sphere' object (a leaf, a lemon, a squashed fruit) realizes to."""

    def __init__(self, c, radii):
        self.c = np.asarray(c, float); self.a = np.maximum(np.asarray(radii, float), 1e-6)

    def eval(self, P):
        q = (np.asarray(P, float) - self.c)
        k1 = np.linalg.norm(q / self.a, axis=1)
        k2 = np.linalg.norm(q / (self.a * self.a), axis=1)
        return np.where(k2 > 1e-12, k1 * (k1 - 1.0) / np.maximum(k2, 1e-12), -self.a.min())


class _BoxSDF:
    def __init__(self, c, half):
        self.c = np.asarray(c, float); self.h = np.asarray(half, float)

    def eval(self, P):
        q = np.abs(np.asarray(P, float) - self.c) - self.h
        return (np.linalg.norm(np.maximum(q, 0.0), axis=1)
                + np.minimum(np.max(q, axis=1), 0.0))


class _CylinderSDF:
    """A vertical capped cylinder of radius r, half-height hy, centred at c (axis = +y)."""
    def __init__(self, c, r, hy):
        self.c = np.asarray(c, float); self.r = float(r); self.hy = float(hy)

    def eval(self, P):
        p = np.asarray(P, float) - self.c
        d_rad = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - self.r      # distance in the xz-plane
        d_ax = np.abs(p[:, 1]) - self.hy                           # distance along the axis
        dx = np.maximum(d_rad, 0.0); dy = np.maximum(d_ax, 0.0)
        return np.minimum(np.maximum(d_rad, d_ax), 0.0) + np.sqrt(dx * dx + dy * dy)


class _ConeSDF:
    """A vertical cone, base radius r at the bottom tapering to a point at the top, half-height hy, centred at c."""
    def __init__(self, c, r, hy):
        self.c = np.asarray(c, float); self.r = float(r); self.hy = float(hy)

    def eval(self, P):
        p = np.asarray(P, float) - self.c
        qy = p[:, 1] + self.hy                                     # 0 at the base, 2*hy at the tip
        rad = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
        frac = np.clip(qy / (2.0 * self.hy), 0.0, 1.0)
        r_at = self.r * (1.0 - frac)                              # radius shrinks to 0 at the tip
        d_side = rad - r_at
        d_ax = np.abs(p[:, 1]) - self.hy
        dx = np.maximum(d_side, 0.0); dy = np.maximum(d_ax, 0.0)
        return np.minimum(np.maximum(d_side, d_ax), 0.0) + np.sqrt(dx * dx + dy * dy)


class _TorusSDF:
    """A torus lying in the xz-plane: major radius R (ring), minor radius rt (tube), centred at c."""
    def __init__(self, c, R, rt):
        self.c = np.asarray(c, float); self.R = float(R); self.rt = float(rt)

    def eval(self, P):
        p = np.asarray(P, float) - self.c
        qx = np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - self.R
        return np.sqrt(qx * qx + p[:, 1] ** 2) - self.rt


class _RotatedSDF:
    """Wrap an axis-aligned primitive SDF and ROTATE it about its centre by (axis, angle_rad). WHY THIS EXISTS: the
    local primitive SDFs are axis-aligned (a cone always points +y), so a real rosette of leaves splaying OUTWARD was
    impossible -- the long-standing 'rotation is not modelled' kept negative. A rotation is rigid (distance-preserving),
    so evaluating the inner SDF at query points rotated by the INVERSE rotation about the centre is an EXACT rotated
    SDF -- the same trick the node palette's sdf_rotate uses. Reuses holographic_sdf._rot_matrix for the matrix (do not
    reinvent the wheel). Default-off: realize_scene only wraps an object that carries a 'rotation' field."""
    def __init__(self, inner, center, axis, angle):
        from holographic.mesh_and_geometry.holographic_sdf import _rot_matrix
        self.inner = inner
        self.c = np.asarray(center, float)
        self.Rinv = _rot_matrix(axis, -float(angle))         # rotate query points by -angle == rotate the object by +angle
        self.r = getattr(inner, "r", None)                   # pass through a radius hint for bounds (rotation-invariant)

    def eval(self, P):
        P = np.asarray(P, float)
        local = self.c + (P - self.c) @ self.Rinv.T          # per-point: c + Rinv @ (P - c); then the inner (which
        return self.inner.eval(local)                        # subtracts the SAME centre) sees the rotated field


def _base_size(obj):
    """Resolve an object's size word to (uniform_scale, (sx,sy,sz) stretch)."""
    s = obj.get("size")
    if s is None:
        return 1.0, (1.0, 1.0, 1.0)
    kind, val = SIZES[s]
    if kind == "uniform":
        return float(val), (1.0, 1.0, 1.0)
    return 1.0, tuple(val)


def realize_scene(objects, spacing=2.7, base_radius=0.7):
    """Turn parsed objects into a list of renderables: dicts with sdf (an object with .eval), color (rgb), and
    material render kwargs. Positions come from the relations (heuristic): roots spread along x; 'inside' sits at the
    container centre (and shrinks); 'leaning'/'on'/'beside'/'diagonal' offset from the reference. HONEST: rotation is
    not modelled (the SDF combinators are axis-aligned), so 'diagonal' is faked by an offset + stretch, not a true
    tilt -- a kept limitation."""
    n = len(objects)
    pos = [np.array([0.0, 0.0, 0.0]) for _ in range(n)]
    scale = [1.0] * n
    roots = [i for i, o in enumerate(objects) if o["relation"] is None and
             not any(oo["relation"] and oo["relation"][1] == i for oo in objects)]
    # spread the root/container objects along x
    k = 0
    for i in range(n):
        if objects[i]["relation"] is None:
            pos[i] = np.array([(k - (max(len(roots), 1) - 1) / 2.0) * spacing, 0.0, 0.0]); k += 1
    # apply relations
    for i, o in enumerate(objects):
        rel = o["relation"]
        if rel is None:
            continue
        relword, ref = rel
        if ref < 0 or ref >= n:
            continue
        if relword == "inside":
            pos[i] = pos[ref].copy(); scale[i] = 0.55         # nested, shrunk to fit (but visible through glass)
        elif relword in ("on", "above"):
            pos[i] = pos[ref] + np.array([0.0, 1.1, 0.0])
        elif relword == "under":
            pos[i] = pos[ref] + np.array([0.0, -1.1, 0.0])
        elif relword == "behind":
            pos[i] = pos[ref] + np.array([0.0, 0.0, -2.4])
        elif relword == "front":
            pos[i] = pos[ref] + np.array([0.0, 0.0, 2.4])
        else:                                                 # leaning / beside / diagonal
            off = np.array([2.1, 0.5, 0.3]) if relword in ("leaning", "diagonal") else np.array([2.4, 0.0, 0.0])
            pos[i] = pos[ref] + off

    out = []
    for i, o in enumerate(objects):
        uni, stretch = _base_size(o)
        # EXPLICIT per-axis stretch (default-off, additive): o['stretch']=(sx,sy,sz) overrides the size-word stretch --
        # what photo-matched authoring needs (a plate is a squashed cylinder, a leaf a flattened sphere). Absent ->
        # the size-word behaviour, byte-identical.
        if o.get("stretch") is not None:
            stretch = tuple(float(s) for s in o["stretch"])
        # B2-transform: an optional explicit OFFSET (translate) and SCALE_MUL (continuous scale) on top of the
        # relation-derived layout. Absent -> (0,0,0) and 1.0, i.e. BYTE-IDENTICAL to the pre-transform layout.
        obj_pos = pos[i] + np.asarray(o.get("offset", (0.0, 0.0, 0.0)), float)
        # ABSOLUTE position override (default-off): if an object carries an explicit 'position', it is placed THERE,
        # bypassing the relation/row layout entirely. This is what lets scene_from_image drop each object at the
        # location its photo region implies. Absent -> the relation+offset layout as before (byte-identical).
        if o.get("position") is not None:
            obj_pos = np.asarray(o["position"], float)
        r = base_radius * uni * scale[i] * float(o.get("scale_mul", 1.0))
        # EXACT RGB (default-off, additive): a (r,g,b) tuple/list as o['color'] is used directly -- what a
        # photo-matched scene needs (colour picked from the image, not the nearest named colour). A string name
        # goes through COLORS exactly as before, byte-identical.
        if isinstance(o["color"], (tuple, list)) and len(o["color"]) == 3:
            col = tuple(float(c) for c in o["color"])
        else:
            col = COLORS.get(o["color"], (0.8, 0.8, 0.8))
        mat = MATERIAL_RENDER.get(o["material"], MATERIAL_RENDER[None])
        if o["shape"] == "sphere":
            # a STRETCHED sphere is an ellipsoid (iq's bounded approximation) -- default (1,1,1) stays the exact
            # sphere, byte-identical. What a leaf, a lemon, or a squashed fruit actually is.
            if tuple(stretch) != (1.0, 1.0, 1.0):
                sdf = _EllipsoidSDF(obj_pos, np.array(stretch, float) * r)
            else:
                sdf = _SphereSDF(obj_pos, r)
        elif o["shape"] == "cylinder":
            sx, sy, sz = stretch
            sdf = _CylinderSDF(obj_pos, r * 0.72 * max(sx, sz), r * 1.05 * sy)
        elif o["shape"] == "cone":
            sx, sy, sz = stretch
            sdf = _ConeSDF(obj_pos, r * 0.9 * max(sx, sz), r * 1.1 * sy)
        elif o["shape"] == "torus":
            sdf = _TorusSDF(obj_pos, r * 0.72, r * 0.3)
        else:
            half = np.array(stretch) * r
            sdf = _BoxSDF(obj_pos, half)
        # optional ROTATION about the object's centre -- (axis, angle_degrees). Absent -> unrotated, BYTE-IDENTICAL to
        # the axis-aligned layout. This is what lets leaves splay into a rosette (closes the axis-aligned negative).
        rot = o.get("rotation")
        if rot:
            axis, angle_deg = rot
            sdf = _RotatedSDF(sdf, obj_pos, axis, np.deg2rad(float(angle_deg)))
        out.append({"sdf": sdf, "color": col, "material": mat, "mat_name": o["material"], "name": _obj_name(o)})
    return out


def _obj_name(o):
    bits = [o[k] for k in ("color", "size", "material") if o[k]]
    # an exact-RGB colour (tuple) reads as 'rgb(0.75,0.62,0.25)' in the descriptive name
    bits = [b if isinstance(b, str) else "rgb(%.2f,%.2f,%.2f)" % tuple(float(c) for c in b) for b in bits]
    return " ".join(bits + [o["shape"]])


class _PlaneSDF:
    """A horizontal ground plane at height y0 -- gives objects something to stand on and catch their shadows, which
    is most of what makes a render read as 'real' rather than floating shapes."""

    def __init__(self, y0):
        self.y0 = float(y0)

    def eval(self, P):
        return np.asarray(P, float)[:, 1] - self.y0


class _WallSDF:
    """A vertical BACKDROP plane at z=z0 (normal +z) -- the wall behind the scene, so a render has a real background to
    compare against a photo's wall instead of empty sky. With the ground plane it forms a floor+wall corner, which is
    most of what an indoor product photo's frame is."""

    def __init__(self, z0):
        self.z0 = float(z0)

    def eval(self, P):
        return np.asarray(P, float)[:, 2] - self.z0


class _UnionSDF:
    """min over a list of object SDFs -- the whole scene as ONE field, so a single march sees every object (and
    therefore casts inter-object shadows and AO). `ids(P)` reports which object is nearest at each point."""

    def __init__(self, sdfs):
        self.sdfs = list(sdfs)

    def eval(self, P):
        return np.min(np.stack([s.eval(P) for s in self.sdfs], axis=0), axis=0)

    def ids(self, P):
        return np.argmin(np.stack([s.eval(P) for s in self.sdfs], axis=0), axis=0)


def _reflectivity(mat_name):
    return {"mirror": 0.85, "brushed": 0.55, "glossy": 0.35, "metal": 0.5, "glass": 0.12}.get(mat_name, 0.05)


def _roughness(mat_name):
    """Reflection-lobe half-angle (radians) per material: 0 = a sharp mirror; larger = a blurrier glossy reflection.
    This is the width the ray-differential FRAME reconstructs -- surface micro-imperfections spread the reflection."""
    return {"brushed": 0.16, "glossy": 0.09, "metal": 0.05}.get(mat_name, 0.0)


def _perp_frames(D):
    """Per-row orthonormal (u, v) perpendicular to each unit direction in D (M,3) -- the plane the glossy lobe's
    marginal rays are tilted into. Vectorised."""
    D = np.asarray(D, float)
    a = np.where((np.abs(D[:, 0]) < 0.9)[:, None], np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    u = np.cross(D, a); u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    v = np.cross(D, u)
    return u, v


def _shade_reflection_rays(union, O, D, colors, sun_dir, amb, sun_i):
    """Trace + shade one bounce of reflection rays (sky where they miss). Factored so the sharp mirror ray AND the
    glossy frame's tilted rays use identical shading."""
    from holographic.rendering.holographic_raymarch import sphere_trace, sdf_normal, sky_dome
    from holographic.rendering.holographic_shadowhome import Shadow      # visibility via the Shadow home  R8
    rc = sky_dome(D, sun_dir=tuple(sun_dir))
    hm, tm, Pmh = sphere_trace(union, O, D)
    if np.any(hm):
        Nm = sdf_normal(union, Pmh[hm]); idm = union.ids(Pmh[hm])
        lamm = np.clip((Nm * sun_dir).sum(1), 0, 1)
        shm = Shadow.soft(union, Pmh[hm] + Nm * 3e-3, sun_dir); aom = Shadow.ambient_occlusion(union, Pmh[hm], Nm)
        rc[hm] = np.clip(colors[idm] * (amb * aom + lamm * shm * sun_i)[:, None], 0, 1)
    return rc


def cloud_field(center, radius, density=6.0, seed=0, grid=56, octaves=6, gain=0.62, noise_grid=None):
    """A density field (points(N,3)->density>=0) for a CONVINCING CUMULUS CLOUD.

    Shape recipe (this is what makes it read as a real cloud, not an egg):
      * a SMOOTH-UNION of many spherical lobes -- a wide low body plus a cluster of billows rising on top -- so the
        puffs MERGE into one cauliflower mass instead of a single ellipsoid;
      * a FLAT BASE (density hard-cut below the cloudbase, feathered just above) -- cumulus sit on a level bottom;
      * DOMAIN-WARPED fBm carving the billows, with stronger erosion at the low-density RIM so the edges break up
        into wisps rather than a clean outline.
    Render it with volume_render(..., self_shadow=True) and a modest sigma (~6-8) so light PENETRATES and you get
    the lit-crown / sky-blue-shadowed-base modelling; too high a density/sigma and you only see a flat white shell.
    `mind.make_cloud()` wires all of that up for you.

    The fBm is BAKED onto a `grid`^3 lattice ONCE (the noise query is per-point, ~2 ms/point, so baking dominates
    cost: grid=24 ~30 s, grid=32 ~60 s, grid=48 ~4 min); after that the field is a fast trilinear lookup so the
    render is ~1 s. Reuse one field across frames of the same cloud.

    Pass `noise_grid` (a (grid,grid,grid) array already baked over bounds=center+-radius*1.8, e.g. the completed
    result of a holographic_renderjobs.make_noise_bake_job) to SKIP the bake entirely -- the one way to make a
    cloud from a background/paused-and-resumed bake job without re-baking. See holographic_noise.FractalNoise."""
    from holographic.sampling_and_signal.holographic_noise import FractalNoise
    c = np.asarray(center, float); r = float(radius)
    lo = c - r * 1.8; hi = c + r * 1.8
    bounds = [(float(lo[k]), float(hi[k])) for k in range(3)]
    if noise_grid is not None:
        noise = np.asarray(noise_grid, float)
        grid = noise.shape[0]
    else:
        fbm = FractalNoise(3, dim=1024, bounds=bounds, octaves=octaves, lacunarity=2.2,
                           gain=gain, base_bandwidth=2.4 / max(r, 1e-3), seed=seed * 131 + 7)
        noise = fbm.sample_grid_fast(grid)      # exact-but-vectorised bake (~40x faster than per-point sample_grid)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-9)     # -> [0,1]
    noise = noise - noise.mean()                               # centre, then stretch to full swing so the
    noise = 0.5 + 0.5 * noise / (np.abs(noise).max() + 1e-9)   # rim erosion has high-contrast noise to carve with
    span = (hi - lo)

    # multi-lobe cumulus: a wide body + billows that rise on top (positions in units of r, relative to center)
    rng = np.random.default_rng(seed * 977 + 3)
    lobes = [(np.array([0.0, -0.05, 0.0]), 1.0), (np.array([-0.55, -0.10, 0.20]), 0.62),
             (np.array([0.60, -0.08, -0.15]), 0.66), (np.array([0.15, 0.15, -0.50]), 0.55),
             (np.array([-0.20, 0.12, 0.50]), 0.58), (np.array([0.0, 0.32, 0.0]), 0.55),
             (np.array([-0.35, 0.28, -0.10]), 0.42), (np.array([0.38, 0.30, 0.15]), 0.44)]
    for _ in range(10):
        ang = rng.uniform(0, 2 * np.pi); rad = rng.uniform(0.20, 0.70)
        lobes.append((np.array([rad * np.cos(ang), rng.uniform(0.10, 0.50), rad * np.sin(ang) * 0.85]),
                      rng.uniform(0.28, 0.44)))
    base_y = -0.55                                             # cloudbase height (in units of r)

    def _sample_noise(P):
        g = np.clip((P - lo) / span * (grid - 1), 0, grid - 1 - 1e-6)
        i = np.floor(g).astype(int); f = g - i
        i0, j0, k0 = i[:, 0], i[:, 1], i[:, 2]; i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        c00 = noise[i0, j0, k0] * (1 - fx) + noise[i1, j0, k0] * fx
        c01 = noise[i0, j0, k1] * (1 - fx) + noise[i1, j0, k1] * fx
        c10 = noise[i0, j1, k0] * (1 - fx) + noise[i1, j1, k0] * fx
        c11 = noise[i0, j1, k1] * (1 - fx) + noise[i1, j1, k1] * fx
        c0 = c00 * (1 - fy) + c10 * fy; c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz

    def field(P):
        P = np.asarray(P, float)
        q = (P - c) / r                                        # normalise to lobe space
        # smooth-union of lobe falloffs (polynomial smin on the "inside-ness" 1 - dist/r_lobe)
        val = np.full(q.shape[0], -1e9)
        for lc, lr in lobes:
            d = 1.0 - np.linalg.norm(q - lc, axis=1) / lr
            k = 0.30
            h = np.clip(0.5 + 0.5 * (val - d) / k, 0.0, 1.0)
            val = val * h + d * (1 - h) + k * h * (1 - h)
        env = np.clip(val, 0.0, 1.0)
        env *= np.clip((q[:, 1] - base_y) / 0.16, 0.0, 1.0)    # flat but softly-feathered base
        warp = 0.15 * (_sample_noise(P * 1.9 + 3.1) - 0.5)     # domain warp
        n = _sample_noise(P + warp[:, None])                   # big billows (low freq)
        n2 = _sample_noise(P * 2.7 + 5.0)                      # fine cauliflower detail (high freq)
        # RISING-THRESHOLD erosion: the core (env high) keeps its density; toward the rim (env low) the noise has
        # to clear a rising bar to survive, so the boundary dissolves into disconnected wisps instead of a hard
        # shell. Then a high gamma (d**2) thins those wisps into a genuinely SEE-THROUGH translucent skirt while
        # the dense core stays opaque -- this is what stops the cloud reading as solid cartoon clay.
        thresh = 0.20 + 0.75 * (1.0 - env)                     # low bar in the core, high bar at the rim
        detail = 0.55 * n + 0.45 * n2                          # combined noise in [0,1]
        d = env - thresh * (1.0 - detail)                      # noise pokes holes; strongest at the rim
        d = np.clip(d, 0.0, 1.0)
        return d ** 2.0 * density                              # gamma 2: translucent wispy edges, opaque core
    return field


def volumetric_field(center, radius, density=1.4, turbulence=0.7, seed=0):
    """A density field (points(N,3)->density>=0) for a volumetric object: a soft sphere falloff modulated by cheap
    deterministic turbulence so smoke/fog reads as wispy rather than a clean ball. Fed to volume_render."""
    c = np.asarray(center, float); r = float(radius)

    def field(P):
        P = np.asarray(P, float)
        d = np.linalg.norm(P - c, axis=1)
        base = np.clip((r - d) / r, 0.0, 1.0) ** 2 * density   # soft sphere
        if turbulence > 0:                                     # sum-of-sines value noise (deterministic, no deps)
            s = seed * 0.7
            n = (np.sin(P[:, 0] * 3.9 + s) * np.sin(P[:, 1] * 3.3 + s) * np.sin(P[:, 2] * 4.1 - s)
                 + 0.5 * np.sin(P[:, 0] * 7.7) * np.sin(P[:, 1] * 8.3) * np.sin(P[:, 2] * 7.1))
            base = base * np.clip(1.0 + turbulence * n, 0.0, None)
        return base
    return field


def _scene_setup(objects, ground, sky, sun, glass_tint, rs=None, lighting=None, sun_scale=1.0,
                 ground_color=None, backdrop=None):
    """Build the immutable per-scene context (SDFs, colours, materials, lighting) once; reused by every ray batch so
    the adaptive refinement does not rebuild it. Pass `rs` to use pre-realized renderables (so volumetric objects can
    be split out and the layout computed once). `lighting` is an optional LIGHTING preset name (noon/sunset/golden/
    overcast/night/studio/dramatic/...) that sets the sun direction, colour, intensity and ambient tint.
    `ground_color` recolours the floor (default -> the neutral gray, byte-identical); `backdrop` (an rgb) adds a
    vertical wall behind the scene of that colour -- both let a render match a photo's floor + wall."""
    if rs is None:
        rs = realize_scene(objects)
    if not rs and not ground:
        return None
    if ground:
        lows = []
        for r in rs:
            s = r["sdf"]
            if isinstance(s, _SphereSDF):
                lows.append(float(s.c[1] - s.r))
            elif isinstance(s, _BoxSDF):
                lows.append(float(s.c[1] - s.h[1]))
        y0 = (min(lows) - 0.08) if lows else -0.9
        gcol = tuple(float(c) for c in ground_color[:3]) if ground_color is not None else (0.62, 0.63, 0.66)
        rs = list(rs) + [{"sdf": _PlaneSDF(y0), "color": gcol, "material": MATERIAL_RENDER["matte"],
                          "mat_name": "matte", "name": "ground"}]
    if backdrop is not None:                                  # a vertical wall behind the scene (photo backdrop)
        backs = []
        for r in rs:
            s = r["sdf"]
            c = getattr(s, "c", None)
            if c is not None:
                backs.append(float(c[2]))
        z0 = (min(backs) - 1.6) if backs else -3.0
        bcol = tuple(float(c) for c in backdrop[:3])
        rs = list(rs) + [{"sdf": _WallSDF(z0), "color": bcol, "material": MATERIAL_RENDER["matte"],
                          "mat_name": "matte", "name": "backdrop"}]
    if not rs:
        return None
    sdfs = [r["sdf"] for r in rs]
    sun_dir = np.array([-0.45, 0.8, -0.35]); sun_dir = sun_dir / np.linalg.norm(sun_dir)
    sun_col = np.array([1.0, 1.0, 1.0]); amb = 0.34 if sky in ("cloudy", "partly") else 0.24
    amb_col = np.array([1.0, 1.0, 1.0]); sun_i = 1.0 if sun == "bright" else 0.7
    preset = LIGHTING.get(lighting) if lighting else None
    if preset is not None:                                    # a named time-of-day / mood preset overrides the sun
        sun_dir = np.asarray(preset["dir"], float); sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-9)
        sun_col = np.asarray(preset["sun_col"], float); sun_i = preset["sun_i"]
        amb = preset["amb"]; amb_col = np.asarray(preset["amb_col"], float)
    # B2b: relative brightness. sun_scale multiplies the FINAL sun intensity (after any preset), so a spoken
    # "brighter"/"dimmer" scales the scene without a new preset. Default 1.0 -> byte-identical to before.
    sun_i = float(sun_i) * float(sun_scale)
    return {"sdfs": sdfs, "union": _UnionSDF(sdfs),
            "colors": np.array([r["color"] for r in rs]),
            "refl": np.array([_reflectivity(r["mat_name"]) for r in rs]),
            "rough": np.array([_roughness(r["mat_name"]) for r in rs]),                # glossy reflection lobe width
            "is_glass": np.array([r["mat_name"] == "glass" for r in rs]),
            "is_sss": np.array([r["mat_name"] == "wax" for r in rs]),               # subsurface scattering
            "is_translucent": np.array([r["mat_name"] == "translucent" for r in rs]),   # diffuse see-through
            "sun_dir": sun_dir, "glass_tint": np.array(glass_tint),
            "sun_col": sun_col, "amb_col": amb_col,
            "amb": amb, "sun_i": sun_i}


def _shade_rays(ctx, O, D):
    """Shade an ARBITRARY batch of rays against the scene (used for the base grid AND for edge subrays). Returns
    (col (M,3), hit (M,), t (M,), ids (M,) with -1 = background)."""
    from holographic.rendering.holographic_raymarch import sphere_trace, sdf_normal, sky_dome
    from holographic.rendering.holographic_shadowhome import Shadow      # visibility via the Shadow home  R8
    union = ctx["union"]; sdfs = ctx["sdfs"]; colors = ctx["colors"]; refl = ctx["refl"]; is_glass = ctx["is_glass"]
    sun_dir = ctx["sun_dir"]; amb = ctx["amb"]; sun_i = ctx["sun_i"]
    sun_col = ctx.get("sun_col", np.array([1.0, 1.0, 1.0]))
    amb_col = ctx.get("amb_col", np.array([1.0, 1.0, 1.0]))
    hit, t, P = sphere_trace(union, O, D, relax=ctx.get("relax", 1.0))
    col = sky_dome(D, sun_dir=tuple(sun_dir))
    ids_full = -np.ones(len(D), int)
    if np.any(hit):
        ids = union.ids(P[hit]); ids_full[hit] = ids
        N = sdf_normal(union, P[hit])
        lam = np.clip((N * sun_dir).sum(1), 0, 1)
        sh = Shadow.soft(union, P[hit] + N * 3e-3, sun_dir)     # marches the WHOLE union -> inter-object shadows
        ao = Shadow.ambient_occlusion(union, P[hit], N)
        # coloured lighting: warm/cool sun tint on the direct term, sky-tinted ambient fill (per LIGHTING preset)
        shade_rgb = (amb * ao)[:, None] * amb_col[None, :] + (lam * sh * sun_i)[:, None] * sun_col[None, :]
        albedo = colors[ids]                                    # default: the nearest object's colour
        r_amt = refl[ids]                                       # reflectivity and roughness default to the object's
        rough_all = ctx.get("rough", np.zeros(len(refl)))[ids]
        rf = ctx.get("region_field")                           # OPTIONAL: shade by REGION -- a boundary's material wins
        if rf is not None:                                      # (a biome planet, OR one body with many materials)
            rmask = rf.classify(P[hit]) >= 0
            if np.any(rmask):
                albedo = albedo.copy(); albedo[rmask] = rf.material_at(P[hit])[rmask]
                r_amt = r_amt.copy(); r_amt[rmask] = rf.reflect_at(P[hit], default=0.0)[rmask]      # per-region reflect
                rough_all = rough_all.copy(); rough_all[rmask] = rf.roughness_at(P[hit])[rmask]     # per-region rough
        shaded = albedo * shade_rgb
        refl_dir = D[hit] - 2.0 * (D[hit] * N).sum(1)[:, None] * N
        reflected = sky_dome(refl_dir, sun_dir=tuple(sun_dir))   # default: a mirror reflects the sky dome
        mirror = r_amt > 0.05                                    # only trace a reflection ray for reflective surfaces
        if np.any(mirror):                                       # ONE-BOUNCE OBJECT REFLECTION: the mirror reflects
            Pm = P[hit][mirror] + N[mirror] * 3e-3              # other OBJECTS, not just the sky (wired into render_scene)
            Dm = refl_dir[mirror]
            rc = _shade_reflection_rays(union, Pm, Dm, colors, sun_dir, amb, sun_i)   # the sharp centre reflection
            rough_m = rough_all[mirror]
            glossy = rough_m > 1e-3
            if np.any(glossy):                                  # GLOSSY: reconstruct the roughness LOBE with the frame
                Pg = Pm[glossy]; Dg = Dm[glossy]; rg = rough_m[glossy]   # (Moose's pencil): 4 marginal rays tilted by
                u, v = _perp_frames(Dg)                         # the roughness angle + the centre -> a 5-tap lobe average
                acc = rc[glossy].copy()
                for su, sv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    Dt = Dg + rg[:, None] * (su * u + sv * v)
                    Dt = Dt / (np.linalg.norm(Dt, axis=1, keepdims=True) + 1e-12)
                    acc = acc + _shade_reflection_rays(union, Pg, Dt, colors, sun_dir, amb, sun_i)
                rc[glossy] = acc / 5.0                          # 5-ray frame stands in for the whole glossy bundle
            reflected[mirror] = rc
        shaded = shaded + r_amt[:, None] * reflected
        sss_here = ctx["is_sss"][ids]                           # SUBSURFACE SCATTERING: thin parts transmit the sun
        if np.any(sss_here):                                    # and GLOW (wax/jade/marble/skin), via Beer-Lambert
            from holographic.rendering.holographic_raymarch import subsurface
            tS = subsurface(union, P[hit][sss_here], N[sss_here], sun_dir)   # transmission: thin -> ~1 (bright)
            glow = colors[ids][sss_here] * (0.7 * tS[:, None]) * sun_i       # forward-scatter glow through the body
            shaded[sss_here] = shaded[sss_here] + glow
        col[hit] = np.clip(shaded, 0, 1)
        glass_here = is_glass[ids]
        if np.any(glass_here) and not np.all(is_glass):
            ng_i = [i for i in range(len(sdfs)) if not is_glass[i]]
            un2 = _UnionSDF([sdfs[i] for i in ng_i]); ng_colors = colors[ng_i]
            idx_hit = np.where(hit)[0]; gidx = idx_hit[glass_here]
            Pg = P[gidx]; Dg = D[gidx]
            h2, t2, P2 = sphere_trace(un2, Pg + Dg * 6e-3, Dg)
            behind = sky_dome(Dg, sun_dir=tuple(sun_dir))
            if np.any(h2):
                N2 = sdf_normal(un2, P2[h2])
                lam2 = np.clip((N2 * sun_dir).sum(1), 0, 1)
                sh2 = Shadow.soft(union, P2[h2] + N2 * 3e-3, sun_dir)
                ao2 = Shadow.ambient_occlusion(un2, P2[h2], N2)
                behind[h2] = np.clip(ng_colors[un2.ids(P2[h2])] * (amb * ao2 + lam2 * sh2 * sun_i)[:, None], 0, 1)
            Ng = sdf_normal(union, Pg)
            fres = (1.0 - np.clip(np.abs((Ng * (-Dg)).sum(1)), 0, 1)) ** 3
            col[gidx] = np.clip(0.12 * col[gidx] + 0.78 * behind * ctx["glass_tint"] + 0.35 * fres[:, None], 0, 1)
        trans_here = ctx["is_translucent"][ids]                 # TRANSLUCENCY: a diffuse, tinted see-through (frosted)
        if np.any(trans_here):                                  # -- the object behind shows through, blurred & coloured
            ng_i = [i for i in range(len(sdfs)) if not ctx["is_translucent"][i]]
            if ng_i:
                unT = _UnionSDF([sdfs[i] for i in ng_i]); ngT = colors[ng_i]
                idx_hit2 = np.where(hit)[0]; tidx = idx_hit2[trans_here]
                Pt = P[tidx]; Dt = D[tidx]
                hT, tT, PT = sphere_trace(unT, Pt + Dt * 6e-3, Dt)
                behindT = sky_dome(Dt, sun_dir=tuple(sun_dir))
                if np.any(hT):
                    NT = sdf_normal(unT, PT[hT]); lamT = np.clip((NT * sun_dir).sum(1), 0, 1)
                    shT = Shadow.soft(union, PT[hT] + NT * 3e-3, sun_dir); aoT = Shadow.ambient_occlusion(unT, PT[hT], NT)
                    behindT[hT] = np.clip(ngT[unT.ids(PT[hT])] * (amb * aoT + lamT * shT * sun_i)[:, None], 0, 1)
                # diffuse mix: half the object's own (tinted) colour, half the (scattered) light from behind
                col[tidx] = np.clip(0.5 * col[tidx] + 0.5 * behindT * colors[ids][trans_here] * 1.4, 0, 1)
    return col, hit, t, ids_full


def _edge_mask(idbuf, depth, frame, depth_rel=0.06, lum_thr=0.06):
    """The 2-D analog of adaptive_anchors' curvature detector: mark the pixels where the image BENDS -- a material-id
    boundary, a depth discontinuity (silhouette), or a strong luminance gradient (shadow / AO edge). These are the
    only pixels that alias, so the only ones worth supersampling. Dilated by one pixel so AA covers the edge."""
    H, W = idbuf.shape
    E = np.zeros((H, W), bool)
    dh = idbuf[:, :-1] != idbuf[:, 1:]; E[:, :-1] |= dh; E[:, 1:] |= dh
    dv = idbuf[:-1, :] != idbuf[1:, :]; E[:-1, :] |= dv; E[1:, :] |= dv
    fd = np.where(np.isfinite(depth) & (depth < 1e29), depth, 0.0)
    scale = max(1e-6, float(np.median(fd[fd > 0])) if np.any(fd > 0) else 1.0)
    ddh = np.abs(fd[:, :-1] - fd[:, 1:]) > depth_rel * scale; E[:, :-1] |= ddh; E[:, 1:] |= ddh
    ddv = np.abs(fd[:-1, :] - fd[1:, :]) > depth_rel * scale; E[:-1, :] |= ddv; E[1:, :] |= ddv
    lum = frame.mean(2)
    lh = np.abs(lum[:, :-1] - lum[:, 1:]) > lum_thr; E[:, :-1] |= lh; E[:, 1:] |= lh
    lv = np.abs(lum[:-1, :] - lum[1:, :]) > lum_thr; E[:-1, :] |= lv; E[1:, :] |= lv
    D = E.copy()
    D[:-1] |= E[1:]; D[1:] |= E[:-1]; D[:, :-1] |= E[:, 1:]; D[:, 1:] |= E[:, :-1]
    return D


def _scene_aabb(rs, margin=1.0):
    """Axis-aligned bounding box of the realized primitives (+ margin) -- the volume to bake the SDF over. Spheres
    contribute centre +/- radius; boxes centre +/- half-extent. Falls back to a default cube if extents are unknown."""
    lows = []; highs = []
    for r in rs:
        s = r.get("sdf") if isinstance(r, dict) else r
        c = np.asarray(getattr(s, "c", (0.0, 0.0, 0.0)), float)
        if hasattr(s, "r"):
            ext = np.full(3, float(s.r))
        elif hasattr(s, "h"):
            ext = np.asarray(s.h, float)
        else:
            ext = np.full(3, 1.0)
        lows.append(c - ext); highs.append(c + ext)
    if not lows:
        return np.full(3, -3.0), np.full(3, 3.0)
    lo = np.min(lows, axis=0) - margin; hi = np.max(highs, axis=0) + margin
    return lo, hi


def render_scene(objects, camera, width=256, height=256, post=None, sun="bright", sky="clear",
                 glass_tint=(0.75, 0.9, 0.85), ss=2, ground=True, dither=0.004, adaptive=True, stats=None,
                 fog=None, fog_density=0.12, fog_color=(0.80, 0.85, 0.92), fog_max_dist=14.0, region_field=None,
                 bake=None, relax=1.0, lighting=None, sun_scale=1.0, ground_color=None, backdrop=None):
    """Render a realized scene in a SINGLE pass over the UNION SDF, so objects cast shadows and ambient occlusion on
    EACH OTHER. Per-pixel material/colour comes from the nearest object (the material-id buffer); reflective
    materials pick up a sky reflection; GLASS is see-through. Optional `post` is a holographic_postfx.PostChain.

    ANTI-ALIASING -- two modes for `ss`>1:
      * ADAPTIVE (default): render the base grid once (1 ray/pixel), find the pixels where the image BENDS
        (material/silhouette/depth/luminance edges -- the 2-D analog of holographic_adaptive_cache's curvature rule),
        and supersample ss^2 rays ONLY there. Interior flat pixels keep their single sample. AA cost then scales with
        EDGE LENGTH (~perimeter), not pixel count (~area) -- typically a small fraction of brute SSAA's rays for the
        same edge quality. Pass a dict as `stats` to read back rays / edge fraction / brute_rays.
      * BRUTE (`adaptive=False`): render the whole frame at ss x and box-average -- ss^2 rays everywhere.
    ss=1 disables AA. `ground` adds a floor that catches shadows; `dither` breaks 8-bit gradient banding.

    HONEST kept limits: glass see-through is a single refraction-free layer; reflections sample only the sky dome;
    adaptive AA refines edges found in the BASE pass, so a feature thinner than one base pixel can be missed (the
    classic adaptive-sampling caveat) -- raise the base resolution if that bites."""
    from holographic.rendering.holographic_postfx import supersample as _ssaa
    ss = max(1, int(ss))
    rs_all = realize_scene(objects)                            # realize once; split surface vs volumetric objects
    vol_idx = [i for i, o in enumerate(objects) if isinstance(o, dict) and o.get("material") in _VOLUMETRIC]
    surf_rs = [rs_all[i] for i in range(len(objects)) if i not in vol_idx]
    ctx = _scene_setup(None, ground, sky, sun, glass_tint, rs=surf_rs, lighting=lighting, sun_scale=sun_scale,
                       ground_color=ground_color, backdrop=backdrop)
    ctx["region_field"] = region_field                          # optional: shade hit points by region material
    ctx["relax"] = relax                                        # >1 = opt-in over-relaxed marching (grazing scenes)
    if bake is not None and ctx.get("union") is not None and len(surf_rs) > 0:
        from holographic.mesh_and_geometry.holographic_sdfbake import GridSDF                   # PRECOMPUTE the SDF into a grid once, then sample
        if isinstance(bake, GridSDF):
            ctx["union"] = bake                                  # REUSE a grid baked once (amortise over many frames)
        else:
            _blo, _bhi = _scene_aabb(surf_rs, margin=1.5)       # O(1) per sample -> cost independent of #primitives;
            ctx["union"] = GridSDF.bake(ctx["union"], _blo, _bhi, bake)   # shadows/AO/reflections/normals reuse it
    if ctx is None:
        if not vol_idx:
            return np.zeros((height, width, 3))
        ctx = _scene_setup(None, True, sky, sun, glass_tint, rs=[], sun_scale=sun_scale)   # ground+sky backdrop for the volume

    eye, dirs0 = camera.ray_dirs(width, height)
    D0 = dirs0.reshape(-1, 3); O0 = np.broadcast_to(eye, D0.shape).copy()
    col0, hit0, t0, ids0 = _shade_rays(ctx, O0, D0)
    frame = col0.reshape(height, width, 3).copy()
    depth = np.where(hit0, t0, 1e30).reshape(height, width)
    rays = len(D0)
    if stats is not None:                                       # export for the incremental (dirty-region) renderer
        stats["idbuf"] = ids0.reshape(height, width); stats["ctx"] = ctx; stats["eye"] = eye
        stats["depth"] = depth

    if ss > 1 and adaptive:
        E = _edge_mask(ids0.reshape(height, width), depth, frame)
        if np.any(E):
            _, dirs_s = camera.ray_dirs(width * ss, height * ss)
            Ds = dirs_s.reshape(height * ss, width * ss, 3)
            ei, ej = np.where(E)
            O_e = np.broadcast_to(eye, (len(ei), 3)).copy()
            acc = np.zeros((len(ei), 3))
            for di in range(ss):                                # ss^2 subrays per edge pixel, averaged
                for dj in range(ss):
                    c, _, _, _ = _shade_rays(ctx, O_e, Ds[ei * ss + di, ej * ss + dj])
                    acc += c
            frame[ei, ej] = acc / (ss * ss)
            rays += len(ei) * ss * ss
        if stats is not None:
            stats["edge_fraction"] = float(E.mean()); stats["rays"] = rays
            stats["brute_rays"] = width * height * ss * ss
    elif ss > 1:                                                # brute SSAA
        _, dirs_s = camera.ray_dirs(width * ss, height * ss)
        Ds = dirs_s.reshape(-1, 3); Os = np.broadcast_to(eye, Ds.shape).copy()
        cols, hits, ts, _ = _shade_rays(ctx, Os, Ds)
        frame = _ssaa(cols.reshape(height * ss, width * ss, 3), ss)
        dd = np.where(hits, ts, 1e30).reshape(height * ss, width * ss, 1)
        depth = _ssaa(dd, ss)[..., 0]
        rays = len(Ds)
        if stats is not None:
            stats["rays"] = rays; stats["brute_rays"] = rays; stats["edge_fraction"] = 1.0
    elif stats is not None:
        stats["rays"] = rays; stats["brute_rays"] = rays; stats["edge_fraction"] = 0.0

    # composite VOLUMETRIC objects (fog / smoke / fire) over the surface frame via volume_render
    if vol_idx:
        from holographic.rendering.holographic_render import volume_render, Light
        for i in vol_idx:
            s = rs_all[i]["sdf"]
            center = s.c if hasattr(s, "c") else np.zeros(3)
            radius = float(s.r) if isinstance(s, _SphereSDF) else (float(np.max(s.h)) if isinstance(s, _BoxSDF) else 0.7)
            mat = objects[i].get("material")
            light_i = Light("directional", direction=tuple(-ctx["sun_dir"]))
            if mat == "cloud":
                # a convincing cloud: multi-lobe fBm density + physically-motivated lighting (self-shadow,
                # forward-scatter phase, powder, multi-scatter -- see volume_render / cloud_field)
                field = cloud_field(center, radius * 1.3, density=3.4, seed=i)
                # tight, cloud-SHAPED bounds -- see the identical note in UnifiedMind.make_cloud
                bounds = (center + radius * np.array([-2.0, -1.1, -2.0]), center + radius * np.array([2.0, 1.4, 2.0]))
                col = COLORS.get(objects[i].get("color"))
                alb = tuple(col) if col else (1.0, 0.99, 0.97)
                vrgb, valpha = volume_render(field, camera, bounds, width=width, height=height, steps=128,
                                             mode="smoke", sigma=6.5, albedo=alb, lights=[light_i],
                                             self_shadow=True, shadow_steps=16, shadow_sigma=9.0,
                                             ambient=(0.40, 0.52, 0.74),
                                             phase_g=0.75, powder=True, multi_scatter=4)
            else:
                mode = "fire" if mat == "fire" else "smoke"
                col = COLORS.get(objects[i].get("color"))
                alb = tuple(col) if (col and mode == "smoke") else (0.92, 0.93, 0.96)
                field = volumetric_field(center, radius * 1.25, density=1.6 if mode == "smoke" else 2.2,
                                         turbulence=0.7, seed=i)
                bounds = (center - radius * 1.8, center + radius * 1.8)
                vrgb, valpha = volume_render(field, camera, bounds, width=width, height=height, steps=80,
                                             mode=mode, sigma=14.0, albedo=alb, lights=[light_i])
            a = valpha[..., None]
            frame = frame * (1.0 - a) + vrgb * a               # over-composite the medium

    frame = np.clip(frame, 0, 1)
    if fog is not None:                                          # closed-form HOLOGRAPHIC fog along each camera ray
        from holographic.misc.holographic_volint import render_fog               #  no marching  -- the volint field, in the real path
        dd = np.where(np.isfinite(depth) & (depth < 1e29), depth, fog_max_dist)
        frame = render_fog(camera, width, height, fog, density_scale=fog_density, fog_color=fog_color,
                           max_dist=fog_max_dist, background=frame, depth=dd)
    if post is not None:
        frame = post.apply(frame, depth=depth)
    if dither > 0:
        rng = np.random.default_rng(0)
        frame = np.clip(frame + (rng.random(frame.shape) - 0.5) * dither, 0, 1)
    return frame


class SceneRenderer:
    """A render cache that re-renders only what CHANGED. The first render() is a full adaptive render; a subsequent
    material/colour edit re-shades ONLY the changed object's pixels -- found via the cached material-id buffer -- not
    the whole frame. Geometry is untouched on a material edit, so there is no re-tracing: the saving is the ratio of
    the object's screen area to the whole frame. This is the "only reprocess what changed when an object/material/
    light changes" idea, on the same union substrate.

    HONEST scope: a material/colour edit updates the object's DIRECTLY-VISIBLE pixels exactly; its appearance seen
    THROUGH glass or in reflections is not incrementally refreshed (call render() for a full refresh). A geometry/
    position edit moves the silhouette, so it falls back to a full render (the id buffer is stale)."""

    def __init__(self, camera, width=256, height=256, sun="bright", sky="clear", ground=True, ss=2,
                 glass_tint=(0.75, 0.9, 0.85)):
        self.cam = camera; self.w = width; self.h = height
        self.sun = sun; self.sky = sky; self.ground = ground; self.ss = ss; self.glass_tint = glass_tint
        self.objects = None; self.frame = None; self.idbuf = None

    def render(self, objects):
        """Full adaptive render; caches the frame and the base material-id buffer for later dirty re-renders."""
        st = {}
        self.objects = [dict(o) for o in objects]
        self.frame = render_scene(objects, self.cam, self.w, self.h, sun=self.sun, sky=self.sky,
                                  ground=self.ground, ss=self.ss, dither=0.0, adaptive=True, stats=st)
        self.idbuf = st["idbuf"]
        return self.frame

    def set_attr(self, obj_index, field, value):
        """Edit one object's colour/material and re-shade ONLY its visible pixels. Returns (frame, stats) where stats
        reports how many pixels (rays) were re-rendered vs the full-frame count."""
        if self.frame is None:
            raise RuntimeError("call render() before set_attr()")
        self.objects[obj_index][field] = value
        ctx = _scene_setup(self.objects, self.ground, self.sky, self.sun, self.glass_tint)
        mask = self.idbuf == obj_index                          # the changed object's directly-visible pixels
        ei, ej = np.where(mask)
        if len(ei):
            eye, dirs = self.cam.ray_dirs(self.w, self.h)
            D = dirs.reshape(self.h, self.w, 3)[ei, ej]
            c, _, _, _ = _shade_rays(ctx, np.broadcast_to(eye, (len(ei), 3)).copy(), D)
            self.frame = self.frame.copy()
            self.frame[ei, ej] = np.clip(c, 0, 1)
        return self.frame, {"rerendered_pixels": int(len(ei)), "full_pixels": self.w * self.h}


def _selftest_render():
    """Quick check that the adaptive renderer matches brute SSAA closely at far fewer rays."""
    from holographic.rendering.holographic_render import Camera
    objs = parse_description("a big red ball beside a blue box")["objects"]
    cam = Camera(eye=(0.4, 1.6, 5.0), target=(0, 0, 0), fov_deg=42.0)
    sa, sb = {}, {}
    fa = render_scene(objs, cam, width=140, height=140, ss=3, adaptive=True, dither=0.0, stats=sa)
    fb = render_scene(objs, cam, width=140, height=140, ss=3, adaptive=False, dither=0.0, stats=sb)
    assert sa["rays"] < sb["rays"] * 0.5, (sa["rays"], sb["rays"])
    assert np.abs(fa - fb).mean() < 0.002
    print("render selftest ok: adaptive AA matches brute (mean diff %.5f) at %.1fx fewer rays"
          % (float(np.abs(fa - fb).mean()), sb["rays"] / sa["rays"]))


def _pbr_props(o):
    """Resolve an object's (albedo, metallic, roughness, emission, ior) for the path-traced PBR renderer. ior>1
    marks a dielectric (glass) the path tracer will refract through; 0 = opaque."""
    mat = o.get("material")
    metallic, roughness, em = PBR_PARAMS.get(mat, PBR_PARAMS[None])
    col = COLORS.get(o.get("color"))
    if metallic >= 0.5 and col is None:
        albedo = np.array(METAL_TINT.get(mat, (0.95, 0.95, 0.95)), float)
    else:
        albedo = np.array(col if col is not None else (0.8, 0.8, 0.8), float)
    emission = albedo * em if em > 0 else np.zeros(3)
    ior = 1.5 if mat == "glass" else 0.0                       # glass -> refractive dielectric
    if mat == "glass" and col is None:
        albedo = np.array([0.93, 0.97, 0.95])                  # faint tint for clear glass transmission
    return albedo, float(metallic), float(roughness), emission, float(ior)


def _scene_material_fn(props, union):
    """A material(P) callback for the path tracer: at each hit point, look up the nearest object's PBR props."""
    albedos = np.array([p[0] for p in props]); mets = np.array([p[1] for p in props])
    roughs = np.array([p[2] for p in props]); emis = np.array([p[3] for p in props])
    iors = np.array([p[4] if len(p) > 4 else 0.0 for p in props])

    def fn(P):
        ids = union.ids(P)
        return albedos[ids].copy(), mets[ids], roughs[ids], emis[ids].copy(), iors[ids]
    return fn


def render_scene_pbr(objects, camera, width=200, height=200, spp=24, max_bounce=4, post=None,
                     ground=True, sky="clear", sun="bright", tonemap=True, dither=0.004, env=None,
                     adaptive_spp=0, noise_pct=70, stats=None, lighting=None):
    """HYPERREAL render path: route a described scene through the engine's Monte-Carlo PATH TRACER
    (holographic_pathtrace) with REAL per-object Cook-Torrance/GGX materials (holographic_brdf) -- true multi-bounce
    global illumination, soft shadows, colour bleeding, glossy GGX highlights, EMISSIVE objects that light the scene,
    and REFRACTIVE GLASS (dielectric reflect/refract by Fresnel, light bends through). Per-object (albedo, metallic,
    roughness, emission, ior) come from the material grounding. This is the slow, offline, method-parity renderer.

    ADAPTIVE SAMPLING (`adaptive_spp`>0): render `spp` samples everywhere, measure per-pixel variance, then spend the
    extra `adaptive_spp` samples ONLY on the noisiest pixels (above the `noise_pct` percentile). Most pixels converge
    fast; fireflies/penumbrae/caustics do not -- so this puts compute where the noise is, the sample-domain analog of
    the edge-adaptive AA. Pass a dict as `stats` to read back the sample budget vs the uniform-equivalent.

    HONEST kept limits: glass is a SMOOTH dielectric (no rough/frosted glass, no wavelength dispersion; transmission
    is albedo-tinted, not true Beer-Lambert over path length); no next-event estimation (small bright emitters are
    noisy -- raise spp); single-scatter GGX energy loss at high roughness is inherited. Noise falls as 1/sqrt(spp)."""
    from holographic.rendering.holographic_pathtrace import path_trace
    from holographic.rendering.holographic_raymarch import sky_dome
    ctx = _scene_setup(objects, ground, sky, sun, (0.75, 0.9, 0.85), lighting=lighting)  # preset sets the sun_dir the sky dome lights from
    if ctx is None:
        return np.zeros((height, width, 3))
    union = ctx["union"]; sdfs = ctx["sdfs"]
    props = [_pbr_props(o) for o in objects]
    if len(sdfs) > len(objects):                               # the ground plane _scene_setup appended
        props.append((np.array([0.62, 0.63, 0.66]), 0.0, 0.9, np.zeros(3), 0.0))
    matfn = _scene_material_fn(props, union)
    sun_dir = ctx["sun_dir"]
    skyfn = (lambda D: sky_dome(D, sun_dir=tuple(sun_dir), env=env))

    if adaptive_spp > 0:                                       # variance-driven adaptive sampling
        img, var = path_trace(union, camera, width=width, height=height, spp=spp, max_bounce=max_bounce,
                              material=matfn, sky=skyfn, return_variance=True)
        noise = np.sqrt(var)
        thr = float(np.percentile(noise, noise_pct))
        mask = (noise > thr).reshape(-1)
        if mask.any():
            extra = path_trace(union, camera, width=width, height=height, spp=adaptive_spp, max_bounce=max_bounce,
                               material=matfn, sky=skyfn, active=mask, seed=9973)
            m2 = mask.reshape(height, width)
            img = img.copy()
            img[m2] = (img[m2] * spp + extra[m2] * adaptive_spp) / (spp + adaptive_spp)   # sample-count weighted
        if stats is not None:
            stats["base_samples"] = spp * width * height
            stats["extra_samples"] = adaptive_spp * int(mask.sum())
            stats["total_samples"] = stats["base_samples"] + stats["extra_samples"]
            stats["uniform_equiv_samples"] = (spp + adaptive_spp) * width * height
            stats["noisy_fraction"] = float(mask.mean())
        hdr = img
    else:
        hdr = path_trace(union, camera, width=width, height=height, spp=spp, max_bounce=max_bounce,
                         material=matfn, sky=skyfn)
    frame = hdr
    if tonemap:                                                # ACES filmic tonemap of the HDR result
        from holographic.rendering.holographic_postfx import aces
        frame = aces(frame)
    if post is not None:
        frame = post.apply(np.clip(frame, 0, 1))
    frame = np.clip(frame, 0, 1)
    if dither > 0:
        rng = np.random.default_rng(0)
        frame = np.clip(frame + (rng.random(frame.shape) - 0.5) * dither, 0, 1)
    return frame


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
_CONTROLLABLE = {
    "size": dict(kind="slider", min=0.3, max=2.5, value=1.0, step=0.05, label="size"),
    "radius": dict(kind="slider", min=0.2, max=2.0, value=0.7, step=0.05, label="radius"),
    "metallic": dict(kind="slider", min=0.0, max=1.0, value=0.5, step=0.05, label="metallic"),
    "roughness": dict(kind="slider", min=0.0, max=1.0, value=0.3, step=0.05, label="roughness"),
    "reflect": dict(kind="slider", min=0.0, max=1.0, value=0.3, step=0.05, label="reflectivity"),
    "ior": dict(kind="slider", min=1.0, max=2.4, value=1.5, step=0.02, label="index of refraction"),
    "brightness": dict(kind="slider", min=0.0, max=4.0, value=2.5, step=0.1, label="sun brightness"),
    "material": dict(kind="select", options=sorted(set(MATERIALS.values())), value="matte", label="material"),
    "color": dict(kind="select", options=sorted(COLORS), value="gray", label="color"),
}


def control_spec(command):
    """Turn a control phrase into a list of UI control descriptors (sliders / selects / toggles) a front-end can
    render directly. e.g. 'control the ball size and metallic-ness' -> a size slider and a metallic slider, each
    scoped to the matched target. The engine produces the SPEC; the browser muscle draws the widgets. Honest: this
    matches controllable KEYWORDS in the command against a fixed table -- it does not invent new controls."""
    cmd = command.lower()
    words = cmd.replace("-", " ").split()
    # target: an object phrase ('the ball', 'all materials') or the scene
    target = "scene"
    for w in SHAPES:
        if w in words:
            target = w; break
    if "all" in words:
        target = "all " + target if target != "scene" else "all"

    controls = []
    seen = set()
    # synonym hooks into the controllable table
    hooks = {"big": "size", "bigger": "size", "size": "size", "scale": "size", "radius": "radius",
             "metal": "metallic", "metallic": "metallic", "rough": "roughness", "roughness": "roughness",
             "reflect": "reflect", "reflective": "reflect", "mirror": "reflect", "shiny": "reflect",
             "glass": "ior", "ior": "ior", "refract": "ior", "bright": "brightness", "brightness": "brightness",
             "sun": "brightness", "material": "material", "colour": "color", "color": "color"}
    for w in words:
        key = hooks.get(w)
        if key and key not in seen:
            spec = dict(_CONTROLLABLE[key]); spec["target"] = target; spec["param"] = key
            controls.append(spec); seen.add(key)
    if not controls:                                          # nothing matched -> offer the generic transform knobs
        for key in ("size", "material", "color"):
            spec = dict(_CONTROLLABLE[key]); spec["target"] = target; spec["param"] = key
            controls.append(spec)
    return {"command": command, "target": target, "controls": controls}


def _selftest():
    from holographic.misc.holographic_unified import UnifiedMind
    mind = UnifiedMind(dim=1024, seed=0)
    desc = ("A red ball sitting inside of a box with a glass material, with a metallic elongated box leaning on the "
            "glass box diagonally. The sun is bright in the sky, which is partly cloudy")
    scene = parse_description(desc)
    objs = scene["objects"]
    assert len(objs) == 3, [o for o in objs]
    assert objs[0] == {"shape": "sphere", "color": "red", "material": None, "size": None, "relation": ("inside", 1)}
    assert objs[1]["shape"] == "box" and objs[1]["material"] == "glass"
    assert objs[2]["material"] == "metal" and objs[2]["size"] == "elongated"
    assert objs[2]["relation"][0] == "leaning" and objs[2]["relation"][1] == 1   # leans on the glass box (index 1)
    assert scene["environment"]["sun"] == "bright" and scene["environment"]["sky"] == "partly"

    # bidirectional: encode each object, decode its attributes back
    recs = encode_objects(objs, mind)
    d0 = decode_object(recs[0], mind)
    assert d0["shape"] == "sphere" and d0["color"] == "red", d0

    # query the BUNDLED scene by slot (unbind + cleanup recovers attributes through the superposition crosstalk)
    sv, recs, roles = encode_scene(objs, mind)
    q1 = query_scene(sv, roles, mind, 1)
    assert q1["shape"] == "box" and q1["material"] == "glass", q1

    # batch edit: make all materials reflective
    mirrored = batch_set(objs, "material", "mirror")
    assert all(o["material"] == "mirror" for o in mirrored)
    assert find_objects(objs, material="glass") == [1]

    # control spec from a command
    spec = control_spec("control the ball size and how metallic it is")
    keys = [c["param"] for c in spec["controls"]]
    assert "size" in keys and "metallic" in keys and spec["target"] == "ball"

    # PHOTO-MATCHED AUTHORING (default-off, additive): an exact (r,g,b) colour is used verbatim; an explicit
    # per-axis 'stretch' overrides the size-word stretch and turns a 'sphere' into an ELLIPSOID -- what building a
    # scene from a photo needs (colours picked from the image, a plate/leaf/lemon that is a squashed primitive). A
    # named colour with no stretch is BYTE-IDENTICAL to before (an exact sphere).
    _plain = realize_scene([{"shape": "sphere", "color": "red", "material": None, "size": None, "relation": None}])[0]
    assert isinstance(_plain["sdf"], _SphereSDF)                              # default path unchanged
    _rgb = realize_scene([{"shape": "sphere", "color": (0.6, 0.5, 0.2), "material": None, "size": None,
                           "relation": None, "stretch": (1.2, 0.8, 1.0)}])[0]
    assert isinstance(_rgb["sdf"], _EllipsoidSDF) and _rgb["color"] == (0.6, 0.5, 0.2)   # exact rgb + ellipsoid
    _e = _EllipsoidSDF((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))                      # on-surface point reads ~0
    assert abs(float(_e.eval(np.array([[2.0, 0.0, 0.0]]))[0])) < 0.05

    print("semantic selftest ok: parsed 3 objects + environment from the example sentence; objects encode/decode "
          "bidirectionally; the bundled scene is queryable by slot; batch edit + control spec work.")


if __name__ == "__main__":
    _selftest()
