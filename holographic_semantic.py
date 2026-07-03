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
    "ball": "sphere", "sphere": "sphere", "orb": "sphere", "globe": "sphere",
    "box": "box", "cube": "box", "block": "box", "crate": "box", "brick": "box",
}
COLORS = {                                                   # word -> linear rgb
    "red": (0.85, 0.18, 0.18), "green": (0.20, 0.70, 0.25), "blue": (0.20, 0.40, 0.85),
    "yellow": (0.90, 0.80, 0.20), "orange": (0.95, 0.55, 0.15), "purple": (0.60, 0.25, 0.70),
    "pink": (0.95, 0.55, 0.70), "cyan": (0.25, 0.80, 0.85), "white": (0.90, 0.90, 0.90),
    "black": (0.08, 0.08, 0.08), "gray": (0.50, 0.50, 0.50), "grey": (0.50, 0.50, 0.50),
}
MATERIALS = {                                                # word -> canonical material
    "glass": "glass", "glassy": "glass", "transparent": "glass", "clear": "glass",
    "metal": "metal", "metallic": "metal", "chrome": "metal", "steel": "metal", "iron": "metal",
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
    "wax": "wax", "waxy": "wax", "jade": "wax", "marble": "wax", "skin": "wax", "candle": "wax",   # subsurface (SSS)
    "translucent": "translucent", "frosted": "translucent", "milky": "translucent",  # see-through diffuse
}
# materials that are VOLUMETRIC (participating media), rendered with volume_render, not as surfaces
_VOLUMETRIC = {"fog", "smoke", "fire"}
SIZES = {                                                    # word -> ('uniform', s) or ('stretch', (sx,sy,sz))
    "big": ("uniform", 1.5), "large": ("uniform", 1.5), "huge": ("uniform", 2.0),
    "small": ("uniform", 0.62), "tiny": ("uniform", 0.42), "little": ("uniform", 0.62),
    "elongated": ("stretch", (1.0, 1.0, 2.2)), "long": ("stretch", (1.0, 1.0, 2.2)),
    "flat": ("stretch", (1.6, 0.4, 1.6)), "tall": ("stretch", (1.0, 2.0, 1.0)),
    "wide": ("stretch", (2.0, 1.0, 1.0)),
}
RELATIONS = {                                                # word -> canonical relation
    "inside": "inside", "within": "inside", "in": "inside",
    "on": "on", "atop": "on", "above": "above", "over": "above", "under": "under", "below": "under",
    "leaning": "leaning", "beside": "beside", "next": "beside", "near": "beside",
    "diagonal": "diagonal", "diagonally": "diagonal",
}
ENVIRO_SKY = {"clear": "clear", "cloudy": "cloudy", "overcast": "cloudy", "partly": "partly"}
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
            from holographic_text import learn_word_vectors
            self.enc = learn_word_vectors(corpus or DEFAULT_GROUNDING_CORPUS, dim=dim, window=window, seed=seed)

    def resolve(self, word, field):
        """Map `word` to the nearest known `field` value (table first, then learned), or None."""
        if word in self.table and self.table[word][0] == field:
            return self.table[word][1]
        if self.enc is None:
            return None
        from holographic_ai import cosine
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
        from holographic_ai import cosine
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
    env = {"sun": None, "sky": None}
    obj_tokens = []
    for clause in _split_clauses(text):
        toks = clause.split()
        has_obj = any(t in SHAPES for t in toks) or any(MATERIALS.get(t) in _VOLUMETRIC for t in toks)
        env_subject = any(t in ("sun", "sky", "sunny") for t in toks)
        cloud_words = any(t in ("cloud", "clouds", "cloudy", "overcast") for t in toks)
        is_env = env_subject or (cloud_words and not has_obj)   # 'cloud' alone is env only with no object content
        if is_env:
            if "sun" in toks or "sunny" in toks:
                env["sun"] = "bright" if ("bright" in toks or "sunny" in toks) else "soft"
            if "partly" in toks and ("cloudy" in toks or "cloud" in toks or "clouds" in toks):
                env["sky"] = "partly"
            elif "cloudy" in toks or "overcast" in toks:
                env["sky"] = "cloudy"
            elif "clear" in toks or "blue" in toks:
                env["sky"] = "clear"
        else:
            obj_tokens.extend(toks)

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
    from holographic_ai import bind, bundle, derived_atom
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
    from holographic_ai import unbind
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


class _BoxSDF:
    def __init__(self, c, half):
        self.c = np.asarray(c, float); self.h = np.asarray(half, float)

    def eval(self, P):
        q = np.abs(np.asarray(P, float) - self.c) - self.h
        return (np.linalg.norm(np.maximum(q, 0.0), axis=1)
                + np.minimum(np.max(q, axis=1), 0.0))


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
        else:                                                 # leaning / beside / diagonal
            off = np.array([2.1, 0.5, 0.3]) if relword in ("leaning", "diagonal") else np.array([2.4, 0.0, 0.0])
            pos[i] = pos[ref] + off

    out = []
    for i, o in enumerate(objects):
        uni, stretch = _base_size(o)
        r = base_radius * uni * scale[i]
        col = COLORS.get(o["color"], (0.8, 0.8, 0.8))
        mat = MATERIAL_RENDER.get(o["material"], MATERIAL_RENDER[None])
        if o["shape"] == "sphere":
            sdf = _SphereSDF(pos[i], r)
        else:
            half = np.array(stretch) * r
            sdf = _BoxSDF(pos[i], half)
        out.append({"sdf": sdf, "color": col, "material": mat, "mat_name": o["material"], "name": _obj_name(o)})
    return out


def _obj_name(o):
    bits = [o[k] for k in ("color", "size", "material") if o[k]]
    return " ".join(bits + [o["shape"]])


class _PlaneSDF:
    """A horizontal ground plane at height y0 -- gives objects something to stand on and catch their shadows, which
    is most of what makes a render read as 'real' rather than floating shapes."""

    def __init__(self, y0):
        self.y0 = float(y0)

    def eval(self, P):
        return np.asarray(P, float)[:, 1] - self.y0


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
    from holographic_raymarch import sphere_trace, sdf_normal, ambient_occlusion, soft_shadow, sky_dome
    rc = sky_dome(D, sun_dir=tuple(sun_dir))
    hm, tm, Pmh = sphere_trace(union, O, D)
    if np.any(hm):
        Nm = sdf_normal(union, Pmh[hm]); idm = union.ids(Pmh[hm])
        lamm = np.clip((Nm * sun_dir).sum(1), 0, 1)
        shm = soft_shadow(union, Pmh[hm] + Nm * 3e-3, sun_dir); aom = ambient_occlusion(union, Pmh[hm], Nm)
        rc[hm] = np.clip(colors[idm] * (amb * aom + lamm * shm * sun_i)[:, None], 0, 1)
    return rc


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


def _scene_setup(objects, ground, sky, sun, glass_tint, rs=None):
    """Build the immutable per-scene context (SDFs, colours, materials, lighting) once; reused by every ray batch so
    the adaptive refinement does not rebuild it. Pass `rs` to use pre-realized renderables (so volumetric objects can
    be split out and the layout computed once)."""
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
        rs = list(rs) + [{"sdf": _PlaneSDF(y0), "color": (0.62, 0.63, 0.66), "material": MATERIAL_RENDER["matte"],
                          "mat_name": "matte", "name": "ground"}]
    if not rs:
        return None
    sdfs = [r["sdf"] for r in rs]
    sun_dir = np.array([-0.45, 0.8, -0.35]); sun_dir = sun_dir / np.linalg.norm(sun_dir)
    return {"sdfs": sdfs, "union": _UnionSDF(sdfs),
            "colors": np.array([r["color"] for r in rs]),
            "refl": np.array([_reflectivity(r["mat_name"]) for r in rs]),
            "rough": np.array([_roughness(r["mat_name"]) for r in rs]),                # glossy reflection lobe width
            "is_glass": np.array([r["mat_name"] == "glass" for r in rs]),
            "is_sss": np.array([r["mat_name"] == "wax" for r in rs]),               # subsurface scattering
            "is_translucent": np.array([r["mat_name"] == "translucent" for r in rs]),   # diffuse see-through
            "sun_dir": sun_dir, "glass_tint": np.array(glass_tint),
            "amb": 0.34 if sky in ("cloudy", "partly") else 0.24,
            "sun_i": 1.0 if sun == "bright" else 0.7}


def _shade_rays(ctx, O, D):
    """Shade an ARBITRARY batch of rays against the scene (used for the base grid AND for edge subrays). Returns
    (col (M,3), hit (M,), t (M,), ids (M,) with -1 = background)."""
    from holographic_raymarch import sphere_trace, sdf_normal, ambient_occlusion, soft_shadow, sky_dome
    union = ctx["union"]; sdfs = ctx["sdfs"]; colors = ctx["colors"]; refl = ctx["refl"]; is_glass = ctx["is_glass"]
    sun_dir = ctx["sun_dir"]; amb = ctx["amb"]; sun_i = ctx["sun_i"]
    hit, t, P = sphere_trace(union, O, D, relax=ctx.get("relax", 1.0))
    col = sky_dome(D, sun_dir=tuple(sun_dir))
    ids_full = -np.ones(len(D), int)
    if np.any(hit):
        ids = union.ids(P[hit]); ids_full[hit] = ids
        N = sdf_normal(union, P[hit])
        lam = np.clip((N * sun_dir).sum(1), 0, 1)
        sh = soft_shadow(union, P[hit] + N * 3e-3, sun_dir)     # marches the WHOLE union -> inter-object shadows
        ao = ambient_occlusion(union, P[hit], N)
        shade = amb * ao + lam * sh * sun_i
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
        shaded = albedo * shade[:, None]
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
            from holographic_raymarch import subsurface
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
                sh2 = soft_shadow(union, P2[h2] + N2 * 3e-3, sun_dir)
                ao2 = ambient_occlusion(un2, P2[h2], N2)
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
                    shT = soft_shadow(union, PT[hT] + NT * 3e-3, sun_dir); aoT = ambient_occlusion(unT, PT[hT], NT)
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
                 bake=None, relax=1.0):
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
    from holographic_postfx import supersample as _ssaa
    ss = max(1, int(ss))
    rs_all = realize_scene(objects)                            # realize once; split surface vs volumetric objects
    vol_idx = [i for i, o in enumerate(objects) if isinstance(o, dict) and o.get("material") in _VOLUMETRIC]
    surf_rs = [rs_all[i] for i in range(len(objects)) if i not in vol_idx]
    ctx = _scene_setup(None, ground, sky, sun, glass_tint, rs=surf_rs)
    ctx["region_field"] = region_field                          # optional: shade hit points by region material
    ctx["relax"] = relax                                        # >1 = opt-in over-relaxed marching (grazing scenes)
    if bake is not None and ctx.get("union") is not None and len(surf_rs) > 0:
        from holographic_sdfbake import GridSDF                   # PRECOMPUTE the SDF into a grid once, then sample
        if isinstance(bake, GridSDF):
            ctx["union"] = bake                                  # REUSE a grid baked once (amortise over many frames)
        else:
            _blo, _bhi = _scene_aabb(surf_rs, margin=1.5)       # O(1) per sample -> cost independent of #primitives;
            ctx["union"] = GridSDF.bake(ctx["union"], _blo, _bhi, bake)   # shadows/AO/reflections/normals reuse it
    if ctx is None:
        if not vol_idx:
            return np.zeros((height, width, 3))
        ctx = _scene_setup(None, True, sky, sun, glass_tint, rs=[])   # ground+sky backdrop for the volume

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
        from holographic_render import volume_render, Light
        for i in vol_idx:
            s = rs_all[i]["sdf"]
            center = s.c if hasattr(s, "c") else np.zeros(3)
            radius = float(s.r) if isinstance(s, _SphereSDF) else (float(np.max(s.h)) if isinstance(s, _BoxSDF) else 0.7)
            mat = objects[i].get("material")
            mode = "fire" if mat == "fire" else "smoke"
            col = COLORS.get(objects[i].get("color"))
            alb = tuple(col) if (col and mode == "smoke") else (0.92, 0.93, 0.96)
            field = volumetric_field(center, radius * 1.25, density=1.6 if mode == "smoke" else 2.2,
                                     turbulence=0.7, seed=i)
            bounds = (center - radius * 1.8, center + radius * 1.8)
            vrgb, valpha = volume_render(field, camera, bounds, width=width, height=height, steps=80,
                                         mode=mode, sigma=14.0, albedo=alb,
                                         lights=[Light("directional", direction=tuple(-ctx["sun_dir"]))])
            a = valpha[..., None]
            frame = frame * (1.0 - a) + vrgb * a               # over-composite the medium

    frame = np.clip(frame, 0, 1)
    if fog is not None:                                          # closed-form HOLOGRAPHIC fog along each camera ray
        from holographic_volint import render_fog               # (no marching) -- the volint field, in the real path
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
    from holographic_render import Camera
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
                     adaptive_spp=0, noise_pct=70, stats=None):
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
    from holographic_pathtrace import path_trace
    from holographic_raymarch import sky_dome
    ctx = _scene_setup(objects, ground, sky, sun, (0.75, 0.9, 0.85))
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
        from holographic_postfx import aces
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
    from holographic_unified import UnifiedMind
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
    assert scene["environment"] == {"sun": "bright", "sky": "partly"}

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

    print("semantic selftest ok: parsed 3 objects + environment from the example sentence; objects encode/decode "
          "bidirectionally; the bundled scene is queryable by slot; batch edit + control spec work.")


if __name__ == "__main__":
    _selftest()
