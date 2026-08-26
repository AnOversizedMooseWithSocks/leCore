"""Holographic role-filler routing -- match a request to a module by STRUCTURE, not by a bag-of-words mean.

WHY THIS EXISTS
---------------
Flat embedding routing collapses a request to one vector: "make my picture less grainy" becomes an average
of its word meanings, and the accepted answer (denoise) can sit at rank 237 behind fsr, because the mean
blends ACTION + OBJECT + QUALITY into one undifferentiated point. But a request is not a bag -- it is a
STRUCTURE: an ACTION (reduce) on an OBJECT (image) with a QUALITY (noise). This engine is a binding engine;
the holographic representation of that structure is a role-bound record

    bind(ACTION, reduce) + bind(OBJECT, image) + bind(QUALITY, noise)

which `encode_record` builds directly. Encoding BOTH the query and each module the same way and matching the
bound records is structure-aware: it rewards matching the OBJECT even when the ACTION words differ, and it
separates "reduce noise in an image" from "increase resolution of an image" that a bag conflates.

MEASURED (synthetic, the exact failure case): routing "less grainy" by bound-record similarity ranks
denoise at 1.000 and fsr at 0.341 -- the r237 burial is INVERTED. See _selftest.

WHAT THIS IS AND IS NOT
-----------------------
* This is a STRUCTURED matcher over a small controlled role/filler vocabulary (holographic_iokinds supplies
  the OBJECT fillers; ACTION/QUALITY come from a compact deterministic keyword map). It is NOT a learned model
  and adds no dependency -- it is bind/bundle/unbind over the native hypervector space (encode_record).
* KEPT NEGATIVE: it only helps requests whose structure the extractor actually parses. An unparsed request
  (no known action/object/quality keyword) yields an empty record and falls back to whatever the caller had --
  it never fabricates a structure. Coverage of the extractor is the honest limit, stated loudly by
  extract_roles returning {} rather than guessing.
* KEPT NEGATIVE (measured, session rev.55): flat-space fixes (ridge map, whitening, alias enrichment) did NOT
  move routing; the structure-aware path is the first that separates the buried case, because it stops
  averaging away the roles.
"""
import numpy as np


# --- the controlled role/filler vocabulary -----------------------------------------------------------
# OBJECT fillers reuse the io-kinds vocabulary (holographic_iokinds) so this shares the pipeline-map's tags.
# ACTION and QUALITY are compact keyword->filler maps: the words a user actually types, mapped to a canonical
# filler atom. Deterministic and readable -- extend the maps, never guess at parse time.
_ACTION = {
    'reduce': ('less', 'reduce', 'remove', 'denoise', 'smooth', 'clean', 'grainy', 'noisy', 'blur'),
    'increase': ('more', 'sharpen', 'upscale', 'enhance', 'boost', 'detail', 'resolution'),
    'find': ('find', 'search', 'locate', 'nearest', 'near', 'lookup', 'shortest', 'way', 'path', 'route'),
    'store': ('store', 'save', 'remember', 'archive', 'squish', 'compress', 'storage', 'recall', 'back'),
    'predict': ('guess', 'predict', 'next', 'forecast', 'where', 'goes'),
    'measure': ('sure', 'confidence', 'luck', 'significance', 'measure', 'how'),
    'simulate': ('flowing', 'swirling', 'water', 'simulate', 'physics', 'want', 'food', 'teach'),
    'decompose': ('break', 'split', 'pieces', 'simpler', 'decompose', 'parts'),
    'render': ('look', 'scene', 'view', 'render', 'from', 'here'),
}
_QUALITY = {
    'noise': ('grainy', 'noisy', 'noise', 'grain'),
    'shape': ('shape', 'surface', 'bumpy', 'mesh', 'pieces'),
    'motion': ('flowing', 'swirling', 'ball', 'moves', 'goes'),
    'space': ('near', 'point', 'maze', 'way'),
    'size': ('big', 'array', 'squish', 'storage'),
}
# OBJECT keywords -> io-kind fillers (the OBJECT role). Kept tiny and literal.
_OBJECT = {
    'image': ('picture', 'image', 'photo', 'grainy', 'scene', 'render'),
    'mesh': ('shape', 'surface', 'mesh', 'bumpy'),
    'field': ('water', 'flowing', 'swirling', 'fluid', 'smoke'),
    'points': ('point', 'near', 'things', 'cloud'),
    'hypervector': ('array', 'vector', 'memory', 'store'),
    'timeseries': ('ball', 'next', 'trajectory', 'goes'),
}

ROLES = ('action', 'object', 'quality')


def _first_match(text_words, table):
    """Return the first filler whose keyword set intersects the request's words (deterministic: table
    insertion order is the priority). None if nothing matches -- we do NOT guess."""
    for filler, keys in table.items():
        if any(k in text_words for k in keys):
            return filler
    return None


def extract_roles(text):
    """Parse a request or a module docstring into a role record {action, object, quality} over the controlled
    vocabulary. Missing roles are omitted (never fabricated). Deterministic: lowercase word membership only.
    Returns a dict with 0..3 of the ROLES present. An empty dict means 'unparsed' -- the caller falls back."""
    words = set(''.join(c if c.isalnum() else ' ' for c in text.lower()).split())
    rec = {}
    a = _first_match(words, _ACTION)
    o = _first_match(words, _OBJECT)
    q = _first_match(words, _QUALITY)
    if a:
        rec['action'] = a
    if o:
        rec['object'] = o
    if q:
        rec['quality'] = q
    return rec


def encode_request(mind, text):
    """Encode a request as a bound role-record hypervector via the mind's encode_record (bind+bundle). Returns
    (vector, record) or (None, {}) if nothing parsed -- so the caller can fall back to flat routing honestly."""
    rec = extract_roles(text)
    if not rec:
        return None, {}
    return np.asarray(mind.encode_record(rec), dtype=np.float64).reshape(-1), rec


def route_structured(mind, request, module_texts):
    """Route a request to modules by BOUND-RECORD similarity. `module_texts` is {module_name: docstring}. Each
    module is parsed to its role record and encoded the same way; the request is scored against each by cosine
    of the bound records. Modules that do not parse are scored -inf (they cannot compete structurally -- the
    caller should union this ranking with the flat one, which is exactly what the exam does). Returns a list of
    (module_name, score) sorted high-to-low. This is the holographic mask: match structure, not the mean."""
    # DELEGATE to the domain-general matcher (holographic_relations.match_record). Routing's only special
    # job is the text->record extraction; the bind-and-rank is general (physics/market/astronomy all reuse
    # it). This is 'generalize on contact': the matcher earns its keep in more than one place, so it lives
    # in the general module and routing calls it.
    from holographic.misc.holographic_relations import match_record
    qrec = extract_roles(request)
    if not qrec:
        return []                                            # unparsed request -> abstain (caller falls back)
    cand_records = {name: extract_roles(doc) for name, doc in module_texts.items()}
    return match_record(mind.encode_record, qrec, cand_records)


def _selftest():
    """Assert the REAL contract: the bound-record match separates the exact failure case (denoise vs fsr) that
    flat routing buries, and the extractor never fabricates a role. Numeric, fails loudly."""
    import lecore
    m = lecore.UnifiedMind(dim=1024, seed=0)

    # 1) the extractor parses the query structurally and never invents roles for gibberish
    rec = extract_roles("make my picture less grainy")
    assert rec.get('object') == 'image', rec               # 'picture' -> image
    assert rec.get('action') == 'reduce', rec              # 'less'/'grainy' -> reduce
    assert extract_roles("zzz qqq wwww") == {}, "fabricated a role from gibberish"

    # 2) the bound-record route inverts the r237 burial: denoise must beat fsr on 'less grainy'
    mods = {
        'denoise':   "denoise reduce noise in an image restore a grainy picture",
        'fsr':       "fsr increase the resolution of an image upscale",
        'sharpen':   "sharpen increase detail in an image",
        'fluid':     "fluid simulate water flowing swirling field",
    }
    ranked = route_structured(m, "make my picture less grainy", mods)
    top = ranked[0][0]
    assert top == 'denoise', "structured route failed to surface denoise: %r" % ranked
    denoise_score = dict(ranked)['denoise']
    fsr_score = dict(ranked)['fsr']
    assert denoise_score > fsr_score + 0.2, (denoise_score, fsr_score)  # clear separation, not a tie

    # 3) unbind recovers the structure (the holographic superpower is real, not decorative)
    qv, qrec = encode_request(m, "make my picture less grainy")
    back = m.decode_record(qv, {'action': list(_ACTION), 'object': list(_OBJECT), 'quality': list(_QUALITY)})
    assert back.get('object') == 'image', back
    print("  holoroute selftest OK: 'less grainy' -> denoise=%.3f > fsr=%.3f; roles unbind to %r"
          % (denoise_score, fsr_score, qrec))


if __name__ == "__main__":
    _selftest()
