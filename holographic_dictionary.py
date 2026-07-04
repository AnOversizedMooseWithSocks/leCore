"""holographic_dictionary.py -- a vendored, comprehensive English DICTIONARY + TAXONOMY, for contextual awareness.

WHAT THIS IS
------------
The engine has always had the MACHINERY to learn word meaning from a dictionary (holographic_lexicon, the mind's
learn_dictionary / define), but no actual dictionary shipped with it -- you had to supply one. This vendors a real,
comprehensive one: ~144,000 English words, each with a definition, part of speech, synonyms, an example, and its
"is_a" parent (hypernym) -- so the engine has genuine world-knowledge to lean on, not just its internal machinery.

The data is Princeton WordNet 3.0 (via NLTK), redistributed under the WordNet license (see LICENSE_WORDNET.txt).
It is stored LZMA-compressed (~3.3 MB -- ~45% smaller than the old gzip, same exact round-trip) and loaded
LAZILY with the standard library only (json + lzma/gzip) -- NLTK was used
once at build time to extract it and is NOT a runtime dependency, honouring the NumPy/stdlib-only rule.

  dictionary.json.xz : {word: {"d": definition, "p": pos, "s": [synonyms], "e": example, "h": is_a-parent}}

TWO RESOURCES IN ONE FILE
  * DICTIONARY -- define(word), synonyms(word), part_of_speech(word), example(word).
  * ENCYCLOPEDIA / TAXONOMY -- is_a(word) and hypernym_chain(word) walk the "kind of" hierarchy
    (dog -> domestic animal -> animal -> ... -> entity), which is exactly what holographic_encyclopedia consumes.

Users can supply their own or a larger dictionary by replacing the vendored file, or pass their own map to the
Lexicon / learn_dictionary machinery -- this is just the batteries-included default.
"""
import gzip
import lzma
import json
import os

# The vendored dictionary ships in the lecore_data PACKAGE, so it resolves the same from a source clone and from a
# pip-installed wheel. It is stored LZMA-compressed (.json.xz, ~3.3 MB) -- lzma is stdlib and packs this JSON ~45%
# tighter than gzip did (5.9 -> 3.3 MB) with the same exact, lossless round-trip. We still accept the older gzip
# (.json.gz) file, and the old repo-relative data/ path, so an older checkout keeps working.
_CANDIDATES = ("dictionary.json.xz", "dictionary.json.gz")     # prefer lzma; fall back to gzip


def _resolve_data_path():
    # try the packaged location first (xz then gz), then the legacy repo-relative data/ dir (xz then gz)
    try:
        import lecore_data
        for name in _CANDIDATES:
            p = lecore_data.file("knowledge", name)
            if os.path.exists(p):
                return p
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _CANDIDATES:
        p = os.path.join(here, "data", "knowledge", name)
        if os.path.exists(p):
            return p
    return os.path.join(here, "data", "knowledge", _CANDIDATES[0])   # a sensible default for the error message


_DATA_PATH = _resolve_data_path()

_DICT = None                                            # lazily-loaded {word: entry}; None until first use


def _open_compressed(path):
    """Open the vendored dictionary for text reading, picking the decompressor from the extension: lzma for .xz,
    gzip for .gz. Both are stdlib and both are exactly lossless -- the choice is purely which packs smaller."""
    if path.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8")
    return gzip.open(path, "rt", encoding="utf-8")


def _load():
    """Load the compressed dictionary once, on first use (stdlib only). Cached for the process lifetime."""
    global _DICT
    if _DICT is None:
        if not os.path.exists(_DATA_PATH):
            raise FileNotFoundError("vendored dictionary not found at %s -- is data/knowledge/ present?" % _DATA_PATH)
        with _open_compressed(_DATA_PATH) as f:
            _DICT = json.load(f)
    return _DICT


# -- lazy-load control (the language layer is OPT-IN) -------------------------------------------------------
# The dictionary NEVER loads just from importing leCore or building a UnifiedMind -- it decompresses into a plain dict
# in RAM the first time you actually call a language function (lookup/has/define/...), and only then. So a user who
# imports the library to build on top pays nothing for the language layer unless they use it. These helpers make that
# behaviour visible and controllable, instead of leaving it implicit.
def is_loaded():
    """True once the dictionary has been decompressed into RAM (i.e. after the first language call)."""
    return _DICT is not None


def preload():
    """Force the one-time decompress+parse NOW (e.g. at app startup) so the FIRST lookup later isn't the slow one.
    Returns the number of words. Optional -- lookups auto-load on first use anyway."""
    return len(_load())


def unload():
    """Drop the dictionary from RAM (frees ~22 MB). The next language call will transparently reload it. Handy if you
    used a language feature once and want the memory back."""
    global _DICT
    _DICT = None


def stats():
    """A small, honest picture of the language layer's state: whether it's loaded, how many words, which file backs it
    and how it's compressed. Reading this does NOT trigger a load."""
    import os
    path = _DATA_PATH
    return {"loaded": _DICT is not None,
            "words": (len(_DICT) if _DICT is not None else None),   # None until loaded, so this stays load-free
            "source": os.path.basename(path),
            "compression": "lzma (xz)" if path.endswith(".xz") else ("gzip" if path.endswith(".gz") else "?"),
            "on_disk_bytes": (os.path.getsize(path) if os.path.exists(path) else None)}


# -- membership + size --------------------------------------------------------------------------------------
def has(word):
    return _norm(word) in _load()


def size():
    """How many words the dictionary holds."""
    return len(_load())


def words():
    """All words (a large list -- ~144k). Prefer search()/has() unless you really need the whole list."""
    return list(_load())


def _norm(word):
    return str(word).strip().lower()


# -- the dictionary -----------------------------------------------------------------------------------------
def entry(word):
    """The full record for a word: {definition, part_of_speech, synonyms, example, is_a}. None if unknown."""
    e = _load().get(_norm(word))
    if e is None:
        return None
    out = {"word": _norm(word), "definition": e.get("d", ""), "part_of_speech": e.get("p", "")}
    if "s" in e:
        out["synonyms"] = list(e["s"])
    if "e" in e:
        out["example"] = e["e"]
    if "h" in e:
        out["is_a"] = e["h"]
    return out


def define(word):
    """The definition string for a word ('' if unknown). The plain 'what does this word mean?' lookup."""
    e = _load().get(_norm(word))
    return e.get("d", "") if e else ""


def synonyms(word):
    """The word's synonyms (empty list if none/unknown)."""
    e = _load().get(_norm(word))
    return list(e.get("s", [])) if e else []


def part_of_speech(word):
    """noun / verb / adj / adv ('' if unknown)."""
    e = _load().get(_norm(word))
    return e.get("p", "") if e else ""


def example(word):
    """An example sentence using the word ('' if none/unknown)."""
    e = _load().get(_norm(word))
    return e.get("e", "") if e else ""


def search(prefix, k=25):
    """Words starting with a prefix (sorted) -- a simple, deterministic autocomplete over the vocabulary."""
    p = _norm(prefix)
    return sorted(w for w in _load() if w.startswith(p))[:k]


# -- the taxonomy (encyclopedia side) -----------------------------------------------------------------------
def is_a(word):
    """The word's immediate 'is a kind of' parent (hypernym), or None. E.g. is_a('dog') -> 'domestic animal'."""
    e = _load().get(_norm(word))
    return e.get("h") if e else None


def hypernym_chain(word, max_depth=12):
    """Walk the is_a hierarchy upward: dog -> domestic animal -> animal -> ... The 'what kind of thing is this?'
    chain, useful for contextual reasoning and for seeding holographic_encyclopedia. Stops at the top or on a cycle."""
    chain = []
    seen = set()
    cur = _norm(word)
    for _ in range(max_depth):
        parent = is_a(cur)
        if not parent or parent in seen:
            break
        chain.append(parent)
        seen.add(parent)
        cur = _norm(parent)
    return chain


# -- bridges to the engine's learning machinery -------------------------------------------------------------
def definition_words(word):
    """The content words appearing in a word's definition (lowercased, alpha, de-duplicated) -- the input shape the
    Lexicon / mind.learn_dictionary expects for bootstrapping meaning from definitions."""
    import re
    e = _load().get(_norm(word))
    if not e:
        return []
    toks = re.findall(r"[a-z]+", e.get("d", "").lower())
    seen, out = set(), []
    for t in toks:
        if len(t) > 2 and t != _norm(word) and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def definition_map(vocab):
    """A {word: [definition words]} map over a given vocabulary -- feed straight into mind.learn_dictionary() so the
    encoder bootstraps meaning from the REAL dictionary. Only words present in the dictionary are included."""
    d = _load()
    return {w: definition_words(w) for w in vocab if _norm(w) in d}


def manifest():
    """The self-describing manifest for the vendored resource (source, license, entry count, fields)."""
    path = os.path.join(os.path.dirname(_DATA_PATH), "manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"name": "vendored dictionary", "entries": size()}


def _selftest():
    assert size() > 100000, size()
    # dictionary lookups
    assert "force" in define("gravity").lower()
    assert part_of_speech("dog") == "noun"
    assert has("algorithm") and not has("zzzznotaword")
    assert "member" in define("dog").lower()
    syn = synonyms("happy")
    assert isinstance(syn, list)
    # entry shape
    e = entry("photosynthesis")
    assert e["part_of_speech"] == "noun" and "synthesis" in e["definition"].lower() and "is_a" in e
    # taxonomy: dog is a kind of ... animal (somewhere up the chain)
    assert is_a("dog") is not None
    chain = hypernym_chain("dog")
    assert any("animal" in c or "entity" in c for c in chain), chain
    # search / autocomplete
    assert all(w.startswith("algor") for w in search("algor"))
    # bridge to the learning machinery
    dm = definition_map(["gravity", "dog", "notarealword_xyz"])
    assert "gravity" in dm and "dog" in dm and "notarealword_xyz" not in dm
    assert all(isinstance(v, list) for v in dm.values())
    # unknown word is graceful
    assert entry("zzzznotaword") is None and define("zzzznotaword") == "" and is_a("zzzznotaword") is None
    m = manifest()
    print("OK: holographic_dictionary self-test passed (%d words vendored; define/synonyms/pos/example, is_a taxonomy "
          "+ hypernym_chain, search, definition_map bridge -- %s, %s)" % (size(), m.get("source"), m.get("license")))


if __name__ == "__main__":
    _selftest()
