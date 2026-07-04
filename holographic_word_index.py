"""holographic_word_index.py -- an OPTIONAL semantic index over the dictionary: find words by MEANING.

The dictionary looks a word UP by its spelling. This does the fuzzy reverse: given a description ('unexpected good
luck', 'a young dog') find the words whose DEFINITIONS mean that ('serendipity', 'puppy'). It's built by RANDOM
INDEXING -- a classic, readable VSA move: every token has a fixed random vector, a definition's MEANING vector is the
bundle (normalised sum) of its content words' vectors, and a query is encoded the same way and matched by cosine. This
is APPROXIMATE by nature -- a lossy, geometry-preserving meaning space -- which is exactly what VSA is good at, and why
it lives here on the SEARCH path and not on the exact-lookup path.

OPT-IN and separate: nothing in this file runs unless you call build_semantic_index(). Importing leCore, building a
UnifiedMind, or doing plain word lookups never touches it -- so a user who just wants the library to build on top pays
nothing for this. Building the index loads the dictionary once and encodes the words you ask for.

HONEST about scale: the index is (N words x dim) float32. A few thousand common words is a few MB; the whole 144k at
dim=256 is ~150 MB. So you choose the vocabulary (pass `words=`) and the `dim`. Deterministic (token vectors seeded by
hashlib, per the no-Python-hash rule); NumPy + stdlib only.

KEPT NEGATIVE (loud): this is RANDOM INDEXING over short dictionary glosses, not a trained embedding. find(description)
is reliable for the TOP hit or two ('a young dog'->puppy, 'a body of water'->river/ocean); the tail is noisy, and
similar(word) is hit-or-miss because it only sees which CONTENT WORDS two definitions share -- 'sprint' ("a quick run")
and 'run' ("a score in baseball...") don't match because the dictionary's primary sense of 'run' is the baseball one.
It captures the ONE definition stored per word, so word-sense collisions bite. Treat it as fuzzy suggestion, not truth.
For sharper meaning use the co-occurrence space (holographic_meaning_predict) trained on a real corpus.
"""
import hashlib
import re

import numpy as np

# small, obvious stop-word list so meaning vectors are built from CONTENT words, not glue words
_STOP = set(("a an the of to and or in on at for with without by from as is are was were be been being that this "
             "these those it its into over under out up down not no than then so such may can will would used use "
             "usually especially any some each other another one two which who whose whom where when what").split())


def _token_vector(token, dim, seed, _cache):
    """A fixed, deterministic random vector for a token (random indexing's index vector). Seeded by hashlib -- NOT
    Python's hash() -- so it's identical every run and across machines. Cached so a token that appears in many
    definitions is only generated once."""
    v = _cache.get(token)
    if v is not None:
        return v
    h = int.from_bytes(hashlib.sha256(("%d:%s" % (seed, token)).encode()).digest()[:8], "big")
    r = np.random.default_rng(h).standard_normal(dim).astype(np.float32)
    r /= (np.linalg.norm(r) + 1e-9)
    _cache[token] = r
    return r


def _content_tokens(text):
    """The lower-case content words of a piece of text (letters only, stop-words dropped)."""
    return [t for t in re.findall(r"[a-z]+", str(text).lower()) if t not in _STOP and len(t) > 1]


def _meaning_vector(text, dim, seed, cache):
    """A meaning vector for a phrase: the bundle (normalised sum) of its content tokens' vectors. Empty text -> zeros."""
    toks = _content_tokens(text)
    if not toks:
        return np.zeros(dim, dtype=np.float32)
    acc = np.zeros(dim, dtype=np.float32)
    for t in toks:
        acc += _token_vector(t, dim, seed, cache)
    n = np.linalg.norm(acc)
    return acc / n if n > 0 else acc


class SemanticIndex:
    """Words placed in a meaning space so you can search by description. Hold one and call find()/similar(). It stores
    a row-normalised (N, dim) matrix + the word list + the token cache (so queries encode the same way it was built)."""

    def __init__(self, words, matrix, dim, seed, cache):
        self.words = list(words)
        self.M = matrix                                    # (N, dim) float32, each row unit-length
        self.dim = dim
        self.seed = seed
        self._cache = cache
        self._pos = {w: i for i, w in enumerate(self.words)}

    def __len__(self):
        return len(self.words)

    def _rank(self, qvec, k, skip=None):
        if np.linalg.norm(qvec) == 0:
            return []
        scores = self.M @ qvec                             # cosine (rows are unit, qvec we normalise here)
        scores = scores / (np.linalg.norm(qvec) + 1e-9)
        order = np.argsort(-scores)
        out = []
        for i in order:
            if skip is not None and i == skip:
                continue
            out.append((self.words[i], float(scores[i])))
            if len(out) >= k:
                break
        return out

    def find(self, description, k=10):
        """The words whose definitions best match a free-text `description`, most-similar first, as (word, score)."""
        q = _meaning_vector(description, self.dim, self.seed, self._cache)
        return self._rank(q, k)

    def similar(self, word, k=10):
        """Words with the most similar MEANING to `word`. If `word` is in the index its own vector is used; otherwise
        its definition is encoded on the fly (so you can probe words you didn't index)."""
        word = word.lower()
        if word in self._pos:
            i = self._pos[word]
            return self._rank(self.M[i], k, skip=i)
        from holographic_dictionary import define
        d = define(word)
        if not d:
            return []
        return self._rank(_meaning_vector(d, self.dim, self.seed, self._cache), k)

    def vector(self, word):
        """The meaning vector of an indexed word (or None)."""
        i = self._pos.get(word.lower())
        return None if i is None else self.M[i]


def build_semantic_index(words=None, dim=256, seed=0, include_synonyms=True, max_words=None):
    """Build a SemanticIndex over the dictionary. `words` limits it to a vocabulary you care about (default: every
    word, which is ~150 MB at dim=256 -- pass a list or `max_words` to keep it small). Each word's meaning vector is
    built from its definition (and its synonyms, if include_synonyms). Loads the dictionary on first use.

    This is the OPT-IN entry point -- the only thing in this module that does work."""
    from holographic_dictionary import _load, define, synonyms as _syn

    D = _load()
    if words is None:
        words = list(D)
    words = [w.lower() for w in words if w.lower() in D]
    if max_words is not None:
        words = words[:max_words]

    cache = {}
    rows = np.zeros((len(words), dim), dtype=np.float32)
    for i, w in enumerate(words):
        text = define(w) or ""
        if include_synonyms:
            text = text + " " + " ".join(_syn(w))          # a word's synonyms sharpen its meaning vector
        rows[i] = _meaning_vector(text, dim, seed, cache)
    return SemanticIndex(words, rows, dim, seed, cache)


def _selftest():
    # build a small index over related + unrelated words, then search BY DESCRIPTION (the strong path)
    vocab = ["dog", "puppy", "cat", "kitten", "wolf", "car", "truck", "serendipity", "luck", "chance",
             "happy", "joyful", "sad", "river", "stream", "ocean", "mountain", "wealth", "money", "king"]
    idx = build_semantic_index(words=vocab, dim=512, seed=0)
    assert len(idx) == len(vocab)

    # find-by-description is reliable for the TOP results (this is the capability)
    def top(q, k=3):
        return [w for w, _ in idx.find(q, k=k)]
    assert "puppy" in top("a young dog") or "dog" in top("a young dog")
    assert "serendipity" in top("good fortune") or "luck" in top("good fortune")
    assert any(w in top("a body of water") for w in ("river", "ocean", "stream"))
    assert any(w in top("a wheeled vehicle") for w in ("car", "truck"))

    # similar() clusters obvious cases (animals near animals) -- but see the kept negative on quality
    assert any(w in [x for x, _ in idx.similar("puppy", k=4)] for w in ("dog", "kitten"))

    # a word not in the index is encoded from its definition on the fly (no crash)
    assert isinstance(idx.similar("hound", k=3), list)

    print("OK: holographic_word_index self-test passed (find-by-description nails the top hits: 'a young dog'->%s, "
          "'good fortune'->%s, 'a body of water'->%s; similar() clusters obvious cases; approximate random-indexing, "
          "opt-in)" % (top("a young dog")[:2], top("good fortune")[:2], top("a body of water")[:2]))


if __name__ == "__main__":
    _selftest()
