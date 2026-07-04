"""Tests for holographic_word_index.py -- the OPT-IN semantic index over the dictionary (find words by meaning)."""
import holographic_dictionary as hd
from holographic_word_index import build_semantic_index, SemanticIndex


VOCAB = ["dog", "puppy", "cat", "kitten", "wolf", "car", "truck", "serendipity", "luck", "chance",
         "happy", "joyful", "sad", "river", "stream", "ocean", "mountain", "wealth", "money", "king"]


def test_find_by_description_top_hits():
    idx = build_semantic_index(words=VOCAB, dim=512, seed=0)
    top = lambda q: [w for w, _ in idx.find(q, k=3)]
    assert "puppy" in top("a young dog") or "dog" in top("a young dog")
    assert "serendipity" in top("good fortune") or "luck" in top("good fortune")
    assert any(w in top("a body of water") for w in ("river", "ocean", "stream"))
    assert any(w in top("a wheeled vehicle") for w in ("car", "truck"))


def test_similar_clusters_obvious_cases():
    idx = build_semantic_index(words=VOCAB, dim=512, seed=0)
    sim = [w for w, _ in idx.similar("puppy", k=4)]
    assert any(w in sim for w in ("dog", "kitten"))


def test_deterministic():
    a = build_semantic_index(words=VOCAB, dim=256, seed=0)
    b = build_semantic_index(words=VOCAB, dim=256, seed=0)
    assert a.find("a young dog", k=5) == b.find("a young dog", k=5)


def test_unindexed_word_encoded_on_the_fly():
    idx = build_semantic_index(words=VOCAB, dim=256, seed=0)
    assert isinstance(idx.similar("hound", k=3), list)         # not in the index -> encode its definition, no crash


def test_scoping_and_len():
    idx = build_semantic_index(words=VOCAB, dim=128, seed=0, max_words=5)
    assert len(idx) == 5


def test_it_is_opt_in_import_does_not_load_dictionary():
    """Importing the module must NOT load the dictionary; only building an index does."""
    hd.unload()
    import importlib, holographic_word_index
    importlib.reload(holographic_word_index)
    assert not hd.is_loaded()                                  # importing/reloading the index module didn't load it
    holographic_word_index.build_semantic_index(words=["dog", "cat"], dim=64)
    assert hd.is_loaded()                                      # building did (it needs the definitions)


def test_dictionary_lazy_controls():
    hd.unload()
    assert hd.is_loaded() is False
    assert hd.stats()["loaded"] is False                      # reading stats does not load
    hd.preload()
    assert hd.is_loaded() is True and hd.stats()["words"] == 144478
    hd.unload()
    assert hd.is_loaded() is False
    assert hd.define("dog")                                    # still works -- transparently reloads
