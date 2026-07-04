"""Tests for holographic_dictionary (the vendored WordNet dictionary + taxonomy)."""
import holographic_dictionary as d


def test_size_is_comprehensive():
    assert d.size() > 100000


def test_define_and_pos():
    assert "force" in d.define("gravity").lower()
    assert d.part_of_speech("dog") == "noun"
    assert d.define("algorithm")


def test_entry_shape():
    e = d.entry("photosynthesis")
    assert e["part_of_speech"] == "noun" and "synthesis" in e["definition"].lower()
    assert "is_a" in e


def test_synonyms_and_example():
    assert isinstance(d.synonyms("happy"), list)
    assert isinstance(d.example("dog"), str)


def test_taxonomy_is_a_and_chain():
    assert d.is_a("dog") is not None
    chain = d.hypernym_chain("dog")
    assert any("animal" in c or "entity" in c for c in chain)


def test_search_prefix():
    res = d.search("algor")
    assert res and all(w.startswith("algor") for w in res)


def test_unknown_word_is_graceful():
    assert d.entry("zzzznotaword") is None
    assert d.define("zzzznotaword") == ""
    assert d.is_a("zzzznotaword") is None
    assert d.synonyms("zzzznotaword") == []


def test_definition_map_bridge():
    dm = d.definition_map(["gravity", "dog", "notarealword_xyz"])
    assert "gravity" in dm and "dog" in dm and "notarealword_xyz" not in dm
    assert all(isinstance(v, list) and v for v in dm.values())


def test_manifest_has_provenance():
    m = d.manifest()
    assert "WordNet" in m.get("source", "") and m.get("entries", 0) > 100000
    assert "license" in m


def test_case_insensitive():
    assert d.has("Dog") and d.has("DOG") and d.define("Gravity")
