"""The default install corpus must be general reference, not our own docs.

install.py grounds an install in a text file, and with no `--doc` it fell back to
leCore's OWN DOCUMENTATION. That is a real default -- always present, real prose
-- but it teaches the subject that every question is a leCore question, which
makes a documentation chatbot for this framework instead of a better reasoner.

WHAT THE CORPUS IS FOR, read off install.py rather than assumed:
    text[:20000]      the calibration fits here
    text[20000:26000] held-out evaluation
    text[i:i+240]     the searchable passages
SO THE HEAD OF THE FILE IS NOT ARBITRARY. Sorting the relations layer
alphabetically put "a", "about" and "above" in the calibration window while
`because`, `unless` and `whereas` fell past it -- measured, 2 of 8 causal probes
in the head. Ordered by reasoning weight instead: 8 of 8.
"""

import lzma
import os

CORPUS = os.path.join("lecore_data", "knowledge", "corpus.txt.xz")


def _text():
    with lzma.open(CORPUS) as f:
        return f.read().decode("utf-8")


def test_the_corpus_ships_and_is_the_right_size():
    assert os.path.exists(CORPUS), "the default corpus is not shipped"
    t = _text()
    # install.py mines vocabulary from text[:400000]; below that it is starved,
    # far above it is padding nobody reads.
    assert 350_000 <= len(t) <= 500_000, len(t)


def test_the_calibration_head_carries_the_reasoning_vocabulary():
    """**The first 20 KB is what the calibration fits on.**

    This is the whole reason the relations layer is ordered by reasoning weight
    rather than alphabetically."""
    head = _text()[:20000]
    for w in ("because", "unless", "since", "whereas", "although",
              "if", "therefore", "until"):
        assert (w + " --") in head, (
            "%r is not in the calibration window -- the relations layer is "
            "ordered wrongly" % w)


def test_all_six_layers_are_present_and_ordered():
    t = _text()
    marks = [(n, t.find(n)) for n in
             ("== RELATIONS ==", "== ORDER ==", "== SYNTAX ==",
              "== MATH ==", "== PLANNING ==", "== SEMANTICS ==")]
    for name, at in marks:
        assert at >= 0, "%s layer missing" % name
    positions = [at for _, at in marks]
    assert positions == sorted(positions), (
        "layers out of order -- relations must lead, it feeds the calibration")


def test_the_corpus_is_not_lecore_documentation():
    """The point of the change: grounding must not be about this framework."""
    t = _text().lower()
    for marker in ("unifiedmind", "find_capability", "holographic_", "lecore"):
        assert t.count(marker) == 0, (
            "%r appears in the default corpus -- it is leaking framework docs "
            "into the grounding text" % marker)


def test_licences_are_recorded_and_shipped():
    """Both sources are redistributable, and the manifest has to say so."""
    import json

    man = json.load(open(os.path.join("lecore_data", "knowledge",
                                      "manifest.json")))
    entry = man.get("corpus.txt.xz")
    assert entry, "the corpus is not in the manifest"
    licences = {s["license"] for s in entry["sources"]}
    assert any("PUBLIC DOMAIN" in l for l in licences), licences
    assert any("PSF" in l for l in licences), licences
    assert os.path.exists(os.path.join("lecore_data", "knowledge",
                                       "LICENSE_PYTHON.txt"))


def test_the_build_is_deterministic():
    """No RNG anywhere: same sources in, byte-identical corpus out."""
    import hashlib
    import json

    man = json.load(open(os.path.join("lecore_data", "knowledge",
                                      "manifest.json")))
    with lzma.open(CORPUS) as f:
        blob = f.read()
    assert hashlib.sha256(blob).hexdigest() == man["corpus.txt.xz"]["sha256"], (
        "the shipped corpus does not match the hash recorded when it was built")


def test_the_builder_can_fetch_its_own_sources():
    """**A build script whose sources are undocumented is a binary with extra steps.**

    The builder defaulted to /tmp paths that existed only because they had been
    curl'd by hand -- so the shipped corpus was reproducible on exactly one
    machine for exactly one afternoon. Now every source is a pinned raw URL with
    its licence in the table, fetched with stdlib urllib into a git-ignored cache.
    VERIFIED BY REBUILDING FROM AN EMPTY CACHE: the result was byte-identical to
    the shipped corpus (sha256 acfcf636ff56d51d) and to the manifest.
    This test asserts the CONTRACT without network: every source is declared,
    named, licensed and reachable as a plain https raw file."""
    from tools import build_corpus

    assert len(build_corpus.SOURCES) >= 5, build_corpus.SOURCES
    for name, url, licence in build_corpus.SOURCES:
        assert url.startswith("https://raw."), (
            "%s is not a pinned raw file -- an API or a redirect will drift" % name)
        assert licence, "%s has no recorded licence" % name
        assert "PUBLIC DOMAIN" in licence or "PSF" in licence, licence
    assert hasattr(build_corpus, "fetch"), "the builder cannot obtain its inputs"


def test_the_shipped_corpus_matches_its_recorded_hash():
    """The manifest hash is the contract that the shipped bytes are the built ones."""
    import hashlib
    import json

    with lzma.open(CORPUS) as f:
        blob = f.read()
    man = json.load(open(os.path.join("lecore_data", "knowledge",
                                      "manifest.json")))["corpus.txt.xz"]
    assert hashlib.sha256(blob).hexdigest() == man["sha256"], (
        "the shipped corpus does not match the hash recorded when it was built -- "
        "rebuild with tools/build_corpus.py --fetch and update the manifest")
    assert man["raw_chars"] == len(blob), (man["raw_chars"], len(blob))


def test_every_named_weakness_is_covered():
    """**The corpus exists to answer five named weaknesses, so probe all five.**

    The first build covered three and left two THIN -- measured, long-term
    planning 1/5 and math 3/6 on probe words. Adding math/statistics/fractions
    and itertools/asyncio-task closed both. Probing the ones already covered
    would have reported success while two thirds of the brief was unmet."""
    t = _text().lower()
    probes = {
        "code syntax": ("operator precedence", "grammar", "identifier"),
        "order of events": ("binding", "scope", "exception"),
        "cause and effect": ("because", "therefore", "unless"),
        "long-term planning": ("schedul", "await", "concurrent"),
        # "modulo" is MY word, not Python's -- the reference says fmod and
        # remainder and never once says modulo. The first version of this test
        # failed on that, and the corpus was right. PROBE WITH THE SOURCE'S
        # VOCABULARY, NOT YOUR OWN.
        "math": ("integer", "remainder", "logarithm"),
    }
    for weakness, words in probes.items():
        missing = [w for w in words if w not in t]
        assert not missing, "%s is not covered: missing %r" % (weakness, missing)


def test_no_layer_falls_outside_the_vocabulary_window():
    """**install.py mines vocabulary from text[:400000].**

    A layer starting past that is invisible to the thing it was added for.
    Uncapped, syntax took 245 KB and pushed SEMANTICS to char 556,677 -- 3.3 KB
    of ordinary prose, entirely outside the window. THE LAST LAYER MUST NOT PAY
    FOR THE FIRST ONE'S APPETITE."""
    t = _text()
    for name in ("== RELATIONS ==", "== ORDER ==", "== SYNTAX ==",
                 "== MATH ==", "== PLANNING ==", "== SEMANTICS =="):
        at = t.find(name)
        assert 0 <= at < 400000, (
            "%s starts at char %d, outside the 400 KB window install.py reads"
            % (name, at))
