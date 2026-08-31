"""Build the DEFAULT grounding corpus for an install.

WHY THIS EXISTS. `assimilation/install.py` grounds the install in a text file,
and with no `--doc` it fell back to leCore's OWN DOCUMENTATION. That is a real
default -- the docs are present and they are prose -- but it teaches the model
that every question is a leCore question, which turns the subject into a
documentation chatbot for this framework instead of a better reasoner.

WHAT THE CORPUS IS ACTUALLY FOR, read off install.py rather than assumed:

    text[:20000]        fit_ids      -- calibration fitting
    text[20000:26000]   eval_ids     -- held-out evaluation
    text[:400000]       words        -- the vocabulary it mines
    text[i:i+120]       negatives    -- contrastive samples
    text[i:i+240]       ~200 passages -- the searchable memory

So the target is roughly 400 KB, and THE FIRST 26 KB CARRIES THE CALIBRATION.
Size is not the goal; coverage of the things models are weak at is.

THE FOUR LAYERS, each chosen against a named weakness:

  RELATIONS   Webster's 1913 closed-class words. WordNet -- which the shipped
              dictionary is built from -- covers only nouns, verbs, adjectives
              and adverbs BY DESIGN, so `because`, `unless`, `since`, `whereas`
              and `if` are structurally absent. Their definitions are
              RELATIONAL ("because" = "by or for the cause that"), which is
              what teaches cause and effect rather than naming a thing.
  ORDER       Python's `executionmodel` reference: naming, binding, scope
              resolution, exception propagation. Order of events, written as a
              specification rather than a tutorial.
  SYNTAX      Python's `lexical_analysis`, `expressions`, `compound_stmts`.
              Precedence, associativity, grammar productions -- code syntax
              stated formally.
  SEMANTICS   Webster definitions of ordinary content words, for plain prose
              that is not about any one domain.

LICENCES, all redistributable and recorded in the manifest:
  Webster's Unabridged Dictionary 1913 -- PUBLIC DOMAIN (US, pre-1929)
  Python documentation                 -- PSF License Agreement

DETERMINISM: sources are read in a fixed order, entries sorted, and the output
is byte-identical across runs. No RNG anywhere in this file.
"""

import hashlib
import json
import lzma
import os
import re

# The layer order IS the file order, and it matters: the first 20 KB is what the
# calibration fits on, so relations and execution order come before syntax
# minutiae. A shuffled corpus would fit on whatever happened to land first.
LAYERS = ("relations", "order", "syntax", "math", "planning", "semantics")

_RST_DIRECTIVE = re.compile(r"^\s*\.\.\s+\w+::.*$")
_RST_ROLE = re.compile(r":[a-z:]+:`([^`]*)`")
_RST_LINK = re.compile(r"`([^`<]*)\s*<[^>]*>`_+")


def clean_rst(text):
    """reStructuredText -> plain prose.

    Directives, roles and link targets are MARKUP, not language: leaving
    ``:meth:`str.join``` in the corpus teaches the tokenizer about Sphinx. The
    role's *content* is kept, because that is a real word."""
    out = []
    for line in text.splitlines():
        if _RST_DIRECTIVE.match(line):
            continue
        if line.strip().startswith((".. ", "..\t")) or line.strip() == "..":
            continue
        line = _RST_LINK.sub(r"\1", line)
        line = _RST_ROLE.sub(r"\1", line)
        line = line.replace("``", "").replace("**", "").replace("*", "")
        out.append(line.rstrip())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def webster_entries(path, words):
    """`word -- definition` lines for the named words, in the order given."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    out = []
    for w in words:
        body = data.get(w)
        if not body:
            continue
        body = re.sub(r"\s+", " ", str(body)).strip()
        if len(body) < 20:
            continue
        out.append("%s -- %s" % (w, body[:700]))
    return out


def closed_class(supplement):
    """The closed-class word list, read from the shipped supplement.

    Read rather than re-declared, so the corpus and the dictionary supplement
    can never drift apart -- one list, one source.

    ORDERED BY REASONING WEIGHT, NOT ALPHABETICALLY. The first 20 KB of the
    corpus is what the calibration fits on, and sorted() put "a", "about" and
    "above" there while `because`, `unless` and `whereas` fell past the cutoff --
    MEASURED, 2 of 5 causal probes in the head. Alphabetical order is arbitrary
    with respect to what the corpus is FOR."""
    with lzma.open(supplement) as f:
        blob = json.load(f)
    have = set(blob.get("words") or {})
    first = [w for w in (
        "because", "since", "unless", "although", "though", "whereas", "if",
        "else", "when", "whether", "until", "before", "after", "while",
        "therefore", "hence", "thus", "consequently", "otherwise", "however",
        "nevertheless", "moreover", "provided", "lest", "so", "and", "or",
        "but", "not", "then", "once", "every", "each", "all", "any", "no",
    ) if w in have]
    return first + sorted(have - set(first))


def build(webster, py_docs, supplement, budget=420000, per_layer_cap=70000,
          semantics_reserve=60000):
    """Assemble the corpus. Returns (text, manifest).

    `per_layer_cap` bounds each documentation layer and `semantics_reserve`
    holds room for ordinary prose, because install.py mines its vocabulary from
    text[:400000] and the last layer would otherwise be pushed outside it."""
    parts, manifest = [], []

    def add(layer, body, source, licence):
        if not body:
            return
        parts.append("== %s ==\n\n%s" % (layer.upper(), body))
        manifest.append({"layer": layer, "source": source, "license": licence,
                         "chars": len(body)})

    # 1. RELATIONS -- first, because the calibration fits on the head of the file
    rel = closed_class(supplement)
    add("relations", "\n\n".join(webster_entries(webster, rel)),
        "Webster's Unabridged Dictionary, 1913 (closed-class words)",
        "PUBLIC DOMAIN (US, pre-1929)")

    # 2. ORDER, then 3. SYNTAX -- a fixed reading order, not a directory listing
    order_files = ["executionmodel.rst"]
    syntax_files = ["lexical_analysis.rst", "expressions.rst",
                    "compound_stmts.rst", "simple_stmts.rst"]
    # MATH and PLANNING close the two weaknesses the first build left thin.
    # math/statistics/fractions state numeric behaviour EXACTLY -- domains,
    # error conditions, exact-vs-approximate -- which is the part a model
    # guesses at. itertools and asyncio-task are composition and scheduling:
    # steps that depend on other steps, which is what "orchestration" means
    # when it is written down rather than gestured at.
    math_files = ["math.rst", "statistics.rst", "fractions.rst"]
    plan_files = ["itertools.rst", "asyncio-task.rst"]
    for layer, names in (("order", order_files), ("syntax", syntax_files),
                         ("math", math_files), ("planning", plan_files)):
        bodies = []
        for n in names:
            p = os.path.join(py_docs, n)
            if os.path.exists(p):
                with open(p, encoding="utf-8", errors="ignore") as f:
                    bodies.append(clean_rst(f.read()))
        # CAP EACH DOC LAYER. install.py mines vocabulary from text[:400000], so
        # a layer that runs past that is invisible to the thing it was added for.
        # Uncapped, syntax alone took 245 KB and pushed SEMANTICS to char 556,677
        # -- 3.3 KB of ordinary prose, entirely outside the window. THE LAST
        # LAYER MUST NOT PAY FOR THE FIRST ONE'S APPETITE.
        body = "\n\n".join(bodies)
        add(layer, body[:per_layer_cap],
            "Python Language Reference (%s)" % ", ".join(names),
            "PSF License Agreement")

    # 4. SEMANTICS -- ordinary prose, filling whatever budget remains
    used = sum(len(p) for p in parts)
    room = max(semantics_reserve, budget - used)
    if room > 1000:
        with open(webster, encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        # deterministic: sorted headwords, ordinary length, no proper nouns
        pool = [w for w in sorted(data)
                if w.isalpha() and w.islower() and 4 <= len(w) <= 12]
        picked, total = [], 0
        for w in pool:
            body = re.sub(r"\s+", " ", str(data[w])).strip()
            if len(body) < 40:
                continue
            line = "%s -- %s" % (w, body[:400])
            if total + len(line) > room:
                break
            picked.append(line)
            total += len(line) + 2
        add("semantics", "\n\n".join(picked),
            "Webster's Unabridged Dictionary, 1913 (content words)",
            "PUBLIC DOMAIN (US, pre-1929)")

    text = "\n\n".join(parts)
    return text, manifest


SOURCES = (
    # (local name, URL, licence). Pinned to a raw file, not an API, so a fetch is
    # one GET with no auth and no version negotiation.
    ("webster.json",
     "https://raw.githubusercontent.com/matthewreagan/"
     "WebstersEnglishDictionary/master/dictionary_compact.json",
     "PUBLIC DOMAIN (US, pre-1929)"),
    ("executionmodel.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/reference/"
     "executionmodel.rst", "PSF License Agreement"),
    ("lexical_analysis.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/reference/"
     "lexical_analysis.rst", "PSF License Agreement"),
    ("expressions.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/reference/"
     "expressions.rst", "PSF License Agreement"),
    ("compound_stmts.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/reference/"
     "compound_stmts.rst", "PSF License Agreement"),
    ("simple_stmts.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/reference/"
     "simple_stmts.rst", "PSF License Agreement"),
    # MATH and PLANNING were the two named weaknesses the first corpus left
    # thin -- measured 3/6 and 1/5 on probe words against the built text.
    ("math.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/"
     "math.rst", "PSF License Agreement"),
    ("statistics.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/"
     "statistics.rst", "PSF License Agreement"),
    ("fractions.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/"
     "fractions.rst", "PSF License Agreement"),
    ("itertools.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/"
     "itertools.rst", "PSF License Agreement"),
    ("asyncio-task.rst",
     "https://raw.githubusercontent.com/python/cpython/main/Doc/library/"
     "asyncio-task.rst", "PSF License Agreement"),
)


def fetch(cache=".corpus_sources"):
    """Download the sources into `cache`, skipping what is already there.

    THE BUILDER HAD NO WAY TO GET ITS OWN INPUTS. It defaulted to /tmp paths that
    existed only because I had curl'd them by hand, so the corpus was
    reproducible on exactly one machine for exactly one afternoon -- and the
    thing it builds SHIPS. A build script whose sources are undocumented is a
    binary with extra steps.
    stdlib urllib, per the constitution; every URL is a raw file over https, and
    the licence travels in the table above so nobody has to go looking."""
    import urllib.request

    os.makedirs(cache, exist_ok=True)
    got = []
    for name, url, licence in SOURCES:
        dest = os.path.join(cache, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            got.append((name, dest, licence, "cached"))
            continue
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        got.append((name, dest, licence, "%.1f KB" % (len(data) / 1e3)))
    return got


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # DEFAULT TO THE CACHE, NOT TO /tmp. The old defaults named files that
    # existed only on the machine where they were curl'd by hand.
    ap.add_argument("--cache", default=".corpus_sources",
                    help="where fetched sources live (git-ignored)")
    ap.add_argument("--fetch", action="store_true",
                    help="download any missing sources first (stdlib urllib)")
    ap.add_argument("--webster", default=None)
    ap.add_argument("--py-docs", default=None)
    ap.add_argument("--supplement",
                    default="lecore_data/knowledge/function_words.json.xz")
    ap.add_argument("--out", default="lecore_data/knowledge/corpus.txt.xz")
    ap.add_argument("--budget", type=int, default=420000)
    a = ap.parse_args(argv)

    if a.fetch:
        for name, dest, licence, how in fetch(a.cache):
            print("   fetched %-22s %-10s %s" % (name, how, licence))
    webster = a.webster or os.path.join(a.cache, "webster.json")
    py_docs = a.py_docs or a.cache
    if not os.path.exists(webster):
        raise SystemExit(
            "no sources at %s -- run with --fetch to download them "
            "(Webster 1913: public domain; Python docs: PSF)" % a.cache)
    text, manifest = build(webster, py_docs, a.supplement, a.budget)
    if len(text) < 100000:
        raise SystemExit("corpus is only %d chars -- a source was missing; "
                         "check --webster and --py-docs" % len(text))
    blob = text.encode("utf-8")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(lzma.compress(blob, preset=9))

    print("corpus %s" % a.out)
    for m in manifest:
        print("   %-10s %7.1f KB  %s" % (m["layer"], m["chars"] / 1e3, m["license"]))
    print("   %-10s %7.1f KB raw -> %.1f KB compressed"
          % ("TOTAL", len(blob) / 1e3, os.path.getsize(a.out) / 1e3))
    print("   sha256 %s" % hashlib.sha256(blob).hexdigest()[:16])
    return manifest


if __name__ == "__main__":
    main()
