#!/usr/bin/env python3
"""tools/backlog_probe.py -- "does this engine already do X?", answered honestly.

WHY THIS EXISTS. Twice in one program an audit reached the wrong conclusion about whether a backlog item was
already built, and both times the instrument was at fault, not the eye:

  * `find_capability` ALWAYS returns its best three, even when the best is nothing. A query about a faculty the
    engine lacks comes back with three confident-looking names. Read as a hit, it says "built" when it is not.
    (Concluded W5 was unbuilt from a fallback -- it had shipped, wired, catalogued.)
  * grepping for a symbol name you INVENTED (`def factored_blur`) proves only that you guessed the name wrong.
    (Concluded W1 was absent; it ships as `low_rank_field` / `factored_field_report`.)

Those are the same error wearing two costumes: an absence of evidence from an instrument that cannot report
absence. The fix is to ask both questions and to read the SCORE, not the name.

    python3 tools/backlog_probe.py "blur a field without decompressing it"
    python3 tools/backlog_probe.py --symbol hierarchical_recall
    python3 tools/backlog_probe.py --file backlog_items.txt

For each query it prints:
  * the top catalog capabilities WITH SCORES and the confidence verdict (a dominant top vs a flat fallback list)
  * whether the top capability is a live mind faculty (wired) or a curated home
  * any module-level `def`/`class` whose name contains the query's distinctive words

VERDICTS: SHIPPED (confident catalog hit), LIKELY (a hit, but no clear lead -- read the names), ABSENT (no score).
None of them is a substitute for reading the code; all of them beat guessing.
"""

import argparse
import pathlib
import re
import sys


def _mind():
    # run from anywhere: the repo root is this file's parent's parent
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import lecore
    return lecore.UnifiedMind(dim=64, seed=0)


def _symbols(words, root=None):
    """Every top-level `def`/`class` whose name contains one of `words`. A cheap existence check that does not
    depend on guessing the exact name -- pass the CONCEPT's words, not a name you made up."""
    pat = re.compile(r"^(?:def|class)\s+(\w+)", re.M)
    hits = []
    root = pathlib.Path(root) if root else pathlib.Path(__file__).resolve().parent.parent / "holographic"
    for f in sorted(root.rglob("*.py")):
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in pat.findall(txt):
            low = name.lower()
            if any(w in low for w in words):
                hits.append((f.stem.replace("holographic_", ""), name))
    return hits


_STOP = {"a", "an", "the", "of", "in", "on", "to", "and", "or", "it", "its", "with", "without",
         "into", "from", "for", "that", "this", "is", "are", "be", "by", "as", "at", "my", "me"}


def _content_words(q, min_len=5):
    return [w for w in re.findall(r"[a-z]+", q.lower()) if w not in _STOP and len(w) >= min_len]


def probe(mind, query, k=4, show_symbols=True):
    """Print the honest verdict for one query. Returns the verdict string."""
    conf = mind.capability_confidence(query)
    scored = mind.find_scored(query, k=k)

    if not scored or conf["score"] == 0.0:
        verdict = "ABSENT"
    elif conf["confident"]:
        verdict = "SHIPPED"
    else:
        verdict = "LIKELY"

    print("\n%s" % query)
    print("  verdict: %-8s  top score %.1f, margin %.1f over the runner-up" % (verdict, conf["score"], conf["margin"]))
    for cap, s in scored:
        kind = "faculty" if getattr(cap, "native", False) else "home   "
        print("    %5.1f  [%s] %s" % (s, kind, cap.name))
    if verdict == "LIKELY":
        print("    ^ no clear lead: these may be FALLBACKS. Read the names before concluding anything.")

    if show_symbols:
        words = _content_words(query)
        syms = _symbols(words) if words else []
        if syms:
            print("  symbols containing %s:" % (words,))
            for mod, name in syms[:8]:
                print("    %s.%s" % (mod, name))
        elif words:
            print("  no module symbol contains any of %s (weak evidence: the name may simply differ)" % (words,))
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query", nargs="*", help="a plain-English description of the capability")
    ap.add_argument("--symbol", help="check for a module symbol containing this substring, and nothing else")
    ap.add_argument("--file", help="a file of one query per line (blank lines and #comments skipped)")
    ap.add_argument("-k", type=int, default=4, help="how many capabilities to show")
    args = ap.parse_args(argv)

    if args.symbol:
        hits = _symbols([args.symbol.lower()])
        if hits:
            for mod, name in hits:
                print("%s.%s" % (mod, name))
        else:
            print("no symbol contains %r" % args.symbol)
        return 0 if hits else 1

    queries = []
    if args.file:
        for line in pathlib.Path(args.file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    if args.query:
        queries.append(" ".join(args.query))
    if not queries:
        ap.print_help()
        return 2

    mind = _mind()
    counts = {"SHIPPED": 0, "LIKELY": 0, "ABSENT": 0}
    for q in queries:
        counts[probe(mind, q, k=args.k)] += 1
    print("\n%d shipped, %d likely (read them), %d absent" % (counts["SHIPPED"], counts["LIKELY"], counts["ABSENT"]))
    return 0


def _selftest():
    """The tool must correctly call the two cases that fooled a human auditor: a shipped faculty it once called
    absent, and a fallback it once read as a hit."""
    mind = _mind()

    # W1 ships (as low_rank_field / factored_field_report). A confident catalog hit.
    assert mind.capability_confidence("blur a field without decompressing it")["confident"] is True

    # A WGSL emitter does NOT ship. The catalog still returns names; the SCORE says they are fallbacks.
    wgsl = mind.capability_confidence("emit a kernel as a webgpu compute shader")
    assert wgsl["confident"] is False

    # symbol search finds a real name without being told it
    assert any(n == "hierarchical_recall" for _, n in _symbols(["hierarchical"]))
    assert _symbols(["factoredblurthatdoesnotexist"]) == []

    print("OK: backlog_probe self-test passed (a shipped faculty scores confident; a missing one scores as a "
          "FALLBACK even though the catalog still names three capabilities; symbol search finds real names and "
          "reports nothing for invented ones)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        sys.exit(main())
