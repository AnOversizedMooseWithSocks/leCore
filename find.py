#!/usr/bin/env python3
"""
find.py -- "search before you build": describe what you need, get the modules/functions that do it.

Runs over the CI-generated REFERENCE.md with nothing but the standard library. It doesn't understand
code -- it ranks entries by how many of YOUR words appear in their descriptions (keyword overlap).
Because the reference is plain English, that's enough to surface the thing you didn't know existed.

    python find.py "search a big pile of vectors for the nearest ones"
    python find.py --ref /path/to/REFERENCE.md "cache a slow computation"

HONEST LIMIT: it only matches words you and the docstring BOTH use. If the docstring says "recall"
and you search "nearest", it can miss. The fix isn't a fancier search -- it's a capability TAG per
module in the reference (see the note at the bottom), so you query the concept, not the wording.
"""
import os, re, sys
from collections import Counter

# words too common to carry a search signal (so "the field" doesn't match half the codebase)
STOP = set("the a an of to for and or in on at is are be by with from into as it that this we our you "
           "your i so -- them then which who what when where how not no run one".split())


def tokenize(text):
    """lowercase words of 3+ letters, minus the stop words"""
    return [w for w in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if w not in STOP]


def load_reference(path):
    """Parse REFERENCE.md into flat searchable entries: one per module (its docstring), one per function."""
    entries, module, doc = [], None, []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^###\s+`?(holographic_[a-z0-9_]+)\.py`?", line)
        if m:                                        # a new module heading: flush the previous docstring
            if module and doc:
                entries.append({"label": module, "text": " ".join(doc)})
            module, doc = m.group(1), []
            continue
        if module and line.startswith(">"):          # a docstring line (blockquote) for this module
            doc.append(line.lstrip("> ").rstrip())
            continue
        fn = re.match(r"^-\s+`def\s+([a-z0-9_]+)\(.*?\)`\s*--\s*(.*)", line)
        if fn and module:                            # a public function -> its own searchable entry
            entries.append({"label": f"{module}.{fn.group(1)}()", "text": f"{fn.group(1)} {fn.group(2)}"})
    if module and doc:
        entries.append({"label": module, "text": " ".join(doc)})
    return entries


def score(query_words, entry_words):
    """count overlapping words; give half credit when a query word is CONTAINED in an entry word
    (so 'search' still matches 'searching')."""
    ec = Counter(entry_words)
    hits = 0.0
    for w in set(query_words):
        if w in ec:
            hits += 1.0
        elif any(w in e for e in ec):
            hits += 0.5
    return hits


def find_reference(argv):
    """--ref PATH, or REFERENCE.md next to this script, or in the current directory."""
    if "--ref" in argv:
        i = argv.index("--ref")
        return argv[i + 1], argv[:i] + argv[i + 2:]
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "REFERENCE.md"), "REFERENCE.md"):
        if os.path.exists(cand):
            return cand, argv
    return "REFERENCE.md", argv


def main():
    ref, argv = find_reference(sys.argv[1:])
    query = " ".join(argv) or "nearest neighbour search recall"
    entries = load_reference(ref)
    qwords = tokenize(query)

    ranked = [(score(qwords, tokenize(e["text"])), e) for e in entries]
    ranked = sorted([r for r in ranked if r[0] > 0], key=lambda r: r[0], reverse=True)

    print(f'\nsearching {len(entries)} entries in {ref} for: "{query}"\n')
    for s, e in ranked[:12]:
        print(f'  [{s:>3.1f}] {e["label"]:<40} {e["text"][:80]}')
    if not ranked:
        print("  (no keyword overlap -- try different words, or add a capability tag to the reference)")


if __name__ == "__main__":
    main()
