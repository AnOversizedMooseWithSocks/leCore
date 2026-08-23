"""A HARD corpus, and an honest look at how hard it is, before any retrieval number is quoted.

THE COMPLAINT THIS ANSWERS: every benchmark so far used `hash_atom("doc%d")` -- one fresh
near-orthogonal atom per document. That fixture is the friendliest input this code could ever
receive: no shared vocabulary, no length variation, no duplicates, and the right answer is the
only vector anywhere near the query. The rule D >= 9*sqrt(K) was derived under exactly those
conditions, so it is a claim about the fixture until it is tested on something real.

THIS CORPUS: every module in the tree, split into overlapping passages of real source and prose.
It is deliberately nasty in the ways real corpora are nasty -- heavy vocabulary sharing (the same
twenty identifiers appear everywhere), Zipfian term frequency, lengths spanning an order of
magnitude, and genuine NEAR-DUPLICATES (boilerplate headers, repeated selftest scaffolding).

QUERIES ARE HELD OUT, NOT THE DOCUMENT ITSELF. Querying with the whole document is a lookup, not
a retrieval: the target is the only vector containing every term, so any method wins. Here a
query is a random subset of one passage's terms, and crucially the SUBSET IS SMALL, which is the
regime where shared vocabulary actually competes.
"""
import glob
import os
import pathlib
import re
from collections import Counter

import numpy as np

STOP = set("""the a an of to and or is not but if then than so its it as by be from at in on for
with that this it can use used using we our you your what when how which are was were will would
should could may might must have has had do does did been being there their they them he she""".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if w not in STOP]


def load_passages(target=6000, lo=30, hi=140, stride=3):
    """Overlapping windows over real source. Overlap is deliberate: it manufactures the
    near-duplicate pairs a clean fixture never has, and those are what break a coarse index."""
    # STRATIFIED, NOT TRUNCATED. Reading files in order and stopping at `target` takes an
    # ALPHABETICAL PREFIX of the tree: measured, that dropped three entire families
    # (sampling_and_signal, simulation_and_physics, unified) and 26% of modules from every
    # experiment in this arc. Collect per file, then INTERLEAVE, so a truncated corpus is a
    # sample of the whole tree rather than its first chunk.
    per_file = []
    for path in sorted(glob.glob("holographic/*/holographic_*.py")):
        try:
            src = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        toks = tokens(src)
        name = os.path.basename(path)[:-3]
        chunks = []
        for i in range(0, max(1, len(toks) - lo), hi // stride):
            w = toks[i:i + hi]
            if len(w) >= lo:
                chunks.append((name + "#%d" % i, w))
        if chunks:
            per_file.append(chunks)
    # Even STRIDE over files, so a target smaller than the file count still spans the tree.
    if per_file and target < len(per_file):
        step = len(per_file) / float(target)
        per_file = [per_file[int(i * step)] for i in range(target)]
    out = []
    depth = 0
    while len(out) < target and per_file:
        wrote = False
        for chunks in per_file:
            if depth < len(chunks):
                out.append(chunks[depth]); wrote = True
                if len(out) >= target:
                    break
        if not wrote:
            break
        depth += 1
    out = out[:target]
    # COVERAGE ASSERTION, not a target count. A truncated sorted list looks exactly like a
    # complete corpus from the inside; this is what makes that failure loud.
    # PATH SPLITTING MUST BE OS-AGNOSTIC. glob returns backslashes on Windows, so split("/")[1]
    # raised IndexError there and this loader silently failed -- which sent a GPU benchmark to a
    # synthetic corpus for the two rows it existed to measure. Use the path API, not the separator.
    def _family(f):
        return pathlib.PurePath(f).parts[1]

    def _stem(f):
        return pathlib.PurePath(f).stem

    fams = {_family(f) for f in glob.glob("holographic/*/holographic_*.py")}
    got = {_family(f) for f in glob.glob("holographic/*/holographic_*.py")
           if _stem(f) in {n.split("#")[0] for n, _ in out}}
    if fams - got:
        raise RuntimeError(
            "corpus covers %d of %d families -- MISSING %s. A biased sample is worse than a small "
            "one because it looks complete. Raise `target` or widen the stride."
            % (len(got), len(fams), sorted(fams - got)))
    return out


def difficulty_report(docs):
    """State how hard the corpus is BEFORE quoting any retrieval score on it."""
    lens = np.array([len(set(t)) for _, t in docs])
    df = Counter()
    for _, t in docs:
        df.update(set(t))
    vocab = len(df)
    freqs = np.array(sorted(df.values(), reverse=True))
    top20 = freqs[:20].sum() / freqs.sum()

    # Near-duplicate rate: Jaccard over a random sample of pairs, plus each doc's nearest peer.
    rng = np.random.default_rng(0)
    sets = [set(t) for _, t in docs]
    pairs = rng.integers(0, len(docs), size=(4000, 2))
    jac = []
    for a, b in pairs:
        if a == b:
            continue
        u = len(sets[a] | sets[b])
        jac.append(len(sets[a] & sets[b]) / u if u else 0.0)
    jac = np.array(jac)

    sample = rng.choice(len(docs), min(300, len(docs)), replace=False)
    nn = []
    for i in sample:
        best = 0.0
        for j in rng.choice(len(docs), 400, replace=False):
            if j == i:
                continue
            u = len(sets[i] | sets[j])
            if u:
                best = max(best, len(sets[i] & sets[j]) / u)
        nn.append(best)
    nn = np.array(nn)

    print("CORPUS DIFFICULTY (stated before any retrieval number)")
    print("  passages                        %d" % len(docs))
    print("  distinct terms per passage      median %d, range %d..%d"
          % (np.median(lens), lens.min(), lens.max()))
    print("  vocabulary                      %d terms" % vocab)
    print("  top-20 terms are               %.1f%% of all term occurrences (Zipf, heavy head)"
          % (100 * top20))
    print("  random pair Jaccard             median %.3f, 95th pct %.3f" % (np.median(jac), np.percentile(jac, 95)))
    print("  NEAREST-NEIGHBOUR Jaccard       median %.3f, 90th pct %.3f  <- near-duplicates"
          % (np.median(nn), np.percentile(nn, 90)))
    print("  passages with a peer >0.5 Jaccard  %.1f%%" % (100 * (nn > 0.5).mean()))
    return dict(K=len(docs), vocab=vocab, nn_median=float(np.median(nn)))


if __name__ == "__main__":
    docs = load_passages()
    difficulty_report(docs)
