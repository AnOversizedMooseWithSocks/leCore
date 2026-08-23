"""Re-test the index on data that was NOT chosen to flatter it.

EVERY SCALE RESULT SO FAR USED `hash_atom("doc%d")`: one synthetic near-orthogonal atom per
document, queried with an EXACT COPY of the stored vector. That is the friendliest fixture that
exists, and the capacity rule (g <= D/9) and the dimension rule (D >= 9*sqrt(K)) were fitted on
it. Real text breaks all three assumptions at once:

  * documents are BAGS OF SHARED TOKENS, so their vectors are CORRELATED, not near-orthogonal
  * token frequency is ZIPFIAN, so a few tokens dominate every vector and pull them together
  * documents are near-duplicates of each other far more often than random vectors ever are
  * a query is a FRAGMENT, not the document -- so the target's own score is much weaker

This harness builds the corpus out of THIS REPOSITORY's own prose and code comments -- ~740
modules and the markdown docs -- chunked into passages, and queries each passage with a
CONTIGUOUS SPAN HELD OUT OF IT, which is what a person actually types. Then it re-runs the two
rules and reports whether they survive.
"""
import glob, re
import numpy as np
import holographic.agents_and_reasoning.holographic_hashatom as HA

STOP = set(("the a an of to and or is are was were be been being in on for with that this it as "
            "by at from not but if then than so its it's which what when how you your we our can "
            "could use used using into over under about only just also more most other some such "
            "no nor too very s t don now d ll m o re ve y").split())


def tokens(text):
    return [w for w in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if w not in STOP]


def build_corpus(target_chunks=4000, chunk_words=90, stride=70):
    """Real prose: every module docstring/comment block plus every markdown doc, chunked.

    Overlapping windows (stride < chunk) are DELIBERATE: they manufacture the near-duplicate
    neighbours a real archive has and a random fixture never does. If the index can only tell
    apart documents that share no vocabulary, it is not an index.
    """
    blobs = []
    for path in sorted(glob.glob("holographic/*/*.py")):
        src = open(path, encoding="utf-8", errors="ignore").read()
        for m in re.finditer(r'"""(.{200,4000}?)"""', src, re.S):
            blobs.append(m.group(1))
    for path in sorted(glob.glob("*.md")) + sorted(glob.glob("docs/*.md")):
        blobs.append(open(path, encoding="utf-8", errors="ignore").read()[:200000])

    chunks = []
    for b in blobs:
        w = tokens(b)
        for i in range(0, max(1, len(w) - chunk_words + 1), stride):
            c = w[i:i + chunk_words]
            if len(c) >= 40:
                chunks.append(c)
            if len(chunks) >= target_chunks:
                return chunks
    return chunks


def corpus_stats(chunks):
    from collections import Counter
    cnt = Counter(t for c in chunks for t in set(c))
    freqs = np.array(sorted(cnt.values())[::-1], dtype=float)
    # Zipf slope: log-rank vs log-frequency. ~-1 is natural language.
    r = np.arange(1, min(len(freqs), 2000) + 1)
    slope = float(np.polyfit(np.log(r), np.log(freqs[:len(r)]), 1)[0])
    lens = np.array([len(set(c)) for c in chunks])
    return dict(K=len(chunks), vocab=len(cnt), zipf_slope=slope,
                len_med=int(np.median(lens)), len_iqr=(int(np.percentile(lens, 25)),
                                                       int(np.percentile(lens, 75))))


def encode_all(chunks, D):
    return np.stack([HA.encode_hash(sorted(set(c)), D) for c in chunks])


def held_out_queries(chunks, D, span=12, seed=0):
    """A CONTIGUOUS span lifted out of the passage -- what a person types, not a bag sample.

    Contiguity matters: a random token sample spreads across the whole passage and is far easier
    than a phrase, so sampling randomly would quietly re-flatter the index.
    """
    rng = np.random.default_rng(seed)
    Q, T = [], []
    for i, c in enumerate(chunks):
        if len(c) <= span:
            continue
        s = int(rng.integers(0, len(c) - span))
        Q.append(HA.encode_hash(sorted(set(c[s:s + span])), D, normalise=False))
        T.append(i)
    return np.stack(Q), np.array(T)


def two_level(V, g):
    return np.stack([V[i:i + g].sum(0) for i in range(0, len(V), g)])


def walk_acc(V, top, g, Q, T, beam):
    hit = 0
    for q, t in zip(Q, T):
        cand = np.argsort(top @ q)[::-1][:beam]
        rows = np.concatenate([np.arange(c * g, min((c + 1) * g, len(V))) for c in cand])
        hit += int(rows[int(np.argmax(V[rows] @ q))]) == t
    return hit / len(T)


def flat_acc(V, Q, T):
    return float(np.mean([int(np.argmax(V @ q)) == t for q, t in zip(Q, T)]))


if __name__ == "__main__":
    chunks = build_corpus()
    st = corpus_stats(chunks)
    print("REAL CORPUS FROM THIS REPO")
    print("   chunks %d | vocab %d | Zipf slope %.2f (natural language is ~-1) | "
          "distinct tokens/chunk median %d IQR %s"
          % (st["K"], st["vocab"], st["zipf_slope"], st["len_med"], st["len_iqr"]))

    D0 = 512
    V0 = encode_all(chunks, D0)
    off = np.abs(V0 @ V0.T - np.eye(len(V0)))
    print("   MEAN off-diagonal |cos| between real chunks: %.4f  (synthetic hash atoms: %.4f)"
          % (off.mean(), float(np.abs(np.stack([HA.hash_atom("d%d" % i, D0) for i in range(400)])
                                      @ np.stack([HA.hash_atom("d%d" % i, D0) for i in range(400)]).T
                                      - np.eye(400)).mean())))
    print("   MAX  off-diagonal |cos|: %.4f  -- near-duplicates are REAL here\n" % off.max())

    Q0, T0 = held_out_queries(chunks, D0)
    K = len(V0)
    print("QUERY = a contiguous 12-token span held out of the passage (not the passage vector)")
    print("   flat scan accuracy at D=512: %.3f over %d queries\n" % (flat_acc(V0, Q0, T0), len(T0)))

    print("RULE UNDER TEST 1 -- does g <= D/9 still predict the knee on real text?")
    print("   D     cap=D/9   g     leaves/cell  beam4   beam16   flat")
    for D in (256, 512, 1024):
        V = encode_all(chunks, D)
        Q, T = held_out_queries(chunks, D)
        fa = flat_acc(V, Q, T)
        for g in (int(D / 9), int(round(np.sqrt(K))), 64):
            g = max(2, g)
            top = two_level(V, g)
            print("   %-5d %-9d %-5d %-12d %-7.3f %-8.3f %.3f"
                  % (D, D // 9, g, g, walk_acc(V, top, g, Q, T, 4),
                     walk_acc(V, top, g, Q, T, 16), fa))
    print("\nRULE UNDER TEST 2 -- D >= 9*sqrt(K) demands D >= %d here (K=%d)"
          % (int(9 * np.sqrt(K)), K))
