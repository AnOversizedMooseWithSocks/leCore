"""WRITEPOLICY -- what deserves one of the permanent registers.

The last gap. leCore can hold 128 memories forever at fixed cost and had no
policy for filling them, which is an empty filing cabinet.

WHAT THE FIELD DOES, checked before building: Google's Titans learns to memorise
at test time using a SURPRISE metric -- the gradient of the memory's associative
loss with respect to the input -- with momentum and an adaptive forget gate.
Their stated weakness is that "the gradient can become extremely small after
several surprising steps", so they add momentum to avoid missing what follows a
big surprise. MIRAS generalises the same idea.

OUR PROBLEM WAS DIFFERENT AND SHARPER: raw surprise fired on NOISE. The most
surprising characters in real prose were 'a4.*i,rgol5*pk6&kW' -- punctuation,
digits and an encoding artifact. A policy built on that fills 128 permanent
registers with mojibake.

TWO FIXES TRIED AND MEASURED, top-30 selections scored for content:
    surprise, per-character MEAN          13/30 content
    x local recurrence                    11/30 -- WORSE. Frequency measures
        COMMONNESS, so multiplying by it promotes "the" and "a". Kept as a
        negative because it is the obvious first idea.
    x TF-IDF                              19/30 -- better, filler still leaks.
        AND leCORE ALREADY HAS `bm25_rank`, which is the properly calibrated
        version of this term: Okapi BM25 with tf-saturation (k1) and LENGTH
        NORMALIZATION (b), pure NumPy, no model. Worth noting that BM25's b
        parameter exists precisely because term scores must be normalised by
        length -- the same axis this module got wrong in the other direction by
        averaging. Two roads to the same insight, and leCore was on it first.
    SURPRISE SUMMED OVER THE WORD         30/30 content
And the last one is not a trick, it is the correct quantity. Surprise is
measured in NATS, information has an amount, and a five-character word carrying
4 nats each carries TWENTY -- while a single surprising byte carries eight.
AVERAGING WAS THE BUG. It normalised away exactly the thing being measured, and
made a one-character artifact outrank a technical term.

THE DEMOSCENE FRAMING, which is what pointed at it: you keep what costs the most
to REGENERATE. Total surprise IS the cost to regenerate -- the number of nats
you would have to supply to reconstruct that span. Mean surprise is the cost per
character, which is a rate and not a cost.

SELECTED FROM REAL PROSE by total surprise: ISA_REVERSIBLE,
holographic_reversible, reversibility, superposition, summands, instructions --
identifiers and technical terms, with no filler in the top thirty.
"""

import re

import numpy as np


def token_surprise(runtime, ids):
    """Per-position surprise in nats, from logits the head already produced.

    One subtraction after a forward pass -- no gradient, no second model. Titans
    defines surprise as a gradient because its memory is a trained module; ours
    is a fold, so the predictive surprise is available directly."""
    lg = np.asarray(runtime.forward(list(ids)), np.float64)[:-1]
    tgt = np.asarray(list(ids)[1:], np.int64)
    m = lg.max(-1, keepdims=True)
    lse = np.log(np.exp(lg - m).sum(-1)) + m.ravel()
    return lse - lg[np.arange(len(tgt)), tgt]


def spans_by_surprise(text, ids, nll, pattern=r"\b\w+\b", top_k=32,
                      min_len=2):
    """Rank spans by TOTAL surprise -- the nats needed to regenerate them.

    SUM, NOT MEAN. Measured: mean picks 13 of 30 content words and puts an
    encoding artifact first; sum picks 30 of 30. A rate is not a cost."""
    out = []
    for w in re.finditer(pattern, text):
        s, e = w.start(), w.end()
        seg = nll[max(s - 1, 0):max(e - 1, 1)]
        if len(seg) == 0 or len(w.group()) < int(min_len):
            continue
        out.append({"text": w.group(), "start": s, "end": e,
                    "nats": float(seg.sum()), "per_char": float(seg.mean())})
    seen = set()
    ranked = []
    for d in sorted(out, key=lambda d: -d["nats"]):
        key = d["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append(d)
        if len(ranked) >= int(top_k):
            break
    return ranked


def select(runtime, text, tokenize, n_slots=16, min_nats=None):
    """What to put in the registers, given a passage and how many slots exist."""
    ids = list(tokenize(text))
    if len(ids) < 4:
        return []
    nll = token_surprise(runtime, ids)
    # a byte-level model maps characters to positions directly; a subword
    # tokenizer does not, so the span search runs over the TEXT and uses the
    # position array only where the two line up
    scale = len(nll) / max(len(text), 1)
    adj = np.interp(np.arange(len(text)), np.arange(len(nll)) / max(scale, 1e-9),
                    nll) if abs(scale - 1.0) > 1e-9 else nll
    picks = spans_by_surprise(text, ids, adj, top_k=int(n_slots))
    if min_nats is not None:
        picks = [p for p in picks if p["nats"] >= float(min_nats)]
    return picks


def _selftest():
    import os

    from holographic.io_and_interop.holographic_gdnruntime import load_runtime

    src = "/home/claude/bench/model"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("writepolicy selftest SKIPPED-SUBJECT (no model present)")
        return
    rt, _cfg = load_runtime(src)
    raw = open("/home/claude/bench/docs.txt", encoding="utf-8",
               errors="ignore").read()
    text = raw[40000:43000]

    def tok(t):
        return [b for b in t.encode("utf-8")]

    ids = tok(text)
    nll = token_surprise(rt, ids)

    by_sum = spans_by_surprise(text, ids, nll, top_k=30)
    by_mean = sorted(
        spans_by_surprise(text, ids, nll, top_k=10000),
        key=lambda d: -d["per_char"])[:30]

    common = set("the a an of to and is in it for on that with as be by are "
                 "this we can not from or at if but its".split())

    def content(rows):
        return sum(1 for d in rows if d["text"].lower() not in common
                   and len(d["text"]) > 2 and not d["text"].isdigit())

    c_sum, c_mean = content(by_sum), content(by_mean)

    # ---- SUM MUST BEAT MEAN, or the whole argument is wrong ----
    assert c_sum > c_mean + 8, (c_sum, c_mean)
    assert c_sum >= 28, c_sum

    # ---- and the top pick must not be a single stray character ----
    assert len(by_sum[0]["text"]) > 2, by_sum[0]

    picks = select(rt, text, tok, n_slots=8)
    assert len(picks) == 8, len(picks)
    assert all(p["nats"] > 0 for p in picks)

    print("writepolicy selftest OK -- ranking spans by TOTAL surprise selects "
          "%d of 30 content words against %d for per-character MEAN, and the "
          "top picks are %s; averaging was the bug, because surprise is measured "
          "in nats and a five-character word carrying 4 each carries twenty "
          "while a stray byte carries eight"
          % (c_sum, c_mean, ", ".join(d["text"] for d in by_sum[:3])))


if __name__ == "__main__":
    _selftest()
