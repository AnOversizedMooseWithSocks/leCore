"""chunk_text must not lose facts on break-free input (the runaway-paragraph path).

chunk_text's own docstring names the failure mode: a fact split across two
chunks is retrievable from neither. The paragraph path honours that, but the
fallback for ONE oversized paragraph used to be plain `p[:max_chars]` windows
-- fixed, non-overlapping, the exact failure the docstring rejects. Input with
no blank lines at all (a pasted log, a minified file, a NIAH haystack) is one
runaway paragraph, so the WHOLE document took that path, and any fact
straddling a max_chars boundary ended up intact in no chunk. Measured on the
production harness (600-char chunks, 86 needle offsets): blob input lost 8/86
offsets (9%); the same prose with paragraph breaks lost 0/86. No downstream
ranker can recover a fact that no longer exists.

The fix strides max_chars - overlap on the degenerate path only, making any
fact shorter than `overlap` unloseable. This test sweeps a needle across
offsets in the same filler text presented both ways (deterministic, seeded)
and asserts: zero losses on both arms now, byte-identical paragraph-path
output regardless of overlap, and (pinning the mechanism, not just the
outcome) that overlap=0 still reproduces the original loss on the blob arm.
"""
import random

from holographic.caching_and_storage.holographic_knowledgestore import chunk_text

MAX_CHARS = 600
N_OFFSETS = 86
NEEDLE = "the vault access code is MAGENTA-4471-OTTER."


def _paragraphs(n_paras=80, para_chars=300, seed=7):
    """Deterministic filler as a list of ~300-char paragraphs."""
    rng = random.Random(seed)
    words = ["alpha", "signal", "ledger", "harbor", "quartz", "meadow",
             "cipher", "lantern", "orbit", "thicket", "velvet", "casing"]
    out = []
    for _ in range(n_paras):
        buf, total = [], 0
        while total < para_chars:
            w = rng.choice(words)
            buf.append(w)
            total += len(w) + 1
        out.append(" ".join(buf))
    return out


def _variants(offset_idx):
    """One document, two presentations, identical prose: the needle spliced at
    a word boundary inside one paragraph, swept across the corpus by
    offset_idx; returned as a break-free blob and as authored paragraphs."""
    paras = _paragraphs()
    p_i = offset_idx * (len(paras) - 1) // max(1, N_OFFSETS - 1)
    host = paras[p_i]
    cut = host.index(" ", (offset_idx * 37) % (len(host) // 2) + 1)
    paras[p_i] = host[:cut] + " " + NEEDLE + host[cut:]
    para_text = "\n\n".join(paras)                 # the author's own breaks
    blob_text = para_text.replace("\n\n", " ")     # same prose, no breaks at all
    return blob_text, para_text


def _lost(text, **kw):
    return not any(NEEDLE in c for c in chunk_text(text, max_chars=MAX_CHARS, **kw))


def test_no_needle_lost_on_blob_or_paragraphed_input():
    blob_lost, para_lost = [], []
    for i in range(N_OFFSETS):
        blob_text, para_text = _variants(i)
        if _lost(blob_text):
            blob_lost.append(i)
        if _lost(para_text):
            para_lost.append(i)
    assert not para_lost, f"paragraph path lost the needle at offsets {para_lost}"
    assert not blob_lost, f"degenerate path lost the needle at offsets {blob_lost}"


def test_overlap_zero_reproduces_the_original_loss():
    """The mechanism, pinned: with overlap disabled the fallback is the old
    fixed-window loop, and boundary-straddling needles die on blob input at
    ~needle_len/max_chars of offsets. If this ever stops failing, the sweep
    above has gone soft and is no longer testing anything."""
    blob_lost = [i for i in range(N_OFFSETS) if _lost(_variants(i)[0], overlap=0)]
    para_lost = [i for i in range(N_OFFSETS) if _lost(_variants(i)[1], overlap=0)]
    assert not para_lost, "paragraph path must never lose the needle, overlap or not"
    assert blob_lost, ("expected the overlap-free fixed-window fallback to destroy "
                       "boundary-straddling needles on break-free input")


def test_paragraph_path_byte_identical_regardless_of_overlap():
    """overlap only touches the runaway-paragraph fallback: on input whose
    paragraphs all fit under max_chars, output must not depend on it."""
    _, para_text = _variants(11)
    assert chunk_text(para_text, max_chars=MAX_CHARS, overlap=0) == \
        chunk_text(para_text, max_chars=MAX_CHARS, overlap=300)


def test_degenerate_chunks_respect_max_chars_and_cover_the_text():
    blob_text, _ = _variants(0)
    chunks = chunk_text(blob_text, max_chars=MAX_CHARS)
    assert chunks and all(len(c) <= MAX_CHARS for c in chunks)
    # coverage: every max_chars-aligned probe of the source appears in some chunk
    for pos in range(0, len(blob_text) - 40, MAX_CHARS):
        probe = blob_text[pos:pos + 40]
        assert any(probe in c for c in chunks), f"text near {pos} not covered"
    # tiny overlap values must not stall: stride stays positive
    assert chunk_text("x" * 5000, max_chars=100, overlap=10**6)
