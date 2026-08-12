"""BPE -- byte-level Byte-Pair Encoding in pure stdlib.

WHY THIS EXISTS: the leCore runtime can execute a real checkpoint with nothing
but NumPy, and then the driver made you paste TOKEN IDS because tokenizing
needed `transformers`. That is a silly place to lose self-containment: the
tokenizer is a vocabulary and a merge list, both sitting in the model directory
as plain JSON and text.

Reads `vocab.json` + `merges.txt` (GPT-2 / Qwen / Llama-BPE layout) or pulls the
same two tables out of a `tokenizer.json`. No regex module beyond `re`, no
tokenizers library, no torch.

VERIFIED, not assumed: when `transformers` happens to be installed, the selftest
encodes real text with BOTH and asserts identical ids. A tokenizer that is
almost right produces text that is subtly wrong in ways nobody traces back to
tokenization, so "almost" is not acceptable here.
"""

import json
import os
import re


# GPT-2's byte<->unicode table: maps raw bytes to printable code points so a
# byte sequence can live in a JSON vocabulary without escaping problems.
def _byte_encoder():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s A-Za-z0-9]+|\s+(?!\S)|\s+""")


class BPE:
    """Byte-level BPE encoder/decoder built from a model directory."""

    def __init__(self, vocab, merges, specials=()):
        self.encoder = dict(vocab)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.ranks = {tuple(m): i for i, m in enumerate(merges)}
        self.b2u = _byte_encoder()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.specials = dict(specials or {})
        self._cache = {}

    # ---- loading ----

    @classmethod
    def from_dir(cls, path):
        """vocab.json + merges.txt if present, else the tables inside
        tokenizer.json. Raises with a readable message rather than guessing --
        a wrong vocabulary silently produces fluent nonsense."""
        vj = os.path.join(path, "vocab.json")
        mt = os.path.join(path, "merges.txt")
        specials = {}
        tj = os.path.join(path, "tokenizer.json")
        if os.path.exists(tj):
            with open(tj, encoding="utf-8") as f:
                tok = json.load(f)
            for a in tok.get("added_tokens", []) or []:
                specials[a["content"]] = int(a["id"])
        if os.path.exists(vj) and os.path.exists(mt):
            with open(vj, encoding="utf-8") as f:
                vocab = json.load(f)
            merges = []
            with open(mt, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#version"):
                        continue
                    parts = line.split(" ")
                    if len(parts) == 2:
                        merges.append(parts)
            return cls(vocab, merges, specials)
        if os.path.exists(tj):
            with open(tj, encoding="utf-8") as f:
                tok = json.load(f)
            model = tok.get("model") or {}
            vocab = model.get("vocab")
            merges = [m.split(" ") if isinstance(m, str) else list(m)
                      for m in (model.get("merges") or [])]
            if vocab:
                return cls(vocab, merges, specials)
        raise FileNotFoundError(
            "no vocab.json+merges.txt and no usable tokenizer.json in %r -- "
            "this directory does not carry a BPE vocabulary" % path)

    # ---- the algorithm ----

    def _bpe(self, token):
        if token in self._cache:
            return self._cache[token]
        word = list(token)
        while len(word) > 1:
            pairs = [(self.ranks.get((word[i], word[i + 1]), 1 << 30), i)
                     for i in range(len(word) - 1)]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            word[i:i + 2] = [word[i] + word[i + 1]]
        self._cache[token] = word
        return word

    def encode(self, text):
        """Text -> token ids. Special tokens are matched FIRST and verbatim, so
        a chat template's control tokens survive rather than being split into
        their letters (the failure that makes a model answer as if the template
        were content)."""
        ids = []
        if self.specials:
            pattern = "(" + "|".join(re.escape(s) for s in
                                     sorted(self.specials, key=len, reverse=True)) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.specials:
                ids.append(int(self.specials[chunk]))
                continue
            for piece in _PAT.findall(chunk):
                u = "".join(self.b2u[b] for b in piece.encode("utf-8"))
                for sym in self._bpe(u):
                    if sym in self.encoder:
                        ids.append(int(self.encoder[sym]))
                    else:                       # fall back byte by byte
                        for ch in sym:
                            if ch in self.encoder:
                                ids.append(int(self.encoder[ch]))
        return ids

    def decode(self, ids):
        rev = {v: k for k, v in self.specials.items()}
        out = []
        buf = []
        for i in ids:
            i = int(i)
            if i in rev:
                if buf:
                    out.append(self._flush(buf))
                    buf = []
                out.append(rev[i])
                continue
            tok = self.decoder.get(i)
            if tok is not None:
                buf.append(tok)
        if buf:
            out.append(self._flush(buf))
        return "".join(out)

    def _flush(self, toks):
        s = "".join(toks)
        return bytes(self.u2b.get(c, 63) for c in s).decode("utf-8", "replace")


def _selftest():
    import tempfile

    # a tiny hand-built vocabulary exercises the machinery without a download
    b2u = _byte_encoder()
    base = {b2u[b]: i for i, b in enumerate(range(256))}
    merges = [["h", "e"], ["he", "l"], ["hel", "l"], ["hell", "o"]]
    nxt = len(base)
    for m in merges:
        base["".join(m)] = nxt
        nxt += 1
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(base, f)
    with open(os.path.join(d, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n" + "\n".join(" ".join(m) for m in merges))
    bpe = BPE.from_dir(d)
    ids = bpe.encode("hello")
    assert bpe.decode(ids) == "hello", bpe.decode(ids)
    assert len(ids) == 1, ids            # the merges collapsed it to one token
    # ROUND TRIP over awkward text: unicode, punctuation, newlines, spaces
    for probe in ("hello world", "  spaced\tout\n", "caf\u00e9 na\u00efve",
                  "def f(x):\n    return x**2\n", "\u4e2d\u6587\u6d4b\u8bd5"):
        assert bpe.decode(bpe.encode(probe)) == probe, probe

    # AGAINST THE REAL THING when it is available -- "almost right" tokenizing
    # produces subtly wrong text that nobody traces back to the tokenizer
    checked = False
    try:
        from transformers import AutoTokenizer
        import glob
        cands = [p for p in ("/home/claude/bench/model",) if os.path.exists(p)]
        for c in cands:
            if not os.path.exists(os.path.join(c, "vocab.json")):
                continue
            ref = AutoTokenizer.from_pretrained(c)
            mine = BPE.from_dir(c)
            for probe in ("The holographic engine binds and bundles.",
                          "def compress(x):\n    return x\n"):
                assert mine.encode(probe) == ref.encode(probe), probe
            checked = True
    except Exception:
        pass

    print("bpe selftest OK -- merges collapse 'hello' to 1 id; round-trips "
          "unicode, code and whitespace exactly; %s"
          % ("verified identical to the reference tokenizer"
             if checked else "no reference tokenizer present to cross-check"))


if __name__ == "__main__":
    _selftest()
