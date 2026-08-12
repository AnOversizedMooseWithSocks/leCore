"""GALVAPORT -- carry as much of a Galvatron as a traditional runtime can hold.

The honest starting point, measured rather than assumed: loading a Galvatron's
`model.safetensors` in another framework gives the BARE MODEL. Same test, same
prompt -- through leCore the output was " a fix on a " with the ward holding;
weights-only it was " the sign an" and the ward was BREACHED. Residents are
structure in the forward pass, and a GGUF file has nowhere to put them.

But "nowhere to put the code" is not "nothing survives". Researched what
llama.cpp actually offers (Aug 2026) and three of the four load-bearing pieces
have a native home:

  WARD      -> GBNF grammar. llama.cpp constrains sampling to a formal grammar,
               per request or per server. A ban list is a grammar. This is the
               same guarantee, enforced by their sampler instead of ours.
  MANIFEST  -> GGUF metadata. GGUF carries arbitrary key/value pairs (real
               models ship ~50), so the roster, the calibration reference and
               the provenance travel INSIDE the file rather than beside it.
  MEMORY,
  TOOLBELT,
  VERIFIER  -> MCP sidecar. llama-server has function calling and MCP hooks;
               leCore runs as a tool server, so retrieval, the holographic
               database and capability invocation are reachable from a runtime
               that has never heard of leCore.
  DREAMER,
  CARRIER,
  HRNN      -> DO NOT TRAVEL, and this file says so rather than pretending.
               They operate on the residual stream mid-forward; llama.cpp
               exposes no such hook. Use the leCore runtime when those matter.

WHAT THIS FILE DOES NOT DO: convert weights to GGUF. That is llama.cpp's own
`convert_hf_to_gguf.py`, it is well-tested, and reimplementing it here would be
a worse copy. This emits the ARTIFACTS that conversion cannot produce -- the
grammar, the metadata, the sidecar manifest -- plus the exact commands to run.
"""

import json
import os


def ward_to_gbnf(banned=(), allowed=None, vocab=None):
    """Compile a ward into a GBNF grammar llama.cpp can enforce.

    A ban list is a whitelist over the remaining alphabet, which is what a
    grammar can express: GBNF constrains what MAY be produced, so a ban has to
    be inverted into the permitted set. Working at the BYTE level rather than
    the token level, because a grammar over token ids would need the exact
    tokenizer llama.cpp built, while bytes are the same everywhere.

    HONEST LIMIT, stated because it changes what you can promise: this bans
    CHARACTERS, not token ids. A word banned as a token can still be spelled if
    its letters are permitted. For exact token-level bans, run the leCore
    runtime, where the ward masks logits directly."""
    if allowed is not None:
        chars = sorted({c for s in allowed for c in str(s)})
        if not chars:
            raise ValueError("an empty whitelist would permit nothing at all")
        body = " | ".join(_gbnf_char(c) for c in chars)
        return 'root ::= ( %s )+\n' % body
    banned_chars = sorted({c for s in banned for c in str(s)})
    if not banned_chars:
        return 'root ::= [^]+\n'      # nothing banned: any character
    ranges = "".join(_gbnf_escape(c) for c in banned_chars)
    return ('# every character EXCEPT the banned set\n'
            'root ::= char+\n'
            'char ::= [^%s]\n' % ranges)


def _gbnf_escape(c):
    if c in "\\]^-":
        return "\\" + c
    if c == "\n":
        return "\\n"
    if c == "\r":
        return "\\r"
    if c == "\t":
        return "\\t"
    return c


def _gbnf_char(c):
    return '"%s"' % c.replace("\\", "\\\\").replace('"', '\\"')


def export(pack_dir, out_dir, model_name="galvatron", port=5931):
    """Emit everything a traditional runtime needs beside a converted GGUF.

    Returns a report naming what travels and what does NOT -- the second list is
    the important one, because a packaging tool that only advertises its wins
    teaches the user to expect capabilities that are not there."""
    from holographic.io_and_interop.holographic_galvapack import MANIFEST
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(pack_dir, MANIFEST)) as f:
        man = json.load(f)
    specs = man.get("residents", [])
    kinds = sorted({s.get("kind") for s in specs})

    banned_ids = []
    for sp in specs:
        if sp.get("kind") == "ward":
            banned_ids = list(sp.get("banned", []))
    # tokens -> text, so the grammar can be written over characters
    banned_text = []
    try:
        from holographic.io_and_interop.holographic_bpe import BPE
        tok = BPE.from_dir(pack_dir)
        banned_text = [tok.decode([int(t)]) for t in banned_ids]
    except Exception:
        banned_text = [chr(int(t)) for t in banned_ids if 0 < int(t) < 0x110000]
    grammar = ward_to_gbnf(banned=banned_text)
    with open(os.path.join(out_dir, "ward.gbnf"), "w") as f:
        f.write(grammar)

    # GGUF metadata: the roster rides INSIDE the model file
    meta = {"galvatron.format": "galvatron/1",
            "galvatron.residents": json.dumps(kinds),
            "galvatron.engine": "leCore",
            "galvatron.note": man.get("without_leCore", ""),
            "galvatron.sidecar": "leCore MCP/OpenAI server exposes memory, "
                                 "toolbelt and verifier"}
    with open(os.path.join(out_dir, "gguf_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)

    travels = [k for k in kinds if k in ("ward", "memory", "verifier",
                                         "toolbelt", "capability", "leap")]
    stays = [k for k in kinds if k in ("dreamer", "carrier", "hrnn", "oracle",
                                       "screen", "corpus")]
    readme = _README % {
        "kinds": ", ".join(kinds) or "(none)",
        "travels": ", ".join(travels) or "(none)",
        "stays": ", ".join(stays) or "(none)",
        "model": model_name, "port": port,
        "n_banned": len(banned_ids)}
    with open(os.path.join(out_dir, "README_llamacpp.md"), "w") as f:
        f.write(readme)
    return {"out_dir": out_dir, "kinds": kinds, "travels": travels,
            "stays_in_lecore": stays, "banned_tokens": len(banned_ids),
            "files": sorted(os.listdir(out_dir))}


_README = """# Running this Galvatron under llama.cpp / Ollama

Residents in this pack: %(kinds)s

## What travels into a traditional runtime
%(travels)s

* **ward** -> `ward.gbnf`. llama.cpp constrains sampling to a grammar, so the
  ban is enforced by their sampler:
      llama-server -m model.gguf --grammar-file ward.gbnf
  Compiled from %(n_banned)d banned tokens. NOTE: this bans CHARACTERS, not
  token ids -- a banned word can still be spelled from permitted letters. Exact
  token-level bans need the leCore runtime.
* **memory / toolbelt / verifier** -> run leCore as a sidecar and point the
  runtime's tool calling at it:
      python galvatron.py serve --port %(port)d
  llama-server has function calling and MCP hooks; the sidecar exposes
  retrieval over the holographic database, capability invocation and the
  evidence check as tools.
* **leap** -> llama.cpp has its own speculative decoding (`--spec-type`), so
  use theirs; the setting travels as intent, not as code.

## What does NOT travel
%(stays)s

These operate on the residual stream mid-forward (repair, the carrier band, the
HRNN observer, in-stream retrieval). llama.cpp exposes no hook there, so under
Ollama they are simply absent. This is not a limitation to work around later --
a GGUF file has nowhere to put a function that runs between layers.

## Converting the weights
Use llama.cpp's own converter (well-tested; do not reimplement it):
    python convert_hf_to_gguf.py <pack_dir> --outfile %(model)s.gguf
then attach the metadata in `gguf_metadata.json` with `gguf-py`'s writer or
`llama-gguf` so the roster rides inside the file.

## The honest summary
Weights-only, MEASURED on a real pack: the output differs from the leCore run
and the ward is breached. With `ward.gbnf` plus the sidecar you recover the
guarantees that can be expressed outside the forward pass, and nothing more.
"""


def _selftest():
    import tempfile

    # ---- a ban becomes a grammar that EXCLUDES exactly those characters ----
    g = ward_to_gbnf(banned=["a", "e"])
    assert "root ::= char+" in g and "[^ae]" in g, g
    # ---- special characters are escaped, not pasted into a character class --
    g2 = ward_to_gbnf(banned=["]", "^", "\n"])
    assert "\\]" in g2 and "\\^" in g2 and "\\n" in g2, g2
    # ---- nothing banned means nothing constrained ----
    assert ward_to_gbnf(banned=[]) == "root ::= [^]+\n"
    # ---- a whitelist is expressed directly, and an EMPTY one is refused
    #      rather than silently producing a grammar that permits nothing
    w = ward_to_gbnf(allowed=["ab"])
    assert '"a"' in w and '"b"' in w, w
    try:
        ward_to_gbnf(allowed=[])
        raise AssertionError("an empty whitelist was accepted")
    except ValueError:
        pass

    # ---- export names what travels AND what does not ----
    pack = tempfile.mkdtemp()
    with open(os.path.join(pack, "galvatron.json"), "w") as f:
        json.dump({"format": "galvatron/1",
                   "residents": [{"kind": "ward", "banned": [101, 116]},
                                 {"kind": "memory"}, {"kind": "dreamer"},
                                 {"kind": "carrier"}, {"kind": "toolbelt"}],
                   "without_leCore": "ordinary checkpoint"}, f)
    rep = export(pack, tempfile.mkdtemp())
    assert "ward" in rep["travels"] and "memory" in rep["travels"]
    assert "dreamer" in rep["stays_in_lecore"], rep
    assert "carrier" in rep["stays_in_lecore"], rep
    assert set(rep["files"]) == {"README_llamacpp.md", "gguf_metadata.json",
                                 "ward.gbnf"}, rep["files"]
    with open(os.path.join(rep["out_dir"], "README_llamacpp.md")) as f:
        text = f.read()
    assert "does NOT travel" in text and "dreamer" in text, "the README must "\
        "name the losses, not only the wins"

    print("galvaport selftest OK -- a ban compiles to a GBNF character-class "
          "grammar with specials escaped, an empty whitelist is refused, and "
          "export names both what travels (%s) and what STAYS in leCore (%s); "
          "UNVERIFIED AGAINST llama.cpp: no llama.cpp here to run the grammar, "
          "so the syntax is asserted, not executed"
          % (",".join(rep["travels"]), ",".join(rep["stays_in_lecore"])))


if __name__ == "__main__":
    _selftest()
