"""Slice a REAL checkpoint down to something small enough to send.

Moose asked whether any LLM is small enough to upload here and still advanced
enough to test with. Measured, the answer is no -- not even the smallest:
    SmolLM2-135M    270 MB bf16,  74 MB at 4-bit
    Gemma 3 270M    540 MB,      148 MB
    Qwen3-0.6B     1200 MB,      330 MB
    Qwen3.5-0.8B   1746 MB,      480 MB   (his model)
against an upload budget of roughly 30 MB.

BUT A SLICE FITS, and stays genuinely trained. Two cuts:
    LAYERS     keep the first N. The result is a real, runnable transformer
               whose weights were trained -- lobotomised, so its perplexity is
               poor, but every tensor is a TRAINED tensor with a real spectrum,
               real heavy tails, and real activation geometry.
    VOCABULARY keep the first V rows of the embedding and head. This is the cut
               that matters: the embedding is usually MOST of a small model
               (49152 x 576 in SmolLM2), so slicing layers alone barely helps.

WHY A TRAINED SLICE BEATS A SYNTHETIC FIXTURE, and why both are needed:
build_mini_qwen gives STRUCTURE -- the right tensor names, layer pattern, tied
embeddings, bf16, vision tower -- and it caught eight structural defects. What
it cannot give is TRAINED STATISTICS. Every guard in this pipeline reverted its
bakes on the synthetic fixture because random weights have no structure for a
VSA circuit to exploit; whether they revert on trained weights is a different
question and needs trained weights to answer.

WHAT THE SLICE IS HONEST ABOUT: it is not the model. Its perplexity is not the
model's perplexity and never will be. It is for testing whether the PIPELINE
does the right thing to real trained tensors -- which is the only question that
has actually been failing.
"""

import json
import os
import shutil
import sys

import numpy as np

# RUNNABLE FROM ANYWHERE. This is a tool people invoke by path -- from the
# assimilation folder, from a shell, from a shortcut -- and importing the engine
# only works if the repo root is on sys.path. Requiring the caller to be in the
# right directory is a footgun disguised as a convention.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def slice_model(src_dir, out_dir, layers=4, vocab=4096, dtype=None,
                drop_vision=True):
    """Keep the first `layers` blocks and `vocab` token rows. Returns a report."""
    from holographic.io_and_interop.holographic_unicron import (
        load_safetensors, save_safetensors, source_dtypes)

    # ACCEPT A FILE OR A DIRECTORY. Pointing at model.safetensors is the
    # obvious thing to type, and refusing it with a confusing error is a worse
    # answer than simply handling it.
    if os.path.isfile(src_dir):
        src_file = src_dir
        src_dir = os.path.dirname(os.path.abspath(src_dir)) or "."
        files = [os.path.basename(src_file)]
    else:
        files = [f for f in sorted(os.listdir(src_dir))
                 if f.endswith(".safetensors")]
    if not files:
        raise ValueError(
            "no .safetensors found in %r -- point this at a model DIRECTORY "
            "(or directly at a .safetensors file). Found: %s"
            % (src_dir, ", ".join(sorted(os.listdir(src_dir))[:8]) or "nothing"))
    w = {}
    for f in files:
        w.update(load_safetensors(os.path.join(src_dir, f)))
    dts = source_dtypes(src_dir)

    keep = {}
    dropped_layers = set()
    dropped_vision = 0
    for name, val in w.items():
        a = np.asarray(val)
        # THE VISION TOWER IS NAMED "blocks.", NOT "layers.", so a layer slice
        # leaves it entirely intact -- 153 of 488 tensors on a real Qwen3.5, and
        # the reason a first attempt only shrank 2.6x. A text test model does
        # not need it, and dropping it is explicit rather than incidental.
        if drop_vision and (".visual." in name or name.startswith("visual.")):
            dropped_vision += 1
            continue
        if "layers." in name:
            try:
                L = int(name.split("layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                keep[name] = a
                continue
            if L >= int(layers):
                dropped_layers.add(L)
                continue
        # slice the vocabulary on any tensor whose first axis IS the vocabulary
        if (name.endswith("embed_tokens.weight") or "lm_head" in name) \
                and a.ndim == 2 and a.shape[0] > int(vocab):
            a = a[:int(vocab)]
        keep[name] = np.ascontiguousarray(a)
    if drop_vision:
        # a config that still advertises a vision tower will send a loader
        # looking for tensors that are no longer there
        pass

    os.makedirs(out_dir, exist_ok=True)
    save_safetensors(os.path.join(out_dir, "model.safetensors"),
                     keep, dtypes={k: (dtype or dts.get(k, "F32"))
                                   for k in keep})

    # ---- the config must MATCH the slice, or nothing will load it ----
    cfg_path = os.path.join(src_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        tc["num_hidden_layers"] = int(layers)
        tc["vocab_size"] = int(vocab)
        if isinstance(tc.get("layer_types"), list):
            tc["layer_types"] = tc["layer_types"][:int(layers)]
        if drop_vision:
            cfg.pop("vision_config", None)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # tokenizer files travel unchanged: ids above the slice simply never appear,
    # and rewriting a tokenizer is a far bigger risk than an unused entry
    for f in os.listdir(src_dir):
        p = os.path.join(src_dir, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") \
                and f != "config.json":
            shutil.copy(p, os.path.join(out_dir, f))

    src_mb = sum(os.path.getsize(os.path.join(src_dir, f)) for f in files) / 1e6
    out_mb = os.path.getsize(os.path.join(out_dir, "model.safetensors")) / 1e6
    return {"out_dir": out_dir, "layers_kept": int(layers),
            "layers_dropped": len(dropped_layers), "vocab": int(vocab),
            "tensors": len(keep), "vision_dropped": dropped_vision,
            "source_megabytes": round(src_mb, 1),
            "megabytes": round(out_mb, 2),
            "shrunk": round(src_mb / max(out_mb, 1e-9), 1)}


def _selftest():
    import tempfile

    from holographic.io_and_interop.holographic_gdnruntime import load_runtime

    src = "/tmp/fw"
    if not os.path.exists(os.path.join(src, "model.safetensors")):
        print("make_test_model selftest SKIPPED-SUBJECT (no fixture present)")
        return
    rt0, cfg0 = load_runtime(src)
    out = tempfile.mkdtemp()
    rep = slice_model(src, out, layers=3, vocab=512)

    # ---- IT MUST STILL BE A LOADABLE, RUNNABLE MODEL ----
    rt, cfg = load_runtime(out)
    ids = [int(i % 500) for i in range(5, 60)]
    logits = rt.forward(ids)
    assert np.all(np.isfinite(logits)), "sliced model produced non-finite logits"
    assert int(cfg["n_layers"]) == 3, cfg["n_layers"]
    assert logits.shape[-1] == 512, logits.shape

    # ---- AND SMALLER. How MUCH smaller depends on where a model's mass
    #      sits: slicing the vocabulary dominates when the embedding is most of
    #      the model (SmolLM2-135M: 49152 x 576), and slicing layers dominates
    #      when it is not. Asserting a fixed ratio would be asserting a property
    #      of the fixture rather than of the tool.
    assert rep["megabytes"] < rep["source_megabytes"], rep
    assert rep["shrunk"] > 1.5, rep

    print("make_test_model selftest OK -- sliced a %.0f MB checkpoint to %.2f MB "
          "(%.1fx) by keeping %d of %d layers and %d vocabulary rows; the result "
          "still LOADS and produces finite logits, and every tensor in it is a "
          "tensor from the original"
          % (rep["source_megabytes"], rep["megabytes"], rep["shrunk"],
             rep["layers_kept"], rep["layers_kept"] + rep["layers_dropped"],
             rep["vocab"]))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        rep = slice_model(sys.argv[1], sys.argv[2],
                          layers=int(sys.argv[3]) if len(sys.argv) > 3 else 4,
                          vocab=int(sys.argv[4]) if len(sys.argv) > 4 else 4096)
        print("sliced %.0f MB -> %.2f MB (%.1fx): %d layers, %d vocab, %d "
              "tensors%s"
              % (rep["source_megabytes"], rep["megabytes"], rep["shrunk"],
                 rep["layers_kept"], rep["vocab"], rep["tensors"],
                 ", dropped %d vision tensors" % rep["vision_dropped"]
                 if rep["vision_dropped"] else ""))
        print("wrote %s" % rep["out_dir"])
        if rep["megabytes"] > 30:
            print("  NOTE: still over ~30 MB. Try fewer layers or a smaller "
                  "vocab, e.g. 2 2048, or pass dtype 'I8'.")
    else:
        _selftest()
