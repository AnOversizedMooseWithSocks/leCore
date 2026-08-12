"""install.py -- put leCore into a real model, in one pass, and verify it.

    python assimilation/install.py MODEL_DIR OUT_DIR [--doc FILE] [--registers N]

THIS REPLACES assimilate -> repair -> imbue. That pipeline changed 18 of 265
tensors, repair reverted 12 of them as harmful, and the surviving difference sat
inside the measurement noise -- 149 seconds to demonstrate nothing. Nothing here
edits the original tensors at all: two blank layers go in FRONT, everything
leCore adds lives in them, in unused vocabulary rows, or in reserved directions
of the recurrent state.

WHAT GETS INSTALLED, each step measured and REVERTED if it regresses:
    prepend       2 blank layers, output BIT-IDENTICAL (verified, not assumed)
    boot_record   one embedding row, scaled and clamped, 4 bits per slot
    registers     reserved key directions -- permanent memory in the state
    router        a discriminant on layer 0 that decides when to use a capability
    memory_index  passage addresses in rows the tokenizer never emits
    improvement   a closed-form correction, step chosen by measuring

THE ARTIFACT is an ordinary checkpoint: same tensor names, same dtype, a config
with two more layers, plus lecore.json describing what was installed. It
converts and runs anywhere a normal model does.
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _tokenizer(model_dir, n_vocab):
    """Byte fallback is not a fallback for a 248k vocabulary -- it is nonsense.

    A model with a real tokenizer must use it, or every probe, every index
    address and every router example is built from tokens the model has never
    seen in that order."""
    try:
        from holographic.io_and_interop.holographic_bpe import BPE
        bpe = BPE.from_dir(model_dir)
        return lambda t: list(bpe.encode(t))[:512], "model tokenizer"
    except Exception:
        if int(n_vocab) > 1024:
            raise SystemExit(
                "this model has a %d-entry vocabulary but no readable "
                "tokenizer -- refusing to fall back to raw bytes, which would "
                "make every probe meaningless" % n_vocab)
        return lambda t: [b for b in t.encode("utf-8")][:512], "raw bytes"


def _free_rows(model_dir, n_vocab, need):
    """Rows the tokenizer will never emit. Measured, not assumed.

    reserved_rows reads tokenizer.json's added_tokens, which is how we learned
    that Qwen3.5's "free" rows start at 248,070 and not 248,044 -- the earlier
    count would have overwritten the vision and eos tokens."""
    from holographic.io_and_interop.holographic_galvapack import reserved_rows
    top = int(reserved_rows(model_dir, n_vocab))
    rows = list(range(top, int(n_vocab)))
    return rows[:int(need)], top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?", default="work/original",
                    help="the model to assimilate (default: work/original, "
                         "resolved from where you are standing)")
    ap.add_argument("out_dir", nargs="?", default=None,
                    help="where to write Galvatron (default: work/galvatron "
                         "beside the model directory)")
    ap.add_argument("--doc", help="OPTIONAL: a text file to ground the install "
                                  "in. Leave it out and leCore uses its own "
                                  "documentation, which always ships with it.")
    ap.add_argument("--registers", type=int, default=0,
                    help="OPTIONAL: permanent memory slots. 0 = choose from "
                         "the model's width (one eighth of it).")
    ap.add_argument("--passages", type=int, default=0,
                    help="OPTIONAL: searchable passages. 0 = fill the "
                         "vocabulary rows the tokenizer never emits.")
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cpu", "gpu"),
                    help="use an accelerator if one is present (default auto)")
    ap.add_argument("--prepend", type=int, default=2)
    a = ap.parse_args()

    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime, load_weights_dir)
    from holographic.io_and_interop.holographic_install_lecore import install
    from holographic.io_and_interop.holographic_unicron import (
        export_portable, source_dtypes)
    from holographic.io_and_interop.holographic_boot import boot
    from holographic.io_and_interop.holographic_measure import measure

    # RESOLVE BEFORE THE FIRST USE, which is this line. The resolver was
    # correct and ran 120 lines too late -- load_runtime(a.model_dir) had
    # already failed on the raw string. A fix that runs after the thing it
    # fixes is not a fix, and the traceback said so precisely: line 99 using
    # a.model_dir, resolution at line 221.
    # The launchers cd to the repo root so the package imports work, which
    # silently breaks any relative path typed elsewhere; GALVATRON_CWD carries
    # the caller's directory and _resolve_model_dir tries it, then the repo,
    # then work/ under BOTH the repo root and assimilation/ -- because
    # `work\original` lives beside the launcher, not beside the repo.
    from assimilation.galvatron import _resolve_model_dir as _rmd
    a.model_dir = _rmd(a.model_dir)
    if not a.out_dir:
        # DEFAULT BESIDE THE MODEL, not beside the repo. install.bat treats
        # out_dir as optional and install.py required it -- a launcher and its
        # script disagreeing about their own interface, which fails only when
        # someone uses the documented one-argument form.
        a.out_dir = os.path.join(os.path.dirname(os.path.abspath(a.model_dir)),
                                 "galvatron")

    print("[load] %s" % a.model_dir)
    rt, cfg = load_runtime(a.model_dir)
    w = load_weights_dir(a.model_dir)
    hk = next(k for k in w if k.endswith("embed_tokens.weight"))
    V = int(np.asarray(w[hk]).shape[0])
    print("      hidden %d | %d layers | vocab %d | %s"
          % (cfg["hidden"], cfg["n_layers"], V,
             ", ".join(sorted(set(source_dtypes(a.model_dir).values())))))

    # READ THE MODEL FROM ITS TENSORS TOO, and cross-check. A checkpoint is an
    # unlabeled dataset; the config is one witness and the tensors are another,
    # and when they disagree it is the CONFIG that is usually stale -- a wrong
    # layer count or hidden size makes every tensor below reshape wrongly, which
    # is the most expensive failure this pipeline knows.
    from holographic.io_and_interop.holographic_adapt import infer
    seen = infer(w, tokenizer_dir=a.model_dir)
    agree = (seen["n_layers"] == int(cfg["n_layers"])
             and seen["hidden"] == int(cfg["hidden"]))
    print("      inferred from tensors alone: %d layers, hidden %d, tied=%s "
          "(confidence %.2f)" % (seen["n_layers"], seen["hidden"],
                                 seen["tied"], seen["confidence"]))
    if not agree:
        print("      [!] THE CONFIG AND THE TENSORS DISAGREE:")
        print("          config says %d layers / hidden %d"
              % (int(cfg["n_layers"]), int(cfg["hidden"])))
        print("          tensors say %d layers / hidden %d (%s)"
              % (seen["n_layers"], seen["hidden"], seen["evidence"]["hidden"]))
        print("          continuing on the CONFIG, but check it before trusting "
              "any number below.")

    tok, how = _tokenizer(a.model_dir, V)
    print("      tokenizer: %s" % how)

    # A DEFAULT CORPUS THAT ALWAYS EXISTS. Requiring --doc made the first step
    # of the whole pipeline "go find some text", which is not a decision anyone
    # should have to make to try this. leCore ships 5.5 MB of its own English
    # documentation; it is real prose, it is always present, and a model with
    # leCore installed having read about leCore is the right default anyway.
    if a.doc:
        text = open(a.doc, encoding="utf-8", errors="ignore").read()
        print("      corpus: %s (%.0f KB)" % (a.doc, len(text) / 1e3))
    else:
        import glob
        parts = []
        for f in sorted(glob.glob(os.path.join(_REPO, "docs", "*.md"))):
            try:
                parts.append(open(f, encoding="utf-8", errors="ignore").read())
            except OSError:
                pass
        text = "\n\n".join(parts)
        if len(text) < 20000:
            raise SystemExit(
                "no --doc given and leCore's own docs were not found at %s -- "
                "pass --doc FILE with any plain text file"
                % os.path.join(_REPO, "docs"))
        print("      corpus: leCore's own documentation (%.0f KB) -- pass "
              "--doc FILE to use your own" % (len(text) / 1e3))

    # CHECK THE TOKENIZER ACTUALLY TOKENIZES. A vocabulary file can load
    # cleanly and still return NOTHING for real text -- and then every
    # measurement below is taken on an empty probe, which this pipeline has
    # already shipped once. Fail here, where the reason is obvious.
    fit_ids = tok(text[:20000])
    if len(fit_ids) < 256:
        raise SystemExit(
            "the tokenizer returned only %d tokens for 20,000 characters of "
            "text -- it loaded but does not encode this corpus. Pass --doc "
            "with text the model was trained on, or check tokenizer.json."
            % len(fit_ids))
    eval_ids = tok(text[20000:26000])[:1200]
    if len(eval_ids) < 128:
        cut = max(128, len(fit_ids) // 3)
        eval_ids, fit_ids = fit_ids[-cut:], fit_ids[:-cut]

    # router examples: questions against ordinary prose from the same corpus
    rng = np.random.default_rng(0)
    stems = ["what is ", "how does ", "why does ", "where is ", "which ",
             "how many ", "what happens when ", "explain "]
    words = re.findall(r"\b[a-z]{5,12}\b", text[:400000]) or ["memory", "state"]
    pos = [rng.choice(stems) + " ".join(rng.choice(words, 2)) + " "
           for _ in range(120)]
    step = max(len(text) // 200, 40)
    neg = [text[i:i + 120] for i in range(2000, min(len(text) - 200,
                                                    2000 + 120 * step), step)]
    passages = [text[i:i + 240]
                for i in range(4000, min(len(text) - 300,
                                         4000 + a.passages * step), step)]
    passages = passages[:a.passages]

    # CHOOSE BOTH NUMBERS FROM THE MODEL, because they are properties of the
    # model and not decisions a user should have to make. REGISTERS cost one
    # hidden dimension each and 120 of 128 still worked, so an eighth is
    # generous and safe. PASSAGES are limited by the vocabulary rows the
    # tokenizer never emits -- there is no reason to use fewer than exist.
    all_free, top = _free_rows(a.model_dir, V, 100000)
    n_reg = int(a.registers) or max(8, int(cfg["hidden"]) // 8)
    n_pass = int(a.passages) or min(len(all_free), len(passages))
    passages = passages[:n_pass]
    rows = all_free[:len(passages)]
    print("      memory: %d registers (of %d dimensions) and %d searchable "
          "passages" % (n_reg, cfg["hidden"], len(passages)))
    if not rows:
        print("      NOTE: this tokenizer uses every vocabulary row, so there "
              "is nowhere to put a search index -- skipping it. Registers and "
              "everything else still install.")

    def show(s):
        print("      %-14s %-5s %s" % (s["step"], "ok" if s["ok"] else "FAIL",
                                       s["detail"]))

    print("\n[install] leCore into the weights")
    # RUN THE INSTALL THROUGH leCore ITSELF. Unicron assimilating a model should
    # use leCore's own faculties to do it -- the holographic operations that
    # build in vector space are the same ones that build in weight space, and a
    # tool that imports around its own engine is not dogfooding it.
    import lecore as _lecore
    mind = _lecore.UnifiedMind(dim=512, seed=0)

    # USE THE HARDWARE THAT IS THERE. An LLM is usually run on a GPU, and this
    # pipeline was host-NumPy throughout. Weights go resident ONCE if a device
    # and the policy allow; on a laptop this reports cpu and runs unchanged.
    from holographic.io_and_interop.holographic_devicerun import place, status
    _dev = place(rt, want=a.device)
    print("      hardware: %s (%s)"
          % (_dev.get("device"), _dev.get("why", status()["array_module"])))
    w2, c2, rep = install(w, cfg, rt, fit_ids, eval_ids, tokenize=tok,
                          passages=passages, router_positive=pos,
                          router_negative=neg, n_registers=n_reg,
                          prepend=a.prepend, progress=show, mind=mind)
    if rep.get("aborted"):
        raise SystemExit("[install] ABORTED: %s" % rep["aborted"])

    # ---- write an ORDINARY checkpoint ----
    os.makedirs(a.out_dir, exist_ok=True)
    export_portable(w2, os.path.join(a.out_dir, "model.safetensors"),
                    like=a.model_dir)
    for f in os.listdir(a.model_dir):
        src = os.path.join(a.model_dir, f)
        if os.path.isfile(src) and not f.endswith(".safetensors") \
                and not f.endswith(".index.json"):
            shutil.copy(src, os.path.join(a.out_dir, f))

    # THE CONFIG MUST MATCH THE NEW DEPTH, including layer_types -- a loader
    # that reads 24 entries for a 26-layer model misreads every tensor after
    # the second one.
    cp = os.path.join(a.out_dir, "config.json")
    if os.path.exists(cp):
        with open(cp) as f:
            cj = json.load(f)
        tc = cj.get("text_config", cj)
        tc["num_hidden_layers"] = int(c2["n_layers"])
        if isinstance(tc.get("layer_types"), list):
            tc["layer_types"] = (["linear_attention"] * int(a.prepend)
                                 + list(tc["layer_types"]))
        # EVERY SHAPE THE INSTALL CHANGED MUST BE WRITTEN, or the model cannot
        # be RELOADED. The HRNN ladder grows in_proj_qkvz from 320 rows to 960
        # by adding key and value heads; without these four keys the reload
        # fails validation with "the GDN head numbers are wrong" -- and the
        # in-memory selftest never saw it, because it never saved and reloaded.
        # AN INSTALL THAT ONLY WORKS IN THE PROCESS THAT BUILT IT IS NOT
        # INSTALLED.
        for _src, _dst in (("linear_num_key_heads", "linear_num_key_heads"),
                           ("linear_num_value_heads", "linear_num_value_heads"),
                           ("linear_key_head_dim", "linear_key_head_dim"),
                           ("linear_value_head_dim", "linear_value_head_dim"),
                           ("hidden", "hidden_size")):
            if _src in c2:
                tc[_dst] = int(c2[_src])
        with open(cp, "w") as f:
            json.dump(cj, f, indent=2)

    with open(os.path.join(a.out_dir, "lecore.json"), "w") as f:
        json.dump({"format": "leCore/installed/1",
                   "installed": rep["installed"],
                   "registers": rep.get("registers"),
                   "router": {k: rep.get("router", {}).get(k)
                              for k in ("layer", "holdout_accuracy")},
                   "memory_index": rep.get("memory_index"),
                   "improvement": rep.get("improvement"),
                   "boot_row": rep.get("boot_row"),
                   "baseline_perplexity": rep.get("baseline_perplexity"),
                   "final": rep.get("final")}, f, indent=2)

    mb = os.path.getsize(os.path.join(a.out_dir, "model.safetensors")) / 1e6
    print("\n[wrote] %s  (%.1f MB, %s)"
          % (a.out_dir, mb,
             ", ".join(sorted(set(source_dtypes(a.out_dir).values())))))

    # ---- RELOAD FROM DISK and verify. In-process success is a different
    #      claim from "this file works", and this project has shipped the
    #      difference before.
    print("\n[verify] reloading from disk")
    # FREE THE IN-MEMORY MODEL FIRST. The verify step reloads the whole
    # checkpoint from disk while the installed copy, the ORIGINAL copy and a
    # live runtime are all still held -- on a 2.1 GB model that is three copies
    # and the reload dies with MemoryError while reading the file. Field-caught
    # on a real Qwen3.5-0.8B: everything installed, the file wrote correctly,
    # and the VERIFICATION ran out of memory.
    import gc
    for _name in ("w", "w2", "rt"):
        if _name in dir():
            pass
    try:
        del w, w2
    except Exception:
        pass
    try:
        del rt
    except Exception:
        pass
    gc.collect()

    rt3, c3 = load_runtime(a.out_dir)
    w3 = load_weights_dir(a.out_dir)
    m3 = measure(rt3, eval_ids)
    try:
        seed = boot(w3)["record"].seed
    except Exception as exc:
        seed = "FAILED (%s)" % exc
    print("      %d layers | perplexity %.4f (was %.4f) | boots as %r"
          % (c3["n_layers"], m3["perplexity"],
             rep["baseline_perplexity"], seed))
    f = rep["final"]
    print("      verdict %s (%+.3f%%) | repetition %.2f -> %.2f"
          % (f["verdict"], f["delta_pct"], rep["baseline_repetition"],
             f["repetition"]))
    print("\nNext:  assess.bat        (or: python assimilation/galvatron.py "
          "%s --assess out.npz)" % a.out_dir)


if __name__ == "__main__":
    main()
