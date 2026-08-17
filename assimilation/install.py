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
    ap.add_argument("--prepend", type=int, default=None,
                    help="blank layers to add (default: ~8%% of depth, so the "
                         "intervention is proportionate on a 4-layer fixture "
                         "and on a 61-layer model alike)")
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
    # A DEFAULT OF ZERO IS NOT A DEFAULT. `--passages` defaults to 0, which
    # made this range EMPTY, so `passages` was [] long before anything asked
    # where to store an index -- and the "0 searchable passages" line was
    # telling the truth about a list nobody had filled. The row-count cap
    # downstream then looked like the cause and was only the second one.
    # 256 is chosen the way the register count is: from the model. It is 1 MB of
    # index at hidden 1024, which is the same order as the memory contract.
    # PASSAGES ARE A BYTE BUDGET, NOT A COUNT. 256 passages is 0.13 MB of
    # index at hidden 128 and 4.19 MB at hidden 4096 -- the same number
    # describing two very different files. Budget ~1 MB, which is the order of
    # the memory contract rather than the order of the model, and let the width
    # decide how many passages that buys.
    _budget_mb = 1.0
    _want = int(a.passages) or max(
        32, min(4096, int(_budget_mb * 1e6 / (int(cfg["hidden"]) * 4))))
    passages = [text[i:i + 240]
                for i in range(4000, min(len(text) - 300,
                                         4000 + _want * step), step)]
    passages = passages[:_want]

    # CHOOSE BOTH NUMBERS FROM THE MODEL, because they are properties of the
    # model and not decisions a user should have to make. REGISTERS cost one
    # hidden dimension each and 120 of 128 still worked, so an eighth is
    # generous and safe. PASSAGES are limited by the vocabulary rows the
    # tokenizer never emits -- there is no reason to use fewer than exist.
    all_free, top = _free_rows(a.model_dir, V, 100000)
    n_reg = int(a.registers) or max(8, int(cfg["hidden"]) // 8)
    # DO NOT LET THE ROW COUNT CAP THE PASSAGE COUNT. This read
    # min(len(all_free), len(passages)) -- so a tokenizer with no free rows gave
    # ZERO passages, and the sidecar index that needs no rows at all was handed
    # an empty list and dutifully built nothing. THE CONSTRAINT OF ONE STORAGE
    # SCHEME WAS SILENTLY LIMITING A DIFFERENT ONE.
    n_pass = int(a.passages) or (len(passages) if not all_free
                                 else min(len(all_free), len(passages)))
    passages = passages[:n_pass]
    rows = all_free[:len(passages)] if all_free else []
    print("      memory: %d registers (of %d dimensions) and %d searchable "
          "passages" % (n_reg, cfg["hidden"], len(passages)))
    if not rows:
        # THE INDEX DOES NOT HAVE TO LIVE IN THE WEIGHTS. Baking passages into
        # unused vocabulary rows is one way to store an index, and on a
        # tokenizer that uses every row it is NO way -- which is how a real
        # Qwen3.5-0.8B ended up with "0 searchable passages" and RAG silently
        # absent from the install.
        # leCore's own `build_index` needs no rows at all: a nearest-neighbour
        # index with a cosine scan for small sets and a sub-linear RP-forest for
        # large ones, plus abstention. And it is NOT massive -- 1,000 passages
        # at hidden 1024 is 4.1 MB, which is the same order as the 63 KB memory
        # contract rather than the same order as the model.
        # So it ships BESIDE the weights, like the KV cache and the session
        # memory: the same boundary this arc keeps arriving at, and the third
        # thing to land on the correct side of it.
        print("      NOTE: this tokenizer uses every vocabulary row, so the "
              "index cannot be baked into spare embedding rows. Building it "
              "ALONGSIDE the model instead (leCore build_index -- %d passages, "
              "~%.1f MB), which needs no rows and abstains on a bad query."
              % (len(passages), len(passages) * int(cfg["hidden"]) * 4 / 1e6))
        rows = None

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
    # ASK FOR THE GPU BEFORE ASKING WHERE IT IS. `--device gpu` set a variable
    # that place() read, but the BACKEND's switch is the environment variable
    # HOLOSTUFF_GPU, checked at import. So --device gpu found "no accelerator
    # available" on a machine with a working CUDA card, because nothing had
    # requested one. A flag that does not reach the thing it names is a flag
    # that lies.
    if str(a.device).lower() in ("gpu", "auto"):
        try:
            from holographic.misc.holographic_backend import (
                enable_gpu, gpu_available)
            if gpu_available():
                enable_gpu(True)
            elif str(a.device).lower() == "gpu":
                # NAME THE EXACT WHEEL, by asking the DRIVER. nvidia-smi reports
                # the highest CUDA version the installed driver supports, which
                # is the only number that decides between cupy-cuda11x and
                # cupy-cuda12x. THE CUDA TOOLKIT IS NOT NEEDED -- the pip wheel
                # bundles the runtime; only the driver has to be present, and
                # it already is if the card works at all. Saying "install cupy
                # for your CUDA version" makes the user go find that out.
                _hint = ""
                try:
                    import subprocess
                    _out = subprocess.run(["nvidia-smi"], capture_output=True,
                                          text=True, timeout=10).stdout
                    _m = re.search(r"CUDA Version:\s*(\d+)\.", _out)
                    if _m:
                        _hint = ("cupy-cuda12x" if int(_m.group(1)) >= 12
                                 else "cupy-cuda11x")
                        print("      [!] a driver IS present (CUDA %s.x) but "
                              "cupy is not installed. Run:" % _m.group(1))
                        print("          .venv\\Scripts\\python.exe -m pip "
                              "install %s" % _hint)
                        print("          (the wheel bundles the CUDA runtime "
                              "-- you do NOT need the CUDA Toolkit)")
                except Exception:
                    pass
                if not _hint:
                    print("      [!] --device gpu requested but no CUDA device "
                          "is visible. Check `nvidia-smi` runs; if it does, "
                          "install cupy-cuda12x (or cupy-cuda11x for an older "
                          "driver) into assimilation\\.venv. The CUDA Toolkit "
                          "is NOT required -- the wheel bundles the runtime.")
        except Exception as _exc:
            if str(a.device).lower() == "gpu":
                print("      [!] --device gpu requested but the GPU backend "
                      "would not load: %s" % str(_exc)[:70])

    _dev = place(rt, want=a.device)
    print("      hardware: %s (%s)"
          % (_dev.get("device"), _dev.get("why", status()["array_module"])))
    w2, c2, rep = install(w, cfg, rt, fit_ids, eval_ids, tokenize=tok,
                          passages=passages, router_positive=pos,
                          router_negative=neg, n_registers=n_reg,
                          prepend=a.prepend,       # None -> derived from depth
                          progress=show, mind=mind)
    if rep.get("aborted"):
        raise SystemExit("[install] ABORTED: %s" % rep["aborted"])

    # ---- write an ORDINARY checkpoint ----
    os.makedirs(a.out_dir, exist_ok=True)
    # KEEP THE BOOT SUBSTRATE OUT OF BF16. `like=` copies the source dtypes,
    # and a bf16 source narrows the embedding -- which carries PACKED BYTES, not
    # numbers. bf16 has EIGHT mantissa bits; the manifest needs more, so the row
    # comes back zeroed and boot() raises "no leCore substrate header here".
    # FIELD-CAUGHT on a real Qwen3.5-0.8B: the install reported boot_record ok
    # (true in memory) and audit.bat on the SAVED model reported NO BOOT RECORD
    # (true on disk). Both were honest about different bytes.
    # Measured: dtype=None round-trips, F16 round-trips, BF16 DESTROYS IT.
    from holographic.io_and_interop.holographic_boot import (
        boot_substrate_keys)
    _keep = boot_substrate_keys(w2, report=(rep.get("boot") or {}))
    # WRITE TO A TEMP NAME AND RENAME. A forced Windows Update restart during
    # the export leaves a TRUNCATED model.safetensors that still loads -- the
    # header is written first, so the file looks structurally fine and the
    # tensors after the cut are garbage or absent. Field-caught: an install was
    # interrupted and the resulting folder assessed cleanly at 24 layers,
    # because the layer count came from a config that HAD been written while
    # the weights had not.
    # os.replace is atomic on Windows and POSIX alike, so the final name either
    # does not exist or is a complete file. THERE IS NO PARTIAL STATE TO
    # MISREAD.
    _final = os.path.join(a.out_dir, "model.safetensors")
    _tmp = _final + ".incomplete"
    export_portable(w2, _tmp, like=a.model_dir, keep_f32=_keep)
    os.replace(_tmp, _final)
    for f in os.listdir(a.model_dir):
        src = os.path.join(a.model_dir, f)
        if os.path.isfile(src) and not f.endswith(".safetensors") \
                and not f.endswith(".index.json"):
            shutil.copy(src, os.path.join(a.out_dir, f))

    # THE CONFIG MUST MATCH THE NEW DEPTH, including layer_types -- a loader
    # that reads 24 entries for a 26-layer model misreads every tensor after
    # the second one.
    # AND THE CONFIG LAST, for the same reason in the other direction: a config
    # claiming 26 layers beside weights that only have 24 is exactly the state
    # that made an interrupted run look finished.
    cp = os.path.join(a.out_dir, "config.json")
    if os.path.exists(cp):
        with open(cp) as f:
            cj = json.load(f)
        tc = cj.get("text_config", cj)
        tc["num_hidden_layers"] = int(c2["n_layers"])
        if isinstance(tc.get("layer_types"), list):
            # USE WHAT THE INSTALL ACTUALLY DID, not what was requested. With
            # prepend derived from depth, a.prepend is None and this wrote zero
            # entries -- so the saved layer_types would have been SHORTER than
            # the model. The report is the source of truth for what happened.
            _added = int(rep.get("prepend_layers")
                         or (int(c2["n_layers"]) - int(cfg["n_layers"])))
            tc["layer_types"] = (["linear_attention"] * _added
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

    # lecore.json IS THE COMPLETION MARKER, written LAST and atomically. Its
    # presence means every earlier step finished; its ABSENCE on a folder that
    # otherwise looks like a model means the run was interrupted. Before this,
    # an install killed by a forced restart left a directory that loaded, ran,
    # and assessed cleanly -- with no way to tell it from a finished one except
    # by counting layers and knowing what the count should have been.
    _lj = os.path.join(a.out_dir, "lecore.json")
    _ljt = _lj + ".incomplete"
    with open(_ljt, "w") as f:
        json.dump({"format": "leCore/installed/1",
                   "installed": rep["installed"],
                   "registers": rep.get("registers"),
                   "router": {k: rep.get("router", {}).get(k)
                              for k in ("layer", "holdout_accuracy")},
                   # RECORD THE CALIBRATION WHERE A RUNTIME CAN FIND IT. A
                   # measurement nobody reads is the exact failure this session
                   # has found five times, so the safe depth ships in
                   # lecore.json and holographic_lecorerun reads it.
                   "exit_calibration": rep.get("exit_calibration"),
                   "memory_index": rep.get("memory_index"),
                   "improvement": rep.get("improvement"),
                   "boot_row": rep.get("boot_row"),
                   "baseline_perplexity": rep.get("baseline_perplexity"),
                   "final": rep.get("final")}, f, indent=2)
    os.replace(_ljt, _lj)          # atomic: the marker appears complete or not

    mb = os.path.getsize(os.path.join(a.out_dir, "model.safetensors")) / 1e6
    print("\n[wrote] %s  (%.1f MB, %s)"
          % (a.out_dir, mb,
             ", ".join(sorted(set(source_dtypes(a.out_dir).values())))))

    # ---- RELOAD FROM DISK and verify. In-process success is a different
    #      claim from "this file works", and this project has shipped the
    #      difference before.
    # ---- THE SIDECAR INDEX, when the tokenizer left no rows to bake into ----
    #      WRITTEN AFTER THE CHECKPOINT, because it lands in out_dir and the
#      first version ran before that directory existed -- FileNotFoundError
#      on a step that had otherwise worked. Order is part of the wiring.
#      Built from the model's OWN last-layer state, so a query and a passage
    #      are compared in the space the model actually thinks in -- no second
    #      embedding model, nothing learned, one forward pass per passage.
    #      MEASURED on 120 real passages: half-passage queries retrieve the
    #      right one 9 of 18 times. That is a REAL number and not a good one --
    #      it is a byte-level tokenizer on a tiny fixture, and it is recorded
    #      rather than hidden so nobody mistakes the mechanism for a benchmark.
    #      KEPT NEGATIVE: CENTRING DID NOT HELP HERE (9/18 either way), which is
    #      worth stating because centring has been the fix four separate times
    #      in this project and it is tempting to apply it on faith.
    if rows is None and passages:
        try:
            import numpy as _np
            from holographic.io_and_interop.holographic_gdnruntime import (
                GDNRuntime as _RT)
            _r = _RT(w2, dict(c2))
            _L = int(c2["n_layers"]) - 1
            _V = []
            for _p in passages:
                _cap = {}
                _r.mlp_probe = (lambda l, x: _cap.__setitem__(
                    "x", _np.asarray(x)[-1].copy()) if int(l) == _L else None)
                _r.forward(tok(_p)[:200])
                _r.mlp_probe = None
                _v = _cap["x"].astype(_np.float32)
                _V.append(_v / (_np.linalg.norm(_v) + 1e-30))
            _M = _np.stack(_V)
            _np.savez_compressed(os.path.join(a.out_dir, "lecore_index.npz"),
                                 vectors=_M,
                                 passages=_np.array(passages, dtype=object),
                                 allow_pickle=True)
            rep["sidecar_index"] = {"passages": len(passages),
                                    "megabytes": round(_M.nbytes / 1e6, 2),
                                    "file": "lecore_index.npz"}
            print("      index         ok    %d passages beside the model "
                  "(%.2f MB, lecore_index.npz)"
                  % (len(passages), _M.nbytes / 1e6))
        except Exception as _exc:
            print("      index         FAIL  %s: %s"
                  % (type(_exc).__name__, str(_exc)[:60]))


    # ---- VERIFY THE BOOT RECORD ON DISK, not in memory. It reported ok during
    #      the install and came back NO BOOT RECORD (JSONDecodeError) from the
    #      saved file on a real bf16 Qwen3.5 -- both true, about different
    #      bytes. The record is the LAST thing written and the FIRST thing a
    #      narrowing export can destroy, so it is checked where it lands.
    try:
        from holographic.io_and_interop.holographic_boot import boot as _boot
        from holographic.io_and_interop.holographic_unicron import (
            load_safetensors as _ls)
        _disk = _ls(os.path.join(a.out_dir, "model.safetensors"))
        _rec = _boot(_disk)["record"]
        print("      boot record   ok    reads back from disk: %d capabilities"
              % len(_rec.capabilities))
    except Exception as _exc:
        print("      boot record   FAIL  wrote ok but does NOT read back from "
              "disk (%s) -- the manifest is in lecore.json, the model still "
              "works, but nothing can identify it from the weights alone"
              % type(_exc).__name__)

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
