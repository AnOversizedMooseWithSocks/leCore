"""GALVABUNDLE -- the model IS the engine. One directory that contains leCore,
the weights, the resident stack, and its own bootstrap; boots on a machine where
leCore was never installed, and serves an ordinary-looking API.

The distinction from galvapack: a PACK references scaffolding the host must
already have. A BUNDLE carries it. Because the engine is NumPy/Flask/stdlib
only, "carry the engine" is a directory copy -- there is no build step, no
compiled extension, no dependency tree to resolve. That property was a design
constraint from the beginning and this is where it pays: a superior model is
distributable precisely because its scaffolding is small and pure.

WHAT IS IN A BUNDLE
    model.safetensors    ordinary weights (also usable alone, anywhere)
    galvatron.json       declarative resident manifest -- data, never code
    engine/              the leCore source tree (the full capability catalog)
    capabilities.json    the bundle's advertised feature set, generated from
                         the live catalog at build time
    run.py               bootstrap: `python run.py serve --port N`
    README.md            what it is, how to run it, and what it needs

THE FULL FEATURE SET AS PART OF THE MODEL: a bundle does not merely embed the
engine, it ADVERTISES it. `capability_tools` turns the live catalog into
OpenAI-style tool schemas, so a client that speaks tool-calling sees the whole
of leCore as functions the model can use, and /v1/capabilities + /v1/invoke let
any client call them directly. The model's feature set is the engine's feature
set -- which is the point of bundling rather than linking.

HONEST BOUNDARIES, unchanged and restated: GGUF harnesses (Ollama, llama.cpp)
have no hook surface, so for them a bundle offers its plain safetensors and
nothing more -- run the bundle's own server if you want the residents. And a
bundle is only as portable as its own rules: NumPy is required, Flask is
required for the server, and both are stated in the README rather than assumed.
"""

import json
import os
import shutil

import numpy as np


BOOTSTRAP = '''"""Galvatron bundle bootstrap -- runs without leCore installed.

The engine ships inside this directory; this script puts it on sys.path and
starts the model with its residents. No install step, no network.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine"))


def main():
    ap = argparse.ArgumentParser(description="run this Galvatron bundle")
    ap.add_argument("mode", nargs="?", default="serve",
                    choices=["serve", "info", "generate", "chat", "sessions"])
    ap.add_argument("--port", type=int, default=5930)
    ap.add_argument("--tokens", type=int, default=256,
                    help="maximum new tokens per reply (generation also STOPS "
                         "early at an end-of-turn token)")
    ap.add_argument("--prompt", default="0,1,2,3")
    ap.add_argument("--no-residents", action="store_true",
                    help="load the plain model (what a bare harness would see)")
    ap.add_argument("--session", default="default",
                    help="conversation name; resumed automatically if it exists")
    ap.add_argument("--new", action="store_true", help="start it over")
    a = ap.parse_args()

    import lecore
    from holographic.io_and_interop import holographic_galvapack as pack

    mind = None if a.no_residents else lecore.UnifiedMind(dim=512, seed=0)
    # --no-residents is an EXPLICIT request for the bare model, which is not the
    # same as "no mind was available": the latter must still enforce guards.
    gv, report = pack.load_pack(HERE, mind=mind,
                                with_guards=not a.no_residents)
    if a.mode == "info":
        print(json._default_decoder.decode(open(
            os.path.join(HERE, "galvatron.json")).read())
            if False else open(os.path.join(HERE, "galvatron.json")).read())
        print("load report:", report)
        return
    if a.mode == "generate":
        # A BUNDLE MUST ACCEPT WHAT A PERSON TYPES (cp51): this path required
        # COMMA-SEPARATED TOKEN IDS, so `generate --prompt "the lever"` died on
        # int("the lever") -- the first thing anyone tries. If the bundle carries a
        # tokenizer, text is encoded and the reply is DECODED back to text; the
        # id-list form still works for anyone scripting against it.
        raw = str(a.prompt or "")
        looks_like_ids = raw.replace(",", " ").split() and all(
            t.strip().lstrip("-").isdigit() for t in raw.split(",") if t.strip())
        tok = None
        try:
            from holographic.io_and_interop.holographic_bpe import BPE
            if os.path.exists(os.path.join(HERE, "tokenizer.json")):
                tok = BPE.from_dir(HERE)
        except Exception:
            tok = None
        if looks_like_ids:
            ids = [int(t) for t in raw.split(",") if t.strip() != ""]
        elif tok is not None:
            ids = list(tok.encode(raw))
            if not ids:
                print("the bundle's tokenizer encoded 0 tokens for that prompt -- "
                      "pass comma-separated ids instead")
                return
        else:
            print("this bundle carries no tokenizer.json, so it cannot encode text -- "
                  "pass comma-separated token ids (e.g. --prompt 12,44,7)")
            return
        out, _ = gv.generate(ids, n_new=a.tokens)
        if tok is not None and not looks_like_ids:
            try:
                print(tok.decode(list(out)))
                return
            except Exception:
                pass
        print(",".join(str(t) for t in out))
        return
    # conversations live inside the bundle, so a bundle carries its own history
    sess_root = os.path.join(HERE, "sessions")
    from holographic.io_and_interop.holographic_session import (
        SessionStore, runtime_fingerprint)
    store = SessionStore(sess_root, fingerprint=runtime_fingerprint(gv.rt))
    if a.mode == "sessions":
        rows = store.list()
        if not rows:
            print("no conversations yet")
        for m in rows:
            print("%-24s %6d tokens" % (m["name"], m.get("n_tokens", 0)))
        return
    # THE BUNDLE CARRIES ITS OWN VOCABULARY. Encoding raw bytes into a
    # large-vocab model produces fluent-looking nonsense, so use the BPE tables
    # shipped beside the weights when they are present, and fall back to bytes
    # only for genuinely byte-level models.
    _tok = None
    try:
        from holographic.io_and_interop.holographic_bpe import BPE
        _tok = BPE.from_dir(HERE)
    except Exception:
        _tok = None
    _nv = int(gv.rt.lm_head.shape[0])

    # END-OF-TURN IDS, read from the tokenizer's own added tokens rather than
    # hardcoded: every chat template names its stop differently and a guessed id
    # would silently never fire.
    _stops = set()
    for _name in ("<|im_end|>", "<|endoftext|>", "<|end|>", "</s>",
                  "<|eot_id|>", "<|end_of_text|>"):
        if _tok is not None and _name in getattr(_tok, "specials", {}):
            _stops.add(int(_tok.specials[_name]))
    try:
        import json as _json
        with open(os.path.join(HERE, "config.json")) as _f:
            _cfg_json = _json.load(_f)
        for _k in ("eos_token_id", "bos_token_id"):
            _v = _cfg_json.get(_k)
            if isinstance(_v, int):
                _stops.add(int(_v))
            elif isinstance(_v, list):
                _stops.update(int(x) for x in _v if isinstance(x, int))
        _stops.discard(int(_cfg_json.get("bos_token_id", -1)))
    except Exception:
        pass

    def _gen_stop(g, ids, n_new, state, stops):
        """Generate up to n_new tokens, stopping at an end-of-turn token."""
        seq = list(ids)
        st = state
        if st is None:
            logits, st = g.rt.prefill(seq)
        else:
            logits = st.logits
        for _ in range(int(n_new)):
            gl = g._guard(logits) if hasattr(g, "_guard") else logits
            nxt = int(gl.argmax())
            seq.append(nxt)
            if nxt in stops:
                break
            logits, st = g.rt.step(nxt, st, hooks=g._hooks()
                                   if hasattr(g, "_hooks") else None)
        return seq, st

    def _encode(text):
        if _tok is not None:
            return _tok.encode(text)
        return [b for b in text.encode("utf-8") if b < _nv]

    def _decode(ids):
        if _tok is not None:
            return _tok.decode(ids)
        return bytes(bytearray(int(t) % 256 for t in ids)).decode("utf-8", "replace")

    if a.mode == "chat":
        if a.new:
            store.delete(a.session)
        state, history = None, []
        try:
            state, man, _m = store.load(a.session)
            history = man.get("tokens") or []
            print("resumed %r (%d tokens)" % (a.session, len(history)))
        except (FileNotFoundError, OSError):
            print("new conversation %r" % a.session)
        while True:
            try:
                line = input("\\nyou> ")
            except (EOFError, KeyboardInterrupt):
                print("\\nsaved; run `chat` again to resume %r" % a.session)
                return
            if not line.strip():
                continue
            if line.strip() == "/quit":
                print("saved; run `chat` again to resume %r" % a.session)
                return
            ids = _encode(line)
            # STOP AT THE END OF THE TURN, not at the budget. Without this the
            # model runs the full token count every time and a finished sentence
            # gets cut mid-word, which reads as a broken model rather than as a
            # missing stop condition (it did).
            if state is None:
                out, state = _gen_stop(gv, ids, a.tokens, None, _stops)
                history = ids
            else:
                _lg, state = gv.rt.extend(ids, state)
                history = list(history) + ids
                out, state = _gen_stop(gv, history, a.tokens, state, _stops)
            history = out
            store.save(a.session, state, tokens=history)
            _new = out[len(history):]
            _shown = [t for t in _new if t not in _stops]
            print("bot> %s" % _decode(_shown))
    print("serving Galvatron on http://127.0.0.1:%d  (residents: %d%s, "
          "persistent sessions in ./sessions)"
          % (a.port, report["residents"], ", DEGRADED" if report["degraded"] else ""))
    app = pack.make_app(gv, model_name=os.path.basename(HERE.rstrip("/")),
                        mind=mind, session_root=sess_root)
    app.run(port=a.port, use_reloader=False)


if __name__ == "__main__":
    import json
    main()
'''


def capability_tools(mind, limit=None):
    """Turn the live catalog into OpenAI-style tool schemas -- the bundle's
    advertised feature set. Generated from the RUNNING mind (and, for native
    faculties, from the real method signature), so a bundle cannot claim a
    capability the engine it carries does not have, and a client is told the
    actual parameter NAMES rather than a useless generic blob.

    Caught in build: the first version emitted {"args": object} for everything,
    which is unusable by any tool-calling client -- it advertises that a function
    exists while hiding how to call it. Probing signatures live fixes that and
    keeps the schema honest as the engine changes."""
    import inspect
    rows = mind.capabilities().rows
    tools, seen = [], set()
    for r in rows if limit is None else rows[:limit]:
        name = r.get("name")
        if not isinstance(name, str) or name in seen or name.startswith("_"):
            continue
        seen.add(name)
        props, required = {}, []
        fn = getattr(mind, name, None)
        if callable(fn):
            try:
                for pname, prm in inspect.signature(fn).parameters.items():
                    if pname == "self" or prm.kind in (prm.VAR_POSITIONAL,
                                                       prm.VAR_KEYWORD):
                        continue
                    props[pname] = {"type": "string"}
                    if prm.default is inspect._empty:
                        required.append(pname)
            except (TypeError, ValueError):
                pass
        tools.append({
            "type": "function",
            "function": {"name": name,
                         "description": (r.get("doc") or "")[:300],
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}})
    return tools


def bundle(path, weights, cfg, residents=(), engine_root=None, notes="",
           include_engine=True, like_dir=None):
    """Write a self-contained bundle. `engine_root` defaults to the leCore tree
    this process is running from."""
    from holographic.io_and_interop import holographic_galvapack as pack
    import lecore

    os.makedirs(path, exist_ok=True)
    pack.save_pack(path, weights, cfg, residents=residents, notes=notes,
                   like_dir=like_dir)

    n_files = 0
    if include_engine:
        root = engine_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dst = os.path.join(path, "engine")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        # only what the engine needs to RUN: the package tree plus the top-level
        # entry modules. Tests, docs, tools and delivery zips are excluded --
        # a bundle is a runtime, not a repository (and shipping the zip inside
        # the zip is the recursive-artifact trap).
        shutil.copytree(os.path.join(root, "holographic"),
                        os.path.join(dst, "holographic"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in ("lecore.py", "holographic_service.py"):
            src = os.path.join(root, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst, f))
        for _r, _d, fs in os.walk(dst):
            n_files += len(fs)

    mind = lecore.UnifiedMind(dim=256, seed=0)
    tools = capability_tools(mind)
    with open(os.path.join(path, "capabilities.json"), "w") as f:
        json.dump({"count": len(tools), "tools": tools}, f)
    with open(os.path.join(path, "run.py"), "w") as f:
        f.write(BOOTSTRAP)
    # A SECOND, UNAMBIGUOUS NAME. Repositories routinely already contain a
    # run.py (this one does -- the assimilation driver), and "python run.py
    # info" from the wrong directory fails in a way that looks like the bundle
    # is broken. galvatron.py cannot be confused with anything else.
    with open(os.path.join(path, "galvatron.py"), "w") as f:
        f.write(BOOTSTRAP)
    with open(os.path.join(path, "README.md"), "w") as f:
        f.write(
            "# Galvatron bundle\n\n"
            "Self-contained: the leCore engine ships in `engine/`, so this runs\n"
            "on a machine where leCore was never installed.\n\n"
            "    python run.py chat                  # conversation that PERSISTS\n"
            "    python run.py sessions              # list saved conversations\n"
            "    python run.py serve --port 5930     # OpenAI-compatible API\n"
            "    python run.py generate --prompt 1,2,3 --tokens 8\n"
            "    python run.py serve --no-residents  # what a bare harness sees\n\n"
            "Requires: numpy (always), flask (for `serve`). Nothing else.\n\n"
            "`model.safetensors` is an ordinary checkpoint -- usable alone in any\n"
            "harness, converts to GGUF via llama.cpp's convert_hf_to_gguf.py.\n"
            "Residents (%d declared) are runtime behaviour and do NOT survive that\n"
            "conversion; run this bundle's server if you want them.\n\n"
            "Advertised capabilities: %d (see capabilities.json).\n"
            % (len(list(residents)), len(tools)))
    return {"path": path, "engine_files": n_files, "capabilities": len(tools),
            "bytes": sum(os.path.getsize(os.path.join(dp, f))
                         for dp, _, fs in os.walk(path) for f in fs)}


def _selftest():
    """The claim under test is ISOLATION: a bundle must run in a subprocess whose
    only leCore on sys.path is the one inside the bundle -- with the dev tree
    explicitly removed from the environment. Anything less proves nothing about
    distributability."""
    try:
        import torch
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except ImportError:
        print("galvabundle selftest SKIPPED-REFERENCE (torch/transformers absent)")
        return
    import subprocess
    import sys
    import tempfile

    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    cfg_t = Qwen3NextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=112,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        num_experts=0, tie_word_embeddings=True, rms_norm_eps=1e-6)
    ref = Qwen3NextForCausalLM(cfg_t).eval().float()
    weights = {k: v.detach().numpy().astype(np.float64)
               for k, v in ref.state_dict().items()}
    cfg = dict(hidden=64, n_layers=4, rms_eps=1e-6, rope_theta=10000.0,
               linear_num_value_heads=4, linear_num_key_heads=2,
               linear_key_head_dim=8, linear_value_head_dim=16, conv_kernel=4,
               n_heads=4, n_kv_heads=2, head_dim=16, partial_rotary_factor=0.25)
    ids = [int(t) for t in rng.integers(0, 97, size=8)]

    rt0 = GDNRuntime(weights, cfg)
    bare, _ = rt0.generate_fast(ids, n_new=6)
    banned = sorted(set(bare[len(ids):]))
    specs = [{"kind": "ward", "banned": banned}]

    out_dir = os.path.join(tempfile.mkdtemp(), "galv_bundle")
    rep = bundle(out_dir, weights, cfg, residents=specs, notes="selftest bundle")
    assert rep["engine_files"] > 100, rep
    assert rep["capabilities"] > 500, rep
    # the advertised schemas must carry REAL parameter names -- a tool list that
    # says only {"args": object} tells a client nothing it can call
    with open(os.path.join(out_dir, "capabilities.json")) as f:
        adv = json.load(f)
    named = [t for t in adv["tools"]
             if t["function"]["parameters"]["properties"]]
    assert len(named) > 0.8 * adv["count"], (len(named), adv["count"])
    fc = [t for t in adv["tools"]
          if t["function"]["name"] == "find_capability"]
    assert fc and "problem" in fc[0]["function"]["parameters"]["properties"]
    for f in ("model.safetensors", "galvatron.json", "run.py", "README.md",
              "capabilities.json"):
        assert os.path.exists(os.path.join(out_dir, f)), f
    assert os.path.isdir(os.path.join(out_dir, "engine", "holographic"))

    # ISOLATED RUN: cwd elsewhere, PYTHONPATH cleared, dev tree not importable.
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        [sys.executable, os.path.join(out_dir, "run.py"), "generate",
         "--prompt", ",".join(str(t) for t in ids), "--tokens", "6"],
        cwd=tempfile.mkdtemp(), env=env, capture_output=True, text=True,
        timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = [int(t) for t in proc.stdout.strip().splitlines()[-1].split(",")]
    assert got[:len(ids)] == ids, got
    # the ward travelled inside the bundle and held in a foreign process
    assert not (set(got[len(ids):]) & set(banned)), (got, banned)

    # and the bare path still works: --no-residents reproduces the plain model
    proc2 = subprocess.run(
        [sys.executable, os.path.join(out_dir, "run.py"), "generate",
         "--prompt", ",".join(str(t) for t in ids), "--tokens", "6",
         "--no-residents"],
        cwd=tempfile.mkdtemp(), env=env, capture_output=True, text=True,
        timeout=900)
    assert proc2.returncode == 0, proc2.stderr[-2000:]
    plain = [int(t) for t in proc2.stdout.strip().splitlines()[-1].split(",")]
    assert plain == bare, (plain, bare)

    print("galvabundle selftest OK -- %.1f MB bundle, %d engine files, %d "
          "advertised capabilities; ran in an ISOLATED subprocess with no leCore "
          "on the path, ward held across the process boundary, --no-residents "
          "reproduced the bare model exactly"
          % (rep["bytes"] / 1e6, rep["engine_files"], rep["capabilities"]))


if __name__ == "__main__":
    _selftest()
