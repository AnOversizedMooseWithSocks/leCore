#!/usr/bin/env python3
"""Chat harness for the assimilated (or original) model -- "how do I run it?"

    ./assimilation/chat.sh                       # chat with the ASSIMILATED model
    ./assimilation/chat.sh --original            # chat with the untouched original
    ./assimilation/chat.sh --both                # SAME prompt to both, side by side

--both is the harness worth using: perplexity (--eval) is the number, but reading
the two models answer the same question is the fastest way to FEEL whether the
assimilation kept the model's mind. Type a message, get a reply; 'quit' exits.

Runs entirely locally out of assimilation/work/. Uses the transformers runtime
(installed into the venv by chat.sh on first use) because Qwen3.5's hybrid
DeltaNet architecture ships its own modeling code -- our NumPy engine reads and
rewrites the WEIGHTS; running the model is the runtime's job, and pretending
otherwise would be exactly the kind of unmeasured claim we do not make.
"""
import argparse
import os
import sys

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")


def load(model_dir, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading %s ..." % model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    want = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=want, trust_remote_code=True).to(device).eval()
    except TypeError:   # older transformers: dtype kwarg not accepted yet
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=want, trust_remote_code=True).to(device).eval()
    return tok, model


def reply(tok, model, device, history, user_msg, max_new=256, greedy=False):
    """One chat turn. Uses the tokenizer's own chat template when it has one
    (Qwen ships one); otherwise falls back to a plain prompt.

    Empty replies get a DIAGNOSTIC line instead of silence: how many tokens were
    generated and what they were (specials included). An empty reply has two very
    different causes -- the model emitting EOS immediately / only special or
    thinking tokens (a MODEL-behaviour fact, possibly assimilation damage), vs a
    template/decode artifact (a HARNESS fact) -- and a blank "model>" line hides
    which one happened. Field report on record: chat ran, no crash, all replies
    empty, cause indistinguishable."""
    import torch
    history = history + [{"role": "user", "content": user_msg}]
    if getattr(tok, "chat_template", None):
        try:
            # Qwen3-family templates take enable_thinking; without it the model
            # may spend its whole budget inside a think block that decodes empty
            enc = tok.apply_chat_template(history, add_generation_prompt=True,
                                          enable_thinking=False,
                                          return_tensors="pt")
        except TypeError:
            enc = tok.apply_chat_template(history, add_generation_prompt=True,
                                          return_tensors="pt")
        ids = enc if torch.is_tensor(enc) else enc["input_ids"]
    else:
        ids = tok("\n".join(m["content"] for m in history) + "\n",
                  return_tensors="pt").input_ids
    ids = ids.to(device)
    attn = torch.ones_like(ids)
    kwargs = dict(attention_mask=attn, max_new_tokens=max_new,
                  min_new_tokens=1, pad_token_id=tok.eos_token_id)
    if greedy:
        kwargs["do_sample"] = False
        # even greedy gets the repetition penalty: the official card warns the
        # 0.8B is prone to degenerate loops without a presence penalty, and a
        # damaged OR healthy model deserves the card's operating point
        kwargs["repetition_penalty"] = 1.3
    else:
        # official Qwen3.5 card, non-thinking text mode: temperature=1.0,
        # top_p=1.0, top_k=20, presence_penalty=2.0. transformers generate()
        # has no presence_penalty; repetition_penalty is the closest lever.
        kwargs.update(do_sample=True, temperature=1.0, top_p=1.0, top_k=20,
                      repetition_penalty=1.3)
    with torch.no_grad():
        out = model.generate(ids, **kwargs)
    new_tokens = out[0][ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    if not text:
        n = int(new_tokens.shape[0]) if hasattr(new_tokens, "shape") else len(new_tokens)
        raw = tok.decode(new_tokens, skip_special_tokens=False)
        print("[diagnostic] empty reply: %d token(s) generated; raw (specials "
              "kept): %r" % (n, raw[:200]))
        print("[diagnostic] run chat with --both -- if the ORIGINAL answers and "
              "the ASSIMILATED does not, the assimilation damaged the model and "
              "that is a RESULT to report, not a harness bug.")
    history.append({"role": "assistant", "content": text})
    return text, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", action="store_true", help="run the untouched model")
    ap.add_argument("--galvatron", action="store_true",
                    help="run the INSTALLED model (work/galvatron -- the one with "
                         "leCore in its weights); with --both, pairs it against "
                         "the original")
    ap.add_argument("--model", default=None, metavar="DIR",
                    help="run an arbitrary model directory")
    ap.add_argument("--memory", default=None, metavar="DIR",
                    help="external leCore memory for this chat (default: "
                         "assimilation/work/chat_memory, or $LECORE_PARTITION). "
                         "Taught facts answer FIRST with provenance; teach: q = a "
                         "stores durably; wrong: q vetoes; everything persists "
                         "across sessions like any real chat harness's store")
    ap.add_argument("--no-memory", action="store_true",
                    help="raw model only (the old behaviour)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (default is GPU when available, with "
                         "automatic CPU fallback if the GPU fails)")
    ap.add_argument("--pack", default=None, metavar="PACK_DIR",
                    help="serve a galvapack PACK (residents LIVE in the forward "
                         "pass, numpy runtime -- the door the resident plane "
                         "actually walks through; plain transformers loads show "
                         "the weights alone)")
    ap.add_argument("--both", action="store_true",
                    help="same prompt to original AND assimilated, side by side")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--greedy", action="store_true",
                    help="deterministic decoding (best for before/after comparison)")
    args = ap.parse_args()

    # ------------------------------------------------------------- memory end
    # THE HARNESS GROWS A MIND (cp86): a chat harness without external memory
    # has nowhere for leCore to unfurl -- no taught knowledge, no grounded arm,
    # no continuity between sessions. This is the same job openwebui's store
    # does, played by the engine itself: numpy + stdlib, happy inside this venv.
    mind = None
    memdir = None
    if not args.no_memory:
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        try:
            import lecore
            memdir = (args.memory or os.environ.get("LECORE_PARTITION") or
                      os.path.join(WORK, "chat_memory"))
            mind = lecore.autoboot(partition=memdir, llm=None)
            n_rows = len({str(t[0]) for t in
                          mind.zoo["ladder"].taught_log if len(t) > 3})
            print("memory: %s (%d rows; teach: q = a | wrong: q)"
                  % (memdir, n_rows))
        except Exception as exc:
            print("memory unavailable (%s: %s) -- raw model chat"
                  % (type(exc).__name__, str(exc)[:60]))
            mind = None

    def memory_turn(msg):
        """Returns (handled, text). Grounded answers and the teach/veto verbs
        are memory's business; everything else falls through to the model."""
        if mind is None:
            return False, None
        low = msg.lower()
        if low.startswith("teach:") and "=" in msg:
            q_, a_ = msg[6:].split("=", 1)
            rep = mind.teach(q_.strip(), a_.strip())
            mind.learning_save(memdir)
            ok = not (isinstance(rep, dict) and rep.get("taught") is False)
            return True, ("[MEMORY] taught and saved" if ok else
                          "[MEMORY] refused: %s" % rep.get("reason"))
        if low.startswith(("wrong:", "veto:")):
            q_ = msg.split(":", 1)[1].strip()
            try:
                mind.answer_feedback(q_, ok=False)
                mind.learning_save(memdir)
                return True, ("[MEMORY] vetoed durably (tombstoned across "
                              "restarts; a deliberate re-teach lifts it)")
            except Exception as exc:
                return True, "[MEMORY] veto failed: %s" % exc
        g = mind.ask_grounded(msg)
        if g.get("answer") and not g.get("escalate"):
            return True, "[MEMORY %s] %s" % (g.get("provenance"),
                                             g["answer"])
        return False, None

    if args.pack:
        # THE RESIDENT DOOR (cp83, coverage item 2): chat used to load plain
        # transformers even for galvatron, so residents were absent from every
        # efficacy conversation. A pack chat runs the numpy Galvatron with its
        # resident stack live.
        from holographic.io_and_interop.holographic_galvapack import load_pack
        g, rep = load_pack(args.pack)
        print("pack loaded: %s" % {k: rep[k] for k in sorted(rep)
                                   if k in ("residents", "degraded", "skipped")})
        print("type a message; 'quit' exits. teach: t = x writes the live db.\n")
        while True:
            try:
                msg = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if msg.lower() in ("quit", "exit", ""):
                break
            handled, mtext = memory_turn(msg)
            if handled:
                print(mtext)
                continue
            if msg.startswith("teach:") and "=" in msg:
                t, x = msg[6:].split("=", 1)
                print(g.teach(t.strip(), x.strip()))
                continue
            out = g.ask(msg) if hasattr(g, "ask") else g.generate_text(msg) \
                if hasattr(g, "generate_text") else "(pack has no text door; " \
                "use run_galvatron for the HTTP server)"
            print("[PACK] %s" % str(out)[:1000])
        return

    orig_dir = os.path.join(WORK, "original")
    assim_dir = os.path.join(WORK, "assimilated")
    # FIELD-CAUGHT (first efficacy run): --both silently paired original vs
    # ASSIMILATED, so the "what is your BIOS" question went to a model with no
    # BIOS installed -- the user noticed the banner said 'assimilated'. The
    # right subject for that question is work/galvatron.
    galv_dir = os.path.join(WORK, "galvatron")
    subj_dir = (args.model or (galv_dir if args.galvatron else assim_dir))
    for d in ([orig_dir, subj_dir] if args.both else
              [orig_dir] if args.original else [subj_dir]):
        if not os.path.isdir(d):
            sys.exit("model dir missing: %s\nrun ./assimilation/assimilate.sh first" % d)

    import torch
    # GPU BY DEFAULT (cp87): there is no need to go slow on purpose. --cpu
    # forces CPU; otherwise CUDA is used when present and any GPU failure
    # falls back to CPU with the reason printed. NOTE for Windows boxes: a
    # plain `pip install torch` ships the CPU-ONLY wheel -- the launcher now
    # installs the CUDA build when nvidia-smi is present, so an A4500 is
    # actually used instead of silently idling.
    if args.cpu:
        device = "cpu"
        print("device: cpu (forced by --cpu)")
    elif torch.cuda.is_available():
        device = "cuda"
        try:
            print("device: cuda (%s)" % torch.cuda.get_device_name(0))
        except Exception:
            print("device: cuda")
    else:
        device = "cpu"
        print("device: cpu (torch reports no CUDA -- if this box has an "
              "NVIDIA GPU, delete assimilation/.venv and rerun the launcher "
              "so the CUDA wheel installs)")

    def _load_with_fallback(d_):
        nonlocal device
        try:
            return load(d_, device)
        except Exception as exc:
            if device == "cuda":
                print("GPU load failed (%s: %s) -- falling back to CPU"
                      % (type(exc).__name__, str(exc)[:80]))
                device = "cpu"
                return load(d_, "cpu")
            raise

    subj_name = os.path.basename(subj_dir).upper()
    if args.both:
        tok_a, mod_a = _load_with_fallback(orig_dir)
        # THE ACTUAL LOAD (field-caught twice): the label fix alone left this
        # line loading assim_dir while the banner claimed GALVATRON -- a label
        # that lies is worse than a wrong default. The subject loads subj_dir.
        tok_b, mod_b = _load_with_fallback(subj_dir)
        hist_a, hist_b = [], []
        print("\nside-by-side: [ORIGINAL] vs [%s]." % subj_name + " 'quit' to exit.\n")
        while True:
            try:
                msg = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if msg.lower() in ("quit", "exit", ""):
                break
            # HARNESS VERBS (teach/veto) are commands, not questions -- they
            # go to memory and neither model runs.
            low_ = msg.lower()
            if mind is not None and (low_.startswith(("teach:", "wrong:",
                                                      "veto:"))):
                _h, mtext = memory_turn(msg)
                print("\n%s\n" % mtext)
                continue
            # THE ORIGINAL IS THE BASELINE (cp87): it never sees external
            # memory -- raw generation only, so the comparison stays honest.
            # The SUBJECT side answers from memory first when it can.
            import time as _t
            t0 = _t.time()
            a, hist_a = reply(tok_a, mod_a, device, hist_a, msg,
                              args.max_new, args.greedy)
            ta = _t.time() - t0
            handled, mtext = memory_turn(msg)
            if handled:
                b_label = "%s via MEMORY" % subj_name
                b, tb = mtext, 0.0
            else:
                t0 = _t.time()
                b, hist_b = reply(tok_b, mod_b, device, hist_b, msg,
                                  args.max_new, args.greedy)
                tb = _t.time() - t0
                b_label = subj_name
            print("\n[ORIGINAL %.1fs]\n%s\n\n[%s%s]\n%s\n"
                  % (ta, a, b_label,
                     "" if handled else " %.1fs" % tb, b))
    else:
        which = orig_dir if args.original else subj_dir
        tok, model = _load_with_fallback(which)
        hist = []
        print("\nchatting with %s. 'quit' to exit.\n"
              % ("ORIGINAL" if args.original else "ASSIMILATED"))
        while True:
            try:
                msg = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if msg.lower() in ("quit", "exit", ""):
                break
            handled, mtext = memory_turn(msg)
            if handled:
                print("\n%s\n" % mtext)
                continue
            text, hist = reply(tok, model, device, hist, msg, args.max_new, args.greedy)
            print("\nmodel> %s\n" % text)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        # A double-clicked console window closes before the error can be read;
        # print the full traceback and hold the window open. Measured need: the
        # first live chat run crashed with NO visible error.
        import traceback
        traceback.print_exc()
        print("\n[the error above is the reason the chat could not start]")
        try:
            input("press Enter to close...")
        except EOFError:
            pass
        sys.exit(1)
