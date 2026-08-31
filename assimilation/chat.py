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
    ap.add_argument("--both", action="store_true",
                    help="same prompt to original AND assimilated, side by side")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--greedy", action="store_true",
                    help="deterministic decoding (best for before/after comparison)")
    args = ap.parse_args()

    orig_dir = os.path.join(WORK, "original")
    assim_dir = os.path.join(WORK, "assimilated")
    for d in ([orig_dir, assim_dir] if args.both else
              [orig_dir] if args.original else [assim_dir]):
        if not os.path.isdir(d):
            sys.exit("model dir missing: %s\nrun ./assimilation/assimilate.sh first" % d)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: %s" % device)

    if args.both:
        tok_a, mod_a = load(orig_dir, device)
        tok_b, mod_b = load(assim_dir, device)
        hist_a, hist_b = [], []
        print("\nside-by-side: [ORIGINAL] vs [ASSIMILATED]. 'quit' to exit.\n")
        while True:
            try:
                msg = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if msg.lower() in ("quit", "exit", ""):
                break
            a, hist_a = reply(tok_a, mod_a, device, hist_a, msg, args.max_new, args.greedy)
            b, hist_b = reply(tok_b, mod_b, device, hist_b, msg, args.max_new, args.greedy)
            print("\n[ORIGINAL]\n%s\n\n[ASSIMILATED]\n%s\n" % (a, b))
    else:
        which = orig_dir if args.original else assim_dir
        tok, model = load(which, device)
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
