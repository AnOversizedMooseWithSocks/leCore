"""fixture_gate.py -- THE GATE cp106 PAID FOR (cp107): refuse to report a number from a
learned-structure experiment when the artifact has no learned structure.

Three of four gap-list builds in cp106 (SAE-to-codebook, prune-and-heal, compute-to-lookup)
returned negatives that were later traced to ONE root cause: the test fixture sat at chance
(perplexity 2249-2304 against vocab_size 2048). The dictionary found nothing because there
was nothing; the lookup table failed because the hidden states were near-isotropic noise;
pruning "improved" perplexity because perplexity was already at the ceiling. Each of those
runs cost minutes. The check that would have caught all three costs one second.

VERDICTS
  TRAINED    ppl <= trained_frac * vocab   -- learned-structure experiments are meaningful
  WEAK       ppl <  vocab but above frac   -- interpret with care; report the ratio
  AT-CHANCE  ppl >= vocab                  -- BLOCKED: report BLOCKED, never a number

The gate deliberately reports BLOCKED rather than raising: a blocked experiment is a
result ("this artifact cannot answer this question"), not a crash.

Usage:
    python3 tools/fixture_gate.py /path/to/model [--text FILE] [--tokens N]
    from tools.fixture_gate import gate;  v = gate("/tmp/mini_baked")
Exit code 0 when TRAINED or WEAK, 2 when AT-CHANCE, 3 when the artifact cannot be read.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAINED_FRAC = 0.5  # ppl at or below half of chance counts as genuinely trained


def _vocab_size(model_dir):
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    tc = cfg.get("text_config", cfg)
    v = tc.get("vocab_size") or cfg.get("vocab_size")
    if not v:
        raise ValueError("no vocab_size in %s" % cfg_path)
    return int(v)


def _token_stream(model_dir, n_tokens):
    """A structured, non-uniform stream. Uniform-random ids are unpredictable BY
    CONSTRUCTION and drive every model to chance, which is exactly the confound this
    gate exists to catch -- so we never use them."""
    return [(i * 7 + 3) % 400 + 1 for i in range(n_tokens)]


def gate(model_dir, n_tokens=256, trained_frac=TRAINED_FRAC):
    """Return a verdict dict. Never raises for a merely-bad model; only for an unreadable one."""
    from holographic.io_and_interop import holographic_gdnruntime as g

    vocab = _vocab_size(model_dir)
    rt, _cfg = g.load_runtime(model_dir)
    toks = _token_stream(model_dir, n_tokens)
    ppl = float(rt.perplexity(toks))
    ratio = ppl / float(vocab)
    if ppl >= vocab:
        verdict = "AT-CHANCE"
        blocked = True
    elif ratio <= trained_frac:
        verdict = "TRAINED"
        blocked = False
    else:
        verdict = "WEAK"
        blocked = False
    return {
        "model": model_dir,
        "vocab_size": vocab,
        "perplexity": ppl,
        "ppl_over_chance": ratio,
        "verdict": verdict,
        "blocked": blocked,
        "n_tokens": len(toks),
        "note": ("learned-structure experiments (SAE extraction, prune-and-heal, "
                 "weight folding, compute-to-lookup) are BLOCKED on this artifact"
                 if blocked else "learned-structure experiments are meaningful here"),
    }


def require_trained(model_dir, what="this experiment", **kw):
    """Guard for experiment scripts. Returns the verdict when usable; prints and returns
    None when blocked, so the caller can report BLOCKED instead of a meaningless number."""
    v = gate(model_dir, **kw)
    if v["blocked"]:
        print("BLOCKED: %s needs learned structure, but %s is AT-CHANCE "
              "(ppl %.1f >= vocab %d)" % (what, model_dir, v["perplexity"], v["vocab_size"]))
        return None
    return v


def main(argv):
    if not argv:
        print(__doc__)
        return 3
    model_dir = argv[0]
    n_tokens = 256
    if "--tokens" in argv:
        n_tokens = int(argv[argv.index("--tokens") + 1])
    try:
        v = gate(model_dir, n_tokens=n_tokens)
    except Exception as exc:  # unreadable artifact is a different failure than a bad one
        print("UNREADABLE %s: %s" % (model_dir, exc))
        return 3
    print("fixture gate: %s" % model_dir)
    print("  vocab_size   : %d" % v["vocab_size"])
    print("  perplexity   : %.1f  (%.2fx chance)" % (v["perplexity"], v["ppl_over_chance"]))
    print("  VERDICT      : %s" % v["verdict"])
    print("  %s" % v["note"])
    return 2 if v["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
