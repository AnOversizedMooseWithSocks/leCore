"""GALVATRON DRIVER -- run a REAL checkpoint inside leCore, with residents.

This is the script that turns the whole arc into numbers on Moose's machine. It
needs no torch and no transformers: leCore owns the forward pass, so a model
directory plus NumPy is the entire dependency list.

    python assimilation/galvatron.py MODEL_DIR --ppl "some text tokens"
    python assimilation/galvatron.py MODEL_DIR --generate 1,2,3 --tokens 20
    python assimilation/galvatron.py MODEL_DIR --demo          # residents live
    python assimilation/galvatron.py MODEL_DIR --report        # unicron_report

WHAT EACH MODE ANSWERS
  --ppl       the standing EVAL DEBT: perplexity computed IN-ENGINE, so an
              assimilated model can finally be priced against its original
              without a second runtime. Run it on both directories and compare.
  --demo      the residents on the real model: ward (bans hold), salience
              (does a TRAINED model's hesitation actually vary?), oracle
              (memory steers), and a snapshot/branch rewind check.
  --report    unicron_report over the checkpoint: regime census, structure,
              levers, and the refutations.

HONEST NOTE ON SPEED: this runtime is correctness-first NumPy. On a 0.8B it is
slow -- use short prompts. The point of these numbers is truth, not throughput;
the fast path (GDN state cache) is already measured at 4.8-12.9x over recompute
and is what `generate_fast` uses here.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _reference_logits(model_dir, ids):
    """Reference next-token logits for the same ids, via transformers (this
    python or the assimilation venv). This is the ONE check that settles a
    tensor-layout question: names can be guessed, numbers cannot."""
    import subprocess
    snippet = (
        "import sys,json,torch;from transformers import AutoModelForCausalLM;"
        "ids=json.loads(sys.argv[2]);"
        "m=AutoModelForCausalLM.from_pretrained(sys.argv[1],"
        "trust_remote_code=True,dtype=torch.float32).eval();"
        "print(json.dumps(m(torch.tensor([ids])).logits[0,-1].tolist()))")
    try:
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,
                                                 dtype=torch.float32).eval()
        return m(torch.tensor([list(ids)])).logits[0, -1].tolist()
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for base in (here, root):
        for py in (os.path.join(base, ".venv", "Scripts", "python.exe"),
                   os.path.join(base, ".venv", "bin", "python")):
            if not os.path.exists(py):
                continue
            print("      (reference runtime from %s -- this loads the model in "
                  "torch and can take a minute)" % py)
            try:
                out = subprocess.run(
                    [py, "-c", snippet, model_dir, json.dumps([int(i) for i in ids])],
                    capture_output=True, text=True, timeout=1800)
                if out.returncode == 0 and out.stdout.strip():
                    return json.loads(out.stdout.strip().splitlines()[-1])
                print("      reference run failed: %s"
                      % (out.stderr.strip().splitlines()[-1:] or ["(no output)"])[0])
            except Exception as exc:
                print("      reference run error: %s" % exc)
    return None


def _reference_ids(model_dir, text):
    """Cross-check against the reference tokenizer, using the ASSIMILATION VENV
    when this interpreter has no transformers.

    The venv is where assimilation installed torch/transformers, and it is the
    only place on a normal setup that can answer -- so look there rather than
    reporting "not available" and leaving the check undone."""
    import subprocess
    snippet = (
        "import sys,json;from transformers import AutoTokenizer;"
        "print(json.dumps(AutoTokenizer.from_pretrained(sys.argv[1],"
        "trust_remote_code=True).encode(sys.argv[2])))")
    try:
        from transformers import AutoTokenizer            # this python?
        return list(AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True).encode(text))
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    cands = []
    for base in (here, root):
        cands += [os.path.join(base, ".venv", "Scripts", "python.exe"),
                  os.path.join(base, ".venv", "bin", "python"),
                  os.path.join(base, "venv", "Scripts", "python.exe"),
                  os.path.join(base, "venv", "bin", "python")]
    for py in cands:
        if not os.path.exists(py):
            continue
        try:
            out = subprocess.run([py, "-c", snippet, model_dir, text],
                                 capture_output=True, text=True, timeout=300)
            if out.returncode == 0 and out.stdout.strip():
                print("      (reference from %s)" % py)
                return list(json.loads(out.stdout.strip().splitlines()[-1]))
        except Exception:
            continue
    return None


def _resolve_model_dir(arg):
    """Find the model directory the user MEANT.

    The launchers cd to the repo root before python starts (so the package
    imports work), which silently breaks any relative path typed from another
    directory -- the caller's cwd is preserved in GALVATRON_CWD for exactly this
    reason. We also look in the usual places, because "work/assimilated" is
    almost always right about the NAME and wrong only about the prefix."""
    cand = []
    a = os.path.expanduser(str(arg).rstrip("/\\"))
    # NORMALISE THE SEPARATOR. A Windows user -- or install.bat's own default --
    # passes `work\original`, and on a POSIX-flavoured shell (git-bash, MSYS,
    # WSL) the backslash is a literal character in a filename, not a separator,
    # so every candidate below is built wrong and the path is REFUSED even
    # though it exists. Measured: "work/original" resolved and "work\original"
    # did not, on the same directory.
    # BOTH SEPARATOR FORMS FEED EVERY CANDIDATE. My first attempt normalised
    # once at the top and left the derived candidates using the original -- so
    # "work\original" still failed from another directory even though
    # "work/original" worked on the SAME folder. A normalisation that does not
    # reach the places the value is USED has not normalised anything.
    forms = [a]
    swapped = (a.replace(chr(92), "/") if chr(92) in a
               else a.replace("/", chr(92)))
    if swapped != a:
        forms.append(swapped)
    home = os.environ.get("GALVATRON_CWD")
    here = os.path.dirname(os.path.abspath(__file__))          # assimilation/
    root = os.path.dirname(here)
    for f in forms:
        cand.append(f)
        if os.path.isabs(f):
            continue
        if home:
            cand.append(os.path.join(home, f))
        cand += [os.path.join(here, f), os.path.join(root, f)]
        base = os.path.basename(f.replace(chr(92), "/"))
        # `work\original` lives beside the LAUNCHER, not beside the repo root,
        # which is where the default in install.bat points.
        cand += [os.path.join(here, "work", base),
                 os.path.join(root, "work", base)]
    for c in cand:
        if os.path.isdir(c) and any(f.endswith(".safetensors")
                                    for f in os.listdir(c)):
            return c
    # nothing matched: say what DOES exist rather than just failing
    found = []
    for base in (here, root, os.path.join(here, "work"),
                 os.path.join(root, "work"), home or here):
        if not base or not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            try:
                if os.path.isdir(d) and any(f.endswith(".safetensors")
                                            for f in os.listdir(d)):
                    found.append(d)
            except OSError:
                continue
    msg = ["model directory %r not found (looked in %d places)"
           % (arg, len(cand))]
    if found:
        msg.append("these directories DO contain a checkpoint:")
        for d in dict.fromkeys(found):
            msg.append("    " + d)
        msg.append("pass one of those (a full path always works)")
    else:
        # SAY HOW TO GET ONE. "not found" is a diagnosis; the next COMMAND is
        # what the person actually needs, and a fresh clone or a deleted folder
        # is the likeliest reason to be reading this at all. assimilate.bat
        # already downloads anonymously, resumably, and skips if present -- it
        # just was not mentioned anywhere the failure could be seen.
        msg.append("")
        msg.append("no checkpoint anywhere nearby. To fetch one:")
        msg.append("    assimilate.bat        downloads Qwen3.5-0.8B into "
                   "work\\original")
        msg.append("                          (~1.6 GB, anonymous, resumable, "
                   "skips if present)")
        msg.append("    assimilate.bat --model Qwen/Qwen3.5-2B    other sizes")
        msg.append("then:")
        msg.append("    install.bat ./work/original")
    raise SystemExit("\n".join(msg))


MIN_CHUNK_TOKENS = 48


def _grounding_corpus(spec):
    """Build a grounding corpus without asking the user to supply one.

    leCore already ships text the model provably never trained on: this repo's
    own notes, and a 144k-entry WordNet dictionary. Both are better
    hallucination probes than an invented file -- the notes because no public
    model has seen them, the dictionary because obscure definitions are exactly
    where a small model confabulates confidently."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    spec = (spec or "").strip()
    if spec and os.path.exists(spec):
        with open(spec, encoding="utf-8", errors="ignore") as f:
            out = [p.strip() for p in f.read().split("\n\n") if len(p.strip()) > 40]
        return out[:400], os.path.basename(spec)

    def _lecore_docs():
        out = []
        import glob as _glob
        for pat in ("docs/*.md", "*.md"):
            for fp in sorted(_glob.glob(os.path.join(root, pat)))[:8]:
                try:
                    with open(fp, encoding="utf-8", errors="ignore") as f:
                        out += [p.strip() for p in f.read().split("\n\n")
                                if 60 < len(p.strip()) < 700]
                except OSError:
                    continue
        return out[:300]

    def _wordnet(n=200):
        import lzma as _lzma
        for cand in (os.path.join(root, "lecore_data", "knowledge",
                                  "dictionary.json.xz"),
                     os.path.join(here, "lecore_data", "knowledge",
                                  "dictionary.json.xz")):
            if not os.path.exists(cand):
                continue
            with _lzma.open(cand) as f:
                d = json.load(f)
            words = sorted(d)
            rng = np.random.default_rng(0)
            picks = rng.choice(len(words), size=min(n, len(words)), replace=False)
            out = []
            for i in picks:
                w = words[int(i)]
                e = d[w]
                e = e[0] if isinstance(e, list) and e else e
                if isinstance(e, dict) and e.get("d"):
                    out.append("%s: %s" % (w.replace("_", " "), e["d"]))
            return out
        return []

    if spec == "wordnet":
        wn = _wordnet(300)
        if wn:
            return wn, "bundled WordNet dictionary"
    if spec == "lecore":
        docs = _lecore_docs()
        if docs:
            return docs, "this repository's own notes"
    docs, wn = _lecore_docs(), _wordnet(120)
    if docs or wn:
        return (docs + wn), ("leCore notes (%d) + WordNet sample (%d)"
                             % (len(docs), len(wn)))
    return (["Bread is baked from flour, water, salt and yeast in a hot oven."],
            "fallback")


def _prove(rt, cfg, tok, n_vocab, prompt, doc_path, n_tokens):
    """Show what leCore adds to a checkpoint that contains none of it.

    The weights are ordinary; every capability below is RUNTIME structure in
    leCore's forward pass. Each test prints the bare model first and the
    resident-equipped model second, so the difference is visible rather than
    asserted -- and each is a thing a plain harness running this same
    checkpoint cannot do at all."""
    import lecore
    from holographic.agents_and_reasoning.holographic_galvatron import (
        Galvatron, OracleResident, WardResident)
    from holographic.agents_and_reasoning.holographic_knowres import (
        CorpusResident, SalienceTrigger)
    from holographic.agents_and_reasoning.holographic_swarm import (
        EvidenceStore, verified_generate)
    import numpy as _np

    mind = lecore.UnifiedMind(dim=512, seed=0)
    ids = _tokens_from(prompt, n_vocab, tok)
    say = lambda t: _detok(t, tok, n_vocab)
    probe_layer = max(0, int(cfg["n_layers"]) - 2)
    print("\n=== 0. THE BARE MODEL (what any harness gives you) ===")
    bare, _ = rt.generate_fast(ids, n_new=n_tokens)
    print("    %r" % say(bare[len(ids):]))

    print("\n=== 1. WARD: make tokens IMPOSSIBLE, not discouraged ===")
    banned = sorted(set(bare[len(ids):]))
    warded, _ = Galvatron(rt, guards=[WardResident(banned=banned)]).generate(
        ids, n_new=n_tokens)
    leaked = set(warded[len(ids):]) & set(banned)
    print("    banned every token it just used -> %r" % say(warded[len(ids):]))
    print("    ban breached: %s   (a prompt cannot promise this)" % bool(leaked))

    print("\n=== 2. ORACLE: editable memory keyed on the live hidden state ===")
    cap = {}
    rt.forward(ids, hooks={probe_layer:
                           lambda h: cap.__setitem__("h", h.copy()) or None})
    target = int(_np.argsort(rt.forward(ids)[-1])[-6])
    # SCALE THE MEMORY TO THE MODEL, not to a magic number: a fixed gain is
    # either silent or dictatorial depending on embedding scale (the same lesson
    # the swarm digest taught). Size the injection against the logit margin the
    # memory has to overcome.
    # Inject at the LAST layer: a vector added earlier is reshaped by every
    # layer after it, so an analytic estimate of "how much is enough" made at
    # layer n-2 does not survive to the logits (it did not -- measured).
    # Sweep the gain instead and REPORT what the model actually needed: the
    # number is informative, and a swept demo cannot quietly fail.
    last = int(cfg["n_layers"]) - 1
    capL = {}
    rt.forward(ids, hooks={last: lambda h: capL.__setitem__("h", h.copy()) or None})
    lg0 = rt.forward(ids)[-1]
    top_before = int(_np.argmax(lg0))
    direction = _np.asarray(rt.embed[target], _np.float64)
    top_after, used = top_before, None
    for gain in (1, 2, 4, 8, 16, 32, 64, 128):
        orc = OracleResident(mind, int(cfg["hidden"]), layer=last,
                             gain=1.0, threshold=0.0)
        orc.remember(capL["h"][-1], float(gain) * direction)
        g = Galvatron(rt, residents=[orc])
        top_after = int(_np.argmax(rt.forward(ids, hooks=g._hooks())[-1]))
        if top_after == target:
            used = gain
            break
    if used:
        print("    (memory strength needed: %gx the target embedding)" % used)
    print("    next token %r -> %r  (target %r)"
          % (say([top_before]), say([top_after]), say([target])))
    print("    memory installed and effective WITHOUT touching a weight: %s"
          % (top_after == target))

    print("\n=== 3. SALIENCE: does the model's own hesitation vary? ===")
    sal = SalienceTrigger(rt)
    sal.calibrate(cap["h"], quantile=0.8)
    sc = _np.array([sal.score(x) for x in cap["h"]])
    print("    lens entropy over %d positions: mean %.3f spread %.3f "
          "(min %.3f max %.3f)" % (len(sc), sc.mean(), sc.std(), sc.min(), sc.max()))
    hi = int(_np.argmax(sc)); lo = int(_np.argmin(sc))
    print("    most uncertain at %r, most confident at %r"
          % (say([ids[hi]]), say([ids[lo]])))
    print("    -> retrieval can fire on hesitation instead of a fixed schedule")

    print("\n=== 4. CORPUS: ground the answer in a document it never saw ===")
    passages, source = _grounding_corpus(doc_path)
    print("    corpus: %s (%d passages)" % (source, len(passages)))
    cr = CorpusResident(mind, passages, int(cfg["hidden"]), layer=probe_layer,
                        query_fn=lambda h: prompt, gain=2.0)
    base_lg = rt.forward(ids)
    out_lg = rt.forward(ids, hooks={probe_layer: cr.hook})
    if cr.log:
        print("    retrieved: %r" % cr.log[0]["passage"][:90])
        print("    reached the residual stream: %s   (no context window used)"
              % bool(_np.max(_np.abs(out_lg - base_lg)) > 1e-6))

    print("\n=== 5. FACT CHECK: refuse to assert what no source supports ===")
    # SPAN MUST SCALE WITH THE CORPUS. A 3-token span is a real constraint
    # against three passages and a rubber stamp against three hundred -- common
    # trigrams appear somewhere in any large corpus, so the checker vetoed
    # NOTHING (measured: 0 of 3 proposals). Longer spans keep "grounded" meaning
    # grounded as the source set grows.
    span = 3 if len(passages) < 20 else (5 if len(passages) < 200 else 6)
    ev = EvidenceStore([_tokens_from(p, n_vocab, tok) for p in passages],
                       span=span)
    print("    evidence: %d passages, %d-token spans must be supported"
          % (len(passages), span))
    unchecked, _ = rt.generate_fast(ids, n_new=min(12, n_tokens))
    got, rep = verified_generate(rt, ids, ev, n_new=min(12, n_tokens), k=4)
    print("    unchecked : %r   <- asserted freely, grounded in nothing"
          % say(unchecked[len(ids):]))
    print("    checked   : %r   (%d proposals, %d vetoed)"
          % (say(got[len(ids):]), rep["proposals"], rep["vetoes"]))
    if rep["exhausted"]:
        print("    the checker ran out of grounded options and STOPPED rather "
              "than assert something unsupported. Silence is the correct answer "
              "when the sources cannot back a claim -- that is the contract.")
    # and prove the checker is not simply refusing everything
    ok_text = passages[0][:60]
    ok_ids = _tokens_from(ok_text, n_vocab, tok)
    print("    sanity: a span taken FROM the sources passes the checker: %s"
          % (not ev.unsupported(ok_ids)))

    print("\n=== 6. TIME TRAVEL: snapshot, branch, rewind exactly ===")
    _lg, st = rt.prefill(ids)
    snap = st.copy()
    a1, _ = rt.generate_fast(ids, n_new=6, state=st)
    a2, _ = rt.generate_fast(ids, n_new=6, state=snap.copy())
    print("    rewind reproduces the timeline token-for-token: %s" % (a1 == a2))
    print("\n    NOTE: none of this is IN the checkpoint. Export these weights "
          "to GGUF and every capability above disappears -- they are leCore "
          "running the forward pass, not parameters.")


def _wrap(text, width):
    """Wrap on word boundaries so a passage can be READ and checked."""
    words, line, out = str(text).split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w) if line else w
    if line:
        out.append(line)
    return out or [""]


def _chunk_passages(text, tok, n_vocab, want):
    """Split TEXT into passages WITHOUT re-tokenizing the pieces.

    THE BUG THIS FIXES, caught because the tool disagreed with itself: passages
    were tokenized one at a time, so a sentence starting a passage lost its
    leading space and became DIFFERENT TOKENS than the same sentence inside the
    full text ("I" vs " I"). The concatenated result scored 22.09 where the same
    text scored 16.56 through --ppl. Re-tokenizing a fragment does not measure
    the fragment; it measures a different string.

    So: tokenize the WHOLE text once, then locate sentence boundaries as token
    OFFSETS into that single sequence. Returns (ids, [(start, end), ...]) so the
    caller scores exactly the tokens the model would have seen.
    """
    import re as _re
    text = str(text)
    ids = _tokens_from(text, n_vocab, tok)
    want = max(1, int(want))
    # character offsets of sentence ends
    # cut immediately AFTER the punctuation, BEFORE the following space: BPE
    # merges a space with the word that follows it, so taking the offset after
    # the whitespace pushed the next sentence's first token into the previous
    # passage ("...it melts. I" / "had a bunch..." in a real run).
    offs = [m.end() for m in _re.finditer(r"[.!?]", text)]
    if not offs or offs[-1] < len(text):
        offs.append(len(text))
    if len(offs) < 2 or want == 1:
        return ids, [(0, len(ids))]
    # token index of each boundary, by encoding the PREFIX (never the piece)
    bounds = []
    for off in offs:
        n = len(_tokens_from(text[:off], n_vocab, tok)) if off < len(text) else len(ids)
        bounds.append(min(max(n, 0), len(ids)))
    # INTERIOR boundaries only: the end of the text is not a cut point, and
    # including it made the selection collapse onto duplicates (3 sentences with
    # --chunks 3 produced 2 passages).
    interior = sorted(set(b for b in bounds if 0 < b < len(ids)))
    n = min(want, len(interior) + 1)
    if n <= 1 or not interior:
        return ids, [(0, len(ids))]
    if len(interior) <= n - 1:
        picks = interior
    else:
        picks = [interior[int(round(i * (len(interior) - 1) / float(n - 2)))]
                 if n > 2 else interior[len(interior) // 2]
                 for i in range(n - 1)]
    cuts = [0] + sorted(set(picks)) + [len(ids)]
    spans = [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
    return ids, spans


def _detok(ids, tok, n_vocab):
    """Ids -> text by whatever vocabulary this model actually has."""
    if tok is not None:
        return tok.decode(ids)
    if n_vocab <= 256:
        return bytes(bytearray(int(t) % 256 for t in ids)).decode("utf-8", "replace")
    return ",".join(str(int(t)) for t in ids)


def _load_tokenizer(model_dir):
    """The model directory already carries its vocabulary (vocab.json +
    merges.txt, or tokenizer.json). leCore reads it with stdlib -- no
    transformers, no tokenizers library -- so the driver speaks TEXT."""
    from holographic.io_and_interop.holographic_bpe import BPE
    for d in (model_dir, os.path.join(model_dir, ".."),
              os.path.join(os.path.dirname(model_dir.rstrip("/\\")), "original")):
        try:
            return BPE.from_dir(d)
        except (FileNotFoundError, OSError, ValueError):
            continue
    return None


def _tokens_from(arg, n_vocab, tok=None):
    """Accept TEXT (tokenized with the model's own vocabulary) or explicit ids.
    Ids are detected only when the whole argument is comma-separated numbers, so
    ordinary prose is never mistaken for a token list."""
    txt = str(arg)
    parts = [p for p in txt.replace(" ", ",").split(",") if p != ""]
    if parts and all(p.lstrip("-").isdigit() for p in parts):
        ids = [int(p) for p in parts]
        bad = [v for v in ids if not (0 <= v < n_vocab)]
        if bad:
            raise SystemExit("token id %d out of range for vocab %d"
                             % (bad[0], n_vocab))
        return ids
    if tok is None:
        if n_vocab <= 256:
            # a byte-level model already HAS a vocabulary: the bytes
            return [b for b in txt.encode("utf-8") if b < n_vocab]
        raise SystemExit(
            "no vocabulary found in the model directory, so text cannot be "
            "tokenized -- pass comma-separated token ids instead, or point at "
            "a directory containing vocab.json+merges.txt or tokenizer.json")
    return tok.encode(txt)


def main():
    ap = argparse.ArgumentParser(description="run a real checkpoint in leCore")
    ap.add_argument("model_dir")
    ap.add_argument("--ppl", help="token ids to score (perplexity, in-engine)")
    ap.add_argument("--generate", help="prompt token ids")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--demo", action="store_true", help="residents on the real model")
    ap.add_argument("--leap", action="store_true",
                    help="speculative decoding with a learned route drafter "
                         "(output identical to greedy, measured both ways)")
    ap.add_argument("--report", action="store_true", help="unicron_report")
    ap.add_argument("--lazy", action="store_true", help="compressed resident weights")
    ap.add_argument("--chat", action="store_true",
                    help="interactive conversation with PERSISTENT context")
    ap.add_argument("--session", default="default",
                    help="conversation name (default: 'default' -- resumed "
                         "automatically if it exists)")
    ap.add_argument("--new", action="store_true",
                    help="start this conversation over, discarding its context")
    ap.add_argument("--list-sessions", action="store_true",
                    help="show saved conversations and their sizes")
    ap.add_argument("--fork", metavar="NAME",
                    help="copy --session into NAME (two futures, one past)")
    ap.add_argument("--forget", metavar="NAME", help="delete a saved conversation")
    ap.add_argument("--sessions-dir", default=None,
                    help="where conversations live (default: MODEL_DIR/sessions)")
    ap.add_argument("--ingest", action="append", default=[], metavar="FILE",
                    help="file the model should remember and be able to cite "
                         "(repeatable; also usable mid-chat with /ingest FILE)")
    ap.add_argument("--recall", metavar="QUERY",
                    help="search everything this model has ever been told")
    ap.add_argument("--check-tokenizer", metavar="TEXT", nargs="?", const=
                    "The holographic engine binds and bundles hypervectors.",
                    help="verify leCore's stdlib BPE against the reference "
                         "tokenizer (needs transformers; run it once)")
    ap.add_argument("--repair", metavar="ORIGINAL_DIR",
                    help="make THIS assimilated model at least as good as the "
                         "original: per-tensor, walk back toward the original "
                         "and keep whichever blend measures best")
    ap.add_argument("--imbue", metavar="OUT_DIR",
                    help="build an IMBUED GALVATRON here: these weights plus the "
                         "resident roster, their calibration, the grounding "
                         "corpus and leCore itself, runnable anywhere")
    ap.add_argument("--ban", metavar="TEXT",
                    help="text whose tokens the imbued model must never emit")
    ap.add_argument("--prove", nargs="?", const="", metavar="PROMPT",
                    help="prove what leCore adds ON TOP of these weights: ward, "
                         "oracle memory, corpus grounding, fact-check veto and "
                         "time travel, each shown bare vs resident-equipped")
    ap.add_argument("--doc", metavar="FILE|wordnet|lecore",
                    help="grounding corpus: a file, 'wordnet' (the bundled "
                         "144k-entry dictionary), or 'lecore' (this repo's own "
                         "docs -- text the model provably never saw). "
                         "Default: lecore docs plus a wordnet sample.")
    ap.add_argument("--compare", metavar="OTHER_DIR",
                    help="load a SECOND model and report the perplexity delta "
                         "over the same text, with per-chunk spread")
    ap.add_argument("--chunks", type=int, default=6,
                    help="split the text into this many passages so the delta "
                         "gets an error bar instead of a single number")
    ap.add_argument("--verify", nargs="?", const="The holographic engine binds.",
                    metavar="TEXT",
                    help="THE definitive check: run leCore and the reference "
                         "implementation on the same text and compare logits")
    ap.add_argument("--assess", nargs="?", const="assessment.npz",
                    metavar="OUT.npz",
                    help="measure this model and write an assessment bundle "
                         "(BIOS, POST, perplexity, tokens/sec, gates, full "
                         "spectra, activations, top-64 logits, harden audit). "
                         "A PROFILE, not the model -- safe to send.")
    ap.add_argument("--sidecar", nargs="?", const="lecore.sidecar.npz",
                    metavar="FILE",
                    help="build a leCore SIDECAR beside this model instead of "
                         "editing it: the base stays byte-identical, the "
                         "sidecar carries the boot record and circuits, and "
                         "--merge-sidecar writes a deployable checkpoint")
    ap.add_argument("--merge-sidecar", nargs=2, metavar=("FILE", "OUT_DIR"),
                    help="merge a sidecar into an ordinary checkpoint for "
                         "llama.cpp / Ollama")
    ap.add_argument("--bios", action="store_true",
                    help="enumerate this model before touching it: layout, "
                         "block structure, vocabulary slack, carrier capacity, "
                         "and whether leCore is already installed")
    ap.add_argument("--install", nargs="?", const="", metavar="OUT_DIR",
                    help="install the leCore layer into these weights (boot "
                         "record + engine payload) and write the result to "
                         "OUT_DIR; with no OUT_DIR, AUDIT the model instead")
    ap.add_argument("--transform", action="store_true",
                    help="analyse this model's block structure and print the "
                         "targeted upgrade plan (which layers to preserve, "
                         "which to grow, where to compress the KV cache)")
    ap.add_argument("--testkit-all", nargs="?", const="kits", metavar="DIR",
                    help="export layers as separate files into DIR (default: "
                         "kits/ beside where you are standing), plus a shared "
                         "base.npz. Use --layers to pick which.")
    ap.add_argument("--layers", metavar="LIST",
                    help="which layers --testkit-all should write, e.g. "
                         "'0,12,23' or 'first,mid,last' (default: first, "
                         "middle and last -- all 24 is ~980 MB)")
    ap.add_argument("--testkit", metavar="OUT.npz",
                    help="export a compact profile of THIS model (spectra, "
                         "gates, a real activation stream, one layer of real "
                         "weights) for offline experimentation. Not the model.")
    ap.add_argument("--keys", nargs="?", const="0,1,2,3", metavar="LAYERS",
                    help="dump tensor names for these layers (diagnostic)")
    ap.add_argument("--knows", action="store_true",
                    help="inventory of what the model has been told")
    ap.add_argument("--scope", choices=("all", "session", "none"),
                    help="what THIS conversation may reference: everything "
                         "(all), only itself (session), or nothing (none = a "
                         "clean slate). Sticky: saved per conversation.")
    ap.add_argument("--prune", nargs="*", metavar="FILTER",
                    help="delete knowledge: session=NAME kind=KIND source=SRC "
                         "days=N (previewed unless --yes is given)")
    ap.add_argument("--yes", action="store_true",
                    help="actually perform a --prune instead of previewing it")
    a = ap.parse_args()

    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    from holographic.io_and_interop.holographic_session import (
        SessionStore, runtime_fingerprint)

    a.model_dir = _resolve_model_dir(a.model_dir)
    print("[1/3] loading %s%s ..." % (a.model_dir, " (lazy)" if a.lazy else ""))
    t0 = time.time()
    rt, cfg = load_runtime(a.model_dir, lazy=a.lazy)
    n_vocab = int(np.asarray(rt.lm_head).shape[0])
    tok = _load_tokenizer(a.model_dir)
    print("      tokenizer: %s"
          % ("%d entries from the model directory (stdlib BPE)" % len(tok.encoder)
             if tok else ("byte-level (vocab %d)" % n_vocab if n_vocab <= 256
                          else "NONE FOUND -- token ids only")))
    print("      OK in %.1fs | hidden %d, layers %d, vocab %d, "
          "GDN %dV/%dK heads, attn %dQ/%dKV head_dim %d"
          % (time.time() - t0, cfg["hidden"], cfg["n_layers"], n_vocab,
             cfg["linear_num_value_heads"], cfg["linear_num_key_heads"],
             cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]))

    # PERSISTENCE IS ON BY DEFAULT for a normal run: conversations live beside
    # the model unless told otherwise, so "run it again tomorrow" resumes
    # instead of starting from nothing. Nobody should have to wire a store to
    # get the behaviour every chat interface already has.
    # leCore's own artifacts go in a DOT-DIRECTORY, not loose in the model
    # folder: a plain "sessions/" directory beside the weights got picked up by
    # the assimilation file copy and crashed it (PermissionError on a directory).
    # A model directory belongs to the model.
    sess_root = a.sessions_dir or os.path.join(a.model_dir, ".lecore",
                                               "sessions")
    legacy = os.path.join(a.model_dir, "sessions")
    if os.path.isdir(legacy) and not os.path.isdir(sess_root):
        os.makedirs(os.path.dirname(sess_root), exist_ok=True)
        try:
            os.rename(legacy, sess_root)          # move old sessions, do not lose them
        except OSError:
            pass
    store = SessionStore(sess_root, fingerprint=runtime_fingerprint(rt))

    # KNOWLEDGE lives beside the sessions and spans them: what you told the
    # model in one conversation is findable from another, because a fact does
    # not belong to the thread that happened to mention it.
    from holographic.caching_and_storage.holographic_knowledgestore import (
        KnowledgeStore)
    import lecore as _lecore
    know = KnowledgeStore(os.path.join(sess_root, "_knowledge"),
                          session=a.session)
    _mind = _lecore.UnifiedMind(dim=512, seed=0)
    for f in a.ingest:
        made = know.add_file(f)
        print("      ingested %s -> %d chunks" % (f, len(made)))
    if a.list_sessions:
        rows = store.list()
        if not rows:
            print("      no saved conversations yet (they appear after --chat)")
        for m in rows:
            age = (time.time() - m.get("saved_at", 0)) / 3600.0
            print("      %-24s %6d tokens   last used %.1f h ago"
                  % (m["name"], m.get("n_tokens", 0), age))
        return
    if a.forget:
        print("      forgot %r: %s" % (a.forget, store.delete(a.forget)))
        return
    if a.fork:
        man = store.fork(a.session, a.fork)
        print("      forked %r -> %r (%d tokens, independent from here on)"
              % (a.session, a.fork, man.get("n_tokens", 0)))
        return
    if a.new:
        store.delete(a.session)
        dropped = know.prune(session=a.session) if a.session else []
        print("      started %r over (context cleared, %d knowledge entries "
              "from it removed)" % (a.session, len(dropped)))

    if a.scope:
        know.set_scope(a.scope, session=a.session)
        print("      %r may now reference: %s" % (a.session, a.scope))
        if not a.chat:
            return
    if a.prune is not None:
        f = {}
        for tok in a.prune:
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            if k == "session":
                f["session"] = v
            elif k == "kind":
                f["kinds"] = (v,)
            elif k == "source":
                f["sources"] = (v,)
            elif k == "days":
                f["older_than"] = float(v) * 86400.0
        if not f:
            print("      --prune needs a filter: session=NAME kind=KIND "
                  "source=SRC days=N")
            return
        doomed = know.prune(dry_run=not a.yes, **f)
        print("      %s %d entr%s:"
              % ("deleted" if a.yes else "would delete", len(doomed),
                 "y" if len(doomed) == 1 else "ies"))
        for d in doomed[:12]:
            print("        [%s/%s] %.70s" % (d["kind"], d["source"], d["preview"]))
        if not a.yes and doomed:
            print("      re-run with --yes to actually delete")
        return
    if a.knows:
        print("      scope of %r: %s" % (a.session, know.get_scope(a.session)))
        cat = know.catalog()
        print("      %d entries, %d chars" % (cat["entries"], cat["chars"]))
        for k, v in sorted(cat["by_kind"].items()):
            print("        %-10s %d" % (k, v))
        for src, v in sorted(cat["by_source"].items())[:12]:
            print("        from %-24s %d" % (src, v))
        return
    if a.recall:
        for h in know.search(_mind, a.recall, top=5):
            print("      [%s/%s%s] %.140s"
                  % (h["kind"], h["source"],
                     ("/" + h["author"]) if h.get("author") else "",
                     h["text"].replace("\n", " ")))
        return
    if a.ingest and not a.chat:
        return

    if a.chat:
        _chat(rt, cfg, store, a.session, a.tokens, n_vocab, know, _mind, tok)
        return

    if a.verify:
        ids = _tokens_from(a.verify, n_vocab, tok)[:16]
        print("[verify] leCore forward over %d tokens ..." % len(ids))
        mine = rt.forward(ids)[-1]
        ref = _reference_logits(a.model_dir, ids)
        if ref is None:
            print("      no reference runtime available (transformers not in "
                  "this python and no venv found) -- cannot cross-check")
            return
        ref = np.asarray(ref, np.float64)
        rel = float(np.max(np.abs(mine - ref)) / max(np.max(np.abs(ref)), 1e-30))
        agree = int(np.argmax(mine)) == int(np.argmax(ref))
        print("      leCore  top-5: %s" % np.argsort(mine)[-5:][::-1].tolist())
        print("      reference top-5: %s" % np.argsort(ref)[-5:][::-1].tolist())
        print("      max relative logit difference: %.3e" % rel)
        print("      SAME ARGMAX: %s" % agree)
        if rel < 1e-3 and agree:
            print("      VERIFIED -- leCore reproduces the reference on YOUR "
                  "checkpoint. Every number measured from here is the model.")
        else:
            print("      MISMATCH -- send this output. The tensor layout is "
                  "being read differently than the reference reads it; the "
                  "top-5 lists above say how badly.")
        return

    if a.assess is not None:
        from holographic.io_and_interop.holographic_assess import assess
        out_path = a.assess
        if not os.path.isabs(out_path):
            home = os.environ.get("GALVATRON_CWD") or os.path.dirname(
                os.path.abspath(__file__))
            out_path = os.path.join(home, out_path)
        print("[assess] measuring %s" % a.model_dir)
        rep = assess(a.model_dir, out_path,
                     text=(a.ppl if a.ppl and not a.ppl.startswith("@") else None),
                     progress=lambda step, d: print("      %-12s %s" % (step, d)))
        print("      wrote %s (%.2f MB)" % (rep["path"], rep["megabytes"]))
        print("      perplexity %.4f | %.1f tokens/sec | harden %s"
              % (rep["perplexity"], rep["tokens_per_second"], rep["harden"]))
        for c in rep["contains"]:
            print("        - %s" % c)
        print("      This is a PROFILE, not the model: no weight tensors, no "
              "training data, no text beyond the probe.")
        return

    if a.merge_sidecar:
        from holographic.io_and_interop.holographic_sidecar import merge
        f, od = a.merge_sidecar
        rep = merge(a.model_dir, f, od)
        print("[sidecar] merged %d deltas -> %s" % (rep["applied"], rep["out_dir"]))
        return

    if a.sidecar is not None:
        import numpy as _np
        from holographic.io_and_interop.holographic_sidecar import (
            new_sidecar, add_rows, save, load, apply_to, load_sidecar)
        from holographic.io_and_interop.holographic_boot import (
            BootRecord, write_boot)
        from holographic.io_and_interop.holographic_gdnruntime import (
            load_weights_dir, GDNRuntime)
        from holographic.io_and_interop.holographic_measure import (
            measure, better_than)
        w = load_weights_dir(a.model_dir)
        side = new_sidecar(a.model_dir, notes="built by galvatron --sidecar")
        # the boot record goes in the SIDECAR's row set, so the base file is
        # never written to and a bad record can be deleted from a manifest
        emb = next(k for k in w if k.endswith("embed_tokens.weight"))
        probe = _tokens_from("The capital of France is Paris. Water freezes at "
                             "zero degrees. A recurrent state carries what the "
                             "past can tell the future.", n_vocab, tok)[:256]
        booted, brep = write_boot({emb: _np.array(w[emb], copy=True)},
                                  BootRecord(seed="leCore",
                                             dim=int(cfg["hidden"])))
        row = int(brep["row"])
        add_rows(side, emb, {row: _np.asarray(booted[emb])[row]},
                 why="boot record: seed leCore")
        out_path = a.sidecar if os.path.isabs(a.sidecar) else os.path.join(
            a.model_dir, a.sidecar)
        srep = save(side, out_path)
        print("[sidecar] %s  (%.3f MB, %d deltas, boot row %d)"
              % (srep["path"], srep["megabytes"], srep["deltas"], row))
        if len(probe) >= 32:
            on, _ap = apply_to(w, load_sidecar(out_path), gain=1.0)
            m0 = measure(rt, probe)
            m1 = measure(GDNRuntime(on, dict(rt.cfg)), probe)
            v = better_than(m1, m0)
            print("      base %.4f | with sidecar %.4f (%+.2f%%) -> %s"
                  % (m0["perplexity"], m1["perplexity"], v["delta_pct"],
                     v["verdict"]))
        print("      the base model was NOT modified.")
        return

    if a.bios:
        from holographic.io_and_interop.holographic_bios import report, fits
        from holographic.io_and_interop.holographic_gdnruntime import (
            load_weights_dir)
        w = load_weights_dir(a.model_dir)
        ids = _tokens_from("The capital of France is Paris.", n_vocab, tok)[:16]
        p = report(w, cfg, model_dir=a.model_dir, probe_ids=ids)
        print("[bios] %s" % a.model_dir)
        print("      POST               : %s (%s)"
              % ("PASS" if p["post"]["ok"] else "FAIL", p["post"]["detail"]))
        print("      tensor root        : %s" % p["root"])
        print("      layers             : %d  (%d linear-attn, %d attention, "
              "blocks of %d)" % (p["n_layers"], len(p["gdn_layers"]),
                                 len(p["attn_layers"]), p["block_period"]))
        print("      projection layout  : %s" % p["projection_layout"])
        print("      hidden / vocab     : %d / %d declared, %d defined "
              "(%d free rows)" % (p["hidden"], p["vocab_declared"],
                                  p["vocab_defined"], p["vocab_free_rows"]))
        print("      carrier dtypes     : %s" % ", ".join(p["carrier_dtypes"]))
        print("      surface capacity   : %.2f / %.2f / %.2f MB at 1 / 2 / 4 bits"
              % tuple(p["carrier_bytes"][b] / 1e6 for b in (1, 2, 4)))
        print("      leCore installed   : %s%s"
              % (p["lecore_installed"],
                 " (seed %r)" % p["seed"] if p["seed"] else ""))
        return

    if a.install is not None:
        from holographic.io_and_interop.holographic_install import install, audit
        from holographic.io_and_interop.holographic_gdnruntime import (
            load_weights_dir)
        from holographic.io_and_interop.holographic_unicron import export_portable
        w = load_weights_dir(a.model_dir)
        ids = _tokens_from("The capital of France is Paris.", n_vocab, tok)[:32]
        if not a.install:
            rep = audit(w, cfg=cfg, probe_ids=ids)
            print("[install] AUDIT %d/%d" % (rep["passed"], rep["total"]))
            for c in rep["checks"]:
                print("   %-32s %s   %s"
                      % (c["check"], "PASS" if c["ok"] else "FAIL", c["detail"]))
                if not c["ok"]:
                    print("        why it matters: %s" % c["why"])
            if not rep["clean"]:
                print("   An install that writes cleanly and audits short is a "
                      "model carrying dead weight it will never use.")
            return
        import io as _io, os as _os, tarfile as _tar
        here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        buf = _io.BytesIO()
        with _tar.open(fileobj=buf, mode="w:xz", preset=6) as t:
            t.add(_os.path.join(here, "holographic"), arcname="holographic")
        blob = buf.getvalue()
        from holographic.caching_and_storage.holographic_substrate import (
            capacity_bytes)
        room = capacity_bytes(w, 1)
        print("[install] engine payload %.2f MB | surface at 1 bit %.2f MB"
              % (len(blob) / 1e6, room / 1e6))
        if len(blob) > room:
            # SAY WHY, DO NOT JUST FAIL. A small model genuinely cannot carry
            # the engine, and the honest answer is the boot record alone --
            # which is still a working leCore layer, because everything except
            # the DATA regenerates from the seed.
            print("      this model is too small to carry the engine "
                  "(%.1fx over). Installing the BOOT RECORD only -- the "
                  "codebook, capability table and instruction set all "
                  "regenerate from the seed, so the layer still works; only "
                  "the bundled source does not travel."
                  % (len(blob) / max(room, 1)))
            blob = None
        w2, rep = install(w, cfg, payload=blob,
                          progress=lambda step, d: print("      %s %s" % (step, d)))
        _os.makedirs(a.install, exist_ok=True)
        export_portable(w2, _os.path.join(a.install, "model.safetensors"))
        import shutil as _sh
        for f in _os.listdir(a.model_dir):
            fp = _os.path.join(a.model_dir, f)
            if _os.path.isfile(fp) and not f.endswith(".safetensors"):
                _sh.copy(fp, _os.path.join(a.install, f))
        chk = audit(w2, payload=blob, cfg=cfg, probe_ids=ids)
        print("      wrote %s | AUDIT %d/%d" % (a.install, chk["passed"],
                                                chk["total"]))
        for c in chk["checks"]:
            if not c["ok"]:
                print("      FAILED %s: %s" % (c["check"], c["detail"]))
        return

    if a.transform:
        from holographic.io_and_interop.holographic_transform import analyse, plan
        from holographic.io_and_interop.holographic_gdnruntime import load_weights_dir
        w = load_weights_dir(a.model_dir)
        an = analyse(w, cfg)
        print("[transform] %d blocks of %d: %d linear-attention layers, "
              "%d full-attention"
              % (len(an["attn_layers"]), an["block_period"],
                 len(an["gdn_layers"]), len(an["attn_layers"])))
        for pos, med in sorted(an["median_by_position"].items()):
            print("      position %d in block: median half-life %8.1f tokens"
                  % (pos, med))
        p = plan(w, cfg)
        buckets = {}
        for act in p["actions"]:
            buckets.setdefault(act["do"], []).append(act["layer"])
        print()
        for what, layers in sorted(buckets.items()):
            print("      %-12s %s" % (what, layers))
        print()
        print("      %s" % p["actions"][0]["why"])
        print("      Apply it with: mind.unicron_retarget(w, cfg, apply=True)")
        return

    if a.testkit_all:
        from holographic.io_and_interop.holographic_testkit import export_all
        out_dir = a.testkit_all
        if not os.path.isabs(out_dir):
            # default under the ASSIMILATION folder, where the user is standing,
            # not under the repo root the launcher cd'd to
            home = os.environ.get("GALVATRON_CWD") or os.path.dirname(
                os.path.abspath(__file__))
            out_dir = os.path.join(home, out_dir)
        n_layers = int(cfg["n_layers"])
        spec = (a.layers or "first,mid,last").strip().lower()
        if spec in ("all", "*"):
            want = None
        else:
            named = {"first": 0, "mid": n_layers // 2, "middle": n_layers // 2,
                     "last": n_layers - 1}
            want = []
            for tok_ in spec.replace(" ", "").split(","):
                if not tok_:
                    continue
                want.append(named[tok_] if tok_ in named else int(tok_))
            want = sorted(set(want))
        print("[testkit] writing %s to %s"
              % ("ALL %d layers" % n_layers if want is None
                 else "layers %s" % want, out_dir))
        def _p(L, path, mb):
            print("      layer %2d -> %-18s %6.1f MB"
                  % (L, os.path.basename(path), mb), flush=True)
        rep = export_all(a.model_dir, out_dir, progress=_p, layers=want)
        print("      %d files, %.1f MB total (%d of %d layers)"
              % (len(rep["files"]), rep["total_megabytes"], rep["layers"],
                 rep["of_layers"]))
        print("      base.npz carries spectra, gates, activations and logits;")
        print("      each layer_NN.npz stands alone -- send whichever are wanted.")
        return

    if a.testkit:
        from holographic.io_and_interop.holographic_testkit import export
        rep = export(a.model_dir, a.testkit,
                     probe=(a.ppl if a.ppl and not a.ppl.startswith("@") else None))
        print("      wrote %s (%.2f MB, %d arrays)"
              % (rep["path"], rep["megabytes"], rep["arrays"]))
        for c in rep["contains"]:
            print("        - %s" % c)
        print("      layer exported: %s" % rep["layer_exported"])
        print("      This is a PROFILE, not the checkpoint: no full weight set, "
              "no training data, no text beyond the probe.")
        return

    if a.keys:
        for L in [int(x) for x in str(a.keys).split(",") if x.strip().isdigit()]:
            ks = rt.layer_keys(L)
            kind = "GDN (linear_attn)" if rt._is_gdn(L) else "full attention"
            print("      layer %-3d %-18s %d tensors" % (L, kind, len(ks)))
            for k in ks:
                print("          %s" % k)
        return

    if a.check_tokenizer:
        mine = _tokens_from(a.check_tokenizer, n_vocab, tok)
        print("      vocabulary : %s"
              % ("stdlib BPE, %d entries" % len(tok.encoder) if tok
                 else "byte-level (%d)" % n_vocab))
        print("      leCore ids : %s" % (mine[:24] + (["..."] if len(mine) > 24 else [])))
        print("      round trip : %r" % _detok(mine, tok, n_vocab))
        ref = _reference_ids(a.model_dir, a.check_tokenizer)
        if ref is not None and (not ref or len(ref) > 8 * len(mine) + 8):
            # A reference that returns nothing (or wildly more tokens than
            # characters) did not actually load this model's vocabulary -- it
            # is an ABSENT reference, not a disagreement. Reporting it as a
            # MISMATCH is a false alarm about the scariest possible failure,
            # which is worse than reporting nothing at all.
            print("      reference tokenizer loaded but produced %d ids -- "
                  "treating it as ABSENT rather than as a mismatch" % len(ref))
            ref = None
        if ref is None:
            print("      no reference tokenizer found (transformers is not in "
                  "this python and no assimilation venv was located) -- the "
                  "round trip above is still the useful check")
        else:
            print("      reference  : %s" % ref)
            same = (list(mine) == list(ref))
            print("      MATCH: %s" % same)
            if not same:
                print("      MISMATCH -- send this output. A tokenizer that is "
                      "almost right makes the MODEL look broken, and every "
                      "number measured after it would be wrong.")
        return

    if a.report:
        import lecore
        mind = lecore.UnifiedMind(dim=512, seed=0)
        print("[2/3] unicron_report ...")
        rep = mind.unicron_report(dict(rt.w) if not hasattr(rt.w, "_codes")
                                  else {k: rt.w[k] for k in rt.w},
                                  sample_layers=12)
        c = rep["census"]
        print("      regimes: %d examined | %d heavy-tail | %d spike+bulk | "
              "%d policy-skipped" % (c["examined"], c["heavy_tail"],
                                     c["spike_bulk"], c["policy_skipped"]))
        if rep["heads"]:
            print("      inferred attention heads (blind): %s   [%s]"
                  % (rep["heads"]["inferred_heads"],
                     rep["heads"].get("reason", "")))
        for role, d in (rep["depth"] or {}).items():
            print("      depth sharing %-28s shared_frac %.3f (chance %.3f)"
                  % (role, d["shared_frac"], d["chance"]))
        for lv in rep["levers"]:
            print("      LEVER  %-46s %s" % (lv["lever"], lv["verdict"]))
        for wmsg in rep["warnings"]:
            print("      WARN   %s" % wmsg)

    if a.repair:
        from holographic.io_and_interop.holographic_galvapack import (
            repair_regressions)
        orig_dir = _resolve_model_dir(a.repair)
        text = a.ppl or ("The capital of France is Paris. Water freezes at zero "
                         "degrees and boils at one hundred. A recurrent state "
                         "carries what the past can tell the future.")
        if text.startswith("@"):
            with open(text[1:], encoding="utf-8", errors="ignore") as f:
                text = f.read()
        ids = _tokens_from(text, n_vocab, tok)[:256]
        print("[repair] scoring on %d tokens; every changed tensor is tested "
              "against the original" % len(ids))
        def _prog(i, name, ppl):
            print("      [%3d] %-52s ppl %.4f" % (i + 1, name[-52:], ppl), flush=True)
        _w, rep = repair_regressions(orig_dir, a.model_dir, ids,
                                     out_dir=(a.imbue or None), progress=_prog)
        print("      changed %d | reverted %d, blended %d, kept %d"
              % (rep["changed"], rep["reverted"], rep["blended"], rep["kept"]))
        print("      original    %.4f" % rep["perplexity_original"])
        print("      assimilated %.4f" % rep["perplexity_assimilated"])
        print("      REPAIRED    %.4f   beats the original: %s"
              % (rep["perplexity_repaired"], rep["beats_original"]))
        if rep.get("out_dir"):
            print("      wrote %s" % rep["out_dir"])
        else:
            print("      (add --imbue OUT_DIR to write the repaired weights)")
        return

    if a.imbue:
        import lecore as _lc
        from holographic.io_and_interop.holographic_galvapack import imbue as _imbue
        # THE CORPUS IS THE USER'S DATA, NOT OURS. --prove may fall back to
        # leCore's notes because it is a demonstration; a model someone is going
        # to SHIP must not silently carry this repository's documentation.
        if a.doc:
            corpus, source = _grounding_corpus(a.doc)
        else:
            corpus, source = [], ("none -- pass --doc FILE to give it a "
                                  "grounding corpus")
        banned = _tokens_from(a.ban, n_vocab, tok) if a.ban else []
        print("[imbue] corpus: %s (%d passages); banned tokens: %d"
              % (source, len(corpus), len(banned)))
        rep = _imbue(a.model_dir, a.imbue, _lc.UnifiedMind(dim=512, seed=0),
                     corpus=corpus, banned=banned)
        print("      wrote %s  (%.1f MB)" % (a.imbue, rep.get("bytes", 0) / 1e6))
        print("      residents: %d  %s" % (rep["residents"], rep["kinds"]))
        for sk in rep.get("skipped", []):
            print("      skipped: %s" % (sk,))
        print("      calibrated on %d probe tokens" % rep.get("calibrated_on", 0))
        print("      run it:  python %s/galvatron.py chat"
              % os.path.abspath(a.imbue).replace("\\", "/"))
        print("      (an absolute path, because the repo has its own run.py "
              "and running the wrong one gives a confusing argparse error)")
        print("      NOTE: model.safetensors inside is an ORDINARY checkpoint. "
              "Load it elsewhere and every resident is gone -- they are "
              "reconstructed from the manifest by leCore, not stored in weights.")
        return

    if a.prove is not None:
        _prove(rt, cfg, tok, n_vocab, a.prove or
               "The capital of France is", a.doc, a.tokens)
        return

    if a.compare and a.ppl:
        other_dir = _resolve_model_dir(a.compare)
        text = a.ppl
        if text.startswith("@"):
            with open(text[1:], encoding="utf-8", errors="ignore") as f:
                text = f.read()
        ids, spans = _chunk_passages(text, tok, n_vocab, int(a.chunks))
        chunks = [ids[a0:b0] for a0, b0 in spans]
        sizes = [len(c) for c in chunks]
        print("[compare] %d passages of %d-%d tokens (scored IN CONTEXT: one "
              "pass over the whole text, losses bucketed per passage)"
              % (len(chunks), min(sizes), max(sizes)))
        print("[compare] loading %s ..." % other_dir)
        rt2, cfg2 = load_runtime(other_dir, lazy=a.lazy)
        # ONE forward per model over the FULL text -- every token keeps its real
        # preceding context, and the passage numbers become comparable
        nll1 = rt.token_nll(ids)
        nll2 = rt2.token_nll(ids)
        # nll[i] scores token i+1, so a span [a,b) of tokens maps to nll[a-1:b-1]
        bounds = [(max(a0 - 1, 0), min(b0 - 1, len(nll1))) for a0, b0 in spans]
        rows = []
        for i, (lo, hi) in enumerate(bounds):
            if hi <= lo:
                continue
            p1 = float(np.exp(nll1[lo:hi].mean()))
            p2 = float(np.exp(nll2[lo:hi].mean()))
            rows.append((p1, p2))
            # SHOW THE WHOLE PASSAGE. A 26-character preview made complete
            # sentences look like truncated fragments, and a reader cannot
            # verify the split from an ellipsis -- the display was lying about
            # data that was correct.
            txt = _detok(chunks[i], tok, n_vocab).strip()
            print("      passage %d (%d tokens)  %10.4f -> %10.4f   %+.2f%%"
                  % (i + 1, len(chunks[i]), p1, p2, 100.0 * (p2 - p1) / p1))
            for line in _wrap(txt, 92):
                print("          %s" % line)
        A = np.array([r[0] for r in rows]); Bv = np.array([r[1] for r in rows])
        rel = 100.0 * (Bv - A) / A
        whole1 = float(np.exp(nll1.mean())); whole2 = float(np.exp(nll2.mean()))
        print()
        print("      A = %s" % a.model_dir)
        print("      B = %s" % other_dir)
        print("      WHOLE TEXT        A %.4f   B %.4f   (%+.2f%%)"
              % (whole1, whole2, 100.0 * (whole2 - whole1) / whole1))
        print("      per-passage mean  A %.4f   B %.4f" % (A.mean(), Bv.mean()))
        print("      RETENTION DELTA   %+.2f%%  (spread %.2f, range %+.2f%% .. %+.2f%%)"
              % (rel.mean(), rel.std(), rel.min(), rel.max()))
        print("      B was worse on %d of %d passages" % (int((Bv > A).sum()), len(rows)))
        if np.allclose(rel, 0.0):
            print("      IDENTICAL: same perplexity on every passage.")
            return
        if len(rel) < 2:
            print("      ONE PASSAGE: no spread, so no error estimate exists.")
            return
        stderr = rel.std() / max(np.sqrt(len(rel)), 1.0)
        print("      standard error of the mean: %.2f%%  (n=%d passages)"
              % (stderr, len(rel)))
        if stderr <= 1e-9:
            print("      every passage shifted identically -- check the two "
                  "directories actually differ.")
            return
        if abs(rel.mean()) < 2.0 * stderr:
            have = sum(sizes)
            need_tok = int(have * (2.0 * rel.std() / max(abs(rel.mean()), 1e-9)) ** 2)
            print("      NOT DISTINGUISHABLE on this text: within 2 standard "
                  "errors of zero. Resolving a %+.2f%% effect at this spread "
                  "needs roughly %d tokens of text (you gave %d) -- more TEXT, "
                  "not more passages of the same text."
                  % (rel.mean(), max(need_tok, have * 2), have))
        else:
            print("      MEASURED: %.1f standard errors from zero."
                  % (abs(rel.mean()) / stderr))
        return

    if a.ppl:
        text = a.ppl
        if text.startswith("@"):
            with open(text[1:], encoding="utf-8", errors="ignore") as f:
                text = f.read()
        ids = _tokens_from(text, n_vocab, tok)
        print("[3/3] perplexity over %d tokens (in-engine, no torch) ..." % len(ids))
        t0 = time.time()
        p = rt.perplexity(ids)
        print("      PERPLEXITY %.4f   (%.1fs)" % (p, time.time() - t0))
        print("      run this on the ORIGINAL and the ASSIMILATED directory; the "
              "delta is the retention number the transform reports as UNVERIFIED")

    if a.generate and not a.demo:
        ids = _tokens_from(a.generate, n_vocab, tok)
        t0 = time.time()
        out, _st = rt.generate_fast(ids, n_new=a.tokens)
        new_ids = out[len(ids):]
        print("      generated (%.1fs): %s"
              % (time.time() - t0, repr(_detok(new_ids, tok, n_vocab))))

    if a.leap:
        from holographic.agents_and_reasoning.holographic_leap import (
            RouteMemory, leap_generate)
        ids = _tokens_from(a.generate or "The holographic engine", n_vocab, tok)
        print("[leap] plain generation ...")
        t0 = time.time()
        base, _ = rt.generate_fast(ids, n_new=a.tokens)
        t_plain = time.time() - t0
        print("       %.2fs for %d tokens (%.2f tok/s)"
              % (t_plain, a.tokens, a.tokens / max(t_plain, 1e-9)))
        print("[leap] cold memory (route never walked) ...")
        t0 = time.time()
        got, mem, rep = leap_generate(rt, ids, n_new=a.tokens, k=8)
        t_cold = time.time() - t0
        print("       %.2fs | acceptance %.2f | identical: %s"
              % (t_cold, rep["acceptance_rate"], got == base))
        print("[leap] warm memory (same route again) ...")
        t0 = time.time()
        got2, _m, rep2 = leap_generate(rt, ids, n_new=a.tokens, memory=mem, k=8)
        t_warm = time.time() - t0
        print("       %.2fs | acceptance %.2f | identical: %s | SPEEDUP %.2fx"
              % (t_warm, rep2["acceptance_rate"], got2 == base,
                 t_plain / max(t_warm, 1e-9)))
        if got != base or got2 != base:
            raise SystemExit("OUTPUT DIVERGED -- this must never happen; report it")

    if a.demo:
        import lecore
        from holographic.agents_and_reasoning.holographic_galvatron import (
            Galvatron, OracleResident, WardResident)
        from holographic.agents_and_reasoning.holographic_knowres import (
            SalienceTrigger)
        mind = lecore.UnifiedMind(dim=512, seed=0)
        ids = _tokens_from(a.generate or "The holographic engine", n_vocab, tok)
        probe_layer = max(0, cfg["n_layers"] - 2)

        print("[demo] bare generation ...")
        bare, _ = rt.generate_fast(ids, n_new=8)
        print("       bare: %r" % _detok(bare[len(ids):], tok, n_vocab))

        cap = {}
        rt.forward(ids, hooks={probe_layer:
                               lambda h: cap.__setitem__("h", h.copy()) or None})

        print("[demo] WARD: banning exactly what it just said ...")
        ward = WardResident(banned=sorted(set(bare[len(ids):])))
        warded, _ = Galvatron(rt, guards=[ward]).generate(ids, n_new=8)
        leaked = set(warded[len(ids):]) & set(bare[len(ids):])
        print("       warded: %r | ban breached: %s"
              % (_detok(warded[len(ids):], tok, n_vocab), bool(leaked)))

        print("[demo] SALIENCE: does a TRAINED model's hesitation actually vary?")
        sal = SalienceTrigger(rt)
        sal.calibrate(cap["h"], quantile=0.8)
        scores = np.array([sal.score(x) for x in cap["h"]])
        print("       lens entropy over %d positions: mean %.3f  spread %.3f  "
              "min %.3f  max %.3f" % (len(scores), scores.mean(), scores.std(),
                                      scores.min(), scores.max()))
        print("       (on the tiny RANDOM test model spread was 0.007 -- a real "
              "spread here is the result that makes salience gating meaningful)")

        print("[demo] ORACLE: a memory keyed on a live hidden state ...")
        target = int(np.argsort(rt.forward(ids)[-1])[-5])   # a plausible-but-not-top token
        orc = OracleResident(mind, cfg["hidden"], layer=probe_layer,
                             gain=1.0, threshold=0.0)
        orc.remember(cap["h"][-1], 8.0 * np.asarray(rt.embed[target], np.float64))
        g = Galvatron(rt, residents=[orc])
        top = int(np.argmax(rt.forward(ids, hooks=g._hooks())[-1]))
        print("       base top %d -> with memory %d (target %d) | steered: %s"
              % (int(np.argmax(rt.forward(ids)[-1])), top, target, top == target))

        print("[demo] TIME TRAVEL: snapshot, branch, rewind ...")
        _lg, st = rt.prefill(ids)
        snap = st.copy()
        a1, _ = rt.generate_fast(ids, n_new=5, state=st)
        a2, _ = rt.generate_fast(ids, n_new=5, state=snap.copy())
        print("       rewind reproduces timeline exactly: %s" % (a1 == a2))


def _make_schedule(rt, cfg, n_vocab):
    """The leCore per-turn schedule, or None if this model cannot carry one.

    Returns fn(ids) -> a short human line, or None. Built once per session
    because the reservation and codebook regenerate from a seed and must not
    change between turns -- a register file with a different basis each turn is
    not a register file."""
    import numpy as _np
    try:
        from holographic.caching_and_storage.holographic_keyreserve import (
            reserve, delta_write, delta_read)
        from holographic.agents_and_reasoning.holographic_hybrid import split
    except Exception:
        return None
    H = int(cfg["hidden"])
    K = reserve(H, min(32, H // 4), seed=0)
    # CODEBOOK CHOICE IS A CROSSOVER, NOT A PREFERENCE. A random codebook is a
    # K x D matmul against BLAS; a Hadamard codebook cleans up in ONE transform,
    # O(D log D). MEASURED: at K=256/D=512 the matmul WINS (0.54x) and at
    # K=1024/D=512 Hadamard wins 2.15x; at a real vocabulary it is not close --
    # K=131,072 / D=1024 reads 635x, and the codebook is GENERATED from a seed
    # rather than stored, so it is 64 bits against 1,074 MB.
    # My first measurement ran at K=1024, D=512 and read 0.93x. THE WRONG SCALE
    # ANSWERED THE WRONG QUESTION, and a small fixture is exactly where that
    # mistake is easy to make.
    _use_ht = int(n_vocab) >= 512
    if _use_ht:
        import lecore as _lc
        _hc = _lc.UnifiedMind(dim=H, seed=0).hadamard_codebook(dim=H, seed=0)
        CB = None
    else:
        _hc = None
        g = _np.random.default_rng(0)
        CB = g.standard_normal((int(n_vocab), H))
        CB /= _np.linalg.norm(CB, axis=1, keepdims=True) + 1e-30

    def _atom(i):
        return _hc.atom(int(i) % _hc.K) if _hc is not None else CB[int(i)]

    def _clean(v):
        if _hc is not None:
            r = _hc.cleanup(v)
            return int(r[0] if isinstance(r, tuple) else r)
        sc = CB @ v
        return int(_np.argmax(sc))

    def _score(v):
        if _hc is not None:
            r = _hc.correlations(v)
            return float(_np.max(_np.abs(_np.asarray(r))))
        return float(_np.max(CB @ v))
    # ACT-R ACTIVATION, so a full file EVICTS rather than REFUSES. Measured
    # without it: 30 tokens filled 32 registers on turn one and every later turn
    # reported "registers full". A memory that stops accepting after one turn is
    # a buffer. Base-level activation A = ln(sum_j age^-0.5) ranks by recency AND
    # frequency together, and the lowest-activation slot is the one to overwrite.
    box = {"S": _np.zeros((H, H)), "n": 0, "uses": {}, "clock": 0}
    carried = set()          # what EARLIER turns put in the file

    def run(ids):
        ids = list(ids)
        if len(ids) < 16:
            return ""
        lg = _np.asarray(rt.forward(ids), _np.float64)[:-1]
        sp = split(lg, quantile=0.90)
        tgt = _np.asarray(ids[1:])
        from holographic.agents_and_reasoning.holographic_actr import (
            base_level)
        stored = evicted = 0
        for t in _np.flatnonzero(sp["store"]):
            box["clock"] += 1
            if box["n"] < len(K):
                slot = box["n"]
                box["n"] += 1
            else:
                # EVICT THE LEAST ACTIVE, not the oldest. A slot used three
                # times long ago can outrank one used once recently, and that
                # is the whole point of base-level activation.
                slot = min(range(len(K)),
                           key=lambda j: base_level(box["uses"].get(j, [0.0]),
                                                    box["clock"]))
                evicted += 1
            box["S"] = delta_write(box["S"], K[slot], _atom(int(tgt[t])))
            box["uses"].setdefault(slot, []).append(float(box["clock"]))
            stored += 1
        if not stored:
            return "nothing uncertain enough to store this turn"
        # ---- THE READ SIDE. Storing without consulting is a write-only
        # memory, and that is what this loop was until now: it counted readable
        # slots and never asked one a question. MEASURED on a second encounter
        # with the same material: the model is 9.4% top-1 on the positions the
        # store holds and THE STORE IS 100%. The gap is the whole reason to
        # carry registers at all, and it was going unspent every turn.
        # WHAT THE PRIOR TURNS CAN ANSWER FOR THIS ONE. The obvious version of
        # this counter was CIRCULAR: checking whether a recalled token appears
        # in the set just stored from measures "did I store what I stored" and
        # reads ~100% by construction. The honest question is whether registers
        # written on EARLIER turns cover positions THIS turn is unsure about, so
        # the carried set is captured BEFORE this turn's writes.
        ok = 0
        for j in range(box["n"]):
            r = delta_read(box["S"], K[j])
            if _score(r / (_np.linalg.norm(r) + 1e-30)) > 0.5:
                ok += 1
        uncertain_now = set(int(x) for x in tgt[sp["store"]])
        hits = len(uncertain_now & carried)
        for j in range(box["n"]):
            r = delta_read(box["S"], K[j])
            rn = r / (_np.linalg.norm(r) + 1e-30)
            if _score(rn) > 0.5:
                carried.add(_clean(rn))
        return ("stored %d uncertain token(s)%s, %d/%d registers readable, "
                "%d of this turn's uncertain tokens were ALREADY held"
                % (stored, (" (evicted %d by lowest activation)" % evicted)
                   if evicted else "", ok, box["n"], hits))

    return run


def _chat(rt, cfg, store, session, n_tokens, n_vocab, know=None, mind=None,
          tok=None):
    """Interactive conversation with context that SURVIVES THE PROCESS.

    Each turn is appended to the session's inference state and saved, so the
    next run of this script picks the conversation up mid-thought -- no
    re-prefill of the history, no transcript replay, and no external harness
    required. Commands: /new /list /fork NAME /switch NAME /quit."""
    # BUILT ONCE PER SESSION, not per turn: the reservation and codebook
    # regenerate from a seed and must not change between turns, or the register
    # file has a different basis each time and is not a register file.
    _sched = _make_schedule(rt, cfg, n_vocab)
    if _sched is not None:
        print("      leCore schedule active -- uncertain tokens go to registers")
    state, history = None, []
    try:
        state, man, _m = store.load(session)
        history = man.get("tokens") or []
        print("      resumed %r (%d tokens of context)" % (session, len(history)))
    except (FileNotFoundError, OSError):
        print("      new conversation %r" % session)
    print("      commands: /new  /list  /fork NAME  /switch NAME  /quit")
    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print("\n      saved. run again to resume %r." % session)
            return
        if not line.strip():
            continue
        if line.startswith("/"):
            cmd = line.split()
            if cmd[0] == "/ingest" and len(cmd) > 1 and know is not None:
                try:
                    print("      ingested %s -> %d chunks"
                          % (cmd[1], len(know.add_file(cmd[1]))))
                except OSError as exc:
                    print("      could not read %s: %s" % (cmd[1], exc))
                continue
            if cmd[0] == "/recall" and len(cmd) > 1 and know is not None:
                for h in know.search(mind, " ".join(cmd[1:]), top=4):
                    print("      [%s/%s] %.120s"
                          % (h["kind"], h["source"], h["text"].replace("\n", " ")))
                continue
            if cmd[0] == "/note" and len(cmd) > 1 and know is not None:
                know.add_note(" ".join(cmd[1:]), author="user")
                print("      noted"); continue
            if cmd[0] == "/knows" and know is not None:
                print("      scope: %s | %s"
                      % (know.get_scope(session), know.catalog())); continue
            if cmd[0] == "/scope" and know is not None:
                if len(cmd) > 1:
                    try:
                        know.set_scope(cmd[1], session=session)
                        print("      %r may now reference: %s" % (session, cmd[1]))
                    except ValueError as exc:
                        print("      %s" % exc)
                else:
                    print("      scope of %r: %s  (all|session|none)"
                          % (session, know.get_scope(session)))
                continue
            if cmd[0] == "/prune" and len(cmd) > 1 and know is not None:
                f = {}
                for tok in cmd[1:]:
                    if "=" not in tok:
                        continue
                    k, v = tok.split("=", 1)
                    if k == "session":
                        f["session"] = v
                    elif k == "kind":
                        f["kinds"] = (v,)
                    elif k == "source":
                        f["sources"] = (v,)
                    elif k == "days":
                        f["older_than"] = float(v) * 86400.0
                if not f:
                    print("      /prune session=NAME | kind=KIND | source=SRC | days=N")
                    continue
                d = know.prune(**f)
                print("      deleted %d entries" % len(d)); continue
            if cmd[0] == "/quit":
                print("      saved. run again to resume %r." % session)
                return
            if cmd[0] == "/new":
                store.delete(session); state, history = None, []
                print("      started %r over" % session); continue
            if cmd[0] == "/list":
                for m in store.list():
                    print("      %-20s %6d tokens" % (m["name"], m.get("n_tokens", 0)))
                continue
            if cmd[0] == "/fork" and len(cmd) > 1:
                if state is not None:
                    store.save(session, state, tokens=history)
                store.fork(session, cmd[1])
                print("      forked to %r" % cmd[1]); continue
            if cmd[0] == "/switch" and len(cmd) > 1:
                if state is not None:
                    store.save(session, state, tokens=history)
                session = cmd[1]
                try:
                    state, man, _m = store.load(session)
                    history = man.get("tokens") or []
                    print("      switched to %r (%d tokens)" % (session, len(history)))
                except (FileNotFoundError, OSError):
                    state, history = None, []
                    print("      switched to new conversation %r" % session)
                continue
            print("      unknown command"); continue
        # EVERY TURN IS FILED, automatically. The user should not have to
        # decide in advance which sentence will matter in three weeks.
        if know is not None:
            know.add(line, kind="turn", source="user", session=session)
        ids = tok.encode(line) if tok else [int(b) for b in line.encode("utf-8")
                                           if int(b) < n_vocab]
        if state is None:
            out, state = rt.generate_fast(ids, n_new=n_tokens)
            history = ids
        else:
            _lg, state = rt.extend(ids, state)
            history = list(history) + ids
            out, state = rt.generate_fast(history, n_new=n_tokens, state=state)
            # ---- THE leCORE SCHEDULE, run on the turn just produced.
            # The chat loop carried state across turns and used NONE of the
            # installed architecture -- the same disease the usage audit found
            # in the library modules. This is the loop from
            # holographic_lecorerun, applied here: read the model's OWN entropy
            # off the logits it just made, and store what it could not predict
            # so the NEXT turn can recall it exactly. Measured elsewhere at 100%
            # recall against 9% top-1 on identical positions.
            if _sched is not None:
                try:
                    _rep = _sched(history)
                    if _rep:
                        print("      [leCore] %s" % _rep)
                except Exception as _exc:
                    print("      [leCore] schedule skipped: %s"
                          % type(_exc).__name__)
        history = out
        store.save(session, state, tokens=history)
        new_ids = out[-n_tokens:]
        text = _detok(new_ids, tok, n_vocab)
        if know is not None:
            know.add(text, kind="output", source="model", session=session)
        print("bot> %s" % text)
        if know is not None and mind is not None:
            rel = know.search(mind, line, top=1, kinds=("turn", "document", "note"))
            if rel and rel[0]["score"] > 0 and rel[0]["text"][:40] not in line:
                print("      [recalled %s/%s: %.90s]"
                      % (rel[0]["kind"], rel[0]["source"],
                         rel[0]["text"].replace("\n", " ")))


if __name__ == "__main__":
    main()
