"""A console for the one model.

This is a small web UI for `UnifiedMind` -- the top-level model that perceives any
input into one holographic space and runs every operation over it. You pick a real
corpus, the server PULLS it on demand (NLTK's data lives on GitHub, the same place the
test data came from), trains one `UnifiedMind` on it, and then you can exercise all four
operations against the same trained mind:

  * TRAIN     -- learn a corpus's documents (classification + recall) and its raw text
                 (generation), all into one self-organizing memory.
  * CLASSIFY  -- 'what is this?' nearest self-organized prototype, routed to its modality.
  * RECALL    -- 'what's like this?' nearest stored individual.
  * ORGANIZE  -- show how the memory split each label into sub-prototypes, and reorganize.
  * GENERATE  -- continue text in the style of what was learned.

Run:  python unified_app.py   then open http://127.0.0.1:5001

If a corpus is not present it is downloaded via nltk (needs a network connection the
first time); everything degrades gracefully and tells you what to install.
"""

import os
import re
from collections import defaultdict

import numpy as np
from flask import Flask, request, jsonify, render_template_string

from holographic_unified import UnifiedMind
from holographic_text import STOPWORDS

app = Flask(__name__)

# one trained mind lives here between requests (single-process dev server)
STATE = {"mind": None, "dataset": None, "labels": [], "test": [], "raw_len": 0,
         "trained_ids": [], "trained_on": []}

# Trained-brain cache, now keyed by the FULL ordered training STACK (tuple of dataset
# ids). One mind can be trained on several datasets in sequence -- e.g. curriculum to
# lay down a base structure, then books, then reuters -- and the brain remembers what
# it was trained on (STATE["trained_on"]). Building a stack trains each dataset into
# one mind in order; intermediate prefixes are cached too, so extending a stack reuses
# the work already done. Snapshots are deep copies so a cached prefix is never
# corrupted when a longer stack builds on top of it. A "fresh" request rebuilds.
TRAINED = {}                       # (id, id, ...) -> (state_snapshot, build_result)


def _snapshot_state():
    import copy
    return copy.deepcopy(dict(STATE))


def _restore(snapshot):
    import copy
    STATE.clear()
    STATE.update(copy.deepcopy(snapshot))


def build_stack(ids, fresh=False):
    """Train one mind on an ordered stack of datasets, reusing cached prefixes. ids is
    a tuple of dataset ids in training order. Returns the build-result for the last
    dataset added, enriched with the full 'trained_on' list and a 'cached' flag."""
    import copy
    ids = tuple(ids)
    if not fresh and ids in TRAINED:
        snap, result = TRAINED[ids]
        _restore(snap)
        out = dict(result); out["cached"] = True
        out["trained_on"] = list(STATE.get("trained_on", []))
        out["trained_ids"] = list(STATE.get("trained_ids", []))
        return out

    if len(ids) == 1:
        # base of the stack: a fresh mind trained on the one dataset
        mind = UnifiedMind(dim=2048, seed=0, text_window=3)
        STATE.clear()
        STATE.update({"mind": mind, "trained_ids": [], "trained_on": []})
        result = _absorb_into(mind, ids[0])
    else:
        # build (or reuse) the prefix, take an independent copy of its mind, and train
        # the last dataset on top -- so the prefix's cached mind is left untouched
        build_stack(ids[:-1], fresh=False)
        mind = copy.deepcopy(STATE["mind"])
        prev_ids = list(STATE.get("trained_ids", []))
        prev_on = list(STATE.get("trained_on", []))
        STATE["mind"] = mind
        STATE["trained_ids"] = prev_ids
        STATE["trained_on"] = prev_on
        result = _absorb_into(mind, ids[-1])

    TRAINED[ids] = (_snapshot_state(), dict(result))
    out = dict(result); out["cached"] = False
    out["trained_on"] = list(STATE.get("trained_on", []))
    out["trained_ids"] = list(STATE.get("trained_ids", []))
    return out


# kept for backward compatibility (single-dataset load)
def build_cached(dataset_id, fresh=False):
    return build_stack((dataset_id,), fresh=fresh)


def build(dataset_id):
    """Backward-compatible single-dataset build: train a fresh mind on one dataset
    (resets the working stack first). Equivalent to a 'Start fresh' load."""
    STATE.clear()
    STATE.update({"mind": None, "dataset": None, "labels": [], "test": [],
                  "raw_len": 0, "trained_ids": [], "trained_on": []})
    mind = UnifiedMind(dim=2048, seed=0, text_window=3)
    STATE.update({"mind": mind, "trained_ids": [], "trained_on": []})
    return _absorb_into(mind, dataset_id)


# ---------------------------------------------------------------------------
# pulling + shaping real corpora (NLTK data, hosted on GitHub)
# ---------------------------------------------------------------------------

def _ensure(pkg):
    """Make sure an NLTK corpus is available, downloading it on demand."""
    import nltk
    nltk.data.path.insert(0, "/home/claude/nltk_data")
    try:
        nltk.download(pkg, quiet=True)
        return True
    except Exception:
        return False


def _content(tokens):
    return [w for w in tokens if w not in STOPWORDS]


def _detok(words):
    """Join corpus word-tokens back into readable text: punctuation attaches to
    the preceding word ('the dog , ran .' -> 'the dog, ran.'). Generation wants
    text that LOOKS like writing -- case and punctuation kept (measured on
    Austen: true raw costs ~12% bits/char vs the scrubbed diet but the output
    becomes prose; see README)."""
    out = []
    for w in words:
        if out and (w in {",", ".", ";", ":", "!", "?", ")", "''", "'"}
                    or w.startswith("'")):
            out[-1] += w
        elif out and out[-1] in {"(", "``"}:
            out[-1] += w
        else:
            out.append(w)
    return " ".join(out).replace("``", '"').replace("''", '"')


def load_reuters():
    from nltk.corpus import reuters
    single = [(f, reuters.categories(f)[0]) for f in reuters.fileids()
              if len(reuters.categories(f)) == 1]
    top = ["earn", "acq", "crude", "trade", "money-fx", "interest",
           "money-supply", "ship", "sugar", "coffee"]
    by = defaultdict(list)
    for f, c in single:
        if c in top:
            by[c].append(f)
    items, raw = [], []
    for c, fids in by.items():
        for f in fids[:150]:
            toks = [w.lower() for w in reuters.words(f) if w.isalpha()]
            items.append((_content(toks), c))
            # generation learns from the TRUE article text -- case and
            # punctuation kept (the scrubbed token diet made every generated
            # sentence lowercase and unpunctuated; user-caught)
            raw.append(re.sub(r"\s+", " ", reuters.raw(f)))
    return items, " ".join(raw), "Reuters financial newswire -- 10 confusable categories (grain/crude/money-fx share vocabulary)"


def load_brown():
    from nltk.corpus import brown
    cats = ["news", "romance", "science_fiction", "government", "hobbies"]
    items, raw = [], []
    for c in cats:
        words = [w.lower() for w in brown.words(categories=c) if w.isalpha()]
        true_words = list(brown.words(categories=c))[:20000]
        for k in range(0, min(len(words), 18000) - 300, 300):
            chunk = words[k:k + 300]
            items.append((_content(chunk), c))
        raw.append(_detok(true_words))             # generation: real text
    return items, " ".join(raw), "Brown corpus -- five prose genres, in 300-word chunks"


def load_gutenberg():
    from nltk.corpus import gutenberg
    books = {"austen-emma.txt": "Austen", "carroll-alice.txt": "Carroll",
             "shakespeare-hamlet.txt": "Shakespeare", "melville-moby_dick.txt": "Melville",
             "chesterton-brown.txt": "Chesterton"}
    items, docs = [], []
    for fid, author in books.items():
        if fid not in gutenberg.fileids():
            continue
        words = [w.lower() for w in gutenberg.words(fid) if w.isalpha()][:9000]
        for k in range(0, len(words) - 200, 200):
            chunk = words[k:k + 200]
            items.append((_content(chunk), author))
        # generation corpus is a LIST OF (text, source) docs -> PROVENANCE:
        # generated or pasted text can be traced back to the author who taught
        # the transitions it used (measured 92% top-1 on held-out passages)
        docs.append((re.sub(r"\s+", " ", gutenberg.raw(fid))[:40000], author))
    return items, docs, ("Project Gutenberg -- classify the author, generate in their "
                         "style, and TRACE text back to its source author")


def load_europarl():
    from nltk.corpus import europarl_raw as eu
    items, raw = [], []
    for lang in ("english", "french", "german", "spanish", "italian"):
        words = [w.lower() for w in getattr(eu, lang).words()[:12000] if w.isalpha()]
        for k in range(0, len(words) - 120, 120):
            chunk = words[k:k + 120]
            items.append((_content(chunk), lang))
            raw.append(" ".join(chunk))
    return items, " ".join(raw), "Europarl -- five languages; classify the language, generate in it"


def _code_content(tokens):
    """The stopword lesson, transferred to code: pure-punctuation tokens are shared
    by EVERY module and dilute the bag exactly like prose stopwords dilute topics.
    Dropping them lifted held-out subsystem classification from 42% to 70% --
    one move, the same principle. (Generation keeps punctuation: code without
    parentheses is not code.)"""
    return [t for t in tokens if any(c.isalnum() or c == "_" for c in t)]


def load_self():
    """INCEPTION, and the only dataset that needs no network: the system learns its
    own source code. Snippets (docstrings/comments stripped, so it is pure code) are
    labeled by subsystem; the held-out question is 'which subsystem is this code
    from?', and generation continues code in the project's own style."""
    import io, tokenize
    mods = [("holographic_image.py", "code:image"), ("holographic_creature.py", "code:creature"),
            ("holographic_tree.py", "code:tree"), ("holographic_text.py", "code:text"),
            ("holographic_ai.py", "code:engine")]
    items, raw = [], []
    here = os.path.dirname(os.path.abspath(__file__))
    for fname, lab in mods:
        lines, line = [], []
        src = open(os.path.join(here, fname), encoding="utf-8").read()
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if tok.type in (tokenize.NEWLINE, tokenize.NL):
                if line:
                    lines.append(" ".join(line)); line = []
            elif tok.type not in (tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
                line.append(tok.string)
        if line:
            lines.append(" ".join(line))
        # classification snippets: content tokens only (see _code_content)
        toks = _code_content(" ".join(lines).split())
        for k in range(0, len(toks) - 24, 24):
            items.append((" ".join(toks[k:k + 24]), lab, "code"))
        raw.append("\n".join(lines))               # generation keeps punctuation
    return items, "\n".join(raw), ("This project's own source -- classify which subsystem a "
                                   "snippet is from; generate code in its style (no download needed)")


WORLD = {
    "france":  dict(capital="paris", currency="franc", language="french", continent="europe"),
    "belgium": dict(capital="brussels", currency="franc", language="french", continent="europe"),
    "sweden":  dict(capital="stockholm", currency="krona", language="swedish", continent="europe"),
    "japan":   dict(capital="tokyo", currency="yen", language="japanese", continent="asia"),
    "mexico":  dict(capital="mexico_city", currency="peso", language="spanish", continent="america"),
    "usa":     dict(capital="washington", currency="dollar", language="english", continent="america"),
    "peru":    dict(capital="lima", currency="sol", language="spanish", continent="america"),
    "egypt":   dict(capital="cairo", currency="pound", language="arabic", continent="africa"),
    "kenya":   dict(capital="nairobi", currency="shilling", language="swahili", continent="africa"),
    "vietnam": dict(capital="hanoi", currency="dong", language="vietnamese", continent="asia"),
}


def load_world():
    """Record-shaped knowledge, no network: ten countries, each absorbed from
    EIGHT noisy observations (one attribute dropped at random per copy) -- the
    measured result this leans on is that role decode survives prototype
    averaging at 100%. This is the dataset the RELATIONS panel runs on: explain
    why two countries are similar, find by attribute, chain multi-hop asks --
    all over the mind's own absorbed memory."""
    rng = np.random.default_rng(7)
    items, raw = [], []
    for name, attrs in WORLD.items():
        for _ in range(8):
            drop = rng.choice(list(attrs))
            items.append(({k: v for k, v in attrs.items() if k != drop}, name, "record"))
        # VARIED templates, not one fixed rendering: the first version repeated
        # ten identical sentences x40, and the chunk schema memorized whole
        # sentences as single chunks -- any seed that wasn't an exact chunk
        # encoded to nothing and generation ignored it (user-visible as the
        # seed being echoed then restarted). Varied phrasing gives the schema
        # genuine sub-sentence structure to chunk, so seeds condition properly.
        cap = lambda s: s.replace("_", " ").title()
        n, attrs_c = cap(name), {k: cap(v) for k, v in attrs.items()}
        raw += [f"The capital of {n} is {attrs_c['capital']}. ",
                f"{n} uses the {attrs['currency']} and speaks {attrs_c['language']}. ",
                f"{n} is in {attrs_c['continent']}. ",
                f"In {n} they speak {attrs_c['language']}, and the capital is "
                f"{attrs_c['capital']}. ",
                f"The currency of {n} is the {attrs['currency']}. ",
                f"{attrs_c['capital']} is the capital of {n}, a country in "
                f"{attrs_c['continent']}. "]
    # EIGHT INDEPENDENTLY SHUFFLED PASSES, not one pass tiled: tiling the same
    # block makes the WHOLE BLOCK the optimal compression unit, so the chunk
    # schema legitimately learns a corpus-sized mega-chunk and generation
    # replays the corpus wholesale, seed ignored (caught by tracing the PPM:
    # the top candidate after any context was a several-hundred-atom chunk).
    # The schema was right; the diet was degenerate.
    passes = []
    for p in range(8):
        block = list(raw)
        np.random.default_rng(p).shuffle(block)
        passes.append(" ".join(block))
    return items, " ".join(passes), ("ten countries as role-bound records, each "
                                     "learned from eight incomplete observations")


DATASETS = {
    "world": ("Countries (records)", [], load_world),
    "curriculum": ("Dictionary + encyclopedia (curriculum)", [], None),
    "self": ("This project's own source", [], load_self),
    "reuters": ("Reuters categories", ["reuters"], load_reuters),
    "brown": ("Brown genres", ["brown"], load_brown),
    "gutenberg": ("Books (Gutenberg authors)", ["gutenberg"], load_gutenberg),
    "europarl": ("Europarl languages", ["europarl_raw"], load_europarl),
}


# A small, self-contained curriculum: a dictionary (word meaning) and an
# encyclopedia (is_a relations) over two clean domains (animals, minerals). Hand-
# built so it needs no network and the structure is legible in the UI.
CURRICULUM_DEFS = {
    "cat": ["animal", "feline", "pet"], "dog": ["animal", "canine", "pet"],
    "lion": ["animal", "feline", "wild", "predator"], "wolf": ["animal", "canine", "wild", "predator"],
    "fox": ["animal", "canine", "wild"], "tiger": ["animal", "feline", "wild", "predator"],
    "animal": ["living", "creature", "mobile"], "feline": ["cat", "lion", "tiger", "animal"],
    "canine": ["dog", "wolf", "fox", "animal"], "pet": ["animal", "tame"],
    "wild": ["untamed", "animal"], "predator": ["animal", "hunts"], "living": ["alive"],
    "creature": ["living", "animal"], "mobile": ["moving"], "tame": ["gentle"],
    "untamed": ["wild"], "alive": ["living"], "gentle": ["mild"], "mild": ["gentle"],
    "hunts": ["predator"], "moving": ["mobile"],
    "rock": ["mineral", "hard", "solid"], "stone": ["mineral", "hard", "solid"],
    "granite": ["rock", "hard", "igneous"], "marble": ["rock", "hard", "metamorphic"],
    "iron": ["metal", "mineral", "hard"], "copper": ["metal", "mineral", "shiny"],
    "mineral": ["solid", "inert", "natural"], "metal": ["mineral", "shiny", "conductive"],
    "hard": ["solid"], "solid": ["firm"], "shiny": ["bright"], "inert": ["still"],
    "natural": ["formed"], "conductive": ["conducts"], "firm": ["solid"],
    "bright": ["shiny"], "still": ["inert"], "igneous": ["rock"], "metamorphic": ["rock"],
    "formed": ["natural"], "conducts": ["conductive"],
}

CURRICULUM_FACTS = {
    "dog": {"is_a": "canine"}, "wolf": {"is_a": "canine"}, "fox": {"is_a": "canine"},
    "cat": {"is_a": "feline"}, "lion": {"is_a": "feline"}, "tiger": {"is_a": "feline"},
    "canine": {"is_a": "carnivore"}, "feline": {"is_a": "carnivore"},
    "carnivore": {"is_a": "mammal"}, "mammal": {"is_a": "animal"}, "animal": {"is_a": "organism"},
    "granite": {"is_a": "rock"}, "marble": {"is_a": "rock"}, "rock": {"is_a": "mineral"},
    "iron": {"is_a": "metal"}, "copper": {"is_a": "metal"}, "metal": {"is_a": "mineral"},
    "mineral": {"is_a": "matter"}, "matter": {"is_a": "substance"},
}


# ---------------------------------------------------------------------------
# training one UnifiedMind on a pulled corpus
# ---------------------------------------------------------------------------

def _absorb_into(mind, dataset_id):
    """Train one dataset INTO an existing mind (the stack's working brain), updating
    STATE cumulatively: append to trained_on, merge labels, refresh the prose used by
    the generation/codec experiments. Returns the build-result for this dataset."""
    if dataset_id == "curriculum":
        return _absorb_curriculum_into(mind)
    name, pkgs, loader = DATASETS[dataset_id]
    for p in pkgs:
        _ensure(p)
    items, raw, desc = loader()
    items = [(i if len(i) == 3 else (i[0], i[1], "text")) for i in items]

    # split each label 70/30 for an honest held-out accuracy number
    by = defaultdict(list)
    for x, lab, mod in items:
        by[lab].append((x, mod))
    rng = np.random.default_rng(0)
    train, test = [], []
    for lab, docs in by.items():
        docs = list(docs); rng.shuffle(docs)
        cut = int(len(docs) * 0.7)
        train += [(x, lab, mod) for x, mod in docs[:cut]]
        test += [(x, lab, mod) for x, mod in docs[cut:]]
    rng.shuffle(train)

    # absorb() adds to the one memory (it does not reset it), so calling it again
    # layers this dataset on top of whatever the mind already learned.
    mind.absorb(train)
    seq_mod = "code" if any(m == "code" for _, _, m in train) else "text"
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        seq_data = [(t[:40000], s) for t, s in raw]
    else:
        seq_data = raw[:160000]
    mind.learn_sequence(seq_data, n=6, modality=seq_mod)

    acc = sum(mind.classify(x)[0] == lab for x, lab, _ in test) / max(1, len(test))
    test = [(x, lab) for x, lab, _ in test]
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        seq_raw = " ".join(t for t, _ in raw)
    else:
        seq_raw = raw if isinstance(raw, str) else ""

    # accumulate STATE across the stack
    prev_labels = set(STATE.get("labels", []))
    all_labels = sorted(prev_labels | set(by))
    trained_on = list(STATE.get("trained_on", [])) + [name]
    trained_ids = list(STATE.get("trained_ids", [])) + [dataset_id]
    # the freshly added prose is what the prose experiments should read now
    STATE.update({"mind": mind, "dataset": " + ".join(trained_on), "labels": all_labels,
                  "is_code": seq_mod == "code", "test": test, "raw_len": len(raw),
                  "desc": desc, "seq_raw": seq_raw[:400000],
                  "trained_on": trained_on, "trained_ids": trained_ids})
    # any predictor/codec built on an earlier stack is now stale -- drop so it rebuilds
    for attr in ("_meaning_pred", "_verifier", "_codec"):
        if hasattr(mind, attr):
            delattr(mind, attr)
    return {
        "ok": True, "dataset": " + ".join(trained_on), "desc": desc,
        "labels": all_labels,
        "counts": mind.memory.live.counts_by_label(),
        "prototypes": mind.memory.live.size(),
        "trained": len(train), "held_out": len(test),
        "accuracy": round(100 * acc),
        "gen_chars": min(len(raw), 160000),
        "added": name,
    }


def _absorb_curriculum_into(mind):
    """Train the curriculum (dictionary then encyclopedia) INTO an existing mind, as
    a base structure other datasets can be layered on. Accumulates STATE."""
    mind.learn_dictionary(CURRICULUM_DEFS, iters=3)        # layer 1: meaning
    mind.learn_encyclopedia(CURRICULUM_FACTS)              # layer 3: relations

    # honest layer measurements
    onehop_ok = sum(1 for c, rel in CURRICULUM_FACTS.items()
                    if mind.read_role(c, "is_a")[0] == rel["is_a"])
    onehop = round(100 * onehop_ok / len(CURRICULUM_FACTS))
    # dictionary domain coherence: nearest meaning-neighbour of a few probes
    probes = {}
    for w in ("cat", "wolf", "rock", "iron"):
        probes[w] = [(n, round(s, 2)) for n, s in mind.define(w, 3)]

    name = "Dictionary + encyclopedia (curriculum)"
    prev_labels = set(STATE.get("labels", []))
    all_labels = sorted(prev_labels | set(CURRICULUM_FACTS))
    trained_on = list(STATE.get("trained_on", [])) + [name]
    trained_ids = list(STATE.get("trained_ids", [])) + ["curriculum"]
    STATE.update({"mind": mind, "dataset": " + ".join(trained_on),
                  "labels": all_labels, "is_code": False,
                  "test": STATE.get("test", []), "raw_len": STATE.get("raw_len", 0),
                  "desc": "a dictionary (word meaning) then an encyclopedia (is_a relations) "
                          "learned natively into the mind, as a base structure",
                  "trained_on": trained_on, "trained_ids": trained_ids})
    for attr in ("_meaning_pred", "_verifier", "_codec"):
        if hasattr(mind, attr):
            delattr(mind, attr)
    return {
        "ok": True, "dataset": " + ".join(trained_on),
        "desc": "a dictionary (word meaning) and an encyclopedia (is_a relations) "
                "learned into one mind -- meaning from definitions, knowledge from relations",
        "labels": all_labels,
        "counts": mind.memory.live.counts_by_label(),
        "prototypes": mind.memory.live.size(),
        "trained": len(CURRICULUM_DEFS), "held_out": 0,
        "accuracy": onehop,                               # encyclopedia one-hop is_a accuracy
        "gen_chars": 0, "added": name,
        "curriculum": {"defs": len(CURRICULUM_DEFS), "facts": len(CURRICULUM_FACTS),
                       "onehop_is_a": onehop, "probes": probes},
    }


# ---------------------------------------------------------------------------
# routes -- one per operation
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/unified/datasets")
def datasets():
    import importlib
    out = []
    have_nltk = importlib.util.find_spec("nltk") is not None
    for did, (name, pkgs, _) in DATASETS.items():
        out.append({"id": did, "name": name, "available": (have_nltk or not pkgs)})
    return jsonify({"datasets": out, "nltk": have_nltk})


@app.route("/api/unified/load", methods=["POST"])
def load():
    did = request.json.get("id")
    fresh = bool(request.json.get("fresh", False))
    mode = request.json.get("mode", "replace")     # "replace" = fresh base; "add" = on top
    if did not in DATASETS:
        return jsonify({"ok": False, "error": "unknown dataset"})
    try:
        if mode == "add" and STATE.get("trained_ids"):
            ids = tuple(STATE["trained_ids"]) + (did,)
        else:
            ids = (did,)
        return jsonify(build_stack(ids, fresh=fresh))
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e} "
                        "(if this is a missing corpus, a network connection is needed "
                        "the first time to pull it from GitHub)"})


@app.route("/api/unified/trained", methods=["GET"])
def trained_status():
    """What the CURRENT brain has been trained on (the ordered stack), plus which
    stacks are cached so they can be rebuilt instantly. Lets you judge whether a
    fresh brain is needed before adding more."""
    cached = []
    for ids, (snap, _r) in TRAINED.items():
        cached.append(" + ".join(snap.get("trained_on", [])) or snap.get("dataset"))
    return jsonify({"active": STATE.get("dataset"),
                    "trained_on": list(STATE.get("trained_on", [])),
                    "prototypes": (STATE["mind"].memory.live.size() if STATE.get("mind") else 0),
                    "labels": list(STATE.get("labels", [])),
                    "cached_stacks": sorted(set(c for c in cached if c))})


def _need_mind():
    return STATE["mind"] is None


@app.route("/api/unified/curriculum", methods=["POST"])
def curriculum_query():
    """Two queries over the curriculum-trained brain: define(word) returns the
    nearest words by dictionary-learned MEANING, and climb(concept) walks the
    is_a chain up the absorbed encyclopedia with path-traced throughput. Both run
    over the SAME mind, showing meaning and relational knowledge side by side."""
    if _need_mind():
        return jsonify({"error": "load the Dictionary + encyclopedia (curriculum) dataset first"})
    mind = STATE["mind"]
    word = (request.json.get("word") or "").strip().lower()
    if not word:
        return jsonify({"error": "enter a word"})
    out = {"word": word}
    # layer 1: meaning neighbours from the dictionary
    near = mind.define(word, 6) if hasattr(mind, "define") else []
    out["meaning"] = [{"word": w, "sim": round(s, 3)} for w, s in near]
    # layer 3: is_a chain from the encyclopedia, with per-hop confidence
    if hasattr(mind, "climb"):
        chain, tp = mind.climb(word, min_throughput=0.0)
        out["is_a_chain"] = chain
        out["throughput"] = round(float(tp), 3)
    else:
        out["is_a_chain"] = [word]
        out["throughput"] = 0.0
    if not out["meaning"] and len(out["is_a_chain"]) <= 1:
        out["note"] = ("this word is not in the curriculum -- try one of: "
                       "dog, wolf, cat, lion, rock, iron, granite, copper")
    return jsonify(out)


@app.route("/api/unified/answer", methods=["POST"])
def answer():
    """The QUESTION ROUTER over the live mind: recognizes a question's SHAPE and
    dispatches to the brain's real operation (define / is_a / role / classify /
    recall), falling back to labelled text completion. Not a chatbot -- an honest
    front door that answers from knowledge when it can and says when it is only
    continuing text."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    q = (request.json.get("question") or "").strip()
    if not q:
        return jsonify({"error": "ask a question"})
    return jsonify(STATE["mind"].answer(q))


@app.route("/api/unified/resolution", methods=["POST"])
def resolution():
    """How much holographic RESOLUTION does classifying this input need? Returns
    the per-truncation winner ladder and the dimension at which the answer
    stabilises -- a low value means the match is robust to heavy truncation
    (cheap to find coarse-to-fine), full width means it was a close call."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    text = (request.json.get("text") or "").strip()
    if not text:
        return jsonify({"error": "type some text"})
    r = STATE["mind"].resolution_profile(text)
    if not r["profile"]:
        return jsonify({"error": "no prototypes to compare against"})
    return jsonify(r)


@app.route("/api/unified/classify", methods=["POST"])
def classify():
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    text = (request.json.get("text") or "").lower()
    # code keeps every token -- stopword stripping would gut it of its keywords
    # (for, in, if, return are all English "stopwords"); prose keeps the content
    # words it was trained on. Classification is UNTAGGED: the mind discovers
    # the query's modality itself (type inference, then the content gate).
    if STATE.get("is_code"):
        text = request.json.get("text") or ""        # code keeps its case (HoloTree != holotree)
        toks = _code_content(text.split())           # ...and drops punctuation, like training did
    else:
        toks = _content(text.split())
    if not toks:
        return jsonify({"error": "type some words the model might know"})
    mind = STATE["mind"]
    label, score = mind.classify(toks)
    (rlabel, _), rscore = mind.recall(toks)
    # MULTI-RAY: also classify by firing several resampled views and combining
    # them z-scored -- robust to a noisy single encoding (path tracing's many
    # rays per pixel). Agreement is the fraction of rays that backed the winner.
    rob_label, agreement = mind.classify_robust(" ".join(toks) if isinstance(toks, list)
                                                else toks)
    return jsonify({"label": label, "score": round(float(score), 3),
                    "recall": {"label": rlabel, "score": round(float(rscore), 3)},
                    "robust": {"label": rob_label, "agreement": agreement}})


@app.route("/api/unified/organize", methods=["POST"])
def organize():
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    mind = STATE["mind"]
    before = mind.memory.live.counts_by_label()
    choice = mind.maintain_now()
    after = mind.memory.live.counts_by_label()
    # the mind keeps its own journal of maintenance events (self-narrating
    # reorganization: splits NAMED by the contrast-judged role decode where the
    # data is record-shaped) -- surface its latest account verbatim
    story = mind.journal[-1]["story"] if mind.journal else ""
    return jsonify({"before": before, "after": after,
                    "choice": (choice[0] if choice else "keep"),
                    "story": story,
                    "journal": [e["story"] for e in mind.journal[-5:]],
                    "note": "each label may hold several sub-prototypes when the memory "
                            "found it multi-modal; one each means it stayed simple."})


@app.route("/api/unified/relations", methods=["POST"])
def relations():
    """The relations operations over the mind's OWN memory: explain two learned
    labels per-role, find which record holds an attribute, or chain a multi-hop
    ask -- every readout cleaned up to a symbol (the measured law)."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    mind = STATE["mind"]
    if not mind._fillers:
        return jsonify({"error": "this dataset has no record structure -- load "
                                 "'Countries (records)' for the relations panel"})
    p = request.get_json(force=True)
    op = p.get("op")
    try:
        if op == "explain":
            rows = mind.explain(p["a"].strip(), p["b"].strip())
            return jsonify({"explain": [
                {"role": r, "a": va, "b": vb, "shared": bool(sh), "conf": round(c, 2)}
                for r, va, vb, sh, c in rows]})
        if op == "find":
            lab, score = mind.find(p["role"].strip(), p["value"].strip())
            reads = {r: mind.read_role(lab, r)[0] for r in sorted(mind._fillers)}
            return jsonify({"find": {"label": lab, "score": round(float(score), 3),
                                     "record": reads}})
        if op == "ask":
            hops = [tuple(h) for h in p["hops"]]
            # traced chain: the answer plus its THROUGHPUT (the ray's accumulated
            # confidence as it bounced through memory) and the per-hop confidences
            ans, tp, confs = mind.ask_traced(p["start"].strip(), *hops)
            return jsonify({"ask": {"answer": ans, "throughput": round(float(tp), 3),
                                    "hops": confs}})
        return jsonify({"error": f"unknown op {op!r}"})
    except KeyError as e:
        return jsonify({"error": f"unknown label or role: {e}"})


@app.route("/api/unified/generate", methods=["POST"])
def generate():
    if _need_mind() or STATE["mind"]._gen is None:
        return jsonify({"error": "load a dataset first"})
    j = request.json
    # the seed keeps its CASE: the corpus is learned raw (caps + punctuation),
    # so a lowercased seed matches no learned context and generation restarts
    # with its own capitalized opener -- the doubled-echo bug, user-caught
    seed = j.get("seed") or "The "
    length = int(j.get("length", 200))
    temp = float(j.get("temperature", 0.45))
    top_p = max(0.1, min(float(j.get("top_p", 1.0)), 1.0))   # <1.0 = nucleus (more coherent)
    text = STATE["mind"].generate(seed, max(20, min(length, 600)),
                                  max(0.1, min(temp, 1.2)), top_p=top_p)
    return jsonify({"text": text})


@app.route("/api/unified/codec", methods=["POST"])
def codec_endpoint():
    """Go both directions: losslessly compress the loaded prose to a rank-code and
    decompress it back to the exact original, reporting the achievable size and the
    honest controls (random barely shrinks; structure shrinks a lot). Also attribute
    a passage to its source if the dataset offers two."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    if STATE.get("is_code"):
        return jsonify({"error": "the codec demo expects prose \u2014 load Reuters, Brown, or Books"})
    mind = STATE["mind"]
    raw = STATE.get("seq_raw")
    if not raw:
        return jsonify({"error": "no corpus available"})
    from holographic_text import _tokens
    if not hasattr(mind, "_meaning_pred"):
        sents = [_tokens(s) for s in raw.split(".") if s.strip()][:1500]
        mind.build_meaning_predictor([s for s in sents if s], order=2)
    toks = _tokens(raw)[:600]
    in_vocab = [t for t in toks if t in mind._meaning_pred.idx]
    r = mind.compress_lossless(in_vocab[:300])
    recon = mind.decompress_lossless(r["code"])
    import numpy as _np
    vocab = mind._meaning_pred.vocab
    rng = _np.random.default_rng(0)
    randtoks = [vocab[rng.integers(len(vocab))] for _ in range(300)]
    rand_ratio = mind.compress_lossless(randtoks)["cost"]["ratio"]
    return jsonify({
        "lossless": bool(recon == in_vocab[:300]),
        "ratio": round(r["cost"]["ratio"], 2),
        "bits_per_token": round(r["cost"]["bits_per_token"], 2),
        "baseline": round(r["cost"]["baseline"], 2),
        "random_ratio": round(rand_ratio, 2),
        "n": r["cost"]["n"],
        "note": ("encoder and decoder share the predictor, so the rank stream decompresses "
                 "to the exact original -- lossless. The size is bounded by structure: real "
                 "text shrinks well, random data barely (no free lunch), and a perfectly "
                 "predictable stream would collapse to almost the seed alone")})


@app.route("/api/unified/population", methods=["POST"])
def population_endpoint():
    """Many NPCs, one mind: train a small shared base, branch several lightweight
    NPCs that each learn a PRIVATE fact, and show inheritance, isolation,
    propagation, and the memory saving versus one full brain per NPC."""
    from holographic_unified import UnifiedMind
    base = UnifiedMind(dim=512, seed=0)
    world = [("sword", "weapon"), ("axe", "weapon"), ("apple", "food"),
             ("bread", "food"), ("gold", "treasure"), ("gem", "treasure"),
             ("river", "place"), ("cave", "place")]
    for x, lab in world:
        base.learn(x, lab)
    shared = base.share()
    # three NPCs, each learns something only they know
    npc_specs = [("Alaric the alchemist", "potion", "alchemy"),
                 ("Bryn the smith", "anvil", "smithing"),
                 ("Cora the scout", "map", "navigation")]
    npcs = [shared.branch(name).learn(word, fact) for name, word, fact in npc_specs]
    n_pop = int(request.json.get("population", 50))
    n_pop = max(2, min(500, n_pop))

    def cost_for(n):
        # n NPCs each with one private prototype
        return shared.population_cost([shared.branch(f"x{i}").learn(f"w{i}", f"f{i}")
                                       for i in range(n)])
    cost = cost_for(n_pop)
    rows = []
    # inheritance: each NPC classifies a shared item it never learned
    for (name, word, fact), npc in zip(npc_specs, npcs):
        rows.append({"npc": name, "inherits_shared": f"sword -> {npc.classify('sword')}",
                     "private": f"{word} -> {npc.classify(word)}",
                     "knows_privately": npc.knows_privately()})
    # isolation: NPC[1] does not see NPC[0]'s private fact
    isolation = {"npc": npcs[1].name, "on_alarics_word": npcs[0].knows_privately()[0]
                 if npcs[0].knows_privately() else "",
                 "result": npcs[1].classify("potion")}
    # propagation: Alaric shares his learning; now everyone sees it
    before = npcs[1].classify("potion")
    npcs[0].propagate()
    after = npcs[1].classify("potion")
    return jsonify({
        "base_labels": sorted(base.memory.live.labels()),
        "npcs": rows, "isolation": isolation,
        "propagation": {"word": "potion", "before": before, "after": after},
        "cost": cost, "population": n_pop,
        "note": (f"{n_pop} NPCs sharing one frozen base cost {cost['shared_total']} prototypes "
                 f"vs {cost['separate_total']} for a full brain each -- a {cost['saving_x']:.0f}x saving "
                 f"that grows with the base. Each NPC inherits all shared knowledge, keeps its own "
                 f"private learning isolated, and can propagate that learning back to everyone")})


@app.route("/api/unified/bigtext", methods=["POST"])
def bigtext_endpoint():
    """Experiment (run on demand): run the heavy text instruments on a LARGE slice of
    the loaded corpus -- structure score (real vs shuffled vs random), lossless codec
    ratio, and self-discovered word-boundary F1 -- and report the numbers. This is the
    kind of run kept out of the test suite; trigger it here and read the results."""
    if _need_mind():
        return jsonify({"error": "load a text dataset first"})
    if STATE.get("is_code"):
        return jsonify({"error": "the big-text run expects prose \u2014 load Reuters, Brown, or Books"})
    raw = STATE.get("seq_raw")
    if not raw:
        return jsonify({"error": "no corpus available"})
    mind = STATE["mind"]
    n_tokens = int(request.json.get("tokens", 3000))
    n_tokens = max(500, min(20000, n_tokens))
    import numpy as _np
    from holographic_text import _tokens
    try:
        toks = _tokens(raw)
        if len(toks) < 500:
            return jsonify({"error": "corpus too small for a big-text run"})
        out = {"corpus_tokens": len(toks), "used_tokens": min(n_tokens, len(toks))}
        # 1) structure score: real vs shuffled vs random (if the verifier is built)
        if not hasattr(mind, "_verifier"):
            sents = [_tokens(s) for s in raw.split(".") if s.strip()][:2000]
            mind.build_meaning_predictor([s for s in sents if s], order=2)
        if hasattr(mind, "verify_structure"):
            real = toks[:n_tokens]
            sh = real[:]; _np.random.default_rng(0).shuffle(sh)
            vocab = mind._meaning_pred.vocab
            rnd = [vocab[i] for i in _np.random.default_rng(1).integers(0, len(vocab), len(real))]
            out["structure"] = {
                "real": round(float(mind.verify_structure(real)["score"]), 2),
                "shuffled": round(float(mind.verify_structure(sh)["score"]), 2),
                "random": round(float(mind.verify_structure(rnd)["score"]), 2)}
        # 2) lossless codec on a big slice
        if hasattr(mind, "compress_lossless"):
            in_vocab = [t for t in toks[:n_tokens] if t in mind._meaning_pred.idx]
            r = mind.compress_lossless(in_vocab[:min(2000, len(in_vocab))])
            out["codec"] = {"lossless": bool(r.get("lossless")),
                            "ratio": round(float(r["cost"]["ratio"]), 3),
                            "bits_per_token": round(float(r["cost"]["bits_per_token"]), 2),
                            "tokens": r["cost"]["n"]}
        # 3) self-discovered boundaries on a big char slice
        if hasattr(mind, "discover_units"):
            text = "".join(c for c in raw.lower() if c.isalpha() or c == " ")[:n_tokens * 6]
            chars, truth = [], set()
            for ch in text:
                if ch == " ":
                    if chars:
                        truth.add(len(chars) - 1)
                else:
                    chars.append(ch)
            stream = "".join(chars)
            from holographic_segment import Segmenter, boundary_f1
            seg = Segmenter(dim=512, order=4, seed=0).fit(stream)
            bounds = seg.boundaries(stream)
            rng = _np.random.default_rng(0)
            rand = set(rng.choice(len(stream), len(bounds), replace=False).tolist())
            out["segmentation"] = {"chars": len(stream),
                                   "f1": round(boundary_f1(bounds, truth)["f1"], 2),
                                   "random_f1": round(boundary_f1(rand, truth)["f1"], 2)}
        out["note"] = ("a heavy run on a large slice: structure score separates real text from "
                       "shuffled and random; the codec round-trips losslessly at the shown ratio; "
                       "and word boundaries are self-discovered well above a random cut. Bump the "
                       "token count for a bigger experiment")
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})


@app.route("/api/unified/market", methods=["POST"])
def market_endpoint():
    """Experiment (run on demand): does real market data carry the structure the
    engine's instrument looks for? Raw returns are efficient-market-like (no symmetric
    band), but VOLATILITY CLUSTERING -- |returns| being positively autocorrelated --
    is the signature that distinguishes real returns from a shuffle. Reports the lag-1
    |return| autocorrelation and how many sigma it beats shuffled controls, on the
    larger datasets, with the shuffle control shown for honesty."""
    import os
    import json as _json
    import numpy as _np
    from holographic_signal_structure import volatility_clustering, clustering_zscore
    which = (request.json.get("dataset") or "sol").lower()
    try:
        if which.startswith("dai") or which.startswith("weth"):
            path = "data/dai_weth_big.json"
            if not os.path.exists(path):
                return jsonify({"error": "data/dai_weth_big.json not present"})
            a = _np.array(_json.load(open(path))["ohlcv"], float)
            close = a[:, 4] if a.shape[1] >= 5 else a[:, -2]
            series = _np.diff(close) / close[:-1]
            label = f"DAI/WETH candles ({len(series)} close-to-close returns)"
        else:
            from holographic_market import load_ticks, move_series
            if not os.path.exists("data/sol_5min.npz"):
                return jsonify({"error": "data/sol_5min.npz not present"})
            ts, px = load_ticks()
            series, _burst = move_series(ts, px)
            label = f"SOL ticks ({len(series)} within-burst moves)"
        acf1 = volatility_clustering(series)
        z = clustering_zscore(series, n_shuffle=100)
        rng = _np.random.default_rng(0)
        sh = _np.asarray(series, float).copy(); rng.shuffle(sh)
        return jsonify({
            "dataset": label, "n": int(len(series)),
            "vol_clustering_acf1": round(float(acf1), 3),
            "zscore": round(float(z), 2),
            "shuffled_acf1": round(float(volatility_clustering(sh)), 3),
            "structured": bool(z > 2.0),
            "note": ("real returns show volatility clustering -- bursts of activity cluster in "
                     "time, so |returns| is positively autocorrelated. The z-score is how many "
                     "sigma that beats a shuffle of the same returns; >2 is meaningful structure, "
                     "and the shuffle control collapses to ~0. Raw signed returns carry almost no "
                     "such structure (efficient-market-like) -- the clustering is the honest signal")})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})


@app.route("/api/unified/sprites", methods=["POST"])
def sprites_endpoint():
    """Cross-sprite compression on the real 712-sprite set: a sprite sheet is one
    structured body, so packing the whole set against shared references beats
    compressing each sprite on its own. Reports the measured sizes per method."""
    import os
    folder = "features/sprites"
    if not os.path.isdir(folder):
        return jsonify({"error": "sprite set not found (features/sprites)"})
    try:
        import io
        from PIL import Image
        import pack_sprites
        import numpy as _np
        items = pack_sprites.load_folder(folder)
        if not items:
            return jsonify({"error": "no sprites found"})
        total = len(items)
        n = int(request.json.get("n", 200))
        n = max(20, min(total, n))
        items = items[:n]
        blob = pack_sprites.pack(items)               # set-pack: shared references, bit-exact
        # honest baseline: each sprite compressed on its own as an optimized PNG
        def png_size(a):
            b = io.BytesIO(); Image.fromarray(a).save(b, "PNG", optimize=True); return len(b.getvalue())
        per_file = sum(png_size(a) for _nm, a in items)
        # verify the pack is lossless (the whole point: structure, not loss)
        exact = all(_np.array_equal(a, b) for (_, a), (_, b) in zip(items, pack_sprites.unpack(blob)))
        saving = round(per_file / len(blob), 2) if blob else None
        return jsonify({"total_sprites": total, "used": n,
                        "set_pack": len(blob), "per_file_png": per_file,
                        "saving_x": saving, "lossless": bool(exact),
                        "note": (f"{n} of the {total}-sprite set: packing them as one body against "
                                 f"shared references is {saving}x smaller than compressing each "
                                 f"sprite alone ({len(blob):,} vs {per_file:,} bytes), and it is "
                                 f"bit-exact -- the cross-sprite structure that separate files hide")})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})


@app.route("/api/unified/vault", methods=["POST"])
def vault_endpoint():
    """The image repository as one perceptual memory: ingest the photo set, find
    near-duplicates by a size/format-invariant fingerprint, cluster them, and answer
    query-by-example -- the same space the rest of the engine uses, applied to
    pictures. Falls back to the sprite set if no photo sample is present."""
    import os
    from image_vault import ImageVault
    folder = "features/photo_sample" if os.path.isdir("features/photo_sample") else "features/sprites"
    if not os.path.isdir(folder):
        return jsonify({"error": "no image set found"})
    try:
        v = ImageVault().add_folder(folder)
        if len(v) == 0:
            # photo_sample holds .npy arrays; load those directly
            import numpy as _np
            import glob as _glob
            for p in sorted(_glob.glob(os.path.join(folder, "*.npy"))):
                v.add(_np.load(p), name=os.path.basename(p))
        if len(v) == 0:
            return jsonify({"error": "image set is empty"})
        clusters = v.clusters(threshold=0.9)
        report = v.report(threshold=0.9)
        rows = sorted([{"method": m, "bytes": int(b),
                        "psnr": (None if p == float("inf") else round(p, 1))}
                       for m, b, p in report], key=lambda r: r["bytes"])[:6]
        # query-by-example: the first image should match itself best
        sim = v.most_similar(v.images[0], k=min(3, len(v)))
        return jsonify({"count": len(v), "folder": folder,
                        "n_clusters": len(clusters),
                        "biggest_cluster": max((len(c) for c in clusters), default=1),
                        "rows": rows,
                        "query": [{"name": nm, "sim": round(s, 3)} for nm, s in sim],
                        "note": (f"{len(v)} images grouped into {len(clusters)} perceptual clusters "
                                 f"by a format-invariant fingerprint; query-by-example and honest "
                                 f"size/fidelity comparison run over the same vault")})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})


@app.route("/api/unified/factorize", methods=["POST"])
def factorize_endpoint():
    """Searching in superposition: bind three random factors into one composite
    vector, then let the resonator network pull them back out -- recovering the
    factorization from a combinatorial space far larger than it ever enumerates."""
    import numpy as _np
    from holographic_resonator import ResonatorNetwork, map_codebook, map_bind
    n = int(request.json.get("codebook_size", 50))
    n = max(10, min(120, n))
    dim = 1500
    books = [map_codebook(n, dim, s) for s in range(3)]
    rng = _np.random.default_rng(int(request.json.get("seed", 0)))
    true = [int(rng.integers(n)) for _ in range(3)]
    c = map_bind(*[books[f][true[f]] for f in range(3)])
    r = ResonatorNetwork(books).factor(c, restarts=25)
    return jsonify({
        "true": list(true), "recovered": list(r["factors"]), "solved": r["solved"],
        "restarts": r["restarts"], "iterations": r["iterations"],
        "search_space": r["search_space"], "codebook_size": n,
        "note": ("three random vectors were bound into one; the resonator recovered which "
                 "vector came from each codebook by searching in superposition -- it never "
                 "enumerated the " + format(r["search_space"], ",") + " possible combinations")})


@app.route("/api/unified/discover", methods=["POST"])
def discover_endpoint():
    """Self-discovery of structure: strip the spaces from the loaded prose and let
    the system rediscover the word boundaries from branching entropy alone -- no
    labels. Reports the discovered chunks, the F1 against the true (removed) word
    boundaries, and the compression payoff (discovered chunks vs single characters)."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    if STATE.get("is_code"):
        return jsonify({"error": "self-discovery expects prose \u2014 load Reuters, Brown, or Books"})
    raw = STATE.get("seq_raw")
    if not raw:
        return jsonify({"error": "no corpus available"})
    text = "".join(c for c in raw.lower() if c.isalpha() or c == " ")[:20000]
    # ground-truth boundaries (positions before a removed space) and the bare stream
    chars, truth = [], set()
    for c in text:
        if c == " ":
            if chars:
                truth.add(len(chars) - 1)
        else:
            chars.append(c)
    stream = "".join(chars)
    if len(stream) < 200:
        return jsonify({"error": "corpus too small"})
    from holographic_segment import Segmenter, boundary_f1, chunk_compression
    seg = Segmenter(dim=512, order=4, seed=0).fit(stream)
    bounds = seg.boundaries(stream)
    chunks = seg.segment(stream)
    f1 = boundary_f1(bounds, truth)
    cb, sb = chunk_compression(stream, chunks)
    import numpy as _np
    rng = _np.random.default_rng(0)
    rand = set(rng.choice(len(stream), len(bounds), replace=False).tolist())
    rf1 = boundary_f1(rand, truth)["f1"]
    return jsonify({
        "sample": [" ".join("".join(c) for c in chunks[:30])],
        "f1": round(f1["f1"], 2), "precision": round(f1["precision"], 2),
        "recall": round(f1["recall"], 2), "random_f1": round(rf1, 2),
        "chunk_bits": round(cb, 2), "symbol_bits": round(sb, 2),
        "note": ("the system never saw spaces -- it found the word boundaries from where "
                 "its next-character prediction becomes uncertain (branching entropy peaks), "
                 "and the discovered chunks compress far better than single characters")})


@app.route("/api/unified/deliberate", methods=["POST"])
def deliberate_endpoint():
    """Think before answering: the system drafts a response, judges it, and refines
    -- keeping the best and stopping early once it is good enough. Returns the
    chosen response plus the trace of drafts and the iteration count, so the inner
    deliberation (and how the thinking time adapts to the query) is visible."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    if STATE.get("is_code"):
        return jsonify({"error": "deliberation expects prose \u2014 load Reuters, Brown, or Books"})
    mind = STATE["mind"]
    query = (request.json.get("query") or "").strip().lower()
    if not query:
        return jsonify({"error": "type a query"})
    if not hasattr(mind, "_meaning_pred") or not hasattr(mind, "_verifier"):
        raw = STATE.get("seq_raw")
        if not raw:
            return jsonify({"error": "no corpus available"})
        from holographic_text import _tokens
        sents = [_tokens(s) for s in raw.split(".") if s.strip()][:1500]
        mind.build_meaning_predictor([s for s in sents if s], order=2)
    r = mind.deliberate(query, max_iters=8, target_quality=0.45)
    return jsonify({
        "query": query,
        "response": " ".join(r["response"]),
        "iterations": r["iterations"], "quality": round(r["quality"], 3),
        "relevance": round(r["relevance"], 3), "structure": round(r["structure"], 2),
        "trace": [{"draft": " ".join(t["draft"][:14]), "quality": t["quality"]} for t in r["trace"]],
        "note": ("the system drafts, judges, and refines -- keeping the best draft; the "
                 "number of iterations is the thinking time, and it adapts to how hard the "
                 "query is (easy ones settle fast, hard ones take longer)")})


@app.route("/api/unified/respond", methods=["POST"])
def respond_endpoint():
    """Query-and-generate: answer a typed query with a continuation steered toward
    what the query is about, kept coherent by the structure guard. Reports the
    response with its relevance (on-query) and structure (coherent) -- both
    measured. Trains the meaning predictor + verifier lazily from the loaded prose."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    if STATE.get("is_code"):
        return jsonify({"error": "query-and-generate expects prose \u2014 load Reuters, Brown, or Books"})
    mind = STATE["mind"]
    query = (request.json.get("query") or "").strip().lower()
    if not query:
        return jsonify({"error": "type a query"})
    qw = float(request.json.get("query_weight", 5.0))
    if not hasattr(mind, "_meaning_pred") or not hasattr(mind, "_verifier"):
        raw = STATE.get("seq_raw")
        if not raw:
            return jsonify({"error": "no corpus available"})
        from holographic_text import _tokens
        sents = [_tokens(s) for s in raw.split(".") if s.strip()][:1500]
        mind.build_meaning_predictor([s for s in sents if s], order=2)
    steered = mind.respond_report(query, length=30, query_weight=qw)
    plain = mind.respond_report(query, length=30, query_weight=0.0)
    return jsonify({
        "query": query,
        "steered": {"text": " ".join(steered["response"]),
                    "relevance": round(steered["relevance"], 3),
                    "structure": round(steered["structure"], 2)},
        "unsteered": {"text": " ".join(plain["response"]),
                      "relevance": round(plain["relevance"], 3),
                      "structure": round(plain["structure"], 2)},
        "note": ("the query pulls generation toward its meaning while the structure "
                 "guard keeps it coherent; relevance should rise over the unsteered "
                 "baseline without structure collapsing")})


@app.route("/api/unified/predictive", methods=["POST"])
def predictive():
    """The predictive loop, surfaced: the mind observes the loaded corpus one
    symbol at a time, anticipating each next token and learning from its surprise.
    Reports the learning curve (accuracy as it sees more), the surprise/free-energy
    trace (does it converge?), and a generation-by-anticipation sample. Also
    measures generalisation -- accuracy on contexts never seen exactly, where an
    exact n-gram is blind."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    if STATE.get("is_code"):
        return jsonify({"error": "the predictive loop expects prose \u2014 load Reuters, Brown, or Books"})
    mind = STATE["mind"]
    raw = STATE.get("seq_raw")
    if not raw:
        return jsonify({"error": "no corpus available for this dataset"})
    from holographic_text import _tokens
    toks = _tokens(raw)[:8000]
    if len(toks) < 200:
        return jsonify({"error": "corpus too small"})
    train, held = toks[:int(len(toks) * 0.8)], toks[int(len(toks) * 0.8):]
    # learning curve: fresh predictor at each budget, accuracy on a held probe
    probe = held[:400]
    curve = []
    for frac in (0.15, 0.4, 0.7, 1.0):
        mind.build_predictor(order=2)
        mind.observe_sequence(train[:int(len(train) * frac)])
        rep = mind.prediction_report(probe)
        curve.append({"tokens": int(len(train) * frac), "accuracy": round(rep["accuracy"], 3),
                      "entries": len(mind._predictor._ctx)})
    # full model: trace + generalisation + a generated sample
    steps = mind.observe_sequence(train)            # full pass (the live trace)
    sfe = [s.self_free_energy for s in steps]
    seen = set(tuple(train[max(0, i - 2):i]) for i in range(1, len(train)))
    unseen = [(held[max(0, i - 2):i], held[i]) for i in range(1, len(held))
              if tuple(held[max(0, i - 2):i]) not in seen]
    gen_hits = sum(mind.anticipate(ctx)[0] == act for ctx, act in unseen[:300])
    seed = train[:2]
    sample = mind.generate_predictive(seed, length=24)
    # PROOF OF STRUCTURE: build the meaning predictor + verifier, then show that a
    # locally-coherent greedy generation collapses (low structure score) while
    # steered generation -- defending coherence as a process -- stays in the band.
    proof = None
    try:
        mind.build_meaning_predictor([_tokens(s) for s in raw.split(".") if s.strip()][:1500], order=2)
        greedy = list(seed)
        for _ in range(40):
            w, _c = mind.anticipate_meaning(greedy[-2:])
            greedy.append(w if w else seed[0])
        steered = list(seed) + mind.generate_structured(seed, length=40, beam=6)
        real_v = mind.verify_structure(train[:300])
        proof = {
            "real_score": round(real_v["score"], 2), "threshold": round(real_v["threshold"], 2),
            "greedy_score": round(mind.verify_structure(greedy[2:])["score"], 2),
            "steered_score": round(mind.verify_structure(steered[2:])["score"], 2),
            "greedy_sample": " ".join(greedy[2:22]),
            "steered_sample": " ".join(steered[2:22])}
        # better structure -> better compression, on the same text vs a shuffle
        import random as _rnd
        shuffled = train[:300][:]
        _rnd.Random(0).shuffle(shuffled)
        proof["compress_real"] = round(mind.compress_cost(train[:300])["ratio"], 2)
        proof["compress_shuffled"] = round(mind.compress_cost(shuffled)["ratio"], 2)
    except Exception:
        proof = None
    return jsonify({
        "curve": curve,
        "free_energy_start": round(float(np.mean(sfe[:100])), 3),
        "free_energy_end": round(float(np.mean(sfe[-100:])), 3),
        "generalization": {"correct": int(gen_hits), "total": min(300, len(unseen)),
                           "pct": round(100 * gen_hits / max(1, min(300, len(unseen))))},
        "seed": " ".join(seed), "sample": " ".join(sample), "proof": proof,
        "note": ("accuracy rises with exposure; free energy (smoothed prediction error) "
                 "falls as the model learns to anticipate; and it scores on contexts it "
                 "never saw exactly -- generalising by resonance, where exact lookup is blind")})


@app.route("/api/unified/topic_pull", methods=["POST"])
def topic_pull():
    """The honest topic-pull experiment, surfaced live: sweep the topic_weight
    and report coherence, transition validity, and lexical diversity. The kept
    negative -- coherence that 'rises' only as diversity collapses is the metric
    being gamed by repetition, not real on-topic language; it shows why deeper
    conditioning alone does not make this brain an LLM."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    mind = STATE["mind"]
    if STATE.get("is_code"):
        return jsonify({"error": "topic pull is a prose experiment \u2014 load Reuters, Brown, or Books"})
    seed = (request.json.get("seed") or "the").strip().lower()
    # train the word generator lazily from the loaded corpus (first use only)
    if not hasattr(mind, "_wordgen"):
        raw = STATE.get("seq_raw")
        if not raw:
            return jsonify({"error": "no corpus available for this dataset"})
        from holographic_text import _tokens
        sents = [_tokens(s) for s in raw.split(".") if s.strip()][:1200]
        mind.learn_word_generator([s for s in sents if s][:1200], order=1)
    seeds = [seed, "the " + seed, seed + " and the"]
    rows = mind.topic_pull_tradeoff(seeds, weights=(0.0, 2.0, 8.0, 16.0), length=40)
    sample0 = " ".join(mind.generate_words(seed, length=30, topic_weight=0.0, seed_rng=1))
    sampleH = " ".join(mind.generate_words(seed, length=30, topic_weight=16.0, seed_rng=1))
    return jsonify({"rows": rows, "sample_baseline": sample0, "sample_hot": sampleH})


@app.route("/api/unified/attribute", methods=["POST"])
def attribute():
    """WHO taught this text? Ranks the dataset's sources by how much of the
    passage's transitions each one taught -- provenance from the same tables
    generation reads. Works on pasted text OR the last generated output."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    mind = STATE["mind"]
    text = (request.json.get("text") or "").strip()
    if not text:
        return jsonify({"error": "nothing to attribute"})
    tr = mind.trace(text)
    if tr is None:
        return jsonify({"error": "this dataset carries no source provenance -- "
                                 "load 'Project Gutenberg' (per-author sources)"})
    return jsonify({
        "verdict": tr["verdict"], "basis": tr["basis"],
        "span": tr["span"][:120], "span_source": tr["span_source"],
        "by_material": [{"source": s, "weight": round(w, 3)} for s, w in tr["by_material"]],
        "ranked": [{"source": s, "weight": round(w, 3)} for s, w in tr["by_style"]]})


@app.route("/api/unified/recall", methods=["POST"])
def recall():
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    text = (request.json.get("text") or "").lower()
    toks = _content(text.split())
    if not toks:
        return jsonify({"error": "type some words"})
    (label, example), score = STATE["mind"].recall(toks, "text")
    snippet = " ".join(example[:18]) if isinstance(example, list) else str(example)
    return jsonify({"label": label, "score": round(float(score), 3), "example": snippet})


# --- generative panels: drive the existing decoders FORWARD to produce output ----------

def _img_data_url(arr):
    """A numpy HxWx3 image (uint8 or float in [0,1]) -> a base64 PNG data URL the browser
    can render directly. Used by the generative panels to show what they made."""
    import io as _io
    from PIL import Image
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    buf = _io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/api/unified/compose", methods=["POST"])
def compose_endpoint():
    """Forward compositional generation: pick attribute tags, run the resonator FORWARD to
    build a NEW scene vector, render it, and prove it is real by factoring the vector and
    auto-tagging the pixels straight back to the spec it was built from. Goes through the
    loaded mind's OWN scene faculty (one brain), falling back to a fresh mind if none is
    loaded yet so the panel still works standalone."""
    import holographic_scene as hs
    from holographic_compose import render_scene, render_fidelity
    j = request.json or {}
    mind = STATE.get("mind") or UnifiedMind(dim=1024, seed=0)
    rng = np.random.default_rng(int(j.get("seed", 0)))
    n = max(1, min(int(j.get("objects", 2)), 4))
    # compose n NOVEL objects (random tags from the full vocabulary)
    tags = [{"colour": str(rng.choice(hs.COLOURS)), "shape": str(rng.choice(hs.SHAPES)),
             "texture": str(rng.choice(hs.TEXTURES))} for _ in range(n)]
    img = render_scene(tags, S=120, seed=int(rng.integers(1000)))
    recovered = mind.decompose_scene(mind.compose_scene(tags), n, sweeps=2)
    key = lambda d: (d["colour"], d["shape"], d["texture"])
    exact = {key(t) for t in tags} == {key(g) for g in recovered}
    # render fidelity on the first object (shape+colour the renderer actually paints)
    sh_ok, col_ok = render_fidelity(tags[0], S=120, seed=0)
    # a short colour-sweep animation of the first object, composed through the mind
    sweep = ["red", "yellow", "green", "cyan", "blue"]
    frames, anim_ok = [], 0
    for v in sweep:
        t = dict(tags[0]); t["colour"] = v
        vec = mind.compose_scene([t])
        frames.append(render_scene([t], S=120, seed=0))
        anim_ok += (mind.decompose_scene(vec, 1)[0]["colour"] == v)
    anim = anim_ok / len(sweep)
    return jsonify({
        "composed": [(t["colour"], t["shape"], t["texture"]) for t in tags],
        "recovered": [(g["colour"], g["shape"], g["texture"]) for g in recovered],
        "exact": bool(exact),
        "image": _img_data_url(img),
        "render_shape_ok": bool(sh_ok), "render_colour_ok": bool(col_ok),
        "anim_frames": [_img_data_url(im) for im in frames],
        "anim_faithful": round(float(anim), 2),
        "note": ("the resonator runs FORWARD to compose a scene that was never stored, then "
                 "BACKWARD to prove it: a generated scene is real because it analyses straight "
                 "back to the spec it was built from -- all through the loaded mind's own "
                 "scene faculty")})


@app.route("/api/unified/nested", methods=["POST"])
def nested_endpoint():
    """Fractal composition: the SAME bind+superpose that builds a scene from objects, run
    ONE LEVEL UP to build a scene-of-scenes. Compose several sub-scenes, group them, then
    peel each group back out and factor it -- same above, same below. Renders each recovered
    sub-scene and reports per-group round-trip fidelity. Goes through the loaded mind's own
    faculty (one brain)."""
    import holographic_scene as hs
    from holographic_compose import render_scene
    j = request.json or {}
    mind = STATE.get("mind") or UnifiedMind(dim=1024, seed=0)
    rng = np.random.default_rng(int(j.get("seed", 0)))
    n_groups = max(2, min(int(j.get("groups", 2)), 4))
    per = max(1, min(int(j.get("per_group", 2)), 3))

    def rt():
        return {"colour": str(rng.choice(hs.COLOURS)), "shape": str(rng.choice(hs.SHAPES)),
                "texture": str(rng.choice(hs.TEXTURES))}
    groups = {f"group{i+1}": [rt() for _ in range(per)] for i in range(n_groups)}
    sizes = {k: len(v) for k, v in groups.items()}
    recovered = mind.decompose_nested(mind.compose_nested(groups), sizes)
    key = lambda d: (d["colour"], d["shape"], d["texture"])
    out_groups = []
    exact_count = 0
    for k in groups:
        ok = {key(t) for t in groups[k]} == {key(g) for g in recovered[k]}
        exact_count += ok
        out_groups.append({
            "name": k,
            "composed": [(t["colour"], t["shape"], t["texture"]) for t in groups[k]],
            "recovered": [(g["colour"], g["shape"], g["texture"]) for g in recovered[k]],
            "exact": bool(ok),
            "image": _img_data_url(render_scene(groups[k], S=110, seed=0)),
        })
    return jsonify({
        "groups": out_groups,
        "exact_groups": exact_count, "total_groups": len(groups),
        "note": ("a sub-scene is to the super-scene exactly what an object is to a scene -- the "
                 "same bind+superpose and unbind+factor at two levels (fractal). Recovery is "
                 "near-perfect at 2-3 groups and degrades gracefully beyond as group-binding "
                 "cross-talk accumulates -- the flat scene's capacity limit, one level up")})


@app.route("/api/unified/highcap", methods=["POST"])
def highcap_endpoint():
    """High-capacity binding: cram many key->value pairs into ONE vector and read them back.
    Compares the real-valued HRR core against the mind's FHRR (complex-phasor) faculty
    (high_capacity_memory) at the same load -- FHRR holds far more pairs before readback
    breaks. Goes through the loaded mind's own faculty (one brain)."""
    import numpy as _np
    from holographic_ai import random_vector, bind, unbind, cosine
    j = request.json or {}
    n = max(4, min(int(j.get("pairs", 40)), 80))
    dim = 256                                            # low-ish dim so capacity bites visibly
    mind = STATE.get("mind") or UnifiedMind(dim=dim, seed=0)

    # real-HRR baseline: one trace of n bound pairs, recover each by nearest value
    rng = _np.random.default_rng(int(j.get("seed", 0)))
    keys = [random_vector(dim, rng) for _ in range(n)]
    vals = [random_vector(dim, rng) for _ in range(n)]
    trace = sum(bind(keys[i], vals[i]) for i in range(n))
    real_ok = sum(int(_np.argmax([cosine(unbind(trace, keys[i]), v) for v in vals])) == i
                  for i in range(n))

    # FHRR via the mind's faculty
    mem, voc = mind.high_capacity_memory()
    mem.trace = _np.zeros_like(mem.trace)                # fresh trace for the demo
    pairs = {f"k{i}": f"v{i}" for i in range(n)}
    for k, v in pairs.items():
        mem.learn(voc.get(k), voc.get(v))
    vocab = [f"v{i}" for i in range(n)]
    fhrr_ok = sum(voc.cleanup(mem.recall(voc.get(k)), candidates=vocab)[0] == v
                  for k, v in pairs.items())
    return jsonify({
        "pairs": n, "dim": dim,
        "real_hrr_recovered": real_ok, "real_hrr_frac": round(real_ok / n, 3),
        "fhrr_recovered": fhrr_ok, "fhrr_frac": round(fhrr_ok / n, 3),
        "note": ("the real-valued HRR core is the readable default and is perfect at the "
                 "few-pair loads the mind normally runs; for a LARGE key->value trace the FHRR "
                 "faculty (complex unit-phasor atoms, binding by complex multiply) holds far "
                 "more pairs -- the high-capacity option the VSA literature recommends, kept "
                 "opt-in so the common path stays real-valued and readable")})


@app.route("/api/unified/morph", methods=["POST"])
def morph_endpoint():
    """Image morph: spherical interpolation between two rendered shapes IN THE
    DCT-COEFFICIENT DOMAIN, vs a pixel crossfade. The honest difference is ghosting -- a
    crossfade midpoint IS the double-exposure; the coefficient morph blends structure.
    Goes through the loaded mind's morph faculty (one brain)."""
    import holographic_scene as hs
    from holographic_generate import crossfade_images, ghosting
    j = request.json or {}
    mind = STATE.get("mind") or UnifiedMind(dim=1024, seed=0)
    rng = np.random.default_rng(int(j.get("seed", 0)))
    # render two distinct shapes to morph between
    a_spec = (str(rng.choice(hs.SHAPES)), str(rng.choice([c for c in hs.COLOURS if c != "grey"])))
    b_spec = (str(rng.choice(hs.SHAPES)), str(rng.choice([c for c in hs.COLOURS if c != "grey"])))
    A = hs.make_scene([a_spec], S=96, seed=1).astype(float) / 255.0
    B = hs.make_scene([b_spec], S=96, seed=2).astype(float) / 255.0
    steps = 9
    morph = mind.morph_scene(A, B, steps=steps)        # the mind's own morph faculty
    cross = crossfade_images(A, B, steps=steps)
    gm = ghosting(morph[steps // 2], A, B)
    gx = ghosting(cross[steps // 2], A, B)
    return jsonify({
        "a": f"{a_spec[1]} {a_spec[0]}", "b": f"{b_spec[1]} {b_spec[0]}",
        "morph_frames": [_img_data_url(f) for f in morph],
        "crossfade_mid": _img_data_url(cross[steps // 2]),
        "morph_mid": _img_data_url(morph[steps // 2]),
        "ghost_morph": round(float(gm), 3), "ghost_crossfade": round(float(gx), 3),
        "note": ("coefficient-domain slerp blends structure; the pixel crossfade midpoint is "
                 "literally the double-exposure of the two pictures (ghosting near zero distance)")})


@app.route("/api/unified/nucleus", methods=["POST"])
def nucleus_endpoint():
    """Text generation with nucleus (top-p) decoding over the loaded mind's n-gram, vs the
    plain temperature sampling already on the Generate panel. Trimming the unlikely tail
    raises the real-word fraction (coherence) at a modest diversity cost -- reported."""
    if _need_mind() or STATE["mind"]._gen is None:
        return jsonify({"error": "load a dataset first"})
    from holographic_generate import generate_text, real_word_fraction, distinct_ngram_fraction
    mind = STATE["mind"]
    # find a FLAT HolographicNGram among the loaded generators (it exposes _distribution);
    # hierarchical schema generators don't, so nucleus decoding doesn't apply to them.
    ng = None
    for g in getattr(mind, "_gens", {}).values():
        if g.get("kind") == "flat" and hasattr(g["gen"], "_distribution"):
            ng = g["gen"]; break
    if ng is None and hasattr(mind._gen, "_distribution"):
        ng = mind._gen
    if ng is None:
        return jsonify({"error": "this dataset's generator is hierarchical, not a flat "
                        "n-gram -- nucleus sampling needs the character n-gram (Brown, "
                        "Reuters, or Books)"})
    j = request.json or {}
    seed = j.get("seed") or "the "
    length = max(40, min(int(j.get("length", 300)), 600))
    top_p = max(0.5, min(float(j.get("top_p", 0.85)), 1.0))
    rep = max(0.0, min(float(j.get("rep_penalty", 0.0)), 0.9))
    vocab = set((STATE.get("seq_raw") or "").lower().split())
    nuc = generate_text(ng, seed, length=length, temperature=0.6, top_p=top_p,
                        rep_penalty=rep, rng=np.random.default_rng(0))
    tmp = ng.generate(seed, length, 0.6)
    out = {"nucleus": nuc, "temperature": tmp,
           "nucleus_distinct4": round(distinct_ngram_fraction(nuc), 3),
           "temperature_distinct4": round(distinct_ngram_fraction(tmp), 3)}
    if vocab:
        out["nucleus_realword"] = round(real_word_fraction(nuc, vocab), 3)
        out["temperature_realword"] = round(real_word_fraction(tmp, vocab), 3)
    out["note"] = ("nucleus keeps the smallest set of likeliest next-characters summing to "
                   "top_p and samples within it -- fewer tail characters that break words")
    return jsonify(out)


@app.route("/api/unified/persist", methods=["POST"])
def persist_endpoint():
    """Save the loaded mind's whole LEARNED MEMORY (its SelfOrganizingMind: the encoder's
    learned meaning space plus the classified prototype bank) through the frozen core's
    versioned save/load, reload it, and prove it is identical -- same word neighbours AND
    the same classifications. A trained result is saved to disk and restored, not
    recomputed, with a version stamp that fails loudly on a format mismatch."""
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    import os
    import tempfile
    import holographic_core as core
    from holographic_ai import cosine
    mind = STATE["mind"]
    mem = getattr(mind, "memory", None)
    if mem is None or not hasattr(mem, "to_state"):
        return jsonify({"error": "this mind has no persistable memory yet -- load a dataset first"})
    ctx = getattr(getattr(mem.encoder, "_text", None), "context", None) or {}
    # which precision to persist at: float32 (default), int8, or auto (dynamic per-array).
    # The recent quantization work lives in core.save; expose it here so it is reachable
    # from the live stack, not just the library.
    quant = (request.json or {}).get("quant", "float32")
    save_kw = {"int8": dict(quant="int8"), "auto": dict(quant="auto")}.get(quant, dict(compress=True))
    # mkstemp atomically creates the file and returns an open fd, avoiding the race that
    # makes the older mktemp() insecure; we just need the unique path for save/load.
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    core.save(mem, path, **save_kw)                     # the whole SelfOrganizingMind
    back = core.load(path)
    size = os.path.getsize(path)
    os.remove(path)
    # float32 baseline size, so the panel can SHOW what the chosen quantization saved
    fd2, p2 = tempfile.mkstemp(suffix=".npz"); os.close(fd2)
    core.save(mem, p2, compress=True); base_size = os.path.getsize(p2); os.remove(p2)
    # verify: prototypes survive, word neighbours survive, and -- the real test --
    # classifications match on a sample of what the mind already learned
    words = list(ctx)
    probe = next((w for w in ("city", "money", "market", "the", words[0] if words else "")
                  if w in ctx), words[0] if words else None)
    def neighbours(enc):
        pv = enc._text.wordvec(probe)
        sims = sorted(((cosine(pv, enc._text.wordvec(w)), w) for w in words if w != probe), reverse=True)
        return [w for _s, w in sims[:5]]
    same_neighbours = (probe is None) or (neighbours(mem.encoder) == neighbours(back.encoder))
    labels = list(mem.live.labels()) if hasattr(mem.live, "labels") else []
    samples = words[:30]
    same_class = all(mem.classify(w) == back.classify(w) for w in samples) if samples else True
    bad_version_caught = False
    try:
        st = core.to_state(mem); st["state_version"] = core.STATE_VERSION + 99
        core.from_state(st)
    except ValueError:
        bad_version_caught = True
    return jsonify({
        "words": len(words),
        "prototypes": mem.live.size(),
        "labels": sorted(str(l) for l in labels)[:12],
        "bytes": size,
        "quant": quant,
        "float32_bytes": base_size,
        "shrink": round(base_size / size, 2) if size else 1.0,
        "probe": probe,
        "neighbours_before": neighbours(mem.encoder) if probe else [],
        "neighbours_after": neighbours(back.encoder) if probe else [],
        "same_neighbours": bool(same_neighbours),
        "same_classifications": bool(same_class),
        "version_guard_works": bool(bad_version_caught),
        "note": ("the whole learned memory -- meaning space and prototype bank -- round-trips "
                 "through one version-stamped .npz: identical word neighbours and identical "
                 "classifications, so a trained mind is saved and reloaded, not recomputed; an "
                 "incompatible format version is refused rather than loaded silently wrong. The "
                 "precision is selectable -- float32, int8, or dynamic auto (per-array, by the "
                 "data's own separation) -- and classifications survive the chosen level")})


PAGE = r"""
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UnifiedMind console</title>
<style>
  :root{--bg:#0e1116;--card:#171c24;--line:#2a313c;--ink:#e7edf5;--muted:#8b97a7;
        --teal:#3fd9c8;--teal2:#7af0e2;--amber:#f5b94d;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:22px 26px;border-bottom:1px solid var(--line)}
  h1{margin:0;font-size:20px}.sub{color:var(--muted);margin-top:4px;font-size:13px}
  main{max-width:920px;margin:0 auto;padding:22px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:18px}
  .card h2{margin:0 0 10px;font-size:15px;color:var(--teal2);letter-spacing:.3px}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  select,input,textarea,button{font:inherit}
  select,input,textarea{background:#0f141b;border:1px solid var(--line);color:var(--ink);
        border-radius:8px;padding:9px 11px}
  textarea{width:100%;min-height:54px;resize:vertical}
  button{background:var(--teal);color:#04201d;border:0;border-radius:8px;padding:9px 16px;
        font-weight:600;cursor:pointer}
  button:hover{background:var(--teal2)}
  button.ghost{background:#1d242e;color:var(--ink);border:1px solid var(--line)}
  .muted{color:var(--muted)}.out{margin-top:12px;white-space:pre-wrap;font-size:14px}
  .pill{display:inline-block;background:#0f141b;border:1px solid var(--line);border-radius:999px;
        padding:3px 10px;margin:3px 4px 0 0;font-size:12.5px}
  .big{font-size:26px;font-weight:700;color:var(--amber)}
  .disabled{opacity:.5;pointer-events:none}
  code{background:#0f141b;border:1px solid var(--line);border-radius:5px;padding:1px 5px}
  /* ---- searchable / categorized / collapsible example cards ---- */
  .toolbar{position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0 10px;
        margin-bottom:6px;border-bottom:1px solid var(--line)}
  #search{width:100%;font-size:15px;padding:11px 13px}
  .cats{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}
  .cat{cursor:pointer;background:#0f141b;border:1px solid var(--line);color:var(--muted);
        border-radius:999px;padding:5px 13px;font-size:13px;user-select:none}
  .cat:hover{color:var(--ink)}
  .cat.active{background:var(--teal);color:#04201d;border-color:var(--teal);font-weight:600}
  .count{color:var(--muted);font-size:12.5px;margin-top:8px}
  .card.collapsible>.card-head{cursor:pointer;display:flex;align-items:flex-start;gap:10px}
  .card-head .twirl{color:var(--muted);transition:transform .15s;margin-top:2px;font-size:12px}
  .card.open .card-head .twirl{transform:rotate(90deg)}
  .card-head .titles{flex:1}
  .card-head h2{margin:0 0 4px}
  .card-desc{color:var(--muted);font-size:13px}
  .card .tags{margin-top:7px}
  .tag{display:inline-block;background:#0f141b;border:1px solid var(--line);color:var(--muted);
        border-radius:6px;padding:1px 8px;margin:2px 4px 0 0;font-size:11.5px}
  .card-body{display:none;margin-top:14px;border-top:1px solid var(--line);padding-top:14px}
  .card.open>.card-body{display:block}
  .card.hidden{display:none}
  .nomatch{color:var(--muted);text-align:center;padding:30px 0}
  mark{background:var(--amber);color:#04201d;border-radius:2px;padding:0 1px}
</style></head><body>
<header>
  <h1>UnifiedMind &mdash; one model, one space</h1>
  <div class="sub">Pull a real corpus, train one mind, then classify / recall / organize / generate against it.</div>
</header>
<main>

  <div class="toolbar">
    <input id="search" placeholder="Search examples &mdash; try &quot;compress&quot;, &quot;predict&quot;, &quot;structure&quot;, &quot;vision&quot;&hellip;" oninput="filterCards()">
    <div class="cats" id="cats"></div>
    <div class="count" id="count"></div>
  </div>

  <div class="card">
    <h2>1 &middot; pull + train</h2>
    <div class="muted">Train one mind on a dataset, or stack several. <b>Start fresh</b> begins a new brain; <b>Add on top</b> layers the dataset onto the current brain &mdash; e.g. dictionary+encyclopedia first to lay down a base structure, then books, then Reuters or country info. Training is cached and shared, so re-building a stack you've made before is instant. The box below always shows what the current brain has been trained on, in order.</div>
    <div class="row" style="margin-top:8px">
      <select id="ds"></select>
      <button onclick="load('replace')">Start fresh</button>
      <button onclick="load('add')">Add on top &darr;</button>
      <label class="muted"><input type="checkbox" id="fresh"> rebuild (ignore cache)</label>
      <span id="loadmsg" class="muted"></span>
    </div>
    <div id="brainstack" class="out" style="margin-top:8px"></div>
    <div id="trained" class="out"></div>
  </div>

  <div class="card">
    <h2>2 &middot; ask <span class="muted">(question router)</span></h2>
    <div class="muted">This mind is <b>not a chatbot</b> &mdash; but most questions have a shape that maps to something it actually knows. Type a question and it routes to the right operation: <i>"what is a dog?"</i> &rarr; meaning + is_a chain, <i>"is a dog an animal?"</i> &rarr; taxonomic check, <i>"what is the capital of france?"</i> &rarr; a record's role, <i>"classify: &lt;text&gt;"</i> &rarr; nearest category, <i>"what is like &lt;text&gt;"</i> &rarr; nearest memory. Anything it can't map, it continues as text &mdash; and says so, rather than pretending to answer.</div>
    <div class="row" style="margin-top:8px">
      <input id="ask_q" placeholder="what is a dog?" style="width:320px"
             onkeydown="if(event.key==='Enter')askQ()">
      <button onclick="askQ()">Ask</button>
    </div>
    <div id="askout" class="out"></div>
  </div>

  <div id="ops" class="disabled">
  <div class="card">
    <h2>2 &middot; classify &amp; recall</h2>
    <textarea id="cq" placeholder="type a sentence in the style of the corpus..."></textarea>
    <div class="row" style="margin-top:8px"><button onclick="classify()">Classify</button>
      <button class="ghost" onclick="recall()">Recall nearest</button>
      <button class="ghost" onclick="resolution()">How much resolution?</button></div>
    <div id="cout" class="out"></div>
  </div>

  <div class="card">
    <h2>3 &middot; organize</h2>
    <div class="muted">How the one memory split each label into sub-prototypes (multi-modal labels get more).</div>
    <div class="row" style="margin-top:8px"><button onclick="organize()">Show &amp; reorganize</button></div>
    <div id="oout" class="out"></div>
  </div>

  <div class="card">
    <h2>3&frac12; &middot; relations <span class="muted">(record datasets)</span></h2>
    <div class="muted">The mind answers WHY over its own memory: per-role explanation of two learned labels, find-by-attribute, and a two-hop chain &mdash; every readout cleaned up to a symbol.</div>
    <div class="row" style="margin-top:8px">
      <input id="rel_a" placeholder="label A, e.g. france" style="width:130px">
      <input id="rel_b" placeholder="label B, e.g. belgium" style="width:130px">
      <button onclick="relExplain()">Explain</button>
    </div>
    <div class="row" style="margin-top:6px">
      <input id="rel_role" placeholder="role, e.g. capital" style="width:130px">
      <input id="rel_val" placeholder="value, e.g. tokyo" style="width:130px">
      <button onclick="relFind()">Find</button>
      <input id="rel_chain" placeholder="chain, e.g. capital>currency, currency>language" style="width:260px">
      <button onclick="relAsk()">Ask</button>
    </div>
    <div id="relout" class="out"></div>
  </div>

  <div class="card">
    <h2>3&frac34; &middot; curriculum <span class="muted">(dictionary + encyclopedia dataset)</span></h2>
    <div class="muted">Load the <b>Dictionary + encyclopedia</b> dataset. The mind learns word MEANING from definitions and relational KNOWLEDGE from is_a facts &mdash; both natively. Enter a word to see its meaning-neighbours (the dictionary layer) and its is_a chain climbed as a path-traced ray (the encyclopedia layer), side by side. Try: dog, wolf, cat, lion, rock, iron, granite, copper.</div>
    <div class="row" style="margin-top:8px">
      <input id="cur_word" placeholder="word, e.g. dog" style="width:160px">
      <button onclick="curriculum()">Look up</button>
    </div>
    <div id="curout" class="out"></div>
  </div>

  <div class="card">
    <h2>4 &middot; generate</h2>
    <div class="row">
      <input id="seed" value="the " style="width:160px" placeholder="seed text">
      <label class="muted">length <input id="len" type="number" value="220" style="width:80px"></label>
      <label class="muted">temp <input id="temp" type="number" step="0.05" value="0.45" style="width:80px"></label>
      <label class="muted">top-p <input id="topp" type="number" step="0.05" min="0.1" max="1" value="1.0" style="width:70px" title="below 1.0 = nucleus decoding (trims the unlikely tail, more coherent)"></label>
      <button onclick="generate()">Generate</button>
    </div>
    <div id="gout" class="out"></div>
  </div>

  <div class="card">
    <h2>4&frac18; &middot; predictive loop <span class="muted">(anticipate &amp; correct)</span></h2>
    <div class="muted">The active layer on top of storage: the mind reads the corpus one token at a time, <em>anticipates</em> each next token, and learns from its surprise. It predicts by resonance &mdash; so a context it never saw exactly still predicts, generalising from similar ones. Watch accuracy rise with exposure and free energy (prediction error) fall.</div>
    <div class="row" style="margin-top:8px"><button onclick="predictive()">Live the sequence</button></div>
    <div id="pout" class="out"></div>
  </div>

  <div class="card">
    <h2>4&frac38; &middot; query &amp; generate <span class="muted">(ask it something)</span></h2>
    <div class="muted">Ask a question; the system generates a continuation steered toward what the query is about, held coherent by the structure guard. It reports how on-query (relevance) and how coherent (structure) the answer is &mdash; and the unsteered baseline beside it, so the query-pull is visible.</div>
    <div class="row" style="margin-top:8px">
      <input id="rq" placeholder="e.g. the school and education for children" style="width:300px">
      <button onclick="respondQuery()">Respond</button>
    </div>
    <div id="rqout" class="out"></div>
  </div>

  <div class="card">
    <h2>4&frac12; &middot; deliberate <span class="muted">(think before answering)</span></h2>
    <div class="muted">Rather than emit the first draft, the system drafts a response, judges it (on-query and coherent), and refines &mdash; keeping the best and stopping once it's good enough. The iteration count is the thinking time, and it adapts: easy queries settle fast, hard ones take longer. The trace below is the inner deliberation made visible.</div>
    <div class="row" style="margin-top:8px">
      <input id="dq" placeholder="e.g. the school and education for children" style="width:300px">
      <button onclick="deliberateQuery()">Think &amp; respond</button>
    </div>
    <div id="dqout" class="out"></div>
  </div>

  <div class="card">
    <h2>4&frac34; &middot; self-discovery <span class="muted">(find the units, no labels)</span></h2>
    <div class="muted">Strip every space from the text and the system rediscovers the word boundaries on its own &mdash; from where its next-character prediction becomes uncertain (branching entropy peaks at unit ends). The discovered chunks then compress far better than single characters: finding the right decomposition shortens the description.</div>
    <div class="row" style="margin-top:8px"><button onclick="discover()">Discover the units</button></div>
    <div id="scout" class="out"></div>
  </div>

  <div class="card">
    <h2>4&frac78; &middot; factorize <span class="muted">(pull a composite apart)</span></h2>
    <div class="muted">Binding combines several vectors into one; the hard inverse is recovering which vector came from each codebook given only the composite. The resonator network does it by searching in superposition &mdash; converging on the factors without ever enumerating the combinatorial space. (Uses self-inverse MAP binding, the algebra factorization needs.)</div>
    <div class="row" style="margin-top:8px"><button onclick="factorize()">Bind three, then factor</button></div>
    <div id="fzout" class="out"></div>
  </div>

  <div class="card">
    <h2>5&frac14; &middot; compose a scene <span class="muted">(run the resonator FORWARD)</span></h2>
    <div class="muted">The inverse of factorize: pick attribute atoms, bind and superpose them into a scene vector that was never stored, then render it. A generated scene is real because it analyses straight back &mdash; the resonator factors the composed vector, and the rendered pixels auto-tag, recovering exactly the colour/shape/texture it was built from. The colour sweep is procedural animation by composition.</div>
    <div class="row" style="margin-top:8px">
      <label class="muted">objects <input id="cmpN" type="number" value="2" min="1" max="4" style="width:54px"></label>
      <button onclick="composeScene()">Compose &amp; verify</button>
    </div>
    <div id="cmpout" class="out"></div>
  </div>

  <div class="card">
    <h2>5&frac13; &middot; nested scene <span class="muted">(fractal: same above, same below)</span></h2>
    <div class="muted">The same bind-and-superpose that builds a scene from objects, run one level up to build a scene-of-scenes. Several sub-scenes are composed, bound to group atoms, and superposed; then each group is peeled back out and factored &mdash; the identical unbind-then-factor at two levels. A sub-scene is to the super-scene exactly what an object is to a scene. Recovery is near-perfect at 2&ndash;3 groups and degrades gracefully beyond, as group cross-talk accumulates (the flat scene's capacity limit, one level up).</div>
    <div class="row" style="margin-top:8px">
      <label class="muted">groups <input id="nstG" type="number" value="2" min="2" max="4" style="width:54px"></label>
      <label class="muted">per group <input id="nstP" type="number" value="2" min="1" max="3" style="width:54px"></label>
      <button onclick="nestedScene()">Compose nested &amp; verify</button>
    </div>
    <div id="nstout" class="out"></div>
  </div>

  <div class="card">
    <h2>5&frac12; &middot; morph <span class="muted">(slerp in the coefficient domain)</span></h2>
    <div class="muted">Spherical interpolation between two rendered shapes <i>in the DCT-coefficient domain</i>, inverse-transformed per frame. The honest difference from a pixel crossfade is ghosting: a crossfade midpoint <i>is</i> the double-exposure of both pictures; the coefficient morph blends structure, so its midpoint sits measurably away from that double image.</div>
    <div class="row" style="margin-top:8px"><button onclick="morphScene()">Morph two shapes</button></div>
    <div id="mphout" class="out"></div>
  </div>

  <div class="card">
    <h2>5&frac34; &middot; nucleus text <span class="muted">(top-p decoding vs temperature)</span></h2>
    <div class="muted">Generate from the loaded mind's character n-gram with nucleus (top-p) decoding: keep the smallest set of likeliest next-characters summing to p and sample within it. Trimming the unlikely tail removes the garbage characters that break words &mdash; measurably more real words than plain temperature sampling, at a little less variety. Needs a flat n-gram dataset (Brown, Reuters, Books).</div>
    <div class="row" style="margin-top:8px">
      <input id="nucSeed" placeholder="seed text, e.g. the " value="the " style="width:160px">
      <label class="muted">top-p <input id="nucP" type="number" value="0.85" min="0.5" max="1" step="0.05" style="width:60px"></label>
      <button onclick="nucleusGen()">Generate both ways</button>
    </div>
    <div id="nucout" class="out"></div>
  </div>

  <div class="card">
    <h2>5&frac78; &middot; save &amp; reload <span class="muted">(persist a trained mind)</span></h2>
    <div class="muted">A trained result should be savable, not recomputed every run. This snapshots the mind's learned meaning space (its word&rarr;vector map) through the frozen core's version-stamped save/load, reloads it, and proves the reloaded space is identical &mdash; same vectors and the same nearest-neighbour structure for a probe word. An incompatible format version is refused rather than loaded silently wrong. The precision is selectable: <b>float32</b> (default), <b>int8</b> (~2&times; smaller), or <b>auto</b> &mdash; dynamic per-array quantization that picks binary/int8/float32 from each array's own separation and size.</div>
    <div class="row" style="margin-top:8px">
      <label class="muted">precision
        <select id="prsQ" style="margin-left:4px">
          <option value="float32">float32</option>
          <option value="int8">int8</option>
          <option value="auto">auto (dynamic)</option>
        </select>
      </label>
      <button onclick="persistMind()">Save, reload, verify</button>
    </div>
    <div id="prsout" class="out"></div>
  </div>

  <div class="card">
    <h2>6 &middot; many NPCs, one mind <span class="muted">(shared base + deltas)</span></h2>
    <div class="muted">For a game with many NPCs: instead of a full brain each, train one base mind, freeze it, and give every NPC a lightweight overlay that holds only what it personally learned. NPCs inherit all shared knowledge, keep private learning isolated, and can propagate it back to everyone. Because all instances share the same atoms, merging is just superposition.</div>
    <div class="row" style="margin-top:8px">
      <label class="muted">population <input id="pop_n" type="number" value="50" min="2" max="500" style="width:90px"></label>
      <button onclick="population()">Spawn a population</button>
    </div>
    <div id="popout" class="out"></div>
  </div>

  <div class="card">
    <h2>7 &middot; sprite pack <span class="muted">(cross-sprite compression)</span></h2>
    <div class="muted">A sprite sheet is one structured body, not hundreds of unrelated files. This packs the real sprite set against shared references and compares it, bit-exact, to compressing each sprite on its own &mdash; the cross-sprite structure that separate files throw away.</div>
    <div class="row" style="margin-top:8px">
      <label class="muted">sprites <input id="spr_n" type="number" value="200" min="20" max="712" style="width:90px"></label>
      <button onclick="sprites()">Pack the set</button>
    </div>
    <div id="sprout" class="out"></div>
  </div>

  <div class="card">
    <h2>8 &middot; image vault <span class="muted">(perceptual repository)</span></h2>
    <div class="muted">The image repository as one perceptual memory: ingest the picture set, group near-duplicates by a size- and format-invariant fingerprint, and answer query-by-example &mdash; pictures living in the same kind of space the rest of the engine uses for words and meaning.</div>
    <div class="row" style="margin-top:8px"><button onclick="vault()">Open the vault</button></div>
    <div id="vaultout" class="out"></div>
  </div>

  <div class="card">
    <h2>9 &middot; market structure <span class="muted">(volatility clustering)</span></h2>
    <div class="muted">An experiment to run: does real market data carry the structure the engine looks for? Raw signed returns are efficient-market-like, but <b>volatility clustering</b> &mdash; bursts cluster in time, so |returns| are positively autocorrelated &mdash; is the honest signature. Reports the lag-1 |return| autocorrelation and how many sigma it beats a shuffle of the same returns (&gt;2 is real structure; the shuffle control should collapse to ~0).</div>
    <div class="row" style="margin-top:8px">
      <select id="mkt_ds"><option value="sol">SOL ticks (~1,500 moves)</option><option value="dai">DAI/WETH candles (~1,000 returns)</option></select>
      <button onclick="market()">Test for structure</button>
    </div>
    <div id="mktout" class="out"></div>
  </div>

  <div class="card">
    <h2>10 &middot; big-text run <span class="muted">(heavy experiment)</span></h2>
    <div class="muted">A heavy run on a large slice of the loaded corpus, kept out of the test suite: structure score (real vs shuffled vs random), the lossless codec ratio, and self-discovered word-boundary F1 &mdash; all at once. Load a sizeable text dataset (Reuters, Brown, Books) first, then run and read the numbers. Bump the token count for a bigger experiment.</div>
    <div class="row" style="margin-top:8px">
      <label class="muted">tokens <input id="bt_n" type="number" value="3000" min="500" max="20000" step="500" style="width:100px"></label>
      <button onclick="bigtext()">Run the big-text experiment</button>
    </div>
    <div id="btout" class="out"></div>
  </div>

  <div class="card">
    <h2>5 &middot; lossless codec <span class="muted">(compress &harr; decompress)</span></h2>
    <div class="muted">Both directions, exactly. The predictor ranks the vocabulary at each step; the rank stream is the compressed code, and because the decoder runs the same predictor it decompresses to the exact original. The size is bounded by structure &mdash; real text shrinks, random data barely (no free lunch), and a perfectly predictable stream would collapse to almost the seed alone.</div>
    <div class="row" style="margin-top:8px"><button onclick="codec()">Compress &amp; restore</button></div>
    <div id="cxout" class="out"></div>
  </div>

  <div class="card">
    <h2>4&frac14; &middot; topic pull <span class="muted">(an honest experiment)</span></h2>
    <div class="muted">Why isn't this brain an LLM? Generation is a shallow n-gram. This re-ranks word-n-gram candidates by alignment to a running topic vector &mdash; deeper conditioning &mdash; and <em>measures</em> whether it buys coherence. Watch what happens to diversity as the pull increases.</div>
    <div class="row" style="margin-top:8px">
      <input id="tp_seed" value="the" style="width:160px" placeholder="seed word">
      <button onclick="topicPull()">Run the sweep</button>
    </div>
    <div id="tpout" class="out"></div>
  </div>

  <div class="card">
    <h2>5&frac14; &middot; source tracing <span class="muted">(who taught this?)</span></h2>
    <div class="muted">WHO taught this text? Paste a passage (or trace a generated one above) and the mind ranks the dataset's sources by how much of the text's actual transitions each one taught &mdash; from the same tables generation reads. Strong on real held-out text (measured ~70&ndash;92% top-1 depending on how distinct the sources are); near-uniform on freely-generated text, which legitimately blends everyone.</div>
    <div class="row" style="margin-top:8px">
      <textarea id="attin" rows="3" style="width:100%;background:#0d1626;color:#cfe;border:1px solid #2a3a55;border-radius:6px;padding:6px" placeholder="paste text to trace to its source..."></textarea>
    </div>
    <div class="row" style="margin-top:6px"><button onclick="trace($('attin').value)">Trace sources &rarr;</button></div>
    <div id="trout2" class="out"></div>
  </div>

</main>
<script>
const $=id=>document.getElementById(id);
async function post(url,body){const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify(body||{})});return r.json();}

async function init(){
  const r=await fetch("/api/unified/datasets").then(x=>x.json());
  $("ds").innerHTML=r.datasets.map(d=>`<option value="${d.id}">${d.name}</option>`).join("");
  if(!r.nltk) $("loadmsg").textContent="(install nltk to pull corpora: pip install nltk)";
  buildCards();
  refreshStack();
}

/* ---- example catalog: category, tags, and a one-line description per card.
   Keyed by a distinctive substring of each card's <h2> text so we never have to
   touch the panels themselves. Cards not listed default to "Setup". ---- */
const CATALOG=[
  {key:"pull + train", cat:"Setup", pinned:true, tags:["corpus","train"],
    desc:"Pull a real corpus and train one mind you can probe with every example below."},
  {key:"question router", cat:"Reason", tags:["question","router","answer"],
    desc:"Ask a question; the router decides whether to answer or complete, and responds."},
  {key:"classify", cat:"Memory", tags:["classify","recall","nearest"],
    desc:"Classify an input and recall the closest stored items from the one shared space."},
  {key:"organize", cat:"Memory", tags:["cluster","organize","structure"],
    desc:"Watch the mind reorganize what it holds into a tidier structure."},
  {key:"relations", cat:"Reason", tags:["records","roles","bind"],
    desc:"Bind role/filler records and query the relations between them."},
  {key:"curriculum", cat:"Setup", tags:["dictionary","encyclopedia","prior"],
    desc:"Seed meaning from a dictionary + encyclopedia prior before reading a corpus."},
  {key:"4 &middot; generate", cat:"Generate", tags:["generate","ngram","text"],
    desc:"Generate text from the trained mind by sampling its sequence memory."},
  {key:"predictive loop", cat:"Predict", tags:["anticipate","surprise","free-energy","learning-curve"],
    desc:"The mind reads a stream, anticipates each next token, and learns from its surprise."},
  {key:"query &amp; generate", cat:"Generate", tags:["query","steer","relevance","structure"],
    desc:"Ask something and get a continuation steered toward the query, kept coherent."},
  {key:"deliberate", cat:"Generate", tags:["think","iterate","adaptive","negotiate"],
    desc:"Draft, judge, and refine before answering &mdash; thinking time adapts to difficulty."},
  {key:"self-discovery", cat:"Structure", tags:["segment","boundaries","decompose","no-labels"],
    desc:"Strip the spaces and the mind rediscovers the word boundaries on its own."},
  {key:"factorize", cat:"Structure", tags:["resonator","factor","superposition","decompose"],
    desc:"Bind several vectors into one, then pull them back apart by searching in superposition."},
  {key:"compose a scene", cat:"Generate", tags:["compose","resonator","forward","render","novel","round-trip"],
    desc:"Run the factorizer forward: compose a scene that was never stored, render it, and verify it analyses straight back."},
  {key:"nested scene", cat:"Generate", tags:["nested","fractal","recursive","scene","self-similar","round-trip"],
    desc:"Fractal composition: the same bind-and-factor that builds a scene from objects, run one level up to build a scene-of-scenes."},
  {key:"morph", cat:"Generate", tags:["morph","slerp","interpolate","coefficient","ghosting","animate"],
    desc:"Slerp between two shapes in the coefficient domain &mdash; blends structure where a crossfade just ghosts."},
  {key:"nucleus text", cat:"Generate", tags:["nucleus","top-p","decode","coherence","sampling"],
    desc:"Generate from the n-gram with nucleus (top-p) decoding &mdash; more real words than plain temperature."},
  {key:"save &amp; reload", cat:"Setup", tags:["persist","save","load","version","reload"],
    desc:"Save the mind's learned meaning space through the versioned core and reload it identically."},
  {key:"many NPCs", cat:"Scale", tags:["npc","shared-base","branch","delta","merge","game"],
    desc:"Run a population of NPCs as one shared mind plus lightweight per-instance deltas."},
  {key:"sprite pack", cat:"Images", tags:["sprites","compression","cross-file","lossless","game-assets"],
    desc:"Pack the real sprite set as one body against shared references &mdash; bit-exact, far smaller."},
  {key:"image vault", cat:"Images", tags:["images","wallpaper","dedup","cluster","query-by-example","perceptual"],
    desc:"Group the picture repository by perceptual fingerprint and query it by example."},
  {key:"market structure", cat:"Market", tags:["market","returns","volatility-clustering","experiment","crypto"],
    desc:"Test whether real market data carries volatility-clustering structure vs a shuffle."},
  {key:"big-text run", cat:"Experiment", tags:["large","structure","codec","segment","heavy","run-it"],
    desc:"Heavy on-demand run: structure, lossless codec, and boundary F1 on a large text slice."},
  {key:"lossless codec", cat:"Compress", tags:["compress","lossless","seed","reversible"],
    desc:"Compress to a rank-code and decompress to the exact original &mdash; both directions."},
  {key:"source tracing", cat:"Reason", tags:["attribution","provenance","who-taught"],
    desc:"Trace a passage back to the sources whose stored transitions best explain it."},
  {key:"topic pull", cat:"Generate", tags:["steer","topic","honest-negative"],
    desc:"An honest experiment: does pulling generation toward a topic actually help?"},
];
const CATS=["All","Setup","Memory","Predict","Generate","Structure","Compress","Reason","Scale","Images","Market","Experiment"];

function meta(h2text){
  for(const m of CATALOG){ if(h2text.includes(m.key)) return m; }
  return {cat:"Setup", tags:[], desc:""};
}
function cleanTitle(h2){
  // strip the leading "N · " numbering and the muted parenthetical for the card title
  let t=h2.replace(/^\s*[\d&frac;\/¼½¾⅛⅜⅝⅞]+\s*&middot;\s*/,"").trim();
  return t;
}

let activeCat="All";
function buildCards(){
  const cards=[...document.querySelectorAll("main > .card")];
  cards.forEach(card=>{
    const h2=card.querySelector("h2"); if(!h2) return;
    const m=meta(h2.innerHTML);
    card.dataset.cat=m.cat;
    card.dataset.tags=(m.tags||[]).join(" ");
    card.dataset.title=h2.textContent;
    card.classList.add("collapsible");
    // build the head (twirl + title + description + tags) and move the rest into a body
    const head=document.createElement("div"); head.className="card-head";
    const titleHTML=h2.innerHTML;
    const tagHTML=(m.tags||[]).map(t=>`<span class="tag">${t}</span>`).join("");
    head.innerHTML=`<span class="twirl">&#9656;</span><div class="titles">`+
      `<h2>${titleHTML}</h2>`+(m.desc?`<div class="card-desc">${m.desc}</div>`:"")+
      (tagHTML?`<div class="tags">${tagHTML}</div>`:"")+`</div>`;
    const body=document.createElement("div"); body.className="card-body";
    // move every original child (except the h2 we copied) into the body
    [...card.childNodes].forEach(n=>{ if(n!==h2) body.appendChild(n); });
    h2.remove();
    card.appendChild(head); card.appendChild(body);
    head.addEventListener("click",()=>card.classList.toggle("open"));
    if(m.pinned) card.classList.add("open");   // keep "pull + train" open
  });
  // category pills
  $("cats").innerHTML=CATS.map(c=>`<span class="cat${c==='All'?' active':''}" data-c="${c}" onclick="setCat('${c}')">${c}</span>`).join("");
  filterCards();
}
function setCat(c){
  activeCat=c;
  [...document.querySelectorAll(".cat")].forEach(p=>p.classList.toggle("active",p.dataset.c===c));
  filterCards();
}
function filterCards(){
  const q=($("search").value||"").trim().toLowerCase();
  const cards=[...document.querySelectorAll("main > .card")];
  let shown=0;
  cards.forEach(card=>{
    const hay=(card.dataset.title+" "+card.dataset.tags+" "+card.dataset.cat+" "+
      (card.querySelector(".card-desc")?.textContent||"")).toLowerCase();
    const matchCat=(activeCat==="All"||card.dataset.cat===activeCat);
    const matchQ=(!q||hay.includes(q));
    const show=matchCat&&matchQ;
    card.classList.toggle("hidden",!show);
    if(show){ shown++; if(q) card.classList.add("open"); }
  });
  $("count").textContent=`${shown} example${shown===1?"":"s"}${q?` matching \u201c${q}\u201d`:""}${activeCat!=="All"?` in ${activeCat}`:""}`;
  let nm=$("nomatch");
  if(shown===0){ if(!nm){ nm=document.createElement("div"); nm.id="nomatch"; nm.className="nomatch";
      nm.textContent="No examples match \u2014 try a different word or category."; document.querySelector("main").appendChild(nm);} }
  else if(nm){ nm.remove(); }
}
async function load(mode){
  mode=mode||"replace";
  const fresh=$("fresh").checked;
  const verb=mode==="add"?"adding on top":(fresh?"retraining from scratch":"pulling + training");
  $("loadmsg").textContent=verb+"\u2026"; $("trained").innerHTML="";
  const r=await post("/api/unified/load",{id:$("ds").value, fresh:fresh, mode:mode});
  if(!r.ok){$("loadmsg").textContent="";$("trained").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("loadmsg").textContent=r.cached?"reused cached training (instant)":(r.added?("added "+r.added):"trained");
  const pills=r.labels.map(l=>`<span class="pill">${l}: ${r.counts[l]||0}</span>`).join("");
  $("trained").innerHTML=
    `<div>${r.desc}</div>
     <div style="margin-top:8px">${r.held_out>0?`held-out accuracy <span class="big">${r.accuracy}%</span>&nbsp;`:""}
        <span class="muted">${r.trained} trained / ${r.held_out} held out &middot;
        ${r.prototypes} prototypes &middot; ${r.gen_chars.toLocaleString()} chars for generation</span></div>
     <div style="margin-top:8px">${pills}</div>`;
  renderStack(r.trained_on||[]);
  $("ops").classList.remove("disabled");
}
function renderStack(list){
  if(!list||!list.length){ $("brainstack").innerHTML=""; return; }
  const chain=list.map((n,i)=>`<span class="pill" style="background:#0f2a26;border-color:var(--teal)">${i+1}. ${n}</span>`).join(' <span class="muted">&rarr;</span> ');
  $("brainstack").innerHTML=`<div><b>current brain trained on:</b> ${chain}</div>`+
    `<div class="muted" style="margin-top:4px">use &ldquo;Add on top&rdquo; to layer another dataset, or &ldquo;Start fresh&rdquo; for a new brain.</div>`;
}
async function refreshStack(){
  try{ const s=await fetch("/api/unified/trained").then(x=>x.json()); renderStack(s.trained_on||[]); }catch(e){}
}
async function classify(){
  const r=await post("/api/unified/classify",{text:$("cq").value});
  if(r.error){$("cout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const rob=r.robust;
  const robline=rob?`<br><span class="muted">multi-ray (5 resampled views, z-combined): <b style="color:var(--teal2)">${rob.label}</b> &mdash; ${Math.round(rob.agreement*100)}% of rays agree</span>`:"";
  $("cout").innerHTML=`classified as <b style="color:var(--teal2)">${r.label}</b> (cos ${r.score})
     <br><span class="muted">nearest stored item is a <b>${r.recall.label}</b> (cos ${r.recall.score})</span>${robline}`;
}
async function recall(){
  const r=await post("/api/unified/recall",{text:$("cq").value});
  if(r.error){$("cout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("cout").innerHTML=`nearest stored item: <b style="color:var(--teal2)">${r.label}</b> (cos ${r.score})
     <br><span class="muted">&ldquo;${r.example}\u2026&rdquo;</span>`;
}
async function organize(){
  const r=await post("/api/unified/organize",{});
  if(r.error){$("oout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const pills=Object.entries(r.after).map(([l,n])=>`<span class="pill">${l}: ${n}</span>`).join("");
  const story=r.story?`<div style="margin-top:8px;color:var(--amber)">the mind's own account: ${r.story}</div>`:"";
  $("oout").innerHTML=`reorganize decided: <b>${r.choice}</b><div style="margin-top:8px">${pills}</div>${story}
     <div class="muted" style="margin-top:6px">${r.note}</div>`;
}
async function generate(){
  $("gout").innerHTML='<span class="muted">generating\u2026</span>';
  const r=await post("/api/unified/generate",{seed:$("seed").value,length:+$("len").value,temperature:+$("temp").value,top_p:+$("topp").value});
  if(r.error){$("gout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  window._lastGen=r.text;
  $("gout").innerHTML=`<span style="color:var(--amber)">${r.text}</span>`+
    `<div style="margin-top:8px"><button onclick="trace()">Trace sources &rarr;</button>`+
    `<span class="muted" style="margin-left:8px">who taught the transitions this used?</span></div>`+
    `<div id="trout" style="margin-top:8px"></div>`;
}
async function trace(text){
  const t=text||window._lastGen||"";
  const fromPaste=!!text;
  const r=await post("/api/unified/attribute",{text:t});
  const node=fromPaste?$("trout2"):($("trout")||$("gout"));
  if(r.error){node.innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const barRow=(e,col)=>{const pct=Math.round(e.weight*100);
    return `<div style="display:flex;align-items:center;gap:8px;margin:3px 0">
      <span style="width:90px">${e.source}</span>
      <span style="flex:1;background:#1a2740;border-radius:4px;overflow:hidden">
        <span style="display:block;height:14px;width:${pct}%;background:${col}"></span></span>
      <span class="muted" style="width:40px;text-align:right">${pct}%</span></div>`;};
  const mat=r.by_material.map(e=>barRow(e,"#67d6a0")).join("");
  const sty=r.ranked.map(e=>barRow(e,"var(--amber)")).join("");
  const spanLine=r.span?`<div style="margin:6px 0;padding:6px;background:#0d1626;border-radius:6px;font-size:12px">
     longest verbatim span &rarr; <b style="color:#67d6a0">${r.span_source}</b>: <span class="muted">&ldquo;${r.span}&rdquo;</span></div>`:"";
  node.innerHTML=`<div style="margin-bottom:6px">verdict: <b style="color:${r.basis==='material'?'#67d6a0':'var(--amber)'}">${r.verdict}</b>
     <span class="muted">(by ${r.basis==='material'?'MATERIAL &mdash; a verbatim span pins it':'STYLE &mdash; no decisive span, stylistic match'})</span></div>
     ${spanLine}
     <div class="muted" style="margin:6px 0 2px">by material (sequence alignment):</div>${mat}
     <div class="muted" style="margin:8px 0 2px">by style (transition bag):</div>${sty}`;
}
async function relExplain(){
  const r=await post("/api/unified/relations",{op:"explain",a:$("rel_a").value,b:$("rel_b").value});
  if(r.error){$("relout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("relout").innerHTML="<table><tr><th>role</th><th>"+$("rel_a").value+"</th><th>"+$("rel_b").value+"</th><th></th></tr>"+
    r.explain.map(e=>`<tr><td>${e.role}</td><td>${e.a}</td><td>${e.b}</td><td>${e.shared?'<span style="color:var(--green)">SHARED</span>':'<span class="muted">differs</span>'}</td></tr>`).join("")+"</table>";
}
async function relFind(){
  const r=await post("/api/unified/relations",{op:"find",role:$("rel_role").value,value:$("rel_val").value});
  if(r.error){$("relout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const rec=Object.entries(r.find.record).map(([k,v])=>`<span class="pill">${k}: ${v}</span>`).join(" ");
  $("relout").innerHTML=`<b>${r.find.label}</b> holds ${$("rel_role").value}=${$("rel_val").value} (score ${r.find.score})<div style="margin-top:6px">${rec}</div>`;
}
async function relAsk(){
  const v=$("rel_val").value;
  const txt=$("rel_chain").value||"capital>currency, currency>language";
  const hops=txt.split(",").map(h=>h.split(">").map(s=>s.trim())).filter(h=>h.length===2&&h[0]&&h[1]);
  if(!hops.length){$("relout").innerHTML='<span class="muted">chain format: matchrole&gt;readrole, ...</span>';return;}
  const r=await post("/api/unified/relations",{op:"ask",start:v,hops:hops});
  if(r.error){$("relout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const path=hops.map(h=>`${h[0]}&rarr;${h[1]}`).join(" , ");
  const a=r.ask;
  const tp=a.throughput;
  const bar=tp!=null?` <span class="muted">(throughput ${tp.toFixed(2)} &mdash; the ray's accumulated confidence; low means trust it less)</span>`:"";
  const answer = a.answer==null
    ? `<span style="color:#f0a0a0">abstained (throughput too low to trust)</span>`
    : `<b style="color:var(--amber)">${a.answer}</b>`;
  $("relout").innerHTML=`ask <b>${v}</b> through [${path}] &rarr; ${answer}${bar}`;
}
async function askQ(){
  const q=($("ask_q").value||"").trim();
  if(!q){$("askout").innerHTML='<span class="muted">ask a question</span>';return;}
  const r=await post("/api/unified/answer",{question:q});
  if(r.error){$("askout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const A='color:var(--amber)';
  let html="";
  if(r.kind==="define"){
    const mean=(r.meaning||[]).map(m=>`${m[0]} <span class="muted">(${m[1].toFixed(2)})</span>`).join(", ")||'<span class="muted">no meaning learned</span>';
    const chain=(r.is_a_chain||[]).join(" &rarr; ");
    html=`<div><span class="muted">meaning &mdash; like:</span> ${mean}</div>`+
         (chain.includes("&rarr;")?`<div style="margin-top:6px"><span class="muted">is_a:</span> <b style="${A}">${chain}</b></div>`:"");
  } else if(r.kind==="is_a"){
    const yes=r.answer;
    html=`<b style="${A}">${yes?"Yes":"No"}</b> &mdash; <b>${r.subject}</b> ${yes?"is a":"is not a"} <b>${r.ancestor}</b>`+
         (yes?` <span class="muted">(${r.hops} hops, throughput ${r.throughput})</span>`:"")+
         `<div style="margin-top:6px"><span class="muted">chain:</span> ${(r.chain||[]).join(" &rarr; ")}</div>`;
  } else if(r.kind==="role"){
    html=`the <b>${r.role}</b> of <b>${r.concept}</b> is <b style="${A}">${r.value}</b> <span class="muted">(confidence ${r.confidence})</span>`;
  } else if(r.kind==="classify"){
    html=`nearest category: <b style="${A}">${r.label}</b> <span class="muted">(score ${r.score})</span>`;
  } else if(r.kind==="recall"){
    html=`nearest memory: <b style="${A}">${r.label}</b> <span class="muted">(score ${r.score})</span>`;
  } else if(r.kind==="completion"){
    html=`<div class="muted">${r.note}</div><div style="margin-top:6px"><b style="${A}">${r.text}</b></div>`;
  } else {
    html=`<span class="muted">${r.note||r.text||"no answer"}</span>`;
  }
  $("askout").innerHTML=html;
}
async function codec(){
  $("cxout").innerHTML='<span class="spin">compressing both ways&hellip;</span>';
  const r=await post("/api/unified/codec",{});
  if(r.error){$("cxout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("cxout").innerHTML=
    `<div>round-trip lossless: <b style="color:${r.lossless?'var(--teal2)':'var(--coral)'}">${r.lossless?'exact &#10003;':'mismatch &#10007;'}</b></div>`+
    `<div style="margin-top:4px">real text: <b style="color:var(--teal2)">${r.bits_per_token}</b> bits/token vs ${r.baseline} baseline &mdash; ratio <b>${r.ratio}</b></div>`+
    `<div style="margin-top:4px">random control: ratio <b style="color:var(--coral)">${r.random_ratio}</b> (barely shrinks &mdash; no free lunch)</div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function population(){
  $("popout").innerHTML='<span class="spin">spawning population&hellip;</span>';
  const r=await post("/api/unified/population",{population:parseInt($("pop_n").value)||50});
  if(r.error){$("popout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  let html=`<div class="muted">shared base knows: ${r.base_labels.join(", ")}</div>`;
  html+=`<div style="margin-top:8px">`;
  for(const n of r.npcs){
    html+=`<div style="margin:4px 0"><b style="color:var(--teal2)">${n.npc}</b> &mdash; `+
      `inherits <code>${n.inherits_shared}</code>, privately <code>${n.private}</code></div>`;
  }
  html+=`</div>`;
  html+=`<div style="margin-top:8px">isolation: <b>${r.isolation.npc}</b> asked about another NPC's private word &rarr; <b style="color:var(--amber)">${r.isolation.result}</b> (doesn't see it)</div>`;
  html+=`<div style="margin-top:4px">propagation: word "<b>${r.propagation.word}</b>" &mdash; before sharing: <b style="color:var(--coral)">${r.propagation.before}</b>, after a NPC propagates: <b style="color:var(--teal2)">${r.propagation.after}</b></div>`;
  html+=`<div style="margin-top:8px">memory: <b style="color:var(--teal2)">${r.cost.shared_total.toLocaleString()}</b> prototypes for ${r.population} NPCs vs <b style="color:var(--coral)">${r.cost.separate_total.toLocaleString()}</b> for separate brains &mdash; <b style="color:var(--amber)">${r.cost.saving_x.toFixed(0)}x</b> smaller</div>`;
  html+=`<div class="muted" style="margin-top:6px">${r.note}.</div>`;
  $("popout").innerHTML=html;
}
async function market(){
  $("mktout").innerHTML='<span class="spin">testing for structure&hellip;</span>';
  const r=await post("/api/unified/market",{dataset:$("mkt_ds").value});
  if(r.error){$("mktout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("mktout").innerHTML=
    `<div>${r.dataset}</div>`+
    `<div style="margin-top:4px">volatility clustering (|return| lag-1 acf): <b style="color:var(--teal2)">${r.vol_clustering_acf1}</b>, `+
    `<b style="color:${r.structured?'var(--teal2)':'var(--coral)'}">${r.zscore}&sigma;</b> above shuffle `+
    `<b style="color:${r.structured?'var(--teal2)':'var(--coral)'}">${r.structured?'&#10003; real structure':'&#10007; not significant'}</b></div>`+
    `<div style="margin-top:4px">shuffle control: acf <b style="color:var(--coral)">${r.shuffled_acf1}</b> (collapses to ~0)</div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function bigtext(){
  $("btout").innerHTML='<span class="spin">running the big-text experiment&hellip;</span>';
  const r=await post("/api/unified/bigtext",{tokens:parseInt($("bt_n").value)||3000});
  if(r.error){$("btout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  let h=`<div class="muted">corpus ${r.corpus_tokens.toLocaleString()} tokens, used ${r.used_tokens.toLocaleString()}</div>`;
  if(r.structure) h+=`<div style="margin-top:6px">structure score &mdash; real <b style="color:var(--teal2)">${r.structure.real}</b>, shuffled <b style="color:var(--amber)">${r.structure.shuffled}</b>, random <b style="color:var(--coral)">${r.structure.random}</b> (higher = more structured)</div>`;
  if(r.codec) h+=`<div style="margin-top:4px">lossless codec &mdash; <b style="color:${r.codec.lossless?'var(--teal2)':'var(--coral)'}">${r.codec.lossless?'exact':'mismatch'}</b>, ratio <b>${r.codec.ratio}</b> (${r.codec.bits_per_token} bits/token over ${r.codec.tokens})</div>`;
  if(r.segmentation) h+=`<div style="margin-top:4px">self-discovered boundaries &mdash; F1 <b style="color:var(--teal2)">${r.segmentation.f1}</b> vs <b style="color:var(--coral)">${r.segmentation.random_f1}</b> random (${r.segmentation.chars.toLocaleString()} chars)</div>`;
  h+=`<div class="muted" style="margin-top:6px">${r.note}.</div>`;
  $("btout").innerHTML=h;
}
async function sprites(){
  $("sprout").innerHTML='<span class="spin">packing the set&hellip;</span>';
  const r=await post("/api/unified/sprites",{n:parseInt($("spr_n").value)||200});
  if(r.error){$("sprout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("sprout").innerHTML=
    `<div>${r.used} of ${r.total_sprites} sprites &mdash; set-pack <b style="color:var(--teal2)">${r.set_pack.toLocaleString()}</b> bytes `+
    `vs per-file PNG <b style="color:var(--coral)">${r.per_file_png.toLocaleString()}</b> bytes</div>`+
    `<div style="margin-top:4px">that's <b style="color:var(--amber)">${r.saving_x}x</b> smaller, and <b style="color:var(--teal2)">${r.lossless?'bit-exact &#10003;':'NOT exact &#10007;'}</b></div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function vault(){
  $("vaultout").innerHTML='<span class="spin">opening the vault&hellip;</span>';
  const r=await post("/api/unified/vault",{});
  if(r.error){$("vaultout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  let q=(r.query||[]).map(x=>`${x.name} (${x.sim})`).join(", ");
  $("vaultout").innerHTML=
    `<div>${r.count} images grouped into <b style="color:var(--teal2)">${r.n_clusters}</b> perceptual clusters `+
    `(biggest holds ${r.biggest_cluster})</div>`+
    `<div style="margin-top:4px">query-by-example, closest matches: <span style="color:var(--amber)">${q}</span></div>`+
    (r.rows&&r.rows.length?`<div style="margin-top:6px" class="muted">smallest encoders: `+
      r.rows.slice(0,3).map(x=>`${x.method} ${x.bytes.toLocaleString()}b`).join(" &middot; ")+`</div>`:"")+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function factorize(){
  $("fzout").innerHTML='<span class="spin">searching in superposition&hellip;</span>';
  const r=await post("/api/unified/factorize",{seed:Math.floor(Math.random()*10000)});
  if(r.error){$("fzout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const match=JSON.stringify(r.true)===JSON.stringify(r.recovered);
  $("fzout").innerHTML=
    `<div>bound factors (hidden from the solver): <b>[${r.true.join(", ")}]</b></div>`+
    `<div style="margin-top:4px">resonator recovered: <b style="color:${match?'var(--teal2)':'var(--coral)'}">[${r.recovered.join(", ")}]</b> ${match?'&#10003; exact':'&#10007;'}</div>`+
    `<div style="margin-top:6px" class="muted">searched a space of <b style="color:var(--amber)">${r.search_space.toLocaleString()}</b> combinations in ${r.restarts} restart(s) &mdash; never enumerated them</div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function composeScene(){
  $("cmpout").innerHTML='<span class="spin">composing &amp; verifying&hellip;</span>';
  const n=parseInt($("cmpN").value||"2");
  const r=await post("/api/unified/compose",{objects:n,seed:Math.floor(Math.random()*10000)});
  if(r.error){$("cmpout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const fmt=a=>a.map(t=>`${t[0]} ${t[1]} (${t[2]})`).join(" + ");
  const frames=r.anim_frames.map(u=>`<img src="${u}" style="width:54px;height:54px;border-radius:6px;margin-right:4px;image-rendering:pixelated">`).join("");
  $("cmpout").innerHTML=
    `<div><img src="${r.image}" style="width:160px;height:160px;border-radius:8px;image-rendering:pixelated"></div>`+
    `<div style="margin-top:6px">composed (never stored): <b>${fmt(r.composed)}</b></div>`+
    `<div style="margin-top:3px">factored back: <b style="color:${r.exact?'var(--teal2)':'var(--coral)'}">${fmt(r.recovered)}</b> ${r.exact?'&#10003; exact round-trip':'&#10007;'}</div>`+
    `<div style="margin-top:3px" class="muted">rendered pixels auto-tag back: shape ${r.render_shape_ok?'&#10003;':'&#10007;'}, colour ${r.render_colour_ok?'&#10003;':'&#10007;'}</div>`+
    `<div style="margin-top:8px"><span class="muted">colour-sweep animation (${Math.round(r.anim_faithful*100)}% of frames on-target):</span><br>${frames}</div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function nestedScene(){
  $("nstout").innerHTML='<span class="spin">composing nested &amp; verifying&hellip;</span>';
  const g=parseInt($("nstG").value||"2"), p=parseInt($("nstP").value||"2");
  const r=await post("/api/unified/nested",{groups:g,per_group:p,seed:Math.floor(Math.random()*10000)});
  if(r.error){$("nstout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const fmt=a=>a.map(t=>`${t[0]} ${t[1]} (${t[2]})`).join(" + ");
  let html=`<div>recovered <b style="color:${r.exact_groups==r.total_groups?'var(--teal2)':'var(--amber)'}">${r.exact_groups}/${r.total_groups}</b> sub-scenes exactly (same machinery, one level up)</div>`;
  for(const grp of r.groups){
    html+=`<div style="margin-top:8px;display:flex;gap:10px;align-items:flex-start">`+
      `<img src="${grp.image}" style="width:80px;height:80px;border-radius:8px;image-rendering:pixelated">`+
      `<div><div class="muted">${grp.name}</div>`+
      `<div>built: <b>${fmt(grp.composed)}</b></div>`+
      `<div>factored back: <b style="color:${grp.exact?'var(--teal2)':'var(--coral)'}">${fmt(grp.recovered)}</b> ${grp.exact?'&#10003;':'&#10007;'}</div></div></div>`;
  }
  html+=`<div class="muted" style="margin-top:6px">${r.note}.</div>`;
  $("nstout").innerHTML=html;
}
async function morphScene(){
  $("mphout").innerHTML='<span class="spin">morphing&hellip;</span>';
  const r=await post("/api/unified/morph",{seed:Math.floor(Math.random()*10000)});
  if(r.error){$("mphout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const frames=r.morph_frames.map(u=>`<img src="${u}" style="width:48px;height:48px;border-radius:6px;margin-right:3px;image-rendering:pixelated">`).join("");
  $("mphout").innerHTML=
    `<div><b>${r.a}</b> &rarr; <b>${r.b}</b></div>`+
    `<div style="margin-top:6px">${frames}</div>`+
    `<div style="margin-top:10px;display:flex;gap:18px;align-items:center">`+
      `<div style="text-align:center"><img src="${r.morph_mid}" style="width:80px;height:80px;border-radius:8px;image-rendering:pixelated"><br><span class="muted">coeff morph midpoint<br>ghost dist <b style="color:var(--teal2)">${r.ghost_morph}</b></span></div>`+
      `<div style="text-align:center"><img src="${r.crossfade_mid}" style="width:80px;height:80px;border-radius:8px;image-rendering:pixelated"><br><span class="muted">crossfade midpoint<br>ghost dist <b style="color:var(--coral)">${r.ghost_crossfade}</b> (the double-exposure)</span></div>`+
    `</div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function nucleusGen(){
  $("nucout").innerHTML='<span class="spin">generating&hellip;</span>';
  const seed=$("nucSeed").value||"the ";
  const top_p=parseFloat($("nucP").value||"0.85");
  const r=await post("/api/unified/nucleus",{seed:seed,top_p:top_p,length:300});
  if(r.error){$("nucout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  let rw="";
  if(r.nucleus_realword!==undefined){
    rw=`<div style="margin-top:6px">real-word fraction: nucleus <b style="color:var(--teal2)">${r.nucleus_realword}</b> vs temperature <b style="color:var(--coral)">${r.temperature_realword}</b> `+
       `<span class="muted">(distinct-4gram ${r.nucleus_distinct4} vs ${r.temperature_distinct4}: more coherent for a little less variety)</span></div>`;
  }
  $("nucout").innerHTML=
    `<div><span class="muted">nucleus (top-p):</span><br><span style="color:var(--amber)">${r.nucleus}</span></div>`+
    `<div style="margin-top:8px"><span class="muted">plain temperature:</span><br>${r.temperature}</div>`+
    rw+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function persistMind(){
  $("prsout").innerHTML='<span class="spin">saving &amp; reloading&hellip;</span>';
  const r=await post("/api/unified/persist",{quant:$("prsQ").value});
  if(r.error){$("prsout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const shrinkTxt = (r.quant && r.quant!=="float32" && r.shrink>1.01)
    ? ` <span class="muted">(${r.quant}: ${r.shrink}&times; smaller than float32's ${r.float32_bytes.toLocaleString()} bytes)</span>` : "";
  $("prsout").innerHTML=
    `<div>saved the whole learned memory &mdash; <b>${r.words.toLocaleString()}</b> word-vectors and `+
    `<b>${r.prototypes}</b> prototypes (${r.labels.join(", ")}) &mdash; to a version-stamped .npz `+
    `(<b style="color:var(--amber)">${r.bytes.toLocaleString()}</b> bytes)${shrinkTxt}, reloaded, verified</div>`+
    `<div style="margin-top:4px">classifications identical after reload: <b style="color:${r.same_classifications?'var(--teal2)':'var(--coral)'}">${r.same_classifications?'&#10003; yes':'&#10007; no'}</b></div>`+
    (r.probe?`<div style="margin-top:4px">nearest neighbours of <b>${r.probe}</b> &mdash; before: <span class="muted">${r.neighbours_before.join(", ")}</span><br>&nbsp;&nbsp;after reload: <span style="color:${r.same_neighbours?'var(--teal2)':'var(--coral)'}">${r.neighbours_after.join(", ")}</span> ${r.same_neighbours?'&#10003;':'&#10007;'}</div>`:"")+
    `<div style="margin-top:4px" class="muted">version guard refuses a mismatched format: ${r.version_guard_works?'&#10003;':'&#10007;'}</div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function discover(){
  $("scout").innerHTML='<span class="spin">discovering units&hellip;</span>';
  const r=await post("/api/unified/discover",{});
  if(r.error){$("scout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("scout").innerHTML=
    `<div style="margin-bottom:6px">word boundaries recovered from spaceless text: F1 <b style="color:var(--teal2)">${r.f1}</b> `+
    `(precision ${r.precision}, recall ${r.recall}) vs <b style="color:var(--coral)">${r.random_f1}</b> for a random cut</div>`+
    `<div style="margin:6px 0"><span class="muted">discovered units:</span><br><span style="color:var(--amber)">${r.sample[0]}</span></div>`+
    `<div style="margin-top:6px">better structure &rarr; better compression: discovered chunks <b style="color:var(--teal2)">${r.chunk_bits}</b> bits/char vs single characters <b>${r.symbol_bits}</b> bits/char</div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function deliberateQuery(){
  const q=($("dq").value||"").trim();
  if(!q){$("dqout").innerHTML='<span class="muted">type a query</span>';return;}
  $("dqout").innerHTML='<span class="spin">thinking&hellip;</span>';
  const r=await post("/api/unified/deliberate",{query:q});
  if(r.error){$("dqout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const best=Math.max(...r.trace.map(t=>t.quality));
  const rows=r.trace.map((t,i)=>{
    const isbest=Math.abs(t.quality-best)<1e-6;
    return `<div style="margin:2px 0;font-size:12px">`+
      `<span class="muted">draft ${i+1}${i===0?' (greedy)':''}:</span> `+
      `<b style="color:${isbest?'var(--teal2)':'#9fb0c8'}">${t.quality.toFixed(3)}</b> `+
      `<span class="muted">${t.draft}&hellip;</span>${isbest?' &larr; kept':''}</div>`;
  }).join("");
  $("dqout").innerHTML=
    `<div style="margin-bottom:6px">thought for <b style="color:var(--amber)">${r.iterations}</b> `+
    `iteration${r.iterations>1?'s':''} &middot; quality ${r.quality} (relevance ${r.relevance}, structure ${r.structure})</div>`+
    `<div style="margin:8px 0"><span style="color:var(--amber)">${r.response}</span></div>`+
    `<div style="border-top:1px solid #1e2c45;padding-top:8px"><div class="muted" style="font-size:13px;margin-bottom:4px">the deliberation (inner drafts):</div>${rows}</div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function respondQuery(){
  const q=($("rq").value||"").trim();
  if(!q){$("rqout").innerHTML='<span class="muted">type a query</span>';return;}
  $("rqout").innerHTML='<span class="spin">generating a response&hellip;</span>';
  const r=await post("/api/unified/respond",{query:q});
  if(r.error){$("rqout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const s=r.steered, u=r.unsteered;
  const better=s.relevance>=u.relevance;
  $("rqout").innerHTML=
    `<div style="margin:6px 0"><span class="muted">steered toward the query</span> `+
    `<span class="muted">(relevance <b style="color:${better?'var(--teal2)':'#9fb0c8'}">${s.relevance}</b>, structure ${s.structure})</span><br>`+
    `<span style="color:var(--amber)">${s.text}</span></div>`+
    `<div style="margin:8px 0;border-top:1px solid #1e2c45;padding-top:8px"><span class="muted">unsteered baseline `+
    `(relevance ${u.relevance}, structure ${u.structure})</span><br><span class="muted">${u.text}</span></div>`+
    `<div class="muted" style="margin-top:6px">${r.note}.</div>`;
}
async function predictive(){
  $("pout").innerHTML='<span class="spin">living the sequence&hellip;</span>';
  const r=await post("/api/unified/predictive",{});
  if(r.error){$("pout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const maxacc=Math.max(...r.curve.map(c=>c.accuracy), 0.01);
  const rows=r.curve.map(c=>{
    const w=Math.round(c.accuracy/maxacc*180);
    return `<div style="display:flex;align-items:center;gap:8px;margin:2px 0">`+
      `<span style="width:64px;font-family:monospace;font-size:12px">${c.tokens} tok</span>`+
      `<span style="display:inline-block;height:12px;width:${Math.max(2,w)}px;background:var(--teal2);border-radius:3px"></span>`+
      `<span class="muted" style="font-size:12px">${(c.accuracy*100).toFixed(1)}%  ·  ${c.entries} entries</span></div>`;
  }).join("");
  const fe=r.free_energy_start>r.free_energy_end;
  $("pout").innerHTML=
    `<div class="muted" style="font-size:13px;margin-bottom:6px">accuracy on a held-out probe as it sees more (learning curve):</div>${rows}`+
    `<div style="margin-top:8px">free energy (prediction error): <b>${r.free_energy_start}</b> &rarr; <b style="color:${fe?'var(--teal2)':'var(--coral)'}">${r.free_energy_end}</b> ${fe?'(falling &mdash; learning to anticipate)':'(not falling)'}</div>`+
    `<div style="margin-top:4px">generalisation to <b>unseen</b> contexts: <b style="color:var(--amber)">${r.generalization.correct}/${r.generalization.total}</b> (${r.generalization.pct}%) &mdash; exact lookup scores these blind</div>`+
    `<div style="margin-top:8px"><span class="muted">generate by anticipation from "${r.seed}":</span><br>${r.seed} <span style="color:var(--amber)">${r.sample}</span></div>`+
    (r.proof?`<div style="margin-top:12px;border-top:1px solid #1e2c45;padding-top:10px">`+
      `<div class="muted" style="font-size:13px;margin-bottom:6px">proof of structure (real text scores ~${r.proof.real_score}, threshold ${r.proof.threshold}; lower = more like salad):</div>`+
      `<div>greedy decoding &rarr; structure <b style="color:var(--coral)">${r.proof.greedy_score}</b>: <span class="muted">${r.proof.greedy_sample}</span></div>`+
      `<div style="margin-top:4px">steered by the verifier &rarr; structure <b style="color:var(--teal2)">${r.proof.steered_score}</b>: <span class="muted">${r.proof.steered_sample}</span></div>`+
      `<div class="muted" style="margin-top:6px">greedy collapses into a loop (a locally-coherent salad single-step checks would rate highly); steering by trajectory structure &mdash; projecting each word onto the running context &mdash; defends coherence.</div>`+
      (r.proof.compress_real!==undefined?`<div style="margin-top:8px">better structure &rarr; better compression: real text compresses to <b style="color:var(--teal2)">${r.proof.compress_real}</b> of baseline, the same words shuffled only <b style="color:var(--coral)">${r.proof.compress_shuffled}</b> &mdash; a predictor is a compressor.</div>`:"")+
      `</div>`:"")+
    `<div class="muted" style="margin-top:8px">${r.note}.</div>`;
}
async function topicPull(){
  const seed=($("tp_seed").value||"the").trim();
  $("tpout").innerHTML='<span class="spin">sweeping the topic pull&hellip;</span>';
  const r=await post("/api/unified/topic_pull",{seed});
  if(r.error){$("tpout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const base=r.rows[0];
  const rows=r.rows.map(x=>{
    const divCollapse=x.diversity<base.diversity-0.2;
    return `<tr><td style="padding:2px 12px 2px 0;font-family:monospace">${x.topic_weight.toFixed(0)}</td>`+
      `<td style="padding:2px 12px 2px 0">${x.coherence.toFixed(3)}</td>`+
      `<td style="padding:2px 12px 2px 0">${x.transition_validity.toFixed(3)}</td>`+
      `<td style="padding:2px 0;color:${divCollapse?'var(--coral)':'#9fb0c8'}">${x.diversity.toFixed(3)}${divCollapse?' &larr; collapsed':''}</td></tr>`;
  }).join("");
  $("tpout").innerHTML=
    `<table style="font-size:13px"><tr class="muted"><th align="left">topic_weight</th><th align="left">coherence</th><th align="left">transition&nbsp;valid</th><th align="left">diversity</th></tr>${rows}</table>`+
    `<div style="margin-top:8px"><span class="muted">baseline (pull 0):</span> ${r.sample_baseline}</div>`+
    `<div style="margin-top:4px"><span class="muted">heavy pull (16):</span> <span style="color:var(--coral)">${r.sample_hot}</span></div>`+
    `<div class="muted" style="margin-top:8px">The coherence number can <em>rise</em> with heavy pull only because the topic vector collapses onto a few frequent words &mdash; diversity craters and the text stops being language. Re-ranking can't add structure the n-gram never proposed; the missing piece is a high-capacity learned next-word function, not this lever.</div>`;
}
async function resolution(){
  const t=($("cq").value||"").trim();
  if(!t){$("cout").innerHTML='<span class="muted">type some text</span>';return;}
  const r=await post("/api/unified/resolution",{text:t});
  if(r.error){$("cout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  const final=r.profile[r.profile.length-1][1];
  const ladder=r.profile.map(([k,lab,sc])=>{
    const hit=lab===final;
    return `<div style="font-family:monospace">${String(k).padStart(5)} dim &rarr; `+
           `<b style="color:${hit?'var(--amber)':'#7a8aa5'}">${lab}</b> `+
           `<span class="muted">(${sc.toFixed(2)})</span>${hit?'':' <span class="muted">— not yet settled</span>'}</div>`;
  }).join("");
  const robust=r.stable_from<=r.full_dim/2;
  $("cout").innerHTML=`<div class="muted">winner per truncation (coarse &rarr; fine):</div>${ladder}`+
    `<div style="margin-top:6px">answer settles at <b style="color:var(--amber)">${r.stable_from}</b> of ${r.full_dim} dims `+
    `&mdash; ${robust?'robust to heavy truncation (cheap to find coarse-to-fine)':'a close call, needs near-full width'}</div>`;
}
async function curriculum(){
  const w=($("cur_word").value||"").trim().toLowerCase();
  if(!w){$("curout").innerHTML='<span class="muted">enter a word</span>';return;}
  const r=await post("/api/unified/curriculum",{word:w});
  if(r.error){$("curout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  if(r.note){$("curout").innerHTML=`<span class="muted">${r.note}</span>`;return;}
  const mean=(r.meaning||[]).map(m=>`${m.word} <span class="muted">(${m.sim.toFixed(2)})</span>`).join(", ")
             || '<span class="muted">no meaning learned for this word</span>';
  const chain=(r.is_a_chain||[]).join(" &rarr; ");
  const tp=r.throughput;
  const tpnote=` <span class="muted">(throughput ${tp.toFixed(2)} &mdash; the is_a ray's accumulated confidence; it fades with depth)</span>`;
  $("curout").innerHTML=
    `<div><span class="muted">dictionary &mdash; meaning neighbours:</span> ${mean}</div>`+
    `<div style="margin-top:8px"><span class="muted">encyclopedia &mdash; is_a chain:</span> `+
    `<b style="color:var(--amber)">${chain}</b>${chain.includes("&rarr;")?tpnote:""}</div>`;
}
init();
</script>
</body></html>
"""

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
