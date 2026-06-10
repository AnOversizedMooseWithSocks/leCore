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

import io
from collections import defaultdict

import numpy as np
from flask import Flask, request, jsonify, render_template_string

from holographic_unified import UnifiedMind
from holographic_text import STOPWORDS

app = Flask(__name__)

# one trained mind lives here between requests (single-process dev server)
STATE = {"mind": None, "dataset": None, "labels": [], "test": [], "raw_len": 0}


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
            raw.append(" ".join(toks))
    return items, " ".join(raw), "Reuters financial newswire -- 10 confusable categories (grain/crude/money-fx share vocabulary)"


def load_brown():
    from nltk.corpus import brown
    cats = ["news", "romance", "science_fiction", "government", "hobbies"]
    items, raw = [], []
    for c in cats:
        words = [w.lower() for w in brown.words(categories=c) if w.isalpha()]
        for k in range(0, min(len(words), 18000) - 300, 300):
            chunk = words[k:k + 300]
            items.append((_content(chunk), c))
            raw.append(" ".join(chunk))
    return items, " ".join(raw), "Brown corpus -- five prose genres, in 300-word chunks"


def load_gutenberg():
    from nltk.corpus import gutenberg
    books = {"austen-emma.txt": "Austen", "carroll-alice.txt": "Carroll",
             "shakespeare-hamlet.txt": "Shakespeare", "melville-moby_dick.txt": "Melville",
             "chesterton-brown.txt": "Chesterton"}
    items, raw = [], []
    for fid, author in books.items():
        if fid not in gutenberg.fileids():
            continue
        words = [w.lower() for w in gutenberg.words(fid) if w.isalpha()][:9000]
        for k in range(0, len(words) - 200, 200):
            chunk = words[k:k + 200]
            items.append((_content(chunk), author))
            raw.append(" ".join(chunk))
    return items, " ".join(raw), "Project Gutenberg -- classify the author, generate in their style"


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


DATASETS = {
    "reuters": ("Reuters categories", ["reuters"], load_reuters),
    "brown": ("Brown genres", ["brown"], load_brown),
    "gutenberg": ("Gutenberg authors", ["gutenberg"], load_gutenberg),
    "europarl": ("Europarl languages", ["europarl_raw"], load_europarl),
}


# ---------------------------------------------------------------------------
# training one UnifiedMind on a pulled corpus
# ---------------------------------------------------------------------------

def build(dataset_id):
    name, pkgs, loader = DATASETS[dataset_id]
    for p in pkgs:
        _ensure(p)
    items, raw, desc = loader()

    # split each label 70/30 for an honest held-out accuracy number
    by = defaultdict(list)
    for toks, lab in items:
        by[lab].append(toks)
    rng = np.random.default_rng(0)
    train, test = [], []
    for lab, docs in by.items():
        docs = list(docs); rng.shuffle(docs)
        cut = int(len(docs) * 0.7)
        train += [(d, lab) for d in docs[:cut]]
        test += [(d, lab) for d in docs[cut:]]
    rng.shuffle(train)

    mind = UnifiedMind(dim=1024, seed=0, text_window=3)
    mind.read([toks for toks, _ in train])          # learn word co-occurrence
    for toks, lab in train:
        mind.learn(toks, lab, "text")               # classification + recall, one memory
    mind.maintain_now()
    mind.learn_sequence(raw[:160000], n=6)          # generation, same space

    acc = sum(mind.classify(toks, "text")[0] == lab for toks, lab in test) / max(1, len(test))
    STATE.update({"mind": mind, "dataset": name, "labels": sorted(by),
                  "test": test, "raw_len": len(raw), "desc": desc})
    return {
        "ok": True, "dataset": name, "desc": desc,
        "labels": sorted(by),
        "counts": mind.memory.live.counts_by_label(),
        "prototypes": mind.memory.live.size(),
        "trained": len(train), "held_out": len(test),
        "accuracy": round(100 * acc),
        "gen_chars": min(len(raw), 160000),
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
        out.append({"id": did, "name": name, "available": have_nltk})
    return jsonify({"datasets": out, "nltk": have_nltk})


@app.route("/api/unified/load", methods=["POST"])
def load():
    did = request.json.get("id")
    if did not in DATASETS:
        return jsonify({"ok": False, "error": "unknown dataset"})
    try:
        return jsonify(build(did))
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e} "
                        "(if this is a missing corpus, a network connection is needed "
                        "the first time to pull it from GitHub)"})


def _need_mind():
    return STATE["mind"] is None


@app.route("/api/unified/classify", methods=["POST"])
def classify():
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    text = (request.json.get("text") or "").lower()
    toks = _content(text.split())
    if not toks:
        return jsonify({"error": "type some words the model might know"})
    mind = STATE["mind"]
    label, score = mind.classify(toks, "text")
    (rlabel, _), rscore = mind.recall(toks, "text")
    return jsonify({"label": label, "score": round(float(score), 3),
                    "recall": {"label": rlabel, "score": round(float(rscore), 3)}})


@app.route("/api/unified/organize", methods=["POST"])
def organize():
    if _need_mind():
        return jsonify({"error": "load a dataset first"})
    mind = STATE["mind"]
    before = mind.memory.live.counts_by_label()
    choice = mind.maintain_now()
    after = mind.memory.live.counts_by_label()
    return jsonify({"before": before, "after": after,
                    "choice": (choice[0] if choice else "keep"),
                    "note": "each label may hold several sub-prototypes when the memory "
                            "found it multi-modal; one each means it stayed simple."})


@app.route("/api/unified/generate", methods=["POST"])
def generate():
    if _need_mind() or STATE["mind"]._gen is None:
        return jsonify({"error": "load a dataset first"})
    j = request.json
    seed = (j.get("seed") or "the ").lower()
    length = int(j.get("length", 200))
    temp = float(j.get("temperature", 0.45))
    text = STATE["mind"].generate(seed, max(20, min(length, 600)), max(0.1, min(temp, 1.2)))
    return jsonify({"text": text})


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
</style></head><body>
<header>
  <h1>UnifiedMind &mdash; one model, one space</h1>
  <div class="sub">Pull a real corpus, train one mind, then classify / recall / organize / generate against it.</div>
</header>
<main>

  <div class="card">
    <h2>1 &middot; pull + train</h2>
    <div class="row">
      <select id="ds"></select>
      <button onclick="load()">Pull &amp; train</button>
      <span id="loadmsg" class="muted"></span>
    </div>
    <div id="trained" class="out"></div>
  </div>

  <div id="ops" class="disabled">
  <div class="card">
    <h2>2 &middot; classify &amp; recall</h2>
    <textarea id="cq" placeholder="type a sentence in the style of the corpus..."></textarea>
    <div class="row" style="margin-top:8px"><button onclick="classify()">Classify</button>
      <button class="ghost" onclick="recall()">Recall nearest</button></div>
    <div id="cout" class="out"></div>
  </div>

  <div class="card">
    <h2>3 &middot; organize</h2>
    <div class="muted">How the one memory split each label into sub-prototypes (multi-modal labels get more).</div>
    <div class="row" style="margin-top:8px"><button onclick="organize()">Show &amp; reorganize</button></div>
    <div id="oout" class="out"></div>
  </div>

  <div class="card">
    <h2>4 &middot; generate</h2>
    <div class="row">
      <input id="seed" value="the " style="width:160px" placeholder="seed text">
      <label class="muted">length <input id="len" type="number" value="220" style="width:80px"></label>
      <label class="muted">temp <input id="temp" type="number" step="0.05" value="0.45" style="width:80px"></label>
      <button onclick="generate()">Generate</button>
    </div>
    <div id="gout" class="out"></div>
  </div>
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
}
async function load(){
  $("loadmsg").textContent="pulling + training\u2026"; $("trained").innerHTML="";
  const r=await post("/api/unified/load",{id:$("ds").value});
  if(!r.ok){$("loadmsg").textContent="";$("trained").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("loadmsg").textContent="";
  const pills=r.labels.map(l=>`<span class="pill">${l}: ${r.counts[l]||0}</span>`).join("");
  $("trained").innerHTML=
    `<div>${r.desc}</div>
     <div style="margin-top:8px">held-out accuracy <span class="big">${r.accuracy}%</span>
        <span class="muted">&nbsp; ${r.trained} trained / ${r.held_out} held out &middot;
        ${r.prototypes} prototypes &middot; ${r.gen_chars.toLocaleString()} chars for generation</span></div>
     <div style="margin-top:8px">${pills}</div>`;
  $("ops").classList.remove("disabled");
}
async function classify(){
  const r=await post("/api/unified/classify",{text:$("cq").value});
  if(r.error){$("cout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("cout").innerHTML=`classified as <b style="color:var(--teal2)">${r.label}</b> (cos ${r.score})
     <br><span class="muted">nearest stored item is a <b>${r.recall.label}</b> (cos ${r.recall.score})</span>`;
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
  $("oout").innerHTML=`reorganize decided: <b>${r.choice}</b><div style="margin-top:8px">${pills}</div>
     <div class="muted" style="margin-top:6px">${r.note}</div>`;
}
async function generate(){
  $("gout").innerHTML='<span class="muted">generating\u2026</span>';
  const r=await post("/api/unified/generate",{seed:$("seed").value,length:+$("len").value,temperature:+$("temp").value});
  if(r.error){$("gout").innerHTML=`<span class="muted">${r.error}</span>`;return;}
  $("gout").innerHTML=`<span style="color:var(--amber)">${r.text}</span>`;
}
init();
</script>
</body></html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
