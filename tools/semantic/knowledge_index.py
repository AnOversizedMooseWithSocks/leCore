#!/usr/bin/env python3
"""knowledge_index.py -- ONE semantic index over leCore's whole knowledge base: 400 module docstrings
AND ~1,250 markdown sections (NOTES_concepts, ABLATIONS, THEORY, ISA, REFERENCE, the guides).

WHY THE MD FILES MATTER MORE THAN THE CODE:
  The code says what a module DOES. The markdown says what was TRIED, what FAILED, and WHY -- the
  kept negatives, the ablation results, the ISA contract, the hard-won lessons. NOTES_concepts.md
  alone is 1.7 MB of append-only research log. That is the institutional memory, and right now it is
  only reachable by grep.

THE MEASURED PROBLEM (keyword baseline on the live tree, before any embedding):
  * module routing, 12 natural asks:        keyword top-1 = 1/12  (7 asks share ZERO words with the answer)
  * "has this been tried?", 6 asks:         keyword top-1 = 2/6   (and it matches on stopwords:
    the two "hits" land on chunks containing the needle by coincidence of common words like "backend")
  Probe-first is constitutional -- "audit the live code before claiming something is missing." Every
  duplicate-module proposal in this project's history was a failed lookup. Grep cannot answer
  "did we already try sharpening splats afterwards?" because the answer is filed under 'splatsharpen'
  and 'kept negative', words the asker never used. THIS is the unexpected problem embeddings solve.

WHAT IT BUILDS
  One index, three document kinds, all at Matryoshka tiers:
      code : one entry per module (name + docstring)
      docs : one entry per markdown heading-section
      note : sections of NOTES_concepts.md are tagged separately (they hold the negatives)

WHAT IT MEASURES
  [1] coverage   : UNK rate + pieces/term over the codebase's real vocabulary
  [2] index      : entries and MB at 768/128/64d
  [3] routing    : 12 asks -> module        (bar: beat keyword's 1/12)
  [4] negatives  : 6 asks -> the right lesson (bar: beat keyword's 0/6)  <- the constitutional test
  [5] neighbors  : do families cluster? sanity that this is geometry, not luck

USAGE (from the repo root):
  python3 knowledge_index.py model.safetensors vocab.txt --repo .
  python3 knowledge_index.py model.safetensors vocab.txt --repo . --ask "did we ever try a C backend"
  python3 knowledge_index.py model.safetensors vocab.txt --repo . --save index_64d.npz

CHUNKING NOTE: nomic handles 8192 tokens, so a whole markdown section fits in one embedding -- no
sliding window needed. Sections are split on markdown headings (the author's own boundaries), which
is the honest chunking: it never invents a boundary the writer didn't intend.
"""
import sys, os, re, ast, argparse, collections, hashlib, json
# PERF: BLAS threads must be set BEFORE numpy imports, or the setting is ignored.
if 'OMP_NUM_THREADS' not in os.environ:
    _n = str(os.cpu_count() or 1)
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(_v, _n)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nomic_forward import WordPiece, Nomic

# ---- test suite A: find the module (asks deliberately avoid module vocabulary)
# MULTI-LABEL. The first run used one label per ask and scored 2/12 -- but the "misses" were
# defensible (coldstore's own docstring: "shrink INACTIVE data ... big arrays"). With 400
# near-synonymous modules, top-1-against-one-label measures MY labelling, not the index.
# So: an accept SET per ask, and we additionally report the RANK of the best accepted answer.
ASKS_MODULE = [
    ("make my picture less grainy", ["holographic_denoise", "holographic_sharpen"]),
    ("figure out the shortest way through a maze", ["holographic_flow", "holographic_slime"]),
    ("how sure are we this match isn't luck", ["holographic_honesty", "holographic_measure",
                                              "holographic_conformal"]),
    ("squish a big array down for storage", ["holographic_ratedistortion", "holographic_coldstore",
                                             "holographic_compress"]),
    ("teach the creature to want food", ["holographic_creature", "holographic_creature_mind",
                                         "holographic_agent"]),
    ("what does this scene look like from here", ["holographic_render", "holographic_scene_render",
                                                  "holographic_raymarch"]),
    ("break a shape into simpler pieces", ["holographic_resonator", "holographic_peel",
                                           "holographic_sbc"]),
    ("remember a picture and get it back later", ["holographic_archive", "holographic_objectarchive",
                                                  "holographic_image_vault", "holographic_image"]),
    ("guess where the ball goes next", ["holographic_dynamics", "holographic_predictive",
                                        "holographic_chaos"]),
    ("smooth out the bumpy surface", ["holographic_meshsmooth", "holographic_graphsignal",
                                     "holographic_meshcurvature"]),
    ("find things near this point quickly", ["holographic_tree", "holographic_pivot",
                                             "holographic_octree", "holographic_spatial"]),
    ("water flowing and swirling", ["holographic_fluid", "holographic_fields"]),
]

# ---- test suite B: THE CONSTITUTIONAL TEST -- "has this been tried?" Answers live in the md corpus.
# `expect` is a substring that MUST appear in the retrieved chunk's text for a hit. Grounded in the
# real documented negatives, not invented: splatsharpen, the C-backend rejection, ldexplore/lookahead,
# the hash()/determinism rule, substeps, and the bind_batch tie-break lesson.
ASKS_NEGATIVE = [
    ("did we already try making the splats sharper afterwards", "splatsharpen"),
    ("is there a reason we don't use a C backend", "backend"),
    ("what happened when we tried looking ahead further in navigation", "lookahead"),
    ("why can't we just use python's built in hash", "hashlib"),
    ("was there a problem with vectorized binding changing results", "bind_batch"),
    ("does jittering the splats help", "jittersplat"),
]

PROBES = ["removing noise from a signal", "gaussian splats and point clouds",
          "statistical confidence and false alarms", "editing polygon meshes",
          "things we measured that did not work"]

MAX_CHARS = 280   # set from --max-chars in main()

STOP = set('the a an of to in for and or is are with by on as it its that this be not from we you '
           'was were has have do does did can just why what when how there more'.split())


def words(s):
    return set(w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP)


def _module_families(repo):
    """module NAME (e.g. 'holographic_denoise') -> family (its holographic/<family>/ directory). The family
    is the natural GROUP level for hierarchical routing: route a query to its family first (an 11-way
    choice), then rank modules WITHIN that family. This turns a flat 505-way search into group-then-leaf,
    the structure hierarchical_recall measured at 100% vs 18.3% for flat_recall (holographic_hierarchy)."""
    import os
    fam = {}
    for root, _, files in os.walk(repo):
        if 'node_modules' in root:
            continue
        # family = the directory directly under 'holographic/'
        parts = root.replace('\\', '/').split('/')
        family = parts[parts.index('holographic') + 1] if 'holographic' in parts and \
            parts.index('holographic') + 1 < len(parts) else 'root'
        for f in files:
            if f.startswith('holographic_') and f.endswith('.py'):
                fam[f[:-3]] = family
    return fam


def _doc_chunks(body, max_chunks=6, min_len=24):
    """Split a docstring into up to max_chunks sentence-ish pieces for MULTI-VECTOR (late-interaction)
    routing. Deterministic: split on sentence enders and the '--' section marker, drop tiny fragments,
    keep the whole thing as one chunk if it does not split. Each chunk is embedded separately; a module is
    then scored by the MAX cosine over its chunks -- so a single strongly-relevant sentence surfaces the
    module even when the averaged vector buries it. The demux mask: match the part, not the mean."""
    import re
    parts = re.split(r'(?<=[.!?])\s+|\s+--+\s+|\s+WHY\b', body)
    chunks = [c.strip() for c in parts if len(c.strip()) >= min_len]
    if not chunks:
        chunks = [body.strip()] if body.strip() else []
    # always include the FULL body as one chunk too (the mean-equivalent), so multivec can never do worse
    # than having the whole-text vector available to max over.
    full = body.strip()
    if full and full not in chunks:
        chunks = [full] + chunks
    return chunks[:max_chunks]


def _module_iogroups(repo):
    """module NAME -> its io-kind GROUP (a produces kind, else a consumes kind), via the module= link. This
    is the SEMANTIC coarse level for hierarchical routing -- 'what datatype does this act on?' -- cleaner
    than the directory family (which dumps 149 modules in 'misc'). Modules with no io-kind fall back to a
    'untyped' group. Same join pipeline_map uses, so finishing the io-kind backfill grows BOTH."""
    try:
        import sys, os
        rr = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if rr not in sys.path: sys.path.insert(0, rr)
        from holographic.caching_and_storage.holographic_catalog import default_catalog
    except Exception:
        return {}
    cat = default_catalog()
    out = {}
    for c in cat._by_name.values():
        mod = getattr(c, 'module', None)
        if mod:
            kinds = list(c.produces) or list(c.consumes)
            if kinds:
                out['holographic_' + mod] = kinds[0]          # one group per module (its primary datatype)
    return out


def _alias_enrichment():
    """module-stem -> ' '.join(catalog aliases), joined via the NEW module= link (trustworthy, not fuzzy).
    Empty dict if the catalog can't be imported (keeps this tool runnable stand-alone). WHY: a module's
    aliases are hand-written PARAPHRASES ('clean a noisy signal'); appending them to the routing document
    pulls the embedding toward the phrasings the asks use -- the 'richer target' hypothesis, made testable
    only now that module= gives a reliable code-module -> capability join."""
    import sys, os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)                     # so holographic.* resolves from tools/semantic
    try:
        from holographic.caching_and_storage.holographic_catalog import default_catalog
    except Exception:
        return {}
    cat = default_catalog()
    out = {}
    for c in cat._by_name.values():
        mod = getattr(c, 'module', None)
        if mod:
            out.setdefault(mod, []).extend(c.aliases)
    return {k: ' '.join(v) for k, v in out.items()}


def collect_code(repo, enrich_aliases=False):
    """One entry per module: its module docstring is the authoritative description (per the guide).

    enrich_aliases (default False -> byte-identical old behavior): append the module's catalog aliases
    (joined via the module= link) to its docstring text before it is embedded, so the routing vector is
    pulled toward the paraphrases users type. Tests the 'richer target' hypothesis against bare routing."""
    enr = _alias_enrichment() if enrich_aliases else {}
    out = []
    for root, _, files in os.walk(repo):
        if 'node_modules' in root:
            continue
        for f in files:
            if not (f.startswith('holographic_') and f.endswith('.py')):
                continue
            txt = open(os.path.join(root, f), encoding='utf-8', errors='ignore').read()
            m = re.search(r'"""(.*?)"""', txt, re.S)
            if m:
                stem = f[:-3].replace('holographic_', '')
                doc = ' '.join(m.group(1).split())
                if enr.get(stem):
                    # ADDITIVE enrichment (confound fix): keep the FULL docstring, then append aliases. An
                    # earlier version prepended+truncated, which EVICTED docstring tail to fit aliases and
                    # made the test 'aliases + partial docstring' vs 'full docstring' -- not clean. Here the
                    # docstring is untouched and aliases are pure addition, so the exam measures the aliases.
                    body = doc[:MAX_CHARS] + ' -- ' + enr[stem]
                else:
                    body = doc[:MAX_CHARS]
                out.append(('code', f[:-3], body))
    return out


# Docs REGENERATED from code/catalog on every docs.yml push -- embedding them duplicates content already
# captured by collect_code/collect_terms, AND churns the cache (their hashes change every rebuild). Skip.
_GENERATED_DOCS = {'REFERENCE.md', 'CAPABILITIES.md', 'FACULTY_MAP.md', 'DOC_MAP.md', 'API_QUICKREF.md'}


def collect_md(repo, window=900, stride=650, skip_generated=True):
    """Markdown -> SLIDING WINDOWS inside each heading-section.

    skip_generated (default True): omit docs rebuilt from code (REFERENCE/CAPABILITIES/FACULTY_MAP/
    DOC_MAP/API_QUICKREF). Measured (rev. 47): these were ~40% of md windows and the sole reason the
    embed cache churned on every docs-bot commit -- their text is regenerated, so their hashes move,
    so they re-embed forever while adding nothing collect_code/collect_terms did not already capture.

    WHY (measured): sections have a median length of 2,150 chars (p90 3,750; max 100k). The old code
    truncated each to --max-chars (280) before embedding, so **87% of every section was never seen**.
    The answer to "did we try sharpening splats?" lives past char 280 -- `splatsharpen` appeared in the
    first 280 chars of only 1 of its 4 chunks. That was the kept-negative lookup failure: truncation,
    not semantics.

    Windows overlap (stride < window) so a sentence never falls in a crack. The heading is prepended
    to every window as its parent pointer, which keeps the section's topic in each embedding."""
    out = []
    for root, _, files in os.walk(repo):
        if 'node_modules' in root:
            continue
        for f in files:
            if not f.endswith('.md'):
                continue
            if skip_generated and f in _GENERATED_DOCS:
                continue                                    # rebuilt from code; see _GENERATED_DOCS
            p = os.path.join(root, f)
            txt = open(p, encoding='utf-8', errors='ignore').read()
            for sec in re.split(r'\n(?=#{1,4}\s)', txt):
                w = sec.split()
                if len(w) <= 20:
                    continue
                head = sec.split('\n')[0].strip('# ').strip()[:70]
                kind = 'note' if 'NOTES_concepts' in p else 'docs'
                body = ' '.join(w)
                label = f"{os.path.relpath(p, repo)} :: {head}"
                for start in range(0, max(len(body) - window + stride, 1), stride):
                    piece = body[start:start + window]
                    if len(piece) < 120 and start > 0:
                        break
                    tag = label if start == 0 else f"{label} [+{start}]"
                    out.append((kind, tag, piece))
    return out


def collect_terms(repo):
    parts = collections.Counter()
    def split(s):
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s)
        return [q for q in re.split(r'[_\s]+', s.lower()) if q]
    for root, _, files in os.walk(repo):
        if 'node_modules' in root:
            continue
        for f in files:
            if not f.endswith('.py'):
                continue
            try:
                tree = ast.parse(open(os.path.join(root, f), encoding='utf-8', errors='ignore').read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for q in split(node.name):
                        parts[q] += 1
                    d = ast.get_docstring(node)
                    if d:
                        for w in re.findall(r"[a-zA-Z]{3,}", d.lower()):
                            parts[w] += 1
    return parts


def kw_rank(ask, entries):
    aw = words(ask)
    return sorted(range(len(entries)), key=lambda i: -len(aw & words(entries[i][2])))


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def parent_of(name):
    """A window is labelled 'file :: heading [+offset]'. Its PARENT is the section itself."""
    return re.sub(r'\s*\[\+\d+\]$', '', name)


def collapse_by_parent(order, entries):
    """Windows overlap, so one long section can occupy every slot of a top-5 (measured: 4.2 windows
    per section on average, one section had 570). Keep the best-scoring window per parent section --
    a scoring/display fix, not a model fix."""
    seen = set()
    out = []
    for j in order:
        p = parent_of(entries[j][1])
        if p in seen:
            continue
        seen.add(p)
        out.append(j)
    return out


class AllButTheTop:
    """Mu & Viswanath (2017), 'All-but-the-Top'. Sentence embeddings occupy a narrow cone: on THIS
    corpus the mean cosine between unrelated modules was 0.778, so everything looked similar and
    ranking was dominated by a common direction. Subtract the mean and project out the top-k
    principal directions -> mean cosine 0.015, family separation 0.079 -> 0.335 (measured).

    The basis is fit on the DOCUMENTS and applied unchanged to queries -- fitting per-query would
    leak and would not be a fixed transform."""

    def __init__(self, E, k=1):
        self.mu = E.mean(0)
        X = E - self.mu
        if k:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            self.V = Vt[:k]
        else:
            self.V = None

    def __call__(self, X):
        X = np.atleast_2d(X) - self.mu
        if self.V is not None:
            X = X - (X @ self.V.T) @ self.V
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('vocab')
    ap.add_argument('--repo', default='.')
    ap.add_argument('--ask'); ap.add_argument('--save')
    ap.add_argument('--cache', default='.knowledge_cache.json',
                    help='content-hashed embedding cache; reruns are near-free')
    ap.add_argument('--max-chars', type=int, default=280,
                    help='truncate CODE docstrings. NOTE: this changed 280->600 in the same run that '
                         'introduced md windowing, confounding a routing regression (7/12 -> 5/12). '
                         'Kept at 280 so the two can be A/B-ed independently; the cache makes it cheap.')
    ap.add_argument('--window', type=int, default=900,
                    help='markdown window size in chars (~220 tokens). Sections are windowed, NOT '
                         'truncated: truncation hid 87%% of every section and caused the '
                         'kept-negative lookup failure.')
    ap.add_argument('--stride', type=int, default=650, help='window stride; overlap = window - stride')
    ap.add_argument('--no-cache', action='store_true')
    # --exam turns the routing suite into a CI GATE: exit 1 if the @768d numbers miss the bars. Bars
    # default to the measured state so a regression (a renamed module, a docstring turned to jargon)
    # fails the build instead of drifting silently. Without --exam the suite just prints, as before.
    ap.add_argument('--exam', action='store_true', help='exit 1 if the routing suite misses the bars')
    # --no-md: embed ONLY code docstrings + terms, skipping the ~5k markdown windows. The routing exam
    # ranks over 'code' entries only, so md is dead weight in the CI routing job -- and embedding it is
    # the hours-long step that times out. With --no-md the committed routing seed already holds every
    # entry this run needs, so a cold CI run embeds ~nothing. Full local builds omit --no-md to keep md.
    ap.add_argument('--no-md', action='store_true', help='skip markdown windows (routing-only; CI uses this)')
    ap.add_argument('--require-top5', type=int, default=8)
    ap.add_argument('--require-median', type=float, default=2)
    ap.add_argument('--require-fused-top1', type=int, default=None,
                    help='GATE the fused (dense + workflow bones) route: fail if top-1 at the champion row '
                         '(beta=0, gamma=0.5, 128d) drops below N. Needs --structural. The measured bar is 7 '
                         '(strict Pareto over flat 6/12). Default None = no fused gate.')
    ap.add_argument('--abtt-sweep', default='',
                    help="comma-list of k values, e.g. '0,1,2,3,4' -- print the routing exam for each k "
                         "so the best all-but-the-top depth is a MEASUREMENT, not a guess. Reuses the same "
                         "eval as [3]; does not change what ships.")
    ap.add_argument('--hier-by', default='family', choices=('family', 'iokind'),
                    help="grouping for --hier coarse routing: 'family' = the holographic/<dir>/ (free, but "
                         "'misc' is a 149-module catch-all); 'iokind' = the module's datatype group via the "
                         "module= link (semantic, needs io-kind tags -- finishing the pipeline map grows it).")
    ap.add_argument('--hybrid', action='store_true',
                    help="HYBRID routing: fuse the dense (nomic cosine) ranking with a pure-NumPy BM25 lexical "
                         "ranking via Reciprocal Rank Fusion. Recovers LEXICAL misses (query words present in "
                         "the target docstring: surface/ball/shape) that dense buries, while keeping the dense "
                         "HITs. The literature-backed fix for vocabulary mismatch; fits NumPy-only (no SPLADE).")
    ap.add_argument('--structural', action='store_true',
                    help="Add a THIRD ranked list to --hybrid: workflow-graph propagation. Seeds the sparse "
                         "author-stated workflow bones with the DENSE scores and spreads one hop, so a module "
                         "whose COLLABORATORS are hit gets lifted even when the query shares NO words with its "
                         "docstring -- the gap dense and BM25 both structurally cannot close. Swept by gamma.")
    ap.add_argument('--max-chunks', type=int, default=6,
                    help='cap on docstring chunks per module for --multivec. The default 6 was sized for '
                         '280-char bodies; at --max-chars 5000 a long docstring splits into ~40 sentences and '
                         'a cap of 6 RE-TRUNCATES what --max-chars just widened. Raise together (e.g. '
                         '--max-chars 5000 --max-chunks 24) to run full-width windowed routing.')
    ap.add_argument('--wf-alpha', type=float, default=0.5,
                    help="Workflow propagation strength (0=seed unchanged, 1=pure neighbour spread).")
    ap.add_argument('--rrf-k', type=int, default=60,
                    help="Reciprocal Rank Fusion damping constant k (default 60, standard).")
    ap.add_argument('--multivec', action='store_true',
                    help="MULTI-VECTOR (late-interaction) routing: embed each module docstring as several "
                         "sentence CHUNKS and score by the MAX cosine over a module's chunks, not the mean. "
                         "Rescues a strong single sentence the averaged vector buries (the demux mask).")
    ap.add_argument('--hier', action='store_true',
                    help="HIERARCHICAL routing: route the query to its module FAMILY first (coarse, ~11-way), "
                         "then rank modules WITHIN that family (fine). Turns the flat 505-way search into "
                         "group-then-leaf -- the structure hierarchical_recall measured 100%% vs 18.3%% flat.")
    ap.add_argument('--enrich-aliases', action='store_true',
                    help="append each module's catalog aliases (via the module= link) to its routing text "
                         "before embedding -- tests whether the richer paraphrase target beats bare routing.")
    ap.add_argument('--abtt', type=int, default=1,
                    help='all-but-the-top: principal directions to remove (0 disables)')
    ap.add_argument('--rope-base', type=float, default=None)
    ap.add_argument('--heads', type=int, default=12)
    ap.add_argument('--ln-eps', type=float, default=1e-12)
    ap.add_argument('--swap-gate', action='store_true', default=True)
    ap.add_argument('--pre-ln', action='store_true', default=False)
    args = ap.parse_args()

    cfgp = os.path.join(os.path.dirname(os.path.abspath(args.model)), 'config.json')
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp))   # json imported at module top; a local import here shadows it
        if args.rope_base is None: args.rope_base = float(cfg.get('rotary_emb_base', 1000.0))
        args.heads = int(cfg.get('n_head', cfg.get('num_attention_heads', args.heads)))
    if args.rope_base is None: args.rope_base = 1000.0

    global MAX_CHARS
    MAX_CHARS = args.max_chars
    tok = WordPiece(args.vocab)

    print("=" * 90)
    print("[1] VOCABULARY COVERAGE over the codebase's real terms")
    print("=" * 90)
    terms = collect_terms(args.repo)
    unk = 0; pieces = []; shattered = []
    for t, c in terms.items():
        ids = tok.encode(t)[1:-1]
        if len(ids) and all(i == tok.unk for i in ids):
            unk += 1
        pieces.append(len(ids))
        if len(ids) >= 4 and c >= 5:
            shattered.append((len(ids), t, c))
    pieces = np.array(pieces)
    print(f"  distinct terms {len(terms)} | all-[UNK] {unk} ({unk/max(len(terms),1):.2%}) | "
          f"single-token {int((pieces==1).sum())} ({(pieces==1).mean():.1%}) | mean pieces/term {pieces.mean():.2f}")
    print("  most-shattered frequent jargon:", [f"{t}({p})" for p, t, _ in sorted(shattered, reverse=True)[:6]])
    print("  READ: WordPiece never fails -- it decomposes. Cost of rare jargon = extra pieces, and")
    print("        multi-piece terms still embed compositionally (why snake_case names work at all).")

    print("\n" + "=" * 90)
    print("[2] BUILDING THE INDEX: code docstrings + markdown sections")
    print("=" * 90)
    entries = collect_code(args.repo, enrich_aliases=getattr(args, 'enrich_aliases', False))
    if not args.no_md:
        entries = entries + collect_md(args.repo, args.window, args.stride)
    kinds = collections.Counter(e[0] for e in entries)
    print(f"  entries: {len(entries)}  {dict(kinds)}")
    model = Nomic(args.model, args)

    # CONTENT-HASHED CACHE. The embedding of a document depends only on its text and the wiring,
    # both of which we hash. Rerunning after a code change re-embeds only what changed.
    cache = {}
    if not args.no_cache and os.path.exists(args.cache):
        try:
            cache = json.load(open(args.cache))
            print(f"  cache: {len(cache)} entries loaded from {args.cache}", file=sys.stderr)
        except Exception:
            cache = {}
    wiring = f"{args.rope_base}|{args.heads}|{args.swap_gate}|{args.pre_ln}"

    def embed_cached(text):
        key = hashlib.sha256((wiring + '||' + text).encode()).hexdigest()[:32]
        hit = cache.get(key)
        if hit is not None:
            return np.array(hit, dtype=np.float32)
        v = model.embed(tok.encode(text))
        cache[key] = [round(float(z), 6) for z in v]      # 6dp keeps cosine to ~1e-6
        return v

    import time as _time
    t0 = _time.time(); hits0 = len(cache)
    E = []
    for i, (k, name, body) in enumerate(entries):
        E.append(embed_cached(f"search_document: {name} -- {body}"))
        if (i + 1) % 50 == 0:
            done, tot = i + 1, len(entries)
            el = _time.time() - t0
            eta = el / done * (tot - done)
            print(f"    {done}/{tot}  elapsed {el/60:.1f}m  eta {eta/60:.1f}m", file=sys.stderr)
    E = np.array(E)
    # NOTE: the cache is flushed at the END of main() so query embeddings are saved too.
    # (An earlier version dumped here and silently never cached the queries.)
    for d in (768, 128, 64):
        print(f"  index @{d:3d}d: {len(entries)*d*4/1e6:6.3f} MB fp32 | {len(entries)*d/1e6:6.3f} MB q8")

    if args.save:
        ab64 = AllButTheTop(E[:, :64], k=args.abtt)
        payload = dict(emb=ab64(E[:, :64]).astype(np.float32),
                       names=[e[1] for e in entries], kinds=[e[0] for e in entries])
        if getattr(args, 'structural', False) and wf_graph is not None:
            # persist the WORKFLOW BONES beside the embeddings, INDEX-ALIGNED to names, so the production
            # router (holographic_router) can run the measured dense+structure fusion at query time without
            # re-deriving the graph. Flat edge arrays (src_idx, dst_idx, weight) -- ~1.1k edges, npz-friendly.
            # The graph keys are bare stems; the index names are holographic_<stem> -- join here, once.
            name_idx = {e[1]: j for j, e in enumerate(entries)}
            src, dst, wts = [], [], []
            for (a, b), w in wf_graph["edges"].items():
                ia = name_idx.get("holographic_" + a)
                ib = name_idx.get("holographic_" + b)
                if ia is not None and ib is not None:
                    src.append(ia); dst.append(ib); wts.append(w)
            payload.update(bone_src=np.array(src, dtype=np.int32),
                           bone_dst=np.array(dst, dtype=np.int32),
                           bone_w=np.array(wts, dtype=np.float32))
            print(f"  bones packed into index: {len(src)} edges")
        np.savez_compressed(args.save, **payload)
        print(f"  saved 64d index -> {args.save}")

    # ---- [3] module routing
    # Fit the anisotropy correction on the full document set, once.
    abtt = AllButTheTop(E, k=args.abtt)
    # BUG FIXED: unit(v) divides by the FROBENIUS norm on a 2-D array (one scalar for the whole
    # matrix), so the old diagnostic printed ~0.000. Normalize ROWS.
    _rows = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    print(f"\n  anisotropy: raw mean|cos| between docs "
          f"{float(np.abs(_rows(E) @ _rows(E).T).mean()):.3f} -> corrected "
          f"{float(np.abs(abtt(E) @ abtt(E).T).mean()):.3f}   (all-but-the-top, k={args.abtt})")

    exam_top5, exam_median = 0, 999.0     # set by the 768d pass below; safe defaults for the gate
    print("\n" + "=" * 90)
    print("[3] MODULE ROUTING (bar: keyword top-1 = 1/12)")
    print("=" * 90)
    code_idx = [i for i, e in enumerate(entries) if e[0] == 'code']
    code_entries = [entries[i] for i in code_idx]
    # A zero-module corpus means --repo points at the wrong tree (the classic: run from tools/semantic
    # with --repo .. instead of ../..). Fail with the FIX in the message, not an IndexError 30 lines down.
    if not code_entries:
        raise SystemExit(
            f"\n  0 code modules found under --repo {args.repo!r} (cwd {os.getcwd()!r}).\n"
            f"  collect_code looks for holographic_*.py; none were under that path.\n"
            f"  If running from tools/semantic, the repo root is '../..', not '..'.")
    def kw_rank_of(a, accept):
        order = kw_rank(a, code_entries)
        for r, j in enumerate(order):
            if code_entries[j][1] in accept: return r + 1
        return len(code_entries)
    kw1 = sum(kw_rank_of(a, acc) == 1 for a, acc in ASKS_MODULE)
    kwmr = np.median([kw_rank_of(a, acc) for a, acc in ASKS_MODULE])
    print(f"  KEYWORD baseline: top-1 {kw1}/{len(ASKS_MODULE)} | median rank of best accepted answer: {kwmr:.0f}"
          f"  (of {len(code_entries)} modules)")
    Q = np.array([embed_cached(f"search_query: {a}") for a, _ in ASKS_MODULE])
    for d in (768, 128, 64):
        # truncate FIRST (Matryoshka), then fit/apply the correction in that subspace
        ab_d = AllButTheTop(E[:, :d], k=args.abtt)
        Ed = ab_d(E[code_idx][:, :d])
        Qd = ab_d(Q[:, :d])
        t1 = t5 = 0; ranks = []; detail = []
        for i, (a, acc) in enumerate(ASKS_MODULE):
            order = np.argsort(-(Ed @ Qd[i]))
            names_r = [code_entries[j][1] for j in order]
            rank = next((r + 1 for r, n in enumerate(names_r) if n in acc), len(names_r))
            ranks.append(rank); t1 += rank == 1; t5 += rank <= 5
            if d == 768:
                detail.append((a, names_r[0], rank, acc[0]))
        print(f"  EMBEDDING @{d:3d}d: top-1 {t1}/{len(ASKS_MODULE)}  top-5 {t5}/{len(ASKS_MODULE)}  "
              f"median rank {np.median(ranks):.0f}  worst {max(ranks)}")
        if d == 768:
            exam_top5, exam_median = t5, float(np.median(ranks))   # the gate reads full-width only
        if d == 768:
            for a, g, rank, want in detail:
                mark = 'HIT ' if rank == 1 else ('top5' if rank <= 5 else f'r{rank:<3d}')
                print(f"     [{mark}] {a:42s} -> {g.replace('holographic_',''):20s} "
                      f"(accepted answer at rank {rank}; e.g. {want.replace('holographic_','')})")

    # ---- [3x] HYBRID routing: fuse dense (cosine) with BM25 (lexical) via Reciprocal Rank Fusion. The dense
    # router buries asks whose query WORDS appear in the target docstring but whose GEOMETRY collapses them
    # apart (meshsmooth 'bumpy surface' r22, dynamics 'ball goes next' r40). BM25 exact-matches those words;
    # RRF fuses the two RANKINGS (no score calibration -- cosine and BM25 are on different scales, only ranks
    # are comparable) so a module ranked well by EITHER rises, and the dense HITs are preserved. Literature:
    # MonaVec 2606.19458 (training-free BM25+dense RRF, rejects SPLADE for our constraint), gains largest for
    # weak dense retrievers (2605.24297). KEPT NEG: cannot help 'grainy' (absent from doc AND query terms).
    wf_graph = None                                            # hoisted: --save reads this even when the
                                                               # hybrid block never runs (no NameError)
    if args.structural and not args.hybrid:
        # --structural adds a THIRD list to the hybrid fusion; on its own it has nothing to fuse INTO. Rather
        # than silently print nothing (measured: a real user ran --structural alone and got no output), turn
        # the fusion on and say so.
        args.hybrid = True
        print("\n  [note] --structural implies --hybrid (it is a third fusion input); enabling --hybrid.")
    if args.hybrid:
        # the exam is launched from tools/semantic, so holographic.* is not on the path yet. Add the repo root
        # (authoritative: args.repo; fall back to __file__-relative) exactly like the catalog-join helper does.
        import os as _os
        for _rr in (_os.path.abspath(args.repo),
                    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))):
            if _rr not in sys.path:
                sys.path.insert(0, _rr)
        from holographic.semantic_router.holographic_bm25 import BM25, reciprocal_rank_fusion
        doc_texts = ["%s -- %s" % (name, body) for (_, name, body) in [entries[i] for i in code_idx]]
        bm = BM25(doc_texts)

        # ---- THE STRUCTURAL THIRD LIST (--structural): the workflow bones. Dense and BM25 both need the query
        # to SHARE something with the target text (meaning or words). A vocabulary-gap ask ('less grainy' vs a
        # docstring that says 'manifold projection / Milanfar') shares NEITHER -- structurally unreachable by
        # both. The workflow graph reaches it a third way: seed each module with its DENSE score, spread ONE hop
        # along the sparse author-stated cross-reference bones, and a module whose COLLABORATORS are hit rises
        # even though its own text was never matched. Rarity-weighted bones keep this precise (median
        # out-degree 2) -- the diffuse io-kind graph (13-24 neighbours) is exactly what NOT to propagate along.
        if args.structural:
            from holographic.semantic_router.holographic_workflowgraph import build_workflow_graph, propagate
            wf_graph = build_workflow_graph(_os.path.abspath(args.repo))
            print("  [structural] workflow bones: %d modules, %d edges, hubs dropped %s"
                  % (wf_graph["n_modules"], len(wf_graph["edges"]), wf_graph["dropped_hubs"][:3]))

        # module-name <-> stem join: the exam names modules 'holographic_meshsmooth'; the workflow graph keys on
        # the bare stem 'meshsmooth' (same convention as the catalog's resolved_module).
        stem_of = [code_entries[j][1][len("holographic_"):] if code_entries[j][1].startswith("holographic_")
                   else code_entries[j][1] for j in range(len(code_entries))]
        idx_of_stem = {s: j for j, s in enumerate(stem_of)}

        def _structural_order(sims):
            """Rank doc indices by workflow-propagated dense score. Seeds are CLAMPED at 0: a negative cosine
            means 'not relevant', and spreading negative activation would actively push a module's collaborators
            DOWN, which is not the claim -- bones carry evidence FOR, never against."""
            seed = {stem_of[j]: max(0.0, float(sims[j])) for j in range(len(sims))}
            ranked = propagate(wf_graph, seed, alpha=args.wf_alpha)
            order = [idx_of_stem[m] for m, _ in ranked if m in idx_of_stem]
            seen = set(order)
            order.extend(j for j in range(len(code_entries)) if j not in seen)   # unranked tail, stable
            return order
        # DENSE-DOMINANCE SWEEP: equal-weight RRF (beta=1) let BM25 overtake the dense HITs (measured 6->3
        # top-1). Down-weight the lexical list: fused uses weights (1.0, beta). beta=0 = pure dense (MUST equal
        # flat [3] -- a built-in sanity check); beta=1 = classic equal RRF. The IR-literature optimum is
        # dense-dominant, so the knee should sit at LOW beta -- keeping the dense HITs while a strong BM25 rank
        # still rescues a dense-buried lexical miss. Sweeps in one run so the knee is measured, not guessed.
        # WEIGHT GRID. Each row is a (beta, gamma) pair: beta = lexical (BM25) weight, gamma = structural
        # (workflow) weight, dense always 1.0. (0,0) MUST reproduce flat [3] -- the built-in sanity check.
        # The beta column isolates lexical; the gamma column isolates STRUCTURE (no lexical confound); the
        # combined rows test whether the three villagers stack. Equal weight (1.0) is included because it is
        # the setting that already FAILED for BM25 (6->3 top-1) -- keeping the refutation visible in the table.
        combos = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.5, 0.0), (1.0, 0.0)]
        if args.structural:
            combos += [(0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.5), (0.0, 1.0),
                       (0.2, 0.2), (0.3, 0.2), (0.2, 0.3)]
        print("\n  --- HYBRID ROUTING (dense + BM25%s, weighted RRF k=%d) ---"
              % (" + WORKFLOW BONES" if args.structural else "", args.rrf_k))
        print("  %5s %5s | %4s  %6s %6s %7s %6s" % ("beta", "gamma", "dim", "top-1", "top-5", "median", "worst"))
        detail_rows = {}                                       # (beta,gamma,dim) -> per-ask detail
        flat_rank = {}                                          # dim -> {ask: rank} from the (0,0) row
        _struct_cache = {}                                     # (d, ask_i) -> structural order (weight-free)
        for beta, gamma in combos:
            for d in (768, 128, 64):                           # 64 = the PRODUCTION index dim; flat 64d is
                                                               # 2/12, the weakest regime -- where fusion is
                                                               # predicted (2605.24297) to gain the most.
                ab_d = AllButTheTop(E[:, :d], k=args.abtt)
                Ed = ab_d(E[code_idx][:, :d]); Qd = ab_d(Q[:, :d])
                t1 = t5 = 0; ranks = []; detail = []
                for i, (a, acc) in enumerate(ASKS_MODULE):
                    sims = Ed @ Qd[i]
                    dense_order = list(np.argsort(-sims))                  # dense ranking (doc indices)
                    bm_order = [j for j, _ in bm.rank(a)]                  # BM25 ranking (doc indices)
                    lists = [dense_order, bm_order]; wts = [1.0, beta]
                    if args.structural and gamma > 0.0:
                        key = (d, i)
                        if key not in _struct_cache:                       # order is weight-free -> cache it
                            _struct_cache[key] = _structural_order(sims)
                        lists.append(_struct_cache[key]); wts.append(gamma)
                    fused = reciprocal_rank_fusion(lists, k=args.rrf_k, weights=wts)
                    names_r = [code_entries[j][1] for j, _ in fused]
                    rank = next((r + 1 for r, n in enumerate(names_r) if n in acc), len(code_entries))
                    ranks.append(rank); t1 += rank == 1; t5 += rank <= 5
                    if (beta, gamma) == (0.0, 0.0):
                        flat_rank.setdefault(d, {})[a] = rank              # baseline rank, for the diff below
                    detail.append((a, names_r[0], rank))
                print("  %5.2f %5.2f | %4dd  %5d/%d %5d/%d %7.0f %6d"
                      % (beta, gamma, d, t1, len(ASKS_MODULE), t5, len(ASKS_MODULE),
                         np.median(ranks), max(ranks)))
                # capture per-ask where the RESULT is: measured, the structural win lands at the SHIP dim
                # (128d), not 768d -- so detail at 768d showed a row with no win and told us nothing about
                # WHICH ask flipped. Capture the winning row at 128d, keyed by (beta, gamma, dim).
                if args.structural and (beta, gamma, d) in {(0.0, 0.5, 128), (0.0, 0.5, 768), (0.0, 0.5, 64)}:
                    detail_rows[(beta, gamma, d)] = detail
                if args.structural and (beta, gamma, d) == (0.0, 0.5, 128):
                    fused_top1_128 = t1                        # the champion row, for the --require-fused-top1 gate
                elif not args.structural and (beta, gamma, d) in {(0.1, 0.0, 128), (0.1, 0.0, 768)}:
                    detail_rows[(beta, gamma, d)] = detail
        # PER-ASK DIFF vs the flat baseline -- shows exactly WHICH asks the fusion moved, and by how much.
        for (b_, g_, d_), rows in sorted(detail_rows.items()):
            print("  per-ask @%dd at beta=%.2f gamma=%.2f  (flat rank -> fused rank):" % (d_, b_, g_))
            for a, top, rank in rows:
                fr = flat_rank.get(d_, {}).get(a)
                delta = ""
                if fr is not None and fr != rank:
                    delta = "  %s r%d -> r%d" % ("BETTER" if rank < fr else "worse ", fr, rank)
                elif fr is not None:
                    delta = "  same  r%d" % fr
                mark = 'HIT ' if rank == 1 else ('top5' if rank <= 5 else 'r%-4d' % rank)
                print("     [%s] %-34s -> %-18s%s" % (mark, a[:34], top.replace('holographic_', '')[:18], delta))
        print("  READ: (0,0) MUST equal flat [3] (sanity). beta column = lexical alone; gamma column = "
              "STRUCTURE alone (no lexical confound) -- does propagating along workflow bones rescue denoise, "
              "which shares NO words with its docstring? Combined rows test if the three villagers stack. If "
              "nothing beats (0,0) top-1 6/12, hybrid+structural is a kept negative on this suite.")

        # ---- [3m] MULTI-VECTOR (late-interaction) routing: the demux mask. Each module docstring is split into
    # sentence chunks (_doc_chunks), each chunk embedded (cached like everything else), and a module scored
    # by the MAX cosine over its chunks -- so a single strongly-relevant sentence surfaces the module even
    # when its MEAN vector buries it (denoise's noise sentence diluted by the rest). Compare to flat above.
    if args.multivec:
        print("\n  --- MULTI-VECTOR ROUTING (max-sim over docstring chunks vs the mean) ---")
        # build, per module, the list of chunk vectors (full-width 768d; truncate per dim below).
        chunk_vecs = []                                     # list over modules -> (n_chunks x 768)
        for k, name, body in [entries[i] for i in code_idx]:
            chs = _doc_chunks(body, max_chunks=args.max_chunks)
            cvs = [embed_cached("search_document: %s -- %s" % (name, c)) for c in chs]
            chunk_vecs.append(np.array(cvs))
        for d in (768, 128):
            ab_d = AllButTheTop(E[:, :d], k=args.abtt)
            Qd = ab_d(Q[:, :d])
            # apply the SAME all-but-the-top correction (fit on the whole-doc E) to every chunk vector
            Cd = [ab_d(cv[:, :d]) for cv in chunk_vecs]
            t1 = t5 = 0; ranks = []
            for i, (a, acc) in enumerate(ASKS_MODULE):
                q = Qd[i] / (np.linalg.norm(Qd[i]) + 1e-12)
                # score each module by its BEST chunk (late interaction / max-sim)
                scores = np.array([float(np.max((cv / (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-12)) @ q))
                                   for cv in Cd])
                order = np.argsort(-scores)
                names_r = [code_entries[j][1] for j in order]
                rank = next((r + 1 for r, n in enumerate(names_r) if n in acc), len(names_r))
                ranks.append(rank); t1 += rank == 1; t5 += rank <= 5
            print("  @%3dd: top-1 %d/%d  top-5 %d/%d  median %.0f  worst %d"
                  % (d, t1, len(ASKS_MODULE), t5, len(ASKS_MODULE), np.median(ranks), max(ranks)))
            if d == 768:
                mv_detail = list(zip([a for a, _ in ASKS_MODULE], ranks))   # per-ask rank at full width
        # PER-ASK: which asks did the chunk max-sim move vs flat? (localizes the demux win -- did denoise
        # surface from r240?). Prints only the asks whose multivec rank differs from what flat would give.
        for a, r in mv_detail:
            mark = 'HIT ' if r == 1 else ('top5' if r <= 5 else 'r%-4d' % r)
            print("     [%s] %s" % (mark, a))
        print("  READ: compare to flat [3]. A win here means the relevant signal lived in ONE sentence the "
              "mean diluted -- the demux/late-interaction mask working. Per-ask above shows WHICH asks moved.")

        # ---- BLEND SWEEP: pure max-sim lets a WRONG module's lone strong chunk overtake a good-mean module
        # (measured: dynamics 40->94). Blend keeps the whole-doc MEAN as the dominant term and adds the best
        # chunk as a nudge:  score = alpha*mean_sim + (1-alpha)*max_chunk_sim.  alpha=1 = flat, alpha=0 = pure
        # multivec. Sweeps alpha in ONE run so the precision(top-1)/recall(top-5) knee is visible, not guessed.
        # The whole-doc vector is already one of the chunks, so this is a principled reweighting, not new data.
        print("\n  --- MULTIVEC BLEND SWEEP (alpha*mean + (1-alpha)*max_chunk; alpha=1 flat, 0 pure multivec) ---")
        print("  %5s | %4s  %6s %6s %7s %6s" % ("alpha", "dim", "top-1", "top-5", "median", "worst"))
        for alpha in (1.0, 0.75, 0.5, 0.25, 0.0):
            for d in (768, 128):
                ab_d = AllButTheTop(E[:, :d], k=args.abtt)
                Ed = ab_d(E[code_idx][:, :d]); Qd = ab_d(Q[:, :d]); Cd = [ab_d(cv[:, :d]) for cv in chunk_vecs]
                t1 = t5 = 0; ranks = []
                for i, (a, acc) in enumerate(ASKS_MODULE):
                    q = Qd[i] / (np.linalg.norm(Qd[i]) + 1e-12)
                    mean_s = Ed @ q                                       # whole-doc similarity per module
                    chunk_s = np.array([float(np.max((cv / (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-12)) @ q))
                                        for cv in Cd])
                    scores = alpha * mean_s + (1.0 - alpha) * chunk_s     # the blend
                    order = np.argsort(-scores)
                    names_r = [code_entries[j][1] for j in order]
                    rank = next((r + 1 for r, n in enumerate(names_r) if n in acc), len(names_r))
                    ranks.append(rank); t1 += rank == 1; t5 += rank <= 5
                print("  %5.2f | %4dd  %5d/%d %5d/%d %7.0f %6d"
                      % (alpha, d, t1, len(ASKS_MODULE), t5, len(ASKS_MODULE), np.median(ranks), max(ranks)))
        print("  READ: look for an alpha that keeps flat's top-1 (alpha=1 row) AND lifts top-5 toward pure "
              "multivec -- the precision/recall sweet spot. If none beats alpha=1, the blend is a kept negative.")

        # ---- [3h] HIERARCHICAL routing: the structural reframe. Flat routing ranks the accepted answer against
    # all 505 modules -- denoise buried at r237. Instead: (1) COARSE -- route the query to its module FAMILY
    # by nearest family PROTOTYPE (mean of the family's module vectors), an ~11-way choice; (2) FINE -- rank
    # modules only WITHIN the chosen family. This is group-then-leaf (holographic_hierarchy's hierarchical_
    # recall: 100%% vs 18.3%% flat). Reports coarse-family accuracy AND the fine rank, so we see WHERE it wins
    # or loses. Honest: if COARSE routes to the wrong family, the answer is unreachable (rank = family size) --
    # that failure mode is shown, not hidden.
    if args.hier:
        # PROMOTED + GENERALIZED: the coarse level is pluggable. 'family' = directory; 'iokind' = the
        # module's datatype group (the pipeline-map join). Same group-then-leaf structure either way.
        if args.hier_by == 'iokind':
            fam_map = _module_iogroups(args.repo)             # module name -> io-kind group (semantic)
            _grp_label = 'io-kind'
        else:
            fam_map = _module_families(args.repo)             # module name -> directory family
        _default = 'untyped' if args.hier_by == 'iokind' else 'root'
        families = sorted(set(fam_map.get(code_entries[j][1], _default) for j in range(len(code_entries))))
        fam_of = [fam_map.get(code_entries[j][1], _default) for j in range(len(code_entries))]
        # COMBINED with --multivec: the FINE step scores in-group modules by max-sim over their docstring
        # CHUNKS (late interaction) instead of the mean vector -- so within the coarse group the ask still
        # surfaces the module whose single best sentence matches, not the one whose average does. Built once.
        combined = bool(args.multivec)
        if combined:
            hchunks = []                                      # module index -> (n_chunks x 768) chunk vectors
            for k, name, body in [entries[i] for i in code_idx]:
                cvs = [embed_cached("search_document: %s -- %s" % (name, c)) for c in _doc_chunks(body, max_chunks=args.max_chunks)]
                hchunks.append(np.array(cvs))
        print("\n  --- HIERARCHICAL ROUTING (coarse %s -> fine %s) ---"
              % (args.hier_by, "module by MAX-SIM over chunks" if combined else "module"))
        for d in (768, 128):
            ab_d = AllButTheTop(E[:, :d], k=args.abtt)
            Ed = ab_d(E[code_idx][:, :d]); Qd = ab_d(Q[:, :d])
            Cd = [ab_d(cv[:, :d]) for cv in hchunks] if combined else None
            # family prototypes: mean (then unit) of each family's module vectors
            proto = {}
            for fam in families:
                idx = [j for j in range(len(Ed)) if fam_of[j] == fam]
                v = Ed[idx].mean(0); proto[fam] = v / (np.linalg.norm(v) + 1e-12)
            P = np.array([proto[f] for f in families])
            t1 = t5 = 0; ranks = []; coarse_ok = 0
            for i, (a, acc) in enumerate(ASKS_MODULE):
                q = Qd[i] / (np.linalg.norm(Qd[i]) + 1e-12)
                fam_pick = families[int(np.argmax(P @ q))]     # COARSE: nearest family prototype
                true_fam = fam_map.get(acc[0], _default)
                coarse_ok += (fam_pick == true_fam)
                # FINE: rank modules within the picked family only -- by max-sim over chunks if combined, else mean
                fidx = [j for j in range(len(Ed)) if fam_of[j] == fam_pick]
                if combined:
                    def _score(j):
                        cv = Cd[j]
                        return float(np.max((cv / (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-12)) @ q))
                    order = sorted(fidx, key=lambda j: -_score(j))
                else:
                    order = sorted(fidx, key=lambda j: -(Ed[j] @ q))
                names_r = [code_entries[j][1] for j in order]
                rank = next((r + 1 for r, n in enumerate(names_r) if n in acc), 10**6)
                ranks.append(rank if rank < 10**6 else len(fidx) + 1)
                t1 += rank == 1; t5 += rank <= 5
            print(f"  @{d:3d}d: coarse-group {coarse_ok}/{len(ASKS_MODULE)} correct | "
                  f"fine top-1 {t1}/{len(ASKS_MODULE)}  top-5 {t5}/{len(ASKS_MODULE)}  median {np.median(ranks):.0f}")
        print("  READ: fine top-1 counts only asks whose group was routed correctly; a coarse miss caps that "
              "ask. %s Compare to flat [3] and to --hier alone -- the win is buried answers surfacing."
              % ("FINE step used max-sim over chunks (--hier --multivec combined)." if combined else ""))

        # ---- [3b] optional: sweep all-but-the-top k so the best depth is measured, not guessed. The synthetic
    # said "more is better"; the real 12-ask suite showed a TRADEOFF (k=1 best for 128d top-1, k=3 best for
    # the tail + 64d). This prints the whole curve in one run so the knee is visible. Reuses E, Q, ASKS.
    if args.abtt_sweep:
        ks = [int(x) for x in args.abtt_sweep.split(',') if x.strip() != '']
        print("\n  --- ABTT SWEEP (all-but-the-top depth k; routing exam per k) ---")
        print(f"  {'k':>3s} | {'dim':>4s}  {'top-1':>6s} {'top-5':>6s} {'median':>7s} {'worst':>6s}")
        for k in ks:
            for d in (768, 128, 64):
                ab = AllButTheTop(E[:, :d], k=k)
                Ed = ab(E[code_idx][:, :d]); Qd = ab(Q[:, :d])
                ranks = []
                for i, (a, acc) in enumerate(ASKS_MODULE):
                    order = np.argsort(-(Ed @ Qd[i]))
                    names_r = [code_entries[j][1] for j in order]
                    ranks.append(next((r + 1 for r, n in enumerate(names_r) if n in acc), len(names_r)))
                t1 = sum(r == 1 for r in ranks); t5 = sum(r <= 5 for r in ranks)
                print(f"  {k:>3d} | {d:>4d}  {t1:>4d}/{len(ASKS_MODULE):<1d} {t5:>4d}/{len(ASKS_MODULE):<1d} "
                      f"{np.median(ranks):>7.0f} {max(ranks):>6d}")
            print()
        print("  READ: pick the k that keeps top-1 high at the SHIP dim (128d) while lowering worst/median.")

    # ---- [4] the constitutional test: has this been tried?
    print("\n" + "=" * 90)
    print("[4] PROBE-FIRST / KEPT-NEGATIVE LOOKUP over the md corpus (bar: keyword top-1 = 2/6)")
    print("=" * 90)
    md_idx = [i for i, e in enumerate(entries) if e[0] in ('docs', 'note')]
    md_entries = [entries[i] for i in md_idx]
    if not md_entries:
        # --no-md skips the markdown corpus, so the negative-lookup (which searches md sections) has
        # nothing to rank. It is a diagnostic section, not the gate -- skip it cleanly. The routing
        # exam in [3] already ran and is what --exam gates on.
        print("  (skipped: --no-md, no markdown corpus to search for kept-negatives)")
    else:

        # SCORER FIX: retrieval ranks WINDOWS but collapse_by_parent returns one window per SECTION.
        # The needle may sit in a different window of the same section, so a correct retrieval was being
        # scored as a miss. Judge the hit at PARENT-SECTION level -- the unit the ranking actually returns.
        _by_parent = {}
        for e in md_entries:
            _by_parent.setdefault(parent_of(e[1]), []).append(e[2])

        def hit(entry, needle):
            p = parent_of(entry[1])
            blob = p + ' ' + ' '.join(_by_parent.get(p, ()))
            return needle.lower() in blob.lower()
        kwn = sum(hit(md_entries[kw_rank(a, md_entries)[0]], nd) for a, nd in ASKS_NEGATIVE)
        print(f"  KEYWORD baseline top-1: {kwn}/{len(ASKS_NEGATIVE)}")
        Qn = np.array([embed_cached(f"search_query: {a}") for a, _ in ASKS_NEGATIVE])
        for d in (768, 128, 64):
            ab_d = AllButTheTop(E[:, :d], k=args.abtt)
            Ed = ab_d(E[md_idx][:, :d])
            Qnd = ab_d(Qn[:, :d])
            t1 = t5 = 0
            detail = []
            for i, (a, nd) in enumerate(ASKS_NEGATIVE):
                full = np.argsort(-(Ed @ Qnd[i]))
                order = collapse_by_parent(full, md_entries)[:5]
                h1 = hit(md_entries[order[0]], nd)
                h5 = any(hit(md_entries[j], nd) for j in order)
                t1 += h1; t5 += h5
                if d == 768:
                    detail.append((a, nd, md_entries[order[0]][1], h1, h5))
            print(f"  EMBEDDING @{d:3d}d: top-1 {t1}/{len(ASKS_NEGATIVE)}  top-5 {t5}/{len(ASKS_NEGATIVE)}")
            if d == 768:
                for a, nd, g, h1, h5 in detail:
                    mark = 'HIT ' if h1 else ('top5' if h5 else 'MISS')
                    print(f"     [{mark}] {a[:44]:44s} -> {g[:44]:44s} (want '{nd}')")

        # ---- [5] neighborhoods
        print("\n" + "=" * 90)
        print("[5] SEMANTIC NEIGHBORHOODS (is this geometry, or luck?)")
        print("=" * 90)
        En = abtt(E)
        for p in PROBES:
            q = abtt(embed_cached(f"search_query: {p}"))[0]
            order = collapse_by_parent(np.argsort(-(En @ q)), entries)[:5]
            print(f"  '{p}'")
            for j in order:
                print(f"      [{entries[j][0]}] {entries[j][1][:66]}")

        if args.ask:
            q = abtt(embed_cached(f"search_query: {args.ask}"))[0]
            order = collapse_by_parent(np.argsort(-(En @ q)), entries)[:6]
            print(f"\n[ASK] '{args.ask}'")
            for j in order:
                print(f"   {float(En[j]@q):.3f} [{entries[j][0]}] {entries[j][1][:60]}")
                print(f"          {entries[j][2][:110]}")

    if not args.no_cache:
        json.dump(cache, open(args.cache, 'w'))
        print(f"  cache: {len(cache)-hits0} new, {len(cache)} total -> {args.cache}", file=sys.stderr)

    print("\nDONE -- paste back. Headlines: [3] beats 1/12? [4] beats 2/6? does 64d hold?")

    # THE GATE, last: only --exam turns a miss into a nonzero exit; it always prints its verdict.
    ok = exam_top5 >= args.require_top5 and exam_median <= args.require_median
    fused_msg = ""
    if args.require_fused_top1 is not None:
        # the FUSED gate guards the production default (route_semantic gamma=0.5 on the 128d index). A bones
        # or fusion regression must fail the build just like a dense regression does. 'absent' fails loudly:
        # if the champion row never ran (e.g. --structural missing), the gate is not silently skipped.
        ft = locals().get("fused_top1_128")
        if ft is None:
            fused_msg = " | fused top-1 ABSENT (need --structural) -> FAIL"
            ok = False
        else:
            f_ok = ft >= args.require_fused_top1
            fused_msg = f" | fused top-1 {ft} (require >= {args.require_fused_top1}) -> {'PASS' if f_ok else 'FAIL'}"
            ok = ok and f_ok
    print(f"  EXAM: top-5 {exam_top5} (require >= {args.require_top5}) | median {exam_median:.0f} "
          f"(require <= {args.require_median}){fused_msg} -> {'PASS' if ok else 'FAIL'}")
    if args.exam and not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
