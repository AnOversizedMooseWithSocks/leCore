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


def collect_code(repo):
    """One entry per module: its module docstring is the authoritative description (per the guide)."""
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
                out.append(('code', f[:-3], ' '.join(m.group(1).split())[:MAX_CHARS]))
    return out


def collect_md(repo, window=900, stride=650):
    """Markdown -> SLIDING WINDOWS inside each heading-section.

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
    entries = collect_code(args.repo) + collect_md(args.repo, args.window, args.stride)
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
        np.savez_compressed(args.save, emb=ab64(E[:, :64]).astype(np.float32),
                            names=[e[1] for e in entries], kinds=[e[0] for e in entries])
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

    print("\n" + "=" * 90)
    print("[3] MODULE ROUTING (bar: keyword top-1 = 1/12)")
    print("=" * 90)
    code_idx = [i for i, e in enumerate(entries) if e[0] == 'code']
    code_entries = [entries[i] for i in code_idx]
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
            for a, g, rank, want in detail:
                mark = 'HIT ' if rank == 1 else ('top5' if rank <= 5 else f'r{rank:<3d}')
                print(f"     [{mark}] {a:42s} -> {g.replace('holographic_',''):20s} "
                      f"(accepted answer at rank {rank}; e.g. {want.replace('holographic_','')})")

    # ---- [4] the constitutional test: has this been tried?
    print("\n" + "=" * 90)
    print("[4] PROBE-FIRST / KEPT-NEGATIVE LOOKUP over the md corpus (bar: keyword top-1 = 2/6)")
    print("=" * 90)
    md_idx = [i for i, e in enumerate(entries) if e[0] in ('docs', 'note')]
    md_entries = [entries[i] for i in md_idx]

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


if __name__ == '__main__':
    main()
