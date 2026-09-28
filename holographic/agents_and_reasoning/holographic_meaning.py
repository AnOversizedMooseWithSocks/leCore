"""holographic_meaning.py -- FIND AN ANSWER BY WHAT A QUESTION MEANS, LEARN NEW WORDINGS AS THEY ARRIVE, AND
REMEMBER HOW THE ANSWER WAS FOUND (sweep 181, owner-directed: Moose 2026-09-24).

WHY THIS EXISTS (measured before a line was written)
    The answer ladder's reflex serves NEAR-EXACT repeats only (Jaccard >= 0.75 over content words -- a
    deliberate gate: a looser one served one-word-different questions each other's answers). Paraphrase
    coverage was "the named sacrifice ... it belongs to T1/synthesis" -- but no rung above it searches the
    TAUGHT rows, so a person's rewording of a taught question was refused. On real human wording:
        CLINC150, one taught phrasing per intent, 4,500 held-out phrasings:  6 served (0.1%), 0 wrong
        Banking77, same protocol, 3,080 held-out phrasings:                  1 served (0.0%), 0 wrong
    Safe, and useless for a person who does not repeat themselves word for word.

WHAT A ROW IS
    One thing memory knows, reachable by MANY wordings:
        kind "answer"   -- a taught answer; the TEXT lives in the ladder's exact store (one source of truth, so
                           a vetoed/forgotten answer can never be served through this door)
        kind "method"   -- HOW to get the answer: {verb, args, slots, live}. "what's the price of solana" is
                           remembered as price(symbol=<symbol>), bound from the question and run FRESH every
                           time -- a live value is never cached, the way to fetch it is
        kind "clarify"  -- a wording that does not say what the person wants ("SOL"): the answer is the
                           clarifying question to ask them
    Each row keeps every PHRASING that has been confirmed to mean it. A confirmed new wording becomes one more
    phrasing of the SAME row -- never a duplicate row with a copied answer.

HOW A QUESTION IS MATCHED (deterministic, NumPy + stdlib, no learned weights shipped)
    features   content-word stems (Porter-style, the bm25 module's), stem bigrams (word order), and
               character 4-grams (morphology and typos: "shading" ~ "shadows" share ^sha/shad)
    weights    IDF over ROWS (a word every row uses says nothing about which row)
    expansion  LEARNED WORD ASSOCIATIONS: every confirmed link teaches which of the new wording's words stand
               for which of the row's words ("balance" <-> "much money"), and those pairs widen FUTURE
               questions for EVERY row -- the meaning is learned, not just the wording
    score      cosine against each phrasing; a row scores its best phrasing
    decision   serve / clarify / escalate. Serve needs confidence; the confidence becomes a CALIBRATED
               probability once enough verdicts exist (isotonic, the same PAV fit SystemOne uses): every
               escalation the model resolves is a free label exactly where the gate is unsure.

THE MODEL END (typed, the four-part shape of SystemOne's escalation_prompt)
    An unsure question goes to the model WITH its nearest rows as evidence; the model returns a typed JSON
    verdict, validated against the contract and retried once:
        {"verdict": "same", "row": "<id>"[, "args": {...}]}       -> link the wording to that row
        {"verdict": "new", "answer": "...", "method": {...}?}      -> a new row (and how it was found)
        {"verdict": "unclear", "clarify": "<question for the person>"}
    A reply that is not JSON is treated as a plain answer ("new", no method) -- a model that ignores the
    contract still answers, it just teaches less.

WHAT THE CLM BACKLOG ADDED (phase C, 2026-09-26 -- each piece documents its own numbers where it lives)
    RowPrototypes / enable_protos   the shared InfoNCE rule on the rows, in this index's own sparse space (E1.1 c3)
    add_row's MERGE RULE            a new answer row with an existing row's answer key joins that row (E2.1; OFF
                                    by default -- Banking77 lost a third of its coverage, MERGE_SAME_ANSWER)
    decide + teacher (TeacherNoise) the gate serves on P(correct) = corrected P(agree) once the model end is
                                    measurably noisy (a conservative bound on eps, TEACHER_EPS_Z); the
                                    teacher-relative ceiling is RETIRED (E2.3, see P_FLOOR)
    note_off_domain                 an explicit off-domain verdict is recorded, nothing trains on it (E1.3)
    bind(reader=)                   from / to read from the question's context words, "no direction" -> ask (E3.1)
    numbers_in                      "20,000" is one number (found on the way; it froze a method's amount)
"""
import hashlib
import json
import math
import re

import numpy as np

# ------------------------------------------------------------------------------------------------ text
# Function words: they carry grammar, not WHICH thing is asked. Question words (what/how/why/when/where/who)
# are kept OUT of this list on purpose -- "how do I" vs "why did" can be the whole difference.
_STOP = frozenset("""
a an the of to in on at for and or is are was were be been being am by with from as it its this that these
those into over under out up down off do does did doing can could would should will shall may might must
i me my mine we us our ours you your yours he him his she her they them their theirs please pls just
really very so some any there here then than too also get got have has had want wanna need like tell
know let lets okay ok im ive id youre
s t d ll re ve m
""".split())
# NEGATION CARRIES MEANING. The first cut listed dont/cant/isnt as stopwords and served "that isn't right" the
# answer to "that is right" (measured on CLINC150: no -> yes at high confidence). Contractions are expanded to
# an explicit "not", which is a content word like any other.
_NEG = {"dont": "do not", "doesnt": "does not", "didnt": "did not", "isnt": "is not", "arent": "are not",
        "wasnt": "was not", "werent": "were not", "cant": "can not", "cannot": "can not", "couldnt": "could not",
        "wont": "will not", "wouldnt": "would not", "shouldnt": "should not", "havent": "have not",
        "hasnt": "has not", "hadnt": "had not", "aint": "is not", "mustnt": "must not"}
_SESSION = re.compile(r"^\[s:[^\]]*\]\s*")


def placeholder_method(method):
    """A METHOD with every credential among its CONSTANT args and slot defaults replaced by a ${VERB_PARAM}
    placeholder (2026-09-27, owner-directed: learn HOW to call, never the key). A method row keeps the verb, the
    slots and the words->value maps; a constant like {"api_key": "..."} would have been the one VALUE it kept --
    now it is the name of the environment variable the call reads at run time (p28 _meaning_run resolves it and
    fails loudly, naming the variable, when it is unset). Slot VALUE maps (solana -> SOL) are vocabulary, not
    credentials, and are left alone."""
    if not isinstance(method, dict):
        return method
    from holographic.io_and_interop.holographic_apilearn import placeholderize
    verb = str(method.get("verb", "")) or "method"
    out = dict(method)
    out["args"] = {k: (v if isinstance(v, dict) else placeholderize({k: v}, verb)[k])
                   for k, v in (method.get("args") or {}).items()}
    if method.get("slots"):
        sl = {}
        for k, spec in method["slots"].items():
            spec = dict(spec)
            if "default" in spec:
                spec["default"] = placeholderize({k: spec["default"]}, verb)[k]
            sl[k] = spec
        out["slots"] = sl
    return out


def normq(text):
    """The ladder's exact-store key: lowercase, whitespace collapsed."""
    return " ".join(str(text).lower().split())


def session_token(text):
    m = _SESSION.match(str(text))
    return m.group(0).strip() if m else ""


def _raw_tokens(text):
    t = _SESSION.sub("", str(text).lower()).replace("\u2019", "'")
    t = re.sub(r"\bwon't\b", "will not", t)
    t = re.sub(r"\bcan't\b", "can not", t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"'s\b", "", t).replace("'", "")
    out = []
    for w in re.findall(r"[a-z0-9]+", t):
        out.extend(_NEG[w].split() if w in _NEG else [w])
    return out


# single number words a person types instead of digits ("one us dollar", "five euros") -- the ones measured in
# the CLINC150 exchange-rate phrasings; multi-word numbers ("twenty five") are not parsed (kept simple)
_NUMWORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
             "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
             "fifteen": "15", "twenty": "20", "thirty": "30", "fifty": "50", "hundred": "100",
             "thousand": "1000"}


def numbers_in(text):
    """Numbers a question gives, in order: digits (thousands separators allowed: "20,000" is 20000), and single
    number words.
    FOUND ON THE WAY (2026-09-26, the direction measurement): "20,000" was read as TWO numbers, "20" and "000", so a
    model's amount "20000" matched neither and generalize_method kept it as a CONSTANT of the method -- the learned
    fx row then answered every later question with amount 20000 and escalated every amount it could not use (it held
    34-49 of CLINC150's exchange wordings). A comma between digit groups of three is a separator, never a list."""
    out = []
    for w in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|[a-z]+", str(text).lower()):
        if w[0].isdigit():
            out.append(w.replace(",", ""))
        elif w in _NUMWORDS:
            out.append(_NUMWORDS[w])
    return out


def _stem(tok):
    from holographic.semantic_router.holographic_bm25 import _normalize, _derivational_stem
    return tok if tok.isdigit() else _derivational_stem(_normalize(tok))


def content_tokens(text):
    """Content words of a question, in order (stopwords dropped, single letters dropped, digits kept)."""
    return [t for t in _raw_tokens(text) if t not in _STOP and (len(t) > 1 or t.isdigit())]


# ------------------------------------------------------------------------------------------------ slots
def mask_slots(tokens, slot_vocab):
    """Replace every known slot VALUE with its slot name, so 'price of sol' and 'price of eth' are the same
    wording to the matcher once both tickers are known values of the same slot. slot_vocab: {value: slot}.
    Multi-word values ("new york") are matched greedily, longest first."""
    out, i = [], 0
    longest = max((len(v.split()) for v in slot_vocab), default=1)
    while i < len(tokens):
        hit = None
        for n in range(min(longest, len(tokens) - i), 0, -1):
            cand = " ".join(tokens[i:i + n])
            if cand in slot_vocab:
                hit = (cand, n)
                break
        if hit:
            out.append("<%s>" % slot_vocab[hit[0]])
            i += hit[1]
        else:
            out.append(tokens[i])
            i += 1
    return out


def meaning_features(text, slot_vocab=None):
    """-> {feature: weight}. w: content stems (1.0); b: stem bigrams (0.5); c: char 4-grams of the raw
    content words (0.6 shared across a word's grams, words of 5+ letters only -- short words are all noise)."""
    toks = content_tokens(text)
    if slot_vocab:
        toks = mask_slots(toks, slot_vocab)
    # A NUMBER IS A VALUE, NOT A MEANING: "20 dollars" and "50 dollars" ask the same thing. (The reflex does the
    # opposite on purpose -- for EXACT recall digits are identifiers; this rung is about what a question means.)
    toks = ["<num>" if re.fullmatch(r"\d+(?:\.\d+)?", t) else t for t in toks]
    f = {}
    stems = [t if t.startswith("<") else _stem(t) for t in toks]
    for s in stems:
        f["w:" + s] = f.get("w:" + s, 0.0) + 1.0
    for a, b in zip(stems, stems[1:]):
        k = "b:%s_%s" % (a, b)
        f[k] = f.get(k, 0.0) + 0.5
    for t in toks:
        if t.startswith("<") or len(t) < 5:
            continue
        w = "^" + t + "$"
        grams = [w[i:i + 4] for i in range(len(w) - 3)]
        for g in grams:
            k = "c:" + g
            f[k] = f.get(k, 0.0) + 0.6 / len(grams)
    return f


def _rid(canonical, kind):
    return hashlib.sha256(("%s|%s" % (kind, normq(canonical))).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ learned rows
class RowPrototypes:
    """THE SHARED CONTRASTIVE RULE ON THE MEANING ROWS (CLM backlog E1.1, third consumer) -- holographic_protostore.
    ProtoStore's update, in the MeaningIndex's OWN sparse IDF space.

    WHY NOT A PLAIN ProtoStore: a row's prototype must start at its centroid in the index's sparse feature space
    (29,041 features on the learned CLINC150 index) and the InfoNCE update makes the learned part DENSE -- measured on
    that index, one pass over its wordings leaves 17.5 M non-zeros of a 17.5 M (features x rows) delta; a top-16
    candidate set still 1.1 M. A dict-of-dicts would need gigabytes; a dense features x rows float64 matrix is 140 MB.
    And the centroid itself MOVES as wordings are linked and the IDF shifts. So the prototype is split in two:

        P_k = C_k + D_k          C_k  the row's live unit centroid (exact, sparse -- what the index already scores)
                                 D_k  the LEARNED delta, held in a signed hashed projection of the same space
                                      (DIM buckets, sha256 of the feature name -> bucket and sign)
        cos(q, P_k) = (q.C_k + h(q).D_k) / sqrt(1 + 2 h(C_k).D_k + |D_k|^2)

    A row no verdict has touched has D_k = 0 and reads EXACTLY its centroid (the readout is bit-identical to the
    index without this class). The update per verdict (q = the question, y = the row the teacher named):
        p = softmax(cos(q, P) / tau) over the rows of the question's session,  t = onehot(y) (or the noise-corrected
        target, eps > 0: t_k ~ p_k * T(k -> y), ProtoStore's form),  D_k += lr * (t_k - p_k) * h(q)
    rows that share y's ANSWER KEY are never pushed (two rows that serve one answer are not rivals), and |t - p| <
    1e-4 touches nothing (ProtoStore's cut). negative(q, k): D_k -= lr * weight * h(q) (a labelled negative, E2.1).

    MEASURED (scratch probes on the learned CLINC150 index, 602 rows, 8,591 wordings; the panel's rule tau 0.05,
    eta/tau = 0.6 -- the step here is lr = 0.6):
        exact dense delta, 1 pass over the wordings      +1.67 top-1 points [+0.91, +2.36]
        hashed delta DIM 2048 / 4096 / 8192, 1 pass      +1.40 / +1.56 / +1.67
        exact dense delta, 4 passes (the panel's config) +2.07 [+1.33, +2.73]
    The hashing costs 0.27 points at 2048 and nothing at 8192. Through THIS class and MeaningIndex.answers (the
    serving path, one candidate per answer key) under the E0.4 gate protocol, one pass, 3 seeds: CLINC150 learned
    index 0.8413 -> +2.71..+3.13 points, Banking77 learned index 0.8450 -> +2.55..+3.32, AURC lower with CI on both
    -- but at the calibrated thresholds it serves more out-of-scope questions than the bar allows (it serves far more
    of everything), so the mind ships it OFF (UnifiedMind.MEANING_PROTOS, part 28 has the full table;
    docs/research/evidence/bench_meaning_protos.json).

    numpy + stdlib only, deterministic (sha256 buckets, stable sorts); state() / from_state() round-trip exactly."""

    def __init__(self, dim=2048, tau=0.05, lr=0.6):
        self.dim, self.tau, self.lr = int(dim), float(tau), float(lr)
        self.rows = []                 # rid per row of D (insertion order = the tie order)
        self._ix = {}
        self.D = np.zeros((0, self.dim))
        self.DD = np.zeros(0)          # |D_k|^2 per row, kept exact as D moves (the prototype's norm needs it)
        self.count = []                # verdicts that named this row (its maturity)
        self.n_updates = 0
        self.n_negatives = 0
        self._hb = {}                  # feature name -> (bucket, sign), cached (sha256 is the only hash used)

    def hash_of(self, feat):
        """(bucket, sign) of a feature NAME: sha256, never hash() (PYTHONHASHSEED-proof)."""
        h = self._hb.get(feat)
        if h is None:
            d = hashlib.sha256(feat.encode("utf-8")).digest()
            h = self._hb[feat] = (int.from_bytes(d[:8], "big") % self.dim, 1.0 if d[8] & 1 else -1.0)
        return h

    def project(self, items):
        """h(x) for a sparse vector given as [(feature name, value)] -> a dense DIM vector."""
        out = np.zeros(self.dim)
        for feat, v in items:
            b, s = self.hash_of(feat)
            out[b] += s * v
        return out

    def row(self, rid, create=False):
        """Row index of rid in D (appended as a zero delta when create=True), or None."""
        i = self._ix.get(rid)
        if i is None and create:
            i = self._ix[rid] = len(self.rows)
            self.rows.append(rid)
            self.D = np.vstack([self.D, np.zeros((1, self.dim))])
            self.DD = np.append(self.DD, 0.0)
            self.count.append(0)
        return i

    def active(self):
        """True once any row carries a learned delta (before that the readout is the plain centroid)."""
        return bool(self.rows) and bool(np.any(self.D))

    def forget(self, rid):
        """A dropped row loses its delta (the slot stays so indices never shift; it is zeroed)."""
        i = self._ix.get(rid)
        if i is not None:
            self.D[i] = 0.0
            self.DD[i] = 0.0

    def state(self):
        """(meta, arrays): meta is JSON-able, the delta travels as a float64 array (bit-exact reload)."""
        return ({"dim": self.dim, "tau": self.tau, "lr": self.lr, "rows": list(self.rows), "count": list(self.count),
                 "n_updates": self.n_updates, "n_negatives": self.n_negatives},
                {"D": np.asarray(self.D, np.float64), "DD": np.asarray(self.DD, np.float64)})

    @classmethod
    def from_state(cls, meta, arrays):
        rp = cls(meta.get("dim", 2048), meta.get("tau", 0.05), meta.get("lr", 0.6))
        rp.rows = list(meta.get("rows") or [])
        rp._ix = {r: i for i, r in enumerate(rp.rows)}
        D = arrays.get("D") if arrays else None
        rp.D = np.asarray(D, np.float64).reshape(len(rp.rows), rp.dim) if D is not None and len(rp.rows) \
            else np.zeros((len(rp.rows), rp.dim))
        DD = arrays.get("DD") if arrays else None
        # |D|^2 is carried (not recomputed) so a reload reads the incrementally-kept norm bit for bit
        rp.DD = np.asarray(DD, np.float64).reshape(-1) if DD is not None and len(DD) == len(rp.rows) \
            else (np.einsum("ij,ij->i", rp.D, rp.D) if len(rp.rows) else np.zeros(0))
        rp.count = list(meta.get("count") or [0] * len(rp.rows))
        rp.n_updates = int(meta.get("n_updates", 0))
        rp.n_negatives = int(meta.get("n_negatives", 0))
        return rp


# ------------------------------------------------------------------------------------------------ the index
class MeaningIndex:
    """Rows reachable by many wordings; see the module docstring. Pure data + NumPy; the mind wires it into the
    answer ladder (holographic_zoo.AnswerLadder) and persists it (lecore.learning.meaning)."""

    # PRIOR GATE (before enough verdicts exist to calibrate). Chosen on the CLINC150 *validation* split with ONE
    # taught wording per intent, for precision >= 0.97 -- and reported on the test splits of CLINC150 AND
    # Banking77, never on the split it was chosen on. MEASURED (val, 1 wording/intent): score>=0.80 & lead>=0.15
    # serves 1.4% at 0.977 precision; the first guess (0.45 / 0.10) served 15% at 0.82 -- and 6.9% WRONG on
    # Banking77's look-alike intents. Cold start is deliberately near-exact: until verdicts calibrate the gate,
    # an unsure question is the MODEL's to answer (and memory learns from its verdict).
    SERVE_SCORE = 0.80
    SERVE_GAP = 0.15
    ESCALATE_FLOOR = 0.12       # below this nothing in memory is related: plain escalation, no candidates
    P_SERVE = 0.95              # once calibrated: serve only at >= 95% estimated correctness
    CALIB_MIN = 40              # verdicts needed (with both outcomes present) before the calibrator gates
    ASSOC_WEIGHT = 0.5          # how strongly a learned association widens a question
    ASSOC_TOP = 3               # learned partners per word used for expansion
    BLEND = 0.8                 # row score = (1-BLEND) * best phrasing + BLEND * row centroid
    # E2.1: a new answer row with an existing row's answer key joins that row (add_row's MERGE RULE). OFF BY DEFAULT --
    # MEASURED (tools/bench_meaning.py online, perfect stand-in teacher, everything else equal): CLINC150 52.7% / 1.6%
    # wrong / 2.4% out-of-scope served with it vs 48.3% / 1.6% / 2.0% without (191 rows vs 604), but Banking77 25.5% /
    # 1.9% / 0.1% vs 37.2% / 2.1% / 1.0% (83 rows vs 437): one row per intent blurs the centroid of Banking77's
    # look-alike intents, the runner-up rises, and the calibrated gate serves a THIRD less. Kept as a per-mind switch.
    MERGE_SAME_ANSWER = False

    def __init__(self):
        self.rows = {}              # rid -> {kind, canonical, session, phrasings[], answer?, method?, clarify?, provenance}
        self._pid_row = []          # phrasing id -> rid
        self._pid_text = []         # phrasing id -> text
        self._by_norm = {}          # normq(phrasing) -> pid  (a wording belongs to exactly one row)
        self._vocab = {}            # feature -> fid
        self._fnames = []           # fid -> feature
        self._ent_pid = []          # sparse entries: (pid, fid, value), grown in chunks
        self._ent_fid = []
        self._ent_val = []
        self._row_feats = {}        # rid -> set(fid)   (for IDF over rows)
        self._df = []               # fid -> number of rows containing it
        self._arrays = None         # cached numpy views, rebuilt when entries change
        self._norm_epoch = -1
        self.assoc = {}             # word stem -> {partner stem: strength}  (learned from confirmed links)
        self.calib = []             # [(confidence g, correct 0/1)] from verdicts, newest last (cap 1024)
        self._cal_fit = None
        self.slot_vocab = {}        # words for a slot value (lowercase) -> slot name   (grows as methods learn)
        self.slot_values = {}       # slot name -> {words: tool value}, pooled across methods
        self.stats = {"served": 0, "escalated": 0, "clarified": 0, "linked": 0, "new_rows": 0,
                      "verdict_same": 0, "verdict_new": 0, "verdict_unclear": 0, "verdict_bad": 0}
        self.protos = None          # RowPrototypes once enable_protos() (E1.1 consumer 3); None = the plain index
        self._pcache = None         # cached cross terms of the learned rows (rebuilt with the index or the delta)
        self.teacher = None         # holographic_protostore.TeacherNoise (E2.3), created on the first re-ask
        self.merged = 0             # wordings that joined an existing row with the same answer instead of a new row
        self._loading = False       # from_state: rows come back exactly as saved (no merge on the way in)
        self._akey_rows = {}        # (answer key, session token) -> [rids], oldest first (the merge rule's lookup)
        self.off_domain = []        # E1.3: [normalised wording, verdict kind] the model marked off-domain (recorded only)

    # ---------------------------------------------------------------- building
    def _fid(self, feat):
        f = self._vocab.get(feat)
        if f is None:
            f = len(self._fnames)
            self._vocab[feat] = f
            self._fnames.append(feat)
            self._df.append(0)
        return f

    def add_row(self, canonical, kind="answer", answer=None, method=None, clarify=None, provenance="taught",
                akey=None):
        """Create (or return) the row whose canonical wording is `canonical`. Idempotent: replaying the
        taught log re-adds rows that already exist and changes nothing.
        akey: the ANSWER KEY -- rows that would serve the same answer text share it. The decision is about
        which ANSWER to serve, so two rows with one answer are not rivals (see decide).

        THE MERGE RULE (backlog E2.1, "merge duplicate rows that share an answer key when safe") -- OFF BY DEFAULT
        (MERGE_SAME_ANSWER: it cost Banking77 a third of its coverage, see the class constant). When on, a NEW answer row
        whose answer key AND session token equal an existing answer row's is not created -- its wording joins the
        OLDEST such row as a confirmed wording (link learn=True: associations, and the learned prototype when
        enabled) and that row's id is returned. Why it is safe:
          * the same answer key = the same (normalised) answer text would be served either way;
          * the session token must match (sessions are a privacy boundary, never merged across);
          * kind "answer" only: a method's key "m:<verb>" names the TOOL, not its arguments or slots, and a clarify
            row's key is its own id -- neither is ever merged;
          * the ladder's exact store still holds the question itself, so an exact repeat is served as before; and
            the meaning rung serves a merged row from ANY of its wordings whose exact answer is still live (p28
            _meaning_answer), so vetoing the oldest wording's answer does not orphan the rest.
        Loading (from_state) never merges: an old partition's duplicate rows come back as saved (answers() already
        ranks them as ONE candidate per answer key). Why it matters: with a perfect teacher 91% of `new` verdicts on
        CLINC150 were in-scope DUPLICATES -- the right row was not among the eight candidates shown, so the teacher
        said `new` with the same answer, and the intent split into rows (602 rows for 151 intents)."""
        rid = _rid(canonical, kind)
        row = self.rows.get(rid)
        if row is None and kind == "answer" and akey is not None and not self._loading and self.MERGE_SAME_ANSWER:
            sess = session_token(canonical)
            for old in self._akey_rows.get((str(akey), sess), ()):
                if old in self.rows and self.rows[old]["kind"] == "answer" and self.rows[old].get("akey") == str(akey):
                    self.link(old, canonical, learn=True)
                    self.merged += 1
                    return old
        if row is None:
            row = {"kind": kind, "canonical": str(canonical), "session": session_token(canonical),
                   "phrasings": [], "provenance": provenance}
            self.rows[rid] = row
            self._row_feats[rid] = set()
            self.stats["new_rows"] += 1
        if akey is not None:
            row["akey"] = str(akey)
        elif "akey" not in row:
            row["akey"] = ("m:" + str((method or {}).get("verb", ""))) if kind == "method" else rid
        lst = self._akey_rows.setdefault((row["akey"], row["session"]), [])
        if rid not in lst:
            lst.append(rid)
        if answer is not None:
            row["answer"] = answer
        if method is not None:
            row["method"] = method
            for sname, spec in (method.get("slots") or {}).items():
                for surf, val in (spec.get("values") or {}).items():
                    self._add_surface(sname, surf, val)
        if clarify is not None:
            row["clarify"] = clarify
        self.link(rid, canonical, learn=False)
        return rid

    def link(self, rid, phrasing, learn=True, eps=0.0, train=True):
        """Add a wording to a row. learn=True (a CONFIRMED new wording) also learns word associations
        between the new wording and the row's closest existing phrasing, and -- when the learned row prototypes
        are enabled (enable_protos) and train=True -- moves them by the shared contrastive rule (learn_verdict;
        eps = the teacher's estimated flip rate for the noise-corrected target, E2.3). Every confirmed wording
        passes here (a `same` verdict, a correction by decision id, a merged `new`), so every verdict trains the
        rows. Returns True if it was new."""
        if rid not in self.rows:
            return False
        n = normq(phrasing)
        if n in self._by_norm:
            return False                        # already a phrasing (of this or another row): first owner wins
        if learn:
            self._learn_assoc(rid, phrasing)
        pid = len(self._pid_row)
        self._pid_row.append(rid)
        self._pid_text.append(str(phrasing))
        self._by_norm[n] = pid
        self.rows[rid]["phrasings"].append(str(phrasing))
        for feat, val in meaning_features(phrasing, self.slot_vocab).items():
            fid = self._fid(feat)
            self._ent_pid.append(pid)
            self._ent_fid.append(fid)
            self._ent_val.append(val)
            if fid not in self._row_feats[rid]:
                self._row_feats[rid].add(fid)
                self._df[fid] += 1
        self._arrays = None
        if learn:
            self.stats["linked"] += 1
            if train and self.protos is not None:
                # AFTER the wording joined its row: the rule then sees the row as the panel trained it (the wording in
                # its own centroid), and pushes only the rows that still compete for it
                self.learn_verdict(phrasing, rid, eps=eps)
        return True

    def drop_row(self, rid):
        """Forget a row (a vetoed answer). Its phrasings stop matching; the wording slots are freed."""
        row = self.rows.pop(rid, None)
        if row is None:
            return False
        for p in row["phrasings"]:
            self._by_norm.pop(normq(p), None)
        for fid in self._row_feats.pop(rid, set()):
            self._df[fid] -= 1
        if self.protos is not None:
            self.protos.forget(rid)
            self._pcache = None
        self._arrays = None
        return True

    # ---------------------------------------------------------------- learned row prototypes (E1.1 consumer 3)
    def enable_protos(self, dim=2048, tau=0.05, lr=0.6):
        """Turn on the LEARNED row prototypes (RowPrototypes): every confirmed wording then moves the rows by the
        shared InfoNCE rule, and rank() reads cos(q, centroid + learned delta) in place of the plain centroid. A
        row no verdict touched reads exactly as before. Idempotent; returns the RowPrototypes."""
        if self.protos is None:
            self.protos = RowPrototypes(dim=dim, tau=tau, lr=lr)
            self._pcache = None
            self._arrays = None                 # the next build keeps the centroid entries the cross term needs
        return self.protos

    def _fid_hash(self):
        """(bucket, sign) arrays over the CURRENT feature ids (extended as the vocabulary grows)."""
        hb = getattr(self, "_fhb", None)
        n = len(self._fnames)
        if hb is None or len(hb[0]) != n:
            b0 = list(hb[0]) if hb is not None else []
            s0 = list(hb[1]) if hb is not None else []
            for f in self._fnames[len(b0):]:
                b, s = self.protos.hash_of(f)
                b0.append(b)
                s0.append(s)
            hb = self._fhb = (np.asarray(b0, np.int64), np.asarray(s0, np.float64))
        return hb

    def _qhash(self, q, qn):
        """h(q / qn): the question's unit sparse vector (as rank() normalises it) in the hashed delta space."""
        hb, hs = self._fid_hash()
        out = np.zeros(self.protos.dim)
        if q:
            f = np.fromiter(q.keys(), np.int64, len(q))
            v = np.fromiter(q.values(), np.float64, len(q)) / max(qn, 1e-12)
            np.add.at(out, hb[f], hs[f] * v)
        return out

    def _proto_terms(self):
        """The cross terms the learned rows need, valid for ONE index build: pmap (index row -> delta row, -1 = no
        delta), HC (h(C_k) per delta row: the row's live unit centroid in the hashed space) and CD = h(C_k).D_k.
        C_k moves with every link and every IDF shift, so HC / CD are rebuilt with the index -- never carried over
        stale; WITHIN a build, learn_verdict / proto_negative update CD in place (exactly: CD_k += a * HC_k.h(q)),
        so a verdict does not pay a rebuild. |D_k|^2 (RowPrototypes.DD) does not depend on the index at all."""
        A = self._build()
        if self._pcache is not None and self._pcache.get("arrays") is A:
            return self._pcache
        rp = self.protos
        rids = list(A["rix"])
        pm = np.array([rp._ix.get(r, -1) for r in rids], np.int64) if rids else np.zeros(0, np.int64)
        T = {"arrays": A, "pmap": pm, "HC": np.zeros((len(rp.rows), rp.dim)), "CD": np.zeros(len(rp.rows))}
        self._pcache = T
        uk, agg, base = A.get("uk"), A.get("agg"), A.get("kbase")
        if uk is not None and len(uk) and np.any(pm >= 0):
            # every delta row's hashed centroid in ONE scatter (a per-row loop cost ~30 ms per build at 600 rows)
            rr = uk // base
            ff = uk % base
            di = pm[rr]
            keep = di >= 0
            hb, hs = self._fid_hash()
            np.add.at(T["HC"], (di[keep], hb[ff[keep]]),
                      hs[ff[keep]] * agg[keep] / np.maximum(A["cnorm"][rr[keep]], 1e-12))
            T["CD"] = np.einsum("ij,ij->i", T["HC"], rp.D)
        return T

    def _proto_row_terms(self, T, r, i):
        """Fill HC[i] = h(C_r) and CD[i] = HC[i].D[i] for index row r (delta row i) from this build's centroid
        entries (uk is sorted by row, so a row's entries are one contiguous slice)."""
        A = T["arrays"]
        uk, agg, base = A.get("uk"), A.get("agg"), A.get("kbase")
        if uk is None or not len(uk):
            return
        lo, hi = np.searchsorted(uk, r * base), np.searchsorted(uk, (r + 1) * base)
        hb, hs = self._fid_hash()
        ff = uk[lo:hi] % base
        h = np.zeros(self.protos.dim)
        np.add.at(h, hb[ff], hs[ff] * agg[lo:hi] / max(float(A["cnorm"][r]), 1e-12))
        T["HC"][i] = h
        T["CD"][i] = float(h @ self.protos.D[i])

    def _proto_delta_row(self, T, rid):
        """The delta row of rid (created -- zero -- on first touch), with this build's cross terms extended."""
        rp = self.protos
        i = rp.row(rid)
        if i is None:
            i = rp.row(rid, create=True)
            T["HC"] = np.vstack([T["HC"], np.zeros((1, rp.dim))])
            T["CD"] = np.append(T["CD"], 0.0)
            r = T["arrays"]["rix"].get(rid)
            if r is not None:
                T["pmap"][r] = i
                self._proto_row_terms(T, r, i)
        return i

    def _proto_add(self, T, i, a, qh, qq):
        """D_i += a * qh, keeping DD (|D_i|^2) and CD (h(C_i).D_i) exact without a rebuild. qq = qh.qh."""
        rp = self.protos
        qd = float(rp.D[i] @ qh)
        rp.DD[i] += 2.0 * a * qd + a * a * qq
        T["CD"][i] += a * float(T["HC"][i] @ qh)
        rp.D[i] += a * qh

    def _proto_cos(self, cen, q, qn):
        """cos(q, C_k + D_k) for every index row: the plain centroid cosine where a row has no learned delta
        (bit-identical), the learned prototype's cosine where it has one."""
        T = self._proto_terms()
        pm = T["pmap"]
        has = pm >= 0
        if not has.any():
            return cen
        qh = self._qhash(q, qn)
        i = pm[has]
        out = np.array(cen, np.float64, copy=True)
        qd = self.protos.D[i] @ qh
        den = np.sqrt(np.maximum(1.0 + 2.0 * T["CD"][i] + self.protos.DD[i], 1e-12))
        out[has] = (out[has] + qd) / den
        return out

    def _row_scores(self, q, qn):
        """-> (arrays, per-phrasing cosine, per-row best phrasing, per-row prototype cosine) -- rank()'s scoring,
        shared with learn_verdict so the rule trains exactly what the index serves."""
        A = self._build()
        scores = np.zeros(len(self._pid_row))
        st, P, Wt = A["starts"], A["p_pid"], A["p_w"]
        for fid, qv in q.items():
            a, b = st[fid], st[fid + 1]
            if b > a:
                np.add.at(scores, P[a:b], Wt[a:b] * qv)
        scores = scores / (np.maximum(A["norm"], 1e-12) * qn)
        scores[~A["live"]] = 0.0
        # per ROW: the best single phrasing (kNN) blended with the row centroid (Rocchio). kNN alone is
        # brittle with few wordings; the centroid rewards the words a row's wordings AGREE on.
        prow, nr = A["prow"], len(A["rix"])
        ok = prow >= 0
        best_p = np.zeros(nr)
        np.maximum.at(best_p, prow[ok], scores[ok])
        cen = np.bincount(prow[ok], weights=scores[ok] * A["norm"][ok] / np.maximum(A["norm"][ok], 1e-12),
                          minlength=nr) / np.maximum(A["cnorm"], 1e-12)
        if self.protos is not None and self.protos.active():
            cen = self._proto_cos(cen, q, qn)
        return A, scores, best_p, cen

    def learn_verdict(self, text, rid, eps=0.0, m=None):
        """ONE VERDICT through the shared rule: `text` means row `rid`. p = softmax(cos(q, P) / tau) over the rows
        of the question's session; t = onehot(rid), or with eps > 0 the noise-corrected posterior t_k ~ p_k *
        T(k -> rid) (T = 1 - eps on the named row, eps / m elsewhere: ProtoStore.update's form); rows sharing the
        named row's answer key are never pushed; D_k += lr * (t_k - p_k) * h(q) where |t - p| >= 1e-4.
        -> {pred, correct, p_truth, touched} (None when the rule is off or the question has no known feature)."""
        rp = self.protos
        if rp is None or rid not in self.rows:
            return None
        q, qn = self._query_vec(text)
        if not q or qn <= 0:
            return None
        A, _, _, pc = self._row_scores(q, qn)
        rids = list(A["rix"])
        sess = session_token(text)
        cand = np.array([i for i, r in enumerate(rids) if self.rows[r]["session"] == sess], np.int64)
        yi_all = A["rix"].get(rid)
        if yi_all is None or not len(cand) or yi_all not in set(cand.tolist()):
            return None
        s = pc[cand]
        yi = int(np.flatnonzero(cand == yi_all)[0])
        z = (s - s.max()) / max(rp.tau, 1e-9)
        p = np.exp(z)
        p /= p.sum()
        if eps > 0:
            mm = m if m is not None else max(1, min(len(cand), 8) - 1)
            t = p * (float(eps) / mm)
            t[yi] = p[yi] * (1.0 - float(eps))
            t = t / t.sum()
        else:
            t = np.zeros_like(p)
            t[yi] = 1.0
        g = t - p
        ky = self.rows[rid].get("akey")
        if ky is not None:
            same = np.array([self.rows[rids[int(c)]].get("akey") == ky for c in cand]) & (cand != yi_all)
            g[same] = 0.0                       # two rows that serve ONE answer are not rivals: never pushed
        sel = np.flatnonzero(np.abs(g) > 1e-4)
        T = self._proto_terms()
        if len(sel):
            qh = self._qhash(q, qn)
            qq = float(qh @ qh)
            for j in sel:
                self._proto_add(T, self._proto_delta_row(T, rids[int(cand[j])]), rp.lr * float(g[j]), qh, qq)
        yrow = self._proto_delta_row(T, rid)
        rp.count[yrow] += 1
        rp.n_updates += 1
        pred = rids[int(cand[int(np.argmax(s))])]
        return {"pred": pred, "correct": pred == rid, "p_truth": float(p[yi]), "touched": int(len(sel))}

    def proto_negative(self, text, rid, weight=1.0):
        """A LABELLED NEGATIVE (backlog E2.1): `text` does NOT mean row `rid` (a wrong meaning serve, reported).
        ProtoStore.negative's rule on the learned delta: D_rid -= lr * weight * h(q). -> {pushed, ...}."""
        rp = self.protos
        if rp is None or rid not in self.rows:
            return {"pushed": False, "why": "no learned prototypes" if rp is None else "unknown row"}
        q, qn = self._query_vec(text)
        if not q or qn <= 0:
            return {"pushed": False, "why": "no known feature"}
        T = self._proto_terms()
        qh = self._qhash(q, qn)
        self._proto_add(T, self._proto_delta_row(T, rid), -rp.lr * float(weight), qh, float(qh @ qh))
        rp.n_negatives += 1
        return {"pushed": True, "row": rid}

    def consolidate_protos(self, epochs=1, seed=0):
        """IDLE-TIME CONSOLIDATION: replay every stored wording (each one a confirmed verdict: its row is its label)
        through learn_verdict, `epochs` times in a seeded order. The online rule sees each verdict once; the panel
        measured the rule at 4 passes over the index's own wordings (+2.07 vs +1.67 points for one pass on the
        learned CLINC150 index, exact delta). Deterministic (np.random.default_rng(seed)). -> {updates, epochs}."""
        if self.protos is None:
            self.enable_protos()
        pairs = [(self._pid_text[i], r) for i, r in enumerate(self._pid_row) if r in self.rows]
        rng = np.random.default_rng(int(seed))
        n = 0
        for _ in range(int(epochs)):
            for i in rng.permutation(len(pairs)):
                t, r = pairs[int(i)]
                if self.learn_verdict(t, r) is not None:
                    n += 1
        return {"updates": n, "epochs": int(epochs)}

    def _learn_assoc(self, rid, phrasing):
        """Which words of the NEW wording stand for which words of the row? Pair the words the new wording
        has that the row's closest phrasing lacks with the words that phrasing has that the new one lacks.
        Strength is split across the pairs, so one confirmation of a long rewording teaches each pair a
        little and a pair confirmed again and again becomes strong."""
        row = self.rows[rid]
        if not row["phrasings"]:
            return
        new_w = {_stem(t) for t in mask_slots(content_tokens(phrasing), self.slot_vocab) if not t.startswith("<")}
        best, best_ov = None, -1
        for p in row["phrasings"]:
            pw = {_stem(t) for t in mask_slots(content_tokens(p), self.slot_vocab) if not t.startswith("<")}
            ov = len(new_w & pw)
            if ov > best_ov:
                best, best_ov = pw, ov
        a_only, b_only = new_w - best, best - new_w
        if not a_only or not b_only or len(a_only) * len(b_only) > 36:
            return                                  # nothing to pair, or too loose to mean anything
        share = 1.0 / (len(a_only) * len(b_only))
        for a in sorted(a_only):
            for b in sorted(b_only):
                for x, y in ((a, b), (b, a)):
                    d = self.assoc.setdefault(x, {})
                    d[y] = d.get(y, 0.0) + share

    # ---------------------------------------------------------------- matching
    def _build(self):
        if self._arrays is not None:
            return self._arrays
        pid = np.asarray(self._ent_pid, np.int64)
        fid = np.asarray(self._ent_fid, np.int64)
        val = np.asarray(self._ent_val, np.float64)
        n_rows = max(len(self.rows), 1)
        df = np.asarray(self._df, np.float64)
        idf = np.log((n_rows + 1.0) / (np.maximum(df, 0.0) + 0.5))
        idf = np.maximum(idf, 0.0)
        live = np.array([r in self.rows for r in self._pid_row], bool) if self._pid_row else np.zeros(0, bool)
        w = val * (idf[fid] if len(fid) else 0.0)
        norm = np.sqrt(np.bincount(pid, weights=w * w, minlength=len(self._pid_row))) if len(pid) else np.zeros(0)
        # ROW CENTROIDS (Rocchio): the sum of a row's unit phrasing vectors. cos(q, centroid) =
        # sum_p(q . p_hat) / |centroid|, so only the centroid NORM is needed -- computed once per build.
        rix = {}
        for rid in self.rows:
            rix[rid] = len(rix)
        prow = np.array([rix.get(r, -1) for r in self._pid_row], np.int64) if self._pid_row else np.zeros(0, np.int64)
        cnorm = np.zeros(max(len(rix), 1))
        uk = agg = None
        if len(pid):
            u = w / np.maximum(norm[pid], 1e-12)                     # unit-normalised entries
            keep = prow[pid] >= 0
            key = prow[pid][keep] * (len(self._fnames) + 1) + fid[keep]
            uk, inv = np.unique(key, return_inverse=True)
            agg = np.bincount(inv, weights=u[keep])
            cnorm = np.sqrt(np.bincount(uk // (len(self._fnames) + 1), weights=agg * agg, minlength=len(rix)))
        order = np.argsort(fid, kind="stable")          # postings by feature: fid -> slice of (pid, weight)
        f_sorted = fid[order]
        starts = np.searchsorted(f_sorted, np.arange(len(self._fnames) + 1))
        self._arrays = {"idf": idf, "norm": norm, "live": live, "p_pid": pid[order], "p_w": w[order],
                        "starts": starts, "prow": prow, "rix": rix, "cnorm": cnorm}
        if self.protos is not None:
            # the centroid ENTRIES (row x feature, summed unit phrasings; / cnorm = the unit centroid), kept only
            # when learned rows need their cross term h(C_k).D_k (RowPrototypes)
            self._arrays.update(uk=uk, agg=agg, kbase=len(self._fnames) + 1)
        return self._arrays

    def _query_vec(self, text):
        """{fid: weight*idf} for a question, widened by learned associations."""
        A = self._build()
        idf = A["idf"]
        q = {}
        feats = meaning_features(text, self.slot_vocab)
        for feat, v in feats.items():
            fid = self._vocab.get(feat)
            if fid is not None:
                q[fid] = q.get(fid, 0.0) + v * idf[fid]
        # learned associations: a word the rows never use can still point at the words they do use
        for feat in list(feats):
            if not feat.startswith("w:"):
                continue
            partners = self.assoc.get(feat[2:])
            if not partners:
                continue
            for p, s in sorted(partners.items(), key=lambda kv: (-kv[1], kv[0]))[:self.ASSOC_TOP]:
                fid = self._vocab.get("w:" + p)
                if fid is None or ("w:" + p) in feats:
                    continue
                q[fid] = q.get(fid, 0.0) + self.ASSOC_WEIGHT * min(1.0, s) * idf[fid]
        # the query's norm counts ALL its words, known or not -- unknown words are evidence of a different
        # question, and ignoring them would let "price of solana stock split history" match "price of solana"
        unknown = sum((v * float(np.log(len(self.rows) + 1.5))) ** 2
                      for feat, v in feats.items() if feat not in self._vocab and feat.startswith("w:"))
        return q, math.sqrt(sum(x * x for x in q.values()) + unknown)

    def rank(self, text, k=5, session=None):
        """-> [(rid, score, best phrasing)] best first. Rows are filtered to the question's session token."""
        if not self.rows:
            return []
        self._build()
        q, qn = self._query_vec(text)
        if not q or qn <= 0:
            return []
        # the centroid term is the LEARNED prototype's cosine for rows a verdict has moved (enable_protos), the plain
        # centroid otherwise -- bit-identical to the index without learned rows (_row_scores)
        A, scores, best_p, cen = self._row_scores(q, qn)
        prow = A["prow"]
        blend = self.BLEND
        row_s = (1.0 - blend) * best_p + blend * cen
        rids = list(A["rix"])
        sess = session_token(text) if session is None else session
        # the phrasing that scored best inside each row (for the evidence shown to the model)
        best_pid = {}
        for pid in np.flatnonzero(scores > 0):
            r = prow[pid]
            if r >= 0 and (r not in best_pid or scores[pid] > scores[best_pid[r]]):
                best_pid[r] = pid
        out = []
        for r in np.argsort(-row_s, kind="stable"):
            s = float(row_s[r])
            if s <= 0.0 or len(out) >= k:
                break
            rid = rids[r]
            if self.rows[rid]["session"] != sess:
                continue
            out.append((rid, s, self._pid_text[best_pid[r]] if r in best_pid else self.rows[rid]["canonical"]))
        return out

    # ---------------------------------------------------------------- deciding
    @staticmethod
    def confidence(ranked):
        """One scalar the calibrator maps to P(correct): the top score plus its lead over the runner-up.
        `ranked` is already one row per ANSWER (see answers), so the runner-up is a real rival."""
        if not ranked:
            return 0.0
        s1 = ranked[0][1]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        return s1 + (s1 - s2)

    def answers(self, text, k=8):
        """rank(), keeping only the best row per ANSWER KEY: rows that would serve the same answer are one
        candidate (measured: without this, duplicate rows of one intent split the lead and escalated questions
        memory already knew)."""
        seen, out = set(), []
        for r in self.rank(text, k=k * 4):
            ak = self.rows[r[0]].get("akey", r[0])
            if ak in seen:
                continue
            seen.add(ak)
            out.append(r)
            if len(out) >= k:
                break
        return out

    def calibrated(self):
        ys = [c for _, c in self.calib]
        return len(ys) >= self.CALIB_MIN and 0 < sum(ys) < len(ys)

    def p_correct(self, g):
        """Isotonic P(correct | confidence), or None before calibration is possible."""
        if not self.calibrated():
            return None
        if self._cal_fit is None:
            from holographic.agents_and_reasoning.holographic_systemone import IsotonicCalibrator
            self._cal_fit = IsotonicCalibrator([g_ for g_, _ in self.calib], [c for _, c in self.calib])
        return float(self._cal_fit.predict(float(g)))

    # RETIRED (CLM backlog E2.3, 2026-09-26): the TEACHER-RELATIVE serve bar. It lowered the bar to P_SERVE x the
    # agreement "ceiling" on the most confident fifth of verdicts, floored at P_FLOOR -- and was OFF by default
    # (P_FLOOR = P_SERVE) because the ceiling cannot tell a noisy TEACHER from confusable ROWS: measured, it read
    # 0.941 (Banking77) and 0.988 (CLINC150) for a PERFECT teacher, and lowering the floor to follow it served 6.8%
    # WRONG on Banking77 at 0.90 and 8.6% at 0.75. What replaced it measures the teacher ITSELF: a re-ask with the
    # candidates permuted (TeacherNoise.eps_hat: 0.000 perfect, 0.084-0.107 at 10% planted noise) and the gate
    # serving on P(correct | g) = TeacherNoise.corrected(P(agree | g)) against the SAME absolute bar (decide).
    # P_FLOOR and ceiling() stay READABLE -- an old partition or a caller that set mind.meaning.P_FLOOR keeps
    # working and ceiling() is still reported -- but neither moves the bar any more (serve_bar is P_SERVE).
    P_FLOOR = 0.95

    def ceiling(self):
        """RETIRED as a gate input (see P_FLOOR above), kept as a diagnostic: how often the model agreed with memory's
        most confident decisions (the top fifth of labels by confidence)."""
        if len(self.calib) < 100:
            return 1.0
        top = sorted(self.calib, key=lambda x: -x[0])[:max(20, len(self.calib) // 5)]
        return sum(c for _, c in top) / float(len(top))

    def serve_bar(self):
        """The bar a calibrated P(correct) must reach to serve: P_SERVE (0.95). The teacher-relative lowering is
        retired (see P_FLOOR above); with the defaults it always returned 0.95, so nothing that ran before moves."""
        return self.P_SERVE

    # How sure the gate must be of the teacher's noise before correcting for it. 0 = the point estimate eps_hat; z > 0 =
    # the LOWEST flip rate the re-asks support at one-sided confidence z (Wilson upper bound on the agreement, solved
    # through TeacherNoise's quadratic). WHY a bound: the correction is steep exactly where it matters -- at 10% noise a
    # calibrated P(agree) of 0.8557 already means P(correct) 0.95 -- so an eps_hat read 0.03 too high (the sampling
    # error of ~100 re-asks at a 1% rate) serves bins that are only ~91% right. MEASURED (bench_meaning online, CLINC150,
    # noisy:0.1): the point estimate met the bar on 1 of 3 seeds (seed 2: eps_hat 0.1355, 2.4% wrong, 3.1% out-of-scope);
    # z = 0.5 on 0 of 2; z = 1 with the 3% re-ask rate (part 28) kept every seed within 2.0% wrong and 2.0%
    # out-of-scope -- evidence: docs/research/evidence/bench_meaning_online.json.
    TEACHER_EPS_Z = 1.0

    def teacher_eps(self):
        """The model end's flip rate the gate corrects for: TeacherNoise.eps_hat, or its conservative lower bound when
        TEACHER_EPS_Z > 0 -- 0.0 without re-asks, below 20 of them, or for a teacher that always agreed.
        The bound itself is TeacherNoise.eps_bound (backlog G2: moved there so every door corrects for a noisy teacher
        with ONE definition; the arithmetic moved unchanged, so this returns the same value bit for bit)."""
        t = self.teacher
        if t is None:
            return 0.0
        return t.eps_bound(self.TEACHER_EPS_Z)

    def note_off_domain(self, query, kind):
        """E1.3: RECORD a verdict's explicit "off_domain": true (the question is about nothing this memory covers).
        Nothing trains from it yet -- a NULL / none-of-these option stays DEFERRED (the panel): the absolute floor
        already catches what a NULL centroid catches (+0.8 / 0.0 points measured), and `new` verdicts must never
        train it (91% of them were in-scope duplicates on CLINC150). The field exists so a scalar NULL logit can be
        trained and measured later on labels that actually mean "off-domain". Newest 1,024 kept."""
        if not hasattr(self, "off_domain") or self.off_domain is None:
            self.off_domain = []
        self.off_domain.append([normq(query), str(kind)])
        del self.off_domain[:-1024]
        self.stats["off_domain"] = self.stats.get("off_domain", 0) + 1

    def observe(self, g, correct):
        """A verdict about a decision made at confidence g: was the top row the right one?"""
        self.calib.append((round(float(g), 6), 1 if correct else 0))
        del self.calib[:-1024]
        self._cal_fit = None

    def conflicts(self, text, rid):
        """Rare words of the question that point AWAY from this row: used by another row's wordings (at most a
        few rows use it -- a referent, not a function word), absent from this row's wordings, and not tied to
        them by a learned association. Unknown words are not conflicts (a rewording brings new words)."""
        row = self.rows.get(rid)
        if row is None:
            return []
        mine = set()
        for p in row["phrasings"]:
            mine |= {_stem(t) for t in mask_slots(content_tokens(p), self.slot_vocab) if not t.startswith("<")}
        rare = max(2, int(0.01 * len(self.rows)))
        out = []
        for t in mask_slots(content_tokens(text), self.slot_vocab):
            if t.startswith("<") or t.isdigit():
                continue
            w = _stem(t)
            if w in mine:
                continue
            fid = self._vocab.get("w:" + w)
            if fid is None or self._df[fid] < 1 or self._df[fid] > rare:
                continue
            if any(pw in mine for pw in (self.assoc.get(w) or {})):
                continue
            out.append(t)
        return out

    def only_slots(self, text):
        """True when a question is NOTHING BUT known slot values ("SOL", "eth?") -- it names a thing without
        saying what to do with it."""
        toks = content_tokens(text)
        if not toks or not self.slot_vocab:
            return False
        masked = mask_slots(toks, self.slot_vocab)
        return all(t.startswith("<") for t in masked)

    def decide(self, text, k=8):
        """-> {"action": "serve"|"clarify"|"escalate", "rid", "row", "ranked", "confidence", "p", "why"}"""
        ranked = self.answers(text, k=k)
        g = self.confidence(ranked)
        out = {"ranked": ranked, "confidence": round(g, 4), "p": None}
        if self.only_slots(text):
            slot_names = sorted({s for s in self.slot_vocab.values()})
            users = [rid for rid, r in self.rows.items() if r["kind"] == "method"
                     and set((r.get("method") or {}).get("slots") or {}) & set(slot_names)]
            vals = [t for t in content_tokens(text)]
            out.update(action="clarify", rid=None, why="the question only names %s -- it does not say what "
                       "to do with it" % " ".join(vals),
                       clarify=self._clarify_text(vals, users))
            self.stats["clarified"] += 1
            return out
        # A CONFIRMED WORDING IS NOT A GUESS (maple swarm run, 2026-09-27). A wording a verdict LINKED to a row (or the
        # row's own canonical) is served as that row -- it is exact memory, like the ladder's T0. MEASURED before: row
        # scores are 0.2 x best phrasing + 0.8 x centroid, so as a row gathers varied wordings an EXACT confirmed
        # wording scores LOWER (0.68-0.78 for a row with seven wordings): after a restart 15 of 48 wordings the swarm's
        # verdicts had linked escalated again, each a repeat model call for a question memory had been told the answer
        # to. Negatives still win (the mind's exact veto is checked after decide), and only answer / clarify rows take
        # this path -- a method row still goes through argument extraction.
        pid = self._by_norm.get(normq(text))
        if pid is not None and pid < len(self._pid_row):
            rid = self._pid_row[pid]
            row = self.rows.get(rid)
            if row is not None and row["session"] == session_token(text) and row["kind"] in ("answer", "clarify"):
                self.stats["exact_wording"] = self.stats.get("exact_wording", 0) + 1
                # the serve still REPORTS the gate's estimate at this confidence (p_correct on the decision record),
                # computed exactly as below; it just does not veto a wording memory was told the meaning of
                p_ex = self.p_correct(g)
                eps_ex = self.teacher_eps()
                if p_ex is not None and eps_ex > 0:
                    p_ex = self.teacher.corrected(p_ex, eps_ex)
                out["p"] = None if p_ex is None else round(p_ex, 4)
                if row["kind"] == "clarify":
                    out.update(action="clarify", rid=rid, clarify=row.get("clarify"), why="a known unclear wording")
                    self.stats["clarified"] += 1
                    return out
                out.update(action="serve", rid=rid, row=row, why="a confirmed wording of this row")
                self.stats["served"] += 1
                return out
        if not ranked or ranked[0][1] < self.ESCALATE_FLOOR:
            # nothing is lexically close -- but the nearest rows still go to the model as evidence: a rewording
            # with NO shared words ("how much money do i have" / "account balance") is exactly what only the
            # model can link, and showing it eight rows costs nothing
            out.update(action="escalate", rid=None, why="nothing in memory is lexically close")
            self.stats["escalated"] += 1
            return out
        rid, s1, _ = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        p = self.p_correct(g)
        eps = self.teacher_eps()
        if p is not None and eps > 0:
            # E2.3: the calibrator learned P(the MODEL agrees | g); a model wrong at rate eps agrees by chance too.
            # P(correct | g) = (P(agree | g) - eps/m) / (1 - eps - eps/m), held to the same absolute bar. eps = 0
            # (a perfect teacher, or fewer than 20 re-asks) leaves p untouched: bit-identical to before.
            out["p_agree"] = round(p, 4)
            p = self.teacher.corrected(p, eps)
        out["p"] = None if p is None else round(p, 4)
        if p is not None:
            bar = self.serve_bar()
            ok = p >= bar
            why = "calibrated P(%s)=%.2f %s %.2f" % ("correct, teacher noise %.3f removed" % eps if eps > 0
                                                     else "agrees with the model", p, ">=" if ok else "<", bar)
        else:
            ok = s1 >= self.SERVE_SCORE and (s1 - s2) >= self.SERVE_GAP
            why = "prior gate: score %.2f (need %.2f), lead %.2f (need %.2f)" % (
                s1, self.SERVE_SCORE, s1 - s2, self.SERVE_GAP)
        row = self.rows[rid]
        if ok:
            clash = self.conflicts(text, rid)
            if clash:
                # NEAR-MISS GUARD (found by tests/test_external_abstention.py): "what colour is my SISTER'S bike"
                # scored 0.80 against "what colour is my bike" and was served the bike's colour. A rare word of
                # the question that belongs to ANOTHER row, and that this row's wordings never use nor are
                # associated with, names a different referent -- escalate, never guess.
                ok = False
                why = "the question says %s, which belongs to other rows, not this one" % ", ".join(clash)
        if ok and row["kind"] == "clarify":
            out.update(action="clarify", rid=rid, clarify=row.get("clarify"), why="a known unclear wording")
            self.stats["clarified"] += 1
            return out
        out.update(action="serve" if ok else "escalate", rid=rid, row=row, why=why)
        self.stats["served" if ok else "escalated"] += 1
        return out

    def _clarify_text(self, vals, method_rids):
        """'What would you like to do with ETH? For example: what's the price of ETH?' -- the examples are the
        methods that take this kind of value, re-worded with the value the person actually named."""
        what = " ".join(v.upper() if len(v) <= 5 else v for v in vals)
        things = []
        for rid in method_rids[:3]:
            canon = self.rows[rid]["canonical"].rstrip(" ?.!")
            for spec in (self.rows[rid]["method"].get("slots") or {}).values():
                for surf in sorted(spec.get("values") or {}, key=lambda x: (-len(x), x)):
                    canon = re.sub(r"(?i)\b%s\b" % re.escape(surf), what, canon)
            things.append(canon + "?")
        if things:
            return "What would you like to do with %s? For example: %s" % (what, " / ".join(things))
        return "What would you like to know or do about %s?" % what

    # ---------------------------------------------------------------- methods (how the answer is found)
    # A SLOT maps the WORDS a person used to the VALUE a tool wants: {"solana": "SOL", "sol": "SOL"}. The model
    # reports both (args + from_question); the mapping is what makes the next question with those words bind
    # without the model. Values are also pooled per slot NAME across methods (self.slot_values), so learning
    # "solana -> SOL" for the price method helps every method with a "symbol" slot.

    def _add_surface(self, slot, surface, value):
        surf = " ".join(_raw_tokens(surface))
        if not surf:
            return None
        self.slot_values.setdefault(slot, {})
        new = surf not in self.slot_values[slot]
        self.slot_values[slot][surf] = value
        self.slot_vocab[surf] = slot
        if new:
            self._arrays = None
        return surf

    def generalize_method(self, query, method):
        """A model tells us HOW it answered: {"verb": "price", "args": {"symbol": "SOL"},
        "from_question": {"symbol": "solana"}, "live": true}. An argument whose words appear in the question
        (from_question, or the value itself) becomes a SLOT that maps those words to the value; a number from the
        question becomes a number slot; anything else is a constant of the method."""
        m = {"verb": str(method.get("verb", "")), "live": bool(method.get("live", False)),
             "args": {}, "slots": {}}
        text = " ".join(_raw_tokens(query))
        nums = numbers_in(query)
        surfaces = method.get("from_question") if isinstance(method.get("from_question"), dict) else {}
        for name, val in (method.get("args") or {}).items():
            sval = str(val).strip()
            if sval in nums and not surfaces.get(name):
                m["args"][name] = {"slot": name}
                m["slots"][name] = {"type": "number"}
                continue
            if name in surfaces and not str(surfaces.get(name) or "").strip():
                # the model says: this argument COMES FROM the question when the person names it, and this
                # value is what it means when they don't ("weather" -> location = where you are)
                m["args"][name] = {"slot": name}
                m["slots"][name] = {"type": "choice", "values": {}, "default": val}
                continue
            surf = " ".join(_raw_tokens(surfaces.get(name) or sval))
            if surf and not surf.replace(" ", "").isdigit() and re.search(r"(^| )%s( |$)" % re.escape(surf), text):
                m["args"][name] = {"slot": name}
                m["slots"][name] = {"type": "choice", "values": {surf: val}, "at": text.find(surf)}
                self._add_surface(name, surf, val)
            else:
                m["args"][name] = val                   # a constant of the method
        return placeholder_method(m)

    def learn_slot_value(self, rid, slot, surface, value=None):
        """A new way to say a slot value ('eth' -> 'ETH', 'ethereum' -> 'ETH'): remember it for this method and
        for every method with a slot of the same name."""
        row = self.rows.get(rid)
        if not row or row["kind"] != "method":
            return False
        spec = row["method"]["slots"].get(slot)
        if not spec or spec.get("type") != "choice":
            return False
        value = surface if value is None else value
        surf = self._add_surface(slot, surface, value)
        if surf is None:
            return False
        new = surf not in spec["values"]
        spec["values"][surf] = value
        return new

    def bind(self, rid, query, reader=None):
        """-> (args, missing): the method's arguments bound from the question (words -> tool values);
        `missing` names slots the question did not fill (then the caller asks the model, or the person).
        Two slots never bind the same words. Slots that take the same KIND of value (from / to currency) share one
        pool of words, and WHICH mention fills WHICH slot is:
          * reader (a holographic_rolecall.ContextRoles, the mind's direction reader) given and taught for these
            slot names: read from the question's CONTEXT words (E3.1 wiring). A reading with a direction fills the
            slots; a reading with NO direction ("the rate between X and Y" -- the reader's taught 'either'
            hypothesis wins) leaves the slots MISSING, so the question is clarified instead of guessed;
          * otherwise (no reader, an untaught reader -- via 'positional' -- or fewer mentions than slots): the
            order the method's first example named them -- positional, the honest minimum. MEASURED: the
            positional rule was wrong on 29 of 74 directional CLINC150 train items (w2-hd), which is why the
            reader exists; its numbers through this door are in tools/bench_meaning.py direction."""
        m = self.rows[rid]["method"]
        src = " ".join(_raw_tokens(query))
        nums = numbers_in(query)
        args, missing, used = {}, [], []
        choice = []
        for name, val in m["args"].items():
            if isinstance(val, dict) and "slot" in val:
                spec = m["slots"][val["slot"]]
                if spec["type"] == "number":
                    if nums:
                        args[name] = nums.pop(0)
                    elif not spec.get("optional"):
                        missing.append(name)
                    continue
                choice.append((spec.get("at", 10 ** 6), name, val["slot"], spec))
            else:
                args[name] = val
        # slots of ONE method that take the same kind of value (from/to currency) share one pool of words: a
        # currency first seen as "to" is still a currency when a later question puts it first
        shared = {}
        for _, _, slot, spec in choice:
            shared.update(self.slot_values.get(slot, {}))
        for _, _, slot, spec in choice:
            shared.update(spec.get("values") or {})
        order = sorted(choice, key=lambda c: (c[0], c[1]))
        if reader is not None and len(choice) > 1 and self._bind_by_context(reader, query, src, order, shared,
                                                                             args, missing, used):
            order = []                                  # the reader filled (or deliberately left open) every slot
        for _, name, slot, spec in order:
            pool = shared if len(choice) > 1 else dict(self.slot_values.get(slot, {}), **(spec.get("values") or {}))
            hits = []
            for surf in pool:
                for mt in re.finditer(r"(?:^| )(%s)(?= |$)" % re.escape(surf), src):
                    a, b = mt.start(1), mt.end(1)
                    if not any(a < ub and ua < b for ua, ub in used):
                        hits.append((a, -(b - a), surf, b))
            if hits:
                a, _, surf, b = min(hits)                       # earliest mention, longest words first
                used.append((a, b))
                args[name] = pool[surf]
            elif "default" in spec:
                args[name] = spec["default"]
            elif not spec.get("optional"):
                missing.append(name)
        self._last_unused = self._unused(src, nums, used)
        return args, missing

    def _unused(self, src, nums_left, used):
        """What the question GIVES that the bound call does not use: numbers left over, and known slot words
        (a currency, a coin) no slot took. A call that ignores part of the question is not trusted."""
        out = ["number %s" % n for n in nums_left]
        for surf in sorted(self.slot_vocab, key=lambda x: (-len(x), x)):
            for mt in re.finditer(r"(?:^| )(%s)(?= |$)" % re.escape(surf), src):
                a, b = mt.start(1), mt.end(1)
                if not any(a < ub and ua < b for ua, ub in used):
                    used = used + [(a, b)]
                    out.append(surf)
        return out

    def _bind_by_context(self, reader, query, src, order, pool, args, missing, used):
        """The E3.1 direction read for a shared value pool (see bind). Mentions = the pool's words found in the
        question, in question order (non-overlapping, longest first). Fills args / missing / used IN PLACE and
        returns True when the reader decided (a direction, or deliberately none); False = fall back to positional
        (an untaught reader, fewer mentions than slots, or a mention the reader's tokenizer cannot locate)."""
        hits = []
        for surf in pool:
            for mt in re.finditer(r"(?:^| )(%s)(?= |$)" % re.escape(surf), src):
                hits.append((mt.start(1), -(mt.end(1) - mt.start(1)), surf, mt.end(1)))
        chosen = []
        for a, _, surf, b in sorted(hits):
            if not any(a < ub and ua < b for ua, ub, _ in chosen) and not any(a < ub and ua < b for ua, ub in used):
                chosen.append((a, b, surf))
        slots = [slot for _, _, slot, _ in order]
        if len(chosen) < len(slots) or len(set(slots)) != len(slots):
            return False
        try:
            rd = reader.read(str(query), [c[2] for c in chosen], slots)
        except ValueError:
            return False
        if rd.get("via") != "context":
            return False                                # untaught for these slot names: positional, as before
        self._last_direction = {"direction": rd.get("direction"), "p": rd.get("p"), "p_none": rd.get("p_none"),
                                "assignment": rd.get("assignment")}
        if not rd.get("direction"):
            for _, name, _, _ in order:
                missing.append(name)                    # the question names the values but not which way: ASK
            used.extend((a, b) for a, b, _ in chosen[:len(slots)])
            return True
        for i, (_, name, _, _) in enumerate(order):
            a, b, surf = chosen[int(rd["order"][i])]
            used.append((a, b))
            args[name] = pool[surf]
        return True

    def bind_checked(self, rid, query, reader=None):
        """bind() plus `unused`: -> (args, missing, unused)."""
        args, missing = self.bind(rid, query, reader=reader)
        return args, missing, list(getattr(self, "_last_unused", []))

    def extend_method(self, rid, query, name, value, surface=None):
        """A 'same' verdict names an argument the method did not have ("20 yen" -> amount=20): the method grows
        an OPTIONAL slot for it -- a number slot when the value is a number in the question, a word slot when its
        words are in the question, else nothing (a constant is never inferred from one question)."""
        row = self.rows.get(rid)
        if not row or row["kind"] != "method" or name in row["method"]["args"]:
            return False
        text = " ".join(_raw_tokens(query))
        nums = numbers_in(query)
        m = row["method"]
        if str(value) in nums and not surface:
            m["args"][name] = {"slot": name}
            m["slots"][name] = {"type": "number", "optional": True}
            return True
        surf = " ".join(_raw_tokens(surface or value))
        if surf and re.search(r"(^| )%s( |$)" % re.escape(surf), text):
            m["args"][name] = {"slot": name}
            m["slots"][name] = {"type": "choice", "values": {surf: value}, "optional": True}
            self._add_surface(name, surf, value)
            return True
        return False

    # ---------------------------------------------------------------- the model end
    def resolution_prompt(self, query, ranked):
        """THE TYPED MODEL-END PROMPT for an unsure question: goal, return format, constraints, the nearest
        rows as evidence, the verification contract, the state. Pure function of its inputs."""
        cands = []
        for rid, s, ph in ranked[:8]:
            r = self.rows[rid]
            c = {"row": rid, "kind": r["kind"], "asked_as": r["phrasings"][:3], "score": round(float(s), 3)}
            if r["kind"] == "method":
                c["method"] = {"verb": r["method"]["verb"], "slots": sorted(r["method"]["slots"])}
            cands.append(c)
        lines = [
            "GOAL: decide what the QUESTION below means, relative to what memory already knows.",
            'RETURN FORMAT: one JSON object, exactly one of:',
            '  {"verdict": "same", "row": "<row id from CANDIDATES>"}   -- it asks the same thing as that row '
            '(if that row is a method: add "args": {...} and "from_question": {...} for the values this '
            'question names)',
            '  {"verdict": "new", "answer": "<the answer>", "method": {"verb": "<tool>", "args": {...}, '
            '"from_question": {"<arg>": "<the exact words in the question that gave it>"}, "live": true|false}}'
            '   -- memory does not know it; include "method" when the answer came from a tool or lookup '
            '(live=true when the value changes over time: prices, weather, balances, time)',
            '  {"verdict": "unclear", "clarify": "<one question to ask the person>"}   -- the question does not '
            'say what they want',
            "CONSTRAINTS: \"same\" only if the question would be fully answered by that row's answer; a question "
            "that merely shares words is NOT the same. Never put a password, key, seed phrase or other secret "
            "in an answer. Prefer \"unclear\" to guessing.",
            "OPTIONAL: add \"off_domain\": true to any verdict when the question is about nothing memory's rows "
            "cover (recorded only). For a method whose slots take the same kind of value (from / to), "
            "from_question names each slot's words; when the question names the values but NO direction (\"the "
            "rate between X and Y\"), give \"from_question\": {\"either\": [\"<words>\", \"<words>\"]}.",
            "CANDIDATES (nearest rows in memory, best first): %s" % json.dumps(cands),
            "VERIFICATION: the reply is parsed as JSON and checked against this contract (row must be one of the "
            "candidate ids); a violation is retried once, then the reply is used as a plain answer.",
            "QUESTION: %s" % str(query)[:2000],
        ]
        return "\n".join(lines)

    @staticmethod
    def parse_verdict(text, allowed_rids):
        """-> (verdict dict, None) or (None, why). Accepts a JSON object anywhere in the reply."""
        s = str(text or "").strip()
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return None, "no JSON object in the reply"
        try:
            v = json.loads(m.group(0))
        except ValueError as e:
            return None, "not valid JSON (%s)" % e
        kind = v.get("verdict")
        if "off_domain" in v and not isinstance(v["off_domain"], bool):
            # E1.3: the optional field is TYPED -- a bool or absent; anything else violates the contract
            return None, "off_domain must be true or false"
        if kind == "same":
            if v.get("row") not in allowed_rids:
                return None, "row %r is not one of the candidates" % v.get("row")
            return v, None
        if kind == "new":
            if not str(v.get("answer") or "").strip() and not isinstance(v.get("method"), dict):
                return None, "a new verdict needs an answer or a method"
            if v.get("method") is not None and not (isinstance(v["method"], dict) and v["method"].get("verb")):
                return None, "method needs a verb"
            return v, None
        if kind == "unclear":
            if not str(v.get("clarify") or "").strip():
                return None, "an unclear verdict needs a clarifying question"
            return v, None
        return None, "verdict must be same, new or unclear"

    # ---------------------------------------------------------------- persistence
    def state(self):
        """JSON-able meta. The learned row prototypes' delta is an ARRAY (state_arrays) -- the container keeps it
        beside the meta, never as JSON numbers. Optional keys (absent when unused, so an older reader and an
        index without them round-trip exactly as before): protos (RowPrototypes meta), teacher (TeacherNoise,
        E2.3), merged (the merge rule's count)."""
        st = {"v": 1,
              "rows": [{"rid": rid, **{k: r[k] for k in ("kind", "canonical", "phrasings", "provenance")},
                        **{k: r[k] for k in ("method", "clarify", "akey") if k in r}}
                       for rid, r in self.rows.items()],
              "assoc": {a: {b: round(s, 6) for b, s in sorted(d.items())} for a, d in sorted(self.assoc.items())},
              "calib": [list(x) for x in self.calib],
              "slot_values": {k: dict(v) for k, v in sorted(self.slot_values.items())},
              "stats": dict(self.stats)}
        if self.protos is not None:
            st["protos"] = self.protos.state()[0]
        if self.teacher is not None:
            st["teacher"] = self.teacher.state()
        if self.merged:
            st["merged"] = int(self.merged)
        if getattr(self, "off_domain", None):
            st["off_domain"] = list(self.off_domain)
        return st

    def state_arrays(self):
        """The array half of the state: {"proto_D": the learned delta} when the learned rows are on, else {}."""
        if self.protos is None:
            return {}
        arr = self.protos.state()[1]
        return {"proto_D": arr["D"], "proto_DD": arr["DD"]}

    @classmethod
    def from_state(cls, st, arrays=None):
        mi = cls()
        mi._loading = True                      # rows come back exactly as saved: the merge rule never runs here
        mi.migrated = 0
        try:
            for r in (st or {}).get("rows") or []:
                if r.get("method"):
                    # MIGRATION ON LOAD (2026-09-27): a method row saved before the placeholder rule with a raw
                    # credential among its constants comes back with ${VERB_PARAM} placeholders instead
                    pm = placeholder_method(r["method"])
                    if pm != r["method"]:
                        r = dict(r, method=pm)
                        mi.migrated += 1
                ph = list(r.get("phrasings") or [])
                # the ANSWER KEY travels with the row (sweep 181 fix, found by the CLM panel swarm: it was not in the
                # state, so after a reload every row became its own answer and duplicate rows of one intent split
                # the lead -- the saved CLINC index had 602 rows and 602 keys, 8.2% of test runner-ups were the same
                # intent)
                rid = mi.add_row(r["canonical"], kind=r.get("kind", "answer"), method=r.get("method"),
                                 clarify=r.get("clarify"), provenance=r.get("provenance", "taught"),
                                 akey=r.get("akey"))
                for p in ph:
                    mi.link(rid, p, learn=False)
        finally:
            mi._loading = False
        mi.assoc = {a: dict(d) for a, d in ((st or {}).get("assoc") or {}).items()}
        mi.calib = [tuple(x) for x in (st or {}).get("calib") or []]
        for slot, vals in ((st or {}).get("slot_values") or {}).items():
            for surf, val in vals.items():
                mi._add_surface(slot, surf, val)
        mi.stats.update((st or {}).get("stats") or {})
        if (st or {}).get("protos") is not None:
            mi.protos = RowPrototypes.from_state(st["protos"], {"D": (arrays or {}).get("proto_D"),
                                                                "DD": (arrays or {}).get("proto_DD")})
            mi._arrays = None
        if (st or {}).get("teacher") is not None:
            from holographic.agents_and_reasoning.holographic_protostore import TeacherNoise
            mi.teacher = TeacherNoise.from_state(st["teacher"])
        mi.merged = int((st or {}).get("merged", 0))
        if (st or {}).get("off_domain"):
            mi.off_domain = [list(x) for x in st["off_domain"]]
        return mi


def _selftest():
    mi = MeaningIndex()
    a = mi.add_row("how do i check my account balance")
    b = mi.add_row("what's the weather like today")
    r = mi.rank("show me my balance")
    assert r and r[0][0] == a, r
    assert mi.rank("weather forecast please")[0][0] == b
    # a confirmed rewording links, never duplicates
    assert mi.link(a, "how much money do i have") and not mi.link(a, "How much money do I have")
    assert len(mi.rows) == 2 and len(mi.rows[a]["phrasings"]) == 2
    # the association learned from that link widens a NEW wording
    assert mi.assoc.get("money") and "balance" in mi.assoc["money"]
    # methods: the value in the question becomes a slot; a new question binds it; a bare value clarifies
    m = mi.generalize_method("what's the price of solana", {"verb": "price", "args": {"symbol": "SOL"},
                                                           "from_question": {"symbol": "solana"}, "live": True})
    assert m["args"]["symbol"] == {"slot": "symbol"} and m["slots"]["symbol"]["values"] == {"solana": "SOL"}
    p = mi.add_row("what's the price of solana", kind="method", method=m)
    mi.learn_slot_value(p, "symbol", "eth", "ETH")
    assert mi.bind(p, "price of eth right now") == ({"symbol": "ETH"}, [])
    assert mi.bind(p, "how much is solana") == ({"symbol": "SOL"}, [])
    assert mi.decide("ETH")["action"] == "clarify"
    # the verdict contract
    v, why = MeaningIndex.parse_verdict('ok: {"verdict": "same", "row": "%s"}' % a, [a, b])
    assert v and why is None
    assert MeaningIndex.parse_verdict('{"verdict": "same", "row": "zzz"}', [a])[0] is None
    assert MeaningIndex.parse_verdict("just an answer", [a])[0] is None
    # round trip
    mi2 = MeaningIndex.from_state(json.loads(json.dumps(mi.state())))
    assert mi2.rank("show me my balance")[0][0] == a and mi2.bind(p, "eth price")[0] == {"symbol": "ETH"}
    return "ok"


if __name__ == "__main__":
    print(_selftest())
