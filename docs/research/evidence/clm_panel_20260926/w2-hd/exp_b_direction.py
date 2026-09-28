"""exp_b_direction.py -- does FROM(x)+TO(y) separate direction, and can the STATE side carry it?

Part 1 (SYNTHETIC): engine HRR bind (circular convolution), unitary role atoms, Gaussian fillers.
Part 2 (REAL WORDING): CLINC150 exchange_rate phrasings. The bench's truth (tools/bench_meaning.py
method_truth) is POSITIONAL (from = first currency named). Here direction labels come from the
amount-adjacency cue (the currency right after a number / 'a' / 'one' is FROM), every disagreement with
the positional truth checked by eye (34 of 34 are the cue being semantically right). A holographic
context-role learner is trained on TRAIN labels and scored on held-out VAL+TEST directional items.
"""
import sys, json, re, time
import numpy as np
sys.path.insert(0, "/home/claude/leCore")
sys.path.insert(0, "/home/claude/leCore/tools")
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, cosine, derived_atom, random_vector
from bench_meaning import method_truth, _find_spans, CURRENCY, NUMWORDS

DATA = "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad/data/clinc150_full.json"
res = {}
T0 = time.perf_counter()

# ---------------------------------------------------------------- part 1: synthetic cosine gaps
for D in (2048, 1024):
    rng = np.random.default_rng(0)
    FROM, TO, VERB = (derived_atom(0, "role:" + r, D, unitary=True) for r in ("FROM", "TO", "VERB"))
    vals = np.stack([random_vector(D, rng) for _ in range(50)])
    verbs = np.stack([random_vector(D, rng) for _ in range(20)])
    swap, swap_v, floor, dec = [], [], [], 0
    for t in range(1000):
        x, y = rng.choice(50, 2, replace=False); v = rng.integers(20)
        a = bind(FROM, vals[x]) + bind(TO, vals[y]); b = bind(FROM, vals[y]) + bind(TO, vals[x])
        swap.append(cosine(a, b))
        av = a + bind(VERB, verbs[v]); bv = b + bind(VERB, verbs[v])
        swap_v.append(cosine(av, bv))
        floor.append(cosine(random_vector(D, rng), random_vector(D, rng)))
        gx = int(np.argmax(vals @ unbind(av, FROM))); gy = int(np.argmax(vals @ unbind(av, TO)))
        dec += (gx, gy) == (x, y)
    f = np.abs(np.array(floor))
    res["synthetic_D%d" % D] = {
        "cos_swapped_no_verb_mean": round(float(np.mean(swap)), 4), "sd": round(float(np.std(swap)), 4),
        "cos_swapped_with_verb_mean": round(float(np.mean(swap_v)), 4), "sd_v": round(float(np.std(swap_v)), 4),
        "cos_unordered_bag_x_plus_y": 1.0,
        "noise_floor_abs_cos_mean": round(float(f.mean()), 4), "noise_floor_abs_cos_p99": round(float(np.quantile(f, 0.99)), 4),
        "decode_from_to_exact": dec / 1000.0}

# ---------------------------------------------------------------- part 2: the state side, real wording
d = json.load(open(DATA))
AMT = set(NUMWORDS) | {"a", "one"}


def toks(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def label(t):
    """-> (kind, from, to, mentions[(code, surf, pos)]) or None. kind 'sym' = no direction in the words."""
    if not method_truth("exchange", t):
        return None
    codes = []
    for pos, surf, code in _find_spans(t, CURRENCY):
        if code not in [c for c, _, _ in codes]:
            codes.append((code, surf, pos))
    s = " " + " ".join(toks(t)) + " "
    frm = None
    for code, surf, pos in codes:
        left = s[:pos].split()
        if left and (left[-1] in AMT or re.fullmatch(r"\d+(\.\d+)?", left[-1])):
            frm = code
            break
    if frm is None:
        if re.search(r"\b(between|and|vs)\b", s) and not re.search(r"\bto\b", s):
            return ("sym", codes[0][0], codes[1][0], codes[:2])
        return ("dir", codes[0][0], codes[1][0], codes[:2])
    return ("dir", frm, [c for c, _, _ in codes if c != frm][0], codes[:2])


D = 2048
ROLES = {r: derived_atom(0, "ctx:" + r, D, unitary=True) for r in ("L1", "L2", "R1")}


def w(tok):
    tok = "<num>" if re.fullmatch(r"\d+(\.\d+)?", tok) else tok
    return derived_atom(0, "w:" + tok, D)


def ctx(text, pos, surf):
    """The mention's context as a role-filler record: LEFT1 (*) word-before + LEFT2 (*) two-before + RIGHT1 (*) after."""
    s = " " + " ".join(toks(text)) + " "
    left = s[:pos].split(); right = s[pos + len(surf):].split()
    parts = []
    for role, tok in (("L1", left[-1] if left else "<s>"), ("L2", left[-2] if len(left) > 1 else "<s>"),
                      ("R1", right[0] if right else "</s>")):
        parts.append(bind(ROLES[role], w(tok)))
    v = np.sum(parts, axis=0)
    return v / np.linalg.norm(v)


def items(split):
    out = []
    for t, i in d[split]:
        if i != "exchange_rate":
            continue
        lab = label(t)
        if lab and lab[0] == "dir":
            out.append((t, lab))
    return out


train, held = items("train"), items("val") + items("test")
P = {"FROM": np.zeros(D), "TO": np.zeros(D)}
for t, (_, frm, to, ms) in train:
    for code, surf, pos in ms:
        P["FROM" if code == frm else "TO"] += ctx(t, pos, surf)


def predict(t, ms):
    sc = [cosine(ctx(t, pos, surf), P["FROM"]) - cosine(ctx(t, pos, surf), P["TO"]) for code, surf, pos in ms]
    j = int(np.argmax(sc))
    return ms[j][0], ms[1 - j][0], sc


def score(rows):
    pos_ok = cue_ok = 0; per = []
    for t, (_, frm, to, ms) in rows:
        pos_ok += (ms[0][0], ms[1][0]) == (frm, to)
        f, tt, sc = predict(t, ms)
        cue_ok += (f, tt) == (frm, to)
        per.append({"q": t, "truth": "%s->%s" % (frm, to), "learned": "%s->%s" % (f, tt), "positional": "%s->%s" % (ms[0][0], ms[1][0])})
    return pos_ok, cue_ok, per


p_tr, c_tr, _ = score(train)
p_h, c_h, per_h = score(held)
test_all = [(t, label(t)) for t, i in d["test"] if i == "exchange_rate" and label(t)]
res["clinc_exchange"] = {
    "train_directional": len(train), "heldout_directional_val_test": len(held),
    "test_items_total": len(test_all), "test_symmetric": sum(1 for _, l in test_all if l[0] == "sym"),
    "positional_truth_semantically_wrong": {"train": sum(1 for t, l in train if (l[3][0][0], l[3][1][0]) != (l[1], l[2])),
                                            "test": sum(1 for t, l in test_all if l[0] == "dir" and (l[3][0][0], l[3][1][0]) != (l[1], l[2]))},
    "heldout_positional_correct": "%d/%d" % (p_h, len(held)),
    "heldout_learned_context_role_correct": "%d/%d" % (c_h, len(held)),
    "train_fit_learned": "%d/%d" % (c_tr, len(train)),
    "heldout_errors": [r for r in per_h if r["truth"] != r["learned"]]}
res["cpu_s"] = round(time.perf_counter() - T0, 1)
print(json.dumps(res, indent=1))
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
