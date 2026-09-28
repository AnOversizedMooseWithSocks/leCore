"""bench_meaning.py -- does memory find a taught answer when a PERSON rephrases the question? (sweep 181)

Real human wording, not templated test strings:
    CLINC150 (Larson et al. 2019, CC BY 3.0): 150 intents x 100 crowd-written training phrasings, 30 test
        phrasings per intent, plus 1,000 out-of-scope test questions that match NO intent.
    Banking77 (Casanueva et al. 2020, CC BY 4.0): 77 fine-grained banking intents, 3,080 test phrasings --
        deliberately confusable neighbours (card_arrival vs card_delivery_estimate, ...).
Neither dataset ships in the repo: pass their folder with --data (files: clinc150_full.json,
banking77_train.csv, banking77_test.csv; both are on GitHub -- clinc/oos-eval, PolyAI-LDN/task-specific-datasets).

MODES
  offline   teach K phrasings per intent (the first K training utterances; answer = "Here is what I know about <intent>."),
            then ask every held-out test phrasing and every out-of-scope question with NO model attached.
            Counts: served-correct, served-WRONG (the dangerous one), escalated/refused; out-of-scope
            served = wrong. This is what the memory alone can do.
  online    teach ONE phrasing per intent, then stream the training phrasings (shuffled, seed 0) through
            ask() with a MODEL END attached; every escalation is resolved by the model's typed verdict and
            learned on the go. Reports the model-call rate per 1,000 questions as the stream runs (it should
            FALL), the wrong-serve rate, and then the offline test with the model detached.
            THE MODEL IS A SCRIPTED STAND-IN (no model is reachable from the sandbox that built this):
            'oracle' answers the typed prompt from the dataset's labels; 'noisy:<rate>' gives a WRONG verdict
            with that probability. That measures the LEARNING LOOP and its robustness to a fallible model,
            not any real model's judgement.

  methods   learn HOW live questions are answered (price / weather / exchange rate) and score the calls; exchange
            direction is scored against tests/data/exchange_direction.json (the old truth was positional)
  direction E3.1: the direction reader vs the positional rule, through the meaning door, on that labelled set
  protos    E1.1 consumer 3: the learned row prototypes under the E0.4 gate protocol (tools/bench_contrastive)

Usage:
    PYTHONHASHSEED=0 python3 tools/bench_meaning.py --data DIR offline [--k 1] [--dataset clinc|banking]
    PYTHONHASHSEED=0 python3 tools/bench_meaning.py --data DIR online --model oracle|noisy:0.1 [--dataset ...]
            [--seed S] [--protos on|off] [--record]     (--record merges into evidence/bench_meaning_online.json)
    PYTHONHASHSEED=0 python3 tools/bench_meaning.py --data DIR methods [--positional] [--record]
    PYTHONHASHSEED=0 python3 tools/bench_meaning.py --data DIR direction
    PYTHONHASHSEED=0 python3 tools/bench_meaning.py --data DIR protos --index idx_clinc.json --index-banking idx_b.json

The online stand-in, the audit and the phase-C numbers (2026-09-26): docs/research/evidence/bench_meaning_online.json
(runs = the shipped defaults, baseline_before_phase_c, kept_negatives), bench_meaning_protos.json,
bench_meaning_direction.json, bench_meaning_methods.json.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------------------------------------ data
def load(data_dir, dataset):
    """-> (train {intent: [phrasings]}, test [(text, intent)], oos_test [text], dev [(text, intent)])"""
    if dataset == "clinc":
        d = json.load(open(os.path.join(data_dir, "clinc150_full.json")))
        train = {}
        for t, i in d["train"]:
            train.setdefault(i, []).append(t)
        return train, [tuple(x) for x in d["test"]], [t for t, _ in d["oos_test"]], [tuple(x) for x in d["val"]]
    if dataset == "banking":
        def rows(fn):
            with open(os.path.join(data_dir, fn), newline="") as f:
                return [(r["text"], r["category"]) for r in csv.DictReader(f)]
        tr = rows("banking77_train.csv")
        train = {}
        for t, i in tr:
            train.setdefault(i, []).append(t)
        # Banking77 has no out-of-scope split: out-of-domain CLINC questions stand in when available
        oos = []
        p = os.path.join(data_dir, "clinc150_full.json")
        if os.path.exists(p):
            oos = [t for t, _ in json.load(open(p))["oos_test"]]
        return train, rows("banking77_test.csv"), oos, []
    raise SystemExit("unknown dataset %r" % dataset)


# answers are written as ordinary sentences: a bare token like "ANSWER[gas]" reads as a credential-shaped
# value to the learning guard (measured: 5 of 150 teaches refused), which would be testing the guard, not recall
ANS = re.compile(r"^Here is what I know about ([a-z0-9_]+)\.$")


def answer_for(intent):
    return "Here is what I know about %s." % intent


def intent_of(answer):
    m = ANS.match(str(answer or "").strip())
    return m.group(1) if m else None


def classify(out, want):
    """-> 'correct' | 'wrong' | 'abstain' | 'clarify' for one ask() result (want=None: out-of-scope)."""
    tier = out.get("tier")
    if tier in (None, "refused") or not str(out.get("answer") or "").strip():
        return "abstain"
    if tier == "clarify":
        return "clarify"
    got = intent_of(out.get("answer"))
    if want is None:
        return "wrong"                       # anything served for an out-of-scope question is wrong
    return "correct" if got == want else "wrong"


def fresh_mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def tally(results):
    n = max(len(results), 1)
    c = {k: sum(1 for r in results if r == k) for k in ("correct", "wrong", "abstain", "clarify")}
    return {k: "%d (%.1f%%)" % (v, 100.0 * v / n) for k, v in c.items()}


# ------------------------------------------------------------------------------------------------ offline
def offline(args):
    train, test, oos, _ = load(args.data, args.dataset)
    m = fresh_mind()
    t0 = time.time()
    for intent in sorted(train):
        for q in train[intent][:args.k]:
            r = m.teach(q, answer_for(intent))
            if isinstance(r, dict) and r.get("taught") is False:
                print("  teach refused:", q, r.get("reason"))
    t_teach = time.time() - t0
    taught = {q for i in train for q in train[i][:args.k]}
    held = [(t, i) for t, i in test if t not in taught]
    t0 = time.time()
    res_in = [classify(m.ask(t), i) for t, i in held]
    res_oos = [classify(m.ask(t), None) for t in oos]
    t_ask = time.time() - t0
    print(json.dumps({"dataset": args.dataset, "k_taught_per_intent": args.k, "intents": len(train),
                      "in_scope_test": len(held), "in_scope": tally(res_in),
                      "out_of_scope_test": len(oos), "out_of_scope": tally(res_oos),
                      "teach_s": round(t_teach, 1), "ask_ms_per_q": round(1000 * t_ask / max(len(held) + len(oos), 1), 2)},
                     indent=1))


# ------------------------------------------------------------------------------------------------ methods
# The oracle's gazetteers: how the STAND-IN model knows what a question asks for. Real phrasings (CLINC150
# weather / exchange_rate) and the hand-written crypto set; the gazetteer is the stand-in's knowledge, not
# leCore's -- leCore only ever sees the typed verdicts.
CITIES = ["seattle", "tallahassee", "austin", "costa mesa", "sparks", "orlando", "tampa", "pittsburgh",
          "sarasota", "chicago", "denver", "georgia", "miami"]
CURRENCY = {"us dollars": "USD", "us dollar": "USD", "usd": "USD", "dollars": "USD", "dollar": "USD",
            "canadian dollars": "CAD", "canadian dollar": "CAD", "cad": "CAD",
            "australian dollars": "AUD", "aud": "AUD", "japanese yen": "JPY", "yens": "JPY", "yen": "JPY",
            "mexican pesos": "MXN", "pesos": "MXN", "peso": "MXN", "mxn": "MXN",
            "british pounds": "GBP", "pounds": "GBP", "pound": "GBP", "sterling": "GBP", "gbp": "GBP",
            "euros": "EUR", "euro": "EUR", "eur": "EUR", "rubles": "RUB", "kroner": "NOK", "lira": "TRY",
            "won": "KRW", "rupees": "INR", "francs": "CHF", "franks": "CHF", "riyal": "SAR"}
NUMWORDS = {"one": "1", "five": "5", "ten": "10", "twenty": "20"}


def _find_spans(text, lexicon):
    """Longest-first, non-overlapping lexicon matches in order of appearance -> [(pos, surface, value)]."""
    t = " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "
    used, out = [], []
    for surf in sorted(lexicon, key=lambda x: (-len(x), x)):
        for mt in re.finditer(r"(?<= )%s(?= )" % re.escape(surf), t):
            a, b = mt.start(), mt.end()
            if not any(a < ub and ua < b for ua, ub in used):
                used.append((a, b))
                out.append((a, surf, lexicon[surf] if isinstance(lexicon, dict) else surf))
    return sorted(out)


_DIRECTION = None


def direction_labels():
    """tests/data/exchange_direction.json -> {normalised text: item} for the direction-labelled exchange questions
    (FROM / TO labelled by hand from the MEANING of the sentence) and the symmetric ones (no direction in the words),
    each with its deterministic split. Loaded once."""
    global _DIRECTION
    if _DIRECTION is None:
        d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data",
                                        "exchange_direction.json")))
        lab = {}
        for it in d["items"]:
            lab[" ".join(it["text"].lower().split())] = dict(it, kind="directional")
        for it in d["skipped"]:
            if it.get("split") and len(it.get("words") or []) >= 2:
                lab[" ".join(it["text"].lower().split())] = dict(it, kind="symmetric")
        _DIRECTION = lab
    return _DIRECTION


def method_truth(domain, text):
    """-> {"verb", "args", "from"} | {"unclear": True} | None (the stand-in cannot label it: skipped).
    EXCHANGE DIRECTION (fixed 2026-09-26, backlog E0.1 / the CLM panel's w2-hd finding): the truth used to be
    POSITIONAL (the first currency named = FROM), wrong on 29 of 74 directional CLINC150 train items, so the old
    'right call' only ever checked the currency SET. It now comes from the hand-labelled set
    (tests/data/exchange_direction.json): a directional item's from / to and the words that named them; a
    symmetric item ('the rate between X and Y') carries "direction": False and the model's no-direction reading
    {"either": [words, words]}; a sentence the set does not label falls back to the old positional reading and
    says so ("direction": None) -- scored on the set only."""
    if domain == "exchange":
        lab = direction_labels().get(" ".join(str(text).lower().split()))
        if lab is not None and lab["kind"] == "directional":
            args = {"from": lab["from"], "to": lab["to"]}
            if lab.get("amount"):
                args["amount"] = lab["amount"]
            return {"verb": "fx", "args": args, "from": {"from": lab["from_words"], "to": lab["to_words"]},
                    "direction": True}
        if lab is not None and lab["kind"] == "symmetric":
            cur = _find_spans(text, CURRENCY)
            codes = []
            for _, surf, code in cur:
                if code not in [c for c, _ in codes]:
                    codes.append((code, surf))
            if len(codes) < 2:
                return None
            return {"verb": "fx", "args": {"from": codes[0][0], "to": codes[1][0]},
                    "from": {"either": [codes[0][1], codes[1][1]]},
                    "from_new": {"from": codes[0][1], "to": codes[1][1]}, "direction": False}
    if domain == "weather":
        hits = _find_spans(text, {c: c.title() for c in CITIES})
        if hits:
            return {"verb": "weather", "args": {"location": hits[0][2]}, "from": {"location": hits[0][1]}}
        return {"verb": "weather", "args": {"location": "current location"}, "from": {"location": ""}}
    if domain == "exchange":
        cur = _find_spans(text, CURRENCY)
        codes = []
        for _, surf, code in cur:
            if code not in [c for c, _ in codes]:
                codes.append((code, surf))
        if len(codes) < 2:
            return None
        nums = re.findall(r"\d+(?:\.\d+)?", text) or [NUMWORDS[w] for w in re.findall(r"[a-z]+", text.lower())
                                                         if w in NUMWORDS]
        args = {"from": codes[0][0], "to": codes[1][0]}
        frm = {"from": codes[0][1], "to": codes[1][1]}
        if nums:
            args["amount"] = nums[0]
        return {"verb": "fx", "args": args, "from": frm, "direction": None}
    raise ValueError(domain)


class MethodModel:
    """The stand-in model for methods: a candidate row with the right VERB -> same (+ the args and the words
    that gave them); otherwise new, with the method and the tool's live answer; a bare coin -> unclear."""

    def __init__(self, truth, row_verb, tools):
        self.truth, self.row_verb, self.tools = truth, row_verb, tools
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        q = re.search(r"^QUESTION: (.*)$", prompt, re.M).group(1)
        cands = json.loads(re.search(r"^CANDIDATES \(nearest rows in memory, best first\): (.*)$", prompt, re.M).group(1))
        t = self.truth.get(q)
        if t is None:
            return json.dumps({"verdict": "new", "answer": "I am not sure."})
        if t.get("unclear"):
            return json.dumps({"verdict": "unclear", "clarify": "What would you like to know about %s?" % q.strip(" ?!$")})
        if "intent" in t:                                   # a static CLINC question (a distractor row's kind)
            right = [c["row"] for c in cands if self.row_verb.get(c["row"]) == "static:" + t["intent"]]
            if right:
                return json.dumps({"verdict": "same", "row": right[0]})
            return json.dumps({"verdict": "new", "answer": answer_for(t["intent"])})
        right = [c["row"] for c in cands if self.row_verb.get(c["row"]) == t["verb"]]
        if right:
            return json.dumps({"verdict": "same", "row": right[0], "args": t["args"],
                               "from_question": t["from"]})
        return json.dumps({"verdict": "new", "answer": self.tools[t["verb"]](**t["args"]),
                           "method": {"verb": t["verb"], "args": t["args"],
                                      "from_question": t.get("from_new", t["from"]), "live": True}})


def methods(args):
    """Learn HOW to answer live questions from a stream, then answer held-out ones with the model detached.
    Scored on the CALL memory makes (verb + arguments), not on text; every live answer must come from a fresh
    tool call (a cached value would carry a stale tick)."""
    tick = [0]

    def tool(verb):
        def fn(**kw):
            tick[0] += 1
            return "%s(%s)#%d" % (verb, ",".join("%s=%s" % kv for kv in sorted(kw.items())), tick[0])
        return fn
    tools = {"price": tool("price"), "weather": tool("weather"), "fx": tool("fx")}
    crypto = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data",
                                         "meaning_crypto.json")))["questions"]
    items = {"crypto": ([], [])}
    for k, qd in enumerate(crypto):
        t = {"unclear": True} if qd.get("unclear") else {"verb": qd["verb"], "args": qd["args"],
                                                         "from": {"symbol": qd["from"]}}
        items["crypto"][k % 2].append((qd["text"], t))
    d = json.load(open(os.path.join(args.data, "clinc150_full.json")))
    for dom, intent in (("weather", "weather"), ("exchange", "exchange_rate")):
        tr = [(t, method_truth(dom, t)) for t, i in d["train"] if i == intent]
        te = [(t, method_truth(dom, t)) for t, i in d["test"] if i == intent]
        items[dom] = ([x for x in tr if x[1]], [x for x in te if x[1]])
    m = fresh_mind()
    if getattr(args, "positional", False):
        m._meaning_direction = False                        # the positional baseline (no direction reader)
    row_verb = {}
    train = {}
    for t, i in d["train"]:
        train.setdefault(i, []).append(t)
    for intent in sorted(train):                            # the static answers memory already has (rivals)
        if intent in ("weather", "exchange_rate"):
            continue
        m.teach(train[intent][0], answer_for(intent))
    for rid, row in m.meaning.rows.items():
        row_verb[rid] = "static:" + str(intent_of(m.zoo["ladder"]._exact.get(" ".join(row["canonical"].lower().split()), {}).get("answer")))
    for verb, fn in tools.items():
        m.meaning_tool_register(verb, fn)
    stream = [(t, tr) for dom in sorted(items) for t, tr in items[dom][0]]
    # the static intents' other phrasings flow through the same stream (so methods compete with 148 intents)
    stream += [(t, {"intent": i}) for i in sorted(train) if i not in ("weather", "exchange_rate")
               for t in train[i][1:6]]
    random.Random(0).shuffle(stream)
    truth = {t: tr for t, tr in stream}
    model = MethodModel(truth, row_verb, tools)
    m.zoo_attach(model)
    m.zoo["llm"] = model
    for q, tr in stream:
        out = m.ask(q)
        if out.get("row") and out.get("verdict") == "new" and "verb" in tr:
            row_verb[out["row"]] = tr["verb"]
        if out.get("verdict") == "new" and "intent" in tr:
            for rid, row in m.meaning.rows.items():
                if rid not in row_verb:
                    row_verb[rid] = "static:" + tr["intent"]
    m.zoo["llm"] = None
    report = {"stream": len(stream), "model_calls_in_stream": model.calls}

    def score(dom):
        c = {"right_call": 0, "WRONG_call": 0, "clarified": 0, "escalated": 0, "stale": 0, "n": 0,
             "unclear_n": 0, "unclear_clarified": 0, "unclear_ACTED": 0}
        for q, tr in items[dom][1]:
            before = tick[0]
            out = m.ask(q)
            if tr.get("unclear"):
                c["unclear_n"] += 1
                if out.get("tier") == "clarify":
                    c["unclear_clarified"] += 1
                elif out.get("call"):
                    c["unclear_ACTED"] += 1
                continue
            c["n"] += 1
            call = out.get("call")
            if tr.get("direction") is False:
                # a SYMMETRIC exchange question ('the rate between X and Y'): it names no direction, so leaving it
                # to the person (clarify / escalate) is the designed answer; acting on it guesses a direction
                c["sym_n"] = c.get("sym_n", 0) + 1
                c["sym_ACTED" if call else "sym_left_open"] = c.get("sym_ACTED" if call else "sym_left_open", 0) + 1
            if out.get("tier") == "clarify":
                c["clarified"] += 1
            elif not call:
                c["escalated"] += 1
            else:
                got = call["args"]
                if tr["verb"] == "fx":
                    same_set = call["verb"] == "fx" and {got.get("from"), got.get("to")} == \
                        {tr["args"]["from"], tr["args"]["to"]} and str(got.get("amount")) == str(tr["args"].get("amount"))
                    # a DIRECTIONAL labelled item needs from AND to right; a symmetric or unlabelled one has no
                    # direction to get right (the old set check, kept for them)
                    ok = same_set and (not tr.get("direction") or (got.get("from"), got.get("to")) ==
                                       (tr["args"]["from"], tr["args"]["to"]))
                    if same_set and not ok:
                        c["WRONG_direction"] = c.get("WRONG_direction", 0) + 1
                else:
                    ok = call["verb"] == tr["verb"] and got == tr["args"]
                c["right_call" if ok else "WRONG_call"] += 1
                if not ok and os.environ.get("BENCH_SHOW"):
                    print("  WRONG %s | want %s | got %s" % (q, tr["args"], call))
                if out.get("answer") and not str(out["answer"]).endswith("#%d" % tick[0]) or tick[0] == before:
                    c["stale"] += 1
        return c
    for dom in sorted(items):
        report[dom] = score(dom)
    if not getattr(args, "positional", False):
        # PAIRED: the same learned memory, the direction reader switched off -- the positional baseline alone
        m._meaning_direction = False
        report["exchange_positional_same_memory"] = score("exchange")
        m._meaning_direction = True
    report["meaning"] = {k: v for k, v in m.meaning_report().items() if k in ("rows", "by_kind", "slot_values")}
    print(json.dumps(report, indent=1))
    if getattr(args, "record", False):
        report["command"] = "PYTHONHASHSEED=0 python3 tools/bench_meaning.py --data DATA methods%s --record" % (
            " --positional" if getattr(args, "positional", False) else "")
        report["scoring"] = ("fx: a DIRECTIONAL labelled item (tests/data/exchange_direction.json) needs from AND to "
                             "right; a symmetric one is scored by currency set and counted sym_ACTED / sym_left_open; "
                             "exchange_positional_same_memory = the same learned memory asked with the direction "
                             "reader off")
        with open(os.path.join(EVIDENCE, "bench_meaning_methods.json"), "w") as f:
            json.dump(report, f, indent=1)


# ------------------------------------------------------------------------------------------------ direction (E3.1)
def direction(args):
    """EXCHANGE DIRECTION THROUGH THE MEANING DOOR, END TO END (backlog E3.1 wiring): the context-role reader
    (holographic_rolecall.ContextRoles, taught by every `same` / `new` verdict's from_question -- p28
    _meaning_direction_learn) against the positional baseline (the mind's _meaning_direction = False: bind fills
    from / to in the order the method's first example named them), on tests/data/exchange_direction.json's
    deterministic split: 128 labelled train + 27 symmetric train sentences stream through ask() with a stand-in model
    that answers from the LABELS (the words that named FROM and TO; {"either": [...]} for a sentence with no
    direction), mixed with 148 static CLINC150 intents as rivals (5 phrasings each, the methods mode's setup); then
    the 51 labelled test + 14 symmetric test sentences are asked with the model DETACHED.
    PAIRED: the memory is learned ONCE (reader on) and the test set is asked twice, reader on / off.
    Two readings of each: END TO END (what ask() did: right / wrong direction / other wrong call / clarified
    / escalated) and BIND LEVEL (MeaningIndex.bind on the learned fx row for EVERY test sentence -- direction alone,
    whatever the serve gate decided)."""
    d = json.load(open(os.path.join(args.data, "clinc150_full.json")))
    lab = direction_labels()
    ex = [v for v in lab.values()]
    tr_items = [v for v in ex if v["split"] == "train"]
    te_items = [v for v in ex if v["split"] == "test"]
    tick = [0]

    def fx(**kw):
        tick[0] += 1
        return "fx(%s)#%d" % (",".join("%s=%s" % kv for kv in sorted(kw.items())), tick[0])
    train = {}
    for t, i in d["train"]:
        train.setdefault(i, []).append(t)
    out = {"command": "PYTHONHASHSEED=0 python3 tools/bench_meaning.py --data DATA direction",
           "protocol": direction.__doc__.split("\n\n")[0].strip(),
           "n": {"train_directional": sum(v["kind"] == "directional" for v in tr_items),
                 "train_symmetric": sum(v["kind"] == "symmetric" for v in tr_items),
                 "test_directional": sum(v["kind"] == "directional" for v in te_items),
                 "test_symmetric": sum(v["kind"] == "symmetric" for v in te_items)}, "arms": {}}
    # ONE learned memory, TWO readings at test time (paired): the stream is learned once with the reader ON (the
    # shipped door), then every test sentence is asked with the reader on and with it off -- so the comparison is
    # the reader alone, never two stream histories that diverged on the first different serve
    t0 = time.time()
    m = fresh_mind()
    row_verb = {}
    for intent in sorted(train):
        if intent in ("weather", "exchange_rate"):
            continue
        m.teach(train[intent][0], answer_for(intent))
    for rid, row in m.meaning.rows.items():
        row_verb[rid] = "static:" + str(intent_of(
            m.zoo["ladder"]._exact.get(" ".join(row["canonical"].lower().split()), {}).get("answer")))
    m.meaning_tool_register("fx", fx)
    stream = [(v["text"], method_truth("exchange", v["text"])) for v in tr_items]
    stream = [(q, t_) for q, t_ in stream if t_]
    stream += [(t, {"intent": i}) for i in sorted(train) if i not in ("weather", "exchange_rate")
               for t in train[i][1:6]]
    random.Random(0).shuffle(stream)
    # the FIRST fx question the model sees must be able to create the method with two slots: a directional one
    first = next(k for k, (q, t_) in enumerate(stream) if "verb" in t_ and t_.get("direction"))
    stream.insert(0, stream.pop(first))
    truth = {q: t_ for q, t_ in stream}
    model = MethodModel(truth, row_verb, {"fx": fx})
    m.zoo_attach(model)
    m.zoo["llm"] = model
    for q, t_ in stream:
        o = m.ask(q)
        if o.get("row") and o.get("verdict") == "new" and "verb" in t_:
            row_verb[o["row"]] = t_["verb"]
        if o.get("verdict") == "new" and "intent" in t_:
            for rid in m.meaning.rows:
                row_verb.setdefault(rid, "static:" + t_["intent"])
    m.zoo["llm"] = None
    fx_rows = [r for r, row in m.meaning.rows.items() if row["kind"] == "method" and row["method"]["verb"] == "fx"]
    rd = m.__dict__.get("_direction_reader_obj")
    out["learned"] = {"fx_rows": len(fx_rows), "model_calls_in_stream": model.calls,
                      "reader_counts": dict(rd.counts) if rd is not None else {}, "stream_s": round(time.time() - t0, 1)}
    for arm in ("positional", "context"):
        m._meaning_direction = (arm == "context")
        e2e = {"right_direction": 0, "WRONG_direction": 0, "WRONG_other": 0, "clarified": 0, "escalated": 0,
               "stale": 0, "sym_clarified": 0, "sym_ACTED": 0, "sym_escalated": 0}
        bind = {"right_direction": 0, "WRONG_direction": 0, "left_open": 0, "other": 0,
                "sym_left_open": 0, "sym_filled": 0}
        for v in te_items:
            tr = method_truth("exchange", v["text"])
            if tr is None:
                continue
            before = tick[0]
            o = m.ask(v["text"])
            call = o.get("call")
            if v["kind"] == "symmetric":
                key = "sym_clarified" if o.get("tier") == "clarify" else ("sym_ACTED" if call else "sym_escalated")
                e2e[key] += 1
            elif o.get("tier") == "clarify":
                e2e["clarified"] += 1
            elif not call:
                e2e["escalated"] += 1
            else:
                got = call["args"]
                if (got.get("from"), got.get("to")) == (tr["args"]["from"], tr["args"]["to"]):
                    e2e["right_direction"] += 1
                elif {got.get("from"), got.get("to")} == {tr["args"]["from"], tr["args"]["to"]}:
                    e2e["WRONG_direction"] += 1
                else:
                    e2e["WRONG_other"] += 1
                if tick[0] == before:
                    e2e["stale"] += 1
            if fx_rows:
                a_, miss = m.meaning.bind(fx_rows[0], v["text"], reader=m._meaning_reader())
                if v["kind"] == "symmetric":
                    bind["sym_left_open" if ("from" in miss or "to" in miss) else "sym_filled"] += 1
                elif "from" in miss or "to" in miss:
                    bind["left_open"] += 1
                elif (a_.get("from"), a_.get("to")) == (tr["args"]["from"], tr["args"]["to"]):
                    bind["right_direction"] += 1
                elif {a_.get("from"), a_.get("to")} == {tr["args"]["from"], tr["args"]["to"]}:
                    bind["WRONG_direction"] += 1
                else:
                    bind["other"] += 1
        out["arms"][arm] = {"end_to_end": e2e, "bind_level": bind}
        print("[direction %s] e2e %s | bind %s (%.0fs)" % (arm, e2e, bind, time.time() - t0), flush=True)
    m._meaning_direction = True
    path = os.path.join(EVIDENCE, "bench_meaning_direction.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


# ------------------------------------------------------------------------------------------------ protos (E1.1 c3)
EVIDENCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "research", "evidence")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _eval_index(mi, row_intent, items):
    """items [(text, intent or None)] -> (g, correct, top row) through MeaningIndex.answers -- the SERVING code path,
    one candidate per answer key, confidence g = s1 + (s1 - s2) exactly as decide() computes it."""
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    g, ok = [], []
    for t, want in items:
        r = mi.answers(t, k=8)
        g.append(MeaningIndex.confidence(r) if r else -9.0)
        ok.append(bool(r) and want is not None and row_intent.get(r[0][0]) == want)
    import numpy as np
    return np.asarray(g, float), np.asarray(ok, bool)


def protos(args):
    """E1.1's third consumer measured under the E0.4 gate protocol (tools/bench_contrastive.gate_protocol):
    the LEARNED ROW PROTOTYPES (MeaningIndex.enable_protos, holographic_meaning.RowPrototypes) against the index as
    it serves today, on two learned indexes (the online oracle runs' saved indexes: --index for CLINC150, the
    sweep-181 idx_clinc.json whose baseline is 0.8413; --index-banking for Banking77).

    Arms, 3 seeds each: '1pass' = every stored wording replayed ONCE through the rule (what the online door does:
    each wording was one verdict), '4pass' = four passes (the panel's configuration; MeaningIndex.consolidate_protos
    is the idle-time faculty). The index is re-loaded per seed.
    Split: CLINC150 calibrates on val in-scope + sha256 half A of oos_test and reports on test + half B (the E0.4
    convention). Banking77's learned index was trained on EVERY training wording, so a train-carved val would leak:
    it calibrates on a sha256 half of its TEST set + half A of CLINC150's oos_test (out-of-DOMAIN, stated) and
    reports on the other half + half B.
    ACCEPTANCE (backlog E1.1 c3): paired-bootstrap CI lower bound > +1.0 top-1 point on BOTH indexes; AURC lower with
    CI (upper bound < 0); realised precision within 0.005 of target; out-of-scope served <= base + 0.5 points."""
    import hashlib
    import numpy as np
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    from tools.bench_contrastive import (_served_counts, aurc, gate_protocol, oos_half_a, paired_bootstrap,
                                         paired_bootstrap_stat)

    def at_coverage(sig, ok, isin, cov):
        """SYMMETRIC DIAGNOSTIC (an oracle threshold on the REPORT items, never a shipping number): serve the
        highest-threshold prefix whose in-scope right-coverage reaches `cov` -> {oos_served, wrong_in, coverage}."""
        u, c_right, c_in, c_oos = _served_counts(np.asarray(sig, float), np.asarray(ok, bool) & isin, isin)
        n_in, n_oos = int(isin.sum()), max(int((~isin).sum()), 1)
        good = np.flatnonzero(c_right / n_in >= cov)
        if not len(good):
            return None
        j = int(good.max())
        return {"coverage": round(float(c_right[j] / n_in), 4), "wrong_in": round(float((c_in[j] - c_right[j]) / n_in), 4),
                "oos_served": round(float(c_oos[j] / n_oos), 4)}
    d = json.load(open(os.path.join(args.data, "clinc150_full.json")))
    oos = [t for t, _ in d["oos_test"]]
    half_a = oos_half_a(oos)
    out = {"command": "PYTHONHASHSEED=0 OMP_NUM_THREADS=1 python3 tools/bench_meaning.py --data DATA protos "
                      "--index idx_clinc.json --index-banking idx_banking.json",
           "protocol": protos.__doc__.split("\n\n")[1].strip(), "settings": {"dim": 2048, "tau": 0.05, "lr": 0.6},
           "inputs_sha256": {"clinc150_full.json": _sha256(os.path.join(args.data, "clinc150_full.json"))},
           "datasets": {}, "configs_tried": "2 arms (1pass / 4pass) x 3 seeds; rule settings fixed at the panel's "
                                            "(tau .05, lr = eta/tau = .6); hashed delta 2048 (scratch: 1pass +1.40 at "
                                            "2048 vs +1.67 exact)"}
    sets = []
    if args.index:
        sets.append(("clinc", args.index))
    if args.index_banking:
        sets.append(("banking", args.index_banking))
    t00 = time.time()
    for name, path in sets:
        st = json.load(open(path))
        out["inputs_sha256"]["index:" + name] = _sha256(path)
        ri = st["row_intent"]
        if name == "clinc":
            cal_in = [tuple(x) for x in d["val"]]
            rep_in = [tuple(x) for x in d["test"]]
        else:
            with open(os.path.join(args.data, "banking77_test.csv"), newline="") as f:
                test = [(r["text"], r["category"]) for r in csv.DictReader(f)]
            out["inputs_sha256"]["banking77_test.csv"] = _sha256(os.path.join(args.data, "banking77_test.csv"))
            is_cal = [int(hashlib.sha256(("bench_meaning/protos/cal:" + t).encode()).hexdigest()[:8], 16) % 2 == 0
                      for t, _ in test]
            cal_in = [x for x, c in zip(test, is_cal) if c]
            rep_in = [x for x, c in zip(test, is_cal) if not c]
        items = cal_in + rep_in + [(t, None) for t in oos]
        is_cal_all = np.concatenate([np.ones(len(cal_in), bool), np.zeros(len(rep_in), bool), half_a])
        is_oos = np.concatenate([np.zeros(len(cal_in) + len(rep_in), bool), np.ones(len(oos), bool)])
        rep_mask = ~is_cal_all

        def run(mi):
            g, ok = _eval_index(mi, ri, items)
            return g, ok, gate_protocol(g, ok, is_oos, is_cal_all, targets=(0.95, 0.97), max_wrong=(0.015,))
        t0 = time.time()
        mi = MeaningIndex.from_state(st["meaning"])
        mi.protos = None                                 # the BASE arm reads plain centroids, whatever was saved
        g0, ok0, gp0 = run(mi)
        res = {"rows": len(mi.rows), "wordings": sum(len(r["phrasings"]) for r in mi.rows.values()),
               "n_cal_in": len(cal_in), "n_rep_in": len(rep_in), "base": gp0, "arms": {}}
        print("[protos %s] base top1 %.4f aurc %.4f (%.0fs)" % (name, gp0["top1"], gp0["aurc"], time.time() - t0),
              flush=True)
        rin = rep_mask & ~is_oos
        rs0 = g0[rep_mask]
        rk0 = ok0[rep_mask]
        ri0 = ~is_oos[rep_mask]
        for seed in (0, 1, 2):
            mi = MeaningIndex.from_state(st["meaning"])
            mi.protos = None
            mi.enable_protos()                               # a FRESH learned delta per seed (never a saved one)
            for ep in range(4):
                mi.consolidate_protos(epochs=1, seed=1000 * seed + ep)
                if ep not in (0, 3):
                    continue
                arm = "1pass" if ep == 0 else "4pass"
                g1, ok1, gp1 = run(mi)
                rs1, rk1 = g1[rep_mask], ok1[rep_mask]
                bt = paired_bootstrap(ok0[rin], ok1[rin], n=1000, seed=0)
                ba = paired_bootstrap_stat(lambda ix: aurc(rs0[ix], rk0[ix], ri0[ix]),
                                           lambda ix: aurc(rs1[ix], rk1[ix], ri0[ix]), len(rs0), n=500, seed=0)
                cmp_ = {"top1_diff_points": round(100 * bt["diff"], 2), "top1_ci_points": [round(100 * bt["lo"], 2),
                                                                                         round(100 * bt["hi"], 2)],
                        "aurc_diff": round(ba["diff"], 5), "aurc_ci": [round(ba["lo"], 5), round(ba["hi"], 5)],
                        "fixed": bt["b_only"], "broke": bt["a_only"]}
                for P in ("P0.95", "P0.97"):
                    b_, a_ = gp0["calibrated"][P], gp1["calibrated"][P]
                    cmp_[P] = {"precision": round(a_["precision"], 4), "precision_base": round(b_["precision"], 4),
                               "coverage": round(a_["coverage"], 4), "coverage_base": round(b_["coverage"], 4),
                               "oos_served": round(a_["oos_served"], 4), "oos_served_base": round(b_["oos_served"], 4)}
                target_ok = all(gp1["calibrated"][P]["precision"] >= float(P[1:]) - 0.005 for P in ("P0.95", "P0.97"))
                oos_ok = all(gp1["calibrated"][P]["oos_served"] <= gp0["calibrated"][P]["oos_served"] + 0.005
                             for P in ("P0.95", "P0.97"))
                # the OOS bar compares the two arms at the SAME target precision, where the learned arm serves far
                # more in-scope questions (P0.97: base 3.3% -- the prior-trap knife edge -- vs ~40%). At MATCHED
                # in-scope coverage (the base's own P0.95 / P0.97 coverage) the out-of-scope share is compared too:
                for P in ("P0.95", "P0.97"):
                    cov_b = gp0["calibrated"][P]["coverage"]
                    cmp_[P]["matched_coverage_diagnostic"] = {"base": at_coverage(rs0, rk0, ri0, cov_b),
                                                              "learned": at_coverage(rs1, rk1, ri0, cov_b)}
                cmp_["bar"] = {"top1_ci_lo_gt_1pt": bt["lo"] > 0.01, "aurc_lower_with_ci": ba["hi"] < 0,
                               "precision_within_0.005": target_ok, "oos_within_base_plus_0.5pt": oos_ok}
                cmp_["passes"] = all(cmp_["bar"].values())
                res["arms"].setdefault(arm, {})[str(seed)] = {"gate": gp1, "vs_base": cmp_,
                                                              "updates": mi.protos.n_updates}
                print("[protos %s] seed %d %s top1 %.4f d %+.2f [%+.2f, %+.2f] aurc %.4f d %+.5f [%+.5f, %+.5f] "
                      "passes %s (%.0fs)" % (name, seed, arm, gp1["top1"], cmp_["top1_diff_points"],
                                             *cmp_["top1_ci_points"], gp1["aurc"], cmp_["aurc_diff"],
                                             *cmp_["aurc_ci"], cmp_["passes"], time.time() - t0), flush=True)
        for arm, runs in res["arms"].items():
            res["arms"][arm]["all_seeds_pass"] = all(r["vs_base"]["passes"] for r in runs.values())
        out["datasets"][name] = res
    out["seconds"] = round(time.time() - t00, 1)
    path = os.path.join(EVIDENCE, "bench_meaning_protos.json")
    if os.path.exists(path):
        # a run over ONE index keeps the other index's measured section (each carries its own input sha256)
        try:
            prev = json.load(open(path))
            for k, v in (prev.get("datasets") or {}).items():
                if k not in out["datasets"]:
                    out["datasets"][k] = v
                    out["inputs_sha256"].setdefault("index:" + k, prev.get("inputs_sha256", {}).get("index:" + k))
        except ValueError:
            pass
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: {a: v["all_seeds_pass"] for a, v in r["arms"].items()} for k, r in out["datasets"].items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", default="clinc", choices=["clinc", "banking"])
    ap.add_argument("mode", choices=["offline", "online", "methods", "protos", "direction"])
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--model", default="oracle")
    ap.add_argument("--seed", type=int, default=0, help="online: stream shuffle + stand-in noise seed (0 = sweep 181)")
    ap.add_argument("--save-index", default=None, help="online: write the learned meaning index (JSON) here")
    ap.add_argument("--index", default=None, help="protos: the learned CLINC150 index (idx_clinc.json)")
    ap.add_argument("--index-banking", default=None, help="protos: a learned Banking77 index (online --save-index)")
    ap.add_argument("--protos", default="default", choices=["default", "on", "off"],
                    help="online: the learned meaning rows on / off for this run (default: the mind's default)")
    ap.add_argument("--positional", action="store_true",
                    help="methods: bind from / to by position (the baseline) instead of the direction reader")
    ap.add_argument("--record", action="store_true",
                    help="online: merge the result into docs/research/evidence/bench_meaning_online.json")
    args = ap.parse_args()
    if args.mode == "offline":
        offline(args)
    elif args.mode == "methods":
        methods(args)
    elif args.mode == "protos":
        protos(args)
    elif args.mode == "direction":
        direction(args)
    else:
        online(args)


class ScriptedModel:
    """THE STAND-IN MODEL END. It reads the typed prompt exactly as a model would (the QUESTION line and the
    CANDIDATES JSON), and answers from the dataset's labels:
        a candidate row with the question's true intent -> {"verdict": "same", "row": <it>}
        no such candidate                               -> {"verdict": "new", "answer": <the intent's answer>}
    noise=r: with probability r the verdict is WRONG -- a random wrong candidate when there are any, otherwise a
    wrong new answer. It measures the learning loop, not any real model's judgement."""

    def __init__(self, truth, row_intent, noise=0.0, seed=0):
        self.truth, self.row_intent, self.noise = truth, row_intent, float(noise)
        self.rng = random.Random(seed)
        self.calls = 0
        self.wrong_given = 0

    def __call__(self, prompt):
        self.calls += 1
        q = re.search(r"^QUESTION: (.*)$", prompt, re.M).group(1)
        cands = json.loads(re.search(r"^CANDIDATES \(nearest rows in memory, best first\): (.*)$", prompt, re.M).group(1))
        want = self.truth.get(q, "oos")
        right = [c["row"] for c in cands if self.row_intent.get(c["row"]) == want]
        if self.rng.random() < self.noise:
            self.wrong_given += 1
            wrong = [c["row"] for c in cands if c["row"] not in right]
            if wrong:
                return json.dumps({"verdict": "same", "row": self.rng.choice(wrong)})
            return json.dumps({"verdict": "new", "answer": answer_for("noise_%d" % self.calls)})
        if right:
            return json.dumps({"verdict": "same", "row": right[0]})
        return json.dumps({"verdict": "new", "answer": answer_for(want if want != "oos" else "oos_%d" % self.calls)})


def online(args):
    train, test, oos, _ = load(args.data, args.dataset)
    m = fresh_mind()
    if getattr(args, "protos", "default") in ("on", "off"):
        m.meaning_protos(args.protos == "on")             # E1.1 c3 arm: the learned rows on / off for this run
    row_intent = {}
    for intent in sorted(train):
        m.teach(train[intent][0], answer_for(intent))
    for rid, row in m.meaning.rows.items():
        row_intent[rid] = intent_of(m.zoo["ladder"]._exact.get(" ".join(row["canonical"].lower().split()), {}).get("answer"))
    stream = [(t, i) for i in sorted(train) for t in train[i][1:]]
    if args.dataset == "clinc":
        stream += [(t, "oos") for t, _ in json.load(open(os.path.join(args.data, "clinc150_full.json")))["oos_train"]]
    random.Random(args.seed).shuffle(stream)               # seed 0 = the sweep-181 stream, bit for bit
    truth = {t: i for t, i in stream}
    noise = float(args.model.split(":")[1]) if args.model.startswith("noisy:") else 0.0
    model = ScriptedModel(truth, row_intent, noise=noise, seed=args.seed)
    m.zoo_attach(model)
    m.zoo["llm"] = model                                   # zoo_attach keeps an existing llm; force ours
    windows, cur = [], {"n": 0, "model": 0, "mem_correct": 0, "mem_wrong": 0, "clarify": 0}
    t0 = time.time()
    for k, (q, want) in enumerate(stream):
        before = model.calls
        out = m.ask(q)
        used_model = model.calls > before
        if out.get("row") and out.get("verdict") == "new":
            row_intent[out["row"]] = want if want != "oos" else "oos"
        if out.get("verdict") == "new" and out.get("row") is None and out.get("learned"):
            rid_ = None
            for rid, row in m.meaning.rows.items():
                if " ".join(row["canonical"].lower().split()) == " ".join(q.lower().split()):
                    rid_ = rid
            if rid_:
                row_intent[rid_] = want
        cur["n"] += 1
        if used_model:
            cur["model"] += 1
        elif out.get("tier") == "clarify":
            cur["clarify"] += 1
        elif str(out.get("answer") or "").strip():
            got = intent_of(out.get("answer"))
            cur["mem_correct" if (got == want or (want == "oos" and got is None)) else "mem_wrong"] += 1
        if cur["n"] == 1000 or k == len(stream) - 1:
            windows.append(dict(cur))
            cur = {"n": 0, "model": 0, "mem_correct": 0, "mem_wrong": 0, "clarify": 0}
    t_stream = time.time() - t0
    m.zoo["llm"] = None                                    # detach the model: what did memory learn?
    if args.save_index:
        arr = m._meaning_arrays()
        json.dump({"meaning": m._meaning_state(), "row_intent": row_intent,
                   "meaning_arrays": {k: v.tolist() for k, v in arr.items()}}, open(args.save_index, "w"))
    res_in = [classify(m.ask(t), i) for t, i in test]
    res_oos = [classify(m.ask(t), None) for t in oos]
    rep = m.meaning_report()
    # E2.1 AUDIT: do stored negatives ever point at a row of the question's OWN intent (a FALSE negative)? Every exact
    # veto (normalised wording, row) is checked against the dataset's label of that wording and the row's intent.
    ntruth = {" ".join(t.lower().split()): i for t, i in stream}
    negs = sorted(getattr(m, "_meaning_neg", set()))
    resolvable = [(q, r) for q, r in negs if q in ntruth and r in row_intent]
    false_neg = [(q, r) for q, r in resolvable if row_intent[r] == ntruth[q] and ntruth[q] != "oos"]
    tr = m.meaning_teacher_report()
    res = {"dataset": args.dataset, "model": args.model, "seed": args.seed, "protos": m.meaning_protos()["on"],
           "stream": len(stream), "model_wrong_verdicts_given": model.wrong_given, "model_calls_total": model.calls,
           "per_1000_questions": [{"model_calls": w["model"], "memory_right": w["mem_correct"],
                                   "memory_WRONG": w["mem_wrong"], "clarify": w["clarify"], "n": w["n"]}
                                  for w in windows],
           "after_stream_model_detached": {"in_scope_test": len(test), "in_scope": tally(res_in),
                                           "out_of_scope_test": len(oos), "out_of_scope": tally(res_oos)},
           "meaning": {k: rep[k] for k in ("rows", "by_kind", "wordings", "associations",
                                           "calibration_labels", "calibrated")},
           "teacher": {"eps_hat": round(tr["eps_hat"], 4), "re_asks": tr["re_asks"], "agreement": tr["agreement"],
                       "held_back": tr["held_back"]},
           "negatives": {"exact_vetoes": len(negs), "resolvable": len(resolvable),
                         "false_negatives_same_intent": len(false_neg),
                         "prototype_negatives": m.negatives_report()["meaning"]["prototype_negatives"]},
           "merged_wordings": m.meaning.merged,
           "stream_s": round(t_stream, 1)}
    print(json.dumps(res, indent=1))
    if getattr(args, "record", False):
        # the evidence file: one entry per (dataset, model, seed, protos) -- a re-run replaces only its own entry
        path = os.path.join(EVIDENCE, "bench_meaning_online.json")
        doc = {}
        if os.path.exists(path):
            try:
                doc = json.load(open(path))
            except ValueError:
                doc = {}
        doc["about"] = ("tools/bench_meaning.py online: teach one wording per intent, stream the rest through ask() "
                        "with the SCRIPTED stand-in model attached (oracle / noisy:r), detach it, ask the test set. "
                        "Measures the learning loop, not any real model. Command per run: PYTHONHASHSEED=0 python3 "
                        "tools/bench_meaning.py --data DATA [--dataset banking] online --model M --seed S "
                        "[--protos on|off] --record")
        key = "%s|%s|seed%d|protos%s" % (args.dataset, args.model, args.seed, "on" if res["protos"] else "off")
        doc.setdefault("runs", {})[key] = res
        with open(path + ".tmp", "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
