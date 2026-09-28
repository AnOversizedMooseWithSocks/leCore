"""bench_core_memory.py -- what does the CORE MEMORY buy a fresh mind? (2026-09-27, owner-directed)

The owner's direction: "The point of the seed memory, and the learning process in general, is to reduce LLM calls and
improve automatic tool calling capabilities ... If a user uses unknown phrasing ... it should be learned permanently."
So the question is measured, not argued: a mind LEARNS in one process (real human wording, a scripted model end),
tools/distill_release.py distils it, and a FRESH mind boots each candidate core memory and meets HELD-OUT questions
with the same model end attached. Counted per arm: model calls per 1,000 questions, and what memory served on its own
-- correct, WRONG (the dangerous one), out-of-scope served (wrong by definition), clarified.

THE ARMS (every arm boots through mind.boot(partition=<a temp copy>, doctrine=True): the seedpack is in all of them)
    a_nothing        an empty partition
    b_old_bundle     the release_bundle of 2026-08-26 (497 rows copied from session memory; --old-bundle DIR)
    c_rows_only      the rows-only distillate of the live partition (release_bundle before this change; --rows-bundle)
    d_extended       the extended distillate of the live partition (release_bundle after it; --ext-bundle)
    L_rows_only      the LEARNER's rows-only distillate -- what the Q&A rule alone carries of what it learned
    L_extended       the LEARNER's extended distillate: + confirmed wordings, associations, methods, negatives,
                     direction reader, api specs, tool reflexes, shareable ProtoStores
    L_extended_cal   L_extended + the meaning gate's calibration labels (distill(carry_calibration=True))
a..d answer "what ships TODAY for this wording" (the live partition learned none of it); L_* isolate what each
distillation rule carries of real learning. THE L ARMS RUN THE DOMAIN PROFILE (distill anchor=False): a CLINC150
answer is not engine knowledge, so the release rule (anchor=True) would carry none of it -- the profile drops ONLY
the engine-anchor / lexicon / length rules and measures what carrying learned wordings buys when a row does ship.
Methods, API specs and tool reflexes need no profile: the release rule carries them as they are.

THE DOMAINS
    clinc    CLINC150 (CC BY 3.0): N intents (every k-th, sorted), 1 taught wording each, the learner streams S more
             training wordings per intent + out-of-scope training questions through ask() with the scripted oracle
             (tools/bench_meaning.ScriptedModel); held-out = test wordings of those intents + out-of-scope test
    banking  Banking77 (CC BY 4.0): same protocol, CLINC out-of-scope test as the off-topic set
    methods  HOW questions (weather / exchange rate from CLINC150, crypto prices from tests/data/meaning_crypto.json):
             the learner learns METHOD rows from the stand-in's typed verdicts (bench_meaning.MethodModel); held-out:
             the test wordings. Scored on the CALL (verb + args), live values are never cached
    tools    a local HTTP API (127.0.0.1) with an API-KEY header: the learner learns it (api_learn with the spec's
             securityScheme), is given the FAKE key ONCE by hand (api_use headers=...), and is taught three tool
             reflexes; held-out: hand-written requests served through mind.serve() with the key in the environment
             (${TOYAIR_API_KEY}) -- and once with the variable UNSET (must fail loudly, never call unauthenticated).
             The tools arm distils with allow_private_hosts=True (the release rule never carries a private host).

KEEP IT TO A FEW MINUTES OF CPU: defaults are 40 CLINC intents / 20 Banking intents, a bounded stream and test set.

Usage:
    PYTHONHASHSEED=0 python3 tools/bench_core_memory.py --data DIR [--record]
        [--old-bundle DIR] [--rows-bundle DIR] [--ext-bundle DIR] [--domains clinc,banking,methods,tools]
--record writes docs/research/evidence/bench_core_memory.json.
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
EVIDENCE = os.path.join(ROOT, "docs", "research", "evidence")
FAKE_KEY = "Hunter2-FAKE-9c1d-toyair"          # a FAKE secret: never a real credential in a test or a bench


def _lecore():
    import lecore
    return lecore


def boot_arm(bundle):
    """A FRESH mind booted on a temp COPY of `bundle` (None = an empty partition), the seedpack included. A copy,
    because boot rolls a partition's learning files over and a bundle must never be modified."""
    lecore = _lecore()
    tmp = tempfile.mkdtemp(prefix="bcm_arm_")
    part = os.path.join(tmp, "p")
    if bundle:
        shutil.copytree(bundle, part)
    else:
        os.makedirs(part)
    m = lecore.UnifiedMind()
    m.boot(partition=part, doctrine=True, llm=lambda p: "")
    m._archive_root = part
    return m, tmp


def _distill(src, out, **kw):
    import distill_release as D
    return D.distill(None, out, src=src, **kw)


# ------------------------------------------------------------------------------------------------ clinc / banking
def _intent_map(m, B):
    """rid -> intent for every answer row a booted mind holds (engine rows map to None: the stand-in never says
    `same` to them)."""
    out = {}
    ex = getattr(m.zoo["ladder"], "_exact", {})
    for rid, row in m.meaning.rows.items():
        out[rid] = B.intent_of((ex.get(" ".join(row["canonical"].lower().split())) or {}).get("answer"))
    return out


def run_stream(m, B, stream, row_intent, seed):
    """Stream (question, intent|'oos') through ask() with the scripted oracle attached (online: every escalation
    is resolved AND learned, exactly bench_meaning.online). -> tallies."""
    truth = {t: i for t, i in stream}
    model = B.ScriptedModel(truth, row_intent, noise=0.0, seed=seed)
    m.zoo_attach(model)
    m.zoo["llm"] = model
    c = {"n": 0, "model_calls": 0, "memory_correct": 0, "memory_WRONG": 0, "oos_served_WRONG": 0, "clarify": 0,
         "first_200_model_calls": 0}
    for k, (q, want) in enumerate(stream):
        before = model.calls
        out = m.ask(q)
        used = model.calls > before
        if out.get("row") and out.get("verdict") == "new":
            row_intent[out["row"]] = want
        if out.get("verdict") == "new" and out.get("row") is None and out.get("learned"):
            from holographic.agents_and_reasoning.holographic_meaning import _rid
            row_intent[_rid(q, "answer")] = want
        c["n"] += 1
        if used:
            c["model_calls"] += 1
            if k < 200:
                c["first_200_model_calls"] += 1
        elif out.get("tier") == "clarify":
            c["clarify"] += 1
        elif str(out.get("answer") or "").strip():
            got = B.intent_of(out.get("answer"))
            if want == "oos":
                c["oos_served_WRONG"] += 1
            elif got == want:
                c["memory_correct"] += 1
            else:
                c["memory_WRONG"] += 1
    m.zoo["llm"] = None
    c["model_calls_per_1000"] = round(1000.0 * c["model_calls"] / max(c["n"], 1), 1)
    return c


def domain_intents(args, dataset, B, scratch, bundles):
    """Learn in one mind, distil three ways, boot every arm fresh, stream the held-out set."""
    train, test, oos, _ = B.load(args.data, dataset)
    n_int = args.clinc_intents if dataset == "clinc" else args.banking_intents
    step = max(1, len(train) // n_int)
    intents = sorted(train)[::step][:n_int]
    iset = set(intents)
    rng = random.Random(0)
    # ---- the learner: 1 taught wording per intent, then a stream through the scripted oracle ----------------------
    learner, ltmp = boot_arm(None)
    for i in intents:
        learner.teach(train[i][0], B.answer_for(i))
    row_intent = _intent_map(learner, B)
    stream = [(t, i) for i in intents for t in train[i][1:1 + args.stream_per_intent]]
    if dataset == "clinc":
        oos_train = [t for t, _ in json.load(open(os.path.join(args.data, "clinc150_full.json")))["oos_train"]]
        stream += [(t, "oos") for t in oos_train[:args.oos_stream]]
    rng.shuffle(stream)
    t0 = time.time()
    learn_tally = run_stream(learner, B, stream, row_intent, seed=0)
    learn_s = time.time() - t0
    outs = {}
    for name, kw in (("L_rows_only", dict(rows_only=True, anchor=False)),
                     ("L_extended", dict(anchor=False)),
                     ("L_extended_cal", dict(anchor=False, carry_calibration=True))):
        d = os.path.join(scratch, "%s_%s" % (dataset, name))
        rep = _distill(learner, d, **kw)
        outs[name] = (d, {"shipped_rows": rep["shipped"], "promoted": len(rep.get("promoted", [])),
                          "excluded_rows": rep["excluded"],
                          "artifacts": {k: {"carried": v["carried"], "excluded": v["excluded"]}
                                        for k, v in (rep.get("artifacts") or {}).items()
                                        if v["carried"] or v["excluded"]}})
    # ---- held-out: test wordings of the chosen intents + off-topic questions ----------------------------------------
    held = [(t, i) for t, i in test if i in iset]
    rng2 = random.Random(1)
    rng2.shuffle(held)
    held = held[:args.test_n]
    oos_held = list(oos)
    rng2.shuffle(oos_held)
    held += [(t, "oos") for t in oos_held[:args.oos_test]]
    rng2.shuffle(held)
    res = {"intents": len(intents), "learner_stream": len(stream), "learner": learn_tally,
           "learner_stream_s": round(learn_s, 1), "held_out": len(held), "arms": {}}
    # THE REFERENCE: the learner itself, never restarted (everything it learned, nothing distilled away)
    ref_first = offline_pass(learner, B, held)
    ref_online = run_stream(learner, B, list(held), row_intent, seed=1)
    res["arms"]["L_learner_no_restart"] = {"first_contact_memory_alone": ref_first, "online": [ref_online],
                                           "online_model_calls_per_1000_mean": ref_online["model_calls_per_1000"]}
    shutil.rmtree(ltmp, ignore_errors=True)
    print("  %-8s %-16s first contact: correct %4d WRONG %3d oos-served %3d abstain %4d | online calls/1000 %6.1f"
          % (dataset, "L_learner_ref", ref_first["correct"], ref_first["WRONG"], ref_first["oos_served_WRONG"],
             ref_first["abstain"], ref_online["model_calls_per_1000"]), flush=True)
    arms = [("a_nothing", None)] + [(k, v) for k, v in bundles] + [(k, outs[k][0]) for k in
                                                                  ("L_rows_only", "L_extended", "L_extended_cal")]
    for name, b in arms:
        if b is not None and not os.path.isdir(b):
            res["arms"][name] = {"skipped": "bundle not found: %s" % b}
            continue
        # (1) FIRST CONTACT, memory alone (model detached): what the booted core memory serves before anything
        # else is learned -- every abstention here is a model call a served mind would make
        m, tmp = boot_arm(b)
        first = offline_pass(m, B, held)
        shutil.rmtree(tmp, ignore_errors=True)
        # (2) ONLINE, model attached, over SEEDS held-out orders: the stream keeps teaching every arm, so this is the
        # model-call count of a served mind; a verified serve (spot-check) counts as a model call
        online = []
        t0 = time.time()
        for sd in range(1, 1 + args.seeds):
            m, tmp = boot_arm(b)
            order = list(held)
            random.Random(100 + sd).shuffle(order)
            online.append(run_stream(m, B, order, _intent_map(m, B), seed=sd))
            shutil.rmtree(tmp, ignore_errors=True)
        tally = {"first_contact_memory_alone": first, "online": online,
                 "online_model_calls_per_1000_mean": round(sum(o["model_calls_per_1000"] for o in online) /
                                                          len(online), 1),
                 "online_memory_WRONG_mean": round(sum(o["memory_WRONG"] for o in online) / len(online), 1),
                 "online_oos_served_WRONG_mean": round(sum(o["oos_served_WRONG"] for o in online) / len(online), 1),
                 "s": round(time.time() - t0, 1)}
        if name in outs:
            tally["distilled"] = outs[name][1]
        res["arms"][name] = tally
        print("  %-8s %-16s first contact: correct %4d WRONG %3d oos-served %3d abstain %4d | online calls/1000 "
              "%6.1f WRONG %5.1f oos-served %4.1f  (%.0fs)"
              % (dataset, name, first["correct"], first["WRONG"], first["oos_served_WRONG"], first["abstain"],
                 tally["online_model_calls_per_1000_mean"], tally["online_memory_WRONG_mean"],
                 tally["online_oos_served_WRONG_mean"], tally["s"]), flush=True)
    return res


def offline_pass(m, B, held):
    """Every held-out question asked ONCE with no model attached -> {correct, WRONG, oos_served_WRONG, clarify,
    abstain, model_calls_per_1000 (= abstentions: each one is a model call a served mind would make)}."""
    m.zoo["llm"] = None
    c = {"correct": 0, "WRONG": 0, "oos_served_WRONG": 0, "clarify": 0, "abstain": 0}
    for q, want in held:
        r = B.classify(m.ask(q), None if want == "oos" else want)
        if want == "oos":
            c["oos_served_WRONG" if r == "wrong" else ("clarify" if r == "clarify" else "abstain")] += 1
        else:
            c[{"correct": "correct", "wrong": "WRONG", "clarify": "clarify", "abstain": "abstain"}[r]] += 1
    c["model_calls_per_1000"] = round(1000.0 * c["abstain"] / max(len(held), 1), 1)
    return c


# ------------------------------------------------------------------------------------------------ methods
def domain_methods(args, B, scratch, bundles):
    """Learn HOW (method rows) from a stream, distil, boot fresh, score the calls on held-out wordings."""
    tick = [0]

    def tool(verb):
        def fn(**kw):
            tick[0] += 1
            return "%s(%s)#%d" % (verb, ",".join("%s=%s" % kv for kv in sorted(kw.items())), tick[0])
        return fn
    tools = {"price": tool("price"), "weather": tool("weather"), "fx": tool("fx")}
    crypto = json.load(open(os.path.join(ROOT, "tests", "data", "meaning_crypto.json")))["questions"]
    items = {"crypto": ([], [])}
    for k, qd in enumerate(crypto):
        t = {"unclear": True} if qd.get("unclear") else {"verb": qd["verb"], "args": qd["args"],
                                                         "from": {"symbol": qd["from"]}}
        items["crypto"][k % 2].append((qd["text"], t))
    d = json.load(open(os.path.join(args.data, "clinc150_full.json")))
    for dom, intent in (("weather", "weather"), ("exchange", "exchange_rate")):
        tr = [(t, B.method_truth(dom, t)) for t, i in d["train"] if i == intent]
        te = [(t, B.method_truth(dom, t)) for t, i in d["test"] if i == intent]
        items[dom] = ([x for x in tr if x[1]], [x for x in te if x[1]])
    train = {}
    for t, i in d["train"]:
        train.setdefault(i, []).append(t)
    statics = [i for i in sorted(train) if i not in ("weather", "exchange_rate")][::10]

    def row_verbs(m):
        rv = {}
        ex = getattr(m.zoo["ladder"], "_exact", {})
        for rid, row in m.meaning.rows.items():
            if row["kind"] == "method":
                rv[rid] = row["method"]["verb"]
            else:
                rv[rid] = "static:" + str(B.intent_of((ex.get(" ".join(row["canonical"].lower().split())) or {})
                                                      .get("answer")))
        return rv

    def stream_through(m, stream, row_verb):
        for verb, fn in tools.items():
            m.meaning_tool_register(verb, fn)
        truth = {t: tr for t, tr in stream}
        model = B.MethodModel(truth, row_verb, tools)
        m.zoo_attach(model)
        m.zoo["llm"] = model
        c = {"n": 0, "model_calls": 0, "right_call": 0, "WRONG_call": 0, "clarified": 0, "escalated_or_model": 0,
             "unclear_n": 0, "unclear_clarified": 0, "unclear_ACTED": 0, "stale": 0}
        for q, tr in stream:
            before, tb = model.calls, tick[0]
            out = m.ask(q)
            used = model.calls > before
            if out.get("row") and out.get("verdict") == "new":
                row_verb[out["row"]] = tr.get("verb") or ("static:" + tr["intent"] if "intent" in tr else None)
            c["n"] += 1
            c["model_calls"] += int(used)
            if tr.get("unclear"):
                c["unclear_n"] += 1
                if out.get("tier") == "clarify":
                    c["unclear_clarified"] += 1
                elif out.get("call") and not used:
                    c["unclear_ACTED"] += 1
                continue
            if "intent" in tr:
                continue                                    # a static distractor: counted in model_calls only
            call = out.get("call")
            if used:
                c["escalated_or_model"] += 1
            elif out.get("tier") == "clarify":
                c["clarified"] += 1
            elif call:
                got = call["args"]
                if tr["verb"] == "fx":
                    ok = call["verb"] == "fx" and {got.get("from"), got.get("to")} == \
                        {tr["args"]["from"], tr["args"]["to"]} and \
                        str(got.get("amount")) == str(tr["args"].get("amount")) and \
                        (not tr.get("direction") or (got.get("from"), got.get("to")) ==
                         (tr["args"]["from"], tr["args"]["to"]))
                else:
                    ok = call["verb"] == tr["verb"] and got == tr["args"]
                c["right_call" if ok else "WRONG_call"] += 1
                if out.get("answer") and tick[0] == tb:
                    c["stale"] += 1                         # a live answer that did not come from a fresh call
            else:
                c["escalated_or_model"] += 1
        m.zoo["llm"] = None
        c["model_calls_per_1000"] = round(1000.0 * c["model_calls"] / max(c["n"], 1), 1)
        return c
    learner, ltmp = boot_arm(None)
    for i in statics:
        learner.teach(train[i][0], B.answer_for(i))
    stream = [(t, tr) for dom in sorted(items) for t, tr in items[dom][0]]
    stream += [(t, {"intent": i}) for i in statics for t in train[i][1:4]]
    random.Random(0).shuffle(stream)
    learn = stream_through(learner, stream, row_verbs(learner))
    outs = {}
    for name, kw in (("L_rows_only", dict(rows_only=True, anchor=False)), ("L_extended", dict(anchor=False)),
                     ("L_extended_cal", dict(anchor=False, carry_calibration=True))):
        dd = os.path.join(scratch, "methods_%s" % name)
        rep = _distill(learner, dd, **kw)
        outs[name] = (dd, {"shipped_rows": rep["shipped"],
                           "artifacts": {k: {"carried": v["carried"], "excluded": v["excluded"]}
                                         for k, v in (rep.get("artifacts") or {}).items()
                                         if v["carried"] or v["excluded"]}})
    held = [(t, tr) for dom in sorted(items) for t, tr in items[dom][1]]
    held += [(t, {"intent": i}) for i in statics for t in train[i][5:8]]
    random.Random(1).shuffle(held)
    res = {"learner_stream": len(stream), "learner": learn, "held_out": len(held), "arms": {}}
    # THE REFERENCE: the learner itself, never restarted -- the ceiling a distillate is measured against
    res["arms"]["L_learner_no_restart"] = stream_through(learner, list(held), row_verbs(learner))
    shutil.rmtree(ltmp, ignore_errors=True)
    print("  methods  %-16s calls/1000 %6.1f" % ("L_learner_ref", res["arms"]["L_learner_no_restart"]
                                                 ["model_calls_per_1000"]), flush=True)
    for name, b in [("a_nothing", None)] + list(bundles) + [(k, outs[k][0]) for k in
                                                          ("L_rows_only", "L_extended", "L_extended_cal")]:
        if b is not None and not os.path.isdir(b):
            res["arms"][name] = {"skipped": "bundle not found"}
            continue
        m, tmp = boot_arm(b)
        tally = stream_through(m, list(held), row_verbs(m))
        if name in outs:
            tally["distilled"] = outs[name][1]
        res["arms"][name] = tally
        shutil.rmtree(tmp, ignore_errors=True)
        print("  methods  %-16s calls/1000 %6.1f  right %3d  WRONG %3d  unclear acted %d/%d"
              % (name, tally["model_calls_per_1000"], tally["right_call"], tally["WRONG_call"],
                 tally["unclear_ACTED"], tally["unclear_n"]), flush=True)
    return res


# ------------------------------------------------------------------------------------------------ tools / api
TOOL_SPEC = {
    "info": {"title": "toyair"},
    "components": {"securitySchemes": {"key": {"type": "apiKey", "in": "header", "name": "X-Api-Key"}}},
    "paths": {
        "/aqi/{zone}": {"get": {"operationId": "air_quality", "summary": "get the air quality index for a zone",
                                "parameters": [{"name": "zone", "in": "path", "required": True}]}},
        "/pollen/{zone}": {"get": {"operationId": "pollen", "summary": "get the pollen count for a zone",
                                   "parameters": [{"name": "zone", "in": "path", "required": True}]}},
        "/uv/{zone}": {"get": {"operationId": "uv_index", "summary": "get the uv index for a zone",
                               "parameters": [{"name": "zone", "in": "path", "required": True}]}}}}
# (pattern taught, tool) -- the teacher's three reflexes; numbers in the query fill `zone`
TOOL_TEACH = [("what is the air quality index in zone 4", "toyair.air_quality"),
              ("how high is the pollen count in zone 9", "toyair.pollen"),
              ("what is the uv index reading for zone 2", "toyair.uv_index")]
# held-out requests, written the way people ask (none is a taught pattern)
TOOL_HELD = [("air quality index for zone 12 please", "toyair.air_quality"),
             ("whats the air quality like in zone 7", "toyair.air_quality"),
             ("check the air quality index of zone 31", "toyair.air_quality"),
             ("is the air quality index bad in zone 5 today", "toyair.air_quality"),
             ("air quality reading zone 18", "toyair.air_quality"),
             ("pollen count in zone 3?", "toyair.pollen"),
             ("how bad is the pollen count for zone 44", "toyair.pollen"),
             ("give me the pollen count, zone 6", "toyair.pollen"),
             ("tell me the pollen levels count in zone 21", "toyair.pollen"),
             ("current pollen count zone 8", "toyair.pollen"),
             ("uv index for zone 11", "toyair.uv_index"),
             ("how strong is the uv index reading in zone 15", "toyair.uv_index"),
             ("what's the uv index reading at zone 1", "toyair.uv_index"),
             ("show the uv index, zone 27", "toyair.uv_index"),
             ("uv index reading zone 13 now", "toyair.uv_index")]


def _toy_server():
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.headers.get("X-Api-Key") == FAKE_KEY
            parts = self.path.strip("/").split("/")
            body = json.dumps({"tool": parts[0], "zone": parts[-1]} if ok else {"error": "unauthorized"}).encode()
            self.send_response(200 if ok else 401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def domain_tools(args, scratch, bundles):
    srv = _toy_server()
    spec = dict(TOOL_SPEC, servers=[{"url": "http://127.0.0.1:%d" % srv.server_address[1]}])
    learner, ltmp = boot_arm(None)
    lr = learner.api_learn(spec)
    # the person gives the key ONCE, by hand, for one call: the call succeeds and teaches HOW to authenticate
    first = learner.api_use("toyair", "air_quality", params={"zone": 1}, headers={"X-Api-Key": FAKE_KEY})
    for pat, tool in TOOL_TEACH:
        svc, ep = tool.split(".")
        learner.tool_reflex_teach(pat, svc, ep, extract_numbers=["zone"])
    learner_blob = json.dumps([list(t) for t in learner.zoo["ladder"].taught_log])
    outs = {}
    for name, kw in (("L_rows_only", dict(rows_only=True)), ("L_extended", dict(allow_private_hosts=True))):
        dd = os.path.join(scratch, "tools_%s" % name)
        rep = _distill(learner, dd, **kw)
        raw = b""
        for base, _, files in os.walk(dd):
            for fn in files:
                raw += open(os.path.join(base, fn), "rb").read()
        # the container is compressed: read it back and scan its text too
        from holographic.io_and_interop.holographic_container import load_container
        with open(os.path.join(dd, "learning", "state.lecore"), "rb") as f_:
            txt = json.dumps([s_.get("meta") for s_ in load_container(f_.read())["sections"]], default=str)
        outs[name] = (dd, {"env": rep.get("env"), "fake_key_in_bundle_bytes": FAKE_KEY.encode() in raw,
                           "fake_key_in_bundle_text": FAKE_KEY in txt,
                           "artifacts": {k: {"carried": v["carried"], "excluded": v["excluded"]}
                                         for k, v in (rep.get("artifacts") or {}).items()
                                         if v["carried"] or v["excluded"]}})
    res_ref = learner
    res = {"learned": {"env": lr.get("env"), "first_call_ok": first.get("ok"),
                       "learned_auth": first.get("learned_auth"),
                       "fake_key_in_learner_rows": FAKE_KEY in learner_blob},
           "held_out": len(TOOL_HELD), "arms": {}}
    saved = os.environ.get("TOYAIR_API_KEY")
    arms = [("L_learner_no_restart", "REF"), ("a_nothing", None)] + list(bundles) + \
        [(k, outs[k][0]) for k in ("L_rows_only", "L_extended")]
    for name, b in arms:
        if b not in (None, "REF") and not os.path.isdir(b):
            res["arms"][name] = {"skipped": "bundle not found"}
            continue
        for env_on in ((True, False) if name == "L_extended" else (True,)):
            if env_on:
                os.environ["TOYAIR_API_KEY"] = FAKE_KEY
            else:
                os.environ.pop("TOYAIR_API_KEY", None)
            if b == "REF":
                m, tmp = res_ref, ltmp                      # the learner itself (the ceiling)
            else:
                m, tmp = boot_arm(b)
            c = {"n": 0, "served_tool": 0, "right_tool": 0, "WRONG_tool": 0, "right_zone": 0, "escalated": 0,
                 "failed_loudly_missing_env": 0, "called_without_auth": 0}
            for q, want in TOOL_HELD:
                out = m.serve(q)
                c["n"] += 1
                if out.get("via") == "tool-reflex":
                    c["served_tool"] += 1
                    c["right_tool" if out.get("tool") == want else "WRONG_tool"] += 1
                    nums = re.findall(r"\d+", q)
                    if str((out.get("result") or {}).get("zone")) == (nums[0] if nums else None):
                        c["right_zone"] += 1
                else:
                    c["escalated"] += 1
                    if "missing_env" in str(out.get("reason", "")) or "TOYAIR_API_KEY" in str(out.get("reason", "")):
                        c["failed_loudly_missing_env"] += 1
                    if "401" in str(out.get("reason", "")):
                        c["called_without_auth"] += 1
            c["model_calls_per_1000"] = round(1000.0 * c["escalated"] / c["n"], 1)   # an escalation = a model call
            if name in outs:
                c["distilled"] = outs[name][1]
            res["arms"][name + ("" if env_on else "_ENV_UNSET")] = c
            shutil.rmtree(tmp, ignore_errors=True)
            print("  tools    %-24s served %2d/%d right %2d WRONG %d zone %2d loud-missing-env %d unauth-calls %d"
                  % (name + ("" if env_on else "_ENV_UNSET"), c["served_tool"], c["n"], c["right_tool"],
                     c["WRONG_tool"], c["right_zone"], c["failed_loudly_missing_env"], c["called_without_auth"]),
                  flush=True)
    if saved is None:
        os.environ.pop("TOYAIR_API_KEY", None)
    else:
        os.environ["TOYAIR_API_KEY"] = saved
    srv.shutdown()
    return res


def main():
    sc = "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad"
    ap = argparse.ArgumentParser(description="what a distilled core memory buys a fresh mind")
    ap.add_argument("--data", required=True, help="folder with clinc150_full.json, banking77_*.csv")
    ap.add_argument("--old-bundle", default=os.path.join(sc, "seedcheck", "release_bundle_before"))
    ap.add_argument("--rows-bundle", default=os.path.join(sc, "release_bundle_rows_only_20260927"))
    ap.add_argument("--ext-bundle", default=os.path.join(ROOT, "release_bundle"))
    ap.add_argument("--domains", default="clinc,banking,methods,tools")
    ap.add_argument("--clinc-intents", type=int, default=40)
    ap.add_argument("--banking-intents", type=int, default=20)
    ap.add_argument("--stream-per-intent", type=int, default=20)
    ap.add_argument("--oos-stream", type=int, default=150)
    ap.add_argument("--test-n", type=int, default=600)
    ap.add_argument("--oos-test", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=2, help="online held-out orders per arm (clinc / banking)")
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()
    import bench_meaning as B
    bundles = [("b_old_bundle", args.old_bundle), ("c_rows_only", args.rows_bundle), ("d_extended", args.ext_bundle)]
    scratch = tempfile.mkdtemp(prefix="bcm_")
    t0 = time.time()
    out = {"about": __doc__.split("\n\n")[0], "bundles": {k: v for k, v in bundles}, "domains": {}}
    for dom in args.domains.split(","):
        print("== %s" % dom, flush=True)
        if dom in ("clinc", "banking"):
            out["domains"][dom] = domain_intents(args, dom, B, scratch, bundles)
        elif dom == "methods":
            out["domains"][dom] = domain_methods(args, B, scratch, bundles)
        elif dom == "tools":
            out["domains"][dom] = domain_tools(args, scratch, bundles)
    out["total_s"] = round(time.time() - t0, 1)
    out["command"] = "PYTHONHASHSEED=0 python3 tools/bench_core_memory.py --data DATA --record"
    shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps({"total_s": out["total_s"]}))
    if args.record:
        with open(os.path.join(EVIDENCE, "bench_core_memory.json"), "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
