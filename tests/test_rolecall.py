"""Phase-E holographic candidates (CLM backlog E3.1 / E3.2 / E5.1): role-filler calls, the chimera check, the
context-role direction reader on real wording, the soft question state, the mind faculties, trajectories.

Every test is small and deterministic (seeded atoms, hashlib-derived vectors); the full measurement with baselines is
tools/bench_rolecall.py -> docs/research/evidence/bench_rolecall.json."""
import hashlib
import json
import math
import os

import numpy as np
import pytest

from holographic.agents_and_reasoning.holographic_rolecall import (
    ContextRoles, ProductCodec, RoleCodec, amounts_in, call_string, encode_step, expected_calibration_error,
    find_span, naive_decode, positional_read, question_state, question_tokens, trajectory_compare,
    trajectory_decode, trajectory_diff, trajectory_encode)

DATA = os.path.join(os.path.dirname(__file__), "data", "exchange_direction.json")
SLOTS = ["from", "to"]


@pytest.fixture(scope="module")
def exchange():
    return json.load(open(DATA))


def _mentions(it):
    """[(words, token_index)] of an item's currency mentions, in question order (extra TO mentions included)."""
    toks = question_tokens(it["text"])
    fa, ta = it["from_at"], it["to_at"]
    ms = [(it["from_words"], fa), (it["to_words"], ta)]
    taken = [(fa, fa + len(question_tokens(it["from_words"]))), (ta, ta + len(question_tokens(it["to_words"])))]
    for w in it.get("to_also_words", []):
        sp = find_span(toks, w, taken)
        taken.append(sp)
        ms.append((w, sp[0]))
    return sorted(ms, key=lambda m: m[1])


def _right(it, asg):
    return asg["from"] == it["from_words"] and asg["to"] in [it["to_words"]] + it.get("to_also_words", [])


def _teach(reader, items):
    for it in items:
        reader.learn(it["text"], {"from": (it["from_words"], it["from_at"]), "to": (it["to_words"], it["to_at"])})
    return reader


# ------------------------------------------------------------------------------------------ the data file
def test_exchange_set_is_hand_labelled_complete_and_split_by_hash(exchange):
    items, skipped = exchange["items"], exchange["skipped"]
    assert len(items) >= 100                                              # the E3.1 acceptance floor
    assert {it["source"] for it in items} == {"clinc150", "handwritten"}
    # every one of CLINC150's 150 exchange_rate sentences is accounted for: labelled or skipped WITH a reason
    assert sum(it["source"] == "clinc150" for it in items) + sum(s["source"] == "clinc150" for s in skipped) == 150
    assert all(s["reason"] for s in skipped)
    assert "CC BY 3.0" in exchange["attribution"]
    for it in items:
        toks = question_tokens(it["text"])
        # the recorded word spans resolve under the stated tokenizer, and FROM != TO
        assert toks[it["from_at"]:it["from_at"] + len(question_tokens(it["from_words"]))] == question_tokens(it["from_words"])
        assert toks[it["to_at"]:it["to_at"] + len(question_tokens(it["to_words"]))] == question_tokens(it["to_words"])
        assert it["from"] != it["to"] and len(it["from"]) == 3 and len(it["to"]) == 3
        # the split field IS the stated hash rule (so every consumer holds out the same items)
        h = int(hashlib.sha256(("exchange_direction:" + " ".join(toks)).encode()).hexdigest()[:8], 16)
        assert it["split"] == ("test" if h % 100 < 35 else "train")
    # the reason this file exists: word order is NOT direction for a large share of real wording
    assert sum(it["from_at"] > it["to_at"] for it in items) >= 50


# ------------------------------------------------------------------------------------------ E3.1 direction
def test_context_role_reader_beats_positional_on_the_held_out_split(exchange):
    items = exchange["items"]
    train = [it for it in items if it["split"] == "train"]
    test = [it for it in items if it["split"] == "test"]
    cr = _teach(ContextRoles(), train)
    ctx = sum(_right(it, cr.read(it["text"], _mentions(it), SLOTS)["assignment"]) for it in test)
    pos = sum(_right(it, positional_read(_mentions(it), SLOTS)) for it in test)
    # measured 49/51 vs 35/51 (bench_rolecall.json 'direction'); pinned with room, never below the baseline
    assert ctx >= 47 and pos <= 36 and ctx > pos + 10, (ctx, pos)


def test_untaught_reader_is_positional_and_says_so():
    r = ContextRoles(dim=1024).read("how many yen for 20 dollars", ["yen", "dollars"], SLOTS)
    assert r["via"] == "positional" and r["p"] is None and r["assignment"] == {"from": "yen", "to": "dollars"}


def test_no_direction_is_learned_not_inferred(exchange):
    """KEPT NEGATIVE pinned: taught only directional verdicts the reader never says 'no direction'; with the
    symmetric TRAIN sentences taught as {'either': [...]} it refuses a direction on the held-out ones and on no
    directional one."""
    items, skipped = exchange["items"], exchange["skipped"]
    sym_tr = [s for s in skipped if s.get("words") and s["split"] == "train"]
    sym_te = [s for s in skipped if s.get("words") and s["split"] == "test"]
    test = [it for it in items if it["split"] == "test"]
    cr = _teach(ContextRoles(), [it for it in items if it["split"] == "train"])
    assert all(cr.read(s["text"], s["words"], SLOTS)["direction"] for s in sym_te)      # the negative
    for s in sym_tr:
        cr.learn(s["text"], {"either": list(s["words"])})
    assert sum(not cr.read(s["text"], s["words"], SLOTS)["direction"] for s in sym_te) >= len(sym_te) - 1
    assert sum(not cr.read(it["text"], _mentions(it), SLOTS)["direction"] for it in test) <= 1


def test_context_roles_persist_and_are_deterministic():
    def build(rule):
        cr = ContextRoles(dim=512, rule=rule)
        cr.learn("convert 100 dollars to euros", {"from": "dollars", "to": "euros"})
        cr.learn("how many yen for 5 pounds", {"from": "pounds", "to": "yen"})
        return cr
    for rule in ("centroid", "infonce"):
        a, b = build(rule), build(rule)
        assert a.digest() == b.digest()
        c = ContextRoles.from_state(*a.state())
        assert c.digest() == a.digest()
        q = ("how many pesos for 3 euros", ["pesos", "euros"], SLOTS)
        assert c.read(*q)["assignment"] == a.read(*q)["assignment"]


# ------------------------------------------------------------------------------------------ E3.2 compose
@pytest.fixture(scope="module")
def codec():
    return RoleCodec(dim=2048, seed=0)


def _books():
    return ["verb%02d" % i for i in range(20)], ["val%02d" % i for i in range(50)]


def test_sum_form_decodes_exactly_and_is_only_a_candidate(codec):
    verbs, vals = _books()
    s = codec.encode_call("verb07", {"from": "val03", "to": "val41"})
    got = codec.compose(s, verbs, {"from": vals, "to": vals})
    assert got["call"] == "verb07(from=val03, to=val41)" and got["verdict"] == "clean"
    assert got["executes"] is False and got["candidate"] is True
    assert got["p_null"] < 0.01 and got["p_model"] > 0.9


def test_noise_as_long_as_the_state_still_decodes(codec):
    verbs, vals = _books()
    rng = np.random.default_rng(0)
    ok = 0
    for t in range(30):
        v, x, y = verbs[t % 20], vals[(3 * t) % 50], vals[(7 * t + 1) % 50]
        s = codec.encode_call(v, {"from": x, "to": y})
        s = s + math.sqrt(3.0 / 2048) * rng.standard_normal(2048)       # the panel's sigma = 1 arm
        got = codec.compose(s, verbs, {"from": vals, "to": vals})
        ok += got["call"] == call_string(v, {"from": x, "to": y}) and got["verdict"] != "ambiguous"
    assert ok == 30


def test_chimera_check_catches_what_per_role_argmax_builds(codec):
    """Equal 2-call blends: the per-role argmax (the panel's decode) builds chimeras most of the time; compose()
    never SERVES one -- it says 'ambiguous'."""
    verbs, vals = _books()
    rng = np.random.default_rng(1)
    naive_chim = served_chim = flagged = 0
    n = 40
    for t in range(n):
        v1, v2 = [verbs[int(i)] for i in rng.choice(20, 2, replace=False)]
        a, b, c, d = [vals[int(i)] for i in rng.choice(50, 4, replace=False)]
        c1, c2 = call_string(v1, {"from": a, "to": b}), call_string(v2, {"from": c, "to": d})
        s = codec.encode_call(v1, {"from": a, "to": b}) + codec.encode_call(v2, {"from": c, "to": d})
        nd = naive_decode(codec, s, verbs, {"from": vals, "to": vals})
        naive_chim += call_string(nd["verb"], nd["args"]) not in (c1, c2)
        got = codec.compose(s, verbs, {"from": vals, "to": vals})
        flagged += got["verdict"] == "ambiguous"
        served_chim += got["verdict"] != "ambiguous" and got["call"] not in (c1, c2)
    assert naive_chim >= n // 2                        # the panel measured 74-76%
    assert served_chim == 0 and flagged >= n - 2


def test_dominant_blend_serves_the_dominant_call(codec):
    verbs, vals = _books()
    s = codec.encode_call("verb01", {"from": "val10", "to": "val20"}) + \
        codec.encode_call("verb02", {"from": "val30", "to": "val40"}, weight=0.3)
    got = codec.compose(s, verbs, {"from": vals, "to": vals})
    assert got["call"] == "verb01(from=val10, to=val20)" and got["verdict"] == "dominant"


def test_structureless_state_proposes_nothing(codec):
    verbs, vals = _books()
    got = codec.compose(np.random.default_rng(5).standard_normal(2048), verbs, {"from": vals, "to": vals})
    assert got["verdict"] == "empty" and got["call"] is None


def test_signature_limits_slots_and_empty_candidate_lists_are_skipped(codec):
    s = codec.encode_call("weather", {"location": "paris"})
    got = codec.compose(s, {"weather": ["location"], "fx": ["from", "to"]},
                        {"location": ["paris", "oslo"], "from": ["USD", "EUR"], "to": ["USD", "EUR"], "amount": []})
    assert got["call"] == "weather(location=paris)" and got["verdict"] == "clean"


def test_option_and_text_roles(codec):
    ov = codec.encode_option("refund", ["i want my money back", "refund my order please"])
    assert codec.decode_role(ov, "OPTION", ["shipping", "refund", "billing"])[0][0] == "refund"
    assert codec.option_text_similarity(ov, "can i get my money back") > \
        codec.option_text_similarity(ov, "where is my parcel")


def test_question_state_composes_real_wording_and_flags_no_direction(exchange, codec):
    items, skipped = exchange["items"], exchange["skipped"]
    cr = _teach(ContextRoles(), [it for it in items if it["split"] == "train"])
    for s in skipped:
        if s.get("words") and s["split"] == "train":
            cr.learn(s["text"], {"either": list(s["words"])})
    ok = n = 0
    for it in [it for it in items if it["split"] == "test" and not it.get("to_also")][:20]:
        vals = {it["from_words"]: it["from"], it["to_words"]: it["to"]}
        nums = amounts_in(it["text"]) or (["1"] if it["amount"] == "1" else [])
        st, _ = question_state(codec, cr, it["text"], "fx", vals, SLOTS, numbers=nums)
        codes = sorted(set(vals.values()))
        got = codec.compose(st, {"fx": ["from", "to", "amount"]}, {"from": codes, "to": codes, "amount": nums})
        want = {"from": it["from"], "to": it["to"]}
        if it["amount"] is not None:
            want["amount"] = it["amount"]
        ok += got["call"] == call_string("fx", want)
        n += 1
    assert ok >= n - 2, (ok, n)
    sym = [s for s in skipped if s.get("words") and s["split"] == "test"]
    amb = 0
    for s in sym:
        w1, w2 = s["words"]
        st, _ = question_state(codec, cr, s["text"], "fx", {w1: "A", w2: "B"}, SLOTS, numbers=[])
        amb += codec.compose(st, {"fx": ["from", "to"]}, {"from": ["A", "B"], "to": ["A", "B"]})["verdict"] == "ambiguous"
    assert amb >= len(sym) - 1


def test_numbers_are_normalised():
    assert amounts_in("Convert 1,250.00 EUR to JPY") == ["1250.00"]
    assert amounts_in("10k yen to usd") == ["10000"] and amounts_in("2.5 million yen") == ["2500000"]
    assert amounts_in("i must know five dollars in yen") == ["5"] and amounts_in("usd to yen") == []


def test_ece_is_zero_for_perfect_calibration_and_positive_otherwise():
    assert expected_calibration_error([1.0, 1.0, 0.0], [1, 1, 0])[0] == 0.0
    assert expected_calibration_error([0.9] * 10, [1] * 5 + [0] * 5)[0] == pytest.approx(0.4)


# ------------------------------------------------------------------------------------------ the mind faculties
@pytest.fixture(scope="module")
def mind():
    from holographic.misc.holographic_unified import UnifiedMind
    return UnifiedMind(dim=256, seed=0)


def test_mind_compose_records_a_candidate_and_learns_its_calibration_by_id(mind):
    calls = []
    mind.meaning_tool_register("fx", lambda **kw: calls.append(kw) or "1.0")   # a live tool the proposal must not call
    vals = ["USD", "EUR", "JPY", "GBP"]
    for i in range(12):
        a, b = vals[i % 4], vals[(i + 1) % 4]
        v = mind.call_encode("fx", {"from": a, "to": b})["vector"]
        if i % 3 == 0:                                   # a blend: the truth is one of the two, often not decoded
            v = v + mind.call_encode("fx", {"from": b, "to": a})["vector"]
        got = mind.call_compose(v, {"fx": ["from", "to"]}, {"from": vals, "to": vals})
        assert got["executes"] is False and got["id"]
        assert (got["served"] is False) == (got["verdict"] in ("ambiguous", "empty"))
        mind.decision_outcome(got["id"], call_string("fx", {"from": a, "to": b}))
    assert calls == []                                   # a proposal never calls the tool
    assert mind.door_calibrator("compose").calibrated()
    got = mind.call_compose(mind.call_encode("fx", {"from": "USD", "to": "JPY"})["vector"], ["fx"],
                            {"from": vals, "to": vals})
    assert got["p_correct"] is not None and got["p_correct"] > 0.5


def test_mind_direction_and_question_compose(mind):
    mind.direction_learn("how many pesos can i get for 20 dollars", {"from": "dollars", "to": "pesos"})
    mind.direction_learn("convert 10 pounds to euros", {"from": "pounds", "to": "euros"})
    mind.direction_learn("what is 5 euros in yen", {"from": "euros", "to": "yen"})
    r = mind.direction_read("how many rupees can i get for 50 pounds", ["rupees", "pounds"])
    assert r["assignment"] == {"from": "pounds", "to": "rupees"}
    got = mind.call_from_question("how many rupees can i get for 50 pounds", "fx", {"rupees": "INR", "pounds": "GBP"})
    assert got["call"] == "fx(amount=50, from=GBP, to=INR)" and got["executes"] is False


def test_mind_trajectory_faculties(mind):
    steps = [{"verb": "search", "args": {"query": "q%d" % i}} for i in range(6)]
    a = mind.trajectory_encode(steps)
    b = mind.trajectory_encode(steps[:2] + [steps[3], steps[2]] + steps[4:])
    # every step shares the VERB(x)search term, so a swapped pair still matches half of each step: the expected
    # cosine is (4 + 2 * 0.5) / 6 = 0.83, not 4 / 6 -- the decode below is what names the edit
    assert mind.trajectory_compare(a, a)["cosine"] > 0.999 and mind.trajectory_compare(a, b)["cosine"] < 0.9
    d = mind.trajectory_diff(a, b, steps)
    assert d["ops"][0]["op"] == "swap" and d["ops"][0]["at"] == 2


def test_mind_product_factor_with_permutation_roles(mind):
    verbs, vals = ["fx", "price", "swap"], ["USD", "EUR", "JPY", "GBP", "CAD", "MXN"]
    pc = ProductCodec(verbs, vals, SLOTS)
    c = pc.encode("fx", {"from": "JPY", "to": "USD"})
    got = mind.call_factor_product(c, verbs, vals, SLOTS, restarts=6, iters=60, m_null=20, accept_p=0.05)
    assert got["call"] == "fx(from=JPY, to=USD)" and got["exit"] == "exact" and got["executes"] is False


# ------------------------------------------------------------------------------------------ E5.1 trajectories
def test_trajectory_swaps_and_insertions_leave_the_same_plan_noise_floor(codec):
    """The draft's permutation-only gate passed trivially; this one also inserts a step, and compares against a
    realistic floor: the SAME plan re-encoded with every step's incidental text changed."""
    rng = np.random.default_rng(3)
    words = ["alpha", "bravo", "delta", "echo", "golf", "hotel", "india", "kilo"]

    def txt():
        return " ".join(words[int(i)] for i in rng.integers(8, size=5))      # bounded vocabulary (n-gram cache)

    def enc(calls):
        return trajectory_encode([encode_step(codec, v, a, text=txt(), text_weight=0.5) for v, a in calls])

    floor, swap, ins = [], [], []
    for t in range(15):
        calls = [("search", {"query": words[int(i)]}) for i in rng.integers(8, size=6)]
        calls = [(v, dict(a, n=str(k))) for k, (v, a) in enumerate(calls)]         # distinct steps
        A = enc(calls)
        floor.append(float(np.dot(A, enc(calls))))
        i = int(rng.integers(5))
        swap.append(float(np.dot(A, enc(calls[:i] + [calls[i + 1], calls[i]] + calls[i + 2:]))))
        k = int(rng.integers(7))
        ins.append(float(np.dot(A, enc(calls[:k] + [("fx", {"from": "USD", "to": "EUR"})] + calls[k:]))))
    lo = min(floor)
    assert max(swap) < lo, (max(swap), lo)
    assert np.mean(np.array(ins) < lo) >= 0.9
    cmp = trajectory_compare(enc(calls), A, floor=floor)
    assert cmp["p_same"] > 0.02


def test_trajectory_decodes_in_order_and_diff_names_insert_and_delete(codec):
    calls = [("fx", {"from": a, "to": b}) for a, b in [("USD", "EUR"), ("EUR", "JPY"), ("JPY", "GBP"), ("GBP", "CAD")]]
    book = {call_string(v, a): codec.encode_call(v, a) for v, a in calls + [("timer", {})]}
    A = trajectory_encode([codec.encode_call(v, a) for v, a in calls])
    B = trajectory_encode([codec.encode_call(v, a) for v, a in calls[:2] + [("timer", {})] + calls[2:]])
    a, b = [n for n, _ in trajectory_decode(A, book)], [n for n, _ in trajectory_decode(B, book)]
    assert a == [call_string(v, x) for v, x in calls]
    assert trajectory_diff(a, b) == [{"op": "insert", "a": [], "b": ["timer()"], "at": 2}]
    assert trajectory_diff(b, a)[0]["op"] == "delete"
