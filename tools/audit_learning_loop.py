#!/usr/bin/env python3
"""audit_learning_loop.py -- DOES leCore KEEP LEARNING FROM USE ACROSS A RESTART? Measured door by door.

WHY THIS EXISTS (the owner, 2026-09-26: "I want to be sure that our self improvement and self learning process is
working as best it can. Do a sweep to look for ways to improve it."). Every learning door in leCore has its own unit
tests, and each of them checks ONE hop: a door learns live, or a store round-trips through learning_save. What no
test did was walk the loop the way a real deployment walks it -- a decision is made, the process is RESTARTED (a
learning_rollover in a fresh mind: what every boot does), the caller reports the outcome by id AFTERWARDS, and the
restarted mind must learn from it, keep what it learned through ANOTHER restart, and never write a secret down.
A door that passes each of its unit tests can still fail that walk -- an outcome hook that is a live closure dies
with the process, a store whose rows are persisted but whose inputs are not cannot learn after a reload -- and
that failure is silent: decision_outcome returns normally and nothing learns.

THE WALK, per door, in a tmp partition (never the real one):
  (a) DECIDE      the door makes a decision and returns an id
  (b) RESTART     learning_save, then a FRESH mind boots the partition with learning_rollover (a cold boot)
  (c) OUTCOME     the outcome is reported by id in the fresh mind (a CORRECTION wherever the door can be wrong:
                  the owner's rule is that a correction must actually fix the answer)
  (d) LEARNED?    the door-specific evidence that the fresh mind learned: the next decision changed, the door's
                  calibrator got the pair, its ProtoStore moved
  (e) SURVIVES?   save + boot a third fresh mind: is what (d) learned still there, bit for bit where it can be?
  (f) GUARD?      a decision whose text carries a FAKE secret ("Hunter2-FAKE-9c1d"): its outcome must not move the
                  door, and no container section on disk may contain the secret
Each door also runs (a)+(c)+(d) with NO restart ("learns live") as the control: a door that fails live is a
different defect from a door that fails only across a restart.

Plus the loop-level checks the owner's standing rules ask for: every learning section survives a rollover (and a
section this build does not know is kept, as the container promises), the ladder's payload keys stay valid after a
tile split, the retile keeps the reflex's outcome fields, escalations and verify's profile survive a restart,
lookup verbs' ledger side effects, record sizes, and what learning_save's drift_vs_previous_save actually measures.

Usage:
    PYTHONHASHSEED=0 python tools/audit_learning_loop.py                  # every door, writes the evidence JSON
    PYTHONHASHSEED=0 python tools/audit_learning_loop.py --label before   # file this run under runs.before
    PYTHONHASHSEED=0 python tools/audit_learning_loop.py --doors typed,rank --no-write
The evidence file keeps every labelled run it has seen (runs.<label>), so a before / after pair lives in one file:
docs/research/evidence/audit_learning_loop.json. Deterministic (PYTHONHASHSEED=0, seeded minds, no wall clock in
any number; timings are reported separately and are the only machine-dependent values). ~30 s on 2 cores.
"""
import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONHASHSEED", "0")

SECRET = "Hunter2-FAKE-9c1d"                 # the ONLY secret this audit ever uses: fake, and greppable
EVIDENCE = os.path.join("docs", "research", "evidence", "audit_learning_loop.json")
PROPS = ("learns_live", "learns_after_restart", "survives_reload", "guard_holds")


# =====================================================================================================================
# the harness: minds, restarts, the secret scan
# =====================================================================================================================
def _mind():
    """The small, fast mind every learning test uses (dim 256; the trace and the typed doors run at their own dims)."""
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _boot(root, factory=_mind):
    """A COLD BOOT on the partition: a fresh mind + learning_rollover (what agent_boot / the service do at start).
    -> (mind, rollover report)."""
    m = factory()
    rep = m.learning_rollover(root)
    return m, rep


def _restart(m, root, factory=_mind):
    """Save this mind's learning and boot a FRESH one on the same partition. -> the fresh mind."""
    m.learning_save(root)
    m2, _ = _boot(root, factory)
    return m2


def _files(root):
    return sorted(glob.glob(os.path.join(str(root), "learning", "*.lecore")))


def _secret_hits(root, needle=SECRET):
    """Every container section (kind) under root whose meta text contains the fake secret, case-insensitively and
    also in the normalised form a question store keeps ('hunter2 fake 9c1d'). Arrays are numeric by construction
    (the container refuses object arrays), so the meta is where a text secret can hide."""
    from holographic.io_and_interop.holographic_container import load_container
    forms = {needle.lower(), re.sub(r"[^a-z0-9]+", " ", needle.lower()).strip()}
    hits = []
    for f in _files(root):
        got = load_container(open(f, "rb").read())
        for sec in got["sections"]:
            blob = json.dumps(sec.get("meta"), default=str).lower()
            blob_n = re.sub(r"[^a-z0-9]+", " ", blob)
            if any(x in blob or x in blob_n for x in forms):
                hits.append(sec["kind"])
    return sorted(set(hits))


def _kinds(root):
    from holographic.io_and_interop.holographic_container import load_container
    out = []
    for f in _files(root):
        out += [s["kind"] for s in load_container(open(f, "rb").read())["sections"]]
    return out


def _sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _so_sig(m):
    """A digest of every cached SystemOne's learned state (meta + arrays): moves iff a typed door learned."""
    parts = []
    for key, so in sorted((m.__dict__.get("_systemone_cache") or {}).items()):
        meta, arr = so.state()
        parts.append(_sha(key, meta, *[arr[k] for k in sorted(arr)]))
    return _sha(parts)


def _store_sig(m, door):
    st = (m.__dict__.get("_protostores") or {}).get(door)
    return None if st is None else st.digest()


def _cal_n(m, door):
    c = (m.__dict__.get("_door_calibrators") or {}).get(door)
    return 0 if c is None else len(c.pairs)


def _tmp(parent, name):
    d = os.path.join(parent, name)
    os.makedirs(d, exist_ok=True)
    return d


# =====================================================================================================================
# THE DOORS
# =====================================================================================================================
TYPED_Q = {"cat": {"type": "choice", "options": ["billing", "shipping"],
                   "examples": {"billing": ["card charged twice", "refund my invoice fee"],
                                "shipping": ["parcel lost in transit", "courier delivery late"]}}}
TYPED_S = "the invoice for my parcel shipment is wrong"      # a fresh fit answers 'billing' (every scorer): measured
TYPED_TRUTH = "shipping"                                      # ... so reporting 'shipping' is a CORRECTION


def door_typed(tmp, scorer="nb"):
    """typed / SystemOne: systemone_decide -> decision_outcome(id, truth). The correction must FIX the answer."""
    def dec(m, s=TYPED_S):
        return m.systemone_decide(s, TYPED_Q, scorer=scorer, encoder="ngram")["cat"]
    ev = {}
    # live
    m = _mind()
    a = dec(m)
    ev["fresh_answer"] = a["value"]
    m.decision_outcome(a["id"], TYPED_TRUTH)
    live = dec(m)["value"] == TYPED_TRUTH and a["value"] != TYPED_TRUTH
    # across a restart: the decision is made BEFORE, the outcome reported AFTER
    root = _tmp(tmp, "typed_" + scorer)
    m, _ = _boot(root)
    a = dec(m)
    m2 = _restart(m, root)
    rep = m2.decision_outcome(a["id"], TYPED_TRUTH)
    after = dec(m2)
    ev["after_restart_answer"] = after["value"]
    ev["after_restart_forwarded"] = rep.get("forwarded")
    learned_after = after["value"] == TYPED_TRUTH
    m3 = _restart(m2, root)
    again = dec(m3)
    survives = again["value"] == TYPED_TRUTH and again["ranked"] == after["ranked"]
    # guard: a secret-carrying state's outcome must not move the door, and must never reach the disk
    sec_state = "my password is %s and %s" % (SECRET, TYPED_S)
    b = dec(m3, sec_state)
    s0 = _so_sig(m3)                    # AFTER the decide: a decide may add its support score to a stream (a number)
    m3.decision_outcome(b["id"], TYPED_TRUTH if b["value"] != TYPED_TRUTH else "billing")
    moved = _so_sig(m3) != s0
    m3.learning_save(root)
    hits = _secret_hits(root)
    ev["guard"] = {"door_moved_by_secret": moved, "secret_sections": hits}
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": (not moved) and not hits, "evidence": ev}


def door_typed_absorb(tmp):
    """typed / SystemOne absorb (guarded EM over UNLABELLED traffic): the absorbed tables are persisted learned
    state -- a pasted secret in the traffic must never become a counted token on disk."""
    ev = {}
    root = _tmp(tmp, "absorb")
    m, _ = _boot(root)
    states = ["card charged twice again", "parcel lost again", "refund my fee", "courier is late",
              "my password is %s card charged" % SECRET, "invoice fee refund please", "delivery never came"]
    q = {"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["card charged twice", "refund my invoice fee", "double charge on my card", "wrong amount billed"],
        "shipping": ["parcel lost in transit", "courier delivery late", "package never arrived", "tracking frozen"]}}}
    out = m.systemone_absorb(states * 3, q, "cat", scorer="nb", encoder="ngram", iters=1)
    ev["absorb"] = {k: out.get(k) for k in ("applied", "reason") if k in out}
    m.learning_save(root)
    hits = _secret_hits(root)
    ev["secret_sections"] = hits
    return {"learns_live": None, "learns_after_restart": None, "survives_reload": None,
            "guard_holds": not hits, "evidence": ev}


TIE_Q = "denoise an image"            # a three-way lexical TIE at 2.5 (tests/test_router_learning.py)
TIE_USED = "Denoise (domain)"


def _router_verdicts(m):
    try:
        return int(m.router_report().get("verdicts", 0))
    except Exception:
        return 0


def door_route(tmp):
    """route_tiered (catalog path): decision_outcome(id, card used) feeds THREE learners -- the router's ProtoStore
    (a live hook, key 'router'), the route door's calibrator (_door_feed, from the record) and the reflex bridge
    (reflex_learn, from the record). The router is the one whose next decision changes (router mode on)."""
    ev = {}
    m = _mind()
    m.router_mode(True)
    r = m.route_tiered(TIE_Q)
    ev["fresh_top"] = r["options"][0]["name"]
    m.decision_outcome(r["id"], TIE_USED)
    r2 = m.route_tiered(TIE_Q)
    live = _router_verdicts(m) == 1 and r2["options"][0]["name"] == TIE_USED and _cal_n(m, "route") == 1
    root = _tmp(tmp, "route")
    m, _ = _boot(root)
    m.router_mode(True)
    r = m.route_tiered(TIE_Q)
    m2 = _restart(m, root)
    ev["router_mode_after_restart"] = m2.router_mode()
    m2.router_mode(True)                              # the owner's switch; its persistence is reported above
    rep = m2.decision_outcome(r["id"], TIE_USED)
    ev["after_restart"] = {"router_verdicts": _router_verdicts(m2), "route_calibrator_pairs": _cal_n(m2, "route"),
                           "bridge_seen": len(m2.__dict__.get("_reflex_seen") or []),
                           "forwarded": rep.get("forwarded"), "hooks": rep.get("hooks")}
    r3 = m2.route_tiered(TIE_Q)
    learned_after = _router_verdicts(m2) == 1 and r3["options"][0]["name"] == TIE_USED
    m3 = _restart(m2, root)
    survives = _router_verdicts(m3) == _router_verdicts(m2) and _cal_n(m3, "route") == _cal_n(m2, "route") \
        and m3.route_tiered(TIE_Q)["options"][0]["name"] == r3["options"][0]["name"]
    # guard
    v0 = _store_sig(m3, "route")
    g = m3.route_tiered("my password is %s, %s" % (SECRET, TIE_Q))
    m3.decision_outcome(g["id"], TIE_USED)
    moved = _store_sig(m3, "route") != v0
    m3.learning_save(root)
    hits = _secret_hits(root)
    ev["guard"] = {"door_moved_by_secret": moved, "secret_sections": hits}
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": (not moved) and not hits, "evidence": ev}


REFLEX_Q = "smooth a bumpy mesh"


def door_route_reflex(tmp):
    """route_tiered(reflex=True): a reported route teaches the bridge; the repeat is answered from experience with
    its own id, whose outcome feeds door_calibrator('bridge')."""
    ev = {}
    m = _mind()
    r = m.route_tiered(REFLEX_Q)
    m.decision_outcome(r["id"], r["answer"])
    rf = m.route_tiered(REFLEX_Q, reflex=True)
    ok_live = rf.get("via") == "reflex"
    if ok_live:
        m.decision_outcome(rf["id"], rf["answer"])
    live = ok_live and _cal_n(m, "bridge") == 1
    root = _tmp(tmp, "route_reflex")
    m, _ = _boot(root)
    r = m.route_tiered(REFLEX_Q)
    m.decision_outcome(r["id"], r["answer"])
    rf = m.route_tiered(REFLEX_Q, reflex=True)
    m2 = _restart(m, root)
    again = m2.route_tiered(REFLEX_Q, reflex=True)
    ev["repeat_after_restart_via"] = again.get("via")
    m2.decision_outcome(rf["id"], rf["answer"])
    learned_after = again.get("via") == "reflex" and _cal_n(m2, "bridge") == 1
    m3 = _restart(m2, root)
    survives = _cal_n(m3, "bridge") == 1 and m3.route_tiered(REFLEX_Q, reflex=True).get("via") == "reflex"
    # guard: a secret-carrying request must not teach the bridge
    n0 = len(m3.__dict__.get("_reflex_seen") or [])
    g = m3.route_tiered("my password is %s, %s" % (SECRET, REFLEX_Q))
    m3.decision_outcome(g["id"], g["answer"] or g["options"][0]["name"])
    moved = len(m3.__dict__.get("_reflex_seen") or []) != n0
    m3.learning_save(root)
    hits = _secret_hits(root)
    ev["guard"] = {"bridge_moved_by_secret": moved, "secret_sections": hits}
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": (not moved) and not hits, "evidence": ev}


TOOL_Q = "convert celsius fahrenheit units please"
TOOL_Q2 = "please convert these celsius temperatures to fahrenheit units"    # a paraphrase below the seen gate


def _tool_mind():
    """tests/test_tool_call_learning.py's two tool reflexes whose patterns TIE on word overlap (overlap picks the
    first taught, f_to_c; the truth for TOOL_Q is c_to_f). api_use is stubbed per mind (it is the network)."""
    m = _mind()
    _stub_api(m)
    m.tool_reflex_teach("convert 77 fahrenheit into celsius units", "convertd", "f_to_c")
    m.tool_reflex_teach("convert 25 celsius into fahrenheit units", "convertd", "c_to_f")
    return m


def _stub_api(m):
    def api_use(service, endpoint, params=None, headers=None):
        return {"ok": True, "data": {"endpoint": endpoint}}
    m.api_use = api_use
    return m


def _tool_boot_mind():
    """A fresh mind for a boot: the tool reflexes come back from the durable taught rows; only api_use is local."""
    return _stub_api(_mind())


def _tool_verdicts(m):
    a = m.__dict__.get("_tool_audit_d")
    return 0 if a is None else int(a.get("verdicts", 0))


def door_tool(tmp):
    """serve()'s tool reflex: a reported correction (the other tool was right) is a labelled verdict for the tool
    door's ProtoStore (a live hook, key 'tooldoor') -- the next PARAPHRASE is picked by the door."""
    ev = {}
    m = _tool_mind()
    r = m.serve(TOOL_Q)
    ev["fresh_pick"] = r.get("tool")
    m.decision_outcome(r["id"], "convertd.c_to_f")
    r2 = m.serve(TOOL_Q2)
    live = _tool_verdicts(m) == 1 and r2.get("tool") == "convertd.c_to_f"
    root = _tmp(tmp, "tool")
    m, _ = _boot(root, _tool_boot_mind)
    _stub_api(m)
    m.tool_reflex_teach("convert 77 fahrenheit into celsius units", "convertd", "f_to_c")
    m.tool_reflex_teach("convert 25 celsius into fahrenheit units", "convertd", "c_to_f")
    r = m.serve(TOOL_Q)
    m2 = _restart(m, root, _tool_boot_mind)
    rep = m2.decision_outcome(r["id"], "convertd.c_to_f")
    r2 = m2.serve(TOOL_Q2)
    ev["after_restart"] = {"tool_door_verdicts": _tool_verdicts(m2), "paraphrase_pick": r2.get("tool"),
                           "picked_by": r2.get("picked_by"), "forwarded": rep.get("forwarded")}
    learned_after = _tool_verdicts(m2) == 1 and r2.get("tool") == "convertd.c_to_f"
    m3 = _restart(m2, root, _tool_boot_mind)
    r3 = m3.serve(TOOL_Q2)
    survives = _tool_verdicts(m3) == _tool_verdicts(m2) and r3.get("tool") == r2.get("tool")
    v0 = _store_sig(m3, "tool")
    g = m3.serve("my password is %s, convert celsius fahrenheit units" % SECRET)
    if g.get("id"):
        m3.decision_outcome(g["id"], "convertd.c_to_f")
    moved = _store_sig(m3, "tool") != v0
    m3.learning_save(root)
    hits = _secret_hits(root)
    ev["guard"] = {"door_moved_by_secret": moved, "secret_sections": hits, "served_via": g.get("via")}
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": (not moved) and not hits, "evidence": ev}


SERVE_ALIAS = "decode a png"                      # a catalog alias: serve() answers it from the router (E4.5)


def door_serve_route(tmp):
    """serve()'s router path: 'use capability X' with the route's id; the outcome is a route outcome."""
    ev = {}
    m = _mind()
    r = m.serve(SERVE_ALIAS)
    ev["served_via"] = r.get("via")
    if r.get("via") != "route":
        return {"learns_live": False, "learns_after_restart": None, "survives_reload": None, "guard_holds": None,
                "evidence": ev}
    cap = r["capability"]
    m.decision_outcome(r["id"], cap)
    live = _router_verdicts(m) == 1 and _cal_n(m, "route") == 1
    root = _tmp(tmp, "serve_route")
    m, _ = _boot(root)
    r = m.serve(SERVE_ALIAS)
    m2 = _restart(m, root)
    rep = m2.decision_outcome(r["id"], cap)
    ev["after_restart"] = {"router_verdicts": _router_verdicts(m2), "route_calibrator_pairs": _cal_n(m2, "route"),
                           "forwarded": rep.get("forwarded")}
    learned_after = _router_verdicts(m2) == 1 and _cal_n(m2, "route") == 1
    m3 = _restart(m2, root)
    survives = _router_verdicts(m3) == _router_verdicts(m2) and _cal_n(m3, "route") == _cal_n(m2, "route")
    g = m3.serve("%s my password is %s" % (SERVE_ALIAS, SECRET))
    m3.learning_save(root)
    hits = _secret_hits(root)
    ev["guard"] = {"secret_served_via": g.get("via"), "secret_sections": hits}
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": g.get("via") != "route" and not hits, "evidence": ev}


BAL = "how do i check my account balance"
BAL_A = "Open the app and tap Accounts."
PIN = "how do i change my pin"
PIN_A = "Settings > Card > Change PIN."
MEAN_WRONG = "how can i see my balance"            # served the BALANCE row by meaning; the audit reports the PIN row
                                                   # as the truth (a correction: the owner's rule is that it must
                                                   # STOP the wrong serve and teach the wording to the right row)


def _meaning_mind():
    m = _mind()
    m.teach(BAL, BAL_A)
    m.teach(PIN, PIN_A)
    for k in range(24):                               # the verdict history a model-attached mind collects early
        m.meaning.observe(1.0 + k * 0.05, True)
        m.meaning.observe(0.1 + k * 0.02, False)
    m.meaning_resolve("how can i see my account balance", {"verdict": "same", "row": _rid(m, BAL)})
    return m


def _rid(m, text):
    return next(r for r, row in m.meaning.rows.items() if row["canonical"] == text)


def door_meaning(tmp):
    """meaning (serve / resolve / correction / method rows): a wrong meaning SERVE corrected by
    decision_outcome(id, right row) must stop the wrong serve and teach the wording; resolved links and learned
    METHOD rows must come back after a restart."""
    ev = {}
    m = _meaning_mind()
    out = m.ask(MEAN_WRONG)
    ev["fresh_serve"] = {"via": out.get("via"), "is_wrong_row": out.get("row") == _rid(m, BAL)}
    if out.get("via") != "meaning":
        return {"learns_live": None, "learns_after_restart": None, "survives_reload": None, "guard_holds": None,
                "evidence": dict(ev, why="no wrong meaning serve to correct in this build")}
    m.decision_outcome(out["id"], _rid(m, PIN))
    live = m.ask(MEAN_WRONG).get("row") != _rid(m, BAL) and MEAN_WRONG in m.meaning.rows[_rid(m, PIN)]["phrasings"]
    root = _tmp(tmp, "meaning")
    m, _ = _boot(root)
    m.teach(BAL, BAL_A)
    m.teach(PIN, PIN_A)
    for k in range(24):
        m.meaning.observe(1.0 + k * 0.05, True)
        m.meaning.observe(0.1 + k * 0.02, False)
    m.meaning_resolve("how can i see my account balance", {"verdict": "same", "row": _rid(m, BAL)})
    out = m.ask(MEAN_WRONG)
    m.meaning_resolve("what's the price of solana", {"verdict": "new", "answer": "-", "method": {
        "verb": "price", "args": {"symbol": "SOL"}, "from_question": {"symbol": "solana"}, "live": True}})
    m2 = _restart(m, root)
    rep = m2.decision_outcome(out["id"], _rid(m2, PIN))
    fixed = m2.ask(MEAN_WRONG)
    m2.meaning_resolve("how much money is in my account", {"verdict": "same", "row": _rid(m2, BAL)})
    method = m2.ask("what's the price of solana")
    ev["after_restart"] = {"forwarded": rep.get("forwarded"), "wrong_row_again": fixed.get("row") == _rid(m2, BAL),
                           "method_call": method.get("call")}
    learned_after = fixed.get("row") != _rid(m2, BAL) and (rep.get("forwarded") or {}).get("meaning") == "corrected" \
        and MEAN_WRONG in m2.meaning.rows[_rid(m2, PIN)]["phrasings"] \
        and (method.get("call") or {}).get("args") == {"symbol": "SOL"}
    m3 = _restart(m2, root)
    survives = m3.ask(MEAN_WRONG).get("row") != _rid(m3, BAL) and \
        m3.ask("how much money is in my account").get("row") == _rid(m3, BAL)
    rows0 = sum(len(r["phrasings"]) for r in m3.meaning.rows.values())
    m3.meaning_resolve("my password is %s, how much money do i have" % SECRET, {"verdict": "same", "row": _rid(m3, BAL)})
    moved = sum(len(r["phrasings"]) for r in m3.meaning.rows.values()) != rows0
    m3.learning_save(root)
    hits = _secret_hits(root)
    ev["guard"] = {"door_moved_by_secret": moved, "secret_sections": hits}
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": (not moved) and not hits, "evidence": ev}


RANK_STATE = "where is my package right now"
RANK_C = ["reset my password", "cancel my subscription", "track my parcel", "update billing address"]
RANK_TRUTH = "cancel my subscription"             # a deliberate CORRECTION: the static door ranks 'track my parcel'
RANK_DICT = [{"text": "refund", "id": "refund", "examples": ["money back please", "return my money"]},
             {"text": "deliver", "id": "deliver", "examples": ["where is my order"]}]


def door_rank(tmp):
    """mind.rank: decision_outcome(id, label) moves the rank door's ProtoStore (restricted to that call's candidate
    set -- which needs the candidates' ENCODINGS) and feeds door_calibrator('rank')."""
    ev = {}
    m = _mind()
    r = m.rank(RANK_STATE, RANK_C)
    ev["fresh_top"] = r["value"]
    rep = m.decision_outcome(r["id"], RANK_TRUTH)
    live = bool((rep.get("forwarded") or {}).get("learned")) and _cal_n(m, "rank") == 1
    root = _tmp(tmp, "rank")
    m, _ = _boot(root)
    r = m.rank(RANK_STATE, RANK_C)
    rd = m.rank("i want my money returned", RANK_DICT)
    m2 = _restart(m, root)
    rep = m2.decision_outcome(r["id"], RANK_TRUTH)
    rep_d = m2.decision_outcome(rd["id"], "refund")
    ev["after_restart"] = {"forwarded": rep.get("forwarded"), "forwarded_dict_candidates": rep_d.get("forwarded"),
                           "rank_calibrator_pairs": _cal_n(m2, "rank")}
    learned_after = bool((rep.get("forwarded") or {}).get("learned")) and \
        bool((rep_d.get("forwarded") or {}).get("learned"))
    after = m2.rank(RANK_STATE, RANK_C)
    m3 = _restart(m2, root)
    again = m3.rank(RANK_STATE, RANK_C)
    survives = _store_sig(m3, "rank") == _store_sig(m2, "rank") and again["ranked"] == after["ranked"]
    v0 = _store_sig(m3, "rank")
    g = m3.rank("my password is %s where is my package" % SECRET, RANK_C)
    m3.decision_outcome(g["id"], RANK_TRUTH)
    moved = _store_sig(m3, "rank") != v0
    m3.learning_save(root)
    hits = _secret_hits(root)
    ev["guard"] = {"door_moved_by_secret": moved, "secret_sections": hits}
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": (not moved) and not hits, "evidence": ev}


def door_compose(tmp):
    """call_compose: a composed call's outcome feeds the 'compose' door's calibrator (its only learner)."""
    ev = {}

    def comp(m):
        v = m.call_encode("fx", {"from": "EUR", "to": "JPY"})["vector"]
        return m.call_compose(v, {"fx": ["from", "to"]}, {"from": ["EUR", "JPY", "USD"], "to": ["EUR", "JPY", "USD"]})
    m = _mind()
    c = comp(m)
    m.decision_outcome(c["id"], c["call"])
    live = _cal_n(m, "compose") == 1
    root = _tmp(tmp, "compose")
    m, _ = _boot(root)
    c = comp(m)
    m2 = _restart(m, root)
    rep = m2.decision_outcome(c["id"], c["call"])
    ev["after_restart"] = {"compose_calibrator_pairs": _cal_n(m2, "compose"), "forwarded": rep.get("forwarded")}
    learned_after = _cal_n(m2, "compose") == 1
    m3 = _restart(m2, root)
    survives = _cal_n(m3, "compose") == _cal_n(m2, "compose")
    # no text state at this door (a hypervector): nothing a secret could ride in
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": None, "evidence": ev}


VERIFY_TRUTH = {"smooth a bumpy mesh surface": "mesh_smooth", "compress a noisy float series": "series_compress",
                "grow crystals on a surface": "crystal_grow"}
VERIFY_TOOLS = ["mesh_smooth", "series_compress", "crystal_grow", "maze_solve"]


def _verify_mind():
    m = _mind()

    def judge(task, tool):
        return VERIFY_TRUTH.get(task) == tool
    m.judge = judge
    return m


def _vstep(m, task, tool):
    try:
        return m.swarm_step(task, tool, done_when="judge passes", evidence={"tool": tool}, worker="w",
                            verify={"verb": "judge", "args": {"task": task, "tool": tool}}, expect=True)
    except ValueError:
        return None


def door_verify(tmp):
    """The verifier pre-check fed by swarm_step: every verified step (passed OR failed) is a labelled example for
    that tool's success / failure prototypes. Learning happens AT the step (no later outcome), so 'after a restart'
    means: the restarted mind keeps the prototypes and keeps learning on top of them."""
    ev = {}
    task = "smooth a bumpy mesh surface"
    m = _verify_mind()
    _vstep(m, task, "maze_solve")                       # a failed verify: a failure example
    _vstep(m, task, "mesh_smooth")
    pre = m.verify_precheck("smooth this bumpy mesh surface", VERIFY_TOOLS)
    live = pre["order"][0] == "mesh_smooth" and pre["scores"]["maze_solve"] < 0
    root = _tmp(tmp, "verify")
    m, _ = _boot(root)
    m.judge = _verify_mind().judge
    _vstep(m, task, "maze_solve")
    st = _vstep(m, task, "mesh_smooth")
    before = m.verify_precheck("smooth this bumpy mesh surface", VERIFY_TOOLS)
    m2 = _restart(m, root, _verify_mind)
    kept = m2.verify_precheck("smooth this bumpy mesh surface", VERIFY_TOOLS)
    v0 = _store_sig(m2, "verify:series_compress")
    _vstep(m2, "compress a noisy float series", "series_compress")
    rep = m2.decision_outcome(st["id"], "mesh_smooth") if st else {}
    learned_after = kept["scores"] == before["scores"] and _store_sig(m2, "verify:series_compress") != v0
    ev["after_restart"] = {"precheck_equal": kept["scores"] == before["scores"],
                           "step_outcome_reflex": (rep.get("reflex") or {}).get("learned")}
    m3 = _restart(m2, root, _verify_mind)
    survives = m3.verify_precheck("compress this noisy float series", VERIFY_TOOLS)["scores"] == \
        m2.verify_precheck("compress this noisy float series", VERIFY_TOOLS)["scores"]
    v0 = _store_sig(m3, "verify:mesh_smooth")
    _vstep(m3, "my password is %s smooth a bumpy mesh surface" % SECRET, "mesh_smooth")
    moved = _store_sig(m3, "verify:mesh_smooth") != v0
    m3.learning_save(root)
    hits = _secret_hits(root)
    ev["guard"] = {"door_moved_by_secret": moved, "secret_sections": hits}
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": (not moved) and not hits, "evidence": ev}


GUARD_Q = "what is the recovery code for my vault"


def door_guard(tmp):
    """The learning guard's learned examples (learn_guard_example, and the pattern refusals it learns from): a
    guard that forgets its corrections re-flags the same false positive after every restart -- and a guard that
    REMEMBERS A REFUSED QUESTION VERBATIM keeps the very secret it refused."""
    ev = {}
    m = _mind()
    i0 = m.semantic_guard.intent(GUARD_Q)["credential"]
    m.learn_guard_example(GUARD_Q, "credential")
    i1 = m.semantic_guard.intent(GUARD_Q)["credential"]
    live = i1 > i0
    root = _tmp(tmp, "guard")
    m, _ = _boot(root)
    m.learn_guard_example(GUARD_Q, "credential")
    want = m.semantic_guard.intent(GUARD_Q)["credential"]
    m2 = _restart(m, root)
    kept = m2.semantic_guard.intent(GUARD_Q)["credential"]
    m2.learn_guard_example("how many yen for a dollar today", "normal")
    learned_after = abs(kept - want) < 1e-9 and "how many yen for a dollar today" in m2.semantic_guard.learned["normal"]
    m3 = _restart(m2, root)
    survives = "how many yen for a dollar today" in m3.semantic_guard.learned["normal"] and \
        abs(m3.semantic_guard.intent(GUARD_Q)["credential"] - want) < 1e-9
    # the guard's OWN secret hygiene: the pattern layer refuses these teaches -- and learns from each refusal
    m3.teach("my password is %s please remember it" % SECRET, "ok noted")
    m3.teach("remember my api key sk-FAKE0123456789abcdefghijklmnop", "stored")
    m3.learning_save(root)
    hits = _secret_hits(root) + [k for k in _secret_hits(root, "sk-FAKE0123456789abcdefghijklmnop")]
    ev["guard"] = {"secret_sections": sorted(set(hits)),
                   "learned_credential_examples": len(m3.semantic_guard.learned["credential"])}
    return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
            "guard_holds": not hits, "evidence": ev}


# ---- the CLM plugin, contract-tested with a stub (the assumed dialect of tests/test_clm_plugin.py) -------------------
def _words(t):
    return set(re.findall(r"[a-z]+", str(t).lower()))


class _ClmStub:
    """A deterministic /v1/systemone stand-in on 127.0.0.1 (choice questions only): options scored by word overlap
    with their criteria, softmax -> probabilities. No model, no network beyond loopback."""

    def __init__(self):
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                out = stub.systemone(body) if self.path == "/v1/systemone" else {"error": "no route"}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def systemone(body):
        st = _words(body["state"])
        answers = {}
        for qid, q in body["questions"].items():
            opts = list(q["criteria"])
            xs = [len(st & _words(o + " " + str(q["criteria"][o]))) for o in opts]
            mx = max(xs)
            e = [math.exp((x - mx) / 0.5) for x in xs]
            pr = [v / sum(e) for v in e]
            answers[qid] = {"choice": opts[int(np.argmax(pr))], "confidence": max(pr),
                            "probabilities": dict(zip(opts, pr))}
        return {"model": body.get("model"), "answers": answers, "usage": {}}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


CLM_Q = {"topic": {"type": "choice", "options": ["billing", "shipping"],
                   "examples": {"billing": ["charged twice on my card"], "shipping": ["courier lost my parcel"]}}}


def door_clm(tmp):
    """The CLM plugin's records (via model_end, question 'clm:<q>'): decision_outcome(id, truth) calibrates the CLM
    door ('clm:<question>') -- its hook was a live closure."""
    import lecore
    stub = _ClmStub()
    try:
        def factory():
            m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
            m._plugin_load("clm", config={"url": stub.url}, register_in_catalog=False)
            return m
        ev = {}
        m = factory()
        r = m.clm_systemone("courier lost my parcel", CLM_Q)["topic"]
        m.decision_outcome(r["id"], "shipping")
        live = _cal_n(m, "clm:topic") == 1
        root = _tmp(tmp, "clm")
        m, _ = _boot(root, factory)
        r = m.clm_systemone("courier lost my parcel", CLM_Q)["topic"]
        m2 = _restart(m, root, factory)
        rep = m2.decision_outcome(r["id"], "shipping")
        ev["after_restart"] = {"clm_calibrator_pairs": _cal_n(m2, "clm:topic"), "forwarded": rep.get("forwarded")}
        learned_after = _cal_n(m2, "clm:topic") == 1
        m3 = _restart(m2, root, factory)
        survives = _cal_n(m3, "clm:topic") == _cal_n(m2, "clm:topic")
        g = m3.clm_systemone("my password is %s courier lost my parcel" % SECRET, CLM_Q)["topic"]
        m3.decision_outcome(g["id"], "shipping")
        m3.learning_save(root)
        hits = _secret_hits(root)
        ev["guard"] = {"secret_sections": hits}
        return {"learns_live": live, "learns_after_restart": learned_after, "survives_reload": survives,
                "guard_holds": not hits, "evidence": ev}
    finally:
        stub.close()


def door_teach(tmp):
    """teach / resolve / answer_feedback: the durable rails. A taught fact is served at T0 after a restart; an
    ESCALATION recorded before a restart is still open after it (a swarm that restarts must not lose its open
    questions) and resolve() clears it; a payload vetoed by answer_feedback stays vetoed."""
    ev = {}
    m = _mind()
    m.teach("what port does the leCore service use", "8080")
    live_teach = m.ask("what port does the leCore service use").get("answer") == "8080"
    m.serve("which colour is the owner's bicycle")
    live_esc = any(e["question"] == "which colour is the owner's bicycle" for e in m.escalations())
    live = live_teach and live_esc
    root = _tmp(tmp, "teach")
    m, _ = _boot(root)
    m.teach("what port does the leCore service use", "8080")
    m.teach("who maintains the build", "the build bot")
    m.serve("which colour is the owner's bicycle")
    m.serve("my password is %s what is it" % SECRET)
    m.answer_feedback("who maintains the build", ok=False)
    m2 = _restart(m, root)
    open_after = [e["question"] for e in m2.escalations()]
    res = m2.resolve("which colour is the owner's bicycle", "red")
    ev["after_restart"] = {"t0": m2.ask("what port does the leCore service use").get("tier"),
                           "escalation_still_open": "which colour is the owner's bicycle" in open_after,
                           "resolve_cleared": res.get("cleared"),
                           "vetoed_served": m2.ask("who maintains the build").get("answer") == "the build bot"}
    learned_after = ev["after_restart"]["t0"] == "T0" and ev["after_restart"]["escalation_still_open"] \
        and res.get("cleared") and not ev["after_restart"]["vetoed_served"]
    m3 = _restart(m2, root)
    survives = m3.ask("which colour is the owner's bicycle").get("answer") == "red" and \
        not any(e["question"] == "which colour is the owner's bicycle" for e in m3.escalations()) and \
        m3.ask("who maintains the build").get("answer") != "the build bot"
    m3.learning_save(root)
    hits = _secret_hits(root)
    ev["guard"] = {"secret_sections": hits}
    return {"learns_live": live, "learns_after_restart": bool(learned_after), "survives_reload": survives,
            "guard_holds": not hits, "evidence": ev}


DOORS = {
    "typed_nb": lambda t: door_typed(t, "nb"),
    "typed_contrastive": lambda t: door_typed(t, "contrastive"),
    "typed_absorb": door_typed_absorb,
    "route_catalog": door_route,
    "route_reflex": door_route_reflex,
    "serve_tool_reflex": door_tool,
    "serve_router": door_serve_route,
    "meaning": door_meaning,
    "rank": door_rank,
    "call_compose": door_compose,
    "verify_swarm_step": door_verify,
    "learn_guard": door_guard,
    "clm_plugin": door_clm,
    "teach_resolve_feedback": door_teach,
}


# =====================================================================================================================
# LOOP-LEVEL CHECKS (not one door each)
# =====================================================================================================================
def check_rollover_sections(tmp):
    """Every learning section survives a rollover, and a section this build does NOT know is kept (the container's
    promise: 'the reader is expected to keep it and write it back out'). One rich mind: every door has learned."""
    from holographic.io_and_interop.holographic_container import load_container, save_container
    from holographic.agents_and_reasoning.holographic_protostore import TeacherNoise
    root = _tmp(tmp, "sections")
    m, _ = _boot(root)
    a = m.systemone_decide(TYPED_S, TYPED_Q, scorer="nb", encoder="ngram")["cat"]
    m.decision_outcome(a["id"], TYPED_TRUTH)
    r = m.route_tiered(TIE_Q)
    m.decision_outcome(r["id"], TIE_USED)
    rk = m.rank(RANK_STATE, RANK_C)
    m.decision_outcome(rk["id"], RANK_TRUTH)
    m.teach(BAL, BAL_A)
    m.meaning.teacher = TeacherNoise(m=7, min_pairs=20)
    for i in range(3):
        m.meaning.teacher.observe("row:a", "row:a" if i else "row:b")
    m.direction_learn("convert dollars to pesos", {"from": "dollars", "to": "pesos"})
    m.learn_guard_example(GUARD_Q, "credential")
    m.serve("which colour is the owner's bicycle")
    m.learning_save(root)
    f1 = _files(root)[-1]
    got = load_container(open(f1, "rb").read())
    secs = got["sections"] + [{"kind": "lecore.future.section", "id": "v9", "meta": {"keep": "me"},
                               "arrays": {"x": np.arange(3, dtype=np.float64)}}]
    open(f1, "wb").write(save_container(secs, meta=got["meta"]))
    before = sorted(s["kind"] for s in secs)
    m2, rep = _boot(root)
    after = sorted(_kinds(root))
    lost = sorted(set(before) - set(after))
    # the direction reader when there are NO meaning rows (it rode inside the meaning section only)
    root_d = _tmp(tmp, "direction_only")
    md, _ = _boot(root_d)
    md.direction_learn("convert dollars to pesos", {"from": "dollars", "to": "pesos"})
    n_dir = int(sum(md._direction_reader().counts.values())) if hasattr(md._direction_reader(), "counts") else None
    md2 = _restart(md, root_d)
    rd2 = md2.__dict__.get("_direction_reader_obj")
    kept_dir = rd2 is not None and bool(getattr(rd2, "counts", None))
    return {"kinds_before": before, "kinds_after": after, "lost_through_rollover": lost,
            "rolled": rep.get("rolled"), "direction_reader_without_meaning_rows_kept": kept_dir,
            "direction_counts_before": n_dir, "ok": not lost and kept_dir}


def check_verify_state(tmp):
    """verify_decision's displacement PROFILE and drift stream: learned from every reported outcome, used by
    every verdict -- do they survive a restart?"""
    root = _tmp(tmp, "verify_state")
    m, _ = _boot(root)
    stream = [("my card was charged twice this month", "billing"), ("the courier never showed up", "shipping"),
              ("refund the double charge on my invoice", "billing"), ("where is my parcel, tracking is frozen",
                                                                       "shipping"),
              ("i was billed twice for one order", "billing"), ("the package arrived late and damaged", "shipping"),
              ("charge on my card is wrong", "billing"), ("my order is lost", "shipping"),
              ("double billed again", "billing")]
    for s, t in stream:
        a = m.systemone_decide(s, TYPED_Q, scorer="nb", encoder="ngram")["cat"]
        m.decision_outcome(a["id"], t)
    v1 = m.verify_decision(stream[0][0], "billing")
    m2 = _restart(m, root)
    v2 = m2.verify_decision(stream[0][0], "billing")
    return {"before": {k: v1["checks"].get(k) for k in ("profile", "drift_z")},
            "after_restart": {k: v2["checks"].get(k) for k in ("profile", "drift_z")},
            "ok": v2["checks"].get("profile") is not None and v2["checks"].get("drift_z") is not None
            and abs((v2["checks"]["profile"] or 0) - (v1["checks"]["profile"] or 0)) < 1e-6}


def check_retile_outcome_fields(tmp):
    """reflex_retile rebuilds the trace from the audit log; the outcome fields (reported failures) are not in the
    audit -- does the retile carry them (as a split and a reload now do, E2.1)?"""
    m = _mind()
    rng = np.random.default_rng(0)
    for i in range(5):
        k = rng.standard_normal(2048)
        m.reflex_write(k, rng.standard_normal(2048))
        m.reflex_outcome(k, False)
    s0, f0 = m.experience.outcome_fields()
    m.reflex_retile(advisory_load=0.03)
    s1, f1 = m.experience.outcome_fields()
    return {"fail_norm_before": float(np.linalg.norm(f0)), "fail_norm_after": float(np.linalg.norm(f1)),
            "ok": bool(abs(float(np.linalg.norm(f1)) - float(np.linalg.norm(f0))) < 1e-6 * max(1.0, np.linalg.norm(f0)))}


def check_ladder_payload_keys(tmp):
    """Both arms: the ladder as shipped, and the BASELINE with the layout sync and the feedback-target check switched
    off (what the ladder did before the audit's fix) -- one run reproduces the before and the after."""
    fixed = _ladder_arm(tmp, legacy=False)
    return {"fixed": fixed, "baseline_no_sync": _ladder_arm(tmp, legacy=True), "ok": fixed["ok"]}


@contextlib.contextmanager
def _legacy_ladder():
    """The pre-fix ladder: payload keys never follow a re-layout, and a correction marks whatever atom the read
    fires (the old answer_feedback / load re-mark rule)."""
    from holographic.agents_and_reasoning import holographic_zoo as Z
    sync, target = Z.AnswerLadder._payload_sync, Z.AnswerLadder._feedback_target

    def old_target(self, queries, hit, tile):
        return "%d:%d" % (int(tile), int(hit.get("atom", -1))) if (hit or {}).get("fired") else None
    Z.AnswerLadder._payload_sync = lambda self: 0
    Z.AnswerLadder._feedback_target = old_target
    try:
        yield
    finally:
        Z.AnswerLadder._payload_sync, Z.AnswerLadder._feedback_target = sync, target


def _ladder_arm(tmp, legacy=False):
    """The ladder's T0 payloads are keyed '<tile>:<atom index>'. A tile SPLIT (every ~61 writes per tile at the
    measured advisory load 0.03) replays the tile into two new ones, so atom indices shift and half the questions
    move to a new tile index. Measures how many payload keys still locate their own question after a live split,
    how many fuzzy near-repeats (a stop word added: same key, not an exact repeat) are still served, and whether a
    correction after the split vetoes ONLY what it corrected, after a restart."""
    ctx = _legacy_ladder() if legacy else contextlib.nullcontext()
    with ctx:
        return _ladder_arm_body(_tmp(tmp, "ladder_legacy" if legacy else "ladder_fixed"))


def _ladder_arm_body(tmp):
    m = _mind()
    lad = m.zoo["ladder"]
    qs = ["what is the storage code for warehouse %s shelf %s" % (w, s) for w in ("north", "south", "east", "west",
          "upper", "lower", "inner", "outer", "old", "new") for s in ("alpha", "beta", "gamma", "delta", "omega",
                                                                         "kappa", "sigma", "theta", "zeta")]
    for i, q in enumerate(qs):
        m.teach(q, "bin %d" % i)
    splits = m.experience.splits
    stale = _stale_payload_keys(m)
    misrouted = _misrouted_questions(m)
    fuzzy = [m.ask(q + " please") for q in qs]
    served_right = sum(1 for i, a in enumerate(fuzzy) if a.get("answer") == "bin %d" % i)
    served_wrong = sum(1 for i, a in enumerate(fuzzy) if a.get("answer") not in (None, "", "bin %d" % i))
    via = {}
    for a in fuzzy:
        via[str(a.get("via"))] = via.get(str(a.get("via")), 0) + 1
    # A CORRECTION AFTER THE SPLIT: answer_feedback(q, ok=False) on three early questions. The payload it vetoes is
    # located by the CURRENT (tile, atom); the bad mark is persisted as the QUESTION the stale key names. After a
    # restart, which questions are no longer served? Only the three that were corrected may be.
    root = _tmp(tmp, "p")
    m.learning_rollover(root)
    for i in (0, 1, 2):
        m.answer_feedback(qs[i], ok=False)
    m2 = _restart(m, root)
    lost = [i for i, q in enumerate(qs) if m2.ask(q).get("answer") != "bin %d" % i]
    wrong_veto = [i for i in lost if i not in (0, 1, 2)]
    return {"taught": len(qs), "splits": splits, "stale_payload_keys": stale, "payload_keys": len(lad._payload_qs),
            "misrouted_questions": misrouted,
            "fuzzy_repeat_served_right": served_right, "fuzzy_repeat_served_wrong": served_wrong, "via": via,
            "corrected": [0, 1, 2], "not_served_after_restart": lost, "wrongly_vetoed": wrong_veto,
            "ok": stale == 0 and served_wrong == 0 and not wrong_veto}


def _stale_payload_keys(m):
    """Payload keys whose (tile, atom) is NOT their question's content-derived payload atom -- a STALE key (the
    layout moved under it). Separately counted by _misrouted_questions: keys that are right but whose question now
    ROUTES to another tile (a trace-routing property: reads of that question fire elsewhere)."""
    from holographic.agents_and_reasoning.holographic_lever7 import key_atom
    lad = m.zoo["ladder"]
    stale = 0
    for pk, q in list(getattr(lad, "_payload_qs", {}).items()):
        try:
            t, k = (int(x) for x in str(pk).split(":"))
            a = np.asarray(m.experience.tiles[t]._atoms[k], float)
        except (ValueError, IndexError):
            stale += 1
            continue
        pa = key_atom("ans#" + str(q)[:64], 2048)
        stale += int(float(a @ pa / (np.linalg.norm(a) * np.linalg.norm(pa) + 1e-12)) < 0.99)
    return stale


def _misrouted_questions(m):
    lad = m.zoo["ladder"]
    n = 0
    for pk, q in list(getattr(lad, "_payload_qs", {}).items()):
        n += int(m.experience._route(lad._qkey(q)) != int(str(pk).split(":")[0]))
    return n


def check_lookup_side_effects(tmp):
    """Lookups that ADD ledger records as a side effect (a reflex fire inside find_capability / suggest / serve,
    a route inside serve): records nobody can report against (find_capability returns no id) still count toward
    the 4,096 records a save keeps."""
    m = _mind()
    r = m.route_tiered(REFLEX_Q)
    m.decision_outcome(r["id"], r["answer"])
    out = {}
    for name, fn in (("find_capability", lambda: m.find_capability(REFLEX_Q)),
                     ("suggest", lambda: m.suggest(REFLEX_Q)),
                     ("serve_seen", lambda: m.serve(REFLEX_Q)),
                     ("serve_unseen", lambda: m.serve("an unrelated question about gardening tools"))):
        n0 = len(m.decision_ledger()._order)
        fn()
        out[name] = len(m.decision_ledger()._order) - n0
    return out


def check_record_sizes(tmp):
    """Bytes of one persisted record as text (json of to_text()) per door -- what the 4,096-record cap costs."""
    m = _mind()
    a = m.systemone_decide(TYPED_S, TYPED_Q, scorer="nb", encoder="ngram")["cat"]
    cands = ["candidate intent number %d with a few words" % i for i in range(150)]
    r = m.rank("which intent is this about", cands)
    rt = m.route_tiered(TIE_Q)
    L = m.decision_ledger()
    size = {n: len(json.dumps(L.get(i).to_text(), sort_keys=True)) for n, i in
            (("typed", a["id"]), ("rank_150", r["id"]), ("route", rt["id"]))}
    size["rank_to_typed_ratio"] = round(size["rank_150"] / float(size["typed"]), 2)
    return size


def check_drift_number(tmp):
    """What learning_save's drift_vs_previous_save measures: the cosine between two partition fingerprints, each a
    bundle over sections of a random vector seeded by sha256 of the FIRST 2,000 CHARACTERS of the section's sorted
    meta (holographic_unified_p22_zoo2.partition_fingerprint). Scenarios: nothing changed; one ask (query log only);
    one typed correction (the learned tables live in ARRAYS); one taught row."""
    root = _tmp(tmp, "drift")
    m, _ = _boot(root)
    m.teach("seed row one", "a")
    m.learning_save(root)

    def save():
        r = m.learning_save(root)
        return {"drift": r["drift_vs_previous_save"], "sections_changed": r.get("sections_changed")}
    out = {"nothing_changed": save()}
    m.ask("what is the weather like on mars")
    out["one_ask"] = save()
    a = m.systemone_decide(TYPED_S, TYPED_Q, scorer="contrastive", encoder="ngram")["cat"]
    m.learning_save(root)
    m.decision_outcome(a["id"], TYPED_TRUTH)
    out["one_typed_correction"] = save()
    m.teach("seed row two", "b")
    out["one_taught_row"] = save()
    out["sections"] = len(_kinds(root))
    return out


def check_guard_phrasings(tmp):
    """Which ordinary ways of pasting a password does the learning guard's PATTERN layer catch? (It decides what
    every door refuses to learn and what a save refuses to write.) Reported, not fixed here: holographic_learnguard
    belongs to another worker this pass."""
    from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
    phr = ["my password is %s", "password: %s", "password %s", "use password %s to log in", "the pw is %s",
           "login with %s", "my passphrase is %s", "pwd=%s", "here is my password, %s", "%s is my password"]
    got = {p % "<S>": bool(sensitive_reason(p % SECRET)) for p in phr}
    return {"caught": sum(got.values()), "of": len(got), "by_phrasing": got}


def check_reflex_floor(tmp, data_dir):
    """reflex_decide(min_confidence=0.0) is the default every caller gets, while its docstring says 0.1 is MEASURED
    (a 600-row Banking77 stream: every wrong fire on a NOVEL row sat below 0.1 -- a measurement that may predate the
    seen gate, which now refuses novel rows by itself). Re-measured here on the typed door (systemone_decide with
    reflex=True, nb scorer, the truth reported after every decision): 13 Banking77 intents, 5 examples each, a stream
    of 300 distinct held-in rows, and the same stream with 300 exact REPEATS interleaved (what the reflex is for).
    Arms: reflex off; the default floor 0.0; the documented floor 0.1. Needs --data (CC BY 4.0 Banking77 CSV)."""
    import csv
    rows = list(csv.DictReader(open(os.path.join(data_dir, "banking77_train.csv"), encoding="utf-8")))
    cats = sorted({r["category"] for r in rows})[::6]
    by = {c: [r["text"] for r in rows if r["category"] == c] for c in cats}
    q = {"intent": {"type": "choice", "options": cats, "examples": {c: by[c][:5] for c in cats}}}
    pool = sorted(((t, c) for c in cats for t in by[c][5:]),
                  key=lambda tc: hashlib.sha256(tc[0].encode()).hexdigest())[:300]
    rng = np.random.default_rng(0)
    rep = []
    for i, tc in enumerate(pool):
        rep.append(tc)
        if i >= 10:
            rep.append(pool[int(rng.integers(0, i))])
    rep = rep[:600]
    streams = {"novel_300": pool, "with_repeats_600": rep}
    out = {"intents": len(cats), "examples_per_intent": 5}
    for sname, stream in streams.items():
        res = {}
        for arm, floor in (("reflex_off", None), ("floor_0.0_default", 0.0), ("floor_0.1_documented", 0.1)):
            m = _mind()
            if floor is not None:
                m.reflex_decide = (lambda state, key="fingerprint", _m=m, _f=floor, **kw:
                                   type(_m).reflex_decide(_m, state, min_confidence=_f, key=key, **kw))
            right = fires = fires_right = 0
            for text, truth in stream:
                a = m.systemone_decide(text, q, scorer="nb", encoder="ngram", reflex=floor is not None)["intent"]
                ok = a.get("value") == truth
                right += ok
                if a.get("via") == "reflex":
                    fires += 1
                    fires_right += ok
                if a.get("id"):
                    m.decision_outcome(a["id"], truth)
            res[arm] = {"accuracy": round(right / float(len(stream)), 4), "reflex_fires": fires,
                        "reflex_fire_accuracy": round(fires_right / float(fires), 4) if fires else None}
        out[sname] = res
    return out


CHECKS = {
    "guard_phrasings": check_guard_phrasings,
    "rollover_sections": check_rollover_sections,
    "verify_state": check_verify_state,
    "retile_outcome_fields": check_retile_outcome_fields,
    "ladder_payload_keys": check_ladder_payload_keys,
    "lookup_side_effects": check_lookup_side_effects,
    "record_sizes": check_record_sizes,
    "drift_number": check_drift_number,
}


# =====================================================================================================================
# the runner
# =====================================================================================================================
def run(doors=None, checks=None, data=None):
    tmp = tempfile.mkdtemp(prefix="audit_learning_")
    out = {"doors": {}, "checks": {}, "seconds": {}}
    try:
        for name, fn in DOORS.items():
            if doors and name not in doors:
                continue
            t0 = time.perf_counter()
            try:
                res = fn(tmp)
            except Exception as e:                       # one door's crash is a finding, never the end of the audit
                res = {p: None for p in PROPS}
                res["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            out["doors"][name] = res
            out["seconds"][name] = round(time.perf_counter() - t0, 2)
            print("%-24s %s  (%.1fs)" % (name, " ".join("%s=%s" % (p.split("_")[0] + "_" + p.split("_")[-1],
                                                                     _fmt(res.get(p))) for p in PROPS),
                                         out["seconds"][name]), flush=True)
            if res.get("error"):
                print("    ERROR", res["error"], flush=True)
        for name, fn in CHECKS.items():
            if checks is not None and name not in checks:
                continue
            t0 = time.perf_counter()
            try:
                res = fn(tmp)
            except Exception as e:
                res = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}
            out["checks"][name] = res
            out["seconds"]["check:" + name] = round(time.perf_counter() - t0, 2)
            print("check %-22s %s" % (name, json.dumps(res, default=str)[:400]), flush=True)
        if data and (checks is None or "reflex_floor" in checks):
            t0 = time.perf_counter()
            try:
                out["checks"]["reflex_floor"] = check_reflex_floor(tmp, data)
            except Exception as e:
                out["checks"]["reflex_floor"] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}
            out["seconds"]["check:reflex_floor"] = round(time.perf_counter() - t0, 2)
            print("check %-22s %s" % ("reflex_floor", json.dumps(out["checks"]["reflex_floor"])[:600]), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _fmt(v):
    return {True: "PASS", False: "FAIL", None: "n/a"}.get(v, str(v))


def matrix(run_out):
    """The door x property matrix as rows of strings (PASS / FAIL / n/a)."""
    return {d: {p: _fmt(r.get(p)) for p in PROPS} for d, r in run_out["doors"].items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--doors", default=None, help="comma list of doors (default: all)")
    ap.add_argument("--checks", default=None, help="comma list of loop-level checks (default: all; 'none' = skip)")
    ap.add_argument("--label", default="latest", help="file this run under runs.<label> in the evidence JSON")
    ap.add_argument("--out", default=EVIDENCE)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--data", default=None, help="dir with banking77_train.csv: also runs the reflex_floor check")
    a = ap.parse_args(argv)
    doors = set(a.doors.split(",")) if a.doors else None
    checks = None if a.checks is None else (set() if a.checks == "none" else set(a.checks.split(",")))
    t0 = time.perf_counter()
    res = run(doors, checks, data=a.data)
    res["matrix"] = matrix(res)
    res["total_seconds"] = round(time.perf_counter() - t0, 1)
    if not a.no_write:
        ev = {}
        if os.path.exists(a.out):
            try:
                ev = json.load(open(a.out))
            except ValueError:
                ev = {}
        ev.setdefault("about", "tools/audit_learning_loop.py: does each leCore learning door learn live, learn from "
                               "an outcome reported AFTER a restart, keep it through another restart, and never "
                               "persist a (fake) secret. Reproduce: PYTHONHASHSEED=0 python tools/audit_learning_loop.py"
                               " --label <name>. Values: true = PASS, false = FAIL, null = not applicable.")
        ev.setdefault("runs", {})[a.label] = res
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(ev, f, indent=1, sort_keys=True, default=str)
        print("wrote", a.out)
    fails = [(d, p) for d, r in res["doors"].items() for p in PROPS if r.get(p) is False]
    print("FAILS:", len(fails), fails)
    return 0


if __name__ == "__main__":
    sys.exit(main())
