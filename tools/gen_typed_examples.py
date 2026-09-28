"""gen_typed_examples.py -- regenerates every JSON block in docs/TYPED_DECISIONS.md from the LIVE engine, so the
syntax reference cannot drift from the code. Run from the repo root: PYTHONHASHSEED=0 PYTHONPATH=. python3 tools/gen_typed_examples.py
(the PYTHONPATH matters when a PyPI leos-core is also installed -- a script imports whatever lecore is first on sys.path).

It PRINTS every block under its '### title'. With --write it also splices them into docs/TYPED_DECISIONS.md: each
'### title' + json block in the doc whose title this script printed is replaced by the fresh block (sweep 182 --
the splice used to be done by hand, and a hand splice is how the doc's "menu tier" example drifted into the refuse
band unnoticed). Blocks this script does not print (section 8b's plan_change / edit_verified / review) are left
exactly as they are, and are listed so nobody mistakes them for regenerated ones.
"""
import json, os, re, sys
import lecore
m=lecore.UnifiedMind(dim=256,seed=0)
_BLOCKS = {}                                   # title -> the printed markdown block, for --write
def show(title, obj, maxlen=1400):
    s=json.dumps(obj, indent=2, default=str)
    if len(s)>maxlen: s=s[:maxlen]+"\n  ... (truncated)"
    block = "### %s\n```json\n%s\n```" % (title, s)
    _BLOCKS[title] = block
    print(block + "\n")
q={"cat":{"type":"choice","options":["billing","shipping"],
          "examples":{"billing":["card charged twice","refund my invoice fee","charge on my statement"],
                      "shipping":["parcel lost in transit","courier delivery late","package never arrived"]}},
   "priority":{"type":"score","min":1,"max":5,"anchors":[["no rush",1],["whenever you can",1.5],["needs attention this week",3],["urgent",5],["right now",5]]},
   "escalate":{"type":"noul","examples":{"yes":["angry customer","threatening to cancel"],"no":["just a question","curious"]}}}
show("the schema (questions)", q)
lint=m.systemone_lint(q, states=["my card was charged twice. also the parcel is late but not lost"], scorer="nb")
show("systemone_lint -> findings", lint)
lab=[["my card was charged a fee",{"cat":"billing"}],["the parcel is lost",{"cat":"shipping"}],["refund the invoice",{"cat":"billing"}],["courier is late",{"cat":"shipping"}]]
a=m.systemone_decide("courier lost the package", q, labeled=lab, scorer="nb", encoder="ngram", margin=0.0, conformal_alpha=0.1)
show("systemone_decide -> answers (one per question; note id, ranked, margin_gap, p_correct, p_null, set)", a)
rep=m.decision_outcome(a["cat"]["id"], "shipping"); show("decision_outcome(id, truth) -> report", rep)
r=m.route_tiered("smooth a bumpy mesh"); r2=dict(r); r2["options"]=r2["options"][:2]; show("route_tiered -> an answer tier (options cut to 2)", r2)
# "denoise an image" is a real MENU tier (z in the menu band); the old query, "turn things into a mesh", had drifted
# into the refuse band while this block was still titled "a menu tier" (fixed in sweep 182)
menu=m.route_tiered("denoise an image"); m2=dict(menu); m2["options"]=m2["options"][:3]; show("route_tiered -> a menu tier (options cut to 3)", m2)
m.decision_outcome(r["id"], r["answer"])
show("route_tiered(reflex=True) after the outcome: the repeat answered from experience", {k:v for k,v in m.route_tiered("smooth a bumpy mesh", reflex=True).items() if k in ("tier","answer","via","confidence","p_correct","p_null","id","reason")})
show("verify_decision(state, answer) -> verdict", m.verify_decision("smooth a bumpy mesh", r["answer"], key="fingerprint"))
s=m.swarm_step("resolve families", "catalog_families", {"decide": False}, done_when="at least 500 resolve", evidence={"resolved": 501}, worker="w1", verify={"verb":"catalog_families","args":{}}, expect=lambda f: sum(1 for v in f.values() if v[0])>=500)
show("swarm_step -> the published record (verify ran BEFORE acceptance)", s)
p=m.plan_from_request("smooth a bumpy mesh and then grow crystals on a surface", encode=False); show("plan_from_request -> steps", {"steps": p["steps"], "root": repr(p["root"])[:120]})
show("decision_records(k=3) -> ledger view", m.decision_records(k=3))

# ---- sweep 182 (the CLM backlog, docs/TYPED_DECISIONS.md section 8e): the wave-2 doors. Everything above is
# unchanged, so these blocks never move an earlier one. Reading r["p"] would warn (DeprecationWarning): read
# p_correct / p_null, and let json.dumps (which never warns) show the deprecated key while it still ships.
cands=[{"text":"shipping","examples":["parcel lost in transit","package never arrived","courier delivery late"]},
       {"text":"billing","examples":["card charged twice","refund my invoice"]},
       {"text":"account","examples":["reset my password","cannot log in"]}]
rk=m.rank("my parcel never arrived", cands); show("rank(state, candidates) -> one typed record", rk)
show("decision_outcome(rank id, truth) -> the rank door learns", m.decision_outcome(rk["id"], "shipping"))
show("verify_precheck(state, actions) after one verified swarm_step (above)",
     m.verify_precheck("resolve families", ["skill_lint", "catalog_families"]))
m.direction_learn("how many euros do i get for 200 dollars", {"from": "dollars", "to": "euros"})
cq=m.call_from_question("how many yen do i get for 50 pounds", "fx", {"yen": "JPY", "pounds": "GBP"})
cq_show={k: v for k, v in cq.items() if k not in ("roles", "reading")}
cq_show["reading"]={k: cq["reading"][k] for k in ("assignment", "margin", "p", "via")}
show("call_from_question -> a candidate call, never executed (roles and the reader's full ranking cut)", cq_show)
show("serve -> 'use capability X' after a memory miss (the router consult)",
     {k: v for k, v in m.serve("decode a png").items() if k != "example"})
esc=m.serve("denoise an image")
show("serve -> an escalation carrying the capability menu (meaning prompt cut)",
     {"served": esc["served"], "via": esc["via"], "route": esc.get("route"),
      "meaning": {"candidates": (esc.get("meaning") or {}).get("candidates"), "prompt": "(the typed resolution prompt)"}})
show("door_calibration_report -> which doors can say p_correct yet", m.door_calibration_report())


def _write(doc=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "TYPED_DECISIONS.md")):
    """--write: replace every '### title' + ```json block of the doc whose title was printed above; report the rest.
    The title line is matched WITHOUT newlines ([^\\n]+) -- with re.S a greedy title would swallow the whole doc."""
    text = open(doc, encoding="utf-8").read()
    done, kept = [], []
    def rep(mt):
        t = mt.group(1)
        if t in _BLOCKS:
            done.append(t)
            return _BLOCKS[t]
        kept.append(t)
        return mt.group(0)
    text = re.sub(r"### ([^\n]+)\n```json\n.*?\n```", rep, text, flags=re.S)
    open(doc, "w", encoding="utf-8").write(text)
    unused = [t for t in _BLOCKS if t not in done]
    sys.stderr.write("--write: %d blocks replaced; left as they are (not generated here): %s; printed but not in the "
                     "doc (add a '### title' block for them by hand): %s\n" % (len(done), kept, unused))


if "--write" in sys.argv:
    _write()
