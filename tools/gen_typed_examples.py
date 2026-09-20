"""gen_typed_examples.py -- regenerates every JSON block in docs/TYPED_DECISIONS.md from the LIVE engine, so the
syntax reference cannot drift from the code. Run from the repo root: PYTHONHASHSEED=0 PYTHONPATH=. python3 tools/gen_typed_examples.py
(the PYTHONPATH matters when a PyPI leos-core is also installed -- a script imports whatever lecore is first on sys.path).
"""
import json, lecore
m=lecore.UnifiedMind(dim=256,seed=0)
def show(title, obj, maxlen=1400):
    s=json.dumps(obj, indent=2, default=str)
    if len(s)>maxlen: s=s[:maxlen]+"\n  ... (truncated)"
    print("### %s\n```json\n%s\n```\n" % (title, s))
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
show("systemone_decide -> answers (one per question; note id, ranked, margin_gap, p, set)", a)
rep=m.decision_outcome(a["cat"]["id"], "shipping"); show("decision_outcome(id, truth) -> report", rep)
r=m.route_tiered("smooth a bumpy mesh"); r2=dict(r); r2["options"]=r2["options"][:2]; show("route_tiered -> an answer tier (options cut to 2)", r2)
menu=m.route_tiered("turn things into a mesh"); m2=dict(menu); m2["options"]=m2["options"][:3]; show("route_tiered -> a menu tier (options cut to 3)", m2)
m.decision_outcome(r["id"], r["answer"])
show("route_tiered(reflex=True) after the outcome: the repeat answered from experience", {k:v for k,v in m.route_tiered("smooth a bumpy mesh", reflex=True).items() if k in ("tier","answer","via","confidence","p","id","reason")})
show("verify_decision(state, answer) -> verdict", m.verify_decision("smooth a bumpy mesh", r["answer"], key="fingerprint"))
s=m.swarm_step("resolve families", "catalog_families", {"decide": False}, done_when="at least 500 resolve", evidence={"resolved": 501}, worker="w1", verify={"verb":"catalog_families","args":{}}, expect=lambda f: sum(1 for v in f.values() if v[0])>=500)
show("swarm_step -> the published record (verify ran BEFORE acceptance)", s)
p=m.plan_from_request("smooth a bumpy mesh and then grow crystals on a surface", encode=False); show("plan_from_request -> steps", {"steps": p["steps"], "root": repr(p["root"])[:120]})
show("decision_records(k=3) -> ledger view", m.decision_records(k=3))
