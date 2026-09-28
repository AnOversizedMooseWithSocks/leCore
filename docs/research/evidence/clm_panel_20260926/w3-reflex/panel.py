import json
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad")
from inv import inv

N = [
    "Kohonen lens (LVQ2.1/GLVQ) DISAGREES with E1.1's softmax-weighted update of every rival: LVQ moves only the nearest "
    "correct and nearest wrong prototype, and only when the sample falls in a window around their boundary; unwindowed "
    "push-away is LVQ2.1's known divergence. GLVQ's relative distance mu=(d+ - d-)/(d+ + d-) is the stable form and is "
    "also the per-decision score the calibrator should read. Run the windowed rule as an arm of E1.1, not only InfoNCE.",
    "Kohonen lens on hard negatives: the draft names family siblings; E0.2 measured only 13.5% of top-1 routing errors "
    "are same-family (50.8% where both families resolve, 3,952-card catalog). LVQ's nearest-wrong prototype -- the "
    "ranked candidate that beat the truth, already in every DecisionRecord -- is the miner by definition; family "
    "siblings are a supplement at best.",
    "Kohonen lens on Q2: relevance learning (GRLVQ, one weight per feature group / role) not a full matrix (GMLVQ); a "
    "matrix with a few verdicts per row is the ridge-W failure again (3/12 -> 1/12). A low-rank matrix only after the "
    "E7 curves show enough verdicts per row, and behind the held-out gate.",
    "Kohonen lens on Q3: out-of-scope is not a compact class, so a single learned NULL prototype is the wrong shape; "
    "prototype reject options combine an absolute distance reject with a relative (mu-near-zero) reject. Keep the "
    "absolute floor authoritative, add the relative reject, and let NULL be several prototypes from confirmed-new "
    "clusters if it earns a held-out gain.",
    "Kohonen lens on E2.3: in LVQ a labelled sample sitting deep on the wrong side (mu strongly contradicts the verdict) "
    "is the textbook label-noise signature; the update window IS the quarantine rule, so the noisy-teacher filter "
    "should be the same inequality, not a second mechanism.",
    "Kohonen lens on Q5: one algorithm, many maps -- one update rule and one calibrator class shared by the reflex, "
    "meaning and SystemOne doors, but separate codebooks and separate calibration label streams per door; the shared "
    "_reflex_calib_pairs (p22_zoo2.py:1037 and p27_decisions.py:147 feed one isotonic fit) shows the coupling cost.",
]
for t in N:
    r = inv("panel_note", member="kohonen", text=t, tags=["clm", "reflex"])
    print(json.dumps(r)[:160])
