import json
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad")
from inv import inv

T = [
    ("can I report a decision outcome by id after the leCore service restarts",
     "No (measured by w3-reflex 2026-09-26 in scratch): after learning_save + learning_load, decision_records() has 0 "
     "records and decision_outcome(<id issued before the save>) raises KeyError 'no decision record'. The DecisionRecord "
     "ledger and its hooks live in process memory only (no lecore.learning section carries them; the section named "
     "'ledger' is the TokenLedger, holographic_unified_p22_zoo2.py:1349). SystemOne's learned count tables "
     "(_systemone_cache, p08_bake.py:265) and verify_decision's profile (_verify_profile, p27_decisions.py:392) are not "
     "persisted either, so typed-decision learning from outcomes is lost on restart; the reflex bridge (labels + seen "
     "keys) and the meaning index are persisted."),
    ("does the p field mean the same thing in route_tiered and typed decisions",
     "No. route_tiered's catalog path returns p = (count of null draws >= top score + 1)/(n+1), a null p-value where LOW "
     "means confident (holographic_catalog.py:427; live: answer tier p 0.0154, refuse tier p 0.754). Its reflex path "
     "returns p = 1 - error_prob, HIGH means confident (holographic_unified_p01_read.py:790, p27_decisions.py:223). "
     "SystemOne p is isotonic P(correct) from margin_gap (holographic_systemone.py:584); the meaning rung p is "
     "P(the model would agree) (holographic_meaning.py:464). route() forwards route_tiered's p unchanged "
     "(p08_bake.py:766). One field name, opposite directions, even inside one door (found by w3-reflex 2026-09-26)."),
    ("do the reflex doors share one calibration",
     "Yes, and they should not: _reflex_calib_pairs is appended by the ladder's fuzzy T0 reflex via answer_feedback "
     "(holographic_unified_p22_zoo2.py:1037, confidence of a _qkey read) and by the reflex bridge via reflex_learn "
     "(p27_decisions.py:147, confidence of a fingerprint/ngram read); calibrate_reflex fits ONE isotonic map from both "
     "(p22:908) and both consume it -- the ladder's calib_veto (holographic_zoo.py:259) and reflex_decide's p (p27:223). "
     "Three other isotonic calibrators exist with different score definitions and minimum counts: SystemOne per question "
     "(margin_gap, >=8, systemone.py:582), MeaningIndex (s1+(s1-s2), >=40, meaning.py:462). Found by w3-reflex 2026-09-26."),
    ("how many stores does the leCore reflex arc keep for one decision",
     "Mapped by w3-reflex 2026-09-26: ONE experience trace (TiledDisplacementTrace, p19_lever7.py:21) superposes three "
     "key encoders -- the ladder's _qkey content bag (holographic_zoo.py:122), the routing fingerprint and hashed n-grams "
     "(p27_decisions.py:85) -- and two atom codebooks (ans#<question> payload atoms zoo.py:455, answer:<label> atoms "
     "p27:74); the calibrated null is computed against the union codebook (holographic_lever7.py:209), so one door's "
     "writes move the other's gate. On the service (967 writes, 8 tiles) 476 of 482 reads fired (98.8%): the trace null "
     "barely refuses; the text vetoes (jaccard 0.75, digits, session) and the seen gate (cos 0.8) do the protecting. "
     "Tool choice is learned twice: UsageTrace via tool_note (p19:323, semantic_key) which serve never reads (only "
     "present_tools, p23_zoo3.py:1124) and the bridge via reflex_decide(key=ngram) (p19:285) which serve does read. "
     "Negatives live in three unconnected forms: the region-wide failure field (lever7.py:264), _meaning_neg exact-wording "
     "vetoes (p28_meaning.py:80) and the ladder's _payload_bad/_vetoed_qs."),
]
for q, a in T:
    r = inv("teach", query=q, answer=a)
    print(q, "->", json.dumps(r)[:200])
