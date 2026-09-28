from inv import inv
import json
W = "scratchpad/swarm/w4-measure/"
T = [
("Does InfoNCE softmax-weighted prototype learning beat the centroid plus best-wording blend in the leCore meaning index on CLINC150?",
 "Yes on top-1, by about 2 points; it misses the backlog's own 3-point bar. Measured by w4-measure on 2026-09-26 (" + W + "p2b_proto.py, the index's own sparse meaning_features x idf, unit-sphere prototypes, update P_k += eta/tau*x*(1[k=y]-softmax(cos/tau)_k), tau/eta/epochs chosen on CLINC val and test reported once, 3 shuffle seeds). "
 "(1) Learned index (final/default/idx_clinc.json, 602 rows, trained on its own 8,591 wordings): test top-1 0.8413 -> 0.8616 (+2.0 points, paired-bootstrap CI [+1.3, +2.7], seeds 0.8616-0.8618; 182 questions fixed, 91 broken). AURC falls 0.0941 -> 0.0780. At the oracle threshold, coverage at precision 0.97 rises 38.8% -> 53.1%. "
 "(2) 20 wordings per intent, the backlog E1.1 gate: 0.7513 -> 0.7697 (+1.8, CI [+1.0, +2.5]). That FAILS the >=3-point bar, and out-of-scope served rises 0.2% -> 0.6% at P0.97. "
 "(3) 100 wordings per intent, the dataset-label ceiling: 0.8184 -> 0.8735 (+5.5). "
 "The miss-only AdaptHD update adds +0.5 on the learned index (CI includes 0), +0.6 at k20 and +4.2 at k100. Masking same-intent duplicate rows out of the softmax changes nothing (0.8617 vs 0.8616)."),
("Does subtracting a mean or background vector from the similarity (contrastive readout, centering, backlog E1.4) help the leCore meaning index?",
 "No: kept negative on the sparse meaning index (w4-measure, " + W + "p1_centering.py, p3_compare.py, CLINC150, knob chosen on val, reported on test). "
 "The literal E1.4 formula, score(q,r) - lam*sim(q,mu), is RANK-INVARIANT: sim(q,mu) is the same for every row, so neither top-1 nor the lead s1-s2 can change. It only shifts the confidence g. AURC is 0.0941 -> 0.0944, i.e. flat. Coverage at P0.97 goes +2.0 points (37.7 -> 39.6%) while at P0.95 it goes -1.3; this is an operating-point reshuffle, not a gain. "
 "True centering cos(q-a*mu, v-a*mu): val picks a=0, and top-1 at a=1 is 0.8382 vs 0.8413. All-but-the-top PCA removal: -0.6 points, CI [-0.9, -0.3]. A CSLS hub penalty looked like +4.3 points of coverage at P0.97, but its realised test precision was 0.959 against the 0.97 target. "
 "Why: the space is not anisotropic. The norm of the mean unit row centroid is 0.096 and the singular spectrum is flat (2.59, 2.37, 2.21...)."),
("Is coverage at exactly 97 percent precision a reliable statistic for comparing abstention or confidence signals in the leCore meaning index?",
 "No. For the learned CLINC150 index it is a knife-edge (w4-measure, " + W + "p2b_proto.py). The base index's precision-coverage curve sits at about 0.97 from roughly 5% to 38% coverage. "
 "A threshold chosen on val in-scope plus val's 100 out-of-scope questions gave test coverage 37.7% at realised precision 0.970. The same rule with a hash half of oos_test (500) as the calibration out-of-scope set gave 3.3% at 0.993. A coverage delta at P0.97 can therefore read +44 points when the real signal gain is +14 (oracle, symmetric). "
 "Use AURC, precision at fixed served fractions (20-60%), and realised test precision at a threshold chosen on a disjoint calibration split, plus a paired-bootstrap CI."),
("How should a precision gate threshold be chosen on CLINC150 when val has far fewer out-of-scope questions than test?",
 "Re-weight val's out-of-scope questions to the deployment prior, or calibrate on a disjoint hash half of oos_test. Val has 100 out-of-scope per 3,000 in-scope (3.2%); test has 1,000 per 4,500 (18%). "
 "Measured on the learned index (w4-measure, " + W + "p4_prior_trap.py): unweighted, the val threshold for P0.97 realises 0.967 on test and the P0.95 threshold realises 0.941 (serving 5.6% of out-of-scope). Re-weighted by 6.7x, they realise 0.970 and 0.948 (4.4% out-of-scope). "
 "A second trap: 100 out-of-scope questions are too few to set a 0.97 gate stably. Each one counts 6.7x after re-weighting."),
("Does a candidate-relative softmax probability abstain on out-of-scope questions as well as an absolute cosine score?",
 "Only when the candidate set is large. Measured on InfoNCE-trained meaning-index prototypes, CLINC150 test plus out-of-scope, AURC (lower is better; w4-measure, " + W + "p3_compare.py): "
 "absolute cosine confidence g = 2*s1 - s2 scores 0.0780; softmax p_max over all 602 rows 0.0797; over the top 8 candidates only (what the typed prompt shows) 0.0855; over the top 2 only 0.0942. The top-2 figure gives back the entire InfoNCE gain (the base index is 0.0941). "
 "CLM's no-none-of-these weakness is a small-supplied-candidate-set phenomenon. Gate it at the candidate count each door actually uses."),
("Can the E1.3 NULL none-of-these hypervector be trained from new verdicts in the leCore meaning index?",
 "Not as the backlog states it. On the learned CLINC150 index (15k-question stream, perfect stand-in teacher; final/default/idx_clinc.json) the teacher's 'new' verdicts created 452 rows. Only 39 are out-of-scope; 413 (91%) are duplicate rows of in-scope intents whose right row was not among the 8 candidates shown. "
 "119 of 150 intents are split over 2-20 rows, and every row has its own answer key (602 rows, 602 akeys). A NULL learned from 'new' verdicts would mostly learn in-scope questions the ranker missed. It needs a verdict that says off-domain. "
 "The same fact makes the E2.1 rule (never store a negative that shares an answer key) vacuous. Audited: 0 of 1,835 stored negatives point at a same-intent row. (w4-measure, " + W + "p0_baseline.py)"),
("What does the unknown-word mass in the meaning index query norm contribute to abstention?",
 "Most of the abstention at high precision. MeaningIndex._query_vec counts unknown words in the query norm. Dropping that term leaves top-1 unchanged (0.8413) but collapses CLINC150 test coverage at precision 0.97 from 37.7% to 3.4%, and at 0.95 from 56.9% to 37.5% (thresholds chosen on val, oos re-weighted). "
 "The part of a question the index cannot represent is the out-of-scope signal. Any new readout (E1.1 prototypes, E1.4, E6.1 superposed encoders) must keep it. (w4-measure, " + W + "p1_centering.py, ablation U)"),
]
for q, a in T:
    r = inv("teach", query=q, answer=a)
    print(q[:70], "->", json.dumps(r)[:160])
