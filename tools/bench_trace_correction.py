"""bench_trace_correction.py -- backlog E1.5: how should the reflex experience trace store a CORRECTION?

The served answer a was wrong and the reported truth is y. Three arms, all on the REAL lever-7 classes
(TiledDisplacementTrace at the mind's advisory_load 0.03, dim 2048) and the REAL correction functions the mind calls
(holographic_lever7.correct_lms_apa / correct_by_provenance):

    current     write(k, y) only -- what reflex_learn did before E1.5 (the BASELINE)
    lms_apa     design (a): codebook LMS + affine projection (bind(u, err), u orthogonal to protected neighbours)
    provenance  design (b): write(k, y), then a signed along-atom negative if a was written AT this key, else an
                outcome-role negative bind(NEG, a) that the read consults
    lms_noproj  design (a) with the protection band switched off (u = k): the kept comparison that shows what the
                affine projection buys (w2's 'signed' arm, generalised to the codebook)

Keys: hashed char-n-gram vectors (holographic_systemone.hashed_ngram_encode, dim 2048 -- the reflex's 'ngram' key) of
REAL CLINC150 questions; labels derived_atom(0, 'answer:' + intent, 2048) -- the bridge's own label atoms. 40
background (question, intent) pairs per trace (load 0.02, under the 0.03 tile advisory), as in the panel's exp_d.
    Case A  the wrong lesson was written at the SAME key (a noisy verdict, later corrected); 1 and 3 corrections
    Case B  the wrong answer LEAKED from a neighbour key k' (cosine c, synthetic mix k' = c k + sqrt(1-c^2) n) whose
            own truth IS a; after the correction at k, does k read y and does k' still read a? c in 0.4 .. 0.9
    Case C  FLIP-FLOP at one key -- the mind's real path (found by tests/test_arc_substrate.py): a door answered y,
            a noisy verdict reported a (itself a correction, stored by the arm), the reflex then served a, and the
            truth y was reported back. Does k read y?
Metrics (fractions over trials): k_reads_truth (argmax over the label codebook, exp_d's read), k_gated_truth (the
gated read the reflex actually serves: fired and cleaned up to y), kprime_keeps_its_answer, bg_ok (20 background rows
still read back right), replay_agrees (the reads at k, k' and the background are identical after (1) a state
round-trip -- what a reload does -- and (2) a re-tile from the audit -- what reflex_retile / a split do).

Acceptance (the backlog): same-key >= 95% (baseline 45%), neighbour kept >= 85% at c = 0.8 (baseline 90%), replay 100%.

Usage:  PYTHONHASHSEED=0 python tools/bench_trace_correction.py [--data DIR] [--trials 150] [--out PATH]
Data: clinc150_full.json (CLINC150, CC BY 3.0) -- not vendored; --data points at the directory holding it.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic.agents_and_reasoning.holographic_ai import derived_atom, unbind  # noqa: E402
from holographic.agents_and_reasoning.holographic_lever7 import (  # noqa: E402
    DisplacementTrace, TiledDisplacementTrace, correct_by_provenance, correct_lms_apa)
from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode  # noqa: E402

D = 2048
ARMS = ("current", "lms_apa", "provenance", "lms_noproj")


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


class Bench:
    def __init__(self, data_dir, n_pool=1200):
        path = os.path.join(data_dir, "clinc150_full.json")
        raw = open(path, "rb").read()
        self.data_sha256 = hashlib.sha256(raw).hexdigest()
        d = json.loads(raw)
        self.intents = sorted({i for _, i in d["train"]})
        self.book = {i: derived_atom(0, "answer:" + i, D) for i in self.intents}
        self.L = np.stack([self.book[i] for i in self.intents])
        # the outcome role for design (b): a unitary atom, as w2 used (bind/unbind with it is exact)
        self.NEG = derived_atom(0, "role:outcome_neg", D, unitary=True)
        self.enc = hashed_ngram_encode(dim=D)
        rng = np.random.default_rng(0)
        self.pool = [d["train"][i] for i in rng.choice(len(d["train"]), n_pool, replace=False)]
        self._cache = {}

    def key(self, text):
        if text not in self._cache:
            self._cache[text] = _unit(np.asarray(self.enc(text), float))
        return self._cache[text]

    # -- reads -------------------------------------------------------------------------------------------------
    def read(self, tr, k, arm):
        """exp_d's read: argmax over the label codebook of the raw read in k's tile (design (b) applies its veto)."""
        tile = tr.tiles[tr._route(k)]
        r = unbind(tile._trace, k)
        s = self.L @ _unit(r)
        if arm.startswith("provenance"):
            neg = self.L @ _unit(unbind(r, self.NEG))
            s = np.where(neg > s, -1.0, s)
        return int(np.argmax(s))

    def gated(self, tr, k, arm):
        """What the reflex SERVES: read_gated (null, trust, volatility, outcome field) then atom -> label by cosine
        >= 0.5 (reflex_decide's rule). -1 when it does not fire or the atom is not a label."""
        g = tr.read_gated(k, neg_role=(self.NEG if arm.startswith("provenance") else None))
        if not g.get("fired"):
            return -1
        s = self.L @ _unit(np.asarray(g["prediction"], float))
        j = int(np.argmax(s))
        return j if s[j] >= 0.5 else -1

    # -- the correction under test -------------------------------------------------------------------------------
    def correct(self, tr, k, y, a, arm, seen):
        if arm == "current":
            tr.write(k, self.L[y])
        elif arm == "lms_apa":
            correct_lms_apa(tr, k, self.intents[y], self.book, seen=seen)
        elif arm == "lms_noproj":
            correct_lms_apa(tr, k, self.intents[y], self.book, seen=seen, protect_cos=2.0)   # nothing protected
        elif arm == "provenance":
            correct_by_provenance(tr, k, self.L[y], self.L[a], self.NEG)
        elif arm == "provenance_noretract":
            correct_by_provenance(tr, k, self.L[y], self.L[a], self.NEG, retract=False)
        else:
            raise ValueError(arm)

    def trial(self, t, arm, case, c=0.8, n_corr=1):
        r = np.random.default_rng(100 + t)
        idx = r.choice(len(self.pool), 42, replace=False)
        bg = [self.pool[i] for i in idx[:40]]
        q_text, q_int = self.pool[idx[40]]
        y = self.intents.index(q_int)
        a = int(r.integers(len(self.intents)))
        while a == y:
            a = int(r.integers(len(self.intents)))
        tr = TiledDisplacementTrace(dim=D, seed=0, advisory_load=0.03)       # the mind's experience trace
        seen = []
        for bt, bi in bg:
            tr.write(self.key(bt), self.L[self.intents.index(bi)])
            seen.append((self.key(bt), bi))                                  # background outcomes were reported
        k = self.key(q_text)
        if case == "C":
            seen.append((k, self.intents[a]))
            self.correct(tr, k, a, y, arm, seen)                             # the noisy verdict, stored by the arm
            kp = None
        elif case == "A":
            tr.write(k, self.L[a])                                           # the wrong lesson at the same key
            seen.append((k, self.intents[a]))
            kp = None
        else:
            noise = r.standard_normal(D)
            noise -= noise.dot(k) * k
            noise /= np.linalg.norm(noise)
            kp = c * k + np.sqrt(1.0 - c * c) * noise                       # cos(k, k') = c
            tr.write(kp, self.L[a])                                          # the neighbour's own (correct) lesson
            seen.append((kp, self.intents[a]))
        bg_before = np.mean([self.read(tr, self.key(bt), arm) == self.intents.index(bi) for bt, bi in bg[:20]])
        seen.append((k, self.intents[y]))                                    # reflex_learn appends before writing
        for _ in range(n_corr):
            self.correct(tr, k, y, a, arm, seen)
        got = self.read(tr, k, arm)
        out = {"k_reads_truth": got == y, "k_reads_wrong": got == a, "k_gated_truth": self.gated(tr, k, arm) == y}
        probes = [k] + ([kp] if kp is not None else []) + [self.key(bt) for bt, _ in bg[:20]]
        if kp is not None:
            out["kprime_keeps_its_answer"] = self.read(tr, kp, arm) == a
            out["kprime_gated_keeps"] = self.gated(tr, kp, arm) == a
        out["bg_ok"] = float(np.mean([self.read(tr, self.key(bt), arm) == self.intents.index(bi) for bt, bi in bg[:20]]))
        out["bg_ok_before"] = float(bg_before)
        live = [self.read(tr, p, arm) for p in probes]
        # (1) a reload: every tile through to_state / from_state (JSON, as the partition carries it)
        re1 = TiledDisplacementTrace(dim=D, seed=0, advisory_load=0.03)
        re1.tiles = [DisplacementTrace.from_state(json.loads(json.dumps(tl.to_state()))) for tl in tr.tiles]
        re1._centroids, re1._counts = list(tr._centroids), list(tr._counts)
        # (2) a re-tile from the audit, the way the mind's reflex_retile rebuilds it (raw entries verbatim)
        re2 = TiledDisplacementTrace(dim=D, seed=0, advisory_load=0.03)
        for tl in tr.tiles:
            for kk, vv, mode in tl.audit_entries():
                if mode:
                    re2.write_raw(kk, vv, mode)
                else:
                    re2.write(np.asarray(kk, float), np.asarray(vv, float))
        out["replay_agrees"] = (live == [self.read(re1, p, arm) for p in probes]
                                and live == [self.read(re2, p, arm) for p in probes])
        out["reload_bit_identical"] = all(np.array_equal(x._trace, y_._trace) for x, y_ in zip(tr.tiles, re1.tiles))
        return out


def aggregate(rows):
    return {k: round(float(np.mean([float(r[k]) for r in rows])), 3) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad/data")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "docs", "research", "evidence", "bench_trace_correction.json"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    b = Bench(args.data)
    res = {"what": "backlog E1.5 reflex-trace correction, real CLINC keys", "dim": D, "trials": args.trials,
           "background_pairs": 40, "trace": "TiledDisplacementTrace(dim=2048, seed=0, advisory_load=0.03)",
           "data_sha256": b.data_sha256, "arms": list(ARMS),
           "baseline_from_panel": {"same_key_1x": 0.453, "same_key_3x": 0.66, "neighbour_kept_c0.8": 0.9},
           "acceptance": {"same_key": 0.95, "neighbour_kept_c0.8": 0.85, "replay": 1.0}}
    for arm in ARMS:
        for n_corr in (1, 3):
            res["caseA_%s_corr%d" % (arm, n_corr)] = aggregate([b.trial(t, arm, "A", n_corr=n_corr)
                                                                 for t in range(args.trials)])
        for c in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            res["caseB_%s_c%.1f" % (arm, c)] = aggregate([b.trial(t, arm, "B", c=c) for t in range(args.trials)])
        res["caseC_%s_flipflop" % arm] = aggregate([b.trial(t, arm, "C") for t in range(args.trials)])
        print(arm, "done", round(time.perf_counter() - t0, 1), "s", flush=True)
    # the verdict, by the numbers (acceptance: same-key 1x >= 0.95, neighbour kept at 0.8 >= 0.85, replay 1.0)
    verdict = {}
    for arm in ARMS:
        a1 = res["caseA_%s_corr1" % arm]
        b8 = res["caseB_%s_c0.8" % arm]
        rep = min(res[k]["replay_agrees"] for k in res if isinstance(res[k], dict) and k.startswith("case") and
                  ("_%s_" % arm) in k)
        verdict[arm] = {"same_key": a1["k_reads_truth"], "same_key_gated": a1["k_gated_truth"],
                        "flipflop": res["caseC_%s_flipflop" % arm]["k_reads_truth"],
                        "flipflop_gated": res["caseC_%s_flipflop" % arm]["k_gated_truth"],
                        "neighbour_kept_c0.8": b8["kprime_keeps_its_answer"], "neighbour_k_truth_c0.8": b8["k_reads_truth"],
                        "replay": rep, "bg_ok_min": min(res[k]["bg_ok"] for k in res if isinstance(res[k], dict)
                                                         and k.startswith("case") and ("_%s_" % arm) in k),
                        "meets_bar": bool(a1["k_reads_truth"] >= 0.95 and b8["kprime_keeps_its_answer"] >= 0.85
                                          and rep >= 1.0)}
    res["verdict"] = verdict
    res["cpu_s"] = round(time.perf_counter() - t0, 1)
    # KEPT NEGATIVE, measured: design (b) WITHOUT retracting an earlier rejection of the truth (the first cut). A stale
    # NEG(y) record outvotes the fresh truth write when a noisy verdict is corrected back (case C only).
    res["caseC_provenance_noretract_flipflop"] = aggregate([b.trial(t, "provenance_noretract", "C")
                                                            for t in range(args.trials)])
    print(json.dumps(verdict, indent=1))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
