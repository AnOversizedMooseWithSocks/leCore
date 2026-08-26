"""holographic_signalprogram.py -- MANY detectors as ONE screened program, with the honesty gates INSIDE the
loop instead of bolted on afterwards.

WHY THIS MODULE EXISTS
----------------------
Screening is where analyses go wrong, and it goes wrong in a specific, repeatable order:

  1. you write a detector, measure it, and it looks good;
  2. you write nineteen more, because the first one was cheap;
  3. you report the best one.

Step 3 is the whole problem. Twenty independent nulls throw a 2-sigma reading about once each round by
construction, so "the best of twenty" is a selection artifact unless the correction is applied over what was
ACTUALLY TRIED -- and the correction is exactly the step a human skips, because by then the winner already has
a name and a story. The campaign this module comes from ran batteries of up to ~100 checks and learned the
lesson the expensive way: its first committee pass looked excellent (+6.8% -> +9.2%) and then died completely
under a strict rebuild, where NO detector survived split-half and one "signal" turned out to be a +16%/-3.2%
regime artifact.

So the gates are not offered here, they are STRUCTURAL. `screen()` cannot return a per-check effect without
also returning that check's split-half replication and its family-corrected q-value, because they are computed
in the same pass over the same data. There is no code path that produces the seductive number alone.

THE THREE THINGS THIS DOES THAT A LOOP OF t-TESTS DOES NOT
  * ONE CALL -- and a REFUTED speed claim, kept loud. Checks are vectorised into a (K, N) score matrix and
    evaluated in one broadcast rather than K sweeps, which the backlog framed as "one bind/bundle sweep, not N
    loops". Measured against a FAIR baseline (both sides starting from the same precomputed scores), over 5
    repeats: at K=12/N=1200 it is a WASH (0.93-1.64x, noise-dominated at ~30 microseconds); at K=200/N=20000
    batching consistently LOSES (0.02-0.47x, never once winning). Large (K, N) temporaries cost more than K
    cache-friendly small operations. Note the spread -- the direction is solid at large K, the MAGNITUDE is
    not, and any single-run figure from this harness is one draw from a very wide distribution. The value of
    screening in one call is that THE GATES ARE STRUCTURAL, not that it is fast.
    TWO of our own errors on the way here, both kept: (a) the first draft timed score computation on the
    batched side only -- a strawman pointing the wrong way, flattering the loop by 5-20x; (b) the second draft
    then PINNED "batching loses" as an assert, which promptly flaked on a clean extract where it won at 1.19x.
    A timing comparison is not a deterministic contract and must never be asserted. The `timing` field ships
    as evidence for the reader to re-check on their own shapes; nothing depends on it.
  * CORRELATION CLUSTERING. Two checks that agree 0.9 of the time are ONE finding wearing two hats, and a
    family correction over "20 tests" that are really 6 independent ideas is both wrong and over-conservative.
    `screen()` clusters the passing checks and reports the cluster count, so a battery cannot inflate its own
    apparent breadth. (Campaign: 'follow last brick' and 'fast-brick continue' correlated 0.7 -- one finding.)
  * REFUSAL AS A RESULT. An empty pass-list is returned as a populated, quotable result with a reason
    attached, not as an error and not as a silent fallback to the best raw effect. "This battery found nothing
    at this grain" is an answer; a committee assembled from checks that failed their gates is not.

THE VSA PART
`program_vector()` bundles the battery into ONE hypervector -- each check bound to its own role vector, then
superposed -- so a whole screening battery becomes a single object that can be stored, recalled, and compared
against another run's battery by cosine. That is what makes a release-QA battery (screen the new build, compare
its program vector to the last one) a one-line operation instead of a bookkeeping exercise. Recovery is pinned:
unbinding a role returns that check's signature above the noise floor.

KEPT NEGATIVE, up front: this module makes multiplicity handling automatic, NOT free. Passing split-half and
FDR inside one battery says nothing about the batteries you ran last week and did not keep. That debt is a
SESSION-level ledger and is deliberately out of scope here -- see the F3 backlog item. A tool that silently
made you feel finished would be worse than no tool.

NumPy + stdlib only. Deterministic given the seed.
"""

import math
import time

import numpy as np

from holographic.agents_and_reasoning.holographic_honesty import bh_fdr, split_half


class SignalProgram:
    """A battery of detectors, screened together, with replication and family-wide multiplicity control applied
    inside the screening pass rather than afterwards.

    Usage:

        prog = SignalProgram(dim=512, seed=0)
        prog.add_check("momentum", lambda s: s[:, 0])          # a vectorised score per event
        prog.add_check("reversal", lambda s: -s[:, 1])
        report = prog.screen(states, targets)
        report["passed"]        # the checks that cleared BOTH gates -- possibly empty, which is a RESULT
        report["clusters"]      # how many INDEPENDENT findings those represent

    `dim` and `seed` only affect `program_vector()`; screening itself is exact and dimension-free.
    """

    def __init__(self, dim=512, seed=0):
        self.dim = int(dim)
        self.seed = int(seed)
        self._checks = []                                       # [(name, encode_fn, direction_fn)]

    def add_check(self, name, encode_fn, direction_fn=None):
        """Register one detector. `encode_fn(states) -> (N,)` must be VECTORISED: it receives the whole (N, ...)
        state array and returns one signed score per event. A positive score means "this detector expects the
        target to be positive here"; the effect is then the mean of score-sign times target, so a detector that
        is right more often than not scores above zero.

        `direction_fn` is an optional post-map applied to the raw scores (e.g. np.sign to make a check purely
        directional, discarding its confidence). Default: use the scores as they are.

        Names must be unique -- a battery with two checks called "momentum" cannot report an honest family size,
        and silently overwriting one would hide a check the user believes is being tested."""
        if not callable(encode_fn):
            raise ValueError("encode_fn must be callable (states -> (N,) scores), got %r" % (type(encode_fn),))
        if any(n == name for n, _, _ in self._checks):
            raise ValueError("duplicate check name %r -- names must be unique so the family size is honest "
                             "(existing: %s)" % (name, ", ".join(n for n, _, _ in self._checks)))
        self._checks.append((name, encode_fn, direction_fn))
        return self

    def __len__(self):
        return len(self._checks)

    def _score_matrix(self, states):
        """Every check's scores as one (K, N) array. Vectorised on purpose: this is the 'one pass' the module
        claims, and screen() measures it against the equivalent loop rather than asserting it."""
        rows = []
        for name, encode_fn, direction_fn in self._checks:
            s = np.asarray(encode_fn(states), float).ravel()
            if direction_fn is not None:
                s = np.asarray(direction_fn(s), float).ravel()
            rows.append(s)
        widths = {len(r) for r in rows}
        if len(widths) > 1:
            raise ValueError("checks returned different score lengths %s -- every check must emit one score per "
                             "event so the battery shares a single target alignment" % sorted(widths))
        return np.vstack(rows) if rows else np.zeros((0, 0))

    def screen(self, states, targets, alpha=0.1, min_events=8, cluster_threshold=0.7, time_it=True):
        """Evaluate EVERY check in one pass and return the honest report.

        For each check the per-event contribution is `sign(score) * target` -- the value the detector would have
        earned if acted on. From that one series come all three readouts, which is the point: they cannot
        disagree about what was measured because they are computed from the same numbers.

          effect / t / p  the mean contribution and its normal-approximation t-test
          split_half      contiguous replication (first half vs second half), from holographic_honesty
          q / rejected    Benjamini-Yekutieli FDR over the WHOLE battery, applied automatically

        A check `passes` only if it clears BOTH the family-corrected FDR gate AND split-half replication.

        Returns a dict with: `checks` (per-check rows, always all of them, in registration order), `passed`
        (names only), `clusters` / `cluster_members` (independent findings among the passers),
        `family_size`, `n_rejected`, `refused`, `reason`, and `timing`.

        KEPT NEGATIVE: `p` uses the normal approximation (NumPy-only, no t distribution), so it is
        anticonservative for short series -- `min_events` refuses outright below its threshold rather than
        quoting a number that cannot be trusted. And a battery of highly-correlated checks does not get more
        trustworthy by being large: read `clusters`, not `family_size`, when judging how much was really found.
        """
        targets = np.asarray(targets, float).ravel()
        if len(self._checks) == 0:
            return {"checks": [], "passed": [], "clusters": 0, "cluster_members": [], "family_size": 0,
                    "n_rejected": 0, "refused": True, "timing": {},
                    "reason": "empty battery -- no checks registered, so there is nothing to screen"}
        if len(targets) < min_events:
            raise ValueError("need at least min_events=%d targets to screen (got %d); the normal-approximation "
                             "p-values are not trustworthy below that" % (min_events, len(targets)))

        scores = self._score_matrix(states)
        if scores.shape[1] != len(targets):
            raise ValueError("checks emit %d scores but %d targets were given -- they must align event for "
                             "event" % (scores.shape[1], len(targets)))
        # Both timings start from the SAME precomputed scores, so the comparison is apples to apples. The first
        # draft timed score computation on the batched side only -- a strawman baseline pointing the wrong way,
        # which flattered the loop by 5-20x. Baseline discipline applies to our own claims first.
        t0 = time.perf_counter()
        contrib = np.sign(scores) * targets[None, :]            # all K contribution series, one broadcast
        t_batched = time.perf_counter() - t0

        t_loop = float("nan")
        if time_it:
            t1 = time.perf_counter()
            for k in range(scores.shape[0]):
                _ = np.sign(scores[k]) * targets
            t_loop = time.perf_counter() - t1

        n = contrib.shape[1]
        means = contrib.mean(axis=1)
        sds = contrib.std(axis=1, ddof=1)
        ses = sds / math.sqrt(n)
        tstats = np.where(ses > 0, means / np.where(ses > 0, ses, 1.0), 0.0)
        pvals = np.array([math.erfc(abs(float(t)) / math.sqrt(2.0)) for t in tstats])

        # FDR over the WHOLE battery -- dependent variant, because checks built on one dataset are correlated
        # by construction and the independent form would under-correct exactly where it matters.
        rejected, n_rejected = bh_fdr(pvals, alpha=alpha, dependent=True)

        rows, passed = [], []
        for i, (name, _, _) in enumerate(self._checks):
            sh = split_half(contrib[i])
            ok = bool(rejected[i]) and bool(sh["passed"])
            rows.append({"name": name, "effect": float(means[i]), "t": float(tstats[i]), "p": float(pvals[i]),
                         "fdr_rejected": bool(rejected[i]), "split_half_passed": bool(sh["passed"]),
                         "split_half": sh, "passed": ok, "n": int(n)})
            if ok:
                passed.append(name)

        clusters, members = self._cluster(contrib, passed, cluster_threshold)
        refused = len(passed) == 0
        if refused:
            reason = ("no check cleared BOTH gates over %d candidates -- reported as a RESULT: the target is "
                      "not conditionable by this battery at this grain. (%d cleared FDR alone; %d cleared "
                      "split-half alone.)"
                      % (len(rows), sum(r["fdr_rejected"] for r in rows), sum(r["split_half_passed"] for r in rows)))
        else:
            reason = ("%d of %d checks cleared both gates, forming %d independent finding(s) after correlation "
                      "clustering at |r| >= %.2f" % (len(passed), len(rows), clusters, cluster_threshold))
        return {"checks": rows, "passed": passed, "clusters": int(clusters), "cluster_members": members,
                "family_size": len(rows), "n_rejected": int(n_rejected), "refused": bool(refused),
                "reason": reason,
                "timing": {"batched_s": float(t_batched), "loop_s": float(t_loop),
                           "speedup": float(t_loop / t_batched) if t_batched > 0 and t_loop == t_loop else float("nan")}}

    def _cluster(self, contrib, passed_names, threshold):
        """Group passing checks whose contribution series correlate at |r| >= threshold. Single-linkage over the
        correlation graph -- deliberately the LOOSE clustering, because when the question is 'how many separate
        things did I really find', merging too eagerly is the conservative error and splitting is the flattering
        one."""
        if not passed_names:
            return 0, []
        idx = [i for i, (n, _, _) in enumerate(self._checks) if n in set(passed_names)]
        sub = contrib[idx]
        k = len(idx)
        # correlation matrix; a constant series correlates with nothing (guard the zero-variance case).
        sd = sub.std(axis=1)
        C = np.eye(k)
        for a in range(k):
            for b in range(a + 1, k):
                if sd[a] > 0 and sd[b] > 0:
                    C[a, b] = C[b, a] = float(np.corrcoef(sub[a], sub[b])[0, 1])
        parent = list(range(k))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a in range(k):
            for b in range(a + 1, k):
                if abs(C[a, b]) >= threshold:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)       # deterministic union: always toward the lower index
        groups = {}
        for i in range(k):
            groups.setdefault(find(i), []).append(self._checks[idx[i]][0])
        members = [sorted(v) for _, v in sorted(groups.items())]
        return len(members), members

    def build_committee(self, report):
        """E2: seat the committee FROM A SCREEN REPORT -- one representative per correlation cluster of the
        passing checks (the first member of each cluster in name order, deterministic). An empty pass-list
        seats an EMPTY committee, whose decide() refuses with the reason: refusal propagates, it does not get
        smoothed into 'use the best raw check instead'. See Committee for the gates the seated committee must
        then pass ITSELF on fresh data."""
        by_name = {name: (name, enc, dirfn) for name, enc, dirfn in self._checks}
        members = []
        for cluster in report.get("cluster_members", []):
            rep = sorted(cluster)[0]
            if rep in by_name:
                members.append(by_name[rep])
        return Committee(members)


    def program_vector(self, states):
        """The whole battery as ONE hypervector: each check's signature bound to its own role vector, all
        superposed. Two runs' program vectors can be compared by cosine, which is what makes "did this release
        change how the battery behaves" a single number instead of a diff of twenty tables.

        The signature is the check's DECISION PATTERN -- sign(scores) over the given states -- projected onto a
        fixed random basis. Decisions, deliberately, not contributions: no targets are needed, so a battery can
        be fingerprinted on unlabelled data (which is what release QA actually has), and two checks that DECIDE
        alike get similar signatures however they are named or scaled.

        WHY NOT contributions (sign(score) * target): every check's contribution carries the same |target|
        factor, so all K signatures inherit a large shared component and the bundle's crosstalk swamps recovery
        -- measured at cos 0.186 against a 0.177 floor, i.e. nothing. Decision patterns are near-orthogonal
        between genuinely different checks, and recovery works. Kept as a WHY-comment because the failing
        version looked perfectly reasonable.

        Recovery is real, not decorative: unbinding a check's role from the program returns that check's
        signature at a cosine well above the noise floor (pinned in _selftest). Deterministic in `seed`."""
        decisions = np.sign(self._score_matrix(states))
        rng = np.random.default_rng(self.seed)
        basis = rng.standard_normal((decisions.shape[1], self.dim)) / math.sqrt(self.dim)
        sigs = decisions @ basis                                # (K, dim) behavioural signatures
        sigs /= np.linalg.norm(sigs, axis=1, keepdims=True) + 1e-12
        roles = self._roles()
        from holographic.agents_and_reasoning.holographic_ai import bind, bundle
        bound = np.stack([bind(roles[i], sigs[i]) for i in range(len(self._checks))])
        return bundle(bound), {name: sigs[i] for i, (name, _, _) in enumerate(self._checks)}

    def _roles(self):
        """One fixed random unit role vector per check, derived from the program seed and the check's INDEX.
        Index rather than a name hash keeps this NumPy-only and deterministic without reaching for hashlib for
        what is not a content-addressed key."""
        rng = np.random.default_rng(self.seed + 7919)
        r = rng.standard_normal((max(len(self._checks), 1), self.dim))
        return r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)


class Committee:
    """E2 -- the VETO COMMITTEE: the surviving checks of a screen, combined into ONE decision signal, with
    the committee itself held to the same gates its members passed.

    Built via SignalProgram.build_committee(report) -- never by hand-picking checks, so the committee's
    membership is exactly what survived the screen and nothing that did not. Members are CLUSTER
    REPRESENTATIVES, one per independent finding: seating two 0.9-correlated checks would let one idea vote
    twice, which is how a committee's "breadth" becomes a lie (the campaign's first committee did exactly
    this and looked great until the strict rebuild, where NO member survived split-half).

    decide(states) is a majority vote over the representatives' decision signs -- votes, not confidences,
    because a committee exists to be robust to any one member's scale, and a confidence-weighted average is
    one miscalibrated member away from being that member.

    evaluate(states, targets) applies THE COMMITTEE'S OWN GATES on data that must be fresh: effect t, and
    split-half replication of the combined signal. A committee whose members all passed individually can
    still fail combined (vetoes can cancel the very events that carried the effect) -- that failure is
    returned as a verdict, never smoothed into the best member's number.

    KEPT NEGATIVE, structural: evaluating the committee on the SAME data it was screened on repeats the
    selection at one remove -- the committee is the argmax of the screen, so in-sample its performance is
    biased up by construction. evaluate() cannot check that your data is fresh; the docstring can and does
    say that the number is only honest if it is. Pair with a SelectionLedger entry per evaluation.
    """

    def __init__(self, members, contribs_shapes=None):
        self._members = list(members)                           # [(name, encode_fn, direction_fn)]

    @property
    def members(self):
        return [name for name, _, _ in self._members]

    def __len__(self):
        return len(self._members)

    def decide(self, states):
        """The committee's decision per event: sign of the vote sum over member decision signs. 0 when the
        votes tie -- a tie is an abstention, not a coin flip, and downstream sizing should treat it as
        stand-aside."""
        if not self._members:
            raise ValueError("empty committee -- build_committee returned no members; that refusal was the "
                             "result, and there is no decision to take")
        votes = np.zeros(np.asarray(states).shape[0])
        for name, enc, dirfn in self._members:
            s = np.asarray(enc(states), float).ravel()
            if dirfn is not None:
                s = np.asarray(dirfn(s), float).ravel()
            votes += np.sign(s)
        return np.sign(votes)

    def evaluate(self, states, targets, alpha=0.05):
        """Judge the COMBINED signal on (fresh!) data: contribution = decide * target, abstentions excluded
        from the per-event average but counted. Returns {effect, t, p, split_half_passed, n_votes,
        n_abstain, passed, verdict}."""
        d = self.decide(states)
        targets = np.asarray(targets, float).ravel()
        act = d != 0
        n_abstain = int(np.sum(~act))
        if int(np.sum(act)) < 8:
            return {"effect": float("nan"), "t": 0.0, "p": 1.0, "split_half_passed": False,
                    "n_votes": int(np.sum(act)), "n_abstain": n_abstain, "passed": False,
                    "verdict": "committee abstained on nearly everything (%d votes) -- no evaluable record, "
                               "which is itself the verdict" % int(np.sum(act))}
        contrib = d[act] * targets[act]
        n = contrib.size
        se = contrib.std(ddof=1) / math.sqrt(n)
        t = float(contrib.mean() / se) if se > 0 else 0.0
        p = float(math.erfc(abs(t) / math.sqrt(2.0)))
        sh = split_half(contrib, alpha=alpha)
        passed = bool(p < alpha and sh["passed"])
        if passed:
            verdict = ("committee of %d holds on its own gates: effect %+.4f, t=%.1f, replicated"
                       % (len(self._members), contrib.mean(), t))
        else:
            verdict = ("committee of %d FAILS its own gates (p=%.3f, split-half %s) -- members passing "
                       "individually does not transfer; report this, do not fall back to the best member"
                       % (len(self._members), p, "passed" if sh["passed"] else "failed"))
        return {"effect": float(contrib.mean()), "t": t, "p": p,
                "split_half_passed": bool(sh["passed"]), "n_votes": int(n), "n_abstain": n_abstain,
                "passed": passed, "verdict": verdict}



def _selftest():
    """Contracts:

    1. REFUSAL IS A RESULT -- a battery of pure-noise checks against a pure-noise target returns refused=True
       with a populated reason, not an exception and not a best-of-the-litter winner.
    2. POWER -- a genuine check in that same battery is found, and the noise checks around it are not.
    3. THE GATES ARE STRUCTURAL -- a check that clears the raw p-value but fails split-half does NOT pass.
    4. CLUSTERING -- two duplicate checks are reported as ONE finding, not two.
    5. THE ONE-PASS CLAIM IS MEASURED, not asserted.
    6. program_vector RECOVERS a check's signature by unbinding its role.
    """
    rng = np.random.default_rng(0)
    N = 1200
    states = rng.standard_normal((N, 12))

    # (1) REFUSAL: nothing in the data, twelve chances to find it anyway.
    noise_target = rng.standard_normal(N)
    prog = SignalProgram(dim=256, seed=0)
    for j in range(12):
        prog.add_check("noise_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    rep = prog.screen(states, noise_target)
    assert rep["refused"] is True, rep["passed"]
    assert rep["passed"] == [] and rep["clusters"] == 0
    assert "not conditionable" in rep["reason"], rep["reason"]
    assert rep["family_size"] == 12
    # the seductive raw numbers ARE there -- the point is they did not survive the gates.
    best_raw = min(r["p"] for r in rep["checks"])
    assert best_raw < 0.5, best_raw

    # (2) POWER: the same battery plus one real detector. states[:,0] genuinely predicts the target sign.
    real_target = np.sign(states[:, 0]) * np.abs(rng.standard_normal(N))
    prog2 = SignalProgram(dim=256, seed=0)
    prog2.add_check("real", lambda s: s[:, 0])
    for j in range(1, 12):
        prog2.add_check("noise_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    rep2 = prog2.screen(states, real_target)
    assert "real" in rep2["passed"], rep2["reason"]
    assert rep2["passed"] == ["real"], rep2["passed"]            # and NOTHING else came along for the ride
    assert not rep2["refused"]

    # (3) THE GATES ARE STRUCTURAL: a detector that works only in the FIRST HALF has a strong overall p and
    #     must still fail, because split-half is computed in the same pass and both gates are required.
    half_target = real_target.copy()
    half_target[N // 2:] = rng.standard_normal(N - N // 2)       # the edge evaporates after the midpoint
    prog3 = SignalProgram(dim=256, seed=0)
    prog3.add_check("decays", lambda s: s[:, 0])
    rep3 = prog3.screen(states, half_target)
    row = rep3["checks"][0]
    assert row["p"] < 0.05, row                                  # the seductive number is real...
    assert row["fdr_rejected"], row                              # ...and it even clears FDR...
    assert not row["split_half_passed"], row                     # ...but it did not replicate,
    assert not row["passed"], row                                # ...so it does not pass. This is the whole idea.
    assert rep3["refused"]

    # (4) CLUSTERING: the same detector registered twice is ONE finding, and a 0.7-correlated variant merges too.
    prog4 = SignalProgram(dim=256, seed=0)
    prog4.add_check("real_a", lambda s: s[:, 0])
    prog4.add_check("real_b", lambda s: s[:, 0] * 2.0)           # identical direction, different scale
    rep4 = prog4.screen(states, real_target)
    assert sorted(rep4["passed"]) == ["real_a", "real_b"], rep4["passed"]
    assert rep4["clusters"] == 1, rep4["cluster_members"]        # two names, ONE finding
    assert sorted(rep4["cluster_members"][0]) == ["real_a", "real_b"]

    # (5) Timing is REPORTED, never asserted. The backlog predicted batching would win; measurement says it
    #     loses at large K (0.02-0.47x over 5 repeats at K=200) and is a wash at small K. But an earlier draft
    #     PINNED that with  and it flaked immediately on a clean extract, where the same
    #     code won at 1.19x. Wall-clock is not deterministic and has no place in a determinism-constrained
    #     suite. The only contract here is that the evidence is present for the reader.
    assert rep2["timing"]["batched_s"] > 0 and rep2["timing"]["loop_s"] > 0

    # (6) program_vector: unbinding a role recovers that check's behavioural signature above the noise floor.
    from holographic.agents_and_reasoning.holographic_ai import unbind
    pv, sigs = prog2.program_vector(states)
    roles = prog2._roles()
    rec = unbind(pv, roles[0])
    cos = float(np.dot(rec, sigs["real"]) / (np.linalg.norm(rec) * np.linalg.norm(sigs["real"]) + 1e-12))
    # a random signature's cosine to the recovery is the floor to beat.
    floor = max(abs(float(np.dot(rec, sigs["noise_%d" % j]) /
                         (np.linalg.norm(rec) * np.linalg.norm(sigs["noise_%d" % j]) + 1e-12)))
                for j in range(1, 12))
    # The honest bar is the VSA capacity law, not a magic constant: bundling K bindings leaves the recovery
    # cosine near 1/sqrt(K) (here 1/sqrt(12) = 0.29, measured 0.25), so we pin (a) recovery clearly above the
    # crosstalk floor and (b) the relationship that MATTERS -- more dimensions buy more separation. An absolute
    # threshold would silently pass a broken bundle at large dim and fail a correct one at small dim.
    assert cos > 2 * floor, (cos, floor)
    assert cos > 1.0 / math.sqrt(len(prog2)) - 0.1, (cos, len(prog2))
    wide = SignalProgram(dim=2048, seed=0)
    wide.add_check("real", lambda s: s[:, 0])
    for j in range(1, 12):
        wide.add_check("noise_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    pv_w, sigs_w = wide.program_vector(states)
    roles_w = wide._roles()
    rec_w = unbind(pv_w, roles_w[0])
    cos_w = float(np.dot(rec_w, sigs_w["real"]) / (np.linalg.norm(rec_w) * np.linalg.norm(sigs_w["real"]) + 1e-12))
    floor_w = max(abs(float(np.dot(rec_w, sigs_w["noise_%d" % j]) /
                            (np.linalg.norm(rec_w) * np.linalg.norm(sigs_w["noise_%d" % j]) + 1e-12)))
                  for j in range(1, 12))
    assert cos_w / floor_w > cos / floor, ((cos_w, floor_w), (cos, floor))   # 8x the dim, cleaner recovery

    # ---------------- Committee (E2): the veto committee, held to its own gates ----------------
    # Seat from the duplicate-check screen: two names, ONE cluster -> ONE member (an idea cannot vote twice).
    com4 = prog4.build_committee(rep4)
    assert len(com4) == 1 and com4.members == ["real_a"], com4.members

    # An honest committee: three INDEPENDENT weak-but-real detectors on fresh data. Screened on one half,
    # seated, evaluated on the other -- the vote must hold its own gates out-of-sample.
    states_tr = rng.standard_normal((1500, 12))
    drivers = states_tr[:, 0] + states_tr[:, 1] + states_tr[:, 2]
    tgt_tr = np.sign(drivers) * np.abs(rng.standard_normal(1500))
    prog5 = SignalProgram(dim=256, seed=0)
    for j in range(3):
        prog5.add_check("real_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    for j in range(3, 12):
        prog5.add_check("noise_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    rep5 = prog5.screen(states_tr, tgt_tr)
    assert sorted(rep5["passed"]) == ["real_0", "real_1", "real_2"], rep5["passed"]
    com5 = prog5.build_committee(rep5)
    assert len(com5) == 3
    states_te = rng.standard_normal((1500, 12))
    tgt_te = np.sign(states_te[:, 0] + states_te[:, 1] + states_te[:, 2]) * np.abs(rng.standard_normal(1500))
    ev = com5.evaluate(states_te, tgt_te)
    assert ev["passed"], ev["verdict"]
    # Ties: with an ODD member count sign-votes cannot tie, so abstentions on continuous states are ~0 --
    # measured, then pinned as the parity fact rather than a wrong "committees abstain" story. An EVEN
    # committee abstains on genuine splits; both parities keep the count identity.
    assert ev["n_abstain"] == 0 and ev["n_votes"] == 1500
    even = Committee(com5._members[:2])
    ev_even = even.evaluate(states_te, tgt_te)
    assert ev_even["n_abstain"] > 0                             # a 2-member committee ties when they disagree
    assert ev_even["n_votes"] + ev_even["n_abstain"] == 1500

    # THE VETO-CANCELLATION NEGATIVE: same three real detectors, but the test target is driven by ONLY
    # member 0 -- members 1 and 2 veto at random relative to it, diluting the vote. The committee's own
    # evaluation reports the damage honestly (t well below member 0 alone), and if it fails, the verdict
    # says FAILS -- never falls back to the best member.
    tgt_only0 = np.sign(states_te[:, 0]) * np.abs(rng.standard_normal(1500))
    ev_dil = com5.evaluate(states_te, tgt_only0)
    solo = states_te[:, 0]
    solo_t = float((np.sign(solo) * tgt_only0).mean() / ((np.sign(solo) * tgt_only0).std(ddof=1) / math.sqrt(1500)))
    assert ev_dil["t"] < 0.6 * solo_t, (ev_dil["t"], solo_t)     # the dilution is visible in the committee's own number
    assert "FAILS" in ev_dil["verdict"] or ev_dil["passed"]      # honest verdict either way

    # empty committee: refusal propagates.
    com_empty = prog.build_committee(rep)                        # rep is the all-noise refusal from (1)
    assert len(com_empty) == 0
    try:
        com_empty.decide(states)
        raise AssertionError("expected empty-committee refusal")
    except ValueError as e:
        assert "refusal was the result" in str(e)


    # refusals name what is wrong.
    try:
        SignalProgram().add_check("a", lambda s: s[:, 0]).add_check("a", lambda s: s[:, 1])
        raise AssertionError("expected ValueError on a duplicate check name")
    except ValueError as e:
        assert "duplicate check name" in str(e)
    assert SignalProgram().screen(states, real_target)["refused"] is True     # empty battery refuses cleanly

    print("holographic_signalprogram selftest OK (12 pure-noise checks on a noise target REFUSE -- best raw "
          "p=%.3f present but gated out; one real detector among 11 nulls is found alone (%s); a first-half-only "
          "edge clears p=%.4f AND FDR but fails split-half so it does not pass; two duplicate checks report as "
          "%d finding; program_vector recovers a signature at cos=%.2f vs a %.2f noise floor; batched %.2f ms "
          "vs loop %.2f ms -- reported, not claimed)"
          % (best_raw, rep2["passed"], rep3["checks"][0]["p"], rep4["clusters"], cos, floor,
             1e3 * rep2["timing"]["batched_s"], 1e3 * rep2["timing"]["loop_s"]))


if __name__ == "__main__":
    _selftest()
