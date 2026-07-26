"""holographic_selectionledger.py -- the SESSION-LEVEL selection ledger: every hypothesis you tried, kept on
the books, so the correction is applied over what was ACTUALLY TRIED rather than over what survived.

WHY THIS MODULE EXISTS
----------------------
Every honesty tool downstream of a p-value shares one blind spot: it can only correct over the tests it was
handed. SignalProgram corrects over one battery and says so loudly; bh_fdr corrects over one array. Neither
can see the battery you ran YESTERDAY, looked at, and quietly discarded -- and that invisible history is the
look-elsewhere effect in its most common working form. The campaign this comes from ran ~100 checks across
many sittings; any single sitting looked disciplined, and only a ledger of all of them would have told the
truth about the family size.

So this is deliberately boring machinery: an APPEND-ONLY ledger. record() every test at the moment it is run
-- including the embarrassing ones, especially the embarrassing ones -- and correct() computes q-values over
the WHOLE book (or a named family within it). The discipline is in the workflow, not the math: the math is
bh_fdr, unchanged, delegated; what this module adds is that nothing falls off the books.

WHAT IT REFUSES TO DO
  * It NEVER deletes. There is no remove(); a withdrawn test stays on the books flagged withdrawn and still
    counts toward the family size, because you still LOOKED at it. (Withdrawal exists for genuine mistakes --
    a mis-keyed p, a test run on the wrong data -- and takes a required reason string.)
  * It refuses a p-value outside [0, 1] and a duplicate (name, family) pair -- re-running the same test is
    recorded as a new SEQUENCE entry under the same name, never an overwrite, so "I ran it until it passed"
    is visible on the books as exactly that.

PERSISTENCE is stdlib-JSON with a hashlib chain: each entry carries the sha256 of (previous hash + canonical
entry), so a ledger file that has had an inconvenient row deleted no longer verifies. This is bookkeeping,
not cryptography -- it will not stop a determined forger, it stops the QUIET edit, which in practice is the
only kind that happens.

KEPT NEGATIVE, up front: a ledger only corrects what is written down. Tests run mentally ("I eyeballed the
chart and moved on") never reach it, and no software can close that gap. The honest claim is "corrected over
everything RECORDED", and the module's report says exactly that in those words.

NumPy + stdlib + hashlib only. Deterministic; entries are ordered by insertion, never by timestamp.
"""

import hashlib
import json

import numpy as np

from holographic.agents_and_reasoning.holographic_honesty import bh_fdr


def _canonical(entry):
    """A stable byte serialisation for hashing -- sorted keys, no whitespace variance, floats via repr so the
    chain is exact rather than approximately-equal."""
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=repr).encode()


class SelectionLedger:
    """The append-only book of every test run in a session (or a project -- lifetime is the caller's choice,
    and LONGER is more honest).

        led = SelectionLedger()
        led.record("momentum_1h", p=0.021, family="routing")
        led.record("momentum_4h", p=0.19,  family="routing")
        ...
        led.correct(alpha=0.1)                 # q-values over EVERYTHING on the books
        led.correct(alpha=0.1, family="routing")
        led.report()                           # the summary with the family sizes stated out loud
    """

    def __init__(self):
        self._entries = []          # list of dicts, append-only
        self._tip = ""              # hash chain tip

    # ------------------------------------------------------------------ recording
    def record(self, name, p, family="default", effect=None, note=""):
        """Put one test on the books AT THE MOMENT IT IS RUN. `p` in [0, 1]; `family` groups tests that answer
        the same question (correction can then run per family or over the whole book). Re-recording the same
        (name, family) appends a new entry with sequence n+1 -- visible, never an overwrite.

        Returns the entry dict (including its chain hash)."""
        p = float(p)
        if not (0.0 <= p <= 1.0) or p != p:
            raise ValueError("p must be a probability in [0, 1], got %r -- a mis-scaled statistic recorded as "
                             "a p-value would silently corrupt every correction downstream" % (p,))
        seq = 1 + sum(1 for e in self._entries if e["name"] == name and e["family"] == family)
        entry = {"index": len(self._entries), "name": str(name), "family": str(family), "p": p,
                 "sequence": seq, "effect": None if effect is None else float(effect),
                 "note": str(note), "withdrawn": False, "withdraw_reason": ""}
        self._tip = hashlib.sha256(self._tip.encode() + _canonical(entry)).hexdigest()
        entry["hash"] = self._tip
        self._entries.append(entry)
        return entry

    def withdraw(self, index, reason):
        """Flag entry `index` as withdrawn -- for genuine mistakes only (wrong data, mis-keyed value). The
        entry STAYS on the books and STILL COUNTS toward the family size: you looked at it, so it is part of
        the selection history whether or not the number was right. A withdrawal without a reason is refused."""
        if not reason or not str(reason).strip():
            raise ValueError("withdrawal requires a reason -- an unexplained withdrawal is indistinguishable "
                             "from deleting an inconvenient result")
        e = self._entries[int(index)]
        # append a withdrawal marker entry so the CHAIN records the act; the flag flip alone would be a silent
        # mutation of hashed history.
        marker = {"index": len(self._entries), "name": e["name"], "family": e["family"], "p": e["p"],
                  "sequence": e["sequence"], "effect": e["effect"],
                  "note": "WITHDRAWAL of index %d: %s" % (int(index), reason),
                  "withdrawn": True, "withdraw_reason": str(reason)}
        self._tip = hashlib.sha256(self._tip.encode() + _canonical(marker)).hexdigest()
        marker["hash"] = self._tip
        self._entries.append(marker)
        e["withdrawn"] = True
        e["withdraw_reason"] = str(reason)
        return marker

    def __len__(self):
        return len(self._entries)

    # ------------------------------------------------------------------ correction
    def correct(self, alpha=0.1, family=None, include_withdrawn_in_family_size=True):
        """q-values over the book. `family=None` corrects over EVERYTHING recorded -- the honest default for
        "what survives this whole session". A named family corrects within it.

        Withdrawn entries are EXCLUDED from the tested set (their p may be wrong) but, by default, still
        COUNTED in the reported family size, and the correction is run at the penalty of that larger family by
        padding with p=1.0 sentinels -- you spent those looks, so the multiplicity cost is real even if the
        numbers were not. Set include_withdrawn_in_family_size=False only when the withdrawn entries were
        never actually looked at (a batch mis-import).

        Returns {rows, family_size, n_tested, n_withdrawn, n_passed, alpha, scope} where each row carries
        {index, name, family, p, sequence, passed}. Uses the dependent (Benjamini-Yekutieli) form throughout:
        tests recorded in one session are correlated by construction."""
        live = [e for e in self._entries if not e["withdrawn"]
                and (family is None or e["family"] == family)]
        withdrawn = [e for e in self._entries if e["withdrawn"] and not e["note"].startswith("WITHDRAWAL")
                     and (family is None or e["family"] == family)]
        n_pad = len(withdrawn) if include_withdrawn_in_family_size else 0
        pvals = [e["p"] for e in live] + [1.0] * n_pad
        if not pvals:
            return {"rows": [], "family_size": 0, "n_tested": 0, "n_withdrawn": len(withdrawn),
                    "n_passed": 0, "alpha": float(alpha),
                    "scope": "empty ledger%s -- nothing to correct, which is a statement about the workflow, "
                             "not about the data" % ("" if family is None else " for family %r" % family)}
        rejected, n_rej = bh_fdr(pvals, alpha=alpha, dependent=True)
        rows = [{"index": e["index"], "name": e["name"], "family": e["family"], "p": e["p"],
                 "sequence": e["sequence"], "passed": bool(rejected[i])} for i, e in enumerate(live)]
        n_passed = sum(r["passed"] for r in rows)               # sentinels can never pass at p=1.0
        return {"rows": rows, "family_size": len(pvals), "n_tested": len(live),
                "n_withdrawn": len(withdrawn), "n_passed": int(n_passed), "alpha": float(alpha),
                "scope": ("corrected over everything RECORDED (%d tests%s) -- tests never written down are "
                          "not, and cannot be, covered"
                          % (len(pvals), "" if family is None else " in family %r" % family))}

    def report(self, alpha=0.1):
        """The session summary, families stated out loud: per family, how many tests, how many re-runs of the
        same name (the 'ran it until it passed' count), how many survive the family's own correction AND the
        whole-book correction -- a test can pass its family and die on the book, and that difference is the
        look-elsewhere effect made visible."""
        fams = {}
        for e in self._entries:
            if e["note"].startswith("WITHDRAWAL"):
                continue
            fams.setdefault(e["family"], []).append(e)
        whole = self.correct(alpha=alpha)
        passed_whole = {(r["name"], r["family"], r["sequence"]) for r in whole["rows"] if r["passed"]}
        out = []
        for fam in sorted(fams):
            entries = fams[fam]
            fam_result = self.correct(alpha=alpha, family=fam)
            reruns = sum(1 for e in entries if e["sequence"] > 1)
            fam_pass = {(r["name"], r["family"], r["sequence"]) for r in fam_result["rows"] if r["passed"]}
            out.append({"family": fam, "n": len(entries), "reruns": reruns,
                        "passed_in_family": len(fam_pass),
                        "passed_on_whole_book": len(fam_pass & passed_whole),
                        "died_on_the_book": sorted(n for n, f, s in (fam_pass - passed_whole))})
        return {"families": out, "total_recorded": len([e for e in self._entries
                                                        if not e["note"].startswith("WITHDRAWAL")]),
                "whole_book": whole}

    # ------------------------------------------------------------------ persistence + verification
    def to_json(self):
        """The whole book as a JSON string (entries + chain tip). Write it wherever; load_json verifies."""
        return json.dumps({"entries": self._entries, "tip": self._tip}, sort_keys=True)

    @classmethod
    def from_json(cls, s, verify=True):
        """Rebuild a ledger from to_json output. With verify=True (default) the hash chain is recomputed
        entry by entry; a book with a deleted or edited row raises instead of loading -- the quiet edit is the
        failure mode this exists to catch."""
        data = json.loads(s)
        led = cls()
        tip = ""
        for e in data["entries"]:
            body = {k: v for k, v in e.items() if k != "hash"}
            tip = hashlib.sha256(tip.encode() + _canonical(body)).hexdigest()
            if verify and tip != e.get("hash"):
                raise ValueError("ledger chain broken at index %d (%r) -- an entry was edited or removed after "
                                 "recording; refusing to load a book that no longer verifies"
                                 % (e.get("index", -1), e.get("name")))
        led._entries = data["entries"]
        led._tip = data["tip"]
        if verify and tip != led._tip:
            raise ValueError("ledger tip mismatch -- the book's tail was altered")
        return led


def _selftest():
    """Contracts:
    1. THE LOOK-ELSEWHERE EFFECT IS VISIBLE: a test that passes its own small family dies on the whole book
       once the book carries the session's other 60 looks.
    2. RE-RUNS ARE ON THE BOOKS as sequences, and 'ran it until it passed' is countable.
    3. WITHDRAWAL keeps the multiplicity cost: family size does not shrink, and the withdrawn p is excluded.
    4. THE CHAIN CATCHES THE QUIET EDIT: a serialised book with one row deleted refuses to load.
    5. Corrections agree with calling bh_fdr directly (delegation, not reimplementation).
    """
    rng = np.random.default_rng(0)

    # (1) one real finding (p=0.0004) inside a 4-test family looks great alone -- then the same session's 60
    #     null looks (uniform p) land on the book, and the family survivor must be judged against all 64.
    led = SelectionLedger()
    led.record("real_effect", 0.0004, family="focused")
    for i, p in enumerate((0.31, 0.62, 0.08)):
        led.record("focused_%d" % i, p, family="focused")
    fam_only = led.correct(alpha=0.05, family="focused")
    assert fam_only["n_passed"] == 1 and fam_only["family_size"] == 4
    for i in range(60):
        led.record("sweep_%d" % i, float(rng.random()), family="sweep")
    whole = led.correct(alpha=0.05)
    assert whole["family_size"] == 64
    # p=0.0004 still clears BY at m=64 (threshold ~ 0.05/(64*H64) ~ 1.6e-4... check honestly):
    # BY threshold for rank 1 at m=64: alpha*1/(64*H(64)); H(64)~4.74 -> 1.65e-4 < 4e-4 -> it should NOT pass.
    real_row = [r for r in whole["rows"] if r["name"] == "real_effect"][0]
    assert real_row["passed"] is False, "p=4e-4 must DIE on a 64-look book (BY rank-1 bar ~1.6e-4) -- the " \
                                        "look-elsewhere effect, made concrete"
    rep = led.report(alpha=0.05)
    fam = [f for f in rep["families"] if f["family"] == "focused"][0]
    assert fam["passed_in_family"] == 1 and fam["passed_on_whole_book"] == 0
    assert fam["died_on_the_book"] == ["real_effect"]

    # a STRONGER finding survives the same book -- the ledger penalises, it does not execute.
    led.record("strong", 1e-7, family="focused")
    whole2 = led.correct(alpha=0.05)
    assert [r for r in whole2["rows"] if r["name"] == "strong"][0]["passed"] is True

    # (2) re-runs: same (name, family) recorded three times -> sequences 1, 2, 3, and the report counts 2 reruns.
    led2 = SelectionLedger()
    for p in (0.30, 0.11, 0.04):
        led2.record("tuned_until_it_passed", p, family="tuning")
    assert [e["sequence"] for e in led2._entries] == [1, 2, 3]
    assert led2.report()["families"][0]["reruns"] == 2

    # (3) withdrawal: p excluded, family size KEPT. Book of 5; withdraw one; correct.
    led3 = SelectionLedger()
    for i in range(5):
        led3.record("t%d" % i, 0.5, family="f")
    led3.withdraw(2, reason="run on the wrong month's data")
    c = led3.correct(alpha=0.1, family="f")
    assert c["n_tested"] == 4 and c["family_size"] == 5 and c["n_withdrawn"] == 1
    try:
        led3.withdraw(0, reason="   ")
        raise AssertionError("expected refusal of a reasonless withdrawal")
    except ValueError as e:
        assert "requires a reason" in str(e)

    # (4) the quiet edit: serialise, delete the worst-looking row, refuse to load.
    s = led.to_json()
    data = json.loads(s)
    kept = [e for e in data["entries"] if e["name"] != "sweep_7"]
    tampered = json.dumps({"entries": kept, "tip": data["tip"]}, sort_keys=True)
    try:
        SelectionLedger.from_json(tampered)
        raise AssertionError("a book with a deleted row must not load")
    except ValueError as e:
        assert "chain broken" in str(e) or "tail was altered" in str(e)
    led_back = SelectionLedger.from_json(s)                     # the untampered book loads and matches
    assert led_back.correct(alpha=0.05)["family_size"] == whole2["family_size"]

    # (5) delegation: the whole-book verdicts equal bh_fdr called directly on the same p-vector.
    live_p = [e["p"] for e in led._entries if not e["withdrawn"]]
    rej, _ = bh_fdr(live_p, alpha=0.05, dependent=True)
    assert [r["passed"] for r in led.correct(alpha=0.05)["rows"]] == [bool(x) for x in rej]

    # p outside [0,1] refused by name
    try:
        SelectionLedger().record("bad", 2.5)
        raise AssertionError("expected refusal")
    except ValueError as e:
        assert "probability" in str(e)

    print("holographic_selectionledger selftest OK (a p=4e-4 family winner DIES on the 64-look book -- BY "
          "rank-1 bar ~1.6e-4 -- while p=1e-7 survives it; re-runs recorded as sequences (2 reruns counted); "
          "a withdrawn test keeps its multiplicity cost (family 5, tested 4); a book with one deleted row "
          "refuses to load on the hash chain; verdicts identical to direct bh_fdr)")


if __name__ == "__main__":
    _selftest()
