"""Part 20b of UnifiedMind's faculty surface -- the TRANSFER doors (contribute, commons_pool, memory_export, memory_import), split out in sweep 114 when the parent part crossed the
2,000-line budget test_unified_split pins (the point of the split was file size; a
part that grows past the cap gets split again, never a raised cap).

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which is still the only import path anyone
uses. Carries no `__init__`; assumes the state UnifiedMind.__init__ sets up.
"""
import numpy as np

from holographic.unified import check_part


class _UnifiedPart20B:

    def contribute(self, dest, author=None):
        """OPT-IN contribution to the openzoo COMMONS (sweep 100): screen this mind's
        SHARED taught rows through a conservative privacy gate and export the survivors
        as a commons bundle, provenance 'commons:<author-or-anon>'. THE RULES, each a
        refusal: (1) session-salted rows never leave -- session isolation IS user
        privacy by construction, only session=='shared' rows are candidates; (2)
        path-shaped text (/, \\, ~/) is rejected -- no file directories travel; (3)
        email shapes, long digit runs (phone/account), and long hex/base64 runs
        (keys/tokens) are rejected; (4) model-cached rows are rejected -- the commons
        takes established knowledge, not unverified cache. Returns the REVIEW SHEET
        {kept, rejected:[(question, reason)]} -- consent is informed or it is not
        consent. KEPT NEG, loud: a lexical screen is a FLOOR, not a proof of
        anonymity; the review sheet is the real gate and the caller reads it before
        shipping the bundle. Opt-out: never call this, or set mind._commons_optout=True
        and even an accidental call refuses.

        CLM backlog wave 2 (the owner's answer to open question 3: YES, learned prototypes travel):
        (5) a row this memory VETOED (answer_feedback ok=False -> _vetoed_qs) or whose payload it
        marked bad (_payload_bad) never leaves -- it used to be exported as a live fact;
        (6) PROVISIONAL model provenance is refused too: 'model:<by>' (a meaning_resolve'd model
        reply) passed because only the exact string 'model-cached' was rejected;
        (7) HONEST COUNTS: a row the learning guard refuses inside the bundle's own teach() is
        moved to `rejected` with the guard's reason -- it used to be counted as kept;
        (8) every door's learned ProtoStore that is marked shareable AND eligible travels with the
        bundle (mind._shareable_stores(): the router, the tool door, and any door that opted in
        with protostore_share(door, True)). ELIGIBLE = every question it learned from passed the
        learning guard and none was session-salted ([s:<name>]) -- a prototype is a SUM of real
        questions and cannot forget one term, so one refused question bars the whole store.
        Withheld stores are listed with the reason. The bundle carries them in its
        lecore.learning.protostores section; memory_import / commons_pool merge them by the
        documented rule (_protostore_merge: verdict-count-weighted, conflicts flagged, local rows
        never silently overwritten)."""
        import os, re
        if getattr(self, "_commons_optout", False):
            return {"refused": "this mind is opted out of the commons"}
        lad = self.zoo["ladder"]
        PATH = re.compile(r"(~/|[A-Za-z]:\\|/[\w.-]+/)")
        MAIL = re.compile(r"\S+@\S+\.\S+")
        DIGITS = re.compile(r"[\d][\d\s().-]{6,}[\d]")
        SECRET = re.compile(r"[A-Fa-f0-9]{24,}|[A-Za-z0-9+/=]{32,}")
        kept, rejected, seen = [], [], set()
        # (5) what this memory itself refused: vetoed questions (normalised as answer_feedback stores them) and the
        # questions whose reflex payload was marked bad
        vetoed = set(getattr(lad, "_vetoed_qs", set()) or set())
        bad_qs = {" ".join(str(getattr(lad, "_payload_qs", {}).get(pk, "")).lower().split())
                  for pk in (getattr(lad, "_payload_bad", set()) or set())} - {""}
        for row in reversed(getattr(lad, "taught_log", []) or []):
            q, a = str(row[0]), str(row[1])
            sess = str(row[2]) if len(row) > 2 else "shared"
            prov = str(row[3]) if len(row) > 3 else "taught"
            if q in seen:
                continue
            seen.add(q)
            if sess != "shared":
                rejected.append((q[:60], "session-salted: user-private by construction"))
                continue
            if prov == "model-cached" or prov.startswith("model:"):
                rejected.append((q[:60], "%s: unverified model provenance" % prov))
                continue
            nq = " ".join(q.lower().split())
            if nq in vetoed:
                rejected.append((q[:60], "vetoed here: this memory refused the answer"))
                continue
            if nq in bad_qs:
                rejected.append((q[:60], "payload marked bad here"))
                continue
            if a == "carried tombstone":
                continue
            blob = q + " " + a
            if PATH.search(blob):
                rejected.append((q[:60], "path-shaped text"))
                continue
            if MAIL.search(blob):
                rejected.append((q[:60], "email shape"))
                continue
            if DIGITS.search(blob):
                rejected.append((q[:60], "long digit run (phone/account shape)"))
                continue
            if SECRET.search(blob):
                rejected.append((q[:60], "hex/base64 run (key/token shape)"))
                continue
            kept.append((q, a, prov))
        who = str(author) if author else "anon"
        dst = self.__class__(dim=self.dim, seed=0)
        n_kept = 0
        for q, a, prov in reversed(kept):
            r_ = dst.teach(q, a)
            if not (isinstance(r_, dict) and r_.get("taught")):
                # (7) the guard refused it inside the bundle: it did NOT travel -- say so, never count it as kept
                rejected.append((q[:60], "refused by the learning guard in the bundle: %s"
                                 % (r_.get("reason") if isinstance(r_, dict) else "not taught")))
                continue
            n_kept += 1
            _bl = getattr(dst.zoo["ladder"], "taught_log", [])
            if _bl and str(_bl[-1][0]) == q and len(_bl[-1]) > 3:
                keepprov = prov if prov.startswith(("wisdom:", "commons:")) else "commons:%s" % who
                _bl[-1] = [_bl[-1][0], _bl[-1][1], _bl[-1][2], keepprov]
        # (8) the learned prototypes that may travel, and the ones that may not (with why)
        stores, withheld = {}, []
        if hasattr(self, "_shareable_stores"):
            elig = self._shareable_stores()
            doors = set(self.__dict__.get("_protostores", {})) | set(self.__dict__.get("_protostore_share", {}))
            for door in sorted(doors):
                if door in elig:
                    continue
                flags = self.__dict__.get("_protostore_share", {}).get(door) or {}
                why = ("not marked shareable" if not flags.get("shareable") else
                       "a question it learned from was refused by the learning guard" if not flags.get("guard_ok") else
                       "it learned from session-salted (user-private) questions" if flags.get("salted") else
                       "empty")
                withheld.append({"door": door, "why": why})
            for door, st in elig.items():
                from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
                meta_, _ = st.state()
                cp = ProtoStore.from_state(meta_, {"A": np.asarray(st.A, np.float64)})
                cp.A = np.asarray(st.A, np.float64).copy()          # float64: the merge on arrival is exact
                cp.confusion = {}                                    # the miner's pairs stay home (not a prototype)
                if door == "route" and getattr(self, "_router", None) and self._router() is not None:
                    cp.__dict__["_commons_extra"] = {"features": self._router().features.digest}
                stores[door] = cp
            dst.__dict__["_commons_stores"] = stores
        # lever 3 (sweep 101): a contribution bundle is pure-taught BY CONSTRUCTION
        # (built row-by-row through teach), so the regen guard always passes and the
        # bundle ships as text -- the middle-out curve for the commons itself.
        dst.learning_save(str(dest), audit="regen")
        return {"kept": n_kept, "rejected": rejected, "dest": str(dest),
                "author": who,
                "stores": {d: {"labels": len(st), "verdicts": int(sum(st.count))} for d, st in stores.items()},
                "stores_withheld": withheld,
                "advice": "read the rejected list AND spot-check the kept rows before "
                          "shipping -- the lexical screen is a floor, not a proof"}

    def commons_pool(self, bundles, root):
        """POOL many contribution bundles into ONE commons partition (sweep 100): a
        fresh mind imports each bundle through the provenance-carrying cp69 pipe
        (conflicts FLAGGED, never silently resolved -- disagreement between users is
        signal), then saves to root. Any contributing mind imports the commons back
        with memory_import(root) -- the give-and-take Moose described: all who
        contribute may draw. Returns per-bundle counts and the flagged conflicts."""
        pool = self.__class__(dim=self.dim, seed=0)
        report = []
        for b in bundles:
            r = pool.memory_import(str(b), on_conflict="flag")
            report.append({"bundle": str(b), "imported": r.get("imported"),
                           "conflicts": r.get("conflicts"), "stores": r.get("stores", [])})
        # Q3: the learned prototypes merged from every bundle ride in the pooled partition's carrier section, so a
        # contributor importing the commons back receives them (merged into its own stores by the same rule)
        pooled = {}
        if hasattr(pool, "_shareable_stores"):
            # the pool exists only to be shared: every door a bundle brought is shareable here (each arrived from an
            # ELIGIBLE store -- guard-clean, never salted -- and the pool itself learns from no question at all)
            for rep_ in report:
                for st_ in rep_["stores"]:
                    if not st_.get("refused"):
                        pool.protostore_share(st_["door"], True)
            pooled = pool._shareable_stores()
            for door, st in pooled.items():
                if door == "route" and pool._router() is not None:
                    st.__dict__["_commons_extra"] = {"features": pool._router().features.digest}
            pool.__dict__["_commons_stores"] = pooled
        # lever 3 (sweep 101): the pooled commons is likewise pure-taught.
        sv = pool.learning_save(str(root), audit="regen")
        return {"bundles": report, "root": str(root),
                "rows": len(getattr(pool.zoo["ladder"], "taught_log", []) or []),
                "stores": {d: {"labels": len(st), "verdicts": int(sum(st.count))} for d, st in pooled.items()},
                "saved": sv.get("saved")}

    def memory_export(self, dest, query=None, sessions=None, provenance=None,
                      include_api_specs=True):
        """SELECTIVE EXPORT (cp69): write a PORTABLE, SELF-CONTAINED memory bundle
        holding only what the filters admit -- substring/topic `query`, a `sessions`
        list, a `provenance` set ("taught","validated","evidenced"). What travels:
        facts with their provenance, promoted conjectures AT their earned rung, api
        spec records (so learned tools work on arrival -- functionality transfers,
        not just text), and this memory's vetoes (tombstones ride along; a shared
        bundle does not resurrect what its maker killed). VERIFIED before blessing:
        a fresh boot of the bundle must answer every exported question. Returns the
        manifest {exported, by_provenance, vetoes, verified}."""
        import os as _os
        lad = self.zoo["ladder"]
        rows, seen = [], set()
        want_prov = set(provenance) if provenance else None
        for t in reversed(getattr(lad, "taught_log", [])):
            # LATEST STATE WINS (cp69): a promoted conjecture's newest row carries
            # its earned rung; iterating forward exported the stale first row and
            # silently demoted validated knowledge back to a bare conjecture.
            if len(t) < 4:
                continue
            q, a, sess, prov = str(t[0]), str(t[1]), str(t[2]), str(t[3])
            if q in seen or prov == "model-cached":
                continue
            if q in (getattr(lad, "_vetoed_qs", set()) or set()):
                continue                    # a vetoed question travels ONLY as a
                                            # tombstone, never as a live fact
            if q.startswith("api spec record:") and not include_api_specs:
                continue
            if want_prov and prov not in want_prov and \
                    not q.startswith("api spec record:"):
                continue
            if sessions and sess not in sessions:
                continue
            if query and str(query).lower() not in (q + " " + a).lower():
                continue
            seen.add(q)
            rows.append((q, a, prov))
        dst = type(self)()
        dst.zoo_attach(lambda p: "")
        for q, a, prov in rows:
            dst.teach(q, a)
            # symmetric with the sweep-99 import fix: teach() writes provenance
            # 'taught'; a bundle that promises 'facts travel with their provenance'
            # must actually carry it, or authorship (wisdom:<model>) dies in transit.
            _bl = getattr(dst.zoo["ladder"], "taught_log", [])
            if prov and prov != "taught" and _bl and str(_bl[-1][0]) == q and len(_bl[-1]) > 3:
                _bl[-1] = [_bl[-1][0], _bl[-1][1], _bl[-1][2], prov]
            if prov in ("validated", "evidenced"):
                dst.conjecture_record(q, a)
                dst.conjecture_promote(q, prov, "carried by memory_export")
        vet = sorted(getattr(lad, "_vetoed_qs", set()) or [])
        for vq in vet:
            dst.teach(vq, "carried tombstone")
            dst.answer_feedback(vq, ok=False)
        _os.makedirs(dest, exist_ok=True)
        # lever 3 (sweep 101): export bundles are pure-taught scratch minds.
        dst.learning_save(dest, audit="regen")
        chk = type(self)()
        chk.zoo_attach(lambda p: "")
        chk.learning_load(dest)
        def _row_ok(q_, a_, p_):
            got = str(chk.ask(q_).get("answer") or "")
            if p_ in ("validated", "evidenced", "conjecture"):
                return bool(got.strip())        # promote rebuilds canonical
                                                # serve-text; require presence
            return got == a_                    # plain taught: exact
        # NAME THE MISSES, DO NOT JUST COUNT THEM. This reported misses=1 out of
        # 497 and nothing else, so "verified: False" told the caller their bundle
        # was imperfect and gave them no way to act -- I had to reload the bundle
        # and diff 497 rows by hand to learn WHICH question did not survive.
        # A verification that cannot name what failed is a smoke alarm with no
        # location: it is right, and useless.
        _missed = [q for q, a, _p in rows if not _row_ok(q, a, _p)]
        misses = len(_missed)
        byp = {}
        for _q, _a, prov in rows:
            byp[prov] = byp.get(prov, 0) + 1
        return {"exported": len(rows), "by_provenance": byp,
                "vetoes": len(vet), "dest": dest,
                "verified": misses == 0, "misses": misses,
                "missed": _missed[:20]}

    def memory_import(self, src, on_conflict="flag"):
        """IMPORT / MERGE (cp69): bring a shared bundle INTO this memory -- the
        transfer door. Facts arrive with provenance intact; validated/evidenced
        conjectures keep their earned rung; api spec records make the sender's
        learned tools CALLABLE here without relearning; the bundle's tombstones are
        honored (their vetoes import as vetoes). CONFLICTS -- a question this memory
        already answers DIFFERENTLY -- are never silently overwritten: default
        on_conflict="flag" keeps the local answer and reports each collision with
        the drift sentinel's verdict; "theirs" adopts the incoming answer as a
        deliberate re-teach. Returns {imported, conflicts, vetoes, skipped}."""
        donor = type(self)()
        donor.zoo_attach(lambda p: "")
        donor.learning_load(src)
        dlad = donor.zoo["ladder"]
        lad = self.zoo["ladder"]
        mine = {}
        for t in getattr(lad, "taught_log", []):
            if len(t) > 3 and t[3] != "model-cached":
                mine.setdefault(str(t[0]), str(t[1]))
        imported, skipped, conflicts = 0, 0, []
        _dseen = set()
        _drows = []
        for t in reversed(getattr(dlad, "taught_log", [])):
            if len(t) < 4 or t[3] == "model-cached" or str(t[0]) in _dseen:
                continue
            _dseen.add(str(t[0]))
            _drows.append(t)
        for t in reversed(_drows):
            q, a, prov = str(t[0]), str(t[1]), str(t[3])
            if a == "carried tombstone":
                continue
            # AN EMPTY LOCAL ANSWER IS NOT A CONFLICTING ANSWER. `q in mine` was
            # true for questions this memory holds a BLANK for -- an abstention
            # that got written to the taught store -- so incoming knowledge was
            # flagged as a conflict and, under the default on_conflict="flag",
            # SILENTLY NOT IMPORTED.
            # MEASURED on a real bundle: 8 conflicts reported, FOUR of them
            # against questions that answered T4 with answer='' here. Half the
            # shared knowledge was withheld on the grounds of disagreeing with
            # nothing -- which is precisely the failure mode that makes sharing
            # research between memories useless.
            # Treat blank-here as absent: accept theirs, and count it as an
            # import rather than a conflict.
            _mine = str(mine.get(q, "") or "")
            if q in mine and _mine.strip():
                if mine[q] == a:
                    skipped += 1
                    continue
                verdict = ""
                try:
                    verdict = self.teach_check(q, a).get("verdict", "")
                except Exception:
                    pass
                conflicts.append({"q": q[:80], "mine": mine[q][:80],
                                  "theirs": a[:80], "verdict": verdict})
                if on_conflict != "theirs":
                    continue
            self.teach(q, a)
            # cp69 promised 'facts arrive with provenance intact' but this teach()
            # flattened every row to 'taught' -- measured in sweep 99 when a bequeathed
            # wisdom row imported with its authorship stripped (wisdom() showed no
            # authors while ask() served the lesson). Re-stamp any NON-DEFAULT
            # provenance onto the row teach just appended; plain 'taught' rows are
            # untouched, byte-identical.
            _lg = getattr(self.zoo["ladder"], "taught_log", [])
            if (prov and prov != "taught" and _lg and str(_lg[-1][0]) == q
                    and len(_lg[-1]) > 3):
                _lg[-1] = [_lg[-1][0], _lg[-1][1], _lg[-1][2], prov]
            if prov in ("validated", "evidenced"):
                self.conjecture_record(q, a)
                self.conjecture_promote(q, prov, "imported from %s" % src)
            imported += 1
        vet = sorted(getattr(dlad, "_vetoed_qs", set()) or [])
        for vq in vet:
            if vq not in mine:
                self.teach(vq, "carried tombstone")
            self.answer_feedback(vq, ok=False)
        if hasattr(self, "_api_toolbox"):
            self._api_toolbox._rehydrated = False       # relearn arrivals lazily
        # Q3: learned prototypes the bundle carries (a contribute / commons_pool bundle) merge into this mind's own
        # stores by the documented rule -- verdict-count-weighted, conflicts FLAGGED and the local row kept, a store in
        # another feature space refused whole (holographic_unified_p33_router._protostore_merge)
        stores = []
        for door, st in sorted((donor.__dict__.get("_commons_stores") or {}).items()):
            if hasattr(self, "_protostore_merge"):
                stores.append(self._protostore_merge(door, st, source=str(src)))
        return {"imported": imported, "skipped_identical": skipped,
                "conflicts": conflicts, "vetoes": len(vet),
                "on_conflict": on_conflict, "stores": stores}



def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p20b_transfer", "_UnifiedPart20B")
    print("holographic_unified_p20b_transfer selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
