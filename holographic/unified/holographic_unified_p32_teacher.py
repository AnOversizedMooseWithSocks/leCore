"""UnifiedMind part 32 -- learning from the teacher: SystemOne's contrastive scorer, one negative store, a noisy
teacher measured by its own self-agreement, and the meaning rows on the shared contrastive rule (the CLM backlog,
phase C: E1.1 consumers 1 and 3, E2.1, E2.3, E1.3).

Where the work lives (this part is the faculty face; the mechanisms sit with the doors they change):
    SystemOne(scorer="contrastive")   holographic_systemone -- one ProtoStore per question, updated on EVERY outcome,
                                      its calibrator fed p_top (tau 0.02); systemone(..., scorer="contrastive")
    the learned meaning rows          holographic_meaning.RowPrototypes / MeaningIndex.enable_protos
    one negative store                part 28 _meaning_neg_add (exact veto + labelled negative on the row prototype),
                                      _meaning_ladder_negative (a vetoed ladder payload trains its meaning row), the
                                      reflex trace's outcome fields persisted (holographic_lever7.DisplacementTrace)
    the noisy teacher                 part 28 _meaning_reask / meaning_teacher_eps; MeaningIndex.decide corrects the
                                      gate; holographic_protostore.TeacherNoise does the arithmetic

Faculties here:
    meaning_protos(on)          switch the learned meaning rows on / off for this mind, or report them
    meaning_consolidate(...)    idle-time replay of every confirmed wording through the rule
    meaning_teacher_report()    how noisy the attached model end is, measured from its own re-asks
    negatives_report()          the one negative store, every form in one view
"""
import numpy as np


class _UnifiedPart32:

    def meaning_protos(self, on=None):
        """Switch the meaning rows' LEARNED prototypes on or off, or report them: returns {on, rows, updates, ...}.

        on=True: every confirmed wording (a `same` verdict, a correction by decision id, a merged `new`) moves the rows
        by the shared InfoNCE rule, and a reported wrong serve is a labelled negative on its row
        (holographic_meaning.RowPrototypes: the row's live centroid plus a learned delta in a hashed projection of
        the index's own sparse space). on=False DISCARDS the learned delta (the index reads plain centroids again).
        on=None only reports. The mind-wide default is UnifiedMind.MEANING_PROTOS (part 28), set by the measurement
        in docs/research/evidence/bench_meaning_protos.json (tools/bench_meaning.py protos)."""
        mi = self.meaning
        if on is True:
            mi.enable_protos()
        elif on is False and mi.protos is not None:
            mi.protos = None
            mi._pcache = None
            mi._arrays = None
        rp = mi.protos
        if rp is None:
            return {"on": False, "default": bool(self.MEANING_PROTOS)}
        live = [i for i, r in enumerate(rp.rows) if r in mi.rows and np.any(rp.D[i])]
        return {"on": True, "default": bool(self.MEANING_PROTOS), "rows_learned": len(live),
                "rows": len(mi.rows), "updates": rp.n_updates, "negatives": rp.n_negatives,
                "dim": rp.dim, "tau": rp.tau, "lr": rp.lr}

    def meaning_consolidate(self, epochs=1, seed=0):
        """Replay every confirmed wording through the meaning rows' learning rule, idle-time: returns {updates, ...}.

        Each stored wording is a verdict the teacher already gave (its row is its label), so a replay costs no model
        call. The online door sees each verdict once; the panel measured the rule at four passes over an index's own
        wordings, and tools/bench_meaning.py protos measures both (1pass / 4pass arms). Turns the learned rows on if
        they were off. Deterministic for a given seed."""
        mi = self.meaning
        out = mi.consolidate_protos(epochs=int(epochs), seed=int(seed))
        out["report"] = self.meaning_protos()
        return out

    def meaning_teacher_report(self):
        """Report how noisy the attached model end is, from its own re-asks: returns {eps_hat, agreement, ...}.

        About MEANING_REASK_RATE (3%) of typed escalations are asked again with the candidates permuted; eps_hat is
        the flip rate that explains the two answers' agreement (holographic_protostore.TeacherNoise, 0.0 until 20
        re-asks). With eps_hat > 0 the serve gate uses P(correct) = corrected P(agree), the learned rows train on the
        noise-corrected target, and a single verdict picking the 5th candidate or lower is held back. CAVEAT kept
        loud: the two asks are independent for a scripted stand-in; a real model may repeat its own mistake, so
        eps_hat can read LOW for it -- measure on the live model end before trusting the number."""
        mi = self.meaning
        t = mi.teacher
        return {"eps_hat": 0.0 if t is None else float(t.eps_hat()),
                "re_asks": 0 if t is None else len(t.pairs),
                "agreement": None if t is None else t.agreement(),
                "min_pairs": 20 if t is None else t.min_pairs, "reask_rate": float(self.MEANING_REASK_RATE),
                "held_back": int(mi.stats.get("held_back", 0)),
                "serve_bar": mi.serve_bar(),
                "ceiling_retired": {"P_FLOOR": mi.P_FLOOR, "ceiling": mi.ceiling()},
                "caveat": "re-ask independence holds for the scripted stand-in; for a real model it is a hypothesis"}

    def negatives_report(self):
        """Report the one negative store, every form in one view: returns {meaning, ladder, trace}.

        meaning: exact vetoes (this wording never served that row again) and labelled negatives that TRAINED a row's
        learned prototype; ladder: payloads marked bad by answer_feedback (and the questions they were taught for);
        trace: the reflex trace's outcome fields per tile (they travel with the partition, a split and a retile)."""
        mi = self.meaning
        lad = self.zoo["ladder"]
        bad = getattr(lad, "_payload_bad", set())
        tiles = []
        for t in self.experience.tiles:
            tiles.append({"fail_norm": float(np.linalg.norm(t._fail_field)),
                          "succ_norm": float(np.linalg.norm(t._succ_field))})
        return {"meaning": {"exact_vetoes": len(getattr(self, "_meaning_neg", set())),
                            "prototype_negatives": 0 if mi.protos is None else int(mi.protos.n_negatives),
                            "merged_wordings": int(mi.merged)},
                "ladder": {"bad_payloads": len(bad),
                           "bad_questions": sorted({getattr(lad, "_payload_qs", {}).get(p, "") for p in bad} - {""})},
                "trace": {"tiles": tiles,
                          "tiles_with_failures": sum(1 for x in tiles if x["fail_norm"] > 0)}}


def _selftest():
    """Part contract, one home: holographic.unified.check_part (every member reaches UnifiedMind, none shadowed)."""
    from holographic.unified import check_part
    n = check_part("holographic.unified.holographic_unified_p32_teacher", "_UnifiedPart32")
    return {"part": "holographic_unified_p32_teacher", "members": n}


if __name__ == "__main__":
    print(_selftest())
