"""UnifiedMind part 30 -- holographic candidates: compose a method call from its parts, check it for chimeras, read a
slot's direction from the question, encode trajectories (the CLM backlog, phase E: E3.1 / E3.2 / E5.1).

Everything here is a thin, stateful face over holographic.agents_and_reasoning.holographic_rolecall (the encoding
spec, the decoders, the measurements in its docstring and in docs/research/evidence/bench_rolecall.json):

    call_encode(verb, args)                     VERB(x)verb + sum_k SLOT_k(x)value_k at D=2048, unitary roles
    call_compose(state, verbs, slot_values)     per-role unbind + cleanup + CHIMERA CHECK -> a CANDIDATE with
                                                verdict, p_correct (this door's own calibrator), p_model, p_null
                                                and a decision id; it NEVER calls a tool
    call_from_question(question, verb, ...)     the question's soft state (direction read from its context words)
                                                -> call_compose
    call_factor_product(composite, ...)         a bound PRODUCT with permutation roles -> the resonator's
                                                noise-tolerant search with a procedure-matched p
    direction_learn / direction_read            per-slot-name context-role prototypes from from_question verdicts
    option_encode(name, examples)               OPTION(x)name + TEXT(x)examples
    trajectory_encode / trajectory_compare / trajectory_diff      sum_t rho^t(step_t), its comparison, its decode

THE OUTCOME PATH. call_compose records a DecisionRecord (via 'typed', question 'compose', state None -- a hypervector
is not text, so the reflex trace never learns a call from it) whose ledger hook feeds the 'compose' door's OWN
DoorCalibrator (p29): mind.decision_outcome(id, truth_call_string) is the only outcome path, and after min_count
reported outcomes p_correct stops being None.
"""
import hashlib

import numpy as np

import holographic.agents_and_reasoning.holographic_rolecall as RC


class _UnifiedPart30:

    # ------------------------------------------------------------------ shared state (one per mind)
    def _rolecall(self):
        """The mind's RoleCodec (D=2048, seed 0: the spec's operating point, independent of the mind's dim)."""
        c = self.__dict__.get("_rolecall_codec")
        if c is None:
            c = self.__dict__["_rolecall_codec"] = RC.RoleCodec(dim=RC.DIM, seed=RC.SEED)
        return c

    def _direction_reader(self):
        r = self.__dict__.get("_direction_reader_obj")
        if r is None:
            r = self.__dict__["_direction_reader_obj"] = RC.ContextRoles(dim=RC.DIM, seed=RC.SEED)
        return r

    # ------------------------------------------------------------------ E3.1 encoding
    def call_encode(self, verb, args):
        """Encode a method call as a role-filler SUM hypervector: returns {'vector', 'call', 'dim'}.

        vector = VERB(x)verb + sum_k SLOT_k(x)args[k] (HRR at D=2048, unitary role atoms named by slot), the form
        call_compose decodes (500/500 per-role decode measured, 0.1 ms). args = {slot: value}."""
        v = self._rolecall().encode_call(str(verb), dict(args or {}))
        return {"vector": v, "call": RC.call_string(str(verb), dict(args or {})), "dim": RC.DIM}

    def option_encode(self, name, examples=()):
        """Encode a typed option as OPTION(x)name + TEXT(x)examples: returns {'vector', 'name', 'dim'}. Decode the
        name back with call_compose-style cleanup (RoleCodec.decode_role(v, 'OPTION', names))."""
        return {"vector": self._rolecall().encode_option(str(name), list(examples or ())), "name": str(name),
                "dim": RC.DIM}

    # ------------------------------------------------------------------ E3.2 compose
    def call_compose(self, state, verbs, slot_values, record=True, z_margin=3.0):
        """Propose the call a hypervector state holds (never executes it): returns a candidate dict with verdict
        and p.

        verbs = [verb, ...] or {verb: [slot names]}; slot_values = {slot: [known values]}. Per-role unbind +
        cleanup, then the CHIMERA CHECK (recompose, explain away, re-read the residual): verdict 'clean' |
        'dominant' | 'ambiguous' (a rival as strong as the pick -- blended readouts built chimeras 74-76% of the
        time) | 'empty'. Returns {verb, args, call, verdict, served, p_correct, p_model, p_null, score, roles,
        id, executes: False}. p_correct comes from this door's own DoorCalibrator ('compose') and is None until
        min_count outcomes were reported via decision_outcome(id, truth_call); for an ambiguous/empty verdict it
        is capped by p_model (the isotonic map clamps below its lowest seen score -- measured: symmetric
        sentences read p 0.83 uncapped). A candidate is a proposal: this faculty calls nothing."""
        state = np.asarray(state, float)
        got = self._rolecall().compose(state, verbs, slot_values, z_margin=z_margin)
        return self._compose_record(got, record, state=state)

    def _compose_record(self, got, record, meta=None, state=None):
        cal = self.door_calibrator("compose")
        pc = cal.p_correct(got["score"])
        if got["verdict"] in ("ambiguous", "empty"):
            pc = min(pc if pc is not None else 1.0, got["p_model"])
        got["p_correct"] = pc
        got["served"] = got["verdict"] in ("clean", "dominant")
        got["executes"] = False
        got["id"] = None
        if record and got["call"] is not None:
            from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
            alts = [got["call"]]
            # the STATE is a hypervector, not text: the record's text state stays None (so reflex_learn, which only
            # learns text states, never writes a call into the experience trace), and the vector's sha256 goes into
            # the question -- two different states that decode to the same call are two decisions, two ids
            digest = hashlib.sha256(np.ascontiguousarray(np.asarray(state, np.float64)).tobytes()).hexdigest()[:12] \
                if state is not None else "nostate"
            # THE OUTCOME HOOK IS A SPEC, NOT A CLOSURE (learning-loop audit, 2026-09-26). It was a live closure over
            # (score, call), so an outcome reported after a restart fed nothing -- MEASURED: compose calibrator pairs
            # 0 of 1 after a restart (tools/audit_learning_loop.py). The restart-proof 'calibrate' kind (p29
            # _decision_hook) does the same thing from the record: door_calibrator('compose').observe(score, the
            # reported call == this call). The compose door still learns ONLY from its own outcomes (panel Q5).
            rec = DecisionRecord(None, "compose:" + digest, alts, got["call"], "typed", margin=got["score"], p=pc,
                                 meta=dict({"verdict": got["verdict"], "p_model": got["p_model"],
                                            "p_null": got["p_null"], "executes": False,
                                            "evidence": {"score": got["score"], "explained": got["explained"]},
                                            "hook": {"kind": "calibrate", "door": "compose",
                                                     "score": float(got["score"]), "answer": got["call"]}},
                                           **(meta or {})))
            got["id"] = self.decision_ledger().add(rec)
        return got

    def call_from_question(self, question, verb, mentions, slots=("from", "to"), number_slot="amount",
                           numbers=None, record=True):
        """Compose the call a question asks for from known pieces: returns a candidate (never executed) + reading.

        mentions = {words in the question: the value they name} (e.g. {'euros': 'EUR', 'yen': 'JPY'});
        slots = the choice slots those values fill. The DIRECTION comes from direction_read (context words, not
        the order the values appear: 49/51 held out vs positional 35/51 on the hand-labelled exchange set), the
        reader's belief over assignments becomes the soft state, and call_compose decodes it -- a split belief
        (a sentence with no direction) comes back 'ambiguous' instead of guessed. Numbers in the question fill
        `number_slot`. Returns call_compose's candidate plus 'reading' (the direction reader's output)."""
        codec = self._rolecall()
        st, rd = RC.question_state(codec, self._direction_reader(), str(question), str(verb), dict(mentions),
                                   list(slots), numbers=numbers, number_slot=number_slot)
        nums = RC.amounts_in(question) if numbers is None else list(numbers)
        vals = sorted(set(dict(mentions).values()))
        sv = {s: vals for s in slots}
        if number_slot and nums:
            sv[number_slot] = sorted(set(nums), key=nums.index)
        got = codec.compose(st, {str(verb): list(sv)}, sv)
        out = self._compose_record(got, record, meta={"question_len": len(str(question))}, state=st)
        out["reading"] = rd
        return out

    def call_factor_product(self, composite, verbs, values, slots, restarts=20, iters=200, accept_p=0.01,
                            m_null=100):
        """Factor a call held as ONE bound product with permutation roles: returns a candidate with p_value.

        composite = verb * rho^1(x_1) * rho^2(x_2) ... over bipolar atoms (RC.ProductCodec(verbs, values,
        slots).encode builds one). Uses the resonator's NOISE-TOLERANT exit (agreement + procedure-matched null):
        at 5% flipped components the exact-exit resonator solved 0/20 in 757 ms; see bench_rolecall.json
        'resonator' for the after numbers. Returns {verb, args, call, p_value, agreement, accepted, exit,
        executes: False}; the first call per codebook shape pays the null fit once."""
        pc = RC.ProductCodec(list(verbs), list(values), list(slots))
        return pc.factor(np.asarray(composite, float), restarts=restarts, iters=iters, accept_p=accept_p,
                         m_null=m_null)

    # ------------------------------------------------------------------ E3.1 direction (question side)
    def direction_learn(self, question, from_question):
        """Teach which words filled which slot in a question (a from_question verdict): returns {'learned', 'n'}.

        from_question = {slot: words} (e.g. {'from': 'dollars', 'to': 'pesos'}); pin an occurrence with
        (words, token_index); several mentions as a list. A sentence that names values but NO direction is taught
        as {'either': [words, words]} -- without it the reader was confidently directional on 41/41 symmetric
        sentences; with 27 taught it called 14/14 held-out ones directionless and 0/51 directional ones.
        A question carrying a secret is never learned (the reader's prototypes are sums of the question's context
        words and they are persisted with the partition) -- the learning guard's pattern layer decides."""
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        why = sensitive_reason(str(question))
        if why:
            return {"learned": False, "why": why}
        return self._direction_reader().learn(str(question), dict(from_question))

    def direction_read(self, question, mentions, slots=("from", "to")):
        """Read which mention fills which slot from the question's context words: returns the assignment + p.

        mentions = the words naming slot values, e.g. ['pesos', 'dollar']. Returns {assignment, p, p_none,
        direction, margin, ranked, via}; via 'positional' (first mention -> first slot) until a slot has been
        taught with direction_learn. Measured: 49/51 held out vs positional 35/51 (tools/bench_rolecall.py)."""
        return self._direction_reader().read(str(question), list(mentions), list(slots))

    # ------------------------------------------------------------------ E5.1 trajectories
    def _step_vec(self, step):
        codec = self._rolecall()
        if isinstance(step, dict):
            return RC.encode_step(codec, step["verb"], step.get("args") or {}, text=step.get("text"),
                                  text_weight=float(step.get("text_weight", 1.0)))
        if hasattr(step, "via") and hasattr(step, "answer"):          # a DecisionRecord
            from holographic.agents_and_reasoning.holographic_decisionrecord import RecordCodec
            rc = self.__dict__.get("_rolecall_recordcodec")
            if rc is None:
                rc = self.__dict__["_rolecall_recordcodec"] = RecordCodec(dim=RC.DIM, seed=RC.SEED)
            return rc.encode(step)
        return np.asarray(step, float)

    def trajectory_encode(self, steps):
        """Encode an ordered list of steps as ONE hypervector sum_t rho^t(step_t): returns {'vector', 'steps'}.

        A step is {'verb', 'args', 'text'?} (a role-filler record; 'text' = incidental text), a DecisionRecord
        (RecordCodec) or a vector. Order swaps AND one inserted step move it above the same-plan noise floor
        (bench_rolecall.json 'trajectory', which also keeps the case that does not: an insertion at the very
        end of a long trajectory)."""
        vs = [self._step_vec(s) for s in steps]
        return {"vector": RC.trajectory_encode(vs), "steps": len(vs)}

    def trajectory_compare(self, a, b, floor=None):
        """Compare two encoded trajectories by cosine: returns {'cosine', 'p_same'?, 'floor_p01'?}.

        a, b = vectors (or trajectory_encode results). With floor = cosines of SAME-plan re-encodings, p_same is
        the chance a same-plan pair looks this different (small -> an edit is present)."""
        va = a["vector"] if isinstance(a, dict) else a
        vb = b["vector"] if isinstance(b, dict) else b
        return RC.trajectory_compare(np.asarray(va, float), np.asarray(vb, float), floor=floor)

    def trajectory_diff(self, a, b, steps):
        """Decode two trajectories against known steps and name the edit: returns {'a', 'b', 'ops'}.

        steps = the candidate step dicts ({'verb', 'args'}); each position is unpermuted and cleaned up against
        them. ops lists swap / insert / delete / replace with positions (difflib), so a cosine drop becomes an
        explanation."""
        codec = self._rolecall()
        book = {}
        for s in steps:
            book[RC.call_string(s["verb"], s.get("args") or {})] = codec.encode_call(s["verb"], s.get("args") or {})
        va = a["vector"] if isinstance(a, dict) else a
        vb = b["vector"] if isinstance(b, dict) else b
        da = [n for n, _ in RC.trajectory_decode(np.asarray(va, float), book)]
        db = [n for n, _ in RC.trajectory_decode(np.asarray(vb, float), book)]
        return {"a": da, "b": db, "ops": RC.trajectory_diff(da, db)}


def _selftest():
    """Part contract, one home: holographic.unified.check_part (every member reaches UnifiedMind, none shadowed),
    plus the one promise this part makes that no other does: a composed call is a candidate, never an action."""
    from holographic.unified import check_part
    n = check_part("holographic.unified.holographic_unified_p30_compose", "_UnifiedPart30")
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=256, seed=0)
    v = m.call_encode("fx", {"from": "EUR", "to": "JPY"})["vector"]
    got = m.call_compose(v, {"fx": ["from", "to"]}, {"from": ["EUR", "JPY", "USD"], "to": ["EUR", "JPY", "USD"]})
    assert got["call"] == "fx(from=EUR, to=JPY)" and got["executes"] is False and got["id"], got
    assert m.decision_outcome(got["id"], got["call"])["was_correct"] is True
    return {"part": "holographic_unified_p30_compose", "members": n}


if __name__ == "__main__":
    print(_selftest())
