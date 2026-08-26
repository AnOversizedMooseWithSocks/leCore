"""Protocol-as-data auditing (backlog D1): the honesty discipline as a STRUCTURAL property of a program
vector, not a habit you hope you followed.

A 'protocol' here is a HoloMachine program (a hypervector) whose APPLY steps name analysis FACULTIES --
encode, a search/recall, a procedure-matched null, a multiplicity (FDR) control, an out-of-sample split,
a decision. Because the protocol is program-AS-DATA, its structure can be READ BACK from the vector
(unbind per position, clean the operand against the faculty codebook -- the VM's own decode) and checked
for anti-patterns. The canonical artifact-factory is a SEARCH step with no procedure-matched NULL; the
second is selecting and then scoring with no out-of-sample SPLIT between them. This turns 'did the
analysis include its honesty steps?' into a structural query you can run on the protocol vector, rather
than a discipline maintained by hand and discovered missing after a fake edge slips through.

WHY this is the sound version of D1 (and where the backlog over-reaches): the audit reads ONE protocol
vector at a time. The backlog imagined checking 'a whole space of protocols in one operation' -- but
reading a per-protocol property out of a SUPERPOSITION of protocols is the capacity cliff (the same flaw
as the superposed null engine), so that framing is dropped. Auditing one program vector, recovered by
unbind+cleanup, is genuinely holographic AND correct.

SCOPE / KEPT NEGATIVES:
  * It is a STRUCTURAL lint, not a data-flow analysis. The single-accumulator VM does not encode per-step
    data lineage, so 'scores on the exact rows it selected on' is approximated by the ORDER check (no
    SPLIT step between SEARCH and DECIDE), not by tracking data identity.
  * Like every read from a bundle, the per-position decode is subject to the capacity cliff: a protocol
    must be SHORT enough (well within the program vector's capacity) for the read to be reliable -- the
    procedure tax. The selftest verifies the structure round-trips exactly at protocol length; long
    protocols would need a larger dim.
  * An unknown faculty name carries NO honesty obligation (role NEUTRAL); the audit only fires on roles it
    recognises, so a missing taxonomy entry fails OPEN (no false alarm), not closed.
"""
import numpy as np

# Roles a protocol step can play. Neutral steps (encode, denoise, ...) carry no honesty obligation.
SEARCH = "search"        # recall / nearest / combination-search / resonate -- proposes candidates from data
NULL = "null"            # procedure-matched null / calibrated false-alarm / permutation-shuffle test
FDR = "fdr"              # multiplicity / look-elsewhere control across a family of candidates
SPLIT = "split"          # out-of-sample / held-out / both-halves split
DECIDE = "decide"        # the exact arbiter: materialize a scalar truth-claim
ENCODE = "encode"        # neutral: build a feature / representation
NEUTRAL = "neutral"      # any other declared step (no obligation)

# Default taxonomy: faculty NAME -> role. The names match real mind faculties (recall, RecallNull, bh_fdr,
# walk_forward_recall, ...) so a protocol that declares them is a real analysis program, not a toy. Extend
# freely; an unknown faculty is NEUTRAL (fails open).
ROLE_TAXONOMY = {
    "encode": ENCODE, "spectral_encode": ENCODE, "feature": ENCODE,
    "recall": SEARCH, "nearest": SEARCH, "combination_search": SEARCH, "resonate": SEARCH,
    "factor": SEARCH, "scan_search": SEARCH,
    "calibrated_null": NULL, "recall_null": NULL, "permutation_test": NULL, "shuffle_null": NULL,
    "calibration": NULL,
    "fdr": FDR, "bh_fdr": FDR, "look_elsewhere": FDR,
    "oos_split": SPLIT, "walk_forward": SPLIT, "holdout": SPLIT, "both_halves": SPLIT,
    "decide": DECIDE, "materialize": DECIDE, "score": DECIDE,
}


def register_protocol_vocabulary(machine, taxonomy=ROLE_TAXONOMY):
    """Ensure every faculty NAME in the taxonomy has an operand atom in the machine, so a protocol may
    DECLARE that step (APPLY <faculty>) even before any runtime handler exists -- the declared structure is
    what the audit reads. Idempotent: only adds names the machine does not already know."""
    for name in taxonomy:
        if name not in machine.fac_atoms:
            machine.fac_atoms[name] = machine._atom(f"fac:{name}")
            machine.faculty_names.append(name)
    return machine


def build_protocol(machine, steps, halt=True):
    """Assemble a protocol from an ordered list of faculty-step NAMES into ONE program vector. Each step
    becomes an APPLY <faculty>; a trailing HALT marks the end. Registers the taxonomy AND any step names not
    already known (so a neutral step like 'denoise' can be declared too). Returns (program_vec, n_steps)."""
    register_protocol_vocabulary(machine, ROLE_TAXONOMY)
    for s in steps:                          # register any step name the protocol declares (incl. neutral ones)
        if s not in machine.fac_atoms:
            machine.fac_atoms[s] = machine._atom(f"fac:{s}")
            machine.faculty_names.append(s)
    program = [("APPLY", s) for s in steps]
    if halt:
        program = program + [("HALT", "HALT")]
    return machine.assemble(program), len(program)


def protocol_role_sequence(machine, program_vec, n_steps, taxonomy=ROLE_TAXONOMY):
    """Read a protocol's step structure BACK FROM ITS VECTOR: decode each of the n_steps positions with the
    VM's own noisy unbind+cleanup, and map each APPLY's faculty to its role. Returns a list of
    (index, opcode, faculty, role) -- the recovered structure, holographically. Stops at a decoded HALT."""
    seq = []
    for i in range(n_steps):
        op, arg = machine.decode_instruction(program_vec, i)
        if op == "HALT":
            break
        role = taxonomy.get(arg, NEUTRAL) if op == "APPLY" else NEUTRAL
        seq.append((i, op, arg, role))
    return seq


# --- the anti-pattern rules. Each takes the recovered sequence + the set of roles present and returns a
#     (code, message) violation or None. Add rules freely; this is the honesty-lint's rulebook. ---

def _rule_search_without_null(seq, present):
    if SEARCH in present and NULL not in present:
        return ("search_without_null",
                "a SEARCH/recall step proposes candidates from the data but no procedure-matched NULL step "
                "confirms they beat noise -- the canonical artifact-factory")
    return None


def _rule_search_decide_without_fdr(seq, present):
    if SEARCH in present and DECIDE in present and FDR not in present:
        return ("search_decide_without_fdr",
                "a SEARCH proposes a FAMILY of candidates and a DECIDE scores them, but no FDR / "
                "look-elsewhere step controls the multiplicity -- scan enough and one clears by luck")
    return None


def _rule_select_then_score_without_split(seq, present):
    # ORDER check (the data-flow stand-in): is there a SPLIT step BETWEEN a SEARCH and a LATER DECIDE?
    search_i = next((i for (i, _o, _f, r) in seq if r == SEARCH), None)
    decide_i = next((i for (i, _o, _f, r) in seq if r == DECIDE), None)
    if search_i is not None and decide_i is not None and decide_i > search_i:
        split_between = any(r == SPLIT and search_i < i < decide_i for (i, _o, _f, r) in seq)
        if not split_between:
            return ("select_then_score_same_data",
                    "the protocol SELECTS (search) and then SCORES (decide) with no out-of-sample SPLIT "
                    "between them -- it judges a candidate on the same data it was chosen on")
    return None


DEFAULT_RULES = [_rule_search_without_null,
                 _rule_search_decide_without_fdr,
                 _rule_select_then_score_without_split]


def audit_protocol(machine, program_vec, n_steps, taxonomy=ROLE_TAXONOMY, rules=DEFAULT_RULES):
    """Audit a protocol VECTOR for honesty anti-patterns. Reads the step structure back from the vector
    (holographic decode), classifies each step's role, and evaluates the structural rules. Returns
    {sound, roles, sequence, violations}: sound is True iff no rule fires; violations is a list of
    (code, message); sequence is the recovered [(faculty, role), ...]."""
    seq = protocol_role_sequence(machine, program_vec, n_steps, taxonomy)
    present = {r for (_i, _o, _f, r) in seq}
    violations = [v for v in (rule(seq, present) for rule in rules) if v is not None]
    return {"sound": len(violations) == 0,
            "roles": sorted(present),
            "sequence": [(f, r) for (_i, _o, f, r) in seq],
            "violations": violations}


def _selftest():
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine
    M = HoloMachine(dim=4096, seed=1)

    # (1) the structure round-trips from the VECTOR: what we built is what the holographic read recovers.
    sound_steps = ["encode", "combination_search", "oos_split", "calibrated_null", "fdr", "decide"]
    pv, n = build_protocol(M, sound_steps)
    rec = [f for (f, r) in audit_protocol(M, pv, n)["sequence"]]
    assert rec == sound_steps, f"decode did not round-trip the protocol structure: {rec} != {sound_steps}"

    # (2) a complete, honest protocol passes (search has a null, a family has FDR, a split sits between
    #     select and decide).
    a_sound = audit_protocol(M, pv, n)
    assert a_sound["sound"], f"a complete protocol should be sound, got {a_sound['violations']}"

    # (3) the headline anti-pattern: a SEARCH with NO procedure-matched NULL is flagged.
    pv2, n2 = build_protocol(M, ["encode", "combination_search", "oos_split", "fdr", "decide"])
    a2 = audit_protocol(M, pv2, n2)
    assert not a2["sound"] and any(c == "search_without_null" for c, _m in a2["violations"]), \
        f"search-without-null not flagged: {a2['violations']}"

    # (4) selecting then scoring with NO split between them is flagged (the data-flow stand-in).
    pv3, n3 = build_protocol(M, ["encode", "combination_search", "calibrated_null", "fdr", "decide"])
    a3 = audit_protocol(M, pv3, n3)
    assert not a3["sound"] and any(c == "select_then_score_same_data" for c, _m in a3["violations"]), \
        f"select-then-score-same-data not flagged: {a3['violations']}"

    # (5) a searched-and-scored FAMILY with no FDR is flagged.
    pv4, n4 = build_protocol(M, ["encode", "recall", "oos_split", "calibrated_null", "decide"])
    a4 = audit_protocol(M, pv4, n4)
    assert not a4["sound"] and any(c == "search_decide_without_fdr" for c, _m in a4["violations"]), \
        f"search-decide-without-fdr not flagged: {a4['violations']}"

    # (6) targeted, not trigger-happy: a protocol with NO search step has no honesty obligation, so an
    #     honest non-search procedure (e.g. a restoration loop: datafit -> denoise) is NOT flagged.
    pv5, n5 = build_protocol(M, ["datafit", "denoise"])           # both NEUTRAL in the taxonomy
    a5 = audit_protocol(M, pv5, n5)
    assert a5["sound"], f"a no-search protocol should not be flagged, got {a5['violations']}"

    print("holographic_protocol selftest OK:")
    print(f"  structure round-trips from the vector: {rec}")
    print(f"  complete protocol -> sound; roles {a_sound['roles']}")
    print(f"  search-without-null   -> flagged ({a2['violations'][0][0]})")
    print(f"  select-then-score     -> flagged ({a3['violations'][0][0]})")
    print(f"  family-without-fdr    -> flagged ({a4['violations'][0][0]})")
    print(f"  no-search protocol    -> sound (not trigger-happy)")


if __name__ == "__main__":
    _selftest()
