"""Schema-guided typed PLANS (and flat records) on the holographic substrate -- the structured branching
output the planning work was missing.

WHY THIS EXISTS
---------------
Encoding a contingency plan as a bind/bundle tree is already proven (holographic_typed.encode_tree /
StructureRecipe lay down exactly this role-filler graph, bit-exact). What was being hand-rolled at every
call site was the TYPE and the DECODE HELPER -- re-deriving "bind action to a role, bundle the branches,
unbind to read it back" by hand each time. This module ships that type and that helper.

The load-bearing idea -- and the reason a user-provided SHAPE is the right design rather than mere sugar --
is that the SHAPE IS THE DECODE KEY. Decoding a foreign vector with no shape is the resonator's job
(holographic_sbc.decompose_structure): a blind, combinatorial search bounded by crosstalk that caps out
(the typed module says so itself -- a StructureRecipe is a GENERATOR, not a parser). But when you KNOW the
shape, decoding is a deterministic walk: unbind exactly the roles the shape names, clean up against exactly
the codebooks it provides, recurse on exactly the field it marks recursive. No search. That is the
decode-vs-evaluate line again -- a known structure turns a decode-SEARCH into a sequence of clean unbinds,
which is why a 3-level plan round-trips every action and scope exactly where the blind parse would not.

WHAT IS HERE
------------
  * encode_record / decode_record : the GENERAL "bring your own shape" path -- a flat record of named
    symbolic fields (a scientific decision record, a classified state) <-> one vector. The shape is the
    per-field value codebooks. This is the structured-output mechanism for any domain, not just planning.
  * PlanNode + PlanShape           : the concrete contingency-plan type the planning seat asked for -- a
    primary action, a scope tag, an optional builder confidence, and a dict of named branches (each a
    PlanNode). Distinct from holographic_plan.Plan (the corridor planner's namedtuple); this is the
    decision tree, not a baked route.
  * encode_plan / decode_plan      : encode the tree to one vector; decode it back GIVEN ITS SHAPE.
  * descend                        : walk to the branch matching the current situation -- the
    generalisation of the machine's IFMATCH from one gated instruction to a named branch tree WITH
    ABSTENTION ("no contingency applies" -> the node's primary action).

PANEL GROUNDING (seats + real published methods, no fabricated opinions)
  * HRR role-filler recursion with per-level normalisation -- Plate (the capacity bound is the per-node
    fan-out cap, kept as a negative). bundle() already normalises each level.
  * The resonator is the UNKNOWN-structure fallback this AVOIDS -- Olshausen's resonator networks.
  * The branch walk is a behavior-tree selector ("best child whose condition fires, else fall through") --
    Togelius / game AI.
  * Abstention is referenced to a MEASURED noise floor and confidence is the MEASURED decode cosine, not a
    magic threshold -- Cranmer / calibrated detection (a simple cousin of RecallNull; upgrade to the full
    calibrated null if a controlled false-MATCH rate is wanted).
  * argmax ties broken deterministically (numpy argmax takes the first max) and the whole path is run-to-run
    identical -- Macklin / bit-exact tie-breaks.

Pure NumPy + holostuff spirit: deterministic, seed-regenerable, no new substrate (just bind/bundle/cleanup).
"""

import numpy as np

from holographic_ai import bind, bundle, unbind, derived_atom, cosine


# =================================================================================================
# The atom source: every role and value atom is a pure function of (seed, name), so a plan/record
# vector regenerates from the seed plus the names (the project's regenerate-from-seed principle).
# =================================================================================================
class ShapeVocab:
    """Seed-deterministic atoms for shaped structures. ROLES are unitary so a field unbinds cleanly;
    VALUES are ordinary atoms, recovered by cosine cleanup against the shape's codebook."""

    def __init__(self, dim, seed=0):
        self.dim = dim
        self.seed = seed

    def role(self, name):
        """An invertible field key -- bind a field under it, unbind to read it back."""
        return derived_atom(self.seed, "role:" + name, self.dim, unitary=True)

    def value(self, name):
        """A symbol atom -- a field value or a branch condition, snapped back by cosine cleanup."""
        return derived_atom(self.seed, "val:" + name, self.dim)

    def noise_floor(self, samples=512, q=0.99):
        """A MEASURED cosine noise floor: the q-quantile of |cos| between independent random atoms of this
        dimension. A cleanup or branch match must clear this to count as real -- the honest, substrate-derived
        abstention threshold (the Cranmer seat's measured floor, a simple cousin of RecallNull). Deterministic
        from the seed."""
        rng = np.random.default_rng(self.seed + 99991)
        a = rng.standard_normal((samples, self.dim))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        half = samples // 2
        cos = np.abs(a[:half] @ a[half:2 * half].T).ravel()
        return float(np.quantile(cos, q))


# =================================================================================================
# General flat record: a dict of named symbolic fields  <->  one vector.  The general "bring a shape"
# mechanism -- any structured output whose fields draw from known vocabularies is one bundle of
# role-bound value atoms, decoded by unbinding each role and cleaning against that field's codebook.
# =================================================================================================
def encode_record(fields, vocab):
    """Encode a flat record {field_name: value_name} as bundle_f bind(role(field), value(value_name))."""
    parts = [bind(vocab.role(f), vocab.value(v)) for f, v in fields.items()]
    return bundle(parts)


def decode_record(vec, schema, vocab):
    """Decode a flat record GIVEN ITS SHAPE. `schema` = {field_name: [possible_value_names]} -- the per-field
    codebooks the caller provides. For each field we unbind exactly its role and clean up against exactly its
    codebook: a deterministic walk, not the resonator's blind search. Returns {field_name: value_name}."""
    out = {}
    for field, value_names in schema.items():
        probe = unbind(vec, vocab.role(field))
        sims = [cosine(probe, vocab.value(v)) for v in value_names]
        out[field] = value_names[int(np.argmax(sims))]
    return out


def decode_record_confident(vec, schema, vocab):
    """As decode_record, but also return the per-field decode cosine -- how cleanly each field resolved
    against the bundle's capacity (the honest confidence). Returns (values, confidences)."""
    values, confidences = {}, {}
    for field, value_names in schema.items():
        probe = unbind(vec, vocab.role(field))
        sims = [cosine(probe, vocab.value(v)) for v in value_names]
        i = int(np.argmax(sims))
        values[field] = value_names[i]
        confidences[field] = round(float(sims[i]), 4)
    return values, confidences


# =================================================================================================
# The contingency-plan type and its shape.
# =================================================================================================
# Field roles for a plan node. Branch subtrees are bound under a per-name branch role so the decoder
# (which knows the names from the shape) can unbind exactly the branch it wants.
_R_ACTION = "plan.action"
_R_SCOPE = "plan.scope"


def _r_branch(name):
    return "plan.branch:" + name


class PlanNode:
    """A node of a contingency plan -- the structured branching output the planner was missing. Carries a
    primary ACTION, a SCOPE tag, an optional builder CONFIDENCE (metadata; see note), and a dict of named
    contingency BRANCHES, each itself a PlanNode. A plan is a tree of these.

    NOTE on confidence: the builder's confidence rides on the OBJECT, not inside the encoded vector --
    forcing a scalar into the bundle would decode lossily and betray the honest-number rule. After a decode
    the returned node's `confidence` is instead the MEASURED decode cosine of its action (how cleanly it
    resolved against the bundle's capacity), which is the operational, honest confidence.

    Distinct from holographic_plan.Plan (the corridor planner's baked-route namedtuple); this is the
    decision tree."""

    def __init__(self, action, scope="global", confidence=1.0, branches=None):
        self.action = action
        self.scope = scope
        self.confidence = confidence
        self.branches = dict(branches) if branches else {}

    def __repr__(self):
        keys = "{" + ", ".join(self.branches) + "}" if self.branches else "{}"
        return f"PlanNode(action={self.action!r}, scope={self.scope!r}, confidence={self.confidence}, branches={keys})"

    def __eq__(self, other):
        return (isinstance(other, PlanNode)
                and self.action == other.action and self.scope == other.scope
                and set(self.branches) == set(other.branches)
                and all(self.branches[k] == other.branches[k] for k in self.branches))


class PlanShape:
    """The SHAPE a caller hands the decoder so a plan vector can be read back -- the decode KEY. Holds the
    action and scope value codebooks (the possible names) and the nested branch-name skeleton (which branches
    exist at each level). With it, decoding is a deterministic unbind-and-clean walk instead of the
    resonator's blind, capacity-bounded search. `branches` maps a branch name to its OWN PlanShape (the
    sub-shape); the codebooks are usually shared down the tree, so `child()` clones this shape with a new
    branch skeleton for convenience."""

    def __init__(self, actions, scopes, branches=None):
        self.actions = list(actions)
        self.scopes = list(scopes)
        self.branches = dict(branches) if branches else {}   # {name: PlanShape}

    def child(self, branch_skeleton):
        """A sub-shape sharing this shape's codebooks, with the given nested branch skeleton (a dict of
        {name: sub_skeleton})."""
        return PlanShape(self.actions, self.scopes, _skeleton_to_shapes(branch_skeleton, self.actions, self.scopes))


def _skeleton_to_shapes(skeleton, actions, scopes):
    """Turn a plain nested-dict branch skeleton {name: {name: {...}}} into {name: PlanShape} sharing the
    codebooks -- so a caller can describe the whole tree shape with one nested dict instead of building
    PlanShape objects by hand."""
    return {name: PlanShape(actions, scopes, _skeleton_to_shapes(sub or {}, actions, scopes))
            for name, sub in (skeleton or {}).items()}


def plan_shape(actions, scopes, branch_skeleton=None):
    """Build a PlanShape from the action/scope codebooks and a plain nested-dict branch skeleton."""
    return PlanShape(actions, scopes, _skeleton_to_shapes(branch_skeleton, actions, scopes))


# =================================================================================================
# Encode / decode the plan tree.
# =================================================================================================
def encode_plan(plan, vocab):
    """Encode a PlanNode tree as ONE hypervector. At each node: bundle the action and scope (role-bound value
    atoms) with each named branch's SUBTREE bound under its branch role. Recursive -- the whole tree collapses
    to one vector. bundle() renormalises at each level, so a deep subtree contributes unit scale alongside the
    node's own action/scope (the HRR nested-structure encoding -- Plate)."""
    parts = [bind(vocab.role(_R_ACTION), vocab.value(plan.action)),
             bind(vocab.role(_R_SCOPE), vocab.value(plan.scope))]
    for name, sub in plan.branches.items():
        parts.append(bind(vocab.role(_r_branch(name)), encode_plan(sub, vocab)))
    return bundle(parts)


def decode_plan(vec, shape, vocab):
    """Decode a plan vector back to a PlanNode GIVEN ITS SHAPE -- the schema-guided walk. At each node we
    unbind exactly the action/scope roles and the KNOWN branch roles (from the shape) and clean up against the
    shape's codebooks -- a deterministic recursion, no resonator search, so it stays exact far past the blind
    parse's cap. The returned node's `confidence` is the MEASURED decode cosine of its action."""
    a_probe = unbind(vec, vocab.role(_R_ACTION))
    a_sims = [cosine(a_probe, vocab.value(a)) for a in shape.actions]
    ai = int(np.argmax(a_sims))
    s_probe = unbind(vec, vocab.role(_R_SCOPE))
    s_sims = [cosine(s_probe, vocab.value(s)) for s in shape.scopes]
    si = int(np.argmax(s_sims))
    node = PlanNode(shape.actions[ai], shape.scopes[si], confidence=round(float(a_sims[ai]), 4))
    for name, sub_shape in shape.branches.items():
        node.branches[name] = decode_plan(unbind(vec, vocab.role(_r_branch(name))), sub_shape, vocab)
    return node


# =================================================================================================
# descend: walk to the branch matching the current situation -- IFMATCH generalised to a tree.
# =================================================================================================
def descend(vec, situation, shape, vocab, floor=None):
    """Walk the plan to the branch matching the current SITUATION and return the actions along that path.

    This is the generalisation of holographic_machine's IFMATCH from one gated instruction to a named branch
    tree. IFMATCH fires the next instruction iff cosine(state, x) >= tol; descend matches the situation
    against the branch CONDITION keys (each branch's name atom) by cosine, takes the best branch that clears
    the noise FLOOR, and recurses. If NO branch clears the floor it ABSTAINS -- returns the node's primary
    action ("no contingency applies"), the honest answer IFMATCH's hard one-branch gate cannot give over many
    branches (Togelius's selector + Cranmer's measured floor).

    `situation` is either a branch-name str (exact selection -- you already know the condition) or a vector
    (cleaned up against the branch name atoms -- a noisy/real state). `floor` defaults to the vocab's measured
    noise floor. Matching at each level uses the SAME situation, so a single situation descends until it no
    longer matches a deeper branch; pass a fresh situation per level (call descend on the sub-vector) for an
    evolving state."""
    if floor is None:
        floor = vocab.noise_floor()

    # this node's primary action (the fall-through / abstain answer)
    a_probe = unbind(vec, vocab.role(_R_ACTION))
    action = shape.actions[int(np.argmax([cosine(a_probe, vocab.value(a)) for a in shape.actions]))]
    path = [action]

    branches = list(shape.branches.keys())
    if not branches:
        return path

    if isinstance(situation, str):
        chosen = situation if situation in shape.branches else None
        score = 1.0 if chosen is not None else 0.0
    else:
        sims = {b: cosine(situation, vocab.value(b)) for b in branches}        # match state to branch name atoms
        chosen = max(branches, key=lambda b: sims[b])
        score = sims[chosen]

    if chosen is not None and score >= floor:                                  # a contingency applies -> descend
        sub_vec = unbind(vec, vocab.role(_r_branch(chosen)))
        return path + descend(sub_vec, situation, shape.branches[chosen], vocab, floor)
    return path                                                                # abstain: no branch applies


# =================================================================================================
def _selftest():
    """Exercise the general record path, the plan round-trip, descend (match + abstain), and determinism."""
    vocab = ShapeVocab(1024, seed=0)

    # --- general flat record (the "bring your own shape" / science path) ---
    actions = ["advance", "hold", "retreat", "scan", "sample", "abort", "reroute", "wait"]
    scopes = ["global", "local", "step", "mission"]
    rec = {"phase": "stationary", "regime": "high_vol", "call": "hold"}
    rec_schema = {"phase": ["stationary", "drifting"], "regime": ["low_vol", "high_vol"], "call": actions}
    assert decode_record(encode_record(rec, vocab), rec_schema, vocab) == rec

    # --- plan round-trip (schema-guided decode of a 3-level contingency tree) ---
    plan = PlanNode("advance", "mission", branches={
        "blocked": PlanNode("reroute", "local", branches={
            "lowfuel": PlanNode("hold", "step"),
            "anomaly": PlanNode("scan", "local")}),
        "contact": PlanNode("hold", "global", branches={
            "degraded": PlanNode("abort", "mission")})})
    shape = plan_shape(actions, scopes,
                       {"blocked": {"lowfuel": {}, "anomaly": {}}, "contact": {"degraded": {}}})
    vec = encode_plan(plan, vocab)
    back = decode_plan(vec, shape, vocab)
    assert back == plan                                                        # structure + every label exact
    assert back.branches["blocked"].branches["anomaly"].action == "scan"
    assert 0.0 < back.confidence <= 1.0                                        # confidence is the measured cosine

    # --- descend: matches the situation to a branch, abstains when none applies (IFMATCH generalised) ---
    assert descend(vec, "blocked", shape, vocab) == ["advance", "reroute"]
    assert descend(vec, "contact", shape, vocab) == ["advance", "hold"]
    assert descend(vec, "clear", shape, vocab) == ["advance"]                  # no branch -> abstain at the root
    sv = vocab.value("blocked")                                               # a state VECTOR also matches
    assert descend(vec, sv, shape, vocab) == ["advance", "reroute"]

    # --- determinism (Macklin): same inputs -> bit-identical vector and identical walk ---
    assert np.array_equal(encode_plan(plan, vocab), encode_plan(plan, vocab))
    assert descend(vec, "blocked", shape, vocab) == descend(vec, "blocked", shape, vocab)

    print("holographic_planshape: ok")


if __name__ == "__main__":
    _selftest()
