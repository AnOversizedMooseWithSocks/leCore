"""holographic_policy.py -- ROUTING POLICIES AS HYPERVECTOR PROGRAMS (backlog v2, E4').

leOS kept its decision trees as Python node objects -- policy living OUTSIDE the substrate, where
it cannot be stored, composed, priced, or audited by the engine's own machinery. Here a policy IS
a HoloMachine program: an alternating sequence of IFMATCH(prototype) / CALL-style action steps,
ASSEMBLED into one hypervector by the VM's own instruction encoding and DECODED back for
execution and audit. The policy is therefore data in the engine's one space: storable in the
function library, content-addressable, inspectable step by step, and (via the installed
certify->compile pipeline) a candidate for exact-weight baking -- the nearest published relative
is Tracr (RASP->transformer weights, an interpretability testbed with one-hot streams and no
learned policies or certificates); the combination of experience-learned routing + vector program
+ decode-audit is this engine's own.

EXECUTION SEMANTICS (deliberately small, honest scope): steps are evaluated in order; an IFMATCH
step whose prototype matches the context (cosine >= its threshold) routes to its paired ACTION
and stops -- first match wins, and the LAST step must be an unconditional fallback (enforced at
build time; a policy with no fallback is refused, the leOS rule kept). Path provenance is always
returned: {action, step, similarity, path} -- an answer without its route is the failure mode the
decision layer exists to prevent.

LEVER-7 SHORTCUT: execute() accepts an optional DisplacementTrace; a gated hit on the context
serves the remembered action atom BEFORE the program runs (the tree-evolution shortcut, with the
full reflex gate rather than a bare 0.80 threshold). OUTCOMES feed back through the same trace.

Deterministic; NumPy + stdlib; atoms seed-derived. The program vector round-trip is asserted in
the selftest: decode(assemble(steps)) == steps, per instruction, via the VM's own codebooks.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, cosine, random_vector
from holographic.agents_and_reasoning.holographic_machine import HoloMachine
from holographic.agents_and_reasoning.holographic_lever7 import key_atom


class PolicyProgram:
    """One routing policy: steps = [(prototype_tag, action_tag, threshold), ...] with the final
    step's prototype None (the mandatory fallback). The program lives as ONE vector, assembled
    and decoded by the HoloMachine's instruction encoding."""

    def __init__(self, name, steps, dim=2048, seed=0):
        if not steps or steps[-1][0] is not None:
            raise ValueError("a policy MUST end in an unconditional fallback step (leOS rule kept)")
        self.name = str(name)
        self.dim = int(dim)
        self.seed = int(seed)
        self.steps = [(p, a, float(t)) for (p, a, t) in steps]
        # data atoms the VM instruction encoding will reference:
        tags = sorted({p for p, _, _ in steps if p} | {a for _, a, _ in steps})
        self.hm = HoloMachine(dim=self.dim, seed=self.seed, data=tags)
        prog = []
        for p, a, _t in self.steps:
            prog.append(("IFMATCH", p if p is not None else a))
            prog.append(("BIND", a))          # the paired action, carried as the next instruction
        self.instructions = prog
        self.vector = self.hm.assemble(prog)  # THE POLICY IS ONE HYPERVECTOR
        self.prototypes = {p: key_atom(self.name + ":proto:" + p, self.dim)
                           for p, _, _ in self.steps if p is not None}

    def set_prototype(self, tag, vec):
        """Install a real prototype vector for an IFMATCH tag (defaults are seed-derived atoms)."""
        v = np.asarray(vec, float)
        self.prototypes[tag] = v / (np.linalg.norm(v) + 1e-12)

    def decode(self):
        """Read the program BACK out of the vector via the VM's own codebooks (positional
        unpermute + cleanup) -- the audit path. Asserted equal to the source in the selftest."""
        # REAL decode only -- the hasattr fallback that returned the stored list made the
        # round-trip assert vacuous (kept negative pinned on HoloMachine.disassemble).
        return self.hm.disassemble(self.vector, len(self.instructions))

    def execute(self, context_vec, trace=None, handlers=None):
        """Route a context. Optional `trace` (DisplacementTrace) is consulted FIRST under its full
        gate (the lever-7 shortcut); optional `handlers` maps action_tag -> callable(context).
        Returns {action, via, step, similarity, path, result?} -- provenance always included."""
        c = np.asarray(context_vec, float)
        path = []
        if trace is not None:
            hit = trace.read_gated(c)
            if hit["fired"]:
                # decode the remembered action atom back to a tag by cleanup over action atoms
                acts = {a: key_atom(self.name + ":act:" + a, self.dim)
                        for _, a, _ in self.steps}
                best = max(acts, key=lambda a: cosine(hit["prediction"], acts[a]))
                if cosine(hit["prediction"], acts[best]) > 0.5:
                    out = {"action": best, "via": "reflex-shortcut", "step": None,
                           "similarity": hit["confidence"], "path": ["reflex"]}
                    if handlers and best in handlers:
                        out["result"] = handlers[best](c)
                    return out
        for i, (p, a, t) in enumerate(self.steps):
            if p is None:
                path.append(f"{i}:fallback->{a}")
                out = {"action": a, "via": "fallback", "step": i, "similarity": None, "path": path}
                if handlers and a in handlers:
                    out["result"] = handlers[a](c)
                return out
            s = cosine(c, self.prototypes[p])
            path.append(f"{i}:IFMATCH({p})={s:.2f}")
            if s >= t:
                out = {"action": a, "via": "match", "step": i, "similarity": s, "path": path}
                if handlers and a in handlers:
                    out["result"] = handlers[a](c)
                return out
        raise AssertionError("unreachable: the fallback step is enforced at construction")

    def to_horn_rules(self):
        """The policy's ROUTE TABLE as ground Horn facts for the engine's logic layer (backlog
        E4.5'): one fact routes(prototype, action) per step, plus fallback(action). HONEST SCOPE
        (stated, not hidden): first-match PRIORITY and thresholds are NOT encoded -- Horn facts
        state WHICH routes exist and that a fallback exists; the ordering semantics live in the
        program vector and its decode. What the Lean export therefore certifies: the route table
        and the fallback's existence, machine-checkably, from a policy that was DATA all along."""
        rules = []
        for i, (p, a, _t) in enumerate(self.steps):
            if p is None:
                rules.append({"head": ["fallback", [a]], "body": [], "name": f"step{i}_fallback"})
            else:
                rules.append({"head": ["routes", [p, a]], "body": [], "name": f"step{i}"})
        rules.append({"head": ["total", ["policy"]], "body": [["fallback", ["proceed"]]]
                      if any(pp is None and aa == "proceed" for pp, aa, _ in self.steps)
                      else [["fallback", [self.steps[-1][1]]]],
                      "name": "totality_from_fallback"})
        return rules

    def remember(self, trace, context_vec, action_tag):
        """Feed a served route back into the lever-7 trace so the shortcut can learn it."""
        return trace.write(np.asarray(context_vec, float),
                           key_atom(self.name + ":act:" + action_tag, self.dim))


def seed_policy_loop_recovery(dim=2048, seed=0):
    """The LOOP_RECOVERY reference policy (leOS seed tree, ported as a program): a looping/echoing
    context routes to break_loop; a drifting-off-topic context to refocus; everything else falls
    through to proceed. Prototypes are installed by the caller from real drift/loop detectors."""
    return PolicyProgram("loop_recovery",
                         [("looping", "break_loop", 0.35),
                          ("drifting", "refocus", 0.35),
                          (None, "proceed", 0.0)], dim=dim, seed=seed)


def _selftest():
    rng = np.random.default_rng(0)
    dim = 2048
    p = seed_policy_loop_recovery(dim)
    # 1. the policy IS one vector, and the program decodes back from it
    assert p.vector.shape == (dim,)
    dec = p.decode()
    assert [tuple(x) for x in dec] == [tuple(x) for x in p.instructions], \
        "decode(assemble(steps)) must round-trip via the VM's own codebooks"
    # 2. routing with provenance
    loop_proto = random_vector(dim, rng); p.set_prototype("looping", loop_proto)
    drift_proto = random_vector(dim, rng); p.set_prototype("drifting", drift_proto)
    r = p.execute(loop_proto)
    assert r["action"] == "break_loop" and r["via"] == "match" and r["path"], "provenance required"
    r2 = p.execute(random_vector(dim, rng))
    assert r2["action"] == "proceed" and r2["via"] == "fallback"
    # 3. a policy without a fallback is refused (the leOS rule, kept)
    try:
        PolicyProgram("bad", [("a", "x", 0.5)], dim=dim)
        raise AssertionError("must refuse a policy with no fallback")
    except ValueError:
        pass
    # 4. the lever-7 shortcut learns a served route and takes it under the full gate
    from holographic.agents_and_reasoning.holographic_lever7 import DisplacementTrace
    tr = DisplacementTrace(dim, seed=1)
    ctx = loop_proto
    p.remember(tr, ctx, "break_loop")
    r3 = p.execute(ctx, trace=tr)
    assert r3["via"] == "reflex-shortcut" and r3["action"] == "break_loop"
    assert p.execute(random_vector(dim, rng), trace=tr)["via"] != "reflex-shortcut", \
        "the shortcut must not fire on a novel context (the gate holds)"
    return {"decoded_steps": len(dec), "route": r["action"], "shortcut": r3["via"]}


if __name__ == "__main__":
    print(_selftest())
