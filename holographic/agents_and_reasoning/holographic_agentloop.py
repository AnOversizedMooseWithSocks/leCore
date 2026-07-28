"""LOOP-1 -- the in-process agent tool-use loop (holographic_agentloop).

WHAT WAS MISSING
----------------
Over HTTP this already works: `GET /tools` hands an agent the manifest, `POST /invoke` dispatches. IN
PROCESS there was no loop -- nothing that hands a model the manifest, parses a tool call out of its reply,
invokes it, feeds the result back and iterates to a stop condition. So every embedder wrote that loop
themselves, which is precisely the drift `invoke` exists to prevent: `invoke` was created to be the single
choke point, and a hand-rolled loop per caller routes around it.

THE DIFFERENTIATOR, AND IT IS NOT THE LOOP
------------------------------------------
Loops are easy. What this one does that a fluent one does not is CONSULT A NULL BEFORE IT ACTS. Before any
step runs, `route_or_abstain` scores the task against a null built from the catalog's own vocabulary at
matched token count. Below the floor the loop REFUSES THE TASK AND SAYS WHY, rather than handing the model
its best guess and executing whatever comes back.

That ordering is the architecture: the deterministic gate sits BELOW the model, not beside it. The
published reason to prefer that arrangement is that models cannot be relied on to abstain for themselves --
reasoning fine-tuning has been measured to DEGRADE abstention, and stated confidence does not convert into
action decisions. So the engine disposes; the model only proposes.

WHAT IT DELEGATES (all of it, deliberately)
  manifest         -> find_capability + describe_skill   (relevant tools, not all ~1,500)
  dispatch         -> mind.invoke                        (the choke point; refuses private/unknown names)
  the abstain gate -> mind.route_or_abstain              (null-referenced, not an argmax)
  arg hashing      -> hashlib.blake2b

ARGS ARE RECORDED AS A DIGEST PLUS A SHORT REPR, NEVER THE LIVE OBJECT. The precedent is on record: a live
object held in a job's args crashed a worker AFTER the job had already succeeded, on stderr, uncatchable. A
transcript that holds the caller's arrays also pins them in memory and can be mutated after the fact, which
makes it evidence of nothing.

NaN IS PARSED, SO NaN IS GUARDED. Python's `json` accepts bare `NaN` and `Infinity` by default, so a model
can emit one and it will arrive as a float. A non-finite number in a tool's arguments is refused here rather
than passed to a faculty, for the same reason the declare ladder guards its gates: a NaN does not lose
comparisons, it wins them.
"""

import hashlib
import json
import math
import re


def _finite(value):
    """True when `value` contains no non-finite number, recursively. Used to refuse model-supplied args."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def arg_fingerprint(args):
    """A stable digest plus a short repr for one call's arguments -- what a transcript records INSTEAD of the
    arguments themselves. Deterministic across processes (blake2b, never hash())."""
    text = json.dumps(args, sort_keys=True, default=repr)
    return {"digest": hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(),
            "repr": text[:120]}


def parse_action(reply):
    """Pull an action out of a model's reply. Returns {"kind": "call"|"done"|"unparsed", ...}.

    Accepts a JSON object with `tool`/`args`, or a `DONE:` line. Deliberately forgiving about surrounding
    prose (models wrap JSON in chatter) and deliberately UNFORGIVING about anything it cannot read: an
    unparsed reply is reported as unparsed, never guessed at. Guessing the intent of an unclear tool call is
    exactly how a loop takes an action nobody asked for."""
    text = (reply or "").strip()
    done = re.search(r"DONE\s*:\s*(.*)", text, re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and obj.get("tool"):
                return {"kind": "call", "tool": str(obj["tool"]),
                        "args": obj.get("args") if isinstance(obj.get("args"), dict) else {}}
        except ValueError:
            pass
    if done:
        return {"kind": "done", "answer": done.group(1).strip()}
    return {"kind": "unparsed", "reply": text[:200]}


class AgentLoop:
    """Runs a task by letting `llm` (any callable text->text) pick tools, with a null-referenced gate in
    front of every action and `invoke` as the only dispatch path."""

    def __init__(self, mind, llm, max_steps=6, z_min=0.8, k_tools=6, seed=0):
        if not callable(llm):
            raise TypeError("agent_loop needs a callable text->text, got %r" % type(llm))
        self.mind = mind
        self.llm = llm
        self.max_steps = int(max_steps)
        self.z_min = float(z_min)
        self.k_tools = int(k_tools)
        self.seed = int(seed)

    def manifest(self, task):
        """The RELEVANT slice of the tool manifest, not all of it. ~1,500 faculties will not fit a prompt,
        and pasting them all would also make the model's choice harder rather than easier -- retrieval is
        the engine's job, so it does it here too."""
        rows = []
        for cap in self.mind.find_capability(task, k=self.k_tools):
            method = getattr(cap, "method", None)
            if not method:
                continue                       # import-only capabilities are not callable; do not advertise them
            try:
                desc = self.mind.describe_skill(method)
                call = desc.get("call") or method
            except Exception:
                call = method
            rows.append({"tool": method, "call": call, "does": (getattr(cap, "does", "") or "")[:160]})
        return rows

    def _prompt(self, task, tools, history):
        lines = ["TASK: %s" % task, "", "TOOLS:"]
        for row in tools:
            lines.append("  %s -- %s" % (row["tool"], row["does"]))
        if history:
            lines += ["", "SO FAR:"]
            for step in history:
                lines.append("  %s -> %s" % (step["tool"], str(step["result"])[:100]))
        lines += ["", 'Reply with EXACTLY one JSON object {"tool": "<name>", "args": {...}}',
                  'or "DONE: <answer>" when the task is complete.']
        return "\n".join(lines)

    def run(self, task):
        """Run the loop. Returns {done, refused, answer, why, steps, gate}.

        REFUSAL IS A RESULT, and it happens BEFORE the model is consulted: if the task does not clear the
        null floor, no step runs at all and `why` carries the router's reason. That is what holds the
        false-action rate down -- not the model's judgement about its own competence."""
        gate = self.mind.route_or_abstain(task, z_min=self.z_min, seed=self.seed)
        if gate.get("abstain"):
            return {"done": False, "refused": True, "answer": None, "steps": [], "gate": gate,
                    "why": "refused before acting: %s" % gate.get("reason", "below the null floor")}

        tools = self.manifest(task)
        if not tools:
            return {"done": False, "refused": True, "answer": None, "steps": [], "gate": gate,
                    "why": "the task cleared the null floor but no CALLABLE capability matched it"}

        steps, history = [], []
        for _ in range(self.max_steps):
            try:
                reply = self.llm(self._prompt(task, tools, history))
            except Exception as exc:
                return {"done": False, "refused": False, "answer": None, "steps": steps, "gate": gate,
                        "why": "the model raised: %s" % exc}

            action = parse_action(reply)
            if action["kind"] == "done":
                return {"done": True, "refused": False, "answer": action["answer"], "steps": steps,
                        "gate": gate, "why": "the model reported completion"}
            if action["kind"] == "unparsed":
                steps.append({"tool": None, "args": None, "result": None,
                              "why": "unparsed reply: %r" % action["reply"]})
                continue                       # do not guess; ask again with the same manifest

            name, args = action["tool"], action["args"]
            if not _finite(args):
                steps.append({"tool": name, "args": arg_fingerprint(args), "result": None,
                              "why": "refused: non-finite number in arguments"})
                continue
            if name not in [row["tool"] for row in tools]:
                steps.append({"tool": name, "args": arg_fingerprint(args), "result": None,
                              "why": "refused: %r is not in the offered manifest" % name})
                continue
            try:
                result = self.mind.invoke(name, args)
                why = "invoked"
            except Exception as exc:
                result, why = None, "invoke raised: %s" % exc
            record = {"tool": name, "args": arg_fingerprint(args), "result": result, "why": why}
            steps.append(record)
            history.append({"tool": name, "result": result})

        return {"done": False, "refused": False, "answer": None, "steps": steps, "gate": gate,
                "why": "ran out of steps (max_steps=%d)" % self.max_steps}


def _selftest():
    import lecore

    mind = lecore.UnifiedMind(dim=256, seed=0)

    # 1. THE PRIMARY PROPERTY: a task with no matching capability is refused BEFORE the model is consulted.
    calls = {"n": 0}

    def counting_llm(_):
        calls["n"] += 1
        return '{"tool": "mesh_smooth", "args": {}}'

    out = AgentLoop(mind, counting_llm).run("purple monkey dishwasher")
    assert out["refused"] and not out["done"], out
    assert calls["n"] == 0, "the model was consulted %d times on a task that should never reach it" % calls["n"]

    # 2. A model naming a tool OUTSIDE the manifest is refused, not dispatched. `invoke` would also refuse
    #    an unknown name, but refusing here keeps the REASON specific.
    out = AgentLoop(mind, lambda _: '{"tool": "os_system", "args": {}}', max_steps=1).run("smooth a bumpy mesh")
    assert out["steps"] and "not in the offered manifest" in out["steps"][0]["why"], out

    # 3. NON-FINITE ARGS ARE REFUSED. json parses bare NaN, so this is reachable from a real model.
    out = AgentLoop(mind, lambda _: '{"tool": "mesh_smooth", "args": {"iters": NaN}}',
                    max_steps=1).run("smooth a bumpy mesh")
    assert out["steps"] and "non-finite" in out["steps"][0]["why"], out

    # 4. DONE terminates and carries the answer.
    out = AgentLoop(mind, lambda _: "DONE: the mesh is smooth").run("smooth a bumpy mesh")
    assert out["done"] and out["answer"] == "the mesh is smooth"

    # 5. AN UNPARSED REPLY IS NOT GUESSED AT -- it is recorded and retried, never turned into an action.
    out = AgentLoop(mind, lambda _: "I think maybe we should try something?", max_steps=2).run(
        "smooth a bumpy mesh")
    assert len(out["steps"]) == 2 and all("unparsed" in s["why"] for s in out["steps"])
    assert not out["done"]

    # 6. ARGS ARE A DIGEST, NEVER THE OBJECT. A transcript holding live objects is evidence of nothing.
    fp = arg_fingerprint({"a": [1, 2, 3]})
    assert set(fp) == {"digest", "repr"} and len(fp["digest"]) == 16
    assert arg_fingerprint({"a": 1}) == arg_fingerprint({"a": 1})
    assert arg_fingerprint({"a": 1}) != arg_fingerprint({"a": 2})

    # 7. A model that raises does not take the loop down.
    def boom(_):
        raise RuntimeError("model offline")

    out = AgentLoop(mind, boom).run("smooth a bumpy mesh")
    assert not out["done"] and "model raised" in out["why"]

    # 8. Guard.
    try:
        AgentLoop(mind, 42)
        raise AssertionError("accepted a non-callable model")
    except TypeError:
        pass

    print("holographic_agentloop: all selftests passed (gate before model, manifest, NaN, digest, guards)")


if __name__ == "__main__":
    _selftest()
