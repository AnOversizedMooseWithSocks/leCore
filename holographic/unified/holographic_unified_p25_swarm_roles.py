"""Part 25 of UnifiedMind's faculty surface -- the swarm use-case doors (sweep 125):
escalations / resolve (a service swarm that learns from its humans), codebase_sync (a
development swarm that shares one understanding of the code and notices when it
changes), and role (a focused agent that reacts to a topic on the bus with the
shared mind at hand).

WHY A PART OF THEIR OWN. Each door is a thin COMPOSITION of rails that already exist
(serve + teach with provenance; study + teach_about + stale_facts; bus.subscribe) --
none re-implements anything -- but they are the doors three kinds of users reach for
by name, and a door the power user cannot reach under its natural verb is undiscovered.

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which is still the only import path anyone
uses. Carries no `__init__`; assumes the state UnifiedMind.__init__ sets up.
"""
import os

from holographic.unified import check_part


class _UnifiedPart25:

    # ---------------------------------------------------------------- customer service
    def escalations(self):
        """The questions this mind could NOT serve (sweep 125): every serve() that
        escalated is recorded here, oldest first, until a human resolves it. A service
        swarm reads this list to route work to people; the list is the swarm's honest
        account of what it does not yet know. Returns [{question, reason, count}]."""
        led = getattr(self, "_escalations", None) or {}
        return [{"question": q, "reason": v["reason"], "count": v["count"]}
                for q, v in led.items()]

    def resolve(self, question, answer, by="human", propagate=None):
        """A HUMAN ANSWERS AN ESCALATION AND THE SWARM LEARNS IT (sweep 125). The answer
        is taught on the durable rails with provenance 'human:<by>' -- so every later
        serve() of the same question is served from memory with no escalation and the
        record shows WHO taught it -- and the escalation is cleared. propagate= is an
        optional commons bundle path: when given, the mind's shared rows are contributed
        there immediately so other partitions can draw the resolution now rather than
        at the next scheduled pool. Returns the teach receipt plus 'cleared' and, when
        propagated, the contribute review sheet."""
        r = self.teach(str(question), str(answer))
        if r.get("taught"):
            lg = getattr(self.zoo["ladder"], "taught_log", [])
            if lg and str(lg[-1][0]) == str(question) and len(lg[-1]) > 3:
                lg[-1] = [lg[-1][0], lg[-1][1], lg[-1][2], "human:%s" % str(by)]
        led = getattr(self, "_escalations", None) or {}
        cleared = led.pop(str(question), None) is not None
        self._escalations = led
        out = {"taught": bool(r.get("taught")), "by": str(by), "cleared": cleared}
        if propagate:
            out["propagated"] = self.contribute(str(propagate), author=str(by))
        return out

    # -------------------------------------------------------------- development swarm
    def codebase_sync(self, root, only_stale=False, budget_lines=120, max_files=400):
        """ONE UNDERSTANDING OF THE CODE FOR THE WHOLE SWARM (sweep 125): study(root)
        digests the tree, and each file's digest is taught as a fact FINGERPRINTED to
        that file (teach_about), so it persists in the partition across restarts and
        sessions, every agent that boots the partition already knows the code, and
        stale_facts() names exactly the files whose facts went stale when the code
        changed. only_stale=True re-teaches just those files -- the cheap re-sync after
        an edit. Returns {files, taught, stale_before, skipped}."""
        root = str(root)
        stale_qs = set()
        if only_stale:
            st = self.stale_facts(root=root)
            stale_qs = set(st.get("stale", [])) | set(st.get("missing", []))
        s = self.study(root, budget_lines=int(budget_lines))
        chunks = []
        for cell in (s["ask"].__closure__ or ()):
            v = cell.cell_contents
            if isinstance(v, list) and v and isinstance(v[0], dict) and "text" in v[0]:
                chunks = v
                break
        by_file = {}
        for c in chunks:
            src = str(c.get("source", "")).split(":", 1)[0]
            if src and os.path.isfile(src):
                by_file.setdefault(src, []).append(str(c["text"]))
        taught = skipped = 0
        for path, texts in sorted(by_file.items())[:int(max_files)]:
            rel = os.path.relpath(path, root)
            q = "what does %s do" % rel
            if only_stale and q not in stale_qs:
                skipped += 1
                continue
            digest = " | ".join(t[:300] for t in texts[:4])
            r = self.teach_about(q, digest, [rel], root=root)
            taught += 1 if (r.get("taught") or {}).get("taught", r.get("taught")) else 0
        return {"files": len(by_file), "taught": taught, "stale_before": sorted(stale_qs),
                "skipped": skipped, "root": root}

    # ----------------------------------------------------------------- lab of roles
    def role(self, name, pattern, handler, emit=None):
        """A FOCUSED AGENT ON THE SHARED BUS (sweep 125): subscribe handler(payload, mind)
        to every message whose topic matches pattern (the bus's own matching); if the
        handler returns something and emit= names a topic, the result is published
        there under the role's name -- so roles chain (chat -> researcher -> findings
        -> reporter -> reports) without knowing about each other. The role shares this
        mind's memory: what one learns, all recall; the topic keeps it focused. Returns
        the role record; roles() lists them."""
        bus = self.bus()
        reg = getattr(self, "_roles", None) or {}
        def _run(msg, _name=str(name), _emit=emit, _handler=handler):
            payload = msg.get("payload") if isinstance(msg, dict) else msg
            try:
                out = _handler(payload, self)
            except Exception as e:                   # a role that crashes must not take the bus down
                out = {"error": str(e)[:200], "role": _name}
            reg[_name]["handled"] += 1
            if out is not None and _emit:
                bus.publish(str(_emit), out, sender=_name)
            return out
        reg[str(name)] = {"name": str(name), "pattern": str(pattern), "emit": emit, "handled": 0}
        bus.subscribe(str(pattern), _run)
        self._roles = reg
        return reg[str(name)]

    def roles(self):
        """The roles registered on this mind's bus, with how many messages each handled."""
        return list((getattr(self, "_roles", None) or {}).values())


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p25_swarm_roles", "_UnifiedPart25")
    print("holographic_unified_p25_swarm_roles selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
