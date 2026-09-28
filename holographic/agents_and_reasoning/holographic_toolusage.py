"""holographic_toolusage.py -- the READ VIEW of the tool door's audit (what `mind.tool_usage` returns).

WHY THIS EXISTS
---------------
E4.4 (CLM backlog, ONE tool learner): tool choice used to be learned twice -- by the tool reflex and by a separate
UsageTrace (holographic_lever7) that serve() never read. The learner is now the tool door's ProtoStore
(holographic_unified_p33_router), and the old UsageTrace object is gone from the mind. Callers and tests still read
`mind.tool_usage.counts / .failures / .dim` and call `.note(...)` / `.predict(...)`, so this view keeps that surface.

It holds NO STATE of its own: every read goes back to the mind (the tallies live in mind._tool_audit(), the key
dimension on mind.experience, the learner behind mind.tool_note / mind.tool_predict). A view that copied the tallies
would go stale the moment a verdict landed -- a view that holds a reference cannot.

WHY A MODULE OF ITS OWN (backlog G2): the class first lived beside _UnifiedPart19 in
holographic_unified_p19_lever7.py, which broke tests/test_unified_split.py's rule of ONE class per mixin part (the
rule exists so a part's methods can be checked for cross-part name collisions -- a second class in a part is a place
where that check goes blind). Behaviour is unchanged: the same five members, the same forwarding.

numpy-free, stdlib-free: it only forwards.
"""


class ToolUsageView:
    """The tool door's audit, read live from the mind: .counts (successful uses per tool), .failures (failed uses
    per tool), .dim (the tool door's key dimension), .note(task_vec, tool, success) = mind.tool_note, and
    .predict(task_vec, k) = mind.tool_predict. Construct with the mind; it keeps nothing else."""

    def __init__(self, mind):
        self._m = mind

    @property
    def dim(self):
        """The key dimension task vectors must have (the mind's experience trace dimension)."""
        return int(self._m.experience.dim)

    @property
    def counts(self):
        """{tool: successful uses} -- the audit dict itself (live, not a copy)."""
        return self._m._tool_audit()["counts"]

    @property
    def failures(self):
        """{tool: failed uses} -- the audit dict itself (live, not a copy)."""
        return self._m._tool_audit()["failures"]

    def note(self, task_vec, tool, success=True):
        """Record a judged tool use (forwards to mind.tool_note)."""
        return self._m.tool_note(task_vec, tool, success)

    def predict(self, task_vec, k=3):
        """Rank known tools for a task (forwards to mind.tool_predict)."""
        return self._m.tool_predict(task_vec, k)


def _selftest():
    """The view forwards and holds no state: a stand-in mind's tallies are read live through it."""
    class _Stub:
        class experience:
            dim = 16

        def __init__(self):
            self.a = {"counts": {}, "failures": {}}

        def _tool_audit(self):
            return self.a

        def tool_note(self, v, tool, success=True):
            self.a["counts" if success else "failures"][tool] = 1
            return {"tool": tool}

        def tool_predict(self, v, k=3):
            return [("t", 1.0)][:k]

    m = _Stub()
    u = ToolUsageView(m)
    assert u.dim == 16 and u.counts == {} and u.failures == {}
    u.note(None, "t")
    assert u.counts == {"t": 1}                      # read live after the note (a copy would still be empty)
    u.note(None, "bad", success=False)
    assert u.failures == {"bad": 1}
    assert u.predict(None, k=1) == [("t", 1.0)]
    return "ok"


if __name__ == "__main__":
    print(_selftest())
