"""Regression traps for the delegation-drift seam audit (sweep 131).

The claim: a faculty that DELEGATES must be able to forward everything its delegate accepts, and where
it deliberately cannot, the narrowing is on the record rather than silent. Two ways this dies quietly --
the count creeps back up as new faculties land, or the tool gets "fixed" into never reporting anything.
Both are pinned here.
"""
import json
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from tools.delegation_drift import (  # noqa: E402
    BUDGET, _bound_at_call_site, _delegate_names, _local_dict_keys, _params, audit_quiet)


def test_the_tool_selftest_passes():
    """The audit must be able to FAIL on a synthetic drift, or it is decoration -- the exact failure it
    exists to catch in others."""
    p = subprocess.run([sys.executable, os.path.join("tools", "delegation_drift.py"), "--selftest"],
                       cwd=_REPO, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONHASHSEED="0"))
    assert p.returncode == 0, p.stdout[-600:] + p.stderr[-600:]


def test_drift_may_only_shrink():
    """The budget gate. Not zero -- seven findings remain and are named in the budget file -- but a new
    faculty that forgets to plumb a parameter through cannot land silently."""
    p = subprocess.run([sys.executable, os.path.join("tools", "delegation_drift.py"), "--gate"],
                       cwd=_REPO, capture_output=True, text=True, timeout=900,
                       env=dict(os.environ, PYTHONHASHSEED="0"))
    assert p.returncode == 0, p.stdout[-900:]


def test_the_budget_file_states_a_reason():
    """A budget without a reason is a mute button. Every number here is a decision someone must be able
    to read back."""
    b = json.load(open(os.path.join(_REPO, "tools", "delegation_drift_budget.json")))
    assert isinstance(b["missing_budget"], int)
    assert len(b["why"]) > 200, "record WHY the ceiling is where it is, not just that it is"


def test_a_bound_parameter_is_never_reported_as_unreachable():
    """THE 45-of-99 FIX. A parameter the wrapper binds itself is decided, not lost; a name-only check
    cannot tell those apart and reported both for a year."""
    def delegate(a, b, c=1, d=2):
        pass

    def keyword_bound(a, b):
        """See holographic_x.delegate."""
        return delegate(a, b, c=self_c, d=7)

    def positionally_bound(a, b):
        """See holographic_x.delegate."""
        return delegate(a, b, 3, 4)

    tgt, _ = _params(delegate)
    assert set(_bound_at_call_site(keyword_bound, "delegate", tgt)) == {"a", "b", "c", "d"}
    assert set(_bound_at_call_site(positionally_bound, "delegate", tgt)) == {"a", "b", "c", "d"}


def test_an_unreadable_forward_reports_rather_than_excuses():
    """The conservative direction, pinned. A false positive is one budget line; a false negative is the
    silence this tool exists to break -- and assuming `**kw` covered everything really did hide two
    parameters (creature_tree's tip_inset and mount_flare) behind a call that never forwarded them."""
    def delegate(a, b, c=1, d=2):
        pass

    def readable(a, b):
        """See holographic_x.delegate."""
        kw = dict(c=1)
        return delegate(a, b, **kw)

    def opaque(a, b, extra):
        """See holographic_x.delegate."""
        kw = extra
        return delegate(a, b, **kw)

    tgt, _ = _params(delegate)
    assert "c" in _bound_at_call_site(readable, "delegate", tgt)
    assert "d" not in _bound_at_call_site(readable, "delegate", tgt)
    # `a` and `b` are still bound -- they are passed POSITIONALLY, which the reader sees. What an
    # unreadable **kw must not do is claim the parameters only it could have carried.
    opaque_bound = _bound_at_call_site(opaque, "delegate", tgt)
    assert set(opaque_bound) == {"a", "b"} and "c" not in opaque_bound and "d" not in opaque_bound


def test_an_aliased_import_is_still_the_delegate():
    """`from mod import fn as _d` is idiomatic in these wrappers; matching only the bare name read
    `_d(self, ...)` as a call to something else and reported three faculties that plainly supply
    mind=self."""
    import ast
    tree = ast.parse("from m import delegate as _d\n_d(1)\n")
    assert _delegate_names(tree, "delegate") == {"delegate", "_d"}
    assert _local_dict_keys(ast.parse("kw = dict(x=1)"), ast.Name(id="kw")) == {"x"}
    assert _local_dict_keys(ast.parse("kw = whatever"), ast.Name(id="kw")) is None


def test_private_parameters_are_never_public_api():
    """holo_octree's `_depth` and logic_prove's `_return_table` are recursion plumbing. A faculty that
    exposed them would be the bug, not the fix -- so they are dropped before they can reach a budget."""
    r = audit_quiet()
    flagged = {p for row in r["missing"] for p in row["missing"]}
    assert not [p for p in flagged if p.startswith("_")], flagged


def test_restored_parameters_actually_reach_their_delegate():
    """The burn-down's own claim, spot-checked live: the value a caller passes must arrive at the
    delegate. Output-diffing cannot separate 'not wired' from 'no visible effect on this input'; a spy
    on the delegate can."""
    import inspect

    import lecore
    import numpy as np
    from tools import delegation_drift as dd

    mind = lecore.UnifiedMind(dim=64, seed=0)
    cases = [("shape_memory_probe", "holographic_morphogen", "shape_memory_probe", "bins", 3,
              dict(n_shapes=2, n_cells=12, trials=2)),
             ("quantum_velocity", "holographic_probability_current", "velocity_field", "eps", 1e-3,
              dict(psi=np.ones((8, 8), complex))),
             ("curl_noise", "holographic_curlnoise", "curl_noise", "dx", 2.5,
              dict(res=8, octaves=1, seed=0))]

    class Caught(Exception):
        def __init__(self, kw):
            self.kw = kw

    for faculty, modname, fn, param, value, base in cases:
        mod = dd._find_module(modname)
        orig = getattr(mod, fn)
        order = list(inspect.signature(orig).parameters)

        def spy(*a, **kw):
            bound = dict(kw)
            for i, v in enumerate(a):
                if i < len(order):
                    bound[order[i]] = v
            raise Caught(bound)

        setattr(mod, fn, spy)
        try:
            with pytest.raises(Caught) as got:
                getattr(mind, faculty)(**dict(base, **{param: value}))
            assert got.value.kw.get(param) == value, "%s did not forward %s" % (faculty, param)
        finally:
            setattr(mod, fn, orig)


def test_the_narrowings_still_carry_their_reason():
    """BUDGET entries are decisions on the record. An entry without a reason is a hole in the audit
    wearing an exemption's clothes."""
    assert BUDGET, "the gait_* narrowings are still the worked example of a deliberate one"
    for name, why in BUDGET.items():
        assert len(why) > 20, "%s is exempt without saying why" % name


def test_the_instrument_is_reachable_through_the_mind():
    """A capability find_capability cannot surface and /invoke cannot call does not exist -- which is
    what this audit itself was until now: a tools/ script, uncatalogued, measuring a 99-item backlog."""
    import lecore
    r = lecore.UnifiedMind(dim=64, seed=0).delegation_drift()
    assert r["checked"] > 1000 and isinstance(r["total_missing"], int)
    # THE TEST THAT FAILED ON SUCCESS. This asserted `r["missing"]` was non-empty in order to
    # inspect a record's shape -- and the same sweep that made the instrument reachable burned the
    # backlog from 99 to 0, so the assertion started failing BECAUSE the work had worked. A test
    # that requires a finding is a test that goes red the day the finding is fixed. Shape is now
    # checked on whichever section actually has rows; `missing` being empty is the goal state.
    assert isinstance(r["missing"], list)
    if r["missing"]:
        assert set(r["missing"][0]) == {"faculty", "delegate", "missing", "overlap"}
    else:
        assert r["supplied"], "no findings AND no supplied records -- the reader did not run"
        assert set(r["supplied"][0]) == {"faculty", "delegate", "parameter", "bound_to"}
    assert r["supplied"] and set(r["supplied"][0]) == {"faculty", "delegate", "parameter", "bound_to"}
