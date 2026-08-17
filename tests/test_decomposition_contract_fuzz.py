"""A1's GATE, pinned. The decomposition contract itself already shipped (COMPLETE / CAUSAL /
HONEST RESIDUAL); what the backlog actually asked for was the fuzz gate proving it catches
broken decompositions. Measured over 500 random cases: 234 broken all caught, 0 missed,
0 false alarms. This keeps a sample of that running forever.
"""
import numpy as np


def _case(t):
    r = np.random.default_rng(1000 + t)
    x = np.cumsum(r.normal(size=int(r.integers(64, 256)))) * 0.1
    kind = int(r.integers(4))
    if kind == 0:
        return x, (lambda y: {"a": y * 0.3, "b": y * 0.7}), True
    if kind == 1:
        return x, (lambda y: {"lo": y * 0.5, "hi": y * 0.25, "residual": y * 0.25}), True
    if kind == 2:
        return x, (lambda y: {"a": y * 0.3, "b": y * 0.5}), False       # does not sum back
    return x, (lambda y: {"only": y * 1.1}), False                      # scaled copy


def test_contract_catches_broken_decompositions():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    missed = alarms = caught = passed = 0
    for t in range(60):
        x, f, expect = _case(t)
        ok = m.decomposition_contract(f, x)["complete"]
        if expect and ok:
            passed += 1
        elif not expect and not ok:
            caught += 1
        elif expect and not ok:
            alarms += 1
        else:
            missed += 1
    assert missed == 0, "a broken decomposition was certified COMPLETE"
    assert alarms == 0, "a valid decomposition was refused"
    assert caught > 0 and passed > 0, "the fuzz exercised only one side"
