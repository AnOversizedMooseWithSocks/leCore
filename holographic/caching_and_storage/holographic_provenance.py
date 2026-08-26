"""holographic_provenance.py -- tag a vector with WHERE it came from, one model for the whole stack.

Every vector that enters leCore from a source -- an actor, an external model, a bus sender, a fork, a farm node --
can carry a SOURCE ROLE: a deterministic role vector derived from the source's name. Bind a value to its source role
and the value now records where it came from; unbind by the same role to recover it and confirm its origin.

This is the single provenance convention the rest of the composability stack reads: a Principal's actor id, the bus's
`sender`, opponent's "which source", and merge_forks' "which fork" are all the SAME idea -- a named source with a
deterministic role. Because the role is a pure function of the name (no registry, no shared state), any node derives
the same role for the same source, so "who produced this?" is asked and answered identically everywhere.

numpy/stdlib only; deterministic.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, derived_atom


def source_role(source, dim=1024, seed=0):
    """A deterministic role vector for a named SOURCE. The same source name always yields the same role -- it's a pure
    function of (seed, name) via derived_atom, so no registry is needed and every node agrees. Use it to TAG a value
    with its origin (bind) or to CHECK a value's origin (unbind)."""
    return derived_atom(seed, "source:" + str(source), dim)


def from_external(vec, source, dim=None, seed=0):
    """Bring a vector computed OUTSIDE leCore into the shared space, tagged with its origin: bind(source_role(source),
    unit(vec)). `dim` defaults to the vector's own length. The result no longer looks like the bare input -- it looks
    like 'this value, from that source' -- which is what keeps two sources' contributions from being confused."""
    v = np.asarray(vec, float)
    n = np.linalg.norm(v)
    u = v / n if n > 0 else v
    return bind(source_role(source, dim or len(u), seed), u)


def of_source(tagged, source, dim=None, seed=0):
    """Recover a from_external() value by unbinding its source role. If `tagged` really carries this source's tag, the
    result is (close to) the original unit value; if it was some other source's, the result is noise -- so the cosine
    of the result against a candidate original is a provenance CHECK."""
    t = np.asarray(tagged, float)
    return unbind(t, source_role(source, dim or len(t), seed))


def _selftest():
    rng = np.random.default_rng(0)
    dim = 1024

    # same name -> same role, everywhere (no registry, pure function)
    assert np.allclose(source_role("alice", dim), source_role("alice", dim))
    # different names -> (near) orthogonal roles, so tagged values don't collide
    ra, rb = source_role("alice", dim), source_role("bob", dim)
    assert abs(float(np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))) < 0.1

    # tag a value with its source, then recover it by unbinding the same role. Single bind/unbind recovery is APPROX
    # in HRR and improves with dimension (~0.7 at 1024, higher at 4096+); what matters for provenance is that the
    # RIGHT source recovers a clear signal while the WRONG source recovers noise.
    v = rng.standard_normal(dim); v /= np.linalg.norm(v)
    tagged = from_external(v, "alice", dim)
    recovered = of_source(tagged, "alice", dim)
    cos = float(np.dot(recovered, v) / (np.linalg.norm(recovered) * np.linalg.norm(v)))
    assert cos > 0.6, cos                                       # right source -> clear signal

    # the WRONG source recovers noise (provenance check fails), well below the right source
    wrong = of_source(tagged, "bob", dim)
    cos_wrong = float(np.dot(wrong, v) / (np.linalg.norm(wrong) * np.linalg.norm(v)))
    assert cos_wrong < 0.3 and cos > 2 * abs(cos_wrong), (cos, cos_wrong)   # right clearly beats wrong

    print("OK: holographic_provenance self-test passed (same name -> same role; different names -> orthogonal; "
          "from_external tags a value and of_source recovers it with the right source, noise with the wrong one)")


if __name__ == "__main__":
    _selftest()
