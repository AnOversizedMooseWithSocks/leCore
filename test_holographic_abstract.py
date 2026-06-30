"""Tests for abstract_program: a trace -> a reusable program that transfers (ABS-1)."""
import numpy as np
from holographic_unified import UnifiedMind
from holographic_ai import bind, cosine


def test_abstracts_a_transform_that_transfers():
    um = UnifiedMind(dim=1024, seed=0); M = um._machine()
    KEY = M.data_atoms[M.data_names[0]]
    xs = [M.data_atoms[d] for d in M.data_names[1:6]]
    examples = [(x, bind(x, KEY)) for x in xs[:3]]
    res = um.abstract_program(examples, name="apply_key")
    assert res["generalizes"] and res["worst"] >= 0.9
    # TRANSFER to a held-out input the abstraction never saw
    out, _ = um.run_procedure("apply_key", init_acc=xs[3])
    assert cosine(out, bind(xs[3], KEY)) > 0.9


def test_abstract_beats_prototype_on_transfer():
    um = UnifiedMind(dim=1024, seed=1); M = um._machine()
    KEY = M.data_atoms[M.data_names[0]]
    xs = [M.data_atoms[d] for d in M.data_names[1:6]]
    train = xs[:3]; held = xs[3]
    res = um.abstract_program([(x, bind(x, KEY)) for x in train], name="ak")
    out, _ = um.run_procedure("ak", init_acc=held)
    prog_t = cosine(out, bind(held, KEY))
    sims = [cosine(held, x) for x in train]
    proto_t = cosine(bind(train[int(np.argmax(sims))], KEY), bind(held, KEY))
    assert prog_t > 0.9 > proto_t                                  # program transfers; prototype returns stale output


def test_no_shared_transform_does_not_generalize():
    um = UnifiedMind(dim=1024, seed=2); M = um._machine()
    rng = np.random.default_rng(0)
    # examples with NO single VM-expressible shared transform -> should not claim a false abstraction
    examples = [(M.data_atoms[M.data_names[i]], rng.standard_normal(1024)) for i in range(3)]
    res = um.abstract_program(examples)
    assert not res["generalizes"]
