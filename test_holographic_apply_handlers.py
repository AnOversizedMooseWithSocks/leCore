"""Tests for register_apply_handler: making faculties (incl. stateful spatial ops + agent behaviours)
callable from HoloMachine programs as APPLY <name> (WIRE-1)."""

import numpy as np
from holographic_unified import UnifiedMind
from holographic_ai import cosine


def _mind():
    return UnifiedMind(dim=1024, seed=0)


def test_registered_handler_runs_in_a_program():
    um = _mind(); M = um._machine(); d0 = M.data_names[0]
    # a simple, verifiable unary faculty: scale-and-renormalize is identity in cosine
    target = np.random.default_rng(1).standard_normal(1024); target /= np.linalg.norm(target)
    um.register_apply_handler("emit_target", lambda acc: target)
    out, _ = um.run_procedure([("APPLY", "emit_target"), ("HALT", d0)], init_acc=M.data_atoms[d0])
    assert cosine(out, target) > 0.999                       # APPLY actually invoked the registered handler


def test_in_program_equals_direct_call():
    um = _mind(); M = um._machine(); d0 = M.data_names[0]
    A = np.random.default_rng(2).standard_normal((1024, 1024)) * 0.01

    def lin(acc):
        v = A @ acc
        return v / (np.linalg.norm(v) + 1e-12)
    um.register_apply_handler("lin", lin)
    x = M.data_atoms[d0]
    out, _ = um.run_procedure([("APPLY", "lin"), ("HALT", d0)], init_acc=x)
    assert cosine(out, lin(x)) > 0.999                       # the program's APPLY == calling the faculty directly


def test_agent_behaviour_callable_from_program():
    um = _mind()
    ag = um.agent(["grab", "lift", "place"], dim=1024, seed=0)
    s = np.random.default_rng(5).standard_normal(1024)
    ag.reward(s, "lift", 1.0)
    um.register_apply_handler("agent_act", lambda acc: ag.action_vec[ag.decide(acc).get("action", "grab")])
    d0 = um._machine().data_names[0]
    out, _ = um.run_procedure([("APPLY", "agent_act"), ("HALT", d0)], init_acc=s)
    assert cosine(out, ag.action_vec["lift"]) > 0.999        # the agent's learned choice ran inside the program


def test_handlers_chain_in_a_program():
    um = _mind(); M = um._machine(); d0 = M.data_names[0]
    log = []
    um.register_apply_handler("a", lambda acc: (log.append("a"), acc)[1])
    um.register_apply_handler("b", lambda acc: (log.append("b"), acc)[1])
    um.run_procedure([("APPLY", "a"), ("APPLY", "b"), ("HALT", d0)], init_acc=M.data_atoms[d0])
    assert log == ["a", "b"]                                  # both APPLY steps fired, in order


def test_registered_name_overrides_builtin():
    um = _mind(); M = um._machine(); d0 = M.data_names[0]
    sentinel = np.random.default_rng(3).standard_normal(1024); sentinel /= np.linalg.norm(sentinel)
    um.register_apply_handler("denoise", lambda acc: sentinel)   # override the built-in denoise
    out, _ = um.run_procedure([("APPLY", "denoise"), ("HALT", d0)], init_acc=M.data_atoms[d0])
    assert cosine(out, sentinel) > 0.999


def test_non_callable_rejected():
    um = _mind()
    try:
        um.register_apply_handler("bad", 42)
        assert False, "should have rejected a non-callable handler"
    except TypeError:
        pass
