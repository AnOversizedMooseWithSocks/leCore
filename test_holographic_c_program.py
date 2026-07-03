import numpy as np
import pytest

import holographic_c
from holographic_ai import bind, bundle, cosine
from holographic_machine import HoloMachine


pytestmark = pytest.mark.skipif(
    not holographic_c.available() or not getattr(holographic_c._BACKEND, "holo_program_run_basic", None),
    reason="C holographic program runner is not built",
)


def test_c_program_runner_matches_python_vm_for_core_ops():
    m = HoloMachine(dim=2048, seed=7)
    prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    program_vec = m.assemble(prog)
    py_acc, py_trace = m.run(program_vec, max_steps=len(prog))
    c_acc, c_trace = m.run_c_basic(program_vec, max_steps=len(prog))
    expected = bundle([bind(m.data_atoms["a"], m.data_atoms["b"]), m.data_atoms["c"]])
    assert py_trace == [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")]
    assert c_trace == py_trace
    assert cosine(c_acc, py_acc) > 0.999999
    assert cosine(c_acc, expected) > 0.999999


def test_c_program_runner_handles_ifmatch_hit_and_miss():
    m = HoloMachine(dim=2048, seed=7, data=["enemy_near", "calm", "flee_signal"])
    hit = [("IFMATCH", "enemy_near"), ("LOAD", "flee_signal"), ("HALT", "")]
    hit_vec = m.assemble(hit)
    acc_hit, trace_hit = m.run_c_basic(
        hit_vec,
        init_acc=m.data_atoms["enemy_near"],
        max_steps=len(hit),
    )
    assert trace_hit == [("IFMATCH", "enemy_near"), ("LOAD", "flee_signal")]
    assert cosine(acc_hit, m.data_atoms["flee_signal"]) > 0.999999

    acc_miss, trace_miss = m.run_c_basic(
        hit_vec,
        init_acc=m.data_atoms["calm"],
        max_steps=len(hit),
    )
    assert trace_miss == [("IFMATCH", "enemy_near")]
    assert cosine(acc_miss, m.data_atoms["calm"]) > 0.999999


def test_c_program_runner_rejects_host_bound_ops():
    m = HoloMachine(dim=2048, seed=7)
    m.define("tag_b", [("BIND", "b"), ("HALT", "")])
    program_vec = m.assemble([("LOAD", "a"), ("CALL", "tag_b"), ("HALT", "")])
    with pytest.raises(RuntimeError):
        m.run_c_basic(program_vec, max_steps=3)
