"""Tests for holographic_command (R4: allowlisted command runner + orchestrator Tool bridge)."""
from holographic.scene_and_pipeline.holographic_command import CommandRunner, CommandError, command_as_tool
from holographic.scene_and_pipeline.holographic_orchestrator import CircuitBreaker
from holographic.agents_and_reasoning.holographic_ai import Vocabulary


def _runner():
    r = CommandRunner(timeout=10)
    r.register("echo", ["echo", "{input}"], doc="echo back")
    r.register("upper", ["python3", "-c", "import sys;print(sys.argv[1].upper())", "{input}"], doc="uppercase")
    return r


def test_allowlist_gate():
    r = _runner()
    try:
        r.run("rm", {"input": "-rf /"}); assert False
    except CommandError:
        pass
    assert set(r.registered()) == {"echo", "upper"}


def test_no_shell_injection():
    r = _runner()
    out = r.run("echo", {"input": "hello; rm -rf /"})
    assert out["ok"] and out["stdout"].strip() == "hello; rm -rf /"     # ';' not interpreted -- no shell


def test_command_runs_and_captures():
    r = _runner()
    up = r.run("upper", {"input": "distributed"})
    assert up["ok"] and up["stdout"].strip() == "DISTRIBUTED" and up["returncode"] == 0


def test_missing_placeholder_fails():
    r = _runner()
    try:
        r.run("echo", {}); assert False
    except CommandError:
        pass


def test_unknown_executable_refused_at_register():
    r = CommandRunner()
    try:
        r.register("nope", ["definitely_not_a_real_program_xyz", "{input}"]); assert False
    except CommandError:
        pass


def test_timeout():
    r = CommandRunner(timeout=1)
    r.register("sleep", ["python3", "-c", "import time;time.sleep(5)"])
    try:
        r.run("sleep"); assert False
    except CommandError:
        pass


def test_as_orchestrator_tool_and_circuit_breaker():
    r = _runner()
    vocab = Vocabulary(1024, 0)
    tool = command_as_tool(r, "upper", "text", "text", ["uppercase", "text"], vocab)
    assert tool.fn("go") == "GO\n"

    r.register("fail", ["python3", "-c", "import sys;sys.exit(1)"])
    ftool = command_as_tool(r, "fail", "any", "any", ["broken"], vocab)
    breaker = CircuitBreaker(fail_max=2, cooldown=5)
    for _ in range(2):
        try:
            ftool.fn("x")
        except CommandError:
            breaker.report(ftool, ok=False)
    assert not breaker.available(ftool)
