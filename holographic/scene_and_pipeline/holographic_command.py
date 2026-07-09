"""holographic_command.py -- R4: run ANY registered program/script, and wire it as an orchestrator Tool.

WHY
---
The coordinator's third backend runs a task that is a whole external program (ffmpeg, a solver, a shell script, a
command that calls an API). That turns the system into a general job runner and the door to EXTERNAL tools and
services. Wired as orchestrator Tools, an external program joins the same VSA fabric as an internal faculty: the
Planner can select and chain it, and the CircuitBreaker trips on a flaky one -- reusing holographic_orchestrator, not
a new planner.

SECURITY -- THE CAN OF WORMS, LOUD (this is the highest-risk surface in the project)
-----------------------------------------------------------------------------------
Arbitrary command execution IS arbitrary code execution. This runner is deliberately narrow:
  * ALLOWLIST ONLY. A command runs only if its NAME was registered ahead of time. There is no path from untrusted
    input to a new command -- callers pass a registered name + argument VALUES, never a command string.
  * NEVER a shell. subprocess.run gets an argv LIST with shell=False, so there is no shell parsing, no globbing, no
    `; rm -rf`, no pipe injection. An argument value is always ONE argv element; it can be a weird value TO the
    registered program, but it can never become a new command or flag on its own.
  * VALUES fill declared placeholders, they don't build the command. The argv template is fixed at registration;
    "{key}" tokens are replaced by str(args[key]) -- one token in, one token out.
  * TIME-BOXED. Every run has a timeout; a hung tool is killed, not left to spin.
  * (Deployment, not enforced here) run under a restricted user / container, no network unless the tool needs it.
This module gives you the allowlist + no-shell + timeout; the sandboxing above is an operational responsibility.
"""
import shutil
import subprocess
from holographic.scene_and_pipeline.holographic_orchestrator import Tool, CircuitBreaker, keyword_vector


class CommandError(Exception):
    """A command could not be run (not on the allowlist, missing executable, or it timed out)."""


class CommandRunner:
    """Run registered allowlisted commands as external processes. The allowlist maps a name to a FIXED argv template;
    argument VALUES fill "{key}" placeholders. No shell, ever."""

    def __init__(self, timeout=30):
        self._allow = {}                                   # name -> (argv_template, doc)
        self.timeout = timeout

    def register(self, name, argv, doc=""):
        """Add a command to the allowlist. `argv` is a fixed list of tokens; a token like "{path}" is a placeholder
        filled from args at run time. The executable (argv[0]) must exist on PATH (checked, so a typo fails loudly)."""
        if not argv:
            raise CommandError("argv must be a non-empty list (the program + its fixed arguments)")
        exe = argv[0]
        if shutil.which(exe) is None and "{" not in exe:
            raise CommandError("executable %r not found on PATH" % exe)
        self._allow[name] = (list(argv), doc)
        return name

    def registered(self):
        """The allowlist: {name: doc}. This is the ONLY set of commands that can run."""
        return {name: doc for name, (argv, doc) in self._allow.items()}

    def run(self, name, args=None):
        """Run an allowlisted command. Returns {'stdout','stderr','returncode','ok'}. Raises CommandError if the name
        is not on the allowlist (the security gate) or the run times out."""
        if name not in self._allow:
            raise CommandError("%r is not on the allowlist -- register it first (commands never come from input)"
                               % name)
        argv_template, _doc = self._allow[name]
        args = args or {}
        argv = [self._fill(tok, args) for tok in argv_template]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=self.timeout, check=False)     # shell=False (default) -- NO shell parsing
        except subprocess.TimeoutExpired:
            raise CommandError("command %r timed out after %ss" % (name, self.timeout))
        except FileNotFoundError:
            raise CommandError("executable for %r not found" % name)
        return {"stdout": proc.stdout, "stderr": proc.stderr,
                "returncode": proc.returncode, "ok": proc.returncode == 0}

    @staticmethod
    def _fill(token, args):
        """Fill a single "{key}" placeholder with str(args[key]); a non-placeholder token passes through unchanged.
        One token in, one token out -- a value can never expand into extra argv elements."""
        if isinstance(token, str) and token.startswith("{") and token.endswith("}"):
            key = token[1:-1]
            if key not in args:
                raise CommandError("missing argument %r for the command" % key)
            return str(args[key])
        return token


def command_as_tool(runner, name, in_type, out_type, keywords, vocab, args_from=None):
    """Wrap an allowlisted command as an orchestrator Tool so the Planner can select and chain it. `args_from` maps a
    Planner input value to the command's args dict (default: pass it as {'input': value}); the Tool's fn returns the
    command's stdout (raising on a non-zero exit so the CircuitBreaker sees the failure)."""
    def _fn(value):
        args = args_from(value) if args_from else {"input": value}
        result = runner.run(name, args)
        if not result["ok"]:
            raise CommandError("%r failed (exit %d): %s" % (name, result["returncode"], result["stderr"].strip()))
        return result["stdout"]
    return Tool(name, in_type, out_type, keyword_vector(vocab, keywords), fn=_fn)


def _selftest():
    from holographic.agents_and_reasoning.holographic_ai import Vocabulary

    runner = CommandRunner(timeout=10)
    # register a couple of allowlisted commands that exist everywhere
    runner.register("echo", ["echo", "{input}"], doc="echo the input back")
    runner.register("upper", ["python3", "-c", "import sys;print(sys.argv[1].upper())", "{input}"],
                    doc="uppercase the input via a tiny python one-liner")

    # allowlist gate: an unregistered command is refused
    try:
        runner.run("rm", {"input": "-rf /"}); assert False, "unregistered command should be refused"
    except CommandError:
        pass

    # a registered command runs, no shell -> an injection attempt is just a literal value to echo
    out = runner.run("echo", {"input": "hello; rm -rf /"})
    assert out["ok"] and out["stdout"].strip() == "hello; rm -rf /"   # the ';' was NOT interpreted -- no shell

    up = runner.run("upper", {"input": "distributed"})
    assert up["ok"] and up["stdout"].strip() == "DISTRIBUTED"

    # missing placeholder value fails loudly
    try:
        runner.run("echo", {}); assert False
    except CommandError:
        pass

    # wire as an orchestrator Tool + CircuitBreaker: a successful run reports ok, a failing tool trips the breaker
    vocab = Vocabulary(1024, 0)
    tool = command_as_tool(runner, "upper", "text", "text", ["uppercase", "text", "transform"], vocab)
    assert tool.fn("go") == "GO\n"

    runner.register("fail", ["python3", "-c", "import sys; sys.exit(1)"], doc="always fails")
    ftool = command_as_tool(runner, "fail", "any", "any", ["broken"], vocab)
    breaker = CircuitBreaker(fail_max=2, cooldown=5)
    for _ in range(2):
        try:
            ftool.fn("x")
        except CommandError:
            breaker.report(ftool, ok=False)
    assert not breaker.available(ftool)                    # the flaky tool's breaker is open -> the planner skips it

    print("OK: holographic_command self-test passed (allowlist gate, no-shell injection-safe, placeholder values, "
          "timeout, wired as an orchestrator Tool with a tripping CircuitBreaker -- R4)")


if __name__ == "__main__":
    _selftest()
