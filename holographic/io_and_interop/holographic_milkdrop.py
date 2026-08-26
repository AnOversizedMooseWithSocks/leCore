"""Milkdrop `.milk` preset READER + a safe expression evaluator (holographic_milkdrop).

WHY THIS MODULE EXISTS
----------------------
A Milkdrop / projectM preset is not a shader blob -- it is a text file of MATH EQUATIONS. An INI-style header of
`key=value` settings, then per-frame equations (`per_frame_1=...`), per-vertex/per-pixel warp equations
(`per_pixel_1=...`), an init block, and (in newer presets) HLSL warp/comp shaders. The equations are written in
`ns-eel2`, a small C-like expression language (variables, `+ - * / %`, comparisons, `?:` via `if(c,a,b)`, and a
fixed function set: sin/cos/abs/pow/sqrt/min/max/...). Those equations, driven by audio features (bass/mid/treb from
an FFT), are what actually move a preset frame to frame.

leCore already has the audio side (`audio_param_bus` -> bass/mid/treb envelopes) and the feedback/warp machinery. What
was missing is the bridge: READ a `.milk` file into a structured, EVALUABLE form and run its per-frame equations
deterministically. That is what this module does -- the equation layer, not the GPU pixel pipeline (that stays the
renderer's job). Literal 41k-preset compatibility would need the warp-mesh + HLSL passes too; this is the honest,
bounded first layer: parse + evaluate the math.

SECURITY (hard constraint): the engine NEVER uses Python's `eval()` on preset text. This module ships a real
recursive-descent tokenizer + parser + evaluator over a whitelisted grammar, so a hostile preset can do arithmetic
and nothing else -- no attribute access, no imports, no calls outside the function whitelist.

WHAT IT PROVIDES
  * eval_expr(src, vars) -- evaluate ONE ns-eel2 expression string against a variable dict. Safe, deterministic.
  * MilkExpr(src) -- a compiled, reusable expression (parse once, evaluate many).
  * parse_milk(text) -- parse a `.milk` file's text into a MilkPreset (settings + init/frame/pixel equation lists).
  * MilkPreset.run_frame(state, audio) -- evaluate the per-frame equations once, updating and returning the state.

KEPT NEGATIVES (loud)
  * This is the EQUATION layer only. The per-pixel warp mesh and the HLSL warp/comp shaders are parsed and stored but
    NOT executed here -- running them is the renderer's job (the feedback/warp machinery already exists to wire them
    to). A preset's motion driven by per_frame equations DOES run; its pixel-shader look does not, yet.
  * ns-eel2 has assignment-as-expression and `;`-separated statements per equation line; we support that. It also has
    a `megabuf`/`gmegabuf` memory and a few esoteric ops we do NOT implement (they are rare in real presets) -- an
    unknown function raises rather than silently returning 0, so a caller knows the preset used something unsupported.
"""

import math

import numpy as np


# =====================================================================================================
# The safe expression evaluator: tokenizer -> recursive-descent parser -> AST -> evaluate. No eval().
# =====================================================================================================

# The ns-eel2 function whitelist. A preset may call ONLY these; anything else raises (never silently 0), so an
# unsupported preset is loud rather than subtly wrong. All are pure and deterministic.
_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "abs": abs, "sqrt": lambda x: math.sqrt(x) if x >= 0 else 0.0, "sqr": lambda x: x * x,
    "exp": math.exp, "log": lambda x: math.log(x) if x > 0 else 0.0,
    "log10": lambda x: math.log10(x) if x > 0 else 0.0,
    "pow": lambda a, b: math.pow(a, b) if not (a < 0 and not float(b).is_integer()) else 0.0,
    "floor": math.floor, "ceil": math.ceil, "int": lambda x: float(int(x)), "sign": lambda x: (x > 0) - (x < 0),
    "min": min, "max": max, "sigmoid": lambda x, y: 1.0 / (1.0 + math.exp(-x * y)) if -x * y < 700 else 0.0,
    "rand": lambda x: 0.0,                       # DETERMINISM: rand() is pinned to 0 (a preset RNG would break repro)
    "bnot": lambda x: 1.0 if x == 0.0 else 0.0,
    "if": lambda c, a, b: a if c != 0.0 else b,  # ns-eel2's ternary; both branches are pre-evaluated (eager), as EEL
    "band": lambda a, b: 1.0 if (a != 0.0 and b != 0.0) else 0.0,
    "bor": lambda a, b: 1.0 if (a != 0.0 or b != 0.0) else 0.0,
    "equal": lambda a, b: 1.0 if a == b else 0.0, "above": lambda a, b: 1.0 if a > b else 0.0,
    "below": lambda a, b: 1.0 if a < b else 0.0,
    "fmod": lambda a, b: math.fmod(a, b) if b != 0.0 else 0.0,
}


class _Tok:
    """A token: (kind, value). kinds: num, name, op, lparen, rparen, comma, semi, assign, end."""
    __slots__ = ("kind", "val")

    def __init__(self, kind, val):
        self.kind = kind
        self.val = val


def _tokenize(src):
    """Turn an ns-eel2 expression string into a token list. Whitelisted characters only; anything unexpected raises."""
    toks = []
    i, n = 0, len(src)
    ops = "+-*/%^<>=!&|"
    while i < n:
        ch = src[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and (src[j].isdigit() or src[j] in ".eE" or (src[j] in "+-" and j > i and src[j - 1] in "eE")):
                j += 1
            toks.append(_Tok("num", float(src[i:j])))
            i = j
        elif ch.isalpha() or ch == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            toks.append(_Tok("name", src[i:j]))
            i = j
        elif ch == "(":
            toks.append(_Tok("lparen", ch)); i += 1
        elif ch == ")":
            toks.append(_Tok("rparen", ch)); i += 1
        elif ch == ",":
            toks.append(_Tok("comma", ch)); i += 1
        elif ch == ";":
            toks.append(_Tok("semi", ch)); i += 1
        elif ch in ops:
            # two-char operators (==, <=, >=, !=, &&, ||); else single.
            if i + 1 < n and src[i:i + 2] in ("==", "<=", ">=", "!=", "&&", "||"):
                toks.append(_Tok("op", src[i:i + 2])); i += 2
            elif ch == "=":
                toks.append(_Tok("assign", ch)); i += 1
            else:
                toks.append(_Tok("op", ch)); i += 1
        else:
            raise ValueError("milkdrop expr: illegal character %r in %r" % (ch, src))
    toks.append(_Tok("end", None))
    return toks


# AST nodes are plain tuples: ("num", v) / ("var", name) / ("assign", name, expr) / ("bin", op, a, b) /
# ("unary", op, a) / ("call", name, [args]) / ("seq", [stmts]).

class _Parser:
    """Recursive-descent parser for the ns-eel2 subset. Precedence: || && == etc + - * / % unary ^ ."""

    def __init__(self, toks):
        self.toks = toks
        self.pos = 0

    def _peek(self):
        return self.toks[self.pos]

    def _next(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def parse(self):
        # a line is `;`-separated statements; the value is the last statement's value.
        stmts = [self._statement()]
        while self._peek().kind == "semi":
            self._next()
            if self._peek().kind == "end":
                break
            stmts.append(self._statement())
        if self._peek().kind != "end":
            raise ValueError("milkdrop expr: trailing tokens")
        return ("seq", stmts) if len(stmts) > 1 else stmts[0]

    def _statement(self):
        # assignment: NAME = expr  (ns-eel2 assignment is also an expression, but line-level is the common case)
        if self._peek().kind == "name" and self.toks[self.pos + 1].kind == "assign":
            name = self._next().val
            self._next()                                   # consume '='
            return ("assign", name, self._expr())
        return self._expr()

    def _expr(self):
        return self._logic_or()

    def _logic_or(self):
        a = self._logic_and()
        while self._peek().kind == "op" and self._peek().val == "||":
            self._next(); a = ("bin", "||", a, self._logic_and())
        return a

    def _logic_and(self):
        a = self._compare()
        while self._peek().kind == "op" and self._peek().val == "&&":
            self._next(); a = ("bin", "&&", a, self._compare())
        return a

    def _compare(self):
        a = self._add()
        while self._peek().kind == "op" and self._peek().val in ("==", "!=", "<", ">", "<=", ">="):
            op = self._next().val
            a = ("bin", op, a, self._add())
        return a

    def _add(self):
        a = self._mul()
        while self._peek().kind == "op" and self._peek().val in ("+", "-"):
            op = self._next().val
            a = ("bin", op, a, self._mul())
        return a

    def _mul(self):
        a = self._unary()
        while self._peek().kind == "op" and self._peek().val in ("*", "/", "%"):
            op = self._next().val
            a = ("bin", op, a, self._unary())
        return a

    def _unary(self):
        if self._peek().kind == "op" and self._peek().val in ("-", "+", "!"):
            op = self._next().val
            return ("unary", op, self._unary())
        return self._power()

    def _power(self):
        a = self._atom()
        if self._peek().kind == "op" and self._peek().val == "^":
            self._next()
            return ("bin", "^", a, self._unary())          # right-assoc power
        return a

    def _atom(self):
        t = self._peek()
        if t.kind == "num":
            self._next(); return ("num", t.val)
        if t.kind == "lparen":
            self._next(); e = self._expr()
            if self._next().kind != "rparen":
                raise ValueError("milkdrop expr: missing )")
            return e
        if t.kind == "name":
            name = self._next().val
            if self._peek().kind == "lparen":             # a function call
                self._next()
                args = []
                if self._peek().kind != "rparen":
                    args.append(self._expr())
                    while self._peek().kind == "comma":
                        self._next(); args.append(self._expr())
                if self._next().kind != "rparen":
                    raise ValueError("milkdrop expr: missing ) in call to %s" % name)
                return ("call", name, args)
            return ("var", name)
        raise ValueError("milkdrop expr: unexpected token %r" % (t.val,))


def _eval_node(node, env):
    """Evaluate a parsed AST node against `env` (a dict of variable values). Pure; mutates env only on assignment."""
    k = node[0]
    if k == "num":
        return node[1]
    if k == "var":
        return float(env.get(node[1], 0.0))               # unknown variable reads as 0 (ns-eel2 semantics)
    if k == "assign":
        v = _eval_node(node[2], env)
        env[node[1]] = v
        return v
    if k == "seq":
        v = 0.0
        for s in node[1]:
            v = _eval_node(s, env)
        return v
    if k == "unary":
        v = _eval_node(node[2], env)
        if node[1] == "-":
            return -v
        if node[1] == "+":
            return v
        return 1.0 if v == 0.0 else 0.0                   # '!'
    if k == "bin":
        op = node[1]
        a = _eval_node(node[2], env)
        b = _eval_node(node[3], env)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b if b != 0.0 else 0.0             # ns-eel2: divide by zero is 0, not an exception
        if op == "%":
            return math.fmod(a, b) if b != 0.0 else 0.0
        if op == "^":
            return math.pow(a, b) if not (a < 0 and not float(b).is_integer()) else 0.0
        if op == "==":
            return 1.0 if a == b else 0.0
        if op == "!=":
            return 1.0 if a != b else 0.0
        if op == "<":
            return 1.0 if a < b else 0.0
        if op == ">":
            return 1.0 if a > b else 0.0
        if op == "<=":
            return 1.0 if a <= b else 0.0
        if op == ">=":
            return 1.0 if a >= b else 0.0
        if op == "&&":
            return 1.0 if (a != 0.0 and b != 0.0) else 0.0
        if op == "||":
            return 1.0 if (a != 0.0 or b != 0.0) else 0.0
        raise ValueError("milkdrop expr: unknown operator %r" % op)
    if k == "call":
        name = node[1]
        fn = _FUNCS.get(name)
        if fn is None:
            raise ValueError("milkdrop expr: unsupported function %r (not in the ns-eel2 whitelist)" % name)
        args = [_eval_node(a, env) for a in node[2]]
        return float(fn(*args))
    raise ValueError("milkdrop expr: bad node %r" % (k,))


class MilkExpr:
    """A compiled ns-eel2 expression: parse once, evaluate many. `MilkExpr(src).eval(vars)` returns a float and
    updates `vars` in place for any assignments in the expression."""

    def __init__(self, src):
        self.src = src
        self.ast = _Parser(_tokenize(src)).parse()

    def eval(self, env):
        return _eval_node(self.ast, env)


def eval_expr(src, vars=None):
    """Evaluate ONE ns-eel2 expression string `src` against the variable dict `vars` (unknown vars read as 0).
    Safe (no Python eval; a whitelisted grammar), deterministic. Returns a float; `vars` is updated in place for any
    assignments. The one-shot form of MilkExpr for a caller that does not need to reuse the compiled expression."""
    env = {} if vars is None else vars
    return MilkExpr(src).eval(env)


# =====================================================================================================
# The .milk preset parser: an INI-ish header + numbered equation families.
# =====================================================================================================

class MilkPreset:
    """A parsed Milkdrop preset: `settings` (a dict of the scalar key=value header), and the four equation families
    as ORDERED lists of compiled MilkExprs -- `init` (per-frame-init, run once), `frame` (per_frame_*, run each
    frame), `pixel` (per_pixel_*, the warp-mesh equations), and the raw `warp`/`comp` HLSL shader text (stored, not
    executed here). `run_frame` evaluates the per-frame family."""

    def __init__(self, settings, init, frame, pixel, warp_shader, comp_shader):
        self.settings = settings
        self.init = init                                  # list[MilkExpr]
        self.frame = frame                                # list[MilkExpr]
        self.pixel = pixel                                # list[MilkExpr]
        self.warp_shader = warp_shader                    # raw HLSL text (not run here)
        self.comp_shader = comp_shader

    def initial_state(self):
        """The starting variable state: the numeric settings whose names look like preset variables (q1..q32, and the
        standard motion vars) plus the run-once `init` equations evaluated. Returns a fresh dict."""
        env = {}
        for kk, vv in self.settings.items():
            try:
                env[kk] = float(vv)
            except (TypeError, ValueError):
                pass
        for e in self.init:
            e.eval(env)
        return env

    def run_frame(self, state, audio=None, time=0.0, frame=0):
        """Evaluate the per_frame equations ONCE against `state`, injecting the standard driver variables first:
        `time`, `frame`, and the audio reactivity (`bass`, `mid`, `treb`, and their `_att` attenuated versions) from
        an `audio` dict (as produced by audio_param_bus, or any {bass,mid,treb}). Missing audio -> 1.0 (the Milkdrop
        default when there is no signal). Mutates and returns `state` -- the per-frame vars (q1..q32, zoom, rot, ...)
        are now updated for the renderer to read."""
        audio = audio or {}
        for band in ("bass", "mid", "treb"):
            state[band] = float(audio.get(band, 1.0))
            state[band + "_att"] = float(audio.get(band + "_att", state[band]))
        state["time"] = float(time)
        state["frame"] = float(frame)
        for e in self.frame:
            e.eval(state)
        return state


def parse_milk(text):
    """Parse the TEXT of a `.milk` preset into a MilkPreset. Recognises the INI-ish `key=value` settings and the
    numbered equation families per_frame_init_N / per_frame_N / per_pixel_N (also the older `wave_N` etc are read as
    settings), and captures the `warp`/`comp` HLSL shader blocks verbatim (stored, not executed). Equation numbering
    is preserved (equations run in ascending N, as Milkdrop does). A malformed equation raises with its line, so a
    bad preset is loud."""
    settings = {}
    init_eqs, frame_eqs, pixel_eqs = {}, {}, {}
    warp_lines, comp_lines = [], []
    section = None                                        # None | 'warp' | 'comp' for the shader blocks
    for raw in text.splitlines():
        line = raw.rstrip("\n").rstrip("\r")
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("["):
            # a `[preset00]` header or a comment; shaders can contain // so only skip at top level
            if section in ("warp", "comp") and not stripped.startswith("["):
                (warp_lines if section == "warp" else comp_lines).append(line)
            continue
        if "=" not in line:
            if section in ("warp", "comp"):
                (warp_lines if section == "warp" else comp_lines).append(line)
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        low = key.lower()
        if low.startswith("warp_"):                       # warp shader lines: warp_1=`...`
            warp_lines.append(val)
            section = "warp"
            continue
        if low.startswith("comp_"):
            comp_lines.append(val)
            section = "comp"
            continue
        section = None
        if low.startswith("per_frame_init_"):
            init_eqs[int(low.rsplit("_", 1)[1])] = val.strip()
        elif low.startswith("per_frame_"):
            frame_eqs[int(low.rsplit("_", 1)[1])] = val.strip()
        elif low.startswith("per_pixel_"):
            pixel_eqs[int(low.rsplit("_", 1)[1])] = val.strip()
        else:
            settings[key] = val.strip()

    def _compile(d):
        return [MilkExpr(d[n]) for n in sorted(d)]

    return MilkPreset(settings, _compile(init_eqs), _compile(frame_eqs), _compile(pixel_eqs),
                      "\n".join(warp_lines), "\n".join(comp_lines))


def _selftest():
    # --- expression evaluator: arithmetic, precedence, functions, assignment, safety ---
    assert abs(eval_expr("2 + 3 * 4") - 14.0) < 1e-12, "precedence"
    assert abs(eval_expr("(2 + 3) * 4") - 20.0) < 1e-12, "parens"
    assert abs(eval_expr("2 ^ 10") - 1024.0) < 1e-12, "power"
    assert abs(eval_expr("sqrt(sqr(3) + sqr(4))") - 5.0) < 1e-12, "nested calls"
    assert abs(eval_expr("if(above(5,3), 1, 2)") - 1.0) < 1e-12, "if + above"
    assert abs(eval_expr("1 / 0") - 0.0) < 1e-12, "divide by zero is 0 (ns-eel2 semantics)"
    env = {}
    assert abs(eval_expr("x = 3; y = x * 2; y + 1", env) - 7.0) < 1e-12, "assignment + sequence"
    assert env["x"] == 3.0 and env["y"] == 6.0, "assignments persist in the env"
    assert abs(eval_expr("bass * 2", {"bass": 1.5}) - 3.0) < 1e-12, "variable read"
    assert abs(eval_expr("undefined_var + 1") - 1.0) < 1e-12, "unknown var reads as 0"
    # SAFETY: an unsupported function raises (never silently 0), and illegal characters raise.
    try:
        eval_expr("__import__(1)")
        assert False, "must reject unknown function"
    except ValueError:
        pass
    try:
        eval_expr("x @ y")
        assert False, "must reject illegal character"
    except ValueError:
        pass

    # --- a minimal SYNTHETIC preset (written here, NOT copied from any real preset) ---
    text = (
        "[preset00]\n"
        "// a tiny synthetic preset for the self-test\n"
        "fRating=3\n"
        "zoom=1.0\n"
        "per_frame_init_1=q1 = 0\n"
        "per_frame_1=q1 = q1 + 1\n"
        "per_frame_2=zoom = 1.0 + 0.1 * sin(time) + 0.05 * bass\n"
        "per_pixel_1=rot = rot + 0.01\n"
        "warp_1=`shader text that is stored not run`\n"
    )
    preset = parse_milk(text)
    assert preset.settings["fRating"] == "3" and preset.settings["zoom"] == "1.0"
    assert len(preset.frame) == 2 and len(preset.init) == 1 and len(preset.pixel) == 1
    assert "stored not run" in preset.warp_shader, "the warp shader is captured verbatim"

    state = preset.initial_state()
    assert state["q1"] == 0.0, "per_frame_init set q1 to 0"
    # run three frames with a bass beat on frame 2; q1 counts frames, zoom reacts to time + bass.
    preset.run_frame(state, audio={"bass": 1.0}, time=0.0, frame=0)
    assert state["q1"] == 1.0, "per_frame_1 increments q1 each frame"
    preset.run_frame(state, audio={"bass": 1.0}, time=0.0, frame=1)
    assert state["q1"] == 2.0, "q1 counts frames deterministically"
    z_quiet = state["zoom"]
    preset.run_frame(state, audio={"bass": 3.0}, time=0.0, frame=2)   # a bass hit
    assert state["zoom"] > z_quiet, "zoom reacts to the bass envelope (audio-driven motion runs)"
    # determinism: same inputs -> same state
    s2 = preset.initial_state()
    preset.run_frame(s2, audio={"bass": 1.0}, time=0.0, frame=0)
    assert s2["q1"] == 1.0 and abs(s2["zoom"] - eval_expr("1.0 + 0.1*sin(0) + 0.05*1.0")) < 1e-9

    print("holographic_milkdrop selftest: ok (safe ns-eel2 evaluator: precedence/functions/assignment/sequence, "
          "rejects unknown funcs + illegal chars, divide-by-zero=0; parse_milk reads a synthetic preset's settings + "
          "per_frame_init/per_frame/per_pixel equation families + captures the warp shader; run_frame counts frames "
          "and drives zoom from the bass envelope, deterministically)")


if __name__ == "__main__":
    _selftest()
