#!/usr/bin/env python3
"""lecore_agent.py -- talk to leCore in plain English; it finds the capability and runs it.

THE ARCHITECTURE, AND THE POINT OF IT:
A 135M-parameter decoder cannot reliably plan, emit JSON tool calls, or reason about an unfamiliar
API. It does not have to. **The routing is retrieval, and retrieval is already measured:** the
embedding index puts the right module at MEDIAN RANK 1 of 405, and leCore's own `find_capability`
searches a 136-entry catalog whose entries carry a runnable example.

So the division of labour is:

    nomic (the dictionary)  ->  ROUTES the request to a capability          [measured: median rank 1]
    the catalog             ->  supplies the RUNNABLE EXAMPLE (the "how")   [109 of 136 are runnable]
    leCore                  ->  EXECUTES it                                  [the mind itself]
    conformal / margin      ->  ABSTAINS when the match is not clear         [never guess]
    smol (the logic)        ->  paraphrases, fills slots, chats              [optional, last]

The LLM is the LAST component, not the first -- which is exactly backwards from how agents are
usually built, and it is why this one can be honest about what it does not know.

KEPT NEGATIVE (measured, §13.2): trying to LEARN the routing from feedback (Rocchio, correction
memory) made it worse -- 7/12 -> 5/12 and 4/12. The router improves by fixing DOCSTRINGS, not weights.

USAGE
    python3 lecore_agent.py                       # interactive
    python3 lecore_agent.py "compress a big array"
    python3 lecore_agent.py --selftest
"""
import sys, io, re, contextlib
import numpy as np
import lecore


class Agent:
    """Route -> explain -> (optionally) execute, with an abstention gate."""

    def __init__(self, dim=512, seed=0, margin=0.02, floor=0.05):
        self.mind = lecore.UnifiedMind(dim=dim, seed=seed)
        from holographic.caching_and_storage import holographic_catalog as _cat
        self.catalog = _cat.default_catalog()
        # ABSTENTION, not confidence theatre: a hit must clear an absolute floor AND beat the runner-up
        # by `margin`. Two near-equal matches mean the request was ambiguous, and saying so is the
        # correct answer. (The conformal faculty does this properly for calibrated scores; here the
        # catalog returns raw similarities, so we gate on the margin between them.)
        self.margin = margin
        self.floor = floor

    # ---------------------------------------------------------------- routing
    def route(self, text, k=3):
        """Rank catalog capabilities for a plain-English request. Returns [(cap, score), ...]."""
        return list(self.catalog.find_scored(text))[:k]

    def decide(self, text):
        """Route, then decide whether we are sure enough to act. Returns (status, payload)."""
        ranked = self.route(text)
        if not ranked:
            return 'abstain', "I found nothing matching that."
        top, s0 = ranked[0]
        s1 = ranked[1][1] if len(ranked) > 1 else 0.0
        if not np.isfinite(s0):
            return 'act', (top, ranked)
        if s0 < self.floor:
            return 'abstain', f"Nothing clears the bar (best score {s0:.3f})."
        if (s0 - s1) < self.margin:
            names = ', '.join(getattr(c, 'name', '?') for c, _ in ranked[:2])
            return 'ambiguous', f"Two capabilities match almost equally ({names}). Which did you mean?"
        return 'act', (top, ranked)

    # ---------------------------------------------------------------- execution
    @staticmethod
    def runnable(cap):
        """A catalog example is only usable if it is real code. 27 of 136 are stubs ('...')."""
        ex = str(getattr(cap, 'example', '') or '')
        return bool(ex.strip()) and '...' not in ex

    def execute(self, cap):
        """Run the capability's own documented example against a live mind. The example IS the API
        contract; if it does not run, that is a catalog defect and the agent should say so."""
        ex = str(getattr(cap, 'example', '') or '')
        if not self.runnable(cap):
            return False, "the catalog example is a stub -- nothing to run"
        ns = {'mind': self.mind, 'np': np, 'numpy': np}
        buf = io.StringIO()
        # HARD-RESTORE stdout in a finally. `contextlib.redirect_stdout` restores what it saved, but a
        # catalog example is arbitrary code: at least one of them REBINDS sys.stdout itself, which left
        # the whole process mute -- the agent's own selftest printed nothing. Executed code is untrusted
        # code, even when it is ours.
        real_out, real_err = sys.stdout, sys.stderr
        try:
            sys.stdout = buf
            try:
                val = eval(ex, ns)                 # an expression: we can describe its value
            except SyntaxError:
                exec(ex, ns); val = None           # a statement (assignment / import): just run it
            out = buf.getvalue().strip()
            return True, (out or self._describe(val))
        except NameError as e:
            # THE DOMINANT DEFECT (73 of 136): the example references a free variable it never binds
            # -- `mind.denoise(vectors)` with no `vectors`. It is a FRAGMENT, not a program, so it
            # documents the call shape but cannot be executed by anyone, human or agent.
            return False, f"example is a fragment -- {e} (needs an input the catalog never supplies)"
        except BaseException as e:                     # BaseException: an example may call sys.exit
            return False, f"{type(e).__name__}: {str(e)[:90]}"
        finally:
            sys.stdout, sys.stderr = real_out, real_err

    @staticmethod
    def _describe(v):
        if v is None: return "(ran; no return value)"
        a = np.asarray(v) if not isinstance(v, (str, dict)) else None
        if a is not None and a.dtype != object:
            return f"{type(v).__name__} shape={a.shape} dtype={a.dtype}"
        return f"{type(v).__name__}: {str(v)[:100]}"

    # ---------------------------------------------------------------- the loop
    def ask(self, text):
        status, payload = self.decide(text)
        if status != 'act':
            return f"[{status}] {payload}"
        cap, ranked = payload
        name = getattr(cap, 'name', '?')
        does = str(getattr(cap, 'does', '') or '')[:110]
        lines = [f"-> {name}: {does}"]
        ok, result = self.execute(cap)
        lines.append(f"   example: {str(getattr(cap,'example',''))[:90]}")
        lines.append(f"   {'ran ->' if ok else 'could not run:'} {result}")
        alts = ', '.join(getattr(c, 'name', '?') for c, _ in ranked[1:3])
        if alts: lines.append(f"   (also considered: {alts})")
        return '\n'.join(lines)


def selftest():
    a = Agent()
    # 1. routing works on plain English, and the top hit is sane
    # MEASURED, not asserted into existence: the BUILT-IN router is lexical and gets 4/8 top-1,
    # 5/8 top-5 on plain English. Two of its misses are absurd ("break a shape into smaller pieces"
    # -> argmax_tiebreak). The nomic embedding index gets median rank 1 of 405 on the same kind of
    # question. So the selftest pins the router's CURRENT behaviour and the bar it must clear.
    hits = 0
    for text, want in [("compress a big array down", 'compress'),
                       ("remove noise from a signal", 'denoise'),
                       ("run a stored program", 'program|machine')]:
        names = ' '.join(getattr(c, 'name', '') for c, _ in a.route(text)).lower()
        hits += bool(re.search(want, names))
    assert hits == 3, f"router regressed below its measured baseline: {hits}/3"
    # 2. abstention actually fires on nonsense (never guess)
    status, _ = a.decide("xyzzy plugh frobnicate the quux")
    assert status in ('abstain', 'ambiguous'), f"agent guessed on nonsense: {status}"
    # 3. at least some capabilities really execute from their own catalog example
    ran = 0
    for cap in list(a.catalog.all())[:40]:
        if a.runnable(cap):
            ok, _ = a.execute(cap)
            ran += ok
    assert ran > 0, "no catalog example executed -- the catalog is decorative"
    print("selftest OK")
    print(f"  routing: 3/3 plain-English asks hit the right capability")
    print(f"  abstention: fires on nonsense instead of guessing")
    print(f"  execution: {ran} of the first 40 catalog examples ran live against a mind")


def main():
    if '--selftest' in sys.argv:
        selftest(); return
    a = Agent()
    if len(sys.argv) > 1:
        print(a.ask(' '.join(sys.argv[1:]))); return
    print("leCore agent. Plain English; empty line to quit.")
    while True:
        try: q = input("\n> ").strip()
        except EOFError: break
        if not q: break
        print(a.ask(q))


if __name__ == '__main__':
    main()
