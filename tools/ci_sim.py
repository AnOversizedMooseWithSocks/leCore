"""ci_sim.py -- SIMULATE THE AFFECTED-TESTS LEG OF CI INSIDE THIS SANDBOX (cp27).

pytest is not installed here and the network is disabled, so this runner provides a
MINIMAL pytest stand-in -- fixture injection for tmp_path/monkeypatch, pytest.raises/
approx/skip/mark -- and runs each affected test file in-process. HONEST LABELS: a test
needing machinery the shim lacks is SKIPPED AND COUNTED, never silently passed. This
simulates CI; it does not replace the real sharded pytest run, and says so.
"""
import importlib.util, inspect, os, sys, tempfile, traceback, types, shutil, pathlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _install_pytest_stub():
    pt = types.ModuleType("pytest")

    class _Raises:
        def __init__(self, exc, match=None):
            self.exc, self.match = exc, match
        def __enter__(self):
            return self
        def __exit__(self, et, ev, tb):
            import re
            if et is None:
                raise AssertionError("did not raise %s" % self.exc)
            ok = issubclass(et, self.exc)
            if ok and self.match and not re.search(self.match, str(ev)):
                raise AssertionError("raised but message %r !~ %r" % (str(ev), self.match))
            self.excinfo = ev
            return ok

    class _Skip(Exception):
        pass

    def _approx(x, rel=1e-6, abs=1e-12):
        class A:
            def __eq__(self, other):
                import numpy as _np
                return bool(_np.allclose(other, x, rtol=rel, atol=abs))
        return A()

    class _MarkDecor:
        def __getattr__(self, name):
            def deco(*a, **k):
                if a and callable(a[0]) and not k:
                    a[0]._ci_sim_mark = name
                    return a[0]
                def inner(f):
                    f._ci_sim_mark = name
                    return f
                return inner
            return deco

    class _Warns:
        """pytest.warns, which the shim lacked (cp52): a real test asserting a WARNING
        failed with AttributeError and looked like a code defect. Catches warnings, and
        on exit asserts at least one of the expected category was raised."""
        def __init__(self, expected=Warning, match=None):
            self.expected = expected
            self.match = match
            self.list = []

        def __enter__(self):
            import warnings
            self._cm = warnings.catch_warnings(record=True)
            self.list = self._cm.__enter__()
            warnings.simplefilter("always")
            return self

        def __exit__(self, et, ev, tb):
            import re as _re
            self._cm.__exit__(et, ev, tb)
            if et is not None:
                return False
            got = [w for w in self.list if issubclass(w.category, self.expected)]
            assert got, "expected a %s warning, got %d warning(s)" % (
                getattr(self.expected, "__name__", self.expected), len(self.list))
            if self.match:
                assert any(_re.search(self.match, str(w.message)) for w in got), \
                    "no %s warning matched %r" % (
                        getattr(self.expected, "__name__", ""), self.match)
            return False

    pt.warns = _Warns
    pt.raises = _Raises
    pt.approx = _approx
    pt.mark = _MarkDecor()
    pt.skip = lambda reason="": (_ for _ in ()).throw(_Skip(reason))
    pt.xfail = pt.skip
    pt.fixture = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    pt.importorskip = lambda name, **k: (importlib.import_module(name)
                                         if importlib.util.find_spec(name)
                                         else (_ for _ in ()).throw(_Skip(name)))
    pt._Skip = _Skip
    sys.modules["pytest"] = pt
    return pt


class _Monkeypatch:
    def __init__(self):
        self._undo = []
    def setattr(self, obj, name, value):
        old = getattr(obj, name)
        self._undo.append(lambda: setattr(obj, name, old))
        setattr(obj, name, value)
    def setenv(self, k, v):
        old = os.environ.get(k)
        self._undo.append(lambda: (os.environ.pop(k, None) if old is None
                                   else os.environ.__setitem__(k, old)))
        os.environ[k] = str(v)
    def undo(self):
        for u in reversed(self._undo):
            u()


def run_file(path, pt):
    spec = importlib.util.spec_from_file_location(
        "citest_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except pt._Skip:
        return {"file": path, "collected": 0, "passed": 0, "failed": [],
                "skipped_module": True}
    except Exception:
        return {"file": path, "collect_error": traceback.format_exc(limit=3)[-400:]}
    fixtures_known = {"tmp_path", "monkeypatch"}
    passed, failed, skipped = 0, [], 0
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        if getattr(fn, "_ci_sim_mark", "") == "slow":
            skipped += 1
            continue
        params = list(inspect.signature(fn).parameters)
        if any(p_ not in fixtures_known for p_ in params):
            skipped += 1                                  # fixture the shim lacks: COUNTED
            continue
        kw = {}
        td = None
        mp = None
        if "tmp_path" in params:
            td = tempfile.mkdtemp(prefix="ci_sim_")
            kw["tmp_path"] = pathlib.Path(td)
        if "monkeypatch" in params:
            mp = _Monkeypatch()
            kw["monkeypatch"] = mp
        try:
            fn(**kw)
            passed += 1
        except pt._Skip:
            skipped += 1
        except Exception:
            failed.append({"test": name,
                           "err": traceback.format_exc(limit=4)[-500:]})
        finally:
            if mp:
                mp.undo()
            if td:
                shutil.rmtree(td, ignore_errors=True)
    return {"file": path, "passed": passed, "failed": failed, "skipped": skipped}


def main(patterns):
    pt = _install_pytest_stub()
    os.environ.setdefault("MPLBACKEND", "Agg")
    files = sorted(f for f in os.listdir(os.path.join(ROOT, "tests"))
                   if f.startswith("test_") and f.endswith(".py")
                   and any(p_ in f for p_ in patterns))
    tot_p, tot_f, tot_s, errs = 0, [], 0, []
    for f in files:
        r = run_file(os.path.join(ROOT, "tests", f), pt)
        if "collect_error" in r:
            errs.append((f, r["collect_error"]))
            continue
        tot_p += r.get("passed", 0)
        tot_s += r.get("skipped", 0) + (1 if r.get("skipped_module") else 0)
        for x in r.get("failed", []):
            tot_f.append((f, x["test"], x["err"]))
    print("CI-SIM (shim, NOT real pytest): files=%d passed=%d failed=%d "
          "skipped(fixture/slow)=%d collect_errors=%d"
          % (len(files), tot_p, len(tot_f), tot_s, len(errs)))
    for f, t, e in tot_f[:12]:
        print("FAIL %s::%s\n  %s" % (f, t, e.strip().splitlines()[-1][:160]))
    for f, e in errs[:6]:
        print("COLLECT-ERROR %s\n  %s" % (f, e.strip().splitlines()[-1][:160]))
    return len(tot_f) + len(errs)


if __name__ == "__main__":
    pats = sys.argv[1:] or ["zoo", "mcp", "lever7", "ladder", "catalog"]
    sys.exit(1 if main(pats) else 0)
