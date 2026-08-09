"""Pins for tools/regen_docs.py -- the ONE DOOR for generated docs.

WHY THESE EXIST (a measured CI failure, kept loud). ci.yml drift-GATES a generated file: it regenerates the
file and `git diff --exit-code`s it, failing the build if the committed copy is stale. Two independent bugs
made that gate fire on builds where nothing was wrong, and both are pinned here:

  1. THE LIST WAS INVISIBLE. The set of doc generators existed only as copy-pasted steps in docs.yml, so a
     close-out ran the two everyone remembers (capdoc, docgen), skipped apiquickref, and CI went red. A gate
     on a file that no reachable list tells you to regenerate is a guaranteed red build. test_every_ci_drift
     _gated_file_has_a_generator is the pin: it reads ci.yml's ACTUAL gates and demands the list cover them.

  2. THE GATE MEASURED THE CALENDAR. apiquickref.py stamped `date.today()` into API_QUICKREF.md, so the
     regenerated copy differed from the committed one on any push made on a later day -- with zero code
     changed. capdoc.py, same gate and no timestamp, never false-failed: it was the control that proved the
     diagnosis. test_gated_generators_are_deterministic pins it -- a gated generator must be a PURE FUNCTION
     OF THE SOURCE, or the gate is a clock.
"""

import re
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import regen_docs  # noqa: E402
import docgen  # noqa: E402


def _ci_gated_files():
    """The files ci.yml actually drift-gates, read from the workflow -- not from a hardcoded copy that rots."""
    ci = ROOT / ".github" / "workflows" / "ci.yml"
    if not ci.exists():
        pytest.skip("ci.yml not present in this tree")
    gated = set()
    for m in re.finditer(r"git diff --exit-code ([^\n|&]+)", ci.read_text(errors="replace")):
        gated.update(p.strip() for p in m.group(1).split() if p.strip().endswith((".md", ".json")))
    return gated


def test_canonical_list_is_well_formed():
    """The table is non-empty and no two generators claim the same output (they would overwrite each other)."""
    assert regen_docs.GENERATORS
    outs = regen_docs.outputs()
    assert len(outs) == len(set(outs)), "two generators claim the same output file"


def test_every_ci_drift_gated_file_has_a_generator():
    """THE PIN. Every file ci.yml gates must be produced by a generator in the canonical list -- otherwise the
    gate is unfixable-by-construction and the build is red forever. This is the exact bug that shipped."""
    gated = _ci_gated_files()
    assert gated, "expected ci.yml to drift-gate at least one generated file"
    uncovered = gated - set(regen_docs.outputs())
    assert not uncovered, ("ci.yml drift-gates file(s) that tools/regen_docs.py never regenerates: %s -- add "
                           "the generator to regen_docs.GENERATORS or drop the gate" % sorted(uncovered))


def test_gated_generators_are_deterministic():
    """A gated generator must be a pure function of the source: run it twice, get the same bytes. Catches the
    date.today() class of bug BEFORE it turns a drift gate into a calendar check (measured: it did)."""
    gated = _ci_gated_files()
    gens = [(g, outs) for g, outs in regen_docs.GENERATORS if set(outs) & gated]
    assert gens, "no generator produces a gated file -- the pin above should have caught this"
    for gen, outs in gens:
        if not (ROOT / gen).exists():
            pytest.skip("%s not present in this tree" % gen)
        first = {}
        # TWO DIFFERENT HASH SEEDS, not two runs of the same one. The docstring above claims this catches
        # dict-order bugs -- it could not, while both passes inherited one PYTHONHASHSEED: an
        # order-dependent generator produces the SAME bytes twice under a fixed seed and sails through.
        # Varying the seed is what actually exercises the claim (str hashing, and therefore set/dict
        # iteration order, is salted per process). Verified by hand first: every gated output is currently
        # byte-identical across seeds, so this pins a property that already holds rather than announcing one.
        for seed in ("0", "1618033"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run([sys.executable, gen], cwd=str(ROOT), env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert proc.returncode == 0, "%s failed: %s" % (gen, proc.stdout.decode("utf-8", "replace")[-500:])
            for o in outs:
                data = (ROOT / o).read_bytes()
                if o in first:
                    assert data == first[o], ("%s is NON-DETERMINISTIC -- %s changed between two runs under "
                                              "DIFFERENT hash seeds with no source edit (a date/hostname stamp, "
                                              "or iteration over an unsorted set/dict). ci.yml gates this file, "
                                              "so that is a guaranteed red build -- and docs.yml would commit "
                                              "the churn." % (gen, o))
                first[o] = data


def test_reference_generation_is_stable_for_duplicate_basenames(tmp_path):
    """Two families can legitimately contain the same module basename. Their
    order must come from the full path, not os.walk or input order, or the docs
    bot and a developer on another filesystem will commit an endless reorder."""
    paths = []
    for family, summary in (("misc", "Misc implementation."), ("mesh_and_geometry", "Mesh implementation.")):
        path = tmp_path / "holographic" / family / "holographic_duplicate.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""%s"""\n' % summary, encoding="utf-8")
        paths.append(path)

    modules = [docgen.read_module(path) for path in paths]
    forward = tmp_path / "forward.md"
    reverse = tmp_path / "reverse.md"
    docgen.write_reference(modules, forward)
    docgen.write_reference(list(reversed(modules)), reverse)

    assert forward.read_bytes() == reverse.read_bytes()


def test_no_date_stamp_in_gated_outputs():
    """The specific regression, named. A gated doc must not carry a generation DATE -- that is what made the
    API_QUICKREF gate fail on every new calendar day (measured: `on 2026-07-09` -> `on 2026-07-16`)."""
    for o in sorted(_ci_gated_files()):
        path = ROOT / o
        if not path.exists():
            continue
        head = path.read_text(errors="replace")[:2000]
        assert not re.search(r"(generated|auto-generated).{0,40}\bon\s+\d{4}-\d{2}-\d{2}", head, re.I), (
            "%s carries a generation date stamp; ci.yml gates it, so the gate will fire on the next new day" % o)


def test_missing_generator_fails_loudly_rather_than_being_skipped():
    """A generator that is not on disk must be REPORTED, never silently passed over -- a reconciliation has
    dropped files in this repo before, and a silent skip turns that into a green run that regenerated nothing."""
    missing, _failed, _changed = regen_docs.run(check=False, root=str(ROOT / "tests" / "_no_such_dir"))
    assert set(missing) == {g for g, _ in regen_docs.GENERATORS}, "absent generators must all be reported"


def _ci_jobs(text):
    """Split ci.yml into {job_name: raw_text} using INDENTATION ONLY -- no yaml module.

    WHY NOT PyYAML: it is not in requirements.txt and must not be. The core promises NumPy/Flask/stdlib, CI
    installs exactly that, and a test that imports a fourth-party parser to inspect a config file quietly
    widens the dependency surface the whole project is built to avoid. It also fails on the runner, which is
    how this was caught. A job is a 2-space-indented `name:` line; everything more-indented below belongs to
    it. That is enough to answer both questions here, and it sidesteps the line-wrapping bug the yaml.dump
    version had as a bonus."""
    jobs, name, buf, in_jobs = {}, None, [], False
    for line in text.splitlines(True):
        if re.match(r'^jobs:\s*$', line):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if re.match(r'^\S', line):                                # back to top level -> jobs section ended
            break
        head = re.match(r'^  ([A-Za-z][\w-]*):\s*$', line)
        if head:
            if name:
                jobs[name] = "".join(buf)
            name, buf = head.group(1), []
            continue
        if name:
            buf.append(line)
    if name:
        jobs[name] = "".join(buf)
    return jobs


def test_ci_gates_run_once_and_every_shard_is_covered():
    """CI SHAPE, pinned after the 20-minute timeout. Two properties, both of which fail SILENTLY:

    1. THE GATES RUN ONCE. They used to sit inside the `pytest` job; sharding that job four ways would have
       run every lint/audit/doc-drift gate FOUR TIMES per push for identical results. They now live in their
       own unsharded `gates` job.
    2. THE SHARD COUNT IN ci.yml MATCHES THE ONE PASSED TO shard_tests.py. A matrix of 4 with a
       `--num-shards 3` command would silently drop a quarter of the suite -- green CI, untested code."""
    import re
    ci = ROOT / ".github" / "workflows" / "ci.yml"
    if not ci.is_file():
        pytest.skip("no ci.yml in this tree")
    jobs = _ci_jobs(ci.read_text(encoding="utf-8"))
    assert "pytest" in jobs and "gates" in jobs, sorted(jobs)

    assert "Gate --" in jobs["gates"], "the gates job has no gate steps -- did they drift into a sharded job?"
    for name, body in jobs.items():
        sharded = re.search(r'^\s+shard:\s*\[', body, re.M)
        if sharded and "Gate --" in body:
            raise AssertionError("job %r is sharded AND runs gates -- every gate would run once per shard" % name)

    for name, body in jobs.items():
        m = re.search(r'^\s+shard:\s*\[([^\]]*)\]', body, re.M)
        if not m:
            continue
        declared = len([x for x in m.group(1).split(",") if x.strip()])
        used = {int(n) for n in re.findall(r'--num-shards\s+(\d+)', body)}
        assert used, "job %r declares a shard matrix but never passes --num-shards" % name
        assert used == {declared}, \
            "job %r has %d shards in the matrix but passes --num-shards %s -- part of the suite would " \
            "never run" % (name, declared, sorted(used))
