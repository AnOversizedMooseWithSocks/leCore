#!/usr/bin/env python3
"""run_all.py -- run the whole assimilation program, in order, caching each stage. stdlib only.

WHY A DRIVER AND NOT ONE BIG SCRIPT:
Every probe here already exists, is self-adapting, and has its own selftest. Folding them into a
monolith would duplicate logic that is already correct -- the exact mistake this program has made
twice (hand-rolled TT-SVD, hand-rolled k-means, both worse than the shipped versions). So this file
ORCHESTRATES. It never reimplements a probe. If a stage's script changes, the stage re-runs.

CACHING RULE
A stage is skipped iff its FINGERPRINT is unchanged. The fingerprint is a sha256 of:
    the stage's command, the stage's own script content, and (path, size, mtime_ns) of every input.
Model weights are 1.7 GB, so we do not content-hash them -- size+mtime is deterministic, cheap, and
wrong only if someone rewrites a file byte-identically in place, which is not a failure mode we have.
Script CONTENT is hashed, because a one-character edit must invalidate the result. (hashlib, never
`hash()` -- PYTHONHASHSEED discipline applies to the driver too.)

Outputs land in  scripts/.progress/<stage>.json   {fingerprint, seconds, returncode, stdout}
and the human-readable report is regenerated from those, so it is never stale.

USAGE
    python3 run_all.py                 # run everything that is not fresh
    python3 run_all.py --list          # what would run, and why
    python3 run_all.py --only census   # one stage (substring match)
    python3 run_all.py --force         # ignore the cache
    python3 run_all.py --report        # rebuild the summary from cached stages, run nothing
"""
import argparse, hashlib, json, pathlib, shutil, subprocess, sys, time

import lecore_paths as P

PROGRESS = P.SCRIPTS / '.progress'
REPORT = P.SCRIPTS / 'ASSIMILATION_REPORT.md'


# --------------------------------------------------------------------------- stage table
def _opt(fn):
    """A path that may not exist yet (weights not downloaded). Returns None instead of raising."""
    try:
        return fn()
    except FileNotFoundError:
        return None


def stages():
    """Ordered. `needs` = inputs whose absence SKIPS the stage (not an error).
    Cheap, weightless probes first, so a fresh machine gets answers in seconds.

    `timeout` is per stage, in seconds. Found the hard way: the exact duplicate census over a
    49,152-row embedding table is ~2.4e9 dot-products and the driver sat there forever with no
    way to tell "working" from "wedged". A long job must announce its budget."""
    qw, nw, nv, sw = _opt(P.qwen_weights), _opt(P.nomic_weights), _opt(P.nomic_vocab), _opt(P.smol_weights)
    qc, qt = _opt(P.qwen_config), _opt(P.qwen_tokenizer)
    out = []
    # STATIC FIRST. `distill_router` once died 0.4s into a 1800s budget on an UnboundLocalError that
    # needed no weights, no numpy, and no compute to find. A cheap check that runs before an expensive
    # stage is the same principle as the reachability audit.
    # ERRORS halt (they guarantee a crash); WARNINGS print and the run continues. The first version
    # halted on a warning about four standalone tools that are SUPPOSED to take arguments.
    out.append(dict(name='lint', script='lint_scripts.py', cmd=[sys.executable, 'lint_scripts.py'],
                    needs=[], timeout=120, why='static: alias shadowing + syntax (errors halt; warnings do not)'))
    out.append(dict(name='paths', script='lecore_paths.py', cmd=[sys.executable, 'lecore_paths.py'],
                    needs=[], timeout=60, why='verify the layout before anything else'))
    if qc and qt:
        out.append(dict(name='qwen_config', script='qwen_config_probe.py',
                        cmd=[sys.executable, 'qwen_config_probe.py'], needs=[qc, qt], timeout=300,
                        why='architecture + parameter budget + tokenizer coverage; NO weights needed'))
    if nw and nv:
        out.append(dict(name='distill_router', script='distill_router.py',
                        cmd=[sys.executable, 'distill_router.py'], needs=[nw, nv], timeout=1800,
                        why='the bag-of-tokens floor: how far does a token table get, alone?'))
        if P.CACHE.is_file():
            out.append(dict(name='distill_map', script='distill_map.py',
                            cmd=[sys.executable, 'distill_map.py'], needs=[nw, nv, P.CACHE], timeout=1800,
                            why='N31: is the encoder a LINEAR correction to the token table? (ridge, closed form)'))
    if qw or sw:
        model = qw or sw
        out.append(dict(name='llm_census', script='llm_census.py',
                        cmd=[sys.executable, 'llm_census.py', str(model)], needs=[model], timeout=5400,
                        why='families, duplicates, rank, q8/q4, shared subspace, embedding-table quant'))
        out.append(dict(name='model_analysis', script='model_analysis_program.py',
                        cmd=[sys.executable, 'model_analysis_program.py', str(model)], needs=[model], timeout=5400,
                        why='the VSA program: chunk_scout -> scalar frontier -> TT/Tucker -> codebook'))
    return out


# --------------------------------------------------------------------------- fingerprints
def fingerprint(st):
    h = hashlib.sha256()
    h.update(' '.join(st['cmd']).encode())
    if st['name'] == 'lint':
        # the lint reads EVERY script, so every script is one of its inputs
        for p in sorted(P.SCRIPTS.glob('*.py')):
            h.update(p.read_bytes())
    script = P.SCRIPTS / st['script']
    if script.is_file():
        h.update(script.read_bytes())              # a one-character edit must invalidate the stage
    else:
        h.update(b'<missing script>')
    for p in st['needs']:
        p = pathlib.Path(p)
        s = p.stat()
        h.update(f'{p.name}:{s.st_size}:{s.st_mtime_ns}'.encode())   # weights: too big to hash
    return h.hexdigest()[:16]


def cached(st):
    f = PROGRESS / f"{st['name']}.json"
    if not f.is_file():
        return None
    try:
        rec = json.loads(f.read_text())
    except json.JSONDecodeError:
        return None
    return rec if rec.get('fingerprint') == fingerprint(st) else None


def run_stage(st):
    budget = st.get('timeout', 3600)
    print(f"\n{'='*92}\n[{st['name']}] {st['why']}\n  (budget {budget}s)\n{'='*92}", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(st['cmd'], cwd=P.SCRIPTS, capture_output=True, text=True, timeout=budget)
        rc, so, se = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        rc = 124
        so = (e.stdout or b'').decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or '')
        se = f'TIMEOUT after {budget}s -- raise this stage\'s budget, or the input is bigger than expected'
    dt = time.time() - t0
    out = so + (('\n[stderr]\n' + se) if rc else '')
    print(out, flush=True)
    rec = dict(fingerprint=fingerprint(st), seconds=round(dt, 1),
               returncode=rc, stdout=out, when=time.strftime('%Y-%m-%d %H:%M:%S'))
    PROGRESS.mkdir(exist_ok=True)
    # Cache only SUCCESS. A cached failure would be skipped next run and silently poison the report --
    # the "stale example" failure mode, one layer up.
    if rc == 0:
        (PROGRESS / f"{st['name']}.json").write_text(json.dumps(rec, indent=1))
    else:
        (PROGRESS / f"{st['name']}.failed.json").write_text(json.dumps(rec, indent=1))
    status = 'OK' if rc == 0 else ('TIMED OUT' if rc == 124 else f'FAILED (rc={rc})')
    print(f"[{st['name']}] {status} in {dt:.1f}s", flush=True)
    return rec


# --------------------------------------------------------------------------- report
def build_report():
    """Regenerated from the cache every time, so it cannot go stale -- the same discipline that
    forbids hand-editing generated docs."""
    lines = ["# leCore assimilation -- run report", "",
             f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} from `.progress/`", ""]
    for st in stages():
        f_ok, f_bad = PROGRESS / f"{st['name']}.json", PROGRESS / f"{st['name']}.failed.json"
        rec = cached(st) or (json.loads(f_ok.read_text()) if f_ok.is_file()
                             else json.loads(f_bad.read_text()) if f_bad.is_file() else None)
        if not rec:
            lines += [f"## {st['name']} -- NOT RUN", f"*{st['why']}*", ""]
            continue
        fresh = 'fresh' if cached(st) else '**STALE** (inputs or script changed)'
        rc = rec['returncode']
        lines += [f"## {st['name']} -- {'OK' if rc == 0 else f'FAILED rc={rc}'} ({fresh})",
                  f"*{st['why']}*  \u00b7 {rec['seconds']}s \u00b7 {rec['when']}", "",
                  "```", rec['stdout'].rstrip()[:20000], "```", ""]
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nreport -> {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', default=None, help='substring of a stage name')
    ap.add_argument('--report', action='store_true', help='rebuild the report; run nothing')
    a = ap.parse_args()

    sts = [s for s in stages() if not a.only or a.only in s['name']]
    if not sts:
        print(f"no stage matches {a.only!r}; have: {[s['name'] for s in stages()]}")
        return

    if a.report:
        build_report(); return

    if a.list:
        print(f"  scripts: {P.SCRIPTS}\n")
        for s in sts:
            hit = cached(s)
            state = f"cached ({hit['seconds']}s, {hit['when']})" if hit else "WILL RUN"
            print(f"  {s['name']:16s} {state}")
            print(f"  {'':16s} {s['why']}")
        skipped = [n for n in ('qwen_config', 'distill_router', 'llm_census', 'model_analysis')
                   if n not in {s['name'] for s in stages()}]
        if skipped:
            print(f"\n  skipped (inputs absent): {skipped}")
        return

    failures = []
    for s in sts:
        hit = None if a.force else cached(s)
        if hit:
            print(f"[{s['name']}] cached ({hit['seconds']}s, {hit['when']}) -- skipping", flush=True)
            continue
        rec = run_stage(s)
        if rec['returncode']:
            failures.append(s['name'])
            if s['name'] == 'lint':
                # A lint failure means a later stage will crash. Do not burn the budget finding out.
                print("\n  ! lint failed -- fix the reported lines before running the long stages.", flush=True)
                build_report()
                sys.exit(1)
            print(f"  ! {s['name']} failed; continuing (later stages may not depend on it)", flush=True)

    build_report()
    if failures:
        print(f"\nFAILED STAGES: {failures}")
        sys.exit(1)
    print("\nall stages OK")


if __name__ == '__main__':
    main()
