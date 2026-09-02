"""Guard (sweeps 122-123): every `# guide-check` snippet in the human guides actually runs.

The guide is the instruction manual openzoo operators follow; it had fallen a dozen
sweeps behind the engine (serve, reflexes, wisdom, commons, study, orient were all
missing). A guide snippet that does not run is a guide that lies -- same rule as the
README (test_readme_examples). Only blocks that open with `# guide-check` are executed:
the others need a model directory or a live service and are documentation, not claims.
Each block runs in its OWN subprocess from the repo root so blocks cannot lean on each
other's names.

WHY THE BLOCKS RUN CONCURRENTLY (sweep 129, and it is a regression this test suffered rather
than a speed-up anyone wanted): the guides grew from 11 guide-check blocks to 25, sequential
wall time crossed the conftest's 15 s per-test budget, and the watchdog SKIPPED this test.
A skipped guard guards nothing -- the guide could have started lying and nothing would have
said so -- and `@pytest.mark.slow` is the wrong remedy here because slow-marked tests are
DESELECTED by default, which is the same silence with better manners. The blocks are already
independent by construction (that is the point of one subprocess each), so running them in a
thread pool preserves the isolation exactly and only removes the waiting. MEASURED on this box:
25 blocks, 15.8 s sequential at the moment of the skip -- one 8.1 s LOD block was over half of it,
and rewriting that block to assert the same thing on a smaller mesh brought the sequential figure
to 8.3 s -- then pooling takes it to 5.0 s, slowest single block 1.3 s. Both fixes were needed: the
pool alone left only 3.7 s of headroom under the budget, which the next added block would eat.
"""
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GUIDES = [os.path.join(_REPO, "FEATURE_GUIDE.md"),
           os.path.join(_REPO, "docs", "WHY_A_HOLOGRAPHIC_VM.md"),
           os.path.join(_REPO, "docs", "USE_CASES.md")]
# integrations/openzoo/PLATFORM_GUIDE.md is gitignored by design (private platform material); when it
# is present locally its blocks are checked too, but CI must never depend on it (sweep 128).
_OPTIONAL_GUIDES = [os.path.join(_REPO, "integrations", "openzoo", "PLATFORM_GUIDE.md")]


def _checked_blocks(path):
    assert os.path.exists(path), (
        "guide missing from the checkout: %s -- it exists in the delivery zip; `git add` it "
        "(sweep 127: CI lacked integrations/openzoo/PLATFORM_GUIDE.md, which also shifted the "
        "doc-coverage count by the verbs only that guide mentions)" % os.path.relpath(path, _REPO))
    text = open(path, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return [b for b in blocks if b.lstrip().startswith("# guide-check")]


def test_guide_check_blocks_exist():
    for g in _GUIDES:
        assert len(_checked_blocks(g)) >= 3, "%s lost its runnable door examples" % os.path.basename(g)


def _run_block(job):
    """One block in its own subprocess. Returns (guide, index, block, CompletedProcess) so the
    failure message can name WHICH block broke -- a pool that reported only 'something failed'
    would be a worse guard than the sequential loop it replaced."""
    guide, index, block = job
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=_REPO)
    p = subprocess.run([sys.executable, "-c", block], cwd=_REPO, env=env,
                       capture_output=True, text=True, timeout=240)
    return guide, index, block, p


def test_every_guide_check_block_runs():
    jobs = [(g, i, b)
            for g in _GUIDES + [o for o in _OPTIONAL_GUIDES if os.path.exists(o)]
            for i, b in enumerate(_checked_blocks(g))]
    # Ordered results, so a failure reports the same block number a reader counts in the file.
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_run_block, jobs))
    for guide, index, block, p in results:
        assert p.returncode == 0, "%s block %d failed:\n%s\n--- stderr ---\n%s" % (
            os.path.basename(guide), index, block[:400], p.stderr[-1200:])
