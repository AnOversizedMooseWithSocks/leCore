"""Guard (sweeps 122-123): every `# guide-check` snippet in the human guides actually runs.

The guide is the instruction manual openzoo operators follow; it had fallen a dozen
sweeps behind the engine (serve, reflexes, wisdom, commons, study, orient were all
missing). A guide snippet that does not run is a guide that lies -- same rule as the
README (test_readme_examples). Only blocks that open with `# guide-check` are executed:
the others need a model directory or a live service and are documentation, not claims.
Each block runs in its OWN subprocess from the repo root so blocks cannot lean on each
other's names.
"""
import os
import re
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GUIDES = [os.path.join(_REPO, "FEATURE_GUIDE.md"),
           os.path.join(_REPO, "docs", "WHY_A_HOLOGRAPHIC_VM.md"),
           os.path.join(_REPO, "docs", "USE_CASES.md"),
           os.path.join(_REPO, "integrations", "openzoo", "PLATFORM_GUIDE.md")]


def _checked_blocks(path):
    text = open(path, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return [b for b in blocks if b.lstrip().startswith("# guide-check")]


def test_guide_check_blocks_exist():
    for g in _GUIDES:
        assert len(_checked_blocks(g)) >= 3, "%s lost its runnable door examples" % os.path.basename(g)


def test_every_guide_check_block_runs():
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=_REPO)
    for g in _GUIDES:
        for i, block in enumerate(_checked_blocks(g)):
            p = subprocess.run([sys.executable, "-c", block], cwd=_REPO, env=env,
                               capture_output=True, text=True, timeout=240)
            assert p.returncode == 0, "%s block %d failed:\n%s\n--- stderr ---\n%s" % (
                os.path.basename(g), i, block[:400], p.stderr[-1200:])
