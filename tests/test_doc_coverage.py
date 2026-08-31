"""Guard (sweep 123): the set of UnifiedMind verbs that no human document mentions may
shrink, never grow. tools/doc_coverage.py measures it against README, FEATURE_GUIDE,
WHATS_NEW, docs/*.md and the integration guides; the budget file is the recorded count.
A new verb shipped without a line in a human doc trips this -- document it, or --rebase
with a reason in the commit."""
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_unmentioned_verbs_do_not_grow():
    p = subprocess.run([sys.executable, os.path.join("tools", "doc_coverage.py"), "--check"],
                       cwd=_REPO, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONHASHSEED="0"))
    assert p.returncode == 0, p.stdout[-800:] + p.stderr[-400:]
