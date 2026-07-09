"""Guard: the Python examples in README.md actually run.

The README is also the PyPI project page (setup.py's long_description), so a broken snippet there is the first thing a
new user or agent sees -- and they copy-paste it. This extracts every ```python block, concatenates them in order (a
reader follows the README top to bottom, so a later block may use names a earlier one defined), and runs the whole
thing in a SUBPROCESS from the repo root. If any line raises, the README lied and this fails. Bash blocks (```bash) are
ignored -- only Python is executed."""
import os
import re
import sys
import subprocess


def _python_blocks(readme_path):
    src = open(readme_path, encoding="utf-8").read()
    return re.findall(r"```python\n(.*?)```", src, re.DOTALL)


def test_readme_python_examples_run():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blocks = _python_blocks(os.path.join(root, "README.md"))
    assert blocks, "no ```python blocks found in README.md"
    script = "\n".join(blocks)                       # concatenate: later blocks may rely on earlier ones
    env = dict(os.environ, PYTHONHASHSEED="0")       # the engine is deterministic; match its contract
    r = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True,
                       env=env, timeout=240)
    assert r.returncode == 0, "a README python example failed:\n%s\n%s" % (r.stdout, r.stderr)
