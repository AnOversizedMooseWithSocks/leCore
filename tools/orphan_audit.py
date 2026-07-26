"""orphan_audit.py -- CLI over holographic_orphanaudit: reachability at FUNCTION granularity.

The logic lives in the engine (holographic/io_and_interop/holographic_orphanaudit.py) so that leCore can
audit itself through a mind faculty as well as from the command line. This file is the CI entry point only.

    python3 tools/orphan_audit.py            # summary + exit code
    python3 tools/orphan_audit.py --list     # the orphan and test-only names
    python3 tools/orphan_audit.py --json     # machine-readable
    python3 tools/orphan_audit.py --selftest
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic.io_and_interop.holographic_orphanaudit import main, _selftest

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main(sys.argv[1:]))
