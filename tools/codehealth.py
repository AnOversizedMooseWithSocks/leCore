"""codehealth.py -- CLI over holographic_codehealth: complexity x exposure x exercise.

    python3 tools/codehealth.py             # the attention list
    python3 tools/codehealth.py --complex   # plus the raw most-complex ranking
    python3 tools/codehealth.py --selftest  # includes rank cross-validation against radon, if installed
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic.io_and_interop.holographic_codehealth import main, _selftest

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main(sys.argv[1:]))
