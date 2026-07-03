# lecore.py -- the friendly front door.
#
# The engine is a flat set of `holographic_*.py` modules. This module re-exports the handful of things
# most callers want, so that after `pip install lecore` you can simply:
#
#     import lecore
#     mind = lecore.UnifiedMind(...)
#
# ...without needing to know the internal module names. The full engine is still available directly
# (e.g. `from holographic_render import ...`) for anything not re-exported here.

# The main entry point -- always present.
from holographic_unified import UnifiedMind

# Convenience: re-export the raw VSA algebra if it's present. Guarded so that a future rename can never
# break `import lecore` itself -- the mind is what matters, the loose ops are a nicety.
try:
    from holographic_ai import random_vector, unitary_vector, bind, unbind, involution
except Exception:                        # pragma: no cover
    pass

__version__ = "0.1.0"
