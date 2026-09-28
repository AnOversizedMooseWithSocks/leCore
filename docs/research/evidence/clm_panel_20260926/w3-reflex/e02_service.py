"""E0.2 service half: latency of 20 inv('serve') round trips vs a no-work verb and find_capability, same 20 queries.
Queries = the first 20 held-out aliases (hash rule) in sorted order, so the set is reproducible."""
import hashlib
import json
import sys
import time

sys.path.insert(0, "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad")
sys.path.insert(0, "/home/claude/leCore")
from inv import inv
import numpy as np

from holographic.misc.holographic_skills import _catalog
uniq = sorted({a for c in _catalog().all() for a in (c.aliases or ())})
held = [a for a in uniq if int(hashlib.sha256(a.encode()).hexdigest()[:2], 16) % 2 == 1]
# spread the 20 over the sorted list (every len/20-th) so they are not all one alphabetical neighbourhood
qs = [held[i * (len(held) // 20)] for i in range(20)]


def timed(name, **kw):
    t = time.perf_counter()
    r = inv(name, **kw)
    return 1e3 * (time.perf_counter() - t), r


out = {"queries": qs}
for label, fn in (("noop reflex_stats", lambda q: timed("reflex_stats")),
                  ("find_capability", lambda q: timed("find_capability", problem=q)),
                  ("serve", lambda q: timed("serve", query=q))):
    ms, vias = [], {}
    for q in qs:
        t, r = fn(q)
        ms.append(t)
        if label == "serve" and isinstance(r, dict):
            v = r.get("via")
            vias[v] = vias.get(v, 0) + 1
    out[label] = {"mean": float(np.mean(ms)), "p50": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
                  "max": float(max(ms)), "first": ms[0]}
    if vias:
        out[label]["via"] = vias
    print(label, json.dumps(out[label]), flush=True)
json.dump(out, open("/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad/swarm/w3-reflex/e02_service.json", "w"), indent=1)
