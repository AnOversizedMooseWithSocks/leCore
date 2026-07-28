"""Top-level, picklable workers -- a lambda cannot cross a process boundary."""
import numpy as np

def light(bucket, cache=None):
    return np.asarray(bucket, float).sum()

def heavy(bucket, cache=None):
    a = np.asarray(bucket, float)
    out = 0.0
    for _ in range(120):
        out += float(np.sum(np.sqrt(np.abs(a * 3.7 + 1.0))))
    return out

def with_cache(bucket, cache=None):
    """Reads the SHARED cache, so a regression to per-bucket pickling is observable rather than silent."""
    a = np.asarray(bucket, float)
    c = 0.0 if cache is None else float(np.sum(cache))
    return float(a.sum()) + c

def bulk(bucket, cache=None, reps=4000):
    a = np.asarray(bucket, float)
    out = 0.0
    for _ in range(reps):
        out += float(np.sum(np.sqrt(np.abs(a * 3.7 + 1.0))))
    return out
