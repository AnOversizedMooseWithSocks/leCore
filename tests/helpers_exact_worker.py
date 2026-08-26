"""Module-level workers for the Coordinator tests.

A lambda cannot cross a process boundary (`_pickle.PicklingError`), so a LocalPool test needs importable workers.
That is not a limitation of the exact reduce -- it is how multiprocessing pickles callables -- and discovering it
in a probe rather than in CI is the reason this file exists.
"""
import numpy as np


def contribs(bucket, cache):
    """The exact-reduce worker contract: return the bucket's CONTRIBUTIONS, not their sum."""
    return np.asarray(bucket, float)


def bucket_sum(bucket, cache):
    """The old float contract: sum inside the bucket. Kept so a test can show it still drifts."""
    return np.asarray(bucket, float).sum(axis=0)
