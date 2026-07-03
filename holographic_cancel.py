"""holographic_cancel.py -- COOPERATIVE CANCELLATION for long operations (modeling-app backlog, item F).

Progress hooks already exist (path_trace(on_progress=...)), but there is no way to STOP a running render, sim, or
heavy modeling op -- so an app can't stay responsive or honour a "Stop" button. This adds the missing half: a tiny
cancel TOKEN a long loop checks between chunks (tiles / passes / steps) and bails out of, returning whatever it has
so far (a partial render is far better than a frozen UI).

It is deliberately minimal and old-school: a single boolean flag with a cooperative check -- no threads, no
exceptions in the hot path, nothing to get wrong. The op checks `token.should_stop()` (or just calls the token,
which is truthy when cancelled) at a cheap cadence; the app calls `token.cancel()`. Because the check is only read
between chunks, it adds no per-sample cost and stays deterministic. NumPy-free; stdlib only.
"""


class CancelToken:
    """A cooperative cancel flag. Pass it to a long operation as `should_stop=token`; the operation checks it
    between chunks and returns early when it is set. The app sets it with `cancel()` (e.g. from a Stop button)."""

    def __init__(self):
        self._cancelled = False

    def cancel(self):
        """Request cancellation. The next chunk boundary the operation checks will stop it."""
        self._cancelled = True

    def reset(self):
        """Clear the flag so the token can be reused for the next operation."""
        self._cancelled = False

    @property
    def cancelled(self):
        return self._cancelled

    def should_stop(self):
        """The cooperative check a loop calls between chunks."""
        return self._cancelled

    def __call__(self):
        """Callable form, so a token can be passed directly as `should_stop=token`."""
        return self._cancelled

    def __bool__(self):
        return self._cancelled


def run_cancellable(iterable, token, on_step=None):
    """Iterate `iterable`, yielding items until `token` is cancelled -- a thin helper for wrapping any step loop
    (a sim advancing frames, an iterative solver) in cancellation. `on_step(i, item)` is an optional per-step hook.
    Stops cleanly at the next item after cancellation (cooperative, not preemptive)."""
    for i, item in enumerate(iterable):
        if token is not None and token.should_stop():
            return
        if on_step is not None:
            on_step(i, item)
        yield item


def _selftest():
    """A token starts un-cancelled, flips on cancel(), resets, and works as a callable / in a boolean test;
    run_cancellable stops a loop early at the next step after cancellation."""
    tok = CancelToken()
    assert not tok.cancelled and not tok.should_stop() and not tok() and not bool(tok)
    tok.cancel()
    assert tok.cancelled and tok.should_stop() and tok() and bool(tok)
    tok.reset()
    assert not tok.cancelled

    # run_cancellable: cancel after the 3rd item -> the loop stops there
    tok.reset()
    seen = []
    def step(i, item):
        seen.append(item)
        if item == 2:
            tok.cancel()
    out = list(run_cancellable(range(100), tok, on_step=step))
    assert out == [0, 1, 2], out            # stopped at the next check after cancelling on item 2
    assert len(seen) == 3

    print("holographic_cancel selftest OK: CancelToken flips on cancel()/resets/works as callable+bool; "
          "run_cancellable stops a 100-item loop after 3 items once cancelled -- cooperative, deterministic")


if __name__ == "__main__":
    _selftest()
