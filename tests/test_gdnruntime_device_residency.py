"""to_device must move EVERY weight the forward pass reads, not just self.w.

Field failure this pins (Moose, `./install.bat` on an A4500):

    [install] leCore into the weights
        hardware: gpu (weights resident)
    ...
      File "holographic_gdnruntime.py", line 746, in forward
        h = self.embed[ids]
      File "cupy/_core/core.pyx", line 1807, in ..._ndarray_base.__array__
    TypeError: Implicit conversion to a NumPy array is not allowed.
               Please use `.get()` to construct a NumPy array explicitly.

`self.embed` and `self.lm_head` are bound in __init__ as `np.asarray(weights[...])`
-- SEPARATE ATTRIBUTES, not entries in `self.w` -- so to_device's loop over
`self.w` left them on the host. forward() then builds `ids = xp.asarray(...)` on
the device and evaluates `self.embed[ids]`: numpy indexing with a device index,
which numpy tries to convert and cupy refuses.
THE RESIDENCY REPORT WAS TRUE AND USELESS: some of the weights moved.

THE STAND-IN REFUSES CONVERSION ON PURPOSE. This file already records why a
previous parity test missed this class of bug: it "aliased numpy AS cupy to prove
residency without hardware, and numpy-as-cupy accepts np.asarray happily. A FAKE
DEVICE TESTS THE PLUMBING AND NOT THE CONTRACT." So the stand-in here is not an
ndarray subclass (subclassing bypasses __array__ and reproduces nothing -- tried,
and it silently passed); it is a wrapper, like cupy, whose __array__ raises the
real message.
"""

import numpy as np
import pytest


class _DeviceArray:
    """A minimal cupy stand-in: a wrapper that REFUSES implicit conversion."""

    def __init__(self, a):
        self._a = np.asarray(a)

    def __array__(self, *a, **k):
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Please use `.get()` to construct a NumPy array explicitly.")

    def get(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype


def test_a_device_array_refuses_conversion_like_cupy():
    """The stand-in must reproduce the CONTRACT, or the test below proves nothing."""
    host = np.zeros((10, 4))
    with pytest.raises(TypeError, match="Implicit conversion"):
        host[_DeviceArray(np.arange(3))]

    # and the trap that made a previous attempt useless: an ndarray SUBCLASS
    # does not go through __array__ at all, so it accepts the indexing happily
    class _Subclass(np.ndarray):
        def __array__(self, *a, **k):
            raise TypeError("never reached")

    host[np.arange(3).view(_Subclass)]      # no raise -- which is the point


def test_to_device_moves_embed_and_lm_head_not_only_the_weight_dict(monkeypatch):
    """**Every tensor forward() reads must move, or residency is a half-truth.**

    Asserts the ATTRIBUTES moved, not that a forward succeeded: a forward needs a
    whole model, while the defect is exactly one loop that iterated the wrong
    collection."""
    from holographic.io_and_interop import holographic_gdnruntime as G

    class _Fake(G.GDNRuntime):
        def __init__(self):                      # no model, no config
            self.w = {"a": np.zeros((2, 2)), "b": np.ones((2, 2))}
            self.embed = np.zeros((8, 4))
            self.lm_head = self.embed            # TIED, the 0.8B case
            self._dev = None
            self.cfg = {}

    xp = type("xp", (), {})()                    # any non-numpy sentinel
    monkeypatch.setattr(G, "np", np, raising=False)

    rt = _Fake()
    moved = []

    def _to(x):
        moved.append(x)
        return _DeviceArray(x)

    # drive the tail of to_device directly with a stubbed mover, so the test does
    # not need a CUDA device to assert the contract
    for k in list(rt.w):
        rt.w[k] = _to(np.asarray(rt.w[k]))
    tied = rt.lm_head is rt.embed
    rt.embed = _to(np.asarray(rt.embed))
    rt.lm_head = rt.embed if tied else _to(np.asarray(rt.lm_head))

    assert isinstance(rt.embed, _DeviceArray), "embed stayed on the host"
    assert isinstance(rt.lm_head, _DeviceArray), "lm_head stayed on the host"
    assert rt.lm_head is rt.embed, (
        "a tied model must keep ONE object -- moving them separately doubles the "
        "memory and silently un-ties the head from the embedding")
    assert len(moved) == 3, moved


def test_to_device_source_moves_the_prebound_tensors():
    """The fix must be IN to_device, not only in this test's re-enactment.

    Reads the source, because the behaviour needs a CUDA device to exercise and
    the regression is a missing few lines rather than a wrong value."""
    import inspect

    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

    src = inspect.getsource(GDNRuntime.to_device)
    assert "self.embed" in src, (
        "to_device does not touch self.embed -- forward() will index a host "
        "array with a device index on the first line of the pass")
    assert "self.lm_head" in src, "to_device does not touch self.lm_head"
    assert "is self.embed" in src or "_tied" in src, (
        "to_device must preserve the tied identity lm_head is embed")


def test_the_weight_getter_does_not_convert_a_device_weight():
    """**Moving the tensors was necessary and NOT sufficient.**

    After embed/lm_head moved, `_g_opt` still did `np.asarray(self.w[key])` -- and
    that getter is what EVERY layer calls, so on a real device it is the first
    thing to die. The runtime's own docstring names the site: "FIELD-CAUGHT on an
    A4500, at the FIRST weight read (_g on input_layernorm)".
    FOUND BY leCore's OWN file_grep, not by reading: three `np.asarray(self.w[`
    sites survived a fix I had already called complete."""
    import numpy as np

    from holographic.io_and_interop.holographic_gdnruntime import GDNRuntime

    class _Dev:
        __module__ = "cupy._core.core"      # what the follow-the-data check reads

        def __init__(self, a):
            self._a = np.asarray(a)

        def __array__(self, *x, **k):
            raise TypeError("Implicit conversion to a NumPy array is not allowed.")

        def get(self):
            return self._a

    rt = GDNRuntime.__new__(GDNRuntime)
    rt.root = "model."
    rt.w = {"model.layers.0.input_layernorm.weight": _Dev(np.ones(8))}
    got = rt._g_opt(0, "input_layernorm.weight")
    assert isinstance(got, _Dev), (
        "the weight getter converted a device weight to numpy -- this is the "
        "first read of every layer and it will raise on a real GPU")

    # a HOST weight must still come back as a float64 numpy array, unchanged
    rt.w["model.layers.0.input_layernorm.weight"] = np.ones(8, np.float32)
    host = rt._g_opt(0, "input_layernorm.weight")
    assert isinstance(host, np.ndarray) and host.dtype == np.float64, host.dtype
