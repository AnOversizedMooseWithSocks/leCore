"""CI wrapper for adaptive encoder kernel placement (A3). The module ships its asserts in `_a3_selftest`:
ScalarEncoder.fit_resolution warps the encoder's input axis by the value-density CDF (with a resolution floor),
so a non-uniform distribution decodes markedly better under noise, while a uniform distribution ties (the warp
is the identity -- the CACHE-3 control); an unfitted encoder is the plain Fourier encoder. This collects that
check."""
from holographic_encoders import _a3_selftest


def test_holographic_adaptive_encoder_selftest():
    _a3_selftest()
