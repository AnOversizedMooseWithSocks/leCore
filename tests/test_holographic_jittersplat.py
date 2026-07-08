"""CI wrapper for the jittered sub-pixel splat accumulation kept negative (ACCUM-1). The module ships its asserts in
`_selftest`: jittered accumulation beats the base refit only by supersampling, and a finer-grid refit given the same
sub-pixel samples is strictly better -- so jittering does not sharpen past the refit. This collects that recorded
negative into the suite."""
from holographic.rendering.holographic_jittersplat import _selftest


def test_holographic_jittersplat_selftest():
    _selftest()
