"""CI wrapper for the adaptive-stop diffusion over the composed manifold (B3). The module ships its asserts in
`_b3_selftest`: stopping once the decoded structure has settled (stable past a floor) reaches the SAME structure
as the full fixed schedule on every seed at ~half the steps, with a final crisp snap restoring validity to
1.000 -- essentially free, novelty and diversity preserved, off by default. This collects that check."""
from holographic.agents_and_reasoning.holographic_hopfield import _b3_selftest


def test_holographic_diffusion_earlystop_selftest():
    _b3_selftest()
