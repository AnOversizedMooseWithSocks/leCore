"""CI wrapper for the cross-cutting PROBE SWEEP -- six transfers the panel pre-judged as no-ops, each measured and
kept as a negative (see holographic_probesweep for the measured reasons). All six confirm their prior: A1/A2/B4/D5
fail to concentration of measure (the kernel is already near-optimal), B2/D3 fail because the technique's
precondition does not hold (no incoherent tail to gate; a hard safety constraint is not an estimator to blend).
One test per probe so each negative is individually visible in the suite."""
from holographic_probesweep import _probe_a1, _probe_a2, _probe_b2, _probe_b4, _probe_d3, _probe_d5


def test_probe_a1_lowdiscrepancy_codebook_noop():
    _probe_a1()


def test_probe_a2_negative_lobe_cleanup_negative():
    _probe_a2()


def test_probe_b2_throughput_gated_generation_redundant():
    _probe_b2()


def test_probe_b4_lowdiscrepancy_sampling_noop():
    _probe_b4()


def test_probe_d3_mis_decision_negative():
    _probe_d3()


def test_probe_d5_observation_denoise_noop():
    _probe_d5()
