"""Regression traps for the operator resource policy (GPU backlog A6).

The property under test is that the policy is a VETO, not a suggestion, and that it CAPS rather than
COMMANDS. A policy that merely advised would look identical in every test that only reads it back — so
every test here checks the effect on a real gate (`should_pool`, `use_gpu`, `cpu_budget`), not the stored
value.
"""
import os

import pytest

import lecore
from holographic.scene_and_pipeline.holographic_policy import ResourcePolicy
from holographic.scene_and_pipeline.holographic_coordinator import cpu_budget


@pytest.fixture
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


# --------------------------------------------------------------------------------------
# Caps, not commands.
# --------------------------------------------------------------------------------------

def test_a_policy_cannot_conjure_hardware():
    """Permission is not hardware. Asking for 9999 cores on a small box yields what the box has -- the
    policy grants permission to use up to N, it does not allocate N."""
    assert ResourcePolicy(cpu_cores=9999).cores() == cpu_budget()


def test_a_policy_caps_downward(mind):
    mind.resource_policy(cpu_cores=1)
    assert mind.cpu_budget() == 1


def test_capping_cores_does_not_force_pooling(mind):
    # THE DISTINCTION THAT MAKES THIS A CAP. Even with pooling allowed and cores available, the measured
    # gate still decides -- a policy that forced parallelism would be a way to make things slower.
    mind.resource_policy(cpu_cores=8, pool="allow")
    refused, why = mind.should_pool(8, 0.01, cores=8)
    assert refused is False and "dispatch" in why


# --------------------------------------------------------------------------------------
# The vetoes actually bite.
# --------------------------------------------------------------------------------------

def test_pool_deny_refuses_even_a_job_that_would_pay(mind):
    mind.resource_policy(pool="deny")
    refused, why = mind.should_pool(64, 5000.0, cores=64)
    assert refused is False, "a job that clears every measured gate was pooled despite pool='deny'"
    assert "policy" in why


def test_gpu_off_is_a_veto_before_the_backend_is_touched(mind):
    """`gpu='off'` must mean the device is never initialised, not merely that we stop using it. Observable
    without owning a GPU: use_gpu(True) returns False either way here, but the policy path returns False
    BEFORE reaching enable_gpu, which is what makes the setting meaningful on a machine that HAS a device."""
    mind.resource_policy(gpu="off")
    assert mind.use_gpu(True) is False


def test_gpu_auto_and_on_both_permit(mind):
    for setting in ("auto", "on"):
        mind.resource_policy(gpu=setting)
        assert mind._resource_policy().gpu_allowed() is True


# --------------------------------------------------------------------------------------
# Precedence and provenance.
# --------------------------------------------------------------------------------------

def test_an_explicit_policy_beats_the_environment():
    # explicit > policy > env > default, never reversed: you must be able to reproduce a call by reading it.
    os.environ["HOLOSTUFF_GPU"] = "1"
    try:
        assert ResourcePolicy(gpu="off").gpu_allowed() is False
        assert ResourcePolicy()._resolve("gpu") == ("on", "env:HOLOSTUFF_GPU")
    finally:
        del os.environ["HOLOSTUFF_GPU"]


def test_the_existing_env_var_is_folded_in_not_duplicated():
    # HOLOSTUFF_GPU already existed. It became a precedence layer rather than a second, parallel switch --
    # a hand-maintained second copy of a list is always the stale one.
    os.environ["HOLOSTUFF_GPU"] = "0"
    try:
        assert ResourcePolicy().gpu_allowed() is False
    finally:
        del os.environ["HOLOSTUFF_GPU"]


def test_every_value_reports_where_it_came_from(mind):
    report = mind.resource_policy(cpu_cores=2)
    assert report["fields"]["cpu_cores"]["source"] == "policy"
    assert report["fields"]["pool"]["source"] == "default"
    assert "effective" in report["fields"]["cpu_cores"]
    assert "detected" in report["fields"]["cpu_cores"]


def test_numerics_affecting_settings_are_flagged_separately(mind):
    """THE SAFETY PROPERTY. Capping cores is performance-only because the pooled path is verified
    bit-identical; enabling GPU is not, because GPU matches NumPy only to a tolerance. Presenting them as
    the same kind of knob is how someone silently changes their results."""
    plain = mind.resource_policy(cpu_cores=2, pool="deny")
    assert plain["bit_exact"] is True and plain["numerics_affecting"] == []

    risky = mind.resource_policy(gpu="on")
    assert risky["bit_exact"] is False and "gpu" in risky["numerics_affecting"]


# --------------------------------------------------------------------------------------
# Refusals.
# --------------------------------------------------------------------------------------

def test_bad_settings_raise_rather_than_silently_defaulting():
    for bad in ({"nonsense": 1}, {"gpu": "yes"}, {"pool": "maybe"}, {"cpu_cores": 0}, {"cpu_cores": -4}):
        with pytest.raises(ValueError):
            ResourcePolicy(**bad)


def test_the_default_policy_permits_everything(mind):
    # Additive: a mind with no policy set must behave exactly as before this existed.
    assert mind._resource_policy().pool_allowed() is True
    assert mind._resource_policy().gpu_allowed() is True
    assert mind.cpu_budget() == cpu_budget()


def test_the_policy_is_discoverable(mind):
    for query in ("limit how many cores it uses", "turn off the gpu", "configure resource limits",
                  "stop it using all my cores"):
        assert "Resource policy" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the resource policy" % query


# --------------------------------------------------------------------------------------
# The env layer reaches EVERY mind -- service, worker node, notebook (backlog W7 wiring).
# --------------------------------------------------------------------------------------

def test_cores_and_pool_are_readable_from_the_environment():
    """THE PRECEDENCE LAYER LIVES IN ResourcePolicy AND NOWHERE ELSE. A first version parsed
    LECORE_CPU_CORES inside the HTTP service, which would have capped the multi-user surface and left every
    farm WORKER NODE uncapped — and would have been a second copy of the list. A node operator caps their own
    machine with env vars, which is the only honest place for it: a coordinator cannot decide how much of a
    remote box it may consume."""
    os.environ["LECORE_CPU_CORES"] = "4"
    os.environ["LECORE_ALLOW_POOL"] = "0"
    try:
        policy = ResourcePolicy()
        assert policy._resolve("cpu_cores") == (4, "env:LECORE_CPU_CORES")
        assert policy.pool_allowed() is False
    finally:
        del os.environ["LECORE_CPU_CORES"]
        del os.environ["LECORE_ALLOW_POOL"]


def test_a_malformed_cap_is_ignored_rather_than_guessed():
    """LECORE_CPU_CORES='lots' must not silently become 1 (a crippling cap nobody asked for) nor unlimited
    (a cap that does not cap). Falling through to auto-detect is the only reading that cannot surprise."""
    for bad in ("lots", "", "-2", "0", "3.5"):
        os.environ["LECORE_CPU_CORES"] = bad
        try:
            assert ResourcePolicy()._resolve("cpu_cores")[1] == "default", "%r was interpreted" % bad
        finally:
            del os.environ["LECORE_CPU_CORES"]


def test_an_explicit_policy_still_beats_the_env_for_cores():
    os.environ["LECORE_CPU_CORES"] = "4"
    try:
        assert ResourcePolicy(cpu_cores=1).cores() == 1
    finally:
        del os.environ["LECORE_CPU_CORES"]


def test_the_service_mind_inherits_the_env_without_its_own_parsing():
    # The service needs no env code of its own: any mind it builds picks the policy up. Asserted against the
    # source so a future "helpful" second parser in the service fails this test.
    import pathlib

    src = pathlib.Path("holographic_service.py").read_text()
    assert "LECORE_CPU_CORES" in src, "the service no longer documents where its cap comes from"
    assert "os.environ.get(\"LECORE_CPU_CORES\")" not in src, \
        "the service is parsing the env itself again; that is the second-copy bug"
