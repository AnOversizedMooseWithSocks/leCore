"""POLICY-1 -- the resource policy an OPERATOR sets (holographic_policy).

WHY THIS EXISTS
---------------
`cpu_budget()` answers WHAT IS PHYSICALLY AVAILABLE. That is not the same question as WHAT THIS PROCESS IS
ALLOWED TO TAKE. On a shared box, a CI runner, a laptop on battery, or a machine running leOS alongside the
user's actual work, the operator's answer is SMALLER and the engine cannot infer it. Auto-detection is a
good default for a library and the wrong default for a deployed system.

Before this, there was nowhere to say "four cores, no GPU". What existed was scattered and unrelated:
HOLOSTUFF_GPU / LECORE_CC_CACHE / LECORE_ZIG_CACHE as environment variables with three different prefixes,
`use_gpu()` as a global toggle with no cap, `cpu_budget()` as pure auto-detection with no override, and
`zig_dispatch_policy` / `cache_policy` as ADVISORY ORACLES that answer "what would be chosen" while nobody
can constrain the choice.

THE RULES, each of which is load-bearing
----------------------------------------
1. A POLICY CAPS; IT DOES NOT COMMAND. `cpu_cores=4` means NEVER MORE THAN 4, not ALWAYS 4. The measured
   gates still decide inside the cap -- `should_pool` may still refuse because the work per bucket does not
   justify a pool. A setting that FORCED parallelism would discard the measurement discipline the rest of
   this engine is built on, and would be a way to make things slower on purpose.

2. PRECEDENCE IS explicit-argument > policy > environment > auto-detect, NEVER reversed. A per-call
   argument must always win or debugging becomes non-local: you would not be able to reproduce a call by
   reading the call.

3. PERFORMANCE-ONLY AND NUMERICS-CHANGING SETTINGS ARE MARKED, AND DIFFERENTLY. `cpu_cores` and `pool` are
   performance-only: the pooled path is VERIFIED BIT-IDENTICAL, so capping cores cannot change an answer.
   `gpu` is NOT -- GPU matches NumPy only to a tolerance. Presenting them as the same kind of knob invites
   someone to flip GPU on globally and silently change their results, so `describe()` flags it.

4. THE EFFECTIVE POLICY IS PART OF A RESULT'S PROVENANCE. A run on 4 cores with no GPU is not the same
   experiment as one on 32 with a device. `describe()` returns the effective values AND WHERE EACH CAME
   FROM, so it can be recorded beside a measurement -- the same discipline that makes `bundle_capacity`
   carry its curve and `machine_place_unit` carry its baseline.

5. THE ENVIRONMENT VARIABLES THAT ALREADY EXIST ARE FOLDED IN, NOT DUPLICATED. HOLOSTUFF_GPU becomes one
   precedence layer of the `gpu` field. `make_repo_zip` already taught this the hard way: A HAND-MAINTAINED
   SECOND COPY OF A LIST IS ALWAYS THE STALE ONE.

6. NO CONFIG FILE FORMAT AND NO NEW DEPENDENCY. A dict, keyword arguments, and environment variables. File
   formats are the host application's job; core stays NumPy/Flask/stdlib.
"""

import os

#: Field -> (default, whether changing it can change NUMERIC RESULTS). The second flag is the one that
#: matters: an operator capping cores is choosing a speed, an operator enabling GPU is choosing a tolerance.
_FIELDS = {
    "cpu_cores": (None, False),          # None = auto-detect via cpu_budget()
    "pool": ("allow", False),            # "allow" | "deny"
    "gpu": ("auto", True),               # "auto" | "on" | "off"   -- NUMERICS-AFFECTING
    "device_memory_mb": (None, True),    # advisory ceiling for device allocations
}

#: Environment variable per field. THE PRECEDENCE LAYER LIVES HERE AND NOWHERE ELSE, so ANY mind picks these
#: up -- one built by the HTTP service, one built inside a farm WORKER NODE, one built in a notebook. A first
#: version read LECORE_CPU_CORES inside the service instead, which would have capped the multi-user surface
#: and left every worker node uncapped, and would have been a second copy of this list. The packaging tool
#: already taught that lesson: A HAND-MAINTAINED SECOND COPY OF A LIST IS ALWAYS THE STALE ONE.
#: A node operator therefore caps THEIR OWN machine with env vars, which is the only honest place for it --
#: a coordinator cannot and should not decide how much of a remote box it may consume.
_ENV = {"gpu": "HOLOSTUFF_GPU", "cpu_cores": "LECORE_CPU_CORES", "pool": "LECORE_ALLOW_POOL"}


class ResourcePolicy:
    """What this process is ALLOWED to use. Caps the measured gates; never forces them."""

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(_FIELDS)
        if unknown:
            raise ValueError("unknown policy field(s): %s (known: %s)"
                             % (", ".join(sorted(unknown)), ", ".join(sorted(_FIELDS))))
        self._set = dict(kwargs)
        if self._set.get("gpu") not in (None, "auto", "on", "off"):
            raise ValueError("gpu must be 'auto', 'on' or 'off', got %r" % (self._set["gpu"],))
        if self._set.get("pool") not in (None, "allow", "deny"):
            raise ValueError("pool must be 'allow' or 'deny', got %r" % (self._set["pool"],))
        cores = self._set.get("cpu_cores")
        if cores is not None and (not isinstance(cores, int) or cores < 1):
            raise ValueError("cpu_cores must be a positive int or None (auto), got %r" % (cores,))

    def _resolve(self, field):
        """(value, source) for one field, honouring explicit > env > default."""
        if field in self._set and self._set[field] is not None:
            return self._set[field], "policy"
        env_name = _ENV.get(field)
        if env_name and os.environ.get(env_name) is not None:
            raw = os.environ[env_name]
            if field == "gpu":
                return ("on" if raw not in ("", "0", "false", "False") else "off"), "env:%s" % env_name
            if field == "pool":
                return ("allow" if raw not in ("", "0", "false", "False") else "deny"), "env:%s" % env_name
            if field == "cpu_cores":
                # A MALFORMED CAP IS IGNORED, NOT GUESSED AT. LECORE_CPU_CORES="lots" must not silently
                # become 1 (a crippling cap nobody asked for) nor unlimited (a cap that does not cap).
                # Falling through to auto-detect is the only reading that cannot surprise an operator.
                return (int(raw), "env:%s" % env_name) if raw.isdigit() and int(raw) > 0 \
                    else (_FIELDS[field][0], "default")
            return raw, "env:%s" % env_name
        return _FIELDS[field][0], "default"

    def cores(self):
        """The core CAP as an int: the policy value, or the detected budget when unset.

        Takes the MINIMUM of policy and physical availability -- an operator asking for 32 cores on a 4-core
        box gets 4, because a policy grants permission, it does not conjure hardware."""
        from holographic.scene_and_pipeline.holographic_coordinator import cpu_budget

        detected = cpu_budget()
        value, _src = self._resolve("cpu_cores")
        return max(1, min(int(value), detected)) if value is not None else detected

    def pool_allowed(self):
        return self._resolve("pool")[0] != "deny"

    def gpu_allowed(self):
        """False only when explicitly forbidden. 'auto' and 'on' both permit; whether a device is actually
        PRESENT is a separate question that use_gpu answers."""
        return self._resolve("gpu")[0] != "off"

    def describe(self):
        """The effective policy WITH the source of every value -- provenance, not just configuration.

        `numerics_affecting` lists the fields whose current value can change results rather than only
        speed, so a recorded run says plainly whether its numbers are bit-exact or tolerance-bound."""
        rows, numerics = {}, []
        for field in sorted(_FIELDS):
            value, source = self._resolve(field)
            rows[field] = {"value": value, "source": source, "numerics_affecting": _FIELDS[field][1]}
            if _FIELDS[field][1] and value not in (None, "off", "auto"):
                numerics.append(field)
        from holographic.scene_and_pipeline.holographic_coordinator import cpu_budget
        rows["cpu_cores"]["effective"] = self.cores()
        rows["cpu_cores"]["detected"] = cpu_budget()
        return {"fields": rows, "numerics_affecting": numerics,
                "bit_exact": not numerics,
                "note": ("all settings are performance-only; results are bit-exact" if not numerics else
                         "%s can change NUMERIC results, not just speed" % ", ".join(numerics))}


def _selftest():
    # 1. A POLICY CAPS, IT DOES NOT COMMAND -- and it cannot conjure hardware it does not have.
    from holographic.scene_and_pipeline.holographic_coordinator import cpu_budget
    assert ResourcePolicy(cpu_cores=9999).cores() == cpu_budget(), "a policy invented cores"
    assert ResourcePolicy(cpu_cores=1).cores() == 1

    # 2. Defaults permit everything; explicit denial is honoured.
    assert ResourcePolicy().pool_allowed() and ResourcePolicy().gpu_allowed()
    assert not ResourcePolicy(pool="deny").pool_allowed()
    assert not ResourcePolicy(gpu="off").gpu_allowed()

    # 3. PRECEDENCE: an explicit policy value beats the environment.
    os.environ["HOLOSTUFF_GPU"] = "1"
    try:
        assert ResourcePolicy(gpu="off").gpu_allowed() is False, "env overrode an explicit policy"
        assert ResourcePolicy()._resolve("gpu") == ("on", "env:HOLOSTUFF_GPU")
    finally:
        del os.environ["HOLOSTUFF_GPU"]

    # 4. PROVENANCE: every value says where it came from, and numerics-affecting settings are flagged.
    plain = ResourcePolicy(cpu_cores=2).describe()
    assert plain["bit_exact"] is True, "capping cores was reported as numerics-affecting"
    assert plain["fields"]["cpu_cores"]["source"] == "policy"
    risky = ResourcePolicy(gpu="on").describe()
    assert risky["bit_exact"] is False and "gpu" in risky["numerics_affecting"]

    # 5. Refusals: an unknown field or a bad value is an error, never a silent default.
    for bad in ({"nonsense": 1}, {"gpu": "yes"}, {"pool": "maybe"}, {"cpu_cores": 0}):
        try:
            ResourcePolicy(**bad)
            raise AssertionError("accepted %r" % bad)
        except ValueError:
            pass

    print("holographic_policy: all selftests passed (caps not commands, precedence, provenance, refusals)")


if __name__ == "__main__":
    _selftest()
