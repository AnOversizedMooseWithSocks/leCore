"""PLACE-1 -- ONE placement decision over CPU, process pool, and device (holographic_placement).

WHY THIS EXISTS
---------------
Three oracles already answer three placement questions and none of them knows about the others:

    machine_place_unit(name, baseline_ns, n_calls)   -- should this run on a machine-model UNIT?
    should_pool(n_buckets, est_ms_per_bucket)        -- should this run on a PROCESS POOL?
    should_offload(n_bytes, flops_per_byte)          -- should this run on a DEVICE?

A caller wanting "where should this work go" had to consult all three, in some order, and reconcile them by
hand — and nothing at all reconciled them with `resource_policy`, so an oracle could happily recommend a
device the operator had forbidden. This composes them into one answer.

IT COMPOSES, IT DOES NOT REIMPLEMENT. Every verdict below comes from the existing oracle for that question;
this module contributes the ORDER, the policy veto, and one honest report. `machine_place` already does the
break-even arithmetic for any unit, and a device is simply a unit whose setup cost is the transfer.

THE ORDER IS NOT ARBITRARY, and it is the part worth arguing about:

  1. THE POLICY VETO COMES FIRST. No amount of arithmetic makes a forbidden device faster, and an oracle
     that recommends what the operator has banned is worse than no oracle — it produces a plan that cannot
     be executed and a user who stops trusting the advice.
  2. CHEAPEST-CORRECT WINS TIES. When two placements both pay, prefer the one with the weaker requirements:
     CPU over pool over device. A pool costs a process and an interpreter; a device costs a transfer and,
     unlike the pool, IT CHANGES THE NUMBERS — GPU matches NumPy only to a tolerance while the pooled path
     is verified bit-identical. Those are not equivalent risks and the ordering says so.
  3. THE DEVICE IS LAST FOR THE SAME REASON. It is the only placement that can change a result.

HONEST ABOUT ITS OWN CONFIDENCE. `should_offload`'s thresholds are arithmetic from PCIe bandwidth, not
measurements — no host<->device crossover has ever been measured in this project — so any device
recommendation is returned marked `provisional`. A caller can act on it, but nobody should mistake it for a
measured result.
"""


def place_work(n_bytes=None, flops_per_byte=None, n_buckets=None, est_ms_per_bucket=None,
               baseline_ns=None, n_calls=1, unit=None, policy=None, mind=None):
    """Where should this work run? Returns {placement, why, considered, provisional}.

    `placement` is one of 'cpu' | 'pool' | 'device' | 'unit'. Every candidate that was evaluated appears in
    `considered` with its own verdict and reason, so the answer carries the evidence rather than only the
    conclusion — the same discipline that makes `bundle_capacity` return its curve and `declare` return its
    descent.

    Supply only the parameters you can estimate: a candidate with missing inputs is reported as
    'not evaluated' rather than silently skipped, because a placement nobody costed and a placement that
    lost are different facts."""
    considered = {}

    allows_pool = allows_device = True
    if policy is not None:
        allows_pool, allows_device = bool(policy.pool_allowed()), bool(policy.gpu_allowed())

    # --- machine-model unit -------------------------------------------------------------------------
    if unit is not None and baseline_ns is not None and mind is not None:
        report = mind.machine_place_unit(unit, baseline_ns, n_calls)
        considered["unit"] = {"verdict": bool(report.get("use_unit")),
                              "why": "speedup %.2fx, break-even n=%.3g"
                                     % (report.get("speedup", 0.0), report.get("break_even_n", float("inf")))}
    else:
        considered["unit"] = {"verdict": None, "why": "not evaluated: needs unit= and baseline_ns="}

    # --- process pool -------------------------------------------------------------------------------
    if not allows_pool:
        considered["pool"] = {"verdict": False, "why": "the resource policy forbids process pools"}
    elif n_buckets is not None and est_ms_per_bucket is not None:
        from holographic.scene_and_pipeline.holographic_coordinator import should_pool

        ok, why = should_pool(n_buckets, est_ms_per_bucket,
                              cores=policy.cores() if policy is not None else None)
        considered["pool"] = {"verdict": bool(ok), "why": why}
    else:
        considered["pool"] = {"verdict": None,
                              "why": "not evaluated: needs n_buckets= and est_ms_per_bucket="}

    # --- device -------------------------------------------------------------------------------------
    if not allows_device:
        considered["device"] = {"verdict": False, "why": "the resource policy forbids the GPU"}
    elif n_bytes is not None and flops_per_byte is not None:
        from holographic.io_and_interop.holographic_gpureport import should_offload

        ok, why = should_offload(n_bytes, flops_per_byte, policy=policy)
        considered["device"] = {"verdict": bool(ok), "why": why}
    else:
        considered["device"] = {"verdict": None,
                                "why": "not evaluated: needs n_bytes= and flops_per_byte="}

    # CHEAPEST-CORRECT FIRST. unit -> pool -> device: weakest requirements win ties, and the device is last
    # because it is the only placement that can change the NUMBERS rather than only the speed.
    for name in ("unit", "pool", "device"):
        if considered[name]["verdict"]:
            return {"placement": name, "why": considered[name]["why"], "considered": considered,
                    "provisional": name == "device"}
    return {"placement": "cpu", "considered": considered, "provisional": False,
            "why": "no accelerated placement paid for itself; plain in-process CPU is the answer"}


def _selftest():
    import lecore
    from holographic.scene_and_pipeline.holographic_policy import ResourcePolicy

    mind = lecore.UnifiedMind(dim=64, seed=0)

    # 1. NOTHING SUPPLIED -> CPU, and every candidate says it was NOT EVALUATED rather than that it lost.
    out = place_work()
    assert out["placement"] == "cpu"
    assert all(out["considered"][k]["verdict"] is None for k in ("unit", "pool", "device"))

    # 2. THE POLICY VETO BEATS ARITHMETIC THAT WOULD OTHERWISE SAY YES.
    out = place_work(n_bytes=10 ** 9, flops_per_byte=100.0, policy=ResourcePolicy(gpu="off"))
    assert out["placement"] == "cpu"
    assert out["considered"]["device"]["verdict"] is False
    assert "forbids" in out["considered"]["device"]["why"]

    out = place_work(n_buckets=64, est_ms_per_bucket=500.0, policy=ResourcePolicy(pool="deny"))
    assert out["considered"]["pool"]["verdict"] is False

    # 3. A DEVICE RECOMMENDATION IS MARKED PROVISIONAL -- its thresholds are arithmetic, not measurements.
    out = place_work(n_bytes=10 ** 8, flops_per_byte=50.0)
    if out["placement"] == "device":
        assert out["provisional"] is True

    # 4. CHEAPEST-CORRECT WINS: with a unit that pays AND a device that pays, the unit is chosen.
    out = place_work(unit="t2_baked_grid", baseline_ns=1e7, n_calls=100,
                     n_bytes=10 ** 8, flops_per_byte=50.0, mind=mind)
    assert out["placement"] == "unit", out

    # 5. Evidence travels with the answer.
    assert set(out["considered"]) == {"unit", "pool", "device"}
    assert all("why" in row for row in out["considered"].values())

    print("holographic_placement: all selftests passed (veto first, cheapest-correct, provisional flagged)")


if __name__ == "__main__":
    _selftest()
