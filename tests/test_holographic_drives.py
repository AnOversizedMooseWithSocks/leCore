"""Tests for homeostatic drives scheduling faculties through a nested process (DRIVE-1)."""

import numpy as np
from holographic.misc.holographic_drives import DriveSystem, make_nested_process, drive_process, _iter_nodes


def test_drive_system_picks_most_starved_applicable_need():
    D = DriveSystem()
    D.satisfy("clarity", 0.9)                                 # clarity nearly met; understanding still starved
    # both applicable -> the hungrier one (understanding) is pressing
    assert D.pressing({"clarity", "understanding"}) == "understanding"
    # only clarity applicable -> clarity, even though it's more satisfied
    assert D.pressing({"clarity"}) == "clarity"
    assert D.pressing(set()) is None


def test_balance_is_the_worst_served_need():
    D = DriveSystem()
    D.satisfy("clarity", 1.0); D.satisfy("understanding", 1.0)   # coverage still 0
    assert D.balance() == 0.0
    D.satisfy("coverage", 0.5)
    assert abs(D.balance() - 0.5) < 1e-9


def test_denoising_enables_recognition():
    # recognition only succeeds on a cleaned signal -> a 'recognize'-first schedule recognises nothing early
    root, cb = make_nested_process(depth=4, branching=2, dim=96, noise=2.2, seed=0)
    drive = drive_process(root, cb, energy=26, policy="drive", seed=0)
    assert drive["denoise_gain"] > 0.1                        # cleanup really lifts cosine to the codebook
    assert drive["recognized"] > 0                           # and cleaned nodes get recognised


def test_drive_matches_best_fixed_and_beats_naive():
    pols = ("drive", "denoise", "recognize", "descend", "random")
    bal = {p: [] for p in pols}
    for s in range(6):
        for p in pols:
            root, cb = make_nested_process(depth=4, branching=2, dim=96,
                                           noise=1.6 + 1.2 * ((s % 3) / 2),
                                           p_recognizable=0.3 + 0.5 * ((s % 5) / 4), seed=s)
            bal[p].append(drive_process(root, cb, energy=22, policy=p, seed=s)["balance"])
    m = {p: float(np.mean(bal[p])) for p in pols}
    best_fixed = max(m["denoise"], m["recognize"], m["descend"])
    assert m["drive"] >= best_fixed - 0.02                    # matches the best fixed priority, no order given
    assert m["drive"] >= m["random"] + 0.08                   # beats naive scheduling
    assert m["drive"] >= m["descend"] + 0.08                  # beats the worst fixed order


def test_process_tree_is_heterogeneous():
    root, _ = make_nested_process(depth=4, branching=2, dim=64, seed=3)
    nodes = list(_iter_nodes(root))
    assert any(n["truth"] is not None for n in nodes)        # some recognisable
    assert any(n["truth"] is None for n in nodes)            # some pure noise
    assert any(n["children"] for n in nodes)                 # some internal (descendable)


def test_energy_budget_is_respected():
    root, cb = make_nested_process(depth=4, branching=2, dim=64, seed=0)
    r = drive_process(root, cb, energy=10, policy="drive", seed=0)
    assert r["energy_left"] >= 0 and r["visited"] <= 10
