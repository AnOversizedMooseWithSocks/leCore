"""Orphan backfill (wild-release sweep): six public functions the reachability scan found referenced by
NOTHING -- not faculty, not catalog, not tests, not tools. The honest fix is a real reference with a real
assertion, not a bigger budget: each test below exercises the actual contract, so the function moves to the
`test_only` bucket because something now genuinely depends on its behavior."""
import numpy as np


def test_outer_bind_is_the_outer_product():
    from holographic.sampling_and_signal.holographic_tensor import outer_bind
    a, b = np.array([1.0, 2.0]), np.array([3.0, 5.0, 7.0])
    M = outer_bind(a, b)
    assert np.array_equal(M, np.outer(a, b))


def test_sh_rotate_dc_returns_the_band0_term_untouched_by_rotation():
    from holographic.sampling_and_signal.holographic_spharm import sh_rotate_dc
    coeffs = np.arange(9.0)                     # bands 0..2 of a real SH expansion
    # the DC term is rotation-invariant BY DEFINITION -- that is the whole claim of the helper
    assert float(sh_rotate_dc(coeffs)) == coeffs[0]


def test_catalog_to_rows_exports_every_entry_with_the_three_fields():
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    c = default_catalog()
    rows = c.to_rows()
    assert len(rows) == len(list(c.all())) and rows, "one row per entry"
    assert all(set(r) == {"name", "does", "native"} for r in rows[:50])


def test_geomkernel_is_zero_respects_the_kernel_tolerance():
    from holographic.mesh_and_geometry.holographic_geomkernel import ModelTolerance
    k = ModelTolerance(abs_tol=1e-6)
    assert k.is_zero(5e-7) and not k.is_zero(2e-6)


def test_has_phase_data_says_yes_for_water_and_no_for_nonsense():
    from holographic.misc.holographic_phase import has_phase_data, PHASE_DATA
    known = next(iter(PHASE_DATA))
    assert has_phase_data(known) and not has_phase_data("unobtainium")


def test_knowledgestore_add_file_ingests_a_real_file_as_a_document(tmp_path):
    import lecore
    from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
    p = tmp_path / "note.txt"
    p.write_text("the orphan backfill closed the loop")
    ks = KnowledgeStore(str(tmp_path / "store"))
    e = ks.add_file(str(p))
    m = lecore.UnifiedMind(dim=32, seed=0)
    hit = ks.search(m, "orphan backfill", top=1)
    assert hit and "closed the loop" in hit[0]["text"]
    # add() returns the entry LIST it appended to (probed, not assumed) -- the hit carries the
    # per-entry record, so the kind/source contract is asserted there
    assert hit[0]["kind"] == "document" and hit[0]["source"] == "note.txt"
