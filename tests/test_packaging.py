"""Tests that the runtime data is packaged correctly, so a pip-installed wheel is not missing its data.

These guard the exact gap we fixed: holographic_dictionary needs a data file at runtime, and the flat py_modules
layout does not ship a loose data/ folder -- so the data lives in the importable lecore_data package. If someone
moves the data or forgets package_data, these fail in plain pytest (ci.yml), before a broken wheel is ever built."""
import os
import ast


def test_lecore_data_package_resolves_dictionary():
    import lecore_data
    assert lecore_data.exists("knowledge", "dictionary.json.xz")
    assert lecore_data.exists("knowledge", "LICENSE_WORDNET.txt")     # provenance ships too


def test_dictionary_loads_via_the_package():
    import holographic.misc.holographic_dictionary as d
    assert d.size() > 100000
    assert "force" in d.define("gravity").lower()


def test_heat_enrichment_data_is_packaged():
    import lecore_data
    assert lecore_data.exists("definitions", "native", "materials", "enrich.json")
    import holographic.simulation_and_physics.holographic_heat as h
    assert len(h._load_enrichment()) > 0                             # the file is found and parsed


def test_setup_declares_lecore_data_and_its_files():
    """setup.py must list lecore_data as a package and include its data globs -- else the wheel drops the data."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "setup.py"), encoding="utf-8").read()
    assert '"lecore_data"' in src and "package_data" in src
    assert "knowledge/*" in src                                      # the dictionary glob is present


def test_setup_ships_the_optional_c_bridge():
    """A wheel must include the Python bridge used with an external or locally built C library."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "setup.py"), encoding="utf-8").read()
    assert '"holographic_c"' in src


def test_build_script_stages_the_data_package():
    """build_package.sh must copy lecore_data into the staging folder, or the wheel is built without it."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "build_package.sh"), encoding="utf-8").read()
    assert "lecore_data" in src


def test_the_vendor_neutral_gpu_path_has_its_own_extra():
    """`wgpu` must be installable as an extra, and SEPARATELY from `gpu`.

    They are not interchangeable and must not be merged: `gpu` is CuPy, which is NVIDIA/CUDA-only and is
    tied to your CUDA version (so it is deliberately excluded from `all` and best installed by hand). `wgsl`
    is wgpu, which ships prebuilt wheels for every platform, needs no system toolchain, and runs on Vulkan /
    Metal / DX12 / WebGPU — including a SOFTWARE adapter with no GPU at all, which is how the CI lane checks
    correctness on an ordinary runner."""
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "setup.py"), encoding="utf-8").read()
    assert '"wgsl":' in src and '"wgpu"' in src, "the vendor-neutral GPU extra is gone"
    assert '"gpu":' in src and '"cupy"' in src, "the CuPy extra was removed; it is still supported"


def test_wgpu_is_in_all_and_cupy_is_not():
    """The asymmetry is deliberate and is the reason these are two extras rather than one. CuPy is excluded
    from `all` because of CUDA-version coupling — that reason does not apply to wgpu, so leaving wgpu out of
    `all` would deny the portable GPU path to everyone who installs the convenience extra."""
    import os
    import re

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "setup.py"), encoding="utf-8").read()
    block = re.search(r'"all":\s*\[(.*?)\]', src, re.DOTALL)
    assert block, "the `all` extra is gone"
    contents = block.group(1)
    assert "wgpu" in contents, "wgpu dropped out of `all`; the portable GPU path is no longer installed by it"
    assert "cupy" not in contents, "CuPy entered `all`; it is CUDA-version coupled and must stay opt-in"


def test_the_gpu_modules_live_in_discovered_packages():
    """find_packages() picks these up automatically, so no setup.py edit is needed when a module is added —
    but only if it lives inside the holographic/ tree. A module dropped at the top level would be silently
    missing from the wheel, which is exactly the kind of gap that shows up only after publishing."""
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for relative in ("holographic/io_and_interop/holographic_wgpurun.py",
                     "holographic/io_and_interop/holographic_gpureport.py",
                     "holographic/scene_and_pipeline/holographic_policy.py",
                     "holographic/scene_and_pipeline/holographic_placement.py"):
        assert os.path.exists(os.path.join(here, relative)), "%s moved out of the packaged tree" % relative
