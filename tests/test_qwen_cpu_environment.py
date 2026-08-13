"""Static portability gates for the frozen Qwen CPU experiment environment."""

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiments" / "qwen35_acceptance"
REQUIREMENTS = EXPERIMENT / "requirements-cpu.txt"
EXPECTED = {
    "numpy": "2.4.6",
    "Pillow": "12.3.0",
    "safetensors": "0.8.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "5.14.0",
}


def _contract_dependency_versions():
    tree = ast.parse((EXPERIMENT / "contract.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and
                any(isinstance(target, ast.Name) and
                    target.id == "OFFICIAL_DEPENDENCY_VERSIONS"
                    for target in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError("contract.py has no OFFICIAL_DEPENDENCY_VERSIONS")


def test_cpu_requirements_pin_the_matching_official_stack():
    text = REQUIREMENTS.read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in text
    for package, version in EXPECTED.items():
        assert "%s==%s" % (package, version) in text
    assert "torch==2.11.0+cpu" in text
    assert "torchvision==0.26.0+cpu" in text
    assert 'sys_platform == "darwin"' in text


def test_runtime_preflight_and_requirements_cannot_drift():
    # numpy and safetensors are runner inputs but the official smoke preflight
    # deliberately gates the model/vision stack it imports dynamically.
    expected_preflight = {
        key: EXPECTED[key]
        for key in ("Pillow", "torch", "torchvision", "transformers")
    }
    assert _contract_dependency_versions() == expected_preflight
    runner = (EXPERIMENT / "run.py").read_text(encoding="utf-8")
    assert "from contract import METRIC_NAMES, OFFICIAL_DEPENDENCY_VERSIONS" in runner


def test_ci_and_operator_docs_install_the_frozen_file():
    relative = "experiments/qwen35_acceptance/requirements-cpu.txt"
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8")
    readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8")
    benchmark = (EXPERIMENT / "BENCHMARKING.md").read_text(encoding="utf-8")
    assert "pip install -r %s" % relative in workflow
    assert relative in readme
    assert relative in benchmark
