"""The schema gate added to the registry: route raw input to the tool whose learned schema
understands (compresses) it best -- the measured-better gate for fine, same-alphabet routing."""
from holographic_ai import random_vector
from holographic_orchestrator import Tool, ToolRegistry


def test_schema_gate_routes_input_to_the_tool_that_understands_it():
    import numpy as np
    reg = ToolRegistry()
    prose = "she walked through the quiet garden and thought about the long evening ahead. " * 30
    verse = "of arms and the man i sing who first from troy upon the shore of fate did come. " * 30
    nums = " ".join(str(int(v)) for v in (np.sin(np.linspace(0, 60 * np.pi, 1500)) * 30 + 60))
    for name, data in [("prose", prose), ("verse", verse), ("numbers", nums)]:
        reg.fit_schema(reg.add(Tool(name, "t", "t", random_vector(256, np.random.default_rng(0)))),
                       data[:1800], cuts=(0, 60, 160))
    assert reg.select_by_understanding(prose[1800:])[0][1].name == "prose"
    assert reg.select_by_understanding(verse[1800:])[0][1].name == "verse"
    assert reg.select_by_understanding(nums[1800:])[0][1].name == "numbers"
