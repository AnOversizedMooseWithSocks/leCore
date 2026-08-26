"""The schema gate added to the registry: route raw input to the tool whose learned schema
understands (compresses) it best -- the measured-better gate for fine, same-alphabet routing."""
from holographic.agents_and_reasoning.holographic_ai import random_vector
from holographic.scene_and_pipeline.holographic_orchestrator import Tool, ToolRegistry


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


# --- Differentiable Orchestration: optimize a whole chain by analytic-gradient ascent (no autodiff) ---

import numpy as np  # noqa: E402  (appended block; the module above may not import numpy at top level)


def _corr_tools(rng, N, D):
    base = rng.normal(size=(N, D)) + 2.0 * rng.normal(size=(1, D))   # shared component -> correlated
    return base / np.linalg.norm(base, axis=1, keepdims=True)


def test_differentiable_recovers_a_correlated_composed_chain():
    from holographic.scene_and_pipeline.holographic_orchestrator import chain_signature, optimize_toolchain
    rng = np.random.default_rng(0); N, D, L = 12, 256, 4
    V = _corr_tools(rng, N, D)
    true = list(rng.choice(N, L, replace=False))
    idx, score = optimize_toolchain(V, chain_signature(V[true]), L, steps=300)
    assert sum(1 for a, b in zip(idx, true) if a == b) >= L - 1 and score > 0.9


def test_differentiable_beats_independent_greedy_on_correlated_tools():
    from holographic.scene_and_pipeline.holographic_orchestrator import chain_signature, optimize_toolchain
    rng = np.random.default_rng(1); N, D, L = 12, 256, 4
    V = _corr_tools(rng, N, D)
    true = list(rng.choice(N, L, replace=False)); goal = chain_signature(V[true])
    greedy = list(np.argsort(V @ goal)[::-1][:L])               # position-blind per-tool score (existing style)
    g_sig = chain_signature(V[greedy])
    g_cos = float(g_sig @ goal) / (np.linalg.norm(g_sig) * np.linalg.norm(goal))
    _, d_cos = optimize_toolchain(V, goal, L, steps=300)
    assert d_cos > g_cos + 0.05


def test_differentiable_planning_is_deterministic():
    from holographic.scene_and_pipeline.holographic_orchestrator import optimize_toolchain
    rng = np.random.default_rng(2); V = rng.normal(size=(10, 64)); g = rng.normal(size=64)
    assert optimize_toolchain(V, g, 4, steps=50)[0] == optimize_toolchain(V, g, 4, steps=50)[0]
