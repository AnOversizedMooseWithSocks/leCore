# leCore Product Wedge

The first product surface is **LocalAgentCore**: a small facade for local agent
memory, capability routing, and readiness evidence.

It deliberately narrows the promise. The full repo is a broad research engine;
this surface is the five-minute path for a builder who wants a deterministic,
inspectable local substrate.

```python
from holographic_product import LocalAgentCore

core = LocalAgentCore(dim=512, seed=0)
core.remember("local agents need deterministic durable memory", label="memory")
core.remember("capability routing should act when confident", label="routing")

print(core.recall("deterministic local memory")[0])
print(core.route("render a scene with global illumination"))
print(core.dashboard())
```

The same object is available from the friendly import surface:

```python
import lecore

core = lecore.product.LocalAgentCore()
```

## What It Productizes

- **Memory:** local text memories encoded through `UniversalEncoder` and recalled
  through the shared `Index` home.
- **Routing:** plain-English tasks routed through the existing skill catalog.
- **Evidence:** a JSON/static-HTML dashboard with memory counts, capability
  counts, determinism checks, and optional C-kernel availability.
- **Persistence:** `save(path)` and `LocalAgentCore.load(path)` round-trip the
  stable state as JSON. Vectors are rebuilt from seed, text context, and entries.
- **Paid API publishing:** `holographic_x402_api.py` serves the product wedge as
  an optional x402-paid FastAPI service. See [`X402_API.md`](X402_API.md).

## Honest Scope

This is not a neural database, a hosted service, or a general semantic model.
Out of the box it matches by the deterministic holographic text geometry it is
given. Better domain recall comes from adding domain memories, teaching text
context, or layering a specialized encoder on the same facade.

The product rule is simple: the public wedge stays small, auditable, local, and
measured. The research garden remains available behind it.
