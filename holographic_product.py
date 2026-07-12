"""holographic_product.py -- the small product-facing leCore facade.

WHY THIS EXISTS
---------------
The research engine is intentionally broad: memory, geometry, rendering,
simulation, jobs, skills, and more all share the same holographic substrate.
That is useful for research, but a first-time product user needs one narrow,
reliable door.

`LocalAgentCore` is that door. It packages the current production wedge:

  * local deterministic text memory (`remember` / `recall`)
  * agent skill routing through the existing capability catalog (`route`)
  * an evidence snapshot and static HTML dashboard (`dashboard`)

It does not replace `UnifiedMind` or hide the research surface. It is a small,
boring facade over the stable pieces, meant to be easy to install, test, demo,
and embed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import importlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from holographic.caching_and_storage.holographic_index import Index
from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder


_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: Any) -> List[str]:
    """Deterministic product tokenization: lower-case content tokens, no hidden NLP dependency."""
    if isinstance(text, (list, tuple)):
        return [str(t).lower() for t in text if str(t).strip()]
    return _WORD_RE.findall(str(text).lower())


@dataclass
class MemoryEntry:
    """One stored memory item."""

    id: str
    text: str
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation of this memory entry."""
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Build a memory entry from `to_dict` data."""
        return cls(
            id=str(data["id"]),
            text=str(data.get("text", "")),
            label=data.get("label"),
            metadata=dict(data.get("metadata") or {}),
        )


class LocalAgentCore:
    """Product facade for local agent memory, skill routing, and evidence.

    The API is deliberately small:

        core = LocalAgentCore()
        core.remember("local agents need deterministic memory", label="memory")
        core.recall("deterministic local memory")
        core.route("render a scene")
        core.dashboard()

    Text memory uses the existing `UniversalEncoder` and `Index` homes. It is
    deterministic, local-only, and query-safe: `recall()` does not mutate the
    stored corpus or teach the encoder new query words.
    """

    def __init__(self, dim: int = 512, seed: int = 0, route_threshold: float = 0.6):
        self.dim = int(dim)
        self.seed = int(seed)
        self.route_threshold = float(route_threshold)
        self._entries: List[MemoryEntry] = []
        self._encoder = UniversalEncoder(self.dim, seed=self.seed)
        self._vectors: Optional[np.ndarray] = None
        self._index: Optional[Index] = None
        self._next_id = 1

    # ---- memory ---------------------------------------------------------------------------------------
    @property
    def entries(self) -> List[MemoryEntry]:
        """A copy of the stored entries, in insertion order."""
        return list(self._entries)

    def memory_summary(self) -> Dict[str, Any]:
        """Return constant-time memory status without running evidence probes."""
        return {
            "entries": len(self._entries),
            "dim": self.dim,
            "index_method": self._index.method if self._index is not None else None,
            "query_mutates_store": False,
        }

    def remember(
        self,
        text: Any,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store one local memory. Returns the stored entry as a plain dict."""
        entry_id = str(id) if id is not None else self._allocate_id()
        if any(e.id == entry_id for e in self._entries):
            raise ValueError("memory id already exists: %s" % entry_id)
        entry = MemoryEntry(entry_id, str(text), label, dict(metadata or {}))
        self._entries.append(entry)
        self._rebuild_index()
        return entry.to_dict()

    def remember_many(self, items: Iterable[Any]) -> List[Dict[str, Any]]:
        """Store several memories. Each item may be text or a dict with text/label/metadata/id."""
        stored = []
        for item in items:
            if isinstance(item, dict):
                stored.append(self.remember(
                    item.get("text", ""),
                    label=item.get("label"),
                    metadata=item.get("metadata"),
                    id=item.get("id"),
                ))
            else:
                stored.append(self.remember(item))
        return stored

    def recall(self, query: Any, k: int = 3, abstain: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return the nearest stored memories for `query`, best first.

        `abstain` is passed to `Index.nearest`; when set, noisy matches can
        return an empty list instead of a guess.
        """
        if not self._entries or self._index is None:
            return []
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
            raise ValueError("k must be a positive integer")
        if abstain is not None:
            if isinstance(abstain, bool) or not isinstance(abstain, (int, float, np.number)):
                raise ValueError("abstain must be a number between 0 and 1")
            if not 0.0 <= float(abstain) <= 1.0:
                raise ValueError("abstain must be between 0 and 1")
        if not _tokens(query):
            return []
        q = self._encode_text(query)
        hits = self._index.nearest(q, k=min(int(k), len(self._entries)), abstain=abstain)
        by_id = {entry.id: entry for entry in self._entries}
        out = []
        for entry_id, score in hits:
            entry = by_id[str(entry_id)]
            row = entry.to_dict()
            row["score"] = float(score)
            out.append(row)
        return out

    # ---- agent routing --------------------------------------------------------------------------------
    def suggest(self, task: str, k: int = 5) -> List[Dict[str, Any]]:
        """Suggest capabilities for a plain-English task."""
        from holographic.misc import holographic_skills as skills

        return skills.suggest(task, k=k)

    def route(self, task: str) -> Dict[str, Any]:
        """Route a task to one capability when confident, otherwise return options."""
        from holographic.misc import holographic_skills as skills

        routed = skills.route(task, act_threshold=self.route_threshold)
        out = {"task": str(task)}
        out.update(routed)
        return out

    # ---- evidence / dashboard -------------------------------------------------------------------------
    def evidence(self) -> Dict[str, Any]:
        """Return a machine-readable product readiness snapshot."""
        from holographic.caching_and_storage.holographic_catalog import default_catalog

        c_kernel = self._c_kernel_status()
        route_probe = self.route("search a big pile of vectors")
        return {
            "name": "leCore LocalAgentCore",
            "status": "ready" if self._deterministic_probe() else "check",
            "memory": {
                "entries": len(self._entries),
                "dim": self.dim,
                "index_method": self._index.method if self._index is not None else None,
                "query_mutates_store": False,
            },
            "routing": {
                "capabilities": len(default_catalog()),
                "probe_decision": route_probe.get("decision"),
                "probe_skill": (route_probe.get("skill") or {}).get("name"),
            },
            "c_kernel": c_kernel,
            "checks": {
                "deterministic_encoding": self._deterministic_probe(),
                "local_only": True,
                "no_model_weights": True,
            },
        }

    def dashboard(self, html: bool = False) -> Any:
        """Return the evidence dashboard as a dict, or static HTML with `html=True`."""
        data = self.evidence()
        return self.dashboard_html(data) if html else data

    @staticmethod
    def dashboard_html(data: Dict[str, Any]) -> str:
        """Render an evidence snapshot as a dependency-free static HTML dashboard."""
        memory = data.get("memory", {})
        routing = data.get("routing", {})
        c_kernel = data.get("c_kernel", {})
        checks = data.get("checks", {})

        def esc(value: Any) -> str:
            return html.escape("" if value is None else str(value))

        rows = [
            ("Status", data.get("status")),
            ("Memories", memory.get("entries")),
            ("Dimension", memory.get("dim")),
            ("Index", memory.get("index_method") or "empty"),
            ("Capabilities", routing.get("capabilities")),
            ("Route Probe", "%s: %s" % (routing.get("probe_decision"), routing.get("probe_skill"))),
            ("C Kernel", "available" if c_kernel.get("available") else "not built"),
            ("C Path", c_kernel.get("path") or ""),
            ("Deterministic", checks.get("deterministic_encoding")),
            ("Local Only", checks.get("local_only")),
            ("No Model Weights", checks.get("no_model_weights")),
        ]
        body = "\n".join(
            "<tr><th>%s</th><td>%s</td></tr>" % (esc(k), esc(v))
            for k, v in rows
        )
        return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>leCore LocalAgentCore Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; }
    main { max-width: 760px; }
    h1 { font-size: 1.7rem; margin-bottom: 0.25rem; }
    p { color: #52616b; }
    table { border-collapse: collapse; width: 100%%; margin-top: 1.25rem; }
    th, td { border-bottom: 1px solid #d8dee4; padding: 0.65rem 0.4rem; text-align: left; }
    th { width: 11rem; color: #334155; }
  </style>
</head>
<body>
  <main>
    <h1>leCore LocalAgentCore</h1>
    <p>Local deterministic memory, skill routing, and readiness evidence.</p>
    <table>
      %s
    </table>
  </main>
</body>
</html>""" % body

    # ---- persistence ----------------------------------------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        """Serialize configuration and entries. Vectors are seed/context-derived and rebuilt on load."""
        return {
            "dim": self.dim,
            "seed": self.seed,
            "route_threshold": self.route_threshold,
            "next_id": self._next_id,
            "entries": [entry.to_dict() for entry in self._entries],
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "LocalAgentCore":
        """Rebuild a core from `to_state` data."""
        core = cls(
            dim=int(state.get("dim", 512)),
            seed=int(state.get("seed", 0)),
            route_threshold=float(state.get("route_threshold", 0.6)),
        )
        core._entries = [MemoryEntry.from_dict(row) for row in state.get("entries", [])]
        core._next_id = int(state.get("next_id", len(core._entries) + 1))
        core._rebuild_index()
        return core

    def save(self, path: Any) -> str:
        """Atomically write the product state to JSON and return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_state(), indent=2, sort_keys=True)
        fd, temporary = tempfile.mkstemp(prefix=".%s." % p.name, suffix=".tmp", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, p)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return str(p)

    @classmethod
    def load(cls, path: Any) -> "LocalAgentCore":
        """Load a product state saved by `save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state(data)

    # ---- internals ------------------------------------------------------------------------------------
    def _allocate_id(self) -> str:
        entry_id = "m%d" % self._next_id
        self._next_id += 1
        return entry_id

    def _encode_text(self, text: Any) -> np.ndarray:
        toks = _tokens(text)
        return self._encoder.encode(toks, modality="text")

    def _rebuild_index(self) -> None:
        self._encoder = UniversalEncoder(self.dim, seed=self.seed)
        for entry in self._entries:
            toks = _tokens(entry.text)
            if toks:
                self._encoder.learn_text([toks])
        if not self._entries:
            self._vectors = None
            self._index = None
            return
        self._vectors = np.stack([self._encode_text(entry.text) for entry in self._entries])
        self._index = Index(self._vectors, labels=[entry.id for entry in self._entries], method="exact", seed=self.seed)

    def _deterministic_probe(self) -> bool:
        a = self._encode_text("deterministic local memory")
        b = self._encode_text("deterministic local memory")
        return bool(np.allclose(a, b))

    @staticmethod
    def _c_kernel_status() -> Dict[str, Any]:
        try:
            holographic_c = importlib.import_module("holographic_c")

            return {
                "available": bool(holographic_c.available()),
                "path": holographic_c.backend_path(),
            }
        except Exception as exc:  # pragma: no cover - defensive dashboard reporting
            return {"available": False, "path": None, "error": "%s: %s" % (type(exc).__name__, exc)}


def demo() -> LocalAgentCore:
    """Build a tiny ready-to-query product demo."""
    core = LocalAgentCore(dim=512, seed=0)
    core.remember("local agents need deterministic durable memory", label="agent-memory")
    core.remember("capability routing should act when confident and choose when ambiguous", label="routing")
    core.remember("the C kernel accelerates the audited vector algebra hot path", label="c-kernel")
    return core


def _selftest() -> None:
    core = demo()
    assert core.recall("deterministic local memory")[0]["label"] == "agent-memory"
    assert core.route("start pause resume cancel a job")["decision"] == "act"
    assert core.dashboard()["checks"]["deterministic_encoding"]
    print("OK: holographic_product self-test passed")


if __name__ == "__main__":
    _selftest()
