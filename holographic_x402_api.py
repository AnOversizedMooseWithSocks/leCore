"""holographic_x402_api.py -- publish LocalAgentCore as an x402-paid API.

WHY THIS EXISTS
---------------
`LocalAgentCore` is the narrow product wedge. This module makes it sellable as
an HTTP API without making x402, FastAPI, or uvicorn core dependencies.

The boundary is intentionally conservative:

  * public read/compute routes are x402-paid
  * health/pricing routes are free
  * memory writes are admin-token gated, not pay-to-write

That keeps the paid surface useful while preventing customers from poisoning a
shared memory store just because they paid for one request.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
from html import escape
import argparse
import json
import logging
import os
from pathlib import Path
import queue
import re
from string import Template
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from holographic_product import LocalAgentCore, demo


DEFAULT_FACILITATOR_URL = "https://x402.org/facilitator"
DEFAULT_NETWORK = "eip155:84532"  # Base Sepolia, safe default for testnet publishing.
DEFAULT_PRICE = "$0.0011"
DEFAULT_TENANT_ID = "public"
TENANT_HEADER = "X-leCore-Tenant"
TENANT_TOKEN_HEADER = "X-leCore-Tenant-Token"
IDEMPOTENCY_HEADER = "Idempotency-Key"
_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
MAX_QUERY_CHARS = 8192
MAX_TASK_CHARS = 8192
MAX_MEMORY_CHARS = 65536
MAX_RECALL_K = 100
MEMORY_BACKEND_CORE = "core"
MEMORY_BACKEND_NOSQLITE = "nosqlite"
NOSQLITE_ENCODER = "lecore_text"
NOSQLITE_INDEX = "embedding_neural"
NOSQLITE_DIMENSIONS = 384


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaidRoute:
    """One x402-protected route."""

    method: str
    path: str
    description: str
    price: Optional[str] = None
    mime_type: str = "application/json"

    @property
    def key(self) -> str:
        """The route key shape expected by x402 middleware, e.g. `POST /v1/recall`."""
        return "%s %s" % (self.method.upper(), self.path)


REGULAR_PAID_ROUTES: Tuple[PaidRoute, ...] = (
    PaidRoute("POST", "/v1/recall", "Recall nearest memories from a LocalAgentCore instance"),
    PaidRoute("POST", "/v1/route", "Route a plain-English task to a leCore capability"),
    PaidRoute("GET", "/v1/dashboard", "Read the LocalAgentCore evidence dashboard"),
)

DEFAULT_PAID_ROUTES: Tuple[PaidRoute, ...] = REGULAR_PAID_ROUTES
TESTNET_NETWORKS = frozenset({"eip155:84532"})


def _price_amount(price: str) -> Decimal:
    """Parse a dollar-denominated x402 price without a floating-point round trip."""
    try:
        amount = Decimal(price[1:])
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("x402 price must be a positive dollar amount, e.g. '$0.001'") from exc
    if not amount.is_finite() or amount <= 0:
        raise ValueError("x402 price must be a positive dollar amount, e.g. '$0.001'")
    return amount


LANDING_PAGE_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>leCore x402 API</title>
<meta name="description" content="Try local agent memory, capability routing, and readiness dashboards as x402-paid HTTP primitives.">
<style>
:root{--paper:#fbfaf5;--ink:#171714;--muted:#6f6b61;--line:#ded8c8;--acid:#b7ff3c;--cyan:#28d6ff;--coral:#ff6b57;--gold:#f4c542;--graphite:#20201c}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:inherit;text-decoration:none}
.hero{background:var(--graphite);color:var(--paper);display:grid;grid-template-columns:minmax(0,1.08fr) minmax(320px,.92fr);min-height:88vh;overflow:hidden;padding:28px clamp(20px,5vw,72px) 54px;position:relative}
.field{inset:0;overflow:hidden;position:absolute}.field:before{background-image:linear-gradient(rgba(251,250,245,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(251,250,245,.08) 1px,transparent 1px);background-size:56px 56px;content:"";inset:-80px;opacity:.45;position:absolute;transform:rotate(-7deg)}.field:after{background:radial-gradient(circle at 18% 20%,rgba(183,255,60,.42),transparent 25%),radial-gradient(circle at 70% 22%,rgba(40,214,255,.35),transparent 28%),radial-gradient(circle at 78% 74%,rgba(255,107,87,.28),transparent 26%),linear-gradient(135deg,rgba(32,32,28,.12),rgba(32,32,28,.95));content:"";inset:0;position:absolute}
.trace{border:1px solid rgba(251,250,245,.16);border-radius:999px;position:absolute}.trace.a{height:52vw;right:-14vw;top:5vw;width:52vw}.trace.b{border-color:rgba(183,255,60,.24);height:34vw;left:38vw;top:25vh;width:34vw}.trace.c{border-color:rgba(255,107,87,.2);height:42vw;left:-16vw;top:42vh;width:42vw}
.node{animation:pulse 5s ease-in-out infinite;background:var(--acid);border-radius:999px;box-shadow:0 0 18px currentColor;color:var(--acid);height:var(--size);left:var(--x);opacity:.82;position:absolute;top:var(--y);width:var(--size);z-index:1}.node:nth-child(3n){background:var(--cyan);color:var(--cyan)}.node:nth-child(5n){background:var(--coral);color:var(--coral)}
@keyframes pulse{0%,100%{transform:translate3d(0,0,0) scale(.84)}50%{transform:translate3d(12px,-10px,0) scale(1.18)}}
.topbar{align-items:center;display:flex;gap:24px;grid-column:1/-1;justify-content:space-between;position:relative;z-index:2}.brand,.nav{align-items:center;display:flex}.brand{font-weight:760;gap:10px}.mark{align-items:center;background:var(--acid);color:var(--ink);display:inline-flex;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;height:34px;justify-content:center;width:34px}.nav{color:rgba(251,250,245,.78);font-size:14px;gap:18px}
.copy{align-self:center;max-width:760px;padding:90px 0 34px;position:relative;z-index:2}.status{color:rgba(251,250,245,.78);display:flex;flex-wrap:wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;gap:10px;letter-spacing:0;margin:0 0 20px;text-transform:uppercase}.status span{border:1px solid rgba(251,250,245,.22);padding:8px 10px}
h1,h2,h3,p{margin-top:0}h1{font-size:clamp(56px,9vw,124px);line-height:.9;margin-bottom:24px;max-width:900px}.lede{color:rgba(251,250,245,.82);font-size:clamp(19px,2.2vw,28px);line-height:1.32;max-width:720px}.actions,.close-actions{display:flex;flex-wrap:wrap;gap:12px;margin-top:34px}.button{align-items:center;border:1px solid currentColor;display:inline-flex;font-weight:720;min-height:48px;padding:14px 18px;transition:transform 160ms ease,background 160ms ease,color 160ms ease}.button:hover{transform:translateY(-2px)}.primary{background:var(--acid);border-color:var(--acid);color:var(--ink)}.secondary{color:var(--paper)}.secondary.dark{color:var(--ink)}
.terminal{align-self:end;background:rgba(251,250,245,.94);border:1px solid rgba(251,250,245,.26);box-shadow:0 28px 90px rgba(0,0,0,.3);color:var(--ink);margin-bottom:28px;max-width:460px;position:relative;z-index:2}.terminal-top{align-items:center;border-bottom:1px solid var(--line);display:flex;gap:7px;padding:12px 14px}.terminal-top span{background:var(--coral);border-radius:999px;height:10px;width:10px}.terminal-top span:nth-child(2){background:var(--gold)}.terminal-top span:nth-child(3){background:var(--cyan)}pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:clamp(13px,1.3vw,16px);line-height:1.6;margin:0;overflow-x:auto;padding:22px;white-space:pre-wrap}
.strip{background:var(--acid);color:var(--ink);display:grid;grid-template-columns:repeat(3,1fr)}.strip div{border-right:1px solid rgba(23,23,20,.22);padding:18px clamp(18px,4vw,54px)}.strip div:last-child{border-right:0}.strip strong,.strip span{display:block}.strip strong{font-size:13px;text-transform:uppercase}.strip span{font-size:clamp(18px,2vw,28px);font-weight:760;margin-top:5px}
.section{padding:clamp(68px,9vw,128px) clamp(20px,5vw,72px)}.split,.proof{display:grid;gap:clamp(32px,6vw,80px);grid-template-columns:minmax(0,.9fr) minmax(320px,1.1fr)}.eyebrow{color:var(--coral);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;font-weight:720;letter-spacing:0;margin-bottom:16px;text-transform:uppercase}.section h2{color:var(--ink);font-size:clamp(34px,5vw,72px);line-height:.98;margin-bottom:0;max-width:930px}.reason-list{border-top:1px solid var(--line)}.reason-list p{border-bottom:1px solid var(--line);color:#3a372f;font-size:clamp(18px,2vw,25px);line-height:1.35;margin:0;padding:22px 0}.heading{margin-bottom:clamp(28px,5vw,56px)}
.routes,.cases{display:grid;gap:14px}.routes{grid-template-columns:repeat(3,minmax(0,1fr))}.card,.case,.proof-panel{background:#fff;border:1px solid var(--line);border-radius:8px}.card{min-height:315px;padding:24px}.method{color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;margin-bottom:42px}.card h3{font-size:30px;margin-bottom:10px}.card code{background:#f2eee3;display:inline-block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;margin-bottom:18px;padding:7px 9px}.card p:last-child{color:var(--muted);font-size:17px;line-height:1.48}
.use{background:#f2eee3}.cases{grid-template-columns:repeat(4,minmax(0,1fr))}.case{display:flex;gap:16px;min-height:220px;padding:22px}.case span{color:var(--coral);font-size:30px;line-height:1}.case p{color:#3b382f;font-size:18px;line-height:1.42}
.proof{background:var(--graphite);color:var(--paper)}.proof h2,.proof p{color:var(--paper)}.proof-copy p:last-child{color:rgba(251,250,245,.75);font-size:19px;line-height:1.55;margin-top:26px;max-width:690px}.proof-panel{background:rgba(251,250,245,.96);color:var(--ink);padding:8px}.proof-panel dl{display:grid;gap:1px;grid-template-columns:repeat(2,1fr);margin:0}.proof-panel div{background:#fbfaf5;min-height:150px;padding:22px}.proof-panel dt{color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;margin-bottom:28px;text-transform:uppercase}.proof-panel dd{font-size:clamp(24px,3vw,40px);font-weight:760;margin:0;overflow-wrap:anywhere}
.close{align-items:center;display:grid;grid-template-columns:minmax(0,1fr) auto}.close h2{max-width:980px}.close-actions{justify-content:flex-end;margin-top:0}
@media(max-width:980px){.hero,.split,.proof,.close{grid-template-columns:1fr}.hero{min-height:auto}.terminal{align-self:start;margin-bottom:0;max-width:100%}.routes,.cases{grid-template-columns:1fr}.case,.card{min-height:0}.close-actions{justify-content:flex-start;margin-top:24px}}
@media(max-width:680px){.hero{padding:20px 18px 36px}.topbar,.nav,.strip{align-items:flex-start;display:grid;grid-template-columns:1fr}.nav{gap:8px}.copy{padding-top:62px}.status span,.button{width:100%}.strip div{border-bottom:1px solid rgba(23,23,20,.22);border-right:0}.proof-panel dl{grid-template-columns:1fr}.section{padding-left:18px;padding-right:18px}}
</style>
</head>
<body>
<main>
<section class="hero" aria-labelledby="hero-title">
<div class="field" aria-hidden="true"><span class="trace a"></span><span class="trace b"></span><span class="trace c"></span>$nodes</div>
<nav class="topbar" aria-label="Primary"><a class="brand" href="#hero-title" aria-label="leCore x402 API home"><span class="mark">lc</span><span>leCore x402 API</span></a><div class="nav"><a href="#why">Why try it</a><a href="#routes">Routes</a><a href="#proof">Preview status</a></div></nav>
<div class="copy"><p class="status"><span>$environment_label</span><span>$network_label</span><span>$price_per_thousand</span></p><h1 id="hero-title">leCore x402 API</h1><p class="lede">Try local agent memory, capability routing, and a readiness dashboard as paid HTTP primitives. $payment_notice</p><div class="actions"><a class="button primary" href="/pricing">View preview terms</a><a class="button secondary" href="/v1/dashboard">See x402 challenge</a></div></div>
<aside class="terminal" aria-label="Live API snapshot"><div class="terminal-top"><span></span><span></span><span></span></div><pre>GET /health
200 OK
memory.entries: 3
paid: true

GET /v1/dashboard
402 Payment Required
network: $network
asset: $payment_asset</pre></aside>
</section>
<section class="strip" aria-label="Live deployment details"><div><strong>Endpoint</strong><span>https://lecore.rati.foundation</span></div><div><strong>Stage</strong><span>$environment_label</span></div><div><strong>Buyer shape</strong><span>inspect, pay, call</span></div></section>
<section id="why" class="section split"><div><p class="eyebrow">Why try it</p><h2>Most agents do not need a platform. They need a few reliable cognitive calls.</h2></div><div class="reason-list"><p>Test the x402 payment flow against one answerable primitive at a time.</p><p>The API is narrow enough to trust: read/compute routes are paid, memory writes stay admin-gated.</p><p>It exposes the useful part of leCore first: local agent memory plus capability routing.</p><p>$payment_notice</p></div></section>
<section id="routes" class="section"><div class="heading"><p class="eyebrow">What the payment unlocks</p><h2>Three paid routes, each small enough to understand.</h2></div><div class="routes"><article class="card"><p class="method">POST</p><h3>Recall</h3><code>/v1/recall</code><p>Pull nearest memories from a compact local agent core without shipping a whole application stack.</p></article><article class="card"><p class="method">POST</p><h3>Route</h3><code>/v1/route</code><p>Send a plain-language task and get the leCore capability it should use, with evidence attached.</p></article><article class="card"><p class="method">GET</p><h3>Dashboard</h3><code>/v1/dashboard</code><p>Read the readiness surface: memory counts, capability map, abstention behavior, and route coverage.</p></article></div></section>
<section class="section use"><div class="heading"><p class="eyebrow">Good first buyers</p><h2>Teams who want the leCore idea without adopting the whole repo.</h2></div><div class="cases"><article class="case"><span aria-hidden="true">+</span><p>Agent memory for prototypes that should remember without a database rollout.</p></article><article class="case"><span aria-hidden="true">+</span><p>Capability routing for tools that need to pick the right leCore subsystem before doing work.</p></article><article class="case"><span aria-hidden="true">+</span><p>Evidence dashboards for teams deciding whether a local vector system is ready to productize.</p></article><article class="case"><span aria-hidden="true">+</span><p>A working x402 seller endpoint to copy when you want pay-per-call APIs instead of subscriptions.</p></article></div></section>
<section id="proof" class="section proof"><div class="proof-copy"><p class="eyebrow">Preview status</p><h2>It is deployed, health-checked, and ready to integrate.</h2><p>The free endpoints show health and preview terms. Paid endpoints return an x402 challenge on $network_name. The receiving address is public, while admin writes stay out of the customer path.</p></div><div class="proof-panel"><dl><div><dt>Preview price</dt><dd>$price_per_thousand</dd></div><div><dt>Network</dt><dd>$network_name</dd></div><div><dt>Receiver</dt><dd>$pay_to_short</dd></div><div><dt>Status</dt><dd>Healthy</dd></div></dl></div></section>
<section class="section close"><p class="eyebrow">The pitch</p><h2>Try it when you want a local-memory agent primitive without adopting the whole repo.</h2><div class="close-actions"><a class="button primary" href="/pricing">Inspect preview terms</a><a class="button secondary dark" href="/health">Check live health</a></div></section>
</main>
</body>
</html>""")


@dataclass(frozen=True)
class X402Config:
    """Seller configuration for the x402-paid API."""

    pay_to: str
    price: str = DEFAULT_PRICE
    network: str = DEFAULT_NETWORK
    facilitator_url: str = DEFAULT_FACILITATOR_URL
    scheme: str = "exact"
    routes: Tuple[PaidRoute, ...] = DEFAULT_PAID_ROUTES

    def __post_init__(self) -> None:
        if not self.pay_to:
            raise ValueError("pay_to is required")
        if not self.price.startswith("$"):
            raise ValueError("x402 price must include a dollar prefix, e.g. '$0.001'")
        _price_amount(self.price)
        if not self.network:
            raise ValueError("network is required")
        if not self.facilitator_url:
            raise ValueError("facilitator_url is required")

    @classmethod
    def from_env(cls, require_pay_to: bool = True) -> "X402Config":
        """Build config from LECORE_X402_* environment variables."""
        pay_to = os.environ.get("LECORE_X402_PAY_TO", "")
        if require_pay_to and not pay_to:
            raise ValueError("set LECORE_X402_PAY_TO to the receiving wallet address")
        return cls(
            pay_to=pay_to or "0xYourAddress",
            price=os.environ.get("LECORE_X402_PRICE", DEFAULT_PRICE),
            network=os.environ.get("LECORE_X402_NETWORK", DEFAULT_NETWORK),
            facilitator_url=os.environ.get("LECORE_X402_FACILITATOR_URL", DEFAULT_FACILITATOR_URL),
        )

    def to_public_dict(self) -> Dict[str, Any]:
        """Public, JSON-safe view of the payment configuration."""
        return {
            "pay_to": self.pay_to,
            "price": self.price,
            "network": self.network,
            "facilitator_url": self.facilitator_url,
            "scheme": self.scheme,
        }


def optional_dependency_help() -> str:
    """Install hint for the optional paid API dependencies."""
    return 'Install the optional API dependencies with: pip install ".[x402]" (includes FastAPI and EVM x402 support)'


def _landing_nodes() -> str:
    """CSS-positioned visual nodes for the marketing page hero."""
    nodes = []
    for index in range(34):
        size = 9 if index % 5 == 0 else 7 if index % 3 == 0 else 5
        nodes.append(
            '<i class="node" style="--x:%s%%;--y:%s%%;--delay:%ss;--size:%spx"></i>'
            % ((index * 29) % 100, (index * 47 + 11) % 100, (index % 9) * -0.45, size)
        )
    return "".join(nodes)


def _short_address(address: str) -> str:
    """Compact public wallet display."""
    if len(address) <= 12:
        return address
    return "%s...%s" % (address[:6], address[-4:])


def _network_name(network: str) -> str:
    """Human label for known x402 network ids."""
    return {"eip155:84532": "Base Sepolia", "eip155:8453": "Base"}.get(network, network)


def normalize_tenant_id(value: Optional[Any]) -> str:
    """Return a path-safe tenant id for private memory routing."""
    if value is None:
        return DEFAULT_TENANT_ID
    if not isinstance(value, str):
        raise ValueError("tenant id must be a string")
    tenant_id = value.strip().lower()
    if not tenant_id:
        tenant_id = DEFAULT_TENANT_ID
    if not _TENANT_ID_RE.match(tenant_id):
        raise ValueError("tenant id must be 1-64 chars of lowercase letters, numbers, '.', ':', '_' or '-'")
    return tenant_id


def tenant_access_token(tenant_id: str, secret: str) -> str:
    """Deterministic tenant bearer token derived from a server-side secret."""
    normalized = normalize_tenant_id(tenant_id)
    return hmac.new(secret.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256).hexdigest()


def normalize_idempotency_key(value: Optional[Any]) -> Optional[str]:
    """Validate an optional caller-provided retry key without persisting the raw value."""
    if value is None:
        return None
    if not isinstance(value, str) or not _IDEMPOTENCY_KEY_RE.match(value):
        raise ValueError("Idempotency-Key must be 1-256 letters, numbers, '.', '_', ':', or '-'")
    return value


@contextmanager
def _process_file_lock(path: Path) -> Any:
    """Hold an exclusive process lock for one persisted tenant state file."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows CI/users
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI/users
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    """Durably replace a small JSON control record without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.urandom(8).hex()))
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


class TenantCoreStore:
    """Thread-safe LocalAgentCore registry with optional per-tenant persistence."""

    def __init__(
        self,
        default_core: LocalAgentCore,
        state_dir: Optional[Any] = None,
    ):
        self._default_dim = default_core.dim
        self._default_seed = default_core.seed
        self._default_route_threshold = default_core.route_threshold
        self._cores: Dict[str, LocalAgentCore] = {DEFAULT_TENANT_ID: default_core}
        self._versions: Dict[str, Tuple[int, int, int]] = {}
        self._tenant_locks: Dict[str, threading.RLock] = {DEFAULT_TENANT_ID: threading.RLock()}
        self._registry_lock = threading.RLock()
        self._state_dir = Path(state_dir) if state_dir else None
        if self._state_dir is not None:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            public_path = self._path_for(DEFAULT_TENANT_ID)
            if public_path is not None and public_path.exists():
                with _process_file_lock(public_path):
                    self._cores[DEFAULT_TENANT_ID] = LocalAgentCore.load(public_path)
                    self._versions[DEFAULT_TENANT_ID] = self._version(public_path)

    def loaded_tenants(self) -> List[str]:
        """Return tenant ids currently loaded in memory."""
        with self._registry_lock:
            return sorted(self._cores)

    def summary(self, tenant_id: str) -> Dict[str, Any]:
        """Return a cheap cached status summary without probing capabilities."""
        normalized = normalize_tenant_id(tenant_id)
        with self._lock_for(normalized):
            core = self._get_cached(normalized)
            return core.memory_summary()

    def read(self, tenant_id: str, fn: Any) -> Any:
        """Run a read-style operation while holding the tenant lock."""
        normalized = normalize_tenant_id(tenant_id)
        with self._lock_for(normalized):
            return fn(self._get_fresh(normalized))

    def write(self, tenant_id: str, fn: Any) -> Any:
        """Run a mutating operation, then persist that tenant if configured."""
        normalized = normalize_tenant_id(tenant_id)
        with self._lock_for(normalized):
            path = self._path_for(normalized)
            if path is None:
                core = self._get_cached(normalized)
                return fn(core)
            with _process_file_lock(path):
                core = (
                    LocalAgentCore.load(path)
                    if path.exists()
                    else LocalAgentCore.from_state(self._get_cached(normalized).to_state())
                )
                result = fn(core)
                core.save(path)
                with self._registry_lock:
                    self._cores[normalized] = core
                    self._versions[normalized] = self._version(path)
                return result

    def _lock_for(self, tenant_id: str) -> threading.RLock:
        with self._registry_lock:
            lock = self._tenant_locks.get(tenant_id)
            if lock is None:
                lock = threading.RLock()
                self._tenant_locks[tenant_id] = lock
            return lock

    def _get_cached(self, tenant_id: str) -> LocalAgentCore:
        with self._registry_lock:
            core = self._cores.get(tenant_id)
            if core is None:
                core = LocalAgentCore(
                    dim=self._default_dim,
                    seed=self._default_seed,
                    route_threshold=self._default_route_threshold,
                )
                self._cores[tenant_id] = core
            return core

    def _get_fresh(self, tenant_id: str) -> LocalAgentCore:
        path = self._path_for(tenant_id)
        if path is not None and path.exists():
            version = self._version(path)
            with self._registry_lock:
                cached_version = self._versions.get(tenant_id)
            if cached_version != version:
                with _process_file_lock(path):
                    core = LocalAgentCore.load(path)
                    version = self._version(path)
                with self._registry_lock:
                    self._cores[tenant_id] = core
                    self._versions[tenant_id] = version
                return core
        return self._get_cached(tenant_id)

    @staticmethod
    def _version(path: Path) -> Tuple[int, int, int]:
        stat = path.stat()
        return stat.st_ino, stat.st_mtime_ns, stat.st_size

    def _path_for(self, tenant_id: str) -> Optional[Path]:
        if self._state_dir is None:
            return None
        return self._state_dir / ("%s.json" % normalize_tenant_id(tenant_id))


class NoSQLiteError(RuntimeError):
    """Raised when the optional NoSQLite command process cannot serve a request."""


class NoSQLiteProcess:
    """Serialize JSON-line requests to one long-lived NoSQLite CLI process.

    NoSQLite's filesystem mode intentionally takes an exclusive writer lock for
    the life of the process. The API therefore keeps exactly one child process
    per application process and serializes its stdin/stdout protocol here.
    """

    def __init__(
        self,
        binary: str,
        data_dir: Any,
        durability: str = "sync",
        timeout_seconds: float = 10.0,
    ):
        self._binary = str(binary)
        self._data_dir = Path(data_dir)
        self._durability = durability
        self._timeout_seconds = float(timeout_seconds)
        self._lock = threading.RLock()
        self._process: Optional[Any] = None
        self._stdout: Any = queue.Queue()
        self._stderr: Any = queue.Queue()
        self._generation = 0

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def ensure_started(self) -> int:
        """Start the child lazily and return its generation number."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self._generation
            self._stop_process_unlocked()
            command = [self._binary, "--data-dir", str(self._data_dir), "--durability", self._durability]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                raise NoSQLiteError("could not start NoSQLite: %s" % exc) from exc

            self._process = process
            self._stdout = queue.Queue()
            self._stderr = queue.Queue()
            self._start_reader(process.stdout, self._stdout)
            self._start_reader(process.stderr, self._stderr)
            try:
                banner = self._read_line_unlocked("startup")
            except NoSQLiteError:
                self._stop_process_unlocked()
                raise
            if not banner.startswith("nosqlite ready;"):
                self._stop_process_unlocked()
                raise NoSQLiteError("unexpected NoSQLite startup response: %s" % banner.strip())
            self._generation += 1
            return self._generation

    def command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send one command and return the object response from NoSQLite."""
        with self._lock:
            self.ensure_started()
            process = self._process
            if process is None or process.stdin is None:
                raise NoSQLiteError("NoSQLite process has no writable stdin")
            try:
                process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except OSError as exc:
                self._stop_process_unlocked()
                raise NoSQLiteError("failed to send a command to NoSQLite: %s" % exc) from exc
            try:
                line = self._read_line_unlocked("command")
            except NoSQLiteError:
                self._stop_process_unlocked()
                raise
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NoSQLiteError("invalid NoSQLite response: %s" % line.strip()) from exc
            if not isinstance(response, dict):
                raise NoSQLiteError("NoSQLite response must be an object")
            if response.get("ok") == "error":
                raise NoSQLiteError(str(response.get("message") or "unknown NoSQLite error"))
            return response

    def close(self) -> None:
        """Release the child process and its filesystem writer lock."""
        with self._lock:
            process = self._process
            if process is None:
                return
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write('{"shutdown":1}\n')
                    process.stdin.flush()
                    self._read_line_unlocked("shutdown", timeout_seconds=2.0)
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired, NoSQLiteError):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
            finally:
                self._clear_process_unlocked()

    def _start_reader(self, stream: Any, output: Any) -> None:
        def read_lines() -> None:
            if stream is None:
                return
            for line in stream:
                output.put(line)

        threading.Thread(target=read_lines, daemon=True).start()

    def _read_line_unlocked(self, phase: str, timeout_seconds: Optional[float] = None) -> str:
        timeout = self._timeout_seconds if timeout_seconds is None else timeout_seconds
        deadline = time.monotonic() + timeout
        while True:
            process = self._process
            if process is not None and process.poll() is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                return self._stdout.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
        process = self._process
        state = ""
        if process is not None and process.poll() is not None:
            state = " (process exited with code %s)" % process.returncode
        stderr = self._stderr_text_unlocked()
        if stderr:
            state += ": %s" % stderr
        raise NoSQLiteError("NoSQLite %s timed out%s" % (phase, state))

    def _stderr_text_unlocked(self) -> str:
        lines = []
        while True:
            try:
                lines.append(self._stderr.get_nowait().strip())
            except queue.Empty:
                break
        return " ".join(line for line in lines if line)[:2000]

    def _clear_process_unlocked(self) -> None:
        self._process = None

    def _stop_process_unlocked(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._clear_process_unlocked()


class NoSQLiteMemoryStore:
    """Tenant-isolated semantic memory backed by the pinned NoSQLite CLI."""

    def __init__(
        self,
        binary: str,
        data_dir: Any,
        durability: str = "sync",
        dimensions: int = NOSQLITE_DIMENSIONS,
    ):
        if durability not in {"sync", "buffered"}:
            raise ValueError("NoSQLite durability must be 'sync' or 'buffered'")
        self._dimensions = int(dimensions)
        self._process = NoSQLiteProcess(binary, data_dir, durability=durability)
        self._lock = threading.RLock()
        self._encoder_generation: Optional[int] = None
        self._ready_collections: set[str] = set()
        self._synced_collections: set[Tuple[int, str]] = set()

    @property
    def running(self) -> bool:
        return self._process.running

    def remember(self, tenant_id: str, memory: Dict[str, Any]) -> None:
        """Persist one LocalAgentCore-compatible memory entry in its tenant collection."""
        normalized = normalize_tenant_id(tenant_id)
        with self._lock:
            collection = self._ensure_collection(normalized)
            self._insert_memory(collection, normalized, memory)

    def sync(self, tenant_id: str, memories: Iterable[Dict[str, Any]]) -> None:
        """Backfill the durable core mirror once per tenant and CLI generation."""
        normalized = normalize_tenant_id(tenant_id)
        with self._lock:
            collection = self._ensure_collection(normalized)
            key = (self._process.generation, collection)
            if key in self._synced_collections:
                return
            for memory in memories:
                self._insert_memory(collection, normalized, memory)
            self._synced_collections.add(key)

    def recall(
        self,
        tenant_id: str,
        query: str,
        k: int,
        abstain: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return NoSQLite semantic hits in the LocalAgentCore response shape."""
        normalized = normalize_tenant_id(tenant_id)
        with self._lock:
            collection = self._ensure_collection(normalized)
            response = self._process.command({
                "semanticSearch": collection,
                "encoder": NOSQLITE_ENCODER,
                "index": NOSQLITE_INDEX,
                "text": query,
                "k": k,
            })
        documents = response.get("documents")
        if not isinstance(documents, list):
            raise NoSQLiteError("NoSQLite semantic search returned no documents array")
        hits = []
        for document in documents:
            if not isinstance(document, dict):
                continue
            score = document.get("_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                continue
            if abstain is not None and float(score) < abstain:
                continue
            metadata = document.get("metadata")
            label = document.get("label")
            hits.append({
                "id": str(document.get("_id", "")),
                "text": str(document.get("text", "")),
                "label": label if isinstance(label, str) else None,
                "metadata": dict(metadata) if isinstance(metadata, dict) else {},
                "score": float(score),
            })
        return hits

    def close(self) -> None:
        self._process.close()

    def _ensure_collection(self, tenant_id: str) -> str:
        generation = self._process.ensure_started()
        if self._encoder_generation != generation:
            self._ready_collections.clear()
            self._synced_collections.clear()
            self._ignore_duplicate({
                "createEncoder": NOSQLITE_ENCODER,
                "provider": "holographic-hash-v1",
                "kind": "text",
                "dimensions": self._dimensions,
                "seed": 0,
            })
            self._encoder_generation = generation
        collection = self._collection_name(tenant_id)
        if collection not in self._ready_collections:
            self._ignore_duplicate({"create": collection})
            self._ignore_duplicate({
                "createIndexes": collection,
                "indexes": [{
                    "neural": "embedding",
                    "dimensions": self._dimensions,
                    "name": NOSQLITE_INDEX,
                }],
            })
            self._ready_collections.add(collection)
        return collection

    def _ignore_duplicate(self, command: Dict[str, Any]) -> None:
        try:
            self._process.command(command)
        except NoSQLiteError as exc:
            if "already exists" not in str(exc):
                raise

    def _insert_memory(self, collection: str, tenant_id: str, memory: Dict[str, Any]) -> None:
        document = {
            "_id": str(memory["id"]),
            "text": str(memory["text"]),
            "label": memory.get("label"),
            "metadata": dict(memory.get("metadata") or {}),
            "tenant": tenant_id,
        }
        try:
            self._process.command({
                "insert": collection,
                "encode": {"encoder": NOSQLITE_ENCODER, "field": "text", "into": "embedding"},
                "documents": [document],
            })
        except NoSQLiteError as exc:
            if "duplicate value for `_id`" not in str(exc):
                raise

    @staticmethod
    def _collection_name(tenant_id: str) -> str:
        digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:24]
        return "lecore_memory_%s" % digest


class MemoryTransactionError(RuntimeError):
    """The durable memory write journal could not be read or completed safely."""


class MemoryTransactionConflict(MemoryTransactionError):
    """One idempotency key was reused for a different memory write."""


class MemoryMirrorPending(NoSQLiteError):
    """A durable core commit needs the same transaction projected to NoSQLite."""

    def __init__(self, tenant_id: str, transaction_id: str, cause: NoSQLiteError):
        super().__init__(str(cause))
        self.tenant_id = tenant_id
        self.transaction_id = transaction_id


class TenantMemoryTransactions:
    """Durable, idempotent memory writes spanning LocalAgentCore and NoSQLite.

    A query-layer transaction can roll in-memory tables back. This API crosses a
    durable JSON core and an external NoSQLite process, so it instead records an
    intent first, commits the core entry with a stable id, then projects that
    entry to NoSQLite. If the process dies between steps, the journal replays the
    same idempotent projection on the next request or app start.
    """

    _VERSION = 1
    _PLANNED = "planned"
    _CORE_COMMITTED = "core_committed"
    _COMPLETE = "complete"

    def __init__(self, core_store: TenantCoreStore, state_dir: Any):
        self._core_store = core_store
        self._root = Path(state_dir) / ".x402-memory-transactions"
        self._root.mkdir(parents=True, exist_ok=True)

    def remember(
        self,
        tenant_id: str,
        text: str,
        label: Optional[str],
        metadata: Optional[Dict[str, Any]],
        idempotency_key: Optional[str],
        mirror: Optional[NoSQLiteMemoryStore],
    ) -> Dict[str, Any]:
        """Commit one memory and return its stable transaction status.

        Supplying the same `idempotency_key` with the same request returns the
        original memory id. Reusing that key for a different request is refused.
        """
        tenant = normalize_tenant_id(tenant_id)
        key = normalize_idempotency_key(idempotency_key)
        request = {
            "tenant": tenant,
            "text": str(text),
            "label": label,
            "metadata": dict(metadata or {}),
        }
        transaction_id = self._transaction_id(tenant, key)
        path = self._path_for(tenant, transaction_id)
        with _process_file_lock(path):
            record = self._load_or_create(path, transaction_id, request, key, mirror is not None)
            return self._apply_locked(path, record, mirror)

    def resume(
        self,
        tenant_id: str,
        transaction_id: str,
        mirror: Optional[NoSQLiteMemoryStore],
    ) -> Dict[str, Any]:
        """Resume a known journal record without minting a second transaction."""
        tenant = normalize_tenant_id(tenant_id)
        if not re.fullmatch(r"[0-9a-f]{64}", transaction_id):
            raise MemoryTransactionError("invalid memory transaction id")
        path = self._path_for(tenant, transaction_id)
        with _process_file_lock(path):
            record = self._load(path)
            self._validate_record(record, path)
            if record["tenant"] != tenant or record["transaction_id"] != transaction_id:
                raise MemoryTransactionError("memory transaction does not match its tenant")
            return self._apply_locked(path, record, mirror)

    def recover_pending(self, mirror: Optional[NoSQLiteMemoryStore]) -> Dict[str, int]:
        """Replay incomplete durable writes, leaving unavailable mirrors pending."""
        recovered = 0
        pending = 0
        invalid = 0
        for path in sorted(self._root.glob("*/*.json")):
            with _process_file_lock(path):
                try:
                    record = self._load(path)
                    if record.get("state") == self._COMPLETE:
                        continue
                    result = self._apply_locked(path, record, mirror)
                    if result["transaction"]["state"] == self._COMPLETE:
                        recovered += 1
                    else:
                        pending += 1
                except NoSQLiteError as exc:
                    pending += 1
                    LOG.warning("NoSQLite transaction recovery remains pending: %s", exc)
                except MemoryTransactionError as exc:
                    invalid += 1
                    LOG.error("could not recover memory transaction %s: %s", path.name, exc)
        return {"recovered": recovered, "pending": pending, "invalid": invalid}

    def _apply_locked(
        self,
        path: Path,
        record: Dict[str, Any],
        mirror: Optional[NoSQLiteMemoryStore],
    ) -> Dict[str, Any]:
        self._validate_record(record, path)
        memory = dict(record["memory"])
        stored = self._core_store.write(
            record["tenant"],
            lambda core: self._ensure_core_memory(core, memory),
        )
        if record["state"] == self._PLANNED:
            record["state"] = self._CORE_COMMITTED
            _atomic_write_json(path, record)

        if record["requires_mirror"]:
            if mirror is None:
                return self._result(record, stored)
            try:
                mirror.remember(record["tenant"], stored)
            except NoSQLiteError as exc:
                raise MemoryMirrorPending(record["tenant"], record["transaction_id"], exc) from exc

        if record["state"] != self._COMPLETE:
            record["state"] = self._COMPLETE
            _atomic_write_json(path, record)
        return self._result(record, stored)

    @staticmethod
    def _ensure_core_memory(core: LocalAgentCore, memory: Dict[str, Any]) -> Dict[str, Any]:
        for entry in core.entries:
            if entry.id != memory["id"]:
                continue
            stored = entry.to_dict()
            if stored != memory:
                raise MemoryTransactionConflict("memory id %s already holds different content" % memory["id"])
            return stored
        return core.remember(
            memory["text"],
            label=memory.get("label"),
            metadata=memory.get("metadata"),
            id=memory["id"],
        )

    def _load_or_create(
        self,
        path: Path,
        transaction_id: str,
        request: Dict[str, Any],
        key: Optional[str],
        requires_mirror: bool,
    ) -> Dict[str, Any]:
        if path.exists():
            record = self._load(path)
            self._validate_record(record, path)
            if record["request_fingerprint"] != self._fingerprint(request):
                raise MemoryTransactionConflict("Idempotency-Key was already used for a different memory write")
            return record
        record = {
            "version": self._VERSION,
            "transaction_id": transaction_id,
            "tenant": request["tenant"],
            "request_fingerprint": self._fingerprint(request),
            "idempotency_key_hash": self._hash(key) if key is not None else None,
            "requires_mirror": bool(requires_mirror),
            "state": self._PLANNED,
            "memory": {
                "id": "tx_%s" % transaction_id[:32],
                "text": request["text"],
                "label": request["label"],
                "metadata": request["metadata"],
            },
        }
        _atomic_write_json(path, record)
        return record

    @staticmethod
    def _load(path: Path) -> Dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryTransactionError("invalid transaction journal %s" % path.name) from exc
        if not isinstance(value, dict):
            raise MemoryTransactionError("transaction journal %s is not an object" % path.name)
        return value

    def _validate_record(self, record: Dict[str, Any], path: Path) -> None:
        required = {"version", "transaction_id", "tenant", "request_fingerprint", "requires_mirror", "state", "memory"}
        if not required.issubset(record) or record.get("version") != self._VERSION:
            raise MemoryTransactionError("unsupported transaction journal %s" % path.name)
        if record["state"] not in {self._PLANNED, self._CORE_COMMITTED, self._COMPLETE}:
            raise MemoryTransactionError("unknown transaction state in %s" % path.name)
        memory = record["memory"]
        if not isinstance(memory, dict) or set(memory) != {"id", "text", "label", "metadata"}:
            raise MemoryTransactionError("invalid memory transaction payload in %s" % path.name)
        if not isinstance(memory["id"], str) or not isinstance(memory["text"], str):
            raise MemoryTransactionError("invalid memory transaction value in %s" % path.name)
        if memory["label"] is not None and not isinstance(memory["label"], str):
            raise MemoryTransactionError("invalid memory transaction label in %s" % path.name)
        if not isinstance(memory["metadata"], dict):
            raise MemoryTransactionError("invalid memory transaction metadata in %s" % path.name)
        if normalize_tenant_id(record["tenant"]) != record["tenant"]:
            raise MemoryTransactionError("invalid transaction tenant in %s" % path.name)

    def _path_for(self, tenant_id: str, transaction_id: str) -> Path:
        tenant_digest = self._hash(tenant_id)[:24]
        return self._root / tenant_digest / (transaction_id + ".json")

    @staticmethod
    def _hash(value: Optional[str]) -> str:
        return hashlib.sha256((value or "").encode("utf-8")).hexdigest()

    def _transaction_id(self, tenant_id: str, key: Optional[str]) -> str:
        material = key if key is not None else os.urandom(32).hex()
        return self._hash("%s\0%s" % (tenant_id, material))

    @classmethod
    def _fingerprint(cls, value: Dict[str, Any]) -> str:
        return cls._hash(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))

    @staticmethod
    def _result(record: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memory": memory,
            "transaction": {
                "id": record["transaction_id"],
                "state": record["state"],
                "idempotent": record.get("idempotency_key_hash") is not None,
            },
        }


def pricing_summary(config: X402Config) -> Dict[str, Any]:
    """Describe the customer-facing price and whether it is a production charge."""
    per_thousand = _price_amount(config.price) * Decimal("1000")
    per_thousand_display = "$%s" % per_thousand.quantize(Decimal("0.01"))
    testnet = config.network in TESTNET_NETWORKS
    environment = "testnet_preview" if testnet else "production"
    payment_asset = "testnet USDC" if testnet else "USDC"
    payment_notice = (
        "This Base Sepolia developer preview uses testnet USDC and does not accept production payments."
        if testnet
        else "Payments settle in USDC through x402."
    )
    return {
        "environment": environment,
        "environment_label": "Testnet developer preview" if testnet else "Production API",
        "payment_asset": payment_asset,
        "per_request": config.price,
        "per_1000_requests": per_thousand_display,
        "display_price": "%s per 1,000 requests" % per_thousand_display,
        "payment_notice": payment_notice,
    }


def _required_text(payload: Dict[str, Any], key: str, maximum: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % key)
    if len(value) > maximum:
        raise ValueError("%s must be at most %d characters" % (key, maximum))
    return value


def _recall_k(payload: Dict[str, Any]) -> int:
    value = payload.get("k", 3)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("k must be an integer")
    if not 1 <= value <= MAX_RECALL_K:
        raise ValueError("k must be between 1 and %d" % MAX_RECALL_K)
    return value


def _abstain_threshold(payload: Dict[str, Any]) -> Optional[float]:
    value = payload.get("abstain")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("abstain must be a number between 0 and 1")
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("abstain must be between 0 and 1")
    return threshold


def normalize_memory_backend(value: Any) -> str:
    """Validate the memory backend selector without accepting silent fallbacks."""
    if not isinstance(value, str):
        raise ValueError("memory backend must be a string")
    backend = value.strip().lower() or MEMORY_BACKEND_CORE
    if backend not in {MEMORY_BACKEND_CORE, MEMORY_BACKEND_NOSQLITE}:
        raise ValueError("memory backend must be 'core' or 'nosqlite'")
    return backend


def env_flag(value: Optional[str]) -> bool:
    """Parse the small explicit boolean surface used by deployment settings."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def landing_page_html(config: X402Config) -> str:
    """Render the buyer-facing landing page served from `/`."""
    network_name = _network_name(config.network)
    summary = pricing_summary(config)
    return LANDING_PAGE_TEMPLATE.substitute(
        nodes=_landing_nodes(),
        network=escape(config.network),
        network_label=escape("%s x402" % network_name),
        network_name=escape(network_name),
        pay_to_short=escape(_short_address(config.pay_to)),
        environment_label=escape(summary["environment_label"]),
        payment_asset=escape(summary["payment_asset"]),
        payment_notice=escape(summary["payment_notice"]),
        price_per_thousand=escape(summary["display_price"]),
    )


def payment_manifest(config: X402Config) -> List[Dict[str, Any]]:
    """Plain JSON route manifest, useful for docs, `/pricing`, and tests."""
    out = []
    for route in config.routes:
        price = route.price or config.price
        row = {
            "route": route.key,
            "description": route.description,
            "mime_type": route.mime_type,
            "accepts": [{
                "scheme": config.scheme,
                "price": price,
                "network": config.network,
                "pay_to": config.pay_to,
            }],
        }
        out.append(row)
    return out


def x402_route_configs(config: X402Config) -> Dict[str, Any]:
    """Build x402 SDK RouteConfig objects for the protected routes."""
    try:
        from x402.http import PaymentOption
        from x402.http.types import RouteConfig
    except ImportError as exc:
        raise RuntimeError(optional_dependency_help()) from exc

    routes = {}
    for route in config.routes:
        routes[route.key] = RouteConfig(
            accepts=[
                PaymentOption(
                    scheme=config.scheme,
                    pay_to=config.pay_to,
                    price=route.price or config.price,
                    network=config.network,
                )
            ],
            mime_type=route.mime_type,
            description=route.description,
        )
    return routes


def x402_resource_server(config: X402Config) -> Any:
    """Create an x402 resource server wired to the configured facilitator."""
    try:
        from x402.http import FacilitatorConfig, HTTPFacilitatorClient
        from x402.mechanisms.evm.exact import ExactEvmServerScheme
        from x402.server import x402ResourceServer
    except ImportError as exc:
        raise RuntimeError(optional_dependency_help()) from exc

    facilitator = HTTPFacilitatorClient(FacilitatorConfig(url=config.facilitator_url))
    server = x402ResourceServer(facilitator)
    server.register(config.network, ExactEvmServerScheme())
    return server


def create_app(
    core: Optional[LocalAgentCore] = None,
    config: Optional[X402Config] = None,
    paid: bool = True,
    admin_token: Optional[str] = None,
    tenant_secret: Optional[str] = None,
    tenant_state_dir: Optional[Any] = None,
    memory_backend: Optional[str] = None,
    nosqlite_binary: Optional[str] = None,
    nosqlite_data_dir: Optional[Any] = None,
    nosqlite_durability: Optional[str] = None,
    nosqlite_shadow: Optional[bool] = None,
) -> Any:
    """Create the FastAPI application for paid or local serving.

    With `paid=True`, the public `/v1/*` read/compute routes are protected by
    x402 middleware. Set `paid=False` for local development smoke tests.

    x402 proves that a request paid. Private tenant memory is intentionally a
    separate authorization layer using `X-leCore-Tenant-Token`.
    """
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise RuntimeError(optional_dependency_help()) from exc

    core = core or demo()
    store = TenantCoreStore(core, state_dir=tenant_state_dir)
    memory_backend = normalize_memory_backend(
        memory_backend if memory_backend is not None else os.environ.get("LECORE_X402_MEMORY_BACKEND", MEMORY_BACKEND_CORE)
    )
    nosqlite_shadow = (
        bool(nosqlite_shadow)
        if nosqlite_shadow is not None
        else env_flag(os.environ.get("LECORE_X402_NOSQLITE_SHADOW"))
    )
    nosqlite_store: Optional[NoSQLiteMemoryStore] = None
    if memory_backend == MEMORY_BACKEND_NOSQLITE or nosqlite_shadow:
        if not tenant_state_dir:
            raise ValueError("LECORE_X402_TENANT_STATE_DIR is required when NoSQLite is enabled")
        data_dir = nosqlite_data_dir or os.environ.get("LECORE_X402_NOSQLITE_DATA_DIR")
        if not data_dir:
            raise ValueError("LECORE_X402_NOSQLITE_DATA_DIR is required when NoSQLite is enabled")
        nosqlite_store = NoSQLiteMemoryStore(
            nosqlite_binary or os.environ.get("LECORE_X402_NOSQLITE_BIN", "nosqlite"),
            data_dir,
            durability=nosqlite_durability or os.environ.get("LECORE_X402_NOSQLITE_DURABILITY", "sync"),
        )
    memory_transactions = TenantMemoryTransactions(store, tenant_state_dir) if tenant_state_dir else None

    @asynccontextmanager
    async def lifespan(_: Any) -> Any:
        try:
            if memory_transactions is not None:
                recovery = memory_transactions.recover_pending(nosqlite_store)
                if recovery["recovered"] or recovery["pending"] or recovery["invalid"]:
                    LOG.info("memory transaction recovery: %s", recovery)
            yield
        finally:
            if nosqlite_store is not None:
                nosqlite_store.close()

    app = FastAPI(title="leCore x402 API", version="0.1.0", lifespan=lifespan)
    app.state.memory_backend = memory_backend
    app.state.nosqlite_shadow = nosqlite_shadow
    app.state.nosqlite_store = nosqlite_store
    app.state.memory_transactions = memory_transactions
    config = config or (X402Config.from_env(require_pay_to=paid) if paid else X402Config.from_env(require_pay_to=False))
    tenant_secret = tenant_secret or os.environ.get("LECORE_X402_TENANT_SECRET")

    if paid:
        try:
            from x402.http.middleware.fastapi import PaymentMiddlewareASGI
        except ImportError as exc:
            raise RuntimeError(optional_dependency_help()) from exc
        app.add_middleware(
            PaymentMiddlewareASGI,
            routes=x402_route_configs(config),
            server=x402_resource_server(config),
        )

    def require_admin(header_value: Optional[str]) -> None:
        if not admin_token:
            raise HTTPException(status_code=403, detail="admin writes are disabled")
        if not header_value or not hmac.compare_digest(header_value, admin_token):
            raise HTTPException(status_code=401, detail="invalid admin token")

    def require_tenant_access(tenant_id: str, token: Optional[str]) -> None:
        normalized = normalize_tenant_id(tenant_id)
        if normalized == DEFAULT_TENANT_ID:
            return
        if not tenant_secret:
            raise HTTPException(status_code=403, detail="private tenants require LECORE_X402_TENANT_SECRET")
        expected = tenant_access_token(normalized, tenant_secret)
        if not token or not hmac.compare_digest(token, expected):
            raise HTTPException(status_code=401, detail="invalid tenant token")

    def tenant_from_header(header_value: Optional[str]) -> str:
        try:
            return normalize_tenant_id(header_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def tenant_from_payload(payload: Dict[str, Any], header_value: Optional[str]) -> str:
        try:
            payload_value = payload.get("tenant")
            payload_tenant = normalize_tenant_id(payload_value) if payload_value is not None else None
            header_tenant = normalize_tenant_id(header_value) if header_value is not None else None
            if payload_tenant is not None and header_tenant is not None and payload_tenant != header_tenant:
                raise ValueError("tenant id in payload does not match %s" % TENANT_HEADER)
            return payload_tenant or header_tenant or DEFAULT_TENANT_ID
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def validated(callable_: Any, *args: Any) -> Any:
        try:
            return callable_(*args)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def tenancy_public_dict() -> Dict[str, Any]:
        return {
            "default_tenant": DEFAULT_TENANT_ID,
            "tenant_header": TENANT_HEADER,
            "tenant_token_header": TENANT_TOKEN_HEADER,
            "private_tenants_enabled": bool(tenant_secret),
        }

    def memory_public_dict() -> Dict[str, Any]:
        return {
            "backend": memory_backend,
            "nosqlite_shadow": bool(nosqlite_shadow),
            "nosqlite_configured": nosqlite_store is not None,
            "durable_transactions": memory_transactions is not None,
        }

    def nosqlite_unavailable(error: NoSQLiteError) -> HTTPException:
        LOG.warning("NoSQLite memory backend is unavailable: %s", error)
        return HTTPException(status_code=503, detail="NoSQLite memory backend is unavailable")

    def sync_nosqlite_tenant(tenant_id: str) -> None:
        if nosqlite_store is None:
            return
        memories = store.read(tenant_id, lambda tenant_core: [entry.to_dict() for entry in tenant_core.entries])
        nosqlite_store.sync(tenant_id, memories)

    def shadow_recall(tenant_id: str, query: str, k: int, abstain: Optional[float], core_hits: List[Dict[str, Any]]) -> None:
        if nosqlite_store is None:
            return
        try:
            sync_nosqlite_tenant(tenant_id)
            shadow_hits = nosqlite_store.recall(tenant_id, query, k=k, abstain=abstain)
        except NoSQLiteError as exc:
            LOG.warning("NoSQLite shadow recall failed: %s", exc)
            return
        if [hit.get("id") for hit in core_hits] != [hit.get("id") for hit in shadow_hits]:
            LOG.info("NoSQLite shadow recall differs from LocalAgentCore")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def landing() -> str:
        return landing_page_html(config)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "name": "leCore x402 API",
            "paid": bool(paid),
            "memory": store.summary(DEFAULT_TENANT_ID),
            "memory_backend": memory_public_dict(),
            "tenancy": {
                "default_tenant": DEFAULT_TENANT_ID,
                "loaded_tenants": len(store.loaded_tenants()),
                "private_tenants_enabled": bool(tenant_secret),
            },
        }

    @app.get("/pricing")
    def pricing() -> Dict[str, Any]:
        return {
            "ok": True,
            "x402": config.to_public_dict(),
            "pricing": pricing_summary(config),
            "tenancy": tenancy_public_dict(),
            "memory_backend": memory_public_dict(),
            "routes": payment_manifest(config),
        }

    def recall_response(
        payload: Dict[str, Any],
        x_lecore_tenant: Optional[str],
        x_lecore_tenant_token: Optional[str],
    ) -> Dict[str, Any]:
        tenant_id = tenant_from_payload(payload, x_lecore_tenant)
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        query = validated(_required_text, payload, "query", MAX_QUERY_CHARS)
        k = validated(_recall_k, payload)
        abstain = validated(_abstain_threshold, payload)
        if memory_backend == MEMORY_BACKEND_NOSQLITE:
            if nosqlite_store is None:  # pragma: no cover - guarded during app setup
                raise HTTPException(status_code=503, detail="NoSQLite memory backend is not configured")
            try:
                sync_nosqlite_tenant(tenant_id)
                hits = nosqlite_store.recall(tenant_id, query, k=k, abstain=abstain)
            except NoSQLiteError as exc:
                raise nosqlite_unavailable(exc) from exc
        else:
            hits = store.read(
                tenant_id,
                lambda tenant_core: tenant_core.recall(query, k=k, abstain=abstain),
            )
            if nosqlite_shadow:
                shadow_recall(tenant_id, query, k, abstain, hits)
        return {
            "ok": True,
            "tenant": tenant_id,
            "query": query,
            "hits": hits,
        }

    @app.post("/v1/recall")
    def recall(
        payload: Dict[str, Any],
        x_lecore_tenant: Optional[str] = Header(default=None, alias=TENANT_HEADER),
        x_lecore_tenant_token: Optional[str] = Header(default=None, alias=TENANT_TOKEN_HEADER),
    ) -> Dict[str, Any]:
        return recall_response(payload, x_lecore_tenant, x_lecore_tenant_token)

    def route_response(
        payload: Dict[str, Any],
        x_lecore_tenant: Optional[str],
        x_lecore_tenant_token: Optional[str],
    ) -> Dict[str, Any]:
        tenant_id = tenant_from_payload(payload, x_lecore_tenant)
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        task = validated(_required_text, payload, "task", MAX_TASK_CHARS)
        routed = store.read(tenant_id, lambda tenant_core: tenant_core.route(task))
        return {"ok": True, "tenant": tenant_id, "route": routed}

    @app.post("/v1/route")
    def route(
        payload: Dict[str, Any],
        x_lecore_tenant: Optional[str] = Header(default=None, alias=TENANT_HEADER),
        x_lecore_tenant_token: Optional[str] = Header(default=None, alias=TENANT_TOKEN_HEADER),
    ) -> Dict[str, Any]:
        return route_response(payload, x_lecore_tenant, x_lecore_tenant_token)

    def dashboard_response(
        x_lecore_tenant: Optional[str],
        x_lecore_tenant_token: Optional[str],
    ) -> Dict[str, Any]:
        tenant_id = tenant_from_header(x_lecore_tenant)
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        data = store.read(tenant_id, lambda tenant_core: tenant_core.dashboard())
        return {"ok": True, "tenant": tenant_id, "dashboard": data}

    @app.get("/v1/dashboard")
    def dashboard(
        x_lecore_tenant: Optional[str] = Header(default=None, alias=TENANT_HEADER),
        x_lecore_tenant_token: Optional[str] = Header(default=None, alias=TENANT_TOKEN_HEADER),
    ) -> Dict[str, Any]:
        return dashboard_response(x_lecore_tenant, x_lecore_tenant_token)

    @app.post("/admin/remember")
    def remember(
        payload: Dict[str, Any],
        x_admin_token: Optional[str] = Header(default=None),
        x_lecore_tenant: Optional[str] = Header(default=None, alias=TENANT_HEADER),
        idempotency_key: Optional[str] = Header(default=None, alias=IDEMPOTENCY_HEADER),
    ) -> Dict[str, Any]:
        require_admin(x_admin_token)
        tenant_id = tenant_from_payload(payload, x_lecore_tenant)
        text = validated(_required_text, payload, "text", MAX_MEMORY_CHARS)
        key = validated(normalize_idempotency_key, idempotency_key)
        label = payload.get("label")
        metadata = payload.get("metadata")
        if label is not None and not isinstance(label, str):
            raise HTTPException(status_code=400, detail="label must be a string")
        if metadata is not None and not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="metadata must be an object")

        transaction = None
        if memory_transactions is not None:
            try:
                committed = memory_transactions.remember(
                    tenant_id,
                    text,
                    label,
                    metadata,
                    key,
                    nosqlite_store,
                )
            except MemoryTransactionConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except MemoryTransactionError as exc:
                raise HTTPException(status_code=500, detail="memory transaction could not be completed") from exc
            except NoSQLiteError as exc:
                if memory_backend == MEMORY_BACKEND_NOSQLITE:
                    raise nosqlite_unavailable(exc) from exc
                LOG.warning("NoSQLite shadow write failed: %s", exc)
                if not isinstance(exc, MemoryMirrorPending):  # pragma: no cover - mirror errors are wrapped above
                    raise nosqlite_unavailable(exc) from exc
                committed = memory_transactions.resume(exc.tenant_id, exc.transaction_id, None)
            memory = committed["memory"]
            transaction = committed["transaction"]
        else:
            if key is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Idempotency-Key requires LECORE_X402_TENANT_STATE_DIR for durable retries",
                )
            memory = store.write(
                tenant_id,
                lambda tenant_core: tenant_core.remember(text, label=label, metadata=metadata),
            )
            if nosqlite_store is not None:  # pragma: no cover - NoSQLite requires durable tenant state
                try:
                    nosqlite_store.remember(tenant_id, memory)
                except NoSQLiteError as exc:
                    if memory_backend == MEMORY_BACKEND_NOSQLITE:
                        raise nosqlite_unavailable(exc) from exc
                    LOG.warning("NoSQLite shadow write failed: %s", exc)
        return {
            "ok": True,
            "tenant": tenant_id,
            "memory": memory,
            "transaction": transaction,
        }

    @app.post("/admin/tenant-token")
    def issue_tenant_token(payload: Dict[str, Any], x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        require_admin(x_admin_token)
        if not tenant_secret:
            raise HTTPException(status_code=403, detail="tenant tokens require LECORE_X402_TENANT_SECRET")
        tenant_id = tenant_from_payload(payload, None)
        return {
            "ok": True,
            "tenant": tenant_id,
            "tenant_header": TENANT_HEADER,
            "tenant_token_header": TENANT_TOKEN_HEADER,
            "tenant_token": tenant_access_token(tenant_id, tenant_secret),
        }

    return app


def load_core(path: Optional[str]) -> LocalAgentCore:
    """Load a persisted core if present, otherwise return the demo core."""
    if path and Path(path).exists():
        return LocalAgentCore.load(path)
    return demo()


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point for local x402 API serving."""
    p = argparse.ArgumentParser(description="Serve LocalAgentCore as an x402-paid API")
    p.add_argument("--host", default=os.environ.get("LECORE_X402_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("LECORE_X402_PORT", "4021")))
    p.add_argument("--state", default=os.environ.get("LECORE_X402_STATE"))
    p.add_argument("--pay-to", default=os.environ.get("LECORE_X402_PAY_TO", ""))
    p.add_argument("--price", default=os.environ.get("LECORE_X402_PRICE", DEFAULT_PRICE))
    p.add_argument("--network", default=os.environ.get("LECORE_X402_NETWORK", DEFAULT_NETWORK))
    p.add_argument("--facilitator-url", default=os.environ.get("LECORE_X402_FACILITATOR_URL", DEFAULT_FACILITATOR_URL))
    p.add_argument("--admin-token", default=os.environ.get("LECORE_X402_ADMIN_TOKEN"))
    p.add_argument("--tenant-secret", default=os.environ.get("LECORE_X402_TENANT_SECRET"))
    p.add_argument("--tenant-state-dir", default=os.environ.get("LECORE_X402_TENANT_STATE_DIR"))
    p.add_argument(
        "--memory-backend",
        choices=(MEMORY_BACKEND_CORE, MEMORY_BACKEND_NOSQLITE),
        default=os.environ.get("LECORE_X402_MEMORY_BACKEND", MEMORY_BACKEND_CORE),
    )
    p.add_argument("--nosqlite-bin", default=os.environ.get("LECORE_X402_NOSQLITE_BIN", "nosqlite"))
    p.add_argument("--nosqlite-data-dir", default=os.environ.get("LECORE_X402_NOSQLITE_DATA_DIR"))
    p.add_argument(
        "--nosqlite-durability",
        choices=("sync", "buffered"),
        default=os.environ.get("LECORE_X402_NOSQLITE_DURABILITY", "sync"),
    )
    p.add_argument("--nosqlite-shadow", action="store_true", default=None)
    p.add_argument("--unpaid-dev", action="store_true", help="Disable x402 middleware for local development only")
    args = p.parse_args(list(argv) if argv is not None else None)

    paid = not args.unpaid_dev
    config = X402Config(
        pay_to=args.pay_to or ("0xYourAddress" if not paid else ""),
        price=args.price,
        network=args.network,
        facilitator_url=args.facilitator_url,
    )
    app = create_app(
        load_core(args.state),
        config=config,
        paid=paid,
        admin_token=args.admin_token,
        tenant_secret=args.tenant_secret,
        tenant_state_dir=args.tenant_state_dir,
        memory_backend=args.memory_backend,
        nosqlite_binary=args.nosqlite_bin,
        nosqlite_data_dir=args.nosqlite_data_dir,
        nosqlite_durability=args.nosqlite_durability,
        nosqlite_shadow=args.nosqlite_shadow,
    )
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(optional_dependency_help()) from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
