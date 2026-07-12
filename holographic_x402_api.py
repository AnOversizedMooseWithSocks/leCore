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

from dataclasses import dataclass
import hashlib
import hmac
from html import escape
import argparse
import os
from pathlib import Path
import re
from string import Template
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

from holographic_product import LocalAgentCore, demo


DEFAULT_FACILITATOR_URL = "https://x402.org/facilitator"
DEFAULT_NETWORK = "eip155:84532"  # Base Sepolia, safe default for testnet publishing.
DEFAULT_PRICE = "$0.0011"
LEOS_SITE_URL = "https://discoverleos.com/"
LEOS_TOKEN_CA = "5xgsnby6P9zqGK71J7H4yJLxzqPvNbC7rDZxNzjHmj7e"
LEOS_TOKEN_PRICE = "$0.0010"
DEFAULT_TENANT_ID = "public"
TENANT_HEADER = "X-leCore-Tenant"
TENANT_TOKEN_HEADER = "X-leCore-Tenant-Token"
_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")


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

LEOS_PAID_ROUTES: Tuple[PaidRoute, ...] = (
    PaidRoute("POST", "/leos/v1/recall", "Recall nearest memories at the leOS CA offer price", price=LEOS_TOKEN_PRICE),
    PaidRoute("POST", "/leos/v1/route", "Route a task at the leOS CA offer price", price=LEOS_TOKEN_PRICE),
    PaidRoute("GET", "/leos/v1/dashboard", "Read the dashboard at the leOS CA offer price", price=LEOS_TOKEN_PRICE),
)

DEFAULT_PAID_ROUTES: Tuple[PaidRoute, ...] = REGULAR_PAID_ROUTES + LEOS_PAID_ROUTES


LANDING_PAGE_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>leCore x402 API</title>
<meta name="description" content="Buy local agent memory, capability routing, and readiness dashboards as x402-paid HTTP primitives.">
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
.token-offer{background:#fff}.token-panel{align-items:stretch;border:1px solid var(--line);display:grid;gap:1px;grid-template-columns:minmax(180px,.8fr) minmax(0,1.8fr) auto}.token-panel>div,.token-panel>a{background:#fbfaf5;padding:22px}.token-panel span{color:var(--muted);display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;margin-bottom:16px;text-transform:uppercase}.token-panel strong{font-size:clamp(28px,4vw,54px);line-height:1}.token-panel code{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:clamp(13px,1.8vw,19px);overflow-wrap:anywhere}.token-panel .button{justify-content:center;min-width:170px}
.use{background:#f2eee3}.cases{grid-template-columns:repeat(4,minmax(0,1fr))}.case{display:flex;gap:16px;min-height:220px;padding:22px}.case span{color:var(--coral);font-size:30px;line-height:1}.case p{color:#3b382f;font-size:18px;line-height:1.42}
.proof{background:var(--graphite);color:var(--paper)}.proof h2,.proof p{color:var(--paper)}.proof-copy p:last-child{color:rgba(251,250,245,.75);font-size:19px;line-height:1.55;margin-top:26px;max-width:690px}.proof-panel{background:rgba(251,250,245,.96);color:var(--ink);padding:8px}.proof-panel dl{display:grid;gap:1px;grid-template-columns:repeat(2,1fr);margin:0}.proof-panel div{background:#fbfaf5;min-height:150px;padding:22px}.proof-panel dt{color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;margin-bottom:28px;text-transform:uppercase}.proof-panel dd{font-size:clamp(24px,3vw,40px);font-weight:760;margin:0;overflow-wrap:anywhere}
.close{align-items:center;display:grid;grid-template-columns:minmax(0,1fr) auto}.close h2{max-width:980px}.close-actions{justify-content:flex-end;margin-top:0}
@media(max-width:980px){.hero,.split,.proof,.close{grid-template-columns:1fr}.hero{min-height:auto}.terminal{align-self:start;margin-bottom:0;max-width:100%}.routes,.cases,.token-panel{grid-template-columns:1fr}.case,.card{min-height:0}.close-actions{justify-content:flex-start;margin-top:24px}}
@media(max-width:680px){.hero{padding:20px 18px 36px}.topbar,.nav,.strip{align-items:flex-start;display:grid;grid-template-columns:1fr}.nav{gap:8px}.copy{padding-top:62px}.status span,.button{width:100%}.strip div{border-bottom:1px solid rgba(23,23,20,.22);border-right:0}.proof-panel dl{grid-template-columns:1fr}.section{padding-left:18px;padding-right:18px}}
</style>
</head>
<body>
<main>
<section class="hero" aria-labelledby="hero-title">
<div class="field" aria-hidden="true"><span class="trace a"></span><span class="trace b"></span><span class="trace c"></span>$nodes</div>
<nav class="topbar" aria-label="Primary"><a class="brand" href="#hero-title" aria-label="leCore x402 API home"><span class="mark">lc</span><span>leCore x402 API</span></a><div class="nav"><a href="#why">Why buy</a><a href="#routes">Routes</a><a href="#leos">leOS</a><a href="#proof">Proof</a></div></nav>
<div class="copy"><p class="status"><span>Live on AWS</span><span>$network_label</span><span>$price per call</span><span>leOS CA $leos_token_price</span></p><h1 id="hero-title">leCore x402 API</h1><p class="lede">Buy the small, useful surface of leCore: local agent memory, capability routing, and a readiness dashboard, sold as paid HTTP primitives instead of another subscription dashboard.</p><div class="actions"><a class="button primary" href="/pricing">View pricing</a><a class="button secondary" href="/v1/dashboard">See x402 challenge</a><a class="button secondary" href="$leos_site_url" target="_blank" rel="noopener">leOS site</a></div></div>
<aside class="terminal" aria-label="Live API snapshot"><div class="terminal-top"><span></span><span></span><span></span></div><pre>GET /health
200 OK
memory.entries: 3
paid: true

GET /v1/dashboard
402 Payment Required
network: $network
asset: USDC</pre></aside>
</section>
<section class="strip" aria-label="Live deployment details"><div><strong>Endpoint</strong><span>https://lecore.rati.foundation</span></div><div><strong>Payment</strong><span>x402 exact scheme</span></div><div><strong>Buyer shape</strong><span>inspect, pay, call</span></div></section>
<section id="why" class="section split"><div><p class="eyebrow">Why you would buy it</p><h2>Because most agents do not need a platform. They need a few reliable cognitive calls.</h2></div><div class="reason-list"><p>You pay for an answerable primitive, not a monthly seat.</p><p>The API is narrow enough to trust: read/compute routes are paid, memory writes stay admin-gated.</p><p>It exposes the useful part of leCore first: local agent memory plus capability routing.</p><p>The implementation is deployed, health-checked, and already returning x402 payment challenges.</p></div></section>
<section id="leos" class="section token-offer"><div class="heading"><p class="eyebrow">leOS token offer</p><h2>A slightly cheaper CA-only price for the leOS token.</h2></div><div class="token-panel" aria-label="leOS token CA offer"><div><span>Token price</span><strong>$leos_token_price</strong></div><div><span>CA</span><code>$leos_token_ca</code></div><a class="button primary" href="$leos_site_url" target="_blank" rel="noopener">leOS website</a></div></section>
<section id="routes" class="section"><div class="heading"><p class="eyebrow">What the payment unlocks</p><h2>Three paid routes, each small enough to understand.</h2></div><div class="routes"><article class="card"><p class="method">POST</p><h3>Recall</h3><code>/v1/recall</code><p>Pull nearest memories from a compact local agent core without shipping a whole application stack.</p></article><article class="card"><p class="method">POST</p><h3>Route</h3><code>/v1/route</code><p>Send a plain-language task and get the leCore capability it should use, with evidence attached.</p></article><article class="card"><p class="method">GET</p><h3>Dashboard</h3><code>/v1/dashboard</code><p>Read the readiness surface: memory counts, capability map, abstention behavior, and route coverage.</p></article></div></section>
<section class="section use"><div class="heading"><p class="eyebrow">Good first buyers</p><h2>Teams who want the leCore idea without adopting the whole repo.</h2></div><div class="cases"><article class="case"><span aria-hidden="true">+</span><p>Agent memory for prototypes that should remember without a database rollout.</p></article><article class="case"><span aria-hidden="true">+</span><p>Capability routing for tools that need to pick the right leCore subsystem before doing work.</p></article><article class="case"><span aria-hidden="true">+</span><p>Evidence dashboards for teams deciding whether a local vector system is ready to productize.</p></article><article class="case"><span aria-hidden="true">+</span><p>A working x402 seller endpoint to copy when you want pay-per-call APIs instead of subscriptions.</p></article></div></section>
<section id="proof" class="section proof"><div class="proof-copy"><p class="eyebrow">Proof it is real</p><h2>It is already deployed, priced, and protected.</h2><p>The free endpoints show health and pricing. Paid endpoints return a real x402 payment challenge. The receiving address is public, while admin writes stay out of the paid customer path.</p></div><div class="proof-panel"><dl><div><dt>Price</dt><dd>$price</dd></div><div><dt>Network</dt><dd>$network_name</dd></div><div><dt>Receiver</dt><dd>$pay_to_short</dd></div><div><dt>Status</dt><dd>Healthy</dd></div></dl></div></section>
<section class="section close"><p class="eyebrow">The pitch</p><h2>Buy it when you want a local-memory agent primitive that can pay for itself one request at a time.</h2><div class="close-actions"><a class="button primary" href="/pricing">Inspect the offer</a><a class="button secondary dark" href="/health">Check live health</a></div></section>
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
    tenant_id = str(value or DEFAULT_TENANT_ID).strip().lower()
    if not tenant_id:
        tenant_id = DEFAULT_TENANT_ID
    if not _TENANT_ID_RE.match(tenant_id):
        raise ValueError("tenant id must be 1-64 chars of lowercase letters, numbers, '.', ':', '_' or '-'")
    return tenant_id


def tenant_access_token(tenant_id: str, secret: str) -> str:
    """Deterministic tenant bearer token derived from a server-side secret."""
    normalized = normalize_tenant_id(tenant_id)
    return hmac.new(secret.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256).hexdigest()


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
        self._lock = threading.RLock()
        self._state_dir = Path(state_dir) if state_dir else None
        if self._state_dir is not None:
            self._state_dir.mkdir(parents=True, exist_ok=True)

    def loaded_tenants(self) -> List[str]:
        """Return tenant ids currently loaded in memory."""
        with self._lock:
            return sorted(self._cores)

    def read(self, tenant_id: str, fn: Any) -> Any:
        """Run a read-style operation while holding the tenant lock."""
        with self._lock:
            return fn(self._get_locked(tenant_id))

    def write(self, tenant_id: str, fn: Any) -> Any:
        """Run a mutating operation, then persist that tenant if configured."""
        with self._lock:
            normalized = normalize_tenant_id(tenant_id)
            core = self._get_locked(normalized)
            result = fn(core)
            self._save_locked(normalized, core)
            return result

    def _get_locked(self, tenant_id: str) -> LocalAgentCore:
        normalized = normalize_tenant_id(tenant_id)
        core = self._cores.get(normalized)
        if core is not None:
            return core
        path = self._path_for(normalized)
        if path is not None and path.exists():
            core = LocalAgentCore.load(path)
        else:
            core = LocalAgentCore(
                dim=self._default_dim,
                seed=self._default_seed,
                route_threshold=self._default_route_threshold,
            )
        self._cores[normalized] = core
        return core

    def _path_for(self, tenant_id: str) -> Optional[Path]:
        if self._state_dir is None:
            return None
        return self._state_dir / ("%s.json" % normalize_tenant_id(tenant_id))

    def _save_locked(self, tenant_id: str, core: LocalAgentCore) -> None:
        path = self._path_for(tenant_id)
        if path is not None:
            core.save(path)


def leos_token_offer() -> Dict[str, Any]:
    """Public metadata for the leOS CA-only offer."""
    return {
        "name": "leOS CA offer",
        "site": LEOS_SITE_URL,
        "ca": LEOS_TOKEN_CA,
        "price": LEOS_TOKEN_PRICE,
        "discount_routes": [route.key for route in LEOS_PAID_ROUTES],
        "note": "Only the CA is needed for this token offer.",
    }


def landing_page_html(config: X402Config) -> str:
    """Render the buyer-facing landing page served from `/`."""
    network_name = _network_name(config.network)
    offer = leos_token_offer()
    return LANDING_PAGE_TEMPLATE.substitute(
        nodes=_landing_nodes(),
        price=escape(config.price),
        network=escape(config.network),
        network_label=escape("%s x402" % network_name),
        network_name=escape(network_name),
        pay_to_short=escape(_short_address(config.pay_to)),
        leos_site_url=escape(offer["site"], quote=True),
        leos_token_ca=escape(offer["ca"]),
        leos_token_price=escape(offer["price"]),
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
        if route.price:
            row["offer"] = "leos_ca"
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
) -> Any:
    """Create the FastAPI app.

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

    app = FastAPI(title="leCore x402 API", version="0.1.0")
    core = core or demo()
    store = TenantCoreStore(core, state_dir=tenant_state_dir)
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
        if header_value != admin_token:
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
            return normalize_tenant_id(payload.get("tenant") or header_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def tenancy_public_dict() -> Dict[str, Any]:
        return {
            "default_tenant": DEFAULT_TENANT_ID,
            "tenant_header": TENANT_HEADER,
            "tenant_token_header": TENANT_TOKEN_HEADER,
            "private_tenants_enabled": bool(tenant_secret),
        }

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def landing() -> str:
        return landing_page_html(config)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        default_evidence = store.read(DEFAULT_TENANT_ID, lambda tenant_core: tenant_core.evidence())
        return {
            "ok": True,
            "name": "leCore x402 API",
            "paid": bool(paid),
            "memory": default_evidence["memory"],
            "tenancy": {
                "default_tenant": DEFAULT_TENANT_ID,
                "loaded_tenants": len(store.loaded_tenants()),
                "private_tenants_enabled": bool(tenant_secret),
            },
        }

    @app.get("/pricing")
    async def pricing() -> Dict[str, Any]:
        return {
            "ok": True,
            "x402": config.to_public_dict(),
            "token_offer": leos_token_offer(),
            "tenancy": tenancy_public_dict(),
            "routes": payment_manifest(config),
        }

    @app.post("/v1/recall")
    @app.post("/leos/v1/recall")
    async def recall(
        payload: Dict[str, Any],
        x_lecore_tenant: Optional[str] = Header(default=None, alias=TENANT_HEADER),
        x_lecore_tenant_token: Optional[str] = Header(default=None, alias=TENANT_TOKEN_HEADER),
    ) -> Dict[str, Any]:
        tenant_id = tenant_from_payload(payload, x_lecore_tenant)
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        query = payload.get("query")
        if query is None:
            raise HTTPException(status_code=400, detail="POST /v1/recall needs {query}")
        hits = store.read(
            tenant_id,
            lambda tenant_core: tenant_core.recall(query, k=int(payload.get("k", 3)), abstain=payload.get("abstain")),
        )
        return {
            "ok": True,
            "tenant": tenant_id,
            "query": query,
            "hits": hits,
        }

    @app.post("/v1/route")
    @app.post("/leos/v1/route")
    async def route(
        payload: Dict[str, Any],
        x_lecore_tenant: Optional[str] = Header(default=None, alias=TENANT_HEADER),
        x_lecore_tenant_token: Optional[str] = Header(default=None, alias=TENANT_TOKEN_HEADER),
    ) -> Dict[str, Any]:
        tenant_id = tenant_from_payload(payload, x_lecore_tenant)
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        task = payload.get("task")
        if task is None:
            raise HTTPException(status_code=400, detail="POST /v1/route needs {task}")
        routed = store.read(tenant_id, lambda tenant_core: tenant_core.route(str(task)))
        return {"ok": True, "tenant": tenant_id, "route": routed}

    @app.get("/v1/dashboard")
    @app.get("/leos/v1/dashboard")
    async def dashboard(
        x_lecore_tenant: Optional[str] = Header(default=None, alias=TENANT_HEADER),
        x_lecore_tenant_token: Optional[str] = Header(default=None, alias=TENANT_TOKEN_HEADER),
    ) -> Dict[str, Any]:
        tenant_id = tenant_from_header(x_lecore_tenant)
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        data = store.read(tenant_id, lambda tenant_core: tenant_core.dashboard())
        return {"ok": True, "tenant": tenant_id, "dashboard": data}

    @app.post("/admin/remember")
    async def remember(
        payload: Dict[str, Any],
        x_admin_token: Optional[str] = Header(default=None),
        x_lecore_tenant: Optional[str] = Header(default=None, alias=TENANT_HEADER),
    ) -> Dict[str, Any]:
        require_admin(x_admin_token)
        tenant_id = tenant_from_payload(payload, x_lecore_tenant)
        text = payload.get("text")
        if text is None:
            raise HTTPException(status_code=400, detail="POST /admin/remember needs {text}")
        memory = store.write(
            tenant_id,
            lambda tenant_core: tenant_core.remember(text, label=payload.get("label"), metadata=payload.get("metadata")),
        )
        return {
            "ok": True,
            "tenant": tenant_id,
            "memory": memory,
        }

    @app.post("/admin/tenant-token")
    async def issue_tenant_token(payload: Dict[str, Any], x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
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
    )
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(optional_dependency_help()) from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
