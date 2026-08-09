"""holographic_x402_api.py -- publish the leCore Agent Memory & Routing API.

WHY THIS EXISTS
---------------
`LocalAgentCore` remains the embedded implementation facade. This module
translates it into a hosted HTTP API without making x402, FastAPI, or uvicorn
core dependencies.

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
from urllib.parse import urlsplit

from holographic_product import LocalAgentCore, demo
from lecore import __version__ as LECORE_VERSION


DEFAULT_FACILITATOR_URL = "https://x402.org/facilitator"
DEFAULT_NETWORK = "eip155:84532"  # Base Sepolia, safe default for testnet publishing.
DEFAULT_PRICE = "$0.0011"
DEFAULT_PUBLIC_URL = "https://lecore.rati.foundation"
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

SERVICE_NAME = "leCore Agent Memory & Routing API"
HERO_TITLE = "Agent memory and routing, paid per call."
X402_BUYER_GUIDE_URL = "https://docs.x402.org/getting-started/quickstart-for-buyers"
API_DESCRIPTION = """Hosted, tenant-scoped agent memory, capability routing, and
readiness data over HTTPS, with x402 payment on each protected request.

## Request flow

1. Read `GET /pricing` for the network, asset, price, and protected-route manifest.
2. Call a protected `/v1/*` route. An unsigned request returns `402 Payment Required`.
3. Decode the `Payment-Required` response header with an x402 v2 client.
4. Sign the selected payment option and retry with the resulting `Payment-Signature` header.
5. Decode the successful `Payment-Response` header for settlement details.

The interactive reference describes the contract but does not sign payments. See the
[x402 buyer quickstart](https://docs.x402.org/getting-started/quickstart-for-buyers)
for wallet and client setup.
`GET /health`, `GET /pricing`, `/docs`, `/redoc`, and `/openapi.json`
are free. Private tenant calls additionally require `X-leCore-Tenant` and
`X-leCore-Tenant-Token`; payment proves payment, not tenant authorization.
"""
OPENAPI_TAGS = [
    {
        "name": "Discovery",
        "description": "Free service health, pricing, network, and route discovery.",
    },
    {
        "name": "Paid API",
        "description": "Hosted read and compute operations protected by the x402 v2 payment flow.",
    },
]


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
    PaidRoute("POST", "/v1/recall", "Recall nearest memories from tenant-scoped agent memory"),
    PaidRoute("POST", "/v1/route", "Route a plain-English task to a leCore capability"),
    PaidRoute("GET", "/v1/dashboard", "Read the service readiness dashboard"),
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


def _normalize_public_url(value: str) -> str:
    """Return a canonical public base URL safe to advertise in x402 challenges."""
    if not isinstance(value, str):
        raise ValueError("public_url must be a string")
    value = value.strip().rstrip("/")
    if not value or any(char.isspace() or char == "\\" or ord(char) == 127 for char in value):
        raise ValueError("public_url must be an absolute http(s) URL")
    try:
        parts = urlsplit(value)
        _ = parts.port
    except ValueError as exc:
        raise ValueError("public_url must be an absolute http(s) URL") from exc
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.netloc.endswith(":")
        or parts.hostname is None
        or not parts.hostname.strip(".")
    ):
        raise ValueError("public_url must be an absolute http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("public_url must not contain credentials")
    if parts.query or parts.fragment or "?" in value or "#" in value:
        raise ValueError("public_url must not contain a query or fragment")
    return value


def x402_payment_required_responses() -> Dict[int, Dict[str, Any]]:
    """OpenAPI response metadata shared by every x402-protected operation."""
    return {
        402: {
            "description": (
                "Payment required or settlement failed. An unsigned or invalid-payment "
                "request includes Payment-Required; a paid request whose settlement "
                "fails can instead include Payment-Response."
            ),
            "headers": {
                "Payment-Required": {
                    "description": (
                        "Base64-encoded x402 v2 PaymentRequired challenge, present for "
                        "unsigned or invalid-payment requests."
                    ),
                    "schema": {"type": "string", "format": "byte"},
                },
                "Payment-Response": {
                    "description": (
                        "Base64-encoded x402 v2 settlement response, present when a "
                        "paid request reaches the handler but settlement fails."
                    ),
                    "schema": {"type": "string", "format": "byte"},
                },
            },
            "content": {
                "application/json": {
                    "schema": {"type": "object", "maxProperties": 0},
                    "example": {},
                },
                "text/html": {
                    "schema": {"type": "string"},
                    "example": "<!doctype html><title>Payment Required</title>",
                },
            },
        },
    }


def paid_request_openapi(
    required: List[str],
    properties: Dict[str, Dict[str, Any]],
    example: Dict[str, Any],
    example_summary: str,
) -> Dict[str, Any]:
    """Return an accurate OpenAPI request body while runtime validation stays compatible."""
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": required,
                        "properties": properties,
                        "additionalProperties": True,
                    },
                    "examples": {
                        "public": {
                            "summary": example_summary,
                            "value": example,
                        },
                    },
                },
            },
        },
    }


def _json_success_response(
    description: str,
    schema: Dict[str, Any],
    example: Dict[str, Any],
    *,
    payment_receipt: bool = True,
) -> Dict[str, Any]:
    """Return one documented JSON success response."""
    response = {
        "description": description,
        "content": {
            "application/json": {
                "schema": schema,
                "example": example,
            },
        },
    }
    if payment_receipt:
        response["headers"] = {
            "Payment-Response": {
                "description": "Base64-encoded x402 v2 settlement response.",
                "schema": {"type": "string", "format": "byte"},
            },
        }
    return response


def _error_response(description: str, detail: str) -> Dict[str, Any]:
    """Return the shared JSON error envelope used by FastAPI routes."""
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["detail"],
                    "properties": {"detail": {"type": "string"}},
                    "additionalProperties": False,
                },
                "example": {"detail": detail},
            },
        },
    }


def paid_operation_responses(
    success: Dict[str, Any],
    *,
    invalid_detail: str,
    backend_unavailable: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Document paid success, payment, tenant, and validation responses."""
    responses = {
        200: success,
        400: _error_response("Invalid request.", invalid_detail),
        401: _error_response("Private-tenant authorization failed.", "invalid tenant token"),
        403: _error_response(
            "The deployment cannot authorize the selected private tenant.",
            "private tenants require LECORE_X402_TENANT_SECRET",
        ),
        502: {
            "description": "The x402 facilitator could not verify or settle the payment.",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["error"],
                        "properties": {"error": {"type": "string"}},
                        "additionalProperties": True,
                    },
                    "example": {"error": "Payment verification failed"},
                },
            },
        },
    }
    responses.update(x402_payment_required_responses())
    if backend_unavailable:
        responses[503] = _error_response(
            "The configured memory backend is temporarily unavailable.",
            "NoSQLite memory backend is unavailable",
        )
    return responses


def _capability_openapi_schema() -> Dict[str, Any]:
    """Return the stable public shape of one routed capability."""
    return {
        "type": "object",
        "required": ["name", "does", "call"],
        "properties": {
            "name": {"type": "string"},
            "does": {"type": "string"},
            "call": {"type": "string"},
        },
        "additionalProperties": False,
    }


def health_success_openapi(
    *,
    paid: bool,
    private_tenants_enabled: bool,
    memory_backend: str,
    nosqlite_shadow: bool,
    nosqlite_configured: bool,
    durable_transactions: bool,
) -> Dict[str, Any]:
    """Document the free health and deployment-state response."""
    schema = {
        "type": "object",
        "required": ["ok", "name", "paid", "memory", "memory_backend", "tenancy"],
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "name": {"type": "string"},
            "paid": {"type": "boolean"},
            "memory": {
                "type": "object",
                "required": ["entries", "dim", "index_method", "query_mutates_store"],
                "properties": {
                    "entries": {"type": "integer", "minimum": 0},
                    "dim": {"type": "integer", "minimum": 1},
                    "index_method": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "query_mutates_store": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "memory_backend": {
                "type": "object",
                "required": ["backend", "nosqlite_shadow", "nosqlite_configured", "durable_transactions"],
                "properties": {
                    "backend": {"type": "string", "enum": ["core", "nosqlite"]},
                    "nosqlite_shadow": {"type": "boolean"},
                    "nosqlite_configured": {"type": "boolean"},
                    "durable_transactions": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "tenancy": {
                "type": "object",
                "required": ["default_tenant", "loaded_tenants", "private_tenants_enabled"],
                "properties": {
                    "default_tenant": {"type": "string"},
                    "loaded_tenants": {"type": "integer", "minimum": 1},
                    "private_tenants_enabled": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
    example = {
        "ok": True,
        "name": SERVICE_NAME,
        "paid": paid,
        "memory": {
            "entries": 3,
            "dim": 512,
            "index_method": "exact",
            "query_mutates_store": False,
        },
        "memory_backend": {
            "backend": memory_backend,
            "nosqlite_shadow": nosqlite_shadow,
            "nosqlite_configured": nosqlite_configured,
            "durable_transactions": durable_transactions,
        },
        "tenancy": {
            "default_tenant": DEFAULT_TENANT_ID,
            "loaded_tenants": 1,
            "private_tenants_enabled": private_tenants_enabled,
        },
    }
    return _json_success_response(
        "Service health and deployment state returned.",
        schema,
        example,
        payment_receipt=False,
    )


def pricing_success_openapi(
    config: X402Config,
    *,
    private_tenants_enabled: bool,
    memory_backend: str,
    nosqlite_shadow: bool,
    nosqlite_configured: bool,
    durable_transactions: bool,
) -> Dict[str, Any]:
    """Document the free x402 discovery manifest."""
    string_map = {"type": "object", "additionalProperties": {"type": "string"}}
    schema = {
        "type": "object",
        "required": ["ok", "documentation", "x402", "pricing", "tenancy", "memory_backend", "routes"],
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "documentation": {
                "type": "object",
                "required": ["swagger_ui", "reference", "openapi_schema"],
                "properties": {
                    "swagger_ui": {"type": "string", "format": "uri"},
                    "reference": {"type": "string", "format": "uri"},
                    "openapi_schema": {"type": "string", "format": "uri"},
                },
                "additionalProperties": False,
            },
            "x402": {
                "type": "object",
                "required": ["pay_to", "price", "network", "facilitator_url", "scheme", "public_url"],
                "properties": {
                    "pay_to": {"type": "string"},
                    "price": {"type": "string"},
                    "network": {"type": "string"},
                    "facilitator_url": {"type": "string", "format": "uri"},
                    "scheme": {"type": "string"},
                    "public_url": {"type": "string", "format": "uri"},
                },
                "additionalProperties": False,
            },
            "pricing": string_map,
            "tenancy": {
                "type": "object",
                "required": ["default_tenant", "tenant_header", "tenant_token_header", "private_tenants_enabled"],
                "properties": {
                    "default_tenant": {"type": "string"},
                    "tenant_header": {"type": "string"},
                    "tenant_token_header": {"type": "string"},
                    "private_tenants_enabled": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "memory_backend": {
                "type": "object",
                "required": ["backend", "nosqlite_shadow", "nosqlite_configured", "durable_transactions"],
                "properties": {
                    "backend": {"type": "string", "enum": ["core", "nosqlite"]},
                    "nosqlite_shadow": {"type": "boolean"},
                    "nosqlite_configured": {"type": "boolean"},
                    "durable_transactions": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "routes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["route", "description", "mime_type", "accepts"],
                    "properties": {
                        "route": {"type": "string"},
                        "description": {"type": "string"},
                        "mime_type": {"type": "string"},
                        "accepts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["scheme", "price", "network", "pay_to"],
                                "properties": {
                                    "scheme": {"type": "string"},
                                    "price": {"type": "string"},
                                    "network": {"type": "string"},
                                    "pay_to": {"type": "string"},
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }
    example = {
        "ok": True,
        "documentation": documentation_manifest(config),
        "x402": config.to_public_dict(),
        "pricing": pricing_summary(config),
        "tenancy": {
            "default_tenant": DEFAULT_TENANT_ID,
            "tenant_header": TENANT_HEADER,
            "tenant_token_header": TENANT_TOKEN_HEADER,
            "private_tenants_enabled": private_tenants_enabled,
        },
        "memory_backend": {
            "backend": memory_backend,
            "nosqlite_shadow": nosqlite_shadow,
            "nosqlite_configured": nosqlite_configured,
            "durable_transactions": durable_transactions,
        },
        "routes": payment_manifest(config),
    }
    return _json_success_response(
        "Pricing and x402 discovery manifest returned.",
        schema,
        example,
        payment_receipt=False,
    )


def recall_success_openapi() -> Dict[str, Any]:
    """Document the successful memory-recall response."""
    hit_schema = {
        "type": "object",
        "required": ["id", "text", "label", "metadata", "score"],
        "properties": {
            "id": {"type": "string"},
            "text": {"type": "string"},
            "label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "metadata": {"type": "object", "additionalProperties": True},
            "score": {"type": "number"},
        },
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "required": ["ok", "tenant", "query", "hits"],
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "tenant": {"type": "string"},
            "query": {"type": "string"},
            "hits": {"type": "array", "items": hit_schema},
        },
        "additionalProperties": False,
    }
    example = {
        "ok": True,
        "tenant": "public",
        "query": "deterministic memory",
        "hits": [{
            "id": "m2",
            "text": "Prefer explicit capability routing when confidence is low.",
            "label": "routing",
            "metadata": {"source": "public-preview"},
            "score": 0.82,
        }],
    }
    return _json_success_response("Memory recall completed.", schema, example)


def route_success_openapi() -> Dict[str, Any]:
    """Document the successful capability-routing response."""
    capability = _capability_openapi_schema()
    route_schema = {
        "type": "object",
        "required": ["task", "decision", "confidence"],
        "properties": {
            "task": {"type": "string"},
            "decision": {"type": "string", "enum": ["act", "choose", "unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "prompt": {"type": "string"},
            "skill": capability,
            "options": {"type": "array", "items": capability},
        },
        "additionalProperties": True,
    }
    schema = {
        "type": "object",
        "required": ["ok", "tenant", "route"],
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "tenant": {"type": "string"},
            "route": route_schema,
        },
        "additionalProperties": False,
    }
    example = {
        "ok": True,
        "tenant": "public",
        "route": {
            "task": "search a large vector collection",
            "decision": "act",
            "confidence": 0.91,
            "skill": {
                "name": "Index (search)",
                "does": "Search a vector index for nearest entries.",
                "call": "index.nearest(query, k=5)",
            },
        },
    }
    return _json_success_response("Capability routing completed.", schema, example)


def dashboard_success_openapi() -> Dict[str, Any]:
    """Document the successful service-readiness response."""
    schema = {
        "type": "object",
        "required": ["ok", "tenant", "dashboard"],
        "properties": {
            "ok": {"type": "boolean", "const": True},
            "tenant": {"type": "string"},
            "dashboard": {
                "type": "object",
                "required": ["name", "status", "memory", "routing", "c_kernel", "checks"],
                "properties": {
                    "name": {"type": "string"},
                    "status": {"type": "string", "enum": ["ready", "check"]},
                    "memory": {
                        "type": "object",
                        "required": ["entries", "dim", "index_method", "query_mutates_store"],
                        "properties": {
                            "entries": {"type": "integer", "minimum": 0},
                            "dim": {"type": "integer", "minimum": 1},
                            "index_method": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "query_mutates_store": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                    "routing": {
                        "type": "object",
                        "required": ["capabilities", "probe_decision", "probe_skill"],
                        "properties": {
                            "capabilities": {"type": "integer", "minimum": 0},
                            "probe_decision": {"type": "string"},
                            "probe_skill": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                        "additionalProperties": False,
                    },
                    "c_kernel": {"type": "object", "additionalProperties": True},
                    "checks": {
                        "type": "object",
                        "required": ["deterministic_encoding", "no_model_weights", "self_contained_engine"],
                        "properties": {
                            "deterministic_encoding": {"type": "boolean"},
                            "no_model_weights": {"type": "boolean"},
                            "self_contained_engine": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": False,
    }
    example = {
        "ok": True,
        "tenant": "public",
        "dashboard": {
            "name": SERVICE_NAME,
            "status": "ready",
            "memory": {
                "entries": 3,
                "dim": 512,
                "index_method": "exact",
                "query_mutates_store": False,
            },
            "routing": {
                "capabilities": 656,
                "probe_decision": "act",
                "probe_skill": "Index (search)",
            },
            "c_kernel": {"available": False, "path": None},
            "checks": {
                "deterministic_encoding": True,
                "no_model_weights": True,
                "self_contained_engine": True,
            },
        },
    }
    return _json_success_response("Readiness dashboard returned.", schema, example)


def public_response_headers(
    path: str,
    status_code: int,
    public_url: str,
    content_type: str = "",
    network: str = DEFAULT_NETWORK,
) -> Dict[str, str]:
    """Return browser and cache policy headers for one public response."""
    if path == "/docs":
        content_security_policy = (
            "default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://fastapi.tiangolo.com; connect-src 'self'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
    elif path == "/redoc":
        content_security_policy = (
            "default-src 'none'; script-src https://cdn.jsdelivr.net; "
            "style-src 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )
    elif path == "/":
        content_security_policy = (
            "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )
    else:
        content_security_policy = (
            "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )

    private_or_dynamic = (
        status_code >= 400
        or path == "/health"
        or path.startswith("/v1/")
        or path.startswith("/admin/")
    )
    headers = {
        "Cache-Control": (
            "no-store" if private_or_dynamic else "public, max-age=60, must-revalidate"
        ),
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-Permitted-Cross-Domain-Policies": "none",
    }
    # x402's pinned browser paywall is a self-contained wallet application with
    # inline script/style. Permit only its same-origin retry, the configured
    # chain's public RPC, and the two optional Coinbase telemetry endpoints.
    browser_paywall = (
        status_code == 402
        and path.startswith("/v1/")
        and content_type.lower().startswith("text/html")
    )
    if browser_paywall:
        rpc_source = {
            "eip155:84532": "https://sepolia.base.org",
            "eip155:8453": "https://mainnet.base.org",
        }.get(network, "https:")
        content_security_policy = (
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self' %s https://rpc.wallet.coinbase.com "
            "https://cca-lite.coinbase.com https://as.coinbase.com; "
            "img-src 'self' data:; font-src 'none'; media-src 'none'; "
            "object-src 'none'; frame-src 'none'; worker-src 'none'; "
            "manifest-src 'none'; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'" % rpc_source
        )
    headers["Content-Security-Policy"] = content_security_policy
    if urlsplit(public_url).scheme == "https":
        headers["Strict-Transport-Security"] = "max-age=31536000"
    return headers


LANDING_PAGE_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$service_name</title>
<meta name="description" content="Call hosted agent memory, capability routing, and readiness endpoints over HTTPS with x402 payment per request.">
<meta name="theme-color" content="#20201c">
<meta property="og:title" content="$service_name">
<meta property="og:description" content="Hosted agent memory and capability routing with x402 payment per request.">
<link rel="canonical" href="$public_url/">
<link rel="alternate" type="application/json" href="/openapi.json" title="OpenAPI schema">
<style>
:root{--paper:#fbfaf5;--ink:#171714;--muted:#6f6b61;--line:#ded8c8;--acid:#b7ff3c;--cyan:#28d6ff;--coral:#ff6b57;--coral-text:#b6402f;--gold:#f4c542;--graphite:#20201c}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:inherit;text-decoration:none}a:focus-visible,button:focus-visible{outline:3px solid currentColor;outline-offset:4px}.skip{background:var(--acid);color:var(--ink);font-weight:760;left:16px;padding:12px 16px;position:fixed;top:12px;transform:translateY(-180%);transition:transform 140ms ease;z-index:20}.skip:focus{transform:translateY(0)}
.hero{background:var(--graphite);color:var(--paper);display:grid;grid-template-columns:minmax(0,1.1fr) minmax(300px,.8fr);min-height:720px;overflow:hidden;padding:24px clamp(20px,5vw,72px) 38px;position:relative}
.field{inset:0;overflow:hidden;position:absolute}.field:before{background-image:linear-gradient(rgba(251,250,245,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(251,250,245,.08) 1px,transparent 1px);background-size:56px 56px;content:"";inset:-80px;opacity:.45;position:absolute;transform:rotate(-7deg)}.field:after{background:radial-gradient(circle at 18% 20%,rgba(183,255,60,.42),transparent 25%),radial-gradient(circle at 70% 22%,rgba(40,214,255,.35),transparent 28%),radial-gradient(circle at 78% 74%,rgba(255,107,87,.28),transparent 26%),linear-gradient(135deg,rgba(32,32,28,.12),rgba(32,32,28,.95));content:"";inset:0;position:absolute}
.trace{border:1px solid rgba(251,250,245,.16);border-radius:999px;position:absolute}.trace.a{height:52vw;right:-14vw;top:5vw;width:52vw}.trace.b{border-color:rgba(183,255,60,.24);height:34vw;left:38vw;top:25vh;width:34vw}.trace.c{border-color:rgba(255,107,87,.2);height:42vw;left:-16vw;top:42vh;width:42vw}
.node{animation:pulse 5s ease-in-out infinite;background:var(--acid);border-radius:999px;box-shadow:0 0 18px currentColor;color:var(--acid);height:var(--size);left:var(--x);opacity:.82;position:absolute;top:var(--y);width:var(--size);z-index:1}.node:nth-child(3n){background:var(--cyan);color:var(--cyan)}.node:nth-child(5n){background:var(--coral);color:var(--coral)}
@keyframes pulse{0%,100%{transform:translate3d(0,0,0) scale(.84)}50%{transform:translate3d(12px,-10px,0) scale(1.18)}}
.topbar{align-items:center;display:flex;gap:24px;grid-column:1/-1;justify-content:space-between;position:relative;z-index:2}.brand,.nav{align-items:center;display:flex}.brand{font-weight:760;gap:10px}.mark{align-items:center;background:var(--acid);color:var(--ink);display:inline-flex;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;height:34px;justify-content:center;width:34px}.nav{color:rgba(251,250,245,.78);font-size:14px;gap:8px}.nav a{align-items:center;display:inline-flex;min-height:44px;padding:0 8px}
.copy{align-self:center;max-width:760px;padding:54px 0 24px;position:relative;z-index:2}.status{color:rgba(251,250,245,.78);display:flex;flex-wrap:wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;gap:10px;letter-spacing:0;margin:0 0 20px;text-transform:uppercase}.status span{border:1px solid rgba(251,250,245,.22);padding:8px 10px}
h1,h2,h3,p{margin-top:0}h1{font-size:clamp(48px,7vw,96px);line-height:.94;margin-bottom:22px;max-width:820px}.lede{color:rgba(251,250,245,.82);font-size:clamp(18px,1.8vw,24px);line-height:1.38;max-width:720px}.actions,.close-actions{display:flex;flex-wrap:wrap;gap:12px;margin-top:28px}.button{align-items:center;border:1px solid currentColor;display:inline-flex;font-weight:720;min-height:48px;padding:14px 18px;transition:transform 160ms ease,background 160ms ease,color 160ms ease}.button:hover{transform:translateY(-2px)}.primary{background:var(--acid);border-color:var(--acid);color:var(--ink)}.secondary{color:var(--paper)}.secondary.dark{color:var(--ink)}
.terminal{align-self:center;background:rgba(251,250,245,.96);border:1px solid rgba(251,250,245,.26);box-shadow:0 28px 90px rgba(0,0,0,.3);color:var(--ink);max-width:500px;position:relative;z-index:2}.terminal-top{align-items:center;border-bottom:1px solid var(--line);display:flex;gap:7px;padding:12px 14px}.terminal-top span{background:var(--coral);border-radius:999px;height:10px;width:10px}.terminal-top span:nth-child(2){background:var(--gold)}.terminal-top span:nth-child(3){background:var(--cyan)}pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:clamp(13px,1.2vw,15px);line-height:1.6;margin:0;overflow-x:auto;padding:22px;white-space:pre-wrap}
.strip{background:var(--acid);color:var(--ink);display:grid;grid-template-columns:repeat(3,1fr)}.strip div{border-right:1px solid rgba(23,23,20,.22);padding:18px clamp(18px,4vw,54px)}.strip div:last-child{border-right:0}.strip strong,.strip span{display:block}.strip strong{font-size:13px;text-transform:uppercase}.strip span{font-size:clamp(18px,2vw,28px);font-weight:760;margin-top:5px}
.section{padding:clamp(64px,8vw,112px) clamp(20px,5vw,72px)}.proof{display:grid;gap:clamp(32px,6vw,80px);grid-template-columns:minmax(0,.9fr) minmax(320px,1.1fr)}.eyebrow{color:var(--coral-text);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;font-weight:720;letter-spacing:0;margin-bottom:16px;text-transform:uppercase}.section h2{color:var(--ink);font-size:clamp(34px,5vw,68px);line-height:1;margin-bottom:0;max-width:930px}.heading{margin-bottom:clamp(28px,5vw,52px)}
.quickstart{background:#f2eee3}.quick-grid{display:grid;gap:clamp(28px,5vw,64px);grid-template-columns:minmax(280px,.78fr) minmax(0,1.22fr)}.flow{counter-reset:steps;display:grid;gap:0;list-style:none;margin:34px 0 0;padding:0}.flow li{border-top:1px solid var(--line);display:grid;gap:12px;grid-template-columns:36px 1fr;padding:20px 0}.flow li:before{align-items:center;background:var(--graphite);border-radius:999px;color:var(--paper);content:counter(steps);counter-increment:steps;display:flex;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;height:28px;justify-content:center;width:28px}.flow strong{display:block;font-size:18px;margin-bottom:5px}.flow p{color:var(--muted);line-height:1.5;margin:0}.flow a{text-decoration:underline;text-underline-offset:3px}.command-panel{align-self:start;background:var(--graphite);border-radius:8px;box-shadow:0 24px 70px rgba(23,23,20,.2);color:var(--paper);overflow:hidden}.command-head{align-items:center;border-bottom:1px solid rgba(251,250,245,.16);display:flex;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;justify-content:space-between;padding:14px 18px}.command-head a{color:var(--acid);min-height:32px;padding:7px 0}.command-panel pre{font-size:14px;padding:22px}.response-note{border-top:1px solid rgba(251,250,245,.16);color:rgba(251,250,245,.72);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;line-height:1.55;margin:0;padding:18px 22px}.response-note strong{color:var(--acid)}
.routes{display:grid;gap:14px;grid-template-columns:repeat(3,minmax(0,1fr))}.card,.proof-panel{background:#fff;border:1px solid var(--line);border-radius:8px}.card{min-height:315px;padding:24px}.method{color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;margin-bottom:42px}.card h3{font-size:30px;margin-bottom:10px}.card code{background:#f2eee3;display:inline-block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;margin-bottom:18px;padding:7px 9px}.card p:last-child{color:var(--muted);font-size:17px;line-height:1.48}
.proof{background:var(--graphite);color:var(--paper)}.proof h2,.proof p{color:var(--paper)}.proof-copy p:last-child{color:rgba(251,250,245,.75);font-size:19px;line-height:1.55;margin-top:26px;max-width:690px}.proof-panel{background:rgba(251,250,245,.96);color:var(--ink);padding:8px}.proof-panel dl{display:grid;gap:1px;grid-template-columns:repeat(2,1fr);margin:0}.proof-panel div{background:#fbfaf5;min-height:150px;padding:22px}.proof-panel dt{color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;margin-bottom:28px;text-transform:uppercase}.proof-panel dd{font-size:clamp(24px,3vw,40px);font-weight:760;margin:0;overflow-wrap:anywhere}
.close{align-items:center;display:grid;grid-template-columns:minmax(0,1fr) auto}.close h2{max-width:980px}.close-actions{justify-content:flex-end;margin-top:0}
.footer{align-items:center;border-top:1px solid var(--line);color:var(--muted);display:flex;font-size:14px;gap:24px;justify-content:space-between;padding:24px clamp(20px,5vw,72px)}.footer nav{display:flex;flex-wrap:wrap;gap:18px}.footer a{align-items:center;display:inline-flex;min-height:44px;text-decoration:underline;text-underline-offset:3px}
@media(max-width:980px){.hero,.proof,.close,.quick-grid{grid-template-columns:1fr}.hero{min-height:auto}.terminal{align-self:start;margin-bottom:0;max-width:100%}.routes{grid-template-columns:1fr}.card{min-height:0}.close-actions{justify-content:flex-start;margin-top:24px}}
@media(max-width:680px){.hero{padding:18px 18px 34px}.topbar{align-items:flex-start;display:grid;grid-template-columns:1fr}.nav{display:grid;gap:4px;grid-template-columns:repeat(2,minmax(0,1fr));width:100%}.nav a{min-height:44px;padding:0}.copy{padding-top:42px}.status span,.button{width:100%}.strip{align-items:flex-start;display:grid;grid-template-columns:1fr}.strip div{border-bottom:1px solid rgba(23,23,20,.22);border-right:0}.proof-panel dl{grid-template-columns:1fr}.section{padding-left:18px;padding-right:18px}.footer{align-items:flex-start;flex-direction:column}.footer nav{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));width:100%}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.node{animation:none}.button,.skip{transition:none}.button:hover{transform:none}}
</style>
</head>
<body>
<a class="skip" href="#quickstart">Skip to quickstart</a>
<main>
<section class="hero" aria-labelledby="hero-title">
<div class="field" aria-hidden="true"><span class="trace a"></span><span class="trace b"></span><span class="trace c"></span>$nodes</div>
<nav class="topbar" aria-label="Primary"><a class="brand" href="#hero-title" aria-label="$service_name home"><span class="mark">lc</span><span>leCore API</span></a><div class="nav"><a href="#quickstart">Quickstart</a><a href="/docs">API docs</a><a href="/redoc">Reference</a><a href="/pricing">Pricing</a></div></nav>
<div class="copy"><p class="status"><span>$price_per_request</span><span>$environment_label</span><span>$network_label</span></p><h1 id="hero-title">$hero_title</h1><p class="lede">A hosted HTTPS API for querying seeded preview memory, routing tasks to leCore capabilities, and reading service readiness. $payment_notice</p><div class="actions"><a class="button primary" href="#quickstart">Make the first request</a><a class="button secondary" href="/docs">Explore API docs</a><a class="button secondary" href="/pricing">View pricing</a></div></div>
<aside class="terminal" aria-label="Unsigned x402 request example"><div class="terminal-top"><span></span><span></span><span></span></div><pre>curl -i $public_url/v1/dashboard

HTTP/2 402 Payment Required
Payment-Required: &lt;base64 challenge&gt;

# Sign, then retry with:
Payment-Signature: &lt;base64 payment&gt;</pre></aside>
</section>
<section class="strip" aria-label="Deployment details"><div><strong>Endpoint</strong><span><a href="$public_url">$public_url</a></span></div><div><strong>Stage</strong><span>$environment_label</span></div><div><strong>Protocol</strong><span>x402 v2</span></div></section>
<section id="quickstart" class="section quickstart" tabindex="-1"><div class="quick-grid"><div><p class="eyebrow">Four-step quickstart</p><h2>Inspect the terms before signing anything.</h2><ol class="flow"><li><div><strong>Read the free manifest</strong><p><a href="/pricing">GET /pricing</a> returns the exact route, network, asset, receiver, and price.</p></div></li><li><div><strong>Make an unsigned request</strong><p>The protected route returns <code>402</code> with a base64 <code>Payment-Required</code> challenge.</p></div></li><li><div><strong>Sign with an x402 v2 client</strong><p>Use the <a href="$buyer_guide_url">x402 buyer guide</a> to configure a testnet wallet and payment client.</p></div></li><li><div><strong>Retry and verify settlement</strong><p>Send <code>Payment-Signature</code>; a successful response includes <code>Payment-Response</code>.</p></div></li></ol></div><div class="command-panel"><div class="command-head"><span>First request: no wallet required</span><a href="/docs#/Paid%20API/getDashboard">Open route docs</a></div><pre><code>curl -i $public_url/v1/dashboard</code></pre><p class="response-note"><strong>Expected:</strong> HTTP 402 plus <code>Payment-Required</code>. This safely exposes the payment contract without moving testnet funds.</p><div class="command-head"><span>Inspect exact preview terms</span><a href="/pricing">Open live JSON</a></div><pre><code>curl -sS $public_url/pricing</code></pre></div></div></section>
<section id="routes" class="section"><div class="heading"><p class="eyebrow">Paid API surface</p><h2>Three explicit operations, with no account or subscription.</h2></div><div class="routes"><article class="card"><p class="method">POST</p><h3>Recall</h3><code>/v1/recall</code><p>Query the seeded public preview memory or an operator-provisioned private tenant.</p></article><article class="card"><p class="method">POST</p><h3>Route</h3><code>/v1/route</code><p>Send a plain-language task and receive an explicit act, choose, or unknown decision with evidence.</p></article><article class="card"><p class="method">GET</p><h3>Dashboard</h3><code>/v1/dashboard</code><p>Read memory, capability-routing, and deterministic-engine readiness for one tenant.</p></article></div></section>
<section id="proof" class="section proof"><div class="proof-copy"><p class="eyebrow">Preview boundaries</p><h2>Deployed, health-checked, and ready to test.</h2><p>Base Sepolia testnet only. The public memory dataset is read-only; private tenants and memory writes are currently operator-provisioned. Operator routes require separate authorization and are absent from the public OpenAPI schema.</p></div><div class="proof-panel"><dl><div><dt>Per request</dt><dd>$price_per_request</dd></div><div><dt>Per 1,000</dt><dd>$price_per_thousand</dd></div><div><dt>Network</dt><dd>$network_name</dd></div><div><dt>API version</dt><dd>$api_version</dd></div></dl></div></section>
<section class="section close"><p class="eyebrow">Start testing</p><h2>See the full request and response contract.</h2><div class="close-actions"><a class="button primary" href="/docs">Open API docs</a><a class="button secondary dark" href="/openapi.json">View OpenAPI schema</a><a class="button secondary dark" href="/health">Check live health</a></div></section>
</main>
<footer class="footer"><span>$service_name · testnet preview</span><nav aria-label="Footer"><a href="/docs">Swagger</a><a href="/redoc">ReDoc</a><a href="/pricing">Pricing</a><a href="$buyer_guide_url">x402 buyer guide</a></nav></footer>
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
    public_url: str = DEFAULT_PUBLIC_URL

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
        object.__setattr__(self, "public_url", _normalize_public_url(self.public_url))

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
            public_url=os.environ.get("LECORE_X402_PUBLIC_URL", DEFAULT_PUBLIC_URL),
        )

    def to_public_dict(self) -> Dict[str, Any]:
        """Public, JSON-safe view of the payment configuration."""
        return {
            "pay_to": self.pay_to,
            "price": self.price,
            "network": self.network,
            "facilitator_url": self.facilitator_url,
            "scheme": self.scheme,
            "public_url": self.public_url,
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
        """Return the number of successful NoSQLite process starts."""
        with self._lock:
            return self._generation

    @property
    def running(self) -> bool:
        """Return whether the managed NoSQLite process is currently alive."""
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
        """Return whether the underlying NoSQLite process is currently alive."""
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
        """Release the underlying NoSQLite process and writer lock."""
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
        service_name=escape(SERVICE_NAME),
        hero_title=escape(HERO_TITLE),
        api_version=escape(LECORE_VERSION),
        buyer_guide_url=escape(X402_BUYER_GUIDE_URL),
        nodes=_landing_nodes(),
        public_url=escape(config.public_url),
        network_label=escape("%s x402" % network_name),
        network_name=escape(network_name),
        environment_label=escape(summary["environment_label"]),
        payment_notice=escape(summary["payment_notice"]),
        price_per_request=escape("%s per request" % summary["per_request"]),
        price_per_thousand=escape(summary["display_price"]),
    )


def documentation_manifest(config: X402Config) -> Dict[str, str]:
    """Return canonical public documentation URLs for discovery responses."""
    return {
        "swagger_ui": config.public_url + "/docs",
        "reference": config.public_url + "/redoc",
        "openapi_schema": config.public_url + "/openapi.json",
    }


def public_dashboard(data: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the embedded SDK dashboard into the hosted API vocabulary."""
    out = dict(data)
    out["name"] = SERVICE_NAME
    checks = dict(out.get("checks") or {})
    if "local_only" in checks:
        checks["self_contained_engine"] = bool(checks.pop("local_only"))
    out["checks"] = checks
    return out


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
            resource=config.public_url + route.path,
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
    """Create the FastAPI application for paid or unpaid development serving.

    With `paid=True`, the public `/v1/*` read/compute routes are protected by
    x402 middleware. Set `paid=False` only for an unpaid development smoke test.

    x402 proves that a request paid. Private tenant memory is intentionally a
    separate authorization layer using `X-leCore-Tenant-Token`.
    """
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise RuntimeError(optional_dependency_help()) from exc

    config = config or (X402Config.from_env(require_pay_to=paid) if paid else X402Config.from_env(require_pay_to=False))
    public = urlsplit(config.public_url)
    if paid and public.scheme != "https" and public.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("paid mode public_url must use https outside localhost")

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

    app = FastAPI(
        title=SERVICE_NAME,
        description=API_DESCRIPTION,
        version=LECORE_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=OPENAPI_TAGS,
        servers=[{"url": config.public_url, "description": "Public API"}],
        openapi_external_docs={
            "description": "x402 buyer quickstart",
            "url": X402_BUYER_GUIDE_URL,
        },
        lifespan=lifespan,
    )
    app.state.memory_backend = memory_backend
    app.state.nosqlite_shadow = nosqlite_shadow
    app.state.nosqlite_store = nosqlite_store
    app.state.memory_transactions = memory_transactions
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

    @app.middleware("http")
    async def apply_public_response_policy(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        for name, value in public_response_headers(
            request.url.path,
            response.status_code,
            config.public_url,
            response.headers.get("content-type", ""),
            config.network,
        ).items():
            response.headers[name] = value
        return response

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

    @app.get(
        "/health",
        tags=["Discovery"],
        operation_id="getHealth",
        summary="Check service health",
        description="Free liveness, memory-state, backend, and tenancy summary. No x402 payment is required.",
        responses={
            200: health_success_openapi(
                paid=bool(paid),
                private_tenants_enabled=bool(tenant_secret),
                memory_backend=memory_backend,
                nosqlite_shadow=bool(nosqlite_shadow),
                nosqlite_configured=nosqlite_store is not None,
                durable_transactions=memory_transactions is not None,
            ),
        },
    )
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "name": SERVICE_NAME,
            "paid": bool(paid),
            "memory": store.summary(DEFAULT_TENANT_ID),
            "memory_backend": memory_public_dict(),
            "tenancy": {
                "default_tenant": DEFAULT_TENANT_ID,
                "loaded_tenants": len(store.loaded_tenants()),
                "private_tenants_enabled": bool(tenant_secret),
            },
        }

    @app.get(
        "/pricing",
        tags=["Discovery"],
        operation_id="getPricing",
        summary="Discover pricing and protected routes",
        description=(
            "Free discovery document for the x402 network, payment asset, price, "
            "tenant headers, documentation URLs, and protected-route manifest."
        ),
        responses={
            200: pricing_success_openapi(
                config,
                private_tenants_enabled=bool(tenant_secret),
                memory_backend=memory_backend,
                nosqlite_shadow=bool(nosqlite_shadow),
                nosqlite_configured=nosqlite_store is not None,
                durable_transactions=memory_transactions is not None,
            ),
        },
    )
    def pricing() -> Dict[str, Any]:
        return {
            "ok": True,
            "documentation": documentation_manifest(config),
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

    @app.post(
        "/v1/recall",
        tags=["Paid API"],
        operation_id="recallMemory",
        summary="Recall agent memory",
        description=(
            "Recall the nearest entries from tenant-scoped agent memory. An "
            "unsigned request returns the x402 challenge documented in the 402 response."
        ),
        responses=paid_operation_responses(
            recall_success_openapi(),
            invalid_detail="query must be a non-empty string",
            backend_unavailable=True,
        ),
        openapi_extra=paid_request_openapi(
            required=["query"],
            properties={
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_CHARS,
                    "pattern": r"\S",
                    "description": "Text to match against stored agent memory.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RECALL_K,
                    "default": 3,
                    "description": "Maximum number of memories to return.",
                },
                "abstain": {
                    "anyOf": [{"type": "number", "minimum": 0, "maximum": 1}, {"type": "null"}],
                    "description": "Optional minimum similarity score.",
                },
                "tenant": {
                    "type": "string",
                    "description": (
                        "Tenant id. Leading/trailing whitespace is removed and letters are "
                        "lowercased; the normalized id must match X-leCore-Tenant when both are supplied."
                    ),
                },
            },
            example={"query": "deterministic agent memory", "k": 3},
            example_summary="Recall public-tenant memory",
        ),
    )
    def recall(
        payload: Dict[str, Any],
        x_lecore_tenant: Optional[str] = Header(
            default=None,
            alias=TENANT_HEADER,
            description="Tenant id, trimmed and lowercased by the service. Omit for the public tenant.",
        ),
        x_lecore_tenant_token: Optional[str] = Header(
            default=None,
            alias=TENANT_TOKEN_HEADER,
            description=(
                "Required whenever the resolved tenant is private, whether selected "
                "by header or JSON body."
            ),
        ),
        _payment_signature: Optional[str] = Header(
            default=None,
            alias="Payment-Signature",
            description=(
                "Omit to receive the x402 challenge; include the base64 x402 v2 "
                "payment payload when retrying."
            ),
            json_schema_extra={"format": "byte"},
        ),
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

    @app.post(
        "/v1/route",
        tags=["Paid API"],
        operation_id="routeTask",
        summary="Route a task to a capability",
        description=(
            "Route a plain-English task to the best matching leCore capability. "
            "An unsigned request returns the x402 challenge documented in the 402 response."
        ),
        responses=paid_operation_responses(
            route_success_openapi(),
            invalid_detail="task must be a non-empty string",
        ),
        openapi_extra=paid_request_openapi(
            required=["task"],
            properties={
                "task": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_TASK_CHARS,
                    "pattern": r"\S",
                    "description": "Plain-English task to route.",
                },
                "tenant": {
                    "type": "string",
                    "description": (
                        "Tenant id. Leading/trailing whitespace is removed and letters are "
                        "lowercased; the normalized id must match X-leCore-Tenant when both are supplied."
                    ),
                },
            },
            example={"task": "find the best capability for semantic memory retrieval"},
            example_summary="Route a memory-related task",
        ),
    )
    def route(
        payload: Dict[str, Any],
        x_lecore_tenant: Optional[str] = Header(
            default=None,
            alias=TENANT_HEADER,
            description="Tenant id, trimmed and lowercased by the service. Omit for the public tenant.",
        ),
        x_lecore_tenant_token: Optional[str] = Header(
            default=None,
            alias=TENANT_TOKEN_HEADER,
            description=(
                "Required whenever the resolved tenant is private, whether selected "
                "by header or JSON body."
            ),
        ),
        _payment_signature: Optional[str] = Header(
            default=None,
            alias="Payment-Signature",
            description=(
                "Omit to receive the x402 challenge; include the base64 x402 v2 "
                "payment payload when retrying."
            ),
            json_schema_extra={"format": "byte"},
        ),
    ) -> Dict[str, Any]:
        return route_response(payload, x_lecore_tenant, x_lecore_tenant_token)

    def dashboard_response(
        x_lecore_tenant: Optional[str],
        x_lecore_tenant_token: Optional[str],
    ) -> Dict[str, Any]:
        tenant_id = tenant_from_header(x_lecore_tenant)
        require_tenant_access(tenant_id, x_lecore_tenant_token)
        data = store.read(tenant_id, lambda tenant_core: tenant_core.dashboard())
        data = public_dashboard(data)
        return {"ok": True, "tenant": tenant_id, "dashboard": data}

    @app.get(
        "/v1/dashboard",
        tags=["Paid API"],
        operation_id="getDashboard",
        summary="Read the readiness dashboard",
        description=(
            "Read memory, routing, native-kernel, and deterministic-engine readiness "
            "for one tenant. An unsigned request returns the documented x402 challenge."
        ),
        responses=paid_operation_responses(
            dashboard_success_openapi(),
            invalid_detail="tenant id is invalid",
        ),
    )
    def dashboard(
        x_lecore_tenant: Optional[str] = Header(
            default=None,
            alias=TENANT_HEADER,
            description="Tenant id, trimmed and lowercased by the service. Omit for the public tenant.",
        ),
        x_lecore_tenant_token: Optional[str] = Header(
            default=None,
            alias=TENANT_TOKEN_HEADER,
            description="Required whenever the resolved tenant is private.",
        ),
        _payment_signature: Optional[str] = Header(
            default=None,
            alias="Payment-Signature",
            description=(
                "Omit to receive the x402 challenge; include the base64 x402 v2 "
                "payment payload when retrying."
            ),
            json_schema_extra={"format": "byte"},
        ),
    ) -> Dict[str, Any]:
        return dashboard_response(x_lecore_tenant, x_lecore_tenant_token)

    @app.post("/admin/remember", include_in_schema=False)
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

    @app.post("/admin/tenant-token", include_in_schema=False)
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
    """CLI entry point for running the x402 API service."""
    p = argparse.ArgumentParser(description="Serve the leCore Agent Memory & Routing API with x402 payments")
    p.add_argument("--host", default=os.environ.get("LECORE_X402_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("LECORE_X402_PORT", "4021")))
    p.add_argument("--state", default=os.environ.get("LECORE_X402_STATE"))
    p.add_argument("--pay-to", default=os.environ.get("LECORE_X402_PAY_TO", ""))
    p.add_argument("--price", default=os.environ.get("LECORE_X402_PRICE", DEFAULT_PRICE))
    p.add_argument("--network", default=os.environ.get("LECORE_X402_NETWORK", DEFAULT_NETWORK))
    p.add_argument("--facilitator-url", default=os.environ.get("LECORE_X402_FACILITATOR_URL", DEFAULT_FACILITATOR_URL))
    p.add_argument("--public-url", default=os.environ.get("LECORE_X402_PUBLIC_URL", DEFAULT_PUBLIC_URL))
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
    p.add_argument("--unpaid-dev", action="store_true", help="Disable x402 middleware for development only")
    args = p.parse_args(list(argv) if argv is not None else None)

    paid = not args.unpaid_dev
    config = X402Config(
        pay_to=args.pay_to or ("0xYourAddress" if not paid else ""),
        price=args.price,
        network=args.network,
        facilitator_url=args.facilitator_url,
        public_url=args.public_url,
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
    uvicorn.run(app, host=args.host, port=args.port, server_header=False)


if __name__ == "__main__":
    main()
